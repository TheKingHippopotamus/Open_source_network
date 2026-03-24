"""
Optional dense embedding layer for Open Source Network.

Uses pre-computed numpy arrays for tool embeddings.
Gracefully degrades if numpy or embedding files are not available.
Can generate embeddings using sentence-transformers (optional dependency).

Files expected in data_dir:
  embeddings.npy         — float32 array of shape (N, D)
  embedding_index.json   — {"slug": row_index, ...}

Typical usage
-------------
    from engine.embeddings import EmbeddingEngine

    engine = EmbeddingEngine(data_dir=Path("data"), tools=tools)
    if engine.enabled:
        sim = engine.similarity(query_vec, "qdrant")
        neighbors = engine.most_similar("qdrant", limit=5)

CLI rebuild
-----------
    python -m engine.embeddings --rebuild
    python -m engine.embeddings --rebuild --model all-MiniLM-L6-v2 --db path/to/db.json --out path/to/data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# numpy is optional — only needed for dense operations at runtime.
# sentence-transformers is NEVER imported here at module level; it is only
# used inside build_embeddings(), which is called from the CLI entry-point.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Text representation helpers
# ---------------------------------------------------------------------------

def _build_tool_text(tool: Dict) -> str:
    """
    Construct a single text string that represents a tool for embedding.

    Fields are weighted by repetition — tags and problem_domains appear
    multiple times so the embedding is pulled toward those concepts.
    """
    parts: List[str] = []

    # Tags: highest signal — repeated 3x
    for tag in tool.get("tags", []):
        clean = tag.replace("-", " ")
        parts.extend([clean] * 3)

    # Problem domains: high signal — repeated 2x
    for domain in tool.get("problem_domains", []):
        clean = domain.replace("-", " ")
        parts.extend([clean] * 2)

    # Tagline: concise human description — once
    tagline = tool.get("tagline", "").strip()
    if tagline:
        parts.append(tagline)

    # Use cases: rich semantic content — once each
    for uc in tool.get("use_cases_detailed", []):
        uc_clean = uc.strip()
        if uc_clean:
            parts.append(uc_clean)

    # Category / sub-category — once
    category = tool.get("category", "").strip()
    sub_category = tool.get("sub_category", "").strip()
    if category:
        parts.append(category)
    if sub_category:
        parts.append(sub_category)

    # What the tool replaces — useful for "alternative to X" queries
    for r in tool.get("replaces", []):
        r_clean = r.strip()
        if r_clean:
            parts.append(r_clean)

    return " ".join(parts)


# ===========================================================================
# EmbeddingEngine
# ===========================================================================

class EmbeddingEngine:
    """
    Manages dense vector embeddings for semantic search over OSS tools.

    The engine loads pre-computed embeddings from disk at construction time.
    If the embedding files do not exist, or if numpy is unavailable, the
    engine operates in disabled mode — every method returns a safe default
    so callers never need to guard against None or raised exceptions.

    Parameters
    ----------
    data_dir : Path
        Directory that contains (or should contain) ``embeddings.npy`` and
        ``embedding_index.json``.
    tools : List[Dict]
        Full tool records from db.json.  Used to validate the index and
        as a reference for ``most_similar`` result construction.
    """

    def __init__(self, data_dir: Path, tools: List[Dict]) -> None:
        self.enabled: bool = False
        self._embeddings = None   # numpy ndarray (N, D) or None
        self._index: Dict[str, int] = {}  # slug -> row index
        self._tools: List[Dict] = tools
        self._data_dir: Path = data_dir
        self._slug_to_tool: Dict[str, Dict] = {
            t["slug"]: t for t in tools if "slug" in t
        }

        if not _NUMPY_AVAILABLE:
            # numpy not installed — silently operate in disabled mode
            return

        emb_path = data_dir / "embeddings.npy"
        idx_path = data_dir / "embedding_index.json"

        if not emb_path.exists() or not idx_path.exists():
            return

        try:
            embeddings = np.load(str(emb_path))
            with open(idx_path, "r", encoding="utf-8") as fh:
                raw_index = json.load(fh)

            # Accept both {"slug": int_row} and {int_row: "slug"} layouts
            if not isinstance(raw_index, dict):
                return

            if raw_index:
                first_val = next(iter(raw_index.values()))
                if isinstance(first_val, int):
                    # Canonical format: {slug: row}
                    index = {str(k): int(v) for k, v in raw_index.items()}
                else:
                    # Inverted format: {row: slug} — flip it
                    index = {str(v): int(k) for k, v in raw_index.items()}
            else:
                index = {}

            # Basic sanity: embedding array must be 2-D
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                return

            self._embeddings = embeddings.astype(np.float32)
            self._index = index
            self.enabled = True

        except Exception:
            # Any I/O or parsing failure — stay disabled, never raise
            self._embeddings = None
            self._index = {}
            self.enabled = False

    # -----------------------------------------------------------------------
    # Runtime query methods
    # -----------------------------------------------------------------------

    def similarity(
        self,
        query_embedding: "Optional[np.ndarray]",
        slug: str,
    ) -> float:
        """
        Compute cosine similarity between a query embedding and a stored tool
        embedding identified by ``slug``.

        Parameters
        ----------
        query_embedding : np.ndarray or None
            A 1-D float32 vector of the same dimensionality as the stored
            embeddings.  If None or if the engine is disabled, returns 0.0.
        slug : str
            Tool identifier to look up.

        Returns
        -------
        float
            Cosine similarity in [0.0, 1.0].  Negative cosine (dissimilar)
            is clamped to 0.0.  Returns 0.0 on any error.
        """
        if not self.enabled or query_embedding is None:
            return 0.0

        row = self._index.get(slug)
        if row is None:
            return 0.0

        try:
            tool_vec = self._embeddings[row]
            norm_q = float(np.linalg.norm(query_embedding))
            norm_t = float(np.linalg.norm(tool_vec))
            if norm_q == 0.0 or norm_t == 0.0:
                return 0.0
            cosine = float(
                np.dot(query_embedding.astype(np.float32), tool_vec)
                / (norm_q * norm_t)
            )
            return max(0.0, cosine)
        except Exception:
            return 0.0

    def most_similar(
        self,
        slug: str,
        limit: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Return the ``limit`` most similar tools to the tool identified by
        ``slug``, ranked by cosine similarity (descending).

        The query tool itself is excluded from results.

        Parameters
        ----------
        slug : str
            Reference tool slug.
        limit : int
            Maximum number of neighbors to return.

        Returns
        -------
        List[Tuple[str, float]]
            List of ``(slug, similarity)`` pairs, highest similarity first.
            Returns an empty list when the engine is disabled or the slug is
            not in the index.
        """
        if not self.enabled:
            return []

        row = self._index.get(slug)
        if row is None:
            return []

        try:
            query_vec = self._embeddings[row]
            norm_q = float(np.linalg.norm(query_vec))
            if norm_q == 0.0:
                return []

            # Batch cosine: dot(all_embeddings, query_vec) / (norms * norm_q)
            dots = self._embeddings @ query_vec  # shape (N,)
            norms = np.linalg.norm(self._embeddings, axis=1)  # shape (N,)

            # Avoid divide-by-zero for zero vectors
            safe_norms = np.where(norms == 0.0, 1.0, norms)
            cosines = dots / (safe_norms * norm_q)
            cosines = np.clip(cosines, 0.0, 1.0)

            # Build (slug, score) sorted list, excluding self
            row_to_slug: Dict[int, str] = {v: k for k, v in self._index.items()}
            scored: List[Tuple[str, float]] = []
            for idx, score in enumerate(cosines.tolist()):
                if idx == row:
                    continue
                candidate_slug = row_to_slug.get(idx)
                if candidate_slug is None:
                    continue
                if score > 0.0:
                    scored.append((candidate_slug, float(score)))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]

        except Exception:
            return []

    def get_embedding(self, slug: str) -> "Optional[np.ndarray]":
        """
        Return the raw embedding vector for a tool, or None if unavailable.

        Useful for composing with external query encoders.
        """
        if not self.enabled:
            return None

        row = self._index.get(slug)
        if row is None:
            return None

        try:
            return self._embeddings[row].copy()
        except Exception:
            return None

    @property
    def dimension(self) -> int:
        """Embedding dimensionality, or 0 if the engine is disabled."""
        if not self.enabled or self._embeddings is None:
            return 0
        return int(self._embeddings.shape[1])

    @property
    def num_embeddings(self) -> int:
        """Number of stored tool embeddings, or 0 if disabled."""
        if not self.enabled or self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])

    # -----------------------------------------------------------------------
    # Static: one-time build (only called via CLI)
    # -----------------------------------------------------------------------

    @staticmethod
    def build_embeddings(
        tools: List[Dict],
        output_dir: Path,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Generate and persist tool embeddings using sentence-transformers.

        This method is intentionally NOT called at runtime.  It is designed
        to be run once via the CLI entry-point:

            python -m engine.embeddings --rebuild

        Produces two files in ``output_dir``:
          embeddings.npy         — float32 array, shape (N, D)
          embedding_index.json   — {"slug": row_index, ...}

        Parameters
        ----------
        tools : List[Dict]
            Full tool records from db.json.
        output_dir : Path
            Destination directory for the output files.
        model_name : str
            sentence-transformers model identifier.  Defaults to
            ``all-MiniLM-L6-v2`` which produces 384-D embeddings and weighs
            ~90 MB — a good balance of quality vs. size for this corpus.

        Raises
        ------
        ImportError
            If sentence-transformers or numpy are not installed.
        ValueError
            If ``tools`` is empty.
        """
        # Import here — sentence-transformers is a heavy optional dep
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required to rebuild embeddings.\n"
                "Install with: pip install sentence-transformers"
            ) from exc

        try:
            import numpy as np_local
        except ImportError as exc:
            raise ImportError(
                "numpy is required to rebuild embeddings.\n"
                "Install with: pip install numpy"
            ) from exc

        if not tools:
            raise ValueError("Cannot build embeddings: tools list is empty.")

        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        # Build texts and slug order
        slugs: List[str] = []
        texts: List[str] = []
        for tool in tools:
            slug = tool.get("slug", "").strip()
            if not slug:
                continue
            slugs.append(slug)
            texts.append(_build_tool_text(tool))

        if not slugs:
            raise ValueError(
                "Cannot build embeddings: no tools with valid 'slug' fields found."
            )

        print(f"Encoding {len(texts)} tools...")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=False,  # store raw; normalise at query time
            convert_to_numpy=True,
        ).astype("float32")

        print(f"Embedding shape: {embeddings.shape}  "
              f"({embeddings.nbytes / 1024:.1f} KB)")

        output_dir.mkdir(parents=True, exist_ok=True)
        emb_path = output_dir / "embeddings.npy"
        idx_path = output_dir / "embedding_index.json"

        np_local.save(str(emb_path), embeddings)
        print(f"Saved embeddings -> {emb_path}")

        index = {slug: idx for idx, slug in enumerate(slugs)}
        with open(idx_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)
        print(f"Saved index      -> {idx_path}")

        print("Done.")


# ===========================================================================
# CLI entry-point
# ===========================================================================

def _cli_rebuild(db_path: Path, output_dir: Path, model_name: str) -> None:
    """Load db.json and call build_embeddings."""
    if not db_path.exists():
        print(f"ERROR: db.json not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(db_path, "r", encoding="utf-8") as fh:
            tools = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse {db_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(tools, list):
        print(
            f"ERROR: Expected a JSON array in {db_path}, got {type(tools).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        EmbeddingEngine.build_embeddings(
            tools=tools,
            output_dir=output_dir,
            model_name=model_name,
        )
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def _cli_info(data_dir: Path, tools_path: Path) -> None:
    """Print current embedding status without rebuilding."""
    tools: List[Dict] = []
    if tools_path.exists():
        try:
            with open(tools_path, "r", encoding="utf-8") as fh:
                tools = json.load(fh)
        except Exception:
            pass

    engine = EmbeddingEngine(data_dir=data_dir, tools=tools)

    print("Embedding status")
    print(f"  numpy available    : {_NUMPY_AVAILABLE}")
    print(f"  engine enabled     : {engine.enabled}")
    print(f"  data directory     : {data_dir.resolve()}")
    print(f"  embeddings.npy     : {(data_dir / 'embeddings.npy').exists()}")
    print(f"  embedding_index    : {(data_dir / 'embedding_index.json').exists()}")
    if engine.enabled:
        print(f"  num_embeddings     : {engine.num_embeddings}")
        print(f"  dimension          : {engine.dimension}")


if __name__ == "__main__":
    import argparse

    # Resolve default paths relative to this file so the CLI works from any
    # working directory.
    _module_dir = Path(__file__).parent
    _repo_root = _module_dir.parent
    _default_db = _repo_root / "db.json"
    _default_out = _repo_root / "data"

    parser = argparse.ArgumentParser(
        description="Open Source Network — embedding management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Rebuild with default settings\n"
            "  python -m engine.embeddings --rebuild\n\n"
            "  # Rebuild with a specific model and paths\n"
            "  python -m engine.embeddings --rebuild \\\n"
            "      --model sentence-transformers/all-mpnet-base-v2 \\\n"
            "      --db ./db.json --out ./data\n\n"
            "  # Show current embedding status\n"
            "  python -m engine.embeddings --info\n"
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Generate embeddings.npy and embedding_index.json from db.json",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print current embedding status and exit",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help=(
            "sentence-transformers model to use for encoding "
            "(default: all-MiniLM-L6-v2)"
        ),
    )
    parser.add_argument(
        "--db",
        default=str(_default_db),
        metavar="PATH",
        help=f"Path to db.json (default: {_default_db})",
    )
    parser.add_argument(
        "--out",
        default=str(_default_out),
        metavar="DIR",
        help=f"Output directory for embedding files (default: {_default_out})",
    )

    args = parser.parse_args()

    if args.rebuild:
        _cli_rebuild(
            db_path=Path(args.db),
            output_dir=Path(args.out),
            model_name=args.model,
        )
    elif args.info:
        _cli_info(
            data_dir=Path(args.out),
            tools_path=Path(args.db),
        )
    else:
        parser.print_help()
        sys.exit(0)
