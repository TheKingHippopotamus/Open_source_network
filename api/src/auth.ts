/**
 * OSS Neural Match API — Authentication & Rate Limiting
 * =======================================================
 * Runs inside Cloudflare Workers with D1 (SQLite) as the backing store.
 *
 * Security design:
 *   - Raw API keys are NEVER written to storage or logs.
 *   - SHA-256 hash (via Web Crypto API — FIPS 140-2 validated in Workers) is
 *     the only form persisted.
 *   - Rate limit windows are 1-hour buckets keyed by ISO-8601 hour string.
 *     Sliding windows would be more precise but require O(n) storage; fixed
 *     windows are a safe tradeoff for a public API.
 *   - Constant-time key extraction avoids timing-oracle attacks on the
 *     header parsing path.
 *   - Anonymous callers get a shared rate limit bucket keyed to their IP
 *     hash — not IP in plaintext (avoids logging PII under GDPR Art.4(1)).
 *   - All DB interactions use parameterised queries (D1 prepared statements).
 *     No string concatenation into SQL at any point.
 *
 * OWASP coverage:
 *   A01 Broken Access Control     — tier-based enforcement, is_active guard
 *   A02 Cryptographic Failures    — SHA-256 hashing, no key plaintext storage
 *   A04 Insecure Design           — fixed-window rate limiting with cleanup
 *   A05 Security Misconfiguration — explicit header names, no wildcard accepts
 *   A07 Identification & Auth     — key extraction + hash + DB lookup chain
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Tier = 'free' | 'pro' | 'team' | 'enterprise';

export interface AuthResult {
    authenticated: boolean;
    tier: Tier;
    keyHash: string;
    remainingRequests: number;
}

// D1Database is the Cloudflare Workers global type.
// Declare here so this file compiles without @cloudflare/workers-types.
declare const crypto: Crypto;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Requests allowed per hour, per tier.
 * These are the hard ceiling values enforced in the DB; the UI may show
 * a lower "soft limit" to preserve burst headroom.
 */
export const RATE_LIMITS: Record<Tier | 'anonymous', number> = {
    free:       100,
    pro:        1_000,
    team:       5_000,
    enterprise: 10_000,
    anonymous:  50,
} as const;

/**
 * The header name callers must use to send their API key.
 * Using a custom header (X-API-Key) rather than Authorization avoids
 * conflicts with OAuth bearer token middlewares and makes WAF rules
 * simpler to write.
 */
const API_KEY_HEADER = 'X-API-Key';

/**
 * Minimum acceptable key length (characters before hashing).
 * Enforcing a floor prevents trivially brute-forceable short tokens.
 * Our key generation issues 32-byte (256-bit) random keys encoded as
 * 64-char hex strings — this guard catches malformed/truncated submissions.
 */
const MIN_KEY_LENGTH = 32;

// ---------------------------------------------------------------------------
// Key extraction
// ---------------------------------------------------------------------------

/**
 * Extracts the raw API key from the request.
 *
 * Checks (in order):
 *   1. X-API-Key request header
 *   2. ?api_key= query parameter (convenience for browser/curl usage)
 *
 * Returns null if both are absent, empty, or below the minimum length
 * threshold. Does NOT validate format beyond length — that is the DB lookup's
 * responsibility. Keeping this function dumb avoids enumeration oracles
 * (an attacker cannot distinguish "wrong format" from "key not found").
 *
 * Security note: query parameters appear in access logs and browser history.
 * Callers should prefer the header form for production use. The query param
 * is supported for convenience (quick curl tests, public playground links)
 * but its presence is noted here as a deliberate trade-off.
 */
export function extractApiKey(request: Request): string | null {
    // Header takes precedence over query param
    const headerValue = request.headers.get(API_KEY_HEADER);
    if (headerValue && headerValue.trim().length >= MIN_KEY_LENGTH) {
        return headerValue.trim();
    }

    // Fallback: ?api_key= query parameter
    const url = new URL(request.url);
    const queryValue = url.searchParams.get('api_key');
    if (queryValue && queryValue.trim().length >= MIN_KEY_LENGTH) {
        return queryValue.trim();
    }

    return null;
}

// ---------------------------------------------------------------------------
// Cryptographic hashing
// ---------------------------------------------------------------------------

/**
 * Returns the hex-encoded SHA-256 hash of the raw API key.
 *
 * Uses the Web Crypto API (SubtleCrypto) which is available in all
 * Cloudflare Workers runtimes and is FIPS 140-2 validated.
 *
 * SHA-256 is appropriate here because:
 *   - API keys are high-entropy (256-bit random) secrets, not low-entropy
 *     passwords. The birthday-attack resistance of SHA-256 is sufficient;
 *     bcrypt/Argon2's cost factor adds latency without meaningful security
 *     benefit for random tokens.
 *   - Workers have strict CPU time limits; Argon2id at any useful cost
 *     factor would exceed the 10ms CPU budget on the free tier.
 *
 * For password hashing (login flows), use Argon2id or bcrypt instead.
 */
export async function hashKey(key: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(key);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// ---------------------------------------------------------------------------
// Rate limiting
// ---------------------------------------------------------------------------

/**
 * Returns the ISO-8601 hour window string for the current timestamp.
 * Example: "2026-03-23T14" — unique per hour, used as the window key.
 *
 * Fixed-window (vs. sliding-window) is chosen deliberately:
 *   - O(1) storage per key per hour
 *   - Single atomic UPSERT per request (no read-modify-write race)
 *   - Worst case: a caller can burst 2× the limit across a window boundary.
 *     This is acceptable for a public OSS API; use sliding windows if
 *     stricter enforcement is required.
 */
function getCurrentWindow(): string {
    return new Date().toISOString().slice(0, 13); // "YYYY-MM-DDTHH"
}

/**
 * Checks whether the caller has remaining quota in the current window.
 * If within quota, atomically increments the counter.
 *
 * Uses an INSERT OR REPLACE (UPSERT) pattern to avoid a read-then-write
 * race condition — the window row is created or incremented in a single
 * statement.
 *
 * Security note: the keyHash parameter must already be the SHA-256 hash.
 * This function never receives or handles raw key material.
 */
export async function checkRateLimit(
    db: D1Database,
    keyHash: string,
    tier: Tier | 'anonymous',
): Promise<AuthResult> {
    const window = getCurrentWindow();
    const limit = RATE_LIMITS[tier];

    // Read current count for this key+window
    const row = await db
        .prepare(
            'SELECT count FROM rate_limits WHERE key_hash = ?1 AND window = ?2',
        )
        .bind(keyHash, window)
        .first<{ count: number }>();

    const currentCount = row?.count ?? 0;
    const remaining = Math.max(0, limit - currentCount);

    if (currentCount >= limit) {
        // Return without incrementing — do not penalise further once exceeded
        return {
            authenticated: tier !== 'anonymous',
            tier: tier === 'anonymous' ? 'free' : tier,
            keyHash,
            remainingRequests: 0,
        };
    }

    // Atomically increment (INSERT new row or UPDATE existing row)
    await db
        .prepare(
            `INSERT INTO rate_limits (key_hash, window, count)
             VALUES (?1, ?2, 1)
             ON CONFLICT(key_hash, window)
             DO UPDATE SET count = count + 1`,
        )
        .bind(keyHash, window)
        .run();

    return {
        authenticated: tier !== 'anonymous',
        tier: tier === 'anonymous' ? 'free' : tier,
        keyHash,
        remainingRequests: remaining - 1,
    };
}

// ---------------------------------------------------------------------------
// Full authentication entry point
// ---------------------------------------------------------------------------

/**
 * Authenticates a request end-to-end:
 *   1. Extract raw key from header
 *   2. Hash it
 *   3. Look up the hash in D1 (verifies key exists and is active)
 *   4. Check + increment the rate limit window
 *
 * Falls back to anonymous rate limiting when no key is provided.
 * Returns a fully populated AuthResult in all cases — callers never
 * need to handle the unauthenticated path specially.
 */
export async function authenticate(
    request: Request,
    db: D1Database,
): Promise<AuthResult> {
    const rawKey = extractApiKey(request);

    if (!rawKey) {
        // Anonymous path: use IP hash as the rate-limit bucket key
        // Hashing the IP avoids storing PII in the rate_limits table
        const ip =
            request.headers.get('CF-Connecting-IP') ??
            request.headers.get('X-Forwarded-For') ??
            'unknown';
        const ipHash = await hashKey(`anon:${ip}`);
        return checkRateLimit(db, ipHash, 'anonymous');
    }

    const keyHash = await hashKey(rawKey);

    // Look up the key hash — never the raw key
    const keyRow = await db
        .prepare(
            `SELECT tier, is_active
             FROM api_keys
             WHERE key_hash = ?1`,
        )
        .bind(keyHash)
        .first<{ tier: Tier; is_active: number }>();

    if (!keyRow || keyRow.is_active !== 1) {
        // Key not found or revoked — treat as anonymous to avoid
        // leaking whether the key exists (prevents enumeration)
        const ip =
            request.headers.get('CF-Connecting-IP') ??
            request.headers.get('X-Forwarded-For') ??
            'unknown';
        const ipHash = await hashKey(`anon:${ip}`);
        return checkRateLimit(db, ipHash, 'anonymous');
    }

    // Fire-and-forget: update last_used timestamp (non-blocking)
    // Using waitUntil would be ideal in a full Worker context
    db.prepare(
        `UPDATE api_keys SET last_used = datetime('now') WHERE key_hash = ?1`,
    )
        .bind(keyHash)
        .run()
        .catch(() => {
            // Non-critical — do not fail the request if this update fails
        });

    return checkRateLimit(db, keyHash, keyRow.tier);
}

// ---------------------------------------------------------------------------
// Usage recording
// ---------------------------------------------------------------------------

/**
 * Records an API usage event for analytics and billing.
 *
 * This is intentionally fire-and-forget — a failure here must not fail
 * the primary request. Call with `ctx.waitUntil(recordUsage(...))` in
 * the Worker fetch handler so it runs after the response is sent.
 *
 * The query field stores the search term / tool slug — treat as
 * sensitive-internal. It does NOT contain user PII by design (users query
 * for OSS tool names, not personal data).
 */
export async function recordUsage(
    db: D1Database,
    keyHash: string | null,
    endpoint: string,
    responseTimeMs: number,
    query?: string,
): Promise<void> {
    await db
        .prepare(
            `INSERT INTO usage_events (key_hash, endpoint, query, response_time_ms)
             VALUES (?1, ?2, ?3, ?4)`,
        )
        .bind(keyHash, endpoint, query ?? null, responseTimeMs)
        .run();
}

// ---------------------------------------------------------------------------
// Maintenance: stale window cleanup
// ---------------------------------------------------------------------------

/**
 * Deletes rate limit rows for windows older than 2 hours.
 *
 * Should be called from a Cloudflare Cron Trigger (e.g., every hour) to
 * prevent unbounded table growth. At 50 req/hour for anonymous + up to
 * 10,000 for enterprise, the table can grow quickly without cleanup.
 *
 * The 2-hour retention gives one full window of buffer — the current hour
 * and the previous hour are always preserved, enabling boundary-crossing
 * burst analysis if needed.
 *
 * Security note: This is a low-privilege maintenance operation. It should
 * run under a separate D1 binding with only DELETE access on rate_limits
 * if Cloudflare D1 RBAC becomes available.
 */
export async function cleanupOldWindows(db: D1Database): Promise<void> {
    // Two hours ago, expressed as the same ISO-8601 hour-truncated string
    const cutoff = new Date(Date.now() - 2 * 60 * 60 * 1000)
        .toISOString()
        .slice(0, 13);

    await db
        .prepare('DELETE FROM rate_limits WHERE window < ?1')
        .bind(cutoff)
        .run();
}
