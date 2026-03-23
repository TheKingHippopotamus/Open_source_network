-- =============================================================================
-- Open Source Network API — D1 Database Schema
-- =============================================================================
-- Security design principles applied:
--   1. Raw API keys are NEVER stored. Only SHA-256 hashes are persisted.
--   2. Tier CHECK constraint enforced at the DB layer — not just application layer.
--   3. is_active flag allows instant key revocation without schema changes.
--   4. Rate limit windows are hash-keyed — no way to reverse a key from the DB.
--   5. Usage events store only hashed key references — no PII, no raw identifiers.
--   6. Indices are explicit to prevent full-table scans under load (DoS vector).
-- =============================================================================

-- API Keys
-- Stores the SHA-256 hash of the API key, never the key itself.
-- The tier column is constrained at the DB layer as a second line of defense
-- after application-level validation.
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash    TEXT    PRIMARY KEY,
    tier        TEXT    NOT NULL DEFAULT 'free'
                        CHECK(tier IN ('free', 'pro', 'team', 'enterprise')),
    email       TEXT,
    stripe_customer_id  TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    last_used   TEXT,
    is_active   INTEGER NOT NULL DEFAULT 1
                        CHECK(is_active IN (0, 1))
);

-- Rate Limiting Windows
-- key_hash + window (e.g. "2026-03-23T14:00") forms a composite PK.
-- This prevents duplicate rows and makes atomic upsert safe.
-- count is a monotonic counter reset when a new window row is created.
CREATE TABLE IF NOT EXISTS rate_limits (
    key_hash    TEXT    NOT NULL,
    window      TEXT    NOT NULL,
    count       INTEGER NOT NULL DEFAULT 0 CHECK(count >= 0),
    PRIMARY KEY (key_hash, window)
);

-- Usage Events (append-only analytics log)
-- key_hash is nullable — unauthenticated/anonymous requests are tracked
-- without any identifying information.
-- query is stored for analytics purposes only — it contains OSS tool names,
-- never user PII. Treat as sensitive-internal classification.
-- response_time_ms allows latency SLO monitoring without a separate APM service.
CREATE TABLE IF NOT EXISTS usage_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash            TEXT,
    endpoint            TEXT    NOT NULL,
    query               TEXT,
    timestamp           TEXT    NOT NULL DEFAULT (datetime('now')),
    response_time_ms    INTEGER CHECK(response_time_ms IS NULL OR response_time_ms >= 0)
);

-- =============================================================================
-- Indices
-- Without these, rate limit checks and usage queries become full-table scans
-- that degrade linearly under load — a trivially exploitable DoS surface.
-- =============================================================================

-- Lookup active keys by hash (auth hot path — must be O(1))
CREATE INDEX IF NOT EXISTS idx_api_keys_active
    ON api_keys(key_hash, is_active);

-- Cleanup job: find rate limit rows older than the current window
CREATE INDEX IF NOT EXISTS idx_rate_limits_window
    ON rate_limits(window);

-- Usage analytics: time-range queries for billing and SLO monitoring
CREATE INDEX IF NOT EXISTS idx_usage_timestamp
    ON usage_events(timestamp);

-- Usage analytics: per-endpoint breakdown
CREATE INDEX IF NOT EXISTS idx_usage_endpoint
    ON usage_events(endpoint);

-- Usage analytics: per-key billing aggregation
CREATE INDEX IF NOT EXISTS idx_usage_key_hash
    ON usage_events(key_hash);
