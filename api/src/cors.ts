/**
 * Open Source Network API — CORS & Security Headers
 * ================================================
 * Applied to every response by the Worker fetch handler.
 *
 * Security header rationale (per header):
 *
 *   X-Content-Type-Options: nosniff
 *     Prevents browsers from MIME-sniffing responses away from the declared
 *     Content-Type. Mitigates polyglot content attacks where an attacker
 *     uploads a file that a browser might execute as script.
 *     CWE-116 / OWASP A05.
 *
 *   X-Frame-Options: DENY
 *     Blocks the API responses from being framed in an <iframe>.
 *     Prevents clickjacking attacks against any browser-based clients.
 *     Superseded by CSP frame-ancestors for modern browsers, but retained
 *     for IE11/legacy compatibility.
 *     CWE-1021 / OWASP A05.
 *
 *   Strict-Transport-Security: max-age=31536000; includeSubDomains
 *     Forces HTTPS for 1 year (including subdomains). Prevents SSL stripping
 *     attacks on first visit after the initial HSTS response is cached.
 *     max-age=31536000 (1 year) is the industry standard minimum.
 *     preload is intentionally omitted until the domain is confirmed stable —
 *     adding preload commits to HTTPS permanently and is hard to undo.
 *     CWE-319 / OWASP A02.
 *
 *   Content-Security-Policy: default-src 'none'
 *     The API returns only JSON. It loads no scripts, styles, images, or
 *     frames. "default-src 'none'" is the strictest possible CSP — any
 *     attempt to interpret the response as a navigable document will have
 *     all resource loads blocked. Prevents XSS even if an attacker
 *     manages to inject HTML into a response body.
 *     CWE-79 / OWASP A03.
 *
 *   X-XSS-Protection: 0
 *     Disabled intentionally. The legacy XSS auditor (IE/early Chrome) has
 *     known bypasses and can itself be exploited to suppress legitimate
 *     content. Modern browsers ignore it; CSP above is the correct control.
 *     RFC-aligned security posture: remove broken controls, don't keep them.
 *
 *   Referrer-Policy: no-referrer
 *     API responses should not leak the caller's URL in Referer headers to
 *     any third-party resources (there are none, but defence-in-depth).
 *
 *   Permissions-Policy
 *     Explicitly disables all browser feature APIs. The API endpoint has
 *     no legitimate use for camera, microphone, geolocation, or payment APIs.
 *
 * CORS rationale:
 *   Access-Control-Allow-Origin: *
 *     This is a public API serving public OSS data. Credential-bearing
 *     cross-origin requests are disabled (no cookies, no withCredentials).
 *     Wildcard origin is appropriate and expected for public APIs.
 *
 *   Access-Control-Allow-Methods: GET, OPTIONS
 *     Write methods (POST, PUT, DELETE, PATCH) are not exposed. Restricting
 *     to GET+OPTIONS prevents CORS-based CSRF on mutation endpoints.
 *
 *   Access-Control-Allow-Headers: X-API-Key, Content-Type
 *     Explicit allowlist — only the headers the API actually reads. Prevents
 *     header injection through overly permissive CORS allowlists.
 *
 *   Access-Control-Max-Age: 86400
 *     Preflight cache of 24 hours reduces OPTIONS latency for repeat callers.
 *     Browsers cap this at 7200s (Chrome) or 86400s (Firefox); 86400 is the
 *     safe cross-browser maximum.
 */

// ---------------------------------------------------------------------------
// Security headers (applied to all responses)
// ---------------------------------------------------------------------------

export const SECURITY_HEADERS: Readonly<Record<string, string>> = {
    'X-Content-Type-Options':           'nosniff',
    'X-Frame-Options':                  'DENY',
    'Strict-Transport-Security':        'max-age=31536000; includeSubDomains',
    'Content-Security-Policy':          "default-src 'none'",
    // X-XSS-Protection intentionally omitted — legacy auditor is harmful,
    // CSP handles XSS. Setting to '0' explicitly disables the auditor where
    // still present.
    'X-XSS-Protection':                 '0',
    'Referrer-Policy':                  'no-referrer',
    'Permissions-Policy':               'camera=(), microphone=(), geolocation=(), payment=()',
} as const;

// ---------------------------------------------------------------------------
// CORS headers (applied to all responses + preflight)
// ---------------------------------------------------------------------------

export const CORS_HEADERS: Readonly<Record<string, string>> = {
    'Access-Control-Allow-Origin':      '*',
    'Access-Control-Allow-Methods':     'GET, OPTIONS',
    'Access-Control-Allow-Headers':     'X-API-Key, Content-Type',
    'Access-Control-Max-Age':           '86400',
} as const;

// ---------------------------------------------------------------------------
// Rate limit / auth headers (added per-response by the auth layer)
// ---------------------------------------------------------------------------

/**
 * Returns the standard rate-limit response headers given the current state.
 * Modelled on the IETF draft-ietf-httpapi-ratelimit-headers spec.
 *
 * RateLimit-Limit:     Maximum requests in the current window
 * RateLimit-Remaining: Requests left in the current window
 * RateLimit-Reset:     Unix timestamp when the window resets
 */
export function buildRateLimitHeaders(
    limit: number,
    remaining: number,
): Record<string, string> {
    // Reset at the top of the next hour
    const now = new Date();
    const resetMs =
        new Date(
            now.getFullYear(),
            now.getMonth(),
            now.getDate(),
            now.getHours() + 1, // next hour
            0,
            0,
            0,
        ).getTime() / 1000;

    return {
        'RateLimit-Limit':     String(limit),
        'RateLimit-Remaining': String(Math.max(0, remaining)),
        'RateLimit-Reset':     String(Math.floor(resetMs)),
    };
}

// ---------------------------------------------------------------------------
// Response builder helpers
// ---------------------------------------------------------------------------

/**
 * Merges security + CORS + optional extra headers into a Headers object.
 * Use this as the single source of truth for header composition — never
 * construct response headers ad-hoc in route handlers.
 */
export function buildResponseHeaders(
    extra: Record<string, string> = {},
    contentType = 'application/json',
): Headers {
    const headers = new Headers({
        'Content-Type': contentType,
        ...SECURITY_HEADERS,
        ...CORS_HEADERS,
        ...extra,
    });
    return headers;
}

/**
 * Handles CORS preflight (OPTIONS) requests.
 * Returns 204 No Content with full CORS + security headers.
 * Must be called before any auth checks — preflight requests do not
 * carry credentials and must always succeed.
 */
export function handlePreflight(): Response {
    return new Response(null, {
        status: 204,
        headers: buildResponseHeaders(),
    });
}

/**
 * Builds a JSON error response with the correct status code and headers.
 * Centralising error responses ensures security headers are never omitted
 * on error paths — a common oversight that leaves error pages unprotected.
 */
export function errorResponse(
    status: number,
    message: string,
    extra: Record<string, string> = {},
): Response {
    return new Response(
        JSON.stringify({ error: message, status }),
        {
            status,
            headers: buildResponseHeaders(extra),
        },
    );
}

/**
 * Builds a successful JSON response.
 */
export function jsonResponse(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data: unknown,
    status = 200,
    extra: Record<string, string> = {},
): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: buildResponseHeaders(extra),
    });
}
