# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in Open Source Network, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Send a detailed report to: security@open-source-network.dev

Include in your report:
- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept, if available)
- Affected versions or endpoints
- Any suggested mitigations you have identified

We will acknowledge receipt within 48 hours and aim to provide an initial assessment within 7 days. We follow a 90-day coordinated disclosure timeline — we will work with you to resolve the issue before any public disclosure.

We do not currently operate a formal bug bounty program, but we will credit researchers in our release notes with your permission.

---

## Security Practices

### API Key Security

API keys authenticate callers to the tiered rate-limiting system.

- Keys are issued as 256-bit (32-byte) cryptographically random values, hex-encoded as 64-character strings.
- Raw keys are **never stored**. Only the SHA-256 hash of each key is persisted in the database.
- Keys are transmitted exclusively via the `X-API-Key` request header over HTTPS.
- Compromised keys can be revoked instantly via the `is_active` flag without any schema changes.
- Last-used timestamps are recorded to support anomaly detection (e.g. a key used from two continents simultaneously).

If you believe your API key has been compromised, revoke it immediately via your account dashboard and generate a replacement. Do not share keys in public repositories, issue trackers, or support tickets.

### Transport Security

All API traffic is served over HTTPS with TLS enforced at the Cloudflare edge.

- HTTP Strict Transport Security (HSTS) is set with a one-year max-age, including subdomains.
- TLS 1.0 and 1.1 are disabled at the edge. TLS 1.2 and TLS 1.3 are supported.
- The website enforces the same HSTS policy via Cloudflare Pages `_headers` configuration.

### Rate Limiting

Rate limiting protects the service against abuse and ensures fair access across all tiers.

| Tier       | Requests per hour |
|------------|-------------------|
| Anonymous  | 50                |
| Free       | 100               |
| Pro        | 1,000             |
| Team       | 5,000             |
| Enterprise | 10,000            |

Rate limits are enforced server-side in Cloudflare D1 using fixed one-hour windows. The current window, remaining quota, and reset time are returned in standard `RateLimit-*` response headers on every request.

Attempts to bypass rate limiting via key rotation, IP spoofing, or header manipulation are treated as abuse and may result in account suspension.

### Response Security Headers

Every API response includes the following headers:

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Content-Type-Options` | `nosniff` | Prevents MIME-type sniffing |
| `X-Frame-Options` | `DENY` | Blocks framing / clickjacking |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | Enforces HTTPS |
| `Content-Security-Policy` | `default-src 'none'` | Blocks all resource loads from API responses |
| `X-XSS-Protection` | `0` | Disables the legacy XSS auditor (superseded by CSP) |
| `Referrer-Policy` | `no-referrer` | Prevents URL leakage in referrer headers |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=(), payment=()` | Disables unused browser APIs |

### Input Handling

- All database queries use parameterised prepared statements (D1 prepared statements). String concatenation into SQL is not used anywhere in the codebase.
- Search queries are treated as untrusted input regardless of auth tier.
- Query strings are stored in usage analytics as-is, but are never executed or interpreted — they are OSS tool names, not user-supplied code.

---

## Data Handling

Open Source Network indexes publicly available open-source software metadata. The database contains no private or proprietary data.

**What we store:**

| Data | Location | Retention |
|------|----------|-----------|
| API key hash (SHA-256) | D1 database | Until key is deleted |
| Key tier and Stripe customer ID | D1 database | Until account is deleted |
| Email address (optional) | D1 database | Until account is deleted |
| Rate limit counters (key hash + hour window) | D1 database | 2 hours |
| Usage events (key hash, endpoint, query, response time) | D1 database | 90 days, then purged |

**What we do NOT store:**

- Raw API keys (only SHA-256 hashes are persisted)
- IP addresses in plaintext (anonymous rate limits use a hashed IP)
- User passwords (authentication is API-key only; no password login)
- Personal search history linked to identifiable individuals
- Any third-party tracking data

The OSS tool database itself contains only publicly available metadata: names, descriptions, GitHub URLs, license types, tags, and graph relationships between tools. No PII is present in the tool database.

---

## Dependency Management

Dependencies are pinned and audited as part of the development workflow:

- `npm audit` is run on every pull request for the API (TypeScript/Workers).
- `pip-audit` is run on every pull request for the Python MCP server.
- Critical and high severity dependency vulnerabilities block merge.
- Medium severity vulnerabilities are tracked and resolved within 30 days.

---

## Supported Versions

We support the current release only. Security patches are applied to the latest version; no backports are maintained.

---

## Scope

The following are **in scope** for vulnerability reports:

- Authentication bypass or API key enumeration
- Rate limit bypass (allows one tier to exceed another's quota)
- SQL injection or database access beyond intended queries
- Sensitive data disclosure (API key exposure, user data exposure)
- CORS misconfiguration allowing credential theft
- Security header bypass on the API or website

The following are **out of scope**:

- Vulnerabilities in third-party services (Cloudflare, Stripe)
- Denial of service via resource exhaustion (rate limiting is a business control, not a security vulnerability)
- Missing security headers on external third-party embeds
- Social engineering attacks against staff
- Physical security

---

## Compliance Posture

Open Source Network is not subject to HIPAA (no health data) or PCI-DSS (payment data is handled exclusively by Stripe; no cardholder data transits our systems). We apply GDPR data minimisation principles to all data we collect, regardless of jurisdictional requirements.
