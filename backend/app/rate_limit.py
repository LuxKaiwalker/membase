"""In-memory rate limiting for unauthenticated, security-sensitive
endpoints -- currently just POST /api/auth/login (see docs/DECISIONS.md
"Password login with per-device tokens" for why /login had none
before).

In-memory and per-process by design, not a general-purpose limiter: no
Redis, no DB table. scripts/run-prod.sh always runs this app with
exactly one worker (see its own comment for why -- the app already has
other process-local singletons that assume this), so there is exactly
one process to hold this state. A restart resets everyone's counters,
which is an acceptable loosening of the limit, not a security hole.
"""

import time
from collections import defaultdict

# 10 failed attempts per 15 minutes per source IP, then locked out
# until the oldest failure in the window ages out. A single trusted
# user mistyping a password a few times should never hit this; a
# script trying passwords in a loop will. This is a brute-force guard
# -- it only counts *failures*, and only per-IP.
_WINDOW_SECONDS = 15 * 60
_MAX_ATTEMPTS = 10

_failures: dict[str, list[float]] = defaultdict(list)

# Separate, coarser flood guard: at most 5 requests to /login per
# second *in total*, across every caller combined, checked before the
# per-IP guard above. Its job is "don't let anyone hammer this one
# endpoint", not "catch a determined attacker" -- every request counts
# here regardless of source IP or outcome, unlike the per-IP counter.
_GLOBAL_WINDOW_SECONDS = 1.0
_GLOBAL_MAX_REQUESTS = 5

_global_requests: list[float] = []


def seconds_until_allowed(key: str) -> float | None:
    """None if `key` may attempt a login right now; otherwise how many
    seconds until it may."""
    now = time.monotonic()
    attempts = _failures[key]
    attempts[:] = [t for t in attempts if now - t < _WINDOW_SECONDS]
    if len(attempts) < _MAX_ATTEMPTS:
        return None
    return _WINDOW_SECONDS - (now - attempts[0])


def record_failure(key: str) -> None:
    _failures[key].append(time.monotonic())


def record_success(key: str) -> None:
    _failures.pop(key, None)


def seconds_until_globally_allowed() -> float | None:
    """None if /login may be called right now; otherwise how many
    seconds until the global request budget has room again."""
    now = time.monotonic()
    _global_requests[:] = [t for t in _global_requests if now - t < _GLOBAL_WINDOW_SECONDS]
    if len(_global_requests) < _GLOBAL_MAX_REQUESTS:
        return None
    return _GLOBAL_WINDOW_SECONDS - (now - _global_requests[0])


def record_global_request() -> None:
    _global_requests.append(time.monotonic())


def reset_all() -> None:
    """Test-only: clear all tracked state between tests."""
    _failures.clear()
    _global_requests.clear()
