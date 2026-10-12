"""Password login with per-device bearer tokens (see docs/DECISIONS.md
"Password login with per-device tokens"). This app has exactly one
user by design -- there is no username, no user table, just one shared
password read fresh from `Settings.membase_auth_password_file` on every
attempt (see verify_password) -- but a login issues a fresh,
independently-revocable token per device, so a lost phone or an old
browser session can be cut off without changing the password
everywhere else. This is the "necessary infrastructure" for multiple
devices (a future mobile app included) sharing one account: each
device logs in once, keeps its own token, and the backend already
serves any authenticated client identically -- no separate sync
protocol needed, since every device is just talking to the same shared
API and database.

Tokens are stored as a SHA-256 hash only (`AuthToken.token_hash`) -- the
raw token is returned to the client exactly once, at login, and never
persisted, logged, or shown again.
"""

import hashlib
import secrets
import datetime as dt

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import AuthToken

# No scheme docs/auto-redirect -- just extracts the Authorization header
# if present; absence or malformed-ness is handled explicitly below so
# the error message is the same "invalid credentials" either way (never
# hint at *why* a token didn't work).
_bearer = HTTPBearer(auto_error=False)


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


def generate_token() -> str:
    # 32 random bytes, URL-safe base64 -- plenty of entropy for a
    # bearer token that's never brute-forceable in any realistic sense;
    # no configurable length, this isn't a value worth exposing a
    # setting for.
    return secrets.token_urlsafe(32)


def verify_password(password: str) -> bool:
    # Read the file on every call, not once at startup -- the whole
    # point of a file-backed password (over the old env-var one) is
    # that editing it over SSH takes effect immediately. get_settings()
    # itself is @lru_cache'd, but that only caches the *path*, not the
    # file's contents.
    try:
        stored = get_settings().membase_auth_password_file.read_text().strip()
    except OSError:
        return False
    # An empty/missing file must never match an empty password --
    # that would turn "forgot to set a password" into "no password
    # required", the opposite of fail-closed.
    if not stored:
        return False
    # compare_digest, not `==` -- a plain equality check leaks how many
    # leading characters matched via response-time differences; this is
    # the one place that distinction actually matters, since this *is*
    # the password check.
    return secrets.compare_digest(password, stored)


def issue_token(db: Session, device_label: str) -> tuple[str, AuthToken]:
    """Call only after verify_password() has already succeeded. Returns
    the raw token (show it to the client now; it's unrecoverable after
    this) and the persisted row (id/created_at, for the client to know
    what it just created)."""
    raw_token = generate_token()
    token = AuthToken(token_hash=_hash_token(raw_token), device_label=device_label.strip()[:200])
    db.add(token)
    db.commit()
    db.refresh(token)
    return raw_token, token


async def require_auth(
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthToken:
    """FastAPI dependency -- apply to every router that serves this
    app's actual data (see app/main.py). Deliberately returns the
    AuthToken row (not just None/True) so a caller that wants to know
    *which* device made this request can -- nothing currently does, but
    the shape costs nothing and avoids a second lookup if that changes.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token_hash = _hash_token(credentials.credentials)
    token = db.query(AuthToken).filter(AuthToken.token_hash == token_hash).one_or_none()
    if token is None or token.revoked_at is not None:
        raise HTTPException(status_code=401, detail="Invalid or revoked token")
    token.last_used_at = dt.datetime.now(dt.UTC)
    db.commit()
    return token
