import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.auth import issue_token, require_auth, verify_password
from app.db import get_db
from app.models import AuthToken
from app.rate_limit import (
    LOGIN_LOCK,
    record_failure,
    record_global_request,
    record_success,
    seconds_until_allowed,
    seconds_until_globally_allowed,
)
from app.schemas import LoginRequest, LoginResponse, SessionOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)) -> LoginResponse:
    client_key = request.client.host if request.client else "unknown"

    # Both rate-limit checks and their corresponding record_*() calls
    # happen under one lock, making the whole "check, then act on what
    # we saw" sequence atomic across concurrently-running requests --
    # login() is a plain `def`, so FastAPI really does thread-pool
    # concurrent calls to it. Without this, several truly-simultaneous
    # requests could all pass a check before any of them records,
    # transiently exceeding either limit. See app/rate_limit.py.
    with LOGIN_LOCK:
        # Global flood guard first -- applies to every call regardless
        # of source IP or outcome, so it's checked ahead of, and
        # independent from, the per-IP brute-force guard below.
        global_wait = seconds_until_globally_allowed()
        if global_wait is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Try again shortly.",
                headers={"Retry-After": str(int(global_wait) + 1)},
            )
        record_global_request()

        # Keyed by source IP. Checked before the password comparison
        # so a locked-out caller can't use response timing to tell
        # "still locked out" apart from "wrong password".
        wait = seconds_until_allowed(client_key)
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many attempts. Try again later.",
                headers={"Retry-After": str(int(wait) + 1)},
            )
        password_ok = verify_password(payload.password)
        if password_ok:
            record_success(client_key)
        else:
            record_failure(client_key)

    if not password_ok:
        # Same message a missing/invalid bearer token gets (see
        # app/auth.py require_auth) -- never confirm or deny which part
        # of a failed attempt was wrong.
        raise HTTPException(status_code=401, detail="Invalid credentials")
    raw_token, token = issue_token(db, payload.device_label)
    return LoginResponse(token=raw_token, session=SessionOut.model_validate(token))


@router.post("/logout", status_code=204)
def logout(current: AuthToken = Depends(require_auth), db: Session = Depends(get_db)) -> None:
    # Revokes only the token used to authenticate *this* request --
    # other devices' sessions are untouched. Use DELETE /sessions/{id}
    # to revoke a different one.
    current.revoked_at = datetime.now(UTC)
    db.commit()


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(
    current: AuthToken = Depends(require_auth), db: Session = Depends(get_db)
) -> list[AuthToken]:
    return (
        db.query(AuthToken)
        .filter(AuthToken.revoked_at.is_(None))
        .order_by(AuthToken.created_at.desc())
        .all()
    )


@router.delete("/sessions/{session_id}", status_code=204)
def revoke_session(
    session_id: uuid.UUID,
    current: AuthToken = Depends(require_auth),
    db: Session = Depends(get_db),
) -> None:
    token = db.get(AuthToken, session_id)
    if token is None or token.revoked_at is not None:
        raise HTTPException(status_code=404, detail="Session not found")
    token.revoked_at = datetime.now(UTC)
    db.commit()
