import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.auth import list_sessions, login, logout, revoke_session
from app.auth import issue_token, require_auth, verify_password
from app.config import get_settings
from app.models import AuthToken
from app.schemas import LoginRequest

# Password login with per-device bearer tokens (see docs/DECISIONS.md
# "Password login with per-device tokens"). Calls route functions and
# the require_auth dependency directly, same pattern as
# tests/test_captures_api.py -- FastAPI route/dependency functions are
# plain callables; Depends() defaults just become ordinary arguments
# when not going through a real request.


def _bearer(raw_token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=raw_token)


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequest:
    """Stand-in for FastAPI's Request -- login() only ever reads
    .client.host off it (see app/api/auth.py), so that's all this
    needs. Each test gets its own IP by default so one test's failed
    attempts never rate-limit another's."""

    def __init__(self, host: str = "test-client") -> None:
        self.client = _FakeClient(host)


def _configured_password() -> str:
    return get_settings().membase_auth_password_file.read_text().strip()


def test_verify_password_accepts_the_configured_password():
    assert verify_password(_configured_password()) is True


def test_verify_password_rejects_anything_else():
    assert verify_password(_configured_password() + "x") is False


@pytest.mark.asyncio
async def test_login_with_correct_password_issues_a_usable_token(db):
    response = login(
        LoginRequest(password=_configured_password(), device_label="unit test"),
        _FakeRequest(),
        db,
    )

    assert response.token
    assert response.session.device_label == "unit test"
    # The returned token must actually authenticate.
    token_row = await require_auth(db, _bearer(response.token))
    assert token_row.id == response.session.id


@pytest.mark.asyncio
async def test_login_with_wrong_password_is_rejected(db):
    with pytest.raises(HTTPException) as exc_info:
        login(LoginRequest(password="definitely-wrong", device_label=""), _FakeRequest(), db)
    assert exc_info.value.status_code == 401


@pytest.fixture()
def _bypass_global_rate_limit(monkeypatch):
    """The global 5-req/s flood guard (see app/rate_limit.py) would
    otherwise trip inside tests that deliberately make many rapid
    login() calls to exercise the *per-IP* guard -- those are a
    different concern, tested separately below, so bypass it here."""
    import app.api.auth as auth_api

    monkeypatch.setattr(auth_api, "seconds_until_globally_allowed", lambda: None)
    monkeypatch.setattr(auth_api, "record_global_request", lambda: None)


def test_login_is_rate_limited_per_ip_after_repeated_failures(db, _bypass_global_rate_limit):
    request = _FakeRequest(host="1.2.3.4")
    for _ in range(10):
        with pytest.raises(HTTPException) as exc_info:
            login(LoginRequest(password="wrong", device_label=""), request, db)
        assert exc_info.value.status_code == 401

    # 11th attempt in the window is locked out even with the right password.
    with pytest.raises(HTTPException) as exc_info:
        login(LoginRequest(password=_configured_password(), device_label=""), request, db)
    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers

    # A different source IP is unaffected.
    other = login(
        LoginRequest(password=_configured_password(), device_label="other device"),
        _FakeRequest(host="5.6.7.8"),
        db,
    )
    assert other.token


def test_successful_login_clears_the_failure_count(db, _bypass_global_rate_limit):
    request = _FakeRequest(host="9.9.9.9")
    for _ in range(6):
        with pytest.raises(HTTPException) as exc_info:
            login(LoginRequest(password="wrong", device_label=""), request, db)
        assert exc_info.value.status_code == 401

    # One correct login resets the counter, so this same IP can rack up
    # another run of failures without being locked out from the first
    # run carrying over (6 + 9 > the limit of 10, so this only passes
    # if the counter actually reset).
    login(LoginRequest(password=_configured_password(), device_label=""), request, db)
    for _ in range(9):
        with pytest.raises(HTTPException) as exc_info:
            login(LoginRequest(password="wrong", device_label=""), request, db)
        assert exc_info.value.status_code == 401


def test_global_rate_limit_caps_requests_per_second_across_all_ips(db):
    # First 5 requests (spread across distinct IPs, to isolate from
    # the per-IP guard) fit the global budget and reach the password
    # check; the next 5 -- still distinct IPs -- are throttled purely
    # by the global guard, proving it isn't per-IP.
    for i in range(10):
        with pytest.raises(HTTPException) as exc_info:
            login(LoginRequest(password="wrong", device_label=""), _FakeRequest(host=f"10.0.0.{i}"), db)
        if i < 5:
            assert exc_info.value.status_code == 401
        else:
            assert exc_info.value.status_code == 429
            assert "Retry-After" in exc_info.value.headers


def test_global_rate_limit_resets_after_its_window(db, monkeypatch):
    import app.rate_limit as rate_limit

    clock = [1_000.0]
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: clock[0])

    for i in range(5):
        with pytest.raises(HTTPException):
            login(LoginRequest(password="wrong", device_label=""), _FakeRequest(host=f"172.16.0.{i}"), db)

    clock[0] += 1.1  # past the 1-second global window
    response = login(
        LoginRequest(password=_configured_password(), device_label=""), _FakeRequest(host="172.16.0.99"), db
    )
    assert response.token


@pytest.mark.asyncio
async def test_require_auth_rejects_missing_credentials(db):
    with pytest.raises(HTTPException) as exc_info:
        await require_auth(db, None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_auth_rejects_an_unknown_token(db):
    with pytest.raises(HTTPException) as exc_info:
        await require_auth(db, _bearer("not-a-real-token"))
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_auth_rejects_a_revoked_token(db):
    raw_token, token = issue_token(db, "doomed device")
    logout(current=token, db=db)

    with pytest.raises(HTTPException) as exc_info:
        await require_auth(db, _bearer(raw_token))
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_auth_updates_last_used_at(db):
    raw_token, token = issue_token(db, "my phone")
    assert token.last_used_at is None

    await require_auth(db, _bearer(raw_token))
    db.refresh(token)
    assert token.last_used_at is not None


@pytest.mark.asyncio
async def test_logout_revokes_only_the_current_token(db):
    _raw_a, token_a = issue_token(db, "device a")
    raw_b, token_b = issue_token(db, "device b")

    logout(current=token_a, db=db)

    assert token_a.revoked_at is not None
    assert token_b.revoked_at is None
    # device b's token still works.
    await require_auth(db, _bearer(raw_b))


@pytest.mark.asyncio
async def test_list_sessions_excludes_revoked_tokens(db):
    _raw_a, token_a = issue_token(db, "kept")
    raw_b, token_b = issue_token(db, "revoked")
    logout(current=token_b, db=db)

    sessions = list_sessions(current=token_a, db=db)

    labels = {s.device_label for s in sessions}
    assert "kept" in labels
    assert "revoked" not in labels


@pytest.mark.asyncio
async def test_revoke_session_by_id(db):
    _raw_a, token_a = issue_token(db, "caller")
    _raw_b, token_b = issue_token(db, "target")

    revoke_session(session_id=token_b.id, current=token_a, db=db)

    db.refresh(token_b)
    assert token_b.revoked_at is not None


@pytest.mark.asyncio
async def test_revoke_session_rejects_an_already_revoked_or_unknown_id(db):
    import uuid

    _raw_a, token_a = issue_token(db, "caller")

    with pytest.raises(HTTPException) as exc_info:
        revoke_session(session_id=uuid.uuid4(), current=token_a, db=db)
    assert exc_info.value.status_code == 404
