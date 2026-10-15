import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import captures, recall, units
from app.config import get_settings
from app.llm import get_llm_provider
from app.llm.base import LLMError
from app.web import routes as web_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="membase")

# Authentication is NOT handled in this app -- see docs/DECISIONS.md
# "Retire app-level auth for nginx Basic Auth". A reverse proxy
# (nginx) sitting in front of this process is expected to gate every
# request with HTTP Basic Auth before it ever reaches here; this app
# trusts whatever already reached it. Never bind this app directly to
# a public interface without that proxy in front -- see
# docs/DEPLOYMENT.md.
#
# CORS is still this app's job, not the proxy's: the frontend is
# published separately (GitHub Pages), a different origin from
# wherever this backend is reached, so cross-origin fetch() calls need
# an explicit allow-list. Only the configured origin is allowed; no
# wildcard, and no credentials mode (Basic Auth credentials are sent
# as an explicit header by the frontend's own JS, not as a browser-
# managed cookie, so `allow_credentials` is correctly left False).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().frontend_origin],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(captures.router)
app.include_router(units.router)
app.include_router(recall.router)
app.include_router(web_routes.router)

_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    # Safety net: ingestion/retrieval pipelines already catch LLMError and
    # degrade gracefully, so this should rarely fire. Never log the request
    # body (may contain raw personal memory text) -- just the failure type.
    logger.warning("Unhandled LLM failure on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "The local LLM is temporarily unavailable. Please try again."},
    )


@app.on_event("shutdown")
async def _close_llm_provider() -> None:
    # get_llm_provider() is @lru_cache'd -- this closes the same
    # singleton's httpx.AsyncClient connection pool the app has been
    # using the whole time, rather than leaving it to the process exit
    # to clean up. aclose() is OllamaProvider-specific, not part of
    # the LLMProvider interface (a test double has no need for it).
    aclose = getattr(get_llm_provider(), "aclose", None)
    if aclose is not None:
        await aclose()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
