import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import attachments, captures, recall, units
from app.llm import get_llm_provider
from app.llm.base import LLMError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="membase")

# Neither authentication nor CORS is handled in this app -- both are
# nginx's job, in front of this process. See docs/DECISIONS.md "Retire
# app-level auth for nginx Basic Auth" and "Move CORS to nginx: an
# app-level CORSMiddleware can't cover a proxy-level rejection" (CORS
# moved here after nginx's own 401/429 responses -- generated without
# ever reaching this app -- turned out to need CORS headers too, which
# only the layer producing every response, not just the proxied ones,
# can guarantee). This app trusts whatever request already reached it.
# Never bind this app directly to a public interface without nginx in
# front -- see docs/DEPLOYMENT.md.
#
# This app is API-only -- no HTML page routes, no /static mount. The
# frontend is published separately to GitHub Pages (see
# scripts/build_frontend.py), which reads backend/app/templates/ and
# backend/app/static/ directly as source files at build time; this
# process never serves them over HTTP. Visiting the backend's own
# domain should 404 on anything that isn't under /api, not render a
# website -- see docs/DECISIONS.md "API-only backend: stop serving the
# frontend's own HTML/static files".

app.include_router(captures.router)
app.include_router(units.router)
app.include_router(recall.router)
app.include_router(attachments.router)


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
