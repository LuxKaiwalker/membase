import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import captures, recall, units
from app.llm.base import LLMError
from app.web import routes as web_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="membase")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
