from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter()


@router.get("/")
async def create_page(request: Request):
    return templates.TemplateResponse(request, "create.html", {})


@router.get("/recall")
async def recall_page(request: Request):
    return templates.TemplateResponse(request, "recall.html", {})


@router.get("/captures")
async def captures_page(request: Request):
    return templates.TemplateResponse(request, "captures.html", {})


@router.get("/docs")
async def docs_page(request: Request):
    return templates.TemplateResponse(request, "docs.html", {})
