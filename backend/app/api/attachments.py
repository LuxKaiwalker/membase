import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import Attachment

router = APIRouter(prefix="/api/attachments", tags=["attachments"])


@router.get("/{attachment_id}/file")
def get_attachment_file(attachment_id: uuid.UUID, db: Session = Depends(get_db)):
    """Serve an attachment's original bytes -- the one place in the API
    that does, so the Store/Browse/Recall UIs can offer a view/download
    link for an image or document a memory came from (see
    docs/DECISIONS.md "Attachment view/download").

    `stored_path` is always server-generated (a uuid-prefixed path under
    settings.uploads_dir, never derived from user input -- see
    app/ingestion/files.py), so there's no path-traversal risk in
    joining it directly. `original_filename` IS user-supplied, so it's
    only ever used for the response's filename (via Path(...).name,
    defense-in-depth against a crafted value), never to build the
    on-disk path.
    """
    attachment = db.get(Attachment, attachment_id)
    if attachment is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    path = get_settings().uploads_dir / attachment.stored_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(
        path,
        media_type=attachment.content_type,
        filename=Path(attachment.original_filename).name,
    )
