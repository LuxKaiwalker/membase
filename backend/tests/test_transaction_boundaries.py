"""The pipeline must not hold a transaction open across LLM calls.

app/db.py sets `idle_in_transaction_session_timeout=180000` on every
connection, on the stated assumption that the ingestion pipeline "makes
at most one LLM call per open transaction". That assumption is load-
bearing: when it broke, Postgres terminated the connection mid-capture
and the failure surfaced minutes later as an
`IdleInTransactionSessionTimeout` raised from an unrelated-looking
commit (see docs/DECISIONS.md "One LLM call per open transaction, in
process_capture_fully too", found seeding scripts/seed_demo.py).

The invariant is invisible in ordinary tests: the fake provider answers
instantly, so a transaction spanning five calls looks exactly like one
spanning none. This observes the engine's own BEGIN/COMMIT events and
the provider's calls on one timeline instead, so the shape is asserted
rather than the timing.
"""

import pytest
from sqlalchemy import event

from app.db import get_engine
from app.ingestion.pipeline import process_capture_fully
from tests.fakes import FakeLLMProvider


class _Timeline:
    """BEGIN / COMMIT / ROLLBACK / llm events, in the order they happen."""

    def __init__(self):
        self.events: list[str] = []

    def worst_window(self) -> int:
        """Most LLM calls seen inside a single open transaction."""
        worst = 0
        current = 0
        inside = False
        for e in self.events:
            if e == "begin":
                inside, current = True, 0
            elif e in ("commit", "rollback"):
                inside = False
                current = 0
            elif e == "llm" and inside:
                current += 1
                worst = max(worst, current)
        return worst


class _RecordingProvider(FakeLLMProvider):
    """FakeLLMProvider that notes every call on the shared timeline."""

    def __init__(self, timeline: _Timeline, **kwargs):
        super().__init__(**kwargs)
        self._timeline = timeline

    async def generate(self, *args, **kwargs):
        self._timeline.events.append("llm")
        return await super().generate(*args, **kwargs)

    async def generate_structured(self, *args, **kwargs):
        self._timeline.events.append("llm")
        return await super().generate_structured(*args, **kwargs)

    async def embed(self, texts):
        # Embedding is a model call over the network like any other, and
        # the comment in db.py counts it: the processing-lease pattern
        # exists precisely so it runs outside a transaction.
        self._timeline.events.append("llm")
        return await super().embed(texts)


@pytest.fixture
def timeline():
    tl = _Timeline()
    engine = get_engine()

    def on_begin(conn):
        tl.events.append("begin")

    def on_commit(conn):
        tl.events.append("commit")

    def on_rollback(conn):
        tl.events.append("rollback")

    event.listen(engine, "begin", on_begin)
    event.listen(engine, "commit", on_commit)
    event.listen(engine, "rollback", on_rollback)
    try:
        yield tl
    finally:
        event.remove(engine, "begin", on_begin)
        event.remove(engine, "commit", on_commit)
        event.remove(engine, "rollback", on_rollback)


def _tiny_jpeg() -> bytes:
    """A real (if minimal) JPEG.

    process_upload() validates that an image actually decodes, so a
    byte string that merely starts with the JPEG magic is rejected
    before any of this gets a chance to run.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 150, 180)).save(buf, format="JPEG")
    return buf.getvalue()


async def _run(db, llm, text, uploads=None):
    from app.ingestion.pipeline import create_capture

    capture = await create_capture(db, text=text, uploads=uploads or [])
    capture_id = capture.id
    # Closed deliberately: the `db` fixture's own session is not what is
    # under test here, and leaving it open adds its idle transaction to
    # the timeline (which is itself the bug scripts/seed_demo.py hit).
    db.commit()
    await process_capture_fully(capture_id, llm=llm)


@pytest.mark.anyio
async def test_text_capture_holds_no_transaction_across_two_llm_calls(db, timeline):
    llm = _RecordingProvider(timeline)
    await _run(db, llm, "Met Ana at the Thursday market and she mentioned her new job in Bern.")

    assert timeline.events.count("llm") >= 2, "expected a multi-call pipeline run"
    assert timeline.worst_window() <= 1, (
        "a transaction stayed open across several LLM calls: "
        f"{timeline.worst_window()} calls in one transaction. "
        "See docs/DECISIONS.md 'One LLM call per open transaction'."
    )


@pytest.mark.anyio
async def test_image_capture_holds_no_transaction_across_two_llm_calls(db, timeline):
    """The case that actually broke.

    An image adds a vision call and a long `attachment_text`, which is
    what pushed the pre-fix pipeline's single transaction past 180
    seconds.
    """
    llm = _RecordingProvider(
        timeline,
        image_caption_response="A flat lake with mountains behind it.\n\nStructured data:\n- Lake: Thun",
    )
    await _run(
        db,
        llm,
        "Photo from the lake trip in 2022.",
        uploads=[("lake.jpg", _tiny_jpeg())],
    )

    assert timeline.events.count("llm") >= 2
    assert timeline.worst_window() <= 1, (
        f"{timeline.worst_window()} LLM calls inside one transaction for an image capture"
    )

