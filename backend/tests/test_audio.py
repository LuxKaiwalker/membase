"""Audio ingestion: an uploaded recording becomes searchable through its
transcript, produced by the SAME generation model that captions images
(gemma4:12b's native multimodal understanding -- no Whisper, no second
model, there is no VRAM for one; see CLAUDE.md "Environment facts").

The request shape and the format support here were established
empirically against Ollama 0.40.3, not from documentation -- see the long
comment above TRANSCRIBABLE_AUDIO_MIMES in app/ingestion/pipeline.py. The
two findings these tests lock in are that audio rides the existing
`images` field, and that only WAV/MP3/FLAC actually decode.

Mirrors tests/test_chunking.py's image-captioning tests, since
transcription deliberately reuses that exact structure and failure
tolerance.
"""

from app.ingestion.pipeline import (
    TRANSCRIBABLE_AUDIO_MIMES,
    create_capture,
    is_audio,
    is_untranscribed_audio,
    process_capture_fully,
)
from app.models import Attachment
from app.retrieval.pipeline import hybrid_search
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish
from tests.test_files import _OGG_BYTES, _wav_bytes


async def test_audio_is_transcribed_into_extracted_text(db):
    """The transcript goes into Attachment.extracted_text -- the same
    column PDF text lands in -- so it flows into searchable_text and
    chunking with no downstream changes at all."""
    llm = FakeLLMProvider(audio_transcript_response="Remember to book the dentist for Thursday.")
    capture = await create_capture_and_finish(
        db, llm, text="Voice memo from the car.", uploads=[("memo.wav", _wav_bytes())]
    )

    attachment = db.query(Attachment).filter(Attachment.capture_id == capture.id).one()
    assert attachment.extraction_status == "ok"
    assert attachment.extracted_text == "Remember to book the dentist for Thursday."
    assert capture.units[0].status == "ready"


async def test_audio_bytes_are_passed_in_the_images_field(db):
    """Finding (1): Ollama 0.40.3 reads audio from the message's existing
    `images` array. A separate `audio` field returns HTTP 200 while
    silently DISCARDING the media, so getting this wrong yields a
    confident, fully hallucinated transcript rather than an error."""
    data = _wav_bytes()
    llm = FakeLLMProvider(audio_transcript_response="a transcript")
    await create_capture_and_finish(db, llm, text="Memo.", uploads=[("memo.wav", data)])

    assert llm.images_received == [[data]]
    assert llm.calls.count("audio_transcript") == 1


async def test_transcript_makes_the_recording_searchable(db):
    """The whole point: nothing in a .wav filename says "dentist". Only
    the transcript makes the recording retrievable."""
    llm = FakeLLMProvider(audio_transcript_response="Remember to book the dentist for Thursday.")
    capture = await create_capture_and_finish(
        db, llm, text="Voice memo.", uploads=[("memo.wav", _wav_bytes())]
    )

    scored = await hybrid_search(db, llm, "dentist")
    assert scored and scored[0].unit.capture_id == capture.id


async def test_transcription_failure_leaves_recording_stored_but_untranscribed(db):
    """Best-effort, identical tolerance to image captioning: an LLMError
    -- which is also how an undecodable container surfaces, since
    Ollama's HTTP 400 becomes an LLMUnavailableError -- must not block
    the capture from forming. The unit still reaches "ready", and the
    original recording is still on disk and downloadable."""
    llm = FakeLLMProvider(fail="unavailable")
    capture = await create_capture(db, text="Memo.", uploads=[("memo.wav", _wav_bytes())])
    attachment = db.query(Attachment).filter(Attachment.capture_id == capture.id).one()
    assert attachment.extraction_status == "unsupported"

    await process_capture_fully(capture.id, llm=llm)
    db.refresh(attachment)
    assert attachment.extraction_status == "unsupported"
    assert attachment.extracted_text is None
    db.refresh(capture)
    assert capture.units[0].status == "ready"


async def test_undecodable_container_is_stored_without_attempting_transcription(db):
    """Finding (2): OGG/M4A/WebM return HTTP 400 "Failed to load image or
    audio file" from this Ollama version. Accepted and stored anyway (never
    lose the original), but transcribing them is a guaranteed-failing call,
    so none is made -- the file just keeps extraction_status
    "unsupported" and stays downloadable via the attachments file
    endpoint."""
    llm = FakeLLMProvider(audio_transcript_response="should never be produced")
    capture = await create_capture_and_finish(
        db, llm, text="Browser recording.", uploads=[("memo.ogg", _OGG_BYTES)]
    )

    attachment = db.query(Attachment).filter(Attachment.capture_id == capture.id).one()
    assert attachment.content_type == "audio/ogg"
    assert attachment.extraction_status == "unsupported"
    assert attachment.extracted_text is None
    assert "audio_transcript" not in llm.calls
    assert llm.images_received == []
    # Still a real, retrievable memory -- just via the user's own text.
    assert capture.units[0].status == "ready"


async def test_no_speech_sentinel_is_not_stored_as_a_transcript(db):
    """A pure 440Hz sine tone made the live model invent a fluent
    paragraph of dialogue instead of reporting silence, so the prompt
    asks for a NO_SPEECH sentinel and it must be treated as "nothing to
    store". Indexing a confabulated transcript as a memory is worse than
    having no transcript."""
    llm = FakeLLMProvider(audio_transcript_response="NO_SPEECH")
    capture = await create_capture_and_finish(
        db, llm, text="Ambient noise recording.", uploads=[("noise.wav", _wav_bytes())]
    )

    attachment = db.query(Attachment).filter(Attachment.capture_id == capture.id).one()
    assert attachment.extraction_status == "unsupported"
    assert attachment.extracted_text is None


async def test_no_speech_sentinel_matched_loosely(db):
    """Asked for "NO_SPEECH", the live model was observed replying
    "NO SPEECH" -- a sentinel that missed by one character would store the
    sentinel itself as the recording's searchable transcript."""
    for reply in ("NO SPEECH", "no_speech", "NO_SPEECH.", '"NO SPEECH"'):
        llm = FakeLLMProvider(audio_transcript_response=reply)
        capture = await create_capture_and_finish(
            db, llm, text="Ambient noise.", uploads=[("noise.wav", _wav_bytes())]
        )
        attachment = db.query(Attachment).filter(Attachment.capture_id == capture.id).one()
        assert attachment.extracted_text is None, reply


async def test_audio_transcript_response_accepts_a_list_for_per_file_transcripts(db):
    llm = FakeLLMProvider(audio_transcript_response=["first transcript", "second transcript"])
    capture = await create_capture_and_finish(
        db, llm, text="Two memos.",
        uploads=[("one.wav", _wav_bytes()), ("two.wav", _wav_bytes(seconds=0.2))],
    )

    attachments = (
        db.query(Attachment)
        .filter(Attachment.capture_id == capture.id)
        .order_by(Attachment.original_filename)
        .all()
    )
    transcripts = {a.original_filename: a.extracted_text for a in attachments}
    assert transcripts == {"one.wav": "first transcript", "two.wav": "second transcript"}


async def test_transcript_is_labeled_as_a_recording_in_the_analysis_prompt(db):
    """An unlabeled transcript would read as the user's own typed words,
    and -- because `visual_concepts` enrichment keys off the "Image N:"
    label -- its concrete nouns could be mined as things "visible in the
    image" of a memory that contains no image."""
    from app.ingestion.pipeline import _attachment_text_for_analysis

    llm = FakeLLMProvider(audio_transcript_response="the lake was calm")
    capture = await create_capture_and_finish(
        db, llm, text="Memo.", uploads=[("memo.wav", _wav_bytes())]
    )
    db.refresh(capture)

    assert "Audio recording 1 (transcript):" in _attachment_text_for_analysis(capture)


def test_transcribable_set_reflects_what_ollama_can_actually_decode():
    """Locks in the measured result so a later "let's also transcribe
    OGG/WebM" change has to re-measure rather than assume. WAV/MP3/FLAC
    returned HTTP 200 with a real transcript; OGG (Opus and Vorbis),
    M4A/AAC and WebM all returned HTTP 400."""
    assert "audio/mpeg" in TRANSCRIBABLE_AUDIO_MIMES
    assert "audio/flac" in TRANSCRIBABLE_AUDIO_MIMES
    # libmagic's spelling for a real WAV, so this is the one that matters.
    assert "audio/x-wav" in TRANSCRIBABLE_AUDIO_MIMES
    for undecodable in ("audio/ogg", "audio/mp4", "audio/x-m4a", "audio/webm", "video/webm"):
        assert undecodable not in TRANSCRIBABLE_AUDIO_MIMES


def test_is_audio_covers_the_webm_sniffing_quirk():
    """libmagic reports an audio-only WebM as "video/webm" and offers no
    way to tell it from a WebM with a video track, so is_audio() has to
    accept that mime for browser-recorded audio to work at all."""
    assert is_audio(Attachment(content_type="video/webm", extraction_status="unsupported"))
    assert is_audio(Attachment(content_type="audio/x-wav", extraction_status="unsupported"))
    assert not is_audio(Attachment(content_type="image/png", extraction_status="unsupported"))


def test_already_transcribed_audio_is_not_transcribed_again():
    assert not is_untranscribed_audio(
        Attachment(content_type="audio/x-wav", extraction_status="ok")
    )
    assert is_untranscribed_audio(
        Attachment(content_type="audio/x-wav", extraction_status="unsupported")
    )
