from dataclasses import dataclass

# Simple deterministic fixed-size chunking, deliberately not a
# sophisticated document pipeline (spec section 27). PDFs chunk per page
# (further split only if a single page is unusually long); everything
# else chunks by a fixed character window with overlap so a fact split
# across a chunk boundary isn't lost entirely.
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150


@dataclass
class Chunk:
    text: str
    page_number: int | None


def _split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    # Not reachable with the module's own constants (max_chars > overlap
    # always holds there), but the sliding-window math below infinite-loops
    # for any caller that passes overlap >= max_chars (each step would
    # advance `start` by zero or backwards) -- fail fast instead.
    if overlap >= max_chars:
        raise ValueError(f"overlap ({overlap}) must be less than max_chars ({max_chars})")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start = end - overlap
    return pieces


def chunk_attachment(
    pages: list[str] | None, extracted_text: str | None, max_chunks: int | None = None
) -> list[Chunk]:
    """`pages` (per-page text) takes priority when available -- gives a
    real page_number per chunk. Falls back to chunking `extracted_text`
    as one undifferentiated stream (page_number=None) for non-paginated
    formats (txt/md).

    `max_chunks` bounds the *embedding* cost of a single huge attachment
    (see docs/DECISIONS.md "Large attachments") -- it truncates how many
    chunks get built, never the original file or its full extracted
    text, which stay completely intact and fully available to plain
    full-text search regardless. A chunk beyond the cap just never gets
    its own fine-grained, independently-embedded search entry.
    """
    if pages:
        chunks = []
        for page_index, page_text in enumerate(pages, start=1):
            for piece in _split_long_text(page_text):
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks
                chunks.append(Chunk(text=piece, page_number=page_index))
        return chunks
    if extracted_text:
        chunks = []
        for piece in _split_long_text(extracted_text):
            if max_chunks is not None and len(chunks) >= max_chunks:
                break
            chunks.append(Chunk(text=piece, page_number=None))
        return chunks
    return []
