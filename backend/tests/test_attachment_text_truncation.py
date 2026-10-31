from types import SimpleNamespace

from app.ingestion.pipeline import (
    MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS,
    _attachment_text_for,
    _attachment_text_for_analysis,
)

# Regression test for a reported bug (see docs/user-todo.md): a capture
# with two image attachments -- the first with little to describe, the
# second a glasses prescription whose caption plus "Structured data:"
# block (SPH/CYL/AXS/PSM) together push the *combined* attachment text
# past MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS. The old flat
# `_attachment_text_for(capture)[:MAX_...]` slice spent the whole budget
# on "Image 1:" and dropped "Image 2:" (the one with the actual values)
# entirely, so clarify.generate_question() -- which never sees the image
# itself, only this text -- truthfully reported the second image's
# description as "cut off". These tests pin the fix: every attachment
# gets a fair share of the budget, and within an image's own share its
# "Structured data:" block is kept whole rather than truncated away.


def _image_attachment(extracted_text: str) -> SimpleNamespace:
    return SimpleNamespace(content_type="image/jpeg", extracted_text=extracted_text)


def test_budget_is_split_fairly_across_attachments_not_spent_on_the_first():
    short_caption = "A small receipt with mostly illegible values."
    long_caption = "A detailed description. " * 100  # long prose caption
    structured_block = "Structured data:\n- SPH: -2.50\n- CYL: -1.50\n- AXS: 173\n- PSM: A 0.00"
    capture = SimpleNamespace(
        attachments=[
            _image_attachment(short_caption),
            _image_attachment(f"{long_caption}\n\n{structured_block}"),
        ]
    )

    analysis_text = _attachment_text_for_analysis(capture)

    assert "Image 1:" in analysis_text
    assert "Image 2:" in analysis_text
    # The second image's exact prescription values must survive, not just
    # its label -- this is the actual bug: they used to be dropped
    # entirely because the first image's share consumed the whole budget.
    assert "SPH: -2.50" in analysis_text
    assert "AXS: 173" in analysis_text
    # Two parts, each capped at its own share of the budget, joined by
    # one "\n\n" separator.
    assert len(analysis_text) <= MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS + 2


def test_structured_data_block_survives_even_when_its_own_captions_long():
    # A single image whose free-text caption alone already exceeds what a
    # multi-attachment capture would give it -- the structured block
    # (the actual answer the clarification step needs) must still come
    # through whole rather than being cut off mid-value.
    long_caption = "Lots of scene detail here. " * 200
    structured_block = "Structured data:\n- SPH: -2.50\n- CYL: -1.50\n- AXS: 173\n- PSM: A 0.00"
    capture = SimpleNamespace(
        attachments=[
            _image_attachment("short"),
            _image_attachment(f"{long_caption}\n\n{structured_block}"),
        ]
    )

    analysis_text = _attachment_text_for_analysis(capture)

    assert structured_block in analysis_text


def test_full_text_for_search_index_remains_uncapped():
    long_caption = "word " * 1000
    capture = SimpleNamespace(attachments=[_image_attachment(long_caption)])

    full_text = _attachment_text_for(capture)

    assert len(full_text) > MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS


def test_single_attachment_still_gets_the_whole_budget():
    huge_text = "X" * 5000
    capture = SimpleNamespace(attachments=[SimpleNamespace(content_type="text/plain", extracted_text=huge_text)])

    analysis_text = _attachment_text_for_analysis(capture)

    assert analysis_text == "X" * MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS
