from types import SimpleNamespace

from app.ingestion.pipeline import (
    MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS,
    _allocate_part_budgets,
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


# --- budget allocation (_allocate_part_budgets) -------------------------
# An even split is only the right answer when every part wants at least
# its share. These pin the max-min fair rule that replaced it, including
# the case it deliberately does *not* fix.


def test_a_small_attachments_unused_share_goes_to_a_larger_one():
    # Even split would hand the 60-char caption 750 chars it cannot use
    # and cut the long part at 750; the surplus belongs to the part that
    # actually has more to say.
    budgets = _allocate_part_budgets([60, 3000], 1500)

    assert budgets == [60, 1440]
    assert sum(budgets) == 1500


def test_surplus_flows_through_several_rounds():
    # Two parts release surplus, and the second round's larger share
    # satisfies a third -- a single pass over an equal share would stop
    # after the first redistribution.
    budgets = _allocate_part_budgets([10, 20, 600, 5000], 1500)

    assert budgets == [10, 20, 600, 870]
    assert sum(budgets) == 1500


def test_no_surplus_anywhere_falls_back_to_an_even_split():
    # The real production capture from docs/user-todo.md: a 1299-char
    # first image (caption + "Structured data:" block) and an 866-char
    # second one, 2165 chars of demand against a 1500-char cap. Every
    # part already wants more than its share, so there is nothing to
    # redistribute and both get 750 -- redistribution cannot create
    # budget, only stop it being wasted. Fitting this capture whole is a
    # question about the size of the cap, not about how it is divided.
    budgets = _allocate_part_budgets([1299, 866], 1500)

    assert budgets == [750, 750]


def test_the_remainder_is_handed_out_rather_than_dropped():
    budgets = _allocate_part_budgets([1000, 1000, 1000], 1000)

    assert sorted(budgets) == [333, 333, 334]
    assert sum(budgets) == 1000


def test_every_attachment_keeps_at_least_one_character():
    # More attachments than the budget has characters. Overshooting the
    # cap slightly is the lesser evil: a part allocated zero vanishes
    # from the prompt with no indication it existed, which is the
    # original bug.
    budgets = _allocate_part_budgets([10] * 2000, MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS)

    assert min(budgets) >= 1
    assert len(budgets) == 2000


def test_allocation_stays_within_budget_whenever_it_can():
    for demands in ([5], [1, 2, 3], [900, 900], [1, 5000], [700] * 3, [1299, 866]):
        budgets = _allocate_part_budgets(demands, MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS)
        assert len(budgets) == len(demands)
        assert sum(budgets) <= MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS
        # Never allocate a part more than it asked for -- that would be
        # budget taken from a part that could have used it.
        assert all(b <= max(d, 1) for b, d in zip(budgets, demands))


def test_uneven_real_capture_shape_keeps_both_images_and_their_values():
    # Same shape as the production capture, but with the second image
    # short enough that redistribution has something to work with: the
    # first image's structured block must survive AND the second image's
    # values must be present in full.
    first = "Scene prose. " * 100 + "\n\nStructured data:\n- RIGHT SPH: -3.50\n- RIGHT CYL: -0.75"
    second = "A slip of paper reading SPH - 2.50 / CYL - 1.50 / AXS 173."
    capture = SimpleNamespace(
        attachments=[_image_attachment(first), _image_attachment(second)]
    )

    analysis_text = _attachment_text_for_analysis(capture)

    assert "RIGHT SPH: -3.50" in analysis_text
    assert second in analysis_text  # whole, not truncated to an even share
    assert len(analysis_text) <= MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS + 2
