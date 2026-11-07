from app.memory.entities import (
    display_name_for,
    is_usable_entity_name,
    normalize_name,
    valid_kind,
    is_usable_visual_concept,
)

# Entity resolution is a normalization function, not an LLM call (see the
# module docstring): every rule below is the actual identity key the graph
# joins on, so each of these is a behavioural contract, not a style check.


def test_normalize_name_casefolds():
    assert normalize_name("ETH") == "eth"
    # Casefold rather than lower() so non-ASCII pairs fold too.
    assert normalize_name("STRASSE") == normalize_name("strasse")


def test_normalize_name_strips_possessives():
    assert normalize_name("Anna's") == "anna"
    assert normalize_name("Anna's bike") == "anna bike"
    # The typographic apostrophe is what a phone keyboard actually produces.
    assert normalize_name("Anna’s") == "anna"


def test_normalize_name_strips_punctuation_but_keeps_hyphens():
    assert normalize_name("ETH, Zurich!") == "eth zurich"
    # Hyphens are part of real names, so they survive where other punctuation
    # does not -- "Jean-Luc" and "Jean Luc" are deliberately NOT the same key.
    assert normalize_name("Jean-Luc") == "jean-luc"
    assert normalize_name("Jean-Luc") != normalize_name("Jean Luc")


def test_normalize_name_collapses_whitespace():
    assert normalize_name("  Lake   Bled\n") == "lake bled"
    assert normalize_name("Lake\tBled") == "lake bled"


def test_normalize_name_strips_a_leading_determiner():
    # "the lake" and "a lake" are the same entity as "lake"; the determiner
    # carries no identity.
    assert normalize_name("the lake") == "lake"
    assert normalize_name("A Lake") == "lake"
    assert normalize_name("my bike") == "bike"
    assert normalize_name("their flat") == "flat"


def test_normalize_name_does_not_empty_out_a_bare_determiner():
    """The determiner strip requires something to follow it. Without that
    guard "the" normalizes to "" and every such mention collides on one
    empty-named entity node."""
    assert normalize_name("the") == "the"
    assert normalize_name("an") == "an"


def test_normalize_name_applies_unicode_nfkc():
    # Decomposed and composed forms of the same word must be one key, or the
    # same entity splits into two nodes depending on the input method.
    assert normalize_name("Café") == normalize_name("Café")
    assert normalize_name("ＥＴＨ") == "eth"  # fullwidth letters


def test_normalize_name_tolerates_a_non_string():
    # Comes straight out of model JSON, so a non-string must not raise inside
    # a background pipeline step.
    assert normalize_name(None) == ""
    assert normalize_name(42) == ""


def test_pronouns_and_placeholders_are_not_usable_entity_names():
    """"I" appears in nearly every first-person memory; as a node it would be
    the single highest-degree thing in the graph, connecting everything to
    everything and making every association score meaningless."""
    assert is_usable_entity_name("I") is False
    assert is_usable_entity_name("me") is False
    assert is_usable_entity_name("someone") is False
    assert is_usable_entity_name("it") is False
    assert is_usable_entity_name("they") is False
    assert is_usable_entity_name("unknown") is False


def test_single_characters_bare_numbers_and_empty_strings_are_not_usable():
    assert is_usable_entity_name("A") is False
    assert is_usable_entity_name("x") is False
    # Dates and measurements have their own representation; as entity nodes
    # they are pure noise.
    assert is_usable_entity_name("2024") is False
    assert is_usable_entity_name("3") is False
    assert is_usable_entity_name("") is False
    assert is_usable_entity_name("   ") is False


def test_real_names_are_usable():
    assert is_usable_entity_name("Anna") is True
    assert is_usable_entity_name("ETH") is True
    assert is_usable_entity_name("Lake Bled") is True
    # A number plus a word is still a name ("3 Peaks"); only *bare* quantities
    # are rejected.
    assert is_usable_entity_name("3 Peaks") is True


def test_display_name_preserves_capitalization_but_normalizes_whitespace():
    # Normalization deliberately discards "ETH" vs "eth" for matching; display
    # must not discard it too.
    assert display_name_for("  ETH   Zurich ") == "ETH Zurich"
    assert display_name_for("Anna's bike") == "Anna's bike"
    assert display_name_for("Lake\n Bled") == "Lake Bled"
    assert display_name_for(None) == ""


def test_valid_kind_accepts_only_the_closed_set():
    assert valid_kind("person") is True
    assert valid_kind("object") is True
    # The DB enforces the same set with a CHECK constraint, so an unknown kind
    # has to be rejected before it reaches an INSERT.
    assert valid_kind("animal") is False
    assert valid_kind("") is False


def test_a_visual_concept_describing_the_medium_is_not_usable():
    """An image's medium and rendering style are facts about the FILE, not about
    anything in the memory.

    They are shared by every image produced the same way, and because they land
    as kind `object` -- which IS an episode anchor -- they actively grouped
    unrelated memories: observed live, an episode membership was justified by
    two pictures sharing "flat-style 2D illustration" and "minimalist
    illustration", i.e. by having been drawn in the same style.
    """
    for rejected in (
        "flat-style 2D illustration",
        "minimalist illustration",
        "geometric style",
        "photograph",
        "a stylized drawing",
        "background",
        "colour palette",
    ):
        assert not is_usable_visual_concept(rejected), rejected


def test_a_visual_concept_naming_something_visible_is_usable():
    for accepted in ("snow-capped peaks", "wooden dock", "lake", "mountain range", "pine trees"):
        assert is_usable_visual_concept(accepted), accepted


def test_visual_concept_admission_is_stricter_than_general_entity_admission():
    """The two checks are deliberately different: "illustration" is a perfectly
    good entity name in a memory about an illustrator, and is only rejected in
    the one field where it describes the attachment rather than the content."""
    assert is_usable_entity_name("minimalist illustration")
    assert not is_usable_visual_concept("minimalist illustration")
