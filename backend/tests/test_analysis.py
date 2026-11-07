from app.ingestion.analysis import build_retrieval_text, is_verbatim

# analysis.py is now just the shared substrate for the restored
# multi-call pipeline (segmentation.py/clarify.py/enrichment.py each
# have their own test file) -- what's left here is the verbatim safety
# net and the deterministic retrieval-text assembly, neither of which
# belongs to any one of those three calls.


def test_word_boundary_violating_candidate_rejected_as_non_verbatim():
    """A candidate that only matches by crossing word boundaries (e.g.
    chopped mid-word) is a character-substring of the source but not an
    actual excerpt of it -- is_verbatim() must reject it, not just check
    for "in" at the character level."""
    assert not is_verbatim("y dog ate foo", "my dog ate food happily")
    assert is_verbatim("dog ate food", "my dog ate food happily")


def test_build_retrieval_text_includes_all_present_fields():
    semantic = {
        "meaning": "A trip to Japan.",
        "aliases": ["Japan trip"],
        "people": ["Anna"],
        "organizations": [],
        "locations": ["Osaka"],
        "topics": ["travel"],
        "definitions": [{"term": "AHV", "definition": "Swiss social security number"}],
        "relations": [{"subject": "I", "predicate": "visited", "object": "Osaka"}],
        "event_date": "2026-01-01",
    }
    text = build_retrieval_text("My trip to Japan.", [], semantic)
    assert "My trip to Japan." in text
    assert "A trip to Japan." in text
    assert "Japan trip" in text
    assert "AHV = Swiss social security number" in text
    assert "I visited Osaka" in text
    assert "People: Anna" in text
    assert "Locations: Osaka" in text
    assert "Event date: 2026-01-01" in text


def test_build_retrieval_text_includes_visual_concepts():
    """An image's visible contents must be embedded AND full-text indexed,
    not only projected into the association graph -- otherwise a plain
    semantic search for "a lake and mountains" still misses the photo."""
    semantic = {
        "meaning": "A hike above a lake.",
        "aliases": [], "people": [], "organizations": [], "locations": [],
        "topics": ["hiking"], "definitions": [], "relations": [], "event_date": "",
        "visual_concepts": ["lake", "mountains", "snow"],
    }
    text = build_retrieval_text("That summer hike.", [], semantic)
    assert "Visible in image: lake, mountains, snow" in text


def test_build_retrieval_text_omits_visual_concepts_when_empty():
    semantic = {
        "meaning": "", "aliases": [], "people": [], "organizations": [], "locations": [],
        "topics": [], "definitions": [], "relations": [], "event_date": "",
        "visual_concepts": [],
    }
    assert "Visible in image" not in build_retrieval_text("Some content.", [], semantic)


def test_build_retrieval_text_includes_clarifications():
    semantic = {"meaning": "", "aliases": [], "people": [], "organizations": [], "locations": [], "topics": [], "definitions": [], "relations": [], "event_date": ""}
    clarifications = [{"question": "What is X?", "answer": "X is Y."}]
    text = build_retrieval_text("Some content.", clarifications, semantic)
    assert "Q: What is X?" in text
    assert "A: X is Y." in text
