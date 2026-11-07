import datetime as dt

from sqlalchemy import func, select

from app.config import get_settings
from app.ingestion.pipeline import delete_capture
from app.memory import graph as graph_module
from app.memory.graph import (
    ASSOC_SAME_CAPTURE,
    ASSOC_SHARED_ENTITY,
    ASSOC_STRUCTURER,
    SAME_CAPTURE_WEIGHT,
    STRUCTURER_LINK_WEIGHT,
    ProjectionResult,
    _ordered_pair,
    entity_bridges,
    entity_specificity,
    mentions_for,
    neighbors,
    project_unit,
    project_unit_safely,
    subgraph,
)
from app.models import (
    ORIGIN_EXPLICIT,
    ORIGIN_EXTRACTED,
    ORIGIN_INFERRED,
    Capture,
    EntityRelation,
    MemoryEntity,
    MemoryUnit,
    UnitAssociation,
    UnitEntityLink,
)
from app.retrieval.pipeline import hybrid_search
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish_unstructured

# The graph is a *projection* over MemoryUnit.semantic (see
# app/memory/graph.py): deterministic, LLM-free, idempotent. Everything here
# therefore builds the projection's input directly rather than paying for even
# a fake ingestion round-trip -- except the handful of tests that specifically
# assert the pipeline and the status gate hold end to end.


def _ready_unit(db, *, text, semantic=None, explicit_metadata=None, capture=None, status="ready"):
    """A capture plus one unit, written straight to the DB."""
    metadata = explicit_metadata or {}
    if capture is None:
        capture = Capture(raw_text=text, explicit_metadata=metadata)
        db.add(capture)
        db.flush()
    unit = MemoryUnit(
        capture_id=capture.id,
        authoritative_content=text,
        status=status,
        semantic=semantic or {},
        explicit_metadata=metadata,
        retrieval_text=text,
        searchable_text=text,
    )
    db.add(unit)
    db.commit()
    db.refresh(unit)
    return unit


def _corpus_padding(db, count=3):
    """`count` extra ready memories that share nothing with anything.

    Association weight is driven by `entity_specificity`, i.e. inverse
    document frequency: in a two-memory corpus a shared entity appears in
    100% of it and scores exactly 0.0 by construction. Padding the corpus is
    what makes a shared entity *distinctive* rather than universal, so this
    is setup for the behaviour under test, not scaffolding around it.
    """
    return [
        _ready_unit(db, text=f"unrelated memory {i}", semantic={"topics": [f"filler-topic-{i}"]})
        for i in range(count)
    ]


def _triples(mentions):
    return {(m.kind, m.surface, m.role) for m in mentions}


def _associate(db, unit_a, unit_b, *, kind=ASSOC_SHARED_ENTITY, weight=0.6, explanation="Both mention Anna (person)."):
    """Ensure an association of this kind exists between two units.

    Idempotent on purpose: several tests call `project_unit` first and then
    this, and projection may legitimately have created the same pair+kind
    already (two units sharing a distinctive entity is exactly what it writes).
    Blindly inserting would then violate uq_unit_associations_pair_kind and
    fail the test for a reason that has nothing to do with what it asserts.
    """
    source_id, target_id = _ordered_pair(unit_a.id, unit_b.id)
    existing = db.execute(
        select(UnitAssociation).where(
            UnitAssociation.source_unit_id == source_id,
            UnitAssociation.target_unit_id == target_id,
            UnitAssociation.kind == kind,
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.weight = weight
        existing.explanation = explanation
        db.commit()
        return
    db.add(
        UnitAssociation(
            source_unit_id=source_id,
            target_unit_id=target_id,
            kind=kind,
            origin=ORIGIN_INFERRED,
            weight=weight,
            shared_entity_ids=[],
            explanation=explanation,
        )
    )
    db.commit()


def _count(db, column, *conditions):
    query = select(func.count(column))
    for condition in conditions:
        query = query.where(condition)
    return int(db.execute(query).scalar() or 0)


def _links(db, unit_id):
    return list(
        db.execute(select(UnitEntityLink).where(UnitEntityLink.unit_id == unit_id)).scalars()
    )


def _entity(db, kind, normalized_name):
    return db.execute(
        select(MemoryEntity).where(
            MemoryEntity.kind == kind, MemoryEntity.normalized_name == normalized_name
        )
    ).scalar_one_or_none()


# --------------------------------------------------------------------------
# mentions_for: semantic fields -> graph nodes (pure function)
# --------------------------------------------------------------------------


def test_mentions_cover_every_typed_semantic_field_with_its_own_kind_and_role():
    semantic = {
        "people": ["Anna"],
        "organizations": ["ETH"],
        "locations": ["Lake Bled"],
        "topics": ["hiking"],
        "definitions": [{"term": "via ferrata", "definition": "a protected climbing route"}],
        "visual_concepts": ["kayak"],
        "relations": [{"subject": "Anna", "predicate": "recommended", "object": "via ferrata"}],
    }

    mentions = mentions_for(semantic)

    assert _triples(mentions) == {
        ("person", "Anna", "people"),
        ("organization", "ETH", "organizations"),
        ("location", "Lake Bled", "locations"),
        ("topic", "hiking", "topics"),
        ("concept", "via ferrata", "definition_term"),
        ("object", "kayak", "visual_concepts"),
        ("person", "Anna", "relation_subject"),
        ("concept", "via ferrata", "relation_object"),
    }
    # With no explicit_metadata, nothing can claim the user stated it.
    assert {m.origin for m in mentions} == {ORIGIN_EXTRACTED}
    definition_mention = next(m for m in mentions if m.role == "definition_term")
    assert definition_mention.excerpt == "a protected climbing route"
    relation_mention = next(m for m in mentions if m.role == "relation_object")
    assert relation_mention.predicate == "recommended"


def test_relation_endpoint_reuses_the_kind_a_typed_field_established():
    """A relation triple carries no kind information of its own. Without
    reusing the kind `people` already established, "Anna" the relation subject
    would become a *second* node ("concept") distinct from "Anna" the person --
    i.e. the one entity the memory is about, split in half."""
    semantic = {
        "people": ["Anna"],
        "relations": [{"subject": "Anna", "predicate": "recommended", "object": "a tactile sensing paper"}],
    }

    mentions = mentions_for(semantic)

    anna_kinds = {m.kind for m in mentions if m.surface == "Anna"}
    assert anna_kinds == {"person"}
    assert {m.role for m in mentions if m.surface == "Anna"} == {"people", "relation_subject"}


def test_an_unmatched_relation_endpoint_falls_back_to_concept():
    semantic = {
        "people": ["Anna"],
        "relations": [{"subject": "Anna", "predicate": "met", "object": "Boris"}],
    }

    mentions = mentions_for(semantic)

    # "Boris" was named by no typed field, so the triple alone cannot say what
    # kind of thing he is -- "concept" is the honest fallback, not "person".
    assert ("concept", "Boris", "relation_object") in _triples(mentions)


def test_a_value_the_user_supplied_is_marked_explicit_and_a_model_only_one_extracted():
    semantic = {"locations": ["ETH", "Lake Bled"]}

    mentions = mentions_for(semantic, {"location": "ETH"})

    origins = {(m.surface, m.origin) for m in mentions}
    # The whole point of the origin column: at retrieval time a user-stated
    # location can be trusted over a model-inferred one.
    assert ("ETH", ORIGIN_EXPLICIT) in origins
    assert ("Lake Bled", ORIGIN_EXTRACTED) in origins


def test_explicit_metadata_only_marks_the_field_it_belongs_to():
    # "ETH" supplied as a location must not make the *topic* "ETH" explicit --
    # origin is per (value, field), not per value.
    semantic = {"topics": ["ETH"]}

    mentions = mentions_for(semantic, {"location": "ETH"})

    assert [m.origin for m in mentions] == [ORIGIN_EXTRACTED]


def test_a_pronoun_relation_subject_produces_no_node():
    semantic = {
        "people": ["Anna"],
        "relations": [{"subject": "I", "predicate": "met", "object": "Anna"}],
    }

    mentions = mentions_for(semantic)

    assert all(m.surface != "I" for m in mentions)
    assert ("person", "Anna", "relation_object") in _triples(mentions)


def test_two_spellings_of_one_name_in_one_field_collapse_to_a_single_mention():
    # The link table is unique on (unit, entity, role), so the same normalized
    # name twice in one field must not produce two mentions of the same role.
    semantic = {"people": ["Anna", "anna's", "  Anna  "]}

    mentions = mentions_for(semantic)

    assert len([m for m in mentions if m.role == "people"]) == 1


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------


def test_projecting_a_ready_unit_creates_entities_and_links(db):
    unit = _ready_unit(
        db,
        text="Hiking with Anna at Lake Bled.",
        semantic={"people": ["Anna"], "locations": ["Lake Bled"], "topics": ["hiking"]},
    )

    result = project_unit(db, unit)
    db.commit()

    assert _entity(db, "person", "anna") is not None
    assert _entity(db, "location", "lake bled") is not None
    assert _entity(db, "topic", "hiking") is not None
    links = _links(db, unit.id)
    assert {link.role for link in links} == {"people", "locations", "topics"}
    assert result.entity_links == 3
    # Role weights are a fixed documented ranking: a named person is a
    # stronger handle on a memory than a theme like "hiking".
    by_role = {link.role: link for link in links}
    assert by_role["people"].weight > by_role["topics"].weight
    # Provenance keeps the user's own spelling and the originating capture, so
    # a link can always be traced back to the words it came from.
    assert by_role["locations"].provenance["surface"] == "Lake Bled"
    assert by_role["locations"].provenance["capture_id"] == str(unit.capture_id)
    assert _entity(db, "person", "anna").mention_count == 1


def test_projecting_the_same_unit_twice_is_idempotent(db):
    unit = _ready_unit(
        db,
        text="Anna recommended a tactile sensing paper at ETH.",
        semantic={
            "people": ["Anna"],
            "organizations": ["ETH"],
            "topics": ["robotics"],
            "event_date": "2024-03-15",
            "relations": [{"subject": "Anna", "predicate": "recommended", "object": "tactile sensing"}],
        },
    )

    project_unit(db, unit)
    db.commit()
    first = (
        _count(db, UnitEntityLink.id),
        _count(db, EntityRelation.id),
        _count(db, MemoryEntity.id),
    )

    # Delete-then-rebuild is what makes it safe to call projection from
    # several pipeline stages without tracking what changed -- a second run
    # must neither duplicate rows nor hit a unique-constraint violation.
    project_unit(db, unit)
    db.commit()

    assert (
        _count(db, UnitEntityLink.id),
        _count(db, EntityRelation.id),
        _count(db, MemoryEntity.id),
    ) == first


def test_two_surfaces_that_fold_to_one_entity_produce_a_single_link(db):
    """Regression: two different spellings in the SAME field can resolve to one
    entity through alias folding, which violates
    uq_unit_entity_links_unit_entity_role unless the dedup key is the resolved
    entity rather than the surface string."""
    unit = _ready_unit(
        db,
        text="Reading up on glaciology.",
        semantic={
            "topics": ["Glaciology", "glaciology research"],
            "aliases": ["glaciology research"],
        },
    )

    project_unit(db, unit)
    db.commit()  # an IntegrityError here is the bug this guards

    topic_links = [link for link in _links(db, unit.id) if link.role == "topics"]
    assert len(topic_links) == 1
    assert _count(db, MemoryEntity.id, MemoryEntity.kind == "topic") == 1
    entity = _entity(db, "topic", "glaciology")
    assert entity is not None
    assert "glaciology research" in entity.aliases


def test_projecting_a_unit_that_is_no_longer_ready_removes_every_row_it_had(db):
    capture = Capture(raw_text="Anna studies robotics. Anna recommended a paper.")
    db.add(capture)
    db.flush()
    semantic = {
        "people": ["Anna"],
        "topics": ["robotics"],
        "relations": [{"subject": "Anna", "predicate": "studies", "object": "robotics"}],
    }
    sibling = _ready_unit(db, text="Anna studies robotics.", semantic=semantic, capture=capture)
    unit = _ready_unit(db, text="Anna recommended a paper.", semantic=semantic, capture=capture)
    project_unit(db, sibling)
    project_unit(db, unit)
    db.commit()
    assert _count(db, UnitEntityLink.id, UnitEntityLink.unit_id == unit.id) > 0
    assert _count(db, EntityRelation.id, EntityRelation.unit_id == unit.id) > 0
    assert _count(db, UnitAssociation.id, UnitAssociation.kind == ASSOC_SAME_CAPTURE) == 1

    # `awaiting_clarification` content is deliberately invisible to retrieval;
    # a graph node for it (or an association edge routing through it) would be
    # a backdoor around that status gate.
    unit.status = "awaiting_clarification"
    db.commit()
    result = project_unit(db, unit)
    db.commit()

    assert result == ProjectionResult()
    assert _count(db, UnitEntityLink.id, UnitEntityLink.unit_id == unit.id) == 0
    assert _count(db, EntityRelation.id, EntityRelation.unit_id == unit.id) == 0
    # Removal has to cover associations where the unit is the *target* too,
    # since a pair is stored once in whichever order the UUIDs sort.
    assert (
        _count(
            db,
            UnitAssociation.id,
            (UnitAssociation.source_unit_id == unit.id) | (UnitAssociation.target_unit_id == unit.id),
        )
        == 0
    )
    # The sibling that is still ready keeps its own rows.
    assert _count(db, UnitEntityLink.id, UnitEntityLink.unit_id == sibling.id) > 0


def test_projecting_a_discarded_unit_creates_nothing(db):
    unit = _ready_unit(
        db,
        text="Something the user cancelled.",
        semantic={"people": ["Anna"], "locations": ["Lake Bled"]},
        status="discarded",
    )

    result = project_unit(db, unit)
    db.commit()

    assert result == ProjectionResult()
    assert _count(db, UnitEntityLink.id) == 0
    assert _count(db, MemoryEntity.id) == 0


def test_relations_become_dated_edges_carrying_the_asserting_unit(db):
    unit = _ready_unit(
        db,
        text="Anna recommended a via ferrata route.",
        semantic={
            "people": ["Anna"],
            "definitions": [{"term": "via ferrata", "definition": "a protected climbing route"}],
            "event_date": "2024-03-15",
            "relations": [{"subject": "Anna", "predicate": "recommended", "object": "via ferrata"}],
        },
    )

    result = project_unit(db, unit)
    db.commit()

    assert result.relations == 1
    relation = db.execute(select(EntityRelation)).scalar_one()
    # A relation with no memory behind it would be a free-floating "fact";
    # unit_id is what keeps every edge attributable.
    assert relation.unit_id == unit.id
    assert relation.subject_entity_id == _entity(db, "person", "anna").id
    assert relation.object_entity_id == _entity(db, "concept", "via ferrata").id
    assert relation.predicate == "recommended"
    assert relation.normalized_predicate == "recommended"
    assert relation.origin == ORIGIN_EXTRACTED
    assert relation.occurred_on == dt.date(2024, 3, 15)
    assert relation.date_precision == "day"
    assert relation.provenance["event_date_raw"] == "2024-03-15"


def test_an_unparseable_event_date_leaves_the_relation_undated(db):
    unit = _ready_unit(
        db,
        text="Anna recommended a paper, I think last summer.",
        semantic={
            "people": ["Anna"],
            "topics": ["tactile sensing"],
            "event_date": "last summer",
            "relations": [{"subject": "Anna", "predicate": "recommended", "object": "tactile sensing"}],
        },
    )

    project_unit(db, unit)
    db.commit()

    relation = db.execute(select(EntityRelation)).scalar_one()
    # Never back-filled with created_at or a plausible guess: "unknown" with a
    # NULL date is the honest answer, and the only one a user can tell apart
    # from a real date.
    assert relation.occurred_on is None
    assert relation.date_precision == "unknown"
    assert relation.provenance["event_date_raw"] == "last summer"


def test_a_relation_with_a_pronoun_endpoint_creates_no_edge_but_keeps_the_mention(db):
    unit = _ready_unit(
        db,
        text="I met Anna.",
        semantic={
            "people": ["Anna"],
            "relations": [{"subject": "I", "predicate": "met", "object": "Anna"}],
        },
    )

    result = project_unit(db, unit)
    db.commit()

    # An edge needs two real endpoints; the usable half still contributes its
    # plain mention, so the memory is no less connected than it should be.
    assert result.relations == 0
    assert _count(db, EntityRelation.id) == 0
    roles = {link.role for link in _links(db, unit.id)}
    assert roles == {"people", "relation_object"}
    assert _count(db, MemoryEntity.id) == 1  # no node for "I"


# --------------------------------------------------------------------------
# Associations
# --------------------------------------------------------------------------


def test_two_units_sharing_a_distinctive_entity_get_an_explained_association(db):
    _corpus_padding(db)
    unit_a = _ready_unit(
        db, text="Hiking with Anna at Lake Bled.", semantic={"people": ["Anna"], "locations": ["Lake Bled"]}
    )
    unit_b = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})

    project_unit(db, unit_a)
    project_unit(db, unit_b)
    db.commit()

    association = db.execute(
        select(UnitAssociation).where(UnitAssociation.kind == ASSOC_SHARED_ENTITY)
    ).scalar_one()
    assert {association.source_unit_id, association.target_unit_id} == {unit_a.id, unit_b.id}
    assert association.origin == ORIGIN_INFERRED  # nobody said these are related
    assert association.weight >= get_settings().membase_graph_min_association_weight
    # The explanation is deterministic text, not an LLM call, so the UI can
    # answer "why are these connected?" without a round trip.
    assert "Anna" in association.explanation
    assert str(_entity(db, "person", "anna").id) in association.shared_entity_ids


def test_two_units_sharing_nothing_get_no_association(db):
    _corpus_padding(db)
    unit_a = _ready_unit(db, text="Hiking with Anna.", semantic={"people": ["Anna"]})
    unit_b = _ready_unit(db, text="Repotted the monstera.", semantic={"topics": ["gardening"]})

    project_unit(db, unit_a)
    project_unit(db, unit_b)
    db.commit()

    # A graph where everything is weakly connected to everything is the same
    # as no graph, just more expensive to query.
    assert _count(db, UnitAssociation.id) == 0


def test_an_association_is_stored_once_per_unordered_pair(db):
    _corpus_padding(db)
    unit_a = _ready_unit(db, text="Hiking with Anna.", semantic={"people": ["Anna"]})
    unit_b = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})

    project_unit(db, unit_a)
    project_unit(db, unit_b)
    project_unit(db, unit_a)  # the other side of an already-stored pair
    db.commit()

    assert _count(db, UnitAssociation.id, UnitAssociation.kind == ASSOC_SHARED_ENTITY) == 1
    association = db.execute(
        select(UnitAssociation).where(UnitAssociation.kind == ASSOC_SHARED_ENTITY)
    ).scalar_one()
    # Canonical ordering is what makes the unique constraint actually prevent
    # a duplicate reversed row.
    assert str(association.source_unit_id) < str(association.target_unit_id)


def test_sibling_units_from_one_capture_are_associated_as_extracted_not_inferred(db):
    capture = Capture(raw_text="Anna studies robotics. I repotted the monstera.")
    db.add(capture)
    db.flush()
    unit_a = _ready_unit(db, text="Anna studies robotics.", semantic={"people": ["Anna"]}, capture=capture)
    unit_b = _ready_unit(db, text="I repotted the monstera.", semantic={"topics": ["gardening"]}, capture=capture)

    project_unit(db, unit_a)
    project_unit(db, unit_b)
    db.commit()

    association = db.execute(
        select(UnitAssociation).where(UnitAssociation.kind == ASSOC_SAME_CAPTURE)
    ).scalar_one()
    assert {association.source_unit_id, association.target_unit_id} == {unit_a.id, unit_b.id}
    # "extracted", not "inferred": the user genuinely did write these in one
    # submission, so nothing here was guessed by the system.
    assert association.origin == ORIGIN_EXTRACTED
    assert association.weight == SAME_CAPTURE_WEIGHT


def test_related_unit_ids_is_projected_as_a_structurer_link_and_left_unchanged(db):
    unit_a = _ready_unit(db, text="Anna recommended a paper.", semantic={"topics": ["robotics"]})
    unit_b = _ready_unit(db, text="I read the paper Anna recommended.", semantic={"topics": ["reading"]})
    unit_b.related_unit_ids = [str(unit_a.id)]
    db.commit()

    project_unit(db, unit_b)
    db.commit()
    db.refresh(unit_b)

    association = db.execute(
        select(UnitAssociation).where(UnitAssociation.kind == ASSOC_STRUCTURER)
    ).scalar_one()
    assert {association.source_unit_id, association.target_unit_id} == {unit_a.id, unit_b.id}
    assert association.origin == ORIGIN_INFERRED
    assert association.weight == STRUCTURER_LINK_WEIGHT
    # The graph reads `related_unit_ids`; it must not become a second writer of
    # it, or the structurer's advisory list stops meaning what it says.
    assert unit_b.related_unit_ids == [str(unit_a.id)]


def test_a_structurer_link_to_a_unit_that_is_not_ready_is_not_projected(db):
    unit_a = _ready_unit(
        db, text="Half-finished note.", semantic={"topics": ["robotics"]}, status="awaiting_clarification"
    )
    unit_b = _ready_unit(db, text="I read the paper.", semantic={"topics": ["reading"]})
    unit_b.related_unit_ids = [str(unit_a.id), "not-a-uuid"]
    db.commit()

    project_unit(db, unit_b)
    db.commit()

    # Same status gate as everywhere else, plus a malformed id must be skipped
    # rather than raise out of a background projection.
    assert _count(db, UnitAssociation.id) == 0


def test_entity_specificity_discounts_entities_that_appear_everywhere(db):
    # df == total: the entity is in every memory, so sharing it says nothing.
    assert entity_specificity(200, 200) == 0.0
    # One memory out of 200: sharing it is almost the whole story.
    assert entity_specificity(1, 200) > 0.95
    # A corpus of one (or an entity nobody links) has no frequency information
    # to discount by, so nothing is discounted.
    assert entity_specificity(1, 1) == 1.0
    assert entity_specificity(1, 0) == 1.0
    assert entity_specificity(0, 50) == 1.0


# --------------------------------------------------------------------------
# Traversal
# --------------------------------------------------------------------------


def test_neighbors_returns_a_one_hop_associate_weaker_than_its_seed(db):
    seed = _ready_unit(db, text="Hiking with Anna.", semantic={"people": ["Anna"]})
    other = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    _associate(db, seed, other, weight=0.6)

    found = neighbors(db, {seed.id: 1.0})

    assert [a.unit.id for a in found] == [other.id]
    # An associate is evidence *about* a match, never as strong as the match
    # itself -- otherwise associative expansion would outrank direct retrieval.
    assert found[0].strength < 1.0
    assert found[0].hops == 1
    assert len(found[0].path) == 1
    assert found[0].why == "Both mention Anna (person)."


def test_neighbors_never_returns_a_seed_itself(db):
    seed_a = _ready_unit(db, text="Hiking with Anna.", semantic={"people": ["Anna"]})
    seed_b = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    _associate(db, seed_a, seed_b)

    found = neighbors(db, {seed_a.id: 1.0, seed_b.id: 0.9})

    # This answers "what else is connected to what I already found", not "what
    # did I find" -- a seed coming back would double-count it as evidence.
    assert found == []


def test_neighbors_never_returns_a_unit_that_is_not_ready(db):
    seed = _ready_unit(db, text="Hiking with Anna.", semantic={"people": ["Anna"]})
    hidden = _ready_unit(
        db, text="Half-finished note about Anna.", semantic={"people": ["Anna"]}, status="awaiting_clarification"
    )
    _associate(db, seed, hidden, weight=0.9)

    # The status gate has to hold through the graph as well as through
    # hybrid_search, or association traversal becomes a way around it.
    assert neighbors(db, {seed.id: 1.0}) == []


def test_neighbors_reaches_two_hops_with_the_whole_path_recorded(db):
    seed = _ready_unit(db, text="Hiking with Anna.", semantic={"people": ["Anna"]})
    middle = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    far = _ready_unit(db, text="I read the paper on the train.", semantic={"topics": ["reading"]})
    _associate(db, seed, middle, weight=0.8, explanation="Both mention Anna (person).")
    _associate(db, middle, far, weight=0.8, explanation="Both mention the paper (concept).")

    found = {a.unit.id: a for a in neighbors(db, {seed.id: 1.0})}

    assert set(found) == {middle.id, far.id}
    assert found[middle.id].hops == 1
    assert found[far.id].hops == 2
    # Per-hop decay: a memory two hops out must carry visibly less weight.
    assert found[far.id].strength < found[middle.id].strength
    assert len(found[far.id].path) == 2
    # The full path is kept so a retrieval result can explain its own
    # provenance back to a directly-matched memory.
    assert found[far.id].why == "Both mention Anna (person). -> Both mention the paper (concept)."


def test_min_strength_filters_out_a_path_that_has_decayed_too_far(db):
    seed = _ready_unit(db, text="Hiking with Anna.", semantic={"people": ["Anna"]})
    middle = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    far = _ready_unit(db, text="I read the paper on the train.", semantic={"topics": ["reading"]})
    _associate(db, seed, middle, weight=0.8)
    _associate(db, middle, far, weight=0.8)

    found = neighbors(db, {seed.id: 1.0}, min_strength=0.4)

    assert [a.unit.id for a in found] == [middle.id]


def test_entity_bridges_returns_the_entities_two_units_share(db):
    unit_a = _ready_unit(
        db,
        text="Hiking with Anna at Lake Bled.",
        semantic={"people": ["Anna"], "locations": ["Lake Bled"], "topics": ["hiking"]},
    )
    unit_b = _ready_unit(
        db,
        text="Kayaking with Anna at Lake Bled.",
        semantic={"people": ["Anna"], "locations": ["Lake Bled"], "topics": ["kayaking"]},
    )
    project_unit(db, unit_a)
    project_unit(db, unit_b)
    db.commit()

    bridges = entity_bridges(db, unit_a.id, unit_b.id)

    assert {e.display_name for e in bridges} == {"Anna", "Lake Bled"}


# --------------------------------------------------------------------------
# Snapshot for the explorer UI
# --------------------------------------------------------------------------


def test_subgraph_without_a_center_returns_prefixed_unit_and_entity_nodes_with_edges(db):
    unit_a = _ready_unit(
        db, text="Hiking with Anna at Lake Bled.", semantic={"people": ["Anna"], "locations": ["Lake Bled"]}
    )
    unit_b = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    project_unit(db, unit_a)
    project_unit(db, unit_b)
    db.commit()

    snapshot = subgraph(db)

    unit_nodes = {n.id for n in snapshot.nodes if n.node_type == "unit"}
    entity_nodes = {n.id for n in snapshot.nodes if n.node_type == "entity"}
    # Node ids are namespaced because the view is bipartite: memories and the
    # things they are about share one id space in the renderer.
    assert unit_nodes == {f"unit:{unit_a.id}", f"unit:{unit_b.id}"}
    assert f"entity:{_entity(db, 'person', 'anna').id}" in entity_nodes
    assert all(n.id.startswith(("unit:", "entity:")) for n in snapshot.nodes)
    mentions_edges = [e for e in snapshot.edges if e.edge_type == "mentions"]
    assert (f"unit:{unit_a.id}", f"entity:{_entity(db, 'location', 'lake bled').id}") in {
        (e.source, e.target) for e in mentions_edges
    }
    assert snapshot.total_units == 2
    assert snapshot.truncated is False  # nothing was dropped


def test_subgraph_reports_itself_truncated_when_the_node_cap_cuts_the_set(db):
    for i in range(3):
        unit = _ready_unit(db, text=f"Hiking with Anna, trip {i}.", semantic={"people": ["Anna"]})
        project_unit(db, unit)
    db.commit()

    snapshot = subgraph(db, max_nodes=1)

    assert len([n for n in snapshot.nodes if n.node_type == "unit"]) == 1
    # Saying so beats silently presenting a partial graph as the whole thing.
    assert snapshot.truncated is True
    assert snapshot.total_units == 3


def test_subgraph_centered_on_a_unit_returns_its_neighbourhood(db):
    center = _ready_unit(
        db, text="Hiking with Anna at Lake Bled.", semantic={"people": ["Anna"], "locations": ["Lake Bled"]}
    )
    associate = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    far_away = _ready_unit(db, text="Repotted the monstera.", semantic={"topics": ["gardening"]})
    project_unit(db, center)
    project_unit(db, associate)
    project_unit(db, far_away)
    db.commit()
    _associate(db, center, associate, weight=0.6)

    snapshot = subgraph(db, center_unit_id=center.id)

    unit_nodes = {n.id for n in snapshot.nodes if n.node_type == "unit"}
    assert unit_nodes == {f"unit:{center.id}", f"unit:{associate.id}"}
    assert f"unit:{far_away.id}" not in unit_nodes
    association_edges = [e for e in snapshot.edges if e.edge_type == "association"]
    assert len(association_edges) == 1  # one row per pair, not one per direction
    assert association_edges[0].kind == ASSOC_SHARED_ENTITY


def test_subgraph_centered_on_a_unit_that_is_not_ready_is_empty(db):
    hidden = _ready_unit(
        db, text="Half-finished note about Anna.", semantic={"people": ["Anna"]}, status="awaiting_clarification"
    )
    neighbour = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    _associate(db, hidden, neighbour)

    snapshot = subgraph(db, center_unit_id=hidden.id)

    # Centering the explorer on non-ready content would display it, which is
    # precisely what the status gate exists to prevent.
    assert snapshot.nodes == []
    assert snapshot.edges == []


def test_subgraph_renders_a_relation_edge_between_two_entities(db):
    unit = _ready_unit(
        db,
        text="Anna recommended a via ferrata route.",
        semantic={
            "people": ["Anna"],
            "definitions": [{"term": "via ferrata", "definition": "a protected climbing route"}],
            "relations": [{"subject": "Anna", "predicate": "recommended", "object": "via ferrata"}],
        },
    )
    project_unit(db, unit)
    db.commit()

    snapshot = subgraph(db)

    relation_edges = [e for e in snapshot.edges if e.edge_type == "relation"]
    assert len(relation_edges) == 1
    assert relation_edges[0].label == "recommended"
    assert relation_edges[0].source == f"entity:{_entity(db, 'person', 'anna').id}"
    assert relation_edges[0].origin == ORIGIN_EXTRACTED


# --------------------------------------------------------------------------
# Pipeline integration and failure containment
# --------------------------------------------------------------------------


async def test_the_fast_pipeline_projects_a_finished_capture_into_the_graph(db):
    llm = FakeLLMProvider(
        enrichment_response={
            "people": ["Anna"],
            "topics": ["robotics"],
            "relations": [{"subject": "Anna", "predicate": "studies", "object": "robotics"}],
        }
    )

    capture = await create_capture_and_finish_unstructured(db, llm, text="Anna studies robotics.")
    db.expire_all()  # projection ran in the background step's own session

    unit = capture.units[0]
    assert unit.status == "ready"
    assert _count(db, UnitEntityLink.id, UnitEntityLink.unit_id == unit.id) > 0
    assert _count(db, EntityRelation.id, EntityRelation.unit_id == unit.id) == 1


async def test_project_unit_safely_swallows_an_unexpected_error_and_leaves_the_session_usable(
    db, fake_llm, monkeypatch
):
    llm = FakeLLMProvider(enrichment_response={"people": ["Anna"], "topics": ["robotics"]})
    capture = await create_capture_and_finish_unstructured(db, llm, text="Anna recommended a robotics paper.")
    db.expire_all()
    unit = capture.units[0]
    links_before = _count(db, UnitEntityLink.id, UnitEntityLink.unit_id == unit.id)
    assert links_before > 0

    def boom(*args, **kwargs):
        raise RuntimeError("projection exploded")

    monkeypatch.setattr(graph_module, "mentions_for", boom)

    result = project_unit_safely(db, unit)

    # The graph is an index, not a memory: a bug here must never surface as a
    # failed ingestion of words the user already typed.
    assert result == ProjectionResult()
    # The savepoint is the point: the caller's session must still be usable,
    # not stuck in a pending-rollback state that fails the next statement.
    db.add(Capture(raw_text="written after the failed projection"))
    db.commit()
    assert _count(db, Capture.id) == 2
    # The failed projection's own deletes were rolled back with it, so the
    # unit's existing graph rows survive untouched.
    assert _count(db, UnitEntityLink.id, UnitEntityLink.unit_id == unit.id) == links_before

    db.refresh(unit)
    assert unit.status == "ready"
    scored = await hybrid_search(db, fake_llm, "robotics paper")
    assert unit.id in [s.unit.id for s in scored]


async def test_deleting_a_capture_cascades_every_graph_row_away(db):
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [{"content": "Anna studies robotics."}, {"content": "Anna recommended a paper."}]
        },
        enrichment_response={
            "people": ["Anna"],
            "topics": ["robotics"],
            "relations": [{"subject": "Anna", "predicate": "studies", "object": "robotics"}],
        },
    )
    capture = await create_capture_and_finish_unstructured(
        db, llm, text="Anna studies robotics. Anna recommended a paper."
    )
    db.expire_all()
    assert _count(db, UnitEntityLink.id) > 0
    assert _count(db, EntityRelation.id) > 0
    assert _count(db, UnitAssociation.id, UnitAssociation.kind == ASSOC_SAME_CAPTURE) == 1

    delete_capture(db, capture)

    # Explicit user deletion is the only way a memory goes away, and it has to
    # take every derived row with it -- a surviving relation would be an
    # assertion with no memory behind it.
    assert _count(db, UnitEntityLink.id) == 0
    assert _count(db, EntityRelation.id) == 0
    assert _count(db, UnitAssociation.id) == 0
    # Entity rows themselves are not user content and have no FK to units;
    # they are expected to survive as empty, unreferenced nodes.
    assert _count(db, MemoryEntity.id) > 0
