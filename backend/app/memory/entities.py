"""Deterministic entity resolution: turning the strings in
`MemoryUnit.semantic` into stable `MemoryEntity` rows.

No LLM call. Resolution is a normalization function plus an alias table, which
buys three things the graph depends on:

1. **Free.** Ingestion already spends ~6 generation calls per capture (see
   docs/DECISIONS.md). The graph adds zero.
2. **Idempotent.** Re-projecting the same unit produces byte-identical links,
   so projection can be a delete-then-rebuild and a backfill can be re-run.
3. **Testable without a model.** Every rule below is a unit test, not a
   prompt that "usually" behaves.

The cost is honest and documented in docs/TODO.md: this will not merge "Anna"
with "Annalisa", nor a misspelling, nor "ETH" with "Eidgenössische Technische
Hochschule" unless some memory's `aliases` field happens to connect them.
Fuzzy or embedding-based entity resolution is a real improvement and a real
risk (wrongly merging two people is worse than failing to merge one), so it is
deliberately not attempted here.
"""

import re
import unicodedata

from app.models import ENTITY_KINDS

# Which semantic field a mention came from -> the entity kind it creates.
# `relations`' subject/object are handled separately (see graph.py): a
# relation's endpoints carry no kind information of their own, so they resolve
# against entities already established by the typed fields, and only fall back
# to "concept" when nothing matches.
FIELD_TO_KIND = {
    "people": "person",
    "organizations": "organization",
    "locations": "location",
    "topics": "topic",
    "definitions": "concept",
    "visual_concepts": "object",
}

_POSSESSIVE_RE = re.compile(r"'s\b|’s\b")
_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
# A parenthesised tail is a gloss, not part of the name. Observed against the
# live model, which routinely emits entities like "Narrator (User)",
# "User (Owner of the AHV number)" and
# "Mountains behind Lake Bled (Julian Alps region)" -- keeping the gloss made
# each of those a distinct node from the plain name, so the same real thing
# fragmented into several entities depending on how chatty the extraction was
# that time.
_PARENTHETICAL_RE = re.compile(r"\s*[\(\[][^)\]]*[\)\]]")

# Leading articles/determiners carry no identity ("the lake" and "a lake" are
# the same entity as "lake"). Stripped only when something follows, so the
# word "the" alone never normalizes to the empty string.
_LEADING_STOPWORDS = ("the ", "a ", "an ", "my ", "our ", "his ", "her ", "their ", "its ")

# Strings that are grammatically a reference to the user or to nobody in
# particular. They would otherwise become the single highest-degree node in the
# graph -- "I" appears in nearly every first-person memory -- connecting
# everything to everything and making association scores meaningless. Dropped
# at resolution time rather than filtered later, so they never reach the DB.
_PRONOUN_LIKE = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "you", "he", "she",
    "it", "they", "them", "him", "her", "someone", "somebody", "anyone",
    "anybody", "everyone", "no one", "nobody", "something", "anything",
    "nothing", "this", "that", "these", "those", "unknown", "n/a", "none",
    "null", "other", "others", "person", "people", "thing", "things", "self",
}

# How the extraction model refers to the person whose memory store this is,
# when the text said "I". Every one of these was observed for real against
# gemma4:12b on a four-memory corpus: "Narrator", "Narrator (User)", "User",
# "User (Owner of the AHV number)", "User within the Swiss social security
# framework".
#
# These must not become entities. The owner of the store is in *every*
# first-person memory, so a node for them is the highest-degree node in the
# graph by a wide margin -- it connects every memory to every other memory and
# makes association weights meaningless. (The document-frequency ceiling in
# graph.py is the second line of defence, but it only engages once the corpus
# is large, so this filter has to catch it first.)
#
# A name is rejected when its FIRST token is one of these, which handles the
# qualified forms ("User within the ...") that a whole-string match misses.
# The set is deliberately tiny and self-reference-specific: "owner" alone is
# not in it, because "Owner" could be a real role in a memory about someone
# else, while "narrator" and "user" are never how a person refers to a third
# party in their own notes.
_SELF_REFERENCE_HEADS = {"i", "me", "my", "myself", "we", "narrator", "user", "author", "speaker"}

# A single character, or a bare number, is never a usable entity name.
_MIN_NAME_CHARS = 2


def normalize_name(name: str) -> str:
    """The join key for entity identity.

    Unicode-NFKC, casefolded, possessives and punctuation stripped, whitespace
    collapsed, leading determiners removed. Hyphens are kept (they are part of
    real names like "Jean-Luc"); other punctuation is not.
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKC", name).strip()
    if not text:
        return ""
    text = text.casefold()
    without_gloss = _PARENTHETICAL_RE.sub("", text).strip()
    # Only drop the gloss if something is left: an entity whose entire name is
    # parenthesised ("(unnamed)") should fall through to the usability check
    # below rather than silently normalize to the empty string.
    if without_gloss:
        text = without_gloss
    text = _POSSESSIVE_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    for stop in _LEADING_STOPWORDS:
        if text.startswith(stop) and len(text) > len(stop):
            text = text[len(stop) :]
            break
    return text.strip()


def is_usable_entity_name(name: str) -> bool:
    """Would this string make a meaningful graph node?

    Rejects pronouns and generic placeholders (see `_PRONOUN_LIKE` for why),
    single characters, and bare numbers. A rejected name is simply not
    projected -- the underlying memory is untouched and still fully
    searchable through every existing path.
    """
    normalized = normalize_name(name)
    if len(normalized) < _MIN_NAME_CHARS:
        return False
    if normalized in _PRONOUN_LIKE:
        return False
    # Bare quantities ("2024", "3 kg") are dates or measurements, both of which
    # have their own representation; as entity nodes they only add noise.
    if not any(ch.isalpha() for ch in normalized):
        return False
    if normalized.split()[0] in _SELF_REFERENCE_HEADS:
        return False
    # A resolvable date is temporal information, not an entity. "July 2023"
    # arrived as a `definitions` term from the live model; as a node it would
    # sit alongside the same memory's own `event_date` saying the same thing,
    # except unusable for temporal comparison. Reusing the real parser (rather
    # than a second date regex here) means the two can never disagree about
    # what counts as a date.
    from app.memory.temporal import PRECISION_UNKNOWN, parse_event_date

    _parsed, precision = parse_event_date(normalized)
    if precision != PRECISION_UNKNOWN:
        return False
    return True


# Words describing the MEDIUM or RENDERING STYLE of an image rather than
# anything in it. A vision caption routinely volunteers these, and
# `semantic.visual_concepts` carries them through: the demonstration corpus
# produced "flat-style 2D illustration", "minimalist illustration" and
# "geometric style" alongside the genuinely useful "snow-capped peaks".
#
# As entities they are actively harmful, not merely noisy. They are shared by
# every image produced the same way, so they behave like the generic entities
# the document-frequency ceiling exists to suppress -- except they are of kind
# `object`, which IS an episode anchor. Observed live: an episode grouped two
# memories because they shared "flat-style 2D illustration" and "minimalist
# illustration", i.e. because the pictures were drawn in the same style. That
# is a fact about the files, not about the experience.
_MEDIUM_WORDS = {
    "illustration", "illustrations", "photograph", "photographs", "photo",
    "photos", "picture", "pictures", "image", "images", "drawing", "drawings",
    "painting", "paintings", "render", "rendering", "graphic", "graphics",
    "artwork", "sketch", "diagram", "screenshot", "style", "styled", "aesthetic",
    "palette", "composition", "foreground", "background", "closeup", "close-up",
}
# Style qualifiers that only ever modify a medium word.
_STYLE_WORDS = {
    "flat", "flat-style", "minimalist", "minimal", "geometric", "stylized",
    "stylised", "abstract", "cartoon", "cartoonish", "2d", "3d", "vector",
    "monochrome", "grayscale", "greyscale", "low-poly", "pixelated",
}


def is_usable_visual_concept(name: str) -> bool:
    """Would this visible-thing description make a meaningful entity?

    Stricter than `is_usable_entity_name`: additionally rejects descriptions of
    the image's medium or rendering style (see `_MEDIUM_WORDS`). A concept is
    rejected when ANY of its words is a medium word, or when every word is a
    style qualifier -- so "snow-capped peaks" and "wooden dock" survive while
    "minimalist illustration", "flat-style 2D illustration" and "geometric
    style" do not.

    Rejecting on any medium word is deliberately blunt. "photograph of a lake"
    loses the lake, which is a real cost -- but the lake almost always also
    appears as its own concept in the same list, and a wrongly-kept medium term
    pollutes the graph for every image in the store while a wrongly-dropped
    compound costs one redundant node.
    """
    if not is_usable_entity_name(name):
        return False
    words = set(normalize_name(name).split())
    if words & _MEDIUM_WORDS:
        return False
    return not words.issubset(_STYLE_WORDS)


def is_alias_of(alias: str, entity_name: str) -> bool:
    """Is `alias` a surface variant of `entity_name`, conservatively?

    True only when one name's words contain the other's as a contiguous
    whole-word run -- "ETH" vs "ETH Zurich" yes, "Markus" vs "Zurich" no.

    This exists because of a real failure. `semantic.aliases` is a
    *memory-level* field: "other names/terms this same thing could be called
    or searched by" about the whole memory, not about one entity in it (see
    app/ingestion/enrichment.py's `_RETRIEVAL_AIDS_SYSTEM`). Folding a
    memory's entire alias list into every entity it mentions -- which is what
    projection did at first -- produced entities like the location
    "Switzerland" carrying the aliases "avs number" and "markus glaciology
    student", and "Zurich" carrying "markus". Since `_find_entity` looks
    entities up by alias, that actively corrupts resolution: a later mention
    of "Markus" could resolve to the city.

    Wrongly merging two entities is worse than failing to merge one (there is
    no un-merge), so the rule errs toward not merging. It therefore misses
    genuine aliases with no shared wording ("ETH Zurich" vs "Swiss Federal
    Institute of Technology"); that limitation is recorded in docs/TODO.md.
    """
    a = normalize_name(alias)
    b = normalize_name(entity_name)
    if not a or not b or a == b:
        return False
    a_words, b_words = a.split(), b.split()
    shorter, longer = (a_words, b_words) if len(a_words) <= len(b_words) else (b_words, a_words)
    n = len(shorter)
    return any(longer[i : i + n] == shorter for i in range(len(longer) - n + 1))


def display_name_for(name: str) -> str:
    """The surface form to show: capitalization preserved, gloss removed.

    Capitalization survives because 'ETH' vs 'eth' is information the
    normalization step deliberately discards for matching but should not
    discard for display.

    A parenthesised tail is dropped, matching `normalize_name`. The gloss is
    the extraction model annotating itself, not part of any name -- live output
    included entities literally called "Slovenia (Implicit location of Lake
    Bled)" and "Mountains behind Lake Bled (Julian Alps region)", which read as
    machine debris wherever they surface. Keeping it here while
    `normalize_name` strips it also meant two spellings of one entity could
    display differently depending on which memory created the row first.
    """
    text = _WS_RE.sub(" ", unicodedata.normalize("NFKC", name or "").strip())
    without_gloss = _WS_RE.sub(" ", _PARENTHETICAL_RE.sub("", text)).strip()
    return without_gloss or text


def valid_kind(kind: str) -> bool:
    return kind in ENTITY_KINDS
