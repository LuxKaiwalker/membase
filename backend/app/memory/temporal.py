"""Deterministic parsing of `semantic.event_date` into a sortable date plus an
honest precision label.

`semantic.event_date` is free text the enrichment call produces (see
app/ingestion/enrichment.py's `_NARRATIVE_SYSTEM`, which asks for an absolute
date and otherwise an empty string). In practice it arrives as anything from
"2024-03-15" to "March 2024" to "summer 2023" to "" -- and occasionally as
something genuinely unparseable.

The one rule this module exists to enforce: **never invent a date.** A string
this parser cannot confidently resolve yields `(None, "unknown")`, not a
plausible guess and not a fallback to `created_at`. A timeline that silently
files an undated memory under the day it was typed is worse than one that
admits the date is unknown, because the user cannot tell the two apart.

`precision` is what the caller must respect when displaying or comparing:

    "day"     -- a specific calendar day is known              (2024-03-15)
    "month"   -- the month is known, the day is not            (March 2024)
    "season"  -- a season is known, the month is approximate   (summer 2023)
    "year"    -- only the year is known                        (2023)
    "unknown" -- nothing usable; `date` is always None

For "month"/"season"/"year" the returned date is a deliberate *representative*
anchor (first of the month, mid-season, first of the year) so the value is
sortable. It is not a claim that the event happened on that day, which is
exactly why precision travels with it everywhere instead of being dropped.
"""

import datetime as dt
import re

PRECISION_DAY = "day"
PRECISION_MONTH = "month"
PRECISION_SEASON = "season"
PRECISION_YEAR = "year"
PRECISION_UNKNOWN = "unknown"

# Ordered loosest-to-tightest, so callers can compare two precisions.
PRECISION_ORDER = (PRECISION_UNKNOWN, PRECISION_YEAR, PRECISION_SEASON, PRECISION_MONTH, PRECISION_DAY)

# How many days either side of the anchor the real event could plausibly fall.
# Used by episode clustering and by "approximately when" queries so a
# month-precision memory and a day-precision one in the same month are
# recognised as temporally compatible instead of 15 days apart.
PRECISION_TOLERANCE_DAYS = {
    PRECISION_DAY: 0,
    PRECISION_MONTH: 15,
    PRECISION_SEASON: 45,
    PRECISION_YEAR: 182,
    PRECISION_UNKNOWN: 0,
}

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# Northern-hemisphere seasons, with a mid-season anchor day. A deliberate
# simplification -- documented rather than silently assumed. The user base is
# one person in Zurich; a southern-hemisphere user would need this inverted,
# and `precision="season"` plus a 45-day tolerance is wide enough that the
# anchor choice does not drive any retrieval decision on its own.
_SEASONS = {
    "spring": (4, 15),
    "summer": (7, 15),
    "autumn": (10, 15),
    "fall": (10, 15),
    "winter": (1, 15),
}

# Phrases that say outright the date is NOT known. These have to short-circuit
# the whole parser, because the loosest pattern below (`_YEAR_RE`, a bare
# four-digit year) will happily find a year inside a sentence whose entire
# point is that there isn't one.
#
# Found for real, not hypothesised: the live model produced
#   "Not specified in the primary memory (though related memories suggest a
#    timeframe between 2021 and 2023)."
# as a unit's `event_date`. `_YEAR_RE` matched "2021", so the memory was
# reported as year-precision 2021 and placed on the timeline accordingly --
# the module deriving a date from a string that explicitly disclaims one,
# which is precisely the one thing it promises never to do.
_NO_DATE_PHRASES = (
    "not specified",
    "unspecified",
    "not stated",
    "not mentioned",
    "not given",
    "not provided",
    "not known",
    "not clear",
    "unclear",
    "unknown",
    "no date",
    "no specific date",
    "cannot be determined",
    "can't be determined",
    "could not be determined",
    "no explicit date",
    "not determinable",
)

_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_ISO_MONTH_RE = re.compile(r"\b(\d{4})-(\d{1,2})\b")
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")

# "15 March 2024", "March 15, 2024", "March 15th 2024"
_DAY_MONTH_YEAR_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?\s+(\d{4})\b", re.IGNORECASE
)
_MONTH_DAY_YEAR_RE = re.compile(
    r"\b([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.IGNORECASE
)
_MONTH_YEAR_RE = re.compile(r"\b([a-z]+)\.?\s+(\d{4})\b", re.IGNORECASE)
_SEASON_YEAR_RE = re.compile(
    r"\b(spring|summer|autumn|fall|winter)\s+(?:of\s+)?(\d{4})\b", re.IGNORECASE
)

# Years outside this range in a personal memory store are far more likely to be
# a misparsed quantity ("I ran 5000 steps") than a real event date.
_MIN_YEAR = 1900
_MAX_YEAR = 2100


def _safe_date(year: int, month: int, day: int) -> dt.date | None:
    if not (_MIN_YEAR <= year <= _MAX_YEAR):
        return None
    try:
        return dt.date(year, month, day)
    except ValueError:
        # A real invalid date ("2024-02-31") is a parse failure, not something
        # to clamp into a nearby valid day -- clamping would fabricate.
        return None


def parse_event_date(raw: str | None) -> tuple[dt.date | None, str]:
    """Parse a free-text event date into `(anchor_date, precision)`.

    Returns `(None, "unknown")` for anything not confidently resolvable --
    including an empty string, a relative expression the enrichment call failed
    to resolve ("last summer", "a few years ago"), and a date that does not
    exist. Never guesses, never falls back to the current date.
    """
    if not isinstance(raw, str):
        return None, PRECISION_UNKNOWN
    text = raw.strip()
    if not text:
        return None, PRECISION_UNKNOWN

    # A string that says the date is unknown is unknown, whatever digits it
    # happens to contain. Checked before every pattern, including the exact
    # ones: a model that writes "date not specified (the ISO 2023-04-01 refers
    # to a related memory)" is still telling us it does not know.
    lowered = text.casefold()
    if any(phrase in lowered for phrase in _NO_DATE_PHRASES):
        return None, PRECISION_UNKNOWN

    # Tightest precision first: a string containing a full ISO date is treated
    # as day-precision even if it also mentions a month name elsewhere.
    m = _ISO_RE.search(text)
    if m:
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            return d, PRECISION_DAY
        # A *complete* date that does not exist ("2024-02-31") or sits outside
        # the plausible year range means the string itself is unreliable. The
        # looser patterns below would happily read "2024-02" out of it and
        # report month precision -- i.e. derive a date from garbage, which is
        # the one thing this module promises not to do.
        return None, PRECISION_UNKNOWN

    m = _DMY_RE.search(text)
    if m:
        # Day-first (15.03.2024): European convention, which is what this
        # deployment's user writes. Ambiguous cases (03.04.2024) are therefore
        # read as 3 April, not 4 March -- a documented convention, not a guess
        # the parser makes differently from one input to the next.
        d = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if d:
            return d, PRECISION_DAY
        return None, PRECISION_UNKNOWN

    m = _DAY_MONTH_YEAR_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(2).lower())
        # An unrecognised word is not a failed date, just a different shape
        # ("summer 2023" reaches the season pattern below), so only a known
        # month with an impossible day short-circuits to unknown.
        if month:
            d = _safe_date(int(m.group(3)), month, int(m.group(1)))
            if d:
                return d, PRECISION_DAY
            return None, PRECISION_UNKNOWN

    m = _MONTH_DAY_YEAR_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            d = _safe_date(int(m.group(3)), month, int(m.group(2)))
            if d:
                return d, PRECISION_DAY
            return None, PRECISION_UNKNOWN

    m = _ISO_MONTH_RE.search(text)
    if m:
        d = _safe_date(int(m.group(1)), int(m.group(2)), 1)
        if d:
            return d, PRECISION_MONTH

    m = _MONTH_YEAR_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            d = _safe_date(int(m.group(2)), month, 1)
            if d:
                return d, PRECISION_MONTH

    m = _SEASON_YEAR_RE.search(text)
    if m:
        month, day = _SEASONS[m.group(1).lower()]
        d = _safe_date(int(m.group(2)), month, day)
        if d:
            return d, PRECISION_SEASON

    m = _YEAR_RE.search(text)
    if m:
        d = _safe_date(int(m.group(1)), 1, 1)
        if d:
            return d, PRECISION_YEAR

    return None, PRECISION_UNKNOWN


def tighter_precision(a: str, b: str) -> str:
    """The more precise of two precision labels."""
    ia = PRECISION_ORDER.index(a) if a in PRECISION_ORDER else 0
    ib = PRECISION_ORDER.index(b) if b in PRECISION_ORDER else 0
    return PRECISION_ORDER[max(ia, ib)]


def looser_precision(a: str, b: str) -> str:
    """The less precise of two precision labels -- what a span covering both
    can honestly claim."""
    ia = PRECISION_ORDER.index(a) if a in PRECISION_ORDER else 0
    ib = PRECISION_ORDER.index(b) if b in PRECISION_ORDER else 0
    return PRECISION_ORDER[min(ia, ib)]


def dates_compatible(
    date_a: dt.date | None, precision_a: str, date_b: dt.date | None, precision_b: str
) -> bool:
    """Could these two dates refer to the same time, given how precisely each
    is actually known?

    An unknown date is compatible with nothing -- not even with another unknown
    date. Two memories that both have no date are not thereby evidence of
    having happened at the same time, and treating them as such is how an
    "episode" ends up being a bag of every undated memory in the store.
    """
    if date_a is None or date_b is None:
        return False
    tolerance = PRECISION_TOLERANCE_DAYS.get(precision_a, 0) + PRECISION_TOLERANCE_DAYS.get(precision_b, 0)
    return abs((date_a - date_b).days) <= tolerance


def describe(date_value: dt.date | None, precision: str) -> str:
    """Render a date at no more precision than is actually known -- so a
    year-precision memory never displays as a specific day."""
    if date_value is None or precision == PRECISION_UNKNOWN:
        return "date unknown"
    if precision == PRECISION_DAY:
        return date_value.isoformat()
    if precision == PRECISION_MONTH:
        return date_value.strftime("%B %Y")
    if precision == PRECISION_SEASON:
        for name, (month, _day) in _SEASONS.items():
            if month == date_value.month and name != "fall":  # "autumn" over "fall"
                return f"{name} {date_value.year}"
        return date_value.strftime("%B %Y")
    return str(date_value.year)
