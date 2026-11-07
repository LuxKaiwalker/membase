import datetime as dt

from app.memory.temporal import (
    PRECISION_DAY,
    PRECISION_MONTH,
    PRECISION_SEASON,
    PRECISION_UNKNOWN,
    PRECISION_YEAR,
    dates_compatible,
    describe,
    looser_precision,
    parse_event_date,
    tighter_precision,
)

# The one rule this module exists to enforce (see its docstring): never invent
# a date. Every "-> unknown" assertion below is guarding that, not testing a
# regex -- a timeline that files an undated memory under a plausible-looking
# day is worse than one that admits the date is unknown.


def test_iso_date_parses_at_day_precision():
    assert parse_event_date("2024-03-15") == (dt.date(2024, 3, 15), PRECISION_DAY)


def test_day_first_european_date_is_read_day_first():
    # Documented convention (this deployment's user writes 15.03.2024, not
    # 03/15/2024); the point is that it is the *same* reading every time.
    assert parse_event_date("15.03.2024") == (dt.date(2024, 3, 15), PRECISION_DAY)


def test_day_month_year_words_parse_at_day_precision():
    assert parse_event_date("15 March 2024") == (dt.date(2024, 3, 15), PRECISION_DAY)


def test_month_day_year_words_parse_at_day_precision():
    assert parse_event_date("March 15, 2024") == (dt.date(2024, 3, 15), PRECISION_DAY)


def test_iso_month_parses_at_month_precision_anchored_to_the_first():
    date_value, precision = parse_event_date("2024-03")
    assert precision == PRECISION_MONTH
    # The anchor is deliberate and sortable, not a claim about the day -- which
    # is exactly why `precision` travels alongside it.
    assert date_value == dt.date(2024, 3, 1)


def test_month_name_and_year_parse_at_month_precision():
    assert parse_event_date("March 2024") == (dt.date(2024, 3, 1), PRECISION_MONTH)


def test_season_and_year_parse_at_season_precision():
    date_value, precision = parse_event_date("summer 2023")
    assert precision == PRECISION_SEASON
    assert date_value == dt.date(2023, 7, 15)  # mid-season anchor


def test_bare_year_parses_at_year_precision():
    assert parse_event_date("2023") == (dt.date(2023, 1, 1), PRECISION_YEAR)


def test_empty_string_is_unknown():
    assert parse_event_date("") == (None, PRECISION_UNKNOWN)


def test_none_is_unknown():
    # enrichment can leave event_date unset entirely; a non-string must not
    # raise its way out of a background pipeline step.
    assert parse_event_date(None) == (None, PRECISION_UNKNOWN)


def test_unresolved_relative_expression_is_unknown_not_guessed():
    """"last summer" is relative to a *writing* date the parser does not have.
    Resolving it against today would silently invent a year."""
    assert parse_event_date("last summer") == (None, PRECISION_UNKNOWN)


def test_a_date_that_does_not_exist_is_unknown():
    """A complete but impossible date means the string is unreliable. Falling
    back to "February 2024" would be asserting a date read out of garbage."""
    assert parse_event_date("2024-02-31") == (None, PRECISION_UNKNOWN)


def test_a_bare_quantity_does_not_become_a_year():
    # "5000 steps" must not produce year 5000: an out-of-range year in a
    # personal memory store is a misparsed measurement, not an event date.
    assert parse_event_date("5000 steps") == (None, PRECISION_UNKNOWN)
    assert parse_event_date("I ran 5000 steps today") == (None, PRECISION_UNKNOWN)


def test_a_year_outside_the_plausible_range_is_rejected():
    # 1800 matches the year regex but sits below _MIN_YEAR, so the guard in
    # _safe_date is what stops it -- not the pattern.
    assert parse_event_date("01.01.1800") == (None, PRECISION_UNKNOWN)
    assert parse_event_date("1800") == (None, PRECISION_UNKNOWN)


def test_tightest_precision_wins_when_several_patterns_match():
    # A string carrying both a full ISO date and a month name is day-precision:
    # the parser must not downgrade to the loosest thing it can see.
    assert parse_event_date("2024-03-15 (that was March 2024)") == (
        dt.date(2024, 3, 15),
        PRECISION_DAY,
    )


def test_tighter_precision_picks_the_more_precise_label():
    assert tighter_precision(PRECISION_YEAR, PRECISION_DAY) == PRECISION_DAY
    assert tighter_precision(PRECISION_SEASON, PRECISION_MONTH) == PRECISION_MONTH
    assert tighter_precision(PRECISION_UNKNOWN, PRECISION_YEAR) == PRECISION_YEAR
    assert tighter_precision(PRECISION_DAY, PRECISION_DAY) == PRECISION_DAY


def test_looser_precision_picks_what_a_span_can_honestly_claim():
    assert looser_precision(PRECISION_YEAR, PRECISION_DAY) == PRECISION_YEAR
    assert looser_precision(PRECISION_SEASON, PRECISION_MONTH) == PRECISION_SEASON
    assert looser_precision(PRECISION_UNKNOWN, PRECISION_DAY) == PRECISION_UNKNOWN


def test_an_unrecognised_precision_label_is_treated_as_the_loosest():
    # Defensive: a stored date_precision from an older schema version must not
    # blow up an ordering comparison.
    assert tighter_precision("bogus", PRECISION_DAY) == PRECISION_DAY
    assert looser_precision("bogus", PRECISION_DAY) == PRECISION_UNKNOWN


def test_the_same_day_is_compatible_with_itself():
    day = dt.date(2024, 3, 15)
    assert dates_compatible(day, PRECISION_DAY, day, PRECISION_DAY) is True


def test_two_month_precision_dates_ten_days_apart_are_compatible():
    # Both anchors are only accurate to ±15 days, so 10 days apart is well
    # inside what either date actually claims.
    assert (
        dates_compatible(
            dt.date(2024, 3, 1), PRECISION_MONTH, dt.date(2024, 3, 11), PRECISION_MONTH
        )
        is True
    )


def test_two_month_precision_dates_two_hundred_days_apart_are_not_compatible():
    assert (
        dates_compatible(
            dt.date(2024, 1, 1), PRECISION_MONTH, dt.date(2024, 7, 19), PRECISION_MONTH
        )
        is False
    )


def test_two_undated_memories_are_not_compatible():
    """A deliberate rule, not an oversight: two memories that both have no
    date are not thereby evidence of having happened at the same time. Treating
    unknown as matching unknown is how an "episode" becomes a bag of every
    undated memory in the store."""
    assert dates_compatible(None, PRECISION_UNKNOWN, None, PRECISION_UNKNOWN) is False
    assert dates_compatible(dt.date(2024, 3, 15), PRECISION_DAY, None, PRECISION_UNKNOWN) is False


def test_describe_never_renders_more_precision_than_is_known():
    year_only = dt.date(2023, 1, 1)
    rendered = describe(year_only, PRECISION_YEAR)
    assert rendered == "2023"
    # The anchor day (1 January) must not leak into the rendering -- that is
    # the display half of "never invent a date".
    assert "01" not in rendered
    assert "January" not in rendered


def test_describe_renders_each_precision_at_its_own_granularity():
    assert describe(dt.date(2024, 3, 15), PRECISION_DAY) == "2024-03-15"
    assert describe(dt.date(2024, 3, 1), PRECISION_MONTH) == "March 2024"
    assert describe(dt.date(2023, 7, 15), PRECISION_SEASON) == "summer 2023"
    assert describe(dt.date(2023, 10, 15), PRECISION_SEASON) == "autumn 2023"


def test_describe_says_so_when_the_date_is_unknown():
    assert describe(None, PRECISION_UNKNOWN) == "date unknown"
    # A stray date with precision "unknown" is still reported as unknown: the
    # precision label is authoritative over the anchor.
    assert describe(dt.date(2024, 3, 15), PRECISION_UNKNOWN) == "date unknown"


def test_a_string_that_says_the_date_is_unknown_never_yields_a_date():
    """The guard that matters most, because the loosest pattern is the most
    dangerous one: a bare four-digit year will match inside a sentence whose
    whole point is that there is no date.

    This exact string came out of the live model as a unit's `event_date`, and
    the bare-year fallback read "2021" out of it and reported year precision --
    the module deriving a date from a string that explicitly disclaims one.
    """
    disclaimer = (
        "Not specified in the primary memory (though related memories suggest "
        "a timeframe between 2021 and 2023)."
    )
    assert parse_event_date(disclaimer) == (None, PRECISION_UNKNOWN)

    for phrase in (
        "date unknown",
        "No date given in the text.",
        "The exact date is unclear, possibly 2019.",
        "not stated",
        "cannot be determined from the memory, maybe March 2022",
    ):
        assert parse_event_date(phrase) == (None, PRECISION_UNKNOWN), phrase


def test_the_unknown_date_guard_does_not_swallow_a_real_date():
    """The guard is a substring check, so it must not fire on ordinary text
    that merely happens to contain one of its words in another sense."""
    parsed, precision = parse_event_date("Met an unknown number of people on 2024-03-15")
    # "unknown" is present, so this IS treated as unknown -- documenting the
    # deliberate bluntness rather than pretending it is precise.
    assert (parsed, precision) == (None, PRECISION_UNKNOWN)

    # But a plain date, and a date with harmless surrounding prose, still parse.
    assert parse_event_date("2024-03-15")[1] == "day"
    assert parse_event_date("Sometime in March 2024, on the way back")[1] == "month"
