"""Theory display-name extraction for eval_hilbig labels."""
from scripts.eval_hilbig import theory_display_name


def test_paren_acronym_name():
    d = ("Weighted Additive (WADD) theory posits that individuals evaluate "
         "options by considering all available features...")
    assert theory_display_name(d, "pi_3") == "Weighted Additive (WADD)"


def test_colon_heading_name():
    d = ("Dynamic Strategy Selection (Threshold Model): Decision-makers "
         "dynamically select between a...")
    assert theory_display_name(d, "pi_4") == "Dynamic Strategy Selection (Threshold Model)"


def test_name_before_posits():
    d = ("Diminishing Returns WADD posits that individuals evaluate options "
         "by applying a concave utility...")
    assert theory_display_name(d, "pi_7") == "Diminishing Returns WADD"


def test_canonical_seed_keywords():
    ttb = ("People compare two options by consulting cues one at a time in "
           "order of validity, stopping at the first cue that discriminates")
    tally = ("People compare two options by counting, across all features, "
             "how often one option has a higher value than the other.")
    assert theory_display_name(ttb, "pi_1") == "Take the Best"
    assert theory_display_name(tally, "pi_2") == "Tallying"


def test_wadd_seed_keyword():
    d = ("People compare two options by computing, for each option, a weighted "
         "sum of its feature values, where each feature is weighted by its "
         "subjective validity.")
    assert theory_display_name(d, "pi_2") == "Weighted Additive (WADD)"


def test_long_colon_heading_still_extracted():
    d = ("Probabilistic Strategy Mixture Model with Flexible Compensatory "
         "Component: Subjects maintain a repertoire of distinct strategies...")
    assert theory_display_name(d, "pi_7") == (
        "Probabilistic Strategy Mixture Model with Flexible Compensatory "
        "Component"
    )


def test_fallback_to_label_when_unknown():
    assert theory_display_name("some opaque prose with no name", "pi_9") == "pi_9"


def test_theory_label_renames_base_to_seed_and_wraps():
    from scripts.eval_hilbig import _theory_label
    lab = _theory_label("Take the Best", "base")
    assert lab.endswith("[seed]")
    assert "[base]" not in lab
    # A long name wraps across lines instead of one long string.
    long = _theory_label("Non-linear Subjective Weighting Model", "surfaced")
    name_part = long.rsplit("\n[", 1)[0]
    assert "\n" in name_part
    assert long.endswith("[surfaced]")
