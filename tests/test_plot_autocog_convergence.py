"""Tests for the AutoCog convergence figure's data extraction.

The figure tells the lineage story of one run (online/.../ttb+wadd): two theory
slots across five rounds, who is killed/admitted each round, the final two
surviving theories, and the per-round leaderboard scores. The only logic worth
testing is the extraction; the matplotlib drawing is not unit-tested.

Every assertion below is a hand-verified ground truth read off this run's
`rounds/round_00X/theories.json` and `leaderboard.md` (see the design notes):

    round 0: slot1 pi_1 KILLED, slot2 pi_2 survived; admit pi_3 -> slot 1
    round 1: slot1 pi_3 survived, slot2 pi_2 KILLED; admit pi_4 -> slot 2
    round 2: slot1 pi_3 KILLED, slot2 pi_4 survived; admit pi_5 -> slot 1
    round 3: slot1 pi_5 KILLED, slot2 pi_4 survived; admit pi_6 -> slot 1
    round 4: slot1 pi_6 KILLED, slot2 pi_4 survived; admit pi_7 -> slot 1
    final two survivors: {pi_4, pi_7}
"""
from pathlib import Path

import pytest

from scripts.plot_autocog_convergence import (
    final_survivors,
    load_post_admit_scores,
    parse_lineage,
    seed_names,
    survivor_names,
    theory_title,
)

RUN_DIR = Path(
    "results/human_decision_making_binary/ttb+wadd"
)


@pytest.fixture(scope="module")
def lineage():
    return parse_lineage(RUN_DIR)


def test_one_round_lineage_entry_per_round(lineage):
    assert [r.round_idx for r in lineage] == [0, 1, 2, 3, 4]


def test_killed_slot_each_round(lineage):
    # (round_idx, slot that was killed) — the other slot survived.
    expected_killed_slot = {0: 1, 1: 2, 2: 1, 3: 1, 4: 1}
    for r in lineage:
        killed = [slot for slot, s in r.slots.items() if s.killed]
        assert killed == [expected_killed_slot[r.round_idx]], r.round_idx


def test_slot_occupant_labels_each_round(lineage):
    # Who occupies each slot at the START of the round (before admission).
    expected = {
        0: {1: "pi_1", 2: "pi_2"},
        1: {1: "pi_3", 2: "pi_2"},
        2: {1: "pi_3", 2: "pi_4"},
        3: {1: "pi_5", 2: "pi_4"},
        4: {1: "pi_6", 2: "pi_4"},
    }
    for r in lineage:
        got = {slot: s.label for slot, s in r.slots.items()}
        assert got == expected[r.round_idx], r.round_idx


def test_replacement_label_and_target_slot_each_round(lineage):
    # (round_idx) -> (admitted label, slot it lands in for the next round).
    expected = {
        0: ("pi_3", 1),
        1: ("pi_4", 2),
        2: ("pi_5", 1),
        3: ("pi_6", 1),
        4: ("pi_7", 1),
    }
    for r in lineage:
        assert (r.replacement.label, r.replacement.slot) == expected[r.round_idx]


def test_final_two_survivors(lineage):
    # After the last round's admission: surviving slot (pi_4) + new theory (pi_7).
    assert final_survivors(lineage) == {"pi_4", "pi_7"}


def test_highlight_label_for_curated_runs():
    # The per-run "headline" theory: starred + coloured yellow. ttb+wadd ->
    # Non-linear Subjective Weighting (pi_4); ttb+tallying -> Diminishing
    # Returns WADD (pi_7).
    from scripts.plot_autocog_convergence import highlight_label

    assert highlight_label(RUN_DIR) == "pi_4"
    assert highlight_label(TALLYING_RUN) == "pi_7"


def test_persistent_survivor_is_the_unkilled_final_slot(lineage, tallying_lineage):
    # ttb+wadd: slot 2 (pi_4) is never killed at the end; ttb+tallying: slot 2
    # holds pi_6 at the last round (slot 1's pi_3 is the one killed -> pi_7).
    from scripts.plot_autocog_convergence import persistent_survivor

    assert persistent_survivor(lineage) == "pi_4"
    assert persistent_survivor(tallying_lineage) == "pi_6"


def test_survivor_colors_highlight_is_yellow_other_is_orange(lineage):
    # The highlighted survivor is sandy (yellow); the other survivor terracotta
    # (orange). For ttb+wadd the highlight pi_4 is also the persistent survivor.
    from scripts.figure_style import NEUTRAL_HARMONY
    from scripts.plot_autocog_convergence import survivor_colors

    colors = survivor_colors(lineage, highlight="pi_4")
    assert colors["pi_4"] == NEUTRAL_HARMONY["sandy"]
    assert colors["pi_7"] == NEUTRAL_HARMONY["terracotta"]


def test_survivor_colors_highlight_can_be_the_newcomer(tallying_lineage):
    # ttb+tallying highlights pi_7 (the newcomer), so the yellow/orange roles
    # are not tied to persistent-vs-newcomer.
    from scripts.figure_style import NEUTRAL_HARMONY
    from scripts.plot_autocog_convergence import survivor_colors

    colors = survivor_colors(tallying_lineage, highlight="pi_7")
    assert colors["pi_7"] == NEUTRAL_HARMONY["sandy"]
    assert colors["pi_6"] == NEUTRAL_HARMONY["terracotta"]


def test_theory_colors_follow_convergence_scheme():
    # The single source of truth shared with the human MSE/Pearson trajectory:
    # seeds -> slate, highlight -> sandy, other survivor -> terracotta,
    # transient -> gray. This run additionally pins TTB (pi_1) to sage.
    from scripts.figure_style import GRAY, NEUTRAL_HARMONY, ROLE_COLOR
    from scripts.plot_autocog_convergence import theory_colors

    colors = theory_colors(RUN_DIR)
    assert colors["pi_1"] == NEUTRAL_HARMONY["sage"]         # TTB pinned (this run)
    assert colors["pi_2"] == ROLE_COLOR["seed"]              # WADD seed -> slate
    assert colors["pi_4"] == NEUTRAL_HARMONY["sandy"]        # highlight (yellow)
    assert colors["pi_7"] == NEUTRAL_HARMONY["terracotta"]   # other survivor (orange)
    for lbl in ("pi_3", "pi_5", "pi_6"):
        assert colors[lbl] == GRAY                           # transient


def test_ttb_color_override_pins_only_named_seed():
    # Exact-colour unit check of the override precedence in _label_color:
    # a pinned label wins over the slate-seed default; unpinned seeds and
    # survivors are untouched.
    from scripts.figure_style import NEUTRAL_HARMONY, ROLE_COLOR
    from scripts.plot_autocog_convergence import _label_color

    seeds = {"pi_1": "TTB", "pi_2": "WADD"}
    survivors = {"pi_4": NEUTRAL_HARMONY["sandy"]}
    overrides = {"pi_1": NEUTRAL_HARMONY["sage"]}
    assert _label_color("pi_1", seeds, survivors, overrides) == NEUTRAL_HARMONY["sage"]
    assert _label_color("pi_2", seeds, survivors, overrides) == ROLE_COLOR["seed"]
    assert _label_color("pi_4", seeds, survivors, overrides) == NEUTRAL_HARMONY["sandy"]
    # No override map -> legacy behaviour (slate for the seed).
    assert _label_color("pi_1", seeds, survivors) == ROLE_COLOR["seed"]


def test_ttb_color_override_applies_to_tallying_run_too():
    # TTB (pi_1) is pinned to sage in BOTH human-dataset runs; the tallying
    # run's other seed (Tallying, pi_2) stays slate.
    from scripts.figure_style import NEUTRAL_HARMONY, ROLE_COLOR
    from scripts.plot_autocog_convergence import theory_colors

    other = theory_colors(TALLYING_RUN)
    assert other["pi_1"] == NEUTRAL_HARMONY["sage"]   # TTB pinned (this run too)
    assert other["pi_2"] == ROLE_COLOR["seed"]        # Tallying seed -> slate


def test_theory_title_strips_description_prefix(lineage):
    titles = {
        r.replacement.label: r.replacement.name for r in lineage
    }
    assert titles["pi_3"] == "Probabilistic Heuristic Selection"
    assert titles["pi_4"] == "Non-linear Subjective Weighting Model"
    assert titles["pi_6"] == "Threshold-Gated Compensatory Model"
    assert titles["pi_7"] == (
        "Probabilistic Strategy Mixture Model with "
        "Flexible Compensatory Component"
    )


def test_theory_title_helper_takes_text_before_colon():
    assert theory_title("Foo Model: subjects do X.") == "Foo Model"
    # No colon -> returned unchanged (trimmed).
    assert theory_title("  Bare description  ") == "Bare description"


def test_theory_title_handles_posits_format():
    # Some theories use "Name posits that ..." instead of "Name: ...".
    assert (
        theory_title("Diminishing Returns WADD posits that individuals ...")
        == "Diminishing Returns WADD"
    )
    # "... theory posits that ..." drops the trailing word "theory".
    assert (
        theory_title("Weighted Additive (WADD) theory posits that they ...")
        == "Weighted Additive (WADD)"
    )
    # A colon still wins when it comes first.
    assert (
        theory_title("Satisficing (Binarized WADD): decision-makers ...")
        == "Satisficing (Binarized WADD)"
    )


def test_seed_legend_label_uses_actual_seeds():
    # The legend must name THIS run's seeds, not a hardcoded pair.
    from scripts.plot_autocog_convergence import _seed_legend_label

    assert _seed_legend_label({"pi_1": "TTB", "pi_2": "TALLYING"}) == (
        "seeds (TTB, TALLYING)"
    )
    assert _seed_legend_label({"pi_1": "TTB", "pi_2": "WADD"}) == (
        "seeds (TTB, WADD)"
    )


def test_parse_args_default_and_override():
    from scripts.plot_autocog_convergence import DEFAULT_RUN_DIR, parse_args

    assert parse_args([]).run_dir == DEFAULT_RUN_DIR
    assert parse_args(["some/run/dir"]).run_dir == Path("some/run/dir")


def test_column_label_marks_seed_then_cycles():
    # The left-most lineage column holds the seeds; the rest are cycles 1..N
    # (the final post-admission column is the last cycle, not a separate label).
    from scripts.plot_autocog_convergence import _column_label

    assert _column_label(0) == "seed"
    assert _column_label(1) == "cycle 1"
    assert _column_label(5) == "cycle 5"


def test_main_writes_lineage_and_leaderboard_as_separate_figures(monkeypatch):
    # The lineage grid and the leaderboard trajectory are now two standalone
    # figures (not stacked panels). main() must save exactly these two bases,
    # in order, into the run's analysis/convergence directory.
    import scripts.plot_autocog_convergence as mod

    saved: list[Path] = []

    def fake_save(fig, path_base):
        saved.append(Path(path_base))
        return [Path(path_base).with_suffix(".svg"), Path(path_base).with_suffix(".png")]

    monkeypatch.setattr(mod, "save_figure", fake_save)
    mod.main(RUN_DIR)

    assert [p.name for p in saved] == ["autocog_lineage", "autocog_leaderboard"]
    for p in saved:
        assert p.parent == RUN_DIR / "analysis" / "convergence"


def test_seed_names_from_run_meta(lineage):
    names = seed_names(RUN_DIR)
    # run_meta seeds are ["ttb", "wadd"]; slot 1 of round 0 is the first seed.
    assert names["pi_1"] == "TTB"
    assert names["pi_2"] == "WADD"


def test_post_admit_scores_known_values():
    rounds = load_post_admit_scores(RUN_DIR)
    assert len(rounds) == 5
    # pi_4 is admitted at round 1 and holds rank #1 every round after.
    assert rounds[1]["pi_4"] == pytest.approx(0.922)
    assert rounds[2]["pi_4"] == pytest.approx(0.878)
    assert rounds[3]["pi_4"] == pytest.approx(0.809)
    assert rounds[4]["pi_4"] == pytest.approx(0.760)
    # pi_4 is the per-round maximum from round 1 onward (rank persistence).
    for r in (1, 2, 3, 4):
        assert max(rounds[r].values()) == pytest.approx(rounds[r]["pi_4"])
    # pi_7 only appears in the final round's post-admit table.
    assert rounds[4]["pi_7"] == pytest.approx(0.510)
    assert "pi_7" not in rounds[3]
    # The seed TTB (pi_1) sits at the floor every round.
    for r in range(5):
        assert rounds[r]["pi_1"] == pytest.approx(0.0)


# --- generalization: a second run with different seeds and naming formats -----
# human_decision_making_cardinal/ttb+tallying — seeds are TTB+Tallying, slot 2
# churns (not slot 1), and the final two are BOTH late arrivals (pi_6, pi_7).
# Ground truth read off this run's theories.json / leaderboard.md:
#   round 0: kill slot1 (pi_1/TTB), admit pi_3 (WADD) -> slot 1
#   round 1: kill slot2 (pi_2/Tallying), admit pi_4 -> slot 2
#   round 2: kill slot2 (pi_4), admit pi_5 -> slot 2
#   round 3: kill slot2 (pi_5), admit pi_6 -> slot 2
#   round 4: kill slot1 (pi_3), admit pi_7 -> slot 1
#   final two: {pi_6, pi_7}
TALLYING_RUN = Path(
    "results/human_decision_making_cardinal/ttb+tallying"
)


@pytest.fixture(scope="module")
def tallying_lineage():
    return parse_lineage(TALLYING_RUN)


def test_tallying_seed_names():
    # Acronyms stay upper-case; "Tallying" is shown title-case.
    assert seed_names(TALLYING_RUN) == {"pi_1": "TTB", "pi_2": "Tallying"}


def test_seed_display_uppercases_acronyms_but_titlecases_tallying():
    from scripts.plot_autocog_convergence import _seed_display

    assert _seed_display("ttb") == "TTB"
    assert _seed_display("wadd") == "WADD"
    assert _seed_display("tallying") == "Tallying"
    # Unknown seeds fall back to upper-case (the previous behaviour).
    assert _seed_display("foo") == "FOO"


def test_abbrev_shortens_weighted_additive_to_wadd():
    from scripts.plot_autocog_convergence import _abbrev

    # The verbose "Weighted Additive (WADD)" theory name is shown as "WADD".
    assert _abbrev("Weighted Additive (WADD)") == "WADD"
    # Other theory names pass through unchanged.
    assert _abbrev("Non-linear Subjective Weighting Model") == (
        "Non-linear Subjective Weighting Model"
    )


def test_tallying_admissions_and_killed_slots(tallying_lineage):
    admit = {
        r.round_idx: (r.replacement.label, r.replacement.slot)
        for r in tallying_lineage
    }
    assert admit == {
        0: ("pi_3", 1),
        1: ("pi_4", 2),
        2: ("pi_5", 2),
        3: ("pi_6", 2),
        4: ("pi_7", 1),
    }


def test_tallying_final_survivors(tallying_lineage):
    assert final_survivors(tallying_lineage) == {"pi_6", "pi_7"}


def test_tallying_survivor_names_parse_both_formats(tallying_lineage):
    # pi_7 uses the "Name posits that ..." format; the parse must still be short.
    names = survivor_names(tallying_lineage)
    assert names["pi_7"] == "Diminishing Returns WADD"
    assert names["pi_6"] == "Threshold-based Binarization (Satisficing WADD)"


def test_tallying_pi3_name_from_posits_format(tallying_lineage):
    # pi_3 ("Weighted Additive (WADD) theory posits that ...") occupies slot 1
    # for rounds 1-4; its displayed name must be the short title.
    pi3_round1 = next(r for r in tallying_lineage if r.round_idx == 1)
    assert pi3_round1.slots[1].label == "pi_3"
    assert pi3_round1.slots[1].name == "Weighted Additive (WADD)"


def test_reasoning_is_per_run_with_empty_fallback(tmp_path):
    # Curated reasoning is keyed per run; the wrong run's text must never leak.
    from scripts.plot_autocog_convergence import reasoning_for

    assert set(reasoning_for(RUN_DIR)) == {0, 1, 2, 3, 4}
    assert set(reasoning_for(TALLYING_RUN)) == {0, 1, 2, 3, 4}
    # An uncurated run gets NO annotations rather than a mismatched run's text.
    (tmp_path / "run_meta.json").write_text(
        '{"run_id": "unknown", "seeds": ["x", "y"]}'
    )
    assert reasoning_for(tmp_path) == {}


def test_tallying_known_scores_are_tightly_clustered():
    rounds = load_post_admit_scores(TALLYING_RUN)
    assert rounds[0]["pi_3"] == pytest.approx(1.0)
    # Final round: the field bunches near 0.5 (no decisive winner).
    assert rounds[4]["pi_6"] == pytest.approx(0.570)
    assert rounds[4]["pi_7"] == pytest.approx(0.556)
    assert rounds[4]["pi_5"] == pytest.approx(0.522)
