"""Tests for the non-binary (graded-rating) task-trial cartoon."""
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from scripts.non_binary_task_trial_figure import (
    A_RATINGS,
    B_RATINGS,
    COLOR_BAR,
    render_non_binary_trial,
)
from scripts.figure_style import NEUTRAL_HARMONY


def test_ratings_are_graded_not_binary():
    """The whole point of the non-binary figure: option cells hold graded
    ratings (values beyond {0, 1}), unlike binary_task_trial."""
    all_vals = set(A_RATINGS) | set(B_RATINGS)
    assert all_vals - {0, 1}, "ratings must contain non-binary values"
    assert max(all_vals) > 1


def test_validity_bars_use_sage():
    assert COLOR_BAR == NEUTRAL_HARMONY["sage"]


def test_render_shows_each_rating_value():
    fig = render_non_binary_trial()
    assert isinstance(fig, Figure)
    rendered = {t.get_text() for ax in fig.axes for t in ax.texts}
    for v in list(A_RATINGS) + list(B_RATINGS):
        assert str(int(v)) in rendered
