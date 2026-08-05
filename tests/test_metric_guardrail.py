"""Tests for Metric guardrail: bad LLM-generated code must not crash construction."""

import pandas as pd
import pytest

from src.metric import Metric


VALID_SOURCE = "def metric(data):\n    return float(len(data))"

SYNTAX_ERROR_SOURCE = (
    "def metric(data):\n"
    "    data['x'] = data['y'].apply(lambda x: ''.join(map(str, int(v) for v in x)))\n"
    "    return 0.0\n"
)

RUNTIME_ERROR_SOURCE = "def metric(data):\n    return 1 / 0"

NO_METRIC_SOURCE = "def not_metric(data):\n    return 0.0"


class TestMetricGuardrail:
    """Metric with bad code should construct successfully but fail at call time."""

    def test_valid_source_constructs_and_calls(self):
        m = Metric(metric_source=VALID_SOURCE)
        # `Metric.__call__` re-applies the metric per `subject_id`, so the
        # input frame must carry one. Returns `Estimate(value, variance)`:
        # value is `metric(data)` on the full frame (3.0 here), variance is
        # the population variance of `metric(subj_df)` across subjects.
        result = m(pd.DataFrame({"a": [1, 2, 3], "subject_id": [0, 0, 1]}))
        assert result.value == 3.0
        assert result.variance is not None

    def test_syntax_error_does_not_crash_construction(self):
        m = Metric(metric_source=SYNTAX_ERROR_SOURCE)
        assert m is not None

    def test_syntax_error_raises_on_call(self):
        m = Metric(metric_source=SYNTAX_ERROR_SOURCE)
        with pytest.raises(SyntaxError):
            m(pd.DataFrame({"y": [[1, 2]]}))

    def test_missing_metric_func_does_not_crash_construction(self):
        m = Metric(metric_source=NO_METRIC_SOURCE)
        assert m is not None

    def test_missing_metric_func_raises_on_call(self):
        m = Metric(metric_source=NO_METRIC_SOURCE)
        with pytest.raises((KeyError, TypeError)):
            m(pd.DataFrame())
