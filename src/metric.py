from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from pydantic import AliasChoices, Field, PrivateAttr

from src.promptable import Promptable


class Estimate(NamedTuple):
    """Result of evaluating a `Metric` on a dataset.

    `value` is the LLM-emitted `metric(data)` evaluated on the full pooled
    dataset (the canonical scalar — what the rest of the system was reading
    when the metric used to return a bare float). `variance` is the
    population variance (`ddof=0`) of `metric(subj_df)` re-applied per
    `subject_id`, summarising between-subject variability around `value`.
    `variance` may be `None` when the per-subject re-application failed
    (e.g. the metric requires multi-subject statistics).

    A `NamedTuple`, so downstream code can equally unpack as `(v, var)` or
    read named attributes (`est.value`, `est.variance`).
    """

    value: float | None
    variance: float | None


def fmt_estimate(est: "Estimate | None") -> str:
    """Compact human/LLM-facing rendering: `v.vvvv (var=X.XXXX)`.

    `None` value -> "n/a"; missing variance -> "n/a" inside the parens.
    Centralised so every prompt and log line uses the same notation, which
    we then explain once in each prompt's system instructions.
    """
    if est is None:
        return "n/a"
    v = "n/a" if est.value is None else f"{est.value:.4f}"
    var = "n/a" if est.variance is None else f"{est.variance:.4f}"
    return f"{v} (var={var})"


class Metric(Promptable):
    """
    A scalar summary of experimental data, operationalized as code.

    A Metric carries a Python source string defining

        metric(data: pd.DataFrame) -> float

    where `data` is a tidy per-trial DataFrame stacking all subjects (rows
    grouped by `subject_id`, in trial order). The contract for `data`'s
    columns lives on the Experiment that produced it (`Experiment.output_columns`).
    """

    metric_source: str = Field(
        ...,
        validation_alias=AliasChoices("metric", "metric_source"),
        description="Python source defining def metric(data: pd.DataFrame) -> float.",
    )
    rationale: str | None = Field(
        default=None,
        description="Reasoning behind this metric; used when interpreting results.",
    )

    _metric_impl: Any = PrivateAttr(default=None)
    _deferred_error: BaseException | None = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        ns: dict[str, Any] = {"np": np, "pd": pd}
        try:
            exec(self.metric_source, ns)
            object.__setattr__(self, "_metric_impl", ns["metric"])
        except Exception as e:
            object.__setattr__(self, "_deferred_error", e)

    def __call__(self, data: pd.DataFrame) -> Estimate:
        """Evaluate the metric, returning `Estimate(value, variance)`.

        - `value`: `self._metric_impl(data)` on the full pooled dataset
          (unchanged from before — the canonical scalar).
        - `variance`: population variance (`ddof=0`) of
          `self._metric_impl(subj_df)` re-applied per `subject_id`,
          summarising between-subject variability around `value`. Each
          `subj_df` is treated as a self-contained single-subject dataset
          and we make sure the `subject_id` column survives the slicing
          so user-supplied `metric_impl`s that touch `data['subject_id']`
          don't crash. If the per-subject re-application fails (the metric
          requires multi-subject statistics, or breaks on a degenerate
          slice), `variance` falls back to `None` and the pooled `value`
          is still returned — the downstream acceptance test (`pi.py`)
          handles a missing variance by rejecting the metric, rather than
          letting one bad per-subject call kill the whole estimate.

        A `NamedTuple` is returned, so callers can equally do `v, var = m(d)`
        or `est = m(d); est.value`.
        """
        if self._deferred_error is not None:
            raise self._deferred_error
        value = self._metric_impl(data)
        variance: float | None
        if "subject_id" not in data.columns:
            variance = None
        else:
            per_subject: list[float] = []
            try:
                for sid, subj_df in data.groupby("subject_id", sort=False):
                    # `groupby` already keeps the grouping column, but make
                    # it explicit on a fresh copy so the metric sees a
                    # well-formed single-subject DataFrame (with the
                    # `subject_id` column populated to a constant `sid`)
                    # regardless of any in-place mutations the metric does.
                    subj_df = subj_df.copy()
                    subj_df["subject_id"] = sid
                    per_subject.append(float(self._metric_impl(subj_df)))
            except Exception:
                # Per-subject re-application failed (e.g. metric needs
                # cross-subject aggregation). Drop variance; keep value.
                per_subject = []
            variance = float(np.var(per_subject)) if per_subject else None
        return Estimate(value=value, variance=variance)
