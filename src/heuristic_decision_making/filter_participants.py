"""Thin compatibility wrapper around the generic sweetbean quality pipeline.

Historical callers imported ``filter_participants`` and
``participant_filter_report`` from this module. The actual filtering logic
now lives in :mod:`sweetbean.util.data_process` so every domain experiment
shares the same single-key / fast-RT / alternation / bot-detection checks
and produces the same per-subject report (incl. ``prolific_pid``).

Keeping the wrappers lets older code paths and notebooks keep importing
``filter_participants`` while the new HDM ``_observations_to_df`` reaches
into the sweetbean helpers directly.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from sweetbean.util.data_process import (
    participant_quality_report as _sweetbean_quality_report,
    split_by_quality as _sweetbean_split,
)


DEFAULT_SINGLE_KEY_THRESHOLD = 0.95
DEFAULT_MIN_MEAN_RT_MS = 1200.0
DEFAULT_ALTERNATION_THRESHOLD = 0.95
# 60 s of cumulative off-tab time across the task. Real participants who
# briefly check a notification stay under this; a participant who tabs away
# to do something else for a minute trips it.
DEFAULT_MAX_BLUR_DURATION_MS = 60_000.0


def participant_filter_report(
    data: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    response_col: str = "response",
    rt_col: str = "rt_ms",
    suspicious_browser_col: str = "is_suspicious_browser",  # kept for back-compat
    honeypot_col: str = "honeypot_filled",                  # kept for back-compat
    single_key_threshold: float = DEFAULT_SINGLE_KEY_THRESHOLD,
    min_mean_rt_ms: float = DEFAULT_MIN_MEAN_RT_MS,
    alternation_threshold: float = DEFAULT_ALTERNATION_THRESHOLD,
    max_blur_duration_ms: float = DEFAULT_MAX_BLUR_DURATION_MS,
) -> pd.DataFrame:
    """Per-subject quality report — wraps the sweetbean pipeline.

    Accepts a ``data`` frame that already has the canonical per-trial
    rows and BotDetection signals flattened (the shape produced by
    :func:`sweetbean.util.data_process.parse_autora_observations`). For
    legacy callers that pass a frame with the old top-level
    ``is_suspicious_browser`` / ``honeypot_filled`` columns, those are
    promoted to the ``bot_detection_*`` names the sweetbean helper
    expects.
    """
    if data.empty:
        return _sweetbean_quality_report(
            data,
            subject_col=subject_col,
            response_col=response_col,
            rt_col=rt_col,
        )

    promoted = data
    rename: dict[str, str] = {}
    if (
        suspicious_browser_col in data.columns
        and "bot_detection_is_suspicious_browser" not in data.columns
    ):
        rename[suspicious_browser_col] = "bot_detection_is_suspicious_browser"
    if (
        honeypot_col in data.columns
        and "bot_detection_honeypot_filled" not in data.columns
    ):
        rename[honeypot_col] = "bot_detection_honeypot_filled"
    if rename:
        promoted = data.rename(columns=rename)

    return _sweetbean_quality_report(
        promoted,
        subject_col=subject_col,
        response_col=response_col,
        rt_col=rt_col,
        single_key_threshold=single_key_threshold,
        min_mean_rt_ms=min_mean_rt_ms,
        alternation_threshold=alternation_threshold,
        max_blur_duration_ms=max_blur_duration_ms,
    )


def filter_participants(
    data: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    response_col: str = "response",
    rt_col: str = "rt_ms",
    suspicious_browser_col: str = "is_suspicious_browser",
    honeypot_col: str = "honeypot_filled",
    single_key_threshold: float = DEFAULT_SINGLE_KEY_THRESHOLD,
    min_mean_rt_ms: float = DEFAULT_MIN_MEAN_RT_MS,
    alternation_threshold: float = DEFAULT_ALTERNATION_THRESHOLD,
    max_blur_duration_ms: float = DEFAULT_MAX_BLUR_DURATION_MS,
) -> pd.DataFrame:
    """Drop low-quality subjects from ``data`` using the sweetbean pipeline.

    Returns a copy of ``data`` with dropped subjects removed. The full
    per-subject report is attached on ``out.attrs["participant_filter_report"]``
    and the list of dropped subject ids on
    ``out.attrs["participant_filter_dropped_subjects"]`` (sorted).
    """
    report = participant_filter_report(
        data,
        subject_col=subject_col,
        response_col=response_col,
        rt_col=rt_col,
        suspicious_browser_col=suspicious_browser_col,
        honeypot_col=honeypot_col,
        single_key_threshold=single_key_threshold,
        min_mean_rt_ms=min_mean_rt_ms,
        alternation_threshold=alternation_threshold,
        max_blur_duration_ms=max_blur_duration_ms,
    )
    if report.empty:
        out = data.copy()
        out.attrs["participant_filter_report"] = report
        return out

    kept, _ = _sweetbean_split(data, report, subject_col=subject_col)
    out = cast(pd.DataFrame, kept)
    out.attrs["participant_filter_report"] = report
    out.attrs["participant_filter_dropped_subjects"] = sorted(
        cast(list[Any], report.loc[report["drop"], subject_col].tolist())
    )
    return out
