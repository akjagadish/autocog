from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict


class OnlineConfig(BaseModel):
    """
    Deployment knobs for running an experiment online.

    Owns *where* and *how* to recruit subjects and *where* the data lives.
    Knows nothing about the experiment design itself (that's on the Experiment class).
    """

    model_config = ConfigDict(extra="ignore")

    firebase_credentials_path: str
    study_url: str

    n_subjects: int = 1
    duration: int = 20
    """Per-condition timeout in minutes (Prolific expects minutes; firebase runner gets minutes*60)."""

    is_prolific: bool = False
    prolific_settings_path: str | None = None
    completion_code: str = ""

    reward: int = 0
    """Per-participant Prolific payment in the smallest currency unit
    (cents / pence). 0 = let upstream auto-compute as
    ``20 * study_completion_time`` (≈ $12/hour). Set explicitly to anchor
    at Prolific's policy minimum (£6/hour ≈ £0.10/min) or any other rate.
    Ignored when ``is_prolific=False``."""

    consent_yaml_path: str = "consent.yaml"

    min_rt_ms: int = 1500
    """Minimum response time (in ms) per choice trial. While > 0, the trial
    cannot be advanced for the first ``min_rt_ms`` after it renders.
    Defaults to 1500 for online runs. Implemented per-stimulus by
    SweetBean's ``HtmlKeyboardResponse(min_rt=...)`` — only consumed by
    experiments that opt into it via ``build_online_experiment(min_rt_ms=...)``."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)
