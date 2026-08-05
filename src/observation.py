"""Observation / Round / Observations — the loop's evidence pool.

Three nested layers:

  * `Observation` — one (experiment, metric, proposer_theory, data,
    real_value, predicted_values) record. The unit of empirical evidence.
  * `Round` — an ordered group of Observations from one debate iteration.
    In the standard two-pi setup each round contains exactly two
    Observations (one per pi).
  * `Observations` — the pool of all Rounds, owns persistence.

Indexing is round-wise (`pool.rounds[i]`, `round.observations[j]`) so the
caller never deals with a flat global index. References returned by `add(...)`
let `main_` thread observations through downstream steps without ever
touching numeric indices.

Cardinality (per Observation):
  one experiment ↔ one metric ↔ one real_value ↔ many predicted_values.

Theories live at the Round level (each Observation's `proposer_theory`
snapshots the theory that authored it). Predictions inside an Observation
are tagged by `label` ("self" / "adversary"), so the theory behind any
given Prediction is recoverable from its containing Round — no need to
duplicate the Theory snapshot inside Prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.arbiter_verdict import ArbiterVerdict
from src.experiment import Experiment
from src.metric import Estimate, Metric
from src.theory import Theory


class Prediction(BaseModel):
    """One theory's predicted metric value for the parent Observation.

    The value is the Observation's `metric` evaluated on data simulated under
    some theory (NOT on the Observation's collected data — that's
    `real_value`). The theory itself is identified by `label` plus the
    containing Round (e.g. `label="adversary"` in obs_1 means the theory
    behind this prediction is the proposer of obs_2 in the same Round).
    """

    value: float | None = None
    variance: float | None = Field(
        default=None,
        description=(
            "Between-subject variance (population, ddof=0) of the same "
            "metric, re-applied per subject_id and aggregated. Carries the "
            "noise / consistency of the prediction alongside the point "
            "estimate `value`. None when the metric can't be applied to a "
            "single-subject slice."
        ),
    )
    label: str | None = Field(
        default=None,
        description="Role tag tying this prediction to a theory in its Round (e.g. 'self' / 'adversary').",
    )

    def as_estimate(self) -> Estimate:
        """Return the (value, variance) pair as an `Estimate` namedtuple,
        for symmetric handling with `Metric.__call__` results."""
        return Estimate(value=self.value, variance=self.variance)


class Observation(BaseModel):
    """Empirical evidence for one (experiment, metric) pair.

    `data` and `real_value` are singletons: one collected dataset, one ground
    truth metric value. `predicted_values` is a list of `Prediction`s — the
    same metric scored against data simulated under each candidate theory.

    `data` is intentionally typed `Any` to admit any pandas DataFrame without
    pydantic trying to coerce it; serialization treats it as opaque.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    experiment: Experiment
    metric: Metric
    proposer_theory: Theory
    proposer_label: str | None = Field(
        default=None,
        description=(
            "Stable label of the AutoCog that authored this Observation "
            "(e.g. 'pi_1', 'pi_1_1', 'pi_3'). Matches the `label` field on "
            "one of this Observation's `predicted_values` (the proposer's own "
            "self-prediction); the other prediction's label identifies the "
            "adversary in this Round."
        ),
    )

    data: Any = Field(
        default=None,
        description="Collected data (simulated or human). pandas DataFrame.",
    )
    real_value: float | None = Field(
        default=None,
        description="Metric evaluated on `data`. One number per Observation.",
    )
    real_variance: float | None = Field(
        default=None,
        description=(
            "Between-subject variance of the metric on `data`. Same "
            "semantics as `Prediction.variance`; reports the consistency "
            "of the real-data measurement around `real_value`."
        ),
    )

    def real_as_estimate(self) -> Estimate:
        """The Observation's real_value packaged as an `Estimate`."""
        return Estimate(value=self.real_value, variance=self.real_variance)

    predicted_values: list[Prediction] = Field(
        default_factory=list,
        description="Metric values predicted by each theory ever evaluated.",
    )

    # --- mutation helpers ----------------------------------------------------

    def set_data(self, data: pd.DataFrame, *, evaluate: bool = True) -> None:
        """Attach collected data and (by default) compute `real_value`.

        The metric is called on a copy because LLM-generated metrics often
        mutate the input frame in place (e.g. add `stim_str` / `correct`
        helper columns). We never want those temporaries to bleed into the
        persisted `obs.data`.
        """
        self.data = data
        if evaluate:
            try:
                est = self.metric(data.copy())
                self.real_value = (
                    float(est.value) if est.value is not None else None
                )
                self.real_variance = (
                    float(est.variance) if est.variance is not None else None
                )
            except Exception as e:
                print(
                    f"Observation.set_data: metric eval failed "
                    f"({type(e).__name__}: {e}); real_value/variance left as None."
                )
                self.real_value = None
                self.real_variance = None

    def add_prediction(
        self,
        theory: Theory,
        *,
        label: str | None = None,
        n_runs: int = 200,
    ) -> Prediction:
        """Simulate `experiment` under `theory`, score `metric`, append the value.

        Only `(value, label)` is stored — the Theory itself isn't snapshotted
        on the Prediction (it's pinned at the Round level via the proposer
        theories). If a prediction with this exact `label` already exists it
        is overwritten in place (kept idempotent for role-tagged predictions).
        """
        sim = self.experiment.simulate(theory, n_runs=n_runs)
        try:
            # `.copy()` shields `sim` from in-place mutations by LLM metrics
            # (mostly defensive — we don't persist `sim`, but stay consistent).
            est = self.metric(sim.copy())
            value: float | None = (
                float(est.value) if est.value is not None else None
            )
            variance: float | None = (
                float(est.variance) if est.variance is not None else None
            )
        except Exception as e:
            print(
                f"Observation.add_prediction: metric eval failed "
                f"({type(e).__name__}: {e}); value/variance left as None."
            )
            value = None
            variance = None

        if label is not None:
            for p in self.predicted_values:
                if p.label == label:
                    p.value = value
                    p.variance = variance
                    return p
        pred = Prediction(value=value, variance=variance, label=label)
        self.predicted_values.append(pred)
        return pred

    def record_prediction(
        self,
        value: float | None,
        *,
        variance: float | None = None,
        label: str | None = None,
    ) -> Prediction:
        """Store an already-computed predicted value (no simulation).

        `variance` is the optional between-subject variance produced
        alongside `value` by `Metric.__call__` (paired half of the
        `Estimate` namedtuple). Pass `None` if it isn't available — the
        Prediction will still record the point estimate.
        """
        pred = Prediction(value=value, variance=variance, label=label)
        self.predicted_values.append(pred)
        return pred

    # --- views ---------------------------------------------------------------

    def prediction_by_label(self, label: str) -> Prediction | None:
        """Return the most recently appended Prediction with `label`, if any.

        The "self" / "adversary" labels combined with the containing Round
        identify the originating theory.
        """
        for p in reversed(self.predicted_values):
            if p.label == label:
                return p
        return None

    def distances(self) -> dict[str, float | None]:
        """Per-label `|prediction.value - real_value|`.

        `None` when either side is missing (no `real_value` yet, or the
        prediction's simulation/metric eval failed). Predictions without a
        `label` are skipped — only labelled theories enter the matrix.
        """
        out: dict[str, float | None] = {}
        for p in self.predicted_values:
            if p.label is None:
                continue
            if self.real_value is None or p.value is None:
                out[p.label] = None
            else:
                out[p.label] = abs(p.value - self.real_value)
        return out

    def normalized_distances(self) -> dict[str, float]:
        """Per-label min-max normalized distance, with NaNs replaced by 1.0.

        Maps each labelled prediction to `[0, 1]` per:
          - `0` if it is the closest theory to `real_value` on this metric,
          - `1` if it is the farthest,
          - linear interpolation in between.
        Edge cases:
          - all theories tied (`d_min == d_max`): everyone gets `0.0`.
          - prediction (or `real_value`) was `None`: the cell is treated as
            the worst possible fit and gets `1.0`, never NaN.
          - no labelled predictions at all: returns `{}`.
        """
        raw = self.distances()
        valid = [d for d in raw.values() if d is not None]
        if not raw:
            return {}
        if not valid:
            return {label: 1.0 for label in raw}
        d_min, d_max = min(valid), max(valid)
        span = d_max - d_min
        out: dict[str, float] = {}
        for label, d in raw.items():
            if d is None:
                out[label] = 1.0
            elif span == 0:
                out[label] = 0.0
            else:
                out[label] = (d - d_min) / span
        return out


class Round(BaseModel):
    """An ordered group of Observations from one debate iteration.

    In the standard two-pi setup, each Round holds two Observations — one
    per pi, in the order the pis proposed. The Arbiter consumes a Round to
    judge between the two theories.

    A Round also captures the *outcome* of arbitration: `next_theory` is
    the replacement theory the loop should carry into the next round
    (produced by the `Improver` on a `"new_model"` verdict, or the
    `TheoryGenerator` on a `"new_theory"` verdict), and `next_theory_idx`
    (1 or 2) records which pi-slot it replaces. Both are `None` until the
    arbitration / regeneration step has run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observations: list[Observation] = Field(default_factory=list)
    verdict: ArbiterVerdict | None = Field(
        default=None,
        description=(
            "Arbiter's structured verdict for this round (None until "
            "arbitration runs). Persisted so a crash between arbitration and "
            "regeneration doesn't force a re-arbitration on resume — the "
            "saved verdict is reused, keeping the target_theory_idx stable."
        ),
    )
    next_theory: Theory | None = Field(
        default=None,
        description="Replacement theory chosen for the next round (None until arbitration runs).",
    )
    next_theory_idx: int | None = Field(
        default=None,
        description="Which pi slot (1 or 2) `next_theory` replaces.",
    )
    next_theory_label: str | None = Field(
        default=None,
        description=(
            "Stable label assigned to `next_theory` (e.g. 'pi_1_1' for an "
            "improved model under pi_1, or 'pi_3' for a brand-new theory)."
        ),
    )

    def add(self, obs: Observation) -> Observation:
        self.observations.append(obs)
        return obs

    def set_verdict(self, verdict: ArbiterVerdict) -> None:
        """Record the arbiter's verdict for this round."""
        self.verdict = verdict

    def set_next_theory(self, theory: Theory, *, idx: int, label: str) -> None:
        """Record the theory (and its label) replacing pi-slot `idx` next round."""
        self.next_theory = theory
        self.next_theory_idx = idx
        self.next_theory_label = label

    def observation_by_label(self, label: str) -> Observation | None:
        """First Observation in this Round authored by pi `label`, if any."""
        for o in self.observations:
            if o.proposer_label == label:
                return o
        return None

    @property
    def proposer_labels(self) -> list[str | None]:
        """Labels of the pis that authored each Observation, in order."""
        return [o.proposer_label for o in self.observations]

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self):  # type: ignore[override]
        return iter(self.observations)

    def __getitem__(self, idx: int) -> Observation:
        return self.observations[idx]


class Observations(BaseModel):
    """The pool of all evidence collected across the loop.

    Round-wise indexing: `pool.rounds[i].observations[j]` (or
    `pool[i][j]`). Both pis (and any future agents) write into the same
    pool via the round currently in progress; "the current state" of the
    loop is just `pool.latest_round`.

    Persistence layout under `workspace`:
      <workspace>/state.json                       — manifest (everything except `data`)
      <workspace>/data/round_RRR_obs_OO.jsonl     — per-observation collected data
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rounds: list[Round] = Field(default_factory=list)
    workspace: Path | None = Field(default=None, exclude=True)

    # --- mutation -----------------------------------------------------------

    def start_round(self) -> Round:
        """Append and return a fresh Round. Caller fills it via `round.add(obs)`."""
        r = Round()
        self.rounds.append(r)
        return r

    def add_round(self, round: Round) -> Round:
        """Append a pre-built Round and auto-save if `workspace` is set."""
        self.rounds.append(round)
        if self.workspace is not None:
            self.save()
        return round

    # --- views --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rounds)

    def __iter__(self):  # type: ignore[override]
        return iter(self.rounds)

    def __getitem__(self, idx: int) -> Round:
        return self.rounds[idx]

    @property
    def latest_round(self) -> Round:
        """The most recently started Round."""
        return self.rounds[-1]

    @property
    def all_observations(self) -> list[Observation]:
        """Flat list of every Observation, across rounds, in insertion order."""
        return [o for r in self.rounds for o in r.observations]

    @property
    def experiments(self) -> list[Experiment]:
        """Flat list of past experiments (used as a proposal ledger)."""
        return [o.experiment for o in self.all_observations]

    @property
    def metrics(self) -> list[Metric]:
        """Flat list of past metrics (used as a proposal ledger)."""
        return [o.metric for o in self.all_observations]

    # --- prediction matrix --------------------------------------------------

    def theory_registry(self) -> dict[str, Theory]:
        """Map every label that has authored or been admitted to its Theory.

        Walks (a) every observation's `proposer_label` -> `proposer_theory`,
        and (b) every round's `next_theory_label` -> `next_theory`. The same
        label is expected to point to the same Theory across the run; later
        entries overwrite earlier ones on collision.
        """
        registry: dict[str, Theory] = {}
        for r in self.rounds:
            for o in r.observations:
                if o.proposer_label is not None and o.proposer_theory is not None:
                    registry[o.proposer_label] = o.proposer_theory
            if r.next_theory_label is not None and r.next_theory is not None:
                registry[r.next_theory_label] = r.next_theory
        return registry

    def theory_distance_vectors(self) -> dict[str, list[float]]:
        """Per-theory vector of normalized distances across observations.

        Component `i` is `obs_i.normalized_distances()[label]` in pool
        insertion order. After `backfill_predictions()` the matrix is dense,
        so every label appears in every observation; defensively a missing
        cell is treated as `1.0` (worst). Observations with no `real_value`
        contribute `1.0` for every label (no signal there either way).
        """
        labels = sorted(self.theory_registry().keys())
        vectors: dict[str, list[float]] = {label: [] for label in labels}
        for obs in self.all_observations:
            nd = obs.normalized_distances()
            for label in labels:
                vectors[label].append(nd.get(label, 1.0))
        return vectors

    def theory_scores(self) -> dict[str, float]:
        """Per-theory score in `[0, 1]`, higher = better.

        For each label, take the L2 norm of its `theory_distance_vectors`
        entry; divide by the longest such norm across all labels (so the
        worst-fitting theory normalises to 1.0); then `score = 1 - that`.
        Edge case: if every vector has norm 0 (everyone tied at 0 on every
        observation), all labels score `1.0`.
        """
        import math

        vectors = self.theory_distance_vectors()
        if not vectors:
            return {}
        norms = {
            label: math.sqrt(sum(d * d for d in v))
            for label, v in vectors.items()
        }
        max_norm = max(norms.values())
        if max_norm == 0:
            return {label: 1.0 for label in norms}
        return {label: 1.0 - n / max_norm for label, n in norms.items()}

    def backfill_predictions(self, *, n_runs: int = 50) -> int:
        """Densify the (observation × theory) prediction matrix.

        For every (obs, label) cell that is missing, simulate `theory` on
        `obs.experiment`, score `obs.metric`, and append a Prediction tagged
        with `label`. Returns the number of new predictions added (0 if the
        matrix was already dense). Idempotent.
        """
        registry = self.theory_registry()
        if not registry:
            return 0
        added = 0
        for obs in self.all_observations:
            existing = {p.label for p in obs.predicted_values}
            for label, theory in registry.items():
                if label in existing:
                    continue
                obs.add_prediction(theory, label=label, n_runs=n_runs)
                added += 1
        return added

    # --- persistence --------------------------------------------------------

    def save(self) -> None:
        """Dump pool state to `<workspace>/state.json` (+ data sidecars).

        Online runs may attach additional per-observation sidecars on the
        DataFrame's ``attrs`` mapping:

        * ``participant_report`` (pd.DataFrame) — one row per recruited
          subject (kept *and* dropped) with prolific_pid, slot_key, drop
          flags, and the reason. Saved as ``<rel>_participants.csv`` next
          to the canonical jsonl. This is the file you reach for to pay
          everyone, including those filtered out for quality.
        * ``excluded_trials`` (pd.DataFrame) — raw per-trial rows for
          dropped subjects, with the drop reason attached. Saved as
          ``<rel>_excluded.jsonl`` so the canonical jsonl stays clean for
          modeling but the dropped data is still auditable.

        Both sidecars are skipped if the corresponding attr is missing or
        empty (e.g. simulated observations, where there is no human filter).
        """
        if self.workspace is None:
            raise ValueError("save() requires a workspace.")
        self.workspace.mkdir(parents=True, exist_ok=True)
        data_dir = self.workspace / "data"
        rounds_manifest: list[dict[str, Any]] = []
        for r_idx, r in enumerate(self.rounds):
            obs_manifest: list[dict[str, Any]] = []
            for o_idx, o in enumerate(r.observations):
                d = o.model_dump(exclude={"data"}, serialize_as_any=True)
                if isinstance(o.data, pd.DataFrame):
                    data_dir.mkdir(parents=True, exist_ok=True)
                    rel = f"data/round_{r_idx:03d}_obs_{o_idx:02d}.jsonl"
                    o.data.to_json(self.workspace / rel, orient="records", lines=True)
                    d["data_path"] = rel

                    rel_base = rel[:-len(".jsonl")]
                    report = o.data.attrs.get("participant_report")
                    if isinstance(report, pd.DataFrame) and not report.empty:
                        report_rel = f"{rel_base}_participants.csv"
                        report.to_csv(self.workspace / report_rel, index=False)
                        d["participant_report_path"] = report_rel
                    excluded = o.data.attrs.get("excluded_trials")
                    if isinstance(excluded, pd.DataFrame) and not excluded.empty:
                        excluded_rel = f"{rel_base}_excluded.jsonl"
                        excluded.to_json(
                            self.workspace / excluded_rel,
                            orient="records",
                            lines=True,
                        )
                        d["excluded_trials_path"] = excluded_rel
                obs_manifest.append(d)
            round_dump: dict[str, Any] = {"observations": obs_manifest}
            if r.verdict is not None:
                round_dump["verdict"] = r.verdict.model_dump()
            if r.next_theory is not None:
                round_dump["next_theory"] = r.next_theory.model_dump()
                round_dump["next_theory_idx"] = r.next_theory_idx
                round_dump["next_theory_label"] = r.next_theory_label
            rounds_manifest.append(round_dump)
        (self.workspace / "state.json").write_text(
            json.dumps({"rounds": rounds_manifest}, indent=2)
        )

    @classmethod
    def load(
        cls,
        workspace: str | Path,
        *,
        experiment_class: type[Experiment],
    ) -> "Observations":
        """Load a previously saved pool from `<workspace>/state.json`.

        `experiment_class` is required to reconstruct the concrete Experiment
        subclass — the manifest only carries its instance fields, not the
        class itself.
        """
        workspace = Path(workspace)
        state_path = workspace / "state.json"
        if not state_path.exists():
            return cls(workspace=workspace)
        payload = json.loads(state_path.read_text())
        rounds = [
            Round(
                observations=[
                    cls._observation_from_dump(o, experiment_class=experiment_class, workspace=workspace)
                    for o in r.get("observations", [])
                ],
                verdict=(
                    ArbiterVerdict(**r["verdict"]) if r.get("verdict") is not None else None
                ),
                next_theory=(
                    Theory(**r["next_theory"]) if r.get("next_theory") is not None else None
                ),
                next_theory_idx=r.get("next_theory_idx"),
                next_theory_label=r.get("next_theory_label"),
            )
            for r in payload.get("rounds", [])
        ]
        return cls(rounds=rounds, workspace=workspace)

    @staticmethod
    def _observation_from_dump(
        o: dict[str, Any],
        *,
        experiment_class: type[Experiment],
        workspace: Path,
    ) -> Observation:
        predictions = [
            Prediction(
                value=p.get("value"),
                variance=p.get("variance"),
                label=p.get("label"),
            )
            for p in o.get("predicted_values", [])
        ]
        return Observation(
            experiment=experiment_class(**o["experiment"]),
            metric=Metric(**o["metric"]),
            proposer_theory=Theory(**o["proposer_theory"]),
            proposer_label=o.get("proposer_label"),
            predicted_values=predictions,
            real_value=o.get("real_value"),
            real_variance=o.get("real_variance"),
            data=(
                pd.read_json(workspace / o["data_path"], orient="records", lines=True)
                if o.get("data_path") and (workspace / o["data_path"]).exists()
                else None
            ),
        )
