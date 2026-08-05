"""Run-level tuning knobs for the inner critique loops.

These were previously hardcoded as default arguments on
`Improver.propose_model` and `TheoryGenerator.propose_theory`. Centralising
them here lets you tweak a single file when you want to change behaviour
across a run; per-call overrides on those methods still win.
"""

from __future__ import annotations

# Improver (model-only regeneration under a fixed theory description).
IMPROVER_MAX_ATTEMPTS: int = 10
"""How many times Improver re-asks the LLM per critique iteration when the
proposed `Model` source fails to compile."""

# TheoryGenerator (brand-new theory + model).
THEORY_GENERATOR_MAX_ATTEMPTS: int = 10
"""How many times TheoryGenerator re-asks the LLM per critique iteration
when the proposed `Theory` source fails to compile."""

# Shared inner critique loop (propose -> simulate -> feedback).
MAX_CRITIQUE_ITERS: int = 10
"""Outer iterations of the propose -> simulate -> feedback loop in both
Improver and TheoryGenerator before falling back to the best iteration."""

FEEDBACK_N_RUNS: int = 50
"""Subjects per simulation when scoring a candidate against the
observation pool inside the critique loop."""

REAL_N_SUBJECTS: int = 25
"""How many real (human / ground-truth) subjects are run per Observation.

This is the N used to collect `Observation.real_value` in `main` (the
ground-truth theory stands in for humans). The same N is reused as the
sample size in `AutoCog.propose_round`'s metric-acceptance check: a metric
is admitted only if a Welch's t-test on the two theories' simulated
estimates and between-subject variances says they would be statistically
distinguishable with this many participants. Centralised so the simulation
threshold and the human-data collection can never drift out of sync."""
