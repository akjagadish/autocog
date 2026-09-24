# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=28):** [1.0, 1.0, 1.0, 1.0, 0.65, 0.65, 0.65, 0.65, 0.65, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 10: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 22: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** ADVERSARIAL LOGIC. PAHI (advocated) and SCWI (competitor) are structurally near-identical anti-validity additive integrators: a rating of 1 counts AGAINST an option, tied cues cancel, choice is a softmax over the anti-load difference plus a symmetric lapse. The ONLY substantive difference is the validity-to-weight map: PAHI uses w(v)=v^gamma with gamma in [1.4, 2.4]; SCWI uses w(v)=v+tau*v^2 with tau in [0.25, 0.45]. Three mathematical facts drive the design. (1) UNIFORM STEEPNESS ORDERING: SCWI's log-local elasticity is (1+2*tau*v)/(1+tau*v), which lies in [1.11, 1.31] for ALL v in [0.5, 1] and ALL tau in its box; PAHI's elasticity is exactly gamma >= 1.4. The elasticity ranges are DISJOINT, so for every v < 1, PAHI's top-to-low weight ratio w(1)/w(v) strictly exceeds SCWI's. (2) CROSSING-WINDOW DISJOINTNESS AT r=2: fix counts of 2 top cues (v=1.00, where PAHI's weight is exactly 1, gamma-invariant - a fixed 'currency') versus 4 low cues at validity v. The choice flips at v* solving 2*w(v) = w(1). SCWI: v* in [0.550, 0.576] across its ENTIRE parameter box - a 0.026-wide, effectively parameter-free window. PAHI: v* = 0.5^(1/gamma) in [0.610, 0.749]. The windows are disjoint, so each subject's measured flip validity classifies the theories with zero free parameters. (3) FLATNESS AT r=2.5: at counts 2:5, SCWI's flip validity v* in [0.449, 0.477] lies BELOW the floor 0.5, so SCWI must prefer the top-cue option for EVERY v in [0.5, 1] and every parameter - a parameter-free flat-H prediction. PAHI flips at 0.4^(1/gamma) in [0.520, 0.683] and must prefer the LOW option at v=0.50 for every gamma - parameter-free flat-L at the floor. Opposite pinned predictions. WHERE THIS SITS RELATIVE TO PRIOR DESIGNS: experiment [4] swept the COUNT RATIO at fixed validities (1.0 vs 0.5) and used a single knife-edge (d=5). This design is its TRANSPOSE: it fixes the count ratio and sweeps the LOW-CUE VALIDITY, making the statistic the CROSSING VALIDITY (its location, its between-subject spread, and its cross-ratio consistency), plus amplified stacked kills at validity mixtures (1.0/0.60/0.50 and 1.0/0.55) that [4] never touched. Only one cell (2x1.00 vs 5x0.50) coincides with [4]'s d=5 anchor, retained deliberately as a cross-experiment replication cell. INSTRUMENT: 28 experts - 4 at 1.00, 5 at 0.65, 5 at 0.60, 9 at 0.55, 5 at 0.50. Two ladders at r=2 (2:4) and r=2.5 (2:5) over v in {0.50, 0.55, 0.60, 0.65}, two stacked kills (4x1.00 vs 4x0.60+5x0.50; 4x1.00 vs 9x0.55, i.e. r=2.25 sitting inside the (2.11, 2.32) gap at v=0.55), all mirrored, plus tie / lapse / agreement controls. 24 unique pairs x K = 96//24 = 4 = 96 trials. KILL CELLS (parameter-free sign separation; SCWI evidence is linear in tau so endpoint checks suffice, PAHI evidence is monotone decreasing in gamma on every kill cell): R2-60 (2x1.00 vs 4x0.60): SCWI evidence +0.148..+0.26, P(top) in [0.535, 0.603]; PAHI evidence -0.044..-0.826, P(top) <= 0.487. This cell is the r=2 crossing detector: SCWI's crossing (<=0.576) is below 0.60, PAHI's (>=0.610) is above. R25-50: SCWI +0.163..+0.313, P >= 0.538; PAHI -0.106..-1.053, P <= 0.468. STK (4x1.00 vs 4x0.60+5x0.50): SCWI +0.311..+0.573, P in [0.572, 0.742]; PAHI -0.149..-1.879, P <= 0.456. STK2 (4x1.00 vs 9x0.55): SCWI +0.375..+0.630, P in [0.588, 0.733]; PAHI -0.103..-1.856, P <= 0.469. R2-55 is a REVERSE kill: SCWI sits exactly at its crossing (evidence -0.156..+0.002, P(top) in [0.464, 0.501], i.e. pinned at chance) while PAHI is confidently opposite (evidence -0.268..-1.047, P(top) <= 0.420). R25-55: SCWI P in [0.62, 0.73]; PAHI prefers the low option for 87% of its gamma range. GAMMA-MAPPING RUNGS: R2-65 (SCWI [0.61, 0.70]; PAHI splits at gamma=1.61, 79% prefer the low option), R25-60 (SCWI [0.70, 0.82]; PAHI splits at gamma=1.79, 61% low), R25-65 (SCWI [0.76, 0.88]; PAHI splits at gamma=2.13, 73% top). POPULATION SHAPE AND HETEROGENEITY SIGNATURES: plotting P(top-option) versus low validity at r=2, SCWI crosses 0.5 inside (0.55, 0.60) for EVERY subject with near-zero between-subject spread (the whole tau box moves v* by only 0.026), whereas PAHI crosses in (0.60, 0.75) with crossing = 0.5^(1/gamma) spread over 0.14 validity units - a fivefold variance dissociation in crossing location. At r=2.5, SCWI never crosses (flat above 0.5 down to v=0.50) while PAHI crosses inside (0.52, 0.68). Moreover PAHI's two crossings must be jointly consistent with a single gamma: v*_{2.5} = 0.8^(1/gamma) * v*_2, i.e. v*_{2.5} in [0.85, 0.91] x v*_2; SCWI has no observable r=2.5 crossing at all. CONTROLS: pair 21 (identical mixed vectors) pins side/response bias at exactly 0.500 for both theories; pair 22 (all-1 vs all-0) calibrates the lapse (both predict P(all-1 option) ~ epsilon/2, confirming anti-validity processing); pairs 23/24 are confident agreement cells (e.g. one 0.50-cue 1 vs two 1.00-cue 1s) calibrating the shared softmax/lapse on the anti-validity side. DECISION RULES: (i) SCWI is supported, and PAHI falsified, if P(top) > 0.5 on all four kill cells (especially STK/STK2 >= 0.57, which PAHI cannot exceed at 0.456/0.469 for any admissible parameters), the r=2 crossing lies in (0.55, 0.60) with tight between-subject clustering, r=2.5 is flat-H all the way down to v=0.50, and R2-55 sits at ~0.50. (ii) PAHI is supported, and SCWI falsified, if P(top) < 0.5 on the kill cells (STK/STK2 <= 0.47, below SCWI's 0.572/0.588 floor), the r=2 crossing lies at or above 0.61 (low-option choices persist through v=0.60 and top-option choices emerge only at 0.65+), an r=2.5 crossing is observed inside (0.52, 0.68) and is consistent with the r=2 crossing through a single gamma, R2-55 is clearly below 0.5, and crossing locations vary widely across subjects. (iii) If subjects prefer options with MORE high-validity 1s (pro-validity integration), or crossings appear below 0.55, or the profile fits neither map, both pure forms are falsified.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI). Subjects integrate all cues additively into a per-option subjective value, but a rating of 1 is treated as evidence AGAINST the option (comprehension inversion, s = -1 for ~99% of subjects). The anti-weight of cue j is a free power law of the instructed validity, |w_j| = v_j^gamma, with gamma a free per-subject parameter drawn from a heterogeneous population distribution (gamma ~ 1.4-2.4, mildly supralinear on average). Choice is a softmax over beta * s * sum_j v_j^gamma * rating_j with a symmetric lapse epsilon. Because gamma, beta, and epsilon are drawn broadly per subject, the population reproduces both the mean-level signatures (negative Exp-1 slope, positive Exp-2 dissociation, positive Exp-3 slope, low Exp-4 conflict-following) and the large between-subject variance seen in the real data.

**Parameters:**
- beta: [1.5, 4.0]
- gamma: [1.4, 2.4]
- epsilon: [0.06, 0.20]
- sign_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = s * v_j**gamma per unit
    # rating, where s = -1 for the (vast majority of) comprehension-inverted
    # subjects and +1 for the rare pro-validity subject; gamma is a free
    # power-law exponent applied to the instructed validity. Option values
    # are the cue-weighted sums; choice is a numerically stable softmax over
    # beta * values, mixed with a symmetric lapse epsilon. History is
    # ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PAHI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s_txt = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s_txt.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    sign_seed = float(parameters["sign_seed"])

    # Heterogeneous comprehension inversion: with population probability
    # 0.99 a subject inverts the rating direction (a 1 counts AGAINST the
    # option); the remaining subjects read ratings pro-validity.
    s = -1.0 if sign_seed < 0.99 else 1.0

    # Power-law anti-validity cue weights: the anti-weight of expert j is
    # v_j**gamma (on average mildly supralinear in the instructed validity).
    w = s * np.power(val, gamma)

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

**Parameters:**
- beta: [1.1, 1.7]
- tau: [0.25, 0.45]
- epsilon: [0.05, 0.15]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = -(v_j + tau * v_j^2)
    # per unit rating (anti-validity weighting, supralinear in the
    # instructed validity). Option values are the cue-weighted sums;
    # choice is a numerically stable softmax over beta * values, mixed
    # with a symmetric lapse. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])

    # Signed (anti-validity) cue weights: a rating of 1 is evidence
    # AGAINST an option, and the inversion strength grows supralinearly
    # with the instructed validity of the expert.
    w = -(val + tau * np.square(val))

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # \"Top-cue adherence\" on steepness-diagnostic ladder cells.\n    # The design's validity vector: 4 cues at 1.00, 5 at 0.65, 5 at 0.60,\n    # 14 at 0.50 (hardcoded from the experimental design).\n    VALID = np.array([1.0] * 4 + [0.65] * 5 + [0.60] * 5 + [0.50] * 14)\n    NF = VALID.size\n\n    def to_vec(x):\n        if isinstance(x, str):\n            s = x.strip().strip('[]()')\n            return np.array([float(t) for t in s.replace(',', ' ').split()])\n        return np.asarray(x, dtype=float).ravel()\n\n    # Excluded cells (signature = (top-cue count, sorted low-cue validities)):\n    #  (2, 4x0.50): both theories prefer the low option -> no separation;\n    #  (2, 5x0.60) and (2, 5x0.65): PAHI splits across its gamma range on\n    #  these cells -> they add between-subject noise without adding gap.\n    EXCLUDE = {\n        (2, (0.5, 0.5, 0.5, 0.5)),\n        (2, (0.6, 0.6, 0.6, 0.6, 0.6)),\n        (2, (0.65, 0.65, 0.65, 0.65, 0.65)),\n    }\n\n    hits = []\n    for a_raw, b_raw, resp in zip(\n        data['option_a_ratings'], data['option_b_ratings'], data['response']\n    ):\n        a = to_vec(a_raw)\n        b = to_vec(b_raw)\n        if a.size != NF or b.size != NF:\n            continue\n        ta = int(np.sum(a[:4] > 0.5))\n        tb = int(np.sum(b[:4] > 0.5))\n        # Exactly one option must carry 1s on the perfect-validity cues.\n        if ta > 0 and tb > 0:\n            continue\n        if ta == 0 and tb == 0:\n            continue\n        if ta > 0:\n            top_vec, low_vec, top_idx, tc = a, b, 0, ta\n        else:\n            top_vec, low_vec, top_idx, tc = b, a, 1, tb\n        if tc not in (2, 4):\n            continue\n        # Top option must have 1s ONLY on perfect cues; low option none.\n        if np.sum(top_vec[4:] > 0.5) > 0:\n            continue\n        if np.sum(low_vec[:4] > 0.5) > 0:\n            continue\n        low_mask = low_vec > 0.5\n        if int(np.sum(low_mask)) < 4:\n            continue\n        sig = (tc, tuple(sorted(float(x) for x in np.round(VALID[low_mask], 2))))\n        if sig in EXCLUDE:\n            continue\n        hits.append(1.0 if int(resp) == top_idx else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "CORE CONTRAST. Both theories are anti-validity additive integrators differing only in the validity-to-weight map. The decisive quantity is the weight ratio between a perfect cue (v=1.00) and a low cue: SCWI's map w(v)=v+tau*v^2 has log-elasticity (1+2*tau*v)/(1+tau*v) in [1.11, 1.31], i.e. nearly linear, so w(1)/w(0.6) is only 1.81-1.90 and w(1)/w(0.5) only 2.04-2.58. PAHI's map w(v)=v^gamma gives w(1)/w(0.6) = (1/0.6)^gamma in [2.05, 3.14] and w(1)/w(0.5)=2^gamma in [2.64, 5.28]. CONSEQUENCE on the ladder cells: on every 2-perfect-cue vs 4/5-low-cue cell and every 4-perfect-cue vs 9-low-cue cell, the low option's total anti-weight EXCEEDS the top option's under SCWI (so SCWI picks the TOP option, since more anti-weight = worse) but is BELOW the top option's under PAHI (so PAHI picks the LOW option). Concretely: cell 2-vs-4x0.60: SCWI evidence +0.148..+0.26 (P(top) ~0.56), PAHI evidence -0.046..-0.728 (P(top) ~0.29); cell 2-vs-5x0.50: SCWI +0.16..+0.31 (P ~0.57), PAHI -0.105..-1.053 (P ~0.25); cell 4-vs-(4x0.60+5x0.50): SCWI +0.31..+0.57 (P ~0.63), PAHI -0.149..-1.78 (P ~0.20); cell 4-vs-9x0.50: SCWI -0.29..+0.06 (P ~0.47), PAHI -0.59..-2.29 (P ~0.12); cell 2-vs-4x0.65: SCWI +0.46..+0.52 (P ~0.65), PAHI mostly negative (P ~0.39). METRIC: the proportion of choices of the top-cue option (the option whose 1s sit exclusively on the four v=1.00 cues), pooled over exactly these five diagnostic cells (12 unique trials x 4 reps = 48 trials per subject). Predicted pooled value: PAHI ~0.25, SCWI ~0.58 - a ~0.33 gap. CELL SELECTION FOR VARIANCE CONTROL: the 2-vs-4x0.50 cell is excluded because BOTH theories prefer the low option there (no separation); the 2-vs-5x0.60 and 2-vs-5x0.65 cells are excluded because PAHI's preference flips inside its gamma range (gamma=1.63 and 2.13), so those cells inflate PAHI's between-subject variance without enlarging the mean gap. The retained cells are ones where PAHI prefers the low option for essentially its ENTIRE parameter box, so its per-subject values are one-sided (range roughly 0.05-0.43, variance ~0.017 including binomial noise over 48 trials), while SCWI's per-subject values are tightly pinned (0.55-0.64, variance ~0.005, dominated by binomial noise) because its whole tau box moves the evidence by only ~0.15. Welch t at N=25: t ~ 10-12, p << 0.001. The metric is a simple proportion, well-defined on any subject's slice and on the pooled data, uses no theory-internal parameters, and the trial classification is purely structural (which option carries the perfect-cue 1s), so it is robust to trial randomization and response coding."
}
```

## Usage

```json
{
  "input_tokens": 9851,
  "output_tokens": 39899,
  "reasoning_tokens": 38186,
  "provider": "Sail Research"
}
```
