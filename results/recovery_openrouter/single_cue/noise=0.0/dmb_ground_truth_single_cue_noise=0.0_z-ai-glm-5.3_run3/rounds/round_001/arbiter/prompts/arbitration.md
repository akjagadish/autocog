# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Noisy Validity-weighted evidence integration with capacity-interference and one-reason lapses (NVW). People do not use a single fixed heuristic; they pool all the expert ratings they can, but each cue's contribution is weighted by its validity raised to a steepness exponent gamma (gamma = 0 recovers Tallying, gamma -> infinity recovers Take-The-Best, so both classic heuristics are limiting cases). Integration is capacity-limited: beyond a working-memory span of about four cues, every additional cue that must be pooled multiplies the reliability (effective inverse temperature) of the integrated comparison by an interference factor delta < 1, so choice consistency is design-dependent and falls sharply as feature count grows. Each cue is additionally attended only with probability alpha (lapses of attention). On a minority of trials — with a probability omega that is modulated by how strongly the most valid discriminating cue's validity dominates the validity of the opposing side's cues — the agent abandons integration and makes a one-reason decision based solely on that top cue. The final choice is a softmax (at the capacity-scaled inverse temperature) over the attended, validity-weighted evidence difference, mixed with an independent lapse to uniform guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # NVW: Noisy Validity-weighted integration with capacity-interference
    # and one-reason lapses.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"NVW expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n}."
        )

    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    delta = float(parameters["delta"])
    epsilon = float(parameters["epsilon"])
    omega_base = float(parameters["omega_base"])
    omega_dom = float(parameters["omega_dom"])

    # (1) Validity-steepness weights: gamma = 0 -> Tallying (equal weights),
    # gamma -> infinity -> Take-The-Best (top cue only).
    w = np.power(v, gamma)

    # (3) Attention: each cue independently attended with probability alpha.
    att = np.random.random(n) < alpha
    a_win = (a > b) & att
    b_win = (b > a) & att
    disc = a_win | b_win

    # (3) Capacity / interference: beyond a working-memory span of ~4 cues,
    # each additional cue that must be pooled multiplies the reliability
    # (effective inverse temperature) of the integrated evidence by delta.
    # This makes decision noise grow with feature count.
    excess = max(0, n - 4)
    beta_eff = beta * (delta ** excess)

    def _sig(z):
        # Numerically stable logistic.
        return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))

    if not np.any(disc):
        # No attended discriminating cue: pure guess.
        p_a = 0.5
    else:
        # (1) Integration: softmax over the weighted evidence difference.
        d = float(np.sum(w[a_win]) - np.sum(w[b_win]))
        p_int = float(_sig(beta_eff * d))

        # (2) One-reason mode: the single most valid attended
        # discriminating cue decides (TTB-like), with confidence scaled
        # by that cue's weight and the effective sensitivity.
        idx = np.where(disc)[0]
        top = int(idx[np.argmax(v[idx])])
        s = 1.0 if a[top] > b[top] else -1.0
        p_1r = float(_sig(beta_eff * w[top] * s))

        # (2) Validity-dominance coupling for the one-reason lapse rate:
        # dominance of the top cue over the opposing side's cue validities.
        opp = b_win if s > 0 else a_win
        if np.any(opp):
            dom = float(v[top] - np.mean(v[opp]))
        else:
            dom = float(v[top] - 0.5)
        omega_t = float(np.clip(omega_base + omega_dom * dom, 0.0, 0.95))

        p_a = (1.0 - omega_t) * p_int + omega_t * p_1r

    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent lapse toward uniform guessing.
    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_2
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.6]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 10: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 11: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 12: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 13: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 14: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 15: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 16: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 18: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 19: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 20: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 21: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 22: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 23: A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  trial 24: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 25: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 26: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 27: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 28: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 29: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 30: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 31: A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 32: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design dissociates NVW (validity-weighted, capacity-limited integration with one-reason lapses) from Tallying by exploiting the fact that Tallying's predicted P(A) depends ONLY on the win-count margin: for any beta and epsilon, Tallying assigns exactly ONE probability to every trial with the same (a_wins, b_wins), and exactly 0.5 to every tie trial. NVW instead assigns a graded probability ordered by the validity-weighted evidence difference d = sum(v^gamma over A-won cues) - sum(v^gamma over B-won cues), plus a small one-reason tilt toward the highest-validity discriminating cue. I use n=5 (one cue beyond the working-memory span, so the interference factor delta is active but integration remains informative: beta_eff = beta*delta in [0.9, 3.3]) with a wide validity spread [0.95, 0.85, 0.75, 0.65, 0.55]. The design contains four dissociation axes, each with mirrored (A/B-swapped) versions to absorb side bias: (1) MARGIN-0 VALIDITY GRADIENT (types 1-5): 2-2 tie trials where the validity allocation varies from strongly pro-A (d=+0.60: A wins the two most valid cues) through d=+0.40, +0.20, 0.00, to a top-cue-conflict probe (d=+0.10 with the top cue favoring B). Tallying predicts EXACTLY 0.5 on all five types for every parameter value; NVW predicts a monotone deviation from 0.5 whose sign is parameter-free (because v^gamma preserves validity order for any gamma>0, A winning {f0,f1} vs B winning {f3,f4} yields d>0 at any gamma in [0.7,1.3]), with magnitude sigma(beta_eff*d) roughly 0.61-0.93 at d=0.60. (2) WITHIN-MARGIN-1 SPREAD (types 6-14): nine 3-2 trials spanning d from 1.35 down to 0.15. Tallying predicts an identical probability sigma(beta) on ALL nine; NVW predicts a monotone spread of at least ~0.12 (and up to ~0.5) across the aligned subseries (types 6-10, where the top cue also favors A, so both the integration and one-reason components agree). (3) ORDER REVERSAL: NVW predicts P(type 1; margin 0, d=0.60) > P(type 14; margin 1, d=0.15) strictly for every admissible parameter setting (the d-gap is positive at all gamma, and the one-reason lapse favors A on type 1 but B on type 14), whereas Tallying predicts the strict opposite ordering (0.5 < sigma(beta)). The sign of P(margin-0, high-d) - P(margin-1, low-d) is therefore a decisive, parameter-free test. (4) MATCHED-d ALLEGIANCE FLIP: types 9 vs 11 (both d=0.75) and 10 vs 12 (both d=0.55) hold counts AND the validity-weighted evidence identical while flipping which option wins the top cue; Tallying predicts exactly zero difference, NVW predicts a small positive difference (the one-reason lapse signature, magnitude up to ~0.07). Types 15-16 are margin-2 anchors probing the steepness of the margin gradient (Tallying: sigma(2beta); NVW: sigma(beta_eff*1.90) vs sigma(beta_eff*1.00), itself a within-margin spread Tallying cannot produce). In total Tallying predicts only THREE distinct choice probabilities across all 32 pairs (margin 0, 1, 2 and their complements), while NVW predicts a graded profile ordered by d with allegiance modulations - a sharp structural signature. With 32 unique pairs, K = 96 // 32 = 3, giving exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the partial OLS coefficient of the validity-weighted evidence difference d = sum(v over A-won cues) - sum(v over B-won cues) in a regression of the chose-A indicator on [intercept, signed tally margin, d], pooled over all trials.

WHY IT IS EXACTLY ZERO UNDER TALLYING (for every beta, epsilon): Tallying's P(A) = sigmoid(beta*(a_wins - b_wins)) (plus a symmetric lapse), i.e., a function of the signed margin m alone. The design's trial set is perfectly mirrored within every margin level: the d-values of all trials sharing a given margin sum to exactly zero (margin 0: +0.55,+0.40,+0.15,0,+0.05,-0.55,-0.40,-0.15,0,-0.05; margin +1: +1.30...+0.20 with the exact negatives in margin -1; margin +2: +1.90,+1.05 with exact negatives). Hence E[d | m] = 0 for each m, so the residual of d after projecting on m is orthogonal to any function of m, including Tallying's (possibly nonlinear) mean choice function. The d-coefficient is therefore exactly zero in expectation under Tallying, for ALL of its parameter values, with only binomial sampling noise around zero.

WHY IT IS ROBUSTLY POSITIVE UNDER NVW (for every admissible parameter): NVW's P(A) is monotone increasing in the validity-weighted evidence difference (v^gamma preserves validity order for gamma in [0.7,1.3], so the sign and rank of d computed with raw v is preserved), and the design provides large within-margin spread of d (margin 0: |d| up to 0.55; margin 1: 0.20 to 1.30; margin 2: 1.05 to 1.90). The within-margin d-slope of the logistic is approximately beta_eff * p(1-p) with beta_eff = beta*delta in [0.9, 3.3], so even at the worst-case corner (beta=5, delta=0.18) the slope is >= ~0.19, and at mid parameters (~2.1) it is ~0.4-0.5. The one-reason lapse component (top-cue direction) is itself aligned with sign(d) on the high-|d| trials and only adds to the slope. The lapse epsilon (<=0.06) shrinks the slope by at most 6%.

POWER AT N=25: The pooled within-margin residual variance of d is ~0.70 and each subject contributes 96 trials, so the per-subject standard error of the d-coefficient is about sqrt(0.25/(96*0.70)) ~ 0.06 (worst case, Tallying with low beta; NVW's is smaller because its p(1-p) is smaller at extreme d). Welch's t for a mean gap of even 0.19 (absolute worst-case NVW corner) vs 0 (Tallying) is then roughly 0.19 / sqrt((0.0037+0.0022)/25) ~ 12-14, far below alpha=0.01; at typical parameters the gap (~0.4-0.5) gives t > 25. The metric works identically on the pooled DataFrame and on a single subject's 96-trial slice (the regression is well-conditioned in both cases), so the between-subject variance is well defined. The contrast is structural (Tallying literally has no within-margin sensitivity; NVW does), not a parameter tuning, and cannot be trivially satisfied by the advocated theory's noise terms.

Source:
import numpy as np
import pandas as pd


def _to_vec(x):
    """Robustly convert a cell (list/tuple/ndarray/str) to a float 1-D array."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, (list, tuple)):
        return np.asarray(list(x), dtype=float)
    s = str(x).strip().replace("[", " ").replace("]", " ").replace(",", " ")
    return np.asarray(s.split(), dtype=float)


def metric(data: pd.DataFrame) -> float:
    df = data.reset_index(drop=True)
    n_rows = len(df)
    if n_rows == 0:
        return 0.0

    A = np.vstack([_to_vec(x) for x in df["option_a_ratings"]])
    B = np.vstack([_to_vec(x) for x in df["option_b_ratings"]])
    n_feat = A.shape[1]

    # Design validities [0.95, 0.85, 0.75, 0.65, 0.60]; fall back gracefully.
    v = np.array([0.95, 0.85, 0.75, 0.65, 0.60], dtype=float)
    if v.shape[0] != n_feat:
        v = np.linspace(0.95, 0.60, n_feat)

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = (np.asarray(df["response"], dtype=float) == 0.0).astype(float)

    a_win = A > B
    b_win = B > A

    # Signed tally margin (Tallying's ONLY explanatory variable).
    margin = a_win.sum(axis=1).astype(float) - b_win.sum(axis=1).astype(float)

    # Validity-weighted evidence difference (NVW's driving quantity).
    d = (a_win * v).sum(axis=1) - (b_win * v).sum(axis=1)

    # OLS: y ~ 1 + margin + d. Return the partial coefficient on d.
    X = np.column_stack([np.ones(n_rows), margin, d])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2])

### RESULTS
- Predicted under pi_3 (simulated): 0.2763 (var=0.0165)
- Predicted under pi_2 (simulated): -0.0069 (var=0.0141)
- Observed on real data: -0.5756 (var=0.0186)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=8):** [0.98, 0.9, 0.82, 0.74, 0.66, 0.6, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  trial 12: A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 15: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 16: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 17: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  trial 18: A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  trial 19: A=[0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  trial 20: A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  trial 21: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  trial 23: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** This design attacks NVW's central and most falsifiable structural claim: capacity interference. NVW asserts that beyond a working-memory span of ~4 cues, each additional pooled cue multiplies the effective inverse temperature by delta < 1, so choice consistency collapses as feature count grows. Tallying has no capacity limit: its softmax inverse temperature beta is design-independent, and its predicted probability depends only on win counts. I therefore set n_features = 8 (four cues beyond the span), which drives NVW's effective sensitivity to beta_eff = beta * delta^4 <= 11 * 0.30^4 = 0.089. Since the maximum possible validity-weighted evidence difference across all trials is |d| <= sum(v^gamma) <= 6.34 (at gamma = 0.7), NVW's predicted P(A) is HARD-BOUNDED to roughly [0.37, 0.64] on every single trial, for every admissible parameter setting (the one-reason lapse uses the same tiny beta_eff, and epsilon/omega only shrink deviations further; attention lapses at alpha <= 1 compress it even more). Tallying, in contrast, predicts a steep, margin-graded ladder: sigmoid(beta), sigmoid(2*beta), sigmoid(4*beta), sigmoid(6*beta) across the margin gradient G1->G4, and near-deterministic choice on the anchor trials E1/E2 (e.g., sigmoid(8*beta) >= 0.98 for beta >= 0.5). The design thus yields three decisive, parameter-free signatures: (1) CAPACITY CLIFF: observed choice proportions on unambiguous anchors (E1: 8-0 split; E2: 7-1 split) exceeding ~0.7 are impossible for NVW at ANY admissible parameter combination, while trivially accommodated by Tallying; likewise, a spread across the margin ladder (G1 vs G4) exceeding ~0.12 is impossible for NVW (its maximum ladder spread is sigmoid(0.089*4.73) - sigmoid(0.089*0.98) ~ 0.08) but natural for Tallying. (2) FIXED-COUNT VALIDITY GRADIENT: trials S1, S2, S3 all have identical win counts (4,3) but validity-weighted evidence d = +1.63, +0.99, -0.37 (order preserved for all gamma in [0.7, 1.3], verified at both endpoints, since v^gamma preserves validity order). Tallying predicts EXACTLY identical probabilities on all three; NVW predicts a strictly decreasing series with S3 below 0.5. Similarly, the tie trials Z1 (d = +1.11), Z2 (d = +0.25), Z3 (d = -0.19) all have counts (4,4): Tallying predicts exactly 0.5 on all three; NVW predicts a strictly ordered series crossing 0.5 between Z2 and Z3. (3) MATCHED-COUNT ALLEGIANCE: G4 and E2 both have counts (7,1) — Tallying predicts identical probabilities; NVW predicts P(G4) > P(E2) strictly, because both the integrated evidence (d = 4.73 vs 3.81) and the one-reason component (top cue favors the tally winner in G4, opposes it in E2) align. Conversely, if NVW is correct, subjects should be near-chance everywhere with only a small d-ordered ripple — a profile Tallying can only mimic by taking beta near 0, which would then destroy the margin ladder Tallying needs elsewhere; the joint pattern (near-chance + d-ordering at fixed counts) is uniquely NVW, while (steep ladder + exact flatness at fixed counts) is uniquely Tallying. Every trial type appears in A/B-mirrored pairs to absorb side and position bias. With 24 unique pairs, K = 96 // 24 = 4, giving exactly 96 trials per subject.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The metric isolates the design's single strongest parameter-free signature: the capacity cliff on unambiguous anchors. Trials 7/8 (7-1 split), 21/22 (8-0 split) and 23/24 (7-1 split) all have tally margins of 6 or 8, giving 6 unique pairs x 4 reps = 24 anchor trials per subject, so the per-subject estimate is based on a decent sample and its between-subject variance is just binomial noise. Tallying predicts P(choose tally winner) = sigmoid(beta*margin)*(1-eps) + eps/2, which is >= ~0.95 for any moderately fitted beta (and still ~0.6 even at the extreme low end beta=0.1, eps=0.5). NVW is hard-bounded: with n_features=8 the capacity interference forces beta_eff = beta*delta^4 <= 11*0.30^4 ~ 0.089, so even the largest possible validity-weighted evidence difference (sum of v^gamma ~ 6.3) gives P <= sigmoid(0.56) ~ 0.64, and the one-reason lapse, attention lapses (alpha < 1 shrink d), and uniform-mixing epsilon all compress this toward 0.5. No admissible NVW parameter combination can push anchor allegiance above ~0.64, while Tallying naturally sits near 1. The A/B mirroring of every anchor pair absorbs side/position bias, so the proportion is a clean readout of margin-driven (count-based) choice rather than response bias. Expected separation: ~0.95+ (self) vs ~0.55-0.60 (adv), with per-subject standard deviations of ~0.05 and ~0.10 respectively, yielding a Welch t-statistic well above 10 at N=25 and p far below 0.01.

Source:
def metric(data: pd.DataFrame) -> float:
    """Anchor-allegiance: proportion of trials with a large tally margin
    (|a_wins - b_wins| >= 6) on which the subject chose the option that
    wins on more features (the tally winner).

    Under Tallying, these are the unambiguous anchors (8-0 and 7-1 splits:
    design trials 7, 8, 21, 22, 23, 24 and their 4 repetitions = 24 trials
    per subject), where the softmax over tallies predicts near-deterministic
    choice for any non-trivial beta. Under NVW, the capacity-interference
    factor delta^4 crushes the effective inverse temperature to
    beta_eff <= 11 * 0.30^4 ~ 0.09, so even the maximum validity-weighted
    evidence difference (~6) yields P ~ sigmoid(0.5) <= 0.64; attention
    lapses and the uniform-mixing epsilon only pull this further toward
    0.5. The two theories therefore make sharply separated predictions on
    this proportion, and the A/B-mirrored anchors cancel side bias.
    """
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    n_anchor = 0
    n_hit = 0

    for ra, rb, resp in zip(a_col, b_col, r_col):
        va = np.asarray(ra, dtype=float)
        vb = np.asarray(rb, dtype=float)
        if va.shape != vb.shape:
            continue

        a_wins = float(np.sum(va > vb))
        b_wins = float(np.sum(vb > va))
        margin = a_wins - b_wins

        if abs(margin) >= 6:
            n_anchor += 1
            winner_is_a = margin > 0
            chose_a = (int(resp) == 0)
            if chose_a == winner_is_a:
                n_hit += 1

    if n_anchor == 0:
        return 0.5
    return float(n_hit) / float(n_anchor)


### RESULTS
- Predicted under pi_3 (simulated): 0.5300 (var=0.0122)
- Predicted under pi_2 (simulated): 0.8808 (var=0.0089)
- Observed on real data: 0.8433 (var=0.0152)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Design validities (fixed by experiment): strictly decreasing,
    # so the TTB cue order is simply feature 0,1,2,3,4.
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_vec(x):
        if isinstance(x, str):
            return np.array(
                [int(v) for v in x.strip("[]()").replace(",", " ").split()]
            )
        return np.asarray(x, dtype=float)

    hits = []
    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = _to_vec(a_cell)
        b = _to_vec(b_cell)
        if a.size == 0 or a.shape != b.shape:
            continue

        # --- TTB winner: first discriminating cue in validity order ---
        ttb_winner = None
        for j in cue_order:
            if j >= a.size:
                continue
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue  # no discriminating cue: TTB guesses, uninformative

        # --- Tallying winner: strict feature-wise win counts ---
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally ties -> Tallying guesses at 0.5, dilutes contrast
        tally_winner = 0 if a_wins > b_wins else 1

        if tally_winner == ttb_winner:
            continue  # heuristics agree: uninformative

        # Pure conflict trial: TTB's option vs the tally majority's option.
        hits.append(1.0 if int(resp) == ttb_winner else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2080 (var=0.0060)
**Predicted under pi_3:** 0.2377 (var=0.0050)
**Predicted under pi_2:** 0.1357 (var=0.0112)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Tally-vs-TTB choice-allegiance contrast.

    For every trial we determine (i) the TALLYING winner (option with more
    strict feature-wise wins) and (ii) the TAKE-THE-BEST winner (winner on
    the highest-validity discriminating cue, validities fixed by design as
    [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]).  The metric is

        mean( response == tally winner ) - mean( response == TTB winner )

    On the 4 trial types where both heuristics pick the same option
    (trials 1, 3, 5, 7 and mirrors) the two indicators are identical and
    cancel.  On the 12 conflict trial types they are exact complements, so
    the metric reduces to 0.75 * (2*P(tally winner | conflict) - 1): strongly
    positive under Tallying, strongly negative under TTB.
    """
    validities = np.array([0.95, 0.9, 0.85, 0.65, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    tally_match = []
    ttb_match = []

    for ca, cb, cr in zip(a_col, b_col, r_col):
        a = np.asarray(ca, dtype=float).ravel()
        b = np.asarray(cb, dtype=float).ravel()
        r = int(cr)

        # Tallying winner: more strict feature-wise wins (ties on a feature
        # contribute to neither side).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1  # undiscriminating tally -> excluded

        # TTB winner: first discriminating cue in descending-validity order.
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break

        if tally_winner >= 0:
            tally_match.append(1.0 if r == tally_winner else 0.0)
        if ttb_winner >= 0:
            ttb_match.append(1.0 if r == ttb_winner else 0.0)

    if len(tally_match) == 0 or len(ttb_match) == 0:
        return 0.0

    return float(np.mean(tally_match) - np.mean(ttb_match))
```

**Observed (real) value:** 0.1583 (var=0.0053)
**Predicted under pi_3:** 0.1658 (var=0.0093)
**Predicted under pi_2:** 0.5479 (var=0.0187)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, neither theory is task-invariant, and each is decisively falsified somewhere. Experiment 1 (pi_3's own design): the observed partial d-coefficient is -0.5756, a large NEGATIVE within-margin validity effect. pi_3 predicted +0.276 (wrong sign, error 0.85) and pi_2 predicted ~0 (structurally required, error 0.57) \u2014 both fail, and the observed sign implies subjects' within-margin sensitivity runs AGAINST stated validities in this design, which neither integration-with-validity-weights nor pure tallying can produce. Experiment 2 (pi_2's design): observed anchor allegiance is 0.8433, matching pi_2 (0.8808) and flatly contradicting pi_3 (0.5300) \u2014 the capacity-interference claim (beta_eff = beta*delta^(n-4)) is decisively falsified: at n=8 features subjects are highly consistent, so noise does not collapse with cue count. Experiment 3: observed TTB-adherence on conflict trials is 0.2080; pi_3 (0.2377) is close, pi_2 (0.1357) is too extreme \u2014 subjects show validity sensitivity beyond win counts. Experiment 4: observed tally-minus-TTB contrast is 0.1583; pi_3 (0.1658) is nearly exact, pi_2 (0.5479) is far too tally-heavy. Net: pi_3 wins Experiments 3 and 4 (nearly exact), pi_2 wins Experiment 2 (nearly exact); both fail Experiment 1, and pi_3 additionally suffers a structural falsification of its capacity mechanism. The overall pattern that a correct theory must capture is: (a) steep, design-independent margin ladder and near-deterministic anchors even at n=8 (kills capacity interference); (b) validity sensitivity beyond counts, but modest and mostly tally-anchored (Exps 3-4); (c) within-margin sensitivity that is ANTI-correlated with validity in Experiment 1's design \u2014 where validities strictly decrease with feature index, the only natural mechanism producing a negative d-coefficient is cue weighting that increases with presentation position (recency/late-cue overweighting), i.e., subjective cue weights only loosely calibrated to communicated validities. Since neither theory can produce (c), and each owns only part of (a)/(b), the field needs a new theory rather than a patched model.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_2 (pure Tallying) with a stronger competitor to the surviving incumbent pi_3. Proposed theory: 'Subjective-Weighted Counting with Cue-Order Recency (SWC)'. Core claims: (1) People integrate ALL discriminating cues additively \u2014 a count-like tally backbone \u2014 with NO capacity interference: the softmax inverse temperature beta is design-independent, so consistency does not degrade with feature count (directly repairs pi_3's falsified Exp-2 behavior and preserves Tallying's Exp-2 success). (2) Cue weights are subjective and only partially calibrated to communicated validities: w_j = (1-lambda)/n + lambda * [(1-rho)*v_j_norm + rho * pos_j], where pos_j is a normalized recency/position term increasing with feature index and rho captures position-driven overweighting of late cues. Because every experiment here presents validities in strictly decreasing feature order, the recency component is anti-correlated with validity, which is the only natural mechanism that yields Experiment 1's large NEGATIVE within-margin d-coefficient (late-cue winners preferred at fixed margin) while still producing mostly-tally behavior. (3) Choice is sigmoid(beta * sum_j w_j * sign(a_j - b_j)) plus a small uniform lapse epsilon; optionally a small per-cue attention lapse alpha as in pi_3. Expected profile: Exp 1 \u2014 negative partial d-coefficient of tunable magnitude (set rho moderate, beta moderate); Exp 2 \u2014 high anchor allegiance (~0.85) and steep margin ladder at n=8 (no delta term); Exp 3 \u2014 mostly tally with a recency tilt away from top-cue winners, giving TTB-adherence near the observed 0.21 rather than pi_2's 0.14; Exp 4 \u2014 mildly positive tally-minus-TTB contrast near 0.16 rather than pi_2's 0.55. Parameter ranges: beta in [0.5, 6], epsilon in [0, 0.08], lambda in [0.2, 1], rho in [0, 0.8], alpha in [0.8, 1]. This theory differs from both incumbents: unlike Tallying it has graded, non-count sensitivity (validity AND position); unlike NVW it has no capacity interference and no validity-dominance one-reason lapses, and it allows within-margin sensitivity of either sign depending on how validity order aligns with presentation order \u2014 exactly the degree of freedom needed to explain all four experiments, including the anomalous Experiment 1 sign flip that both current theories get wrong."
}
```

## Usage

```json
{
  "input_tokens": 12001,
  "output_tokens": 8388,
  "reasoning_tokens": 7271,
  "provider": "Sail Research"
}
```
