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
**Validities (n_features=9):** [0.9, 0.78, 0.76, 0.74, 0.72, 0.7, 0.68, 0.66, 0.64]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]

**Rationale:** This design exploits a structural chasm between TTB and VWEI: TTB's score is binary (winner=1, loser=0), so its softmax NEVER sees cue validity, coalition size, or coalition composition. Consequence: TTB predicts ONE constant choice probability, P(pick TTB's winner) = (1-eps)*sigmoid(beta) + eps/2, on ALL 16 pairs — whether the deciding cue is the 90% expert or the 76% expert, and whether 0 or 7 lower cues oppose it. VWEI, with its bounded parameters (gamma in [2.8,3.2], beta in [0.18,0.23]), predicts a specific, monotone, non-flat profile across the same 16 pairs. The validity vector [0.90, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64] is engineered so that (a) cue 1 (v=0.90, w=10.6) dominates every possible coalition, (b) cue 2 (v=0.78, w=2.03) is beaten by coalitions of 2+ of the next cues (w3+w4=2.68) but not by any single lower cue or by weak cues (w8+w9=0.48) — creating sign flips that are robust across the entire gamma range [2.8, 3.2] and beta range [0.18, 0.23] (verified: pair-7 E ranges -0.93..-0.95; pair-11 E ranges -3.35..-3.41; pair-13 E ranges +0.96..+1.14). Four dissociation signatures: (1) COALITION-SIZE GRADIENT (pairs 5-12): cue 1 ties, cue 2 picks the TTB winner, and the opposing coalition below cue 2 grows from 1 to 2 to 3 to 7 cues. TTB: P(cue-2 winner) flat at sigmoid(beta). VWEI: monotone decline 0.525 -> 0.468 -> 0.426 -> 0.354, crossing 50% between one and two opponents. (2) COMPOSITION AT FIXED COUNT (pairs 13/14 vs 7/8): exactly two opposing cues, but weak ones (experts 8-9) vs strong ones (experts 3-4). TTB: identical predictions (cue 2 decides in all four; validity-blind). VWEI: an ~11-point gap (0.577 vs 0.468) — this weight-magnitude sensitivity at fixed count falsifies TTB (and any count-based heuristic) while confirming VWEI's amplified log-odds weighting. (3) DECIDING-CUE VALIDITY FLATNESS (pairs 1-4 vs 5-14 vs 15-16): when the 90% expert decides (pairs 1-4, agreement baselines), VWEI predicts P(winner)=0.85-0.89; when the 78% expert decides, ~0.35-0.58; when the 76% expert decides against a 3-cue coalition (pairs 15-16, cascade-depth test), 0.447. TTB predicts the SAME probability in all three regimes — a large cross-family validity effect falsifies TTB, flatness falsifies VWEI. (4) SIGN CONFLICTS: on pairs 7-12 and 15-16 the theories point in OPPOSITE directions (TTB follows cue 2/3; VWEI's evidence sum favors the coalition), so a single trial type can flip a subject's majority choice. Every pair is non-degenerate (a discriminating cue always exists), the design is perfectly side-balanced (8 pairs with TTB winner = A, 8 with TTB winner = B; pairs are exact A/B mirrors), and with 16 unique pairs, K = 96//16 = 6 gives 96 trials and 12 samples per mirrored cell — enough per-subject power to estimate the flat-vs-graded profile that adjudicates the theories. A TTB subject is a horizontal line; a VWEI subject is a specific descending curve that no constant can fit (VWEI's spread 0.89->0.35 exceeds anything TTB's two noise parameters can produce, since TTB has no mechanism for any stimulus-driven variation at all).

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Validity-Weighted Evidence Integration (VWEI) with subjective validity amplification. On each trial the decision maker computes, for every rating dimension j, the discriminant d_j = sign(a_j - b_j) in {-1, 0, +1}, weights it by a subjectively amplified validity w_j = [log(v_j/(1-v_j))]^gamma, and accumulates evidence E = sum_j w_j * d_j. Choice probability is a sigmoid of beta * E mixed with a small uniform lapse epsilon. gamma = 1 recovers Bayes-optimal log-odds integration; gamma -> infinity recovers Take-The-Best; the empirically human region sits at gamma near 3, where the top cue dominates coalitions of weaker cues on steep validity gradients (Exp 2) but lower cues still attenuate allegiance on shallow gradients (Exp 1). This round's refinement is purely parametric: the mechanism is unchanged, but the sampling region is re-centered on the hand-verified sweet spot (gamma = 3, beta = 0.2, epsilon = 0), the lapse range is restored to (a tightened version of) the arbiter's [0, 0.2], and the beta floor is raised so that generic flattening noise no longer competes with the gamma-structured weight profile for attenuation work.

**Parameters:**
- gamma: [2.8, 3.2]
- beta: [0.18, 0.23]
- epsilon: [0.0, 0.03]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Validity-Weighted Evidence Integration (VWEI) with subjective
    # validity amplification. Stimulus is the pair of option feature
    # vectors for the current trial: array-like of shape
    # (2, n_features), row 0 = option A, row 1 = option B.
    #
    # Evidence for A over B:  E = sum_j w_j * sign(a_j - b_j),
    # with w_j = [log(v_j / (1 - v_j))]^gamma.
    #   gamma = 1  -> Bayes-optimal log-odds weighting (full integration)
    #   gamma -> inf -> Take-The-Best (top cue dominates every coalition)
    # P(A) = sigmoid(beta * E), mixed with a uniform lapse epsilon.
    # History is ignored: validities are communicated in the
    # instructions, so weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VWEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard the log-odds against v = 1 (infinite weight) and v < 0.5.
    v = np.clip(v, 0.5 + 1e-9, 1.0 - 1e-6)
    log_odds = np.log(v / (1.0 - v))

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Subjective cue weights: amplified log-odds validity.
    w = np.sign(log_odds) * np.abs(log_odds) ** gamma

    # Per-cue discriminant: +1 favors A, -1 favors B, 0 is uninformative.
    d = np.sign(a - b)

    # Accumulated evidence for A (positive) vs B (negative).
    E = float(np.dot(w, d))

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # When E == 0 this is exactly 0.5 (uniform guessing), for any beta.
    x = beta * E
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Validity-sensitivity contrast for the cue-2 cascade design.\n\n    For every trial we compute (a) the Take-The-Best winner (first\n    discriminating cue in validity order) and (b) the VWEI evidence\n    E = sum_j [log(v_j/(1-v_j))]^3 * sign(a_j - b_j), expressed\n    relative to the TTB winner (Erel = E * (+1 if TTB winner is A\n    else -1)).\n\n    Two trial classes:\n      HIGH  : Erel >  3   -> pairs 1-4, where the 90% expert decides\n             and VWEI strongly agrees (P(pick TTB winner) ~ 0.85).\n      LOW   : Erel < -0.5 -> pairs 7-12 and 15-16, where cue 2 (or 3)\n             decides but the weighted evidence of the opposing\n             coalition dominates (P(pick TTB winner) ~ 0.35-0.47).\n\n    Metric = P(choose TTB winner | HIGH) - P(choose TTB winner | LOW).\n\n    TTB predicts the SAME choice probability on every non-tied pair\n    (its score is binary), so this difference is 0 in expectation.\n    VWEI predicts a large positive difference (~ +0.42). The contrast\n    cancels each subject's overall response rate (beta / epsilon),\n    leaving only the stimulus-driven validity sensitivity.\n    \"\"\"\n    v = np.array([0.90, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64])\n    lo = np.log(v / (1.0 - v))\n    w = lo ** 3.0\n    order = np.argsort(-v, kind=\"stable\")\n\n    highs = []\n    lows = []\n\n    for a_cell, b_cell, r in zip(\n        data[\"option_a_ratings\"], data[\"option_b_ratings\"], data[\"response\"]\n    ):\n        a = np.asarray(a_cell, dtype=float)\n        b = np.asarray(b_cell, dtype=float)\n        if a.shape[0] != 9 or b.shape[0] != 9:\n            continue\n\n        # VWEI-style evidence for A over B (gamma = 3).\n        d = np.sign(a - b)\n        E = float(np.dot(w, d))\n\n        # TTB winner: first discriminating cue in validity order.\n        win = None\n        for j in order:\n            if a[j] > b[j]:\n                win = 0\n                break\n            if b[j] > a[j]:\n                win = 1\n                break\n        if win is None:\n            continue  # no discriminating cue (does not occur here)\n\n        # Evidence relative to the TTB winner.\n        s = 1.0 if win == 0 else -1.0\n        Erel = E * s\n\n        r_val = int(r)\n        chose_winner = 1.0 if r_val == win else 0.0\n\n        if Erel > 3.0:\n            highs.append(chose_winner)\n        elif Erel < -0.5:\n            lows.append(chose_winner)\n\n    if not highs or not lows:\n        return 0.0\n\n    return float(np.mean(highs) - np.mean(lows))",
  "rationale": "The core dissociation: TTB's score is binary (winner=1, loser=0), so its softmax+lapse collapses to ONE constant probability of picking the TTB winner on every non-tied trial \u2014 it is completely blind to which cue decides and to what the lower cues say. VWEI instead produces a graded profile: when the 90% expert decides (pairs 1-4, E_rel = +8.6 to +10.6) it picks the TTB winner ~85% of the time, but when cue 2 or cue 3 decides against a weighted opposing coalition (pairs 7-12, 15-16, E_rel = -0.65 to -3.0) it picks the TTB winner only ~35-47% of the time. The metric is the difference in 'pick the TTB winner' rates between these two trial classes. Because it is a within-subject difference of rates, each subject's overall response bias (their beta and epsilon, which under TTB is the ONLY thing that varies) cancels exactly: under TTB the metric is 0 in expectation for every parameter setting; under VWEI it is ~+0.42 (0.85 vs ~0.42 averaged over the 48 low trials). Per subject there are 24 high trials and 48 low trials (6 reps each of the 16 pairs), so the binomial noise of the difference is small (per-subject SD ~0.10-0.13), giving a Welch t at N=25 of roughly 13 \u2014 far beyond the alpha=0.01 threshold. The trial classification is structural (computed from the fixed stimulus vectors with gamma=3 weights), so it is deterministic and identical for every subject and for the pooled data; the sign of E_rel for every classified pair is robust across VWEI's entire gamma range [2.8, 3.2], and pairs with ambiguous evidence (pairs 5-6, 13-14) are deliberately excluded from both classes to keep the contrast clean. The metric works identically on a single subject's slice (24/48 trials) and on the pooled DataFrame."
}
```

## Usage

```json
{
  "input_tokens": 5445,
  "output_tokens": 7601,
  "reasoning_tokens": 6323,
  "provider": "Sail Research"
}
```
