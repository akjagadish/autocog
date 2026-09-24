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
**Validities (n_features=6):** [0.9, 0.89, 0.88, 0.87, 0.7, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 12: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 13: A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 14: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 15: A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 16: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]

**Rationale:** This design targets the specific operating regime PWEI claims for humans (gamma in [35,50], i.e., a steep but finite power transform) and opposes it to TTB's two structural invariants: (i) constant confidence sigmoid(beta) on EVERY trial where any cue discriminates, regardless of which cue, its validity, or what lower cues say, and (ii) 50% only when all cues tie. The validity ladder is deliberately TIGHT at the top (0.90, 0.89, 0.88, 0.87) with a long tail (0.70, 0.60). At any gamma in [35,50], the three sub-top cues' combined power-weight EXCEEDS the top cue's weight (e.g., at gamma=40: 0.89^40+0.88^40+0.87^40 = 0.0193 > 0.90^40 = 0.0148), so PWEI must REVERSE against the top cue on the flagship 1-vs-3 trials (T2/T2m), predicting p(top-cue option) between 0.25 (gamma=35) and 0.45 (gamma=50) — never above 0.5 — while TTB predicts the top-cue option at its full constant confidence p*. This direction conflict is sign-robust across PWEI's entire parameter box. Second, trials T2, T3, T4, T5 all have cue 1 (0.90) as the first discriminating cue favoring the same option, but with graded weighted evidence D (gamma=40: -0.13, -0.02, +0.16, +0.43). TTB predicts a FLAT p* across all four; PWEI predicts a monotone psychometric sweep from ~0.31 to ~0.88 — a ~0.6-wide confidence gradient on trials TTB treats as identical. Third, depth invariance: on T6/T6m cues 1-4 tie and cue 5 (validity only 0.70) decides; TTB predicts the SAME constant p* as on cue-1 trials, while PWEI's weight for 0.70^gamma is ~1e-5 of the total, forcing ~50% choices. T8/T9 probe depth 2: TTB stays at p*, PWEI is at ~50% (T8, near-canceling block) or 0.82 (T9). T1/T1m anchor both models near ceiling; T10 (identical vectors) verifies both models' 50% guessing prediction and catches key bias. Mirrored versions of every dissociating trial control side bias; analysis should pool by cue-winner. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 12 observations per subject (6 + 6 mirrored). Predicted outcomes: if TTB is correct, choice proportions favoring the first-discriminating-cue winner are statistically indistinguishable across ALL of T1-T9 and their mirrors (one common p*), including the 1-vs-3 reversal trials and the 0.70-validity depth trials, with T10 at 50%; if PWEI is correct, confidence tracks the weighted evidence margin D: below 50% for the top cue on T2, near 50% on T3/T6/T8, and graded from ~0.7 to ~0.9 on T4/T5/T9 — a pattern TTB's two-level step function cannot produce, and conversely a flat p* pattern that PWEI's steep-gamma regime cannot produce because the tight ladder forces it to vote against the top cue.

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
**Description:** Power-Weighted Evidence Integration (PWEI), calibrated operating regime. When choosing between two options described by binary cues, people integrate ALL cues additively, but each cue's vote is weighted by a power transform of its stated validity, w_j proportional to v_j^gamma. The net evidence for option A is D = sum_j w_j * sign(a_j - b_j), and choice is a softmax over [D, -D] with inverse temperature beta, plus an independent lapse epsilon to uniform choice. The psychologically critical regime is a STEEP one: gamma around 40 makes the top cue's weight dominate a tightly-packed validity ladder (e.g. a 0.95 cue overriding a 0.90-led five-cue block, as in Experiment 2), producing near-TTB choices there, while the ~0.2 lapse keeps overall confidence graded and attenuated (as in Experiment 1). The steepness parameter gamma spans Tallying (gamma=0) to Take-The-Best (gamma -> infinity); the fitted human regime sits far toward the TTB end but retains graded, evidence-margin-dependent confidence on near-tie weighted conflicts, which is exactly what constant-confidence pure heuristics cannot produce.

**Parameters:**
- gamma: [35, 50]
- beta: [7, 10]
- epsilon: [0.17, 0.23]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Power-Weighted Evidence Integration.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PWEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Power-transformed validity weights, normalized by their sum.
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), so it is absorbed by beta; it keeps the evidence scale
    # O(1) for any gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Weighted evidence difference: each cue votes +/-w_j, ties vote 0.
    D = float(np.sum(w * np.sign(a - b)))
    scores = np.array([D, -D])

    # Numerically stable softmax over the weighted evidence.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse to uniform choice.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Depth/conflict invariance contrast for TTB vs PWEI.\n\n    Three structurally identified trial classes (validity order is fixed:\n    [0.90, 0.89, 0.88, 0.87, 0.70, 0.60]):\n\n    HIGH  - the top cue (0.90) discriminates and every other discriminating\n            cue agrees with it (design trials 1, 2, 9, 10; 24 obs/subject).\n            PWEI evidence margin |D| is maximal (~1 or w_top).\n    DEPTH5- cues 1-4 all tie; only the 0.70-validity cue discriminates\n            (design trials 11, 12; 12 obs/subject). PWEI's weight for\n            0.70^gamma is ~1e-5 of total, so PWEI predicts ~50% choices.\n    REV   - the top cue discriminates one way, the three sub-top cues\n            (0.89, 0.88, 0.87) discriminate the other way, tail cues tie\n            (design trials 3, 4; 12 obs/subject). In PWEI's steep-gamma\n            regime the sub-top block outweighs the top cue, so PWEI\n            chooses AGAINST the TTB winner.\n\n    For each class compute p = P(choose the TTB winner). TTB's structural\n    invariant is a single constant confidence p* on every discriminating\n    trial, so 2*p_HIGH - p_DEPTH5 - p_REV = 0 exactly, at ANY (beta, eps).\n    PWEI predicts p_HIGH ~ 0.9, p_DEPTH5 ~ 0.5, p_REV ~ 0.1-0.37, giving a\n    value ~0.9-1.2 across its entire parameter box.\n    \"\"\"\n    VALS = np.array([0.90, 0.89, 0.88, 0.87, 0.70, 0.60])\n    order = np.argsort(-VALS, kind=\"stable\")  # validity-descending cue indices\n\n    n_high, k_high = 0, 0\n    n_d5, k_d5 = 0, 0\n    n_rev, k_rev = 0, 0\n\n    a_col = data[\"option_a_ratings\"]\n    b_col = data[\"option_b_ratings\"]\n    r_col = data[\"response\"]\n\n    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):\n        a = np.asarray(a_cell, dtype=float)\n        b = np.asarray(b_cell, dtype=float)\n        s = np.sign(a - b)  # +1 favors A, -1 favors B, 0 tie, per cue\n\n        # First discriminating cue in validity order (TTB's search).\n        first = None\n        for j in order:\n            if s[j] != 0.0:\n                first = j\n                break\n        if first is None:\n            continue  # identical options: both theories say 50%\n\n        ttb_winner_is_a = s[first] > 0\n        chose_ttb = (int(resp) == 0) == ttb_winner_is_a\n\n        if first == 4:\n            # DEPTH5: only the 0.70-validity cue discriminates.\n            n_d5 += 1\n            k_d5 += int(chose_ttb)\n        elif first == 0 and s[1] == -s[0] and s[2] == -s[0] and s[3] == -s[0] \\\n                and s[4] == 0.0 and s[5] == 0.0:\n            # REV: top cue vs the three sub-top cues, tail ties.\n            n_rev += 1\n            k_rev += int(chose_ttb)\n        elif first == 0:\n            # HIGH candidate: all other discriminating cues agree with top.\n            others = [s[j] for j in range(1, 6) if s[j] != 0.0]\n            if all(o == s[0] for o in others):\n                n_high += 1\n                k_high += int(chose_ttb)\n        # All other trial types (mixed conflicts, cue-2-led trials, ties)\n        # are deliberately excluded to keep the classes pure.\n\n    p_high = k_high / n_high if n_high > 0 else 0.5\n    p_d5 = k_d5 / n_d5 if n_d5 > 0 else 0.5\n    p_rev = k_rev / n_rev if n_rev > 0 else 0.5\n\n    return 2.0 * p_high - p_d5 - p_rev",
  "rationale": "The metric is a zero-sum linear contrast (weights 2, -1, -1) over choice proportions favoring the TTB winner in three structurally identified trial classes, so it is exactly 0 in expectation for TTB at ANY parameter setting (beta in [0.1,20], epsilon in [0,0.5]): TTB's two structural invariants \u2014 one-reason stopping and constant confidence sigmoid(beta) on every discriminating trial \u2014 force p_HIGH = p_DEPTH5 = p_REV = p*. For PWEI the three classes sit at very different points of its evidence-margin psychometric: HIGH trials (all cues unanimous, or top cue alone deciding) have |D| ~ 0.4-1 normalized, giving p ~ 0.89-0.92 after lapse; DEPTH5 trials have D ~ 6e-5 (0.70^gamma is ~1e-5 of total weight at gamma in [35,50]), giving p ~ 0.50; REV trials have D negative (0.89^g+0.88^g+0.87^g > 0.90^g throughout the box), giving p ~ 0.12-0.37 \u2014 PWEI votes AGAINST the top cue. Hence PWEI's metric value is ~0.9-1.2 across its entire parameter box, vs ~0 for TTB: a sign- and magnitude-robust gap. Per-subject noise is binomial only (24, 12, 12 observations per class): worst-case per-subject SD is ~0.29 (TTB with p*=0.5) and ~0.21 (PWEI), so at N=25 the Welch t-statistic is ~13-16, far below alpha=0.01. The metric works identically on a single subject's 96-trial slice, so between-subject variance is well defined. It is falsifiable (human data showing graded confidence or top-cue reversals would push it away from 0), not trivially true: it directly tests TTB's constant-confidence/depth-invariance prediction against actual choices."
}
```

## Usage

```json
{
  "input_tokens": 4720,
  "output_tokens": 11512,
  "reasoning_tokens": 9942,
  "provider": "Sail Research"
}
```
