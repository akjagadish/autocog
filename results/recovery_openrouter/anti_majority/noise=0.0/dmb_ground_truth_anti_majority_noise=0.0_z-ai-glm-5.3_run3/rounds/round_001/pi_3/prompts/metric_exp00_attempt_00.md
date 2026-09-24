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
**Validities (n_features=6):** [0.95, 0.92, 0.91, 0.9, 0.89, 0.88]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 9: A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 10: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 11: A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 12: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1]
  trial 13: A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 14: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  trial 15: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 16: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 18: A=[1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 19: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 20: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 21: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 22: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 23: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 24: A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Rationale:** This design attacks TTB's central structural invariant — a SINGLE constant choice probability p* = (1-eps)*sigmoid(beta) + eps/2 on EVERY trial where any cue discriminates — using a validity architecture opposite to previously explored tight-top-ladder designs: one DOMINANT top cue (0.95, whose power-weight exceeds the entire rest of the ladder combined at every gamma in [35,50]: 0.95^40=0.1286 vs 0.0886 for cues 2-6) followed by a CONTESTED second tier (0.92 vs a 0.91/0.90/0.89/0.88 block whose combined power-weight exceeds cue 2's weight at every gamma in [35,50], e.g., at gamma=40: 0.0303 vs 0.0228). Consequence 1 (direction dissociation at depth 2): on L4-type trials cue 1 is tied, cue 2 (first discriminator) favors one option, and all four lower cues favor the other. TTB picks the cue-2 winner at p*; PWEI REVERSES, predicting p(cue-2 winner) between 0.33 (gamma=35) and 0.45 (gamma=50) — never above 0.5, sign-robust across PWEI's entire parameter box. Consequence 2 (graded confidence ladder): trials L0/L1/L2/L3/L4 hold cue 1 tied and cue 2 as first discriminator while the opposing block grows from 0 to 4 cues. PWEI predicts a monotone sweep of p(cue-2 winner) of roughly 0.73 / 0.59 / 0.48 / 0.42 / 0.37 (gamma=40, beta=8, eps=0.2), crossing 50% between L1 and L2 — a psychometric function of the weighted evidence margin D that TTB's two-level step function (p* or 50%) cannot produce at any (beta, eps). Consequence 3 (depth gradient on 'clean' trials): T1 (cue 1 decides, rest tied), L0 (cue 2 decides, rest tied), and U2 (cue 3 decides, rest tied) are identical to TTB — first discriminator, no other signal — so TTB predicts one common p*; PWEI predicts declining confidence 0.89 > 0.73 > 0.66 because the deciding cue's share of total weight shrinks with depth. Consequence 4 (agreement anchors): T3 (all cues agree, PWEI ~0.90) and T2 (cue 1 against all five others, PWEI 0.61-0.78, direction agreeing with TTB) show that even TTB's best single p* must simultaneously miss ~0.90 (T3), ~0.73 (L0), ~0.37 (L4) — no constant fits. Controls: C1/C2 (identical vectors) verify both models' 50% guessing and catch key bias; every dissociating trial has a mirrored twin (A/B swapped) to cancel side bias, with analysis pooled by cue-winner side. 24 unique pairs x K=4 = exactly 96 trials, giving 8 observations per cell per subject (4 + 4 mirrored). Decisive quantitative signatures: (i) Spearman correlation between cell choice proportion and PWEI's weighted margin D (strongly positive if PWEI true, ~0 if TTB true); (ii) L4 significantly below 50% for the first-discriminator winner; (iii) monotone L-ladder crossing 50%; (iv) T1 > L0 > U2 depth ordering. TTB predicts all discriminating cells statistically indistinguishable at one common p* and cannot approximate any of these patterns even with its full beta in [0.1,20], epsilon in [0,0.5] range, because its per-trial prediction is parameterized by a single constant independent of cue depth, cue validity, and lower-cue configuration.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # OLS slope of \"chose the TTB (first-discriminating-cue) winner\" on the\n    # PWEI power-weighted evidence margin measured TOWARD that winner.\n    # TTB: choice prob is one constant p* on every discriminating trial\n    #      (any beta, eps)  =>  slope exactly 0.\n    # PWEI: prob is a psychometric function of the weighted margin D\n    #      =>  strongly positive slope (~0.5 over the design's margin range).\n    val = np.array([0.95, 0.92, 0.91, 0.90, 0.89, 0.88], dtype=float)\n    gamma = 40.0  # mid of PWEI's calibrated regime [35, 50]\n    w = np.power(val, gamma)\n    w = w / w.sum()\n    cue_order = np.argsort(-val, kind=\"stable\")\n\n    xs = []\n    ys = []\n    for a_cell, b_cell, r in zip(\n        data[\"option_a_ratings\"], data[\"option_b_ratings\"], data[\"response\"]\n    ):\n        a = np.asarray(a_cell, dtype=float)\n        b = np.asarray(b_cell, dtype=float)\n        s = np.sign(a - b)\n        if not np.any(s != 0):\n            # Identical options: pure guessing for both theories; excluded.\n            continue\n        # TTB winner: first discriminating cue in descending-validity order.\n        winner = None\n        for j in cue_order:\n            if s[j] != 0:\n                winner = 0 if s[j] > 0 else 1\n                break\n        if winner is None:\n            continue\n        # Power-weighted evidence for option A.\n        D = float(np.sum(w * s))\n        # Margin re-signed so that positive = evidence favors the TTB winner.\n        x = D if winner == 0 else -D\n        y = 1.0 if int(r) == winner else 0.0\n        xs.append(x)\n        ys.append(y)\n\n    xs = np.asarray(xs, dtype=float)\n    ys = np.asarray(ys, dtype=float)\n    if xs.size < 3:\n        return 0.0\n    sxx = float(np.sum((xs - xs.mean()) ** 2))\n    if sxx <= 1e-12:\n        return 0.0\n    sxy = float(np.sum((xs - xs.mean()) * (ys - ys.mean())))\n    return float(sxy / sxx)",
  "rationale": "Core idea: TTB's structural invariant is that its choice probability is a single constant p* = (1-eps)*sigmoid(beta) + eps/2 on EVERY trial where any cue discriminates, regardless of cue depth, validity, or lower-cue configuration. PWEI instead produces a graded psychometric function of the power-weighted evidence margin. The metric therefore regresses, trial by trial, the binary indicator 'did the subject choose the TTB winner (the option favored by the first discriminating cue)?' on the PWEI weighted margin D re-signed toward that winner (x = +D if TTB winner is A, -D if B; weights v_j^40 normalized, gamma=40 the midpoint of PWEI's box). Under TTB, E[y|x] = p* for all x, so the OLS slope is exactly 0 for every parameter setting (beta in [0.1,20], eps in [0,0.5]) \u2014 this cancels TTB's parameter heterogeneity entirely, leaving only binomial noise. Under PWEI, E[y|x] = (1-eps)*sigmoid(beta*x)+eps/2, a strictly increasing psychometric curve; across this design's 11 discriminating cell types the margins span x in [-0.08, 1.0] with predicted conformities from 0.374 (L4: cue-2 winner opposed by all four lower cues) up to 0.90 (all-agree), giving a slope of roughly 0.50-0.59 robustly across PWEI's whole parameter box (checked at gamma=35/beta=7/eps=0.23 -> ~0.52; gamma=50/beta=10/eps=0.17-0.23 -> ~0.50). Per-subject noise: 88 discriminating trials per subject, residual variance ~0.2, Var(x) ~0.10, so per-subject slope sd ~0.15 for both theories (TTB's is at most ~0.17 even at p*=0.5). At N=25 the Welch t on the pooled point estimates is approx 0.55 / sqrt((0.022+0.028)/25) ~ 12, far below alpha=0.01. The metric works identically on the pooled DataFrame and on each single-subject slice (each subject sees all 24 pairs x 4 reps), so between-subject variance is well defined. Identical-option catch trials (trials 23/24) are excluded because both models guess there and including them would hand TTB a spurious non-zero slope. Mirror trials are handled automatically since x and y are computed from the actual A/B vectors, so side bias cancels in the regression."
}
```

## Usage

```json
{
  "input_tokens": 5160,
  "output_tokens": 18268,
  "reasoning_tokens": 17067,
  "provider": "Sail Research"
}
```
