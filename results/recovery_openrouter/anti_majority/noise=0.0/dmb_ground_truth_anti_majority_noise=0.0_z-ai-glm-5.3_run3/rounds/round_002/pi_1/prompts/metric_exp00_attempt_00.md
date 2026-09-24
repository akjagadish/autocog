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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.7, 0.65, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 8: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 12: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 14: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 15: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 16: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 17: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 18: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 19: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 21: A=[0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  trial 22: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 23: A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 24: A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Rationale:** This design adversarially dissociates Take-The-Best (TTB) from Sparse-Option Preference / Polarity-Inverted Evidence Integration (SPI) by exploiting their two structural differences: (i) TTB uses ONLY the single highest-validity discriminating cue with conventional polarity (1 = good), producing one constant choice probability p* = (1-eps)*sigmoid(beta) + eps/2 on EVERY trial where any cue discriminates; (ii) SPI integrates ALL cues with near-uniform weights and INVERTED polarity, preferring the option with FEWER 1s, producing a graded psychometric function of the weighted sparsity evidence E = sum_j w_j*(b_j - a_j). Validities are [0.95, 0.85, 0.75, 0.70, 0.65, 0.60]: well-separated and descending so TTB's cascade order is unambiguous, yet close enough that SPI's sign of E is robust across its entire gamma in [0, 2] (verified at gamma = 0, 1, 2). DISSOCIATION 1 (direction, sign-robust across BOTH parameter boxes): Trial T6 presents unanimous evidence (A = all 1s, B = all 0s). TTB must pick the unanimously endorsed option with p >= 0.51 at ANY (beta in [0.1,20], eps in [0,0.5]) because sigmoid(beta) >= 0.5. SPI must pick the all-zeros option: E = -1, and even at its worst case (rho = 0.35, eps = 0.35, beta = 0.5) p(endorsed option) <= 0.476 < 0.5. The same sign-robust conflict holds on T0, T1, T7, T10, where the first-discriminating-cue winner carries more 1s (E = -w1, -w3, -w2, all < 0 for any gamma since the deciding cue always has the largest weight among consulted cues). DISSOCIATION 2 (quantitative sweep TTB cannot fit): Trials T0, T1, T2, T3, T4, T4b, T5 hold cue 1 (0.95) as the first discriminating cue favoring the SAME option, while the lower-cue configuration sweeps SPI's inverted evidence monotonically: E(T0) = -(w1+w2+w3) in [-0.63, -0.50], E(T1) = -w1 in [-0.26, -0.17], E(T2) = -w1+w2 in [-0.05, 0], E(T3) = -w1+w2+w3 in [+0.11, +0.17], E(T4) in [+0.25, +0.32], E(T4b) in [+0.37, +0.49], E(T5) = 1-2*w1 in [+0.48, +0.67] — all sign-robust across gamma in [0,2]. TTB predicts ONE constant p* on every rung (cue 1 is always the sole basis); SPI predicts a monotone psychometric sweep crossing exactly 50% between T2 and T3 (e.g., at gamma=1, beta=6, eps=0.2, rho=0.1: approximately 0.20, 0.32, 0.48, 0.69, 0.80, 0.85, 0.88 for the cue-1 winner). No single constant can fit a ~0.7-wide sweep that straddles 50%, and conversely SPI cannot produce a flat confident p* because its output is strictly monotone in E. DISSOCIATION 3 (near-tie cell): on T2, SPI's evidence is in [-0.05, 0], forcing p ~ 50% (exactly 50% at gamma=0), while TTB is confident at p* > 0.51 — a confidence dissociation at exactly the point where SPI's evidence cancels. DISSOCIATION 4 (depth invariance): T7 (cue 3 decides) and T10 (cue 2 decides) are, to TTB, identical to T1 (same constant p*, only the discriminating cue's identity changes); SPI predicts slightly weaker anti-winner evidence (|E| = w2 or w3 < w1), a subtle ordering TTB forbids. CONTROLS: T9/T9b are identical-vector trials where both theories predict exactly 50% (key-bias check); every dissociating trial has a mirrored twin (A/B swapped) so side bias cancels, with analysis pooled by first-discriminating-cue winner. Decisive quantitative signatures: (a) p(unanimously endorsed option) < 0.5 on T6/T6m if SPI is true, > 0.5 if TTB is true — sign-robust over both full parameter boxes; (b) Spearman correlation between cell choice proportion and SPI's evidence E: strongly positive if SPI is true, ~0 if TTB is true; (c) monotone ladder T0 < T1 < T2 ~ 0.5 < T3 < T4 < T4b < T5 vs. TTB's flat equality of all ten discriminating cells; (d) T2 at ~50% while T1/T3 are confident. 24 unique pairs x K = 4 repetitions = exactly 96 trials, giving 8 observations per subject per pooled cell (4 unmirrored + 4 mirrored).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



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
**Description:** Sparse-Option Preference / Polarity-Inverted Evidence Integration (SPI). When choosing between two options described by binary expert ratings, a substantial share of subjects systematically mis-map cue polarity: an endorsement (rating = 1) is treated as evidence AGAINST an option, so the option with FEWER 1s (the sparser option) is perceived as higher quality — consistent with a 'fewer red flags' reading of the display or a rarity-implies-quality inference. Subjects integrate ALL cues additively with near-uniform weights: the evidence for option A over B is E = sum_j w_j * (b_j - a_j), where w_j = v_j^gamma with gamma small (validity plays at most a mild role). Choice is a softmax over [E, -E] with inverse temperature beta, plus an independent lapse epsilon to uniform. A per-subject polarity-mixing weight rho captures heterogeneity: with probability 1-rho the subject responds according to the inverted (sparse-preferring) polarity, with probability rho according to the conventional polarity. This mixture attenuates the extreme pure-sparse predictions toward the moderately negative values observed. The theory is sharply distinguished from both incumbents: unlike Take-The-Best it uses all cues and inverts the evidence sign (so it is not pinned at ~0 on the depth/conflict-invariance and margin-slope metrics where TTB is structurally stuck), and unlike power-weighted integration it places humans at the FLAT end of the weighting ladder with inverted polarity, producing the negative signatures that steep-gamma PWEI cannot generate at any parameter setting.

**Parameters:**
- gamma: [0, 2]
- beta: [0.5, 20]
- epsilon: [0.05, 0.35]
- rho: [0, 0.35]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Sparse-option Preference / Polarity-Inverted Evidence Integration (SPI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SPI expects a (2, n_features) stimulus; got shape {stim.shape}."
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
    rho = float(parameters["rho"])

    # Near-uniform validity weighting: w_j = v_j^gamma, gamma in [0, 2].
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), absorbed by beta; it keeps the evidence scale O(1) for any
    # gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Polarity-INVERTED evidence for A over B: each cue where B carries
    # the endorsement (1) counts FOR A; each cue where A carries the
    # endorsement counts AGAINST A. Sparse options accumulate evidence.
    E = float(np.sum(w * (b - a)))

    # Softmax over [E, -E] with max-subtraction for numerical stability.
    # Inverted polarity: positive E favors A (the sparser option).
    scores_inv = np.array([E, -E])
    z = beta * (scores_inv - scores_inv.max())
    e = np.exp(z)
    p_inv = e / e.sum()

    # Conventional polarity (endorsement favors the option carrying it):
    # mirror of the inverted distribution.
    scores_norm = -scores_inv
    z2 = beta * (scores_norm - scores_norm.max())
    e2 = np.exp(z2)
    p_norm = e2 / e2.sum()

    # Per-subject polarity mixture: (1 - rho) inverted, rho conventional.
    p_core = (1.0 - rho) * p_inv + rho * p_norm

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Lower-cue evidence-usage slope.\n\n    For every trial with a discriminating cue, determine the Take-The-Best\n    winner (first cue in descending-validity order that differs). Pool trials\n    into cells keyed by the inverted-polarity integrated evidence favoring\n    that winner, E_w = sum_j w_j * (loser_j - winner_j), w_j proportional to\n    validity. Then compute the OLS slope of P(choose the TTB winner) on E_w\n    across cells.\n\n    TTB: the winner's choice probability is the SAME constant p* on every\n    discriminating trial (only the top discriminating cue is used), so the\n    psychometric function over E_w is exactly flat -> slope = 0 for ANY\n    (beta, epsilon).\n\n    SPI: choice probability is a strictly increasing sigmoid of E_w\n    (rho < 0.5 preserves monotonicity, lapse is affine), and the design's\n    cells sweep E_w from -1 (unanimous endorsement) to +0.58 -> large\n    positive slope.\n    \"\"\"\n    validities = np.array([0.95, 0.85, 0.75, 0.70, 0.65, 0.60], dtype=float)\n    w = validities / validities.sum()\n    cue_order = np.argsort(-validities, kind=\"stable\")\n\n    a_cells = data[\"option_a_ratings\"].tolist()\n    b_cells = data[\"option_b_ratings\"].tolist()\n    resp = data[\"response\"].to_numpy()\n\n    n = len(data)\n    E_list = []\n    Y_list = []\n\n    for i in range(n):\n        a = np.asarray(a_cells[i], dtype=float).ravel()\n        b = np.asarray(b_cells[i], dtype=float).ravel()\n        if a.shape[0] != validities.shape[0] or b.shape[0] != validities.shape[0]:\n            continue\n\n        # TTB cascade: first discriminating cue in validity order.\n        winner = -1\n        for j in cue_order:\n            if a[j] > b[j]:\n                winner = 0\n                break\n            if b[j] > a[j]:\n                winner = 1\n                break\n        if winner < 0:\n            continue  # no discriminating cue: uninformative for the contrast\n\n        # Inverted-polarity integrated evidence favoring the TTB winner:\n        # the loser's endorsements (1s) count FOR the winner (sparse option).\n        if winner == 0:\n            E = float(np.sum(w * (b - a)))\n        else:\n            E = float(np.sum(w * (a - b)))\n\n        E_list.append(E)\n        Y_list.append(1.0 if int(resp[i]) == winner else 0.0)\n\n    if len(E_list) < 4:\n        return 0.0\n\n    E = np.asarray(E_list, dtype=float)\n    Y = np.asarray(Y_list, dtype=float)\n\n    # Pool trials into cells by (rounded) evidence value; mirrored twins and\n    # repetitions of the same pair share the same E_w and are averaged.\n    keys = np.round(E, 4)\n    uniq = np.unique(keys)\n    if uniq.shape[0] < 2:\n        return 0.0\n\n    cell_x = np.empty(uniq.shape[0], dtype=float)\n    cell_y = np.empty(uniq.shape[0], dtype=float)\n    for idx in range(uniq.shape[0]):\n        mask = keys == uniq[idx]\n        cell_x[idx] = E[mask].mean()\n        cell_y[idx] = Y[mask].mean()\n\n    x = cell_x - cell_x.mean()\n    y = cell_y - cell_y.mean()\n    denom = float(np.sum(x * x))\n    if denom <= 1e-12:\n        return 0.0\n\n    return float(np.sum(x * y) / denom)",
  "rationale": "The metric is the OLS slope of P(choose the TTB winner) regressed on the inverted-polarity integrated evidence E_w favoring that winner, computed across the 11 evidence cells of this design (each cell = a unique pair + its mirror, 8 trials per subject per cell; 88 discriminating trials per subject, the 2 identical pairs excluded). STRUCTURAL CONTRAST: Under TTB, the probability of choosing the first-discriminating-cue winner is one constant p* = (1-eps)*sigmoid(beta) + eps/2 on EVERY discriminating trial, independent of everything happening on lower-validity cues. Hence the cell psychometric function over E_w is exactly flat and the slope is 0 in expectation for any (beta, epsilon) in the parameter box \u2014 the per-subject slope carries only binomial noise (sd approx. 0.12 in normalized evidence units, since the wide E_w spread from -1 to +0.58 makes the slope estimator very efficient: Var(slope) approx p*(1-p*) / (8 * Var_cells(E_w))). Under SPI, P(choose winner) = (1-eps)*[(1-rho)*sigma(2*beta*E_w) + rho*sigma(-2*beta*E_w)] + eps/2, which is strictly INCREASING in E_w for any rho < 0.5, any lapse, any beta > 0, and any gamma in [0,2] (the sign/ordering of E_w across cells is robust across gamma because the validities are well separated and the design was built so the deciding cue always dominates). The cells sweep E_w from -1 (unanimously endorsed option, trials 8/20) through -0.57 (trials 1/13) up to +0.58 (trials 7/19), so SPI produces a steep positive slope (approx. 0.55 at the design's example parameters gamma=1, beta=6, eps=0.2, rho=0.1; still approx. 0.15 at weak settings beta=2, rho=0.3, eps=0.3). This is the design's core 'flat vs. monotone ladder' dissociation (dissociations 1-3): TTB is structurally pinned at slope 0 while SPI is structurally pinned away from 0 in the positive direction. The metric works identically on the pooled DataFrame and on each subject's slice (every subject completes all 96 trials, so each subject yields all 11 cells), giving a well-defined between-subject variance that is tiny under TTB (pure binomial) and moderate under SPI, so Welch's t-test at N=25 separates the two theories by many standard errors. No trivially-circular quantity is used: the metric only reads choices and the stimulus vectors, and the contrast is TTB's falsifiable one-reason (lower-cue non-use) prediction against SPI's all-cue integration prediction."
}
```

## Usage

```json
{
  "input_tokens": 5767,
  "output_tokens": 18142,
  "reasoning_tokens": 16574,
  "provider": "Sail Research"
}
```
