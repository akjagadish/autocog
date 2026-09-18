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
**Validities (n_features=18):** [0.55, 0.95, 0.5525, 0.55, 0.5, 0.5868, 0.55, 0.95, 0.6343, 0.55, 0.5, 0.6901, 0.55, 0.95, 0.7504, 0.55, 0.5, 0.8177]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Rationale:** The six base pairs isolate conflicts between two discriminating experts separated by exactly two display positions; all other experts tie and therefore cannot terminate either search rule. In every base pair, the earlier expert has validity 0.55 and favors B, whereas the later expert has slightly to substantially higher validity and favors A. The later-minus-earlier validity-log-odds gaps are approximately 0.01, 0.15, 0.35, 0.60, 0.90, and 1.30. Take The Best always ranks the later expert above the 0.55 expert, so it predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 at all six levels, regardless of the size of the validity gap. Noisy Priority Race Search instead predicts a validity-gap-dependent crossover because the earlier B-favoring expert receives a constant two-position serial advantage. Its later-minus-earlier priority difference is sensitivity*D - 2*serial_priority + (bias_later-bias_earlier). At D≈0.01 this difference is at most 5(0.01)-2(0.15)+0.16=-0.09 across the entire parameter range, so the advocated theory robustly predicts that the nominally less-valid, earlier expert wins the race and B is chosen more often. At D≈1.30 the difference is at least 1(1.30)-2(0.45)-0.16=0.24, so it robustly predicts that the more-valid expert wins and A is chosen more often. Intermediate gaps locate each subject's stochastic priority crossover and should produce a rising psychometric profile, potentially with stable cue-specific deviations from perfect monotonicity. Thus the critical signature is both a directional reversal at the smallest gap and a quantitative increase in following the higher-validity expert as the log-odds gap grows; TTB can only fit one condition-invariant response probability. The intervening tied experts preserve the same two-position separation, while tied 95% and 50% experts provide a broad advertised-validity spread without affecting choices. Each base pair has an exact A/B reversal, balancing response-side preferences. The 12 unique pairs are repeated eight times, yielding exactly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Noisy Priority Race Search proposes that communicated cue validities establish persistent but imperfect subjective cue priorities rather than a universally fixed ordering. Each subject transforms validity log-odds into priority, adds a stable cue-specific representation bias, and has a serial-position prior favoring earlier displayed cues. On each trial, independent retrieval fluctuations perturb these priorities. Cues are inspected in the resulting stochastic order, tied cues are skipped, and search stops at the first discriminating cue. Only that cue determines the intended response; evidence from uninspected or lower-priority cues is never integrated. Small validity gaps can therefore produce stable individual departures and trial-level reversals of the stated ordering, whereas large validity gaps produce nearly deterministic lexicographic search. Response execution is separately subject to temperature-controlled errors and occasional lapses.

**Parameters:**
- n_features: n_features
- validities: validities
- priority_bias: [(-0.08, 0.08)] * n_features
- validity_sensitivity: [1.0, 5.0]
- serial_priority: [0.15, 0.45]
- retrieval_noise: [0.04, 0.08]
- response_temperature: [1.6, 2.0]
- lapse_rate: [0.0, 0.05]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Noisy Priority Race Search expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    persistent_bias = np.asarray(parameters["priority_bias"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got {validities.shape}."
        )
    if persistent_bias.ndim != 1 or persistent_bias.size != n_features:
        raise ValueError(
            f"priority_bias must have length {n_features}; got {persistent_bias.shape}."
        )

    validity_sensitivity = float(parameters["validity_sensitivity"])
    serial_priority = float(parameters["serial_priority"])
    retrieval_noise = float(parameters["retrieval_noise"])
    beta = float(parameters["response_temperature"])
    lapse_rate = float(parameters["lapse_rate"])

    # Persistent subjective priority representation. Validity log odds encode
    # communicated diagnosticity. The serial term supplies a weak display-order
    # prior and resolves equal-validity cues without assuming exact equality in
    # their subjective representations.
    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-9)
    validity_signal = np.log(v / (1.0 - v))
    serial_signal = (n_features - 1) - np.arange(n_features, dtype=np.float64)
    priority = (
        validity_sensitivity * validity_signal
        + serial_priority * serial_signal
        + persistent_bias
    )

    differences = stim[0] - stim[1]
    discriminating = np.flatnonzero(differences != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # If retrieval shocks are iid Gumbel, the probability that a particular
    # discriminating cue is reached first is a softmax over its priority. Cues
    # that tie on the current trial may be inspected but cannot terminate
    # search, so they do not enter the race among stopping cues.
    race_logits = priority[discriminating] / max(retrieval_noise, 1e-12)
    race_logits = race_logits - np.max(race_logits)
    first_prob = np.exp(race_logits)
    first_prob /= first_prob.sum()

    # Marginalize over which single cue wins the priority race. This is not
    # evidence integration: in every latent search realization exactly one
    # first discriminating cue determines the intended choice.
    favors_a = differences[discriminating] > 0.0
    intended_a = float(np.sum(first_prob[favors_a]))

    # Conditional on a stopping cue, response temperature permits execution
    # errors of equal size in either direction.
    if beta >= 0.0:
        follow_probability = 1.0 / (1.0 + np.exp(-beta))
    else:
        exp_beta = np.exp(beta)
        follow_probability = exp_beta / (1.0 + exp_beta)

    core_a = (
        intended_a * follow_probability
        + (1.0 - intended_a) * (1.0 - follow_probability)
    )
    p_a = (1.0 - lapse_rate) * core_a + lapse_rate * 0.5
    p_a = float(np.clip(p_a, 0.0, 1.0))

    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    validities = np.array([\n        0.55, 0.95, 0.5525, 0.55, 0.5, 0.5868,\n        0.55, 0.95, 0.6343, 0.55, 0.5, 0.6901,\n        0.55, 0.95, 0.7504, 0.55, 0.5, 0.8177\n    ], dtype=float)\n    log_odds = np.log(validities / (1.0 - validities))\n\n    gaps = []\n    follows_higher = []\n\n    for a_cell, b_cell, response in zip(\n        data['option_a_ratings'],\n        data['option_b_ratings'],\n        data['response']\n    ):\n        a = np.asarray(a_cell, dtype=float)\n        b = np.asarray(b_cell, dtype=float)\n        diff = a - b\n        discriminating = np.flatnonzero(diff != 0)\n        if discriminating.size != 2:\n            continue\n\n        j0, j1 = int(discriminating[0]), int(discriminating[1])\n        if validities[j0] == validities[j1]:\n            continue\n        high = j0 if validities[j0] > validities[j1] else j1\n        low = j1 if high == j0 else j0\n\n        high_favors_a = diff[high] > 0\n        chose_a = int(response) == 0\n        follows_higher.append(float(chose_a == high_favors_a))\n        gaps.append(float(log_odds[high] - log_odds[low]))\n\n    gaps = np.asarray(gaps, dtype=float)\n    follows_higher = np.asarray(follows_higher, dtype=float)\n    if gaps.size == 0:\n        return float('nan')\n\n    unique_gaps = np.unique(np.round(gaps, 10))\n    if unique_gaps.size < 2:\n        return float('nan')\n\n    smallest = unique_gaps[0]\n    largest = unique_gaps[-1]\n    low_mask = np.isclose(gaps, smallest, atol=1e-9, rtol=0.0)\n    high_mask = np.isclose(gaps, largest, atol=1e-9, rtol=0.0)\n\n    return float(\n        np.mean(follows_higher[high_mask])\n        - np.mean(follows_higher[low_mask])\n    )",
  "rationale": "This endpoint crossover contrast subtracts the probability of following the higher-validity cue at the smallest validity-log-odds gap from that probability at the largest gap. It pools each trial with its exact A/B reversal, so response-side preferences are balanced, and it uses 16 observations at each endpoint per subject. Take The Best predicts the same probability of following the higher-validity cue at both endpoints, making the expected contrast zero even when its response-noise parameters vary across subjects. Noisy Priority Race Search predicts a large positive contrast: the two-position serial advantage makes the earlier, slightly less valid cue likely to win at the smallest gap, whereas the large validity advantage dominates serial priority and representation bias at the largest gap. Restricting the statistic to the two parameter-robust endpoints avoids the additional between-subject crossover-location variance present in intermediate conditions."
}
```

## Usage

```json
{
  "input_tokens": 5787,
  "output_tokens": 1818
}
```
