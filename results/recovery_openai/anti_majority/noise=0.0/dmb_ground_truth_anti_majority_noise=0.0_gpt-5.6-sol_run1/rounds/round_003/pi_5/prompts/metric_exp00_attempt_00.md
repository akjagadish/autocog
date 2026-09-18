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
**Validities (n_features=24):** [0.72, 0.7, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.5, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 12: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Rationale:** The six base conditions manipulate same-direction redundancy within the validity cascade while holding Take The Best's decisive cue fixed. In every base pair, the 95%-valid Expert 24 favors A, so TTB always stops there and predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2. Experts 1 and 2 favor B but have lower validities (72% and 70%). Condition 1 contains only the late-displayed 95% A cue; conditions 2 through 6 successively add A-favoring backup cues with validities 93%, 91%, 89%, 87%, and 85%. All remaining cues tie. Under the Accessibility-Gated Validity Cascade, each added backup supplies another opportunity for an A-favoring cue to be accessible before search reaches the B-favoring cues. If Q_m is the probability that all m active A cues are inaccessible and S is the probability that both B cues are inaccessible, intended P(A)=1-Q_m+0.5*Q_m*S. Thus A choice must increase monotonically as backups are added, with a characteristic diminishing-returns profile from the multiplicative accessibility failures. The manipulation is especially diagnostic because all A cues occupy terminal display positions 19–24, where the advocated theory predicts appreciable accessibility failures in a 24-cue list, whereas the two B fallbacks occupy the highly accessible first positions. Execution flips and lapses attenuate but preserve the positive redundancy gradient. TTB cannot produce this condition effect because its 95% cue already discriminates in every condition and lower-validity cues are never consulted. The final six pairs are exact A/B reversals, balancing response-side preferences while preserving the predicted redundancy profile. Twelve unique pairs repeated eight times yield 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Accessibility-Gated Validity Cascade: Communicated validities create a stable, descending cue hierarchy rather than a noisy priority race. On each trial, however, a cue can control choice only if its identity, stated validity, and displayed value are successfully accessible. Accessibility varies stably across people and cues and declines with list length and absolute display position, with a modest accelerating cost for cues near the end of long displays. Search follows the advertised-validity hierarchy among accessible cues and stops at the first accessible cue that discriminates. Lower-ranked evidence is never integrated. If no discriminating cue is accessible, the decision maker guesses. Retrieval failures are distinct from response execution errors: after forming an intended choice, the person can accidentally reverse it or lapse to a random response.

**Parameters:**
- n_features: n_features
- validities: validities
- cue_accessibility_offsets: [(-0.15, 0.15)] * n_features
- baseline_accessibility: [3.2, 3.8]
- list_length_gradient: [0.035, 0.055]
- position_gradient: [0.015, 0.035]
- late_position_cost: [0.003, 0.0055]
- execution_flip: [0.09, 0.13]
- lapse_rate: [0.0, 0.02]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Accessibility-Gated Validity Cascade expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but "
            f"n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    cue_offsets = np.asarray(
        parameters["cue_accessibility_offsets"], dtype=np.float64
    )
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got "
            f"shape {validities.shape}."
        )
    if cue_offsets.ndim != 1 or cue_offsets.size != n_features:
        raise ValueError(
            f"cue_accessibility_offsets must have length {n_features}; "
            f"got shape {cue_offsets.shape}."
        )

    baseline = float(parameters["baseline_accessibility"])
    list_gradient = float(parameters["list_length_gradient"])
    position_gradient = float(parameters["position_gradient"])
    late_position_cost = float(parameters["late_position_cost"])
    execution_flip = float(parameters["execution_flip"])
    lapse_rate = float(parameters["lapse_rate"])

    # Absolute zero-based display position. The quadratic component is a
    # restrained terminal-display cost: it has little effect on early cues
    # but makes cues near the end of long lists appreciably harder to encode
    # and retrieve. Cue offsets represent stable identity/complexity effects.
    positions = np.arange(n_features, dtype=np.float64)
    access_logits = (
        baseline
        - list_gradient * max(n_features - 1, 0)
        - position_gradient * positions
        - late_position_cost * positions * positions
        + cue_offsets
    )

    # Numerically stable logistic transformation.
    access_prob = np.empty(n_features, dtype=np.float64)
    positive = access_logits >= 0.0
    access_prob[positive] = 1.0 / (1.0 + np.exp(-access_logits[positive]))
    exp_x = np.exp(access_logits[~positive])
    access_prob[~positive] = exp_x / (1.0 + exp_x)
    access_prob = np.clip(access_prob, 1e-9, 1.0 - 1e-9)

    differences = stim[0] - stim[1]
    discriminating = differences != 0.0
    if not np.any(discriminating):
        return np.array([0.5, 0.5], dtype=np.float64)

    # Advertised validity fixes the search hierarchy. Stable sorting means
    # equal-validity cues retain their displayed order. Accessibility changes
    # whether a cue participates, never its priority relative to another cue.
    cue_order = np.argsort(-validities, kind="stable")

    # Marginalize exactly over independent trial-level accessibility events.
    # reach_probability is the probability that every earlier discriminating
    # cue in the validity hierarchy was inaccessible. Accessible tying cues do
    # not terminate search and therefore leave this quantity unchanged.
    reach_probability = 1.0
    intended_a = 0.0
    intended_b = 0.0

    for j in cue_order:
        if not discriminating[j]:
            continue

        stop_probability = reach_probability * access_prob[j]
        if differences[j] > 0.0:
            intended_a += stop_probability
        else:
            intended_b += stop_probability

        reach_probability *= 1.0 - access_prob[j]

    # If every discriminating cue is inaccessible, no evidence is available.
    intended_a += 0.5 * reach_probability
    intended_b += 0.5 * reach_probability

    intended_total = intended_a + intended_b
    if not np.isfinite(intended_total) or intended_total <= 0.0:
        intended_a = 0.5
    else:
        intended_a /= intended_total

    # Execution reversal is separate from retrieval failure. A final lapse
    # replaces the executed response with a random choice.
    executed_a = (
        intended_a * (1.0 - execution_flip)
        + (1.0 - intended_a) * execution_flip
    )
    p_a = (1.0 - lapse_rate) * executed_a + lapse_rate * 0.5
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Measure choice of the option favored by the highest-validity cue,\n    # separately for sparse versus highly redundant terminal-cue conditions.\n    high_redundancy = []\n    low_redundancy = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.size < 24 or b.size < 24:\n            continue\n\n        diff = a - b\n        # Features 19--24 (zero-based indices 18--23) are the nested set\n        # of high-validity cues favoring the same option.\n        redundancy = int(np.count_nonzero(diff[18:24]))\n\n        # Feature 24 has validity .95 and is TTB's decisive cue on every\n        # experimental trial. Code whether the response follows that cue.\n        if diff[23] > 0:\n            followed_top_cue = float(int(row[\"response\"]) == 0)\n        elif diff[23] < 0:\n            followed_top_cue = float(int(row[\"response\"]) == 1)\n        else:\n            continue\n\n        if redundancy in (1, 2):\n            low_redundancy.append(followed_top_cue)\n        elif redundancy in (5, 6):\n            high_redundancy.append(followed_top_cue)\n\n    if len(low_redundancy) == 0 or len(high_redundancy) == 0:\n        return float(\"nan\")\n\n    return float(np.mean(high_redundancy) - np.mean(low_redundancy))\n",
  "rationale": "This is an extreme-groups redundancy gain: the increase in following the 95%-valid cue when it is supported by five or six same-direction terminal cues rather than only one or two. TTB predicts an expected value of zero because the 95% cue is already the first discriminating cue in every condition; its beta and lapse parameters affect both groups equally. The accessibility-gated cascade predicts a large positive value because additional terminal cues rescue decisions when still-higher-validity terminal cues are inaccessible. Pooling two conditions at each extreme gives 32 observations per group per subject, reducing binomial variance, while coding choices relative to the top cue and combining exact A/B reversals cancels response-side bias. Intermediate conditions are omitted because the endpoint regions provide a larger predicted contrast per unit of trial noise."
}
```

## Usage

```json
{
  "input_tokens": 6330,
  "output_tokens": 1845
}
```
