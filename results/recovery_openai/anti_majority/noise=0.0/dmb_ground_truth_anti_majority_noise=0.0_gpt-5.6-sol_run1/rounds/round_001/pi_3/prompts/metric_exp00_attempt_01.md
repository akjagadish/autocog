# metric_exp00_attempt_01

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
**Validities (n_features=8):** [0.95, 0.508, 0.508, 0.508, 0.508, 0.508, 0.508, 0.508]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 0, 1]
  trial 4: A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  trial 7: A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 0, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  trial 9: A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 11: A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Rationale:** The 12 pairs comprise six weak-cue conflict levels and their exact A/B reversals, giving 96 trials with eight repetitions per pair. Expert 1 is highly valid (95%) but always ties, so it cannot determine a choice. Experts 2–8 all have validity 50.8%. Even at the advocated theory's maximum validity sensitivity, their subjective confidence is only 1.5*log(0.508/0.492)≈0.048, below its minimum confidence threshold of 0.05. Consequently, no discriminating cue can trigger cascade stopping, and Confidence-Capped Cascade Integration must combine all discriminating weak cues. Within each base pair, Expert 2 favors A and is the first discriminating cue under stable validity-tie ordering, while M of the remaining experts favor B, for M=1,...,6. Take The Best therefore predicts the same above-chance preference for A at every conflict level. The advocated theory instead predicts exact indifference when M=1 and an increasingly strong preference for B as M rises: its integrated evidence is proportional to (1-M)*[log(0.508/0.492)]^integration_exponent. Thus the critical signature is a reversal plus a graded conflict-level effect, neither of which TTB can generate with its single condition-invariant winner probability. Exact option reversals balance response-side biases while preserving these predictions. The tied 95% expert also supplies the required broad validity spread without accidentally activating the advocated theory's stopping rule.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Confidence-Capped Cascade Integration proposes that people translate each communicated cue validity into an absolute diagnosticity signal and inspect cues from most to least valid. A discriminating cue controls the decision immediately when its diagnosticity exceeds a subject-specific confidence threshold. Earlier weak cues do not force premature stopping: search continues until a sufficiently diagnostic cue is encountered. If no discriminating cue reaches threshold, the decision maker switches to compensatory integration, summing all discriminating cues with weights determined by their stated validities. Thus, the theory produces one-cue dominance when reliable cues are available but graded multi-cue sensitivity when every cue is weak. Subject heterogeneity arises from differences in confidence threshold, sensitivity to stated validity, integration weighting, response temperature, and lapse rate.

**Parameters:**
- n_features: n_features
- validities: validities
- confidence_threshold: [0.05, 0.2]
- validity_sensitivity: [0.75, 1.5]
- integration_exponent: [0.5, 2.5]
- beta: [4.0, 16.0]
- epsilon: [0.0, 0.15]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Confidence-Capped Cascade Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got shape {validities.shape}."
        )

    threshold = float(parameters["confidence_threshold"])
    sensitivity = float(parameters["validity_sensitivity"])
    integration_exponent = float(parameters["integration_exponent"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # A validity is the communicated probability that a cue points toward
    # the superior option. Its log odds are therefore an absolute measure
    # of the cue's diagnosticity. Clipping handles validity exactly equal
    # to 0.5 or 1 without numerical singularities.
    v = np.clip(validities, 0.5, 1.0 - 1e-9)
    log_diagnosticity = np.log(v / (1.0 - v))
    subjective_confidence = sensitivity * log_diagnosticity

    # Stable sorting preserves the displayed feature order when two cues
    # have equal communicated validity.
    cue_order = np.argsort(-validities, kind="stable")
    differences = stim[0] - stim[1]

    decision_evidence = None
    for j in cue_order:
        if differences[j] == 0.0:
            continue
        if subjective_confidence[j] >= threshold:
            # The first sufficiently diagnostic discriminating cue caps
            # search. Positive evidence favors A and negative favors B.
            decision_evidence = (
                np.sign(differences[j]) * subjective_confidence[j]
            )
            break

    if decision_evidence is None:
        # When no cue is individually convincing, combine all available
        # weak evidence. The exponent controls how sharply integration
        # privileges validity differences.
        weights = np.power(
            np.maximum(log_diagnosticity, 1e-12), integration_exponent
        )
        decision_evidence = float(
            np.sum(weights * np.sign(differences))
        )

    # Symmetric utilities ensure exact indifference when integrated evidence
    # is zero. Positive evidence favors action 0 (A), negative favors 1 (B).
    logits = beta * np.array(
        [0.5 * decision_evidence, -0.5 * decision_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
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
[0] rationale: This is a within-subject planned contrast measuring the decline in choices of Expert 2's favored option as the number of opposing weak experts rises. Under Take The Best, Expert 2 is always the first discriminating cue, so every conflict level has the same expected choice probability. Because the contrast weights sum exactly to zero, TTB's highly heterogeneous beta and lapse parameters affect the intercept but not the expected metric, keeping between-subject variance low. Confidence-Capped Cascade Integration instead predicts 50% adherence to Expert 2 at conflict level 1 and progressively lower adherence at levels 2–6. The weights emphasize the theoretically strongest and most reliable parts of that nonlinear gradient while using exact A/B reversals to make the coding independent of response side. Positive values therefore indicate the graded compensatory conflict effect predicted by the advocated theory; TTB predicts zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Planned contrast over the six weak-cue conflict levels. The dependent
    # variable is whether the response followed Expert 2, the first
    # discriminating cue used by Take The Best.
    sums = np.zeros(6, dtype=float)
    counts = np.zeros(6, dtype=float)

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # Expert 2 is feature index 1 and always discriminates in this design.
        first_direction = np.sign(d[1])
        if first_direction == 0:
            continue

        # Conflict level is the number of lower-ranked experts pointing in
        # the direction opposite to Expert 2.
        lower_directions = np.sign(d[2:])
        conflict = int(np.sum(lower_directions * first_direction < 0))
        if conflict < 1 or conflict > 6:
            continue

        response = int(row["response"])
        expert2_winner = 0 if first_direction > 0 else 1
        followed_expert2 = float(response == expert2_winner)

        sums[conflict - 1] += followed_expert2
        counts[conflict - 1] += 1.0

    if np.any(counts == 0):
        return float("nan")

    level_means = sums / counts

    # Centered weights sum to zero, eliminating any condition-invariant TTB
    # preference. They approximate the nonlinear conflict profile expected
    # after averaging over the advocated theory's parameter ranges.
    weights = np.array([10.0, 4.0, 0.0, -3.0, -5.0, -6.0])
    return float(np.dot(weights, level_means) / np.sum(np.abs(weights)))

outcome: self_sim=0.0473 (var=0.0068) adversary_sim=0.0066 (var=0.0020) welch_t=+2.176 p=0.0361 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Proportion of choices following the net direction of all weak cues on\n    # trials where that direction opposes Expert 2. These are conflict levels\n    # M=2,...,6; the M=1 cancellation trials are omitted.\n    scores = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        directions = np.sign(a - b)\n\n        # Feature 0 always ties. Features 1 onward are the equally valid weak\n        # cues that the advocated theory integrates.\n        weak_net = float(np.sum(directions[1:]))\n        expert2_direction = float(directions[1])\n\n        # Retain only genuine reversal trials: the weak-cue aggregate points\n        # opposite to the first discriminating cue used by TTB.\n        if weak_net == 0.0 or expert2_direction == 0.0:\n            continue\n        if np.sign(weak_net) == np.sign(expert2_direction):\n            continue\n\n        integrated_winner = 0 if weak_net > 0.0 else 1\n        scores.append(float(int(row[\"response\"]) == integrated_winner))\n\n    if len(scores) == 0:\n        return float(\"nan\")\n    return float(np.mean(scores))",
  "rationale": "This metric measures the central qualitative reversal rather than the comparatively small centered gradient used by the failed metric. At conflict levels 2\u20136, the aggregate of the equally valid weak cues points opposite to Expert 2. Confidence-Capped Cascade Integration therefore favors the aggregate winner, with strength generally increasing across those levels. Take The Best always follows Expert 2, so it chooses the metric-defined winner only through response noise or lapses. Pooling all five reversal levels supplies 80 observations per subject, reducing binomial variability, while exact A/B reversals cancel response-side bias. The expected contrast is consequently a large difference in overall reversal rate rather than a small difference between noisy condition means."
}
```

## Usage

```json
{
  "input_tokens": 5271,
  "output_tokens": 3815
}
```
