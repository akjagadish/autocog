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
**Validities (n_features=8):** [0.85, 0.82, 0.79, 0.76, 0.73, 0.7, 0.67, 0.6]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** The broad validity profile forces DCSC into its compensatory regime while preserving a unique highest-validity cue. With diagnosticities [0.70, 0.64, 0.58, 0.52, 0.46, 0.40, 0.34, 0.20], the maximum DCSC dominance score is approximately 0.215, below the minimum dominance threshold of 0.285; even at the least decisive allowed gap sensitivity, the dominance gate is below 0.008. In every underlying pattern, Expert 1 discriminates for a designated top-cue winner, so Take The Best always selects that option with the same probability q = epsilon/2 + (1-epsilon)*sigmoid(beta), regardless of all lower cues. DCSC predicts a distinctive crossover. When only one lower expert opposes Expert 1, the top-cue winner remains weakly favored because its reliability weight is larger. This preference should be stronger when the lone opponent is the weakest expert (Expert 8) rather than Expert 2, directly testing compressed reliability weighting. Once two or more lower experts form the opposing coalition, DCSC predicts the coalition's option above chance, with increasingly strong but diminishing support as the coalition grows from two to three, five, and seven experts. Thus TTB predicts complete invariance, whereas DCSC predicts both a directional crossover between one- and multi-opponent trials and a nonlinear coalition-size gradient. Unlike the previously explored tied-anchor design, the highest-validity anchor is unique here and DCSC initially agrees with TTB before lower-cue accumulation overturns it. Exact A/B reversals control response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Dominance-Calibrated Saturating Coalitions (DCSC) proposes that people classify the communicated validity profile before evaluating individual options. A cue activates a dominance regime when it accounts for a sufficiently large share of total diagnosticity and the profile as a whole is concentrated rather than broadly distributed across similarly informative cues. In that regime, the highest-validity discriminating cue anchors the decision, while the remaining cues provide only a small, order-invariant adjustment. Without sufficient profile-level dominance, people enter a compensatory regime and accumulate signed reliability evidence from every discriminating cue. Reliability differences are compressed, and repeated support for the same option has diminishing marginal impact, representing perceived redundancy among concordant experts. Thus, several lower-validity cues can collectively defeat the best discriminating cue, but neither early nor late serial position is intrinsically favored. Evidential balance can induce additional deliberation, modeled as a modest increase in choice consistency rather than a directional recency rule. Stable individual differences concern the dominance threshold, sensitivity to profile concentration, compression and saturation of reliability evidence, leakage of coalition evidence into dominance decisions, deliberation under conflict, and response sensitivity.

**Parameters:**
- validities: validities
- dominance_threshold: [0.285, 0.305]
- gap_sensitivity: [70.0, 110.0]
- profile_shape_weight: [1.8, 2.2]
- validity_compression: [0.02, 0.18]
- redundancy_saturation: [0.08, 0.25]
- anchor_strength: [0.8, 1.2]
- dominance_leak: [0.02, 0.10]
- conflict_deliberation: [0.30, 0.60]
- beta: [2.8, 4.8]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"DCSC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive signs favor B and negative signs favor A. Presentation
    # position never enters the evidence weights.
    signs = np.sign(stim[1] - stim[0])
    discriminating = np.flatnonzero(signs != 0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    dominance_threshold = float(parameters["dominance_threshold"])
    gap_sensitivity = float(parameters["gap_sensitivity"])
    profile_shape_weight = float(parameters["profile_shape_weight"])
    validity_compression = float(parameters["validity_compression"])
    redundancy_saturation = float(parameters["redundancy_saturation"])
    anchor_strength = float(parameters["anchor_strength"])
    dominance_leak = float(parameters["dominance_leak"])
    conflict_deliberation = float(parameters["conflict_deliberation"])
    beta = float(parameters["beta"])

    # Communicated validities are converted to nonnegative diagnostic
    # strengths. The small exponent compresses differences in the
    # compensatory regime without reversing the communicated ordering.
    diagnosticity = np.clip(2.0 * validities - 1.0, 1e-6, 1.0)
    cue_weights = diagnosticity ** validity_compression

    # Dominance depends both on the strongest cue's diagnosticity share
    # and on concentration across the complete profile. The normalized
    # Herfindahl term distinguishes a genuinely concentrated profile from
    # one in which the strongest cue has a similar share but substantial
    # diagnosticity remains broadly distributed among lower cues.
    profile_shares = diagnosticity / max(float(np.sum(diagnosticity)), 1e-12)
    dominance_concentration = float(np.max(profile_shares))
    profile_hhi = float(np.sum(profile_shares ** 2))
    uniform_hhi = 1.0 / float(n_features)
    shape_concentration = (profile_hhi - uniform_hhi) / max(
        1.0 - uniform_hhi, 1e-12
    )
    shape_concentration = float(np.clip(shape_concentration, 0.0, 1.0))
    dominance_score = (
        dominance_concentration
        + profile_shape_weight * shape_concentration
    )
    gate_logit = gap_sensitivity * (
        dominance_score - dominance_threshold
    )
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    dominance_gate = 1.0 / (1.0 + np.exp(-gate_logit))

    # Aggregate reliability separately for the two coalitions. The
    # exponential transform gives diminishing marginal impact to repeated
    # concordant evidence and is invariant to the order in which cues are
    # displayed.
    support_b = float(np.sum(cue_weights[(signs > 0)]))
    support_a = float(np.sum(cue_weights[(signs < 0)]))
    rho = max(redundancy_saturation, 1e-8)
    bounded_b = -np.expm1(-rho * support_b) / rho
    bounded_a = -np.expm1(-rho * support_a) / rho

    # Express coalition evidence in approximately cue-vote units so the
    # response scale remains comparable across validity profiles.
    mean_discriminating_weight = float(np.mean(cue_weights[discriminating]))
    coalition_evidence = (bounded_b - bounded_a) / max(
        mean_discriminating_weight, 1e-12
    )

    # The anchor is the most valid cue that actually discriminates. Ties
    # in communicated validity are resolved without using serial position:
    # their signed evidence is averaged, so tied opposing anchors cancel.
    disc_weights = diagnosticity[discriminating]
    best_weight = float(np.max(disc_weights))
    best_mask = np.isclose(disc_weights, best_weight, rtol=1e-10, atol=1e-12)
    anchor_direction = float(np.mean(signs[discriminating][best_mask]))
    anchor_evidence = anchor_strength * anchor_direction

    # In dominance mode coalition evidence only leaks weakly into the
    # anchored judgment. In integration mode it is fully compensatory.
    dominance_evidence = anchor_evidence + dominance_leak * coalition_evidence
    choice_evidence = (
        dominance_gate * dominance_evidence
        + (1.0 - dominance_gate) * coalition_evidence
    )

    # Balanced conflict can prompt more careful implementation of the
    # selected rule, but it supplies no directional or last-cue evidence.
    count_b = int(np.sum(signs > 0))
    count_a = int(np.sum(signs < 0))
    balance = np.exp(-abs(count_b - count_a))
    beta_eff = beta * (1.0 + conflict_deliberation * balance)

    utilities = np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])
    logits = beta_eff * utilities
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    return probs.astype(np.float64)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Difference in adherence to the highest-validity cue between\n    # one-opponent and multi-opponent coalitions.\n    lone_top_choices = []\n    multi_top_choices = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n\n        # Feature 0 is the uniquely highest-validity expert in this design.\n        top_direction = np.sign(b[0] - a[0])  # +1 means B is top-cue winner\n        if top_direction == 0:\n            continue\n\n        lower_directions = np.sign(b[1:] - a[1:])\n        n_opponents = int(np.sum(lower_directions * top_direction < 0))\n        chose_top_winner = float(\n            int(row[\"response\"]) == (1 if top_direction > 0 else 0)\n        )\n\n        if n_opponents == 1:\n            lone_top_choices.append(chose_top_winner)\n        elif n_opponents >= 2:\n            multi_top_choices.append(chose_top_winner)\n\n    # The fixed schedule supplies both classes for every subject. These\n    # fallbacks keep the function scalar-valued on unexpected partial data.\n    if len(lone_top_choices) == 0 or len(multi_top_choices) == 0:\n        return 0.0\n\n    return float(np.mean(lone_top_choices) - np.mean(multi_top_choices))",
  "rationale": "This is a within-subject crossover contrast. It measures how much more often the highest-validity cue's winner is selected when it faces exactly one opposing lower cue than when it faces a coalition of two or more lower cues. Take The Best predicts the same top-cue adherence probability in both conditions for every beta and epsilon, so its expected contrast is zero; differencing also cancels stable response sensitivity and lapse variation. DCSC predicts a large positive contrast: the top-cue option remains near or above chance against one opponent, whereas compensatory accumulation makes the opposing option strongly preferred for multi-cue coalitions. Pooling all 32 one-opponent and 64 multi-opponent observations yields low sampling variance, while coding choices relative to the top-cue winner makes the exact A/B reversals cancel response-side preferences."
}
```

## Usage

```json
{
  "input_tokens": 5363,
  "output_tokens": 2262
}
```
