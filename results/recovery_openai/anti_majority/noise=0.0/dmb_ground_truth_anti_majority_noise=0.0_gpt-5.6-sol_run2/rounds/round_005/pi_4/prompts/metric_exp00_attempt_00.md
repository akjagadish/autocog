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
**Validities (n_features=15):** [0.99, 0.95, 0.9, 0.85, 0.8, 0.51, 0.55, 0.6, 0.65, 0.7, 0.75, 0.75, 0.75, 0.75, 0.75]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 11: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 12: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** This 15-feature design creates a conflict between a fixed validity residual and count margins of 0, 1, 2, 3, 4, and 5. The validity profile has mean 0.75. In every base trial, A is marked by four low-validity experts (0.55, 0.60, 0.65, 0.70), while B is marked by their four high-validity counterparts (0.95, 0.90, 0.85, 0.80). This produces a constant centered-validity residual of -1.00 and an initially tied count. Successive conditions add marks to A on up to five exactly mean-validity experts. These marks increase the A-minus-B count margin without changing the competitor's centered-validity signal. Under Semantic-Polarity Smooth Integration, markedness evidence is exactly -validity_weight + d*[count_weight + 0.75*(1-partial_centering)*validity_weight], where d is the count margin. Its evidence therefore increases linearly at every step, with a slope between 0.7475 and 2.265; it is necessarily positive by margin 2 and rises by at least 2.99 evidence units from margin 1 to margin 5. The competing theory instead predicts an abrupt reversal from the negative validity signal at margin 0 to positive markedness at margin 1: even its weakest first-margin count component exceeds its strongest possible opposing validity component at 15 features. Thereafter its tanh count accumulator is nearly saturated. From margin 1 to margin 5, its unscaled count signal can increase by no more than about 0.16, while its validity signal remains exactly fixed. Thus the critical quantitative signature is a strong, sustained margin-1-to-margin-5 increase in choices of B under Smooth Integration versus an early margin-1 step followed by an almost flat profile under Dual-Timescale Margin Integration. All six pairs are also A/B reversed. The 12 unique pairs repeated eight times yield exactly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Semantic-Polarity Smooth Integration theory: people first interpret what the marked binary state means rather than assuming that 1 is intrinsically desirable. Each subject forms a stable posterior belief that a mark is adverse or suspicious. Conditional on either the adverse interpretation or its residual beneficial alternative, the subject smoothly integrates two continuous signals: the difference in total marks and a partially count-residualized validity-weighted difference. The count signal normally dominates, while the weighted signal supplies graded sensitivity to which experts carry the marks on every trial. The two semantic interpretations generate opposite choice distributions, which are averaged according to the subject's polarity confidence. A small lapse process captures stimulus-independent errors.

**Parameters:**
- validities: validities
- count_weight: [0.38, 1.05]
- validity_weight: [0.7, 1.8]
- partial_centering: [0.1, 0.3]
- adverse_polarity_confidence: [0.78, 1.0]
- lapse_rate: [0.0, 0.12]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Semantic-Polarity Smooth Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # Positive differences mean that A contains more marked states than B.
    count_difference = float(np.sum(a - b))
    partial_centering = float(parameters["partial_centering"])
    effective_validities = validities - partial_centering * float(np.mean(validities))
    weighted_difference = float(np.dot(effective_validities, a - b))

    # There is no count gate: validity contributes continuously on every
    # discriminating trial. Partial centering limits the count component
    # embedded in the weighted sum without removing it completely.
    integrated_markedness = (
        float(parameters["count_weight"]) * count_difference
        + float(parameters["validity_weight"]) * weighted_difference
    )

    adverse_logits = np.array(
        [-0.5 * integrated_markedness, 0.5 * integrated_markedness],
        dtype=float,
    )
    adverse_logits = adverse_logits - np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # The beneficial interpretation reverses the semantic polarity while
    # preserving the same smooth evidence-integration computation.
    beneficial_probs = adverse_probs[::-1]
    polarity_confidence = float(parameters["adverse_polarity_confidence"])
    core_probs = (
        polarity_confidence * adverse_probs
        + (1.0 - polarity_confidence) * beneficial_probs
    )

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs):
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
**Description:** Polarity-Anchored Dual-Timescale Margin Integration theory: subjects maintain a stable semantic interpretation of whether binary marks are adverse or beneficial. Conditional on that interpretation, tally evidence depends only on the signed absolute count margin, never on display density or the number of discriminating cues. A rapidly compressing accumulator supplies strong evidence at the first nonzero margin, while a very small logarithmic tail preserves continued but sharply diminishing growth across moderate and large margins. Count-residualized validity evidence remains continuously available but is softly standardized and subject to a later, steeper capacity curve that slightly reduces average validity influence at 11 features, preserves relatively more influence at 15 features, and strongly attenuates it by 19 and 37 features. Stable subject differences in count strength, compression, validity attention, semantic polarity, precision, and lapses produce persistent behavioral heterogeneity.

**Parameters:**
- validities: validities
- count_gain: [1.8, 2.6]
- fast_margin_scale: [0.25, 0.55]
- log_tail_gain: [0.002, 0.02]
- validity_attention: [0.5, 1.8]
- reliability_standardization: [0.0, 0.06]
- reliability_shrinkage: [0.15, 0.35]
- validity_capacity: [14.0, 16.0]
- display_load_exponent: [5.0, 7.0]
- response_precision: [0.7, 1.45]
- adverse_polarity_confidence: [0.92, 1.0]
- lapse_rate: [0.0, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Polarity-Anchored Dual-Timescale Margin Integration expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    difference = a - b
    count_difference = float(np.sum(difference))
    margin = abs(count_difference)

    # The count representation depends exclusively on the raw tally margin.
    # A dominant fast component makes the first nonzero margin informative
    # and then compresses rapidly. A much smaller logarithmic tail preserves
    # continued but diminishing growth at moderate and large margins.
    if margin > 0.0:
        fast_component = float(parameters["count_gain"]) * np.tanh(
            margin / float(parameters["fast_margin_scale"])
        )
        slow_component = float(parameters["log_tail_gain"]) * np.log1p(margin)
        count_magnitude = fast_component + slow_component
        count_signal = float(np.sign(count_difference)) * count_magnitude
    else:
        count_signal = 0.0

    # Centering removes the portion of validity-weighted evidence that merely
    # restates which option has more marks. The resulting reliability residual
    # remains available at every margin, including exact count ties.
    centered_validities = validities - float(np.mean(validities))
    raw_validity_residual = float(np.dot(centered_validities, difference))

    # Soft shrinkage standardization avoids disproportionately amplifying
    # low-energy reliability profiles while retaining meaningful differences
    # in reliability dispersion. It uses no trial-specific density or conflict.
    profile_energy = float(np.sqrt(np.dot(
        centered_validities, centered_validities
    )))
    standardization_exponent = float(
        parameters["reliability_standardization"]
    )
    shrinkage_scale = np.sqrt(
        float(parameters["reliability_shrinkage"])
        + profile_energy ** 2
    )
    validity_units = raw_validity_residual / (
        shrinkage_scale ** standardization_exponent
    )

    # Reliability inspection decreases continuously with display complexity.
    # The later, steeper knee preserves relatively more access near 15 cues
    # than the previous curve while retaining strong attenuation by 19 and 37.
    capacity = float(parameters["validity_capacity"])
    load_exponent = float(parameters["display_load_exponent"])
    attention_factor = 1.0 / (
        1.0 + (float(n_features) / capacity) ** load_exponent
    )
    validity_signal = (
        float(parameters["validity_attention"])
        * attention_factor
        * validity_units
    )

    markedness_evidence = float(parameters["response_precision"]) * (
        count_signal + validity_signal
    )

    # Positive markedness means that A carries more adverse evidence and hence
    # favors B under the adverse semantic interpretation.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence],
        dtype=float,
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # A stable alternative interpretation treats marks as beneficial and
    # reverses the entire integrated evidence distribution.
    beneficial_probs = adverse_probs[::-1]
    polarity = float(parameters["adverse_polarity_confidence"])
    core_probs = (
        polarity * adverse_probs
        + (1.0 - polarity) * beneficial_probs
    )

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    margins = []\n    adverse_choices = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        count_difference = float(np.sum(a) - np.sum(b))\n        margin = abs(count_difference)\n\n        # Exclude count ties; the diagnostic trajectory is over margins 1--5.\n        if margin < 0.5:\n            continue\n\n        response = int(row[\"response\"])\n        # Under the adverse-mark interpretation, choose B when A has more\n        # marks and choose A when B has more marks.\n        adverse_choice = response if count_difference > 0 else 1 - response\n        margins.append(margin)\n        adverse_choices.append(float(adverse_choice))\n\n    if len(margins) < 2:\n        return float(\"nan\")\n\n    x = np.asarray(margins, dtype=float)\n    y = np.asarray(adverse_choices, dtype=float)\n    x_centered = x - np.mean(x)\n    denominator = float(np.dot(x_centered, x_centered))\n    if denominator <= 0.0:\n        return float(\"nan\")\n\n    # OLS increase in adverse-aligned choice probability per unit margin.\n    return float(np.dot(x_centered, y - np.mean(y)) / denominator)",
  "rationale": "The metric is the within-subject linear slope of orientation-corrected, adverse-aligned choices across absolute count margins 1 through 5. Smooth Integration predicts a sizable positive slope because its evidence continues increasing at every added mean-validity mark. Dual-Timescale Integration predicts a slope near zero because its count accumulator has already nearly saturated at margin 1 while the centered-validity residual remains fixed. Using all five margins and both A/B orientations yields a lower-variance estimate than an endpoint difference, and the balanced repetitions make the pooled estimate directly comparable to the per-subject estimates."
}
```

## Usage

```json
{
  "input_tokens": 5662,
  "output_tokens": 2397
}
```
