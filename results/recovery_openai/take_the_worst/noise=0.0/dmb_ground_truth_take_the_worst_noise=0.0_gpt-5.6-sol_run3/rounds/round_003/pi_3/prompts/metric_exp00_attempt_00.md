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
**Validities (n_features=10):** [0.51, 0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86, 0.91, 0.96]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This sparse-conflict design isolates the competing theory's terminal-gap mechanism without crossing its serial–parallel transition. Validities increase monotonically with display position, yielding rank-position correlation +1 and therefore mapping simplicity 0 in Accessibility-Gated Stochastic Lexicographic Search. Its validity gate is exactly closed: choice is governed by nearly uniform accessibility plus a shallow positional gradient. Each base trial has five discriminating cues. Features 1 and 2 favor the designated option, while three cues favor its opponent. Those three opposing cues move together from features 3–5 through features 8–10. Accessibility-gated search should therefore initially favor the 3-to-2 cue majority and show only a gradual increase in designated-option choice as the opposing cues move later (roughly from about .37–.45 toward .45–.55, depending on attentional parameters), with no special discontinuity at a particular gap. Conflict-Density Adaptive Serial–Parallel Control remains almost purely serial because k=5 is far below its 7.25–7.35 transition. Its leading discriminating cue always favors the designated option. More importantly, opposing residual cues begin receiving a strong terminal-gap penalty once they lie more than 55% of the display beyond feature 1. It consequently predicts substantially higher designated-option choice throughout (approximately .60–.65 for opposing cues at features 3–5) and a comparatively sharp rise to approximately .70–.76 when they reach features 8–10. The critical quantitative dissociation is thus a low, shallow cue-location gradient under accessibility-gated stochastic search versus a high curve with accelerated growth as residual cues cross the terminal-gap boundary under adaptive serial control. Every configuration has an A/B mirror, giving 12 unique pairs repeated eight times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Accessibility-Gated Stochastic Lexicographic Search: People maintain a stable attentional priority over cues based primarily on visual position and learned accessibility. Communicated validities modify that priority only when the validity-to-position mapping is simple enough to encode and maintain. Mapping simplicity is assessed from how closely validity rank follows display order, while a subject-specific mapping-fidelity parameter limits the influence of this information. On each choice, attention follows a noisy priority ordering and search stops at the first cue that discriminates. Thus, stable priorities guide decisions without imposing an identical deterministic cue order on every trial. When validity and position are aligned, validity information gives the accessible first cue a sufficiently large advantage to produce nearly invariant one-cue choices. Under a nonmonotonic mapping, weak validity gating and modest attentional fluctuations distribute search across accessible cues, producing systematic choices that are approximately orthogonal to validity-ranked search without requiring cancellation between strongly polarized subjects. A small lapse probability captures response errors.

**Parameters:**
- validities: validities
- accessibility_weights: [(0.48, 0.52)] * n_features
- position_strength: [0.02, 0.15]
- validity_strength: [3.0, 5.0]
- mapping_fidelity: [0.8, 1.0]
- complexity_power: [3.0, 8.0]
- attention_temperature: [0.30, 0.40]
- epsilon: [0.02, 0.15]

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Accessibility-Gated Stochastic Lexicographic Search.
    # The subject has a stable, subject-specific cue priority. Validity
    # affects this priority strongly only when validity rank is easy to map
    # onto visual position. A Plackett-Luce attention process samples the
    # cue order, and search stops at the first discriminating cue.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(parameters["accessibility_weights"], dtype=float)

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != n_features {n_features}."
        )

    if n_features == 1:
        position = np.zeros(1, dtype=float)
        validity_rank_score = np.ones(1, dtype=float)
        mapping_simplicity = 1.0
    else:
        # Earlier positions have greater default visual accessibility.
        position = np.arange(n_features, dtype=float) / float(n_features - 1)

        # Bounded validity-rank representation: 1 for the most valid cue and
        # 0 for the least valid. Stable sorting makes ties favor earlier cues.
        validity_order = np.argsort(-validities, kind="stable")
        ranks = np.empty(n_features, dtype=float)
        ranks[validity_order] = np.arange(n_features, dtype=float)
        validity_rank_score = 1.0 - ranks / float(n_features - 1)

        # A descending validity sequence is easy to map onto display position.
        # Compute rank-position coherence directly with NumPy so prediction
        # does not depend on an external scipy namespace being initialized.
        x = position - np.mean(position)
        y = validity_rank_score - np.mean(validity_rank_score)
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if not np.isfinite(denom) or denom <= 0.0:
            mapping_simplicity = 0.0
        else:
            rho = float(np.sum(x * y) / denom)
            mapping_simplicity = float(np.clip(-rho, 0.0, 1.0))

    complexity_power = float(parameters["complexity_power"])
    mapping_fidelity = float(parameters["mapping_fidelity"])
    mapping_gate = mapping_fidelity * (mapping_simplicity ** complexity_power)
    mapping_gate = float(np.clip(mapping_gate, 0.0, 1.0))

    position_strength = float(parameters["position_strength"])
    validity_strength = float(parameters["validity_strength"])

    # When the mapping is easy, communicated validity reshapes accessibility.
    # When it is difficult, stable learned accessibility dominates instead.
    learned_component = (1.0 - mapping_gate) * accessibility
    positional_component = position_strength * (1.0 - position)
    validity_component = validity_strength * mapping_gate * validity_rank_score
    priority = learned_component + positional_component + validity_component

    a, b = stim[0], stim[1]
    discriminating = np.flatnonzero(a != b)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Under a Plackett-Luce/Gumbel ordering, the probability that a cue is the
    # first discriminating cue is its softmax weight among discriminating cues.
    attention_temperature = float(parameters["attention_temperature"])
    logits = priority[discriminating] / attention_temperature
    logits = logits - np.max(logits)
    cue_probs = np.exp(logits)
    cue_probs /= cue_probs.sum()

    p_core = np.zeros(2, dtype=np.float64)
    for j, cue_prob in zip(discriminating, cue_probs):
        if a[j] > b[j]:
            p_core[0] += cue_prob
        else:
            p_core[1] += cue_prob

    epsilon = float(parameters["epsilon"])
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Conflict-Density Adaptive Serial–Parallel Control theory. People do not commit to one search architecture for an entire experiment. Instead, each display automatically recruits a control mode based on the absolute number of discriminating cues. Below a capacity-like transition point, attention follows a stable hierarchy dominated by display position and choice is normally based on the first discriminating cue. A constrained accessibility failure can occasionally make that cue unavailable, after which attention continues serially through the remaining discriminating cues using a shallow rank-normalized accessibility gradient. Residual cues supporting both options create modest interference with the initial stopping decision and therefore smoothly increase the probability of continuing beyond the leading cue; unanimous residual evidence does not. A terminal-gap penalty selectively reduces access to residual cues that are separated from the leading cue by a very large positional gap. Instructed validity only provides a weak tie-breaking adjustment and cannot independently promote a substantially later cue. When the number of discriminating cues exceeds the transition point, the display supports parallel comparison and evidence is integrated using a noisy tally with only weak positional and validity weighting. The transition between modes is smooth and stimulus-contingent, rather than determined by experiment identity, trial history, or feedback.

**Parameters:**
- validities: validities
- transition_count: [7.25, 7.35]
- transition_steepness: [7.0, 8.5]
- position_bias: [0.90, 1.05]
- validity_modifier: [0.01, 0.04]
- serial_second_look: [0.36, 0.40]
- mixed_residual_boost: [0.10, 0.14]
- residual_rank_decay: [0.08, 0.12]
- terminal_gap_penalty: [4.0, 5.0]
- serial_precision: [1.65, 1.90]
- positional_discount: [0.06, 0.14]
- tally_precision: [1.05, 1.35]
- epsilon: [0.01, 0.04]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    # Conflict-Density Adaptive Serial-Parallel Control.
    # History is intentionally unused: without outcome feedback, the stable
    # accessibility hierarchy and stimulus-triggered control rule do not learn
    # across trials.
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    diff = a - b
    discriminating = np.flatnonzero(diff != 0.0)
    k = int(discriminating.size)
    if k == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Put validity on a bounded scale. Its influence remains weak in both
    # modes and therefore cannot by itself overcome a full position step.
    v_range = float(np.max(validities) - np.min(validities))
    if np.isfinite(v_range) and v_range > 0.0:
        v_scaled = (validities - np.mean(validities)) / v_range
    else:
        v_scaled = np.zeros(n_features, dtype=np.float64)

    validity_modifier = float(parameters["validity_modifier"])

    # Sparse mode: stable, position-dominated serial hierarchy. The validity
    # adjustment is much smaller than the priority difference between adjacent
    # display positions, so it serves only as a local tie-breaker.
    position_bias = float(parameters["position_bias"])
    positions = np.arange(n_features, dtype=np.float64)
    serial_priority = -position_bias * positions + validity_modifier * v_scaled
    serial_order = np.argsort(-serial_priority, kind="stable")

    discriminating_order = [int(j) for j in serial_order if diff[j] != 0.0]
    first_discriminating = discriminating_order[0]
    first_winner = 0 if diff[first_discriminating] > 0.0 else 1

    # The hierarchy itself remains stable, but a constrained accessibility
    # failure sometimes omits its leading discriminating member. Mixed support
    # among the residual cues creates interference with stopping and modestly
    # increases continuation; unanimous residual support leaves it unchanged.
    # Attention then continues through the residual hierarchy. Conditional on
    # a miss, residual accessibility decays shallowly by ordinal rank. A
    # separate terminal-gap penalty limits access to cues lying very far from
    # the leading cue without reinstating a steep gradient over early/middle
    # cues.
    serial_second_look = float(np.clip(
        parameters["serial_second_look"], 0.0, 1.0
    ))
    if len(discriminating_order) > 2:
        residual_signs = np.sign(diff[np.asarray(discriminating_order[1:], dtype=int)])
        residual_mixedness = 1.0 - abs(float(np.mean(residual_signs)))
        mixed_residual_boost = float(parameters["mixed_residual_boost"])
        serial_second_look += mixed_residual_boost * residual_mixedness
        serial_second_look = float(np.clip(serial_second_look, 0.0, 1.0))

    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - serial_second_look

    if len(discriminating_order) == 1:
        p_selected[first_winner] += serial_second_look
    else:
        residual = np.asarray(discriminating_order[1:], dtype=int)
        residual_rank_decay = float(parameters["residual_rank_decay"])
        terminal_gap_penalty = float(parameters["terminal_gap_penalty"])

        if residual.size > 1:
            residual_rank = np.arange(
                residual.size, dtype=np.float64
            ) / float(residual.size - 1)
        else:
            residual_rank = np.zeros(1, dtype=np.float64)

        if n_features > 1:
            normalized_gap = (
                positions[residual] - positions[first_discriminating]
            ) / float(n_features - 1)
        else:
            normalized_gap = np.zeros(residual.size, dtype=np.float64)

        large_gap = np.maximum(normalized_gap - 0.55, 0.0)
        residual_logits = (
            -residual_rank_decay * residual_rank
            -terminal_gap_penalty * large_gap
        )
        residual_logits -= np.max(residual_logits)
        residual_probs = np.exp(residual_logits)
        residual_probs /= residual_probs.sum()

        for j, cue_prob in zip(residual, residual_probs):
            residual_winner = 0 if diff[j] > 0.0 else 1
            p_selected[residual_winner] += serial_second_look * cue_prob

    serial_precision = float(parameters["serial_precision"])
    if serial_precision >= 0.0:
        serial_reliability = 1.0 / (1.0 + np.exp(-serial_precision))
    else:
        ex = np.exp(serial_precision)
        serial_reliability = ex / (1.0 + ex)

    p_serial = (
        serial_reliability * p_selected
        + (1.0 - serial_reliability) * p_selected[::-1]
    )

    # Dense mode: all discriminating cues contribute in parallel. Position
    # produces only a shallow gradient here, and validity is a weak multiplier
    # rather than a lexicographic ordering mechanism.
    if n_features > 1:
        normalized_position = positions / float(n_features - 1)
    else:
        normalized_position = np.zeros(1, dtype=np.float64)

    positional_discount = float(parameters["positional_discount"])
    tally_weights = np.exp(-positional_discount * normalized_position)
    tally_weights *= 1.0 + validity_modifier * v_scaled
    tally_weights = np.clip(tally_weights, 1e-8, None)

    signed_tally = float(np.dot(tally_weights, diff))
    tally_precision = float(parameters["tally_precision"])
    tally_logits = np.array(
        [tally_precision * signed_tally, 0.0], dtype=np.float64
    )
    tally_logits -= np.max(tally_logits)
    p_tally = np.exp(tally_logits)
    p_tally /= p_tally.sum()

    # A smooth capacity transition makes the architecture contingent on the
    # current conflict count. The narrow parameter ranges encode constrained
    # subject heterogeneity while preserving a common control regime.
    transition_count = float(parameters["transition_count"])
    transition_steepness = float(parameters["transition_steepness"])
    transition_argument = float(np.clip(
        transition_steepness * (float(k) - transition_count), -60.0, 60.0
    ))
    dense_gate = 1.0 / (1.0 + np.exp(-transition_argument))

    p_core = (1.0 - dense_gate) * p_serial + dense_gate * p_tally

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs) -> int:
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Proportion of choices agreeing with the earliest discriminating cue.\"\"\"\n    agreements = []\n\n    for a_cell, b_cell, response in zip(\n        data[\"option_a_ratings\"],\n        data[\"option_b_ratings\"],\n        data[\"response\"]\n    ):\n        a = np.asarray(a_cell, dtype=float)\n        b = np.asarray(b_cell, dtype=float)\n        diff = a - b\n        discriminating = np.flatnonzero(diff != 0)\n        if discriminating.size == 0 or pd.isna(response):\n            continue\n\n        first = int(discriminating[0])\n        leading_choice = 0 if diff[first] > 0 else 1\n        agreements.append(float(int(response) == leading_choice))\n\n    if len(agreements) == 0:\n        return float(\"nan\")\n    return float(np.mean(agreements))",
  "rationale": "This is the position-led stopping rate: the fraction of trials on which the subject chooses the option supported by the earliest discriminating feature. In every designed pair, the earliest cue supports the designated two-cue option, while three other cues support its opponent. Accessibility-gated stochastic search distributes first attention across all five discriminating cues and should therefore produce a rate near 0.4\u20130.5. The competing serial-control theory assigns substantial probability to stopping at the leading cue and predicts a markedly higher rate, approximately 0.6\u20130.7. Pooling all 96 trials gives a low-noise subject-level estimate, while the mirrored A/B trials make the measure insensitive to a general response-side bias."
}
```

## Usage

```json
{
  "input_tokens": 6578,
  "output_tokens": 1654
}
```
