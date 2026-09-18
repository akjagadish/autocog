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
**Validities (n_features=8):** [0.5, 0.95, 0.51, 0.96, 0.52, 0.97, 0.53, 1.0]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Rationale:** The validity sequence is globally anti-aligned with display position: validity ranks increase strongly toward later positions, so the competing theory's rank-position correlation is positive and its mapping-simplicity term is clipped to zero. Consequently, Accessibility-Gated Stochastic Lexicographic Search must ignore communicated validity in this experiment. Each diagnostic trial pits an earlier, low-validity cue against the immediately following, much higher-validity cue. Under the advocated theory, the 0.45-0.47 validity increase is large enough to reverse the local positional ordering for every allowed parameter combination, including the accessibility offsets. It therefore follows the later, high-validity cue and predicts approximately 0.78-0.86 choice probability for its product. The competing theory instead gives the later cue no validity advantage; with only a small one-position accessibility difference, it predicts choices near chance and, on average, a slight preference for the earlier low-validity cue (roughly 0.45-0.53 for the high-validity-cue winner across parameter extremes). Thus the critical signature is systematic selection of locally validity-promoted later cues rather than near-chance choices. Four adjacent-cue conflicts establish that the effect generalizes across the display rather than depending on one expert. Each conflict has an A/B mirror, controlling product-side and response-key biases. The eight unique pairs repeat 12 times each for 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Accessibility-Anchored Fixed Lexicographic Search. Each person constructs one stable cue hierarchy at the beginning of a task and reuses it across trials. Priority is dominated by display position, with earlier cues receiving a large accessibility advantage. Subject-specific long-run accessibility or salience offsets can alter priorities locally, while communicated validity provides only a weak additive adjustment. The strength of that adjustment is expressed as a narrowly distributed ratio to positional anchoring, preventing independent parameter draws from causing unstable population-wide cue-order reversals. Thus validity can resolve competition between nearby positions but cannot promote a late cue merely because it has the highest instructed validity. Choice follows the first discriminating cue in this fixed hierarchy. Rare search slips skip that cue and consult the next discriminating cue, while bounded response imprecision and a small lapse rate generate occasional choices against the selected cue. The hierarchy is not resampled trial by trial and does not depend on the number of conflicting later cues.

**Parameters:**
- validities: validities
- accessibility_weights: [(0.0, 0.03)] * n_features
- position_strength: [0.12, 0.16]
- validity_position_ratio: [3.80, 3.90]
- order_perturbation: [0.0, 0.03]
- response_precision: [1.45, 1.85]
- epsilon: [0.01, 0.04]

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Accessibility-Anchored Fixed Lexicographic Search.
    # Parameters are fixed for the subject, so the resulting hierarchy is
    # stable across all trials. History is intentionally ignored because no
    # outcome feedback is available to support cue-order learning.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(
        parameters["accessibility_weights"], dtype=float
    )

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != "
            f"n_features {n_features}."
        )

    # Position imposes a linear cost for moving attention to later cues.
    # Raw validity differences are deliberately not converted to ranks:
    # communicated validity is only a weak local adjustment and therefore
    # cannot receive an extra boost when its ordering matches the display.
    position_strength = float(parameters["position_strength"])
    validity_position_ratio = float(parameters["validity_position_ratio"])
    validity_weight = validity_position_ratio * position_strength
    positions = np.arange(n_features, dtype=float)
    centered_validity = validities - float(np.mean(validities))

    priority = (
        -position_strength * positions
        + accessibility
        + validity_weight * centered_validity
    )

    # Stable sorting supplies a deterministic tie rule favoring earlier
    # positions. Since all components are subject-level constants, this order
    # is fixed rather than redrawn on every trial.
    cue_order = np.argsort(-priority, kind="stable")

    a, b = stim[0], stim[1]
    discriminating_order = [int(j) for j in cue_order if a[j] != b[j]]
    if len(discriminating_order) == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def cue_winner(j):
        return 0 if a[j] > b[j] else 1

    first_winner = cue_winner(discriminating_order[0])

    # A rare search-control slip skips the first discriminating cue. This is
    # a tightly constrained perturbation of a fixed hierarchy, not a newly
    # sampled Plackett-Luce ordering.
    order_perturbation = float(parameters["order_perturbation"])
    order_perturbation = float(np.clip(order_perturbation, 0.0, 1.0))
    if len(discriminating_order) >= 2:
        second_winner = cue_winner(discriminating_order[1])
    else:
        second_winner = first_winner

    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - order_perturbation
    p_selected[second_winner] += order_perturbation

    # Conditional on the selected cue, response precision controls accidental
    # reversal of its recommendation. The stable logistic form avoids an
    # unconstrained trial-wise attentional softmax.
    response_precision = float(parameters["response_precision"])
    if response_precision >= 0.0:
        response_reliability = 1.0 / (1.0 + np.exp(-response_precision))
    else:
        e = np.exp(response_precision)
        response_reliability = e / (1.0 + e)

    p_core = (
        response_reliability * p_selected
        + (1.0 - response_reliability) * p_selected[::-1]
    )

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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Soft prevalence of subjects reliably choosing the later conflicting cue.\"\"\"\n    if data is None or len(data) == 0:\n        return float(\"nan\")\n\n    scores = []\n    for _, subj in data.groupby(\"subject_id\", sort=False):\n        later_cue_choices = []\n        for _, row in subj.iterrows():\n            a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n            b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n            differing = np.flatnonzero(a != b)\n            if differing.size == 0:\n                continue\n\n            # In this design, the later of the two discriminating cues is the\n            # locally high-validity cue.\n            later = int(np.max(differing))\n            later_winner = 0 if a[later] > b[later] else 1\n            later_cue_choices.append(float(int(row[\"response\"]) == later_winner))\n\n        if len(later_cue_choices) == 0:\n            continue\n\n        rate = float(np.mean(later_cue_choices))\n        # Smoothly score whether the subject shows a reliable (>65%) rather\n        # than near-chance preference for the validity-promoted later cue.\n        z = float(np.clip(20.0 * (rate - 0.65), -60.0, 60.0))\n        scores.append(1.0 / (1.0 + np.exp(-z)))\n\n    if len(scores) == 0:\n        return float(\"nan\")\n    return float(np.mean(scores))",
  "rationale": "The metric is a soft prevalence score for a theoretically diagnostic behavioral pattern: reliably selecting the later, much more valid cue in each adjacent-cue conflict. The advocated fixed hierarchy predicts later-cue choice rates around 0.78\u20130.86, whereas the gated stochastic theory predicts rates near 0.5 because validity is suppressed by its mapping gate. Centering the smooth threshold at 0.65 places it well between those regimes. Averaging all 96 mirrored observations before applying the sigmoid reduces trial noise, while the bounded transformation compresses within-theory subject variation and preserves a large between-theory difference. Computing the transformation separately by subject and then averaging also makes the pooled value an interpretable soft proportion of subjects exhibiting the diagnostic preference."
}
```

## Usage

```json
{
  "input_tokens": 5255,
  "output_tokens": 1711
}
```
