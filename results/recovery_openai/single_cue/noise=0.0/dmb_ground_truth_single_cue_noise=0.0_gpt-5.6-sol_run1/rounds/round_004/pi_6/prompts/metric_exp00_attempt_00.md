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
**Validities (n_features=4):** [0.95, 0.8, 0.65, 0.52]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 7: A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 8: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Rationale:** This is a fixed-length prefix-coherence test with a protected two-reason suffix. In descending-validity order, the four base sequences are AAAA, BAAA, ABAA, and BBAA; thus every base trial has four discriminating reasons and ends with two reasons favoring A. Recency-Gated Reason Overwrite predicts almost identical A-choice rates across these conditions. If q = 1 - cue_attention*overwrite_rate, two terminal A updates reduce any prefix-dependent difference by q^2; over the allowed ranges q^2 is at most about 0.0095, and common fatigue and lapse transformations reduce the observable contrast further. The Competitive Evidence Buffer instead preserves graded traces of both prefix reasons. It predicts P(A): AAAA > BAAA > ABAA > BBAA. Reversing the earliest reason reduces accessible A evidence by twice its surviving trace; reversing the second reason has a larger effect because that trace undergoes one fewer displacement step. Reversing both produces the largest reduction. Conflict normalization reinforces this gradient because the one- and two-B conditions contain increasingly substantial accessible evidence for both options, whereas AAAA is completely coherent. The critical quantitative signature is therefore a monotonic coherence gradient, including a larger AAAA-versus-ABAA contrast than AAAA-versus-BAAA, against the overwrite theory's essentially flat profile. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Validity-Ordered Competitive Evidence Buffer with Conflict-Normalized Readout. Decision makers inspect cues in descending order of instructed validity and encode every discriminating cue as signed evidence rather than as a provisional choice or directional chunk. Each newly encoded reason causes graded displacement of all earlier traces in a capacity-limited working-memory buffer. Displacement becomes stronger as the buffer exceeds its stable subject-specific capacity, but old evidence is never categorically erased. At response, the decision maker sums all surviving validity-scaled traces. Readout normalization depends on the balance of accessible evidence: it is strongest when substantial trace mass supports both options and weak when the accessible buffer contains a coherent majority. Readout uncertainty additionally grows quadratically with capacity overload and receives a count-based attenuation beginning explicitly at seven discriminating reasons, contracting long conflicts toward chance without imposing arrangement-specific effects. Stable individual differences in effective capacity, displacement rate, validity sensitivity, response gain, and conflict normalization generate heterogeneous recency and integration behavior.

**Parameters:**
- validities: validities
- buffer_capacity: [3.0, 3.8]
- displacement_rate: [2.1, 2.9]
- validity_sensitivity: [0.0, 0.08]
- response_gain: [2.0, 6.5]
- integration_normalization: [0.02, 0.34]
- overflow_noise: [0.75, 1.50]
- lapse_rate: [0.03, 0.15]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive evidence buffer expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    capacity = float(parameters["buffer_capacity"])
    displacement = float(parameters["displacement_rate"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    response_gain = float(parameters["response_gain"])
    integration_normalization = float(parameters["integration_normalization"])
    overflow_noise = float(parameters["overflow_noise"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]

    # Inspection follows instructed validity, independently of physical
    # feature position. Stable sorting only resolves exact validity ties.
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Convert instructed reliability into evidence strength. Normalization
    # removes arbitrary differences in the overall validity scale between
    # experiments while preserving all within-experiment validity ratios.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    positive_reliability = reliability[reliability > 0.0]
    reliability_scale = float(np.mean(positive_reliability))
    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:
        reliability_scale = 1.0

    normalized_reliability = np.clip(reliability / reliability_scale, 1e-4, 20.0)
    cue_strength = np.power(normalized_reliability, validity_sensitivity)
    cue_strength = np.clip(cue_strength, 1e-6, 30.0)

    # The buffer contains separate reason traces. A new discriminating reason
    # interferes continuously with every old trace. Displacement is present
    # even below capacity and increases smoothly under overload; there is no
    # categorical slot deletion and no compression into directional runs.
    traces = []
    for j in discriminating:
        current_load = float(len(traces))
        overload_before_entry = max(0.0, current_load + 1.0 - capacity)
        load_factor = 1.0 + overload_before_entry / max(capacity, 1e-6)
        retention = np.exp(-displacement * load_factor / max(capacity, 1e-6))
        retention = float(np.clip(retention, 0.0, 1.0))

        if traces:
            traces = [retention * trace for trace in traces]

        direction = float(np.sign(diff[j]))  # positive favors A
        traces.append(direction * float(cue_strength[j]))

    accessible_evidence = float(np.sum(traces))

    # Preserve the accepted quadratic capacity-overload uncertainty through
    # six reasons. Beginning explicitly at seven discriminating reasons, add
    # a modest count-based attenuation that is not smeared by capacity.
    overflow = max(0.0, float(m) - capacity)
    long_sequence_excess = max(0.0, float(m) - 6.0)
    convex_multiplier = 1.0 + 0.45 * long_sequence_excess
    readout_sd = np.sqrt(
        1.0 + overflow_noise * overflow * overflow * convex_multiplier
    )

    # Normalize integrated evidence only when accessible traces substantially
    # support both options. Squaring the balance ratio leaves coherent
    # majorities largely intact. The effect remains absent for two reasons.
    positive_mass = float(np.sum([trace for trace in traces if trace > 0.0]))
    negative_mass = float(-np.sum([trace for trace in traces if trace < 0.0]))
    total_mass = positive_mass + negative_mass
    if m <= 2 or total_mass <= 0.0 or not np.isfinite(total_mass):
        conflict_load = 0.0
    else:
        balance = 2.0 * min(positive_mass, negative_mass) / total_mass
        balance = float(np.clip(balance, 0.0, 1.0))
        conflict_load = np.sqrt(max(0.0, float(m) - 2.0)) * balance * balance
    integration_divisor = 1.0 + integration_normalization * conflict_load

    decision_variable = (
        response_gain * accessible_evidence
        / (readout_sd * integration_divisor)
    )
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a_core = 1.0 / (1.0 + np.exp(-decision_variable))
    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)

    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
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
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

**Parameters:**
- validities: validities
- cue_attention: [0.95, 1.0]
- overwrite_rate: [0.95, 1.0]
- update_fatigue: [0.06, 0.17]
- lapse_rate: [0.15, 0.41]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recency-Gated Reason Overwrite expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["cue_attention"])
    overwrite = float(parameters["overwrite_rate"])
    fatigue = float(parameters["update_fatigue"])
    lapse = float(parameters["lapse_rate"])

    # Stable sorting represents the displayed high-to-low validity scan;
    # ties retain their display order.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]

    # Exact marginal distribution over the provisional state:
    # A selected, B selected, or no discriminating cue encoded yet.
    p_a, p_b, p_uncommitted = 0.0, 0.0, 1.0
    discriminating_count = 0

    for j in cue_order:
        direction = diff[j]
        if direction == 0:
            continue

        discriminating_count += 1
        old_a, old_b, old_u = p_a, p_b, p_uncommitted

        if direction > 0:
            # An attended A cue initializes an uncommitted decision and
            # overwrites a conflicting B decision with the overwrite rate.
            p_a = old_a + attention * overwrite * old_b + attention * old_u
            p_b = old_b * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)
        else:
            p_b = old_b + attention * overwrite * old_a + attention * old_u
            p_a = old_a * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)

    # If no discriminating cue was encoded, either response is an equal guess.
    p_core = np.array([
        p_a + 0.5 * p_uncommitted,
        p_b + 0.5 * p_uncommitted
    ], dtype=np.float64)

    # Processing the first reason and its immediate successor is reliable.
    # Additional discriminating updates progressively weaken expression of
    # the retained choice without disrupting the crucial second-cue reversal.
    excess_updates = max(0, discriminating_count - 2)
    retention = (1.0 - fatigue) ** excess_updates
    p_core = retention * p_core + (1.0 - retention) * np.array([0.5, 0.5])

    # Independent execution/inattention lapse.
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Coherence-weighted agreement with the least-valid terminal cue.\"\"\"\n    if len(data) == 0:\n        return float(\"nan\")\n\n    pattern_weights = {\n        (1, 1, 1, 1): 1.00,   # fully coherent\n        (-1, 1, 1, 1): 0.80,  # earliest cue conflicts\n        (1, -1, 1, 1): 0.65,  # second cue conflicts\n        (-1, -1, 1, 1): 0.35  # both prefix cues conflict\n    }\n\n    weighted_agreement = 0.0\n    total_weight = 0.0\n\n    for a_cell, b_cell, response in zip(\n        data[\"option_a_ratings\"],\n        data[\"option_b_ratings\"],\n        data[\"response\"]\n    ):\n        a = np.asarray(a_cell, dtype=float)\n        b = np.asarray(b_cell, dtype=float)\n        diff = a - b\n        if diff.size != 4 or diff[3] == 0:\n            continue\n\n        # Orient every trial so that the terminal (least-valid) cue is +1.\n        terminal_sign = float(np.sign(diff[3]))\n        relative_pattern = tuple(\n            int(np.sign(x) * terminal_sign) for x in diff\n        )\n        weight = pattern_weights.get(relative_pattern, 0.0)\n        if weight <= 0.0:\n            continue\n\n        terminal_favoring_response = 0 if terminal_sign > 0 else 1\n        agreement = float(int(response) == terminal_favoring_response)\n        weighted_agreement += weight * agreement\n        total_weight += weight\n\n    if total_weight <= 0.0:\n        return float(\"nan\")\n    return float(weighted_agreement / total_weight)",
  "rationale": "This metric measures choice agreement with the common two-cue suffix while weighting trials by prefix coherence. It gives greatest precision weight to the fully coherent condition, where the competitive-buffer theory predicts near-ceiling agreement, and progressively less weight to conditions whose surviving prefix traces oppose the suffix. The larger penalty for a conflict at the second cue reflects its stronger surviving buffer trace. Thus the statistic captures both the buffer theory's high overall suffix-consistent responding and its graded coherence profile. The overwrite theory instead predicts an almost flat profile because two terminal updates nearly eliminate prefix dependence; its count-based fatigue and larger lapse range also lower terminal agreement across all four conditions. Orienting reversals by the terminal cue removes response-side bias, and pooling all conditions gives a stable per-subject estimate from 96 trials."
}
```

## Usage

```json
{
  "input_tokens": 4916,
  "output_tokens": 4069
}
```
