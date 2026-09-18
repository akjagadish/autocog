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
**Validities (n_features=12):** [0.99, 0.95, 0.9, 0.86, 0.78, 0.7, 0.63, 0.58, 0.54, 0.52, 0.505, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This is a fixed-sequence terminal-reliability gradient. In every base trial, exactly four experts discriminate and their descending-validity direction sequence is A,A,A,B. Experts 1–3 always form the same early A coalition; only the identity—and hence advertised validity—of the terminal B expert changes across 86%, 70%, 58%, 52%, 50.5%, and 50%. Recency-Gated Reason Overwrite ignores validity magnitudes after determining scan order, so it predicts exactly the same B-choice probability in all six base conditions: the direction sequence, discriminating count, terminal overwrite, fatigue factor, and lapse transformation are identical. The Reliability-Compressed Recurrent Evidence Field instead has pre-normalization evidence proportional to r^3*s1 + r^2*s2 + r*s3 - s_terminal. It therefore predicts a graded increase in A choice as terminal validity declines. High-validity terminal cues should retain B control, whereas sufficiently weak terminal cues can be outweighed by the surviving three-cue A coalition, potentially producing a modal crossover. Near the crossover, its near-balance normalization additionally predicts especially shallow, near-chance responding rather than a condition-invariant terminal preference. The reliability gradient weakens toward zero only at the boundary case of zero validity compression, providing a direct quantitative estimate of the advocated compression mechanism. Exact A/B reversals balance response side. Twelve unique pairs repeated eight times yield exactly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Reliability-Compressed Recurrent Evidence Field. Decision makers inspect discriminating experts in descending instructed-validity order and encode each attended rating as signed, compressed log-odds evidence. Evidence remains distributed across cue traces, but accessibility declines smoothly with ordinal distance from response, producing graded recency rather than categorical replacement. Choice depends on the signed sum of accessible traces. Simultaneously active support for both products generates divisive interference, with an additional bounded increase when the accessible recurrent state is genuinely near balance. Increasing evidence load raises readout noise, especially under conflict. Cue-attention uncertainty contributes additional encoding noise, while a small vigilance decrement allows only a modest decline for long unanimous displays. Thus terminal control emerges continuously from terminal reliability, the accessibility gradient, opposing reliability-weighted coalition strength, and total interference. The mechanism has no sequence-class rules, physical-gap effects, fixed capacity boundary, or cross-trial learning.

**Parameters:**
- validities: validities
- validity_compression: [0.0, 0.10]
- memory_retention: [0.38, 0.54]
- cue_attention: [0.86, 0.99]
- conflict_normalization: [0.15, 0.55]
- near_balance_normalization: [1.5, 2.5]
- load_noise: [0.03, 0.16]
- vigilance_decline: [0.008, 0.025]
- decision_gain: [3.5, 6.5]
- lapse_rate: [0.04, 0.16]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recurrent evidence field expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    compression = float(parameters["validity_compression"])
    retention = float(parameters["memory_retention"])
    attention = float(parameters["cue_attention"])
    conflict_normalization = float(parameters["conflict_normalization"])
    near_balance_normalization = float(parameters["near_balance_normalization"])
    load_noise = float(parameters["load_noise"])
    vigilance_decline = float(parameters["vigilance_decline"])
    decision_gain = float(parameters["decision_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0.0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Communicated validity is converted to normative log-odds and then
    # strongly compressed. Normalizing by the experiment's typical positive
    # log-odds preserves its reliability profile without allowing arbitrary
    # differences in scale to determine response gain.
    v = np.clip(validities, 0.5001, 0.999)
    log_odds = np.log(v / (1.0 - v))
    positive = log_odds[log_odds > 0.0]
    reliability_scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:
        reliability_scale = 1.0
    relative_reliability = np.clip(
        log_odds / reliability_scale, 1e-4, 20.0
    )
    cue_strength = np.power(relative_reliability, compression)
    cue_strength = np.clip(cue_strength, 0.05, 8.0)

    # Every reason retains a separate trace. Accessibility follows a smooth
    # exponential serial-position gradient: no reason is exactly overwritten,
    # and no capacity or sequence-length boundary is imposed.
    signed_traces = []
    unsigned_traces = []
    omission_variance = 0.0
    for serial_position, j in enumerate(discriminating):
        lag = (m - 1) - serial_position
        accessibility = retention ** float(lag)
        strength = float(cue_strength[j])
        direction = float(np.sign(diff[j]))

        expected_trace = attention * accessibility * strength
        signed_traces.append(direction * expected_trace)
        unsigned_traces.append(expected_trace)

        # Bernoulli attention uncertainty is represented at readout as the
        # variance of a potentially omitted trace.
        omission_variance += (
            attention * (1.0 - attention)
            * (accessibility * strength) ** 2
        )

    signed_traces = np.asarray(signed_traces, dtype=float)
    unsigned_traces = np.asarray(unsigned_traces, dtype=float)
    integrated_evidence = float(np.sum(signed_traces))

    positive_mass = float(np.sum(signed_traces[signed_traces > 0.0]))
    negative_mass = float(-np.sum(signed_traces[signed_traces < 0.0]))
    total_mass = positive_mass + negative_mass

    # Simultaneous opposing coalitions produce divisive interference. The
    # smooth balance index is zero for unanimous evidence and approaches one
    # when accessible support is evenly divided.
    if total_mass > 0.0 and np.isfinite(total_mass):
        conflict_balance = (
            4.0 * positive_mass * negative_mass
            / max(total_mass * total_mass, 1e-12)
        )
        conflict_balance = float(np.clip(conflict_balance, 0.0, 1.0))
        opposing_mass = 2.0 * min(positive_mass, negative_mass)
        interference_load = opposing_mass / (1.0 + total_mass)
    else:
        conflict_balance = 0.0
        interference_load = 0.0

    # Only an already accessible, genuinely near-balanced state receives an
    # extra bounded normalization. The sixth power leaves asymmetric states
    # essentially unchanged and introduces no sequence or stimulus class.
    near_balance_multiplier = (
        1.0 + near_balance_normalization * conflict_balance ** 6
    )
    conflict_divisor = 1.0 + (
        conflict_normalization
        * interference_load
        * np.sqrt(max(1.0, float(m - 1)))
        * near_balance_multiplier
    )

    # Load mainly increases readout uncertainty under conflict. A small
    # residual component remains for coherent displays, avoiding perfect
    # performance while not producing excessive unanimous-evidence fatigue.
    load_component = np.log1p(max(0.0, float(m - 1)))
    load_multiplier = 0.12 + 0.88 * conflict_balance
    readout_sd = np.sqrt(
        1.0
        + omission_variance
        + load_noise * load_component * load_multiplier
    )

    # A weak, continuous vigilance decline is the only load penalty that also
    # applies fully to unanimous evidence.
    vigilance = np.exp(-vigilance_decline * max(0.0, float(m - 2)))

    decision_variable = (
        decision_gain
        * vigilance
        * integrated_evidence
        / max(conflict_divisor * readout_sd, 1e-12)
    )
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a = 1.0 / (1.0 + np.exp(-decision_variable))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])

    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Weak- versus strong-terminal increase in choosing the three-cue coalition.\"\"\"\n    if data is None or len(data) == 0:\n        return float(\"nan\")\n\n    weak_choices = []\n    strong_choices = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        diff = a - b\n        discriminating = np.flatnonzero(diff != 0)\n        if discriminating.size != 4:\n            continue\n\n        # In this design, the first three discriminating experts constitute\n        # the repeated coalition and the last one is the opposing terminal cue.\n        terminal_idx = int(discriminating[-1])\n        coalition_supports_a = bool(diff[discriminating[0]] > 0)\n        response = int(row[\"response\"])\n        chose_coalition = float(\n            (response == 0 and coalition_supports_a)\n            or (response == 1 and not coalition_supports_a)\n        )\n\n        # Strong terminals: validities .86 and .70 (indices 3 and 5).\n        # Weak terminals: validities .505 and .50 (indices 10 and 11).\n        if terminal_idx in (3, 5):\n            strong_choices.append(chose_coalition)\n        elif terminal_idx in (10, 11):\n            weak_choices.append(chose_coalition)\n\n    if len(weak_choices) == 0 or len(strong_choices) == 0:\n        return float(\"nan\")\n\n    return float(np.mean(weak_choices) - np.mean(strong_choices))",
  "rationale": "The metric measures how much replacing a highly reliable terminal B reason with an almost uninformative terminal reason increases choice of the opposing three-cue coalition. Coalition coding combines the A/B mirror trials and therefore suppresses stable response-side bias. Pooling the two strongest and two weakest terminal conditions supplies 32 observations per side for each subject, reducing Bernoulli noise. The recurrent evidence-field theory predicts a positive contrast because terminal evidence strength is validity-sensitive, whereas overwrite predicts approximately zero because all included trials have the same A,A,A,B direction sequence, four discriminating cues, and identical fatigue. Omitting the intermediate conditions emphasizes the region with the largest predicted reliability effect rather than estimating a noisy six-level slope."
}
```

## Usage

```json
{
  "input_tokens": 5882,
  "output_tokens": 2115
}
```
