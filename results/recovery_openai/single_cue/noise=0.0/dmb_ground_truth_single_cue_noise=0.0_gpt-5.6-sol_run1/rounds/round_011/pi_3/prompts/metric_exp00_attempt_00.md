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
**Validities (n_features=24):** [0.99, 0.88, 0.7, 0.58, 0.54, 0.96, 0.83, 0.65, 0.61, 0.78, 0.51, 0.92, 0.99, 0.88, 0.7, 0.58, 0.54, 0.96, 0.83, 0.65, 0.61, 0.78, 0.51, 0.92]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This is a reliability-role crossover with the physical direction sequence held fixed. The 24 displayed experts contain four interleaved early banks and four interleaved late banks. On every base trial, one early bank favors A and one late bank favors B, so Spatially Indexed Reliability-Gated Revision always encounters the same left-to-right sequence A,A,A,B,B,B. Across the four conditions, however, reliability is transferred progressively from the early A bank to the late B bank: very-high versus low, moderately-high versus moderately-low, moderately-low versus moderately-high, and low versus very-high. The spatial theory should therefore show stable-to-increasing choice of terminal option B across these conditions. Its reliability-sensitive incumbent trace may protect A when the early bank is strongest, whereas increasingly reliable late challengers should revise that incumbent more effectively; positional means and the number, tally, run structure, and physical direction order of discriminating cues are approximately matched. Recency-Gated Reason Overwrite instead sorts by validity. In conditions 1 and 2 it processes A,A,A,B,B,B and predicts B, but in conditions 3 and 4 it processes B,B,B,A,A,A and predicts A. It thus predicts a sharp decrease in B choice between the middle conditions, opposite the spatial theory's increasing terminal-B profile. Within each side of that crossover, overwrite predicts nearly equal choice rates because validity magnitudes are ignored after sorting, and all trials have six discriminating cues and the same fatigue factor. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Spatially Indexed Reliability-Gated Revision. People inspect product features in their displayed left-to-right order rather than rearranging them by instructed validity. Each discriminating cue revises a bounded directional commitment. Revision is leaky, so recent cues ordinarily have greater influence, while repeated same-direction cues build a saturating, reliability-sensitive stability trace rather than an all-or-none coalition. Congruent trace consolidation is selectively nonlinear in relative reliability: repeated reliable cues stabilize an incumbent more strongly than equally numerous weak cues, while ordinary cue revision remains unchanged. This trace remains associated with an established incumbent during a reversal and is eroded gradually by cumulative challenger evidence; consequently, every cue in an opposing run remains partially gated until the challenger has continuously displaced the incumbent. Instructed validity jointly affects cue accessibility, revision magnitude, and the rate at which evidence builds or erodes stability. A weak cue that locally opposes a more reliable preceding cue evokes surprise-driven orienting, temporarily increasing its accessibility and capacity to revise the incumbent commitment. Accessibility also rises gradually toward the right side of the display, although the positional gradient is compressed on very long displays, with a small additional benefit for an isolated discriminating cue at the physical right edge. Increasing display load gradually reduces response precision, including a small decline for long unanimous sequences. Choices are generated from this single evolving commitment state, with no trial-level mixture of strategies.

**Parameters:**
- validities: validities
- spatial_attention: [0.55, 1.15]
- validity_gain: [0.12, 0.38]
- leak_rate: [0.48, 0.67]
- overwrite_strength: [1.75, 2.45]
- surprise_orienting: [1.25, 2.30]
- commitment_saturation: [0.48, 0.88]
- load_noise: [0.020, 0.055]
- lapse_rate: [0.04, 0.13]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Spatially Indexed Reliability-Gated Revision expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    spatial_attention = float(parameters["spatial_attention"])
    validity_gain = float(parameters["validity_gain"])
    leak_rate = float(parameters["leak_rate"])
    overwrite_strength = float(parameters["overwrite_strength"])
    surprise_orienting = float(parameters["surprise_orienting"])
    commitment_saturation = float(parameters["commitment_saturation"])
    load_noise = float(parameters["load_noise"])
    lapse_rate = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Communicated validity is represented as reliability, but it does not
    # determine scan order. Standardization makes the same gain parameter
    # usable across experiments with different validity ranges.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    rel_mean = float(np.mean(reliability))
    rel_sd = float(np.std(reliability))
    if rel_sd < 1e-8:
        standardized_reliability = np.zeros(n_features, dtype=float)
    else:
        standardized_reliability = np.clip(
            (reliability - rel_mean) / rel_sd, -2.5, 2.5
        )

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # commitment is the bounded current decision state. stability_trace is a
    # continuous signed record of commitment-congruent support. Unlike a run
    # label, it is not reassigned after the first opposing update: cumulative
    # challenger evidence must gradually erode and reverse it.
    commitment = 0.0
    stability_trace = 0.0
    previous_direction = 0.0
    previous_reliability = None
    previous_index = None

    # Compress only the positional component of accessibility on unusually
    # long displays. This prevents normalized scan position from creating an
    # unrealistically large endpoint effect while retaining a rightward bias.
    position_scale = float(min(
        1.0, np.sqrt(8.0 / float(max(8, n_features)))
    ))

    # On unusually long displays, intervening locations increase retention
    # slightly. Ordinary displays retain the original recurrence unchanged.
    long_display_fraction = float(np.clip(
        (float(n_features) - 16.0) / 4.0, 0.0, 1.0
    ))
    effective_leak_rate = float(np.clip(
        leak_rate + 0.10 * long_display_fraction, 0.0, 0.90
    ))

    for j in discriminating:
        j = int(j)
        direction = float(np.sign(diff[j]))
        position = float(j) / float(max(1, n_features - 1))

        # Ordinary accessibility follows a smooth left-to-right gradient.
        # Validity changes accessibility without reordering the cues.
        access_logit = (
            1.15
            + spatial_attention * position_scale * (2.0 * position - 1.0)
            + validity_gain * standardized_reliability[j]
        )

        # A physically isolated discriminating item at the right boundary is
        # modestly foregrounded. The conjunction prevents a generic terminal
        # bonus from guaranteeing reversal on every display.
        if previous_index is None:
            gap_fraction = 0.0
        else:
            gap_fraction = float(max(0, j - previous_index - 1)) / float(
                max(1, n_features - 1)
            )
        right_edge = float(j == n_features - 1)
        isolation_bonus = (
            0.32
            * spatial_attention
            * position_scale
            * right_edge
            * np.sqrt(gap_fraction)
        )
        access_logit += isolation_bonus
        accessibility = sigmoid(access_logit)

        # Reliability also scales encoded revision strength, but only
        # compressively, so weak experts remain psychologically available.
        reliability_strength = float(np.exp(
            validity_gain * 0.55 * standardized_reliability[j]
        ))
        reliability_strength = float(np.clip(reliability_strength, 0.45, 2.20))

        # Surprise is specifically a local reliability drop at an opposing
        # boundary. It boosts attention to a weak challenger rather than
        # installing a separate strategy or reversing the validity ordering.
        orienting = 1.0
        if previous_reliability is not None and previous_direction != direction:
            raw_drop = max(0.0, float(previous_reliability - validities[j]))
            incumbent_alignment = abs(commitment)
            orienting += (
                surprise_orienting
                * raw_drop
                * (0.45 + 0.55 * incumbent_alignment)
            )

        # Leakage creates recency. Every cue remains gated while its direction
        # opposes the persistent stability trace. The gate depends on the trace
        # itself, not instantaneous commitment, so it does not disappear merely
        # because an initial challenger has moved commitment toward zero.
        retained = float(np.clip(
            effective_leak_rate * commitment, -0.999999, 0.999999
        ))
        opposing_stability = bool(
            stability_trace != 0.0 and np.sign(stability_trace) != direction
        )
        if opposing_stability:
            trace_gate_mass = min(2.5, abs(stability_trace) ** 0.75)
            stability_gate = 1.0 + (
                commitment_saturation * trace_gate_mass
            )
        else:
            stability_gate = 1.0

        revision = (
            overwrite_strength
            * accessibility
            * reliability_strength
            * orienting
            / stability_gate
        )

        # Updating in bounded-logit coordinates gives gradual saturation for
        # repeated agreement while preserving smooth directional revision.
        bounded_coordinate = np.arctanh(retained)
        commitment = float(np.tanh(
            bounded_coordinate + direction * revision
        ))

        # The signed stability trace persists across a reversal. Congruent
        # evidence consolidates it efficiently, whereas opposing evidence
        # erodes it continuously and reliability-dependently before eventually
        # installing a new incumbent. No switch threshold or run state is used.
        trace_alignment = direction * float(np.tanh(stability_trace))
        assimilation = (
            0.50 + 0.50 * sigmoid(2.5 * trace_alignment)
        )

        # Only commitment-congruent consolidation receives this bounded
        # nonlinear reliability modulation. Reliable repeated cues strengthen
        # an established trace, whereas weak repetition consolidates less.
        # Opposing cues retain the accepted linear erosion rule.
        congruent_stability = bool(
            stability_trace != 0.0 and np.sign(stability_trace) == direction
        )
        if congruent_stability:
            consolidation_gain = (
                0.78
                + 0.44 * sigmoid(2.2 * standardized_reliability[j])
            )
        else:
            consolidation_gain = 1.0

        trace_retention = float(np.sqrt(effective_leak_rate))
        stability_trace = (
            trace_retention * stability_trace
            + direction
            * accessibility
            * reliability_strength
            * assimilation
            * consolidation_gain
        )
        stability_trace = float(np.clip(stability_trace, -6.0, 6.0))

        previous_direction = direction
        previous_reliability = float(validities[j])
        previous_index = j

    m = int(discriminating.size)

    # More reasons and more displayed locations introduce gradual precision
    # loss. There is no fixed fatigue boundary and no conflict-only collapse.
    display_load = float(max(0, n_features - 1))
    evidence_load = float(max(0, m - 2))
    precision = 3.35 / (
        1.0
        + load_noise * display_load
        + 0.55 * load_noise * evidence_load
    )
    precision = float(np.clip(precision, 0.25, 6.0))

    p_a_core = sigmoid(precision * commitment)
    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
    probs = (
        (1.0 - lapse_rate) * core
        + lapse_rate * np.array([0.5, 0.5], dtype=np.float64)
    )
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Score whether each response follows the physically last discriminating\n    # cue, then contrast crossover conditions 1-2 against conditions 3-4.\n    condition_scores = {0: [], 1: [], 2: [], 3: []}\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        diff = a - b\n        discriminating = np.flatnonzero(diff != 0)\n        if discriminating.size == 0:\n            continue\n\n        # In this design, the leftmost discriminating location (0, 1, 2, or 3)\n        # uniquely identifies the reliability-crossover condition, including\n        # the response-side-reversed versions of each pair.\n        condition = int(discriminating[0])\n        if condition not in condition_scores:\n            continue\n\n        last_direction = diff[int(discriminating[-1])]\n        terminal_response = 0 if last_direction > 0 else 1\n        follows_terminal = 1.0 if int(row[\"response\"]) == terminal_response else -1.0\n        condition_scores[condition].append(follows_terminal)\n\n    means = {}\n    for condition, values in condition_scores.items():\n        if len(values) == 0:\n            return float(\"nan\")\n        means[condition] = float(np.mean(values))\n\n    early_reliability_regime = 0.5 * (means[0] + means[1])\n    late_reliability_regime = 0.5 * (means[2] + means[3])\n    return float(early_reliability_regime - late_reliability_regime)\n",
  "rationale": "This is a within-subject difference-in-differences for control by the physically terminal cue. Recency-Gated Reason Overwrite sorts cues by validity: its final sorted reason supports the physical terminal option in conditions 1-2 but the opposite option in conditions 3-4, predicting a strongly positive contrast. Spatially Indexed Reliability-Gated Revision retains physical left-to-right processing and is designed to show stable or increasing terminal-option control as reliability moves to the late run, predicting a near-zero or negative contrast. Coding choices relative to the terminal cue makes exact A/B reversals equivalent and removes response-side bias. Averaging the 24 repetitions in each condition before taking the within-subject contrast should also keep between-subject sampling variance low."
}
```

## Usage

```json
{
  "input_tokens": 6912,
  "output_tokens": 1563
}
```
