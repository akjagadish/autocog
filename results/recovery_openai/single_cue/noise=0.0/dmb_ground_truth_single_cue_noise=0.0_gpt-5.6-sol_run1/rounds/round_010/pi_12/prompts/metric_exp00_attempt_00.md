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
**Validities (n_features=16):** [0.86, 0.98, 0.98, 0.86, 0.53, 0.86, 0.86, 0.53, 0.98, 0.53, 0.53, 0.98, 0.52, 0.52, 0.52, 0.52]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

**Rationale:** This is a boundary-surprise test with scan-order, coalition size, reliability multiset, and aggregate physical accessibility controlled. On every base trial, exactly four experts discriminate. In descending-validity order their directions are always A,A,A,B, with respective validity magnitudes 98%, 86%, 53%, and 52%. Recency-Gated Reason Overwrite therefore predicts exactly the same terminal-B choice probability in all four base conditions: cue order, direction sequence, attention and overwrite process, discriminating count, fatigue, and lapse are identical. Surprise-Bound Coalition Relay instead inspects the displayed order, which is also A,A,A,B, but distinguishes the reliability discontinuity at the sole coalition boundary. In the two large-drop conditions, the A coalition exits at 98% and the terminal B singleton enters at 52%, producing boundary_surprise about 0.42*0.46/(0.50+0.46)=0.201. In the two small-drop conditions, the coalition exits at 53% before the same 52% singleton, producing only about 0.008. This approximately 0.193 contest difference corresponds to a 0.73–1.20 shift in the switching logit over the allowed switch-precision range, so the relay predicts appreciably more choices of the terminal option after the large reliability drop. Other reliability effects are controlled because every condition uses the same 98/86/53/52 multiset and assigns the first three values to one coalition. The four interleaved expert banks also balance display position: large-drop and small-drop conditions have the same mean terminal position and the same aggregate mean position for the three prefix cues. Exact A/B reversals balance response side. Thus the critical statistic is terminal-option choice averaged over the two large-drop banks minus that averaged over the two small-drop banks: the relay predicts a positive contrast, whereas overwrite predicts exactly zero. Eight unique pairs repeated 12 times yield 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Surprise-Bound Coalition Relay. During left-to-right inspection, discriminating cues are compressed into directional coalitions: consecutive cues favoring the same option strengthen one bounded commitment rather than leaving independently summable traces. An opposing coalition initiates a graded contest with the incumbent. Switching is probabilistic and depends jointly on incumbent strength, challenger coherence, relative instructed reliability, terminal singleton salience, and the challenger's absolute display accessibility. Instructed validity therefore modulates encoding and switching thresholds but is not converted into signed additive evidence. Absolute display position is represented separately from ordinal position in the discriminating sequence. The final response averages over unresolved commitment trajectories and is confidence-gated: balanced evidence, ambiguous switches, and fragmented sequences approach chance, whereas a decisive coalition switch can be expressed confidently. Load impairs precision primarily when conflict remains unresolved; unanimous evidence receives only a small load-dependent decline. At a coalition boundary, a modest, regularized drop from the incumbent's exit reliability to the challenger's entry reliability contributes relative surprise, preserving local boundary information without allowing validity differences to dominate the contest.

**Parameters:**
- validities: validities
- base_attention: [0.82, 0.95]
- validity_use: [0.22, 0.52]
- position_accessibility: [0.28, 0.58]
- coalition_capacity: [1.75, 2.45]
- coherence_gain: [0.10, 0.30]
- switch_threshold: [0.62, 0.88]
- switch_precision: [3.8, 6.2]
- relative_surprise: [0.45, 0.90]
- terminal_salience: [0.42, 0.78]
- opposition_erosion: [0.20, 0.42]
- response_gain: [2.5, 4.4]
- conflict_collapse: [0.48, 0.76]
- load_noise: [0.025, 0.085]
- unanimous_decay: [0.010, 0.032]
- lapse_rate: [0.04, 0.14]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Surprise-Bound Coalition Relay expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    base_attention = float(parameters["base_attention"])
    validity_use = float(parameters["validity_use"])
    position_accessibility = float(parameters["position_accessibility"])
    coalition_capacity = float(parameters["coalition_capacity"])
    coherence_gain = float(parameters["coherence_gain"])
    switch_threshold = float(parameters["switch_threshold"])
    switch_precision = float(parameters["switch_precision"])
    relative_surprise = float(parameters["relative_surprise"])
    terminal_salience = float(parameters["terminal_salience"])
    opposition_erosion = float(parameters["opposition_erosion"])
    response_gain = float(parameters["response_gain"])
    conflict_collapse = float(parameters["conflict_collapse"])
    load_noise = float(parameters["load_noise"])
    unanimous_decay = float(parameters["unanimous_decay"])
    lapse = float(parameters["lapse_rate"])

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    m = int(discriminating.size)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Reliability affects attentional accessibility and switch thresholds,
    # never the sign of an additive evidence trace.
    reliability = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    base_logit = np.log(
        np.clip(base_attention, 1e-5, 1.0 - 1e-5)
        / np.clip(1.0 - base_attention, 1e-5, 1.0)
    )

    ordinal_directions = np.sign(diff[discriminating]).astype(float)
    cue_access = []
    for j in discriminating:
        absolute_position = float(j) / float(max(1, n_features - 1))
        attention = sigmoid(
            base_logit
            + validity_use * (float(reliability[j]) - 0.5)
            + position_accessibility * (absolute_position - 0.5)
        )
        # A modest independent accessibility advantage for later display
        # locations is retained after attentional gating.
        access = attention * (
            1.0 + position_accessibility * 0.45 * absolute_position
        )
        cue_access.append(float(access))
    cue_access = np.asarray(cue_access, dtype=float)

    # Form ordinal coalitions. Physical locations remain attached to each
    # coalition, so discriminating ordinal position and display position are
    # not conflated.
    runs = []
    start = 0
    while start < m:
        direction = float(ordinal_directions[start])
        end = start + 1
        while end < m and ordinal_directions[end] == direction:
            end += 1

        loc = slice(start, end)
        run_indices = discriminating[loc]
        raw_access = float(np.sum(cue_access[loc]))
        run_length = int(end - start)

        # Bounded within-coalition integration. Coherence increases strength
        # sublinearly and cannot create an unbounded linear sum.
        bounded = coalition_capacity * (
            1.0 - np.exp(-raw_access / max(coalition_capacity, 1e-8))
        )
        coherence = 1.0 + coherence_gain * (
            1.0 - 1.0 / float(run_length)
        )
        strength = min(coalition_capacity, bounded * coherence)

        weights = cue_access[loc]
        if float(np.sum(weights)) > 0.0:
            mean_reliability = float(
                np.average(reliability[run_indices], weights=weights)
            )
        else:
            mean_reliability = float(np.mean(reliability[run_indices]))

        final_physical_position = float(run_indices[-1]) / float(
            max(1, n_features - 1)
        )
        runs.append({
            "direction": direction,
            "strength": float(strength),
            "length": run_length,
            "reliability": mean_reliability,
            "entry_reliability": float(reliability[run_indices[0]]),
            "exit_reliability": float(reliability[run_indices[-1]]),
            "final_position": final_physical_position,
            "ends_sequence": bool(end == m)
        })
        start = end

    # Each entry is a possible commitment trajectory. Branching implements a
    # genuinely graded switch rather than deterministic overwrite.
    states = [{
        "weight": 1.0,
        "direction": runs[0]["direction"],
        "strength": runs[0]["strength"],
        "reliability": runs[0]["reliability"],
        "exit_reliability": runs[0]["exit_reliability"]
    }]
    switch_ambiguities = []

    for run in runs[1:]:
        next_states = []
        weighted_ambiguity = 0.0
        total_state_weight = 0.0

        for incumbent in states:
            w = float(incumbent["weight"])
            incumbent_strength = float(incumbent["strength"])
            challenger_strength = float(run["strength"])

            reliability_difference = (
                float(incumbent["reliability"])
                - float(run["reliability"])
            )
            threshold_multiplier = 1.0 + validity_use * reliability_difference
            threshold_multiplier = float(np.clip(
                threshold_multiplier, 0.55, 1.55
            ))

            relative_advantage = (
                challenger_strength - incumbent_strength
            ) / max(challenger_strength + incumbent_strength, 1e-8)

            is_terminal_singleton = float(
                run["ends_sequence"] and run["length"] == 1
            )
            singleton_access = (
                is_terminal_singleton
                * terminal_salience
                * (0.55 + 0.45 * float(run["final_position"]))
            )

            # Switching is initiated at the actual coalition boundary. A
            # downward reliability discontinuity can be surprising, but its
            # influence is bounded and strongly regularized. The coefficient
            # is deliberately intermediate rather than the rejected 0.75
            # boundary weight.
            boundary_drop = max(
                0.0,
                float(incumbent["exit_reliability"])
                - float(run["entry_reliability"])
            )
            boundary_surprise = (
                0.42 * boundary_drop / (0.50 + boundary_drop)
            )

            contest = (
                challenger_strength
                - switch_threshold
                * threshold_multiplier
                * incumbent_strength
                + relative_surprise * relative_advantage
                + singleton_access
                + boundary_surprise
            )
            q_switch = sigmoid(switch_precision * contest)

            weighted_ambiguity += w * 4.0 * q_switch * (1.0 - q_switch)
            total_state_weight += w

            # If the challenger is resisted, conflict still erodes incumbent
            # commitment. If it wins, its bounded coalition becomes incumbent.
            retained_strength = max(
                0.08,
                incumbent_strength
                - opposition_erosion * challenger_strength
            )
            switched_strength = min(
                coalition_capacity,
                challenger_strength
                + 0.10 * q_switch * incumbent_strength
            )

            if q_switch < 1.0:
                next_states.append({
                    "weight": w * (1.0 - q_switch),
                    "direction": float(incumbent["direction"]),
                    "strength": float(retained_strength),
                    "reliability": float(incumbent["reliability"]),
                    "exit_reliability": float(incumbent["exit_reliability"])
                })
            if q_switch > 0.0:
                next_states.append({
                    "weight": w * q_switch,
                    "direction": float(run["direction"]),
                    "strength": float(switched_strength),
                    "reliability": float(run["reliability"]),
                    "exit_reliability": float(run["exit_reliability"])
                })

        states = next_states
        if total_state_weight > 0.0:
            switch_ambiguities.append(
                weighted_ambiguity / total_state_weight
            )

    # Accessible support is used only to diagnose unresolved conflict. It is
    # not read out as a linear evidence total.
    positive_mass = float(np.sum(
        cue_access[ordinal_directions > 0.0]
    ))
    negative_mass = float(np.sum(
        cue_access[ordinal_directions < 0.0]
    ))
    total_mass = positive_mass + negative_mass
    if positive_mass <= 0.0 or negative_mass <= 0.0 or total_mass <= 0.0:
        balance = 0.0
    else:
        balance = 2.0 * min(positive_mass, negative_mass) / total_mass
        balance = float(np.clip(balance, 0.0, 1.0))

    ambiguity = (
        float(np.mean(switch_ambiguities))
        if switch_ambiguities else 0.0
    )
    fragmentation = max(0.0, float(len(runs) - 2)) / float(max(1, m - 1))
    unresolved = balance * (
        0.20 + 0.55 * ambiguity + 0.45 * fragmentation
    )
    unresolved = float(np.clip(unresolved, 0.0, 1.0))

    confidence = max(0.08, 1.0 - conflict_collapse * unresolved)
    overload = max(0.0, float(m) - 4.0)
    conflict_divisor = 1.0 + load_noise * overload * unresolved * unresolved

    # Unanimous evidence avoids conflict noise but shows a slight empirical
    # decline at high loads rather than ever-increasing certainty.
    if balance == 0.0:
        unanimous_divisor = 1.0 + unanimous_decay * max(0.0, float(m) - 2.0)
    else:
        unanimous_divisor = 1.0

    trajectory_weight = float(np.sum([s["weight"] for s in states]))
    if trajectory_weight <= 0.0 or not np.isfinite(trajectory_weight):
        return np.array([0.5, 0.5], dtype=np.float64)

    p_a_core = 0.0
    for commitment in states:
        decision_variable = (
            response_gain
            * confidence
            * float(commitment["direction"])
            * float(commitment["strength"])
            / (conflict_divisor * unanimous_divisor)
        )
        p_a_state = sigmoid(decision_variable)
        p_a_core += float(commitment["weight"]) * p_a_state
    p_a_core /= trajectory_weight
    p_a_core = float(np.clip(p_a_core, 0.0, 1.0))

    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    validities = np.array([\n        0.86, 0.98, 0.98, 0.86,\n        0.53, 0.86, 0.86, 0.53,\n        0.98, 0.53, 0.53, 0.98,\n        0.52, 0.52, 0.52, 0.52\n    ], dtype=float)\n\n    large_drop_choices = []\n    small_drop_choices = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        diff = a - b\n        discriminating = np.flatnonzero(diff != 0.0)\n        if discriminating.size < 2:\n            continue\n\n        # Coalitions are defined in displayed (left-to-right) order. In this\n        # design the final discriminating cue is the terminal singleton and\n        # the preceding discriminating cue is the incumbent coalition's exit.\n        terminal_index = int(discriminating[-1])\n        exit_index = int(discriminating[-2])\n        boundary_drop = float(validities[exit_index] - validities[terminal_index])\n\n        response = int(row[\"response\"])\n        terminal_favors_a = bool(diff[terminal_index] > 0.0)\n        chose_terminal = float(\n            (terminal_favors_a and response == 0)\n            or ((not terminal_favors_a) and response == 1)\n        )\n\n        if boundary_drop > 0.20:\n            large_drop_choices.append(chose_terminal)\n        else:\n            small_drop_choices.append(chose_terminal)\n\n    if len(large_drop_choices) == 0 or len(small_drop_choices) == 0:\n        return 0.0\n\n    return float(\n        np.mean(large_drop_choices) - np.mean(small_drop_choices)\n    )",
  "rationale": "This is the within-subject large-versus-small boundary-drop contrast in choosing the terminal singleton. It pools the exact A/B reversals by coding whether the terminal cue's favored option was selected, eliminating response-side preference. Every subject contributes 48 observations to each side of the contrast, keeping sampling variance low. The overwrite theory predicts a near-zero contrast because all conditions have the same descending-validity direction sequence, cue count, fatigue, and lapse process. The coalition-relay theory predicts a positive contrast because switching to the terminal singleton is selectively increased when the incumbent coalition exits at .98 rather than .53 before the same .52 challenger. The statistic measures an observable interaction rather than directly encoding a model parameter or model-generated probability."
}
```

## Usage

```json
{
  "input_tokens": 7145,
  "output_tokens": 2005
}
```
