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
**Validities (n_features=8):** [0.99, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.51]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  trial 5: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]

**Rationale:** This is a history-dependent context-calibration test. Experts 1 and 8 provide the advertised validity spread but are nondiscriminating; the six diagnostic experts have similar strengths and are scanned in order. Two inducer families create sharply different learned task sets. The coherent AAAAAA/B​BBBBB inducer family contains no opposition or reversal and should lower the advocated accumulator's overwrite calibration. The ABBBBB/BAAAAA family has a concentrated two-run boundary followed by a long, highly reversal-susceptible opposing run and should raise overwrite calibration. The critical probes are AAAB and AAAAAB, plus their exact option reversals. Each probe has an early majority favoring one option but a terminal reason favoring the other. After histories dominated by coherent inducers, the advocated theory predicts greater integration and coalition access, hence more choices of the early-majority option. After histories rich in reversal-susceptible inducers, it predicts lower retention, stronger conflict inhibition, and substantially more choices of the terminal option; sufficiently strong calibration should produce a within-probe modal crossover. This effect should follow the advocated model's exponentially weighted pretrial history statistic, not merely the immediately preceding response. Recency-Gated Reason Overwrite is completely history-independent and therefore predicts no inducer-history effect for either probe. It predicts a stable terminal-option preference determined only by the current sequence, with the six-reason probe attenuated more toward chance than the four-reason probe because of its larger fatigue factor. The primary analysis should regress terminal-option choice on the advocated theory's pretrial context score, computed from stimulus history while excluding the current trial, with separate probe intercepts; the advocated theory predicts a positive context slope, whereas overwrite predicts exactly zero. Eight unique pairs are each repeated 12 times, producing 96 trials, and exact A/B reversals balance response side.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Distribution-Calibrated Competitive Accumulation with Susceptibility-Dominant Context Learning. People inspect discriminating cues in descending instructed-validity order and encode each attended cue as signed evidence in a continuously evolving competitive accumulator. Opposing evidence inhibits the active commitment, while a continuous experiment-level task set controls retention and overwrite. The task set is learned primarily from how readily opposing runs reverse a reference accumulator across the experiment's sequence families. Coarse ordinal properties such as terminal singletons, two-run structure, and reversal-location concentration provide only weak, centered calibration rather than directly producing overwrite. Reliability-weighted coalitions can reverse the sequential state, but their access to choice is proportional to normalized evidence margin. Physical gaps are not represented, and reliability depends on conflict geometry and evidence margin rather than discriminating-cue count.

**Parameters:**
- validities: validities
- attention_rate: [0.88, 1.0]
- integration_retention: [0.84, 0.97]
- overwrite_retention: [0.14, 0.36]
- conflict_inhibition: [0.35, 1.20]
- coalition_weight: [0.30, 0.62]
- validity_sensitivity: [0.15, 0.45]
- context_sensitivity: [1.4, 2.4]
- context_threshold: [1.0, 1.6]
- context_learning_rate: [0.05, 0.12]
- susceptibility_coupling: [0.60, 1.10]
- response_gain: [1.8, 3.8]
- lapse_rate: [0.04, 0.15]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive accumulation expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["attention_rate"])
    integration_retention = float(parameters["integration_retention"])
    overwrite_retention = float(parameters["overwrite_retention"])
    conflict_inhibition = float(parameters["conflict_inhibition"])
    coalition_weight = float(parameters["coalition_weight"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    context_sensitivity = float(parameters["context_sensitivity"])
    context_threshold = float(parameters["context_threshold"])
    learning_rate = float(parameters["context_learning_rate"])
    susceptibility_coupling = float(parameters["susceptibility_coupling"])
    response_gain = float(parameters["response_gain"])
    lapse = float(parameters["lapse_rate"])

    cue_order = np.argsort(-validities, kind="stable")

    def ordinal_summary(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
            return None
        d = a - b
        ordered_idx = [int(j) for j in cue_order if j < d.size and d[j] != 0]
        if not ordered_idx:
            return None
        directions = np.asarray([np.sign(d[j]) for j in ordered_idx], dtype=float)
        m = directions.size
        switches = np.flatnonzero(directions[1:] != directions[:-1]) + 1

        run_lengths = []
        run = 1
        for k in range(1, m):
            if directions[k] == directions[k - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

        if switches.size:
            reversal_fractions = switches.astype(float) / float(m)
        else:
            reversal_fractions = np.asarray([], dtype=float)

        # Estimate reversal susceptibility with the same signed evidence units
        # used by the decision process. At every run boundary, compare the
        # incoming opposing run with the reference-retained incumbent. The
        # smooth score records both actual reversals and near reversals.
        local_v = np.clip(validities, 0.5001, 0.999)
        local_reliability = np.log(local_v / (1.0 - local_v))
        local_scale = float(np.mean(local_reliability[local_reliability > 0.0]))
        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = 1.0
        local_normalized = np.clip(local_reliability / local_scale, 0.15, 6.0)
        local_strength = np.power(local_normalized, validity_sensitivity)
        ordered_strength = np.asarray(
            [attention * float(local_strength[j]) for j in ordered_idx],
            dtype=float
        )

        reference_retention = 0.55
        reference_accumulator = 0.0
        susceptibility_scores = []
        k = 0
        while k < m:
            run_direction = float(directions[k])
            run_end = k + 1
            while run_end < m and directions[run_end] == run_direction:
                run_end += 1

            run_evidence = ordered_strength[k:run_end]
            run_length = int(run_end - k)
            if reference_accumulator != 0.0 and np.sign(reference_accumulator) != run_direction:
                retained_incumbent = abs(reference_accumulator) * (
                    reference_retention ** run_length
                )
                attack = 0.0
                for strength in run_evidence:
                    attack = reference_retention * attack + float(strength)
                normalized_advantage = (
                    attack - retained_incumbent
                ) / max(attack + retained_incumbent, 1e-9)
                smooth_reversal = 1.0 / (
                    1.0 + np.exp(-4.0 * np.clip(normalized_advantage, -15.0, 15.0))
                )
                susceptibility_scores.append(float(smooth_reversal))

            for strength in run_evidence:
                reference_accumulator = (
                    reference_retention * reference_accumulator
                    + run_direction * float(strength)
                )
            k = run_end

        if susceptibility_scores:
            reversal_susceptibility = float(np.mean(susceptibility_scores))
        else:
            reversal_susceptibility = 0.35

        return {
            "m": float(m),
            "directions": directions,
            "reversal_fractions": reversal_fractions,
            "terminal_singleton": float(run_lengths[-1] == 1 and len(run_lengths) > 1),
            "two_run": float(len(run_lengths) == 2),
            "switch_density": float(switches.size) / float(max(1, m - 1)),
            "reversal_susceptibility": reversal_susceptibility
        }

    # Learn the task set from ordinal scan sequences only. Physical indices,
    # distances between cues, and nondiscriminating gaps never enter these
    # statistics. Exponential weighting implements continuous contextual
    # learning while retaining a weak neutral prior early in the block.
    summaries = []
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        s = ordinal_summary(past_a[i], past_b[i])
        if s is not None:
            summaries.append(s)

    current_summary = ordinal_summary(stim[0], stim[1])
    if current_summary is None:
        return np.array([0.5, 0.5], dtype=np.float64)
    summaries.append(current_summary)

    ages = np.arange(len(summaries) - 1, -1, -1, dtype=float)
    weights = np.power(np.clip(1.0 - learning_rate, 1e-6, 1.0), ages)
    # The current stimulus can update calibration, but once context exists it
    # receives a bounded weight and cannot sharply change the experiment-level
    # task set solely because of its own conflict geometry.
    if len(summaries) > 1:
        weights[-1] *= 0.35
    weights /= max(float(weights.sum()), 1e-12)

    terminal_singleton_rate = float(np.sum([
        w * s["terminal_singleton"] for w, s in zip(weights, summaries)
    ]))
    two_run_rate = float(np.sum([
        w * s["two_run"] for w, s in zip(weights, summaries)
    ]))
    switch_density = float(np.sum([
        w * s["switch_density"] for w, s in zip(weights, summaries)
    ]))
    reversal_susceptibility = float(np.sum([
        w * s["reversal_susceptibility"] for w, s in zip(weights, summaries)
    ]))

    # A five-bin distribution represents reversal locations in ordinal scan
    # time, not their physical screen positions. Concentrated sequence
    # families provide only weak contextual calibration; whether opposing
    # runs can actually reverse accumulated evidence is the primary signal.
    reversal_hist = np.zeros(5, dtype=float)
    reversal_mass = 0.0
    for w, s in zip(weights, summaries):
        fractions = s["reversal_fractions"]
        if fractions.size == 0:
            continue
        contribution = w / float(fractions.size)
        for fraction in fractions:
            bin_index = min(4, max(0, int(np.floor(5.0 * float(fraction)))))
            reversal_hist[bin_index] += contribution
            reversal_mass += contribution

    if reversal_mass > 0.0:
        reversal_distribution = reversal_hist / reversal_mass
        concentration = float(np.sum(reversal_distribution ** 2))
        positive = reversal_distribution[reversal_distribution > 0.0]
        reversal_entropy = float(
            -np.sum(positive * np.log(positive)) / np.log(5.0)
        )
    else:
        concentration = 0.2
        reversal_entropy = 1.0

    # Validity-profile steepness can alter task calibration, but only weakly;
    # this allows the same ordinal sequence to generalize differently under a
    # changed reliability profile without turning validity into a hard rule.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    mean_rel = max(float(np.mean(reliability)), 1e-8)
    validity_gradient = float(np.std(reliability) / mean_rel)
    validity_gradient = float(np.clip(validity_gradient, 0.0, 2.0))

    # Coarse geometry terms are shrunken and centered around neutral task
    # values. They can adjust calibration but no longer directly force high
    # overwrite merely because a task contains two-run or singleton displays.
    # The accepted process-derived susceptibility statistic carries the main
    # cross-task signal while the centering preserves the prior operating
    # level of the continuous task set.
    context_score = (
        1.30
        + 1.20 * (terminal_singleton_rate - 0.30)
        + 1.20 * (concentration - 0.30)
        + 0.80 * (two_run_rate - 0.50)
        - 1.50 * (reversal_entropy - 0.70)
        - 0.70 * (switch_density - 0.25)
        - 0.20 * validity_gradient
        + susceptibility_coupling * (reversal_susceptibility - 0.35)
    )
    context_logit = context_sensitivity * (context_score - context_threshold)
    context_logit = float(np.clip(context_logit, -60.0, 60.0))
    overwrite_task_set = 1.0 / (1.0 + np.exp(-context_logit))

    # The latent task set continuously interpolates between integration and
    # overwrite. It never selects a discrete primacy or terminal policy.
    base_retention = (
        (1.0 - overwrite_task_set) * integration_retention
        + overwrite_task_set * overwrite_retention
    )
    base_retention = float(np.clip(base_retention, 0.01, 0.999))

    diff = stim[0] - stim[1]
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    directions = np.asarray([np.sign(diff[j]) for j in discriminating], dtype=float)

    mean_positive_rel = float(np.mean(reliability[reliability > 0.0]))
    if not np.isfinite(mean_positive_rel) or mean_positive_rel <= 0.0:
        mean_positive_rel = 1.0
    normalized_rel = np.clip(reliability / mean_positive_rel, 0.15, 6.0)
    cue_strength = np.power(normalized_rel, validity_sensitivity)

    accumulator = 0.0
    coalition = 0.0
    current_run_strength = 0.0
    previous_direction = 0.0

    for j, direction in zip(discriminating, directions):
        evidence = attention * float(cue_strength[j]) * float(direction)
        coalition += evidence

        if previous_direction == 0.0 or direction == previous_direction:
            current_run_strength += abs(evidence)
        else:
            current_run_strength = abs(evidence)

        retention = base_retention
        if accumulator != 0.0 and np.sign(accumulator) != direction:
            # Opposition inhibits the existing commitment in proportion to
            # the new reason's strength relative to the active commitment.
            relative_attack = abs(evidence) / (abs(accumulator) + abs(evidence) + 1e-9)
            run_attack = current_run_strength / (abs(accumulator) + current_run_strength + 1e-9)
            inhibition = conflict_inhibition * overwrite_task_set
            retention *= np.exp(-inhibition * (0.65 * relative_attack + 0.35 * run_attack))

        accumulator = retention * accumulator + evidence
        previous_direction = float(direction)

    # A reliability-weighted coalition has strong access only when it has a
    # substantial normalized margin. Near balance, sequential opposition can
    # determine the sign; clear early majorities remain protected by global
    # coalition evidence. This is continuous evidence gating, not policy mixing.
    total_input = float(np.sum([
        attention * float(cue_strength[j]) for j in discriminating
    ]))
    coalition_margin = abs(coalition) / max(total_input, 1e-9)
    coalition_margin = float(np.clip(coalition_margin, 0.0, 1.0))
    margin_access = 0.10 + 0.90 * coalition_margin / (coalition_margin + 0.20)
    effective_coalition_weight = (
        coalition_weight * (1.0 - overwrite_task_set) * margin_access
    )
    decision_evidence = (
        (1.0 - effective_coalition_weight) * accumulator
        + effective_coalition_weight * coalition
    )

    # Confidence follows the final evidence margin. There is deliberately no
    # fatigue or count penalty, allowing coherent long reversals to remain
    # reliable while balanced conflict naturally remains uncertain.
    margin = abs(decision_evidence) / max(total_input, 1e-9)
    geometry_gain = 0.75 + 0.50 * np.sqrt(np.clip(margin, 0.0, 1.0))
    decision_variable = response_gain * geometry_gain * decision_evidence
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a = 1.0 / (1.0 + np.exp(-decision_variable))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5], dtype=np.float64)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    validities = np.asarray([0.99, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.51], dtype=float)\n    reliability = np.log(np.clip(validities, 0.5001, 0.999) / (1.0 - np.clip(validities, 0.5001, 0.999)))\n    scale = float(np.mean(reliability[reliability > 0.0]))\n    strengths = np.power(np.clip(reliability / scale, 0.15, 6.0), 0.30)\n\n    def summarize(a, b):\n        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)\n        idx = np.flatnonzero(d != 0.0)\n        if idx.size == 0:\n            return None\n        dirs = np.sign(d[idx])\n        m = int(dirs.size)\n        switches = np.flatnonzero(dirs[1:] != dirs[:-1]) + 1\n\n        run_lengths = []\n        run = 1\n        for k in range(1, m):\n            if dirs[k] == dirs[k - 1]:\n                run += 1\n            else:\n                run_lengths.append(run)\n                run = 1\n        run_lengths.append(run)\n\n        accumulator = 0.0\n        susceptibility = []\n        k = 0\n        while k < m:\n            direction = float(dirs[k])\n            end = k + 1\n            while end < m and dirs[end] == direction:\n                end += 1\n            incoming = strengths[idx[k:end]]\n            run_length = end - k\n            if accumulator != 0.0 and np.sign(accumulator) != direction:\n                incumbent = abs(accumulator) * (0.55 ** run_length)\n                attack = 0.0\n                for s in incoming:\n                    attack = 0.55 * attack + float(s)\n                advantage = (attack - incumbent) / max(attack + incumbent, 1e-12)\n                susceptibility.append(1.0 / (1.0 + np.exp(-4.0 * np.clip(advantage, -15.0, 15.0))))\n            for s in incoming:\n                accumulator = 0.55 * accumulator + direction * float(s)\n            k = end\n\n        reversal_susceptibility = float(np.mean(susceptibility)) if susceptibility else 0.35\n        fractions = switches.astype(float) / float(m) if switches.size else np.asarray([], dtype=float)\n        canonical = tuple(int(x) for x in (dirs * dirs[0]))\n        return {\n            'terminal_singleton': float(len(run_lengths) > 1 and run_lengths[-1] == 1),\n            'two_run': float(len(run_lengths) == 2),\n            'switch_density': float(switches.size) / float(max(1, m - 1)),\n            'susceptibility': reversal_susceptibility,\n            'fractions': fractions,\n            'canonical': canonical,\n            'terminal_direction': float(dirs[-1])\n        }\n\n    def context_value(summaries, learning_rate):\n        ages = np.arange(len(summaries) - 1, -1, -1, dtype=float)\n        w = np.power(1.0 - learning_rate, ages)\n        if len(summaries) > 1:\n            w[-1] *= 0.35\n        w = w / max(float(np.sum(w)), 1e-12)\n\n        terminal = float(np.sum([wi * s['terminal_singleton'] for wi, s in zip(w, summaries)]))\n        two_run = float(np.sum([wi * s['two_run'] for wi, s in zip(w, summaries)]))\n        switch_density = float(np.sum([wi * s['switch_density'] for wi, s in zip(w, summaries)]))\n        susceptibility = float(np.sum([wi * s['susceptibility'] for wi, s in zip(w, summaries)]))\n\n        hist = np.zeros(5, dtype=float)\n        mass = 0.0\n        for wi, s in zip(w, summaries):\n            fr = s['fractions']\n            if fr.size == 0:\n                continue\n            contribution = float(wi) / float(fr.size)\n            for f in fr:\n                j = min(4, max(0, int(np.floor(5.0 * float(f)))))\n                hist[j] += contribution\n                mass += contribution\n        if mass > 0.0:\n            p = hist / mass\n            concentration = float(np.sum(p ** 2))\n            positive = p[p > 0.0]\n            entropy = float(-np.sum(positive * np.log(positive)) / np.log(5.0))\n        else:\n            concentration = 0.2\n            entropy = 1.0\n\n        validity_gradient = float(np.clip(np.std(reliability) / max(np.mean(reliability), 1e-8), 0.0, 2.0))\n        score = (1.30\n                 + 1.20 * (terminal - 0.30)\n                 + 1.20 * (concentration - 0.30)\n                 + 0.80 * (two_run - 0.50)\n                 - 1.50 * (entropy - 0.70)\n                 - 0.70 * (switch_density - 0.25)\n                 - 0.20 * validity_gradient\n                 + 0.85 * (susceptibility - 0.35))\n        return 1.0 / (1.0 + np.exp(-1.90 * np.clip(score - 1.30, -30.0, 30.0)))\n\n    def one_subject(frame):\n        summaries = []\n        records = []\n        for _, row in frame.iterrows():\n            s = summarize(row['option_a_ratings'], row['option_b_ratings'])\n            if s is None:\n                continue\n            summaries.append(s)\n\n            # Average across the advocated learning-rate range to reduce\n            # sensitivity to unknown individual parameter values.\n            x = float(np.mean([context_value(summaries, lr) for lr in (0.05, 0.085, 0.12)]))\n\n            # Retain conflict sequences only. The three canonical geometries\n            # are +-----, +++-, and +++++- (including their A/B reversals).\n            if len(s['canonical']) > 1 and len(set(s['canonical'])) > 1:\n                chosen_direction = 1.0 if int(row['response']) == 0 else -1.0\n                y = 1.0 if chosen_direction == s['terminal_direction'] else -1.0\n                records.append((s['canonical'], x, y))\n\n        if len(records) < 4:\n            return 0.0\n\n        residual_x = np.zeros(len(records), dtype=float)\n        keys = [r[0] for r in records]\n        xvals = np.asarray([r[1] for r in records], dtype=float)\n        yvals = np.asarray([r[2] for r in records], dtype=float)\n\n        for key in set(keys):\n            loc = np.asarray([k == key for k in keys], dtype=bool)\n            residual_x[loc] = xvals[loc] - float(np.mean(xvals[loc]))\n\n        rms = float(np.sqrt(np.mean(residual_x ** 2)))\n        if not np.isfinite(rms) or rms < 1e-10:\n            return 0.0\n        z = residual_x / rms\n        return float(np.mean(z * yvals))\n\n    if 'subject_id' in data.columns:\n        values = [one_subject(g) for _, g in data.groupby('subject_id', sort=False)]\n        return float(np.mean(values)) if values else 0.0\n    return one_subject(data)\n",
  "rationale": "This metric measures whether terminal-option choices covary with the advocated theory's continuously learned context state. For each subject, it reconstructs a parameter-robust estimate of the theory's latent overwrite calibration from stimulus history, then computes the standardized covariance between that state and choosing the terminal cue's option on conflict trials. Context is centered separately within the +-----, +++-, and +++++- geometries, so stable differences among probe types and their baseline terminal-choice rates cannot create the effect. Exact A/B reversals are aligned by terminal direction. The advocated account predicts a positive covariance because high learned overwrite calibration increases terminal control; the competing account is history-independent and predicts zero covariance. Pooling is explicitly the mean of subject-level values, ensuring the same definition works for full and single-subject data."
}
```

## Usage

```json
{
  "input_tokens": 7238,
  "output_tokens": 3819
}
```
