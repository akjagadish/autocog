# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Interference-Limited Retrospective Reason Sampling with Length-Calibrated Accessibility. After viewing all cues, a decision maker retrospectively retrieves one or, subject to stable working-memory capacity, occasionally two discriminating reasons. In two-reason conflicts, retrieval has a strong saturated terminal-recency floor, producing reliable last-reason capture without importing subject heterogeneity from longer conflicts. Around four reasons, stable variation in recency remains expressed. When conflicts exceed four reasons, nonlinear source interference flattens recency while accessibility approaches a validity-biased rather than uniform asymptote. Choice follows the signed, validity-sensitive evidence supplied by the retrieved reasons, with a small response lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Retrospective reason sampling expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    capacity = int(parameters["memory_capacity"])
    recency = float(parameters["recency_bias"])
    validity_access = float(parameters["validity_access"])
    interference = float(parameters["interference_rate"])
    continuation = float(parameters["second_reason_rate"])
    evidence_floor = float(parameters["evidence_floor"])
    evidence_gain = float(parameters["evidence_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_indices = np.flatnonzero(diff != 0)
    m = int(cue_indices.size)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Reliability is represented as instructed log-odds. Clipping only
    # prevents infinite evidence for a validity communicated as exactly 1.
    v = np.clip(validities[cue_indices], 0.500001, 0.999)
    reliability = np.log(v / (1.0 - v))
    directions = np.sign(diff[cue_indices])  # +1 favors A, -1 favors B

    # Cues are displayed from highest to lowest validity. Ordinal position
    # among discriminating cues therefore indexes retrospective recency.
    ordinal_recency = np.arange(m, dtype=float)

    # Interference is negligible through four reasons but grows cubically
    # thereafter, rapidly flattening accessibility in long conflicts.
    overload = float(max(0, m - 4))
    effective_recency = recency / (1.0 + interference * overload * overload * overload)

    # A two-reason conflict is represented as a compact terminal episode:
    # its recency gradient is strong and saturates across subjects. Stable
    # recency differences remain fully expressed in four-reason conflicts.
    if m == 2:
        effective_recency = 3.4 + 0.15 * np.tanh(recency - 2.0)

    # Under overload, interference removes temporal-source information but
    # does not make all reasons equally accessible. Retrieval instead tends
    # toward a modestly validity-biased asymptote.
    effective_validity_access = (
        validity_access + 0.55 * overload / (1.0 + overload)
    )

    log_access = (
        effective_recency * ordinal_recency
        + effective_validity_access * reliability
    )
    log_access -= np.max(log_access)
    access = np.exp(log_access)
    first_probs = access / access.sum()

    def probability_a(retrieved_positions):
        # Every retrieved reason supplies substantial directional evidence;
        # instructed validity additionally scales that evidence and resolves
        # conflicts when more than one reason is retrieved.
        positions = list(retrieved_positions)
        evidence = float(np.sum(
            directions[positions]
            * (evidence_floor + reliability[positions])
        ))
        x = float(np.clip(evidence_gain * evidence, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Exact marginalization over the first retrospectively retrieved reason.
    p_a_one = 0.0
    for i in range(m):
        p_a_one += float(first_probs[i]) * probability_a((i,))

    # Capacity is an upper bound: even capacity-two subjects often terminate
    # retrieval after one sufficient reason. If retrieval continues, the
    # second reason is sampled without replacement from the same accessibility
    # distribution and both reasons are integrated.
    if capacity >= 2 and m >= 2 and continuation > 0.0:
        p_a_two = 0.0
        for i in range(m):
            remaining_mass = float(access.sum() - access[i])
            if remaining_mass <= 0.0:
                continue
            for j in range(m):
                if j == i:
                    continue
                p_ordered = float(first_probs[i]) * float(access[j] / remaining_mass)
                p_a_two += p_ordered * probability_a((i, j))
        p_a_core = (1.0 - continuation) * p_a_one + continuation * p_a_two
    else:
        p_a_core = p_a_one

    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=10):** [0.97, 0.94, 0.9, 0.82, 0.76, 0.69, 0.63, 0.59, 0.55, 0.52]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]

**Rationale:** This is a fixed-length, fixed-tally permutation test targeting the competing theory's overload-induced, validity-biased accessibility. All trials contain 10 discriminating cues, with five favoring each option, so discriminating-cue count, fatigue, and total directional tally are held constant. In the two base trials, the final two cues both favor B. The first base sequence is AAAAABBBBB: A owns the five most valid cues, whereas B owns the five least valid cues. The second is BBBAAAAABB: B owns the three most valid cues and the final two cues, while A owns the middle five. Recency-Gated Reason Overwrite predicts similar B preferences in both base conditions because two consecutive terminal B reasons almost completely overwrite any earlier state; with ten reasons, both receive the identical fatigue factor (1-update_fatigue)^8. Even at the lowest attention and overwrite rates, two terminal overwrites leave less than about 1% of a fully conflicting prior state, so the different prefixes should have little effect. In contrast, the retrospective-sampling theory predicts a large permutation effect. At m=10, cubic interference nearly eliminates its ordinal-recency gradient and retrieval approaches a validity-biased asymptote. It should therefore favor A in AAAAABBBBB, where A possesses all five high-validity reasons, but favor B in BBBAAAAABB, where B possesses the three strongest reasons as well as the terminal reasons. Thus, the critical result is not merely overall accuracy but the within-subject contrast: the advocated theory predicts nearly equal terminal-option choice rates across the two base permutations, whereas the competitor predicts a substantial shift, potentially crossing 0.50. Exact A/B reversals balance response side. Four unique pairs repeated 24 times yield 96 trials.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
This is the within-subject permutation effect on choosing the option supported by the final two cues. It subtracts terminal-option choice in AAAAABBBBB-type block trials from terminal-option choice in BBBAAAAABB-type mixed trials, pooling each trial with its exact A/B reversal. The advocated overwrite theory predicts a contrast near zero because both permutations have the same number of updates and the same two-cue terminal suffix. The retrospective-sampling theory predicts a large positive contrast: validity-biased retrieval disfavors the terminal option when it owns only the five weakest cues, but favors it when it owns the three strongest cues as well as the final two. Each component uses 48 repeated trials per subject, and the mirrored versions balance response-side tendencies, yielding substantially lower between-subject noise than a raw choice rate.

Source:
def metric(data: pd.DataFrame) -> float:
    block_terminal = []
    mixed_terminal = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # The block permutation has one option favored by cues 1--5 and
        # the other option favored by cues 6--10.
        is_block = (
            d.size == 10
            and np.all(d[:5] == d[0])
            and np.all(d[5:] == d[5])
            and d[0] == -d[5]
        )

        # In this design the final two cues agree. Identify whether the
        # response selected the option favored by those terminal cues.
        terminal_response = 0 if d[-1] > 0 else 1
        chose_terminal = float(int(row["response"]) == terminal_response)

        if is_block:
            block_terminal.append(chose_terminal)
        else:
            mixed_terminal.append(chose_terminal)

    if len(block_terminal) == 0 or len(mixed_terminal) == 0:
        return float("nan")

    return float(np.mean(mixed_terminal) - np.mean(block_terminal))

### RESULTS
- Predicted under pi_3 (simulated): 0.0150 (var=0.0104)
- Predicted under pi_4 (simulated): 0.4617 (var=0.0125)
- Observed on real data: 0.0033 (var=0.0049)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=8):** [0.99, 0.95, 0.7, 0.69, 0.68, 0.67, 0.55, 0.51]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Rationale:** This is a nested conflict-length crossover test. In every base trial, the discriminating cues occur in descending-validity order as an initial run favoring A followed by a terminal run favoring B. The 2-cue condition uses the nearly matched 69% and 68% experts (AB), and the 4-cue condition uses the nearly matched 70%–67% experts (AABB). The advocated retrospective-sampling theory should therefore favor terminal option B at both short lengths: its two-reason terminal-recency floor is especially strong, and recency remains fully expressed at four reasons. The 7- and 8-cue conditions add highly valid early A reasons and low-validity terminal B reasons. Once conflict exceeds four reasons, the advocated theory's cubic interference sharply flattens recency while accessibility becomes validity-biased. It should consequently cross over to A for AAAABBB and AAAABBBB, aided by A's ownership of the 99% and 95% experts (and by a 4–3 reason advantage at length 7). The competing overwrite theory predicts no such sign reversal: its final attended cue and redundant terminal B run should preserve a B preference at every length. Its update-fatigue mechanism predicts only a smooth contraction of that B preference toward 0.50 as length increases. Thus, the key quantitative signature is an approximately B, B, A, A profile across 2, 4, 7, and 8 discriminating reasons under retrospective sampling, versus B preferences throughout with gradually declining magnitude under overwrite. The shorter conditions deliberately use experts with similar validities, ensuring that their expected B preference is driven by recency rather than a large reliability difference. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield exactly 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This is the proportion of long-conflict trials on which the subject chooses the option supported by the initial, high-validity cue run. It pools the 7- and 8-cue conditions and canonicalizes the A/B reversals, yielding 48 observations per subject and therefore relatively low sampling variance. Retrospective sampling predicts a high value because overload suppresses temporal recency and validity-biased accessibility favors the early 0.99/0.95 cues. Overwrite predicts a much lower value because the terminal run continues to determine the retained choice; fatigue only contracts that terminal preference toward chance. Restricting the metric to the long conditions avoids adding short-condition choice noise where both theories make similar terminal-choice predictions.

Source:
def metric(data: pd.DataFrame) -> float:
    early_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)

        # Restrict to the 7- and 8-reason overload conditions.
        if discriminating.size < 7:
            continue

        # The first discriminating cue identifies the option favored by the
        # high-validity, early run, regardless of physical A/B orientation.
        first_direction = diff[discriminating[0]]
        response = int(row["response"])
        chose_early_option = (
            (first_direction > 0 and response == 0) or
            (first_direction < 0 and response == 1)
        )
        early_choices.append(float(chose_early_option))

    if not early_choices:
        return float("nan")
    return float(np.mean(early_choices))

### RESULTS
- Predicted under pi_3 (simulated): 0.3379 (var=0.0051)
- Predicted under pi_4 (simulated): 0.7804 (var=0.0054)
- Observed on real data: 0.3067 (var=0.0051)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast TTB-consistent choice rates between trials where the
    # equal-weight tally supports versus opposes the TTB winner.
    support_choices = []
    oppose_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        differing = np.flatnonzero(d != 0)
        if differing.size == 0:
            continue

        # Features are already ordered from highest to lowest validity.
        ttb_direction = float(np.sign(d[differing[0]]))  # +1: A, -1: B
        ttb_response = 0 if ttb_direction > 0 else 1
        chose_ttb = float(int(row["response"]) == ttb_response)

        # Positive iff the tally margin points toward the TTB winner.
        tally_a_margin = float(np.sign(d).sum())
        oriented_tally_margin = ttb_direction * tally_a_margin

        if oriented_tally_margin > 0:
            support_choices.append(chose_ttb)
        elif oriented_tally_margin < 0:
            oppose_choices.append(chose_ttb)

    if len(support_choices) == 0 or len(oppose_choices) == 0:
        return float("nan")

    return float(np.mean(support_choices) - np.mean(oppose_choices))
```

**Observed (real) value:** 0.4944 (var=0.0204)
**Predicted under pi_3:** 0.4743 (var=0.0168)
**Predicted under pi_4:** 0.5279 (var=0.0111)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate at which choices follow the highest-validity discriminating cue."""
    if len(data) == 0:
        return float("nan")

    matches = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features are already ordered from highest to lowest validity.
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            # An all-tie pair provides no directional diagnosticity.
            continue

        first = int(discriminating[0])
        ttb_response = 0 if diff[first] > 0 else 1
        matches.append(float(int(row["response"]) == ttb_response))

    if not matches:
        return float("nan")
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1938 (var=0.0049)
**Predicted under pi_3:** 0.1931 (var=0.0026)
**Predicted under pi_4:** 0.2060 (var=0.0019)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices agreeing with the least-valid discriminating cue."""
    if len(data) == 0:
        return float("nan")

    agreements = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        # Features are displayed in descending validity order, so the
        # largest differing index is the final discriminating reason.
        final_idx = int(differing[-1])
        final_cue_choice = 0 if a[final_idx] > b[final_idx] else 1
        agreements.append(float(int(response) == final_cue_choice))

    if not agreements:
        return float("nan")
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.5750 (var=0.0026)
**Predicted under pi_3:** 0.7129 (var=0.0033)
**Predicted under pi_4:** 0.5346 (var=0.0080)

### Experiment 6
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signed association between choice and the lowest-validity (fourth) cue.
    # +1 means the response follows that cue, -1 means it opposes that cue.
    products = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        terminal_direction = np.sign(a[-1] - b[-1])  # +1 favors A, -1 favors B
        if terminal_direction == 0:
            continue
        response = int(row['response'])
        choice_direction = 1.0 if response == 0 else -1.0
        products.append(choice_direction * terminal_direction)
    if len(products) == 0:
        return float('nan')
    return float(np.mean(products))
```

**Observed (real) value:** 0.6650 (var=0.0621)
**Predicted under pi_3:** 0.5262 (var=0.0122)
**Predicted under pi_4:** 0.6104 (var=0.0504)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, pi_3 is the substantially better task-invariant account. The decisive evidence comes from Experiments 1 and 2. In Experiment 1, the observed permutation effect was essentially zero (0.0033), matching pi_3 (0.0150) and sharply contradicting pi_4 (0.4617). In Experiment 2, long conflicts continued to favor the terminal run: the observed early-option rate was 0.3067, close to pi_3's 0.3379 but opposite pi_4's predicted crossover to 0.7804. These are not small calibration errors for pi_4; they jointly reject its central claim that overload flattens recency and exposes validity-biased retrospective accessibility. The low observed between-subject variances in these experiments also make the discrepancies difficult to attribute to heterogeneous strategies. Pi_3 additionally fits Experiment 3 well (0.4743 predicted versus 0.4944 observed) and is nearly exact in Experiment 4 (0.1931 versus 0.1938). Pi_4 is acceptable on those two experiments but not superior. Pi_3 is imperfect: it overpredicts final-cue agreement in Experiment 5 (0.7129 versus 0.5750) and underpredicts terminal-cue association in Experiment 6 (0.5262 versus 0.6650). Pi_4 is closer on both, especially Experiment 6, but neither discrepancy offsets its catastrophic failures in Experiments 1 and 2. The residual pattern is theoretically informative: terminal control is strong for short conflicts and coherent terminal runs, as in Experiments 1, 2, and 6, but weaker when the final cue is isolated within longer, frequently alternating sequences, as in Experiment 5. This suggests that directional run structure and switch-induced interference matter, rather than pi_4's global long-conflict shift toward high-validity cues. Because pi_4's failures directly follow from its defining overload mechanism, merely retuning its current model would not produce a credible task-invariant competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_4 with a new directional-chunk or run-segmentation theory. The new theory should propose that sequentially viewed cues are compressed into contiguous directional runs. A coherent terminal run forms a strong decision chunk and can suppress earlier chunks, whereas an isolated final cue has weaker control, especially after many direction switches. Short sequences should retain high temporal resolution, producing the strong terminal association observed in Experiment 6; long alternating sequences should produce switch interference and moderate final-cue agreement, accounting for Experiment 5. Crucially, increasing conflict length must not trigger a validity-biased reversal toward early cues: coherent terminal runs should continue to dominate in Experiments 1 and 2, while prefix permutations should have little effect when they share the same strong suffix. Instructed validity may scale cue encoding or chunk confidence, but it should not become the dominant retrieval asymptote under overload. This theory differs from pi_3 by making directional chunks, run length, and number of switches\u2014not independent cue-by-cue overwrites followed by count-based fatigue\u2014the core representational mechanism. It differs from pi_4 by rejecting retrospective sampling and its forced long-conflict validity crossover. Its model should therefore include run formation, switch-dependent interference, recency-weighted competition among run summaries, confidence saturation within repeated-direction runs, and a small lapse process. It should be tested against separate conditions crossing total cue count, terminal-run length, and number of directional switches so that it remains distinguishable from pi_3 rather than becoming a flexible reparameterization of it."
}
```

## Usage

```json
{
  "input_tokens": 9655,
  "output_tokens": 2022
}
```
