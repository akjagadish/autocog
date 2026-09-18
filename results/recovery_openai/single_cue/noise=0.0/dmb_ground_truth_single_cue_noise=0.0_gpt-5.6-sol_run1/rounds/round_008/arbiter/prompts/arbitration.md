# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_10") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_10" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_10
Reliability-Gated Decision-Object Arbitration with Prefix-Coherence-Conditioned Terminal Access. People construct at most three bounded objects from each display: an initial high-validity commitment, a reliability-weighted summary of the visible coalition, and a privileged terminal reason or terminal directional run. Arbitration is stimulus-local and history-free. Terminal evidence has a genuine recency advantage, but weak terminal evidence receives reliable access only when it constitutes the sole coherent reversal against a unified prefix at manageable load. A terminal reason following an already fragmented prefix receives no such accessibility floor and may be mildly discounted. Very short conflicts provide too little separation for a strongly privileged counterargument, whereas seven-or-more-reason conflicts reduce access through object competition rather than general count fatigue. The initial commitment remains a weak but non-negligible fallback, coalition evidence saturates, and unanimous evidence experiences only a modest vigilance decline. Stable variation in coherent-terminal accessibility, terminal susceptibility, coalition reliance, attention, and lapses produces heterogeneous conflict choices without learned context.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Decision-object arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    early_commitment = float(parameters["early_commitment"])
    coalition_reliance = float(parameters["coalition_reliance"])
    terminal_susceptibility = float(parameters["terminal_susceptibility"])
    reliability_threshold = float(parameters["terminal_reliability_threshold"])
    reliability_sharpness = float(parameters["terminal_reliability_sharpness"])
    coherent_access_floor = float(parameters["coherent_access_floor"])
    terminal_run_bonus = float(parameters["terminal_run_bonus"])
    terminal_singleton_bonus = float(parameters["terminal_singleton_bonus"])
    conflict_moderation = float(parameters["conflict_moderation"])
    attentional_reliability = float(parameters["attentional_reliability"])
    choice_gain = float(parameters["choice_gain"])
    vigilance_decline = float(parameters["vigilance_decline"])
    lapse_rate = float(parameters["lapse_rate"])

    # Communicated validity determines psychological inspection order. Exact
    # ties retain display order, but physical gaps otherwise have no role.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]
    discriminating = [int(j) for j in cue_order if diff[j] != 0.0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.asarray(
        [float(np.sign(diff[j])) for j in discriminating], dtype=float
    )

    # Convert instructed validities to reliability strengths. Compression
    # preserves whether a validity profile is flat or steep without allowing
    # one extreme cue to create an unbounded accumulator.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    log_odds = np.log(clipped_v / (1.0 - clipped_v))
    positive = log_odds[log_odds > 0.0]
    scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    normalized = np.clip(log_odds / scale, 1e-4, 30.0)
    strengths_all = np.power(normalized, validity_compression)
    strengths = np.asarray(
        [float(strengths_all[j]) for j in discriminating], dtype=float
    )

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # OBJECT 1: an early high-validity commitment. Its availability receives
    # a small boost only when the first reason is unusually reliable.
    first_direction = float(directions[0])
    first_strength = float(strengths[0])
    maximum_strength = max(float(np.max(strengths)), 1e-8)
    first_relative = first_strength / maximum_strength
    early_availability = 1.0 - np.exp(-first_strength)
    w_early = early_commitment * early_availability * (
        1.0 + 0.5 * first_relative
    )

    # OBJECT 2: a bounded reliability-weighted summary of the visible
    # coalition. Directional mass saturates; no sequential accumulator or
    # learned contextual state is constructed. Balance sensitivity is
    # compressed so the coalition does not duplicate the initial object.
    signed_support = float(np.sum(directions * strengths))
    total_support = max(float(np.sum(strengths)), 1e-12)
    coalition_balance = abs(signed_support) / total_support
    coalition_balance = float(np.clip(coalition_balance, 0.0, 1.0))
    coalition_direction = float(np.sign(signed_support))
    coalition_saturation = 1.0 - np.exp(-total_support / 2.0)
    if coalition_direction == 0.0:
        w_coalition = 0.0
    else:
        w_coalition = (
            coalition_reliance
            * coalition_saturation
            * (0.22 + 0.58 * coalition_balance)
        )

    # OBJECT 3: the final directional run in validity order. Run support is
    # represented as one chunk, and its length benefit rapidly saturates.
    terminal_direction = float(directions[-1])
    run_start = m - 1
    while run_start > 0 and directions[run_start - 1] == terminal_direction:
        run_start -= 1
    terminal_run_length = m - run_start
    terminal_run_strength = float(np.sum(strengths[run_start:]))
    run_saturation = 1.0 - np.exp(-terminal_run_strength)
    n_runs = 1 + int(np.sum(directions[1:] != directions[:-1]))

    # Terminal access depends on reliability relative to the strongest
    # currently visible reason. This creates nonuniform recency across
    # validity environments without learning an experiment-level overwrite
    # state. Even weak terminal reasons retain a small intrinsic privilege.
    terminal_relative = float(strengths[-1] / maximum_strength)
    terminal_access = sigmoid(
        (terminal_relative - reliability_threshold)
        / max(reliability_sharpness, 1e-6)
    )

    # A weak terminal counterargument is distinctly retrievable when it is a
    # short, sole reversal against a coherent prefix at manageable load. No
    # such floor is granted after a fragmented prefix; short terminal runs in
    # those displays receive only a mild local discount. Four-cue conflicts
    # and seven-plus coherent reversals are left unchanged.
    prefix = directions[:run_start]
    prefix_coherent = bool(
        prefix.size > 0
        and np.all(prefix == prefix[0])
        and prefix[0] != terminal_direction
    )
    short_terminal_run = terminal_run_length <= max(2, m // 2)
    if m in (5, 6) and prefix_coherent and short_terminal_run:
        singleton_boost = 0.08 * float(terminal_run_length == 1)
        terminal_access = max(
            terminal_access,
            min(0.92, coherent_access_floor + singleton_boost)
        )
    elif m >= 5 and terminal_run_length <= 2 and prefix.size > 1:
        prefix_fragmented = bool(np.any(prefix[1:] != prefix[:-1]))
        if prefix_fragmented:
            terminal_access *= 0.88

    reliability_gate = 0.08 + 0.92 * terminal_access

    # A coherent reversal is maximally accessible at manageable loads. The
    # smooth window weakens this privilege for very short conflicts and for
    # seven-or-more-reason overload without imposing general count fatigue.
    manageable_access = (
        sigmoid((float(m) - 3.5) / 0.35)
        * sigmoid((6.5 - float(m)) / 0.35)
    )
    geometry_multiplier = 1.0 + 0.25 * terminal_run_bonus * run_saturation
    if n_runs == 2:
        singleton = float(terminal_run_length == 1)
        coherent_access = 2.0 + terminal_singleton_bonus * singleton
        geometry_multiplier *= (
            1.0
            + terminal_run_bonus * manageable_access * coherent_access
        )
    elif n_runs >= 3:
        # Fragmented conflicts provide several competing retrieval boundaries;
        # their terminal reason remains privileged but is less diagnostic.
        fragmentation = min(2.0, float(n_runs - 2))
        geometry_multiplier /= (
            1.0 + 0.50 * terminal_run_bonus * fragmentation
        )
    w_terminal = (
        terminal_susceptibility
        * reliability_gate
        * (0.35 + 0.65 * attentional_reliability)
        * geometry_multiplier
    )

    # A conflicting coalition moderates, but cannot categorically erase, the
    # terminal object. Switch count and physical gaps do not enter this gate.
    if coalition_direction != 0.0 and coalition_direction != terminal_direction:
        opposing_advantage = coalition_balance * min(
            2.0, np.sqrt(float(m) / 2.0)
        )
        w_terminal /= 1.0 + conflict_moderation * opposing_advantage

    # Each object has a bounded probability of expressing its favored option.
    # Attentional reliability affects readout globally while the object
    # strengths determine confidence locally.
    early_confidence = sigmoid(
        choice_gain * attentional_reliability * (0.65 + 0.35 * early_availability)
    )
    coalition_confidence = sigmoid(
        choice_gain
        * attentional_reliability
        * (0.40 + 0.50 * coalition_balance)
    )
    terminal_confidence = sigmoid(
        choice_gain
        * attentional_reliability
        * (0.62 + 0.38 * run_saturation)
    )

    def p_a_from_object(direction, confidence):
        if direction > 0.0:
            return float(confidence)
        if direction < 0.0:
            return float(1.0 - confidence)
        return 0.5

    object_weights = np.asarray(
        [w_early, w_coalition, w_terminal], dtype=np.float64
    )
    object_predictions = np.asarray([
        p_a_from_object(first_direction, early_confidence),
        p_a_from_object(coalition_direction, coalition_confidence),
        p_a_from_object(terminal_direction, terminal_confidence)
    ], dtype=np.float64)

    weight_sum = float(np.sum(object_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        p_a_core = 0.5
    else:
        p_a_core = float(np.dot(object_weights, object_predictions) / weight_sum)

    # Only coherent added evidence incurs a mild vigilance decrement. Because
    # all objects agree on such trials, evidence remains strong and the load
    # effect is much smaller than obligatory count fatigue.
    unanimous = bool(np.all(directions == directions[0]))
    extra_lapse = 0.0
    if unanimous:
        extra_lapse = vigilance_decline * min(
            1.0, max(0.0, float(m) - 2.0) / 8.0
        )
    effective_lapse = float(np.clip(lapse_rate + extra_lapse, 0.0, 0.45))

    p_a = (1.0 - effective_lapse) * p_a_core + effective_lapse * 0.5
    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.99, 0.9, 0.8, 0.7, 0.58, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 3: A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 4: A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 8: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** This is a coherent-prefix terminal-access test at the competing theory's privileged five- and six-reason load window. In descending-validity order, the base sequences are AAAAB versus ABAAB at m=5 and AAAAAB versus ABAAAB at m=6. Within each load, the pair has the same weak terminal B reason and a shared multi-cue A,A,B suffix; only an earlier reason changes. Recency-Gated Reason Overwrite predicts nearly identical B-choice rates within each pair. The shared A,A,B suffix repeatedly overwrites the earlier difference, and both conditions at a given load receive exactly the same fatigue and lapse transformations. It predicts only a modest common contraction toward chance from five to six reasons. Decision-Object Arbitration instead predicts a large prefix-coherence contrast. AAAAB and AAAAAB are sole singleton reversals against unified prefixes at manageable load, so the 50%-valid terminal reason receives the coherent-access floor, singleton bonus, and two-run geometry advantage. ABAAB and ABAAAB have fragmented prefixes, so that floor is absent, weak-terminal access is discounted, and four-run fragmentation further suppresses the terminal object; their A-favoring coalitions should consequently exert much greater control. The critical quantitative signature is substantially more terminal-option choice for coherent than fragmented prefixes at both loads, potentially with fragmented trials favoring A, against overwrite's approximately zero within-load contrast. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This is a thresholded prefix-coherence effect on terminal-option choices, pooled over the five- and six-reason conditions and made response-side invariant by identifying the option favored by the final discriminating cue. Recency-gated overwrite gives coherent and fragmented sequences nearly the same terminal-choice probability because they share the same late directional suffix; its pooled raw contrast should therefore lie inside the 10-point dead band, yielding a score near 0. Decision-object arbitration predicts substantially greater terminal access after a unified prefix, so its contrast should pass the dead band and often approach the score's ceiling of 1. The dead band suppresses ordinary binomial fluctuations around the overwrite theory's near-zero effect, while saturation limits between-subject variance caused by heterogeneity in the competing theory's terminal-access parameters. Exact A/B reversals ensure the contrast is not contaminated by response-side preferences.

Source:
def metric(data: pd.DataFrame) -> float:
    coherent_terminal = []
    fragmented_terminal = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        directions = np.sign(diff[diff != 0])
        m = int(directions.size)

        if m not in (5, 6):
            continue

        # The diagnostic displays have a singleton terminal reversal.
        terminal_singleton = bool(
            m >= 2 and directions[-1] != directions[-2]
        )
        if not terminal_singleton:
            continue

        terminal_direction = float(directions[-1])
        response = int(row["response"])
        chose_terminal_option = float(
            (terminal_direction > 0 and response == 0)
            or (terminal_direction < 0 and response == 1)
        )

        prefix = directions[:-1]
        coherent = bool(
            prefix.size > 0
            and np.all(prefix == prefix[0])
            and prefix[0] != terminal_direction
        )

        if coherent:
            coherent_terminal.append(chose_terminal_option)
        else:
            fragmented_terminal.append(chose_terminal_option)

    if len(coherent_terminal) == 0 or len(fragmented_terminal) == 0:
        return float("nan")

    raw_contrast = (
        float(np.mean(coherent_terminal))
        - float(np.mean(fragmented_terminal))
    )

    # A 10-percentage-point dead band treats smaller differences as negligible;
    # effects of 30 percentage points or more receive the ceiling score.
    return float(np.clip((raw_contrast - 0.10) / 0.20, 0.0, 1.0))

### RESULTS
- Predicted under pi_3 (simulated): 0.0000 (var=0.0227)
- Predicted under pi_10 (simulated): 1.0000 (var=0.0480)
- Observed on real data: 0.0000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_10)

### DESIGN
**Validities (n_features=8):** [0.99, 0.95, 0.9, 0.84, 0.78, 0.71, 0.63, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 2: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 6: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Rationale:** This is an adjacent-load window test of coherent-singleton terminal access. In every base trial, cues are encountered in descending-validity order as a coherent run favoring A followed by the same extremely weak terminal cue favoring B. The six sequences are AAB, AAAB, AAAAB, AAAAAB, AAAAAAB, and AAAAAAAB, giving m=3 through m=8 while keeping the terminal expert, terminal direction, singleton-run geometry, and absence of prefix fragmentation fixed. Decision-Object Arbitration predicts a sharply nonmonotonic terminal-B profile. At m=3–4, the conflict is too short for the full coherent-reversal privilege, and the 50% terminal expert has little reliability-gated access. At m=5–6, the coherent-prefix access floor, singleton bonus, and manageable-load geometry jointly produce a large increase in terminal-B choices despite the growing A coalition. At m=7–8, the access floor is removed and object competition suppresses the terminal reversal, so choices should shift back toward the coherent A prefix. The diagnostic signature is therefore a localized peak in terminal-option choice at m=5–6, especially a positive jump from m=4 to m=5 followed by a drop from m=6 to m=7. Recency-Gated Reason Overwrite predicts B at every load because the same final attended cue overwrites the A state. Its core terminal control is nearly constant across these unanimous prefixes, while update fatigue produces only a smooth monotonic contraction of B choice toward 0.50 as m increases. It therefore cannot predict either the m=4-to-5 increase or the window-shaped profile. Exact A/B reversals balance response side. Twelve unique pairs repeated eight times yield 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is a localized window-boundary contrast in terminal-option choice. It is positive when terminal control rises from load 4 to 5 and then falls from load 6 to 7, the specific nonmonotonic profile predicted by coherent-prefix-conditioned terminal access. Under overwrite plus fatigue, terminal choice should contract smoothly with load: the 4-to-5 change is negative while the 6-to-7 term enters with the opposite sign, so the two adjacent changes largely cancel, leaving a value near zero or slightly negative. Coding choices relative to the terminal option and averaging the two A/B-reversed displays removes response-side preference. Each of the four load means uses 16 trials per subject, yielding a comparatively stable within-subject contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    # Code whether the response selected the option favored by the final
    # discriminating expert in descending-validity order. In this experiment,
    # feature columns are already ordered from highest to lowest validity.
    by_load = {4: [], 5: [], 6: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        load = int(discriminating.size)
        if load not in by_load:
            continue

        final_index = int(discriminating[-1])
        terminal_option = 0 if diff[final_index] > 0 else 1
        terminal_chosen = float(int(row["response"]) == terminal_option)
        by_load[load].append(terminal_chosen)

    if any(len(by_load[m]) == 0 for m in (4, 5, 6, 7)):
        return float("nan")

    p = {m: float(np.mean(by_load[m])) for m in (4, 5, 6, 7)}
    # Average of the upward boundary jump (4->5) and downward boundary
    # jump in terminal choice (6->7).
    return float(0.5 * ((p[5] - p[4]) + (p[6] - p[7])))

### RESULTS
- Predicted under pi_3 (simulated): -0.0175 (var=0.0140)
- Predicted under pi_10 (simulated): 0.2694 (var=0.0251)
- Observed on real data: 0.0262 (var=0.0075)

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
**Predicted under pi_10:** 0.7281 (var=0.0159)

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
**Predicted under pi_10:** 0.2408 (var=0.0078)

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
**Predicted under pi_10:** 0.7527 (var=0.0062)

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
**Predicted under pi_10:** 0.5654 (var=0.0275)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.0033 (var=0.0049)
**Predicted under pi_3:** 0.0150 (var=0.0104)
**Predicted under pi_10:** 0.1788 (var=0.0194)

### Experiment 8
**Design**
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.3067 (var=0.0051)
**Predicted under pi_3:** 0.3379 (var=0.0051)
**Predicted under pi_10:** 0.2900 (var=0.0177)

### Experiment 9
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    low_follow = []
    high_follow = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = diff[diff != 0]

        # The strongest predicted chunking contrast occurs for the 8- and
        # 10-reason displays.
        if discriminating.size < 8:
            continue

        directions = np.sign(discriminating)
        n_switches = int(np.sum(directions[1:] != directions[:-1]))

        # Response coding is 0=A and 1=B. Determine whether the response
        # follows the terminal discriminating cue, independent of side.
        terminal_response = 0 if directions[-1] > 0 else 1
        follows_terminal = float(int(row["response"]) == terminal_response)

        if n_switches <= 1:
            low_follow.append(follows_terminal)
        elif n_switches >= 4:
            high_follow.append(follows_terminal)

    if len(low_follow) == 0 or len(high_follow) == 0:
        return float("nan")

    contrast = float(np.mean(low_follow) - np.mean(high_follow))

    # A preregisterable practical-significance diagnostic: whether coherent
    # low-switch sequences have more than an 8-percentage-point terminal-cue
    # advantage over switch-dense sequences.
    return float(contrast > 0.08)

```

**Observed (real) value:** 0.0000 (var=0.2016)
**Predicted under pi_3:** 0.0000 (var=0.1824)
**Predicted under pi_10:** 0.0000 (var=0.2356)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Direction of the terminal discriminating cue in displayed order.
        terminal_direction = diff[discriminating[-1]]
        terminal_response = 0 if terminal_direction > 0 else 1
        agreements.append(float(int(row['response']) == terminal_response))

    if len(agreements) == 0:
        return float('nan')
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.1721 (var=0.0112)
**Predicted under pi_3:** 0.2754 (var=0.0030)
**Predicted under pi_10:** 0.1440 (var=0.0033)

### Experiment 11
**Design**
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Features are already arranged in descending instructed validity.
        first_direction = float(np.sign(diff[int(discriminating[0])]))
        choice_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(first_direction * choice_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** -0.0325 (var=0.0099)
**Predicted under pi_3:** -0.0083 (var=0.0103)
**Predicted under pi_10:** 0.0575 (var=0.0053)

### Experiment 12
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Coherence-weighted agreement with the least-valid terminal cue."""
    if len(data) == 0:
        return float("nan")

    pattern_weights = {
        (1, 1, 1, 1): 1.00,   # fully coherent
        (-1, 1, 1, 1): 0.80,  # earliest cue conflicts
        (1, -1, 1, 1): 0.65,  # second cue conflicts
        (-1, -1, 1, 1): 0.35  # both prefix cues conflict
    }

    weighted_agreement = 0.0
    total_weight = 0.0

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        if diff.size != 4 or diff[3] == 0:
            continue

        # Orient every trial so that the terminal (least-valid) cue is +1.
        terminal_sign = float(np.sign(diff[3]))
        relative_pattern = tuple(
            int(np.sign(x) * terminal_sign) for x in diff
        )
        weight = pattern_weights.get(relative_pattern, 0.0)
        if weight <= 0.0:
            continue

        terminal_favoring_response = 0 if terminal_sign > 0 else 1
        agreement = float(int(response) == terminal_favoring_response)
        weighted_agreement += weight * agreement
        total_weight += weight

    if total_weight <= 0.0:
        return float("nan")
    return float(weighted_agreement / total_weight)
```

**Observed (real) value:** 0.8748 (var=0.0069)
**Predicted under pi_3:** 0.7843 (var=0.0034)
**Predicted under pi_10:** 0.8569 (var=0.0036)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        m = int(np.count_nonzero(diff))

        # The diagnostic manageable-load conflicts contain 5 or 6
        # discriminating cues. The final (lowest-validity) cue is index 9.
        if m not in (5, 6) or diff[-1] == 0:
            continue

        terminal_response = 0 if diff[-1] > 0 else 1
        scores.append(float(int(row["response"]) == terminal_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8425 (var=0.0123)
**Predicted under pi_3:** 0.6863 (var=0.0047)
**Predicted under pi_10:** 0.8008 (var=0.0073)

### Experiment 14
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Side-balanced rate of choosing the 3-to-2 majority in strong-coalition trials."""
    majority_a = []
    majority_b = []
    strong_set = {0, 1, 2, 11, 12}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = set(np.flatnonzero(diff != 0).tolist())
        if discriminating != strong_set:
            continue

        response = int(row["response"])
        majority_direction = float(np.sum(np.sign(diff)))
        if majority_direction > 0:
            majority_a.append(float(response == 0))
        elif majority_direction < 0:
            majority_b.append(float(response == 1))

    if majority_a and majority_b:
        return float(0.5 * (np.mean(majority_a) + np.mean(majority_b)))

    pooled = majority_a + majority_b
    if pooled:
        return float(np.mean(pooled))
    return 0.5

```

**Observed (real) value:** 0.1713 (var=0.0095)
**Predicted under pi_3:** 0.2650 (var=0.0077)
**Predicted under pi_10:** 0.1781 (var=0.0070)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    d = data.copy()
    if len(d) == 0:
        return float('nan')

    # The competing theory's contextual calibration has reached full weight
    # well before this point; the advocated theory has no history dependence.
    d['_trial_index'] = d.groupby('subject_id', sort=False).cumcount()
    d = d.loc[d['_trial_index'] >= 24].copy()
    if len(d) == 0:
        return float('nan')

    def terminal_match(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        terminal_diff = a[-1] - b[-1]
        if terminal_diff > 0:
            terminal_response = 0
        elif terminal_diff < 0:
            terminal_response = 1
        else:
            return np.nan
        return float(int(row['response']) == terminal_response)

    d['_terminal_match'] = d.apply(terminal_match, axis=1)
    d['_stimulus_key'] = d.apply(
        lambda r: (tuple(r['option_a_ratings']), tuple(r['option_b_ratings'])),
        axis=1
    )

    # Equal-weight subject-by-stimulus cells, eliminating variance caused by
    # random trial order and unequal late-block counts of the six pairs.
    cells = d.groupby(
        ['subject_id', '_stimulus_key'], sort=False
    )['_terminal_match'].mean()
    return float(cells.mean())
```

**Observed (real) value:** 0.8810 (var=0.0070)
**Predicted under pi_3:** 0.7201 (var=0.0040)
**Predicted under pi_10:** 0.8495 (var=0.0032)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Classify each trial structurally and code whether the response followed
    # the physically terminal discriminating reason. In this design that
    # reason is also the final reason in instructed-validity scan order.
    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        idx = np.flatnonzero(diff != 0)
        if idx.size == 0:
            continue

        terminal_idx = int(idx[-1])
        terminal_is_b = bool(diff[terminal_idx] < 0)
        chose_terminal = float(int(row["response"]) == int(terminal_is_b))

        if idx.size > 1:
            largest_gap = int(np.max(np.diff(idx) - 1))
        else:
            largest_gap = 0
        foregrounded = bool(terminal_idx == (len(diff) - 1) and largest_gap > 0)
        records.append((foregrounded, terminal_is_b, chose_terminal))

    if not records:
        return float("nan")

    frame = pd.DataFrame(
        records, columns=["foregrounded", "terminal_is_b", "chose_terminal"]
    )

    # Compute the foregrounding contrast separately for terminal-A and
    # terminal-B trials, then average. This exactly removes any stable A/B
    # response-side preference rather than allowing it to inflate variance.
    side_contrasts = []
    for terminal_is_b in (False, True):
        side = frame[frame["terminal_is_b"] == terminal_is_b]
        fg = side.loc[side["foregrounded"], "chose_terminal"]
        clustered = side.loc[~side["foregrounded"], "chose_terminal"]
        if len(fg) > 0 and len(clustered) > 0:
            side_contrasts.append(float(fg.mean() - clustered.mean()))

    if not side_contrasts:
        return float("nan")
    return float(np.mean(side_contrasts))
```

**Observed (real) value:** 0.0192 (var=0.0062)
**Predicted under pi_3:** -0.0217 (var=0.0090)
**Predicted under pi_10:** -0.0117 (var=0.0094)

### Experiment 17
**Design**
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Winner-choice rate at each unanimous evidence load.
    correct_by_load = {2: [], 4: [], 7: [], 10: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        load = int(np.count_nonzero(diff))
        net_direction = float(np.sum(diff))

        if load not in correct_by_load or net_direction == 0.0:
            continue

        response = int(row["response"])
        favored_response = 0 if net_direction > 0.0 else 1
        correct_by_load[load].append(float(response == favored_response))

    rates = {}
    for load in (2, 4, 7, 10):
        values = correct_by_load[load]
        if len(values) == 0:
            return float("nan")
        rates[load] = float(np.mean(values))

    # Endpoint-emphasized, zero-sum decline contrast. Positive values indicate
    # that unanimous evidence becomes less effective as more reasons are added.
    return float(
        0.75 * rates[2]
        + 0.25 * rates[4]
        - 0.25 * rates[7]
        - 0.75 * rates[10]
    )
```

**Observed (real) value:** 0.0471 (var=0.0135)
**Predicted under pi_3:** 0.1525 (var=0.0093)
**Predicted under pi_10:** 0.0652 (var=0.0056)

### Experiment 18
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.asarray([0.99, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.51], dtype=float)
    reliability = np.log(np.clip(validities, 0.5001, 0.999) / (1.0 - np.clip(validities, 0.5001, 0.999)))
    scale = float(np.mean(reliability[reliability > 0.0]))
    strengths = np.power(np.clip(reliability / scale, 0.15, 6.0), 0.30)

    def summarize(a, b):
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        idx = np.flatnonzero(d != 0.0)
        if idx.size == 0:
            return None
        dirs = np.sign(d[idx])
        m = int(dirs.size)
        switches = np.flatnonzero(dirs[1:] != dirs[:-1]) + 1

        run_lengths = []
        run = 1
        for k in range(1, m):
            if dirs[k] == dirs[k - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

        accumulator = 0.0
        susceptibility = []
        k = 0
        while k < m:
            direction = float(dirs[k])
            end = k + 1
            while end < m and dirs[end] == direction:
                end += 1
            incoming = strengths[idx[k:end]]
            run_length = end - k
            if accumulator != 0.0 and np.sign(accumulator) != direction:
                incumbent = abs(accumulator) * (0.55 ** run_length)
                attack = 0.0
                for s in incoming:
                    attack = 0.55 * attack + float(s)
                advantage = (attack - incumbent) / max(attack + incumbent, 1e-12)
                susceptibility.append(1.0 / (1.0 + np.exp(-4.0 * np.clip(advantage, -15.0, 15.0))))
            for s in incoming:
                accumulator = 0.55 * accumulator + direction * float(s)
            k = end

        reversal_susceptibility = float(np.mean(susceptibility)) if susceptibility else 0.35
        fractions = switches.astype(float) / float(m) if switches.size else np.asarray([], dtype=float)
        canonical = tuple(int(x) for x in (dirs * dirs[0]))
        return {
            'terminal_singleton': float(len(run_lengths) > 1 and run_lengths[-1] == 1),
            'two_run': float(len(run_lengths) == 2),
            'switch_density': float(switches.size) / float(max(1, m - 1)),
            'susceptibility': reversal_susceptibility,
            'fractions': fractions,
            'canonical': canonical,
            'terminal_direction': float(dirs[-1])
        }

    def context_value(summaries, learning_rate):
        ages = np.arange(len(summaries) - 1, -1, -1, dtype=float)
        w = np.power(1.0 - learning_rate, ages)
        if len(summaries) > 1:
            w[-1] *= 0.35
        w = w / max(float(np.sum(w)), 1e-12)

        terminal = float(np.sum([wi * s['terminal_singleton'] for wi, s in zip(w, summaries)]))
        two_run = float(np.sum([wi * s['two_run'] for wi, s in zip(w, summaries)]))
        switch_density = float(np.sum([wi * s['switch_density'] for wi, s in zip(w, summaries)]))
        susceptibility = float(np.sum([wi * s['susceptibility'] for wi, s in zip(w, summaries)]))

        hist = np.zeros(5, dtype=float)
        mass = 0.0
        for wi, s in zip(w, summaries):
            fr = s['fractions']
            if fr.size == 0:
                continue
            contribution = float(wi) / float(fr.size)
            for f in fr:
                j = min(4, max(0, int(np.floor(5.0 * float(f)))))
                hist[j] += contribution
                mass += contribution
        if mass > 0.0:
            p = hist / mass
            concentration = float(np.sum(p ** 2))
            positive = p[p > 0.0]
            entropy = float(-np.sum(positive * np.log(positive)) / np.log(5.0))
        else:
            concentration = 0.2
            entropy = 1.0

        validity_gradient = float(np.clip(np.std(reliability) / max(np.mean(reliability), 1e-8), 0.0, 2.0))
        score = (1.30
                 + 1.20 * (terminal - 0.30)
                 + 1.20 * (concentration - 0.30)
                 + 0.80 * (two_run - 0.50)
                 - 1.50 * (entropy - 0.70)
                 - 0.70 * (switch_density - 0.25)
                 - 0.20 * validity_gradient
                 + 0.85 * (susceptibility - 0.35))
        return 1.0 / (1.0 + np.exp(-1.90 * np.clip(score - 1.30, -30.0, 30.0)))

    def one_subject(frame):
        summaries = []
        records = []
        for _, row in frame.iterrows():
            s = summarize(row['option_a_ratings'], row['option_b_ratings'])
            if s is None:
                continue
            summaries.append(s)

            # Average across the advocated learning-rate range to reduce
            # sensitivity to unknown individual parameter values.
            x = float(np.mean([context_value(summaries, lr) for lr in (0.05, 0.085, 0.12)]))

            # Retain conflict sequences only. The three canonical geometries
            # are +-----, +++-, and +++++- (including their A/B reversals).
            if len(s['canonical']) > 1 and len(set(s['canonical'])) > 1:
                chosen_direction = 1.0 if int(row['response']) == 0 else -1.0
                y = 1.0 if chosen_direction == s['terminal_direction'] else -1.0
                records.append((s['canonical'], x, y))

        if len(records) < 4:
            return 0.0

        residual_x = np.zeros(len(records), dtype=float)
        keys = [r[0] for r in records]
        xvals = np.asarray([r[1] for r in records], dtype=float)
        yvals = np.asarray([r[2] for r in records], dtype=float)

        for key in set(keys):
            loc = np.asarray([k == key for k in keys], dtype=bool)
            residual_x[loc] = xvals[loc] - float(np.mean(xvals[loc]))

        rms = float(np.sqrt(np.mean(residual_x ** 2)))
        if not np.isfinite(rms) or rms < 1e-10:
            return 0.0
        z = residual_x / rms
        return float(np.mean(z * yvals))

    if 'subject_id' in data.columns:
        values = [one_subject(g) for _, g in data.groupby('subject_id', sort=False)]
        return float(np.mean(values)) if values else 0.0
    return one_subject(data)

```

**Observed (real) value:** 0.0311 (var=0.0170)
**Predicted under pi_3:** -0.0111 (var=0.0098)
**Predicted under pi_10:** 0.0122 (var=0.0056)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the full set, pi_3 is the stronger task-invariant account, especially on the experiments that directly test the theories\u2019 distinctive structural claims. In Experiment 1, the observed prefix-coherence score was exactly 0 with zero between-subject variance, matching pi_3\u2019s 0 and decisively contradicting pi_10\u2019s ceiling prediction of 1. In Experiment 2, the observed localized load-window contrast was only 0.0262, much closer to pi_3\u2019s -0.0175 than to pi_10\u2019s 0.2694; the observed variance was also lower than either simulated variance. Thus there is no evidence for pi_10\u2019s proposed special terminal-access window at loads 5\u20136. This conclusion is reinforced by Experiment 7: the observed geometry contrast was essentially zero (0.0033), close to pi_3 (0.0150) but far from pi_10 (0.1788). Experiments 9 and 16 likewise provide no reliable switch-density, chunking, or foregrounding advantage, and Experiment 18 provides little evidence for contextual calibration. Pi_3 also closely captures Experiments 3, 4, 8, and 11. Pi_10 nevertheless performs better on several level-based measures\u2014especially Experiments 10, 12\u201315, and 17\u2014and is somewhat better in Experiments 6 and 18. These results show genuine weaknesses in pi_3: it often underestimates strong terminal control in short or manageable conflicts (Experiments 6, 12, 13, and 15), overstates terminal control in other configurations (Experiments 5 and 10), and predicts too much general count fatigue for unanimous evidence (Experiment 17). Therefore pi_3 is not a complete account. However, pi_10\u2019s successes on absolute choice levels do not rescue its defining categorical mechanisms. Its coherence floor, singleton bonus, fragmentation penalty, and sharply bounded 5\u20136 cue access window jointly make large predictions exactly where the data repeatedly show null effects. Removing those mechanisms would change the theory rather than merely improve its parameterization. The best course is consequently to retain pi_3 as the current benchmark while replacing pi_10 with a stronger, less design-specific competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_10 with a theory of distributed, continuously weighted serial evidence integration. Each attended discriminating cue should contribute signed, compressed log-odds evidence to a leaky or interference-limited decision state, with a smooth serial-position accessibility gradient and divisive normalization by total conflicting support. This gives later cues a genuine but graded retrieval advantage without exact overwrite. Terminal influence should depend continuously on terminal reliability, opposing reliability-weighted coalition strength, and total interference\u2014not on prefix coherence, switch count, singleton status, or a hard-coded 5\u20136 cue window. Load should primarily increase readout noise or interference, while unanimous evidence should show only a small vigilance decline. Stable subject heterogeneity can enter through recency-gradient strength, leak/interference, validity compression, attention, decision gain, and lapse rate, but the mechanism should remain stimulus-local and history-free unless future data establish learning. Such a theory should predict near-zero coherent-versus-fragmented and localized-window contrasts in Experiments 1, 2, and 7; preserve tally modulation in Experiment 3; permit very strong terminal control in Experiments 6, 12, 13, and 15; allow strong opposing coalitions to suppress terminal choices in Experiments 10 and 14; and avoid pi_3\u2019s excessive unanimous-evidence fatigue in Experiment 17. It would therefore compete with pi_3 on its successful recency phenomena while explaining why recency is graded rather than an obligatory overwrite process."
}
```

## Usage

```json
{
  "input_tokens": 22149,
  "output_tokens": 2211
}
```
