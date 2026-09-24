# feedback_iter_04

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace DCD with a 'Softened Validity-Distrust with Count Mixture' (SVD) theory. Core claims: (1) RETAIN VPS's confirmed backbone — every endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity, with NO dampening near v=0.5 and NO trust refund (this is what Exp 1 decisively confirmed: charge d_j = c0 + c1*v_j^kappa, kappa >= 1, c0, c1 > 0). (2) Add a dual-process mixture: on each trial the subject is, with probability w, a 'skeptic' who integrates the distrust difference D = P(B) - P(A) via sigmoid(beta*D), and with probability (1-w) a 'counter' who simply chooses the option with FEWER total endorsements (pure tallying on counts, indifferent at matched counts). This count branch explains the systematic shrinkage of observed proportions toward 0.5 relative to pure VPS: in Exp 2's coin-flip exchange cells it pulls the coin-flip tolerance down from 0.61 toward the observed 0.51 (count-following favors the larger coin-flip coalitions less than VPS's distrust does); in Exp 9's count-conflict cells it pulls the high-validity-option rate up from 0.28 toward the observed 0.38; in Exp 10 it pulls the low-validity-coalition rate down from 0.83 toward 0.78; and in Exp 1's matched-count cells it is exactly neutral (counts equal), so the strong VPS-signed result (0.37) is preserved with w near 0.8-0.9. (3) Allow a larger lapse epsilon (0.10-0.25) and/or between-subject heterogeneity in beta to further soften extremes (Exps 3, 5). Parameter sketch: c0 in [0.05, 0.25], c1 in [0.5, 1.3], kappa in [1.5, 2.5], beta in [1.0, 2.5], w in [0.6, 0.95], epsilon in [0.08, 0.20]. This theory differs from DCD in that its moderation comes from an explicit count-heuristic mixture and noise, not from a trust term that reverses sign near chance validity — the specific mechanism Exp 1 falsified — and it differs from VPS by predicting systematically less extreme choice proportions, which VPS over- or under-shoots in Exps 2, 9, and 10. It should be competitive with (and empirically distinguishable from) VPS on cells where counts and validity-distrust conflict at unequal endorsement counts, while agreeing with VPS on the matched-count validity gradient where VPS was strongly confirmed.


## CANDIDATE THEORY
Softened Validity-Distrust with Count Mixture (SVD). Every expert endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity: d_j = c0 + c1*v_j^kappa, with no dampening near v=0.5 and no trust refund. Decision making is a dual-process mixture at the trial level: with probability w the subject is a 'skeptic' who integrates the accumulated-suspicion difference D = P(B) - P(A) through sigmoid(beta*D); with probability (1-w) the subject is a 'counter' who simply chooses the option with FEWER total endorsements (indifferent at matched counts). A uniform lapse epsilon softens extremes. Between-subject heterogeneity in beta (the sigmoid slope) is a core part of the theory: subjects vary in how steeply they translate suspicion differences into choices, so the pooled population is a mixture of shallow and steep sigmoids — sensitive near D=0 (matched-count validity-composition trials) yet compressed at large |D| (unequal-count trials), which decouples the two assay families that a single global beta could not satisfy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Softened Validity-Distrust with Count Mixture (SVD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Branch 1 (skeptic, prob w): each endorsement is a LIABILITY whose
    # magnitude grows with the endorser's RAW claimed validity:
    #   d_j = c0 + c1 * v_j^kappa   (kappa >= 1.5; NO dampening near
    #                                v = 0.5, NO trust refund)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = P(B) - P(A)  (positive favors A)
    #   P_skeptic(A) = sigmoid(beta * D)
    # Branch 2 (counter, prob 1-w): pure tallying on endorsement COUNTS;
    #   choose the option with FEWER total endorsements; exactly
    #   indifferent (0.5) at matched counts.
    # Mixture: P_core(A) = w * P_skeptic(A) + (1-w) * P_count(A),
    # mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # distrust weights are fixed for the whole block. beta is sampled
    # per subject from a WIDE range: between-subject heterogeneity in
    # the sigmoid slope is an explicit part of the theory.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SVD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v = np.clip(v, 0.5, 1.0)  # defensive; no logs taken, v=1 harmless.

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    w_mix = float(parameters["w"])

    # --- Skeptic branch: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion).
    D = pen_b - pen_a
    x = beta * D
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * x))  # stable sigmoid

    # --- Counter branch: pure endorsement-count tallying ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    if cnt_a < cnt_b:
        p_count_a = 1.0      # A has fewer endorsements -> prefer A
    elif cnt_a > cnt_b:
        p_count_a = 0.0
    else:
        p_count_a = 0.5      # matched counts -> indifferent

    # --- Dual-process mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- c0: [0.05, 0.12]
- c1: [0.95, 1.3]
- kappa: [1.5, 1.8]
- beta: [1.2, 2.6]
- w: [0.80, 0.90]
- epsilon: [0.10, 0.16]
- validities: validities

`rationale`:
This is a single-knob minimal-diff edit of the ACCEPTED iter-3 base, implementing the critic's iter-4 diagnosis exactly: the predict/policy source is re-emitted verbatim and only the parameter box changes. Three iterations of evidence have now established that the matched-count validity-composition assays (Exps 6, 7, 8, 9) and the unequal-count / large-D assays (Exps 1, 2, 3, 10) respond to beta in OPPOSITE directions (sensitivity ratio ~0.6-0.7), so no single narrow beta box can satisfy both: iter-3's effective beta ~1.4 left Exps 6-9 ~0.05 too soft; iter-4's [1.8, 2.3] fixed them (Exp 9 went 0.418 -> 0.3755 vs observed 0.3729, essentially solved) but overshot Exps 1, 3, 10 by 0.05-0.075 and was rejected. The anti-oscillation move is to WIDEN beta to [1.2, 2.6] rather than pick another intermediate point. This is functionally different from a single mid beta: because parameters are sampled per subject, a wide range makes the pooled choice curve a mixture of shallow and steep sigmoids — steep near D=0 where the matched-count trials of Exps 6-9 live (retaining most of the sensitivity that closed them at iter-4's mean beta ~1.9) yet compressed at large |D| where the unequal-count trials of Exps 1, 3, 10 live, since low-beta subjects do not saturate and lag behind the steep subjects' extremes. Back-of-envelope this should hold Exps 6-9 near their iter-4 values while pulling Exp 1 back toward ~0.74-0.76, Exp 3 back toward -0.55 to -0.57, and Exp 10 back toward ~0.52-0.53. It also directly attacks the systematic 3-10x underprediction of between-subject variance on every experiment (e.g. Exp 3 real var 0.042 vs 0.013), which the arbiter explicitly sanctioned via between-subject heterogeneity in beta. Everything else is frozen at the validated iter-3 calibration per the critic's explicit instruction: w at [0.80, 0.90] and epsilon at [0.10, 0.16] exactly (iter-2 proved co-moving these with beta is what broke Exps 5 and 10; epsilon is nearly inert on Exp 10 anyway since its rate sits near 0.5), c1 at [0.95, 1.3] and kappa at [1.5, 1.8] (the charge geometry produces correct signs everywhere and the 0.50-vs-0.60 gap is mathematically capped in kappa). The one additional micro-edit is trimming c0's ceiling from 0.25 to 0.12, per the critic's point (4): the flat floor cancels exactly on matched-count assays so it is inert there, but a smaller floor is mildly correct for Exps 1, 2, 5 and the iter-4 trim was not the guardrail problem. Guardrails respected: if Exp 1 still lands high, the in-family remedy flagged by the critic is a slightly lower w ceiling (~0.88), not more beta reduction; if Exp 3 drifts below -0.58, raise the beta floor rather than lower the ceiling. The architecture, mechanism family, and all function code are untouched — this is a parameter-box-only edit inside the arbiter-prescribed SVD family.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0615 -> ACCEPTED
- iter 2: loss=0.0704 -> REJECTED
- iter 3: loss=0.0530 -> ACCEPTED
- iter 4: loss=0.0545 -> REJECTED
- iter 5 (current candidate you are grading): loss=0.0481 -> ACCEPTED
Running-best (last accepted) base: iter 5 at loss=0.0481.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(choose the Take-The-Best winner) on trials where TTB and Tallying
    make OPPOSITE deterministic predictions (i.e., trials where the two
    options win on different numbers of features, so the tallies disagree).

    High value -> TTB behavior; low value -> Tallying behavior.
    Works on the pooled DataFrame and on a single subject's slice.
    """
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_matrix(col):
        rows = [list(r) for r in col]
        return np.array(rows, dtype=float)

    a = _to_matrix(data["option_a_ratings"])
    b = _to_matrix(data["option_b_ratings"])
    resp = np.asarray(data["response"].to_numpy(), dtype=float)

    n_trials = a.shape[0]

    # --- TTB winner per trial (cue cascade, first discriminating cue) ---
    winners = np.full(n_trials, -1, dtype=int)
    for j in cue_order:
        undecided = winners < 0
        a_better = undecided & (a[:, j] > b[:, j])
        b_better = undecided & (b[:, j] > a[:, j])
        winners[a_better] = 0
        winners[b_better] = 1

    # --- Tallying score per trial (number of strict feature-wise wins) ---
    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Conflict trials: tallies differ (so Tallying has a strict winner that,
    # in this design, is always the option OPPOSITE to the TTB winner),
    # and TTB also has a strict winner.
    conflict = (a_wins != b_wins) & (winners >= 0)

    if not np.any(conflict):
        # No diagnostic trials -> uninformative, return chance.
        return 0.5

    agree = float(np.mean(resp[conflict] == winners[conflict]))
    return agree
```

**Observed (real) value:** 0.7117 (var=0.0205)
**Candidate trajectory (this loop):**
  - iter 1: 0.7453 (var=0.0055) (Δ vs real +0.0336)
  - iter 2: 0.7622 (var=0.0040) (Δ vs real +0.0506)
  - iter 3: 0.7472 (var=0.0033) (Δ vs real +0.0356)
  - iter 4: 0.7867 (var=0.0023) (Δ vs real +0.0750)
  - iter 5 (current): 0.7889 (var=0.0030) (Δ vs real +0.0772)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8444 (var=0.0125)
- pi_2: 0.1203 (var=0.0072)
- pi_3: 0.6869 (var=0.0033)
- pi_4: 0.7792 (var=0.0018)
- pi_5: 0.7272 (var=0.0025)
- pi_6: 0.6844 (var=0.0026)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signature: allegiance to the tally winner on CONFLICT trials
    # (trials where the single top-validity discriminating cue favors
    # the tally LOSER). Tallying predicts P(choose tally winner) > 0.5
    # there (it follows the cue majority against the best cue);
    # Take-The-Best predicts P < 0.5 (it follows the top cue against
    # the majority). We return P(W | conflict) - 0.5, so the sign
    # itself identifies the theory for any noise level beta > 0.
    validities = np.array([0.95, 0.90, 0.85, 0.65, 0.60, 0.50])
    cue_order = np.argsort(-validities, kind="stable")

    a_list = data["option_a_ratings"].tolist()
    b_list = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    conflict_w = []
    for a_raw, b_raw, r in zip(a_list, b_list, resp):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)

        a_w = float(np.sum(a > b))
        b_w = float(np.sum(b > a))
        if a_w == b_w:
            # Tally is tied -> heuristic guesses; uninformative cell.
            continue
        winner = 0 if a_w > b_w else 1

        # First discriminating cue in descending-validity order.
        top_favors = None
        for j in cue_order:
            if a[j] > b[j]:
                top_favors = 0
                break
            if b[j] > a[j]:
                top_favors = 1
                break
        if top_favors is None:
            continue  # no discriminating cue at all

        if top_favors == winner:
            continue  # congruent trial: both heuristics point the same way

        # Conflict trial: tally winner opposes the top cue's favorite.
        conflict_w.append(1.0 if int(r) == winner else 0.0)

    if len(conflict_w) == 0:
        return 0.0
    return float(np.mean(conflict_w)) - 0.5
```

**Observed (real) value:** -0.2562 (var=0.0128)
**Candidate trajectory (this loop):**
  - iter 1: -0.2794 (var=0.0040) (Δ vs real -0.0231)
  - iter 2: -0.2753 (var=0.0039) (Δ vs real -0.0191)
  - iter 3: -0.2759 (var=0.0028) (Δ vs real -0.0197)
  - iter 4: -0.2906 (var=0.0025) (Δ vs real -0.0344)
  - iter 5 (current): -0.2825 (var=0.0035) (Δ vs real -0.0262)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3394 (var=0.0118)
- pi_1: -0.3231 (var=0.0164)
- pi_3: -0.2566 (var=0.0023)
- pi_4: -0.2684 (var=0.0040)
- pi_5: -0.2456 (var=0.0038)
- pi_6: -0.1978 (var=0.0023)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Validity-sensitivity contrast for the cue-2 cascade design.

    For every trial we compute (a) the Take-The-Best winner (first
    discriminating cue in validity order) and (b) the VWEI evidence
    E = sum_j [log(v_j/(1-v_j))]^3 * sign(a_j - b_j), expressed
    relative to the TTB winner (Erel = E * (+1 if TTB winner is A
    else -1)).

    Two trial classes:
      HIGH  : Erel >  3   -> pairs 1-4, where the 90% expert decides
             and VWEI strongly agrees (P(pick TTB winner) ~ 0.85).
      LOW   : Erel < -0.5 -> pairs 7-12 and 15-16, where cue 2 (or 3)
             decides but the weighted evidence of the opposing
             coalition dominates (P(pick TTB winner) ~ 0.35-0.47).

    Metric = P(choose TTB winner | HIGH) - P(choose TTB winner | LOW).

    TTB predicts the SAME choice probability on every non-tied pair
    (its score is binary), so this difference is 0 in expectation.
    VWEI predicts a large positive difference (~ +0.42). The contrast
    cancels each subject's overall response rate (beta / epsilon),
    leaving only the stimulus-driven validity sensitivity.
    """
    v = np.array([0.90, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64])
    lo = np.log(v / (1.0 - v))
    w = lo ** 3.0
    order = np.argsort(-v, kind="stable")

    highs = []
    lows = []

    for a_cell, b_cell, r in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        if a.shape[0] != 9 or b.shape[0] != 9:
            continue

        # VWEI-style evidence for A over B (gamma = 3).
        d = np.sign(a - b)
        E = float(np.dot(w, d))

        # TTB winner: first discriminating cue in validity order.
        win = None
        for j in order:
            if a[j] > b[j]:
                win = 0
                break
            if b[j] > a[j]:
                win = 1
                break
        if win is None:
            continue  # no discriminating cue (does not occur here)

        # Evidence relative to the TTB winner.
        s = 1.0 if win == 0 else -1.0
        Erel = E * s

        r_val = int(r)
        chose_winner = 1.0 if r_val == win else 0.0

        if Erel > 3.0:
            highs.append(chose_winner)
        elif Erel < -0.5:
            lows.append(chose_winner)

    if not highs or not lows:
        return 0.0

    return float(np.mean(highs) - np.mean(lows))
```

**Observed (real) value:** -0.5467 (var=0.0421)
**Candidate trajectory (this loop):**
  - iter 1: -0.5342 (var=0.0218) (Δ vs real +0.0125)
  - iter 2: -0.5883 (var=0.0135) (Δ vs real -0.0417)
  - iter 3: -0.5492 (var=0.0129) (Δ vs real -0.0025)
  - iter 4: -0.5954 (var=0.0101) (Δ vs real -0.0488)
  - iter 5 (current): -0.5571 (var=0.0133) (Δ vs real -0.0104)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0108 (var=0.0045)
- pi_3: 0.4433 (var=0.0115)
- pi_2: 0.5433 (var=0.0293)
- pi_4: -0.4146 (var=0.0116)
- pi_5: -0.5708 (var=0.0133)
- pi_6: -0.4858 (var=0.0140)

### Experiment 4
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Signed-evidence slope.

    For every trial we compute the VWEI evidence
        E = sum_j w_j * sign(a_j - b_j),  w_j = logit(v_j)^3
    and the TTB winner (first discriminating cue in validity order;
    validities are strictly descending so cue order = feature order).

    Define the signed regressor  x = +E if TTB's winner is A, x = -E if
    TTB's winner is B, and the outcome y = 1 iff the subject chose TTB's
    winner.  Return the OLS slope of y on x.

    TTB's probability of choosing its own winner is a single constant
    (softmax(beta) mixed with a lapse) on EVERY trial, so its expected
    slope is exactly 0.  VWEI's probability of choosing TTB's winner is
    sigmoid(beta * x) on every trial, giving a clearly positive slope
    (~0.05 at the design center, sign-stable over the whole parameter
    box).  The two theories are maximally separated on this statistic.
    """
    if data is None or len(data) == 0:
        return 0.0

    validities = np.array([0.74, 0.73, 0.70, 0.69, 0.68,
                           0.67, 0.66, 0.65, 0.64], dtype=float)
    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-6)
    w = np.log(v / (1.0 - v)) ** 3.0

    def _to_vec(cell):
        if isinstance(cell, str):
            return [float(ch) for ch in cell if ch in '01']
        return [float(val) for val in cell]

    a = np.array([_to_vec(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([_to_vec(r) for r in data['option_b_ratings']], dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != len(w):
        return 0.0

    resp = data['response'].to_numpy(dtype=float)

    d = np.sign(a - b)                     # +1 favors A, -1 favors B
    E = d @ w                              # VWEI evidence for A

    disc = d != 0
    keep = disc.any(axis=1)                # drop degenerate all-tie rows
    d, E, resp, disc = d[keep], E[keep], resp[keep], disc[keep]
    if d.shape[0] == 0:
        return 0.0

    # First discriminating cue (validities strictly descending -> order 0..8)
    first_idx = disc.argmax(axis=1)
    rows = np.arange(d.shape[0])
    d_first = d[rows, first_idx]
    winner_a = d_first > 0                  # TTB's winner is option A

    # y = 1 iff subject chose TTB's winner; x = evidence signed toward winner
    y = np.where(winner_a, 1.0 - resp, resp)
    x = np.where(winner_a, E, -E)

    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1e-12:
        return 0.0
    yc = y - y.mean()
    slope = float(np.dot(xc, yc)) / denom
    return slope
```

**Observed (real) value:** -0.1938 (var=0.0027)
**Candidate trajectory (this loop):**
  - iter 1: -0.2062 (var=0.0007) (Δ vs real -0.0124)
  - iter 2: -0.2186 (var=0.0005) (Δ vs real -0.0247)
  - iter 3: -0.2095 (var=0.0005) (Δ vs real -0.0157)
  - iter 4: -0.2159 (var=0.0004) (Δ vs real -0.0221)
  - iter 5 (current): -0.2208 (var=0.0003) (Δ vs real -0.0269)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0475 (var=0.0011)
- pi_1: 0.0002 (var=0.0008)
- pi_2: 0.1895 (var=0.0024)
- pi_4: -0.2020 (var=0.0005)
- pi_5: -0.2142 (var=0.0004)
- pi_6: -0.1987 (var=0.0006)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Top-cue allegiance under endorsement-majority conflict.

    Selected trials: the 95%-validity expert (feature index 0) endorses
    exactly one option, AND that option carries strictly MORE total
    endorsements (sum of ratings) than its rival.  On these trials the
    metric is the empirical rate of choosing the option endorsed by the
    top expert.

    VWEI (any gamma, beta, epsilon) must follow the top cue at ~0.98
    here: the amplified log-odds weight of v=.95 dominates every
    possible opposing coalition of the remaining cues, and 50%-expert
    endorsements contribute exactly zero weight.
    SDI must AVOID the top-cue option here (~0.15-0.25): every
    endorsement is a liability, and the top-cue option is also the
    endorsement-majority option.
    """
    df = data.reset_index(drop=True)

    a = np.array([list(x) for x in df['option_a_ratings']], dtype=float)
    b = np.array([list(x) for x in df['option_b_ratings']], dtype=float)
    resp = df['response'].to_numpy(dtype=float)

    if a.ndim != 2 or a.shape[0] == 0:
        return 0.5

    chose_a = resp < 0.5

    # Which option does the top (95%) expert endorse?
    top_a = a[:, 0] > 0.5
    top_b = b[:, 0] > 0.5
    # Keep only trials where exactly one option gets the top expert's
    # endorsement (guards against degenerate rows).
    xor_top = np.logical_xor(top_a, top_b)

    # Total endorsement counts per option.
    cnt_a = a.sum(axis=1)
    cnt_b = b.sum(axis=1)

    # Is the top-cue-endorsed option also the endorsement-majority
    # option (strictly more endorsements)?
    top_option_more = np.where(top_a, cnt_a > cnt_b, cnt_b > cnt_a)

    sel = xor_top & top_option_more
    if not np.any(sel):
        return 0.5

    # Did the subject choose the option endorsed by the top expert?
    followed = np.where(top_a[sel], chose_a[sel], ~chose_a[sel])
    return float(np.mean(followed))
```

**Observed (real) value:** 0.1143 (var=0.0077)
**Candidate trajectory (this loop):**
  - iter 1: 0.0971 (var=0.0029) (Δ vs real -0.0171)
  - iter 2: 0.0743 (var=0.0020) (Δ vs real -0.0400)
  - iter 3: 0.0876 (var=0.0021) (Δ vs real -0.0267)
  - iter 4: 0.0914 (var=0.0020) (Δ vs real -0.0229)
  - iter 5 (current): 0.0952 (var=0.0022) (Δ vs real -0.0190)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1348 (var=0.0038)
- pi_3: 0.9867 (var=0.0003)
- pi_1: 0.8433 (var=0.0110)
- pi_2: 0.9057 (var=0.0054)
- pi_5: 0.0852 (var=0.0016)
- pi_6: 0.1043 (var=0.0024)

### Experiment 6
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Double-dissociation composite built on the two strongest contrasts
    # engineered into this design:
    #
    # COMPONENT 1 ("big-gap single endorsers", pairs 6/18, 8 trials/subject):
    #   both options carry exactly ONE endorsement; the endorsers' validities
    #   differ by >= 0.25 (0.90 vs 0.60).  VWEI overwhelmingly follows the
    #   higher-validity endorser (P ~ 0.89 across its whole parameter box);
    #   SDI is structurally near-blind at matched endorsement counts
    #   (P ~ 0.50-0.55 across its whole box).
    #
    # COMPONENT 2 ("ridge-crest coalition", pairs 10/22, 8 trials/subject):
    #   one option carries a single 0.80-validity endorsement, the other a
    #   coalition of four 0.75-validity endorsements.  Score +1 for choosing
    #   the COALITION.  VWEI favors the coalition (P ~ 0.62); SDI strongly
    #   favors the single endorser (coalition P ~ 0.10).
    #
    # Metric = mean of the two rates.  Expected: VWEI ~ 0.75, SDI ~ 0.31.
    # Both components use only 8 trials each but their gaps point the SAME
    # way after sign alignment, so the composite has a large mean gap and
    # modest binomial-only between-subject variance.

    V = np.array([0.90, 0.85, 0.80, 0.80, 0.75, 0.75, 0.75, 0.75, 0.65, 0.60])

    a_cells = data['option_a_ratings'].tolist()
    b_cells = data['option_b_ratings'].tolist()
    resp = data['response'].to_numpy()

    scores_gap = []    # 1 = chose the higher-validity single endorser
    scores_coal = []    # 1 = chose the 4x0.75 coalition over the single 0.80 endorser

    for a_raw, b_raw, r in zip(a_cells, b_cells, resp):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape != V.shape or b.shape != V.shape:
            continue
        ca = int(round(a.sum()))
        cb = int(round(b.sum()))
        chose_a = (int(r) == 0)

        if ca == 1 and cb == 1:
            # Single endorsement on each side: who endorsed, and how good
            # is that expert?
            ia = int(np.argmax(a))
            ib = int(np.argmax(b))
            va = V[ia]
            vb = V[ib]
            if abs(va - vb) >= 0.25:
                hi_is_a = (va > vb)
                scores_gap.append(1.0 if (chose_a == hi_is_a) else 0.0)

        elif (ca == 1 and cb == 4) or (ca == 4 and cb == 1):
            # Ridge-crest conflict: verify it is exactly the designed cell
            # (single 0.80 expert vs four 0.75 experts) and score the
            # coalition choice.
            if ca == 1:
                single_vec, coal_vec = a, b
                single_is_a = True
            else:
                single_vec, coal_vec = b, a
                single_is_a = False
            i_single = int(np.argmax(single_vec))
            coal_idx = np.where(coal_vec > 0.5)[0]
            if (
                abs(V[i_single] - 0.80) < 1e-9
                and len(coal_idx) == 4
                and bool(np.all(np.abs(V[coal_idx] - 0.75) < 1e-9))
            ):
                # +1 when the subject chose the coalition side
                scores_coal.append(1.0 if (chose_a != single_is_a) else 0.0)

    r_gap = float(np.mean(scores_gap)) if len(scores_gap) > 0 else 0.5
    r_coal = float(np.mean(scores_coal)) if len(scores_coal) > 0 else 0.5
    return 0.5 * (r_gap + r_coal)
```

**Observed (real) value:** 0.1925 (var=0.0159)
**Candidate trajectory (this loop):**
  - iter 1: 0.2687 (var=0.0108) (Δ vs real +0.0762)
  - iter 2: 0.2038 (var=0.0111) (Δ vs real +0.0113)
  - iter 3: 0.2400 (var=0.0108) (Δ vs real +0.0475)
  - iter 4: 0.1825 (var=0.0062) (Δ vs real -0.0100)
  - iter 5 (current): 0.2112 (var=0.0084) (Δ vs real +0.0187)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7625 (var=0.0125)
- pi_4: 0.3037 (var=0.0089)
- pi_1: 0.4863 (var=0.0079)
- pi_2: 0.6925 (var=0.0122)
- pi_5: 0.1862 (var=0.0087)
- pi_6: 0.2387 (var=0.0095)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Validity-following rate in theory-conflict cells"
    #
    # The 15 experts split into a HIGH-validity group (features 0-6:
    # validities 0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65) and a
    # LOW-validity group (features 7-14: validity 0.60). Every trial in
    # this design pits a "pure high" option against a "pure low"
    # option. Two families of cells separate SDI from VPS:
    #
    # Family 1 (matched endorsement counts, validity gap): SDI's
    #   penalty is a near-flat function of validity (delta ~ 0), so it
    #   is pinned at ~0.51 for the high-validity option (its weak
    #   alpha tie-break even pushes slightly ABOVE 0.5). VPS's
    #   penalty rises steeply in validity, so it systematically picks
    #   the LOW-validity option (~0.28).
    #
    # Family 2 (count conflict where the FEWER-endorsed option is the
    #   high-validity one): SDI follows the endorsement count and
    #   picks the high-validity option (~0.72). VPS's validity-scaled
    #   distrust makes the low-validity majority cheaper, so it picks
    #   the high-validity option only ~0.30 of the time.
    #
    # The metric is the average of P(choose the high-validity-group
    #   option) over these two families. SDI predicts ~0.62; VPS
    #   predicts ~0.29.

    V = np.array([0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65,
                  0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])

    f1_hits = 0.0
    f1_n = 0
    f2_hits = 0.0
    f2_n = 0

    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        m = int(min(a.size, b.size, V.size))
        if m == 0:
            continue
        a = a[:m]
        b = b[:m]

        top = V[:m] >= 0.65  # high-validity expert group
        a_top = float(np.sum(a[top]))
        a_bot = float(np.sum(a[~top]))
        b_top = float(np.sum(b[top]))
        b_bot = float(np.sum(b[~top]))

        # Require each option to be endorsed purely by one group
        # (all trials in this design satisfy this; guards oddities).
        if (a_top > 0 and a_bot > 0) or (b_top > 0 and b_bot > 0):
            continue
        ca = a_top + a_bot
        cb = b_top + b_bot
        if ca == 0 or cb == 0:
            continue

        a_is_top = a_top > 0
        b_is_top = b_top > 0
        if a_is_top == b_is_top:
            continue

        chose_a = (int(round(float(r))) == 0)
        chose_top = chose_a if a_is_top else (not chose_a)

        if ca == cb:
            # Family 1: matched counts, validity gap.
            f1_n += 1
            f2_unused = None
            f1_hits += 1.0 if chose_top else 0.0
        else:
            # Family 2: count conflict; keep only cells where the
            # FEWER-endorsed option is the high-validity one
            # (the cells where SDI and VPS point in opposite
            # directions). Cells where the fewer-endorsed option is
            # the low-validity one are agreed upon by both theories
            # and are excluded.
            fewer_is_top = (a_is_top and ca < cb) or (b_is_top and cb < ca)
            if fewer_is_top:
                f2_n += 1
                f2_hits += 1.0 if chose_top else 0.0

    p1 = f1_hits / f1_n if f1_n > 0 else np.nan
    p2 = f2_hits / f2_n if f2_n > 0 else np.nan
    vals = [p for p in (p1, p2) if not np.isnan(p)]
    if not vals:
        return 0.5
    return float(np.mean(vals))
```

**Observed (real) value:** 0.3815 (var=0.0238)
**Candidate trajectory (this loop):**
  - iter 1: 0.4233 (var=0.0069) (Δ vs real +0.0418)
  - iter 2: 0.3514 (var=0.0047) (Δ vs real -0.0301)
  - iter 3: 0.4274 (var=0.0030) (Δ vs real +0.0459)
  - iter 4: 0.3672 (var=0.0058) (Δ vs real -0.0143)
  - iter 5 (current): 0.3809 (var=0.0038) (Δ vs real -0.0006)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6227 (var=0.0042)
- pi_5: 0.2844 (var=0.0060)
- pi_1: 0.8411 (var=0.0121)
- pi_2: 0.3319 (var=0.0035)
- pi_3: 0.9332 (var=0.0008)
- pi_6: 0.3286 (var=0.0033)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Matched-count validity-reversal index.
    #
    # On trials where A and B carry the SAME number of endorsements (k vs k),
    # the two theories make opposite-sign predictions:
    #   VPS: the coalition of LOWER-validity endorsers is preferred
    #        (a high-validity endorsement is a bigger suspected defect),
    #        with preference strengthening in k.
    #   SDI: at matched count its validity sensitivity has the wrong sign
    #        and is bounded, so it stays at or below chance for the
    #        low-validity coalition on every such trial.
    #
    # The metric is the empirical proportion of choices of the
    # LOW-validity coalition across all matched-count trials
    # (design pairs 1-6 and their A/B mirrors 13-18; 48 of the 96
    # trials per subject). VPS predicts ~0.85; SDI predicts ~0.45.
    # It is well-defined on a single subject's slice (48 Bernoulli
    # trials) and on the pooled data.

    v = np.array([
        0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.85, 0.85, 0.75,
        0.65, 0.65, 0.65, 0.60, 0.60, 0.60, 0.60, 0.65, 0.65, 0.65,
    ], dtype=float)

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape[0] != v.shape[0] or b.shape[0] != v.shape[0]:
            continue

        ca = float(a.sum())
        cb = float(b.sum())
        # Keep only matched-count trials with at least one endorsement
        # on each side (the validity-reversal assay cells).
        if ca == 0.0 or cb == 0.0 or ca != cb:
            continue

        # Mean validity of the endorsing coalition on each side.
        va = float(np.mean(v[a > 0]))
        vb = float(np.mean(v[b > 0]))
        if va == vb:
            continue

        low_is_a = va < vb
        chose_a = (int(row["response"]) == 0)
        hits.append(1.0 if (chose_a == low_is_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7808 (var=0.0201)
**Candidate trajectory (this loop):**
  - iter 1: 0.7050 (var=0.0046) (Δ vs real -0.0758)
  - iter 2: 0.8158 (var=0.0031) (Δ vs real +0.0350)
  - iter 3: 0.7296 (var=0.0075) (Δ vs real -0.0513)
  - iter 4: 0.7967 (var=0.0036) (Δ vs real +0.0158)
  - iter 5 (current): 0.7762 (var=0.0036) (Δ vs real -0.0046)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8263 (var=0.0045)
- pi_4: 0.4600 (var=0.0059)
- pi_1: 0.1496 (var=0.0101)
- pi_2: 0.5108 (var=0.0054)
- pi_3: 0.0317 (var=0.0006)
- pi_6: 0.8054 (var=0.0034)

### Experiment 9
**Design**
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Weighted proportion of choosing the side whose endorsements come from
    # MORE VALID experts, on matched-endorsement-count trials.
    #
    # Design validities (n_features=20, fixed): indices 0-5 are 0.50,
    # indices 6-19 are 0.60. On every trial where both options carry the
    # SAME number of endorsements, the flat count terms (c0) of both
    # DCD and VPS cancel exactly, so the choice is driven purely by the
    # validity composition of the endorsers:
    #   - VPS: penalty strictly increasing in v  -> always avoids the
    #     0.60 experts -> weighted P(hi-validity side) well below 0.5
    #     (mid ~0.38, box <= ~0.45).
    #   - DCD: dampened distrust (2v-1)^kappa collapses near chance and
    #     is opposed by the residual trust term alpha*logit(v) ->
    #     essentially at/above chance (mid ~0.51, box >= ~0.49).
    # Trials are weighted by m = |#0.60-endorsements difference|, which
    # emphasizes the cells (m=2,3,4) where the two theories diverge most.
    V = np.array([0.5] * 6 + [0.6] * 14, dtype=float)
    hi = (V > 0.55).astype(float)  # indicator of the 0.60-validity experts

    if data is None or len(data) == 0:
        return 0.5

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    resp = data['response'].to_numpy()

    num = 0.0
    den = 0.0
    for i in range(len(data)):
        a = np.asarray(a_col[i], dtype=float)
        b = np.asarray(b_col[i], dtype=float)
        if a.shape[0] != V.shape[0] or b.shape[0] != V.shape[0]:
            continue
        tA = float(np.sum(a))
        tB = float(np.sum(b))
        # Only matched-count trials: the flat c0 terms cancel and the
        # choice isolates the validity-composition sensitivity.
        if tA != tB:
            continue
        hA = float(np.dot(hi, a))  # number of 0.60-expert endorsements on A
        hB = float(np.dot(hi, b))
        if hA == hB:
            continue  # pure tie in validity composition
        m = abs(hA - hB)
        r = int(resp[i])
        chose_hi = (r == 0) if (hA > hB) else (r == 1)
        den += m
        if chose_hi:
            num += m

    if den <= 0:
        return 0.5
    return num / den
```

**Observed (real) value:** 0.3729 (var=0.0159)
**Candidate trajectory (this loop):**
  - iter 1: 0.4217 (var=0.0037) (Δ vs real +0.0488)
  - iter 2: 0.4173 (var=0.0062) (Δ vs real +0.0444)
  - iter 3: 0.4180 (var=0.0033) (Δ vs real +0.0451)
  - iter 4: 0.3755 (var=0.0035) (Δ vs real +0.0026)
  - iter 5 (current): 0.3968 (var=0.0034) (Δ vs real +0.0239)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5269 (var=0.0032)
- pi_5: 0.3820 (var=0.0036)
- pi_1: 0.7308 (var=0.0072)
- pi_2: 0.5032 (var=0.0040)
- pi_3: 0.5099 (var=0.0041)
- pi_4: 0.4895 (var=0.0044)

### Experiment 10
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # COIN-FLIP ENDORSEMENT TOLERANCE.
    #
    # For every trial, count how many endorsements each option receives
    # from the five chance-validity (v = 0.50) experts (feature indices
    # 15-19 in this experiment's fixed validity vector). Every one of the
    # 24 unique pairs pits an option carrying coin-flip endorsements
    # against an option carrying none, so the trial is scored:
    #   1.0  if the subject chose the option with MORE 0.50-expert
    #        endorsements (the 'coin-flip tolerant' choice),
    #   0.0  otherwise.
    # The metric is the mean of this score over all trials.
    #
    # This is a direct assay of the one structural difference between the
    # two confidence-distrust theories: the geometry of the per-endorsement
    # distrust charge. VPS charges c0 + c1*v^kappa, so a 0.50 endorsement
    # still costs a substantial c1*0.25-0.33 distrust units and the
    # internal exchange rate between mid-validity (0.75) endorsements and
    # coin-flips is ~1.9 coin-flips per 0.75. DCD charges
    # c0 + c1*(2v-1)^kappa, which collapses to the bare floor c0 at
    # v = 0.50 (and refunds part of the 0.75 charge via the residual trust
    # term), yielding an exchange rate of only ~1.3. Hence VPS subjects
    # are systematically MORE willing to accept coin-flip endorsements,
    # across every cell family in the design:
    #   composition cells (mixed {v_hi, 0.50} vs homogeneous mid): VPS
    #     picks the mixed side (0.53-0.59), DCD avoids it (0.33-0.45);
    #   exchange cells (0.75-coalition vs k coin-flips): VPS keeps
    #     choosing the coin-flip side through k=3-4, DCD abandons it;
    #   calibration cells ({0.90} vs {0.50}, {0.50,0.50} vs {0.90,0.90}):
    #     VPS 0.73/0.86 vs DCD 0.65/0.76.
    # Mid-parameter predictions: VPS ~0.59, DCD ~0.45 (disjoint across
    # their full parameter boxes), with per-subject SD ~0.05-0.06 from
    # 96 Bernoulli trials.

    validities = np.array([
        0.92, 0.90, 0.90, 0.90, 0.85, 0.77, 0.77,
        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.72, 0.72,
        0.50, 0.50, 0.50, 0.50, 0.50,
    ])
    coin_mask = np.isclose(validities, 0.5)

    responses = data["response"].to_numpy()
    ratings_a = data["option_a_ratings"].tolist()
    ratings_b = data["option_b_ratings"].tolist()

    scores = []
    for i in range(len(ratings_a)):
        a = np.asarray(ratings_a[i], dtype=float).ravel()
        b = np.asarray(ratings_b[i], dtype=float).ravel()
        n = min(a.shape[0], b.shape[0], validities.shape[0])
        mask = coin_mask[:n]
        ca = float(np.sum(a[:n][mask]))
        cb = float(np.sum(b[:n][mask]))
        if ca == cb:
            # No coin-flip asymmetry on this trial; skip defensively.
            continue
        chose_a = (int(responses[i]) == 0)
        c_chosen = ca if chose_a else cb
        c_other = cb if chose_a else ca
        scores.append(1.0 if c_chosen > c_other else 0.0)

    if len(scores) == 0:
        return 0.5
    return float(np.mean(scores))
```

**Observed (real) value:** 0.5096 (var=0.0121)
**Candidate trajectory (this loop):**
  - iter 1: 0.4994 (var=0.0028) (Δ vs real -0.0102)
  - iter 2: 0.5629 (var=0.0030) (Δ vs real +0.0533)
  - iter 3: 0.5181 (var=0.0027) (Δ vs real +0.0085)
  - iter 4: 0.5560 (var=0.0032) (Δ vs real +0.0465)
  - iter 5 (current): 0.5402 (var=0.0021) (Δ vs real +0.0306)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6071 (var=0.0030)
- pi_6: 0.4429 (var=0.0037)
- pi_1: 0.4433 (var=0.0012)
- pi_2: 0.6129 (var=0.0030)
- pi_3: 0.5190 (var=0.0010)
- pi_4: 0.3810 (var=0.0022)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The SVD candidate is a faithful implementation of the arbiter's prescribed mechanism (raw-validity distrust backbone d_j = c0 + c1*v_j^kappa, no dampening, no trust refund, plus a count-heuristic mixture and lapse), and it is already the best-scoring theory on most assays: it essentially nails Exps 3 (-0.534 vs -0.547), 4 (-0.206 vs -0.194), 5 (0.097 vs 0.114), and 10 (0.499 vs 0.510 — the coin-flip tolerance the mixture was designed to fix), and is close on Exps 1, 2. However, a systematic residual remains: on the assays that isolate the validity-distrust signal, the candidate is consistently TOO SOFT (shrunk too far toward 0.5): Exp 8 matched-count validity reversal 0.705 vs observed 0.781 (undershoot); Exp 9 matched-count hi-validity avoidance 0.422 vs observed 0.373 (overshoot toward 0.5); Exp 6 double-dissociation composite 0.269 vs observed 0.193 (again shrunk toward the 0.25 midpoint); Exp 7 composite 0.423 vs observed 0.382. All four misfits point the SAME direction: the skeptic branch is under-weighted relative to the neutral-at-matched-counts counter branch and the lapse. This is a single-knob calibration issue inside the prescribed family, not a mechanism failure: raising w (and/or beta, and lowering epsilon) simultaneously (i) pulls Exp 8 up toward 0.78, (ii) pulls Exp 9 down toward 0.37, (iii) pulls Exp 6 down toward 0.19, (iv) pulls Exp 7 down toward 0.38, and (v) moves Exp 1 from 0.745 toward the observed 0.712 (VPS-like pi_6 sits at 0.684). The only counter-indication is Exp 2, where the candidate is already slightly too negative (-0.279 vs -0.256) and a stronger skeptic weight would push marginally further negative — but that cost (~0.02) is small relative to the gains available (~0.05-0.08 across four experiments).
Rationale: The candidate was accepted by the gate (loss 0.0615, running best), and the mechanism family is correct per the arbiter's prescription — do NOT change the architecture. The residual misfit is a calibration problem: the mixture softening over-shrinks the validity-distrust signal. Concretely, on every matched-count assay (Exps 6, 8, 9) the counter branch contributes exactly 0.5 and the lapse adds uniform noise, so the effective signal is w*(1-epsilon)*sigmoid(beta*D); the fitted effective w is too low (or epsilon too high / beta too low). Recommended in-family adjustments, all within or at the edge of the arbiter's parameter sketch: (1) push w toward the top of its range (0.90-0.95; the arbiter itself suggested 'w near 0.8-0.9', and the data now argue for the upper end — consider extending the range ceiling slightly to ~0.97); (2) set epsilon near its floor (0.08) rather than mid-range, since the count mixture already supplies much of the needed moderation and the additional lapse is double-counting softness; (3) optionally raise beta modestly (upper half of [1.0, 2.5]) to sharpen the skeptic's sigmoid on the matched-count validity-gradient cells. Keep c0/c1/kappa essentially unchanged — the raw-validity charge geometry is producing the right signs everywhere. Watch Exp 2 as the one guardrail: if the stronger skeptic weight drives the conflict-cell allegiance below about -0.30, back off w slightly rather than touching the distrust exponents.

**Outcome of your advice:** iter 1 candidate loss=0.0615 -> iter 2 candidate loss=0.0704 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 candidate implemented exactly the calibration shift I recommended (w raised to 0.88-0.97, epsilon floored at 0.08-0.12, beta shifted to 1.7-2.5), and the gate REJECTED it (loss 0.0704 vs 0.0615). The per-experiment decomposition shows why: my advice was directionally right on four assays but the step was too blunt. It helped Exp 6 (0.269→0.204 vs observed 0.193, now nearly perfect), Exp 7 (0.423→0.351 vs 0.382, now close), and Exp 8 (0.705→0.816 vs 0.781 — but OVERSHOT past the target by +0.035). Meanwhile it degraded five previously well-fit assays: Exp 1 (0.745→0.762 vs 0.712), Exp 3 (-0.534→-0.588 vs -0.547, overshot), Exp 4 (-0.206→-0.219 vs -0.194), Exp 5 (0.097→0.074 vs 0.114), and most damagingly Exp 10 (0.499→0.563 vs 0.510 — the lapse reduction removed exactly the softening-toward-0.5 that the coin-flip tolerance assay needs). Critically, Exp 9 — one of the four assays the shift was supposed to fix — barely moved at all (0.422→0.417 vs 0.373). That is the diagnostic key: the Exp 9 residual is NOT a mixture-weight problem. Its validities are only 0.50 vs 0.60, so the skeptic's D on those trials is small no matter how large w or beta gets; the sigmoid is already in its near-linear regime there, and multiplying a near-0.5 probability by a bigger w cannot move it. The Exp 9 gap requires a STEEPER validity gradient in the charge itself (more c1, and/or lower kappa so low-validity differences are less compressed relative to the c0 floor), not a stronger mixture weight.
Rationale: My iter-1 advice ('raise w, floor epsilon, raise beta') was rejected by the gate — I will not repeat that same push. The lesson from the loss decomposition is that the misfits split into two groups with opposite responses to a global re-weighting, so the next edit must decouple them. The base reverts to iter-1 (the gate handles this), and the recommended in-family adjustment is: (1) REVERT w, epsilon, beta to roughly the iter-1 / arbiter-original ranges (w ~0.80-0.90 — note Exp 8 overshot at w≈0.9+, so the optimum sits BETWEEN the two iterations' effective values, closer to 0.85; epsilon back to ~0.10-0.16, since the floor broke Exp 10 and Exp 5; beta back toward 1.0-1.8). (2) Instead, attack the residual through the CHARGE GEOMETRY, which iter-2 left untouched: raise c1 toward the upper half of [0.5, 1.3] and/or lower kappa toward its 1.5 floor. This steepens the per-endorsement validity gradient — which is what Exp 9 (0.50-vs-0.60 composition sensitivity, stuck at +0.044 regardless of w) and the Exp 8 undershoot at iter-1 actually need — while keeping the mixture soft enough to preserve the excellent iter-1 fits on Exps 3, 5, and 10. (3) Keep c0 unchanged; the flat floor was producing correct signs everywhere and a bigger c0 would only re-flatten the low-validity contrast that Exp 9 needs. Guardrails to watch: Exp 2 (iter-1 already -0.279 vs -0.256 — a steeper c1 makes the skeptic slightly more TTB-negative there, so if it drifts below -0.30, trade off by nudging beta down rather than c1), and Exp 1 (iter-1 +0.034 high; the c1 increase should pull it toward 0.71 since distrust differences sharpen). This is a targeted, single-region edit within the prescribed SVD family: distrust-charge slope up, mixture/noise back to the calibrated middle.

**Outcome of your advice:** iter 2 candidate loss=0.0704 -> iter 3 candidate loss=0.0530 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The iter-3 candidate is a faithful, minimal-diff implementation of my iter-2 advice inside the arbiter-prescribed SVD family: the architecture is untouched (raw-validity distrust charge d_j = c0 + c1*v_j^kappa, count-mixture, lapse) and only the charge-geometry box was recalibrated. The gate ACCEPTED it (loss 0.0530, new running best), validating the diagnosis that the residual is a charge-slope problem, not a mixture-weight problem. Per-experiment status: Exps 3 (-0.549 vs -0.547), 10 (0.518 vs 0.510), 4 (-0.210 vs -0.194), and 2 (-0.276 vs -0.256) are essentially solved. However, a coherent residual cluster remains on the four validity-distrust assays, all pointing the SAME direction (candidate too soft, shrunk toward 0.5): Exp 8 matched-count validity reversal 0.730 vs 0.781 (-0.051); Exp 9 hi-validity avoidance 0.418 vs 0.373 (+0.045); Exp 7 composite 0.427 vs 0.382 (+0.046); Exp 6 composite 0.240 vs 0.193 (+0.048). Exp 1 remains +0.036 high. Critically, the recalibration barely moved Exp 9 (0.417 -> 0.418), and the math shows why: the per-endorsement 0.50-vs-0.60 charge gap, c1*(0.6^kappa - 0.5^kappa), is maximized at ~0.11*c1 for kappa anywhere in [1, 2.5] — it is nearly flat in kappa, so no c1/kappa combination inside the box can widen it more than ~15-20%. The gap between 0.418 and 0.373 requires roughly a 2x larger effective beta*D on those trials, and the only remaining in-family lever that scales D-sensitivity without touching the mixture softening (which Exps 3, 5, 10 need) is beta itself. Iter-2 confounded beta with w and epsilon and paid for it on Exps 1, 3, 5, 10; beta has never been isolated.
Rationale: The candidate was ACCEPTED (loss 0.0530, running best), so this is a keep-tuning regenerate, not a rejection of the mechanism — the SVD family is correct and should NOT be redesigned. The single remaining failure mode is a coherent under-sensitivity of the skeptic branch on validity-composition assays (Exps 6, 7, 8, 9, all ~0.05 off in the same direction), and the next edit should isolate the one knob not yet tested in isolation. Recommended in-family adjustment: (1) RAISE beta to roughly [1.8, 2.4] — the top of the arbiter's own [1.0, 2.5] sketch — while HOLDING w at [0.80, 0.90] and epsilon at [0.10, 0.16] exactly where they are. Rationale: on every matched-count assay the effective signal is w*(1-epsilon)*sigmoid(beta*D); iter-3 proved w and epsilon are correctly calibrated (Exps 3, 5, 10 all landed), and the c1/kappa recalibration proved the charge gap on Exp 9's 0.50-vs-0.60 cells is mathematically capped (~0.11*c1 regardless of kappa in [1,2.5]), so beta must carry the remaining factor of ~2 in sensitivity. Back-of-envelope: at c1~1.15, m-weighted D on Exp 9 is ~-0.3, and beta~2.2 yields a pooled rate near 0.375, essentially closing the Exp 9 gap while also pushing Exp 8 up toward 0.78, Exp 7 down toward 0.38, and Exp 6 down toward 0.19. (2) Modestly raise c1 to [1.05, 1.30] (upper half of the sketch) to reinforce the same direction; leave kappa in [1.5, 1.8] and c0 unchanged — the flat floor produces correct signs everywhere and lowering kappa buys almost nothing on the 0.5-0.6 gap. (3) Do NOT repeat the iter-2 mistake of co-moving w/epsilon with beta: the iter-2 rejections on Exps 1, 3, 5, 10 were driven by w~0.93 and the epsilon floor, not by beta per se. Guardrails to watch: Exp 3 (currently perfect at -0.549; if beta~2.2 pushes it below -0.58, cap beta at ~2.0), Exp 5 (currently -0.027 low; stronger distrust worsens it — accept up to ~-0.05 as the traded cost), and Exp 2 (already -0.020 too negative; if it drifts below -0.30, back beta off rather than touching the charge). This is a single-knob edit within the prescribed family, decoupled from the mixture/noise calibration that is now demonstrably correct.

**Outcome of your advice:** iter 3 candidate loss=0.0530 -> iter 4 candidate loss=0.0545 -> the gate marked it REJECTED.

### Iteration 4 (most recent)
Verdict: regenerate
Interpretation: The iter-4 candidate implemented my iter-3 advice faithfully (beta raised in isolation to [1.8, 2.3], c1 to [1.05, 1.30], c0 trimmed to [0.05, 0.10], w/epsilon held), and the per-experiment decomposition shows the diagnosis was directionally correct but the step overshot. On the four target assays the beta raise worked almost perfectly: Exp 9 went 0.418 → 0.3755 (observed 0.3729, now +0.003 — essentially solved after three stalled iterations), Exp 8 went 0.730 → 0.797 (observed 0.781, +0.016), Exp 6 went 0.240 → 0.1825 (observed 0.1925, −0.010), Exp 7 went 0.427 → 0.367 (observed 0.382, −0.014). But the guardrails I flagged broke exactly as warned, and the proposer's c0-trim compensation did NOT protect them: Exp 1 rose 0.747 → 0.787 (observed 0.712, error doubled to +0.075), Exp 3 went −0.549 → −0.595 (observed −0.547, past my stated −0.58 cap), Exp 10 went 0.518 → 0.556 (observed 0.510), Exp 2 drifted to −0.291 (observed −0.256). The gate rejected (0.0545 vs 0.0530), so the base reverts to iter-3. The structural lesson is now crisp from three iterations of evidence: the matched-count validity-composition assays (6, 7, 8, 9) and the unequal-count / large-D assays (1, 2, 3, 10) respond to beta in OPPOSITE directions, with sensitivity ratio ~0.6–0.7 — a single global beta cannot satisfy both, and epsilon is nearly useless as a counterweight on Exp 10 (its rate sits near 0.5, where the lapse has almost no leverage) while it simultaneously undoes the 6–9 gains. A second systematic signal the loop has never exploited: the candidate underpredicts between-subject variance on EVERY experiment by 3–10x (e.g. Exp 3 real var 0.042 vs 0.010; Exp 1 real 0.0205 vs 0.0023), and the arbiter explicitly sanctioned between-subject heterogeneity in beta as a softening mechanism. A wide beta range is not merely a compromise between iter-3 and iter-4 — it is functionally different from a single intermediate beta: a mixture of shallow and steep subjects preserves the small-D (matched-count) sensitivity at the mean beta while compressing the large-D extremes (low-beta subjects lag behind the steep subjects' saturation), which is precisely the decoupling the residual pattern demands, and it simultaneously raises the simulated variances toward the observed ones.
Rationale: Do NOT change the SVD architecture — it is the arbiter-prescribed family and is now demonstrably correct in sign on all ten assays. The next edit is a single-region recalibration on top of the unchanged iter-3 base: (1) WIDEN beta to approximately [1.2, 2.6] rather than choosing another narrow box. This is the anti-oscillation move: iter-3's effective beta (~1.4 mid) left Exps 6–9 ~0.05 too soft; iter-4's [1.8, 2.3] fixed them but overshot Exps 1, 3, 10 by 0.05–0.075. A wide range keeps the mean near ~1.8–1.9 (retaining most of the matched-count sensitivity that closed Exps 6–9) while the pooled choice curve becomes a mixture of sigmoids — steep near D=0 (where matched-count trials live) but compressed at large |D| (where the unequal-count trials of Exps 1, 3, 10 live, since low-beta subjects do not saturate). Back-of-envelope: this should hold Exps 6–9 near their iter-4 values while pulling Exp 1 back toward ~0.74–0.76, Exp 3 back toward −0.55 to −0.57, and Exp 10 back toward ~0.52–0.53. It also directly attacks the systematic 3–10x underprediction of between-subject variance, which the loss has never rewarded the candidate for. (2) HOLD w at [0.80, 0.90] and epsilon at [0.10, 0.16] exactly — iter-2 proved co-moving these with beta is what broke Exps 5 and 10, and iter-3/4 confirmed they are correctly calibrated. Do not floor epsilon again: it is nearly inert on Exp 10 (rate ≈ 0.5) and it erodes the 6–9 gains at almost the same rate beta delivers them. (3) Keep c1 at [1.05, 1.30] and kappa at [1.5, 1.8] — the c1 raise was not the guardrail problem and it reinforces the matched-count gradient; the 0.50-vs-0.60 charge gap remains mathematically capped in kappa, so do not spend effort there. (4) Keep c0 small, [0.05, 0.10]–[0.05, 0.12]: the iter-4 trim did not rescue the guardrails (beta dominated), but a small floor is still mildly correct for Exps 1, 2, 5 and it cancels exactly on the matched-count assays. Guardrails to watch: if Exp 1 still lands above ~0.76, the residual is the count branch contributing a deterministic 1.0 on those conflict trials — the in-family remedy is a slightly lower w ceiling (~0.88), NOT more beta reduction, since w uniquely lowers Exp 1 without lowering matched-count sensitivity (its only serious collateral is Exp 10, so touch it last and only ~0.02–0.03 at a time). If Exp 3 drifts below −0.58, raise the beta range floor rather than lowering the ceiling. Summary: one knob, widened not raised — beta heterogeneous across subjects, everything else frozen at the validated iter-3 calibration.

**Outcome of your advice:** iter 4 candidate loss=0.0545 -> CURRENT candidate loss=0.0481 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The iter-5 candidate implemented my iter-4 advice faithfully (beta widened to [1.2, 2.6] as a between-subject mixture, everything else frozen), and the gate ACCEPTED it at loss 0.0481 \u2014 a new running best and a validation of the wide-beta heterogeneity diagnosis. The decoupling worked almost exactly as predicted on the matched-count assays: Exp 7 is now essentially perfect (0.3809 vs 0.3815), Exp 8 essentially perfect (0.7762 vs 0.7808), Exp 6 close (+0.019), Exp 9 acceptable (+0.024), and the Exp 3 guardrail held (-0.557 vs -0.547, back from iter-4's -0.595 overshoot). Exps 2, 5 remain within ~0.02-0.03. However, the wide-beta mixture did NOT pull Exp 1 back as predicted: it landed at 0.7889 (observed 0.7117, error +0.077) \u2014 now the single dominant residual, and the only assay that got worse. Exp 10 also remains +0.031 high, and Exp 4 drifted to -0.027. Two systematic signals remain: (a) every large-D / unequal-count assay (1, 10, 4, 2) sits too extreme in the distrust direction, while every matched-count assay is now well calibrated; (b) between-subject variance is still underpredicted 3-10x on every experiment (Exp 1: 0.0030 vs 0.0205; Exp 3: 0.0133 vs 0.0421). The math on Exp 1 explains why beta heterogeneity could not fix it: on its TTB-vs-Tally conflict trials the counter branch is deterministic (the TTB winner always carries FEWER endorsements, so p_count = 1.0), and the skeptic branch is saturated at large |D| for most of the beta range \u2014 so the pooled rate is pinned near (1-eps)*(w*~0.95 + (1-w)*1) \u2248 0.79 at eps~0.13. The only remaining in-family lever that moves it is epsilon, which the arbiter's sketch explicitly allows up to 0.25. Critically, I must correct an error in my own iter-4 guardrail: I said 'if Exp 1 still lands high, lower the w ceiling' \u2014 that advice had the wrong SIGN, since lowering w shifts weight onto the counter branch's deterministic 1.0 on exactly those trials, raising Exp 1 further. Raising w (if anything) and raising epsilon are the correct directions.",
  "verdict": "regenerate",
  "rationale": "Do NOT change the SVD architecture \u2014 it is the arbiter-prescribed family, it is sign-correct on all ten assays, and the wide-beta heterogeneity just paid off with the best loss of the loop. The next edit is a single-region recalibration on top of this accepted base. (1) RAISE epsilon to approximately [0.14, 0.22] \u2014 within the arbiter's own 0.10-0.25 sketch \u2014 and consider widening its between-subject spread (some subjects near 0.10, some near 0.24). Rationale: epsilon is the only knob with strong leverage on the saturated large-D assays where the residuals now live (Exp 1 +0.077, Exp 10 +0.031, Exp 4 -0.027, Exp 2 -0.026 all soften toward their targets), while its cost on the matched-count assays (6, 7, 8, 9, all within 0.02 of target) is only ~0.01-0.02 per 0.05 of epsilon because their rates sit closer to the mid-range where the lapse has less pull. Back-of-envelope: eps mean ~0.18 pulls Exp 1 from 0.789 toward ~0.74-0.75 and Exp 10 toward ~0.52, at a cost of roughly +0.01 error on Exps 8/9. Wide per-subject epsilon also directly attacks the 3-10x variance underprediction, which a narrow box cannot. (2) HOLD the wide beta box [1.2, 2.6] exactly \u2014 it just closed Exps 7 and 8 and must not be re-narrowed. (3) HOLD c0, c1, kappa at the current calibration \u2014 the charge geometry is producing correct signs and near-correct magnitudes on every validity-composition assay. (4) On w: do NOT lower it despite my iter-4 guardrail note \u2014 that recommendation was sign-wrong (the counter branch votes deterministically FOR the TTB winner on Exp 1's conflict trials, so lowering w raises Exp 1). If a second knob is touched at all, RAISE the w ceiling modestly (to ~0.92-0.93): this slightly lowers Exp 1 and Exp 10 while sharpening the matched-count assays (Exp 8 up toward 0.78, Exp 9 down toward 0.37), partially offsetting the epsilon costs \u2014 but move it at most 0.02-0.03 beyond the current 0.90 ceiling, and watch Exp 2 (if the conflict-cell allegiance drifts below -0.30, back w off rather than epsilon). Guardrails: if Exp 8 falls below ~0.75 or Exp 6 rises above ~0.24, the epsilon raise went too far \u2014 retreat its ceiling to ~0.19 rather than touching beta. Summary: one primary knob (epsilon up / widened), one optional micro-knob (w ceiling up), everything else frozen on the accepted iter-5 base."
}
```

## Usage

```json
{
  "input_tokens": 39294,
  "output_tokens": 5214,
  "reasoning_tokens": 3949,
  "provider": "Sail Research"
}
```
