# feedback_iter_00

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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_8`
- The recommendation below acts on THEORY 2 (= `pi_8`).

Replace pi_8 with a 'Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak' theory, constructed as a synthesis of the surviving parts of both competitors:

1. CORE (retained from both): one-reason decision making; population-central hierarchy anti-validity (consult ascending subjective weight, first strictly discriminating cue decides); exact validity ties anchor the order; history ignored; no discriminating cue gives exactly 0.5.

2. STRONG INTERIOR-GATED LEAK (pi_8's mechanism, uncapped): when the anti-validity cascade's first discriminating expert on a trial carries a strictly interior validity (min + 0.10 < val < max - 0.10), mix toward the standard TTB (descending-validity) decider with probability q = rho_leak * reversal_coherence * mean_gap, with rho_leak ranging up to ~3.0 and NO 0.18 ceiling — target q in [0.35, 0.50] on high-coherence, large-gap designs (this reproduces Exp 1's observed dip of -0.196 at p_f ~ 0.72, and amplifies the negatives in Exp 9 toward -0.31 and Exp 10 toward -0.29). Weakest-carrier and near-top-carrier trials keep the leak OFF, preserving Exp 1's conditionality triad and Exp 5's strong reversed-cascade following (0.727).

3. DIFFUSE HETEROGENEITY (pi_7's machinery, widened): per-subject stable distortion sigma_h*zeta plus per-trial momentary kappa*xi, both gated by cubed binding ambiguity A^3 and applied only when all validities are distinct; positional-salience tilt gamma drawn asymmetrically (roughly [-0.05, +0.12]). Critically, WIDEN the response-discipline ranges beyond even pi_7's: beta in [0.4, 2.2], epsilon in [0.06, 0.28], so the between-subject follow-protection tail reaches ~0.16 (Exp 2 observed 0.1556; pi_7's 0.112 and pi_8's 0.075 both undershoot) and the pooled determinism drops enough to fix Exp 6's overprediction (0.78 observed vs 0.91/0.96).

4. POSITIONAL-PRIMACY MINORITY (new): a population mixture weight of roughly 0.15-0.25 on a pure earliest-listed-expert-first consultation order (with the remainder on the leak-augmented anti-validity core), which lowers the anti-misbound decider follow toward Exp 15's observed 0.15 (both current theories overpredict at 0.28/0.34) while keeping Exp 17's count-vs-primacy contrast near its observed +0.15 (the mixture's average first-cue distribution must be checked against this constraint during fitting).

5. Drop pi_8's 'frozen discipline' and 'distinct-design bonus' machinery entirely (falsified by Exp 2), and drop pi_7's flatness claim (falsified by Exp 1). All other conventions (lexsort tie-breaking with free per-subject tie_break ~80% early position, softmax(beta) noise plus lapse epsilon, no integration across cues) carry over unchanged. The new theory's free parameters are rho_leak, the primacy mixture weight, gamma, sigma_h, kappa, beta, epsilon, tie_break, and per-subject zeta.


## CANDIDATE THEORY
Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak. People are one-reason decision makers: experts are consulted one at a time and the first strictly discriminating expert decides; there is no integration across cues, no history use, exact validity ties anchor the consultation order, and a trial with no discriminating expert yields exactly 0.5. The population-central hierarchy is anti-validity (distrust-the-weakest: consult ascending advertised validity). Three graded, design-scaled mechanisms modulate this core. (1) STRONG INTERIOR-GATED VALIDITY LEAK: when the anti-validity cascade's first discriminating expert on a trial carries a strictly interior validity (min + 0.10 < val < max - 0.10), the decision mixes toward the standard TTB (descending-validity) decider with probability q = rho_leak * reversal_coherence * mean_gap, with NO substantive ceiling (only a numerical validity clamp at 0.75), targeting q in [0.35, 0.50] on high-coherence, large-gap designs. Weakest-expert and near-top-expert carriers switch the leak off, so anchored designs with extreme carriers keep their pure anti-validity signature. (2) DIFFUSE HETEROGENEITY: each subject's weights are w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), where salience is a positional gradient favoring early-listed experts with gamma drawn asymmetrically from [-0.05, +0.12], A = 1 - |Spearman(position, validity)| is binding ambiguity entering CUBED, zeta is a small stable per-subject distortion, and xi is a large momentary per-trial re-draw; all distortion applies only when all advertised validities are distinct (exact ties anchor the distrust order). (3) POSITIONAL-PRIMACY MINORITY: an ambiguity-gated population mixture weight (roughly 0.15-0.20 at maximal ambiguity, scaling linearly down to zero on monotone validity lists) on a pure earliest-listed-expert-first consultation order, so that when the validity-to-position binding is unambiguous the theory collapses onto the leak-augmented anti-validity core (preserving strong reversed-cascade following on monotone designs), while ambiguous bindings let a minority of consultations fall back to reading order. Response discipline is deliberately wide and heterogeneous: softmax(beta) over the binary winner score plus an independent lapse epsilon, with beta in [0.4, 2.2] and epsilon in [0.06, 0.28], giving per-subject follow probabilities that spread far enough to reproduce the between-subject follow-protection tail. Validity ties inside the hierarchy are broken by a free per-subject tie-break (~80% early position).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Diffuse Mixture Frugality with Strong Interior-Gated Validity Leak.
    #
    # One-reason decision making: experts consulted one at a time, first
    # strictly discriminating expert decides; no integration across cues.
    # Population-central hierarchy: ANTI-VALIDITY (ascending advertised
    # validity; distrust the weakest expert).
    #
    # Mechanisms:
    #   (1) STRONG INTERIOR-GATED VALIDITY LEAK: when the anti cascade's
    #       first discriminating expert j* carries a strictly interior
    #       validity (min+0.10 < val[j*] < max-0.10), mix toward the TTB
    #       (descending-validity) decider with probability
    #       q = rho_leak * reversal_coherence * mean_gap,
    #       with NO substantive ceiling (only a numerical validity clamp
    #       at 0.75 so mixture weights stay proper probabilities).
    #   (2) DIFFUSE HETEROGENEITY: w_j = -val_j + gamma*salience_j
    #       + A^3*(sigma_h*zeta_j + kappa*xi_j), applied only when all
    #       validities are distinct; zeta stable per subject, xi re-drawn
    #       every trial.
    #   (3) POSITIONAL-PRIMACY MINORITY: ambiguity-gated mixture weight
    #       w_eff = primacy_weight * A on a pure earliest-listed-first
    #       consultation order. On monotone validity lists (A ~ 0) the
    #       theory collapses onto the leak-augmented anti-validity core,
    #       preserving strong reversed-cascade following on anchored
    #       designs; ambiguous bindings elicit the reading-order fallback.
    #
    # Noise: softmax(beta) over the binary winner score plus independent
    # lapse epsilon, with deliberately WIDE ranges (beta in [0.4, 2.2],
    # epsilon in [0.06, 0.28]) so the between-subject follow-protection
    # tail is reproduced. No discriminating cue -> exactly 0.5. History
    # is ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    sigma_h = float(parameters["sigma_h"])
    kappa = float(parameters["kappa"])
    rho_leak = float(parameters["rho_leak"])
    primacy_weight = float(parameters["primacy_weight"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        raise ValueError(
            f"zeta length {zeta.shape[0]} != n_features {n_features}."
        )

    n_opts = 2
    a, b = stim[0], stim[1]

    if not np.any(a != b):
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- positional-salience gradient (early-listed experts favored) ----
    if n_features > 1:
        sal = ((n_features - 1 - np.arange(n_features, dtype=float))
               / (n_features - 1.0)) - 0.5
    else:
        sal = np.zeros(1, dtype=float)

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- reversal coherence: |Spearman(validity, validity[::-1])| ----
    if n_features > 1:
        ranks_v = avg_ranks(val)
        ranks_r = avg_ranks(val[::-1])
        dv = ranks_v - ranks_v.mean()
        dr = ranks_r - ranks_r.mean()
        denom2 = np.sqrt(float((dv * dv).sum()) * float((dr * dr).sum()))
        if denom2 > 1e-12:
            rho_rev = float((dv * dr).sum()) / denom2
        else:
            rho_rev = 0.0
        coherence = abs(rho_rev)
    else:
        coherence = 0.0

    # ---- mean adjacent gap over UNIQUE validities ----
    uvals = np.unique(val)
    distinct = (uvals.size == n_features)
    if uvals.size > 1:
        mean_gap = float(np.mean(np.diff(uvals)))
    else:
        mean_gap = 0.0

    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)

    # ---- anti-validity weights with diffuse heterogeneity ----
    # Exact validity ties anchor the distrust order (no distortion);
    # all-distinct lists degrade with the CUBE of binding ambiguity.
    # Small stable part (sigma_h * zeta) + large momentary part
    # (kappa * xi, re-drawn fresh on every trial).
    w = -val + gamma * sal
    if distinct and ambiguity > 0.0:
        amb3 = ambiguity * ambiguity * ambiguity
        xi = np.random.uniform(-1.0, 1.0, size=n_features)
        w = w + amb3 * (sigma_h * zeta + kappa * xi)
    order_anti = np.lexsort((secondary, -w))

    def cascade(order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    winner_anti = cascade(order_anti)

    # ---- this trial's anti-validity carrier (for the leak gate) ----
    j_star = None
    for j in order_anti:
        if a[j] != b[j]:
            j_star = int(j)
            break

    def score_prob(winner):
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        return e / e.sum()

    # Start from the (momentary) anti-validity decider.
    p = score_prob(winner_anti)

    # ---- POSITIONAL-PRIMACY MINORITY (ambiguity-gated mixture) ----
    # The reading-order fallback is elicited only when the
    # validity-to-position binding is ambiguous; monotone lists keep the
    # anchored anti-validity core intact.
    w_eff = min(max(primacy_weight * ambiguity, 0.0), 0.5)
    if w_eff > 0.0:
        winner_prim = cascade(pos)  # earliest-listed expert first
        if winner_prim is not None:
            p = (1.0 - w_eff) * p + w_eff * score_prob(winner_prim)

    # ---- STRONG INTERIOR-GATED VALIDITY LEAK (no substantive ceiling) ----
    # Fires only when the anti cascade's carrier is strictly interior:
    # min + 0.10 < val[j*] < max - 0.10. Weakest-carrier and near-top
    # carrier trials keep the leak OFF.
    if j_star is not None and n_features > 1:
        vmin = float(val.min())
        vmax = float(val.max())
        if (vmin + 0.10) < val[j_star] < (vmax - 0.10):
            q = rho_leak * coherence * mean_gap
            # Numerical validity clamp only (keeps mixture weights proper
            # probabilities on extreme small-n designs); far above the
            # substantive target range q in [0.35, 0.50].
            q = min(max(q, 0.0), 0.75)
            if q > 0.0:
                order_ttb = np.lexsort((secondary, -val))
                winner_ttb = cascade(order_ttb)
                if winner_ttb is not None:
                    p = (1.0 - q) * p + q * score_prob(winner_ttb)

    # ---- wide heterogeneous response discipline: softmax + lapse ----
    return (1.0 - epsilon) * p + epsilon * (np.ones(n_opts) / float(n_opts))

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- rho_leak: [1.4, 2.2]
- primacy_weight: [0.15, 0.2]
- gamma: [-0.05, 0.12]
- sigma_h: [0.6, 1.2]
- kappa: [8, 12]
- beta: [0.4, 2.2]
- epsilon: [0.06, 0.28]
- tie_break: [0, 1]
- zeta: [(-1, 1)] * n_features
- validities: validities

`rationale`:
This theory is the synthesis the arbiter prescribed, retaining the surviving core of both competitors and repairing their documented failures. (1) CORE RETAINED: one-reason anti-validity frugality with tie-anchoring, exact 0.5 on non-discriminating trials, and history-free trials — the machinery that already fits the knife-edge and dissociation experiments (Exp 3, 5, 9-12). (2) STRONG UNCAPPED LEAK: pi_8's interior-carrier gate was directionally right but its 0.18 ceiling strangled the magnitude; with rho_leak in [1.4, 2.2], the leak-dip experiment (validities 0.5/0.75/1.0, coherence 1, mean gap 0.25) gets q in [0.35, 0.55], reproducing the observed dip of about -0.196 at p_f ~ 0.71, while small-gap designs (gap 0.1) get only q ~ 0.18, so the conditionality triad (weakest/near-top carriers keep the leak off) is preserved. The same leak pulls the negative-signature experiments (Exp 7, 8) further toward their observed negative values wherever an interior carrier exists. (3) WIDE RESPONSE DISCIPLINE: beta in [0.4, 2.2] and epsilon in [0.06, 0.28] spread per-subject follow probabilities across roughly [0.46, 0.99]; this lifts the between-subject follow-protection tail to about 0.15 (observed 0.1556, where pi_7's 0.112 and pi_8's 0.075 both undershot) and drops pooled determinism enough to shrink the overprediction on the reversed-decider contrast experiment (0.91/0.96 predicted vs 0.78 observed). (4) POSITIONAL-PRIMACY MINORITY, AMBIGUITY-GATED: the arbiter's own preservation constraint — the strong reversed-cascade following of about 0.727 on the monotone-validity design — forces the primacy mixture to be switched off when the validity-position binding is unambiguous (there, earliest-listed equals highest-validity, so an ungated primacy minority would corrode the anchored core and flip that experiment from 0.44 to about 0.31). Gating the mixture weight by binding ambiguity A (linearly, the same gradient pi_6/pi_7 used for their position components) keeps every monotone design anchored while still delivering the primacy fallback on ambiguous designs, where it dilutes pooled decider-following toward the lower observed rates without ever manufacturing spurious pure primacy types (the per-trial probability mixture keeps every subject below the type-classification thresholds of the primacy-typing experiments, which both theories must predict as exactly 0). (5) DROPPED: pi_8's frozen discipline and distinct-design bonus (falsified by the tail experiment) and pi_7's flatness claim (falsified by the leak dip). Net expected movement relative to the running best: large gain on the leak-dip experiment, moderate gains on the tail, reversed-decider contrast, margin-0 follow, and knife-edge experiments, at the cost of small regressions on the monotone reversed-cascade experiment (leak side-effects ~0.02-0.05) — a favorable trade under the aggregate L2 loss.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0778 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0778.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=float)
    cue_order = np.argsort(-validities, kind="stable")

    margins = []
    matches = []
    for a_cell, b_cell, resp in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float).ravel()
        b = np.asarray(b_cell, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] != validities.shape[0]:
            continue
        diff = a - b

        # Take-The-Best cascade: first discriminating cue in validity order.
        ttb_winner = -1
        for j in cue_order:
            if diff[j] > 0:
                ttb_winner = 0
                break
            if diff[j] < 0:
                ttb_winner = 1
                break
        if ttb_winner < 0:
            # Full tie (trial 13): no TTB winner, uninformative for both theories.
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # Tally margin of the TTB-favored option (positive = tally agrees with TTB).
        if ttb_winner == 0:
            margin = a_wins - b_wins
        else:
            margin = b_wins - a_wins

        r = int(resp)
        match = 1.0 if r == ttb_winner else 0.0
        margins.append(margin)
        matches.append(match)

    if not margins:
        return 0.0

    x = np.asarray(margins, dtype=float)
    y = np.asarray(matches, dtype=float)

    # (1) LEVEL: P(choosing the TTB-favored option) on trials where the
    # tally does NOT favor it (tally majority against, or exact tally tie).
    low = x <= 0
    level = float(np.mean(y[low])) if np.any(low) else 0.0

    # (2) SLOPE: OLS slope of the match indicator on the tally margin,
    # across all trials with a TTB winner (margins -3, -2, -1, 0, +1).
    vx = float(np.var(x))
    if len(x) > 1 and vx > 0:
        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
    else:
        slope = 0.0

    return level - slope
```

**Observed (real) value:** 0.3633 (var=0.0060)
**Candidate (simulated) value:** 0.2923 (var=0.0098)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8479 (var=0.0111)
- pi_2: 0.0323 (var=0.0195)
- pi_3: 0.4099 (var=0.0092)
- pi_3_1: 0.2933 (var=0.0063)
- pi_3_2: 0.3314 (var=0.0076)
- pi_4: 0.3205 (var=0.0077)
- pi_5: 0.2898 (var=0.0076)
- pi_6: 0.3008 (var=0.0206)
- pi_7: 0.2898 (var=0.0117)
- pi_8: 0.2622 (var=0.0083)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def _to_matrix(col):
    # col: pandas Series whose cells are lists/tuples/arrays of binary ratings
    return np.array([list(r) for r in col], dtype=float)


def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    A = _to_matrix(data['option_a_ratings'])
    B = _to_matrix(data['option_b_ratings'])
    if A.ndim != 2 or A.shape != B.shape or A.shape[0] == 0:
        return 0.0

    # Tally margin: number of features where A beats B minus number where B beats A.
    margin = (A > B).sum(axis=1) - (B > A).sum(axis=1)

    # Signed choice: +1 if the subject chose A (response == 0), -1 if B (response == 1).
    resp = pd.to_numeric(data['response'], errors='coerce').to_numpy(dtype=float)
    s = 1.0 - 2.0 * resp

    # Alignment between the signed choice and the tally margin.
    # Tallying: E[s | margin] = (1-eps) * tanh(beta*margin/2)  =>  E[s*margin] > 0,
    # scaling like beta * E[margin^2] / 2 for small beta and saturating at mean(|margin|)
    # for large beta (max value 28/16 = 1.75 on this design).
    # TTB: E[s | trial] = (1-eps)*tanh(beta/2)*ttb_sign, and on this stimulus set
    # sum over the 16 trial types of ttb_sign * margin = -2 (mean -0.125), so the
    # TTB value is slightly NEGATIVE for any beta > 0.
    return float(np.mean(s * margin))
```

**Observed (real) value:** 0.0358 (var=0.0181)
**Candidate (simulated) value:** 0.0892 (var=0.0381)
**Other theories' values on this metric (for reference):**
- pi_2: 1.3208 (var=0.1532)
- pi_1: -0.1621 (var=0.0229)
- pi_3: 0.0608 (var=0.0436)
- pi_3_1: 0.1208 (var=0.0447)
- pi_3_2: 0.0708 (var=0.0422)
- pi_4: 0.0400 (var=0.0527)
- pi_5: 0.0462 (var=0.0368)
- pi_6: 0.0396 (var=0.0357)
- pi_7: 0.1029 (var=0.0537)
- pi_8: 0.1058 (var=0.0349)

### Experiment 3
**Design**
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Reversed-cascade follow rate (centered).
    #
    # For every trial we determine the winner implied by a one-reason
    # cascade that consults cues in ASCENDING true validity (i.e., the
    # 0.5-validity expert first, ..., the 0.9-validity expert last) --
    # exactly the subjective hierarchy implied by full misbinding
    # (kappa = 1). Full-tie trials carry no directional prediction and
    # are dropped. The metric is the signed follow rate:
    #   +1  -> subject always picks the reversed-cascade winner
    #    0  -> chance
    #   -1  -> subject always picks the opposite option
    #
    # Predicted values:
    #   Misbinding TTB: p(follow) = (1-eps)*sigmoid(beta) + eps/2 ~ 0.60
    #                   on EVERY discriminating trial  ->  metric ~ +0.21
    #   Tallying:       on the 7 trials where the tally winner OPPOSES
    #                   the lowest-validity cue (trials 1,2,3,5,6,7,15)
    #                   p(follow) < 0.5 for ANY beta > 0; on the 5
    #                   zero-margin trials (4,8,11,12,13) p(follow)
    #                   = 0.5 EXACTLY for any (beta, eps); only the 3
    #                   agreeing trials (9,10,14) push above 0.5.
    #                   Hence the Tallying metric is strictly NEGATIVE
    #                   for every parameterization in its ranges
    #                   (approaching 0 only as beta->0 or eps->0.5).
    validities = np.array([0.90, 0.80, 0.70, 0.60, 0.50])
    # Ascending true validity == descending subjective validity under
    # full misbinding: consult index 4 first, then 3, 2, 1, 0.
    cue_order = np.argsort(validities, kind="stable")

    n_used = 0
    follow = 0
    for a_raw, b_raw, r in zip(data["option_a_ratings"],
                               data["option_b_ratings"],
                               data["response"]):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape != b.shape or a.size != validities.size:
            continue
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            # Full tie (trial 16): no directional prediction.
            continue
        n_used += 1
        if int(r) == winner:
            follow += 1

    if n_used == 0:
        return 0.0
    return 2.0 * (follow / n_used) - 1.0
```

**Observed (real) value:** 0.4533 (var=0.0207)
**Candidate (simulated) value:** 0.4724 (var=0.0287)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1947 (var=0.0151)
- pi_2: -0.1742 (var=0.0091)
- pi_1: -0.4209 (var=0.0150)
- pi_3_1: 0.4276 (var=0.0099)
- pi_3_2: 0.3960 (var=0.0138)
- pi_4: 0.3751 (var=0.0163)
- pi_5: 0.4636 (var=0.0174)
- pi_6: 0.4244 (var=0.0833)
- pi_7: 0.4400 (var=0.0168)
- pi_8: 0.4920 (var=0.0059)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast metric built from two parameter-free anchors:
    #
    # (1) Margin-0 cells with a reversed-cascade decider (T5, T6, T11, T16).
    #     Tallying predicts p(A) = 0.5 EXACTLY on any tie trial, for any
    #     (beta, epsilon), so the signed follow-the-decider score has
    #     expectation exactly 0. The reversed-hierarchy model follows its
    #     decider with q = (1-eps)*sigmoid(beta) + eps/2 in [0.57, 0.64],
    #     so its expected score is 2q-1 in [0.14, 0.28].
    #
    # (2) Disagreement cells (T1, T2, T10, T14): nonzero tally margin whose
    #     sign OPPOSES the reversed decider. Tallying follows the margin
    #     with p >= 0.5 for any beta > 0 (expected signed score 2p-1 >= 0);
    #     the reversed model follows the decider, i.e. follows the tally
    #     with 1-q <= 0.43 (expected signed score 1-2q < 0).
    #
    # metric = mean(follow-decider score on margin-0 cells)
    #        - mean(follow-tally score on disagreement cells)
    #
    # Tallying: 0 - (2p-1) <= 0 (typically around -0.4 to -0.9).
    # Reversed: (2q-1) - (1-2q) = 4q-2 in [0.28, 0.56] > 0.
    # The two theories land on opposite sides of zero with a large gap.
    s1 = []
    s2 = []
    n_rows = len(data)
    for i in range(n_rows):
        row = data.iloc[i]
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        n = len(a)

        # Tally margin: strict feature-wise wins for A minus for B.
        margin = 0
        for j in range(n):
            if a[j] > b[j]:
                margin += 1
            elif b[j] > a[j]:
                margin -= 1

        # Reversed-hierarchy decider: first strictly discriminating
        # feature scanning from the LAST index down (ascending true
        # validity, since validities descend by index).
        dec = 0
        for j in range(n - 1, -1, -1):
            if a[j] > b[j]:
                dec = 1
                break
            if b[j] > a[j]:
                dec = -1
                break
        if dec == 0:
            # Identical options: no decider, no information.
            continue

        resp = int(row["response"])
        chose_a = (resp == 0)

        if margin == 0:
            # Knife-edge cell: tally is exactly 0.5 under Tallying.
            follow_dec = (dec == 1 and chose_a) or (dec == -1 and not chose_a)
            s1.append(1.0 if follow_dec else -1.0)
        elif (margin > 0 and dec == -1) or (margin < 0 and dec == 1):
            # Disagreement cell: tally direction opposes the decider.
            follow_tally = (margin > 0 and chose_a) or (margin < 0 and not chose_a)
            s2.append(1.0 if follow_tally else -1.0)

    m1 = float(np.mean(s1)) if len(s1) > 0 else 0.0
    m2 = float(np.mean(s2)) if len(s2) > 0 else 0.0
    return m1 - m2
```

**Observed (real) value:** 0.7787 (var=0.0600)
**Candidate (simulated) value:** 0.8560 (var=0.1270)
**Other theories' values on this metric (for reference):**
- pi_2: -0.6893 (var=0.0556)
- pi_3: 0.3867 (var=0.0818)
- pi_1: -1.3413 (var=0.2259)
- pi_3_1: 0.8027 (var=0.0699)
- pi_3_2: 0.6893 (var=0.0735)
- pi_4: 0.8467 (var=0.0843)
- pi_5: 0.8960 (var=0.0975)
- pi_6: 0.7893 (var=0.3919)
- pi_7: 0.9093 (var=0.1091)
- pi_8: 0.9560 (var=0.0463)

### Experiment 5
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Decider-follow rate on the margin-0 DISCRIMINATING trials (the
    # "subjective-hierarchy ladder": T4, T7-T11 in the design).
    # On these trials the tally is exactly balanced, so Tallying is
    # pinned to p = 0.50 toward either side for ANY (beta, epsilon)
    # (softmax over equal tallies is uniform, lapse is uniform).
    # Misbinding TTB instead follows its (reversed-hierarchy) first
    # discriminating cue with p = (1-eps)*sigmoid(beta) + eps/2,
    # which is >= 0.63 over the stated parameter ranges. The metric
    # is the proportion of choices that agree with the misbound
    # decider on exactly those trials: ~0.71 under misbinding TTB,
    # ~0.50 under Tallying.
    val = np.array([0.7, 0.9, 0.6, 0.95, 0.75, 0.6], dtype=float)
    # kappa = 1 misbinding: subjective weights are the validity list
    # bound in reverse to feature positions.
    w = val[::-1]
    cue_order = np.argsort(-w, kind="stable")  # f2, f4, f1, f5, f3, f0

    a = np.array([list(x) for x in data['option_a_ratings']], dtype=float)
    b = np.array([list(x) for x in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int).ravel()

    if a.shape[0] == 0:
        return 0.5

    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Margin-0 trials that still have at least one discriminating cue
    # (excludes the full tie T16, where both theories say 0.5).
    mask = (a_wins == b_wins) & ((a_wins + b_wins) > 0)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return 0.5

    follow = []
    for i in idx:
        decider = None
        for j in cue_order:
            if a[i, j] > b[i, j]:
                decider = 0
                break
            if b[i, j] > a[i, j]:
                decider = 1
                break
        if decider is None:
            continue  # shouldn't happen on discriminating trials
        follow.append(1 if resp[i] == decider else 0)

    if len(follow) == 0:
        return 0.5
    return float(np.mean(follow))
```

**Observed (real) value:** 0.6133 (var=0.0052)
**Candidate (simulated) value:** 0.6733 (var=0.0104)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.7178 (var=0.0092)
- pi_2: 0.5072 (var=0.0084)
- pi_1: 0.1383 (var=0.0146)
- pi_3: 0.6261 (var=0.0070)
- pi_3_2: 0.6633 (var=0.0068)
- pi_4: 0.6883 (var=0.0074)
- pi_5: 0.6311 (var=0.0430)
- pi_6: 0.5400 (var=0.0198)
- pi_7: 0.7217 (var=0.0070)
- pi_8: 0.6900 (var=0.0042)

### Experiment 6
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Knife-edge signed contrast on the four margin-0 discriminating cells.

    Tallying predicts p(A) = 0.50 EXACTLY on every margin-0 cell, for any
    (beta, epsilon): equal tallies give an exactly uniform softmax. The
    reversed-hierarchy one-reason model, in contrast, lets a single cue
    decide these same cells, producing a systematic ~0.70/0.30 split whose
    sign is set by which option wins on the cascade's first cue.

    Metric = mean p(choose A | T11, T13) - mean p(choose A | T8, T15),
    where T11/T13 are the margin-0 cells whose cascade-first cue favors A
    and T8/T15 are the margin-0 cells whose cascade-first cue favors B
    (under the design's validities [0.95, 0.85, 0.75, 0.65, 0.6, 0.6]
    reversed -> f5 consulted first).

    Tallying: expectation exactly 0 for ANY parameter values.
    Misbinding TTB: expectation ~ +/-(2p-1) in [0.33, 0.49].
    """
    def to_key(v):
        return tuple([int(x) for x in v])

    a_keys = data["option_a_ratings"].apply(to_key)
    b_keys = data["option_b_ratings"].apply(to_key)
    chose_a = 1.0 - data["response"].astype(float)  # response 0 = A

    # Margin-0 (knife-edge) cells, identified by their rating vectors.
    # Cascade-first cue (f5) favors A here:
    hi_cells = [
        ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)),  # T11: 1-1 split, f5->A
        ((0, 0, 0, 0, 1, 1), (1, 1, 0, 0, 0, 0)),  # T13: 2-2 split, f5->A
    ]
    # Cascade-first cue (f5) favors B here:
    lo_cells = [
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),  # T8:  1-1 split, f5->B
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),  # T15: 3-3 split, f5->B
    ]

    def cell_pa(ak, bk):
        mask = (a_keys == ak) & (b_keys == bk)
        if int(mask.sum()) == 0:
            return np.nan
        return float(chose_a[mask].mean())

    hi = [cell_pa(ak, bk) for ak, bk in hi_cells]
    lo = [cell_pa(ak, bk) for ak, bk in lo_cells]
    hi = [v for v in hi if not np.isnan(v)]
    lo = [v for v in lo if not np.isnan(v)]
    if not hi or not lo:
        return 0.0

    return float(np.mean(hi) - np.mean(lo))
```

**Observed (real) value:** 0.3400 (var=0.0288)
**Candidate (simulated) value:** 0.4567 (var=0.0584)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0367 (var=0.0470)
- pi_3_1: 0.3933 (var=0.0286)
- pi_1: -0.7033 (var=0.0409)
- pi_3: 0.2250 (var=0.0465)
- pi_3_2: 0.3950 (var=0.0319)
- pi_4: 0.3633 (var=0.0488)
- pi_5: 0.4983 (var=0.0354)
- pi_6: 0.4583 (var=0.1237)
- pi_7: 0.4450 (var=0.0288)
- pi_8: 0.4700 (var=0.0333)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Knife-edge decider-direction contrast at balanced tally margins.
    #
    # For every trial we compute (a) the tally margin m = (#features A wins)
    # - (#features B wins), and (b) the winner implied by the advocated
    # misbinding-TTB cascade (subjective hierarchy = instructed validities
    # reversed end-to-end: cue order f4 -> f0 -> f2 -> f1 -> f3 -> f5).
    #
    # We keep only trials with |m| in {0, 3} (excluding the |m|=1 cells,
    # whose decider directions are unbalanced across margins) and where the
    # cascade has a discriminating cue. In this subset the misbinding-A
    # cells are {T1 (m=-3), T3 (m=0), T5 (m=+3), T9 (m=0), T10 (m=0)} and
    # the misbinding-B cells are {T6 (m=+3), T7 (m=0), T8 (m=-3), T11 (m=0),
    # T12 (m=0)} -- perfectly mirror-balanced in margin.
    #
    # Metric = P(choose A | cascade says A) - P(choose A | cascade says B).
    #
    # Under Tallying, p(A) = (1-eps)*sigmoid(beta*m) + eps/2, which is
    # exactly symmetric in m. Within each group the margins are balanced
    # (-3/+3 pairs and 0s), so BOTH group means are exactly 0.5 for ANY
    # (beta, epsilon) -- the metric is pinned to 0 parameter-free.
    #
    # Under misbinding TTB, p(A) = p_follow on cascade-A cells and
    # 1 - p_follow on cascade-B cells, so the metric equals 2*p_follow - 1,
    # which over the advocated parameter ranges lies in ~[0.27, 0.44]
    # (mean ~0.36) regardless of margin.

    df = data.reset_index(drop=True)
    a_lists = df['option_a_ratings'].tolist()
    b_lists = df['option_b_ratings'].tolist()
    resp = df['response'].to_numpy()

    # Design validities and the kappa=1 reversed subjective hierarchy.
    val = np.array([0.6, 0.95, 0.6, 0.85, 0.65, 0.9])
    w = val[::-1]
    cue_order = np.argsort(-w, kind='stable').tolist()  # -> [4, 0, 2, 1, 3, 5]

    a_side = []  # trials where the misbound cascade favors A
    b_side = []  # trials where the misbound cascade favors B

    for i in range(len(df)):
        av = [int(v) for v in a_lists[i]]
        bv = [int(v) for v in b_lists[i]]
        n = len(av)

        margin = (sum(1 for j in range(n) if av[j] > bv[j])
                  - sum(1 for j in range(n) if bv[j] > av[j]))
        if abs(margin) == 1:
            # Unbalanced decider directions at |m|=1; excluded so the
            # Tallying expectation stays exactly 0.
            continue

        winner = -1
        for j in cue_order:
            if j >= n:
                continue
            if av[j] > bv[j]:
                winner = 0
                break
            if bv[j] > av[j]:
                winner = 1
                break
        if winner == -1:
            continue  # full-tie trial (T16): no discriminating cue

        chose_a = 1.0 if resp[i] == 0 else 0.0
        if winner == 0:
            a_side.append(chose_a)
        else:
            b_side.append(chose_a)

    pa = float(np.mean(a_side)) if len(a_side) > 0 else 0.5
    pb = float(np.mean(b_side)) if len(b_side) > 0 else 0.5
    return float(pa - pb)
```

**Observed (real) value:** -0.3107 (var=0.0225)
**Candidate (simulated) value:** -0.1927 (var=0.0257)
**Other theories' values on this metric (for reference):**
- pi_3_2: 0.3613 (var=0.0180)
- pi_2: -0.0100 (var=0.0118)
- pi_1: 0.0160 (var=0.0107)
- pi_3: 0.2040 (var=0.0089)
- pi_3_1: 0.4007 (var=0.0109)
- pi_4: -0.1453 (var=0.0099)
- pi_5: -0.2327 (var=0.0163)
- pi_6: -0.1993 (var=0.0321)
- pi_7: -0.1520 (var=0.0136)
- pi_8: -0.1673 (var=0.0158)

### Experiment 8
**Design**
  A=[1, 0, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

# Design-time constants for this experiment.
VALIDITIES = np.array([0.6, 0.9, 0.8, 0.95, 0.65, 0.75, 0.6])


def metric(data: pd.DataFrame) -> float:
    """Cue-following deviation from chance on tally-tied (margin-0) trials.

    For every trial whose tally margin is zero (A and B win on equal numbers
    of features), find the first strictly discriminating cue under the
    reversed-hierarchy cue order (the competing theory's subjective
    hierarchy). Record whether the subject's choice followed that cue.
    Tallying predicts exactly chance (0 deviation) on every tied trial,
    for ANY (beta, epsilon). One-reason misbinding predicts a systematic
    follow rate of p = (1-eps)*sigmoid(beta) + eps/2, i.e. a deviation of
    roughly +0.12 to +0.25 above chance.
    """
    # Reversed-hierarchy subjective weights and cue order (stable ties).
    w = VALIDITIES[::-1].astype(float)
    cue_order = np.argsort(-w, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    resp_col = data["response"].tolist()

    follows = []
    for a_raw, b_raw, r_raw in zip(a_col, b_col, resp_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.ndim != 1:
            continue

        # Tally margin: number of features A wins minus number B wins.
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin != 0:
            # Only tied tallies carry the parameter-free signature.
            continue

        # First discriminating cue under the reversed hierarchy.
        decider = None
        for j in cue_order:
            if a[j] > b[j]:
                decider = 0  # cue favors A
                break
            if b[j] > a[j]:
                decider = 1  # cue favors B
                break
        if decider is None:
            # Full tie (T16): both theories say 0.5; no information.
            continue

        resp = int(r_raw)
        follows.append(1.0 if resp == decider else 0.0)

    if len(follows) == 0:
        return 0.0

    # Deviation of the tied-trial cue-follow rate from chance.
    return float(np.mean(follows)) - 0.5
```

**Observed (real) value:** -0.2889 (var=0.0205)
**Candidate (simulated) value:** -0.2456 (var=0.0182)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0011 (var=0.0175)
- pi_3_2: 0.1878 (var=0.0164)
- pi_1: 0.3422 (var=0.0196)
- pi_3: 0.0944 (var=0.0115)
- pi_3_1: 0.2311 (var=0.0115)
- pi_4: -0.1644 (var=0.0172)
- pi_5: -0.2356 (var=0.0097)
- pi_6: -0.2433 (var=0.0122)
- pi_7: -0.2122 (var=0.0106)
- pi_8: -0.1967 (var=0.0089)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dissociation metric: fraction of choices matching the advocated
    # (Reversed-Hierarchy Frugality) cascade winner, computed ONLY on
    # trials where the two theories' cue hierarchies yield OPPOSITE
    # predictions. On those trials the advocated theory predicts a
    # match rate of p = (1-eps)*sigmoid(beta) + eps/2 ~ 0.62-0.80,
    # while the competing theory predicts the complementary rate
    # ~ 1-p ~ 0.20-0.40, so the two theories produce visibly
    # different numbers on this scalar.
    val = np.array([0.9, 0.8, 0.95, 0.7, 0.6, 0.65])
    n = len(val)

    # Advocated hierarchy: subjective weights = reversed validity list.
    w_adv = val[::-1]  # [0.65, 0.6, 0.7, 0.95, 0.8, 0.9]
    order_adv = np.argsort(-w_adv, kind="stable")  # f3, f5, f4, f2, f0, f1

    # Competing hierarchy: ascending advertised validity.
    order_comp = np.lexsort((np.arange(n), val))  # f4, f5, f3, f1, f0, f2

    def cascade_winner(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1  # no discriminating cue

    matches = []
    for ra, rb, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = [int(v) for v in ra]
        b = [int(v) for v in rb]
        r = int(resp)

        wa = cascade_winner(a, b, order_adv)
        wc = cascade_winner(a, b, order_comp)

        # Keep only trials that genuinely dissociate the two theories:
        # both cascades terminate and they pick different options.
        if wa < 0 or wc < 0 or wa == wc:
            continue

        matches.append(1.0 if r == wa else 0.0)

    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4280 (var=0.0052)
**Candidate (simulated) value:** 0.4013 (var=0.0070)
**Other theories' values on this metric (for reference):**
- pi_3_2: 0.6607 (var=0.0064)
- pi_4: 0.3193 (var=0.0078)
- pi_1: 0.8787 (var=0.0117)
- pi_2: 0.5133 (var=0.0067)
- pi_3: 0.6033 (var=0.0089)
- pi_3_1: 0.7033 (var=0.0066)
- pi_5: 0.3993 (var=0.0160)
- pi_6: 0.4400 (var=0.0202)
- pi_7: 0.3867 (var=0.0066)
- pi_8: 0.4053 (var=0.0082)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # ---------- design constants ----------
    NF = 6
    VAL = [0.9, 0.65, 0.6, 0.8, 0.95, 0.7]
    # Weakest-Expert-First cascade: cues in ASCENDING validity
    adv_order = sorted(range(NF), key=lambda j: VAL[j])      # [2,1,5,3,0,4]
    # Reversed-Hierarchy (misbinding) cascade: DESCENDING val[::-1]
    rev = VAL[::-1]
    comp_order = sorted(range(NF), key=lambda j: -rev[j])    # [1,5,2,0,4,3]

    # ---------- robust cell parsers ----------
    def parse_vec(x):
        try:
            if isinstance(x, np.ndarray):
                v = np.asarray(x, dtype=int).ravel()
                return v if v.size == NF else None
            if isinstance(x, (list, tuple)):
                v = np.asarray(list(x), dtype=float).ravel()
                if v.size != NF:
                    return None
                return np.rint(v).astype(int)
            s = str(x).strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            elif s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            toks = s.replace(",", " ").replace(";", " ").split()
            if len(toks) == 1:
                t0 = toks[0].strip("'\"")
                if len(t0) == NF and all(ch in "01" for ch in t0):
                    return np.asarray([int(ch) for ch in t0], dtype=int)
                return None
            vals = []
            for t in toks:
                vals.append(int(round(float(t.strip("'\"")))))
            v = np.asarray(vals, dtype=int)
            return v if v.size == NF else None
        except Exception:
            return None

    def parse_resp(x):
        try:
            if isinstance(x, (bool, np.bool_)):
                return int(x)
            if isinstance(x, (int, np.integer)):
                r = int(x)
                return r if r in (0, 1) else None
            if isinstance(x, (float, np.floating)):
                if np.isnan(x):
                    return None
                r = int(round(float(x)))
                return r if r in (0, 1) else None
            s = str(x).strip().upper()
            if s == "A":
                return 0
            if s == "B":
                return 1
            r = int(round(float(s)))
            return r if r in (0, 1) else None
        except Exception:
            return None

    # ---------- column resolution ----------
    cols = list(data.columns)
    lmap = {}
    for c in cols:
        lmap[str(c).lower()] = c

    def resolve(names, keys):
        for nm in names:
            if nm in lmap:
                return lmap[nm]
        for c in cols:
            lc = str(c).lower()
            if any(k in lc for k in keys):
                return c
        return None

    a_col = resolve(["option_a_ratings", "optiona_ratings", "ratings_a",
                     "a_ratings", "option_a"], ["a_rat", "option_a", "ratings_a"])
    b_col = resolve(["option_b_ratings", "optionb_ratings", "ratings_b",
                     "b_ratings", "option_b"], ["b_rat", "option_b", "ratings_b"])
    r_col = resolve(["response", "choice", "chosen", "answer", "pick",
                     "selected"], ["respon", "choic", "chose", "select", "answer", "pick"])
    s_col = resolve(["subject_id", "subjectid", "subject", "subj",
                     "participant"], ["subject", "subj", "participant"])
    if a_col is None or b_col is None or r_col is None:
        return 0.0

    def cascade(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    # ---------- per-subject cascade-follow rates ----------
    n_rows = int(len(data))
    if s_col is not None:
        try:
            subjects = [str(v) for v in list(data[s_col])]
        except Exception:
            subjects = ["s%d" % i for i in range(n_rows)]
    else:
        subjects = ["s%d" % i for i in range(n_rows)]

    a_cells = list(data[a_col])
    b_cells = list(data[b_col])
    r_cells = list(data[r_col])

    stats = {}
    for i in range(n_rows):
        a = parse_vec(a_cells[i])
        b = parse_vec(b_cells[i])
        resp = parse_resp(r_cells[i])
        if a is None or b is None or resp is None:
            continue
        wa = cascade(a, b, adv_order)
        if wa is None:
            continue  # complete tie: both theories guess 50/50
        wc = cascade(a, b, comp_order)
        if wc != wa:
            continue  # keep only trials where both hierarchies dictate the
                       # same option, so the follow rate is theory-neutral
        rec = stats.get(subjects[i])
        if rec is None:
            rec = [0, 0]
            stats[subjects[i]] = rec
        rec[1] += 1
        if resp == wa:
            rec[0] += 1

    rates = []
    for k in sorted(stats.keys()):
        f, t = stats[k]
        if t > 0:
            rates.append(f / float(t))
    if not rates:
        return 0.0

    x = np.asarray(rates, dtype=float)
    n = int(x.size)
    m = float(x.mean())
    if n > 1:
        dm = np.abs(x[:, None] - x[None, :])
        g = float(dm.sum() / (n * (n - 1.0)))  # Gini mean difference (ordered pairs)
    else:
        g = 0.0

    # ---------- population-level pairwise conformity-dispersion index ----------
    # Every ordered subject pair (i, j) contributes the anchor excess of its
    # first member, (x_i - 0.5), plus half the pair's follow-rate distance,
    # 0.5 * |x_i - x_j|.  Summed over all n^2 ordered pairs this equals
    #   n^2 * (mean(x) - 0.5) + 0.5 * n(n-1) * GiniMD(x).
    # For a single-subject slice it collapses to that subject's own excess
    # conformity (x - 0.5): the one diagonal pair contributes no distance.
    index = (float(n) ** 2) * (m - 0.5) + 0.5 * float(n) * (float(n) - 1.0) * g
    return float(index)
```

**Observed (real) value:** 116.5556 (var=0.0041)
**Candidate (simulated) value:** 132.8667 (var=0.0019)
**Other theories' values on this metric (for reference):**
- pi_4: 588.7556 (var=0.0036)
- pi_3_2: 536.1222 (var=0.0024)
- pi_1: -680.7000 (var=0.0054)
- pi_2: 105.3778 (var=0.0026)
- pi_3: 332.3111 (var=0.0024)
- pi_3_1: 567.5556 (var=0.0025)
- pi_5: 184.2778 (var=0.0153)
- pi_6: 184.6889 (var=0.0053)
- pi_7: 119.2111 (var=0.0025)
- pi_8: 107.0444 (var=0.0025)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Anti-misbound decider-follow rate on hierarchy-dissociating cells.

    For each trial we compute the option picked by the competitor's
    hierarchy (weakest-advertised-expert-first: cues sorted by ASCENDING
    validity, ties by display position) and the option picked by the
    advocated theory's anti-misbound re-encoding (w = -val[::-1],
    consulted in descending subjective weight, which is equivalent to
    sorting cues by ASCENDING val[::-1], ties by display position), using
    the experiment's validities [0.80, 0.55, 0.90, 0.55, 0.80, 0.65].

    On this design the two hierarchies' first discriminating cues point to
    OPPOSITE options on 13 of the 16 unique pairs (T1-T13) and agree on
    T14/T15 (T16 is a full tie). The deciders are invariant to either
    theory's free tie_break parameter because no trial opposes two cues
    tied within a hierarchy group. The metric is the proportion of
    choices that follow the anti-misbound decider on those 13
    dissociating cells only.

    Predicted values:
      - Weakest-Expert-First (competing): subjects follow their own
        (central) decider with p_f ~ 0.70, hence follow the anti-misbound
        decider with ~0.30.
      - Advocated: ~95% of subjects are anti-misbinders (m_am =
    ambiguity x coherence-gate ~ 0.95 for this arrangement), so the
        follow rate toward the anti-misbound decider is ~0.70.
    The two theories therefore sit on opposite sides of 0.50 with a
    ~0.40 gap, while per-subject values are tight (each subject
    contributes 13 cells x 6 reps = 78 dissociating trials).
    """
    VAL = [0.80, 0.55, 0.90, 0.55, 0.80, 0.65]
    n = len(VAL)

    # Competitor hierarchy: ascending advertised validity, position ties.
    comp_order = sorted(range(n), key=lambda j: (VAL[j], j))

    # Anti-misbound hierarchy: w = -val[::-1] consulted in DESCENDING
    # subjective weight == ascending val[::-1], position ties.
    rev = VAL[::-1]
    mis_order = sorted(range(n), key=lambda j: (rev[j], j))

    def decider(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    hits = []
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = [int(v) for v in a_raw]
            b = [int(v) for v in b_raw]
        except Exception:
            continue
        if len(a) != n or len(b) != n:
            continue
        cd = decider(a, b, comp_order)
        md = decider(a, b, mis_order)
        # Keep only cells where the two hierarchies genuinely dissociate.
        if cd is None or md is None or cd == md:
            continue
        hits.append(1.0 if int(r) == int(md) else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4897 (var=0.0022)
**Candidate (simulated) value:** 0.4792 (var=0.0027)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4262 (var=0.0087)
- pi_4: 0.5215 (var=0.0033)
- pi_1: 0.3595 (var=0.0039)
- pi_2: 0.4626 (var=0.0037)
- pi_3: 0.5290 (var=0.0024)
- pi_3_1: 0.5915 (var=0.0016)
- pi_3_2: 0.5703 (var=0.0038)
- pi_6: 0.4649 (var=0.0080)
- pi_7: 0.5056 (var=0.0039)
- pi_8: 0.5026 (var=0.0029)

### Experiment 12
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np

    # Reference hierarchy: the WEAKEST-EXPERT-FIRST (ascending-validity)
    # cascade for the design validity vector [0.75, 0.9, 0.65, 0.8, 0.85, 0.6]:
    # consult order f5 (0.60) -> f2 (0.65) -> f0 (0.75) -> f3 (0.80)
    # -> f4 (0.85) -> f1 (0.90). On every diagnostic trial below, the first
    # cue in this order that discriminates A from B defines the reference
    # side, and the metric measures how often the subject's choice lands
    # on that side.
    val = [0.75, 0.9, 0.65, 0.8, 0.85, 0.6]
    order = sorted(range(6), key=lambda j: val[j])

    # Diagnostic multi-cue trials: T3, T4, T7, T8, T9, T10, T11
    # (7 unique pairs x 6 repetitions = 42 trials per subject).
    # These are the cells on which the two theories' hierarchies diverge:
    # the reference decider is always the low-validity expert (f2 at 0.65
    # or f3 at 0.80), while the competing theory's anti-misbound component
    # (and its per-cue zeta distortion of either component) lands its
    # decider on a different, higher-validity expert for a large fraction
    # of subjects.
    diag = set()
    diag.add(((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 1, 1)))  # T3
    diag.add(((1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 1)))  # T4 (mirror of T3)
    diag.add(((1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0)))  # T7
    diag.add(((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0)))  # T8 (mirror of T7)
    diag.add(((1, 1, 0, 0, 1, 1), (1, 1, 1, 1, 0, 1)))  # T9
    diag.add(((1, 1, 1, 1, 0, 1), (1, 1, 0, 0, 1, 1)))  # T10 (mirror of T9)
    diag.add(((0, 1, 0, 0, 0, 0), (1, 0, 1, 0, 1, 0)))  # T11

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    hits = []
    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        try:
            a = tuple([int(v) for v in a_raw])
            b = tuple([int(v) for v in b_raw])
            r = int(r_raw)
        except Exception:
            continue
        if len(a) != 6 or len(b) != 6:
            continue
        if (a, b) not in diag:
            continue
        # Reference decider: first strictly discriminating cue in the
        # ascending-validity (weakest-expert-first) consult order.
        side = None
        for j in order:
            if a[j] != b[j]:
                side = 0 if a[j] > b[j] else 1
                break
        if side is None:
            continue
        hits.append(1.0 if r == side else 0.0)

    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5324 (var=0.0041)
**Candidate (simulated) value:** 0.5643 (var=0.0079)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7076 (var=0.0076)
- pi_5: 0.5895 (var=0.0200)
- pi_1: 0.3476 (var=0.0042)
- pi_2: 0.7433 (var=0.0076)
- pi_3: 0.5343 (var=0.0067)
- pi_3_1: 0.5810 (var=0.0066)
- pi_3_2: 0.5752 (var=0.0040)
- pi_6: 0.5900 (var=0.0093)
- pi_7: 0.5586 (var=0.0062)
- pi_8: 0.5729 (var=0.0032)

### Experiment 13
**Design**
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Follow-the-anti-misbound-decider rate on the four trial types where the
    # anti-misbound hierarchy (subjective weights w = -validities[::-1], i.e.
    # consult {f1,f3} -> f5 -> f2 -> {f0,f4}) is pitted head-to-head against
    # the position-primacy hierarchy, and where the competitor's anti-validity
    # and TTB components split away from its primacy mass: T1, T2, T9, T10
    # (24 of the 96 trials per subject).
    #
    # Deciders are recomputed inside the metric with EXACTLY the same
    # conventions as both theories' predict functions (w = -val[::-1],
    # np.lexsort((secondary, -w))), and a trial is included only if the
    # decider is identical under BOTH tie-break conventions (verified to
    # hold on all four types: the f1/f3 and f0/f4 weight ties never
    # internally oppose here), so the classification is exact and
    # tie-break independent. Zeta distortion is OFF in both theories for
    # this design (validity ties present), so per-subject deciders are
    # deterministic given the mixture component.
    #
    # Predicted values:
    #   ADVOCATED (m_am = 0.912 * sigmoid((0.727-0.45)/0.08) = 0.884):
    #     88.4% anti-misbound subjects follow this decider at p_f ~ 0.73;
    #     11.6% anti-validity subjects oppose it on T1/T2 and follow it on
    #     T9/T10 -> 0.50. Population ~ 0.70, per-subject SD ~ 0.13.
    #   COMPETING (primacy 0.573, anti-val 0.353, recency 0.043, TTB 0.031):
    #     primacy subjects OPPOSE the misbound decider on all four types
    #     (their f0 decider points the other way) -> 1-p_f ~ 0.27;
    #     anti-validity and TTB subjects sit at 0.50 (oppose on T1/T2 via
    #     f5, follow on T9/T10 via f1); recency -> 0.27 (f5/f4 point
    #     against on all four). Population ~ 0.36, per-subject SD ~ 0.15.
    # Gap ~ 0.34 with tight per-subject spreads on both sides: Welch t at
    # N=25 ~ 8-9, far beyond alpha = 0.01. Values near 0.7 support the
    # anti-misbound mixture; values near 0.35 support the position mixture.
    n_features = 6
    val = np.array([0.65, 0.90, 0.50, 0.80, 0.50, 0.90], dtype=float)

    # Anti-misbound subjective weights (identical to the advocated theory's
    # w_base = -val[::-1]).
    w_misbound = -val[::-1]

    def decider(a, b, w, later_first=False):
        # Consult cues in DESCENDING subjective weight; same lexsort
        # convention as both predict functions.
        n = len(w)
        pos = np.arange(n)
        secondary = (n - 1 - pos) if later_first else pos
        order = np.lexsort((secondary, -w))
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    # The four head-to-head trial types: T1, T2, T9, T10.
    trials = [
        ([0, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1]),  # T1
        ([1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 0, 0]),  # T2 (mirror of T1)
        ([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1]),  # T9
        ([1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1]),  # T10 (mirror of T9)
    ]

    target = {}
    for a_list, b_list in trials:
        a = np.asarray(a_list, dtype=int)
        b = np.asarray(b_list, dtype=int)
        d_early = decider(a, b, w_misbound, later_first=False)
        d_late = decider(a, b, w_misbound, later_first=True)
        # Include only if the decider is tie-break independent.
        if d_early is None or d_late is None or d_early != d_late:
            continue
        key = (tuple([int(v) for v in a]), tuple([int(v) for v in b]))
        target[key] = int(d_early)

    if not target:
        return 0.5

    hits = []
    for ra, rb, resp in zip(data["option_a_ratings"],
                            data["option_b_ratings"],
                            data["response"]):
        a = tuple([int(v) for v in np.asarray(ra).ravel()])
        b = tuple([int(v) for v in np.asarray(rb).ravel()])
        key = (a, b)
        if key in target:
            hits.append(1.0 if int(resp) == target[key] else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.1500 (var=0.0143)
**Candidate (simulated) value:** 0.2775 (var=0.0146)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6542 (var=0.0400)
- pi_6: 0.2417 (var=0.0078)
- pi_1: 0.4908 (var=0.0063)
- pi_2: 0.1350 (var=0.0085)
- pi_3: 0.4050 (var=0.0097)
- pi_3_1: 0.2858 (var=0.0074)
- pi_3_2: 0.3150 (var=0.0106)
- pi_4: 0.2950 (var=0.0089)
- pi_7: 0.2758 (var=0.0171)
- pi_8: 0.3358 (var=0.0067)

### Experiment 14
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """
    Primacy-type excess vs validity-respecting-type excess.

    Two cell families are selected by their discriminating-cue sets
    (feature positions where the two options' ratings differ):

    Family 1 ("early-cue cells"; design trials T1-T4, T7-T8, T13, T15):
        sets {1,4}, {2,4}, {1,5}, {1,3}, {1,3,4}. The earliest-listed
        discriminating expert (f1 or f2) favors the opposite option from
        every later discriminating expert, so a position-primacy reader
        and an anti-validity (weakest-first) reader pick opposite options.

    Family 2 ("weak-cue cells"; design trials T5-T6, T9-T10):
        sets {3,4}, {3,5}. The earliest-listed discriminating expert is
        the low-validity f3, opposing f4/f5, so primacy and anti-validity
        readers disagree here as well, but a descending-validity (TTB-like
        or anti-misbound) reader now sides with the anti-validity reader.

    Per subject: X1 = rate of choosing the earliest-cue option on
    Family 1, X2 = same on Family 2. Score = +1 if the subject follows
    the earliest-listed expert on BOTH families (pure position-primacy
    consultation order), -1 if it follows it on Family 1 but reverses on
    Family 2 (validity-respecting order), 0 otherwise. The metric is the
    mean score: the excess of primacy-type subjects over
    validity-respecting-type subjects.
    """
    fam1 = {frozenset((1, 4)), frozenset((2, 4)), frozenset((1, 5)),
            frozenset((1, 3)), frozenset((1, 3, 4))}
    fam2 = {frozenset((3, 4)), frozenset((3, 5))}

    def _resp(v):
        if isinstance(v, str):
            return 0 if v.strip().upper() in ("A", "0") else 1
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1

    scores = []
    for _, sdf in data.groupby("subject_id"):
        f1, f2 = [], []
        for _, row in sdf.iterrows():
            try:
                a = np.asarray(row["option_a_ratings"], dtype=int).ravel()
                b = np.asarray(row["option_b_ratings"], dtype=int).ravel()
            except (TypeError, ValueError):
                continue
            if a.size == 0 or a.size != b.size:
                continue
            diff = np.nonzero(a != b)[0]
            if diff.size == 0:
                continue
            dset = frozenset(int(j) for j in diff)
            in1 = dset in fam1
            in2 = dset in fam2
            if not (in1 or in2):
                continue
            j0 = int(diff[0])  # earliest-listed discriminating expert
            fav0 = 0 if a[j0] == 1 else 1  # option favored by that cue
            # earliest cue must oppose every later discriminating cue
            ok = True
            for j in diff[1:]:
                if (0 if a[int(j)] == 1 else 1) == fav0:
                    ok = False
                    break
            if not ok:
                continue
            r = _resp(row["response"])
            if r not in (0, 1):
                continue
            follow = 1.0 if r == fav0 else 0.0
            if in1:
                f1.append(follow)
            else:
                f2.append(follow)
        if not f1 or not f2:
            continue
        x1 = float(np.mean(f1))
        x2 = float(np.mean(f2))
        if x1 <= 0.5:
            s = 0.0
        elif x2 > 0.5:
            s = 1.0
        elif x2 < 0.5:
            s = -1.0
        else:
            s = 0.0
        scores.append(s)

    if not scores:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Candidate (simulated) value:** 0.0000 (var=0.0400)
**Other theories' values on this metric (for reference):**
- pi_6: 0.1600 (var=0.2544)
- pi_5: -0.3600 (var=0.2304)
- pi_1: -1.0000 (var=0.0000)
- pi_2: -0.1400 (var=0.2004)
- pi_3: 0.0200 (var=0.0196)
- pi_3_1: 0.0000 (var=0.0000)
- pi_3_2: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_7: 0.0000 (var=0.0000)
- pi_8: 0.0000 (var=0.0000)

### Experiment 15
**Design**
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Count-vs-Primacy planned contrast for one-reason binary-feature choice.
    #
    # For every trial we compute, directly from the rating vectors:
    #   diff[j] = +1 if expert j rates A=1,B=0 (favors A), -1 if B=1,A=0,
    #   n      = number of discriminating experts,
    #   d      = sum(diff) = (#A-favoring cues) - (#B-favoring cues),
    #   e      = diff of the EARLIEST (lowest-index) discriminating cue (+1 -> A).
    #
    # The trial weight is  w = d/n - e , i.e. the count-based choice
    # direction minus the primacy-based choice direction, and the metric is
    # the mean of w * y with y = +1 for choosing A, -1 for choosing B.
    #
    # Under the advocated momentary-diffusion theory the first consulted
    # discriminating cue is ~uniform over the discriminators, so
    # E[y] ~ s * d/n  (s = 2*p_f - 1) and the metric concentrates near
    # +s * 0.26 ~ +0.11 (tight across subjects: only binomial noise plus a
    # small s-spread; full-tie trials carry no weight and the single-cue
    # control trial gets w = 0 automatically).
    #
    # Under the competing discrete position-mixture theory the dominant
    # primacy component (mass ~0.54-0.59 at this design's ambiguity, with
    # anti/recency/TTB minorities) makes E[y] ~ s * e on essentially every
    # cell, so the metric is strongly negative (population ~ -0.28), with
    # large between-subject spread from the fixed per-subject type draw.
    # The contrast is validity-vector robust: the earliest-discriminator and
    # count structures are fixed by the trial list, and the sign of the gap
    # holds for either advertised-validity arrangement.
    df = data
    vals = []
    a_col = df["option_a_ratings"]
    b_col = df["option_b_ratings"]
    r_col = df["response"]
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.ndim != 1 or a.size == 0:
            continue
        diff = a - b
        disc = np.where(diff != 0)[0]
        n = disc.size
        if n == 0:
            # Full tie (no discriminating expert): no structural signal.
            continue
        d = float(diff.sum())            # count margin: A-cues minus B-cues
        e = float(diff[disc[0]])        # earliest discriminator: +1 -> A, -1 -> B
        w = d / float(n) - e            # count direction minus primacy direction
        y = 1.0 if int(resp) == 0 else -1.0   # response 0 = chose A
        vals.append(w * y)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.1541 (var=0.0151)
**Candidate (simulated) value:** -0.0033 (var=0.0178)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0890 (var=0.0148)
- pi_6: -0.1977 (var=0.1728)
- pi_1: 0.5590 (var=0.0378)
- pi_2: 0.4101 (var=0.0246)
- pi_3: -0.0365 (var=0.0123)
- pi_3_1: -0.0876 (var=0.0103)
- pi_3_2: -0.0896 (var=0.0137)
- pi_4: -0.0330 (var=0.0125)
- pi_5: 0.0927 (var=0.0619)
- pi_8: 0.1434 (var=0.0178)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-fixed advertised validities (positions f0..f5).
    VALIDITIES = np.array([0.80, 0.50, 0.85, 0.60, 0.90, 0.95])
    # A subject whose choice rate on primacy-vs-anti conflict trials exceeds
    # this threshold is classified as a position-primacy (earliest-listed-
    # expert) decider. 0.55 sits well above the anti-followers' ceiling
    # (~0.27-0.36 incl. the competing theory's momentary diffusion) and well
    # below the primacy-followers' level (~0.73).
    THRESHOLD = 0.55
    MIN_CONFLICT_TRIALS = 10

    def to_arr(x):
        if isinstance(x, str):
            s = x.strip().strip("[]()")
            return np.array([float(v) for v in s.replace(",", " ").split()])
        return np.asarray(x, dtype=float).ravel()

    df = data.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return 0.0

    subj = df["subject_id"].values
    resp = np.asarray(df["response"].values, dtype=float)
    a_col = df["option_a_ratings"].values
    b_col = df["option_b_ratings"].values

    is_conflict = np.zeros(n, dtype=bool)
    prim_dir = np.zeros(n, dtype=int)

    for i in range(n):
        a = to_arr(a_col[i])
        b = to_arr(b_col[i])
        if a.size != b.size or a.size != VALIDITIES.size:
            continue
        disc = a != b
        if not disc.any():
            continue
        d_idx = np.where(disc)[0]
        # Primacy decider: earliest-listed discriminating expert.
        e = int(d_idx[0])
        # Anti-validity decider: weakest advertised discriminating expert.
        anti = int(d_idx[np.argmin(VALIDITIES[d_idx])])
        pdir = 0 if a[e] > b[e] else 1
        adir = 0 if a[anti] > b[anti] else 1
        if pdir != adir:
            is_conflict[i] = True
            prim_dir[i] = pdir

    subjects = pd.unique(df["subject_id"])
    flags = []
    for s in subjects:
        m = (subj == s) & is_conflict
        cnt = int(m.sum())
        if cnt < MIN_CONFLICT_TRIALS:
            flags.append(0.0)
            continue
        rate = float(np.mean(resp[m] == prim_dir[m]))
        flags.append(1.0 if rate > THRESHOLD else 0.0)

    if not flags:
        return 0.0
    return float(np.mean(flags))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Candidate (simulated) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3000 (var=0.2100)
- pi_7: 0.0000 (var=0.0000)
- pi_1: 0.9600 (var=0.0384)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.5200 (var=0.2496)
- pi_3_1: 0.8400 (var=0.1344)
- pi_3_2: 0.7800 (var=0.1716)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0800 (var=0.0736)
- pi_8: 0.0000 (var=0.0000)

### Experiment 17
**Design**
  A=[0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # THE LEAK-DIP METRIC
    # --------------------
    # Core contrast: within-subject difference between the anti-validity
    # decider-follow rate on LEAK-ELIGIBLE trials (interior-validity
    # carrier opposed by the top-validity expert) and on CONTROL trials
    # (every other trial with a discriminating carrier).
    #
    # Fixed design: validities [0.5, 0.5, 0.75, 0.75, 1.0, 1.0], ascending
    # with display index, so the anti-validity consultation order is the
    # index order (ties are inert: tied cues never oppose in any trial).
    #
    # Advocated theory: follow probability is p_f on EVERY discriminating
    # trial regardless of carrier validity, position, margin, or coalition
    #   -> expected dip = EXACTLY 0.
    # Competing theory: on the 8 leak cells the trial-conditional
    # validity-respecting leak mixes q = min(rho_leak*0.956*0.25, 0.18)
    # (>= 0.120 forced in this design) toward the TTB winner, giving
    # follow = (1-q)*p_f + q*(1-p_f) there, while control cells stay at p_f
    #   -> expected dip = -q*(2*p_f - 1) in [-0.093, -0.050], bounded away
    #      from zero by the competing's own gate formulas.
    #
    # Works on a single subject's slice (48 leak + 42 control trials) and
    # on the pooled frame alike.

    VALIDITIES = [0.5, 0.5, 0.75, 0.75, 1.0, 1.0]
    n = len(VALIDITIES)
    vmin = min(VALIDITIES)
    vmax = max(VALIDITIES)

    def to_vec(x):
        if isinstance(x, str):
            return [int(ch) for ch in x if ch in "01"]
        return [int(v) for v in x]

    # anti-validity order: ascending validity (exact ties anchor; tied cues
    # never oppose in this design, so within-tie order is inert)
    order_anti = sorted(range(n), key=lambda j: VALIDITIES[j])
    # standard TTB order: descending validity
    order_ttb = sorted(range(n), key=lambda j: -VALIDITIES[j])

    leak_follows = []
    ctrl_follows = []

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if len(a) != n or len(b) != n:
            return 0.0

        # anti-validity decider: first strictly discriminating cue in
        # ascending-validity order
        carrier = None
        favored = None
        for j in order_anti:
            if a[j] != b[j]:
                carrier = j
                favored = 0 if a[j] > b[j] else 1
                break
        if carrier is None:
            # full-tie trial (T16): no decider, excluded
            continue

        # TTB decider: first strictly discriminating cue in
        # descending-validity order
        winner_ttb = None
        for j in order_ttb:
            if a[j] != b[j]:
                winner_ttb = 0 if a[j] > b[j] else 1
                break

        # leak-eligible cell: strictly INTERIOR carrier validity
        # (min + 0.10 < val < max - 0.10) AND the TTB winner exists and
        # OPPOSES the anti-validity winner. This is exactly the
        # competing theory's trial-conditional gate; the advocated theory
        # has no mechanism tied to this conjunction.
        interior = (vmin + 0.10) < VALIDITIES[carrier] < (vmax - 0.10)
        opposed = (winner_ttb is not None) and (winner_ttb != favored)

        follow = (int(resp) == favored)
        if interior and opposed:
            leak_follows.append(follow)
        else:
            ctrl_follows.append(follow)

    if not leak_follows or not ctrl_follows:
        return 0.0

    dip = float(np.mean(leak_follows)) - float(np.mean(ctrl_follows))
    return dip
```

**Observed (real) value:** -0.1955 (var=0.0143)
**Candidate (simulated) value:** -0.2271 (var=0.0120)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0068 (var=0.0083)
- pi_8: -0.0798 (var=0.0081)
- pi_1: -0.2999 (var=0.0105)
- pi_2: -0.2459 (var=0.0129)
- pi_3: 0.0146 (var=0.0088)
- pi_3_1: -0.0115 (var=0.0098)
- pi_3_2: -0.0243 (var=0.0112)
- pi_4: 0.0244 (var=0.0092)
- pi_5: -0.0107 (var=0.0077)
- pi_6: -0.0369 (var=0.0155)

### Experiment 18
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # TAIL HETEROGENEITY OF ANTI-VALIDITY DECIDER-FOLLOW RATES.
    #
    # Stage 1 (per subject): estimate the subject's decider-follow rate
    #   p_hat over all non-tie trials. The decider on each trial is the
    #   first discriminating expert in the subject's anti-validity
    #   consultation order: weakest advertised validity first; within the
    #   tied 0.5 block the order is positional, with the side
    #   (earlier-position-first vs later-position-first) inferred from the
    #   subject's own weak-block choices (reliable: the per-subject
    #   earlier-follow rate sits near 0.73 or 0.27 across the 60
    #   weak-block trials).
    # Stage 2 (population): the metric is the mean of the largest 20% of
    #   per-subject absolute deviations |p_hat - 0.72|, i.e. the TAIL
    #   dispersion of response discipline across subjects.
    #
    # The advocated theory's frozen parameter windows confine every
    #   subject's follow probability to [0.67, 0.78] (population SD
    #   ~0.015), so its deviation tail is set by the binomial sampling
    #   floor alone (~0.09). The competing theory samples beta over
    #   [0.5, 1.9] and epsilon over [0.08, 0.24], implying per-subject
    #   follow rates spanning [0.59, 0.84] with SD ~0.07, which lifts the
    #   tail to ~0.16.
    VALS = (0.95, 0.78, 0.5, 0.5, 0.5, 0.5)
    CENTER = 0.72
    TAIL_FRAC = 0.2

    def parse_ratings(x):
        if isinstance(x, str):
            x = x.strip().strip("[]").split(",")
        try:
            items = list(x)
        except TypeError:
            return None
        out = []
        for v in items:
            try:
                out.append(int(round(float(v))))
            except (TypeError, ValueError):
                return None
        return out if len(out) == 6 else None

    per_subject = {}
    for sid, a_raw, b_raw, r_raw in zip(
        data["subject_id"],
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = parse_ratings(a_raw)
        b = parse_ratings(b_raw)
        if a is None or b is None:
            continue
        try:
            r = int(r_raw)
        except (TypeError, ValueError):
            continue
        per_subject.setdefault(sid, []).append((a, b, r))

    devs = []
    for trials in per_subject.values():
        # --- infer the subject's weak-block positional side ---
        early_follows = []
        for a, b, r in trials:
            disc = [j for j in range(6) if a[j] != b[j]]
            if len(disc) == 2 and disc[0] >= 2:
                j1 = disc[0]
                endorsed_early = 0 if a[j1] == 1 else 1
                early_follows.append(1 if r == endorsed_early else 0)
        early_side = (
            len(early_follows) > 0
            and 2 * sum(early_follows) > len(early_follows)
        )

        # --- decider-follow rate over all non-tie trials ---
        follows = []
        for a, b, r in trials:
            disc = [j for j in range(6) if a[j] != b[j]]
            if not disc:
                continue  # full tie: no decider, no information
            if early_side:
                j_star = min(disc, key=lambda j: (VALS[j], j))
            else:
                j_star = min(disc, key=lambda j: (VALS[j], -j))
            endorsed = 0 if a[j_star] == 1 else 1
            follows.append(1 if r == endorsed else 0)

        if not follows:
            continue
        p_hat = float(sum(follows)) / float(len(follows))
        devs.append(abs(p_hat - CENTER))

    if not devs:
        return 0.0
    devs = sorted(devs, reverse=True)
    k = max(1, int(np.ceil(TAIL_FRAC * len(devs))))
    return float(sum(devs[:k]) / float(k))
```

**Observed (real) value:** 0.1556 (var=0.0024)
**Candidate (simulated) value:** 0.1496 (var=0.0027)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0753 (var=0.0006)
- pi_7: 0.1120 (var=0.0016)
- pi_1: 0.1473 (var=0.0026)
- pi_2: 0.1478 (var=0.0011)
- pi_3: 0.1944 (var=0.0016)
- pi_3_1: 0.1533 (var=0.0024)
- pi_3_2: 0.1544 (var=0.0027)
- pi_4: 0.1204 (var=0.0018)
- pi_5: 0.0989 (var=0.0011)
- pi_6: 0.1133 (var=0.0016)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is a clear improvement over both parents in several targeted places: the leak-dip signature (Exp 17: -0.2271 vs observed -0.1955, where pi_7/pi_8 sat at ~0.00/-0.08), the follow-protection tail (Exp 18: 0.1496 vs 0.1556, where pi_7/pi_8 undershot at 0.112/0.075), and the negative-signature Exp 8 (-0.2456 vs -0.2889, best of all theories). The knife-edge and dissociation experiments (Exp 3, 9, 11, 14, 16) are all well captured. However, four experiments still miss by more than 0.10, and one of them is a large REGRESSION relative to the running base: (a) Exp 15 (count-vs-primacy contrast): observed +0.1541, candidate -0.0033 \u2014 pi_8 scored 0.1434 there, so the new machinery destroyed a previously good fit. The arbiter explicitly flagged this constraint ('the mixture's average first-cue distribution must be checked against this constraint during fitting') and the proposer's rationale admits it was not checked. The ambiguity-gated primacy minority is firing on this design and dragging choices toward the earliest-cue (primacy) direction, flattening the count-leaning contrast to zero. (b) Exp 13 (anti-misbound decider follow): observed 0.15, candidate 0.2775 \u2014 still +0.128 too high; the primacy mixture helps here (pi_8 was 0.336) but not enough. Note the tension: Exp 13 wants MORE primacy mass, Exp 15 wants LESS \u2014 this cannot be resolved by a global primacy_weight knob; the gate must separate the two designs. The obvious separator is that Exp 13's validity list contains exact ties ([0.65,0.90,0.50,0.80,0.50,0.90]) while Exp 15's arrangement evidently does not, and/or that the current LINEAR ambiguity gate does not discriminate \u2014 both designs apparently get substantial w_eff. (c) Exp 7: observed -0.3107, candidate -0.1927 \u2014 still +0.118 short. Diagnosis: the leak's q = rho_leak * coherence * mean_gap starves this design, whose unique-validity gaps average only ~0.0875, so q is tiny even at rho_leak = 2.2. The raw mean_gap scaling ties leak strength to arbitrary validity-range choices across designs. (d) Exp 6: observed 0.34, candidate 0.4567 (+0.117), with Exp 4 (+0.077) and Exp 5 (+0.06) in the same direction \u2014 pooled decider-follow determinism on knife-edge cells is still systematically too high, i.e., the response-discipline widening helped the tail (Exp 18) but not the pooled mean enough. Secondary: Exp 1 under by 0.071 (0.2923 vs 0.3633) and Exp 2 over by 0.053.",
  "verdict": "regenerate",
  "rationale": "verdict='regenerate'. The candidate was accepted by the gate (it beats the prior base), but there is substantial headroom and one clear regression; one more in-family tuning pass is warranted. All fixes stay inside the prescribed Diffuse Mixture Frugality family: (1) RE-GATE THE POSITIONAL-PRIMACY MINORITY to repair Exp 15 without losing Exp 13's gain. The current linear w_eff = primacy_weight * A fires too broadly. Make the gate supra-linear or thresholded in ambiguity (e.g., w_eff = primacy_weight * max(0, (A - 0.5)/0.5) or A^2/A^3), cap w_eff at ~0.10-0.12, and \u2014 critically \u2014 add the exact-validity-tie qualifier: the reading-order fallback should elicit only on designs with tied validities and/or near-maximal binding ambiguity (Exp 13's [0.5,0.5,...,0.9,0.9] profile), not on all-distinct moderately-ambiguous lists like Exp 15's. Then re-check Exp 15 against the +0.15 constraint explicitly during fitting, as the arbiter instructed. (2) FIX THE LEAK'S GAP SCALING for Exp 7: normalize mean_gap (e.g., divide by the validity range, use mean gap relative to max gap, or use the per-trial carrier's distance to the validity extremes) so that coherent small-range designs like Exp 7 (unique gaps ~0.05-0.20) receive meaningful leak mass; simultaneously trim rho_leak toward ~1.2-1.6 under the new normalization, because Exp 17 already slightly overshoots (-0.2271 vs -0.1955) and Exp 8 is near target. (3) LOWER POOLED DETERMINISM for Exp 4/5/6 without breaking Exp 18's now-good tail: skew the response-discipline distribution rather than widening it further \u2014 e.g., shift epsilon's range up slightly ([0.08, 0.30]) or put more beta mass at the low end ([0.3, 2.2]) \u2014 and verify the Exp 18 tail (currently 0.1496 vs 0.1556, only ~0.006 of headroom) and the well-fit Exp 3/9/11 do not regress. (4) Monitor Exp 1 (under by 0.07) and Exp 2 (over by 0.05) under these changes; the noise redistribution in (3) will likely move both. These are parameter-range and gating-shape adjustments that leave the anti-validity one-reason core, the interior-gated leak, the diffuse heterogeneity, and the primacy minority all intact."
}
```

## Usage

```json
{
  "input_tokens": 37387,
  "output_tokens": 6121,
  "reasoning_tokens": 4821,
  "provider": "Baidu"
}
```
