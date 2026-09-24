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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace pi_4 with an 'Attenuated Frugality with Position-Validty Mixture Heterogeneity' theory. Core commitments: (1) Keep the one-reason cascade backbone (first strictly discriminating cue decides; flat profile over tally margins) — this is the one structural claim both current theories got right and the margin-0 signatures (Exps 3-8) support. (2) Replace the pure anti-validity hierarchy with a per-subject draw from a GRADED mixture over a small candidate set of subjective orders: anti-validity (ascending advertised validity), position-recency (later-listed experts consulted first), position-primacy (earlier-listed first), and a residual standard-TTB (descending validity) component, with mixture weights that vary smoothly with how ambiguous the validity-to-position binding is (no steep sigmoid gates). This mixture naturally produces near-chance pooled behavior on cells where the candidate orders disagree (Exps 1, 2, 11: observed 0.49, 0.53, 0.43) and consistent, attenuated tilts where they align (Exps 5, 6: observed 0.73-0.78; Exps 9, 10: strongly anti-reversed because anti-validity and recency components jointly oppose the reversed cascade there). (3) Lower and more variable response discipline: implied decider-follow probability p_f ~ 0.55-0.70 with mean ~0.62 and per-subject SD ~0.06-0.10 (both current theories overpredict contrast magnitude nearly everywhere — Exp 6: predicted 0.85-0.90 vs observed 0.78; Exp 2: 0.71 vs 0.53; Exp 12: massive overprediction by both). (4) Free per-subject tie-break as before; softmax(beta) + lapse epsilon noise; history ignored. This differs from pi_4 by abandoning the claim of a universal anti-validity hierarchy in favor of position-anchored heterogeneity with attenuated following, and differs from pi_5 by replacing its binary distrust/anti-misbound rule family and its falsified steep reversal-coherence gate with a smooth mixture that predicts chance-level performance on hierarchy-dissociation contrasts where its rivals both make extreme commitments.


## CANDIDATE THEORY
Attenuated Frugality with Position-Validity Mixture Heterogeneity. People are one-reason decision makers: cues are consulted one at a time and the first strictly discriminating cue decides, with no integration across cues (flat psychometric profile over tally margins, matching the margin-0 signatures of the knife-edge experiments). The subjective cue hierarchy is NOT universal: each subject draws a single consultation order from a graded mixture over a small candidate set: (1) anti-validity (distrust-the-weakest: consult the lowest advertised validity first) as the population-central tendency with a constant base weight; (2) position-primacy (consult experts in reading order, earliest-listed first); (3) position-recency (consult later-listed experts first, a minority variant); and (4) a residual standard-TTB component (descending validity). The position components' mixture weights grow smoothly and linearly with binding ambiguity A = 1 - |Spearman(display position, advertised validity)|, with no sigmoid gates of any kind: when the validity-to-position binding is unambiguous (monotone validity lists), the mixture collapses onto the anti-validity order (plus the small TTB residual), producing strong, consistent one-reason contrasts; when the binding is ambiguous (non-monotone lists), the position components enter, the pooled deciders disperse, and hierarchy-dissociation contrasts collapse toward chance. On top of the discrete mixture, a per-cue idiosyncratic weight distortion (zeta, fixed per subject, zero-centered) is applied, smoothly scaled by ambiguity times reversal-coherence and only when all advertised validities are distinct, so that all-distinct ambiguous designs degrade grain-by-grain while exact validity ties anchor the discrete rule mixture. Response discipline is attenuated and heterogeneous: the decider-follow probability p_f = (1-eps)*sigmoid(beta) + eps/2 spans roughly 0.55-0.85 across subjects with mean ~0.73 and SD ~0.06, so pooled contrasts are diluted both by mixture heterogeneity (disagreeing deciders) and by moderate, variable following. Validity ties are broken by a free per-subject tie-break; noise enters as softmax(beta) over the binary winner score plus an independent lapse epsilon; history is ignored (no feedback in this task).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Attenuated Frugality with Position-Validity Mixture Heterogeneity.
    #
    # One-reason decision making: cues are consulted one at a time in
    # DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues (flat psychometric profile
    # over tally margins).
    #
    # Subjective hierarchy: each subject draws ONE consultation order from
    # a graded mixture over a small candidate set:
    #   - ANTI-VALIDITY (central):  w = -validity  (distrust the weakest
    #     advertised expert; consult ascending validity)
    #   - POSITION-PRIMACY:         w = -position  (reading order;
    #     earliest-listed expert first)
    #   - POSITION-RECENCY:         w = +position  (later-listed first;
    #     minority variant)
    #   - RESIDUAL TTB:             w = +validity  (descending validity)
    # The position components' weights grow SMOOTHLY and LINEARLY with
    # binding ambiguity
    #     A = 1 - |Spearman(display position, advertised validity)|
    # with no sigmoid gates: at A ~ 0 (monotone validity lists) the mixture
    # collapses onto anti-validity plus the small TTB residual; at high A
    # the position components enter and pooled deciders disperse, pulling
    # hierarchy-dissociation contrasts toward chance.
    #
    # Per-cue idiosyncratic distortion (zeta, zero-centered, fixed per
    # subject) is applied only when all advertised validities are DISTINCT,
    # smoothly scaled by ambiguity x reversal-coherence, so all-distinct
    # ambiguous designs degrade grain-by-grain while exact validity ties
    # anchor the discrete rule mixture.
    #
    # Noise: softmax(beta) over the binary winner score mixed with an
    # independent lapse epsilon. History is ignored (no feedback).
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
    sigma_h = float(parameters["sigma_h"])
    tie_break = float(parameters["tie_break"])
    comp = float(parameters["comp"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # Graceful fallback: no per-cue distortion.
        zeta = np.zeros(n_features)

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

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    # 0 when the validity list is monotone in display position (the
    # validity-to-position binding is unambiguous -> the mixture
    # concentrates on the anti-validity distrust order); growing with
    # position-validity conflict (ambiguous binding -> position-anchored
    # reading orders enter the mixture).
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: distrust order undefined -> maximal
            # idiosyncrasy.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- reversal coherence: |Spearman(validity, validity[::-1])| ----
    # Used ONLY as a smooth (gate-free) multiplier on the per-cue zeta
    # distortion: idiosyncratic scrambling is strongest when the binding is
    # ambiguous AND the list structure is reversal-coherent enough that
    # small perturbations reshuffle the subjective order.
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

    # ---- graded mixture weights over the candidate subjective orders ----
    # Smooth in ambiguity; NO sigmoid gates. The per-subject draw `comp` ~
    # U(0,1) selects one component via the cumulative normalized weights.
    w_anti = 0.45                  # anti-validity: constant central mass
    w_rec = 0.06 * ambiguity       # position-recency: minority variant
    w_prim = 0.80 * ambiguity      # position-primacy: reading-order default
    w_ttb = 0.04                   # residual standard TTB
    z_tot = w_anti + w_rec + w_prim + w_ttb
    c_anti = w_anti / z_tot
    c_rec = c_anti + w_rec / z_tot
    c_prim = c_rec + w_prim / z_tot

    if comp < c_anti:
        # Distrust-the-weakest: anti-validity subjective weights.
        w_base = -val
    elif comp < c_rec:
        # Position-recency: later-listed experts consulted first.
        w_base = np.arange(n_features, dtype=float)
    elif comp < c_prim:
        # Position-primacy: earlier-listed experts consulted first.
        w_base = -np.arange(n_features, dtype=float)
    else:
        # Residual standard TTB: descending advertised validity.
        w_base = val

    # ---- per-cue idiosyncratic distortion (distinct lists only) ----
    # Exact validity ties anchor the discrete rule mixture (no per-cue
    # noise); all-distinct lists degrade grain-by-grain, smoothly scaled
    # by ambiguity x reversal-coherence.
    distinct = (np.unique(val).shape[0] == n_features)
    zeta_scale = sigma_h * ambiguity * coherence if distinct else 0.0

    w = w_base + zeta_scale * zeta

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # per-subject tie_break parameter: 0 -> earlier feature position
    # first, 1 -> later feature position first among equally weighted cues.
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, -w))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). Implied decider-follow
    # probability p = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.73 with
    # per-subject SD ~0.06 over the sampled (beta, epsilon) ranges:
    # attenuated and variable, but high enough to reproduce the strong
    # margin-0 one-reason signatures of the monotone-validity designs.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.85, 1.8]
- epsilon: [0.10, 0.22]
- sigma_h: [0.4, 1.4]
- zeta: [(-1, 1)] * n_features
- tie_break: {0, 1}
- comp: [0, 1]
- validities: validities

`rationale`:
STRUCTURE (faithful to the arbiter's prescription). (1) The one-reason cascade backbone is kept: first strictly discriminating cue decides, flat over tally margins — the one structural claim both prior theories got right and the margin-0 signatures support. (2) The pure anti-validity hierarchy is replaced by a per-subject categorical draw from a GRADED mixture over {anti-validity, position-recency, position-primacy, residual TTB}, with weights that vary smoothly and linearly with binding ambiguity A = 1 - |Spearman(position, validity)| — no steep sigmoid gates anywhere (the only sigmoid in the model is the response softmax). In monotone designs (A ~ 0) the mixture collapses to anti-validity + a ~8% TTB residual; in ambiguous designs the position components enter, producing near-chance pooled behavior on hierarchy-dissociation cells (Exps 11, 12, 9) and attenuated tilts where orders align. (3) Response discipline is attenuated and heterogeneous: p_f spans ~0.55-0.85 with mean ~0.73 and per-subject SD ~0.06. (4) Free per-subject tie-break, softmax(beta) + lapse epsilon, history ignored, and a smooth ambiguity-times-coherence-scaled per-cue zeta distortion (distinct validities only) that replaces pi_5's steeply gated scrambling.

DELIBERATE DEVIATIONS FROM THE ARBITER'S NUMBERS, WITH EVIDENCE. (a) p_f mean ~0.62 was rejected: the margin-0 knife-edge signatures pin the effective decider-follow at ~0.70-0.73 (e.g., the Exp-4 contrast metric equals 4q-2 for any one-reason model, so observed 0.779 implies q ~ 0.695; the Exp-3 reversed-cascade follow implies q ~ 0.727). Setting p_f ~ 0.62 would underpredict those experiments by >0.2 each — worse than either current theory. The attenuation the arbiter wants is instead delivered where it is actually needed (pooled contrasts in ambiguous designs) by mixture dilution and the widened p_f spread, not by a uniformly low p_f. (b) Within the position family, primacy is weighted far more heavily than recency (0.80A vs 0.06A). Trial-by-trial analysis of the strongly anti-reversed experiments shows the arbiter's premise that 'recency jointly opposes the reversed cascade' there is factually inverted: in those designs the later-listed experts are the high-validity ones, so a recency-first order AGREES with the reversed/TTB-like cascade and pulls the pooled metric toward zero, away from the observed strongly negative values. The position component that genuinely opposes the reversed cascade in those experiments is primacy (reading order, which lands on the low-validity early features). Primacy-dominance reproduces the strong negative anti-reversed contrasts while recency is retained as a minority variant to serve the designs where it helps. (c) The per-cue zeta is retained from the running-best theory (pi_5) but with its steep reversal-coherence gate replaced by a smooth product moderator (ambiguity x coherence), honoring the 'no steep gates' instruction while preserving the grain-by-grain degradation that pi_5's zeta contributed in distinct-validity ambiguous designs.

EXPECTED IMPROVEMENTS over the running best (pi_5, score 0.849), which overpredicts contrast magnitude nearly everywhere: the ~8% TTB residual in monotone designs lowers the Exp-4 contrast from pi_5's 0.90 toward the observed 0.78 and the Exp-6 knife-edge contrast from ~0.50 toward the observed 0.34, while raising the Exp-1 level toward its observed 0.36; in ambiguous designs the position mixture plus smooth zeta pulls the Exp-12 follow rate down from 0.59 toward 0.53, the Exp-10 conformity index down from 184 toward ~117, and the Exp-11 rate toward chance, while the anti+primacy alignment preserves the strongly negative anti-reversed contrasts and the Exp-5 follow rate. The mixture is experiment-invariant by construction: n_features and validities are read symbolically, all weights are smooth functions of rank statistics of the validity list, and the theory reduces to Weakest-Expert-First Frugality in the unambiguous-binding limit.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0866 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0866.

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
**Candidate (simulated) value:** 0.2837 (var=0.0230)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8479 (var=0.0111)
- pi_2: 0.0323 (var=0.0195)
- pi_3: 0.4099 (var=0.0092)
- pi_3_1: 0.2933 (var=0.0063)
- pi_3_2: 0.3314 (var=0.0076)
- pi_4: 0.3205 (var=0.0077)
- pi_5: 0.2898 (var=0.0076)

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
**Candidate (simulated) value:** 0.1054 (var=0.0356)
**Other theories' values on this metric (for reference):**
- pi_2: 1.3208 (var=0.1532)
- pi_1: -0.1621 (var=0.0229)
- pi_3: 0.0608 (var=0.0436)
- pi_3_1: 0.1208 (var=0.0447)
- pi_3_2: 0.0708 (var=0.0422)
- pi_4: 0.0400 (var=0.0527)
- pi_5: 0.0462 (var=0.0368)

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
**Candidate (simulated) value:** 0.4329 (var=0.0559)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1947 (var=0.0151)
- pi_2: -0.1742 (var=0.0091)
- pi_1: -0.4209 (var=0.0150)
- pi_3_1: 0.4276 (var=0.0099)
- pi_3_2: 0.3960 (var=0.0138)
- pi_4: 0.3751 (var=0.0163)
- pi_5: 0.4636 (var=0.0174)

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
**Candidate (simulated) value:** 0.8267 (var=0.3708)
**Other theories' values on this metric (for reference):**
- pi_2: -0.6893 (var=0.0556)
- pi_3: 0.3867 (var=0.0818)
- pi_1: -1.3413 (var=0.2259)
- pi_3_1: 0.8027 (var=0.0699)
- pi_3_2: 0.6893 (var=0.0735)
- pi_4: 0.8467 (var=0.0843)
- pi_5: 0.8960 (var=0.0975)

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
**Candidate (simulated) value:** 0.5367 (var=0.0403)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.7178 (var=0.0092)
- pi_2: 0.5072 (var=0.0084)
- pi_1: 0.1383 (var=0.0146)
- pi_3: 0.6261 (var=0.0070)
- pi_3_2: 0.6633 (var=0.0068)
- pi_4: 0.6883 (var=0.0074)
- pi_5: 0.6311 (var=0.0430)

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
**Candidate (simulated) value:** 0.3800 (var=0.1448)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0367 (var=0.0470)
- pi_3_1: 0.3933 (var=0.0286)
- pi_1: -0.7033 (var=0.0409)
- pi_3: 0.2250 (var=0.0465)
- pi_3_2: 0.3950 (var=0.0319)
- pi_4: 0.3633 (var=0.0488)
- pi_5: 0.4983 (var=0.0354)

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
**Candidate (simulated) value:** -0.2553 (var=0.0269)
**Other theories' values on this metric (for reference):**
- pi_3_2: 0.3613 (var=0.0180)
- pi_2: -0.0100 (var=0.0118)
- pi_1: 0.0160 (var=0.0107)
- pi_3: 0.2040 (var=0.0089)
- pi_3_1: 0.4007 (var=0.0109)
- pi_4: -0.1453 (var=0.0099)
- pi_5: -0.2327 (var=0.0163)

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
**Candidate (simulated) value:** -0.2133 (var=0.0194)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0011 (var=0.0175)
- pi_3_2: 0.1878 (var=0.0164)
- pi_1: 0.3422 (var=0.0196)
- pi_3: 0.0944 (var=0.0115)
- pi_3_1: 0.2311 (var=0.0115)
- pi_4: -0.1644 (var=0.0172)
- pi_5: -0.2356 (var=0.0097)

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
**Candidate (simulated) value:** 0.4280 (var=0.0172)
**Other theories' values on this metric (for reference):**
- pi_3_2: 0.6607 (var=0.0064)
- pi_4: 0.3193 (var=0.0078)
- pi_1: 0.8787 (var=0.0117)
- pi_2: 0.5133 (var=0.0067)
- pi_3: 0.6033 (var=0.0089)
- pi_3_1: 0.7033 (var=0.0066)
- pi_5: 0.3993 (var=0.0160)

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
**Candidate (simulated) value:** 261.7889 (var=0.0106)
**Other theories' values on this metric (for reference):**
- pi_4: 588.7556 (var=0.0036)
- pi_3_2: 536.1222 (var=0.0024)
- pi_1: -680.7000 (var=0.0054)
- pi_2: 105.3778 (var=0.0026)
- pi_3: 332.3111 (var=0.0024)
- pi_3_1: 567.5556 (var=0.0025)
- pi_5: 184.2778 (var=0.0153)

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
**Candidate (simulated) value:** 0.4444 (var=0.0059)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4262 (var=0.0087)
- pi_4: 0.5215 (var=0.0033)
- pi_1: 0.3595 (var=0.0039)
- pi_2: 0.4626 (var=0.0037)
- pi_3: 0.5290 (var=0.0024)
- pi_3_1: 0.5915 (var=0.0016)
- pi_3_2: 0.5703 (var=0.0038)

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
**Candidate (simulated) value:** 0.6024 (var=0.0093)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7076 (var=0.0076)
- pi_5: 0.5895 (var=0.0200)
- pi_1: 0.3476 (var=0.0042)
- pi_2: 0.7433 (var=0.0076)
- pi_3: 0.5343 (var=0.0067)
- pi_3_1: 0.5810 (var=0.0066)
- pi_3_2: 0.5752 (var=0.0040)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate was ACCEPTED by the gate (loss 0.0866, the new running best) and the per-experiment profile supports that verdict. Directionally, the model is correct on ALL 12 experiments \u2014 every sign and ordinal pattern matches the real data: the reversed-cascade follow (Exp 3: 0.433 vs 0.453), the knife-edge contrasts (Exp 4: 0.827 vs 0.779; Exp 6: 0.380 vs 0.340), the anti-reversed signatures (Exp 7: -0.255 vs -0.311; Exp 8: -0.213 vs -0.289), an exact hit on the dissociation rate (Exp 9: 0.428 vs 0.428), near-chance behavior on hierarchy-dissociation cells (Exp 11: 0.444 vs 0.490), and attenuated tilts on the ambiguous designs (Exps 1, 2, 12 all within ~0.08 of observed). This is the only theory in the comparison set that is simultaneously at or near the best value on the knife-edge experiments (3, 4, 6, 9) AND in the right neighborhood on the ambiguous-binding experiments (1, 2, 11, 12) \u2014 the previous theories each failed one side or the other. The proposer's two documented deviations from the arbiter's numbers are empirically justified: keeping p_f ~0.73 rather than 0.62 is pinned by the Exp-3/Exp-4 knife-edge identities (4q-2 = 0.78 implies q \u2248 0.70), and primacy-dominance over recency is required to reproduce the strongly negative anti-reversed contrasts in Exps 7-8 (a recency-heavy mixture would have pulled those toward zero). The mechanism family prescribed by the arbiter (one-reason cascade + graded position-validity mixture + attenuated following) is faithfully implemented, with no sigmoid gates outside the response softmax. Two residual weaknesses remain, neither disqualifying: (a) Exp 10's conformity-dispersion index is still ~2.2x the observed value (262 vs 117) \u2014 the pooled decider-follow on hierarchy-agreeing cells in that highly ambiguous design (A \u2248 0.86) is still too concentrated; and (b) between-subject variances are overpredicted on several experiments (Exp 4: 0.37 vs 0.06; Exp 6: 0.14 vs 0.03; Exp 3: 0.056 vs 0.021), indicating the categorical comp draw plus the wide beta range yields more cross-subject spread than humans show. However, pushing p_f down to fix Exp 10 would break the Exp-3/Exp-4 pins, and pushing position weights up would move Exps 11/12 in opposite, mutually offsetting directions \u2014 the trade-off surface is flat-to-adverse there, so further tuning risks regressing the many well-fit experiments to chase one index whose best-fitting rival (pi_2, 105) fails nearly everywhere else.",
  "verdict": "continue",
  "rationale": "The candidate is the first theory in this loop to capture the full cross-experimental pattern with one experiment-invariant mechanism: the one-reason cascade backbone reproduces every margin-0 knife-edge signature (Exps 3, 4, 6, 8), while the ambiguity-graded position-validity mixture collapses correctly onto the anti-validity order in monotone designs and dilutes pooled contrasts toward chance in ambiguous designs (Exps 1, 2, 11, 12) \u2014 all with point estimates within ~0.08 of observed and an exact match on Exp 9. The aggregate loss (0.0866) is the running best and was accepted by the gate. The remaining misfits (Exp 10's conformity index at ~2.2x observed; overdispersed between-subject variances on Exps 3, 4, 6) are real but sit on an adverse trade-off surface: lowering follow probability or further boosting position weights to fix Exp 10 would demonstrably break the knife-edge experiments that pin q \u2248 0.70 and would push Exps 11/12 in opposite directions. Given that the qualitative structure is correct everywhere, the mechanism family is faithfully implemented, and the loss is the best achieved, the theory should be kept as-is. If future iterations revisit it, the highest-value in-family refinements would be (i) modestly increasing the position-mixture weight specifically at very high ambiguity (Exp 10's regime) while holding the monotone-design collapse fixed, and (ii) narrowing the beta range or softening the categorical comp draw into a per-trial mixture to reduce the overdispersed between-subject variance \u2014 but these are polish, not grounds for regeneration."
}
```

## Usage

```json
{
  "input_tokens": 25135,
  "output_tokens": 4064,
  "reasoning_tokens": 2566,
  "provider": "Parasail"
}
```
