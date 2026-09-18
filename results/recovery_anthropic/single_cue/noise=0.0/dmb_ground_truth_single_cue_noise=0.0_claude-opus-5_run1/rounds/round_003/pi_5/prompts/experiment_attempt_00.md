# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** **Odd-Cue-Out Anchoring under Label Individuation (OCO-LI).**

Four claims, all about *which single row is allowed to anchor the comparison*, and about *whether the anchoring competition can run at all*.

1. **One discriminating row decides, and which one is stochastic.** On each trial the subject settles the A-vs-B comparison by anchoring on a SINGLE row on which the two products differ (a one-reason rule). No summation of evidence occurs, so a validity-weighted integrator is never the right description. Which row wins the anchoring competition varies from trial to trial (not fixed within a subject), so choices are graded within a subject and per-subject metric variance stays essentially binomial — exactly what the population variances in this corpus show.

2. **Anchor salience = ODDNESS, not diagnosticity.** The probability that row j is the anchor is proportional to u_j = exp(lam * j) * (1 - v_j)^delta.
   * *Odd-cue-out (delta > 0):* a disagreement located on a near-chance expert (v ≈ .53) is experienced as a strange, idiosyncratic fact that must be adjudicated, whereas a disagreement on a near-certain expert (v ≈ .97) is discounted as a mere echo of the general impression the subject already has. Communicated validity is therefore used with INVERTED sign. This is why choice is systematically *sub-chance* with respect to stated validity in every design where the labels are usable.
   * *Reading-order recency (lam > 0):* rows are read left-to-right and the later-read discrepancies are the ones still active in working memory when the response is emitted, so they capture the anchor more often. Position is a genuine second determinant (it is required by the 1-cue-late-versus-3-cue-early coalitions), but it is subordinate to, and — crucially — gated by, claim 3.

3. **Label-ambiguity gating (the new mechanism).** The anchoring competition presupposes that each expert can be *individuated*: the subject must be able to bind a particular stated validity to a particular column. That binding is only available when the stated validities are mutually discriminable. Define, for each expert j, its isolation d_j = min_{k≠j} |v_j - v_k| and its individuation s_j = min(1, d_j / tau); the design-level individuation is phi = mean_j s_j. With probability phi the subject runs the odd-cue-out competition above; with probability 1 - phi the columns are interchangeable in memory, no row is 'the odd one', every discriminating row is equally likely to become the anchor, and the choice probability collapses to the *proportion* of discriminating rows favouring each option (a probability-matched tally). When validity labels are duplicated or tightly clustered (e.g. [.95,.70,.55,.95,.70,.55]) phi → 0, the anti-validity and recency effects switch OFF, and a two-row conflict is decided by a coin flip. When all labels are distinct (gaps ≥ .05) phi → 1 and the anti-validity/recency anchoring runs at full strength.

4. **Conflict interference.** The more rows m on which the products differ, the more competing discrepancies must be juggled and the more often the subject loses the thread and simply guesses: P(guess) = 1 - (1 - eps0) * rho^(m-2). Two-row comparisons are made nearly cleanly; five- and six-row comparisons drift toward chance. Difficulty grows with the *number* of discrepancies, not with their diagnosticity — which is why unanimous (dominance) trials are far from ceiling and get worse as more experts speak.

The rule is stationary: there is no feedback, hence no learning across the block.

*Sharp falsifiable contrasts:* (i) hold cue content fixed and only duplicate versus de-duplicate the stated validity vector — OCO-LI predicts the anti-validity effect switches off and on; any recency-only or validity-integration account predicts no change. (ii) With all-distinct labels, put the low-validity discriminating row EARLY and the high-validity row LATE: OCO-LI predicts a near-even split (oddness and recency now oppose each other), whereas a pure recency account predicts the late high-validity option and a Bayesian account predicts it even more strongly.

**Parameters:**
- lam: [1.50, 1.80]
- delta: [0.82, 0.98]
- eps_base: [0.25, 0.31]
- rho: [0.765, 0.82]
- tau: [0.02, 0.045]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Odd-Cue-Out Anchoring under Label Individuation (OCO-LI)
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            flat = arr.reshape(2, -1)
            a, b = flat[0].ravel(), flat[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.995)

    # ---------- parameters ----------
    lam = float(parameters['lam'])
    delta = float(parameters['delta'])
    eps0 = float(np.clip(float(parameters['eps_base']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    tau = float(max(float(parameters['tau']), 1e-6))

    # ---------- label individuation (design-level gate) ----------
    if n <= 1:
        phi = 1.0
    else:
        D = np.abs(v.reshape(-1, 1) - v.reshape(1, -1))
        np.fill_diagonal(D, np.inf)
        d_min = np.min(D, axis=1)
        s = np.clip(d_min / tau, 0.0, 1.0)
        phi = float(np.mean(s))
    phi = float(min(max(phi, 0.0), 1.0))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])

    fav_a = diff[disc] > 0.0

    # (i) odd-cue-out anchoring: salience = recency x oddness
    j = disc.astype(float)
    log_u = lam * j + delta * np.log(np.clip(1.0 - v[disc], 1e-9, None))
    log_u = log_u - np.max(log_u)
    u = np.exp(log_u)
    tot = float(np.sum(u))
    if (not np.isfinite(tot)) or tot <= 0.0:
        p_anchor_a = 0.5
    else:
        p_anchor_a = float(np.sum(u[fav_a])) / tot

    # (ii) ungated fallback: all discriminating rows equally likely to anchor
    p_flat_a = float(np.sum(fav_a)) / float(m)

    p_core_a = phi * p_anchor_a + (1.0 - phi) * p_flat_a

    # ---------- conflict interference ----------
    expo = max(m - 2, 0)
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS).**

Three claims, all about *which single cue gets to decide*, not about integration.

1. **One cue decides, but which one is stochastic.** On every trial the subject settles the comparison on a SINGLE discriminating expert row (a one-reason rule). Which row wins the competition for the decision varies from trial to trial, so choice is graded *within* a subject (this is required by the data: per-subject metric variances are essentially binomial, i.e. subjects are homogeneous but individually stochastic, ruling out fixed per-subject cue orders). The deciding row j is drawn from the discriminating set D with probability proportional to a salience weight u_j.

2. **Salience = serial recency × distinctiveness (redundancy-discounted validity).** u_j = exp(lambda*j) * (1 - v_j)^delta.
   * *Recency (lambda > 0):* rows are read top-to-bottom / left-to-right and the later-read rows are the ones still active in working memory when the choice is made, so they capture the decision. Column position, not stated diagnosticity, is the dominant driver (this is the arbiter's core claim, retained).
   * *Distinctiveness (delta > 0):* subjects treat a near-certain expert (v = .97) as *redundant* — "I already know what he will say, he just echoes the consensus" — whereas a row on which a near-chance expert (v = .53) discriminates feels like idiosyncratic, decision-relevant information that must be adjudicated. Communicated validity is therefore used, but with inverted sign: high-validity rows are discounted as uninformative repeats of the general impression. This is why choices are systematically *sub-chance* with respect to validity in every experiment.

3. **Conflict interference.** The more rows on which the two products differ, the more competing comparisons must be held simultaneously, and the more often the subject loses the thread and simply guesses: P(guess) = 1 - (1-eps0)*rho^(m-2), where m = |D|. Two-row comparisons are made almost cleanly (hence near-deterministic anti-validity choice when only a .97 row and a .53 row differ); five- and six-row comparisons collapse toward coin-flipping. Difficulty grows with the number of differences, not with their diagnosticity.

No learning, no feedback use: the rule is stationary across the block.

*Why the two extra mechanisms are needed on top of pure position-recency (justified deviation from the arbiter's exact form):* pure position models (graded recency Luce, or its Take-The-Last limit) plus a uniform lapse are mathematically unable to reproduce the observed ordering Exp4 (0.15) < Exp1 (0.28) ≈ Exp3 (0.29): Exp4's diagnostic trial is an ADJACENT single-pair contrast (column 1 vs column 2), whose raw position-based prediction (1/(1+e^lambda)) is always *higher* (closer to chance) than the multi-cue Exp1/Exp3 trials, so with a shared lapse floor Exp4 can never come out lowest. Distinctiveness fixes this (a .97-vs-.53 contrast is the most lopsided contrast in the whole corpus). Likewise pure position forces Exp2's tie-trial term to exactly 0.5 and its 1-vs-3/1-vs-5 coalition term to ~2/3, capping the Exp2 metric at 0.167 only in the noiseless limit; distinctiveness lowers the tie term (the .93 row loses to the .60 row that precedes it) so that the observed 0.174 is reachable with realistic noise. Conflict interference then supplies the trial-type-dependent floor that lets Exp1/Exp3 sit at ~0.28 while Exp4 sits at ~0.15.

**Parameters:**
- lam: [1.4, 1.8]
- delta: [0.8, 1.0]
- eps_base: [0.24, 0.32]
- rho: [0.73, 0.83]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            flat = arr.reshape(2, -1)
            a, b = flat[0].ravel(), flat[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.995)

    # ---------- parameters ----------
    lam = float(parameters['lam'])
    delta = float(parameters['delta'])
    eps0 = float(parameters['eps_base'])
    rho = float(parameters['rho'])
    eps0 = float(np.clip(eps0, 0.0, 1.0))
    rho = float(np.clip(rho, 1e-6, 1.0))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])

    # salience of each discriminating row:
    #   recency  exp(lam * column_index)
    #   distinctiveness  (1 - validity) ** delta   (redundancy discounting)
    j = disc.astype(float)
    log_u = lam * j + delta * np.log(np.clip(1.0 - v[disc], 1e-9, None))
    log_u = log_u - np.max(log_u)          # numerically stable
    u = np.exp(log_u)

    tot = float(np.sum(u))
    if not np.isfinite(tot) or tot <= 0.0:
        p_core_a = 0.5
    else:
        s_a = float(np.sum(u[diff[disc] > 0.0]))
        p_core_a = s_a / tot

    # ---------- conflict interference: guess rate grows with |D| ----------
    expo = max(m - 2, 0)
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with clearly spread, strictly ordered validities (0.95, 0.85, 0.75, 0.65, 0.55) so the TTB cue cascade is unambiguous and matches feature index order. The 12 unique pairs (repeated 8x each = 96 trials) are built to dissociate one-reason decision making from unit-weight tallying along three quantitative dimensions:

(1) CONFLICT MAGNITUDE. Pairs 1-4 pit the single highest-validity cue against a coalition of lower cues, varying the tally margin from 4 (A=[1,0,0,0,0] vs B=[0,1,1,1,1]) to 1 (A=[1,0,0,0,0] vs B=[0,1,1,0,0]). TTB predicts identical choice probabilities across these pairs (always the option winning cue 1, with p fixed by beta and epsilon), whereas Tallying predicts an ordered gradient: near-deterministic choice of the coalition option at margin 4 and a shallower, beta-scaled preference at margin 1. This is a quantitative signature, not merely a directional one: TTB is flat over margins, Tallying is monotone increasing.

(2) TALLY-TIE TRIALS. Pairs 5-6 (A=[1,0,1,0,0] vs B=[0,1,0,1,0] and its mirror) give each option exactly two feature wins, so Tallying must guess at exactly 50/50 regardless of its parameters, while TTB is fully decisive in favour of the option winning cue 1. Any systematic deviation from chance here cannot be produced by Tallying at all, making these trials a parameter-free falsification test.

(3) CASCADE POSITION. Pairs 9-11 place the first discriminating cue lower in the cascade (cue 2 after a tie on cue 1; cue 3 after ties on cues 1-2) while the remaining lower cues favour the other option. TTB still predicts the same high, constant probability for the option winning the first discriminating cue; Tallying predicts the opposite choice. If subjects' accuracy on the discriminating-cue option degrades as that cue moves down the cascade, that also constrains the subjective-validity assumption inside TTB.

Pairs 7-8 are agreement controls where both heuristics select the same option, giving a baseline for lapse rate (epsilon) and beta that is uncontaminated by the model contrast. Every pair and its mirror image appear so that the correct response is balanced across the A and B keys for both models, ruling out side bias as an explanation. No pair is degenerate: on every trial at least one model is decisive, and on 10 of 12 the models disagree either in direction or in predicted confidence.

[1] GOAL: dissociate one-reason decision making (TTB) from unit-weight vote counting (Tallying) using (i) a validity order that is deliberately NOT the display order, (ii) a tally-margin gradient over which TTB is flat, (iii) parameter-free tally-tie trials, and (iv) cascade-depth manipulations.

VALIDITIES: six experts shown in display order with validities [0.60, 0.93, 0.55, 0.87, 0.70, 0.78]. The descending cue cascade is therefore feature index 1 (.93) > 3 (.87) > 5 (.78) > 4 (.70) > 0 (.60) > 2 (.55). Because the top cue is Expert 2 rather than Expert 1, any subject who follows a genuine validity cascade must reorder the display, whereas Tallying is completely invariant to the permutation. This also guards against a confound present when validities descend with index: 'use the leftmost cue' can no longer masquerade as TTB.

16 unique pairs, each repeated 6x (96 trials), every pair present together with its A/B mirror so the model-predicted response is exactly balanced across keys (no side-bias explanation).

(1) TALLY-MARGIN GRADIENT (pairs 1-6). One option wins only the top cue (index 1); the other wins a coalition of lower-validity cues, with the tally margin escalating 1 -> 2 -> 4: [0,1,0,0,0,0] vs [1,0,1,0,0,0], vs [1,0,1,0,1,0], vs [1,0,1,1,1,1]. TTB is decisive for the top-cue option with the SAME probability sigmoid(beta)-scaled value on all three, i.e. flat across margins. Tallying reverses direction (choosing the coalition option) and its confidence increases monotonically with margin, from a shallow beta-scaled preference at margin 1 to near-determinism at margin 4. Flat-vs-monotone over margin is a quantitative, not merely directional, signature.

(2) PARAMETER-FREE TALLY TIES (pairs 7-10). Pair 7: [0,1,0,0,0,0] vs [1,0,0,0,0,0] — each option wins exactly one cue, so Tallying MUST be exactly 50/50 for any beta and epsilon, while TTB is decisive for A (cue 1, .93, beats cue 0, .60). Note the naive 'read left to right' strategy predicts the opposite option here, so this pair simultaneously falsifies Tallying and index-order pseudo-TTB. Pair 9: [0,0,1,1,0,0] vs [0,0,0,0,1,1] — a 2-2 tally tie in which TTB's winner is set by a mid-cascade comparison (cue 3, .87, vs cues 5/4), again forcing Tallying to chance while TTB is decisive for A.

(3) CASCADE DEPTH WITH CONSTANT TALLY (pairs 11-14). Pair 11 ([0,0,0,0,0,1] vs [1,0,1,0,1,0]) has the identical tally structure to pair 3 (1 vs 3) but the first discriminating cue sits third in the cascade (index 5) after two ties; pair 13 ([0,0,0,0,1,0] vs [1,0,1,0,0,0]) matches pair 1's margin of 1 but the discriminator is fourth in the cascade (index 4). Tallying predicts exactly the same probabilities as its margin-matched counterparts, while TTB predicts the same high probability for the discriminating-cue option regardless of depth. Any systematic decay in choosing the discriminating-cue option as it moves down the cascade would violate both models as specified and constrain subjective validities inside TTB.

(4) AGREEMENT BASELINE (pairs 15-16). [1,1,1,0,1,0] vs [0,0,0,1,0,1]: the top cue and the tally (4-2) point the same way, so both models predict the same option. These trials identify epsilon and beta free of the model contrast, so the contrast trials can be read as genuine model divergence rather than overall noise.

No pair is degenerate: on 14 of 16 pairs the models disagree in direction, and on the remaining agreement pairs they differ in nothing but supply the noise baseline.

[2] GOAL: dissociate Amplified-Validity Additive Integration (AVAI) from unit-weight Tallying. Both models are permutation-invariant over cue position, so the crucial contrasts are (i) trials where the WIN COUNT is tied but the validity-weighted margin varies parametrically, and (ii) trials where the win count and the weighted sum point in OPPOSITE directions or where a large count margin corresponds to a near-zero weighted margin.

VALIDITIES: six experts in a deliberately non-monotone display order, [0.68, 0.96, 0.52, 0.88, 0.58, 0.78]. Ranked: H1 = idx1 (.96), H2 = idx3 (.88), M = idx5 (.78), L1 = idx0 (.68), L2 = idx4 (.58), L3 = idx2 (.52). Scrambling display order removes any 'read leftmost cue' artifact. With gamma = 2 the AVAI weights are H1=.922, H2=.774, M=.608, L1=.462, L2=.336, L3=.270 \u2014 all the same order of magnitude, exactly the compressed regime AVAI posits.

16 unique pairs (8 base structures, each with its A/B mirror so the predicted response is exactly key-balanced for both models), 6 repetitions each = 96 trials.

(1) TALLY-TIE LADDER (pairs 1-4, plus mirrors). Every one of these is a 2-2 win tie, so Tallying is EXACTLY 50/50 for any beta and epsilon ">AVAI, in contrast, predicts a graded, monotone ladder of choice probabilities because the logistic acts on the continuous weighted difference:
  \u2022 Pair 1 {H1,H2} vs {L2,L3}: d = +1.09 \u2192 p(A) \u2248 .99 (near-deterministic).
  \u2022 Pair 2 {H1,L1} vs {M,L3}: d = +0.51 \u2192 p(A) \u2248 .95.
  \u2022 Pair 3 {H2,L1} vs {M,L2}: d = +0.29 \u2192 p(A) \u2248 .85.
  \u2022 Pair 4 {H1,L3} vs {H2,L1}: d = -0.045 \u2192 p(A) \u2248 .47 (indistinguishable from chance).
A flat 50% profile across all four falsifies AVAI and confirms Tallying; a monotone spread from ~.99 down to ~.50 with the SAME tally structure cannot be produced by Tallying under any parameter setting. Pair 4 additionally pins gamma: the sign of d for {H1,L3} vs {H2,L1} flips from B-favoring to A-favoring near gamma \u2248 2.7, so it is the most gamma-diagnostic trial in the set.

(2) COUNT-VS-WEIGHT CONFLICT LADDER (pairs 5-7, plus mirrors). One option holds only the single best cue (or the two best); the other holds a coalition of weak cues, with the coalition's tally margin escalating 1 \u2192 2 \u2192 3 while the weighted margin sweeps through zero:
  \u2022 Pair 5: A={H1} vs B={L2,L3}. Tallying: B wins by 1 (p(A) < .5, beta-scaled). AVAI: d = +0.31 \u2192 p(A) \u2248 .87. DIRECTIONAL REVERSAL \u2014 the two models predict opposite modal responses.
  \u2022 Pair 6: A={H1,H2} vs B={M,L1,L2,L3}. Tallying: B wins by 2, i.e. MORE confident than pair 5. AVAI: d = +0.02 \u2192 p(A) \u2248 .50, i.e. maximally noisy. A large count margin producing chance-level behaviour is impossible for Tallying and is the signature 'big count margin, small weighted margin' prediction of AVAI.
  \u2022 Pair 7: A={H1} vs B={L1,L2,L3}. Tallying: B wins by 3 (near-deterministic). AVAI: d = -0.15 \u2192 p(A) \u2248 .29 (same direction, much weaker). Across pairs 5-7 Tallying predicts a monotone increase in coalition choice (.5+ \u2192 strong \u2192 near-1) driven purely by margin, whereas AVAI predicts the coalition being chosen at rates .13, .50, .71 \u2014 crossing chance at margin 2 rather than at margin 0. Flat-margin-crossing location is a quantitative, parameter-robust separator.

(3) AGREEMENT BASELINE (pair 8, plus mirror). A={H1,H2,L1,M} vs B={L2,L3}: tally 4-2 and weighted margin both strongly favour A. Both models predict the same near-deterministic response, which identifies epsilon (lapse rate) and the overall beta scale independently of the contrast trials, so deviations on pairs 1-7 can be attributed to model structure rather than global noise.

No pair is degenerate: on 6 of the 8 base structures the models disagree either in direction or by more than 30 percentage points in predicted probability, and the four tie trials constitute a parameter-free test that Tallying can only pass by producing exactly 50% on all of them.

[3] GOAL: dissociate unit-weight Tallying (advocated) from Amplified-Validity Additive Integration (AVAI). Both models are permutation-invariant over display position, so the decisive manipulation is to hold the WIN COUNT constant while sweeping the validity-weighted margin across its whole range, and then to break the rank-order between count margin and weighted margin.

VALIDITIES: six experts in deliberately scrambled display order, [0.58, 0.97, 0.53, 0.80, 0.66, 0.90]. Ranked: H1 = idx1 (.97), H2 = idx5 (.90), M = idx3 (.80), L1 = idx4 (.66), L2 = idx0 (.58), L3 = idx2 (.53). Scrambling removes any 'read the leftmost cue' artifact and prevents position from proxying validity. AVAI weights at gamma = 2: H1=.941, H2=.810, M=.640, L1=.436, L2=.336, L3=.281 (the compressed regime AVAI assumes). Cues that are irrelevant on a trial are set to 1 for BOTH options, so every display is dense and requires actual comparison rather than counting 1s.

8 base structures, each accompanied by its exact A/B mirror (16 unique pairs, 6 repetitions = 96 trials), so the predicted response of BOTH models is exactly key-balanced and side bias cannot mimic either model.

(1) CONSTANT-COUNT-MARGIN LADDER (bases 1-4, plus mirrors). All four are +1 win-count margins for A, so Tallying must produce the SAME p(A) = sigmoid(beta) on every one of them, for any beta and epsilon. AVAI's weighted difference, in contrast, sweeps the entire probability range:
  - Base 1: A wins {H1} only. d = +0.94 -> AVAI p(A) ~ .998.
  - Base 2: A wins {H1,H2}, B wins {L2}. d = +1.42 -> AVAI p(A) ~ 1.00.
  - Base 3: A wins {L2,L3}, B wins {H1}. d = -0.32 -> AVAI p(A) ~ .11 (DIRECTIONAL REVERSAL relative to Tallying).
  - Base 4: A wins {L1,L2,L3}, B wins {H1,H2}. d = -0.70 -> AVAI p(A) ~ .01 (near-deterministic reversal).
A flat profile across these four (all equal, all above .5) is a signature Tallying alone can produce; a monotone sweep from ~1.00 down to ~.01 at constant count margin is impossible for Tallying under any parameters. This is the crown-jewel contrast: flat vs. full-range sweep.

(2) RANK-ORDER VIOLATION (bases 5-6, plus mirrors). Both are +2 count margins for A, so Tallying predicts them equal AND strictly more extreme than every margin-1 pair in block (1).
  - Base 5: A wins {M,L1,L2,L3}, B wins {H1,H2}. d = -0.06 -> AVAI p(A) ~ .41, i.e. essentially chance despite the LARGEST count margin in the design.
  - Base 6: A wins {H1,H2}, B wins nothing. d = +1.75 -> AVAI p(A) ~ 1.00.
AVAI therefore predicts that a margin-2 trial (base 5) is noisier than a margin-1 trial (base 1 or 2) — a rank-order violation Tallying strictly forbids. Base 6 doubles as an agreement baseline that identifies epsilon and the global beta scale free of the model contrast.

(3) TALLY TIES (bases 7-8, plus mirrors). Each option wins exactly the same number of cues, so Tallying is EXACTLY 50/50 for any beta and epsilon — a parameter-free prediction.
  - Base 7: A wins {H1}, B wins {L3}. d = +0.66 -> AVAI p(A) ~ .99.
  - Base 8: A wins {H1,L3}, B wins {H2,L1}. d = -0.02 -> AVAI p(A) ~ .46 (also near chance).
If subjects are at chance on base 7 as well as base 8, AVAI is falsified while Tallying is confirmed with no free parameters; if base 7 is near-deterministic while base 8 is at chance, that graded split is exactly AVAI's magnitude-sensitivity claim and cannot be produced by Tallying. Base 8 is additionally gamma-diagnostic: the sign of d for {H1,L3} vs {H2,L1} is near zero and flips with gamma, so it pins the amplification exponent if AVAI wins.

No pair is degenerate: on 5 of the 8 base structures the models disagree in direction or by more than 40 percentage points, the two tie trials are parameter-free tests of Tallying, and the remaining structures supply the noise/lapse baseline.

[4] GOAL: dissociate Amplified-Validity Additive Integration (AVAI, advocated) from Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS). The two models differ on three orthogonal axes: (i) POSITION-INVARIANCE \u2014 AVAI depends only on the validity-weighted difference, RDS is dominated by exp(lam*j) recency, so column position of the deciding row is everything; (ii) SIGN OF THE VALIDITY EFFECT \u2014 AVAI weights cues by v^gamma (high validity wins), RDS discounts high validity as 'redundant' via (1-v)^delta (low validity wins); (iii) EFFECT OF THE NUMBER OF DISCRIMINATING ROWS \u2014 AVAI's confidence GROWS with more agreeing cues, RDS's conflict-interference term drives choices toward chance as |D| grows.

VALIDITIES: a deliberately PALINDROMIC vector [0.93, 0.57, 0.75, 0.75, 0.57, 0.93]. This is the design's key trick: every validity level appears twice, once early and once late in the display. Validity content can therefore be held EXACTLY constant while column position is manipulated. Under AVAI (weights at gamma=2: .865, .325, .563, .563, .325, .865) any pair in which A wins column j and B wins the mirror column 5-j has weighted difference EXACTLY ZERO, so AVAI must predict exactly 50/50 for ANY gamma, beta and epsilon \u2014 a parameter-free prediction. Under RDS the same trials are strongly lopsided toward the later column (u_j = e^{1.6 j}(1-v_j)^{0.9}: u = [0.09, 2.32, 7.05, 34.9, 281.5, 272.4]).

18 unique pairs (9 base structures, each with its exact A/B mirror so both models' predicted responses are perfectly key-balanced), K = 5 repetitions = 90 trials. Non-discriminating rows are filled with matched 1s (and some matched 0s) so displays are dense and cannot be solved by counting 1s.

BLOCK 1 \u2014 PARAMETER-FREE MIRROR TIES (bases 1, 2, 5; pairs 1-4 and 9-10).
\u2022 Base 1: A wins col0 (.93), B wins col5 (.93). AVAI = .500 exactly. RDS \u2248 .14 (the late .93 row captures the decision, m=2 so interference is minimal).
\u2022 Base 2: A wins col2 (.75), B wins col3 (.75). AVAI = .500 exactly. RDS \u2248 .26 (adjacent columns \u2192 only one factor of e^lam).
\u2022 Base 5: A wins cols 4,5; B wins cols 0,1. AVAI = .500 exactly. RDS \u2248 .72 (both late rows favour A; m=4 pulls it back toward chance).
A flat 50% profile across these three, with the mirrors also at 50%, is uniquely AVAI. RDS is forced to produce an ordered spread (~.14 < ~.26 < ~.72) whose ordering is set purely by column distance and |D|.

BLOCK 2 \u2014 POSITION x VALIDITY INTERACTION (bases 3, 4; pairs 5-8). Both bases contain the identical validity contrast (.93 vs .57), only relocated.
\u2022 Base 3: A wins col0 (.93), B wins col1 (.57). AVAI p(A) \u2248 .96; RDS p(A) \u2248 .17 (DIRECTIONAL REVERSAL, both position and distinctiveness push to B).
\u2022 Base 4: A wins col5 (.93), B wins col4 (.57). AVAI p(A) \u2248 .96 \u2014 EXACTLY the same as base 3; RDS \u2248 .49 (recency of col5 and distinctiveness of col4 nearly cancel).
AVAI predicts base3 = base4; RDS predicts a ~32-point gap between two trials with identical validity content. This is the crown-jewel quantitative dissociation.

BLOCK 3 \u2014 WEIGHTED MARGIN VS LATE-CUE COALITION (base 6; pairs 11-12). A wins cols 0 and 5 (.93,.93), B wins cols 3 and 4 (.75,.57). AVAI d = +0.84 \u2192 p(A) \u2248 .99. RDS \u2248 .48 (the two late B rows roughly balance the late A row, and m=4 interference flattens it). Near-determinism vs chance on the same display.

BLOCK 4 \u2014 CONFLICT-INTERFERENCE GRADIENT (bases 7, 8, 9; pairs 13-18). All discriminating rows favour the same option, with |D| = 2, 4, 6.
\u2022 |D|=2 (cols 0,1): AVAI \u2248 1.00, RDS \u2248 .86.
\u2022 |D|=4 (cols 0,2,3,5): AVAI \u2248 1.00, RDS \u2248 .72.
\u2022 |D|=6 (all rows): AVAI \u2248 1.00, RDS \u2248 .63.
AVAI predicts accuracy that is flat-at-ceiling or weakly INCREASING with the number of agreeing cues (the weighted difference grows monotonically); RDS predicts a monotone DECREASE, because difficulty is driven by the count of differences rather than their diagnosticity. Observing >95% correct on a unanimous 6-0 trial is essentially impossible for RDS (its ceiling there is ~.63 given eps_base and rho ranges), while a drop to ~.63 on unanimous trials is impossible for AVAI. These trials also double as the lapse/beta baseline for AVAI.

No pair is degenerate: on 6 of the 9 bases the models disagree in direction or by more than 30 percentage points, and the three mirror-tie bases constitute a parameter-free test that AVAI can only pass by sitting exactly at chance.

[5] GOAL: dissociate Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS, advocated) from Amplified-Validity Additive Integration (AVAI, competitor). The two models differ on three orthogonal axes: (i) DIRECTION OF THE VALIDITY EFFECT \u2014 AVAI weights each row by v^gamma so the option holding the more valid experts wins, whereas RDS discounts high validity as redundant via (1-v)^delta so the option holding the LESS valid discriminating expert wins; (ii) COLUMN POSITION \u2014 AVAI is completely permutation-invariant, RDS is dominated by exp(lam*j) recency, so the later of two discriminating rows almost always captures the decision; (iii) NUMBER OF DISCRIMINATING ROWS \u2014 AVAI's confidence grows monotonically as more cues agree, RDS's conflict-interference term 1-(1-eps0)rho^(m-2) drags choices toward chance as |D| grows.

VALIDITIES: a REPEATED-BLOCK vector [0.95, 0.70, 0.55, 0.95, 0.70, 0.55] (deliberately not the palindrome used in an earlier design). Every validity level occurs exactly twice \u2014 once in the early block (cols 0-2) and once in the late block (cols 3-5) \u2014 so validity content can be held EXACTLY constant while the position of the high vs. low expert is swapped. Under AVAI (weights at gamma=2: .903, .490, .303, .903, .490, .303) any pair in which A wins column j and B wins column j+3 has weighted difference EXACTLY ZERO, i.e. p(A)=.500 for ANY gamma, beta, epsilon \u2014 a parameter-free prediction. Under RDS the same displays are strongly lopsided toward the LATER column (u_j = e^{1.6j}(1-v_j)^{0.9}: u = [0.067, 1.68, 12.0, 8.19, 204, 1460]).

8 base structures, each accompanied by its exact A/B mirror (16 unique pairs, K=6 \u2192 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows are filled with matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s.

BLOCK 1 \u2014 PARAMETER-FREE CROSS-BLOCK TIES (pairs 1-4).
\u2022 Base 1: A wins col0 (.95), B wins col3 (.95). AVAI = .500 exactly. RDS \u2248 .14 (the late .95 row wins the salience competition, m=2 so interference is minimal).
\u2022 Base 2: A wins col2 (.55), B wins col5 (.55). AVAI = .500 exactly. RDS \u2248 .15.
A flat 50% profile (with mirrors also at 50%) is uniquely AVAI; a strong pull toward the option holding the LATER identical-validity row is uniquely RDS and cannot be produced by any AVAI parameterisation.

BLOCK 2 \u2014 CROWN JEWEL: HIGH-EARLY vs LOW-LATE (pairs 5-8). Both bases contain the identical validity contrast (.95 vs .55), only relocated across the two blocks.
\u2022 Base 3: A wins col0 (.95), B wins col5 (.55). AVAI d=+0.60 \u2192 p(A) \u2248 .98. RDS \u2248 .14 \u2014 a near-total DIRECTIONAL REVERSAL (recency and distinctiveness both hand the decision to the late .55 row).
\u2022 Base 4: A wins col5 (.55), B wins col3 (.95). AVAI d=-0.60 \u2192 p(A) \u2248 .02. RDS \u2248 .86 \u2014 the mirror-image reversal.
AVAI says the .95-holder is chosen ~98% in both bases regardless of where the rows sit; RDS says the option holding the LATE row is chosen ~86% in both bases regardless of its validity. These two bases alone separate the models by ~85 percentage points in opposite directions.

BLOCK 3 \u2014 WEIGHTED MARGIN vs LATE-CUE COALITION (pairs 9-10). Base 5: A wins cols 0 and 3 (both .95), B wins cols 4 (.70) and 5 (.55). AVAI d=+1.01 \u2192 p(A) \u2248 1.00 (its largest margin outside the unanimous trials). RDS \u2248 .28: the two late B rows dominate salience and m=4 interference flattens the rest. Near-determinism vs. sub-chance on the same display.

BLOCK 4 \u2014 CONFLICT-INTERFERENCE GRADIENT (pairs 11-16). All discriminating rows favour the same option, with |D| = 2 (cols 4,5), 4 (cols 0,2,3,5) and 6 (all rows).
\u2022 |D|=2: AVAI \u2248 .99, RDS \u2248 .86.
\u2022 |D|=4: AVAI \u2248 1.00, RDS \u2248 .72.
\u2022 |D|=6: AVAI \u2248 1.00, RDS \u2248 .63.
AVAI predicts accuracy flat at ceiling (weakly increasing with the number of agreeing cues, since d grows monotonically); RDS predicts a monotone DECREASE, because difficulty tracks the count of differences rather than their diagnosticity. Observing >95% correct on the unanimous 6-0 trial is essentially impossible for RDS (ceiling ~.63 given eps_base/rho ranges), while a drop to ~.63 on a unanimous display is impossible for AVAI. These trials also serve as AVAI's lapse/beta baseline.

No pair is degenerate: on 6 of the 8 bases the models disagree in direction or by more than 30 percentage points, and the two cross-block tie bases constitute a parameter-free test that AVAI can only pass by sitting exactly at chance.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate OCO-LI (advocated) from RDS (competitor). The two models are algebraically identical EXCEPT for OCO-LI's label-individuation gate phi: with probability 1-phi the odd-cue-out/recency anchoring competition cannot run (columns are not individuable) and the choice collapses to a probability-matched tally over the discriminating rows. Everything else (salience form exp(lam*j)*(1-v)^delta, conflict interference 1-(1-eps0)*rho^(m-2)) is shared and its parameter ranges essentially overlap. The only way to separate them is therefore a design in which phi is FORCED to 0, and this is achieved parameter-free (independent of tau) by giving every stated validity an exact duplicate.\n\nVALIDITIES: [0.95, 0.55, 0.75, 0.95, 0.75, 0.55]. Each level (.95, .75, .55) appears exactly twice, so d_j = min_k|v_j-v_k| = 0 for every column, s_j = 0, and phi = 0 EXACTLY for any tau in [0.02,0.045]. The arrangement is neither the palindrome nor the repeated-block vector used in earlier designs: the .95 rows sit at columns 0 and 3, the .55 rows at 1 and 5, the .75 rows at 2 and 4, so every same-validity pair spans a different column distance and low/high validities alternate in both display halves. With lam=1.6, delta=0.9 the RDS salience weights are u = [0.067, 2.42, 7.05, 8.19, 172.8, 1453] \\u2014 a five-order-of-magnitude spread \\u2014 whereas OCO-LI (phi=0) treats every discriminating row as interchangeable.\n\n8 base structures, each with its exact A/B mirror (16 unique pairs, K = 6 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows are filled with matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s.\n\nBLOCK 1 \\u2014 SHARED-PARAMETER BASELINES (pairs 1-4). Base 1: unanimous m=2 (A wins cols 0 and 5). Base 2: unanimous m=6 (A wins every row). All discriminating rows point the same way, so both models predict identically (p(A) ~ .86 at m=2, ~ .64 at m=6 for the stated parameter ranges). These pin eps_base and rho for BOTH models, so the contrast trials cannot be explained away as global noise, and they also test the shared claim that unanimity gets WORSE with more agreeing rows.\n\nBLOCK 2 \\u2014 CROWN JEWEL: PARAMETER-FREE m=2 SPLIT TIES (pairs 5-12). Four bases in which exactly one row favours A and exactly one row favours B. Under OCO-LI with phi=0 the flat tally is 1/1 -> p_core = .500, and mixing with the guess term leaves p(A) = .500 EXACTLY, for every base and every parameter setting (lam, delta, tau, eps0, rho all drop out). Under RDS the same four displays are strongly lopsided and, crucially, ORDERED by column position and validity:\n  \\u2022 Base 3: A wins col0 (.95) vs B wins col3 (.95) \\u2014 identical validity, distant columns. RDS ~ .15.\n  \\u2022 Base 4: A wins col2 (.75) vs B wins col4 (.75) \\u2014 identical validity, medium distance. RDS ~ .17.\n  \\u2022 Base 5: A wins col0 (.95) vs B wins col1 (.55) \\u2014 adjacent, recency and anti-validity ALIGNED. RDS ~ .16.\n  \\u2022 Base 6: A wins col1 (.55) vs B wins col3 (.95) \\u2014 recency and anti-validity OPPOSED. RDS ~ .30.\nA flat .50 profile across all four bases (and their mirrors) is producible ONLY by the gated model; RDS with eps_base <= .32 cannot come within 18 points of chance on any of them, and it is additionally forced to produce the ordered spread .15 < .16 < .17 < .30. Flat-vs-ordered-spread at constant m is the decisive quantitative signature.\n\nBLOCK 3 \\u2014 TALLY PROPORTION VS SALIENCE MASS (pairs 13-16).\n  \\u2022 Base 7 (m=3): A wins cols 0 (.95) and 2 (.75), B wins col5 (.55, last column). OCO-LI: proportion 2/3 -> p(A) ~ .60 (ABOVE chance). RDS: the single late .55 row carries 99.5% of the salience mass -> p(A) ~ .22 (well BELOW chance). ~38-point directional disagreement.\n  \\u2022 Base 8 (m=4): A wins col5 alone (.55, last), B wins cols 0,1,2. OCO-LI: proportion 1/4 -> p(A) ~ .39 (below chance, the majority option wins). RDS: the lone late row dominates salience -> p(A) ~ .72 (above chance, the minority option wins). A full directional reversal on the same display.\n\nINTERPRETATION: if choices on the four m=2 split bases sit at 50/50 while the m=3 and m=4 conflict bases track the PROPORTION of discriminating rows (above chance for 2-vs-1, below chance for 1-vs-3), the label-ambiguity gate is confirmed and RDS is falsified with essentially no free parameters. If instead the m=2 splits show the strong sub-chance, position-ordered anti-validity pattern (~.15-.30) and the minority-but-late option is chosen ~72% of the time in Base 8, RDS is confirmed and OCO-LI's gating claim is dead. Together with the already-run all-distinct-label designs (where both models coincide), this experiment supplies the exact 'duplicate-versus-de-duplicate the stated validity vector' contrast that OCO-LI names as its own falsification test.",
  "validities": [
    0.95,
    0.55,
    0.75,
    0.95,
    0.75,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 18978,
  "output_tokens": 10649
}
```
