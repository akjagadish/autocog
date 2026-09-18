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
**Description:** **Flag Counting with Lexicographic Tiebreak (FC-LEX).**

Subjects in this task do not read a binary expert rating of '1' as an endorsement; they read it as a *warning flag*. And, crucially, they do not weight those flags by the validities they were told about in the instructions. On every trial the subject performs a single, extremely cheap operation: count how many 1s each option carries, and reject the option carrying more flags.

Three claims:

1. **Negative polarity.** A '1' is a strike against an option (a flag/complaint/warning), so an option's *badness* is monotone in its number of 1s. The option with the strictly smaller flag count is chosen.

2. **Validity-blind, magnitude-blind counting.** All experts are counted equally: the validities are acknowledged in the instructions but are not converted into cue weights. Furthermore, only the *sign* of the count difference matters. Choice probability is therefore a single flat accuracy h = (1-eps)*sigmoid(beta) + eps/2 on every trial where the counts differ, regardless of whether the difference is 1 flag or 5 flags, and regardless of which experts contributed the flags. There is no psychometric function over a validity-weighted margin: a 1-vs-0 comparison and a 1-vs-5 comparison produce the same accuracy.

3. **Lexicographic tiebreak on equal counts.** When the two options carry exactly the same number of flags the counting rule is silent. The subject then falls back to a one-reason rule: locate the highest-validity expert who discriminates and let that single expert decide. Because the subject is unsure whether that expert's '1' should attract or repel (the counting rule just told them 1s are bad, but the tiebreak is invoked precisely because badness is balanced), the direction of this tiebreak is a free subject-level bias tie_bias, which sits near chance and is mildly repelled-by-1 on average. Equal-count pairs therefore yield near-chance performance, not the strongly graded reversals a weighted-mass model predicts.

The theory is deliberately non-compensatory in *magnitude* while being fully compensatory in *counting*: it agrees with a weighted-evidence account on every design tested so far (because in those designs the option with the single top-validity 1 also happens to carry the fewest flags), but it diverges sharply on equal-count pairs such as [1,0,0,0,0] vs [0,0,0,0,1], on pairs where more 1s coincides with smaller weighted mass, and on margin ladders, where FC-LEX predicts a flat accuracy plateau conditional on the sign of the count difference.

**Parameters:**
- beta: [1.3, 2.7]
- epsilon: [0.05, 0.25]
- tie_bias: [0.10, 0.70]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])
    tie_bias = float(parameters["tie_bias"])
    eps = min(max(eps, 0.0), 1.0)
    tie_bias = min(max(tie_bias, 0.0), 1.0)

    # ---------- 2. validity-blind flag count (negative polarity) ----------
    count_a = float(np.sum(a))
    count_b = float(np.sum(b))
    d = count_a - count_b   # positive => A carries MORE warning flags => reject A

    if abs(d) > 1e-9:
        # Sign-only rule: magnitude of the count difference is ignored.
        winner = 1 if d > 0 else 0   # fewer flags wins
        s = 1.0 / (1.0 + np.exp(-beta))   # sigmoid(beta * sign) for the winner
        p_core = np.zeros(2, dtype=float)
        p_core[winner] = s
        p_core[1 - winner] = 1.0 - s
    else:
        # ---------- 3. lexicographic tiebreak on equal flag counts ----------
        val = parameters.get("validities", None)
        try:
            val = np.asarray(val, dtype=float).ravel()
        except Exception:
            val = np.array([])
        if val.size != n_features:
            if n_features == 1:
                val = np.array([0.9])
            else:
                val = np.linspace(0.9, 0.55, n_features)

        order = np.argsort(-val, kind="stable")
        top1_owner = None   # index of the option carrying the '1' on the top discriminating cue
        for j in order:
            if a[j] > b[j]:
                top1_owner = 0
                break
            if b[j] > a[j]:
                top1_owner = 1
                break

        if top1_owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.zeros(2, dtype=float)
            p_core[top1_owner] = tie_bias
            p_core[1 - top1_owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------- 4. attention lapse ----------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Polarised Weighted Evidence Accumulation (PWEA).**

People do NOT stop at the first discriminating cue. On every trial they integrate *all* expert ratings for each option into a single scalar impression, and then choose between the two impressions with a graded (margin-sensitive) rule.

Three claims:

1. **Compensatory integration with validity-graded weights.** Each cue j contributes w_j = ((v_j - 0.5)/0.5)^gamma to the option's score, where the weights are rescaled to have mean 1 so that the total evidence scale grows with the number of cues. The steepness exponent gamma is a free, subject-level parameter: gamma -> 0 gives equal weighting (pure tallying), gamma ~ 1 gives classic validity-proportional weighting, and large gamma approaches a lexicographic, Take-The-Best-like regime in which the top cue can outvote all the others. The family therefore *nests* both incumbent heuristics as special cases rather than opposing them.

2. **Polarity of endorsement is a free interpretive parameter s.** A binary rating of 1 is not intrinsically 'good'. It is a symbol whose desirability sign the subject must infer from the instructions/response mapping. In this task population the sign is systematically *negative*: an expert '1' is read as a flag/warning (or the A/B response mapping is inverted), so the option carrying the greater weighted mass of 1s is *rejected*. score_i = s * sum_j w_j x_ij with s < 0.

3. **Graded choice on the evidence margin, with genuinely heterogeneous evidence sensitivity.** Choice is a softmax over the two scores with inverse temperature beta, so the probability of choosing an option increases smoothly with the *magnitude* of the weighted difference (a strict prediction that one-reason stopping rules cannot make), plus an independent attention-lapse epsilon that replaces the choice with a coin flip. Exact ties in weighted score produce exactly 50/50. Critically, the effective evidence sensitivity k = beta*|s| is only *moderate* in this population and varies widely across subjects: many subjects are near-threshold on small-margin pairs, so their choices on narrow-margin conflict items are only weakly above chance, whereas the same subjects are near-deterministic on wide-margin (dominance-like) items. This within-family heterogeneity in k and gamma — not lapse alone — is what produces the between-subject dispersion of the data and keeps the pooled conflict-hit rate well below ceiling.

Because s < 0 and gamma stays in the compensatory range, the model reproduces both signature results with one mechanism: in conflict pairs the option carrying the single highest-validity 1 also carries the *smallest* weighted mass of 1s, so it is chosen (looking exactly like TTB), whereas in agreement/dominance pairs the option that both TTB and tallying favour carries the *largest* mass of 1s and is therefore rejected (looking anti-TTB). Because Exp-1 conflict margins are small and Exp-2 conflict/dominance margins are large, moderate k depresses the Exp-1 hit rate while leaving the Exp-2 contrast saturated — exactly the asymmetry the human data show.

**Parameters:**
- polarity: [-1.0, -0.35]
- gamma: [0.1, 1.15]
- beta: [2.0, 9.0]
- epsilon: [0.10, 0.55]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            # flat concatenation of the two options
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    # ---------- 2. validity-derived weights ----------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n_features:
        # graceful fallback: linearly descending validities
        if n_features == 1:
            val = np.array([0.9])
        else:
            val = np.linspace(0.9, 0.55, n_features)

    gamma = float(parameters["gamma"])
    r = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)          # normalised validity strength
    w = np.power(r, max(gamma, 0.0))
    m = float(np.mean(w))
    if not np.isfinite(m) or m <= 0.0:
        w = np.ones(n_features)
    else:
        w = w / m                                       # mean weight == 1 (scale grows with n cues)

    # ---------- 3. signed compensatory integration ----------
    s = float(parameters["polarity"])                   # negative: a '1' counts AGAINST the option
    score_a = s * float(np.dot(w, a))
    score_b = s * float(np.dot(w, b))
    scores = np.array([score_a, score_b], dtype=float)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # ---------- 4. margin-graded softmax + lapse ----------
    z = beta * scores
    z = z - np.max(z)                                    # numerically stable
    e = np.exp(z)
    p_core = e / e.sum()
    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with well-spread validities (0.90, 0.80, 0.70, 0.60, 0.55) so that the validity ordering is unambiguous and TTB's cue cascade is well defined. The design contains three diagnostic trial classes plus controls, each presented in both A/B mirror forms to remove side bias.

(1) CONFLICT trials (pairs 1-6): the highest discriminating cue favors one option while the majority of remaining cues favors the other. E.g. A=[1,0,0,0,0] vs B=[0,1,1,1,1]: TTB stops on expert 1 and picks A with probability ~sigmoid(beta), whereas Tallying counts 1 vs 4 and picks B with probability ~sigmoid(3*beta) \u2014 the two theories predict opposite responses, and Tallying's tally margin (3) is large so its prediction is nearly deterministic for any moderate beta. Pairs 3-4 use a smaller tally margin (1 vs 2), so Tallying predicts a *weaker* (margin-1 softmax) preference for the option TTB rejects; the graded difference across margin sizes is a quantitative signature that separates the models beyond mere direction.

(2) TALLY-TIE trials (pairs 7-10): the feature-wise wins are exactly balanced, so Tallying is forced to 50/50 regardless of beta, while TTB still discriminates decisively. Pairs 7-8 tie 2-2 with the top cue discriminating (TTB confident); pairs 9-10 tie 1-1 with all three high-validity cues equal so TTB is driven by the low-validity expert 4. If subjects show systematic (non-chance) preferences on these trials, Tallying cannot account for it at any parameter setting.

(3) CONTROL trials (pairs 11-14) where both theories agree, providing clean leverage on beta and epsilon so that the diagnostic trials are interpreted with well-identified noise parameters.

Pair 5-6 additionally makes the top cue non-discriminating (both options rated 1 by expert 1), forcing TTB down to expert 2 while Tallying reads a 1-vs-3 margin the other way \u2014 testing whether people truly stop at the first *discriminating* cue rather than integrating.

Advocated theory (TTB) predicts: choice of the option favored by the highest discriminating cue on ALL of pairs 1-10, with roughly constant accuracy independent of how many lower cues oppose it, and above-chance systematic choice on the tally-tie pairs. Tallying predicts the opposite direction on pairs 1-6 with strength graded by tally margin, and exact chance on pairs 7-10.

[1] I advocate TALLYING and design for a quantitative dissociation that the previous experiment (which emphasized direction reversals only) did not isolate: the *graded* dependence of choice probability on the tally margin versus TTB's flat, margin-independent confidence.

Six experts with clearly ordered, well-spread validities (0.90, 0.82, 0.74, 0.66, 0.58, 0.53) make TTB's cue cascade unambiguous (cue order = feature order). Every pair appears in both A/B mirror forms (16 unique pairs, K=6 -> 96 trials) so side bias cancels.

(1) GRADED-AGREEMENT trials (pairs 1-6). Both models favour the same option, but the tally margin is +1 (3 wins vs 2), +3 (4 vs 1), and +5 (5 vs 0). TTB always stops at expert 1 and therefore predicts EXACTLY the same accuracy, (1-eps)*sigmoid(beta)+eps/2, in all three conditions; its binary score never changes. Tallying predicts monotonically increasing accuracy, sigmoid(1*beta) < sigmoid(3*beta) < sigmoid(5*beta). Because these conditions also pin down beta and epsilon, a reliable accuracy gradient across margin 1 -> 3 -> 5 cannot be absorbed by any TTB parameter setting: TTB is forced to attribute all three to one sigmoid(beta). This is the core quantitative signature, and it is measured on trials where the two theories agree in DIRECTION, so it is not contaminated by any direction-reversal strategy shift.

(2) CONFLICT trials (pairs 7-12). Expert 1 favours one option while the remaining experts favour the other, with tally margins of -1 (2 vs 3), -3 (1 vs 4) and -4 (1 vs 5). TTB predicts choice of the expert-1 option with the SAME probability in all three; Tallying predicts choice of the opposite option with strength graded by margin. Direction and gradient are both diagnostic here, and the -1 case is the crucial one: Tallying predicts a weak reversal (sigmoid(beta)) exactly matching in magnitude what it predicts for the +1 agreement pairs.

(3) DESCENT trial (pairs 13-14): experts 1 and 2 both rate the two products 1, so TTB must descend to expert 3 and choose that option, while the tally is 1 vs 3 the other way. This tests whether people truly stop at the first discriminating cue or integrate across the three low-validity experts.

(4) TALLY-TIE trial (pairs 15-16): A wins on experts 1-2, B wins on experts 3-4 (2-2 tie). Tallying is forced to exactly 50/50 for any parameters; TTB predicts a decisive expert-1-based preference. This is the competing theory's unique prediction and provides a hard falsification test of my advocated model; including it makes the design fair rather than stacked.

Predictions if Tallying is right: accuracy rises steeply with |tally margin| in both the agreement and conflict blocks, choices reverse against expert 1 on pairs 7-12 and 13-14, and choices sit at chance on pairs 15-16. If TTB is right: flat accuracy across all margins, expert-1 option chosen throughout pairs 1-12, expert-3 option on 13-14, and clearly above-chance systematic choice on the tally-tie pairs.

[2] I advocate TAKE-THE-BEST (TTB) against Polarised Weighted Evidence Accumulation (PWEA). PWEA makes two commitments that TTB flatly denies: (i) a rating of '1' is read as a NEGATIVE flag (polarity s<0), so the option carrying the greater validity-weighted mass of 1s is REJECTED; and (ii) choice probability scales smoothly with that weighted mass margin. TTB instead says the option holding the '1' on the highest discriminating expert is chosen, with a single, margin-independent accuracy (1-eps)*sigmoid(beta) on every discriminating trial.

Crucially, the two earlier experiments in this series pitted TTB against Tallying, using only 'conflict' pairs of the form A=[1,0,0,0,0] vs B=[0,1,1,1,1]. On exactly those pairs PWEA MIMICS TTB (the top-cue option also carries the least mass of 1s), so nothing so far can separate them. This design therefore concentrates on the trial class where PWEA and TTB predict OPPOSITE choices: pairs where the option favoured by the top expert also carries MORE 1s.

Five experts with well-spread, strictly ordered validities (0.93, 0.84, 0.72, 0.62, 0.54) make TTB's cue cascade unambiguous. gamma=1 weights (mean-1 normalised) are w=[1.87, 1.48, 0.96, 0.52, 0.17]. All 8 base pairs are shown in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials) so any side bias cancels.

CLASS I - POLARITY REVERSAL, LARGE MARGIN (pairs 1-2 and mirrors 9-10):
[1,1,1,1,1] vs [0,0,0,0,0] (mass 5.00 vs 0) and [1,1,1,0,0] vs [0,0,0,1,1] (4.31 vs 0.70). TTB picks the all-/mostly-1 option essentially deterministically; PWEA picks the OPPOSITE option essentially deterministically for every admissible parameter setting (any gamma, any s<0, beta>=2). These are the hard falsification trials: they cannot be reconciled with PWEA by any parameter tuning, and equally, a reliable preference for the sparse-1 option would kill TTB outright.

CLASS II - GRADED REVERSAL (pairs 3, 4, 6 and mirrors): [1,0,0,0,0] vs [0,0,0,0,0] (PWEA margin 1.87), [0,0,0,0,1] vs [0,0,0,0,0] (margin 0.17 at gamma=1), and [1,0,0,0,0] vs [0,0,0,1,0] (margin 1.35). TTB predicts the SAME accuracy toward the option with the single 1 (or with the higher-validity 1) in all three, because only the sign on the first discriminating cue matters. PWEA predicts the opposite direction with strength graded ~10:1 between the top-cue and bottom-cue trials. So the design delivers a within-subject direction test AND a magnitude-gradient test on the same trial family: flat, above-chance, top-cue-consistent accuracy across margins is uniquely TTB; a margin-graded reversal is uniquely PWEA.

CLASS III - CASCADE / NEAR-TIE TRIAL (pair 7 and mirror): [1,1,0,0,0] vs [1,0,1,1,1]. Expert 1 does not discriminate, so TTB descends to expert 2 and decisively picks A. PWEA's weighted masses are 3.35 vs 3.52 at gamma=1 (and reverse to favour A at gamma<=0.5): across essentially the whole admissible gamma range PWEA is pinned near 50/50 here, whereas TTB predicts full, undiminished accuracy. Systematic choice of A on this pair is diagnostic of one-reason stopping and cannot be produced by PWEA at any evidence sensitivity.

CLASS IV - AGREEMENT / CALIBRATION (pairs 5 and 8, plus mirrors): [1,0,0,0,0] vs [0,1,1,1,1] and [1,0,0,0,0] vs [0,1,1,0,0]. Here both theories favour the same option (the classic TTB-vs-Tallying conflict items on which PWEA mimics TTB). They pin down beta and epsilon for BOTH models independently of the diagnostic classes, so the reversal trials are interpreted against well-identified noise parameters, and they guarantee the design is not simply stacked toward one family.

PREDICTIONS. If TTB is right: on every pair the option bearing the '1' on the highest discriminating expert is chosen at a single, roughly constant above-chance rate, including the mostly-1 dominance pairs (Class I), the bottom-cue pairs (Class II), and the cascade pair (Class III). If PWEA is right: choices REVERSE on Classes I-II (sparse-1 option chosen), near-deterministically on Class I and only weakly on the small-margin Class II items, sit at chance on Class III, and agree with TTB only on Class IV.

[3] I advocate Polarised Weighted Evidence Accumulation (PWEA) against Take-The-Best (TTB). The two prior TTB-vs-PWEA designs contrasted a few isolated reversal pairs; this design instead builds a *continuous margin ladder* on which TTB is mathematically forced to a single flat accuracy while PWEA must produce a monotone psychometric function with a parameter-robust crossing point, plus a gamma-graded pair and a robust weighted-tie pair.

Validities (0.95, 0.80, 0.70, 0.62, 0.55) are strictly ordered so TTB's cue cascade is unambiguous. At gamma=1 the mean-1 normalised weights are w=[2.01, 1.34, 0.89, 0.54, 0.22]; at gamma=0.1 they are nearly flat, w=[1.10, 1.05, 1.01, 0.96, 0.88]. Every base pair appears in both A/B orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly.

LADDER (base pairs 1-4, all of the form A=[1,0,0,0,0] versus a B with progressively more low-validity 1s): B = [0,0,0,0,0] / [0,1,0,0,0] / [0,1,1,0,0] / [0,1,1,1,1]. On ALL FOUR, expert 1 discriminates in favour of A, so TTB predicts exactly the same accuracy (1-eps)*sigmoid(beta) toward A on each; nothing in TTB can make these four differ. PWEA's signed weighted-mass margin (A minus B) is +2.01, +0.67, -0.22, -0.98 at gamma=1 and +1.10, +0.04, -0.92, -2.80 at gamma=0.1: monotone in the same order for the whole admissible gamma range, and with s<0 this yields a monotonically INCREASING probability of choosing A across pairs 1->4, crossing 50% between pair 2 and pair 3 for every gamma in [0.1, 1.15]. A four-point graded curve with a sign flip located between pairs 2 and 3 is uniquely PWEA; flat above-chance A-choice is uniquely TTB. Pair 4 is the classic 'conflict' item on which the two families agree in direction, so it also pins down beta/epsilon for both models.

WEIGHT-vs-COUNT GRADIENT (pairs 6 and 7): [1,0,0,0,0] vs [0,0,0,0,1] and [0,0,1,0,0] vs [0,0,0,1,0]. Both have one 1 per option, so TTB again predicts identical accuracy toward the higher-validity-1 option (A) in both. PWEA predicts the opposite direction (B) with margins in an approximately 5:1 ratio (1.79 vs 0.36 at gamma=1), collapsing toward chance for both as gamma->0. These items therefore separate direction (PWEA vs TTB) and simultaneously identify gamma, which no TTB parameter can mimic.

CASCADE REVERSAL (pair 5): [1,1,1,0,0] vs [1,0,0,1,1]. Expert 1 does not discriminate, so TTB descends to expert 2 and picks A decisively. PWEA's margin (w2+w3-w4-w5) is positive for every gamma (1.47 at gamma=1, 0.22 at gamma=0.1), so PWEA chooses B, strongly if gamma is near 1 and weakly if gamma is near 0. This tests one-reason stopping against signed integration on a trial where the top cue is uninformative.

ROBUST WEIGHTED TIE (pair 8): [0,1,0,0,1] vs [0,0,1,1,0]. Counts are equal and the weighted masses nearly cancel across the whole gamma range (|margin| <= 0.19 for gamma in [0.1, 1.15], and it even changes sign), so PWEA is pinned essentially at chance for any beta and |s|. TTB, however, stops at expert 2 and picks A with its full, undiminished accuracy. Systematic A-choice here would falsify PWEA; near-chance responding here combined with decisive responding on pairs 1 and 5 would falsify TTB, because TTB's accuracy must be constant across all discriminating trials.

PREDICTIONS. PWEA: choice of A rises monotonically across ladder pairs 1->4 with the crossover between 2 and 3; strong B-preference on 6, weak B-preference on 7 (ratio ~5:1); B-preference on 5; chance on 8; overall dispersion driven by subject-level k=beta*|s| and gamma. TTB: a single, roughly constant above-chance preference for A on every one of pairs 1-8 (and for the mirrored option in the mirrors), with no dependence on weighted mass, cue count, or margin.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "I advocate FC-LEX (validity-blind flag counting with a lexicographic tiebreak) against PWEA (validity-weighted, margin-graded signed integration). Both models agree that a '1' is a warning flag, so direction alone is often non-diagnostic; the decisive dissociations are (a) MAGNITUDE INVARIANCE and (b) COMPOSITION INVARIANCE of the count rule, plus (c) the equal-count class.\n\nValidities are deliberately skewed: (0.97, 0.66, 0.60, 0.56, 0.52). At gamma=1 the mean-1-normalised PWEA weights are w=[2.90, 0.99, 0.62, 0.37, 0.12] (a ~24:1 top-to-bottom ratio); at gamma=0.1 they are nearly flat (~[1.19,1.04,1.02,1.00,0.94]). This skew makes PWEA's weighted mass diverge sharply from raw flag counts for moderate/high gamma, while low gamma collapses PWEA onto tallying (which is direction-identical to counting) - so the design simultaneously separates the families and identifies gamma.\n\nCLASS 1 - FLAT PLATEAU / MARGIN LADDER (pairs 1-2, 5-6, 7-8). A=[0,0,0,0,0] against B with 1, 4, or 5 top-loaded flags. All three have a count difference in the same direction, so FC-LEX predicts EXACTLY the same accuracy (1-eps)*sigmoid(beta)+eps/2 on all of them - no parameter of FC-LEX can make them differ. PWEA's weighted margins are 2.90, 1.80, 5.00 at gamma=1 and 1.19, 4.00, 5.19 at gamma=0.1: for every admissible gamma the [0,1,1,1,1] and [1,1,1,1,1] items carry a much larger margin than a single low-weight flag, so PWEA must show a monotone psychometric gradient.\n\nCLASS 2 - COMPOSITION INVARIANCE (pairs 1-2 vs 3-4). Both are 0-flags-versus-1-flag, but the flag sits on the 97% expert in one case and on the 52% expert in the other. FC-LEX: identical accuracy (counting is validity-blind). PWEA at gamma=1: margin 2.90 versus 0.12, i.e. near-deterministic versus essentially chance - a ~24:1 evidence ratio on trials that FC-LEX must treat as identical. This is the single cleanest quantitative test in the design, because direction is the same for both models, so no strategy-reversal confound can explain a gradient.\n\nCLASS 3 - COUNT-vs-WEIGHT REVERSAL (pairs 9-10: [1,0,0,0,0] vs [0,1,1,1,1]; pairs 11-12: [1,1,0,0,0] vs [0,0,1,1,1]). Here more flags coincides with LESS weighted mass. FC-LEX chooses the fewer-flag option (A in 9 and 11) at its usual flat rate. PWEA with gamma near 1 chooses the OPPOSITE option (masses 2.90 vs 1.80 and 3.89 vs 1.11), while with gamma near 0.1 it agrees with FC-LEX. A reliable preference for the many-flags option here is impossible for FC-LEX at any parameter setting and pins gamma for PWEA.\n\nCLASS 4 - EQUAL-COUNT TIEBREAK (pairs 13-14: [1,0,0,0,0] vs [0,0,0,0,1]; pairs 15-16: [0,0,1,0,0] vs [0,0,0,1,0]). Counts are tied, so the counting rule is silent and FC-LEX falls back on the top discriminating expert with a subject-level bias tie_bias near chance - and, critically, the SAME p(top-flag option) in both pairs, because both are decided by whichever expert discriminates first. PWEA instead predicts a strongly graded reversal: margin 2.78 for pair 13 versus 0.25 for pair 15 at gamma=1 (a >10:1 ratio), collapsing toward chance for both as gamma->0. So Class 4 gives a within-subject magnitude contrast on trials where FC-LEX is committed to invariance.\n\nAll 8 base pairs appear in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly, and pairs 7-8 (dominance) plus pairs 1-2 provide clean identification of beta/epsilon for FC-LEX and of beta*|s|/epsilon for PWEA.\n\nPREDICTIONS. FC-LEX: one flat above-chance accuracy toward the fewer-flag option across pairs 1-12 regardless of margin size (1 vs 4 vs 5 flags) and regardless of which expert carries the flag; near-chance, magnitude-invariant, subject-idiosyncratic responding on pairs 13-16. PWEA: monotone accuracy growth with weighted margin in Class 1, a large accuracy gap between top-cue and bottom-cue single-flag items in Class 2, direction reversals in Class 3 for moderate/high gamma, and a strongly graded (not invariant) split between pairs 13-14 and 15-16.",
  "validities": [
    0.97,
    0.66,
    0.6,
    0.56,
    0.52
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
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
      0
    ],
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
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
      0
    ],
    [
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
      0
    ],
    [
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 12856,
  "output_tokens": 6527
}
```
