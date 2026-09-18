# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_3
**Polarised Weighted Evidence Accumulation (PWEA).**

People do NOT stop at the first discriminating cue. On every trial they integrate *all* expert ratings for each option into a single scalar impression, and then choose between the two impressions with a graded (margin-sensitive) rule.

Three claims:

1. **Compensatory integration with validity-graded weights.** Each cue j contributes w_j = ((v_j - 0.5)/0.5)^gamma to the option's score, where the weights are rescaled to have mean 1 so that the total evidence scale grows with the number of cues. The steepness exponent gamma is a free, subject-level parameter: gamma -> 0 gives equal weighting (pure tallying), gamma ~ 1 gives classic validity-proportional weighting, and large gamma approaches a lexicographic, Take-The-Best-like regime in which the top cue can outvote all the others. The family therefore *nests* both incumbent heuristics as special cases rather than opposing them.

2. **Polarity of endorsement is a free interpretive parameter s.** A binary rating of 1 is not intrinsically 'good'. It is a symbol whose desirability sign the subject must infer from the instructions/response mapping. In this task population the sign is systematically *negative*: an expert '1' is read as a flag/warning (or the A/B response mapping is inverted), so the option carrying the greater weighted mass of 1s is *rejected*. score_i = s * sum_j w_j x_ij with s < 0.

3. **Graded choice on the evidence margin, with genuinely heterogeneous evidence sensitivity.** Choice is a softmax over the two scores with inverse temperature beta, so the probability of choosing an option increases smoothly with the *magnitude* of the weighted difference (a strict prediction that one-reason stopping rules cannot make), plus an independent attention-lapse epsilon that replaces the choice with a coin flip. Exact ties in weighted score produce exactly 50/50. Critically, the effective evidence sensitivity k = beta*|s| is only *moderate* in this population and varies widely across subjects: many subjects are near-threshold on small-margin pairs, so their choices on narrow-margin conflict items are only weakly above chance, whereas the same subjects are near-deterministic on wide-margin (dominance-like) items. This within-family heterogeneity in k and gamma — not lapse alone — is what produces the between-subject dispersion of the data and keeps the pooled conflict-hit rate well below ceiling.

Because s < 0 and gamma stays in the compensatory range, the model reproduces both signature results with one mechanism: in conflict pairs the option carrying the single highest-validity 1 also carries the *smallest* weighted mass of 1s, so it is chosen (looking exactly like TTB), whereas in agreement/dominance pairs the option that both TTB and tallying favour carries the *largest* mass of 1s and is therefore rejected (looking anti-TTB). Because Exp-1 conflict margins are small and Exp-2 conflict/dominance margins are large, moderate k depresses the Exp-1 hit rate while leaving the Exp-2 contrast saturated — exactly the asymmetry the human data show.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.93, 0.84, 0.72, 0.62, 0.54]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 13: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 15: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 16: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** I advocate TAKE-THE-BEST (TTB) against Polarised Weighted Evidence Accumulation (PWEA). PWEA makes two commitments that TTB flatly denies: (i) a rating of '1' is read as a NEGATIVE flag (polarity s<0), so the option carrying the greater validity-weighted mass of 1s is REJECTED; and (ii) choice probability scales smoothly with that weighted mass margin. TTB instead says the option holding the '1' on the highest discriminating expert is chosen, with a single, margin-independent accuracy (1-eps)*sigmoid(beta) on every discriminating trial.

Crucially, the two earlier experiments in this series pitted TTB against Tallying, using only 'conflict' pairs of the form A=[1,0,0,0,0] vs B=[0,1,1,1,1]. On exactly those pairs PWEA MIMICS TTB (the top-cue option also carries the least mass of 1s), so nothing so far can separate them. This design therefore concentrates on the trial class where PWEA and TTB predict OPPOSITE choices: pairs where the option favoured by the top expert also carries MORE 1s.

Five experts with well-spread, strictly ordered validities (0.93, 0.84, 0.72, 0.62, 0.54) make TTB's cue cascade unambiguous. gamma=1 weights (mean-1 normalised) are w=[1.87, 1.48, 0.96, 0.52, 0.17]. All 8 base pairs are shown in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials) so any side bias cancels.

CLASS I - POLARITY REVERSAL, LARGE MARGIN (pairs 1-2 and mirrors 9-10):
[1,1,1,1,1] vs [0,0,0,0,0] (mass 5.00 vs 0) and [1,1,1,0,0] vs [0,0,0,1,1] (4.31 vs 0.70). TTB picks the all-/mostly-1 option essentially deterministically; PWEA picks the OPPOSITE option essentially deterministically for every admissible parameter setting (any gamma, any s<0, beta>=2). These are the hard falsification trials: they cannot be reconciled with PWEA by any parameter tuning, and equally, a reliable preference for the sparse-1 option would kill TTB outright.

CLASS II - GRADED REVERSAL (pairs 3, 4, 6 and mirrors): [1,0,0,0,0] vs [0,0,0,0,0] (PWEA margin 1.87), [0,0,0,0,1] vs [0,0,0,0,0] (margin 0.17 at gamma=1), and [1,0,0,0,0] vs [0,0,0,1,0] (margin 1.35). TTB predicts the SAME accuracy toward the option with the single 1 (or with the higher-validity 1) in all three, because only the sign on the first discriminating cue matters. PWEA predicts the opposite direction with strength graded ~10:1 between the top-cue and bottom-cue trials. So the design delivers a within-subject direction test AND a magnitude-gradient test on the same trial family: flat, above-chance, top-cue-consistent accuracy across margins is uniquely TTB; a margin-graded reversal is uniquely PWEA.

CLASS III - CASCADE / NEAR-TIE TRIAL (pair 7 and mirror): [1,1,0,0,0] vs [1,0,1,1,1]. Expert 1 does not discriminate, so TTB descends to expert 2 and decisively picks A. PWEA's weighted masses are 3.35 vs 3.52 at gamma=1 (and reverse to favour A at gamma<=0.5): across essentially the whole admissible gamma range PWEA is pinned near 50/50 here, whereas TTB predicts full, undiminished accuracy. Systematic choice of A on this pair is diagnostic of one-reason stopping and cannot be produced by PWEA at any evidence sensitivity.

CLASS IV - AGREEMENT / CALIBRATION (pairs 5 and 8, plus mirrors): [1,0,0,0,0] vs [0,1,1,1,1] and [1,0,0,0,0] vs [0,1,1,0,0]. Here both theories favour the same option (the classic TTB-vs-Tallying conflict items on which PWEA mimics TTB). They pin down beta and epsilon for BOTH models independently of the diagnostic classes, so the reversal trials are interpreted against well-identified noise parameters, and they guarantee the design is not simply stacked toward one family.

PREDICTIONS. If TTB is right: on every pair the option bearing the '1' on the highest discriminating expert is chosen at a single, roughly constant above-chance rate, including the mostly-1 dominance pairs (Class I), the bottom-cue pairs (Class II), and the cascade pair (Class III). If PWEA is right: choices REVERSE on Classes I-II (sparse-1 option chosen), near-deterministically on Class I and only weakly on the small-margin Class II items, sit at chance on Class III, and agree with TTB only on Class IV.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Contrast score: (rate of choosing the option favoured by the highest-validity discriminating expert on POLARITY-REVERSAL pairs) minus (the same rate on AGREEMENT pairs). Reversal pairs are those where the top-cue winner also carries the larger validity-weighted mass of 1s ([11111]v[00000], [11100]v[00011], [10000]v[00000], [10000]v[00010] and mirrors); agreement pairs are the classic conflict items ([10000]v[01111], [10000]v[01100] and mirrors) where the top-cue winner carries the smaller mass of 1s. Under TTB only the sign on the first discriminating cue matters, so the hit rate is the SAME quantity ((1-eps)*sigmoid(beta)+eps/2) on both sets for every subject regardless of beta and epsilon: the difference is identically 0 up to binomial sampling noise (48 vs 24 trials -> per-subject SD ~0.09). Under PWEA a '1' is a negative flag, so on the reversal set the top-cue option is REJECTED (margins of 5.0, ~3.6, ~1.9, ~1.35 mean-weight units times k=beta*|s|>=0.7 make this near-deterministic), while on the agreement set PWEA mimics TTB and the same option is chosen. The difference therefore lands near -(1-eps)*0.9, i.e. roughly -0.4 to -0.8, with epsilon-driven dispersion only. Because the metric is a within-subject difference of two rates that share the same noise parameters, it cancels the lapse/temperature heterogeneity that otherwise inflates between-subject variance, giving a large mean gap (~0.55-0.6) with tight per-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    def key(x):
        return ''.join([str(int(v)) for v in list(x)])

    # Class I / II 'polarity-reversal' pairs: the option favoured by the
    # top-validity expert ALSO carries the larger weighted mass of 1s.
    rev = {
        ('11111', '00000'), ('00000', '11111'),
        ('11100', '00011'), ('00011', '11100'),
        ('10000', '00000'), ('00000', '10000'),
        ('10000', '00010'), ('00010', '10000'),
    }
    # Class IV 'agreement' pairs: top-cue option carries the SMALLER mass of 1s.
    agr = {
        ('10000', '01111'), ('01111', '10000'),
        ('10000', '01100'), ('01100', '10000'),
    }

    hits_rev = []
    hits_agr = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            ka = key(a)
            kb = key(b)
        except Exception:
            continue
        pair = (ka, kb)
        in_rev = pair in rev
        in_agr = pair in agr
        if not (in_rev or in_agr):
            continue
        aa = int(list(a)[0])
        bb = int(list(b)[0])
        if aa == bb:
            continue
        winner = 0 if aa == 1 else 1
        hit = 1.0 if int(r) == winner else 0.0
        if in_rev:
            hits_rev.append(hit)
        else:
            hits_agr.append(hit)

    if len(hits_rev) == 0 or len(hits_agr) == 0:
        return 0.0
    return float(np.mean(hits_rev) - np.mean(hits_agr))

### RESULTS
- Predicted under pi_1 (simulated): 0.0012 (var=0.0066)
- Predicted under pi_3 (simulated): -0.6121 (var=0.0311)
- Observed on real data: -0.5767 (var=0.0708)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.8, 0.7, 0.62, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 8: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 9: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 14: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** I advocate Polarised Weighted Evidence Accumulation (PWEA) against Take-The-Best (TTB). The two prior TTB-vs-PWEA designs contrasted a few isolated reversal pairs; this design instead builds a *continuous margin ladder* on which TTB is mathematically forced to a single flat accuracy while PWEA must produce a monotone psychometric function with a parameter-robust crossing point, plus a gamma-graded pair and a robust weighted-tie pair.

Validities (0.95, 0.80, 0.70, 0.62, 0.55) are strictly ordered so TTB's cue cascade is unambiguous. At gamma=1 the mean-1 normalised weights are w=[2.01, 1.34, 0.89, 0.54, 0.22]; at gamma=0.1 they are nearly flat, w=[1.10, 1.05, 1.01, 0.96, 0.88]. Every base pair appears in both A/B orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly.

LADDER (base pairs 1-4, all of the form A=[1,0,0,0,0] versus a B with progressively more low-validity 1s): B = [0,0,0,0,0] / [0,1,0,0,0] / [0,1,1,0,0] / [0,1,1,1,1]. On ALL FOUR, expert 1 discriminates in favour of A, so TTB predicts exactly the same accuracy (1-eps)*sigmoid(beta) toward A on each; nothing in TTB can make these four differ. PWEA's signed weighted-mass margin (A minus B) is +2.01, +0.67, -0.22, -0.98 at gamma=1 and +1.10, +0.04, -0.92, -2.80 at gamma=0.1: monotone in the same order for the whole admissible gamma range, and with s<0 this yields a monotonically INCREASING probability of choosing A across pairs 1->4, crossing 50% between pair 2 and pair 3 for every gamma in [0.1, 1.15]. A four-point graded curve with a sign flip located between pairs 2 and 3 is uniquely PWEA; flat above-chance A-choice is uniquely TTB. Pair 4 is the classic 'conflict' item on which the two families agree in direction, so it also pins down beta/epsilon for both models.

WEIGHT-vs-COUNT GRADIENT (pairs 6 and 7): [1,0,0,0,0] vs [0,0,0,0,1] and [0,0,1,0,0] vs [0,0,0,1,0]. Both have one 1 per option, so TTB again predicts identical accuracy toward the higher-validity-1 option (A) in both. PWEA predicts the opposite direction (B) with margins in an approximately 5:1 ratio (1.79 vs 0.36 at gamma=1), collapsing toward chance for both as gamma->0. These items therefore separate direction (PWEA vs TTB) and simultaneously identify gamma, which no TTB parameter can mimic.

CASCADE REVERSAL (pair 5): [1,1,1,0,0] vs [1,0,0,1,1]. Expert 1 does not discriminate, so TTB descends to expert 2 and picks A decisively. PWEA's margin (w2+w3-w4-w5) is positive for every gamma (1.47 at gamma=1, 0.22 at gamma=0.1), so PWEA chooses B, strongly if gamma is near 1 and weakly if gamma is near 0. This tests one-reason stopping against signed integration on a trial where the top cue is uninformative.

ROBUST WEIGHTED TIE (pair 8): [0,1,0,0,1] vs [0,0,1,1,0]. Counts are equal and the weighted masses nearly cancel across the whole gamma range (|margin| <= 0.19 for gamma in [0.1, 1.15], and it even changes sign), so PWEA is pinned essentially at chance for any beta and |s|. TTB, however, stops at expert 2 and picks A with its full, undiminished accuracy. Systematic A-choice here would falsify PWEA; near-chance responding here combined with decisive responding on pairs 1 and 5 would falsify TTB, because TTB's accuracy must be constant across all discriminating trials.

PREDICTIONS. PWEA: choice of A rises monotonically across ladder pairs 1->4 with the crossover between 2 and 3; strong B-preference on 6, weak B-preference on 7 (ratio ~5:1); B-preference on 5; chance on 8; overall dispersion driven by subject-level k=beta*|s| and gamma. TTB: a single, roughly constant above-chance preference for A on every one of pairs 1-8 (and for the mirrored option in the mirrors), with no dependence on weighted mass, cue count, or margin.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Within-subject contrast between two base pairs that TTB must treat identically but PWEA must treat oppositely.

Define 'hit' = choosing the option favoured by the Take-The-Best cascade (the option with the 1 on the highest-validity discriminating cue). The metric is hit-rate on the CONFLICT pair {[1,0,0,0,0] vs [0,1,1,1,1]} minus hit-rate on the DOMINANCE pair {[1,0,0,0,0] vs [0,0,0,0,0]}. Both base pairs appear in both A/B orders, 12 trials each per subject, so side bias cancels.

TTB: cue 1 discriminates on BOTH pairs, so the predicted choice probability for the TTB winner is exactly (1-eps)*sigmoid(beta)+eps/2 on both. The difference is identically 0 for every beta and epsilon, i.e. the metric is pinned at 0 with only binomial noise (var ~0.02-0.03), independent of subject parameters.

PWEA (polarity s<0, compensatory weights): on the dominance pair the weighted mass of 1s is 2.0 (gamma=1) or 1.1 (gamma=0.1) for A and 0 for B, so PWEA REJECTS A -> hit rate well below 0.5. On the conflict pair the mass is 2.0 for A vs 3.0 for B (gamma=1) or 1.1 vs 3.9 (gamma=0.1), so PWEA ACCEPTS A -> hit rate well above 0.5, agreeing with TTB. Hence the difference is strongly positive (~0.5-0.9) for essentially the whole admissible parameter box, and crucially beta, |s| and epsilon largely cancel within subject, keeping between-subject variance modest.

So the expected values are ~0 (TTB) vs ~0.6-0.7 (PWEA) with small per-subject spread: a large, tight contrast that is not trivially true of PWEA (it is a signed, direction-reversing prediction about two specific stimulus structures that any one-reason rule cannot produce).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def as_str(x):
        return ''.join([str(int(v)) for v in list(x)])

    dom_hits = []
    conf_hits = []

    for _, row in data.iterrows():
        a = [int(v) for v in list(row['option_a_ratings'])]
        b = [int(v) for v in list(row['option_b_ratings'])]
        if len(a) != len(b) or len(a) == 0:
            continue
        # Cue order = feature index order (validities are strictly descending by design)
        winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hit = 1.0 if resp == winner else 0.0

        sa = as_str(a)
        sb = as_str(b)
        key = tuple(sorted([sa, sb]))

        # DOMINANCE base pair: [1,0,0,0,0] vs [0,0,0,0,0]  (trials 1 and 9)
        if key == ('00000', '10000'):
            dom_hits.append(hit)
        # CONFLICT base pair: [1,0,0,0,0] vs [0,1,1,1,1]  (trials 4 and 12)
        elif key == ('01111', '10000'):
            conf_hits.append(hit)

    if len(dom_hits) == 0 or len(conf_hits) == 0:
        return 0.0

    return float(np.mean(conf_hits) - np.mean(dom_hits))

### RESULTS
- Predicted under pi_1 (simulated): 0.0183 (var=0.0212)
- Predicted under pi_3 (simulated): 0.6550 (var=0.0311)
- Observed on real data: 0.6600 (var=0.0633)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def ttb_winner(a, b):
        # features are ordered by descending validity (index 0 = 0.9, ... index 4 = 0.55)
        for j in range(len(a)):
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    def tally_winner(a, b):
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        if a_wins > b_wins:
            return 0
        if b_wins > a_wins:
            return 1
        return -1

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row["option_a_ratings"]]
            b = [int(v) for v in row["option_b_ratings"]]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        w_ttb = ttb_winner(a, b)
        w_tal = tally_winner(a, b)
        # keep only trials where the two heuristics make opposite, well-defined predictions
        if w_ttb < 0 or w_tal < 0 or w_ttb == w_tal:
            continue
        resp = row["response"]
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        n += 1
        if int(resp) == w_ttb:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.7744 (var=0.0262)
**Predicted under pi_1:** 0.8172 (var=0.0109)
**Predicted under pi_3:** 0.7611 (var=0.0116)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    agree_hits = []
    conflict_hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        diff = a - b
        nz = np.nonzero(diff)[0]
        if nz.size == 0:
            continue
        # features are listed in descending validity order in this design
        ttb_choice = 0 if diff[nz[0]] > 0 else 1
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin == 0:
            continue
        tally_choice = 0 if margin > 0 else 1
        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        chose_ttb = 1.0 if resp == ttb_choice else 0.0
        if tally_choice == ttb_choice and abs(margin) >= 3:
            agree_hits.append(chose_ttb)
        elif tally_choice != ttb_choice and abs(margin) >= 2:
            conflict_hits.append(chose_ttb)
    if len(agree_hits) == 0 or len(conflict_hits) == 0:
        return 0.0
    return float(np.mean(agree_hits) - np.mean(conflict_hits))
```

**Observed (real) value:** -0.6339 (var=0.0508)
**Predicted under pi_1:** -0.0008 (var=0.0076)
**Predicted under pi_3:** -0.6858 (var=0.0254)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments the pattern is unambiguous: the data show a systematic *negative polarity* \u2014 subjects reject the option carrying the greater mass of expert '1's \u2014 combined with compensatory (not one-reason) integration. Take-The-Best (pi_1) is pinned at ~0 on the three signed-contrast metrics (Exp1 0.001 vs observed -0.577; Exp2 0.018 vs observed 0.660; Exp4 -0.001 vs observed -0.634) and only survives Exp3 (0.817 vs 0.774), which is the single experiment restricted to classic TTB-vs-tally conflict items where every negative-polarity model mimics TTB by construction. TTB is therefore falsified in three of four designs and is not task-invariant. PWEA (pi_3) nails all four: -0.612/-0.577, 0.655/0.660, -0.686/-0.634, 0.761/0.774. It is the clear incumbent and should be retained.\n\nHowever, PWEA is currently unopposed by anything that shares its winning ingredient (negative polarity + integration). Note also that a much simpler account reproduces the same four numbers almost exactly: 'choose the option with the FEWER 1s, counted without validity weighting, with a single fixed accuracy h'. Solving Exp1's metric (0.875 - 1.75h = -0.577) gives h = 0.831, which then predicts Exp2 = 2h-1 = 0.662 (observed 0.660), Exp4 = -(2h-1) = -0.662 (observed -0.634), and Exp3 = h = 0.831 (observed 0.774). So all the data so far are consistent with a validity-blind negative count, and none of the designs has cleanly demonstrated that the *validity-graded weighting* and *margin-graded softmax* parts of PWEA are doing any work. Replacing the dead TTB with this parsimonious rival is far more informative than regenerating TTB's parameters, and it sets up sharp future tests (equal-count pairs differing in which cue carries the 1, e.g. [10000] vs [00010] or [00100] vs [00010]; and count-vs-weighted-mass disagreement pairs).",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 (Take-The-Best) with a negative-polarity, VALIDITY-BLIND counting theory \u2014 call it 'Flag Counting with Lexicographic Tiebreak (FC-LEX)'. Core claims: (1) a rating of '1' is read as a warning flag, so an option's badness score is simply the NUMBER of 1s it carries, with all experts counted equally (validities are acknowledged in the instructions but are not used to weight the count); (2) the option with the strictly smaller flag count is chosen; the choice probability depends only on the SIGN of the count difference (or at most saturates after a difference of 1), NOT on validity-weighted magnitude \u2014 implement as p_hit = (1-eps)*sigmoid(beta*sign(dCount)) + eps/2, i.e. a single flat accuracy h across all discriminating trials; (3) when counts tie exactly, the subject falls back on a one-reason lexicographic tiebreak using the highest-validity discriminating cue (with a free polarity flag for whether the '1' on that cue attracts or repels), which yields near-chance-to-moderate performance on equal-count items. Parameter ranges should be chosen so the pooled flat accuracy h lands near 0.80-0.85 (e.g. beta ~ 1.2-2.5, eps ~ 0.05-0.25) with modest between-subject spread, since h ~ 0.83 reproduces Exp1 (-0.58), Exp2 (+0.66), Exp3 (~0.83) and Exp4 (-0.66) simultaneously. This theory agrees with PWEA on every design run so far but diverges sharply where count and validity-weighted mass dissociate: equal-count pairs such as [1,0,0,0,0] vs [0,0,0,0,1] or [0,0,1,0,0] vs [0,0,0,1,0] (FC-LEX: tiebreak/near-flat; PWEA: strongly graded reversal ~5:1), pairs where the option with more 1s has the smaller weighted mass, and Exp3-style margin ladders (FC-LEX: flat accuracy conditional on the sign of the count difference; PWEA: monotone psychometric function in the weighted margin). That contrast is the next decisive test."
}
```

## Usage

```json
{
  "input_tokens": 16159,
  "output_tokens": 5379
}
```
