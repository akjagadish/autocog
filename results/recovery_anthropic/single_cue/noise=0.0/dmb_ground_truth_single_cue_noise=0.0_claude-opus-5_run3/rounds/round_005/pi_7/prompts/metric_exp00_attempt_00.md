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
**Validities (n_features=7):** [0.7, 0.95, 0.86, 0.7, 0.6, 0.78, 0.7]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0]
  trial 5: A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0]
  trial 11: A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1, 1]
  trial 13: A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0]
  trial 14: A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  trial 15: A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  trial 16: A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate the advocated REASON-COUNTING WITH A TWO-REGIME SALIENCE GATE (validity-graded vote, symmetric first/last-row salience at FULL force only when the reason count is deadlocked and at a CONSTANT residual g_res~.25 otherwise, saturating majority bonus, unanimity bonus, sharp silence threshold in which mutual endorsements count only at weight omega~.82, c floored at c_min) from READ-UNTIL-SUFFICIENT INTEGRATION (competitor: compressed diagnosticity x a PRIMACY-dominated attention multiplier, serial read with a sufficiency stop, FULLY RELATIVE evidence R, convex kappa read-out, and a silence discount in which EVERY non-discriminating feature counts identically).

VALIDITIES BY SCREEN POSITION: [.70, .95, .86, .70, .60, .78, .70]. Seven experts is chosen deliberately: with exactly 3 of 7 features silent, the advocated model's effective silence fraction is 3/7 = .429 when the silent experts endorsed NEITHER product but .82*3/7 = .351 when they endorsed BOTH -- values that STRADDLE its sharp threshold sil_f0 ~ .36 (sigmoid weights .93 vs .42 at k=38), while the competitor's f_sil is 3/7 in BOTH cases, i.e. a mathematically forced null. Three experts share validity .70 and sit at pos0, pos3 and pos6 (both screen ends plus the exact middle), which permits validity-matched swaps of screen position. The oracle (.95) is deliberately placed at pos1 rather than pos0, and in EVERY cell the first discriminating cue is the .70 expert at pos0, so the advocated model's authority-capped first-cue commitment is uniformly suppressed (ratio .40/(1.05*.90) = .42 -> q_com ~ .01) and cannot contaminate any contrast. Advocated salience e_sal (tau=1) = [1.00,.37,.15,.10,.15,.37,1.00] (symmetric U); competitor attention (alpha~.9,tau~1.3,gamma<=.15) = [1.90,1.42,1.20,1.09,1.05,1.04,1.06] (primacy only, essentially flat over pos2-pos6), giving competitor weights w = [.66,1.26,.82,.38,.17,.54,.37] -- dominated by the mid-list .95 expert.

8 base cells, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials. Mirroring makes the predicted-choice split exactly 8 A / 8 B for BOTH theories, so side bias cannot mimic anything.

(1) SILENCE-COMPOSITION TRIAD, cells 1-3 (pairs 1-3 + mirrors 9-11). FLAGSHIP. All three carry the IDENTICAL difference vector d = [+1,0,+1,0,+1,+1,0] (unanimous 4-0 for A; .70,.86,.60,.78 speak; pos1, pos3, pos6 silent). They differ ONLY in what the three silent experts said: cell 1 all both-0, cell 2 two both-1 + one both-0, cell 3 all both-1. The competitor's silence term counts non-discriminating features regardless of level and its evidence uses only discriminating cues, so it is FORCED to predict exactly the same probability in all three (stop at pos2 with R = 1, P(A) ~ .94) for every parameter setting. The advocated model's omega-weighted, threshold-sharp term gives c = .21 / .45 / .65 and hence P(A) ~ .78 / .93 / .98 -- a ~20-point MONOTONE ORDERED ladder produced by ratings that discriminate nothing at all.

(2) DEADLOCK-GATE CELL, cell 4 (pairs 4, 12). d = [+1,-1,-1,0,0,0,+1]: the two screen EXTREMES (.70, .70) favour A, the two mid-list heavyweights (.95, .86) favour B, three silent both-1. Because kA = kB = 2 the advocated gate opens to FULL force (g = 1): drive = V + mu*S = -.80 + 2.66 = +1.86, and with c ~ .56 it predicts P(A) ~ .73-.85 across its whole box. The competitor reads on, stops at pos2 with R ~ -.52 and predicts P(A) ~ .29. Opposite direction, ~50 points.

(3) VALIDITY-MATCHED SALIENCE SWAP AT |dk| = 1, cells 5 & 6 (pairs 5-6, 13-14). The residual-gate signature. Cell 5: A supported by pos0(.70), pos2(.86), pos3(.70), opposed by pos1(.95) and pos6(.70). Cell 6: the two .70 experts trade places (A supported by pos0, pos2, pos6; opposed by pos1, pos3). The validity multiset on each side is IDENTICAL, the silence composition is identical (two both-0), so the vote term V = +.25 is identical and any validity-based model must be flat. The advocated model's constant residual salience (g_res ~ .25, mu ~ 1.8) still reads the screen ends: S goes from -.12 to +1.68, so P(A) goes from ~.55 to ~.76 (delta logit ~ .8-1.0 across mu, g_res, tau). The competitor's position attention is primacy-shaped and nearly identical at pos3 (w = .382) and pos6 (w = .370), and its relative evidence gives P(A) ~ .54 vs ~ .53 -- a ~1-point null, if anything in the opposite direction. A 20-point swap effect vs a flat line isolates SYMMETRIC (first-and-last) salience surviving as a constant bonus outside the deadlock regime.

(4) SUFFICIENCY-STOP vs FULL INTEGRATION, cell 7 (pairs 7, 15). d = [+1,+1,0,0,0,-1,-1]: pos0(.70) and pos1(.95) favour A, pos5(.78) and pos6(.70) favour B, three silent both-1. The competitor's running lead after pos1 (|E| = 1.92) clears the bar set by the strongest unread expert (1.05*.535 = .56), so it stops and commits categorically on R = +1: P(A) ~ .94. The advocated model is deadlocked (kA = kB = 2) so its gate opens, but the salience term cancels EXACTLY (e_sal at pos0+pos1 equals e_sal at pos5+pos6), leaving only the small vote V = +.32 discounted by c ~ .59: P(A) ~ .56. A ~38-point confidence gap in the same direction, driven purely by stopping-plus-normalisation vs. full graded integration -- and a cell where the advocated model's salience machinery is provably inert, so the gap cannot be attributed to position.

(5) FOUR-AGAINST-THE-ORACLE CELL, cell 8 (pairs 8, 16). d = [-1,-1,+1,+1,+1,0,+1]: the leading .70 expert AND the oracle .95 expert favour B, while four later experts (.86,.70,.60,.70) favour A; only ONE expert is silent, so both models' silence machinery is essentially off and cannot be blamed. The competitor accumulates -1.92 over the first two positions, clears the .86 bar and commits on R = -1: P(A) ~ .05. The advocated model's near-linear vote (V = +.40 to +.53 across rho) plus the saturating majority bonus (|dk| = 2 -> M ~ .67) gives drive ~ 1.0-1.2 and P(A) ~ .75-.82. Opposite direction, ~70 points, over both full parameter boxes.

DIAGNOSTIC YIELD: (i) a three-cell forced-null triad where the competitor is pinned to one number and the advocated theory predicts a 20-point ordered ladder produced by non-discriminating ratings; (ii) a validity-matched position swap (cells 5 vs 6) that isolates symmetric residual salience against a primacy-only, fully normalising alternative, where the competitor is again forced to a null; (iii) two full direction disagreements (cells 4 and 8, 50 and 70 points) covering both the deadlock regime and the majority-bonus regime; (iv) one same-direction magnitude dissociation (cell 7, 38 points) in which the advocated model's salience term cancels by construction. Every pair has at least one discriminating cue; no cell is degenerate.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Reason-counting with a TWO-REGIME display-salience gate (full force only on a deadlocked vote, a constant residual otherwise), near-linear diagnosticity weighting, a saturating majority bonus, unanimity boost, sharp silence-to-confidence collapse (mutual endorsement ≈ silence), and a non-zero floor on read-out gain.**

This keeps every mechanism of the accepted base and changes exactly three calibration commitments (one of them structural-but-local: the shape of the salience gate).

1. **Direction is a vote over discriminating experts, graded near-linearly by stated validity.** Each discriminating expert contributes x_j = ((v_j-0.5)/0.5)^rho with rho ~0.88-1.02. At this exponent a single .95 expert is worth about two upper-middling experts (.72 + .70), so numerosity and diagnosticity are *balanced* rather than one dominating: three mid experts still outvote one top expert, but two mid experts only tie it. Screen position plays no role in this term.

2. **Display salience is a TIE-BREAKER WITH A RESIDUAL, not a quantity that decays with the size of the majority.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) over the literal display row, multiplied by a gate g = max(exp(-(k_A-k_B)^2/2 sigma^2), g_res) with sigma ~0.60 and g_res ~0.25. The gate has two regimes: when the reason count is exactly deadlocked the first/last rows of the panel decide the choice outright (g = 1); as soon as one side has more reasons, position stops being decisive but does NOT disappear — it persists as a constant, modest attention bonus (g = g_res ≈ 0.25) that is the SAME for a one-reason edge and for a five-against-one landslide. The previous Gaussian-only gate made position vanish geometrically with the majority size, which is unmotivated: there is no reason why a larger majority should erase the extra attention paid to the top and bottom of the screen. The empirical signature is that a three-against-one display whose majority occupies both screen ends is markedly more decisive (~.88) than a three-against-one display whose lone dissenter sits at a screen end (~.82), even when the validities are matched.

3. **Clear majorities are categorical and saturating:** M = lambda*sign(k_A-k_B)*min(max(|k_A-k_B|-1,0),1) with lambda ~0.67. Once one side has at least two more reasons, 'more experts back this one' is itself a discrete reason, and it does not grow further with the size of the majority.

4. **Unanimity among the speaking experts is categorical**: if every expert who noticed a difference points the same way, a constant bonus is added.

5. **Gap inference from the panel is a sharp threshold on how much of the panel was informative, and 'both products are good' is nearly as uninformative as 'neither is'.** Validity is read as sensitivity: a difference only sensitive experts notice implies a marginal true gap. Mutual endorsements count at weight omega ~0.8. c = 1 - eta*sigmoid(k*(f_sil - f0))*vbar.

6. **The collapse of confidence is bounded below:** c is floored at c_min ~0.10, so a panel judged uninformative leaves a small residual tilt rather than a literal coin flip.

7. **A small, display-contingent first-cue commitment survives**: people commit to the first expert who actually discriminates only if no expert further down the list outranks him in stated validity. Individual differences are unimodal jitter in rho, mu, sigma, g_res, lambda, eta, omega, c_min, beta; no strategy subpopulations, no serial stop rule, no learning.

**Parameters:**
- rho: [0.88, 1.02]
- mu: [1.70, 1.95]
- lam: [0.60, 0.74]
- sigma: [0.55, 0.66]
- g_res: [0.22, 0.28]
- tau: [0.85, 1.25]
- ups: [1.25, 1.80]
- beta: [1.05, 1.40]
- eta: [1.45, 1.65]
- sil_k: [30.0, 45.0]
- sil_f0: [0.345, 0.375]
- omega: [0.72, 0.92]
- c_min: [0.07, 0.13]
- phi: [0.95, 1.12]
- s_stop: [6.0, 10.0]
- d_com: [2.20, 2.90]
- epsilon: [0.01, 0.06]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> diagnosticity (= sensitivity)
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    rho = float(np.clip(parameters.get('rho', 0.95), 0.05, 4.0))
    mu = float(np.clip(parameters.get('mu', 1.82), 0.0, 8.0))
    lam = float(np.clip(parameters.get('lam', 0.67), 0.0, 3.0))
    sigma = float(np.clip(parameters.get('sigma', 0.60), 0.15, 3.0))
    g_res = float(np.clip(parameters.get('g_res', 0.25), 0.0, 0.6))
    tau = float(np.clip(parameters.get('tau', 1.0), 0.2, 8.0))
    ups = float(np.clip(parameters.get('ups', 1.50), 0.0, 4.0))
    beta = float(np.clip(parameters.get('beta', 1.20), 0.05, 40.0))
    eta = float(np.clip(parameters.get('eta', 1.55), 0.0, 5.0))
    sil_k = float(np.clip(parameters.get('sil_k', 38.0), 1.0, 80.0))
    sil_f0 = float(np.clip(parameters.get('sil_f0', 0.36), 0.05, 0.9))
    omega = float(np.clip(parameters.get('omega', 0.82), 0.0, 1.0))
    c_min = float(np.clip(parameters.get('c_min', 0.10), 0.0, 0.5))
    phi = float(np.clip(parameters.get('phi', 1.05), 0.3, 3.0))
    s_stop = float(np.clip(parameters.get('s_stop', 8.0), 0.5, 60.0))
    d_com = float(np.clip(parameters.get('d_com', 2.50), 0.5, 8.0))
    eps = float(np.clip(parameters.get('epsilon', 0.035), 0.0, 0.5))

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. reason count, graded by stated validity (no position here)
    # ------------------------------------------------------------------
    x = np.power(vs, rho)
    idxs = np.flatnonzero(disc)
    m = int(idxs.shape[0])
    sgn = np.sign(d[idxs])
    xs = x[idxs]
    V = float(np.sum(sgn * xs))

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))

    # ------------------------------------------------------------------
    # 5. display salience over ABSOLUTE SCREEN POSITION (first & last
    #    rows of the panel), deployed as a TWO-REGIME tie-breaker:
    #    FULL force when the reason count is deadlocked, and a CONSTANT
    #    residual fraction g_res once either side has more reasons --
    #    position stops deciding, but it does not evaporate with the
    #    size of the majority.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    e_sal_full = np.exp(-pos / tau) + np.exp(-(n - 1.0 - pos) / tau)
    e_sal = e_sal_full[idxs]
    S = float(np.sum(sgn * e_sal))
    dk = float(kA - kB)
    g = float(np.exp(-np.clip((dk * dk) / (2.0 * sigma * sigma), 0.0, 60.0)))
    g = float(max(g, g_res))

    # ------------------------------------------------------------------
    # 5b. saturating majority bonus (fires only at |dk| >= 2)
    # ------------------------------------------------------------------
    maj = min(max(abs(dk) - 1.0, 0.0), 1.0)
    M = lam * (1.0 if dk > 0 else (-1.0 if dk < 0 else 0.0)) * maj

    # ------------------------------------------------------------------
    # 6. unanimity among the speakers is categorical
    # ------------------------------------------------------------------
    uni = 0.0
    if kA == 0 or kB == 0:
        uni = ups * (1.0 if kA > 0 else -1.0)

    drive = V + mu * g * S + uni + M

    # ------------------------------------------------------------------
    # 7. gap inference from panel silence (validity = sensitivity).
    #    An expert who endorsed BOTH products has positively asserted the
    #    two are alike on his dimension: he is ALMOST as uninformative
    #    about the quality gap as one who endorsed neither (weight omega).
    #    SHARP collapse, but BOUNDED BELOW at c_min.
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    both0 = (~disc) & (~both1)
    n_sil_eff = float(np.count_nonzero(both0)) + omega * float(np.count_nonzero(both1))
    f_sil = n_sil_eff / float(n)
    f_sil = float(np.clip(f_sil, 0.0, 1.0))
    tot_x = float(np.sum(xs))
    if tot_x <= 0:
        return np.array([0.5, 0.5])
    vbar = float(np.sum(xs * vs[idxs]) / tot_x)
    c = 1.0 - eta * float(_sig(sil_k * (f_sil - sil_f0))) * vbar
    c = float(np.clip(c, c_min, 1.0))

    p_int = _sig(beta * c * drive)

    # ------------------------------------------------------------------
    # 8. authority-capped first-cue commitment (display-contingent)
    # ------------------------------------------------------------------
    j1 = int(idxs[0])
    if j1 >= n - 1:
        q_com = 1.0
    else:
        v_rest = float(np.max(vs[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = vs[j1] / (phi * v_rest)
        q_com = _sig(s_stop * (ratio - 1.0))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = _sig(beta * c * d_com)
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
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
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Read-until-sufficient integration with mildly compressed diagnosticity, silence-calibrated evidence, and an S-SHAPED (convex) read-out of relative evidence.**

People do not run a fixed heuristic (TTB, tally, or weighted-additive). They read the expert ratings in the order the screen presents them, accumulate weighted differences, and stop as soon as their running lead is safe from any single expert they have not yet read. Six claims:

1. **Cue weights = MILDLY COMPRESSED stated diagnosticity x a modest reading-position bonus.** w_j = ((v_j-0.5)/0.5)^rho * (1 + alpha*exp(-j/tau) + gamma*exp(-(n-1-j)/tau)), with rho ~1.0-1.3, i.e. subjective weight is essentially linear-to-slightly-convex in (v-0.5). A .95 expert is only ~2x a .72 expert and ~5-8x a .56 expert -- clearly heavier, and roughly *balanced* by a pair of upper-middling experts (.78 + .72), but routinely outvotable by three mid-validity experts ('tally-like without unit-weight tallying'). This exponent sits at the crossover point where the two screen-extreme flankers of a mid-list .95 expert neither dominate it nor are dominated by it, which is what turns double-dissociation displays into genuine coin flips. Position is an attention multiplier, never the causal driver.

2. **Serial reading with a graded sufficiency stop.** After each rating that actually discriminates, the reader asks whether any single unread expert could still overturn the current lead; the bar is theta = phi * max weight among unread positions and stopping is graded, q = sigmoid(s*(|E|/theta - 1)). This is the only place where screen order has structural force, and it is display-contingent. Because the bar is set by the strongest unread expert, a merely salient first cue stops the read only when its weighted lead genuinely exceeds what any single remaining expert could reverse -- and as diagnosticity compression relaxes (higher rho) a salient-but-mediocre first cue loses that race to an unread .95 expert.

3. **Evidence is read FULLY relatively, with no load machinery.** R = sum(w*d)/sum(w*|d|) over the cues read so far; features on which both products are endorsed (or both unendorsed) are simply dropped, so any dense-vs-sparse contrast is exactly zero.

4. **Silence calibration.** Validity is treated as sensitivity: a difference only a top expert notices while the rest of the panel is silent implies a marginal quality gap and is acted on hesitantly. Gain is multiplied by c = 1 - eta*(fraction silent)^q * (weighted mean diagnosticity of the speaking experts), which bites only when silence is overwhelming, producing a *reversed* single-cue validity ladder. This term is a function of raw diagnosticity, not of the compressed weights, so the ladder is invariant to rho.

5. **Decisiveness is a CONVEX (S-shaped) function of the evidence balance, not a linear one.** A near-balanced set of reasons does not merely produce a slightly tilted preference; it produces genuine indifference. Read-out is logistic(beta * c * sign(R)*|R|^kappa) with kappa ~1.4-1.9. A small residual imbalance (|R| ~ 0.1-0.2, e.g. two screen-extreme cues almost exactly offsetting one top-validity cue) is discounted to near-nothing and the choice is a coin flip; a clear weighted majority (|R| ~ 0.5) is still followed at ~0.78-0.82; and a categorical conclusion (|R| = 1, i.e. a sufficiency stop, or a single discriminating cue) is unchanged, so committed reads stay decisive.

6. **Choice** = logistic(beta * c * sign(R)|R|^kappa) with lapse epsilon. Individual differences are unimodal jitter in rho, alpha, phi, s, kappa, beta, eta and lapse -- no strategy subpopulations, no primacy/recency reader types, no learning (there is no feedback).

**Parameters:**
- rho: [1.00, 1.32]
- alpha: [0.7, 1.2]
- gamma: [0.0, 0.15]
- tau: [1.0, 1.6]
- phi: [0.95, 1.20]
- s: [7.0, 14.0]
- beta: [2.80, 3.50]
- kappa: [1.35, 1.85]
- eta: [1.5, 2.1]
- sil_pow: [2.4, 3.0]
- epsilon: [0.0, 0.06]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        else:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-9, 1.0)      # scaled diagnosticity in (0,1]

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    rho = float(np.clip(parameters.get('rho', 1.15), 0.1, 6.0))
    alpha = float(np.clip(parameters.get('alpha', 0.9), 0.0, 4.0))
    gamma = float(np.clip(parameters.get('gamma', 0.05), 0.0, 3.0))
    tau = float(np.clip(parameters.get('tau', 1.3), 0.2, 8.0))
    phi = float(np.clip(parameters.get('phi', 1.05), 0.3, 3.0))
    s = float(np.clip(parameters.get('s', 10.0), 0.5, 80.0))
    beta = float(np.clip(parameters.get('beta', 3.15), 0.05, 40.0))
    kappa = float(np.clip(parameters.get('kappa', 1.6), 0.5, 5.0))
    eta = float(np.clip(parameters.get('eta', 1.8), 0.0, 5.0))
    q_sil = float(np.clip(parameters.get('sil_pow', 2.7), 1.0, 8.0))
    eps = float(np.clip(parameters.get('epsilon', 0.03), 0.0, 0.5))

    # ------------------------------------------------------------------
    # 4. cue weights: compressed diagnosticity x modest reading-position
    #    attention.  rho ~1.0-1.3 places a .95 cue at roughly the combined
    #    strength of two upper-middling cues, so screen-extreme flankers
    #    neither dominate nor are dominated by a mid-list top expert.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    att = 1.0 + alpha * np.exp(-pos / tau) + gamma * np.exp(-(n - 1.0 - pos) / tau)
    w = np.power(vs, rho) * att
    w = np.clip(w, 1e-12, None)

    # ------------------------------------------------------------------
    # 5. silence calibration: validity = sensitivity.  A difference that
    #    only the most sensitive experts can see, while the rest of the
    #    panel sees none, implies a marginal quality gap -> discounted.
    #    A difference an insensitive expert notices implies a large gap.
    # ------------------------------------------------------------------
    m = int(np.count_nonzero(disc))
    f_sil = float(n - m) / float(n)
    wd = w[disc]
    tot_wd = float(np.sum(wd))
    if tot_wd <= 0:
        return np.array([0.5, 0.5])
    vbar = float(np.sum(wd * vs[disc]) / tot_wd)
    c = 1.0 - eta * (f_sil ** q_sil) * vbar
    c = float(np.clip(c, 0.02, 1.0))

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    # convex (S-shaped) read-out of the relative balance: |R| = 1 is
    # untouched, small residual imbalances are discounted toward chance.
    def _drive(R):
        R = float(np.clip(R, -1.0, 1.0))
        return float(np.sign(R) * (abs(R) ** kappa))

    # ------------------------------------------------------------------
    # 6. serial reading in screen order with graded sufficiency stopping;
    #    evidence is read fully relatively (R in [-1, 1]).
    # ------------------------------------------------------------------
    idxs = np.flatnonzero(disc)
    E = 0.0
    M = 0.0
    remain = 1.0
    p_a = 0.0
    last = int(idxs.shape[0]) - 1
    for t in range(idxs.shape[0]):
        j = int(idxs[t])
        E += w[j] * (1.0 if d[j] > 0 else -1.0)
        M += w[j]
        R = (E / M) if M > 1e-12 else 0.0
        p_here = _sig(beta * c * _drive(R))
        if t >= last:
            q = 1.0
        else:
            tail = w[j + 1:]
            bar = phi * float(np.max(tail)) if tail.size > 0 else 0.0
            if bar <= 1e-12:
                q = 1.0
            else:
                q = _sig(s * (abs(E) / bar - 1.0))
        p_a += remain * q * p_here
        remain *= (1.0 - q)
        if remain <= 1e-12:
            break
    if remain > 1e-12:
        R = (E / M) if M > 1e-12 else 0.0
        p_a += remain * _sig(beta * c * _drive(R))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
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
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Difference-vector signatures for the two full-direction-disagreement cells\n    # (cell 4: deadlock/extremes cell, pairs 4 & 12;  cell 8: four-against-the-oracle, pairs 8 & 16)\n    # value = index of the option favoured by the reason-counting (advocated) account\n    targets = {\n        (1, -1, -1, 0, 0, 0, 1): 0,     # pair 4  -> A\n        (-1, 1, 1, 0, 0, 0, -1): 1,     # pair 12 -> B\n        (-1, -1, 1, 1, 1, 0, 1): 0,     # pair 8  -> A\n        (1, 1, -1, -1, -1, 0, -1): 1,   # pair 16 -> B\n    }\n\n    hits = []\n    for a_r, b_r, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):\n        try:\n            a = [int(v) for v in a_r]\n            b = [int(v) for v in b_r]\n        except Exception:\n            continue\n        if len(a) != len(b):\n            continue\n        d = tuple(int(ai - bi) for ai, bi in zip(a, b))\n        if d in targets:\n            fav = targets[d]\n            try:\n                r = int(resp)\n            except Exception:\n                continue\n            hits.append(1.0 if r == fav else 0.0)\n\n    if len(hits) == 0:\n        return float('nan')\n    return float(np.mean(hits))",
  "rationale": "The design contains two cells where the two accounts predict OPPOSITE choices, and each cell is presented twice (base + side-mirror) x 6 reps = 24 trials per subject, so a simple proportion has small binomial noise.\n\nCell 4 (d = [+1,-1,-1,0,0,0,+1]; the two .70 screen-extreme experts vs. the mid-list .95 and .86 heavyweights, three both-1 silent): the advocated reason-counting model is at an exact vote deadlock (kA = kB = 2), which opens its salience gate to full force, so the first/last screen rows decide and it picks the extremes-backed option with P ~ .73-.85. The read-until-sufficient competitor weights by compressed validity x a primacy-only attention profile, reads on past the deadlock and lands on a clearly negative relative evidence R ~ -.5, giving P ~ .29 for that same option.\n\nCell 8 (d = [-1,-1,+1,+1,+1,0,+1]; leading .70 expert and the .95 oracle favour one side, four later experts the other, only one silent feature so neither model's silence machinery is engaged): the advocated model's near-linear vote plus saturating majority bonus favours the four-expert side at P ~ .75-.82; the competitor accumulates a large early lead from the first two (heavily attention-weighted) positions, clears its sufficiency bar, commits on R = -1 and gives P ~ .05 for the four-expert side.\n\nThe metric is the proportion of choices, over those 24 trials, that agree with the option the advocated theory favours. Predicted values: ~0.78 under reason-counting-with-salience-gate vs. ~0.17 under read-until-sufficient integration -- a ~0.6 gap. Because the cells are mirrored across sides, side bias cannot inflate or deflate the score, and because both cells sit far from indifference in both models the per-subject variance is dominated by binomial noise over 24 trials (~0.007) plus modest parameter jitter, which should keep Welch's t large at N = 25."
}
```

## Usage

```json
{
  "input_tokens": 15632,
  "output_tokens": 3126
}
```
