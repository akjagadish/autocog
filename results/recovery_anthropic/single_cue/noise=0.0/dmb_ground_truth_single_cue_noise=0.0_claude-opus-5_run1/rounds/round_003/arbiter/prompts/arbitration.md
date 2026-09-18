# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
**Odd-Cue-Out Anchoring under Label Individuation (OCO-LI).**

Four claims, all about *which single row is allowed to anchor the comparison*, and about *whether the anchoring competition can run at all*.

1. **One discriminating row decides, and which one is stochastic.** On each trial the subject settles the A-vs-B comparison by anchoring on a SINGLE row on which the two products differ (a one-reason rule). No summation of evidence occurs, so a validity-weighted integrator is never the right description. Which row wins the anchoring competition varies from trial to trial (not fixed within a subject), so choices are graded within a subject and per-subject metric variance stays essentially binomial — exactly what the population variances in this corpus show.

2. **Anchor salience = ODDNESS, not diagnosticity.** The probability that row j is the anchor is proportional to u_j = exp(lam * j) * (1 - v_j)^delta.
   * *Odd-cue-out (delta > 0):* a disagreement located on a near-chance expert (v ≈ .53) is experienced as a strange, idiosyncratic fact that must be adjudicated, whereas a disagreement on a near-certain expert (v ≈ .97) is discounted as a mere echo of the general impression the subject already has. Communicated validity is therefore used with INVERTED sign. This is why choice is systematically *sub-chance* with respect to stated validity in every design where the labels are usable.
   * *Reading-order recency (lam > 0):* rows are read left-to-right and the later-read discrepancies are the ones still active in working memory when the response is emitted, so they capture the anchor more often. Position is a genuine second determinant (it is required by the 1-cue-late-versus-3-cue-early coalitions), but it is subordinate to, and — crucially — gated by, claim 3.

3. **Label-ambiguity gating (the new mechanism).** The anchoring competition presupposes that each expert can be *individuated*: the subject must be able to bind a particular stated validity to a particular column. That binding is only available when the stated validities are mutually discriminable. Define, for each expert j, its isolation d_j = min_{k≠j} |v_j - v_k| and its individuation s_j = min(1, d_j / tau); the design-level individuation is phi = mean_j s_j. With probability phi the subject runs the odd-cue-out competition above; with probability 1 - phi the columns are interchangeable in memory, no row is 'the odd one', every discriminating row is equally likely to become the anchor, and the choice probability collapses to the *proportion* of discriminating rows favouring each option (a probability-matched tally). When validity labels are duplicated or tightly clustered (e.g. [.95,.70,.55,.95,.70,.55]) phi → 0, the anti-validity and recency effects switch OFF, and a two-row conflict is decided by a coin flip. When all labels are distinct (gaps ≥ .05) phi → 1 and the anti-validity/recency anchoring runs at full strength.

4. **Conflict interference.** The more rows m on which the products differ, the more competing discrepancies must be juggled and the more often the subject loses the thread and simply guesses: P(guess) = 1 - (1 - eps0) * rho^(m-2). Two-row comparisons are made nearly cleanly; five- and six-row comparisons drift toward chance. Difficulty grows with the *number* of discrepancies, not with their diagnosticity — which is why unanimous (dominance) trials are far from ceiling and get worse as more experts speak.

The rule is stationary: there is no feedback, hence no learning across the block.

*Sharp falsifiable contrasts:* (i) hold cue content fixed and only duplicate versus de-duplicate the stated validity vector — OCO-LI predicts the anti-validity effect switches off and on; any recency-only or validity-integration account predicts no change. (ii) With all-distinct labels, put the low-validity discriminating row EARLY and the high-validity row LATE: OCO-LI predicts a near-even split (oddness and recency now oppose each other), whereas a pure recency account predicts the late high-validity option and a Bayesian account predicts it even more strongly.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
**Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS).**

Three claims, all about *which single cue gets to decide*, not about integration.

1. **One cue decides, but which one is stochastic.** On every trial the subject settles the comparison on a SINGLE discriminating expert row (a one-reason rule). Which row wins the competition for the decision varies from trial to trial, so choice is graded *within* a subject (this is required by the data: per-subject metric variances are essentially binomial, i.e. subjects are homogeneous but individually stochastic, ruling out fixed per-subject cue orders). The deciding row j is drawn from the discriminating set D with probability proportional to a salience weight u_j.

2. **Salience = serial recency × distinctiveness (redundancy-discounted validity).** u_j = exp(lambda*j) * (1 - v_j)^delta.
   * *Recency (lambda > 0):* rows are read top-to-bottom / left-to-right and the later-read rows are the ones still active in working memory when the choice is made, so they capture the decision. Column position, not stated diagnosticity, is the dominant driver (this is the arbiter's core claim, retained).
   * *Distinctiveness (delta > 0):* subjects treat a near-certain expert (v = .97) as *redundant* — "I already know what he will say, he just echoes the consensus" — whereas a row on which a near-chance expert (v = .53) discriminates feels like idiosyncratic, decision-relevant information that must be adjudicated. Communicated validity is therefore used, but with inverted sign: high-validity rows are discounted as uninformative repeats of the general impression. This is why choices are systematically *sub-chance* with respect to validity in every experiment.

3. **Conflict interference.** The more rows on which the two products differ, the more competing comparisons must be held simultaneously, and the more often the subject loses the thread and simply guesses: P(guess) = 1 - (1-eps0)*rho^(m-2), where m = |D|. Two-row comparisons are made almost cleanly (hence near-deterministic anti-validity choice when only a .97 row and a .53 row differ); five- and six-row comparisons collapse toward coin-flipping. Difficulty grows with the number of differences, not with their diagnosticity.

No learning, no feedback use: the rule is stationary across the block.

*Why the two extra mechanisms are needed on top of pure position-recency (justified deviation from the arbiter's exact form):* pure position models (graded recency Luce, or its Take-The-Last limit) plus a uniform lapse are mathematically unable to reproduce the observed ordering Exp4 (0.15) < Exp1 (0.28) ≈ Exp3 (0.29): Exp4's diagnostic trial is an ADJACENT single-pair contrast (column 1 vs column 2), whose raw position-based prediction (1/(1+e^lambda)) is always *higher* (closer to chance) than the multi-cue Exp1/Exp3 trials, so with a shared lapse floor Exp4 can never come out lowest. Distinctiveness fixes this (a .97-vs-.53 contrast is the most lopsided contrast in the whole corpus). Likewise pure position forces Exp2's tie-trial term to exactly 0.5 and its 1-vs-3/1-vs-5 coalition term to ~2/3, capping the Exp2 metric at 0.167 only in the noiseless limit; distinctiveness lowers the tie term (the .93 row loses to the .60 row that precedes it) so that the observed 0.174 is reachable with realistic noise. Conflict interference then supplies the trial-type-dependent floor that lets Exp1/Exp3 sit at ~0.28 while Exp4 sits at ~0.15.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.95, 0.55, 0.75, 0.95, 0.75, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 7: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 8: A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 9: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 10: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  trial 11: A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  trial 12: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 13: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 14: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 15: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 16: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Rationale:** GOAL: dissociate OCO-LI (advocated) from RDS (competitor). The two models are algebraically identical EXCEPT for OCO-LI's label-individuation gate phi: with probability 1-phi the odd-cue-out/recency anchoring competition cannot run (columns are not individuable) and the choice collapses to a probability-matched tally over the discriminating rows. Everything else (salience form exp(lam*j)*(1-v)^delta, conflict interference 1-(1-eps0)*rho^(m-2)) is shared and its parameter ranges essentially overlap. The only way to separate them is therefore a design in which phi is FORCED to 0, and this is achieved parameter-free (independent of tau) by giving every stated validity an exact duplicate.

VALIDITIES: [0.95, 0.55, 0.75, 0.95, 0.75, 0.55]. Each level (.95, .75, .55) appears exactly twice, so d_j = min_k|v_j-v_k| = 0 for every column, s_j = 0, and phi = 0 EXACTLY for any tau in [0.02,0.045]. The arrangement is neither the palindrome nor the repeated-block vector used in earlier designs: the .95 rows sit at columns 0 and 3, the .55 rows at 1 and 5, the .75 rows at 2 and 4, so every same-validity pair spans a different column distance and low/high validities alternate in both display halves. With lam=1.6, delta=0.9 the RDS salience weights are u = [0.067, 2.42, 7.05, 8.19, 172.8, 1453] \u2014 a five-order-of-magnitude spread \u2014 whereas OCO-LI (phi=0) treats every discriminating row as interchangeable.

8 base structures, each with its exact A/B mirror (16 unique pairs, K = 6 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows are filled with matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s.

BLOCK 1 \u2014 SHARED-PARAMETER BASELINES (pairs 1-4). Base 1: unanimous m=2 (A wins cols 0 and 5). Base 2: unanimous m=6 (A wins every row). All discriminating rows point the same way, so both models predict identically (p(A) ~ .86 at m=2, ~ .64 at m=6 for the stated parameter ranges). These pin eps_base and rho for BOTH models, so the contrast trials cannot be explained away as global noise, and they also test the shared claim that unanimity gets WORSE with more agreeing rows.

BLOCK 2 \u2014 CROWN JEWEL: PARAMETER-FREE m=2 SPLIT TIES (pairs 5-12). Four bases in which exactly one row favours A and exactly one row favours B. Under OCO-LI with phi=0 the flat tally is 1/1 -> p_core = .500, and mixing with the guess term leaves p(A) = .500 EXACTLY, for every base and every parameter setting (lam, delta, tau, eps0, rho all drop out). Under RDS the same four displays are strongly lopsided and, crucially, ORDERED by column position and validity:
  \u2022 Base 3: A wins col0 (.95) vs B wins col3 (.95) \u2014 identical validity, distant columns. RDS ~ .15.
  \u2022 Base 4: A wins col2 (.75) vs B wins col4 (.75) \u2014 identical validity, medium distance. RDS ~ .17.
  \u2022 Base 5: A wins col0 (.95) vs B wins col1 (.55) \u2014 adjacent, recency and anti-validity ALIGNED. RDS ~ .16.
  \u2022 Base 6: A wins col1 (.55) vs B wins col3 (.95) \u2014 recency and anti-validity OPPOSED. RDS ~ .30.
A flat .50 profile across all four bases (and their mirrors) is producible ONLY by the gated model; RDS with eps_base <= .32 cannot come within 18 points of chance on any of them, and it is additionally forced to produce the ordered spread .15 < .16 < .17 < .30. Flat-vs-ordered-spread at constant m is the decisive quantitative signature.

BLOCK 3 \u2014 TALLY PROPORTION VS SALIENCE MASS (pairs 13-16).
  \u2022 Base 7 (m=3): A wins cols 0 (.95) and 2 (.75), B wins col5 (.55, last column). OCO-LI: proportion 2/3 -> p(A) ~ .60 (ABOVE chance). RDS: the single late .55 row carries 99.5% of the salience mass -> p(A) ~ .22 (well BELOW chance). ~38-point directional disagreement.
  \u2022 Base 8 (m=4): A wins col5 alone (.55, last), B wins cols 0,1,2. OCO-LI: proportion 1/4 -> p(A) ~ .39 (below chance, the majority option wins). RDS: the lone late row dominates salience -> p(A) ~ .72 (above chance, the minority option wins). A full directional reversal on the same display.

INTERPRETATION: if choices on the four m=2 split bases sit at 50/50 while the m=3 and m=4 conflict bases track the PROPORTION of discriminating rows (above chance for 2-vs-1, below chance for 1-vs-3), the label-ambiguity gate is confirmed and RDS is falsified with essentially no free parameters. If instead the m=2 splits show the strong sub-chance, position-ordered anti-validity pattern (~.15-.30) and the minority-but-late option is chosen ~72% of the time in Base 8, RDS is confirmed and OCO-LI's gating claim is dead. Together with the already-run all-distinct-label designs (where both models coincide), this experiment supplies the exact 'duplicate-versus-de-duplicate the stated validity vector' contrast that OCO-LI names as its own falsification test.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design duplicates every stated validity, so OCO-LI's individuation gate is exactly phi=0: the odd-cue-out/recency anchoring competition cannot run and choices collapse to a probability-matched tally over the discriminating rows. RDS has no such gate, so its salience weights exp(lam*j)*(1-v)^delta (spanning ~5 orders of magnitude here) make the LAST-READ discriminating row essentially decide the trial.

The metric is the rate of choosing the option favoured by the highest-index (last-read) discriminating row, computed only on conflict displays (at least one row favouring each option) and excluding the unanimous baseline trials 1-4, where both theories make identical predictions and which would only dilute the contrast.

Predicted values (using mid-range parameters): OCO-LI gives exactly 0.50 on the eight m=2 split-tie pairs (parameter-free, since the flat tally is 1/1 and the guess mixture preserves 0.5), ~0.41 on the m=3 pairs (tally 2:1 against the late row) and ~0.39 on the m=4 pairs (tally 3:1 against the late row), for an overall value near 0.46-0.47. RDS gives ~0.70-0.86 on the m=2 pairs, ~0.78 on m=3 and ~0.72 on m=4, overall near 0.79. That is a ~0.32 gap.

Precision is high because the metric pools 72 trials per subject (12 unique pairs x 6 reps), so per-subject binomial noise is ~0.06 SD, and OCO-LI's dominant component (the m=2 ties at exactly 0.5) is completely parameter-independent, keeping between-subject variance tiny. Direction of the metric is not trivially tied to either model's mechanism label: it is a purely descriptive 'last-differing-row' choice rate that any theory can be scored on.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        fav_a = d[disc] > 0.0
        # skip unanimous (dominance) displays: both theories agree there
        if fav_a.all() or (~fav_a).all():
            continue
        # option favoured by the LAST-READ (highest-index) discriminating row
        last_idx = disc[-1]
        late_option = 0 if d[last_idx] > 0.0 else 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hits.append(1.0 if resp == late_option else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_5 (simulated): 0.4711 (var=0.0034)
- Predicted under pi_4 (simulated): 0.7642 (var=0.0020)
- Observed on real data: 0.4383 (var=0.0023)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.955, 0.545, 0.95, 0.54, 0.945, 0.535]

**Trial pairs (n=18):**
  trial 1: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 6: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 10: A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 11: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 12: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 13: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 14: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 15: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 16: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 18: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]

**Rationale:** GOAL: dissociate RDS (advocated) from OCO-LI (competitor). The two models share their salience form exp(lam*j)*(1-v_j)^delta and their conflict-interference term 1-(1-eps0)*rho^(m-2); the ONLY structural difference is OCO-LI's label-individuation gate phi = mean_j min(1, d_j/tau), where d_j is the distance of expert j's stated validity to its nearest neighbour. With probability 1-phi OCO-LI abandons the anchoring competition and probability-matches the raw PROPORTION of discriminating rows. Any design with all-distinct labels (gaps >= .05) drives phi -> 1 and makes the models identical; a design with EXACT duplicates was already run. This design instead uses NEAR-duplicates: v = [0.955, 0.545, 0.950, 0.540, 0.945, 0.535]. Every expert's nearest neighbour is exactly 0.005 away, so d_j = 0.005 for all j and phi = min(1, 0.005/tau) is pinned to [0.11, 0.25] for tau in [0.02, 0.045] (point estimate phi ~ 0.156). This is not degenerate (all six labels are numerically distinct, and high vs low validities differ by 0.41, so the anti-validity/oddness factor is fully alive) but OCO-LI must be ~85% probability-matching while RDS runs the pure salience competition. Layout alternates high/low validity across columns (H,L,H,L,H,L), so recency and anti-validity can be aligned or opposed. RDS salience with lam=1.6, delta=0.9: u = [0.06, 2.44, 1.65, 60.4, 44.2, 1496]. Note u1 > u2 and u3 > u4: one step of oddness slightly outweighs one column of recency, a within-RDS constraint tested by base 5.

9 base structures, each with its exact A/B mirror (18 unique pairs, K = 5 -> 90 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions below use lam=1.6, delta=0.9, eps0=0.28, rho=0.78 (RDS) and the same plus tau=0.032 -> phi=0.156 (OCO-LI).

BLOCK 1 - SHARED BASELINES (bases 1-2, pairs 1-4). Base 1: unanimous m=2 (A wins cols 2,3). Base 2: unanimous m=6 (A wins every row). All discriminating rows point the same way, so p_flat = p_anchor = 1 and BOTH models predict identically (~.86 at m=2, ~.63 at m=6). These pin eps_base and rho for both models, so the contrast trials cannot be dismissed as global noise, and they test the shared claim that unanimity gets WORSE as more rows speak.

BLOCK 2 - m=2 SPLIT TIES, WHERE OCO-LI IS FORCED TOWARD CHANCE (bases 3-5, pairs 5-10). Exactly one row favours A and one favours B, so the flat tally is 1/1 = .500 and OCO-LI can move away from chance only by the factor phi <= .25.
  - Base 3 (A wins col0 .955, B wins col1 .545; adjacent, recency and oddness ALIGNED against A): RDS ~ .16, OCO-LI ~ .45.
  - Base 4 (A wins col0 .955, B wins col5 .535; maximal column distance): RDS ~ .14 (its floor eps0/2), OCO-LI ~ .45.
  - Base 5 (A wins col1 .545, B wins col2 .950; recency and oddness OPPOSED): RDS ~ .57 (oddness narrowly beats one column of recency), OCO-LI ~ .51. This base has little cross-model leverage but is the sharpest internal test of the lam/delta trade-off inside the shared salience rule, and it rules out pure position accounts.
A ~.14-.16 profile on bases 3-4 with a crossover to ~.57 on base 5 is uniquely RDS; a compressed .45/.45/.51 profile hugging chance on all three is the signature of the gate.

BLOCK 3 - CROWN JEWEL: DIRECTIONAL REVERSALS AT ASYMMETRIC m (bases 6, 7, 9; pairs 11-14, 17-18). Here the proportion of discriminating rows and the salience mass point in OPPOSITE directions.
  - Base 6 (m=3): A wins cols 0 (.955) and 2 (.950), B wins col5 (.535, last column). Proportion favours A (2/3) but 99.7% of the salience mass sits on B's late low-validity row. RDS ~ .22 (well BELOW chance); OCO-LI ~ .54 (ABOVE chance). ~32-point directional disagreement.
  - Base 7 (m=4): A wins col5 alone (.535, last), B wins cols 0,1,2. Proportion favours B (3/4); salience favours A. RDS ~ .72; OCO-LI ~ .44. Full directional reversal on one display.
  - Base 9 (m=5): A wins cols 0,1,2,3, B wins col5 alone. Proportion 4/5 for A; salience 96% for B. RDS ~ .34; OCO-LI ~ .56.
Across bases 7, 6, 9 A's share of discriminating rows rises .25 -> .67 -> .80. OCO-LI must therefore produce a MONOTONE INCREASING profile (~.44, .54, .56) tracking the tally proportion, whereas RDS produces a NON-MONOTONE profile dictated entirely by which option owns the last discriminating column (~.72, .22, .34). Monotone-in-proportion versus governed-by-the-last-column is the decisive quantitative signature and needs no free parameters to read off.

BLOCK 4 - BALANCED HIGH-m SPLITS (base 8, pairs 15-16). A wins cols 0,1 (early), B wins cols 4,5 (late); proportion is exactly 1/2 so OCO-LI's ungated component is pinned to .500, while RDS's salience is 99.8% on B's late rows. RDS ~ .28, OCO-LI ~ .47 at m=4. This isolates the position effect at a tally tie with more conflict than block 2, and jointly with base 4 it maps how far interference erodes the anti-validity/recency pull as m grows.

INTERPRETATION: if the m=2 splits sit at .14-.16 (and their mirrors at .84-.86), the 2-vs-1 base 6 sits below chance while the 1-vs-3 base 7 sits above chance, and choices track the OWNER OF THE LAST DISCRIMINATING COLUMN rather than the row count, RDS is confirmed and the label-individuation gate is falsified (with near-duplicate labels the gate should have suppressed exactly these effects). If instead every split trial collapses toward chance and the three asymmetric bases order themselves monotonically by the proportion of rows favouring A, OCO-LI's gating claim is confirmed. No pair is degenerate: on 6 of the 9 bases the models disagree by >18 points or in direction, and the two unanimous bases supply the shared lapse/interference baseline.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
The two theories share everything except OCO-LI's label-individuation gate, which in this near-duplicate-validity design forces phi ~ 0.16, i.e. ~85% probability matching to the raw proportion of discriminating rows. So the decisive question is: on trials where the row-count tally and the single most salient (late, low-validity) discriminating row point in different directions (or the tally is exactly tied and thus uninformative), do choices follow the salient row or hug the tally?

The metric selects exactly those conflict trials (tally tied, or tally majority opposite to the highest-salience discriminating row, using the design's known validities with lam=1.6, delta=0.9) and reports the proportion of choices that follow the salient-row owner. This selection automatically drops the two unanimous baseline bases (where both models agree) and keeps bases 3-9, ~70 of 90 trials per subject, so the metric is tight per subject (binomial with ~70 observations).

Quantitative predictions from the two generative models with central parameters: RDS ~ 0.73-0.76 (salience essentially owns the decision; only the conflict-interference lapse pulls it down), OCO-LI ~ 0.50 (its 84% ungated component is pinned to the tally, which by construction opposes or is neutral w.r.t. the salient row). A ~0.24 gap with per-subject SD around 0.05-0.07 gives a very large Welch t at N=25. The metric is also mirror-symmetric (each base appears with its A/B mirror), so side bias cannot inflate or deflate it.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment); fallback handles other lengths.
    V_DESIGN = np.array([0.955, 0.545, 0.950, 0.540, 0.945, 0.535], dtype=float)
    LAM = 1.6
    DELTA = 0.9

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if V_DESIGN.size >= n:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.545)])[:n]

        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue

        # salience of each discriminating row: recency x (1 - validity)
        j = disc.astype(float)
        logu = LAM * j + DELTA * np.log(np.clip(1.0 - v[disc], 1e-9, None))
        logu = logu - np.max(logu)
        u = np.exp(logu)
        top = disc[int(np.argmax(u))]
        top_owner_a = bool(d[top] > 0.0)  # True -> the salient row favours A

        n_a = int(np.sum(d[disc] > 0.0))
        n_b = m - n_a

        # include only trials where the row-count tally does NOT strictly
        # agree with the most salient (late / low-validity) discriminating row
        if n_a > n_b:
            majority_a = True
            tie = False
        elif n_b > n_a:
            majority_a = False
            tie = False
        else:
            majority_a = None
            tie = True
        if (not tie) and (majority_a == top_owner_a):
            continue

        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == top_owner_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_5 (simulated): 0.4900 (var=0.0045)
- Predicted under pi_4 (simulated): 0.7326 (var=0.0025)
- Observed on real data: 0.7697 (var=0.0057)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """One-reason adherence on model-disagreement trials.

    For every trial we compute (a) the option favoured by the FIRST
    discriminating cue in validity order (features are listed in
    descending validity: 0.95, 0.85, 0.75, 0.65, 0.55) and (b) the option
    favoured by the unit-weight tally of feature-wise wins.  We keep only
    trials where these two disagree in direction OR where the tally is tied
    (so a counting rule must guess).  The metric is the proportion of such
    trials on which the subject picked the option favoured by the first
    discriminating cue.
    """
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]

        # first discriminating cue (features already in descending validity)
        first_winner = None
        for j in range(n):
            if a[j] > b[j]:
                first_winner = 0
                break
            if b[j] > a[j]:
                first_winner = 1
                break
        if first_winner is None:
            continue  # no cue discriminates: uninformative trial

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None  # counting rule must guess

        if tally_winner is not None and tally_winner == first_winner:
            continue  # agreement control trial: not diagnostic

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hits.append(1.0 if r == first_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2820 (var=0.0035)
**Predicted under pi_5:** 0.2722 (var=0.0027)
**Predicted under pi_4:** 0.2843 (var=0.0034)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.6, 0.93, 0.55, 0.87, 0.7, 0.78], dtype=float)

    def analyze(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
        val = validities[:n] if validities.shape[0] >= n else np.arange(n, 0, -1, dtype=float)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_winner = 0
        elif margin < 0:
            tally_winner = 1
        else:
            tally_winner = -1
        order = np.argsort(-val, kind='stable')
        ttb_winner = -1
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        return margin, tally_winner, ttb_winner

    info = [analyze(r) for _, r in data.iterrows()]
    margins = np.array([x[0] for x in info], dtype=float)
    tally_w = np.array([x[1] for x in info], dtype=int)
    ttb_w = np.array([x[2] for x in info], dtype=int)
    resp = data['response'].to_numpy().astype(int)

    # Group 1: tally-tie trials where TTB is decisive (parameter-free for Tallying)
    tie_mask = (tally_w == -1) & (ttb_w >= 0)
    # Group 2: trials where tally margin is >= 2 AND TTB points the other way
    dis_mask = (np.abs(margins) >= 2) & (tally_w >= 0) & (ttb_w >= 0) & (tally_w != ttb_w)

    if tie_mask.sum() == 0 and dis_mask.sum() == 0:
        return float('nan')

    if dis_mask.sum() > 0:
        p_tally = float(np.mean(resp[dis_mask] == tally_w[dis_mask]))
    else:
        p_tally = 0.5
    if tie_mask.sum() > 0:
        p_ttb_tie = float(np.mean(resp[tie_mask] == ttb_w[tie_mask]))
    else:
        p_ttb_tie = 0.5

    return p_tally - p_ttb_tie

```

**Observed (real) value:** 0.1744 (var=0.0127)
**Predicted under pi_5:** 0.1956 (var=0.0184)
**Predicted under pi_4:** 0.1856 (var=0.0146)

### Experiment 5
**Design**
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.68, 0.96, 0.52, 0.88, 0.58, 0.78], dtype=float)

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        try:
            a = np.asarray([float(x) for x in row['option_a_ratings']], dtype=float)
            b = np.asarray([float(x) for x in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        k = int(min(a.shape[0], b.shape[0]))
        if k == 0:
            continue
        a = a[:k]
        b = b[:k]
        if V.shape[0] >= k:
            v = V[:k]
        else:
            v = np.concatenate([V, np.full(k - V.shape[0], 0.55)])[:k]

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # keep only unit-tally TIES with at least one discriminating cue
        if a_wins != b_wins or a_wins == 0:
            continue

        w = v ** 2.0
        d = float(np.sum(w * (a - b)))
        # discard tie trials whose validity-weighted margin is itself
        # near zero (no clear "higher-validity side" to predict)
        if abs(d) < 0.15:
            continue

        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        if resp not in (0, 1):
            continue

        chose_a = (resp == 0)
        picked_high_validity = chose_a if d > 0 else (not chose_a)
        hits += 1.0 if picked_high_validity else 0.0
        n += 1.0

    if n == 0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.2856 (var=0.0160)
**Predicted under pi_5:** 0.2767 (var=0.0068)
**Predicted under pi_4:** 0.3056 (var=0.0052)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    v = np.array([0.58, 0.97, 0.53, 0.80, 0.66, 0.90], dtype=float)
    w = v ** 2.0  # AVAI-style amplified weights (gamma=2)

    num = 0.0
    den = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0], w.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        ww = w[:n]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # not a tally tie
        d = float(np.sum(ww * (a - b)))
        if abs(d) < 0.30:
            continue  # tie trials whose weighted margin is negligible
        den += 1.0
        if d > 0:
            num += 1.0 if int(resp) == 0 else 0.0
        else:
            num += 1.0 if int(resp) == 1 else 0.0

    if den == 0:
        return 0.5
    return num / den
```

**Observed (real) value:** 0.1500 (var=0.0194)
**Predicted under pi_5:** 0.1650 (var=0.0115)
**Predicted under pi_4:** 0.1583 (var=0.0128)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    correct = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        nz = d[d != 0.0]
        if nz.size == 0:
            continue
        # dominance trial: every discriminating cue favours the same option
        if not (np.all(nz > 0) or np.all(nz < 0)):
            continue
        dominant_is_a = bool(np.all(nz > 0))
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        correct.append(1.0 if (chose_a == dominant_is_a) else 0.0)

    if len(correct) == 0:
        return 0.5
    return float(np.mean(correct))
```

**Observed (real) value:** 0.7427 (var=0.0173)
**Predicted under pi_5:** 0.7427 (var=0.0074)
**Predicted under pi_4:** 0.7440 (var=0.0064)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # design validities (fixed by the experimental design)
    v_design = np.array([0.95, 0.70, 0.55, 0.95, 0.70, 0.55], dtype=float)
    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k == 0:
            continue
        if v_design.size >= k:
            vv = v_design[:k]
        else:
            vv = np.linspace(0.95, 0.55, k)
        d = a[:k] - b[:k]
        idx = np.nonzero(d != 0.0)[0]
        if idx.size != 2:
            continue
        j1 = int(idx[0])
        j2 = int(idx[1])
        # the two discriminating cues must point to OPPOSITE options
        if d[j1] * d[j2] > 0:
            continue
        # and the LATER cue must be the LESS valid one
        if not (vv[j2] < vv[j1]):
            continue
        late_choice = 0 if d[j2] > 0 else 1  # 0 = A, 1 = B
        try:
            resp = int(r)
        except Exception:
            continue
        n += 1
        if resp == late_choice:
            hits += 1
    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5067 (var=0.0112)
**Predicted under pi_5:** 0.5083 (var=0.0072)
**Predicted under pi_4:** 0.8367 (var=0.0052)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Pooling all eight experiments, the corpus splits cleanly into two regimes.\n\nREGIME A - designs whose stated validities are all numerically distinct (Exps 2, 3, 5, 6, and the dominance-only Exp 7). Here choices are strongly ANTI-validity and/or recency-driven: Exp 3 follow-highest-validity-first-cue = 0.28, Exp 5 choose-high-validity-side on tally ties = 0.29, Exp 6 = 0.15 (the .97-vs-.53 adjacent contrast is decided ~85% by the later, near-chance expert), and Exp 2 follow-the-late-low-validity-salient-row = 0.77. Both theories share the salience form exp(lam*j)*(1-v)^delta and both fit Exps 3-7 essentially perfectly (pi_5: .272/.196/.277/.165/.743; pi_4: .284/.186/.306/.158/.744; observed .282/.174/.286/.150/.743). Exp 2 was designed to break the individuation gate and it did: pi_4 predicted 0.733 vs observed 0.770, pi_5 predicted 0.490 - a 0.28 miss. So a 0.005 numeric gap between printed labels does NOT abolish the anti-validity/recency competition; OCO-LI's tau range (0.02-0.045) is simply wrong.\n\nREGIME B - designs in which stated validities are EXACTLY duplicated (Exp 8: [.95,.70,.55,.95,.70,.55]; Exp 1: [.95,.55,.75,.95,.75,.55]). Here the same contrasts collapse to chance/tally: Exp 8's diagnostic pairs (col0 or col3 at .95 against col5 at .55 - the exact structure that produced 0.15 in Exp 6) came out at 0.5067, and Exp 1's follow-the-last-discriminating-row rate was 0.438. pi_4 predicted 0.837 and 0.764 on these two - catastrophic, and unfixable by any parameter setting: to push a 5-column, .95-vs-.55 contrast to chance RDS would need a near-total lapse, which would destroy Exps 5, 6 and 7. pi_5 predicted 0.508 and 0.471 - essentially exact on Exp 8 and close on Exp 1.\n\nSo: pi_5 (OCO-LI) is the only account that spans both regimes; its single failure (Exp 2) is a pure calibration error in the gate's scale parameter (with tau ~0.002-0.004, phi=1 in Exp 2 and phi=0 in Exps 1 and 8, and pi_5 would then reproduce all eight numbers). The design-level modulator is therefore EXACT label repetition (non-injective label code), not numeric proximity. pi_4 has no such modulator by construction and is falsified twice over; it is the degenerate member of the pair and cannot be repaired by regenerating its predict function while keeping its description.\n\nOne further diagnostic detail worth exploiting: in Exp 1 the observed 0.438 lies BELOW pi_5's probability-matched 0.471 and above what a hard majority-tally would give (~0.38 on this metric), and in Exp 2 the observed 0.770 lies slightly ABOVE the one-reason prediction 0.733 (compensatory summation over the 4-vs-1 and 3-vs-1 coalitions would have pulled it down). That is mild evidence that the ungated mode is closer to a graded tally than to pure probability matching, and that the gated mode is close to winner-take-all - a seam that a well-designed rival theory can exploit.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_4 (RDS) entirely. Its core commitment - salience competition with no design-level modulator - is refuted by Exps 1 and 8 and cannot be rescued by re-parameterisation.\n\nPropose instead a DUAL-STRATEGY, COMPENSATORY rival that matches OCO-LI everywhere it currently succeeds but differs from it mechanistically in two testable ways:\n\n**Sketch: 'Strategy Selection by Label-Code Injectivity, with Distinctive-Cue SUMMATION' (SSI-DS).**\n1. *Strategy switch (discrete, not graded).* Before the block the subject inspects the printed validity list. If the list is an injective code (all labels distinct as printed symbols - even 0.955 vs 0.950 counts as distinct), the subject infers the experts are individually identifiable and adopts the weighted mode. If ANY label is repeated, the subject infers the experts are interchangeable and adopts an unweighted MAJORITY TALLY mode. The switch is all-or-none in the design (optionally with a small per-trial mixture probability), NOT a graded function of numeric gap size - this is the sharp contrast with OCO-LI's phi = mean_j min(1, d_j/tau).\n2. *Weighted mode = SUMMATION, not one-reason anchoring.* Evidence E = sum over discriminating rows of sign(a_j - b_j) * (1 - v_j)^gamma * exp(lam * j); choice is softmax(beta * E). Validity enters with inverted sign (near-certain experts are treated as redundant echoes; near-chance disagreements are the news that must be adjudicated) and position enters as a mild recency gain. Crucially, coalitions ADD: three low-validity early rows can outweigh one lower-validity late row, whereas OCO-LI's winner-take-all anchor cannot be outvoted.\n3. *Tally mode = graded majority, not probability matching.* p(A) = softmax(kappa * (n_A - n_B)) over the counts of discriminating rows, so a 2-vs-1 split gives ~0.75-0.85 rather than OCO-LI's 0.67, and a 1-vs-1 split gives exactly 0.50.\n4. *Conflict interference / lapse.* Retain the shared g = 1 - (1-eps0)*rho^(m-2) mixture toward 0.5 so that dominance accuracy is ~0.74 and degrades with the number of discriminating rows (Exp 7).\n\nCalibration targets it must hit simultaneously: Exp 8 = 0.51 (tally mode, 1-1 ties -> exactly 0.5); Exp 1 = 0.44 (tally mode; note the observed value sits between hard-majority ~0.38 and probability-matching 0.47, so kappa should be moderate, giving ~0.80 on the 2-vs-1 and 3-vs-1 bases); Exp 2 = 0.77 (weighted mode, phi effectively 1 despite 0.005 gaps); Exp 3 = 0.28; Exp 4 = 0.17; Exp 5 = 0.29; Exp 6 = 0.15; Exp 7 = 0.74. Suggested ranges: gamma in [0.6, 1.4], lam in [0.3, 1.2] (summation needs a smaller recency gain than winner-take-all), beta in [1.5, 6], kappa in [0.8, 2.5], eps0 in [0.15, 0.35], rho in [0.7, 0.9].\n\nThis rival is decidable against OCO-LI by two future designs: (i) all-distinct labels with a 3-low-validity coalition opposing a single later, slightly-lower-validity row (summation predicts the coalition wins, OCO-LI predicts the single late row wins); (ii) a label vector with exactly one duplicated pair among six otherwise well-separated labels (OCO-LI's averaged gate predicts a partial, ~2/3-strength anti-validity effect; SSI-DS predicts a full switch to tallying)."
}
```

## Usage

```json
{
  "input_tokens": 25868,
  "output_tokens": 23487
}
```
