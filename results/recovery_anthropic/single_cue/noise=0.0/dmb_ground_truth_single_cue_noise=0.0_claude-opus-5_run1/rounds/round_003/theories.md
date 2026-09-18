# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

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

**Rationale:** **Why the previous theory (pi_3, AVAI) must go.** Any additive validity-weighted integrator predicts that on unit-tally ties people pick the side holding the higher-validity cues. The data say the opposite in three independent designs (Exp3 .286, Exp4 .150, Exp5-style conflicts), and AVAI scored .94/.99 where humans scored .29/.15. It also puts dominance accuracy at .98 where humans are at .74. Both failures are structural, not parametric. I have therefore replaced it with a one-reason ANCHORING family, as the arbiter directed.

**What I take from the arbiter, and the one place I deviate (with justification).** I adopt (1) single-row anchoring, (2) salience driven by LOW stated validity (odd-cue-out), (4) conflict interference in the number of discriminating rows m, and — the genuinely new mechanism and the theory's empirical bite — (3) LABEL-AMBIGUITY GATING: the anchoring competition only runs to the extent that each expert can be bound to a column via a distinct stated validity; otherwise every discriminating row is equally likely to anchor and a 2-row conflict is a coin flip. I *retain* a reading-order term rather than omitting it. Justification for the deviation: the arbiter asserts position 'breaks Exp2', but the arithmetic says the reverse. Exp2's diagnostic set contains a trial where a single LATE cue (col 5, v=.78) opposes an early coalition (cols 0,2,4). Without a position term the model chases that coalition (p_tally ≈ .99) and Exp2's metric lands at ~.24-.27 versus the observed .174; with reading-order recency the late lone cue wins about 21% of the time and Exp2 comes out at .178. Exp6, the one place recency genuinely misfires (RDS predicted .84 versus an observed .507), is fixed *by the gate*, not by deleting recency: Exp6's validity vector [.95,.70,.55,.95,.70,.55] has every label duplicated, so phi = 0, both anti-validity and recency switch off, and the two-cue conflict returns exactly .50. In Exps 1-4 the minimum pairwise validity gap is .05-.10, so phi = 1 and nothing changes. Exp5's metric uses only dominance trials, where every discriminating row favours the same option, so the gate is provably inert there — the theory is robust to whatever validity vector that design used.

**Hand-checked central predictions** (lam=1.65, delta=0.90, eps0=0.28, rho=0.7925, computing every diagnostic trial of every design by hand): Exp1 .274 (obs .282), Exp2 .178 (obs .174), Exp3 .285 (obs .286), Exp4 .151 (obs .150), Exp5 .743 (obs .743), Exp6 .500 (obs .507). The parameter ranges are centred tightly on this point so that subject-level sampling keeps the pooled means at these values while adding only a small amount of between-subject variance on top of the binomial floor, matching the observed per-subject variances (~.003-.02). The three degrees of freedom are pinned by nearly orthogonal constraints — eps0 by the m=2 conflicts of Exp4, rho by the m=2/4/6 dominance ladder of Exp5, and lam/delta jointly by Exp1's mixed-m diagnostic set and Exp2's coalition-versus-late-cue contrast — so the fit is not free curve-fitting: the gate then predicts Exp6 with no further tuning.

**Why this is a better theory, not just a better fit.** It explains, with one mechanism, four otherwise disconnected regularities: sub-chance validity following, chance-level performance whenever validity labels are non-individuating, the monotone decline of dominance accuracy with the number of speaking experts, and the coalition-versus-single-late-cue asymmetry. And it makes two crisp, cheap, falsifiable predictions: de-duplicating a validity vector while holding cue content fixed should *switch on* the anti-validity effect, and placing the low-validity discriminating row early against a late high-validity row should produce a near-even split, dissociating it from both pure-recency and Bayesian accounts.

**Parameters:**
  - `lam`: `[1.50, 1.80]`
  - `delta`: `[0.82, 0.98]`
  - `eps_base`: `[0.25, 0.31]`
  - `rho`: `[0.765, 0.82]`
  - `tau`: `[0.02, 0.045]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


### slot 2 — `pi_4` — KILLED ✗

**Description:** **Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS).**

Three claims, all about *which single cue gets to decide*, not about integration.

1. **One cue decides, but which one is stochastic.** On every trial the subject settles the comparison on a SINGLE discriminating expert row (a one-reason rule). Which row wins the competition for the decision varies from trial to trial, so choice is graded *within* a subject (this is required by the data: per-subject metric variances are essentially binomial, i.e. subjects are homogeneous but individually stochastic, ruling out fixed per-subject cue orders). The deciding row j is drawn from the discriminating set D with probability proportional to a salience weight u_j.

2. **Salience = serial recency × distinctiveness (redundancy-discounted validity).** u_j = exp(lambda*j) * (1 - v_j)^delta.
   * *Recency (lambda > 0):* rows are read top-to-bottom / left-to-right and the later-read rows are the ones still active in working memory when the choice is made, so they capture the decision. Column position, not stated diagnosticity, is the dominant driver (this is the arbiter's core claim, retained).
   * *Distinctiveness (delta > 0):* subjects treat a near-certain expert (v = .97) as *redundant* — "I already know what he will say, he just echoes the consensus" — whereas a row on which a near-chance expert (v = .53) discriminates feels like idiosyncratic, decision-relevant information that must be adjudicated. Communicated validity is therefore used, but with inverted sign: high-validity rows are discounted as uninformative repeats of the general impression. This is why choices are systematically *sub-chance* with respect to validity in every experiment.

3. **Conflict interference.** The more rows on which the two products differ, the more competing comparisons must be held simultaneously, and the more often the subject loses the thread and simply guesses: P(guess) = 1 - (1-eps0)*rho^(m-2), where m = |D|. Two-row comparisons are made almost cleanly (hence near-deterministic anti-validity choice when only a .97 row and a .53 row differ); five- and six-row comparisons collapse toward coin-flipping. Difficulty grows with the number of differences, not with their diagnosticity.

No learning, no feedback use: the rule is stationary across the block.

*Why the two extra mechanisms are needed on top of pure position-recency (justified deviation from the arbiter's exact form):* pure position models (graded recency Luce, or its Take-The-Last limit) plus a uniform lapse are mathematically unable to reproduce the observed ordering Exp4 (0.15) < Exp1 (0.28) ≈ Exp3 (0.29): Exp4's diagnostic trial is an ADJACENT single-pair contrast (column 1 vs column 2), whose raw position-based prediction (1/(1+e^lambda)) is always *higher* (closer to chance) than the multi-cue Exp1/Exp3 trials, so with a shared lapse floor Exp4 can never come out lowest. Distinctiveness fixes this (a .97-vs-.53 contrast is the most lopsided contrast in the whole corpus). Likewise pure position forces Exp2's tie-trial term to exactly 0.5 and its 1-vs-3/1-vs-5 coalition term to ~2/3, capping the Exp2 metric at 0.167 only in the noiseless limit; distinctiveness lowers the tie term (the .93 row loses to the .60 row that precedes it) so that the observed 0.174 is reachable with realistic noise. Conflict interference then supplies the trial-type-dependent floor that lets Exp1/Exp3 sit at ~0.28 while Exp4 sits at ~0.15.

**Rationale:** I kept the arbiter's core (position/recency-dominant, single-reason, stationary, validity largely mis-used) but repaired two provable failures of the pure position family, and I verified the repaired model analytically against all four metrics before proposing it.

Why pure position-recency cannot work (analytic check): (i) Exp4's diagnostic trial is the single adjacent-column contrast col1(.97) vs col2(.53); its raw position prediction is 1/(1+e^lambda), which is always CLOSER to chance than the multi-cue Exp1/Exp3 trials, so with a shared lapse floor a position-only model must predict Exp4 >= Exp1, Exp3 — yet the data have Exp4 = 0.15 < Exp1 = 0.282 ≈ Exp3 = 0.286. (ii) In Exp2 the two tie trials are exactly antisymmetric under recency (col1-beats-col0 hit, cols4/5-beat-cols2/3 miss), pinning p_ttb_tie at 0.5 and capping the metric at 0.167 only in the noiseless Take-The-Last limit, where Exp1/Exp3/Exp4 all collapse to 0; adding lapse to reach 0.28 there drives Exp2 down to ~0.09. Best achievable pure noisy-TTL fit: (0.225, 0.092, 0.225, 0.225), max error 0.082.

The two added mechanisms each do distinct, checkable work. Distinctiveness (weights (1-v)^delta, i.e. high-validity rows discounted as redundant with the consensus) makes the .97-vs-.53 Exp4 contrast the single most lopsided contrast in the corpus (pushing it to ~0.15, below Exp1/Exp3), and it lowers Exp2's tie term (the .93 row loses ground to the .60 row that precedes it), which is exactly what is needed to lift the Exp2 difference metric to ~0.17 without demanding implausible noise. Conflict interference (guess rate 1-(1-eps0)rho^(m-2)) supplies a trial-type-dependent floor: 2-row comparisons are made cleanly (Exp4 stays at 0.15) while 3-6-row comparisons drift toward chance (Exp1 and Exp3 rise to ~0.28), which is also the only way to get sub-chance-but-not-zero adherence in Exp1 without a huge global lapse.

Hand-computed predictions at the range centre (lam=1.6, delta=0.9, eps0=0.28, rho=0.78): Exp1 0.280 (real 0.282), Exp2 0.175 (0.174), Exp3 0.292 (0.286), Exp4 0.152 (0.150). Checked at both range corners the predictions stay within ~0.02 of targets ((0.297, 0.161, 0.312, 0.140) and (0.267, 0.191, 0.274, 0.167)), so the pooled means are robust to per-subject parameter sampling. The model is also stochastic within subject with near-binomial spread, matching the observed low between-subject variances (notably Exp4 var 0.019 ≈ p(1-p)/6), which fixed-cue-order or subject-mixture accounts cannot do. This is strictly more experiment-invariant than pi_2 (fails Exp3/Exp4 by 0.22/0.34) and pi_3 (fails Exp3/Exp4 by 0.65/0.84), and better than the best pure-recency variant.

**Parameters:**
  - `lam`: `[1.4, 1.8]`
  - `delta`: `[0.8, 1.0]`
  - `eps_base`: `[0.24, 0.32]`
  - `rho`: `[0.73, 0.83]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** SSI-DS+ (Strategy Selection by Label-Code Injectivity, with Distinctive-Cue Summation and Competition-Scaled Interference).

(1) STRATEGY SWITCH. Before the block the subject reads the printed validity list as a naming code. If all labels are distinct printed symbols (0.955 vs 0.950 counts as distinct) the experts are individuable and the subject adopts a WEIGHTED mode; if any label is repeated the experts are interchangeable and the subject adopts an unweighted MAJORITY TALLY mode. The switch is all-or-none at the design level (only a small per-trial slip probability omega), not a graded function of numeric gap size.

(2) WEIGHTED MODE = COMPENSATORY SUMMATION over all discriminating rows: E = sum_j sign(a_j-b_j) * (1-v_j)^gamma * exp(lam*j), rescaled so the strongest discriminating row has magnitude 1, then p(A) = sigmoid(beta*E). Validity enters with INVERTED sign (near-certain experts are redundant echoes of the general impression; near-chance disagreements are the news that must be adjudicated) and position enters as a reading-order recency gain. Coalitions ADD, so several early distinctive rows can outvote a single later row.

(3) TALLY MODE = graded majority p(A) = sigmoid(kappa*(n_A - n_B)): 1-1 splits give exactly 0.50, 2-1 splits ~0.77 (harder than probability matching, softer than a hard majority rule).

(4) CONFLICT INTERFERENCE IS COMPETITION-SCALED (the refinement). Guessing is driven by the load of holding several discrepancies in mind while ADJUDICATING BETWEEN THEM. The load is therefore not the raw number of discriminating rows but the number of rows that actually have to be pitted against one another: when the display is unanimous (min(n_A,n_B) = 0, i.e. every discrepancy points the same way) there is nothing to adjudicate and the extra rows impose only a discounted reading load eta*(m-2) rather than the full (m-2). Formally g = 1 - (1-eps0) * rho^L with L = (m-2) on opposed displays and L = eta*(m-2) on unanimous displays, and the core probability is mixed toward 0.5 by g. This predicts that dominance accuracy degrades with the number of agreeing experts only mildly, whereas conflict displays degrade steeply with the number of opposed experts - the two are dissociated, not a single monotone function of display size.

No feedback, hence no learning; the rule is stationary across the block.

**Rationale:** MINIMAL DIFF on the accepted iter-1 base: I added exactly one parameter (eta) and three lines in the lapse block; gamma, lam, beta, kappa, eps0, rho, omega ranges and every other line are re-emitted verbatim, per the critic's instruction not to retune lam/gamma and not to widen beta/eps0 again.

What changed and why. The critic's key diagnosis is correct: Exp5 (unanimous displays) needs LESS guessing while the sub-chance conflict metrics need MORE, and a single g = 1-(1-eps0)*rho^(m-2) cannot deliver both. I implement the prescribed decoupling, but with the conflict/dominance asymmetry placed on the DOMINANCE side rather than on the conflict side. Load L = (m-2) on opposed displays and L = eta*(m-2) on unanimous displays: adjudication load arises only from rows that must be pitted against one another, whereas agreeing extra rows impose mere reading effort. This is the same 'conflict interference' clause, just stated in terms of competing rather than total rows.

Why not the critic's literal theta^min(n_A,n_B) with rho pushed to 0.85-0.95: I worked the algebra through the actual designs and that form regresses Exp8. The Exp8 metric is computed on OPPOSED trials only (7 trial types, three of them m=2, c=1, with core probabilities 0.80-0.99), so any extra conflict-driven guessing pulls Exp8 toward 0.5 - and Exp8 is already 0.019 BELOW the data (0.751 vs 0.770), i.e. it needs less guessing, not more. With theta=0.84, rho=0.82, eps0=0.23 my hand calculation gives Exp8 ~0.718 (error -0.052) in exchange for a +0.03 gain on Exp1: a net loss the gate would reject, exactly as in iteration 2. Exp1 and Exp8 have essentially the same joint (m, min(n_A,n_B)) distribution over their diagnostic trials, so no count-based conflict term can separate them; chasing Exp1 through the lapse is a dead end.

The eta edit is instead strictly monotone in the right direction and provably touches only Exp5. Every other metric filters to opposed displays: Exp1 keeps only trials where the first cue disagrees with the tally (never unanimous), Exp2 keeps tally ties and TTB/tally disagreements, Exp3/Exp4 keep tally ties, Exp6 keeps exactly two opposed cues, Exp7 explicitly skips unanimous displays, Exp8 drops trials where the majority agrees with the salient row (which includes all dominance trials). So min(n_A,n_B) >= 1 there and g is bit-for-bit unchanged: Exp1 0.253, Exp2 0.170, Exp3 0.283, Exp4 0.137, Exp6 0.508 and Exp7 0.439 (the two currently exact values the critic warned are most sensitive) and Exp8 0.751 are all preserved.

Calibration of eta. Exp5's dominance trials are m=2, 4, 6 in equal proportion. Inverting the base output (0.7193 at mean g = 0.501) gives an effective p_core ~0.94 and dAcc/dg ~ -0.44, so reaching the observed 0.743 requires mean g to fall by ~0.055, i.e. exp(-0.5034*eta) + exp(-1.0068*eta) = 1.186 at rho = 0.7775, giving eta ~ 0.71-0.73. The declared range [0.62, 0.84] brackets that point symmetrically (predicted Exp5 ~0.732-0.753, mean ~0.742) and, as a bonus, is the only parameter that varies dominance accuracy across subjects, which addresses the too-low Exp5 between-subject variance without touching beta or eps0 as the gate demanded. Expected result: seven metrics unchanged and Exp5's 0.023 error reduced to ~0.003, lowering the RMS point error from ~0.0155 to ~0.0133 and so clearing the 0.0232 floor.

**Parameters:**
  - `gamma`: `[0.86, 0.94]`
  - `lam`: `[1.45, 1.55]`
  - `beta`: `[3.2, 4.0]`
  - `kappa`: `[1.10, 1.30]`
  - `eps0`: `[0.22, 0.26]`
  - `rho`: `[0.755, 0.80]`
  - `omega`: `[0.0, 0.04]`
  - `eta`: `[0.62, 0.84]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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
            flat = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = flat[0].ravel(), flat[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed label code) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    beta = float(parameters['beta'])
    kappa = float(parameters['kappa'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    omega = float(np.clip(float(parameters['omega']), 0.0, 0.5))
    eta = float(np.clip(float(parameters['eta']), 0.0, 1.0))

    # ---------- STRATEGY SWITCH: injectivity of the printed label code ----------
    LABEL_TOL = 2.5e-3   # two printed labels closer than this are the same symbol
    if n <= 1:
        injective = True
    else:
        vs = np.sort(v)
        gaps = np.diff(vs)
        injective = bool(np.all(gaps > LABEL_TOL))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])

    # ---------- weighted mode: distinctive-cue SUMMATION ----------
    # log weight = gamma*log(1 - v_j) + lam*j ; rescaled so strongest row = 1
    logw = gamma * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + lam * disc.astype(float)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    E = float(np.sum(signs * w))
    z = float(np.clip(beta * E, -60.0, 60.0))
    p_weighted_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- tally mode: graded majority ----------
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    zt = float(np.clip(kappa * (n_a - n_b), -60.0, 60.0))
    p_tally_a = 1.0 / (1.0 + np.exp(-zt))

    if injective:
        p_core_a = (1.0 - omega) * p_weighted_a + omega * p_tally_a
    else:
        p_core_a = (1.0 - omega) * p_tally_a + omega * p_weighted_a

    # ---------- conflict interference / lapse (competition-scaled load) ----------
    # load counts only rows that must be ADJUDICATED against one another:
    # unanimous (dominance) displays impose only a discounted reading load.
    conflict = float(min(n_a, n_b))
    expo = float(max(m - 2, 0))
    if conflict <= 0.0:
        expo = eta * expo
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p
```

**`policy(probs)`:**
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
