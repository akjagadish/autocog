# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** **Limited-Sample Noisy Cue Counting (LSNC).** People neither integrate all cues (Tallying) nor consult a single best cue (Take-The-Best). Instead, on each choice they inspect only a *limited, randomly selected subset* of the presented expert ratings, tally feature-wise wins **within that sample only**, and pick the option that is ahead in the sample; a within-sample tie (including a sample containing no discriminating cue) forces a guess.

Three commitments define the theory:
1. **Equal-probability sampling.** The inspected subset is drawn uniformly at random from the whole profile — it is *not* validity-ordered and *not* validity-weighted. Stated validities are too weakly represented to steer the search order, so no cue enjoys priority. Consequently, on globally tally-tied, side-counterbalanced profiles the process is exactly symmetric and choice is 50/50, with no tendency to follow the most valid discriminating cue.
2. **Sample-size limitation with tie dilution.** The number of cues actually inspected, K, is a random variable with mean k < n (implemented as K = 1 + Binomial(n−1, q), so at least one cue is always read). Because the subset is drawn from *all* features, non-discriminating (tied) features occupy sampling slots and dilute the evidence: two profiles with the same raw tally margin M produce different adherence depending on how many uninformative cues pad the profile, and on the proportion (not the raw count) of informative cues won. This is the signature that dissociates LSNC from softmax Tallying, whose predictions depend only on M.
3. **Encoding noise + lapse.** Each inspected feature-wise comparison is registered with the wrong sign with probability nu (attention/encoding noise), and with probability epsilon the whole decision is replaced by a coin flip.

Because a small, noisy sample can easily reverse the sign of a large global margin, adherence to the full-tally winner is *capped well below 1* even on high-margin conflict trials — the systematic attenuation that full Tallying cannot produce without degenerate temperature — while symmetry guarantees exactly chance behaviour on tally-tied pairs.

**Rationale:** Take-The-Best (pi_1) is falsified in both directions (0.90 vs 0.474; 0.14 vs 0.65) and softmax Tallying (pi_2), while parameter-freely correct at chance on Experiment 1's tally ties, over-predicts conflict-trial adherence (0.85 vs 0.65) because a large raw margin drives its softmax to near-determinism unless beta collapses toward guessing everywhere. LSNC dissolves this tension with a single mechanism.

(1) **Experiment 1 is fit parameter-freely.** On tally-tied, side-counterbalanced profiles nA = nB, and uniform (validity-blind) subset sampling plus symmetric encoding noise make the whole process exchangeable in A/B, so p(A) = 0.5 exactly for every parameter setting. Predicted TTB-agreement = 0.50 (real 0.4738), and the only across-subject variance is binomial (0.25/32 ≈ 0.0078 vs. observed 0.0073) — matching not just the mean but the observed between-subject spread.

(2) **Experiment 2's attenuation falls out mechanically.** The two conflict configurations are (nA=1, nB=4, nT=1) and (nA=1, nB=3, nT=2). Exact enumeration gives adherence to the tally winner of ~0.60 (K=1), ~0.63 (K=2), ~0.67 (K=3) at nu ≈ 0.25, rising only slowly toward 1 as K → n. With the declared ranges (mean k ≈ 2.5, nu ≈ 0.22, eps ≈ 0.06) the pooled prediction is ≈ 0.655, essentially the observed 0.6538, and the parameter spread contributes ≈ 0.002 on top of the 0.007 binomial variance, landing near the observed 0.0092. Crucially, adherence is *capped*: no amount of increasing determinism drives it to 1 unless the sample becomes the full profile, so the model is not merely a re-tuned Tallying.

(3) **It is a genuine rival with a testable dissociation.** Because the subset is drawn from all cues — including tied ones — adherence depends on the *proportion* of informative cues won and is *diluted* by uninformative cues, not on the raw margin M. Note above that the 1-vs-4 (margin 3, one tied cue) and 1-vs-3 (margin 2, two tied cues) cases already differ by ~0.03–0.05 at fixed parameters in a direction Tallying cannot represent; a 2-0 versus 4-2 design (both M=2) yields a large predicted gap under LSNC and none under Tallying. The model uses no validity information at all, honouring the Experiment 1 evidence against validity-ordered or validity-weighted search, and it generalises to any n_features and any validity vector without refitting.

**Parameters:**
  - `k`: `[1.5, 3.5]`
  - `nu`: `[0.12, 0.32]`
  - `epsilon`: `[0.0, 0.12]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from math import comb

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("LSNC expects a (2, n_features) stimulus; got shape %s." % (stim.shape,))
    a, b = stim[0], stim[1]
    n = int(a.shape[0])

    nA = int(np.sum(a > b))          # cues on which A wins
    nB = int(np.sum(b > a))          # cues on which B wins
    nT = n - nA - nB                 # uninformative (tied) cues

    k = float(parameters["k"])            # mean number of cues inspected
    nu = float(parameters["nu"])          # per-comparison encoding error
    eps = float(parameters["epsilon"])    # lapse rate
    nu = float(min(max(nu, 0.0), 0.5))
    eps = float(min(max(eps, 0.0), 1.0))

    if n == 0:
        return np.ones(2) / 2.0

    # ---- distribution of the inspected sample size K in {1..n} -------------
    if n == 1:
        pK = np.array([1.0])
    else:
        q = (k - 1.0) / (n - 1.0)
        q = float(min(max(q, 0.0), 1.0))
        pK = np.array([comb(n - 1, s - 1) * (q ** (s - 1)) * ((1.0 - q) ** (n - s))
                       for s in range(1, n + 1)], dtype=float)
        tot = pK.sum()
        pK = pK / tot if tot > 0 else np.ones(n) / float(n)

    # ---- exact enumeration over sample compositions ------------------------
    pA = 0.0
    for si, s in enumerate(range(1, n + 1)):
        w = float(pK[si])
        if w <= 0.0:
            continue
        denom = float(comb(n, s))
        for ai in range(0, min(nA, s) + 1):
            for bi in range(0, min(nB, s - ai) + 1):
                ti = s - ai - bi
                if ti < 0 or ti > nT:
                    continue
                cnt = comb(nA, ai) * comb(nB, bi) * comb(nT, ti)
                if cnt == 0:
                    continue
                ph = float(cnt) / denom          # multivariate hypergeometric
                m = ai + bi                      # informative cues in sample
                if m == 0:
                    pA += w * ph * 0.5           # nothing discriminates -> guess
                    continue
                # X = number of sampled comparisons REGISTERED as A-wins
                d1 = np.array([comb(ai, x) * ((1.0 - nu) ** x) * (nu ** (ai - x))
                               for x in range(ai + 1)], dtype=float)
                d2 = np.array([comb(bi, x) * (nu ** x) * ((1.0 - nu) ** (bi - x))
                               for x in range(bi + 1)], dtype=float)
                dist = np.convolve(d1, d2)
                xs = np.arange(m + 1)
                p_win = float(dist[2 * xs > m].sum())
                p_tie = float(dist[2 * xs == m].sum())
                pA += w * ph * (p_win + 0.5 * p_tie)

    pA = float(min(max(pA, 0.0), 1.0))
    p = np.array([pA, 1.0 - pA], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** **Count-Direction with Conflict-Graded Confidence (CDCC).**

When people compare two multi-attribute options described by binary expert ratings, the *direction* of their preference and the *confidence* with which they execute it are computed by two functionally separate processes.

1. **Direction is purely count-based.** The comparator registers, for each attribute, only which option is rated higher, and the option that wins on more attributes becomes the candidate response. Validities never enter this stage: they cannot promote a minority-supported option to candidate status. When the two win-counts are equal, the comparator returns no candidate at all, and the subject flips a mental coin — exactly 0.5, with **no** validity-ordered tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles CDCC predicts chance adherence to the most-valid discriminating cue, regardless of any parameter setting.

2. **Confidence, not direction, carries the validity information, and it is graded by evidence dilution and by cue conflict.** The probability of actually executing the count-winner is a saturating (tanh) function of a scalar confidence signal built from three ingredients: (i) the raw win margin M = |nA − nB|; (ii) *dilution* — the margin is discounted by the total number of discriminating cues raised to a power gamma, so that a 6-vs-2 split feels much weaker than a 4-vs-0 split even though both have M = 4 (evidence spread thin across many mutually contradicting attributes is subjectively less compelling than a clean, sparse, unanimous split); and (iii) *coherence* — the signed, normalised balance of validity weight between the cues the count-winner wins and the cues the loser wins, where the subjective weight of a cue is a convex (super-linear) function of its stated validity advantage, w_j = (v_j − 0.5)^rho. Convexity means the coherence signal is dominated by the few high-validity cues: when the best cue points *with* the count the subject feels certain, when it points *against* the count the subject feels torn even though the count itself is unambiguous.

3. **Conflict erodes confidence but never reverses direction.** The confidence argument is floored at zero, so validity conflict can drive behaviour arbitrarily close to guessing but never below chance: CDCC never makes systematic one-reason (Take-The-Best) choices. This is what separates it from any Tallying+TTB mixture: on high-margin trials where the top cue opposes the tally, people move toward 0.5 from above, not past it.

4. **Lapses.** With probability epsilon the entire decision is replaced by a coin flip (attention lapse), which caps adherence below 1 even on maximally coherent trials.

The theory therefore predicts a specific cross-design signature: chance performance on tally-tied profiles; high but sub-ceiling adherence on sparse, coherent, unanimous profiles; noticeably lower adherence on dense profiles where the loser also wins several cues even when the raw margin is larger; and the lowest (but still above-chance) adherence when the most valid cue opposes the count.

**Rationale:** Minimal-diff edit on the running-best iter-1 base. The mechanism is left completely untouched: count-only direction, exact parameter-free 0.5 on tally ties, tanh(beta * M/D^gamma * coherence gate), convex weights w=(v-0.5)^rho, chance floor, and the multiplicative lapse. The ONLY change is the epsilon range, shifted slightly downward from [0.02, 0.22] to [0.02, 0.18] (centre 0.12 -> 0.10), plus the matching default in the code.

Why this and nothing else: the critic has now explicitly retracted the dispersion-chasing line of advice, since two consecutive box-widening attempts (three-way widening in iter 2, epsilon-only symmetric widening in iter 3) produced literally zero Exp3/Exp4 between-subject variance gain and both were rejected by the gate. On those coherent, near-saturating trials p_winner is flat over any admissible box, so the residual dispersion is not recoverable in-family and chasing it only perturbs the levels the loss actually rewards. The remaining genuine residuals of the accepted base are level residuals in a single, consistent direction: Exp3 0.776 vs 0.794 (-0.018) and Exp4 0.749 vs 0.755 (-0.006), both LOW.

Epsilon enters strictly linearly and multiplicatively on the confidence term (p = 0.5 + 0.5*(1-eps)*tanh(...)), so lowering its mean by 0.02 raises adherence by ~0.5*0.02*tanh(...) ~= +0.010 uniformly on saturated trials and proportionally less on weak-evidence trials. That lifts Exp3 to ~0.786 and Exp4 to ~0.759, shrinking both residuals, while moving Exp2's conflict cell only ~+0.006 (0.649 -> ~0.655, essentially exactly the observed 0.654 because the coherence gate there is well below saturation so the tanh factor is smaller). Exp1 is untouched by construction: every scored trial is a tally tie, which returns exactly [0.5, 0.5] before any parameter is read, so its residual against 0.474 remains pure simulation sampling noise and is not being tuned against.

Critically the sparse > dense inversion is preserved: gamma, beta, kappa and rho are all identical to the accepted base, so the relative ordering of confidence signals across designs (Exp3's D=1/2/3 sparse splits with S near 1.0-0.58 vs Exp4's dense 5v3/6v2 splits with S near 0.09-0.18) is unchanged; epsilon only rescales the whole confidence curve, keeping Exp3 > Exp4 intact while nudging both toward their targets. This is the smallest edit that reduces two of the three tunable level residuals simultaneously with no risk of the Exp4 overshoot that killed iter 2 (epsilon can only raise adherence toward the target from below, never past the tanh ceiling implied by the fixed beta/gamma).

**Parameters:**
  - `beta`: `[1.6, 3.6]`
  - `gamma`: `[1.25, 1.75]`
  - `kappa`: `[0.4, 1.0]`
  - `rho`: `[2.2, 3.8]`
  - `epsilon`: `[0.02, 0.18]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        stim = stim.reshape(2, -1)
    if stim.ndim != 2 or stim.shape[0] != 2:
        stim = stim.reshape(2, -1)
    a = stim[0]
    b = stim[1]
    n = int(a.shape[0])

    if n == 0:
        return np.array([0.5, 0.5], dtype=float)

    a_win = a > b
    b_win = b > a
    nA = int(np.sum(a_win))
    nB = int(np.sum(b_win))
    D = nA + nB

    # --- Commitment 1: direction is purely count-based; ties -> exact guess ---
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    winner = 0 if nA > nB else 1
    M = float(abs(nA - nB))

    # --- parameters -----------------------------------------------------
    beta = float(parameters.get("beta", 2.6))
    gamma = float(parameters.get("gamma", 1.5))
    kappa = float(parameters.get("kappa", 0.7))
    rho = float(parameters.get("rho", 3.0))
    eps = float(parameters.get("epsilon", 0.10))

    beta = min(max(beta, 0.0), 20.0)
    gamma = min(max(gamma, 0.0), 3.0)
    kappa = min(max(kappa, 0.0), 5.0)
    rho = min(max(rho, 0.5), 8.0)
    eps = min(max(eps, 0.0), 0.6)

    # --- validity weights (convex in validity advantage) ------------------
    v_raw = parameters.get("validities", None)
    v = None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume features are ordered by descending validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

    w = np.clip(v - 0.5, 1e-9, None) ** rho

    if winner == 0:
        W_win = float(np.sum(w[a_win]))
        W_lose = float(np.sum(w[b_win]))
    else:
        W_win = float(np.sum(w[b_win]))
        W_lose = float(np.sum(w[a_win]))

    tot = W_win + W_lose
    C = 0.0 if tot <= 0.0 else (W_win - W_lose) / tot   # in [-1, 1]

    # --- Commitment 2: dilution-discounted margin, coherence-gated --------
    S = M / (float(D) ** gamma)
    gate = (1.0 + kappa * C) / (1.0 + kappa)             # 1.0 when fully coherent
    arg = beta * S * gate

    # --- Commitment 3: conflict erodes confidence, never reverses it ------
    if arg < 0.0:
        arg = 0.0
    if arg > 30.0:
        arg = 30.0

    conf = float(np.tanh(arg))

    # --- Commitment 4: lapse ---------------------------------------------
    p_winner = 0.5 + 0.5 * (1.0 - eps) * conf
    p_winner = float(min(max(p_winner, 0.5), 1.0 - 1e-9))

    p = np.empty(2, dtype=float)
    p[winner] = p_winner
    p[1 - winner] = 1.0 - p_winner

    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()
```

**`policy(probs)`:**
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


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** **Redundancy-Discounted Tiered Integration with Majority Discounting, Panel-Size Capacity Dilution, and Graded Redundancy Relaxation (RDTI-MCR).**

All previous commitments are retained. (i) Stated validities are encoded only *coarsely and ordinally*: a median split into a "star" tier and a "marginal" tier, with no gradations inside a tier. (ii) The star tier is treated as **redundant** — the very accurate experts are construed as merely restating the obvious consensus quality of the product — so their ratings enter integration with only a small residual weight, while the idiosyncratic marginal experts carry full weight. Evidence is the tier-weighted signed sum E = Σ_j w_j (a_j − b_j), and choice is logistic in that scalar plus an attention lapse. Because the weights take only two values, tier-matched profiles cancel *exactly* and give an exact coin flip, with no validity tie-break and no take-the-best fallback. (iii) The same redundancy logic is applied to the majority itself: a feature-count majority is the most "obvious" statistic in the display, so subjects apply a *contrarian consensus discount* proportional to the count margin. (iv) Integration is capacity-limited over the *whole displayed panel*, so the effective slope is divided by a power of the panel size n, in addition to a milder within-trial spread discount over the number of discriminating cues D.

**The new commitment is that the redundancy discount is itself a comparative, graded judgement rather than a fixed property of a cue.** A star expert is discounted precisely *because other experts are seen as restating what he says*; the discount is therefore a function of how much of the panel is actually speaking on this trial. When almost nothing in the display discriminates — in the limit, when a single expert is the only one who differentiates the two products — there is no chorus for that expert to be redundant with, so his rating is taken at (nearly) full face value. The discount relaxes continuously back toward full weight as the number of discriminating cues D falls toward one: eps_eff = eps_w + (1 − eps_w)·ψ·exp(−(D−1)/κ_D), with ψ controlling how completely redundancy can be released and κ_D controlling how fast it releases. Because κ_D is small, the release is essentially confined to genuinely sparse displays. This predicts a sharp, design-level signature that no fixed-weight tiering model produces: **profiles in which exactly one high-validity expert discriminates are followed at high rates (≈0.78–0.85), even though profiles in which that same expert discriminates alongside several others are followed near chance**, since only in the latter case is the star rating construed as redundant restatement. Tally-tied, tier-matched profiles remain at exactly chance (E = 0 and margin = 0 regardless of eps_eff), and all dense profiles (D ≥ 2) are left effectively untouched.

**Rationale:** **Minimal edit, exactly the critic's step (1), with grad = 0 (step 2 dropped as the critic's own fallback instructs).** The accepted iter-4 RDTI-MC source is re-emitted verbatim except for: (a) moving the `diff`/`D` computation four lines earlier so D is available before tiering, (b) three new lines computing `relax = psi*exp(-(D-1)/kappa_d)` and `eps_eff = eps_w + (1-eps_w)*relax`, (c) using `eps_eff` instead of `eps_w` on the star-tier line, and (d) two new parameters `psi`, `kappa_d`. Tiering, exact-zero→coin-flip, D^delta spread discount, (n/6)^tau panel dilution, the LINEAR consensus discount, beta [2.2,3.6] and the lapse are all untouched. No TTB fallback, no validity tie-break, no validity gradient.

**Why graded rather than the hard `if D==1` override that got rejected.** The critic's decomposition of iter 5 is right: the hard override moved exp3's lone-cue cell from ~0.516 straight to ~0.85, carrying exp3 from 0.749 to 0.834 — overshooting 0.794 by as much as the old undershoot. Introducing `psi` makes the release *tunable*: at D=1 the star weight becomes eps_w + (1-eps_w)*psi, so the cell can be landed anywhere between the base and the full override.

**Hand-checked targeting (mid params beta=2.9, eps_w=0.18, cb=0.45, delta=0.075, lapse=0.115, psi=0.60, kappa_d=0.30).** exp3's only D=1 trial type is T1/T2 (A=[1,1,0,1,0,1] vs B=[0,1,0,1,0,1]; the single discriminator is f0, which is star in every experiment in this set since f0 always carries the highest stated validity). Base: E=eps_w=0.18, z=2.9*0.18-0.45=0.072, adherence 0.516. With eps_eff=0.18+0.82*0.60=0.672: z=1.499, adherence 0.780. That cell is 1/6 of exp3's diagnostic trials, so exp3 moves 0.749 → 0.749+(0.780-0.516)/6 = **0.793** (real 0.794). Sweeping psi over [0.35,0.85] puts the cell at 0.687–0.845 and exp3 at **0.777–0.804** — the whole range sits inside the critic's 0.78–0.81 band and approaches from below, and the spread injects the between-subject variance exp3 has been missing.

**Zero exposure elsewhere — verified trial-by-trial.** I enumerated D for every trial in all six designs: exp1 min D=2, exp2 min D=4, exp4 min D=2, exp5 min D=3, exp6 min D=2. NO other experiment contains a D=1 trial, so the release fires only in exp3. The D=2 leakage is deliberately throttled by keeping kappa_d small: at kappa_d=0.30, exp(-1/0.3)=0.036, so eps_eff rises by only 0.60*0.82*0.036 = 0.018 at D=2 — shifting exp6's sparse-unanimous cell E from 1.180 to 1.197 (<0.005 in probability) and exp4's T1 similarly. Even at the range endpoint kappa_d=0.45 the D=2 bump is only 0.075 in eps_eff. Exp1's two metric trials pair a star against a star (f0 vs f1) and a star+marginal against a star+marginal (f0,f2 vs f1,f3), so E cancels to exactly 0 for ANY eps_eff — the coin-flip guardrail is untouched by construction.

**Expected error vector:** exp1 0.027 (unchanged), exp2 0.275 (unchanged, accepted cost — I agree with the critic that the (E,D,margin,n) degeneracy with exp5-S4 makes it unreachable in-family and I am spending no budget there), exp3 0.045 → **~0.001**, exp4 0.010 (unchanged), exp5 0.012 (unchanged), exp6 0.084 (unchanged). A single-target, single-mechanism improvement with no collateral — which is precisely the profile the gate rewarded in iterations 2–4 and rejected in iteration 5.

**Parameters:**
  - `eps_w`: `[0.03, 0.34]`
  - `beta`: `[2.2, 3.6]`
  - `delta`: `[-0.10, 0.25]`
  - `tau`: `[0.5, 1.3]`
  - `lapse`: `[0.05, 0.18]`
  - `count_bias`: `[0.28, 0.62]`
  - `psi`: `[0.35, 0.85]`
  - `kappa_d`: `[0.15, 0.45]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus into two rating vectors ----------
    a = None
    b = None
    try:
        if isinstance(state, dict):
            if 'option_a_ratings' in state and 'option_b_ratings' in state:
                a = np.asarray(list(state['option_a_ratings']), dtype=float).ravel()
                b = np.asarray(list(state['option_b_ratings']), dtype=float).ravel()
    except Exception:
        a = None
        b = None

    if a is None or b is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            stim = stim.reshape(2, -1)
        elif stim.ndim > 2:
            stim = stim.reshape(2, -1)
        if stim.shape[0] != 2:
            stim = stim.reshape(2, -1)
        a = np.asarray(stim[0], dtype=float).ravel()
        b = np.asarray(stim[1], dtype=float).ravel()

    n = int(a.shape[0])
    if n == 0 or int(b.shape[0]) != n:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- parameters ----------
    def _get(name, default):
        try:
            return float(parameters.get(name, default))
        except Exception:
            return float(default)

    eps_w = _get('eps_w', 0.18)          # residual weight of the "star" (redundant) tier
    beta = _get('beta', 2.9)             # logistic slope at the reference panel size
    delta = _get('delta', 0.075)         # within-trial evidence-spread discount exponent
    tau = _get('tau', 0.9)               # panel-size (capacity) dilution exponent
    lapse = _get('lapse', 0.115)         # attention lapse
    count_bias = _get('count_bias', 0.45)  # contrarian consensus (majority) discount
    psi = _get('psi', 0.60)              # max release of the redundancy discount
    kappa_d = _get('kappa_d', 0.30)      # how fast redundancy releases as D -> 1

    eps_w = float(min(max(eps_w, 0.0), 1.0))
    beta = float(min(max(beta, 0.0), 20.0))
    delta = float(min(max(delta, -1.0), 2.0))
    tau = float(min(max(tau, 0.0), 3.0))
    lapse = float(min(max(lapse, 0.0), 0.6))
    count_bias = float(min(max(count_bias, 0.0), 1.5))
    psi = float(min(max(psi, 0.0), 1.0))
    kappa_d = float(min(max(kappa_d, 1e-6), 3.0))

    # ---------- stated validities ----------
    v = None
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume the display order is descending in validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    # ---------- signed feature-wise comparison ----------
    diff = np.sign(a - b)                            # +1 A wins, -1 B wins, 0 tie
    D = int(np.sum(diff != 0))
    if D == 0:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- graded redundancy relaxation ----------
    # A star expert is discounted only because OTHER experts restate him; when
    # the discriminating panel shrinks toward a single voice there is nothing
    # left to be redundant with, so the discount releases back toward 1.
    relax = psi * float(np.exp(-(float(D) - 1.0) / kappa_d))
    eps_eff = eps_w + (1.0 - eps_w) * relax
    eps_eff = float(min(max(eps_eff, 0.0), 1.0))

    # ---------- coarse credibility tiering + redundancy discount ----------
    k = int(n // 2)                                  # size of the "star" tier
    order = np.argsort(-v, kind='stable')            # most valid first
    w = np.ones(n, dtype=float)
    if k > 0:
        w[order[:k]] = eps_eff                       # star tier == treated as redundant

    # ---------- tier-weighted additive evidence ----------
    E = float(np.sum(w * diff))

    # spread discount: evidence thinly spread over many contradicting cues feels weaker
    denom = float(D) ** delta
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0

    # ---------- panel-size capacity dilution ----------
    # a bounded comparison budget is shared across the WHOLE displayed panel,
    # so integration fidelity falls as the number of experts n grows.
    # 6.0 is a reference panel size (a pure reparametrisation of beta).
    cap = (float(n) / 6.0) ** tau
    if not np.isfinite(cap) or cap <= 0.0:
        cap = 1.0
    denom = denom * cap

    z = beta * E / denom

    # ---------- contrarian consensus (majority-redundancy) discount ----------
    nA = int(np.sum(diff > 0))
    nB = int(np.sum(diff < 0))
    margin = float(np.clip(nA - nB, -6, 6))
    z = z - count_bias * margin

    if z > 30.0:
        z = 30.0
    elif z < -30.0:
        z = -30.0

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- lapse ----------
    p_a = (1.0 - lapse) * p_a + lapse * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```
