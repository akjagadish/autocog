# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

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


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** **Capacity-Gated Strongly-Saturating Redundancy Validity-Inversion Tallying (GVIT-CAP-S4).**

Five commitments describe how people combine binary expert ratings whose stated validities they have been told.

**(1) The count comparator is a gate, not a weight.** Before any evaluation of *whose* rating is whose, the subject asks a purely ordinal question: does one product win on more attributes than the other? This is answered with unweighted feature-wise comparisons. If the answer is "no" (equal numbers of winning attributes, including displays where nothing discriminates), the display is construed as *non-diagnostic*, no candidate is generated, and the subject flips a mental coin. Stated validities are **never** used as a tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) When a majority exists, integration uses inverse-informativeness weights.** The subjective news value of a rating is a smooth, panel-relative *decreasing* function of stated validity: s_j = (v_j - v_min)/(v_max - v_min) within the displayed panel, w_j = 1 - lambda_eff*f(s_j). A near-perfect expert is heard as merely restating the obvious consensus quality of the product, so his rating adds almost nothing (and, when lambda_eff*f > 1, is mildly counter-indicative); a barely-diagnostic expert is surprising and is treated as genuine private information carrying full weight. Evidence is the single scalar E = sum_j w_j*sign(a_j - b_j).

**(3) The redundancy construal saturates VERY STRONGLY just above the panel floor.** Redundancy is not judged on a linear scale of stated accuracy: "being one of the accurate ones" is a nearly categorical construal. An expert only has to be slightly above the panel's least-accurate voice before he is already heard as echoing the consensus. Formally the discount is a strongly concave function of panel-relative validity, f(s) = s^omega with omega well below 1 (~0.26-0.42), so the discount rises steeply just above the panel floor and then flattens. Two consequences follow that neither a linear nor a mildly concave inversion produces. (i) A majority carried by *middling* experts (the 2nd-5th most accurate voices) carries almost no subjective evidence: mid-panel support is discounted nearly as heavily as star support, so a mid-backed majority facing a lone star is followed only weakly, close to chance. (ii) Conversely, when the count loser is backed by the panel's genuinely idiosyncratic bottom voices, the anti-count evidence is *strengthened*, sharpening the reversal. So the model separates three displays that a linear inversion renders nearly identical: majority-by-mid-experts (followed barely above chance), majority-by-bottom-voices (followed near ceiling), and star-backed majorities (reversed toward floor).

**(4) Redundancy is comparative; evidence spread over many contradicting cues is compressed.** An expert is discounted only because other experts are heard as echoing him, so the inversion scales with how many voices actually speak on the trial: lambda_eff = lambda_n*(D-1)/((D-1)+kappa_D), where D is the number of discriminating cues; when exactly one expert differentiates the products there is no chorus and his rating is taken at face value. Confidence in E is then compressed by the number of mutually contradicting cues, D^gamma.

**(5) The inversion is a capacity-limited construal: it collapses when the panel exceeds working-memory span.** Discounting the star requires holding the validity ordering of the whole panel in mind, which is possible only up to roughly seven experts. Beyond capacity the ordering blurs, so the redundancy construal washes out toward equal weighting (lambda_n = lambda*(1 - psi*o)) and integration becomes noisier (slope divided by 1 + chi*o), where o = logistic((n - 7.5)/0.2). Finally an attention lapse epsilon replaces the decision by a coin flip.

Cross-design signature: exact chance on tally-tied profiles; strong systematic reversal of star-backed majorities on panels of <=7 experts; near-chance for the same star-backed profiles on panels of 8+ experts, with preserved tally adherence there for broad-mixture majorities; near-ceiling following of a lone discriminating expert; only weakly above-chance following of mid-panel-backed majorities that oppose a lone star; and markedly attenuated confidence whenever the weighted evidence is spread across many mutually contradicting cues.

**Rationale:** MINIMAL DIFF from the running-best iter-6 base: the `predict`/`policy` source is re-emitted verbatim except for two cosmetic `_get` defaults (psi 0.48 -> 0.40, epsilon unchanged at 0.17) that the sampler always overrides. No mechanism, equation, gate, weight function, or name changed. Still a GVIT instantiation: one smooth panel-relative inverse-informativeness weight w_j = 1 - lam_eff*s_j^omega, one scalar E, D^gamma dilution, capacity modulation, lapse. No count-margin contrarian term, no validity tie-break on tally-tied displays, no graded panel-size term.

I execute EXACTLY the critic's pre-specified fallback, which was triggered by iteration 8's trip conditions (Exp4 0.712 < 0.72 and Exp6 -0.905 > -0.90 in magnitude terms):

1) KEEP THE psi MOVE, DROP THE epsilon CENTRE MOVE. psi [0.40, 0.56] -> [0.32, 0.48] (centre 0.40). This is the one knob with a twice-measured, correctly-signed, near-orthogonal gradient: dExp8/dpsi ~ +0.55/unit vs dExp4/dpsi ~ +0.10/unit, so lowering psi pulls Exp8 back from 0.557 toward ~0.50-0.52 (real 0.485 — Exp8 was the LARGEST residual of the iter-6 base at sq ~0.0052) while barely touching Exp4 (same n=8) and leaving Exps 1,2,3,5,6,7 (all n<=7, where the capacity indicator o~0) mathematically untouched. epsilon's CENTRE returns to the iter-6 value 0.17; the +0.02 centre raise is what killed iteration 8 by compressing spread on Exps 5 and 6 and costing Exp4's mean.

2) FIX VARIANCE WITHOUT MOVING MEANS — now the binding term. Point-estimate residuals are already small; the loss is being paid on between-subject dispersion, worst on Exp5 (sim ~0.010 vs real 0.0200) and Exp6 (sim ~0.044 vs real 0.0693). I widen SYMMETRICALLY about the unchanged iter-6 centres only the two knobs that scale the reversal magnitude |E| and the lapse: epsilon [0.09,0.25] -> [0.07,0.27] (centre still 0.17) and lam [1.18,1.72] -> [1.10,1.80] (centre still 1.45). Both spread the per-subject reversal magnitude across draws while leaving the expected value approximately fixed, which is exactly the mechanism the two difference metrics (Exp5, Exp6) need to reproduce their large between-subject variance. Exp3/Exp4 variances should rise toward 0.004-0.006 as a side benefit; Exp1 is driven only by the parameter-free tie gate and is untouched by both widenings, so its slight over-dispersion is not aggravated.

3) FREEZE EVERYTHING ELSE, per the critic. gamma [0.74,1.06] centre 0.90 and beta [5.8,7.6] centre 6.7 are restored/left at iter-6 values (the coupled gamma/beta rotation was REJECTED at iter 7 and must not be repeated or hard-reversed). omega stays at [0.26,0.42] (pushed four times, already overshot its guard rails, no longer the arbiter). kappa_d, chi and the capacity constants are unchanged; chi is explicitly avoided because it moves Exp8 and Exp4 in the same direction and Exp4 cannot afford it.

Expected envelope: Exp1 ~0.47-0.51, Exp2 ~0.69-0.72, Exp3 ~0.76-0.79, Exp4 ~0.73-0.75, Exp5 ~-0.75 to -0.78 with var >= 0.010, Exp6 ~-0.95 to -0.97 with var >= 0.045, Exp7 ~0.66-0.69, Exp8 ~0.50-0.53. The psi step alone recovers most of the Exp8 residual that dominated the iter-6 base (~0.0052 of squared error), and the two symmetric widenings add variance credit on Exps 3/4/5/6 without shifting any mean, so the combined edit should land strictly below the 0.0519 floor rather than trading a mean gain for a variance loss as iteration 8 did.

**Parameters:**
  - `lam`: `[1.10, 1.80]`
  - `beta`: `[5.8, 7.6]`
  - `gamma`: `[0.74, 1.06]`
  - `kappa_d`: `[0.01, 0.45]`
  - `psi`: `[0.32, 0.48]`
  - `chi`: `[0.75, 1.15]`
  - `epsilon`: `[0.07, 0.27]`
  - `omega`: `[0.26, 0.42]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus ----------
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
        try:
            stim = np.asarray(state, dtype=float)
            if stim.ndim == 1:
                stim = stim.reshape(2, -1)
            elif stim.ndim > 2:
                stim = stim.reshape(2, -1)
            if stim.shape[0] != 2:
                stim = stim.reshape(2, -1)
            a = np.asarray(stim[0], dtype=float).ravel()
            b = np.asarray(stim[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    n = int(a.shape[0])
    if n == 0 or int(b.shape[0]) != n:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- feature-wise comparison ----------
    d = np.sign(a - b)                       # +1 A wins cue, -1 B wins cue, 0 tie
    nA = int(np.sum(d > 0))
    nB = int(np.sum(d < 0))
    D = nA + nB

    # ---- Commitment 1: the count comparator is a gate. No majority -> guess.
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- parameters ----------
    def _get(name, default):
        try:
            return float(parameters.get(name, default))
        except Exception:
            return float(default)

    lam = _get('lam', 1.45)
    beta = _get('beta', 6.7)
    gamma = _get('gamma', 0.90)
    kappa_d = _get('kappa_d', 0.10)
    psi = _get('psi', 0.40)
    chi = _get('chi', 0.95)
    eps = _get('epsilon', 0.17)
    omega = _get('omega', 0.34)              # saturating-redundancy exponent

    lam = float(min(max(lam, 0.0), 2.5))
    beta = float(min(max(beta, 0.0), 30.0))
    gamma = float(min(max(gamma, 0.0), 2.5))
    kappa_d = float(min(max(kappa_d, 1e-6), 3.0))
    psi = float(min(max(psi, 0.0), 1.0))
    chi = float(min(max(chi, 0.0), 5.0))
    eps = float(min(max(eps, 0.0), 0.6))
    omega = float(min(max(omega, 0.2), 1.5))

    # ---------- stated validities ----------
    v = None
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    rng = v_max - v_min
    if not np.isfinite(rng) or rng <= 1e-9:
        s = np.zeros(n, dtype=float)          # no validity contrast in the panel
    else:
        s = (v - v_min) / rng                 # panel-relative informativeness, in [0,1]

    # ---- Commitment 3: the redundancy construal SATURATES above the floor ----
    # strongly concave transform: being appreciably above the panel's least-accurate
    # voice is already enough to be heard as "one of the accurate ones".
    s = np.power(np.clip(s, 0.0, 1.0), omega)

    # ---- Commitment 5: capacity overload of the panel (~7 experts) --------
    N_CAP = 7.5
    W_CAP = 0.20
    arg_o = (float(n) - N_CAP) / W_CAP
    arg_o = float(min(max(arg_o, -60.0), 60.0))
    o = 1.0 / (1.0 + np.exp(-arg_o))

    lam_n = lam * (1.0 - psi * o)
    if not np.isfinite(lam_n) or lam_n < 0.0:
        lam_n = 0.0

    # ---- Commitment 4: redundancy is comparative (needs a chorus) -------
    lam_eff = lam_n * (float(D) - 1.0) / ((float(D) - 1.0) + kappa_d)
    if not np.isfinite(lam_eff) or lam_eff < 0.0:
        lam_eff = 0.0

    # ---- Commitment 2: inverse-informativeness weights ------------------
    w = 1.0 - lam_eff * s
    E = float(np.sum(w * d))

    # ---- Commitment 4b: compression by the number of contradicting cues --
    denom = float(D) ** gamma
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    overload_noise = 1.0 + chi * o
    if not np.isfinite(overload_noise) or overload_noise <= 0.0:
        overload_noise = 1.0
    denom = denom * overload_noise

    z = beta * E / denom
    if z > 30.0:
        z = 30.0
    elif z < -30.0:
        z = -30.0

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---- attention lapse ------------------------------------------------
    p_a = (1.0 - eps) * p_a + eps * 0.5
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


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** **Superlinear Redundancy-Discounted Tallying with a Bounded Anti-Count Floor, Baselined Seniority-Scaled Dissent Salience, Dissent-Graded Evidence Dilution and Two-Regime Panel-Size Precision Loss (SR-DT-6b).**

People told the validities of a panel of binary expert ratings do not weight experts *by* validity; they weight them by how much *news* a rating carries, which is a decreasing function of how senior the expert is inside the displayed panel. Eight commitments.

**(1) The count comparator is a gate, not a weight.** Before any expert is evaluated, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not (equal win-counts, including wholly non-discriminating displays), the display is construed as non-diagnostic: choice is exactly 0.5, with NO validity tie-break and NO take-the-best fallback. Parameter-free: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) Seniority is measured against the chance baseline and the panel's best voice, and the discount is SUPERLINEAR in seniority.** s_j = (v_j - 0.5)/(v_max - 0.5), clipped to [0,1]; news value w_j = 1 - lambda * s_j^omega with omega > 1. The discount is negligible across the bottom half of the panel, about half weight for genuine mid-panel experts, and only for the top one or two voices does it drive weight to zero or below. Two mid-panel experts therefore roughly offset one floor-level voice, while a single top expert is worth nothing or less.

**(2b) The anti-count is BOUNDED: news value has a mild negative floor.** w_j = max(w_floor, 1 - lambda * s_j^omega) with a small negative w_floor. Endorsement by the panel's star can subtract a little, never a lot.

**(3) Redundancy requires same-side corroboration.** lambda_side = lambda_n * (m_side - 1)/((m_side - 1) + kappa). A lone voice cannot be redundant with anybody.

**(4) A lone voice is heard at MORE than face value, with a seniority-graded bonus on top of a constant dissent baseline.** w = phi0 + (phi - phi0) * s_j for a solitary discriminating cue.

**(5) Confidence is diluted by DISSENT more than by sheer cue count, but only MILDLY so.** Two functionally distinct things can make a display 'busy': many attributes discriminating at all, and some of those attributes pointing the other way. Only the second is genuinely undermining, but its undermining force is *modest and graded*, not overwhelming. Formally the divisor is D^gamma_d * (1 + m_lose)^gamma_c with a sub-linear cue-count exponent gamma_d and a mild-to-moderate dissent exponent gamma_c that is bounded well below 1.0, where m_lose is the number of cues backing the count loser. Consequences: unopposed majorities grow in subjective strength with their size (dense-unanimous adherence exceeds sparse-unanimous adherence), while internally contradicted displays are pulled somewhat — but never all the way — toward indifference. Crucially the *asymmetry* of this dilution is empirically constrained: displays with a single dissenting voice (lone-star dissent, the cells that generate the large negative coherence contrasts) must retain nearly full evidential force, whereas displays with two or more dissenters are noticeably softened. A strong dissent exponent would flatten the former and is ruled out.

**(6) Precision also falls, in two regimes, with panel size:** divisor multiplied by (n/6)^tau * (1 + eta * max(0, n - 7)).

**(7) Ordering fidelity decays smoothly with panel span:** lambda_n = lambda * exp(-rho * max(0, n - 7)).

**(8) The decision stage is low-lapse but only moderately steep.** The lapse rate epsilon alone fixes the ceiling attainable on evidence-saturated displays; the slope beta alone fixes how far intermediate, internally conflicted displays are driven from indifference. Stimulus-driven evidence strength and stimulus-independent attentional failure are separate sources of stochasticity.

**Cross-design signature:** exact chance on tally-tied displays of any validity composition; strong but bounded reversal of chorus-backed star majorities on six/seven-expert panels; near-chance for the same structure on eight-expert panels with visibly reduced pure-tally adherence there; near-ceiling and *increasing-with-size* following of unopposed majorities; distinctly middling adherence whenever a numeric majority is actively contradicted by several cues; and mid-panel-backed majorities facing a lone floor voice sitting essentially at chance.

**Rationale:** Minimal-diff edit on the ACCEPTED iter-8 base (loss 0.0541). `predict` and `policy` are re-emitted byte-identical except for two fallback-default literals (gamma_d 0.72 -> 0.68, gamma_c 0.80 -> 0.66, which are never used because both are always sampled); only TWO parameter-range lines actually change. Every commitment, the count gate, the seniority profile, the corroboration term, the lone-dissent branch, w_floor, beta/epsilon, kappa/rho/tau/eta and the two-regime size factor are untouched.

**(1) Top-priority edit, read off the gradient the rejected iter-9 run measured: gamma_c [0.55, 1.00] -> [0.42, 0.90] (centre 0.775 -> 0.66).** Iter 9 moved gamma_c up by +0.26 and paid Exp6 0.077 and Exp5 0.043 to gain only 0.012 on Exp7 — i.e. the m_lose=1 cells that carry the coherence contrasts are ~5x more sensitive per unit gamma_c than the m_lose=2 cells that carry Exp7. Running that measured slope backwards, a -0.115 step should recover ~0.03-0.04 of Exp6's dominant +0.1375 residual (-0.800 -> ~-0.835), deepen Exp5 from -0.749 *through* (not past) its target toward ~-0.765 (real -0.763), and cost only ~0.005-0.010 each on Exp7 and Exp2. The new interval still contains 0.85-0.90, so the optimiser can recover the accepted behaviour almost exactly and this dimension cannot regress badly; the region >=1.0 that the gate just rejected is excluded entirely. This is a step in the direction OPPOSITE to the rejected iter-9 package, so it is not a repeat of failed advice.

**(2) Companion edit, the untried lever the critic flagged: gamma_d [0.60, 0.85] -> [0.50, 0.85] (widen downward only, accepted interval kept strictly inside; centre 0.725 -> 0.675).** This is deliberately half the width the critic offered ([0.45, 0.85]) because it touches the unanimous cells that the near-exact Exp4 and Exp8 depend on. A slightly smaller sub-linear cue-count exponent sharpens *unopposed* majorities (dissent factor = 1 there) more than sparse ones, which is the only in-family route to a NEGATIVE d2 term in Exp6's metric — currently d2 is pinned near zero because both unanimous cell types sit on the lapse plateau, forcing all of Exp6 onto d1. It simultaneously helps the two remaining sub-target pure-tally metrics: Exp3 0.763 -> ~0.775 (real 0.794) and Exp4 0.749 -> ~0.757 (real 0.755). It also partially offsets edit (1) on Exp3, whose conflicted cells sit slightly below chance and would otherwise drift down as dissent dilution weakens.

**Why these two move together.** Edit (1) buys Exp6's d1 and Exp5; edit (2) buys Exp6's d2, Exp3 and Exp4; their costs are on different cells (Exp7/Exp2 for (1), Exp8/Exp10 for (2)) and are each bounded at ~0.01-0.015 by the smallness of the steps. Predicted net: Exp6 +0.04-0.06, Exp5 +0.01, Exp3 +0.01, Exp4 ~0, against ~0.03 of aggregate cost — enough to clear the 0.0541 floor without disturbing either structural discriminator (Exp1's exact-chance count gate, Exp10's near-zero contrast).

**Explicitly untouched, per the accumulated ACCEPTED/REJECTED record:** the parameter-free count gate (no validity tie-break — Exp1's deviation is Bernoulli noise around an exact 0.5); lam and omega (Exp10's mid-panel half-weight and the C4 balance, the corpus's cleanest discriminator against pi_5's -0.73 and pi_6's +0.33); w_floor (no positive floor — rejected at iter 4); phi/phi0 (withdrawn permanently after the iter-5 rejection); beta and epsilon (both moved in both directions already — frozen, and in particular the epsilon trim that shipped inside the rejected iter-9 package is NOT repeated); kappa/rho/tau/eta. No hard opposition gate (declined at iter 2); dilution remains smooth and graded, per the arbiter's insistence on no thresholds.

**Parameters:**
  - `lam`: `[1.35, 1.90]`
  - `omega`: `[1.45, 2.05]`
  - `beta`: `[11.0, 16.0]`
  - `gamma_d`: `[0.50, 0.85]`
  - `gamma_c`: `[0.42, 0.90]`
  - `kappa`: `[0.02, 0.15]`
  - `rho`: `[0.55, 0.90]`
  - `tau`: `[0.0, 0.4]`
  - `eta`: `[1.6, 2.8]`
  - `phi`: `[1.15, 1.75]`
  - `phi0`: `[1.00, 1.15]`
  - `w_floor`: `[-0.45, -0.10]`
  - `epsilon`: `[0.05, 0.13]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus ----------
    def _pair(st):
        a = None
        b = None
        if isinstance(st, dict):
            for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                           ('a_ratings', 'b_ratings'),
                           ('A', 'B'), ('a', 'b')):
                if ka in st and kb in st:
                    try:
                        a = np.asarray(list(st[ka]), dtype=float).ravel()
                        b = np.asarray(list(st[kb]), dtype=float).ravel()
                    except Exception:
                        a = None
                        b = None
                    break
        if a is None:
            try:
                arr = np.asarray(st, dtype=float)
                arr = arr.reshape(2, -1)
                a = np.asarray(arr[0], dtype=float).ravel()
                b = np.asarray(arr[1], dtype=float).ravel()
            except Exception:
                return None, None
        return a, b

    a, b = _pair(state)
    if a is None or b is None:
        return np.array([0.5, 0.5], dtype=float)
    if a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------- Commitment 1: unweighted count gate ----------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                       # includes the wholly non-discriminating case
        return np.array([0.5, 0.5], dtype=float)
    D = float(mA + mB)
    m_lose = float(min(mA, mB))        # cues backing the count LOSER (dissent)

    # ---------- parameters ----------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    lam = _g('lam', 1.60, 0.0, 3.0)
    omega = _g('omega', 1.70, 0.5, 3.5)
    beta = _g('beta', 13.5, 0.0, 40.0)
    gamma_d = _g('gamma_d', 0.68, 0.0, 2.5)   # sub-linear dilution by sheer cue count
    gamma_c = _g('gamma_c', 0.66, 0.0, 2.5)   # mild dilution by DISSENT
    kappa = _g('kappa', 0.05, 1e-6, 1.0)
    rho = _g('rho', 0.70, 0.0, 2.0)
    tau = _g('tau', 0.20, 0.0, 3.0)
    eta = _g('eta', 2.10, 0.0, 6.0)
    phi = _g('phi', 1.40, 0.5, 2.5)
    phi0 = _g('phi0', 1.08, 0.5, 2.0)
    eps = _g('epsilon', 0.09, 0.0, 0.6)
    w_floor = _g('w_floor', -0.30, -1.5, 0.0)   # bounded anti-count (Commitment 2b)

    # ---------- stated validities ----------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    # ---------- Commitment 2: seniority anchored at chance & the panel's best voice ----------
    vmax = float(np.max(v))
    span = vmax - 0.5
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - 0.5) / span, 0.0, 1.0)
    s_pow = np.power(np.clip(s, 0.0, 1.0), omega)   # superlinear discount profile

    # ---------- Commitment 7: span-graded ordering fidelity ----------
    over = max(0.0, float(n) - 7.0)
    lam_n = lam * float(np.exp(-rho * over))
    if (not np.isfinite(lam_n)) or lam_n < 0.0:
        lam_n = 0.0

    # ---------- Commitment 3: redundancy needs same-side corroboration ----------
    def _side_lam(m):
        mm = float(m)
        if mm <= 1.0:
            return 0.0                       # a lone voice cannot be redundant
        val = lam_n * (mm - 1.0) / ((mm - 1.0) + kappa)
        if (not np.isfinite(val)) or val < 0.0:
            return 0.0
        return float(val)

    lam_A = _side_lam(mA)
    lam_B = _side_lam(mB)

    # ---------- Commitment 2b: news value is bounded below by a mild negative floor ----------
    if mA > 0:
        wA = np.maximum(w_floor, 1.0 - lam_A * s_pow[iA])
    else:
        wA = np.array([], dtype=float)
    if mB > 0:
        wB = np.maximum(w_floor, 1.0 - lam_B * s_pow[iB])
    else:
        wB = np.array([], dtype=float)

    # ---------- Commitment 4: baselined, seniority-scaled lone-dissent salience ----------
    phi_eff = phi if phi > phi0 else phi0
    if mA == 1:
        wA = phi0 + (phi_eff - phi0) * s[iA]
    if mB == 1:
        wB = phi0 + (phi_eff - phi0) * s[iB]

    E = float(np.sum(wA) - np.sum(wB))
    if not np.isfinite(E):
        E = 0.0

    # ---------- Commitment 5: DISSENT-graded dilution (sub-linear in sheer cue count) ----------
    den = (float(D) ** gamma_d) * ((1.0 + m_lose) ** gamma_c)
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    # ---------- Commitment 6: panel-size precision loss (two regimes) ----------
    size_fac = ((float(n) / 6.0) ** tau) * (1.0 + eta * max(0.0, float(n) - 7.0))
    if (not np.isfinite(size_fac)) or size_fac <= 0.0:
        size_fac = 1.0
    den = den * size_fac

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- Commitment 8: attention lapse (ceiling) is separate from slope ----------
    p_a = (1.0 - eps) * p_a + eps * 0.5
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
    if (not np.isfinite(s)) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```
