# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** **PWA-DD: Position-Bound Weighted Additive integration with Faint-Praise discounting and Endorsement-Density Dilution.**

Choice in binary-cue environments is always *compensatory*: on every trial the decision maker reads all displayed columns, forms a single weighted difference, and never stops early. There is no lexicographic cascade and no coherence/unanimity premium. Three claims fix the rest of the mechanism.

(1) **Weights are a mixture of validity and display position, and the mixture is governed by how readable the validity-to-column mapping is.** Stated validities must be bound to *columns*. When the display is perfectly monotone in validity (ascending or descending) that binding is trivial and people use validity-derived weights w_val ∝ logit(v_j)^γ essentially intact (the residual positional leak μ0 is small and person-specific, and in some people is exactly zero). When the column order is not monotone in validity — and especially when the best expert sits in an interior column — binding fails and people fall back almost entirely on a purely positional gradient w_pos ∝ (n−j)^λ that decays left-to-right over displayed columns. The mixture weight μ = μ0 + (1−μ0)·min(1, μ1(1−|τ|) + μ2·[best cue interior]) is therefore an *environmental* quantity, not a per-experiment fudge: it is inert (μ = μ0) whenever columns are validity-sorted, and it predicts that scrambling a sorted display converts validity-driven behaviour into position-driven behaviour and that re-sorting a scrambled display restores it. Crucially the gate SATURATES: any display that is appreciably non-monotone is read essentially positionally, with no graded validity leak surviving.

(2) **Faint-praise endorsement discounting.** Each positive endorsement carries a small fixed credibility cost θ: an option that collects more '1's is slightly *less* attractive, all else equal. This is strictly monotone in endorsement count — there is no step at unanimity. It makes a lone weak endorsement no better (often slightly worse) than an empty panel, and it makes a 5/5 panel no better than a 4/5 panel by exactly the same amount, which is why 'high-coherence' and 'low-coherence' single-cue steps move together rather than apart. θ sits just at the subjective weight of the weakest experts — close enough that a lone weak endorsement is near-neutral to mildly counter-diagnostic rather than strongly aversive — while endorsement by a strong expert still wins comfortably.

(3) **Endorsement-density dilution of decision sensitivity.** The same absolute weighted difference is less decisive when both panels are densely endorsed. Evidence is divisively normalised by the mean endorsement density T = (Σa + Σb)/2n: D = [Σ_j w_j(a_j−b_j) − θ(Σa−Σb)] / (c0 + κ·T). The dilution has TWO separable constants: a floor c0 that sets how crisply *sparse* panels are compared, and a gain κ that sets how much extra mush dense panels accrue. The floor is *not* negligible: even near-empty panels are compared with bounded, distinctly sub-ceiling determinacy, which keeps lone-endorsement-versus-empty cells close to (rather than far below) chance and preserves the graded ladder structure of sparse comparisons. Heavily endorsed panels are compared mushily. Unlike a consensus premium, dilution is *direction-blind* — it never confers an advantage on the more coherent option, it only compresses whatever advantage exists — so it produces the observed shrinkage of a fixed one-cue advantage at high endorsement backgrounds without ever predicting a unanimity bonus.

p(A) = (1−ε)·logistic(βD) + ε/2. Everyone holds the same stationary rule (no feedback is given, so nothing is learned); heterogeneity is broad and continuous — in determinacy β, validity sensitivity γ, discount θ, positional decay λ, dilution (c0, κ), lapse ε and the readability coefficients — never discrete strategy switching. The theory therefore predicts substantial between-subject spread in every conflict metric, largest where margins are intermediate.

**Rationale:** Strict single-knob edit on the iter-5 running-best base, exactly in the direction the critic prescribed, with the critic's own contingency applied. `predict`/`policy` are mechanically byte-identical (only the inert `theta` `.get()` default changed, 0.12 -> 0.1075); ONE parameter box changes and gamma, c0, kap, mu0, mu1, mu2, beta, lam, eps are all left at their iter-5 values (i.e. gamma reverted from the rejected iter-8 [0.9,1.6] back to [1.1,1.9]).

**The edit: theta [0.08, 0.16] (mid 0.12) -> [0.07, 0.145] (mid ~0.1075).** The critic asked for theta DOWN (a direction never previously advised; the only prior theta advice was the iter-1 raise, which is what flipped Exp 5 positive) and explicitly instructed: hand-check Exp 3 first, and if the contrast rises above ~-0.170 take the HALF-step [0.07, 0.145] instead of the full [0.06, 0.13]. I ran that hand-check at mid-box (gamma=1.5, beta=3.8, c0=0.09, kap=1.7, mu0=0.07, lam=0.95, eps=0.125) and the full step fails the guardrail, so the half-step is what I submit.

Hand-check, Exp 3 (ascending display, mu=mu0, w_eff = [.034,.059,.107,.301,.500]): at theta=0.12 the agree cells are D=0.793, G=0.934, H=0.255 (mean 0.661) and the conflict cells are 0.924/0.845/0.912/0.674 (mean 0.839), contrast -0.178 -- matching the simulated -0.182, so the arithmetic is calibrated. At theta=0.095 the H cell rises to 0.315 and D to 0.844, giving contrast **-0.134** (well above the -0.170 guardrail, i.e. as damaging as the rejected gamma pass). At theta=0.1075 H=0.283, D=0.819, F=0.661, giving contrast **-0.157** -- inside the guardrail, a ~0.021 cost.

Same check on the two target residuals. Exp 5 (descending display): the sparse lo-cell numerator is w_j - theta with a tiny denominator (0.26), so it is the most theta-sensitive cell in the design; p_lo rises 0.339 -> 0.370 while the dense hi-cell (T=0.9, denom 1.62) barely moves (0.447 -> 0.453), so hi-minus-lo falls 0.108 -> 0.083 at mid-box, i.e. simulated ~0.048 -> ~0.028 (residual +0.061 -> ~+0.041). Exp 1: the 1-vs-4 conflict pairs have numerator 2*w0-1+3*theta, so lowering theta lowers the top-cue evidence WITHOUT flattening the weight profile (which is why Exp 3's ORDER comparisons keep their magnitude, unlike the gamma pass); mean conflict hit rate 0.778 -> 0.770 at mid-box, i.e. ~0.798 -> ~0.788.

Protected fits verified numerically. Exp 6 (positional, gate saturated): U4-U1 = -0.048 -> -0.079, V2-V1 = -0.287 -> -0.286, L1-L4 = -0.024 -> 0.000, mean -0.120 -> -0.121 -- essentially inert and, if anything, a touch more negative (target -0.150), so the most fragile good fit is safe. Exp 2 sits at chance and theta enters only through count differences there; Exp 4's trials are moderately dense so the divisor blunts the change. Dominance anchor: [1,1,1,1,1] vs [0,0,0,0,0] has num = 1 - 5*theta = 0.462 (up from 0.40), T=0.5, denom 0.94, D=0.492, beta*D=1.87 -> p_core 0.866 -> p ~0.82 after the mid lapse; stronger than before but not implausibly deterministic.

Net expected in sum-of-absolute-residual terms: gains of ~0.010 (Exp 1), ~0.020 (Exp 5), ~0.002 (Exp 6) and ~0.005 (Exp 4) against a ~0.021 cost on Exp 3 -- a net improvement that should land the aggregate just below the 0.0384 floor. Isolating theta also keeps the pass cleanly attributable, the property both ACCEPTED iterations had and every rejected pass lacked.

**Parameters:**
  - `gamma`: `[1.1, 1.9]`
  - `beta`: `[2.0, 5.6]`
  - `theta`: `[0.07, 0.145]`
  - `c0`: `[0.05, 0.13]`
  - `kap`: `[1.1, 2.3]`
  - `mu0`: `[0.0, 0.14]`
  - `mu1`: `[1.4, 2.2]`
  - `mu2`: `[0.20, 0.60]`
  - `lam`: `[0.70, 1.20]`
  - `eps`: `[0.03, 0.22]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # PWA-DD: position-bound weighted additive integration with
    # faint-praise endorsement discounting and endorsement-density dilution.
    #
    #   w_val_j  proportional to logit(v_j) ** gamma      (validity weights)
    #   w_pos_j  proportional to (n - j) ** lam           (left-to-right gradient)
    #   mu       = mu0 + (1-mu0)*min(1, mu1*(1-|tau|) + mu2*[best cue interior])
    #   w        = (1-mu)*w_val + mu*w_pos                (normalised)
    #   num      = w.(a-b) - theta*(sum a - sum b)        (faint-praise discount)
    #   T        = (sum a + sum b) / (2n)                 (endorsement density)
    #   D        = num / (c0 + kap*T)                     (density dilution)
    #   p(A)     = (1-eps)*logistic(beta*D) + eps/2
    #
    # No stopping rule, no coherence/unanimity premium.  History unused:
    # the rule is stationary because no feedback is provided.
    import numpy as np

    # ---------------- unpack the two rating vectors --------------------
    a = None
    b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            half = int(arr.shape[0] // 2)
            a = arr[:half].astype(float)
            b = arr[half:2 * half].astype(float)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            a = arr[0].astype(float)
            b = arr[1].astype(float)

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float).ravel()[:n]
    b = np.asarray(b, dtype=float).ravel()[:n]

    # ---------------- validities ---------------------------------------
    val = parameters.get("validities", None)
    if val is None:
        v = np.linspace(0.90, 0.55, n)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---------------- parameters ---------------------------------------
    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 3.8))
    theta = float(parameters.get("theta", 0.1075))
    c0 = float(parameters.get("c0", 0.09))
    kap = float(parameters.get("kap", 1.7))
    mu0 = float(np.clip(float(parameters.get("mu0", 0.07)), 0.0, 1.0))
    mu1 = float(parameters.get("mu1", 1.8))
    mu2 = float(parameters.get("mu2", 0.4))
    lam = float(parameters.get("lam", 0.95))
    eps = float(np.clip(float(parameters.get("eps", 0.125)), 0.0, 1.0))
    c0 = max(c0, 1e-3)

    # ---------------- validity-derived weights --------------------------
    L = np.log(v / (1.0 - v))
    L = np.maximum(L, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        wv = np.power(L, gamma)
    wv = np.nan_to_num(wv, nan=0.0, posinf=1e12, neginf=0.0)
    s = float(np.sum(wv))
    if (not np.isfinite(s)) or s <= 0.0:
        wv = np.ones(n, dtype=float) / float(n)
    else:
        wv = wv / s

    # ---------------- positional (reading-order) weights ----------------
    idx = np.arange(n, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        wp = np.power(np.maximum(float(n) - idx, 1e-9), lam)
    wp = np.nan_to_num(wp, nan=0.0, posinf=1e12, neginf=0.0)
    s = float(np.sum(wp))
    if (not np.isfinite(s)) or s <= 0.0:
        wp = np.ones(n, dtype=float) / float(n)
    else:
        wp = wp / s

    # ---------------- display readability -------------------------------
    # Kendall tau between column index and stated validity (ties skipped).
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    tot = conc + disc
    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)
    incong = 1.0 - abs(tau)                     # 0 when display is monotone
    jmax = int(np.argmax(v))
    interior = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0

    mu = mu0 + (1.0 - mu0) * min(1.0, max(0.0, mu1 * incong + mu2 * interior))
    mu = float(np.clip(mu, 0.0, 1.0))

    w = (1.0 - mu) * wv + mu * wp
    s = float(np.sum(w))
    if (not np.isfinite(s)) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)
    else:
        w = w / s

    # ---------------- evidence ------------------------------------------
    core = float(np.dot(w, a - b))
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    num = core - theta * (cnt_a - cnt_b)          # faint-praise discount

    dens = (cnt_a + cnt_b) / (2.0 * float(n))     # endorsement density in [0,1]
    denom = c0 + kap * dens
    if (not np.isfinite(denom)) or denom <= 1e-6:
        denom = 1e-6
    D = num / denom

    # ---------------- logistic choice with lapse -------------------------
    z = np.array([beta * D, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if (not np.isfinite(tot)) or tot <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** **Readability-limited validity weighting with a strict-unanimity consensus premium (RVW-DE+/U), with broad determinacy heterogeneity.**

A single stationary, fully compensatory integration rule underlies choice in all cue-based binary-feature environments; the apparent switching between 'take-the-best-like' and 'tallying-like' behaviour across environments is produced by modulators that are properties of the *display and the evidence pattern*, not of the strategy, plus wide between-person variation in decision determinacy.

(1) **Graded validity weighting.** Each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, w_j ∝ logit(v_j)^gamma, gamma spread over ~1.05–2.0 across people. At the population mean this is steep enough that a clearly best expert usually survives a coalition of three or four weak experts, yet weak enough that a near-peer plus a couple of moderate cues can overturn the top cue.

(2) **Binding readability, not position.** Stated validities must be bound to displayed columns. When the display order is *monotone* in validity (ascending OR descending) this binding is trivial and the validity weights are used essentially intact. When the order is scrambled, binding fails partially and the weights are flattened toward EQUAL weighting: w_eff = (1-kappa)·w + kappa/n, with kappa = k0 + k1·(1-|Kendall tau(column index, validity)|) + k2·[best expert not in an extreme column]. Direction-free: reversing a sorted display costs nothing, permuting it costs a lot.

(3) **Readability-gated reading-order fallback.** The left-to-right attention gradient is a fallback recruited only when validity-to-column binding fails: phi_eff = phi·(r0 + r1·kappa). It only *scales* weights, never re-sorts them.

(4) **Strict-unanimity consensus premium (dissent aversion).** The premium attaches to *consensus itself*, and consensus is an essentially all-or-none perceptual property: a panel is 'agreed' only when it is unanimous or one voice away from it. Subjects add a direction-blind premium rho·|2·mean(x)-1|^q with a HIGH exponent q (~5–11, person-specific), so a 5/5 or 0/5 panel earns almost the whole premium while a 4/5 or 1/5 panel earns almost none. The magnitude rho of this premium is substantial and heterogeneous (population mean ~0.39): an option unanimously rated 0 can be *preferred* to one carrying a single weak endorsement (sub-chance accuracy on one-endorsement-vs-nothing pairs), while options that are merely lopsided (4-of-5 endorsements) receive essentially no credibility bonus and therefore cannot dilute genuine weighted-evidence conflicts. The premium is a *unanimity* effect, not a mean-rating effect — that is the theory's sharpest falsifiable commitment (a strongly non-linear step at exactly 0 and n dissenters, not a smooth gradient in endorsement proportion).

(5) **Endorsement discount / pull.** Each positive endorsement carries a small credibility adjustment theta (faint-praise skepticism for most, mild endorsement-count pull for some), inert whenever the two options carry equal endorsement counts.

(6) **Population structure.** Everyone holds the same rule; heterogeneity is broad and continuous — in determinacy beta, in gamma, in lapse, in the modulator magnitudes rho, q, theta, phi and the readability coefficients. No discrete strategy switching.

Evidence for A: D = Σ_j w_eff_j (x_Aj - x_Bj) - theta·(Σx_A - Σx_B) + rho·(coh(x_A) - coh(x_B)); p(A) = (1-eps)·logistic(beta·D) + eps/2.

**Rationale:** MINIMAL DIFF: the iter-6 (running-best, loss 0.0292) source is re-emitted VERBATIM except for ONE parameter box — rho moves from [0.14, 0.54] to [0.19, 0.59] (midpoint +0.05, width unchanged) — plus the matching fallback default inside predict() (0.34 -> 0.39, an inert code-path default). gamma, beta, eps, k0, k1, k2, phi, theta, q, r0, r1 are IDENTICAL to the accepted base, and policy() is untouched. No structural change, no positional re-sorting, no discrete mixtures.

WHY EXACTLY THIS STEP. This is a bracketed interpolation, not a new push. The loop has now bracketed the optimum on the one knob the critic validated as Exp-3-selective: iter 6 (rho mid 0.34) left Exp 3 at -0.168 (+0.040 too HIGH) and Exp 1 at 0.7296 (-0.024 too low); iter 7 (rho mid 0.44) took Exp 3 to -0.276 (-0.068 too LOW) and Exp 1 to 0.7529 (essentially exact). The response is close to linear in rho over that interval (Exp 3 slope ~ -1.09 per unit rho-mid; Exp 1 slope ~ +0.23; Exp 2 slope ~ +0.18; Exp 4 slope ~ 0). A half-step at mid 0.39 is the standard secant/bisection move and, unlike iters 4/5/7, it is a step INSIDE a known bracket rather than an extrapolation, which is why it should not overshoot in either direction.

PREDICTED DIRECTIONS (stated before simulation):
- Exp 3: -0.168 -> ~-0.22, target envelope [-0.24, -0.19]; |error| 0.040 -> ~0.012.
- Exp 1: 0.730 -> ~0.741, envelope [0.735, 0.750]; |error| 0.024 -> ~0.012 (rho helps Exp 1 through the [10001]v[01111] and [11000]v[10111] families, keeping half of iter-7's gain).
- Exp 2: 0.513 -> ~0.522, envelope [0.515, 0.528]; |error| 0.003 -> ~0.012 (the only cost; it is driven by [011100]v[100000], where the near-unanimous-zero TTB winner gains a little premium).
- Exp 4: inert at ~0.640, |error| 0.019 unchanged — all scored Exp-4 trials have |2*mean-1| <= 1/3, where the q~8 premium is ~1e-4, so rho literally cannot move it.
Summed |point error|: base 0.086, iter 7 0.108, this candidate ~0.055. That is the arithmetic reason to expect the gate to accept.

WHAT I DELIBERATELY DID NOT TOUCH, per this loop's own reject record: theta (co-mover of Exps 1 and 2; pushed twice, rejected twice), beta (zero-sum across Exps 1 and 4), the k2/kappa de-saturation edit (falsified — it lowered BOTH Exp 4's mean and its var), k1 (headroom spent once Exp 2 was near-exact), and rho's WIDTH (symmetric widening was falsified in iter 4 because the G/H agreement term saturates at 1 and widening drags Exp 3 toward zero — hence a pure midpoint SHIFT here). I also did not bundle the reserved phi nudge for Exp 4: iters 4, 5 and 7 all showed the loop cannot attribute credit inside a multi-knob or over-sized change, so Exp 4's residual -0.019 is left for a separate single-knob iteration after this step is scored on its own. Variance chasing is likewise deferred — both prior attempts were rejected and the loss is still dominated by point estimates.

Theoretically, the edit is not a fudge: it says the strict-unanimity consensus premium is somewhat stronger in the population mean than previously estimated (rho ~0.39), which is the same single stationary parameter across all four environments and remains the mechanism that uniquely delivers Exp 3's NEGATIVE agreement-minus-conflict sign (sub-chance accuracy on single-endorsement-vs-unanimous-zero pairs) while leaving equal-coherence conflict families untouched.

**Parameters:**
  - `gamma`: `[1.05, 1.95]`
  - `beta`: `[1.2, 13.0]`
  - `eps`: `[0.0, 0.20]`
  - `k0`: `[0.02, 0.16]`
  - `k1`: `[0.55, 1.00]`
  - `k2`: `[0.10, 0.60]`
  - `phi`: `[0.02, 0.38]`
  - `theta`: `[-0.03, 0.07]`
  - `rho`: `[0.19, 0.59]`
  - `q`: `[5.0, 11.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Readability-limited validity weighting with dissent aversion and
    # endorsement discount (RVW-DE+/U).
    #   w_j        proportional to logit(v_j) ** gamma
    #   w_eff      = (1-kappa)*w + kappa/n          (readability flattening)
    #   w_eff     *= (1 + phi_eff * reading-order gradient)
    #                with phi_eff = phi*(r0 + r1*kappa): the reading-order
    #                fallback is recruited only when binding fails.
    #   D          = w_eff.(a-b) - theta*(sum a - sum b) + rho*(coh(a)-coh(b))
    #   p(A)       = (1-eps)*logistic(beta*D) + eps/2
    # kappa is driven by how incongruent the column order is with the
    # stated validity order (direction-free: monotone displays are cheap,
    # scrambled displays are expensive).  coh(x) = |2*mean(x)-1|**q with a
    # HIGH person-specific exponent q: the consensus premium is a strict
    # unanimity effect, not a smooth function of endorsement proportion.
    # History is unused: no feedback is given, the rule is stationary.
    import numpy as np

    # ---------------- unpack the two rating vectors --------------------
    a = None
    b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            half = arr.shape[0] // 2
            a = arr[:half].astype(float)
            b = arr[half:half + half].astype(float)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            a = arr[0].astype(float)
            b = arr[1].astype(float)

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float).ravel()[:n]
    b = np.asarray(b, dtype=float).ravel()[:n]

    # ---------------- validities --------------------------------------
    val = parameters.get("validities", None)
    if val is None:
        v = np.linspace(0.9, 0.55, n)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---------------- parameters --------------------------------------
    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 7.0))
    eps = float(np.clip(float(parameters.get("eps", 0.09)), 0.0, 1.0))
    k0 = float(parameters.get("k0", 0.08))
    k1 = float(parameters.get("k1", 0.78))
    k2 = float(parameters.get("k2", 0.35))
    phi = float(parameters.get("phi", 0.20))
    theta = float(parameters.get("theta", 0.02))
    rho = float(parameters.get("rho", 0.39))
    q = float(parameters.get("q", 8.0))

    # readability-gating of the reading-order fallback (fixed structure)
    r0 = 0.35
    r1 = 1.15

    # ---------------- validity weights --------------------------------
    L = np.log(v / (1.0 - v))
    L = np.maximum(L, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.power(L, gamma)
    w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)
    s = float(np.sum(w))
    if (not np.isfinite(s)) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)
    else:
        w = w / s

    # ---------------- readability flattening --------------------------
    # Kendall tau between column index and stated validity (ties skipped).
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    tot = conc + disc
    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)
    incong = 1.0 - abs(tau)

    jmax = int(np.argmax(v))
    off_extreme = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0

    kappa = k0 + k1 * incong + k2 * off_extreme
    kappa = float(np.clip(kappa, 0.0, 1.0))

    w_eff = (1.0 - kappa) * w + kappa * (1.0 / float(n))

    # ------- reading-order (primacy) gradient, gated by readability ----
    if n > 1 and phi != 0.0:
        phi_eff = phi * (r0 + r1 * kappa)
        phi_eff = float(np.clip(phi_eff, 0.0, 0.45))
        idx = np.arange(n, dtype=float)
        grad = 1.0 + phi_eff * ((float(n - 1) - 2.0 * idx) / float(n - 1))
        grad = np.maximum(grad, 1e-6)
        w_eff = w_eff * grad
    ssum = float(np.sum(w_eff))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        w_eff = np.ones(n, dtype=float) / float(n)
    else:
        w_eff = w_eff / ssum

    # ---------------- evidence ----------------------------------------
    core = float(np.dot(w_eff, a - b))

    # endorsement discount: credibility adjustment per positive rating
    cost = -theta * (float(np.sum(a)) - float(np.sum(b)))

    # dissent aversion: strict-unanimity consensus premium (high exponent q)
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    ua = abs(2.0 * ma - 1.0)
    ub = abs(2.0 * mb - 1.0)
    coh_a = float(np.power(ua, q))
    coh_b = float(np.power(ub, q))
    cons = rho * (coh_a - coh_b)

    D = core + cost + cons

    # ---------------- logistic choice with lapse -----------------------
    z = np.array([beta * D, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if (not np.isfinite(tot)) or tot <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** **ERC-SB (determinacy-corrected): Expectation-Referenced Cue Coding with Saturating Deviation Value, Compressed Endorsement Numerosity, and *Binding-Conditional* Contrast Compression.**

Choice between two binary-cue panels is compensatory, stationary (no feedback is given, so nothing is learned), and rests on one idea: every displayed rating is scored *relative to a subjective endorsement expectation*, and both the deviation from that expectation and the endorsement surplus it is charged against are perceived on compressed psychophysical scales.

(1) *Signed, expectation-referenced cue coding.* Expert j's stated validity is converted to diagnosticity x_j = (logit v_j)^gamma, and the decision maker holds a reference endorser v* with diagnosticity theta = (logit v*)^gamma. A '1' is good news only to the extent that the endorsing expert is more discriminating than the reference; approval from a sub-reference endorser is faint praise and is genuinely counter-diagnostic. This single mechanism replaces both a unanimity premium and a uniform endorsement discount.

(2) *Diminishing sensitivity about the reference.* Subjective value is an S-shaped function of the deviation from the expectation reference, x~_j = theta + A*tanh(s*(x_j - theta)/A): sharp discrimination near v*, saturation on both branches. People categorise experts as 'better' or 'worse' than their reference endorser, so all clearly sub-reference endorsers are about equally damning.

(3) *Compressed endorsement numerosity.* The expectation charge is levied on the compressed magnitude of the endorsement surplus m: charge = theta*sign(m)*|m|^zeta, zeta < 1 (Weber-like). One extra endorsement is felt far more sharply than the fifth relative to the fourth.

(4) *Contrast-count normalisation that is background-blind but binding-conditional.* Sensitivity is divided by a term that grows with the number K of columns on which the panels actually disagree and is completely blind to the shared endorsement background: density ladders with fixed contrast and coherence must be FLAT. The cost of reconciling many simultaneous contrasts depends on whether the validity-to-column mapping has been successfully bound. When the display is validity-monotone, each discriminating column is read off cheaply and the cost is linear in K. When binding has failed (scrambled display / interior best expert), each additional discriminating column must be held in mind without a stable importance anchor, so cost grows *super*-linearly: denom = c0 + kappa*K*(1 + delta*mu^2*(K-1)). Scrambling a display therefore flattens *large* contrasts far more than small ones.

(5) *Readability / column binding.* When the display is monotone in validity (either direction) binding is trivial and the signed values are used intact; when the mapping is scrambled - especially with the best expert interior - values flatten toward their mean and a left-to-right reading gradient is recruited.

(6) *Population structure and determinacy.* One stationary rule for everyone, with broad continuous heterogeneity in determinacy beta, validity sensitivity gamma, crossover v*, reference-sensitivity (A, s), numerosity exponent zeta, normalisation (c0, kappa, delta), reading gradient rho and lapse eps. The added commitment of this version is that *attentional lapsing is rare*: essentially all of the observed regression toward chance is produced by the evidence transformation itself (expectation-referenced saturation and contrast-count normalisation), not by stimulus-independent guessing. Choices are therefore more determinate than a large-lapse account would allow, and the shrinkage of any given contrast must be predictable from its K, its endorsement surplus and the readability of the display - never from a free global guess rate. No discrete strategy switching, no learning, no consensus premium, no density dilution.

**Rationale:** ONE-LINE MINIMAL DIFF on the accepted iter-5 base (ERC-SB, loss 0.0505): the lapse band eps goes from [0.0, 0.14] to [0.0, 0.05]. Every other line of `predict`, every other parameter band, and the whole mechanism (signed expectation-referenced cue coding, S-shaped deviation value, compressed numerosity charge, binding-conditional super-linear contrast-count denominator, readability gate, no density term, no learning) is byte-identical to the accepted base. I also reverted the default value in the `.get("eps", ...)` fallback for consistency; that default is never used when parameters are supplied.

WHY THIS EDIT (and why exactly this size). I executed the critic's new PRIMARY recommendation, and I first checked it quantitatively rather than by hand-waving. Lapse is the one knob that acts as a *pure global scaling of every metric's deviation from its chance/zero point* (all eight metrics are linear in choice probabilities), so its optimal setting can be solved in closed form. Writing d_i for the iter-5 model deviations and r_i for the real deviations:
  d = [+0.2346, +0.0075, -0.1951, +0.1427, -0.0383, -0.0467, +0.1163, -0.2438]
  r = [+0.2533, +0.0100, -0.2081, +0.1587, -0.0133, -0.1500, +0.0675, -0.2950]
(Exps 1, 2, 4, 8 measured from 0.5; Exps 3, 5, 6, 7 are already difference scores.)
The least-squares optimal global scale is s* = sum(d.r)/sum(d.d) = 0.21003/0.19013 = 1.105. Six of the eight experiments individually demand s > 1 (Exp 8 wants 1.21, Exp 4 1.11, Exp 1 1.08, Exp 3 1.07, Exp 6 3.2, Exp 2 ~1.0); only Exps 5 and 7 want s < 1, and both have small absolute errors and small lever arms. So the model is uniformly *under-determinate*, exactly as the critic diagnosed, and lapse is the safe way to add determinacy (beta is not: it moves Exps 2 and 4 anti-correlated, and my hand-check of a +9% beta bump overshoots Exp 1 to 0.764, err +0.011, i.e. no net gain).

SIZE CHOICE. Lapse can only supply s up to 1/(1-E[eps_base]) = 1.075, below the s* = 1.105 optimum, so the reduction should be taken close to the maximum rather than half-heartedly. The critic's suggested [0.0, 0.07] gives s = 1.037 (about a third of the available move); [0.0, 0.05] gives s = 1.054, which is closer to the optimum while still leaving a real, heterogeneous lapse in the population (subject-level eps spread 0 to 0.05, which keeps between-subject variance in the metrics non-degenerate). Predicted sum-of-squared-error over the eight cells: 0.01709 -> 0.01560 (about a 9% reduction; the s* ceiling is 0.01502, so this captures roughly 70% of everything global determinacy can buy).

PREDICTED CELL VALUES AND THE CRITIC'S GUARD BANDS (all met):
  Exp 1: 0.7346 -> 0.746 (real 0.753; guard [0.74, 0.77]) OK
  Exp 2: 0.5075 -> 0.508 (real 0.510; guard [0.49, 0.53]) OK - Exp 2 sits at chance so it is structurally immune to eps, which is precisely why eps is safer here than beta
  Exp 3: -0.195 -> -0.206 (real -0.208; guard [-0.23, -0.20]) OK
  Exp 4: 0.643 -> 0.650 (real 0.659) OK
  Exp 5: -0.038 -> -0.040 (real -0.013; guard [-0.06, 0.01]) OK (small degradation, accepted)
  Exp 6: -0.047 -> -0.049 (real -0.150) - free, tiny gain; still at the in-family ceiling
  Exp 7: 0.116 -> 0.123 (real 0.068; guard <= 0.14) OK (small degradation, accepted)
  Exp 8: 0.256 -> 0.247 (real 0.205; guard [0.20, 0.25]) OK
Aggregate: five experiments improve (Exps 1, 3, 4, 6, 8, total ~+0.037 of absolute error recovered), two degrade by ~0.009 combined, one is inert. That satisfies the critic's acceptance guidance.

KNOBS DELIBERATELY LEFT ALONE (all previously gate-REJECTED or shown exhausted): zeta returned to / kept at the accepted [0.55, 0.85] and not touched again (my iter-6 push moved Exps 7/8 the wrong way); global kappa and the sublinear-K exponent (iter-1 reject); deeper vstar (iter-2 reject); rho in isolation (iter-2 reject); beta (anti-correlated Exps 2/4); and no shared-background/density term at any point - that is out of the arbiter's family and is exactly the term that makes pi_5 miss Exp 7 by +0.27 and Exp 8 by +0.29.

EXP 6 IS A CEILING - I verified the algebra rather than asserting it. Its U contrast pairs (11111 vs 11101) with (00010 vs 00000) and its V contrast pairs (11111 vs 01111) with (10000 vs 00000); in each pair the difference vector, K (=1) and the endorsement surplus (=+1) are identical, and only the shared background differs, so any background-blind model returns exactly 0 for both. Only the L contrast is reachable, and to reach the observed -0.15 the model would need L1-L4 = -0.45, which would require rho ~ 15 (three times the current band) - catastrophic for Exps 2 and 4. No parameter budget was spent there.

**Parameters:**
  - `gamma`: `[1.25, 1.75]`
  - `vstar`: `[0.80, 0.90]`
  - `beta`: `[0.40, 0.88]`
  - `alpha`: `[0.80, 1.15]`
  - `sl`: `[1.8, 2.8]`
  - `zeta`: `[0.55, 0.85]`
  - `c0`: `[0.25, 0.60]`
  - `kap`: `[0.45, 0.85]`
  - `delta`: `[0.65, 1.05]`
  - `mu0`: `[0.0, 0.10]`
  - `mu1`: `[1.5, 2.4]`
  - `mu2`: `[0.35, 0.90]`
  - `rho`: `[4.0, 5.8]`
  - `lam`: `[0.7, 1.3]`
  - `eps`: `[0.0, 0.05]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # ERC-SB: Expectation-Referenced Cue Coding with an S-shaped (saturating)
    #         deviation value, compressed endorsement numerosity and
    #         BINDING-CONDITIONAL contrast-count normalisation.
    #
    #   x_j     = logit(v_j) ** gamma                (raw diagnosticity)
    #   theta   = logit(v*) ** gamma                 (subjective endorsement expectation)
    #   x~_j    = theta + A*tanh(sl*(x_j-theta)/A)   (diminishing sensitivity about theta,
    #                                                 A = alpha*theta)
    #   mu      = readability gate (0 = validity-sorted display, 1 = unreadable)
    #   xf_j    = (1-mu)*x~_j + mu*mean(x~)          (flatten toward equal weighting)
    #   g_j     = ((n-j)/n)^lam  (centred)           (left-to-right reading gradient)
    #   w_j     = xf_j + rho*mu*g_j
    #   m       = sum(a) - sum(b)                    (endorsement surplus)
    #   num     = sum_j w_j (a_j-b_j) - theta*sign(m)*|m|^zeta   (compressed charge)
    #   K       = number of discriminating columns
    #   D       = num / (c0 + kap*K*(1 + delta*mu^2*(K-1)))      (super-linear only
    #                                                 when column binding has failed)
    #   p(A)    = (1-eps)*logistic(beta*D) + eps/2
    #
    # No density dilution, no coherence/unanimity premium, no learning.
    import numpy as np

    # ---------------- unpack the two rating vectors --------------------
    a = None
    b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            half = int(arr.shape[0] // 2)
            a = arr[:half].astype(float)
            b = arr[half:2 * half].astype(float)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            a = arr[0].astype(float)
            b = arr[1].astype(float)

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]

    # ---------------- validities ---------------------------------------
    val = parameters.get("validities", None)
    if val is None:
        v = np.linspace(0.90, 0.55, n)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---------------- parameters ---------------------------------------
    gamma = float(parameters.get("gamma", 1.5))
    vstar = float(np.clip(float(parameters.get("vstar", 0.855)), 0.51, 0.99))
    beta = float(parameters.get("beta", 0.62))
    alpha = float(np.clip(float(parameters.get("alpha", 1.0)), 0.05, 5.0))
    sl = float(max(float(parameters.get("sl", 2.3)), 0.05))
    zeta = float(np.clip(float(parameters.get("zeta", 0.70)), 0.05, 1.0))
    c0 = max(float(parameters.get("c0", 0.425)), 1e-3)
    kap = max(float(parameters.get("kap", 0.65)), 1e-3)
    delta = max(float(parameters.get("delta", 0.85)), 0.0)
    mu0 = float(np.clip(float(parameters.get("mu0", 0.05)), 0.0, 1.0))
    mu1 = float(parameters.get("mu1", 1.9))
    mu2 = float(parameters.get("mu2", 0.6))
    rho = float(parameters.get("rho", 4.9))
    lam = float(parameters.get("lam", 1.0))
    eps = float(np.clip(float(parameters.get("eps", 0.025)), 0.0, 1.0))

    # ---------------- diagnosticity and expectation reference -----------
    L = np.log(v / (1.0 - v))
    L = np.maximum(L, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.power(L, gamma)
    x = np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=0.0)

    Ls = np.log(vstar / (1.0 - vstar))
    Ls = max(Ls, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        theta = float(np.power(Ls, gamma))
    if not np.isfinite(theta) or theta <= 0.0:
        theta = float(np.mean(x)) if np.isfinite(np.mean(x)) else 1.0

    # ---- diminishing sensitivity about the expectation reference -------
    A = max(alpha * theta, 1e-6)
    with np.errstate(over="ignore", invalid="ignore"):
        x = theta + A * np.tanh(sl * (x - theta) / A)
    x = np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)

    # ---------------- readability of the validity->column mapping -------
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    tot = conc + disc
    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)
    incong = 1.0 - abs(tau)                      # 0 when display is monotone
    jmax = int(np.argmax(v))
    interior = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0

    gate = mu1 * incong + mu2 * interior
    gate = min(1.0, max(0.0, gate))
    mu = mu0 + (1.0 - mu0) * gate
    mu = float(np.clip(mu, 0.0, 1.0))

    # ---------------- flattening + positional gradient ------------------
    xbar = float(np.mean(x))
    xt = (1.0 - mu) * x + mu * xbar

    idx = np.arange(n, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        pos = np.power(np.maximum((float(n) - idx) / float(n), 1e-9), lam)
    pos = np.nan_to_num(pos, nan=0.0, posinf=1.0, neginf=0.0)
    g = pos - float(np.mean(pos))                # zero-mean reading gradient

    w = xt + rho * mu * g                        # column values (pre-expectation)

    # ---------------- evidence -----------------------------------------
    diff = a - b
    K = float(np.sum(np.abs(diff) > 1e-9))
    if K < 1.0:
        return np.array([0.5, 0.5])

    core = float(np.dot(w, diff))
    m = float(np.sum(a) - np.sum(b))             # endorsement surplus
    charge = theta * float(np.sign(m)) * (abs(m) ** zeta)
    num = core - charge

    # BINDING-CONDITIONAL contrast compression: linear in K when the
    # validity->column mapping is readable (mu ~ 0), super-linear when
    # binding has failed (mu ~ 1).  Still background-blind.
    bind_fail = mu * mu
    denom = c0 + kap * K * (1.0 + delta * bind_fail * max(K - 1.0, 0.0))
    if (not np.isfinite(denom)) or denom <= 1e-6:
        denom = 1e-6
    D = num / denom
    if not np.isfinite(D):
        D = 0.0

    # ---------------- logistic choice with lapse -------------------------
    z = np.array([beta * D, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if (not np.isfinite(tot)) or tot <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))
```
