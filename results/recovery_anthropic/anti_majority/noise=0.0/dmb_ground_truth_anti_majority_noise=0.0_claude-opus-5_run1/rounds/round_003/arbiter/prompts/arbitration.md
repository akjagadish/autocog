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
**PWA-DD: Position-Bound Weighted Additive integration with Faint-Praise discounting and Endorsement-Density Dilution.**

Choice in binary-cue environments is always *compensatory*: on every trial the decision maker reads all displayed columns, forms a single weighted difference, and never stops early. There is no lexicographic cascade and no coherence/unanimity premium. Three claims fix the rest of the mechanism.

(1) **Weights are a mixture of validity and display position, and the mixture is governed by how readable the validity-to-column mapping is.** Stated validities must be bound to *columns*. When the display is perfectly monotone in validity (ascending or descending) that binding is trivial and people use validity-derived weights w_val ∝ logit(v_j)^γ essentially intact (the residual positional leak μ0 is small and person-specific, and in some people is exactly zero). When the column order is not monotone in validity — and especially when the best expert sits in an interior column — binding fails and people fall back almost entirely on a purely positional gradient w_pos ∝ (n−j)^λ that decays left-to-right over displayed columns. The mixture weight μ = μ0 + (1−μ0)·min(1, μ1(1−|τ|) + μ2·[best cue interior]) is therefore an *environmental* quantity, not a per-experiment fudge: it is inert (μ = μ0) whenever columns are validity-sorted, and it predicts that scrambling a sorted display converts validity-driven behaviour into position-driven behaviour and that re-sorting a scrambled display restores it. Crucially the gate SATURATES: any display that is appreciably non-monotone is read essentially positionally, with no graded validity leak surviving.

(2) **Faint-praise endorsement discounting.** Each positive endorsement carries a small fixed credibility cost θ: an option that collects more '1's is slightly *less* attractive, all else equal. This is strictly monotone in endorsement count — there is no step at unanimity. It makes a lone weak endorsement no better (often slightly worse) than an empty panel, and it makes a 5/5 panel no better than a 4/5 panel by exactly the same amount, which is why 'high-coherence' and 'low-coherence' single-cue steps move together rather than apart. θ sits just at the subjective weight of the weakest experts — close enough that a lone weak endorsement is near-neutral to mildly counter-diagnostic rather than strongly aversive — while endorsement by a strong expert still wins comfortably.

(3) **Endorsement-density dilution of decision sensitivity.** The same absolute weighted difference is less decisive when both panels are densely endorsed. Evidence is divisively normalised by the mean endorsement density T = (Σa + Σb)/2n: D = [Σ_j w_j(a_j−b_j) − θ(Σa−Σb)] / (c0 + κ·T). The dilution has TWO separable constants: a floor c0 that sets how crisply *sparse* panels are compared, and a gain κ that sets how much extra mush dense panels accrue. The floor is *not* negligible: even near-empty panels are compared with bounded, distinctly sub-ceiling determinacy, which keeps lone-endorsement-versus-empty cells close to (rather than far below) chance and preserves the graded ladder structure of sparse comparisons. Heavily endorsed panels are compared mushily. Unlike a consensus premium, dilution is *direction-blind* — it never confers an advantage on the more coherent option, it only compresses whatever advantage exists — so it produces the observed shrinkage of a fixed one-cue advantage at high endorsement backgrounds without ever predicting a unanimity bonus.

p(A) = (1−ε)·logistic(βD) + ε/2. Everyone holds the same stationary rule (no feedback is given, so nothing is learned); heterogeneity is broad and continuous — in determinacy β, validity sensitivity γ, discount θ, positional decay λ, dilution (c0, κ), lapse ε and the readability coefficients — never discrete strategy switching. The theory therefore predicts substantial between-subject spread in every conflict metric, largest where margins are intermediate.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## THEORY 2 — pi_4
**Readability-limited validity weighting with a strict-unanimity consensus premium (RVW-DE+/U), with broad determinacy heterogeneity.**

A single stationary, fully compensatory integration rule underlies choice in all cue-based binary-feature environments; the apparent switching between 'take-the-best-like' and 'tallying-like' behaviour across environments is produced by modulators that are properties of the *display and the evidence pattern*, not of the strategy, plus wide between-person variation in decision determinacy.

(1) **Graded validity weighting.** Each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, w_j ∝ logit(v_j)^gamma, gamma spread over ~1.05–2.0 across people. At the population mean this is steep enough that a clearly best expert usually survives a coalition of three or four weak experts, yet weak enough that a near-peer plus a couple of moderate cues can overturn the top cue.

(2) **Binding readability, not position.** Stated validities must be bound to displayed columns. When the display order is *monotone* in validity (ascending OR descending) this binding is trivial and the validity weights are used essentially intact. When the order is scrambled, binding fails partially and the weights are flattened toward EQUAL weighting: w_eff = (1-kappa)·w + kappa/n, with kappa = k0 + k1·(1-|Kendall tau(column index, validity)|) + k2·[best expert not in an extreme column]. Direction-free: reversing a sorted display costs nothing, permuting it costs a lot.

(3) **Readability-gated reading-order fallback.** The left-to-right attention gradient is a fallback recruited only when validity-to-column binding fails: phi_eff = phi·(r0 + r1·kappa). It only *scales* weights, never re-sorts them.

(4) **Strict-unanimity consensus premium (dissent aversion).** The premium attaches to *consensus itself*, and consensus is an essentially all-or-none perceptual property: a panel is 'agreed' only when it is unanimous or one voice away from it. Subjects add a direction-blind premium rho·|2·mean(x)-1|^q with a HIGH exponent q (~5–11, person-specific), so a 5/5 or 0/5 panel earns almost the whole premium while a 4/5 or 1/5 panel earns almost none. The magnitude rho of this premium is substantial and heterogeneous (population mean ~0.39): an option unanimously rated 0 can be *preferred* to one carrying a single weak endorsement (sub-chance accuracy on one-endorsement-vs-nothing pairs), while options that are merely lopsided (4-of-5 endorsements) receive essentially no credibility bonus and therefore cannot dilute genuine weighted-evidence conflicts. The premium is a *unanimity* effect, not a mean-rating effect — that is the theory's sharpest falsifiable commitment (a strongly non-linear step at exactly 0 and n dissenters, not a smooth gradient in endorsement proportion).

(5) **Endorsement discount / pull.** Each positive endorsement carries a small credibility adjustment theta (faint-praise skepticism for most, mild endorsement-count pull for some), inert whenever the two options carry equal endorsement counts.

(6) **Population structure.** Everyone holds the same rule; heterogeneity is broad and continuous — in determinacy beta, in gamma, in lapse, in the modulator magnitudes rho, q, theta, phi and the readability coefficients. No discrete strategy switching.

Evidence for A: D = Σ_j w_eff_j (x_Aj - x_Bj) - theta·(Σx_A - Σx_B) + rho·(coh(x_A) - coh(x_B)); p(A) = (1-eps)·logistic(beta·D) + eps/2.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.93, 0.84, 0.74, 0.64, 0.55]

**Trial pairs (n=24):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 9: A=[1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 10: A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  trial 11: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 13: A=[0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0]
  trial 14: A=[0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1]
  trial 15: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 16: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  trial 17: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 18: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 19: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  trial 20: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0]
  trial 21: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 22: A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 23: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate PWA-DD (advocated) from RVW-DE+/U (competing). The two theories share graded validity weighting w \u221d logit(v)^gamma, so I deliberately remove the readability nuisance by displaying the five experts in strictly DESCENDING validity (.93,.84,.74,.64,.55): Kendall tau = -1 (incongruence 0) and the best expert sits in an extreme column, so PWA-DD's mixture gate is inert (mu = mu0 <= .14, essentially pure validity weights) and RVW's flattening is minimal (kappa = k0 <= .16, reading-order fallback phi_eff <= .18). Both models therefore compute nearly the SAME weighted core; every remaining prediction difference is attributable to the two rival post-integration mechanisms: PWA-DD's endorsement-DENSITY DILUTION (divide by c0 + kap*T, direction-blind, smooth and monotone in total endorsement density) plus a monotone faint-praise discount theta, versus RVW's STRICT-UNANIMITY consensus premium rho*|2m-1|^q with high q (a step at exactly 0 and n dissenters, direction-carrying).

Mid-parameter weights (gamma=1.5) are approx [.53,.27,.14,.055,.011] by column, so cue1 (.84) is a solidly weighted cue (w >> theta) and cue4 (.55) is a cue whose weight lies BELOW theta for the whole PWA-DD box (w4 <= .05 < theta_min = .07). 24 unique pairs (K=4, 96 trials); every base pair is mirrored so neither response key is systematically correct.

FAMILY S \u2014 STRONG-CUE LADDER OVER ENDORSEMENT BACKGROUND (pairs 1-10). The two options differ on cue1 (.84) and ONLY cue1; the shared background of 1s grows from k=0 to k=4, so the weighted-evidence term and the endorsement-count difference are IDENTICAL in all five cells and only the total endorsement density T (and, incidentally, panel unanimity at the two ends) changes. PWA-DD: D = (w1-theta)/(c0+kap*T) with T=(2k+1)/10, giving a smooth MONOTONE DECAY that never crosses chance: p(cue1-winner) ~ .87 / .71 / .64 / .60 / .58. RVW: D = w1 - theta + rho*(coh_A - coh_B); coherence is ~0 for k=1,2,3 but -0.98*rho at k=0 (loser panel unanimous-zero) and +0.98*rho at k=4 (winner panel unanimous), giving a NON-MONOTONE step profile ~ .27 / .85 / .85 / .86 / .99. This is a full crossover: at k=0 PWA-DD is at its HIGHEST and RVW is SUB-CHANCE; at k=4 PWA-DD is at its LOWEST and RVW is at ceiling. No re-weighting of cues can mimic either pattern.

FAMILY W \u2014 WEAK-CUE LADDER (pairs 11-16, k=0,2,4, discriminating cue4 = .55). Here w4 < theta for every PWA-DD parameter set, so PWA-DD predicts mildly SUB-CHANCE choice of the endorsed option in all three cells, with the magnitude shrinking as density rises: ~ .26 / .42 / .47 (the c0 floor keeps the sparse cell close to, not far below, chance). RVW predicts the unanimity step again: ~ .12 / .52 / .98. The k=4 cell ([1,1,1,1,1] vs [1,1,1,1,0]) is the sharpest single cell in the design: PWA-DD says just below chance, RVW says near ceiling.

FAMILY T \u2014 PURE DENSITY LADDER WITH COUNTS AND COHERENCE EXACTLY MATCHED (pairs 17-22). A wins cue0 (.93), B wins cue1 (.84), and both options carry an identical number of shared background 1s (0, 1 or 3), so endorsement counts are equal (theta inert), the coherence terms cancel EXACTLY (1/5 vs 1/5, 2/5 vs 2/5, 4/5 vs 4/5), and the weighted core is identical across the three cells. RVW must therefore predict the SAME probability in all three (~.85 flat) \u2014 it has no mechanism that reads total density. PWA-DD predicts a strong monotone compression, ~ .85 / .75 / .63, purely from divisive normalisation by T. This family is the cleanest possible test of density dilution, with direction, counts and coherence all pinned.

FAMILY D \u2014 DOMINANCE ANCHOR (pairs 23-24): [1,1,1,1,1] vs [0,0,0,0,0]. Both panels are unanimous so RVW's premium cancels exactly and PWA-DD's discount subtracts 5*theta while T = .5 dilutes; PWA-DD ~ .83, RVW ~ .96. This calibrates beta and eps for both models so the other families are out-of-sample predictions rather than free fits.

DECISION RULE. If PWA-DD is right: Family S is monotone DECREASING and stays above chance everywhere (no dip at k=0, no jump at k=4); Family W is uniformly at-or-below chance and rises toward .5 with density; Family T shows a clear ~20-point monotone drop from sparse to dense with everything else held constant; and no cell shows a discontinuity at the unanimity boundaries. If RVW-DE+/U is right: Family S and Family W both show a sub-chance reversal at k=0, a flat middle plateau, and a large jump at k=4 (steps located at exactly 0 and n dissenters), while Family T is completely FLAT across the density ladder. Between-subject spread should be largest in Family W's middle cell and Family S's k=1..3 plateau under either account, but only PWA-DD predicts spread ordered by margin rather than by unanimity.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
This isolates the single sharpest structural disagreement between PWA-DD and RVW-DE+/U: what happens at the unanimity boundary. Two cell families are contrasted, both of which are one-feature differences so the weighted-evidence core is essentially identical across the two models.

Group P: [0,1,0,0,0] vs [0,0,0,0,0] (lone endorsement by the .84 expert against a completely unendorsed panel). PWA-DD has NO consensus premium; its only relevant machinery is the small faint-praise cost theta (< w1 = .27) and the sparse-density denominator c0 + kap*0.1, which is small, so the weighted advantage is amplified: p(pick endorsed) ~ .85-.90. RVW-DE+/U, by contrast, gives the all-zero panel the FULL strict-unanimity premium rho (~.19-.59), which exceeds w1 - theta, so it predicts a clear sub-chance reversal, p ~ .15-.35.

Group N: [1,1,1,1,1] vs [1,0,1,1,1] and [1,1,1,1,1] vs [1,1,1,1,0] (a unanimous panel against a 4/5 panel). Here PWA-DD's density dilution divides by c0 + kap*0.9, crushing the same weighted advantage to near-chance (~.60 and ~.44, mean ~.52; the weak-cue cell is actually below chance because w4 < theta). RVW-DE+/U hands the unanimous option the whole premium, driving both cells to ceiling (~.98).

The difference (Group P rate minus Group N rate) therefore points in OPPOSITE directions for the two theories: about +0.35 for PWA-DD versus about -0.71 for RVW, a ~1.06 gap. Each term aggregates 8 and 16 trials per subject respectively, and both theories put each cell far from 0.5 or (for PWA's Group N) tightly near chance, so per-subject binomial + parameter variance stays modest (~0.03-0.06), giving a large Welch t at N=25. Crucially the metric is not trivially true of my theory: it would come out near zero or positive-and-small for any purely additive model without density dilution, and it would only reverse sign if a genuine unanimity step exists in human data.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def key(x):
        try:
            return ''.join([str(int(round(float(v)))) for v in list(x)])
        except Exception:
            return ''

    P_key = frozenset(['01000', '00000'])          # lone strong endorsement vs empty panel
    N_keys = [frozenset(['11111', '10111']),        # unanimous panel vs one dissent (strong cue)
              frozenset(['11111', '11110'])]        # unanimous panel vs one dissent (weak cue)

    p_hits = []
    n_hits = []

    for _, row in data.iterrows():
        ka = key(row['option_a_ratings'])
        kb = key(row['option_b_ratings'])
        if len(ka) == 0 or len(kb) == 0 or ka == kb:
            continue
        pairset = frozenset([ka, kb])
        sa = ka.count('1')
        sb = kb.count('1')
        if sa == sb:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        # did the subject pick the option carrying the extra endorsement?
        if sa > sb:
            chose_endorsed = 1.0 if resp == 0 else 0.0
        else:
            chose_endorsed = 1.0 if resp == 1 else 0.0

        if pairset == P_key:
            p_hits.append(chose_endorsed)
        else:
            for nk in N_keys:
                if pairset == nk:
                    n_hits.append(chose_endorsed)
                    break

    mp = float(np.mean(p_hits)) if len(p_hits) > 0 else 0.5
    mn = float(np.mean(n_hits)) if len(n_hits) > 0 else 0.5
    return mp - mn


### RESULTS
- Predicted under pi_5 (simulated): 0.3337 (var=0.0409)
- Predicted under pi_4 (simulated): -0.5475 (var=0.0849)
- Observed on real data: 0.0675 (var=0.0281)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.55, 0.63, 0.72, 0.83, 0.94]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 3: A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0]
  trial 7: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  trial 8: A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 12: A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  trial 13: A=[1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1]
  trial 14: A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  trial 15: A=[0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 16: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 18: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 19: A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 20: A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 21: A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 22: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 23: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate RVW-DE+/U (advocated) from PWA-DD (competing). The two theories agree on graded validity weighting w \u221d logit(v)^gamma, so I remove all readability nuisance by displaying the five experts in strictly ASCENDING validity (.55,.63,.72,.83,.94): Kendall tau = +1 (incongruence 0) and the best expert sits in an extreme (rightmost) column, so RVW's flattening gate is inert (kappa = k0 <= .16, phi_eff <= .20) and PWA-DD's positional gate is inert (mu = mu0 <= .14). Both theories therefore compute nearly identical weighted cores; every remaining prediction difference is attributable to the rival post-integration mechanisms: RVW's STRICT-UNANIMITY consensus premium rho*|2m-1|^q (a direction-CARRYING step located at exactly 0 and n dissenters) versus PWA-DD's ENDORSEMENT-DENSITY DILUTION (divide by c0+kap*T, direction-BLIND and smooth) plus a monotone faint-praise discount theta.

Mid-parameter weights (gamma=1.5) by column: [.011,.049,.115,.251,.574]. So col3 (v=.83, w~.25) is safely ABOVE PWA-DD's theta box (.07-.145) for every subject, while col1 (v=.63, w~.03-.08) is at or below theta for essentially every subject. 12 base pairs, each mirrored (A/B swapped) so neither key is systematically correct: 24 unique pairs, K = 4, 96 trials.

FAMILY U \u2014 UNANIMITY LADDER WITH WEIGHTED EVIDENCE AND ENDORSEMENT-COUNT DIFFERENCE HELD EXACTLY CONSTANT (pairs 1-8). The two options differ on col3 and ONLY col3; a shared background of 1s grows k = 0, 1, 2, 4, so the core (w3) and the count difference (+1) are identical in all four cells and only total density T (=.1,.3,.5,.9) and panel unanimity change. RVW: coherence is ~0 for k=1,2 (2/5 vs 1/5, 3/5 vs 2/5 give |2m-1|^q ~ 0 at q>=5), but at k=0 the LOSER panel is unanimous-zero (-rho) and at k=4 the WINNER panel is unanimous-one (+rho): predicted p(col3-winner) ~ .27 / .83 / .83 / .99 \u2014 a flat middle plateau with a sub-chance dip at k=0 and a jump at k=4, i.e. steps only at exactly 0 and n dissenters. PWA-DD: D = (w3-theta)/(c0+kap*T) gives a smooth MONOTONE DECAY that never crosses chance, ~ .89 / .71 / .64 / .58. The two profiles cross twice; no re-weighting of cues can produce either.

FAMILY W \u2014 WEAK-CUE (col1) VERSION AT THE TWO EXTREMES (pairs 9-12). W1 ([0,1,0,0,0] vs all-zeros): both models sub-chance but very differently (RVW ~.09 from the full unanimity penalty; PWA ~.30 from w1-theta scaled by the c0 floor). W2 ([1,1,1,1,1] vs [1,0,1,1,1]) is the sharpest single cell in the design: RVW says ~.98 (winner unanimous, premium swamps the tiny cue) while PWA says just BELOW chance (~.47), because w1 < theta and the dense panels are maximally diluted. Comparing W2 with U3 also separates them: PWA predicts U3 (.58) > W2 (.47) with W2 sub-chance; RVW predicts both at ceiling and nearly equal.

FAMILY C \u2014 POLARITY SYMMETRY OF THE COHERENCE EFFECT (pairs 13-16). C1 ([1,1,1,1,1] vs [0,0,1,1,1]) and its exact bit-complement C2 ([0,0,0,0,0] vs [1,1,0,0,0]) pit two weak-cue endorsements against panel purity. RVW's premium is direction-blind, so it must favour the COHERENT option in both cases with similar magnitude (~.97 and ~.94). PWA-DD's discount and dilution always favour the SPARSER option, so it predicts a strong asymmetry: ~.40 in C1 (choosing the sparse 3-endorsement panel against the unanimous one) versus ~.80 in C2. C1 is a clean directional reversal between the theories.

FAMILY T \u2014 PURE DENSITY LADDER WITH COUNTS AND COHERENCE EXACTLY MATCHED (pairs 17-22). A wins col4 (.94), B wins col3 (.83), and both panels carry an identical number of shared background 1s (0, 1, 3), so counts are equal (theta inert), coherence cancels EXACTLY (1/5 vs 1/5, 2/5 vs 2/5, 4/5 vs 4/5) and the weighted core is identical across cells. RVW has no mechanism reading total density and must predict a FLAT ~.91 across all three; PWA-DD predicts a ~25-point monotone compression, ~.95 / .83 / .70. This is the cleanest possible test of density dilution with direction, counts and coherence pinned.

FAMILY D \u2014 DOMINANCE ANCHOR (pairs 23-24): [1,1,1,1,1] vs [0,0,0,0,0]. Both panels unanimous so RVW's premium cancels exactly (p ~ .99) while PWA-DD pays 5*theta and divides by c0+kap*0.5 (p ~ .87). This pins beta and eps for both models, making every other family an out-of-sample prediction rather than a free fit.

DECISION RULE. If RVW-DE+/U is right: Family U shows a sub-chance dip at k=0, a flat plateau at k=1-2, and a jump to ceiling at k=4 (discontinuities located only at 0 and n dissenters); W2 is near ceiling; C1 ~ C2 (coherent option preferred regardless of polarity); Family T is completely flat across the density ladder. If PWA-DD is right: Family U declines smoothly and monotonically with density and never crosses chance; W2 sits at or just below chance; C1 and C2 are strongly asymmetric with the sparser panel always favoured; and Family T shows a large monotone drop with everything else held constant.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
This isolates the design cells (base pairs 11/12 and 13/14) where a unanimous 5-of-5 panel is pitted against a nearly-as-dense panel whose only zeros sit on the WEAKEST experts (validities .55/.63, subjective weight ~.03-.11). Here the weighted-evidence core is tiny and identical for both theories, the endorsement-count difference is small, and the panels are both dense (T ~ .8-.9), so the two rival post-integration mechanisms act in opposite directions:

- RVW-DE+/U: the all-ones panel earns nearly the whole strict-unanimity premium rho (mean ~.39, |2m-1|^q with high q makes a 4/5 or 3/5 panel earn essentially nothing). Since rho dwarfs the weak-cue core and the small theta cost for every parameter draw in the stated ranges, D is positive for ALL subjects and the unanimous panel is chosen well above chance (~.85-.98).
- PWA-DD: there is no consensus premium; instead the faint-praise discount theta (.07-.145 per extra endorsement) exceeds the weak-cue weight, and the whole (already negative) numerator is divided by c0+kap*T with T near .85, pushing choices to just BELOW chance (~.40-.48) for essentially every parameter draw.

So the metric should land near ~.9 under the advocated theory and near ~.45 under the competitor: a ~.45 gap with a directional reversal, and tight within-theory spread because the sign of D is parameter-robust in both models. 16 trials per subject contribute, keeping binomial noise modest. It is not trivially true of RVW: it is a behavioural preference for a unanimous panel over an almost-equally-endorsed one, which any compensatory weighting account (including RVW with rho=0) would place at or below chance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = 0
    tot = 0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = [int(x) for x in a_raw]
            b = [int(x) for x in b_raw]
        except Exception:
            continue
        n = min(len(a), len(b))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        a_all = all(x == 1 for x in a)
        b_all = all(x == 1 for x in b)
        # exactly one option must be the unanimous all-ones panel
        if a_all == b_all:
            continue
        other = b if a_all else a
        zeros = [i for i, x in enumerate(other) if x == 0]
        if len(zeros) == 0:
            continue
        # the dissenting (zero) columns must all be WEAK experts:
        # with validities displayed in ascending order, the weak experts
        # occupy the leftmost columns.
        cutoff = max(1, n - 3)
        if any(i >= cutoff for i in zeros):
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        chose_unanimous = (r == 0) if a_all else (r == 1)
        tot += 1
        if chose_unanimous:
            hits += 1
    if tot == 0:
        return 0.5
    return float(hits) / float(tot)

### RESULTS
- Predicted under pi_5 (simulated): 0.4938 (var=0.0166)
- Predicted under pi_4 (simulated): 0.8800 (var=0.0125)
- Observed on real data: 0.2050 (var=0.0274)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # Take-The-Best winner: first discriminating cue in validity order.
        # Feature columns are already ordered by descending validity in this design.
        ttb_winner = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        # Tallying winner: majority of feature-wise wins (ties excluded).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        if tally_winner == ttb_winner:
            continue  # not a conflict trial
        resp = row['response']
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        hits += 1.0 if int(resp) == ttb_winner else 0.0
        n += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.7533 (var=0.0228)
**Predicted under pi_5:** 0.7683 (var=0.0072)
**Predicted under pi_4:** 0.7196 (var=0.0166)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Fixed design validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65
    val = np.array([0.93, 0.58, 0.86, 0.51, 0.74, 0.65], dtype=float)

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape[0] != b.shape[0]:
            continue
        if a.shape[0] == val.shape[0]:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(a.shape[0])

        tally = int(np.sum(a > b)) - int(np.sum(b > a))

        winner = None
        for j in order:
            j = int(j)
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no cue discriminates: uninformative for both models

        ttb_sign = 1 if winner == 0 else -1
        # Keep only trials where the tally does NOT favour the TTB winner:
        # either the tally is exactly tied (Family D) or it points the
        # other way (Family E, directional conflict).
        if tally * ttb_sign > 0:
            continue

        n += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == winner:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5100 (var=0.0136)
**Predicted under pi_5:** 0.4938 (var=0.0059)
**Predicted under pi_4:** 0.5204 (var=0.0106)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = [0.55, 0.62, 0.70, 0.85, 0.93]

    def key(v):
        return ''.join([str(int(round(float(x)))) for x in v])

    # conflict families: TTB (top-validity cue) and a left-to-right positional
    # weighting scheme point to OPPOSITE options
    conflict_keys = {
        '10000|00001', '00001|10000',      # family A
        '01000|00010', '00010|01000',      # family B
        '11000|00011', '00011|11000',      # family E
        '00100|11000', '11000|00100',      # family F
    }
    # agreement families: both accounts favour the same option
    agree_keys = {
        '11111|00000', '00000|11111',      # family D (dominance)
        '00001|00000', '00000|00001',      # family G
        '10000|00000', '00000|10000',      # family H
    }

    def ttb_winner(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    n_conf = 0
    hit_conf = 0
    n_agree = 0
    hit_agree = 0

    for _, row in data.iterrows():
        try:
            a = [int(round(float(x))) for x in row['option_a_ratings']]
            b = [int(round(float(x))) for x in row['option_b_ratings']]
            resp = int(row['response'])
        except Exception:
            continue
        n = len(a)
        if len(b) != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fallback: assume columns are displayed in ascending validity
            v = np.linspace(0.55, 0.93, n)
        order = list(np.argsort(-v, kind='stable'))
        w = ttb_winner(a, b, order)
        if w < 0:
            continue
        k = key(a) + '|' + key(b)
        if k in conflict_keys:
            n_conf += 1
            hit_conf += 1 if resp == w else 0
        elif k in agree_keys:
            n_agree += 1
            hit_agree += 1 if resp == w else 0

    if n_conf == 0 or n_agree == 0:
        return 0.0

    return float(hit_agree) / n_agree - float(hit_conf) / n_conf

```

**Observed (real) value:** -0.2081 (var=0.0368)
**Predicted under pi_5:** -0.1193 (var=0.0132)
**Predicted under pi_4:** -0.2069 (var=0.0176)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (display order): col0..col5
    v_default = np.array([0.60, 0.93, 0.55, 0.86, 0.68, 0.78], dtype=float)

    def build(n):
        v = v_default.copy()
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
        v = np.clip(v, 0.5 + 1e-6, 1 - 1e-6)
        L = np.log(v / (1.0 - v))
        w = np.power(L, 1.5)
        wpos = np.sort(w)[::-1]
        weff = 0.2 * w + 0.8 * wpos          # lam = 0.8, mid of advocated box
        s = weff.sum()
        weff = weff / s if s > 0 else np.ones(n) / n
        cascade = list(np.argsort(-v, kind="stable"))
        return weff, cascade

    cache = {}
    num = 0.0
    den = 0.0

    for a_raw, b_raw, resp in zip(data["option_a_ratings"],
                                  data["option_b_ratings"],
                                  data["response"]):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        if n not in cache:
            cache[n] = build(n)
        weff, cascade = cache[n]

        d = float(np.dot(weff, a - b))

        winner = None
        for j in cascade:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue
        if abs(d) < 0.20:                    # ambiguous / near-cancelling for WADD-pos
            continue
        pos_choice = 0 if d > 0 else 1
        if pos_choice == winner:             # models agree -> uninformative
            continue

        den += 1.0
        try:
            r = int(resp)
        except Exception:
            continue
        num += 1.0 if r == pos_choice else 0.0

    if den == 0:
        return 0.5
    return float(num / den)
```

**Observed (real) value:** 0.6587 (var=0.0379)
**Predicted under pi_5:** 0.6767 (var=0.0145)
**Predicted under pi_4:** 0.6633 (var=0.0149)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def key(x):
        return tuple(int(v) for v in list(x))

    A = [key(x) for x in data['option_a_ratings']]
    B = [key(x) for x in data['option_b_ratings']]
    R = [int(r) for r in data['response']]

    ones5 = (1, 1, 1, 1, 1)
    four = (1, 1, 1, 1, 0)
    zeros = (0, 0, 0, 0, 0)
    weak_single = [(0, 0, 1, 0, 0), (0, 0, 0, 0, 1)]

    hi_n = 0
    hi_d = 0
    lo_n = 0
    lo_d = 0

    for a, b, r in zip(A, B, R):
        if len(a) != 5 or len(b) != 5:
            continue
        # HIGH-coherence cell: 5/5 unanimous panel vs 4/5 panel, differ on cue5 only.
        if a == ones5 and b == four:
            hi_d += 1
            if r == 0:
                hi_n += 1
        elif b == ones5 and a == four:
            hi_d += 1
            if r == 1:
                hi_n += 1
        # LOW-coherence cell: single weak endorsement vs unanimous all-zero panel.
        elif (a in weak_single) and b == zeros:
            lo_d += 1
            if r == 0:
                lo_n += 1
        elif (b in weak_single) and a == zeros:
            lo_d += 1
            if r == 1:
                lo_n += 1

    if hi_d == 0 or lo_d == 0:
        return float('nan')

    p_hi = hi_n / float(hi_d)
    p_lo = lo_n / float(lo_d)
    return float(p_hi - p_lo)

```

**Observed (real) value:** -0.0133 (var=0.0143)
**Predicted under pi_5:** -0.0008 (var=0.0364)
**Predicted under pi_4:** 0.7083 (var=0.0508)

### Experiment 8
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    df = data
    if len(df) == 0:
        return 0.0

    def as_tup(x):
        return tuple(int(v) for v in x)

    A = df['option_a_ratings'].apply(as_tup).values
    B = df['option_b_ratings'].apply(as_tup).values
    R = df['response'].values

    def frac_choose(target, other):
        """Fraction of trials with pair {target, other} on which the
        subject picked `target`. Collapses across A/B display order."""
        num = 0
        den = 0
        for a, b, r in zip(A, B, R):
            if a == target and b == other:
                den += 1
                if int(r) == 0:
                    num += 1
            elif a == other and b == target:
                den += 1
                if int(r) == 1:
                    num += 1
        if den == 0:
            return None
        return float(num) / float(den)

    # ---- cells -------------------------------------------------------
    # Family L (coalition ladder): does the lone top-validity (col2)
    # endorsement keep winning as the opposing coalition grows?
    L1 = frac_choose((0, 0, 1, 0, 0), (1, 0, 0, 0, 0))
    L4 = frac_choose((0, 0, 1, 0, 0), (1, 1, 0, 1, 1))

    # Family U (unanimity step, weighted evidence pinned to col3 only):
    # winner-unanimous cell vs loser-unanimous cell.
    U4 = frac_choose((1, 1, 1, 1, 1), (1, 1, 1, 0, 1))
    U1 = frac_choose((0, 0, 0, 1, 0), (0, 0, 0, 0, 0))

    # Family V (same step at a heavier column, col0).
    V2 = frac_choose((1, 1, 1, 1, 1), (0, 1, 1, 1, 1))
    V1 = frac_choose((1, 0, 0, 0, 0), (0, 0, 0, 0, 0))

    contrasts = []
    if L1 is not None and L4 is not None:
        contrasts.append(L1 - L4)
    if U4 is not None and U1 is not None:
        contrasts.append(U4 - U1)
    if V2 is not None and V1 is not None:
        contrasts.append(V2 - V1)

    if len(contrasts) == 0:
        return 0.0
    return float(np.mean(contrasts))

```

**Observed (real) value:** -0.1500 (var=0.0125)
**Predicted under pi_5:** -0.1183 (var=0.0157)
**Predicted under pi_4:** 0.4600 (var=0.0383)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments the two theories are not close. pi_4's signature mechanism \\u2014 a strict-unanimity consensus premium \\u2014 makes a directional commitment (unanimous panels get a large bonus) that the real data contradict every time it is put under load. Exp 7: pi_4 predicted +0.71, observed -0.013 (pi_5: -0.001, essentially exact). Exp 8: pi_4 predicted +0.46, observed -0.15 (pi_5: -0.118, close). Exp 2 (pi_4's own experiment, designed to be its best case): pi_4 predicted 0.88 choice of the unanimous 5/5 panel; subjects chose it only 0.205 of the time \\u2014 a near-total reversal; pi_5 predicted 0.494, wrong in magnitude but at least on the correct side of the unanimity boundary. Exp 1: pi_4 predicted -0.55, observed +0.0675, pi_5 predicted +0.334 (right sign, too large). pi_4 wins only Exp 5 (-0.207 vs observed -0.208, pi_5 -0.119), and there the winning behaviour \\u2014 relative preference for the all-zero panel over a lone weak endorsement \\u2014 is equally producible by a monotone faint-praise discount, so it does not vindicate the unanimity step; it only shows that pi_5's discount is too small. Exps 3, 4, 6 (validity-conflict and scrambled-display experiments) are fit adequately by both, so they do not discriminate. Conclusion: the *unanimity premium* is dead \\u2014 humans in this domain show anti-endorsement / anti-unanimity behaviour, not a consensus bonus. pi_5's family of mechanisms (graded validity weights + monotone endorsement discount + readability/position fallback) is qualitatively right everywhere. But pi_5 has its own systematic error: it consistently under-shoots the *magnitude* of the anti-endorsement effect. Its density-dilution term (divide by c0 + kap*T) drags dense-panel comparisons toward chance, so it predicted 0.494 where the data show 0.205 (Exp 2), and it predicted -0.119 where the data show -0.208 (Exp 5) and +0.334 where the data show +0.068 (Exp 1). The data say the discount survives, undiluted and large, even in densely endorsed panels: subjects prefer a 4/5 panel to a 5/5 panel by ~4:1 when the dissent lands on a weak expert. That is a strong, sharp signal that pi_5's dilution mechanism (rather than its discount) is the wrong part. So the right move is to replace pi_4 with a theory that keeps everything the data support (validity-graded compensatory weighting, readability/position fallback, no consensus premium) but replaces both the unanimity premium AND density dilution with a mechanism that makes weak-expert endorsements genuinely negative and preserves discriminability in dense panels \\u2014 a strong competitor to pi_5 on exactly the cells where pi_5 fails.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_4 entirely. Its strict-unanimity consensus premium is falsified in four independent experiments (1, 2, 7, 8) with large, sign-reversed errors; no re-parameterisation can save it, because the premium is direction-carrying and the data show the *opposite* direction at both unanimity boundaries.\n\nProposed replacement \\u2014 'Expectation-Referenced Cue Coding (ERC)': choice is compensatory and stationary, but each displayed rating is scored relative to a subjective endorsement threshold rather than as pure positive evidence.\n\n(1) Signed cue values: s_j = c*(logit v_j)^gamma - theta, with theta calibrated so the crossover validity v* sits around .65-.78 (person-specific). A '1' from a high-validity expert is positive evidence; a '1' from a low-validity expert is *negative* ('faint praise is damning' \\u2014 an indiscriminate endorser's approval signals a mediocre product). D = sum_j s_j (a_j - b_j). This is a single mechanism that replaces both pi_4's rho-premium and pi_5's separate theta term.\n\n(2) NO endorsement-density dilution. Instead, if any normalisation is needed, divide by a *contrast count* term (c0 + kap * number of discriminating columns), which is blind to the shared endorsement background. This is the sharp point of departure from pi_5: it predicts that 5/5-vs-4/5 (weak dissent) is strongly sub-chance (~.20, as observed in Exp 2) rather than near-chance, that single-endorsement-vs-empty and 5/5-vs-4/5 cells sit at similar depths below chance (Exp 7 diff ~0, Exp 1 diff ~0), and that pi_5's 'Family T' density ladders (matched counts, matched coherence, growing shared background) should be FLAT, not compressed \\u2014 an immediately testable dissociation.\n\n(3) Retain a readability/column-binding module, since Exps 4 and 6 require it: when the display is monotone in validity the validity-derived signed weights are used intact; when it is scrambled (or the best expert is interior) the weights are flattened toward equal weighting and a left-to-right positional gradient is recruited. Keep this direction-free (ascending or descending both cheap).\n\n(4) Broad continuous heterogeneity in beta, gamma, theta (i.e. in the crossover validity v*), the readability coefficients and lapse; no discrete strategy switching, no learning (no feedback given).\n\nParameter ranges should be chosen so that: Exp 2 metric lands near 0.20-0.30; Exp 1 metric near 0.0-0.15; Exp 7 near 0; Exp 8 near -0.15; Exp 5 near -0.21; Exp 3 near 0.75; Exp 4 near 0.51; Exp 6 near 0.66. The next discriminating experiment should pit ERC against pi_5 on matched-count / matched-coherence density ladders (flat vs compressed) and on cells that vary which expert supplies the extra endorsement (ERC predicts a sign flip as v crosses v*; pi_5 predicts only a magnitude change scaled by the shared background)."
}
```

## Usage

```json
{
  "input_tokens": 28732,
  "output_tokens": 7908
}
```
