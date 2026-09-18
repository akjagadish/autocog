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
**Description:** **Readability-limited validity weighting with a strict-unanimity consensus premium (RVW-DE+/U), with broad determinacy heterogeneity.**

A single stationary, fully compensatory integration rule underlies choice in all cue-based binary-feature environments; the apparent switching between 'take-the-best-like' and 'tallying-like' behaviour across environments is produced by modulators that are properties of the *display and the evidence pattern*, not of the strategy, plus wide between-person variation in decision determinacy.

(1) **Graded validity weighting.** Each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, w_j ∝ logit(v_j)^gamma, gamma spread over ~1.05–2.0 across people. At the population mean this is steep enough that a clearly best expert usually survives a coalition of three or four weak experts, yet weak enough that a near-peer plus a couple of moderate cues can overturn the top cue.

(2) **Binding readability, not position.** Stated validities must be bound to displayed columns. When the display order is *monotone* in validity (ascending OR descending) this binding is trivial and the validity weights are used essentially intact. When the order is scrambled, binding fails partially and the weights are flattened toward EQUAL weighting: w_eff = (1-kappa)·w + kappa/n, with kappa = k0 + k1·(1-|Kendall tau(column index, validity)|) + k2·[best expert not in an extreme column]. Direction-free: reversing a sorted display costs nothing, permuting it costs a lot.

(3) **Readability-gated reading-order fallback.** The left-to-right attention gradient is a fallback recruited only when validity-to-column binding fails: phi_eff = phi·(r0 + r1·kappa). It only *scales* weights, never re-sorts them.

(4) **Strict-unanimity consensus premium (dissent aversion).** The premium attaches to *consensus itself*, and consensus is an essentially all-or-none perceptual property: a panel is 'agreed' only when it is unanimous or one voice away from it. Subjects add a direction-blind premium rho·|2·mean(x)-1|^q with a HIGH exponent q (~5–11, person-specific), so a 5/5 or 0/5 panel earns almost the whole premium while a 4/5 or 1/5 panel earns almost none. The magnitude rho of this premium is substantial and heterogeneous (population mean ~0.39): an option unanimously rated 0 can be *preferred* to one carrying a single weak endorsement (sub-chance accuracy on one-endorsement-vs-nothing pairs), while options that are merely lopsided (4-of-5 endorsements) receive essentially no credibility bonus and therefore cannot dilute genuine weighted-evidence conflicts. The premium is a *unanimity* effect, not a mean-rating effect — that is the theory's sharpest falsifiable commitment (a strongly non-linear step at exactly 0 and n dissenters, not a smooth gradient in endorsement proportion).

(5) **Endorsement discount / pull.** Each positive endorsement carries a small credibility adjustment theta (faint-praise skepticism for most, mild endorsement-count pull for some), inert whenever the two options carry equal endorsement counts.

(6) **Population structure.** Everyone holds the same rule; heterogeneity is broad and continuous — in determinacy beta, in gamma, in lapse, in the modulator magnitudes rho, q, theta, phi and the readability coefficients. No discrete strategy switching.

Evidence for A: D = Σ_j w_eff_j (x_Aj - x_Bj) - theta·(Σx_A - Σx_B) + rho·(coh(x_A) - coh(x_B)); p(A) = (1-eps)·logistic(beta·D) + eps/2.

**Parameters:**
- gamma: [1.05, 1.95]
- beta: [1.2, 13.0]
- eps: [0.0, 0.20]
- k0: [0.02, 0.16]
- k1: [0.55, 1.00]
- k2: [0.10, 0.60]
- phi: [0.02, 0.38]
- theta: [-0.03, 0.07]
- rho: [0.19, 0.59]
- q: [5.0, 11.0]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
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


## COMPETING THEORY
**Description:** **PWA-DD: Position-Bound Weighted Additive integration with Faint-Praise discounting and Endorsement-Density Dilution.**

Choice in binary-cue environments is always *compensatory*: on every trial the decision maker reads all displayed columns, forms a single weighted difference, and never stops early. There is no lexicographic cascade and no coherence/unanimity premium. Three claims fix the rest of the mechanism.

(1) **Weights are a mixture of validity and display position, and the mixture is governed by how readable the validity-to-column mapping is.** Stated validities must be bound to *columns*. When the display is perfectly monotone in validity (ascending or descending) that binding is trivial and people use validity-derived weights w_val ∝ logit(v_j)^γ essentially intact (the residual positional leak μ0 is small and person-specific, and in some people is exactly zero). When the column order is not monotone in validity — and especially when the best expert sits in an interior column — binding fails and people fall back almost entirely on a purely positional gradient w_pos ∝ (n−j)^λ that decays left-to-right over displayed columns. The mixture weight μ = μ0 + (1−μ0)·min(1, μ1(1−|τ|) + μ2·[best cue interior]) is therefore an *environmental* quantity, not a per-experiment fudge: it is inert (μ = μ0) whenever columns are validity-sorted, and it predicts that scrambling a sorted display converts validity-driven behaviour into position-driven behaviour and that re-sorting a scrambled display restores it. Crucially the gate SATURATES: any display that is appreciably non-monotone is read essentially positionally, with no graded validity leak surviving.

(2) **Faint-praise endorsement discounting.** Each positive endorsement carries a small fixed credibility cost θ: an option that collects more '1's is slightly *less* attractive, all else equal. This is strictly monotone in endorsement count — there is no step at unanimity. It makes a lone weak endorsement no better (often slightly worse) than an empty panel, and it makes a 5/5 panel no better than a 4/5 panel by exactly the same amount, which is why 'high-coherence' and 'low-coherence' single-cue steps move together rather than apart. θ sits just at the subjective weight of the weakest experts — close enough that a lone weak endorsement is near-neutral to mildly counter-diagnostic rather than strongly aversive — while endorsement by a strong expert still wins comfortably.

(3) **Endorsement-density dilution of decision sensitivity.** The same absolute weighted difference is less decisive when both panels are densely endorsed. Evidence is divisively normalised by the mean endorsement density T = (Σa + Σb)/2n: D = [Σ_j w_j(a_j−b_j) − θ(Σa−Σb)] / (c0 + κ·T). The dilution has TWO separable constants: a floor c0 that sets how crisply *sparse* panels are compared, and a gain κ that sets how much extra mush dense panels accrue. The floor is *not* negligible: even near-empty panels are compared with bounded, distinctly sub-ceiling determinacy, which keeps lone-endorsement-versus-empty cells close to (rather than far below) chance and preserves the graded ladder structure of sparse comparisons. Heavily endorsed panels are compared mushily. Unlike a consensus premium, dilution is *direction-blind* — it never confers an advantage on the more coherent option, it only compresses whatever advantage exists — so it produces the observed shrinkage of a fixed one-cue advantage at high endorsement backgrounds without ever predicting a unanimity bonus.

p(A) = (1−ε)·logistic(βD) + ε/2. Everyone holds the same stationary rule (no feedback is given, so nothing is learned); heterogeneity is broad and continuous — in determinacy β, validity sensitivity γ, discount θ, positional decay λ, dilution (c0, κ), lapse ε and the readability coefficients — never discrete strategy switching. The theory therefore predicts substantial between-subject spread in every conflict metric, largest where margins are intermediate.

**Parameters:**
- gamma: [1.1, 1.9]
- beta: [2.0, 5.6]
- theta: [0.07, 0.145]
- c0: [0.05, 0.13]
- kap: [1.1, 2.3]
- mu0: [0.0, 0.14]
- mu1: [1.4, 2.2]
- mu2: [0.20, 0.60]
- lam: [0.70, 1.20]
- eps: [0.03, 0.22]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Goal: dissociate Take-The-Best (one-reason, cue-order-driven, magnitude-blind) from Tallying (equal-weight vote count, margin-graded) both in *direction* and in *graded magnitude* of choice probabilities.

Five experts with clearly spread validities (0.92, 0.80, 0.70, 0.60, 0.53) so the cue cascade is unambiguous and subjects are told expert 1 is much better than experts 3-5.

The 16 unique pairs (each repeated 6x, 96 trials) fall into five diagnostic families, each mirrored so that neither response key is systematically correct:

(1) DIRECTIONAL CONFLICT WITH GRADED TALLY MARGIN. Pairs 1-6: the option winning the top (92%) cue loses on all remaining cues, with the tally margin varied at 3 (1 vs 4 wins), 2 (1 vs 3 wins, one tie) and 1 (2 vs 3 wins). TTB predicts the SAME high probability of choosing the top-cue option in all three, independent of margin; Tallying predicts the OPPOSITE option with a probability that increases monotonically with margin (sigmoid(beta*margin)). This is the core quantitative dissociation: a flat-versus-graded profile across margin, plus opposite signs.

(2) TALLY-TIE / TTB-DISCRIMINATES. Pairs 7-8: A=[1,0,1,0,0] vs B=[0,1,0,1,0]. Feature wins are 2-2, so Tallying must guess (exactly 50/50 regardless of beta), while TTB is driven deterministically by the 92% cue. Any systematic departure from 50% here is fatal to Tallying.

(3) TTB-STRONG / TALLY-WEAK ON A SINGLE LOW CUE. Pairs 9-10: the options tie on the four top cues and differ only on the 53% cue. TTB is fully committed (its stopping rule gives the winner probability sigmoid(beta), the same as for a top-cue win), whereas Tallying only has a margin of 1, its weakest signal. Comparing accuracy here against family (5) below separates the two models' beta scalings.

(4) DEEP-CUE CASCADE CONFLICT. Pairs 13-14: options tie on the 92% cue, the 80% cue favors one option, and the three lower cues all favor the other. TTB stops at cue 2 and chooses against a 3-vs-1 tally. This checks that the cascade (not just the single best cue) governs choice.

(5) AGREEMENT ANCHORS. Pairs 11-12 (4 wins vs 1 win, same direction for both models) and 15-16 (only the top cue discriminates, all else tied) provide low-conflict baselines that pin down epsilon and beta for both models so the conflict trials are interpreted against a calibrated noise level.

Predicted pattern if TTB is right: near-ceiling choice of the top-cue option in families 1-4 with NO modulation by tally margin, and above-chance choice in the tally-tie pairs. If Tallying is right: choice reverses in family 1, scales with margin (m=1 < m=2 < m=3), is at chance in family 7-8, and is weak in pairs 9-10.

[1] GOAL: dissociate Tallying (advocated) from Take-The-Best (competing) by exploiting the fact that TTB's choice probability is *constant* (sigmoid(beta)) whenever ANY cue discriminates, whereas Tallying's is a graded function of the vote margin, sigmoid(beta*m). The previously explored design put the gradation manipulation only inside directional-conflict trials; here the core diagnostic is a *no-conflict* margin ladder, which cleanly separates the two models on magnitude alone (no direction confound), and it is cross-validated against conflict and tally-tie families that separate them on direction.

Six experts with widely spread, deliberately non-monotonic validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65. TTB's cue cascade is therefore f0 > f2 > f4 > f5 > f1 > f3, which is NOT the display order, so cue-identity effects cannot be mimicked by a simple left-to-right reading strategy. 16 unique pairs, each repeated 6x (96 trials), fully key-balanced (every pair appears with A and B swapped).

FAMILY C 	rairairs 1-8): MARGIN LADDER WITHOUT CONFLICT. The 93% cue always favors the same option that wins the tally, and the tally margin takes the values m = 1, 2, 3, 4 (each in both key directions). TTB predicts EXACTLY the same accuracy at m=1 as at m=4 (flat line, sigmoid(beta) after lapse). Tallying predicts a steeply rising curve, e.g. at beta=0.6 roughly .65 / .77 / .86 / .92. This is the central quantitative signature: a flat-versus-monotone accuracy profile over margin with the direction of choice held constant, so no directional confound can explain it.

FAMILY D (pairs 9-12): TALLY TIES, TTB DISCRIMINATES. Pairs 9-10 are 2-2 splits where the 93% cue discriminates; pairs 11-12 are 2-2 splits where f0 and f2 TIE and the first discriminating cue is the 74% expert (with the 65%, 58% and 51% experts split against it). Tallying must be exactly 50/50 in all four, irrespective of beta and of which cue is on top. TTB predicts systematic above-chance choice of the top-discriminating-cue option, and — importantly — the SAME magnitude for the f0-driven and f4-driven pairs. Any deviation from chance here falsifies Tallying; equality of the f0- and f4-driven deviations is a further TTB-specific prediction.

FAMILY E (pairs 13-16): GRADED DIRECTIONAL CONFLICT. The 93% cue favors one option while the tally favors the other, with tally margin 4 (pair 13/14: 5 wins vs 1) or 2 (pair 15/16: 3 wins vs 1). TTB predicts the same probability of choosing the f0-winner in both, and this probability must equal its Family-C and Family-D accuracy (all are sigmoid(beta)). Tallying predicts the opposite option, with a much larger effect at m=4 than at m=2, and with beta already pinned down by the Family-C ladder — so the conflict trials are a genuine out-of-sample prediction rather than a free fit.

DECISION RULE: If Tallying is correct, accuracy in Family C rises monotonically with margin, Family D is at chance, and Family E reverses against the 93% expert more strongly at m=4 than m=2, with all effects captured by a single (beta, epsilon). If TTB is correct, one flat probability level sigmoid(beta) accounts for Families C, D (top-cue side) and E (top-cue side) simultaneously, with zero margin modulation anywhere.

[2] GOAL: dissociate Take-The-Best (advocated) from WADD-\u03b3 with Positional Validity Misattribution (competing). The competitor's central falsifiable commitment is that subjective weights are largely bound to *display columns* (importance assumed to fall left-to-right, mixing weight \u03bb \u2265 0.64), whereas TTB binds validities to experts correctly and consults cues in true validity order, stopping at the first discriminator. The blend is the identity only when columns are already sorted by descending validity, so I present the five experts in strictly ASCENDING validity order (col0=.55, col1=.62, col2=.70, col3=.85, col4=.93). This maximises the mismatch: the positional default assigns the LARGEST weight to col0 (the 55% expert, TTB's very last cue) and the SMALLEST to col4 (the 93% expert, TTB's first cue).\n\nEffective competitor weights (normalised) across its parameter range: at \u03bb=.64,\u03b3=1.15 -> [.31,.22,.13,.15,.19]; at \u03bb=.80,\u03b3=1.50 -> [.44,.25,.10,.10,.12]; at \u03bb=1.0,\u03b3=1.85 -> [.60,.29,.08,.03,.01]. In every case col0 > col1 > {col2,col3,col4}. TTB's order is exactly col4 > col3 > col2 > col1 > col0. This gives sign reversals that hold over the entire competitor parameter box.\n\n16 unique pairs (K=6, 96 trials), every family mirrored so neither key is systematically correct:\n\nFAMILY A (pairs 1-2, EXTREME POSITION CONFLICT): only col0 and col4 discriminate, in opposite directions. TTB decisively picks the 93%-cue option; the competitor picks the col0 option with |d| = .13 to .60 (p up to ~1 at high \u03b2). Opposite signs, robust to \u03bb and \u03b3.\n\nFAMILY B (pairs 3-4, MILD POSITION CONFLICT): only col1 (62%) and col3 (85%) discriminate. TTB again picks the higher-validity (col3) option with EXACTLY the same probability as in Family A (sigmoid(\u03b2) is invariant across all discriminating trials); the competitor again reverses but with a smaller margin (|d| = .06 to .26). The TTB-flat vs competitor-graded profile across A > B is the core quantitative signature.\n\nFAMILY C (pairs 5-6, NEAR-CANCELLING): col2 (70%) vs col3 (85%). Their positional weights are nearly identical, so the competitor is pinned near chance (|d| \u2264 .05, and its sign even flips within the parameter box), while TTB is fully committed to the col3 option at its usual sigmoid(\u03b2) level. Any systematic above-chance choice here is fatal for the competitor's positional binding.\n\nFAMILY D (pairs 7-8, DOMINANCE ANCHOR): all five cues favour one option. Both models agree and near ceiling; this pins \u03b2 and \u03b5 for both models so the conflict families are out-of-sample predictions rather than free fits.\n\nFAMILY E (pairs 9-10, TWO-VS-TWO POSITION CONFLICT): A wins the two leftmost/least valid cues (col0,col1), B wins the two rightmost/most valid cues (col3,col4). TTB \u2192 B via col4; competitor \u2192 A with a large margin (|d| = .19 to .57). This is the strongest directional test.\n\nFAMILY F (pairs 11-12, DEEP-CASCADE vs POSITION): the top two validity cues (col4, col3) tie; col2 favours one option and col0+col1 favour the other. TTB stops at col2 and chooses against a 2-vs-1 vote; the competitor chooses the col0+col1 option strongly. Tests that the cascade, not just the single best cue, governs TTB choice, while remaining maximally diagnostic against positional weighting.\n\nFAMILIES G and H (pairs 13-16, MAGNITUDE-ONLY CONTRAST WITH DIRECTION HELD CONSTANT): in G only col4 (93%) discriminates; in H only col0 (55%) discriminates. Both models pick the option with the single 1, so there is no directional confound. TTB predicts IDENTICAL accuracy in G and H (and identical to Families A-F and D, all sigmoid(\u03b2)). The competitor predicts H \u226b G (weights .44 vs .12 at \u03bb=.8,\u03b3=1.5; .60 vs .01 at \u03bb=1), i.e. subjects should be far more decisive when only the worst-but-leftmost expert speaks than when only the best-but-rightmost expert speaks. That inversion is impossible under TTB.\n\nDECISION RULE: If TTB is right, one flat probability level accounts for all eight families, with choices always tracking the higher-validity discriminating expert (col4 > col3 > col2 > col1 > col0) irrespective of column position, and no G-vs-H asymmetry. If the positional WADD is right, choices reverse in Families A, B, E and F, sit at chance in Family C, and show a large H > G accuracy gap with graded magnitudes ordered D > F > E > H > A > B > G > C.

[3] GOAL: dissociate WADD-\u03b3 with Positional Validity Misattribution (advocated) from Take-The-Best (competing). TTB binds stated validities to the correct expert and consults cues in true validity order, stopping at the first discriminator, so its choice probability is a single flat level sigmoid(beta) (plus lapse) on EVERY discriminating trial, always siding with the highest-validity discriminating expert. The advocated model claims that the validity->column binding is largely replaced by the display default 'importance falls left to right' (lam >= 0.64), and that the resulting weighted margin d produces GRADED choice probabilities.

DISPLAY ORDER (deliberately scrambled, not merely ascending): col0=.60, col1=.93, col2=.55, col3=.86, col4=.68, col5=.78. TTB's cascade is col1 > col3 > col5 > col4 > col0 > col2. The positional default instead assigns the weight multiset in descending order to columns, so col0 (a 60% expert) receives the LARGEST subjective weight and col5 (a 78% expert) receives nearly none. Effective normalised weights over the whole competitor-free parameter box: lam=.64,gamma=1.15 -> [.271,.312,.119,.157,.065,.076]; lam=.80,gamma=1.50 -> [.374,.309,.128,.112,.037,.039]; lam=1.0,gamma=1.85 -> [.518,.269,.138,.053,.017,.005]. Across the whole box col0 and col1 dominate, col2 > col3 for most of it, and col4/col5 are near zero even though col5's true validity (.78) is third best.

18 unique pairs (9 families, each mirrored so neither key is systematically correct), K=5 -> 90 trials. Shared 1s are added as background so no option is trivially empty.

FAMILY A (pairs 1-2, col0 .60 vs col3 .86 discriminate in opposite directions): TTB -> the col3 option; WADD-pos -> the col0 option, |d| ~ .26-.47. Sign reversal robust over the entire parameter box.
FAMILY B (pairs 3-4, col0 vs col5 .78): TTB -> col5 option with EXACTLY the same probability as in Family A; WADD-pos -> the col0 option again but with a LARGER margin (|d| ~ .34-.51). A>B ordering with direction held constant is the core flat-vs-graded signature.
FAMILY C (pairs 5-6, col2 .55 vs col4 .68): TTB -> the col4 option (col4 precedes col2 in the true cascade); WADD-pos -> the col2 option but only weakly (|d| ~ .05-.12). Reversal with a small, gradable margin.
FAMILY D (pairs 7-8, col0 .60 vs col1 .93): TTB is decisive for the col1 option. WADD-pos is near-cancelling and its sign flips inside the parameter box (col1 wins at lam=.64/gamma=1.15, col0 wins at lam>=.8). This is the lam-identifying family and, per the heterogeneity claim, should show the LARGEST between-subject spread of any family.
FAMILY E (pairs 9-10, coalition): one option wins col1 (the true 93% expert) alone, the other wins col0 + col5. TTB stops at col1; WADD-pos reverses (B wins for every (lam,gamma) in the box, though with a modest margin .04-.25).
FAMILY F (pairs 11-12, deep cascade vs position): col1 ties, col3 (.86) favours A while col0 + col2 favour B. TTB stops at col3 and picks A against a 2-vs-1 vote; WADD-pos picks B strongly (|d| ~ .23-.42). Strongest directional test that also verifies TTB uses the cascade, not just the single best cue.
FAMILIES G and H (pairs 13-16, MAGNITUDE ONLY, NO DIRECTIONAL CONFOUND): in G only col1 (93%) discriminates; in H only col5 (78%) discriminates. Both models favour the same option in both, so TTB must produce IDENTICAL accuracy in G and H (and identical to A-F). WADD-pos predicts G (|d| ~ .27-.31) far above H (|d| ~ .005-.08, essentially chance): subjects should be near-indifferent when only a 78% expert speaks from the rightmost column.
FAMILY I (pairs 17-18, full dominance): all six cues agree; both models near ceiling. This pins beta and epsilon for both models so every conflict family is an out-of-sample prediction rather than a free fit.

DECISION RULE: If TTB is right, one flat probability level fits families A-I, choices always track the highest-validity discriminating column (col1 > col3 > col5 > col4 > col0 > col2), and G = H exactly. If positional WADD-\u03b3 is right, choices REVERSE in A, B, C, E, F, sit near chance in D and H, and the graded ordering of |accuracy - .5| should be I > F > B > A > G > E > C > D > H, with the between-subject spread peaking in the intermediate-margin families (D, E, C) and collapsing in I and H.

[4] GOAL: dissociate Take-The-Best (advocated) from RVW-DE+/U (competing). The competitor's sharpest falsifiable commitment is the STRICT-UNANIMITY consensus premium: rho*|2*mean(x)-1|^q with high q (5-11), so a panel that is unanimous (0/5 or 5/5) collects almost the whole premium (rho ~ .19-.59, mean .39) while a merely lopsided 4/5 or 1/5 panel collects essentially none. TTB has no such term: on every discriminating trial it sides with the highest-validity discriminating expert at one flat level sigmoid(beta) (plus lapse), regardless of endorsement counts, coherence, or lower cues.

DISPLAY: five experts shown in strictly DESCENDING validity (.95,.85,.75,.65,.55). This is deliberately the competitor's *best case*: Kendall tau = -1 so incongruence = 0, the best expert sits in an extreme column, hence kappa = k0 (.02-.16) and the reading-order fallback phi_eff = phi*(0.35+1.15*k0) is nearly inert. Readability nuisance is therefore removed and the competitor's predictions are driven almost entirely by its graded validity weights and by the unanimity premium. Normalised weights across its gamma box (1.05-1.95) are approximately [.46-.64, .23-.26, .09-.16, .03-.09, .003-.03], i.e. cue1 >> cue2 > cue3 > cue4 > cue5 for every subject.

16 unique pairs (K=6, 96 trials), every pair mirrored so neither response key is systematically correct. TTB predicts A on pairs 1,3,5,7,9,11,13,15 and B on their mirrors, always at the SAME probability.

FAMILY S - SINGLE ENDORSEMENT vs AN EMPTY (UNANIMOUS-ZERO) PANEL (pairs 1-6). One option has exactly one '1' (on cue1, cue3 or cue5) and the other is all zeros. TTB: the endorsed option wins in all three cases at an identical level (~.99 at moderate beta). Competitor: D(endorsed) = w_j - rho*(1 - 0.6^q) ~ w_j - .38, so with beta=7, gamma=1.5, rho=.39, q=8 it predicts p(endorsed) ~ .70 (cue1), .13 (cue3), .06 (cue5) - a steep monotone gradient that CROSSES CHANCE between cue1 and cue3, i.e. explicit sub-chance accuracy when only a weak expert speaks against an all-zero panel. Flat-at-ceiling versus sign-reversing-gradient is the core directional dissociation.

FAMILY P - UNANIMITY STEP WITH WEIGHTED EVIDENCE HELD EXACTLY CONSTANT (pair 5/6 reused as the baseline, plus pairs 7-10). In all three cells the two options differ on cue5 and only cue5, so the weighted-evidence term is identical (w5 ~ .01-.03) and TTB must give the SAME probability in all three. Only the coherence of the panels changes: (i) [0,0,0,0,1] vs [0,0,0,0,0] - loser panel unanimous, full premium; (ii) [1,1,0,0,1] vs [1,1,0,0,0] - both panels middling (2/5 vs 3/5), premium ~ 0; (iii) [1,1,1,1,1] vs [1,1,1,1,0] - winner panel unanimous, full premium in the same direction as the cue. Competitor predicts p(cue5-winner) ~ .06 / .51 / .94 respectively (a non-monotone step at exactly 0 and n dissenters), TTB predicts ~.99 / .99 / .99. This triple is the sharpest quantitative signature in the design and cannot be mimicked by any re-weighting of cues.

FAMILY D - COMPENSATION / GAMMA IDENTIFICATION (pairs 11-12): [1,0,0,0,0] vs [0,1,1,1,1]. Coherence cancels exactly (both panels 1/5 vs 4/5, |2m-1| identical), so this is a pure weighted-integration test. TTB stops at cue1 and is at ceiling for the top-cue option; the competitor is near-cancelling and its sign flips inside its own gamma box (core = +.05 at gamma=1.5, -.08 at gamma=1.05, +.15 at gamma=1.95), so it must predict the LARGEST between-subject spread of any family here and an overall level close to chance.

FAMILY E - DEEP CASCADE vs COALITION (pairs 13-14): [1,1,0,0,0] vs [1,0,1,1,1]. Cue1 ties, cue2 (.85) favours one option, cues 3-5 favour the other. TTB stops at cue2 and is at ceiling; the competitor computes core ~ .02 and lands near chance (p ~ .59). Confirms the cascade (not merely the single best cue) drives TTB, while remaining maximally diagnostic.

FAMILY F - DOMINANCE ANCHOR (pairs 15-16): [1,1,1,1,1] vs [0,0,0,0,0]. Both panels unanimous so the premium cancels exactly; both models are at ceiling. This pins beta and epsilon for BOTH models, so every other family is an out-of-sample prediction rather than a free fit.

DECISION RULE. If TTB is right: a single flat probability level (calibrated by Family F) accounts for Families S, P, D and E alike; accuracy on the single-endorsement pairs is high and IDENTICAL for cue1, cue3 and cue5; the three cue5-only cells (S5, P2, P3) are indistinguishable; and no coherence/unanimity effect appears anywhere. If RVW-DE+/U is right: accuracy on Family S falls steeply with cue validity and goes SUB-CHANCE for cue3 and cue5 (the all-zero panel is preferred to a weakly endorsed one), the cue5-only ladder shows a large non-monotone step (~.06 / ~.5 / ~.94) driven purely by unanimity, Family D shows near-chance means with maximal individual spread, and Family E hovers near chance.

[5] GOAL: dissociate RVW-DE+/U (advocated) from Take-The-Best (competing) on three orthogonal signatures: (i) READABILITY FLATTENING, (ii) the STRICT-UNANIMITY consensus premium, (iii) flat-vs-graded choice probability. TTB's core commitment is that on every discriminating trial it sides with the highest-validity discriminating expert at ONE flat level sigmoid(beta) (plus lapse), regardless of the other cues, of endorsement counts, of coherence, and of where the expert is displayed.

DISPLAY (deliberately scrambled): col0=.62, col1=.83, col2=.93, col3=.54, col4=.72. Kendall tau between column index and validity is exactly 0 (5 concordant / 5 discordant pairs) and the best expert (.93) sits in a MIDDLE column, so under RVW kappa = k0 + k1*1 + k2*1 lies in [0.67, 1.0] across the whole parameter box: validity weights are strongly flattened toward equal weighting and the reading-order fallback (phi_eff = phi*(0.35+1.15*kappa)) is maximally recruited. Effective weights therefore run roughly [.18-.26, .22-.24, .20-.34, .12-.17, .13-.16] by column (mid-parameter ~[.223,.237,.253,.148,.138]) - the .93 expert never commands more than ~1/3 of the weight, and two or three other columns always outweigh it. TTB's cascade, by contrast, is col2 > col1 > col4 > col0 > col3, bound to experts, not columns.

12 base pairs, each mirrored (A/B swapped) so neither key is systematically correct: 24 unique pairs, K = 4, 96 trials. TTB predicts A on 11 of the 12 base pairs (and B on P1) at IDENTICAL probability throughout.

FAMILY L - COALITION LADDER (pairs 1-4 + mirrors). A carries only the .93 endorsement (col2); B carries a growing coalition of other columns: {col0}, {col0,col3}, {col0,col3,col4}, {col0,col1,col3,col4}. TTB stops at col2 and picks A at ceiling in all four, with zero modulation. RVW predicts a steep monotone ladder that CROSSES CHANCE: mid-parameter cores +.03 / -.12 / -.26 / -.49, i.e. p(A) ~ .55 / .30 / .14 / .03 at beta=7. Coherence is exactly cancelling in L1 and L4 (1/5 vs 1/5 and 1/5 vs 4/5 give identical |2m-1|) and negligible elsewhere, so this family is a pure weighted-integration/flattening test. L1 additionally is the near-cancelling cell where the RVW sign flips inside its own parameter box, so it should show the LARGEST between-subject spread of any family.

FAMILY U - UNANIMITY STEP WITH WEIGHTED EVIDENCE HELD EXACTLY CONSTANT (pairs 5-8 + mirrors). All four cells differ on col3 and ONLY col3, so the weighted-evidence term is identical and TTB must give the same ceiling probability in all four. Only panel coherence changes: (U1) 1/5 vs 0/5 - the LOSER is unanimous, so it collects nearly the whole premium; (U2) 3/5 vs 2/5 - neither panel coherent; (U3) 4/5 vs 3/5 - still no premium under high q even though mean rating rose; (U4) 5/5 vs 4/5 - the WINNER is unanimous. RVW predicts p(A) ~ .15 / .74 / .75 / .97: a sub-chance reversal when a lone weak endorsement faces an all-zero panel, a flat plateau across U2-U3 (which a smooth, low-q coherence rule could not produce - it would make U3 clearly exceed U2), and a jump at exactly 0 dissenters. This non-monotone step at the unanimity boundary, with weighted evidence pinned, is the sharpest falsifiable signature and is impossible under any re-weighting of cues.

FAMILY V - PREMIUM ADDITIVITY AT A HEAVIER COLUMN (pairs 11-12 + mirrors). Same manipulation on col0 (weight ~.22 rather than ~.15): [1,0,0,0,0] vs all-zeros, and [1,1,1,1,1] vs [0,1,1,1,1]. RVW predicts p(A) ~ .22 and ~.98 - the same rho-sized step, shifted upward by the larger cue weight (V1 > U1, V2 > U4). TTB again predicts one flat ceiling.

FAMILY P - POSITION VERSUS VALIDITY (pair 9 + mirror). Only col0 (.62, leftmost) and col4 (.72, rightmost) discriminate, endorsement counts and coherence identical, so theta and rho are inert. TTB decisively picks the col4 (.72) option. RVW, with flattened weights plus the recruited reading-order gradient, predicts the OPPOSITE (col0) option at p ~ .64 mid-parameter, ranging from chance to clearly reversed. Any failure to prefer the higher-validity expert here is fatal for TTB.

FAMILY D - DOMINANCE ANCHOR (pair 10 + mirror). [1,1,1,1,1] vs [0,0,0,0,0]: both panels unanimous so the premium cancels exactly, all weights point the same way. Both models are at ceiling; this pins beta and epsilon for BOTH models, making every other family an out-of-sample prediction rather than a free fit.

DECISION RULE. If TTB is right: one flat probability level, calibrated by Family D, accounts for L1-L4, U1-U4, V1-V2 alike (all choosing the option favoured by the first discriminating cue), no ladder gradation, no coherence effect, and Family P shows decisive choice of the .72 option. If RVW-DE+/U is right: the L ladder falls monotonically and crosses chance, the U ladder shows the non-monotone unanimity step (.15 / .74 / .75 / .97) with U2 = U3, V mirrors U at a higher level, Family P reverses or sits at chance, and between-subject spread peaks in L1 and P.

[6] GOAL: dissociate PWA-DD (advocated) from RVW-DE+/U (competing). The two theories share graded validity weighting w \u221d logit(v)^gamma, so I deliberately remove the readability nuisance by displaying the five experts in strictly DESCENDING validity (.93,.84,.74,.64,.55): Kendall tau = -1 (incongruence 0) and the best expert sits in an extreme column, so PWA-DD's mixture gate is inert (mu = mu0 <= .14, essentially pure validity weights) and RVW's flattening is minimal (kappa = k0 <= .16, reading-order fallback phi_eff <= .18). Both models therefore compute nearly the SAME weighted core; every remaining prediction difference is attributable to the two rival post-integration mechanisms: PWA-DD's endorsement-DENSITY DILUTION (divide by c0 + kap*T, direction-blind, smooth and monotone in total endorsement density) plus a monotone faint-praise discount theta, versus RVW's STRICT-UNANIMITY consensus premium rho*|2m-1|^q with high q (a step at exactly 0 and n dissenters, direction-carrying).

Mid-parameter weights (gamma=1.5) are approx [.53,.27,.14,.055,.011] by column, so cue1 (.84) is a solidly weighted cue (w >> theta) and cue4 (.55) is a cue whose weight lies BELOW theta for the whole PWA-DD box (w4 <= .05 < theta_min = .07). 24 unique pairs (K=4, 96 trials); every base pair is mirrored so neither response key is systematically correct.

FAMILY S \u2014 STRONG-CUE LADDER OVER ENDORSEMENT BACKGROUND (pairs 1-10). The two options differ on cue1 (.84) and ONLY cue1; the shared background of 1s grows from k=0 to k=4, so the weighted-evidence term and the endorsement-count difference are IDENTICAL in all five cells and only the total endorsement density T (and, incidentally, panel unanimity at the two ends) changes. PWA-DD: D = (w1-theta)/(c0+kap*T) with T=(2k+1)/10, giving a smooth MONOTONE DECAY that never crosses chance: p(cue1-winner) ~ .87 / .71 / .64 / .60 / .58. RVW: D = w1 - theta + rho*(coh_A - coh_B); coherence is ~0 for k=1,2,3 but -0.98*rho at k=0 (loser panel unanimous-zero) and +0.98*rho at k=4 (winner panel unanimous), giving a NON-MONOTONE step profile ~ .27 / .85 / .85 / .86 / .99. This is a full crossover: at k=0 PWA-DD is at its HIGHEST and RVW is SUB-CHANCE; at k=4 PWA-DD is at its LOWEST and RVW is at ceiling. No re-weighting of cues can mimic either pattern.

FAMILY W \u2014 WEAK-CUE LADDER (pairs 11-16, k=0,2,4, discriminating cue4 = .55). Here w4 < theta for every PWA-DD parameter set, so PWA-DD predicts mildly SUB-CHANCE choice of the endorsed option in all three cells, with the magnitude shrinking as density rises: ~ .26 / .42 / .47 (the c0 floor keeps the sparse cell close to, not far below, chance). RVW predicts the unanimity step again: ~ .12 / .52 / .98. The k=4 cell ([1,1,1,1,1] vs [1,1,1,1,0]) is the sharpest single cell in the design: PWA-DD says just below chance, RVW says near ceiling.

FAMILY T \u2014 PURE DENSITY LADDER WITH COUNTS AND COHERENCE EXACTLY MATCHED (pairs 17-22). A wins cue0 (.93), B wins cue1 (.84), and both options carry an identical number of shared background 1s (0, 1 or 3), so endorsement counts are equal (theta inert), the coherence terms cancel EXACTLY (1/5 vs 1/5, 2/5 vs 2/5, 4/5 vs 4/5), and the weighted core is identical across the three cells. RVW must therefore predict the SAME probability in all three (~.85 flat) \u2014 it has no mechanism that reads total density. PWA-DD predicts a strong monotone compression, ~ .85 / .75 / .63, purely from divisive normalisation by T. This family is the cleanest possible test of density dilution, with direction, counts and coherence all pinned.

FAMILY D \u2014 DOMINANCE ANCHOR (pairs 23-24): [1,1,1,1,1] vs [0,0,0,0,0]. Both panels are unanimous so RVW's premium cancels exactly and PWA-DD's discount subtracts 5*theta while T = .5 dilutes; PWA-DD ~ .83, RVW ~ .96. This calibrates beta and eps for both models so the other families are out-of-sample predictions rather than free fits.

DECISION RULE. If PWA-DD is right: Family S is monotone DECREASING and stays above chance everywhere (no dip at k=0, no jump at k=4); Family W is uniformly at-or-below chance and rises toward .5 with density; Family T shows a clear ~20-point monotone drop from sparse to dense with everything else held constant; and no cell shows a discontinuity at the unanimity boundaries. If RVW-DE+/U is right: Family S and Family W both show a sub-chance reversal at k=0, a flat middle plateau, and a large jump at k=4 (steps located at exactly 0 and n dissenters), while Family T is completely FLAT across the density ladder. Between-subject spread should be largest in Family W's middle cell and Family S's k=1..3 plateau under either account, but only PWA-DD predicts spread ordered by margin rather than by unanimity.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate RVW-DE+/U (advocated) from PWA-DD (competing). The two theories agree on graded validity weighting w \\u221d logit(v)^gamma, so I remove all readability nuisance by displaying the five experts in strictly ASCENDING validity (.55,.63,.72,.83,.94): Kendall tau = +1 (incongruence 0) and the best expert sits in an extreme (rightmost) column, so RVW's flattening gate is inert (kappa = k0 <= .16, phi_eff <= .20) and PWA-DD's positional gate is inert (mu = mu0 <= .14). Both theories therefore compute nearly identical weighted cores; every remaining prediction difference is attributable to the rival post-integration mechanisms: RVW's STRICT-UNANIMITY consensus premium rho*|2m-1|^q (a direction-CARRYING step located at exactly 0 and n dissenters) versus PWA-DD's ENDORSEMENT-DENSITY DILUTION (divide by c0+kap*T, direction-BLIND and smooth) plus a monotone faint-praise discount theta.\n\nMid-parameter weights (gamma=1.5) by column: [.011,.049,.115,.251,.574]. So col3 (v=.83, w~.25) is safely ABOVE PWA-DD's theta box (.07-.145) for every subject, while col1 (v=.63, w~.03-.08) is at or below theta for essentially every subject. 12 base pairs, each mirrored (A/B swapped) so neither key is systematically correct: 24 unique pairs, K = 4, 96 trials.\n\nFAMILY U \\u2014 UNANIMITY LADDER WITH WEIGHTED EVIDENCE AND ENDORSEMENT-COUNT DIFFERENCE HELD EXACTLY CONSTANT (pairs 1-8). The two options differ on col3 and ONLY col3; a shared background of 1s grows k = 0, 1, 2, 4, so the core (w3) and the count difference (+1) are identical in all four cells and only total density T (=.1,.3,.5,.9) and panel unanimity change. RVW: coherence is ~0 for k=1,2 (2/5 vs 1/5, 3/5 vs 2/5 give |2m-1|^q ~ 0 at q>=5), but at k=0 the LOSER panel is unanimous-zero (-rho) and at k=4 the WINNER panel is unanimous-one (+rho): predicted p(col3-winner) ~ .27 / .83 / .83 / .99 \\u2014 a flat middle plateau with a sub-chance dip at k=0 and a jump at k=4, i.e. steps only at exactly 0 and n dissenters. PWA-DD: D = (w3-theta)/(c0+kap*T) gives a smooth MONOTONE DECAY that never crosses chance, ~ .89 / .71 / .64 / .58. The two profiles cross twice; no re-weighting of cues can produce either.\n\nFAMILY W \\u2014 WEAK-CUE (col1) VERSION AT THE TWO EXTREMES (pairs 9-12). W1 ([0,1,0,0,0] vs all-zeros): both models sub-chance but very differently (RVW ~.09 from the full unanimity penalty; PWA ~.30 from w1-theta scaled by the c0 floor). W2 ([1,1,1,1,1] vs [1,0,1,1,1]) is the sharpest single cell in the design: RVW says ~.98 (winner unanimous, premium swamps the tiny cue) while PWA says just BELOW chance (~.47), because w1 < theta and the dense panels are maximally diluted. Comparing W2 with U3 also separates them: PWA predicts U3 (.58) > W2 (.47) with W2 sub-chance; RVW predicts both at ceiling and nearly equal.\n\nFAMILY C \\u2014 POLARITY SYMMETRY OF THE COHERENCE EFFECT (pairs 13-16). C1 ([1,1,1,1,1] vs [0,0,1,1,1]) and its exact bit-complement C2 ([0,0,0,0,0] vs [1,1,0,0,0]) pit two weak-cue endorsements against panel purity. RVW's premium is direction-blind, so it must favour the COHERENT option in both cases with similar magnitude (~.97 and ~.94). PWA-DD's discount and dilution always favour the SPARSER option, so it predicts a strong asymmetry: ~.40 in C1 (choosing the sparse 3-endorsement panel against the unanimous one) versus ~.80 in C2. C1 is a clean directional reversal between the theories.\n\nFAMILY T \\u2014 PURE DENSITY LADDER WITH COUNTS AND COHERENCE EXACTLY MATCHED (pairs 17-22). A wins col4 (.94), B wins col3 (.83), and both panels carry an identical number of shared background 1s (0, 1, 3), so counts are equal (theta inert), coherence cancels EXACTLY (1/5 vs 1/5, 2/5 vs 2/5, 4/5 vs 4/5) and the weighted core is identical across cells. RVW has no mechanism reading total density and must predict a FLAT ~.91 across all three; PWA-DD predicts a ~25-point monotone compression, ~.95 / .83 / .70. This is the cleanest possible test of density dilution with direction, counts and coherence pinned.\n\nFAMILY D \\u2014 DOMINANCE ANCHOR (pairs 23-24): [1,1,1,1,1] vs [0,0,0,0,0]. Both panels unanimous so RVW's premium cancels exactly (p ~ .99) while PWA-DD pays 5*theta and divides by c0+kap*0.5 (p ~ .87). This pins beta and eps for both models, making every other family an out-of-sample prediction rather than a free fit.\n\nDECISION RULE. If RVW-DE+/U is right: Family U shows a sub-chance dip at k=0, a flat plateau at k=1-2, and a jump to ceiling at k=4 (discontinuities located only at 0 and n dissenters); W2 is near ceiling; C1 ~ C2 (coherent option preferred regardless of polarity); Family T is completely flat across the density ladder. If PWA-DD is right: Family U declines smoothly and monotonically with density and never crosses chance; W2 sits at or just below chance; C1 and C2 are strongly asymmetric with the sparser panel always favoured; and Family T shows a large monotone drop with everything else held constant.",
  "validities": [
    0.55,
    0.63,
    0.72,
    0.83,
    0.94
  ],
  "trial_a_ratings": [
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
      0,
      0,
      0
    ],
    [
      0,
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
      0
    ],
    [
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
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
      1,
      1,
      0,
      1
    ],
    [
      0,
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
      1,
      1,
      1
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
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
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
    ]
  ],
  "trial_b_ratings": [
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
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0
    ],
    [
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
      0,
      1
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
      0,
      0,
      0
    ],
    [
      1,
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
      1
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
      0,
      0,
      1
    ],
    [
      0,
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
      1
    ],
    [
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
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
      1,
      1,
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
  "input_tokens": 23378,
  "output_tokens": 17488
}
```
