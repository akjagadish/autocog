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
**Description:** **ERC-SB (determinacy-corrected): Expectation-Referenced Cue Coding with Saturating Deviation Value, Compressed Endorsement Numerosity, and *Binding-Conditional* Contrast Compression.**

Choice between two binary-cue panels is compensatory, stationary (no feedback is given, so nothing is learned), and rests on one idea: every displayed rating is scored *relative to a subjective endorsement expectation*, and both the deviation from that expectation and the endorsement surplus it is charged against are perceived on compressed psychophysical scales.

(1) *Signed, expectation-referenced cue coding.* Expert j's stated validity is converted to diagnosticity x_j = (logit v_j)^gamma, and the decision maker holds a reference endorser v* with diagnosticity theta = (logit v*)^gamma. A '1' is good news only to the extent that the endorsing expert is more discriminating than the reference; approval from a sub-reference endorser is faint praise and is genuinely counter-diagnostic. This single mechanism replaces both a unanimity premium and a uniform endorsement discount.

(2) *Diminishing sensitivity about the reference.* Subjective value is an S-shaped function of the deviation from the expectation reference, x~_j = theta + A*tanh(s*(x_j - theta)/A): sharp discrimination near v*, saturation on both branches. People categorise experts as 'better' or 'worse' than their reference endorser, so all clearly sub-reference endorsers are about equally damning.

(3) *Compressed endorsement numerosity.* The expectation charge is levied on the compressed magnitude of the endorsement surplus m: charge = theta*sign(m)*|m|^zeta, zeta < 1 (Weber-like). One extra endorsement is felt far more sharply than the fifth relative to the fourth.

(4) *Contrast-count normalisation that is background-blind but binding-conditional.* Sensitivity is divided by a term that grows with the number K of columns on which the panels actually disagree and is completely blind to the shared endorsement background: density ladders with fixed contrast and coherence must be FLAT. The cost of reconciling many simultaneous contrasts depends on whether the validity-to-column mapping has been successfully bound. When the display is validity-monotone, each discriminating column is read off cheaply and the cost is linear in K. When binding has failed (scrambled display / interior best expert), each additional discriminating column must be held in mind without a stable importance anchor, so cost grows *super*-linearly: denom = c0 + kappa*K*(1 + delta*mu^2*(K-1)). Scrambling a display therefore flattens *large* contrasts far more than small ones.

(5) *Readability / column binding.* When the display is monotone in validity (either direction) binding is trivial and the signed values are used intact; when the mapping is scrambled - especially with the best expert interior - values flatten toward their mean and a left-to-right reading gradient is recruited.

(6) *Population structure and determinacy.* One stationary rule for everyone, with broad continuous heterogeneity in determinacy beta, validity sensitivity gamma, crossover v*, reference-sensitivity (A, s), numerosity exponent zeta, normalisation (c0, kappa, delta), reading gradient rho and lapse eps. The added commitment of this version is that *attentional lapsing is rare*: essentially all of the observed regression toward chance is produced by the evidence transformation itself (expectation-referenced saturation and contrast-count normalisation), not by stimulus-independent guessing. Choices are therefore more determinate than a large-lapse account would allow, and the shrinkage of any given contrast must be predictable from its K, its endorsement surplus and the readability of the display - never from a free global guess rate. No discrete strategy switching, no learning, no consensus premium, no density dilution.

**Parameters:**
- gamma: [1.25, 1.75]
- vstar: [0.80, 0.90]
- beta: [0.40, 0.88]
- alpha: [0.80, 1.15]
- sl: [1.8, 2.8]
- zeta: [0.55, 0.85]
- c0: [0.25, 0.60]
- kap: [0.45, 0.85]
- delta: [0.65, 1.05]
- mu0: [0.0, 0.10]
- mu1: [1.5, 2.4]
- mu2: [0.35, 0.90]
- rho: [4.0, 5.8]
- lam: [0.7, 1.3]
- eps: [0.0, 0.05]
- validities: validities

**`predict source code`:**
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

[7] GOAL: dissociate RVW-DE+/U (advocated) from PWA-DD (competing). The two theories agree on graded validity weighting w \u221d logit(v)^gamma, so I remove all readability nuisance by displaying the five experts in strictly ASCENDING validity (.55,.63,.72,.83,.94): Kendall tau = +1 (incongruence 0) and the best expert sits in an extreme (rightmost) column, so RVW's flattening gate is inert (kappa = k0 <= .16, phi_eff <= .20) and PWA-DD's positional gate is inert (mu = mu0 <= .14). Both theories therefore compute nearly identical weighted cores; every remaining prediction difference is attributable to the rival post-integration mechanisms: RVW's STRICT-UNANIMITY consensus premium rho*|2m-1|^q (a direction-CARRYING step located at exactly 0 and n dissenters) versus PWA-DD's ENDORSEMENT-DENSITY DILUTION (divide by c0+kap*T, direction-BLIND and smooth) plus a monotone faint-praise discount theta.

Mid-parameter weights (gamma=1.5) by column: [.011,.049,.115,.251,.574]. So col3 (v=.83, w~.25) is safely ABOVE PWA-DD's theta box (.07-.145) for every subject, while col1 (v=.63, w~.03-.08) is at or below theta for essentially every subject. 12 base pairs, each mirrored (A/B swapped) so neither key is systematically correct: 24 unique pairs, K = 4, 96 trials.

FAMILY U \u2014 UNANIMITY LADDER WITH WEIGHTED EVIDENCE AND ENDORSEMENT-COUNT DIFFERENCE HELD EXACTLY CONSTANT (pairs 1-8). The two options differ on col3 and ONLY col3; a shared background of 1s grows k = 0, 1, 2, 4, so the core (w3) and the count difference (+1) are identical in all four cells and only total density T (=.1,.3,.5,.9) and panel unanimity change. RVW: coherence is ~0 for k=1,2 (2/5 vs 1/5, 3/5 vs 2/5 give |2m-1|^q ~ 0 at q>=5), but at k=0 the LOSER panel is unanimous-zero (-rho) and at k=4 the WINNER panel is unanimous-one (+rho): predicted p(col3-winner) ~ .27 / .83 / .83 / .99 \u2014 a flat middle plateau with a sub-chance dip at k=0 and a jump at k=4, i.e. steps only at exactly 0 and n dissenters. PWA-DD: D = (w3-theta)/(c0+kap*T) gives a smooth MONOTONE DECAY that never crosses chance, ~ .89 / .71 / .64 / .58. The two profiles cross twice; no re-weighting of cues can produce either.

FAMILY W \u2014 WEAK-CUE (col1) VERSION AT THE TWO EXTREMES (pairs 9-12). W1 ([0,1,0,0,0] vs all-zeros): both models sub-chance but very differently (RVW ~.09 from the full unanimity penalty; PWA ~.30 from w1-theta scaled by the c0 floor). W2 ([1,1,1,1,1] vs [1,0,1,1,1]) is the sharpest single cell in the design: RVW says ~.98 (winner unanimous, premium swamps the tiny cue) while PWA says just BELOW chance (~.47), because w1 < theta and the dense panels are maximally diluted. Comparing W2 with U3 also separates them: PWA predicts U3 (.58) > W2 (.47) with W2 sub-chance; RVW predicts both at ceiling and nearly equal.

FAMILY C \u2014 POLARITY SYMMETRY OF THE COHERENCE EFFECT (pairs 13-16). C1 ([1,1,1,1,1] vs [0,0,1,1,1]) and its exact bit-complement C2 ([0,0,0,0,0] vs [1,1,0,0,0]) pit two weak-cue endorsements against panel purity. RVW's premium is direction-blind, so it must favour the COHERENT option in both cases with similar magnitude (~.97 and ~.94). PWA-DD's discount and dilution always favour the SPARSER option, so it predicts a strong asymmetry: ~.40 in C1 (choosing the sparse 3-endorsement panel against the unanimous one) versus ~.80 in C2. C1 is a clean directional reversal between the theories.

FAMILY T \u2014 PURE DENSITY LADDER WITH COUNTS AND COHERENCE EXACTLY MATCHED (pairs 17-22). A wins col4 (.94), B wins col3 (.83), and both panels carry an identical number of shared background 1s (0, 1, 3), so counts are equal (theta inert), coherence cancels EXACTLY (1/5 vs 1/5, 2/5 vs 2/5, 4/5 vs 4/5) and the weighted core is identical across cells. RVW has no mechanism reading total density and must predict a FLAT ~.91 across all three; PWA-DD predicts a ~25-point monotone compression, ~.95 / .83 / .70. This is the cleanest possible test of density dilution with direction, counts and coherence pinned.

FAMILY D \u2014 DOMINANCE ANCHOR (pairs 23-24): [1,1,1,1,1] vs [0,0,0,0,0]. Both panels unanimous so RVW's premium cancels exactly (p ~ .99) while PWA-DD pays 5*theta and divides by c0+kap*0.5 (p ~ .87). This pins beta and eps for both models, making every other family an out-of-sample prediction rather than a free fit.

DECISION RULE. If RVW-DE+/U is right: Family U shows a sub-chance dip at k=0, a flat plateau at k=1-2, and a jump to ceiling at k=4 (discontinuities located only at 0 and n dissenters); W2 is near ceiling; C1 ~ C2 (coherent option preferred regardless of polarity); Family T is completely flat across the density ladder. If PWA-DD is right: Family U declines smoothly and monotonically with density and never crosses chance; W2 sits at or just below chance; C1 and C2 are strongly asymmetric with the sparser panel always favoured; and Family T shows a large monotone drop with everything else held constant.

[8] GOAL: dissociate PWA-DD (advocated) from ERC-SB (competing). Both are compensatory and share logit-validity diagnosticity, and both contain the same readability gate, so I remove the readability nuisance entirely by displaying six experts in strictly DESCENDING validity (.95,.86,.78,.70,.62,.54): Kendall tau = -1 (incongruence 0) and the best expert sits in an extreme column, so mu = mu0 (<=.14 / <=.10) for both models. ERC's super-linear binding-failure term is therefore inert and its normaliser is exactly c0+kappa*K. All remaining differences are attributable to the three rival post-integration mechanisms: PWA-DD's endorsement-DENSITY dilution (divide by c0+kap*T, background-sensitive, contrast-count-blind) with a LINEAR faint-praise discount theta calibrated to the weakest cues, versus ERC-SB's expectation-referenced saturating cue values (everything below v* in [.80,.90] is strongly and near-EQUALLY counter-diagnostic), its COMPRESSED numerosity charge theta*|m|^zeta, and its CONTRAST-COUNT normalisation (divide by c0+kap*K, density-blind).

Mid-parameter PWA weights (gamma=1.5, normalised): [.500,.242,.141,.077,.034,.006]; theta ~ .108. Mid-parameter ERC saturated values (v*=.85,gamma=1.5,sl=2.3,alpha=1): [4.55,2.65,0.69,0.21,0.09,0.05] against a reference charge theta = 2.28 per endorsement. 12 base pairs, each mirrored (A/B swapped) so neither key is systematically correct: 24 unique pairs, K = 4 repetitions, 96 trials.

FAMILY V - SINGLE ENDORSEMENT vs AN EMPTY PANEL, SWEPT ACROSS CUE VALIDITY (pairs 1-6 + mirrors). One option carries exactly one '1' (on col0..col5 in turn), the other is all zeros; K=1 and m=+1 in every cell, so ERC's normaliser and charge are identical throughout and only the cue's subjective value moves. PWA-DD predicts a SMOOTH GRADED LADDER that crosses chance between col2 and col3 and keeps descending: ~ .94 / .85 / .62 / .39 / .26 / .20. ERC-SB predicts a STEP followed by a SATURATED FLOOR: ~ .77 / .55 / .29 / .24 / .23 / .23 - cols 2-5 are all sub-reference and, because of the tanh saturation, essentially indistinguishable, and col2 (.78) is REVERSED relative to PWA-DD. The two critical contrasts are (i) V3 vs V5: PWA-DD says a ~19-point spread among the three weakest experts, ERC-SB says <2 points; (ii) V2: PWA-DD says at-or-above chance, ERC-SB says clearly below chance.

FAMILY T - PURE DENSITY LADDER WITH K, CORE AND COUNT DIFFERENCE HELD EXACTLY CONSTANT (pairs 1, 7, 9, 8 + mirrors: shared background k = 0, 1, 2, 5). In all four cells the options differ on col0 and ONLY col0 (K=1, m=+1), so ERC's evidence, charge and normaliser are literally identical and it must be COMPLETELY FLAT at ~.77. PWA-DD divides by c0+kap*T with T = .083/.25/.417/.917 and predicts a large monotone decay ~ .94 / .89 / .82 / .69 that never crosses chance. Flat-versus-25-point-decline with everything else pinned is the cleanest possible test of density dilution against a background-blind account.

FAMILY K - PURE CONTRAST-COUNT LADDER WITH DENSITY, COUNT DIFFERENCE AND CORE HELD ~CONSTANT (pairs 9, 10, 11 + mirrors). All three cells have sum(a)=3, sum(b)=2 (so T=5/12 and m=+1 are identical, making PWA's dilution and discount identical), the col0 (.95) contrast always favours A, and the extra discriminating columns are drawn from the saturated weak cluster so that they nearly cancel: K = 1, 3, 5. PWA-DD has NO contrast-count term and predicts a flat ~ .82 / .84 / .77. ERC-SB divides by c0+kap*K (1.08 / 2.38 / 3.68) and predicts a steep monotone compression ~ .77 / .64 / .57. This is the mirror-image test of Family T and is diagnostic even if one doubts my density claim.

FAMILY D - DOMINANCE ANCHOR (pair 12 + mirror): [1,1,1,1,1,1] vs [0,0,0,0,0,0]. Under PWA-DD the linear discount removes 6*theta from a core of 1 and the density T=.5 dilutes, leaving a clearly above-chance ~.77-.85. Under ERC-SB the compressed charge theta*6^zeta (~8.0-11.4) very nearly cancels or exceeds the summed saturated evidence (~7.8-8.4) and K=6 crushes the normaliser, so it predicts a striking NEAR-CHANCE (.45-.57) response to a fully dominant option. A high-accuracy dominance cell also pins beta and eps for PWA-DD, making Families V, T and K out-of-sample predictions rather than free fits.

DECISION RULE. If PWA-DD is right: Family V is a smooth graded ladder with the three weakest experts clearly separated and col2 at/above chance; Family T declines steeply and monotonically with shared endorsement background; Family K is FLAT across contrast counts; dominance is comfortably above chance. If ERC-SB is right: Family V shows a saturated floor (V2=V3=V4=V5) with col2 already strongly reversed; Family T is perfectly FLAT; Family K falls steeply with the number of discriminating columns; and the dominance pair sits near chance. Between-subject spread should be largest at V1 (col1 = .86 straddles ERC's v* box) and at V2/K1 (PWA's w2 straddles its theta box) under the respective accounts.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate ERC-SB (advocated) from PWA-DD (competing). Both are compensatory, share logit-validity diagnosticity and contain the SAME readability gate, so I remove the readability nuisance entirely: six experts are displayed in strictly DESCENDING validity (.95,.91,.85,.74,.64,.55). Kendall tau = -1 (incongruence 0) and the best expert sits in an extreme column, so mu = mu0 (<=.10 / <=.14) for both models; ERC's super-linear binding-failure term is inert (its normaliser reduces to c0+kap*K) and PWA's positional fallback is inert. Every remaining prediction difference is therefore attributable to the three rival post-integration mechanisms: (i) ERC's EXPECTATION-REFERENCED SATURATION (tanh about theta = logit(v*)^gamma, v* in [.80,.90]) versus PWA's linear normalised validity weights; (ii) ERC's CONTRAST-COUNT normalisation (divide by c0+kap*K, completely background-blind) versus PWA's ENDORSEMENT-DENSITY dilution (divide by c0+kap*T, completely contrast-count-blind); (iii) ERC's COMPRESSED numerosity charge theta*|m|^zeta versus PWA's LINEAR faint-praise discount theta*m.\n\nMid-parameter quantities. ERC (gamma=1.5, v*=.85 -> theta=2.284, alpha=1, sl=2.3, zeta=.7, c0=.425, kap=.65, beta=.62): saturated column values [4.55, 4.22, 2.28, 0.37, 0.11, 0.05] against a per-endorsement charge of 2.28. PWA (gamma=1.5, theta=.108, c0=.09, kap=1.7, beta=3.8, eps=.125): normalised weights [.406, .283, .184, .086, .035, .007].\n\n12 base pairs, each mirrored (A/B swapped) so neither response key is systematically correct: 24 unique pairs, K = 4 repetitions, 96 trials.\n\nFAMILY T \ration - PURE DENSITY LADDER WITH K, CORE AND COUNT DIFFERENCE PINNED (pairs 1, 5, 6 + mirrors). The two options differ on col0 (.95) and ONLY col0, with a shared background of k = 0, 2, 5 ones, so K=1, m=+1 and the weighted core are literally identical in all three cells; only total endorsement density T (=.083/.417/.917) moves. ERC-SB is background-blind and must be PERFECTLY FLAT at ~.78. PWA-DD divides by c0+kap*T and predicts a large monotone decay ~.93 / .77 / .65. Flat versus a ~28-point decline, with everything else pinned, is the cleanest possible test of density dilution.\n\nFAMILY K - PURE CONTRAST-COUNT LADDER WITH DENSITY, COUNTS AND PWA-CORE PINNED (pairs 7, 8, 9 + mirrors). All three cells have sum(a)=sum(b)=3 (so m=0: BOTH discount terms are exactly inert) and T=.5 exactly, and the PWA weighted core is matched to within 10% (.320 / .348 / .350), while the number of discriminating columns is K = 2, 4, 6. PWA-DD has no contrast-count term and must be FLAT (~.75 / .77 / .77, if anything rising). ERC-SB divides by c0+kap*K (1.725 / 3.025 / 4.325) and predicts a steep monotone compression ~.82 / .70 / .60. This is the exact mirror-image of Family T, so the two families cross: whichever model is right must be flat on one ladder and steep on the other.\n\nFAMILY V - SATURATION FLOOR VERSUS GRADED LADDER (pairs 1, 2, 3, 4 + mirrors). One option carries exactly one '1' (on col0=.95, col2=.85, col3=.74 or col5=.55) and the other is all zeros; K=1 and m=+1 throughout, so ERC's normaliser and charge are identical and only the cue's subjective value moves. ERC-SB predicts .78 / .50 / .25 / .22: a saturated FLOOR in which the .74 and .55 experts are essentially indistinguishable (<4 points apart) because both are clearly below the endorsement reference, and the .85 expert sits at almost exactly chance (it straddles the v* box, so this cell should also show the LARGEST between-subject spread in the design). PWA-DD predicts a SMOOTH GRADED ladder .93 / .74 / .42 / .20 - a 22-point separation between col3 and col5 and a solidly above-chance col2. The two critical contrasts are V3-vs-V5 (flat vs 22 points) and V2 (chance vs clearly above chance).\n\nFAMILY S - POSITIVE-BRANCH SATURATION (pairs 10, 11 + mirrors). Only two columns discriminate, counts are equal (m=0, both discount terms inert) and density is identical, so this is a pure cue-value contrast. S1 pits col0 (.95) against col1 (.91): both are far ABOVE the reference, so ERC's tanh flattens them to near-equality (value gap 0.34 out of ~4.5) and it predicts near-chance ~.53-.60; PWA's weights differ substantially (.406 vs .283) and it predicts ~.74. S2 pits col0 against col2 (.85): ERC ~.69, PWA ~.86. The ERC-specific quantitative claim is the RATIO: the .95-vs-.91 contrast must be far closer to chance relative to the .95-vs-.85 contrast (ERC value ratio .15) than any linear validity weighting allows (PWA ratio .55).\n\nFAMILY D - DOMINANCE ANCHOR / NUMEROSITY COMPRESSION (pair 12 + mirror): [1,1,1,1,1,1] vs [0,0,0,0,0,0]. PWA-DD subtracts a linear 6*theta from a core of 1 and divides by c0+kap*.5, leaving a comfortably above-chance ~.77-.85 - this cell also pins its beta and eps, making Families T, K, V and S out-of-sample predictions rather than free fits. ERC-SB's COMPRESSED charge theta*6^zeta (~5.0-10.1) eats most of the summed saturated evidence (~11.6) and K=6 crushes the normaliser, so it predicts a striking NEAR-CHANCE (.55-.65) response to a fully dominant option.\n\nDECISION RULE. If ERC-SB is right: Family T is perfectly flat across the shared endorsement background; Family K falls steeply (~20 points) with the number of discriminating columns; Family V shows a saturated floor (V3 = V5) with the .85 cell at chance; S1 is near chance while S2 is clearly above it; and full dominance is only weakly above chance. If PWA-DD is right: Family T declines steeply and monotonically; Family K is flat; Family V is a smooth graded ladder separating the three weak experts; S1 is clearly above chance; and dominance is decisively above chance.",
  "validities": [
    0.95,
    0.91,
    0.85,
    0.74,
    0.64,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
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
      0,
      0
    ],
    [
      0,
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
      0,
      0
    ],
    [
      0,
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
      0,
      0,
      0
    ],
    [
      0,
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
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
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
      1,
      1
    ],
    [
      0,
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
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
      1,
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
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      1,
      1
    ],
    [
      0,
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
      0,
      0
    ],
    [
      1,
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
      0,
      0
    ],
    [
      0,
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
      0,
      0
    ],
    [
      0,
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
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
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
      0,
      0,
      0
    ],
    [
      0,
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
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
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
      1,
      1
    ],
    [
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
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
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
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
      0,
      0
    ],
    [
      1,
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
  "input_tokens": 28552,
  "output_tokens": 17886
}
```
