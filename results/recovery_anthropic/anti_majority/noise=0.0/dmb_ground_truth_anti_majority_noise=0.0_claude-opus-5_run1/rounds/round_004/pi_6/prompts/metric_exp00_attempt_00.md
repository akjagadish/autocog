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
**Validities (n_features=6):** [0.95, 0.91, 0.85, 0.74, 0.64, 0.55]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 9: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 13: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 14: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 15: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 16: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 17: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 18: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 19: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 20: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 22: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate ERC-SB (advocated) from PWA-DD (competing). Both are compensatory, share logit-validity diagnosticity and contain the SAME readability gate, so I remove the readability nuisance entirely: six experts are displayed in strictly DESCENDING validity (.95,.91,.85,.74,.64,.55). Kendall tau = -1 (incongruence 0) and the best expert sits in an extreme column, so mu = mu0 (<=.10 / <=.14) for both models; ERC's super-linear binding-failure term is inert (its normaliser reduces to c0+kap*K) and PWA's positional fallback is inert. Every remaining prediction difference is therefore attributable to the three rival post-integration mechanisms: (i) ERC's EXPECTATION-REFERENCED SATURATION (tanh about theta = logit(v*)^gamma, v* in [.80,.90]) versus PWA's linear normalised validity weights; (ii) ERC's CONTRAST-COUNT normalisation (divide by c0+kap*K, completely background-blind) versus PWA's ENDORSEMENT-DENSITY dilution (divide by c0+kap*T, completely contrast-count-blind); (iii) ERC's COMPRESSED numerosity charge theta*|m|^zeta versus PWA's LINEAR faint-praise discount theta*m.

Mid-parameter quantities. ERC (gamma=1.5, v*=.85 -> theta=2.284, alpha=1, sl=2.3, zeta=.7, c0=.425, kap=.65, beta=.62): saturated column values [4.55, 4.22, 2.28, 0.37, 0.11, 0.05] against a per-endorsement charge of 2.28. PWA (gamma=1.5, theta=.108, c0=.09, kap=1.7, beta=3.8, eps=.125): normalised weights [.406, .283, .184, .086, .035, .007].

12 base pairs, each mirrored (A/B swapped) so neither response key is systematically correct: 24 unique pairs, K = 4 repetitions, 96 trials.

FAMILY T ation - PURE DENSITY LADDER WITH K, CORE AND COUNT DIFFERENCE PINNED (pairs 1, 5, 6 + mirrors). The two options differ on col0 (.95) and ONLY col0, with a shared background of k = 0, 2, 5 ones, so K=1, m=+1 and the weighted core are literally identical in all three cells; only total endorsement density T (=.083/.417/.917) moves. ERC-SB is background-blind and must be PERFECTLY FLAT at ~.78. PWA-DD divides by c0+kap*T and predicts a large monotone decay ~.93 / .77 / .65. Flat versus a ~28-point decline, with everything else pinned, is the cleanest possible test of density dilution.

FAMILY K - PURE CONTRAST-COUNT LADDER WITH DENSITY, COUNTS AND PWA-CORE PINNED (pairs 7, 8, 9 + mirrors). All three cells have sum(a)=sum(b)=3 (so m=0: BOTH discount terms are exactly inert) and T=.5 exactly, and the PWA weighted core is matched to within 10% (.320 / .348 / .350), while the number of discriminating columns is K = 2, 4, 6. PWA-DD has no contrast-count term and must be FLAT (~.75 / .77 / .77, if anything rising). ERC-SB divides by c0+kap*K (1.725 / 3.025 / 4.325) and predicts a steep monotone compression ~.82 / .70 / .60. This is the exact mirror-image of Family T, so the two families cross: whichever model is right must be flat on one ladder and steep on the other.

FAMILY V - SATURATION FLOOR VERSUS GRADED LADDER (pairs 1, 2, 3, 4 + mirrors). One option carries exactly one '1' (on col0=.95, col2=.85, col3=.74 or col5=.55) and the other is all zeros; K=1 and m=+1 throughout, so ERC's normaliser and charge are identical and only the cue's subjective value moves. ERC-SB predicts .78 / .50 / .25 / .22: a saturated FLOOR in which the .74 and .55 experts are essentially indistinguishable (<4 points apart) because both are clearly below the endorsement reference, and the .85 expert sits at almost exactly chance (it straddles the v* box, so this cell should also show the LARGEST between-subject spread in the design). PWA-DD predicts a SMOOTH GRADED ladder .93 / .74 / .42 / .20 - a 22-point separation between col3 and col5 and a solidly above-chance col2. The two critical contrasts are V3-vs-V5 (flat vs 22 points) and V2 (chance vs clearly above chance).

FAMILY S - POSITIVE-BRANCH SATURATION (pairs 10, 11 + mirrors). Only two columns discriminate, counts are equal (m=0, both discount terms inert) and density is identical, so this is a pure cue-value contrast. S1 pits col0 (.95) against col1 (.91): both are far ABOVE the reference, so ERC's tanh flattens them to near-equality (value gap 0.34 out of ~4.5) and it predicts near-chance ~.53-.60; PWA's weights differ substantially (.406 vs .283) and it predicts ~.74. S2 pits col0 against col2 (.85): ERC ~.69, PWA ~.86. The ERC-specific quantitative claim is the RATIO: the .95-vs-.91 contrast must be far closer to chance relative to the .95-vs-.85 contrast (ERC value ratio .15) than any linear validity weighting allows (PWA ratio .55).

FAMILY D - DOMINANCE ANCHOR / NUMEROSITY COMPRESSION (pair 12 + mirror): [1,1,1,1,1,1] vs [0,0,0,0,0,0]. PWA-DD subtracts a linear 6*theta from a core of 1 and divides by c0+kap*.5, leaving a comfortably above-chance ~.77-.85 - this cell also pins its beta and eps, making Families T, K, V and S out-of-sample predictions rather than free fits. ERC-SB's COMPRESSED charge theta*6^zeta (~5.0-10.1) eats most of the summed saturated evidence (~11.6) and K=6 crushes the normaliser, so it predicts a striking NEAR-CHANCE (.55-.65) response to a fully dominant option.

DECISION RULE. If ERC-SB is right: Family T is perfectly flat across the shared endorsement background; Family K falls steeply (~20 points) with the number of discriminating columns; Family V shows a saturated floor (V3 = V5) with the .85 cell at chance; S1 is near chance while S2 is clearly above it; and full dominance is only weakly above chance. If PWA-DD is right: Family T declines steeply and monotonically; Family K is flat; Family V is a smooth graded ladder separating the three weak experts; S1 is clearly above chance; and dominance is decisively above chance.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    # ---- cell definitions -------------------------------------------------\n    # Each entry is (designated_option, other_option).  We score the fraction\n    # of trials on which the subject chose the designated option.\n    # Group P: cells where PWA-DD predicts a MUCH higher rate than ERC-SB.\n    P_list = [\n        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)),   # V0 : lone .95 cue, sparse\n        ((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0)),   # V2 : lone .85 cue (ERC ~ chance)\n        ((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0)),   # V3 : lone .74 cue\n        ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)),   # S1 : .95 vs .91 (ERC saturated)\n        ((1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)),   # S2 : .95 vs .85\n        ((1, 0, 1, 1, 0, 0), (0, 1, 0, 0, 1, 1)),   # K6 : six discriminating columns\n        ((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)),   # D  : full dominance\n    ]\n    # Group N: cells where ERC-SB predicts a HIGHER rate than PWA-DD.\n    N_list = [\n        ((1, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 1)),   # lone .95 cue on a dense background\n        ((1, 1, 0, 0, 1, 0), (0, 1, 0, 1, 1, 0)),   # K2 : only two discriminating columns\n    ]\n\n    def sig(u, v):\n        return tuple(sorted([tuple(u), tuple(v)]))\n\n    Pmap = {}\n    for w, o in P_list:\n        Pmap[sig(w, o)] = w\n    Nmap = {}\n    for w, o in N_list:\n        Nmap[sig(w, o)] = w\n\n    p_hit = 0.0\n    p_n = 0.0\n    n_hit = 0.0\n    n_n = 0.0\n\n    for a_raw, b_raw, r in zip(data['option_a_ratings'].values,\n                               data['option_b_ratings'].values,\n                               data['response'].values):\n        try:\n            a = tuple(int(round(float(x))) for x in a_raw)\n            b = tuple(int(round(float(x))) for x in b_raw)\n        except Exception:\n            continue\n        k = sig(a, b)\n        win = Pmap.get(k, None)\n        grp = 'P'\n        if win is None:\n            win = Nmap.get(k, None)\n            grp = 'N'\n        if win is None:\n            continue\n        try:\n            resp = int(r)\n        except Exception:\n            continue\n        chosen = a if resp == 0 else b\n        hit = 1.0 if chosen == win else 0.0\n        if grp == 'P':\n            p_hit += hit\n            p_n += 1.0\n        else:\n            n_hit += hit\n            n_n += 1.0\n\n    if p_n == 0.0 and n_n == 0.0:\n        return float('nan')\n    if p_n == 0.0:\n        return float(0.5 - (n_hit / n_n))\n    if n_n == 0.0:\n        return float((p_hit / p_n) - 0.75)\n    return float((p_hit / p_n) - (n_hit / n_n))\n",
  "rationale": "The design contains cells whose predicted choice rates move in OPPOSITE directions under the two theories, so I build a single contrast that adds those gaps rather than differencing noisy single cells.\n\nWorking through both models at mid-range parameters (validities [.95,.91,.85,.74,.64,.55], display validity-sorted so both readability gates are inert):\n\nGroup P (PWA >> ERC): lone .95 cue vs empty (ERC .78 / PWA .93); lone .85 cue vs empty (ERC ~.50, exactly at chance because .85 sits at the expectation reference / PWA .74); lone .74 cue vs empty (ERC .25 / PWA .44); .95-vs-.91 (ERC ~.53, tanh saturation flattens the two strong experts / PWA .74); .95-vs-.85 (ERC .69 / PWA .85); the six-discriminating-column conflict (ERC .60 because the normaliser c0+kap*K explodes / PWA .76, contrast-count-blind); and full dominance (ERC .62 because the compressed numerosity charge theta*6^zeta eats the evidence / PWA .77).\n\nGroup N (ERC > PWA): the same lone .95 advantage on a fully dense background (ERC .78, background-blind / PWA .64, density-diluted) and the two-column conflict (ERC .81 / PWA .74).\n\nMetric = mean(P hits) - mean(N hits). Predicted: ERC-SB ~ -0.22, PWA-DD ~ +0.06, a ~0.28 gap. Crucially the P group pools 56 trials per subject and mixes cells that move in opposite directions under determinacy (beta) heterogeneity, so individual-parameter noise partially cancels: recomputing ERC at beta=0.40 vs 0.88 shifts the P mean only from .55 to .59, and eps/beta heterogeneity in PWA compresses everything toward .5 symmetrically. That gives a large mean separation with a tight per-subject distribution."
}
```

## Usage

```json
{
  "input_tokens": 15549,
  "output_tokens": 11308
}
```
