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
**Validities (n_features=6):** [0.95, 0.86, 0.78, 0.7, 0.62, 0.54]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 9: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 10: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  trial 12: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 19: A=[0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 20: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 21: A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 22: A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  trial 23: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 24: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate PWA-DD (advocated) from ERC-SB (competing). Both are compensatory and share logit-validity diagnosticity, and both contain the same readability gate, so I remove the readability nuisance entirely by displaying six experts in strictly DESCENDING validity (.95,.86,.78,.70,.62,.54): Kendall tau = -1 (incongruence 0) and the best expert sits in an extreme column, so mu = mu0 (<=.14 / <=.10) for both models. ERC's super-linear binding-failure term is therefore inert and its normaliser is exactly c0+kappa*K. All remaining differences are attributable to the three rival post-integration mechanisms: PWA-DD's endorsement-DENSITY dilution (divide by c0+kap*T, background-sensitive, contrast-count-blind) with a LINEAR faint-praise discount theta calibrated to the weakest cues, versus ERC-SB's expectation-referenced saturating cue values (everything below v* in [.80,.90] is strongly and near-EQUALLY counter-diagnostic), its COMPRESSED numerosity charge theta*|m|^zeta, and its CONTRAST-COUNT normalisation (divide by c0+kap*K, density-blind).

Mid-parameter PWA weights (gamma=1.5, normalised): [.500,.242,.141,.077,.034,.006]; theta ~ .108. Mid-parameter ERC saturated values (v*=.85,gamma=1.5,sl=2.3,alpha=1): [4.55,2.65,0.69,0.21,0.09,0.05] against a reference charge theta = 2.28 per endorsement. 12 base pairs, each mirrored (A/B swapped) so neither key is systematically correct: 24 unique pairs, K = 4 repetitions, 96 trials.

FAMILY V - SINGLE ENDORSEMENT vs AN EMPTY PANEL, SWEPT ACROSS CUE VALIDITY (pairs 1-6 + mirrors). One option carries exactly one '1' (on col0..col5 in turn), the other is all zeros; K=1 and m=+1 in every cell, so ERC's normaliser and charge are identical throughout and only the cue's subjective value moves. PWA-DD predicts a SMOOTH GRADED LADDER that crosses chance between col2 and col3 and keeps descending: ~ .94 / .85 / .62 / .39 / .26 / .20. ERC-SB predicts a STEP followed by a SATURATED FLOOR: ~ .77 / .55 / .29 / .24 / .23 / .23 - cols 2-5 are all sub-reference and, because of the tanh saturation, essentially indistinguishable, and col2 (.78) is REVERSED relative to PWA-DD. The two critical contrasts are (i) V3 vs V5: PWA-DD says a ~19-point spread among the three weakest experts, ERC-SB says <2 points; (ii) V2: PWA-DD says at-or-above chance, ERC-SB says clearly below chance.

FAMILY T - PURE DENSITY LADDER WITH K, CORE AND COUNT DIFFERENCE HELD EXACTLY CONSTANT (pairs 1, 7, 9, 8 + mirrors: shared background k = 0, 1, 2, 5). In all four cells the options differ on col0 and ONLY col0 (K=1, m=+1), so ERC's evidence, charge and normaliser are literally identical and it must be COMPLETELY FLAT at ~.77. PWA-DD divides by c0+kap*T with T = .083/.25/.417/.917 and predicts a large monotone decay ~ .94 / .89 / .82 / .69 that never crosses chance. Flat-versus-25-point-decline with everything else pinned is the cleanest possible test of density dilution against a background-blind account.

FAMILY K - PURE CONTRAST-COUNT LADDER WITH DENSITY, COUNT DIFFERENCE AND CORE HELD ~CONSTANT (pairs 9, 10, 11 + mirrors). All three cells have sum(a)=3, sum(b)=2 (so T=5/12 and m=+1 are identical, making PWA's dilution and discount identical), the col0 (.95) contrast always favours A, and the extra discriminating columns are drawn from the saturated weak cluster so that they nearly cancel: K = 1, 3, 5. PWA-DD has NO contrast-count term and predicts a flat ~ .82 / .84 / .77. ERC-SB divides by c0+kap*K (1.08 / 2.38 / 3.68) and predicts a steep monotone compression ~ .77 / .64 / .57. This is the mirror-image test of Family T and is diagnostic even if one doubts my density claim.

FAMILY D - DOMINANCE ANCHOR (pair 12 + mirror): [1,1,1,1,1,1] vs [0,0,0,0,0,0]. Under PWA-DD the linear discount removes 6*theta from a core of 1 and the density T=.5 dilutes, leaving a clearly above-chance ~.77-.85. Under ERC-SB the compressed charge theta*6^zeta (~8.0-11.4) very nearly cancels or exceeds the summed saturated evidence (~7.8-8.4) and K=6 crushes the normaliser, so it predicts a striking NEAR-CHANCE (.45-.57) response to a fully dominant option. A high-accuracy dominance cell also pins beta and eps for PWA-DD, making Families V, T and K out-of-sample predictions rather than free fits.

DECISION RULE. If PWA-DD is right: Family V is a smooth graded ladder with the three weakest experts clearly separated and col2 at/above chance; Family T declines steeply and monotonically with shared endorsement background; Family K is FLAT across contrast counts; dominance is comfortably above chance. If ERC-SB is right: Family V shows a saturated floor (V2=V3=V4=V5) with col2 already strongly reversed; Family T is perfectly FLAT; Family K falls steeply with the number of discriminating columns; and the dominance pair sits near chance. Between-subject spread should be largest at V1 (col1 = .86 straddles ERC's v* box) and at V2/K1 (PWA's w2 straddles its theta box) under the respective accounts.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Weighted mean 'target rate' over a selected set of stimulus cells.\n    # 'Target' on a trial = the option carrying the 1 in the LEFTMOST column\n    # on which the two panels disagree (i.e. the option endorsed by the most\n    # valid discriminating expert).  Cells are keyed mirror-invariantly as\n    # \"<target panel>|<other panel>\".\n    W = {\n        \"100000|000000\": 1.0,   # V0 : lone .95 endorsement vs empty panel\n        \"010000|000000\": 2.0,   # V1 : lone .86 endorsement vs empty panel\n        \"001000|000000\": 2.0,   # V2 : lone .78 endorsement vs empty panel\n        \"000100|000000\": 1.0,   # V3 : lone .70 endorsement vs empty panel\n        \"110000|010000\": 0.8,   # T1 : col0 contrast on a 1-endorsement background\n        \"110100|010010\": 1.5,   # K3 : three discriminating columns, col0 leads\n        \"100101|001010\": 1.5,   # K5 : five discriminating columns, col0 leads\n        \"111111|000000\": 2.0,   # D  : full dominance\n    }\n\n    hits = {}\n    tot = {}\n\n    a_col = data[\"option_a_ratings\"].tolist()\n    b_col = data[\"option_b_ratings\"].tolist()\n    r_col = data[\"response\"].tolist()\n\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        try:\n            a = [int(round(float(x))) for x in list(a_raw)]\n            b = [int(round(float(x))) for x in list(b_raw)]\n        except Exception:\n            continue\n        n = min(len(a), len(b))\n        if n == 0:\n            continue\n        a = a[:n]\n        b = b[:n]\n        first = -1\n        for j in range(n):\n            if a[j] != b[j]:\n                first = j\n                break\n        if first < 0:\n            continue\n        if a[first] == 1:\n            tgt = a\n            oth = b\n            tgt_resp = 0\n        else:\n            tgt = b\n            oth = a\n            tgt_resp = 1\n        key = \"\".join([str(v) for v in tgt]) + \"|\" + \"\".join([str(v) for v in oth])\n        if key not in W:\n            continue\n        try:\n            rr = int(resp)\n        except Exception:\n            continue\n        tot[key] = tot.get(key, 0) + 1\n        if rr == tgt_resp:\n            hits[key] = hits.get(key, 0) + 1\n        else:\n            hits[key] = hits.get(key, 0)\n\n    num = 0.0\n    den = 0.0\n    for k, w in W.items():\n        if tot.get(k, 0) > 0:\n            rate = float(hits.get(k, 0)) / float(tot[k])\n            num += w * rate\n            den += w\n    if den <= 0.0:\n        return float(\"nan\")\n    return float(num / den)",
  "rationale": "PWA-DD and ERC-SB differ systematically in HOW OFTEN the option endorsed by the leading discriminating expert is chosen, and the differences are largest in a specific subset of cells. Rather than betting everything on one fragile per-cell contrast (8 trials/cell -> binomial sd ~0.15), I aggregate a weighted average of 'target rates' across eight cells whose predicted PWA-minus-ERC gaps are all positive and sizeable: lone-endorsement cells V0/V1/V2/V3 (PWA: smooth graded ladder .94/.85/.62/.39; ERC: step then saturated floor .77/.55/.29/.24), the contrast-count cells K3/K5 (PWA flat ~.84/.77 because it has no K normaliser; ERC compressed to ~.64/.57), the one-endorsement-background col0 contrast T1 (.89 vs .77) and the dominance pair (PWA comfortably above chance ~.81; ERC crushed near chance ~.51 by its compressed numerosity charge plus K=6 normalisation). Weights are proportional to the expected gap, so the metric is essentially the optimal linear discriminant over cells. Predicted values: ~0.75 under PWA-DD vs ~0.51 under ERC-SB (gap ~0.24). Because it pools ~64 trials per subject, binomial noise contributes only sd ~0.055 per subject, and the remaining between-subject spread comes from determinacy/lapse heterogeneity, which is modest relative to the mean gap. The metric is computable identically on the pooled frame and on any single subject's slice."
}
```

## Usage

```json
{
  "input_tokens": 15262,
  "output_tokens": 12679
}
```
