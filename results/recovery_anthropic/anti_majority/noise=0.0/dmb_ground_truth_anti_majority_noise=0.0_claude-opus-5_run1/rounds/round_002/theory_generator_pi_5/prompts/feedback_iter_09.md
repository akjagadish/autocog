# feedback_iter_09

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace Take-The-Best entirely with a compensatory, display-bound weighted-additive theory WITHOUT any coherence/unanimity term. Sketch:

**Name:** Position-Bound Weighted Additive integration with faint-praise endorsement discounting (PWA-FP).

**Core:** D = \u03a3_j w_eff_j (a_j - b_j) - \u03b8\u00b7(\u03a3a - \u03a3b); p(A) = (1-\u03b5)\u00b7logistic(\u03b2 D) + \u03b5/2. All cues are always integrated (no stopping rule) \u2014 this is what buys Exp 4's chance-level conflict performance and Exp 3's ~0.75.

**Weights:** w_j is a convex mixture of (i) a validity-derived weight (e.g. \u221d logit(v_j)^\u03b3) bound to the correct expert, and (ii) a purely positional weight that decays left-to-right over displayed columns, w_pos \u221d (n - j)^\u03bb. The mixture coefficient is a *free* readability parameter that should be allowed to depend on how easy the validity-to-column mapping is (near-pure validity weighting when the display is monotone in validity, near-pure positional weighting when it is scrambled, best cue in a middle column). This is deliberately close to pi_4's kappa mechanism (which the data support: Exp 5 obs -0.208, Exp 6 obs 0.659) but formulated as an explicit position-vs-validity mixture rather than a flattening-toward-equal plus a small gradient, so it can generate the strong positional dominance seen in Exp 6 (0.66) while still letting the .93 cue win in Exp 5's monotone display.

**Crucially, no consensus/coherence premium.** The rho\u00b7|2\u00b7mean-1|^q term must be dropped outright. In its place put a small, strictly MONOTONE endorsement-count discount \u03b8 (0 to ~0.05 of total weight): each extra '1' slightly lowers an option's appeal (faint-praise skepticism). This reproduces (a) Exp 1's ~0 contrast (both the 5/5-vs-4/5 cell and the single-weak-endorsement-vs-empty cell are shifted in the *same* direction, so the difference stays near zero, and can be very slightly negative), (b) Exp 2's small negative -0.15 (U4-U1 and V2-V1 near zero or slightly negative because \u03b8 penalises the higher-endorsement option equally in both cells, while L1-L4 is small once flattened weights let the 4-cue coalition partly offset the .93 cue), and (c) Exp 5's negative agree-minus-conflict value via near-chance/sub-chance responding on the weak-single-endorsement-vs-empty pair, without ever predicting the huge unanimity steps that Exps 1 and 2 refuted. Constrain \u03b8 so that the dominance anchor ([1..1] vs [0..0]) stays at ceiling.

**Parameter ranges:** \u03b2 wide (2\u201312), \u03b3 ~0.8\u20132.0, positional-mixture weight wide (0\u20130.9) and tied to display monotonicity, \u03b8 small (0\u20130.06), \u03b5 0\u20130.15. No stopping/frugality parameter, no q, no rho.

This new theory competes with pi_4 on exactly the mechanism the data can still adjudicate \u2014 smooth endorsement-count effects versus all-or-none unanimity, and flattening-toward-equal versus position-sorted re-weighting \u2014 while both remain compensatory, so future experiments will be genuinely diagnostic rather than re-testing frugality, which is now settled.


## CANDIDATE THEORY
**PWA-DD: Position-Bound Weighted Additive integration with Faint-Praise discounting and Endorsement-Density Dilution.**

Choice in binary-cue environments is always *compensatory*: on every trial the decision maker reads all displayed columns, forms a single weighted difference, and never stops early. There is no lexicographic cascade and no coherence/unanimity premium. Three claims fix the rest of the mechanism.

(1) **Weights are a mixture of validity and display position, and the mixture is governed by how readable the validity-to-column mapping is.** Stated validities must be bound to *columns*. When the display is perfectly monotone in validity (ascending or descending) that binding is trivial and people use validity-derived weights w_val ∝ logit(v_j)^γ essentially intact; the residual positional leak μ0 is *small* and person-specific, and in a substantial fraction of people is exactly zero — readable displays are read almost purely by stated validity. When the column order is not monotone in validity — and especially when the best expert sits in an interior column — binding fails and people fall back almost entirely on a purely positional gradient w_pos ∝ (n−j)^λ that decays left-to-right over displayed columns. The mixture weight μ = μ0 + (1−μ0)·min(1, μ1(1−|τ|) + μ2·[best cue interior]) is therefore an *environmental* quantity, not a per-experiment fudge: it is inert (μ = μ0) whenever columns are validity-sorted, and it predicts that scrambling a sorted display converts validity-driven behaviour into position-driven behaviour and that re-sorting a scrambled display restores it. Crucially the gate SATURATES: any display that is appreciably non-monotone is read essentially positionally, with no graded validity leak surviving.

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
    mu0 = float(np.clip(float(parameters.get("mu0", 0.04)), 0.0, 1.0))
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

`parameters`:
- gamma: [1.1, 1.9]
- beta: [2.0, 5.6]
- theta: [0.07, 0.145]
- c0: [0.05, 0.13]
- kap: [1.1, 2.3]
- mu0: [0.0, 0.08]
- mu1: [1.4, 2.2]
- mu2: [0.20, 0.60]
- lam: [0.70, 1.20]
- eps: [0.03, 0.22]
- validities: validities

`rationale`:
Single-knob edit on the iter-9 running-best base. `predict`/`policy` are mechanically identical (only the inert `mu0` `.get()` default changed, 0.07 -> 0.04); ONE parameter box changes: **mu0 [0.0, 0.14] (mid 0.07) -> [0.0, 0.08] (mid 0.04)**. gamma, beta, theta, c0, kap, mu1, mu2, lam, eps are untouched.

WHY I DEVIATE FROM THE CRITIC'S SPECIFIC KNOB (gamma up) BUT NOT FROM ITS DIAGNOSIS. The critic's diagnosis is exactly right: the loss is now concentrated in Exp3 (+0.0715, contrast too shallow) and Exp5 (+0.065, hi-minus-lo too positive), and Exp1 has flipped to slight UNDER-shoot (0.7408 vs 0.7533). But gamma-up pays for Exp3 with Exp5, by arithmetic that is sign-certain: on Exp5's monotone descending display the lo cells are f((w2-theta)/0.26) and f((w4-theta)/0.26), both with negative numerators; raising gamma lowers w2 (0.109 -> 0.0945) and w4 (0.0159 -> 0.0120), driving both lo cells further sub-chance while the dense hi cell (denominator 1.62) barely moves. Mid-box hand-check at gamma 1.5 -> 1.7: hi-lo widens 0.078 -> 0.104 (+0.026), which in squared-residual terms cancels the ~0.040 Exp3 gain almost exactly. The critic's expectation that Exp5 would 'tighten' rests on the iter-8 mirror, but the critic itself flags Exp5 jitter at +-0.03, and the iter-8 Exp4 movement (0.011 on an experiment that is arithmetically inert to gamma) pins the noise floor, so that mirror is not evidence.

WHY mu0 IS NOW OPEN AGAIN. The critic closed mu0 because it 'trades Exp1 against Exp3 one-for-one'. That verdict was formed when Exp1 was +0.045 too HIGH. The iter-9 theta pass overshot Exp1 to -0.0125 BELOW the observed 0.753, so the sign of the trade has flipped: mu0 down now moves Exp1 UP toward the target AND deepens Exp3 at the same time. The premise of the closure no longer holds.

HAND-CHECK (mid-box gamma=1.5, beta=3.8, theta=0.1075, c0=0.09, kap=1.7, lam=0.95, eps=0.125), mu0 0.07 -> 0.05 shown, edit is to 0.04. Exp3 ascending weights go [.0338,.0601,.1089,.2870,.5103] -> [.0274,.0557,.1069,.2902,.5196]. Conflict cells all rise (A .924->.926, B .834->.840, E .911->.914, F .663->.672; mean .833->.838) because purer validity weighting on an ascending display increases validity-following; the weakest-cue agree cell H falls .284->.270 (its numerator w0-theta becomes more negative), so agree falls .680->.675. Contrast -0.153 -> -0.163 at mid-box for a -0.02 step, i.e. -0.014 at the -0.03 step actually submitted. Exp1 (descending) moves the right way too: conflict mean .7766 -> .7792 for -0.02 (types 1/3/5/13 = .786/.703/.899/.729), i.e. Exp1 rises off its undershoot rather than overshooting (well below the 0.79 guardrail).

COST AND GUARDRAILS. Exp5 is the only fit that pays: mu0 down slightly lowers w2/w4 on the descending display, hi-lo 0.0784 -> 0.0827 at -0.02 (about a third of the damage gamma-up would do for a comparable Exp3 gain). Exps 2, 4 and 6 are protected by construction: Exp4 (tau=0.067, interior best) and Exp6 (interior best) saturate mu=1, so mu0 cancels exactly; Exp2 (tau=-0.333) has mu1*incong>=0.93, so mu changes by <0.003 across the whole mu0 box - Exp6's -0.1575 and Exp2's 0.509 are untouched. The dominance anchor is completely unaffected (num = 1-5*theta = 0.462, denom 0.94, p ~ 0.82). Expected net in squared-residual terms: Exp3 0.0715 -> ~0.03-0.045, Exp1 0.0125 -> ~0.008, against Exp5 0.065 -> ~0.072 - a clear aggregate gain that should land below the 0.0370 floor, and the pass changes exactly one box so it stays cleanly attributable, the property every ACCEPTED iteration in this loop shared.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0557 -> ACCEPTED
- iter 2: loss=0.0484 -> ACCEPTED
- iter 3: loss=0.0691 -> REJECTED
- iter 4: loss=0.0475 -> ACCEPTED
- iter 5: loss=0.0384 -> ACCEPTED
- iter 6: loss=0.0428 -> REJECTED
- iter 7: loss=0.0457 -> REJECTED
- iter 8: loss=0.0420 -> REJECTED
- iter 9: loss=0.0370 -> ACCEPTED
- iter 10 (current candidate you are grading): loss=0.0487 -> REJECTED
Running-best (last accepted) base: iter 9 at loss=0.0370.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7875 (var=0.0059) (Δ vs real +0.0342)
  - iter 2: 0.7979 (var=0.0078) (Δ vs real +0.0446)
  - iter 3: 0.8383 (var=0.0046) (Δ vs real +0.0850)
  - iter 4: 0.7762 (var=0.0093) (Δ vs real +0.0229)
  - iter 5: 0.7979 (var=0.0053) (Δ vs real +0.0446)
  - iter 6: 0.7483 (var=0.0064) (Δ vs real -0.0050)
  - iter 7: 0.7783 (var=0.0084) (Δ vs real +0.0250)
  - iter 8: 0.7638 (var=0.0077) (Δ vs real +0.0104)
  - iter 9: 0.7408 (var=0.0064) (Δ vs real -0.0125)
  - iter 10 (current): 0.7767 (var=0.0091) (Δ vs real +0.0233)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8375 (var=0.0108)
- pi_2: 0.1425 (var=0.0084)
- pi_3: 0.7375 (var=0.0102)
- pi_4: 0.7196 (var=0.0166)

### Experiment 2
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
**Candidate trajectory (this loop):**
  - iter 1: 0.5004 (var=0.0064) (Δ vs real -0.0096)
  - iter 2: 0.5579 (var=0.0085) (Δ vs real +0.0479)
  - iter 3: 0.5225 (var=0.0069) (Δ vs real +0.0125)
  - iter 4: 0.5342 (var=0.0095) (Δ vs real +0.0242)
  - iter 5: 0.5121 (var=0.0087) (Δ vs real +0.0021)
  - iter 6: 0.5267 (var=0.0070) (Δ vs real +0.0167)
  - iter 7: 0.5117 (var=0.0065) (Δ vs real +0.0017)
  - iter 8: 0.5117 (var=0.0076) (Δ vs real +0.0017)
  - iter 9: 0.5092 (var=0.0046) (Δ vs real -0.0008)
  - iter 10 (current): 0.5142 (var=0.0063) (Δ vs real +0.0042)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3058 (var=0.0050)
- pi_1: 0.8204 (var=0.0123)
- pi_3: 0.5371 (var=0.0083)
- pi_4: 0.5204 (var=0.0106)

### Experiment 3
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
**Candidate trajectory (this loop):**
  - iter 1: -0.1257 (var=0.0082) (Δ vs real +0.0824)
  - iter 2: -0.1715 (var=0.0082) (Δ vs real +0.0365)
  - iter 3: -0.2093 (var=0.0088) (Δ vs real -0.0013)
  - iter 4: -0.1771 (var=0.0130) (Δ vs real +0.0310)
  - iter 5: -0.1822 (var=0.0149) (Δ vs real +0.0258)
  - iter 6: -0.1581 (var=0.0101) (Δ vs real +0.0500)
  - iter 7: -0.1590 (var=0.0069) (Δ vs real +0.0490)
  - iter 8: -0.1333 (var=0.0096) (Δ vs real +0.0747)
  - iter 9: -0.1365 (var=0.0095) (Δ vs real +0.0715)
  - iter 10 (current): -0.1497 (var=0.0078) (Δ vs real +0.0583)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0092 (var=0.0059)
- pi_3: 0.6219 (var=0.0370)
- pi_2: 0.4544 (var=0.0220)
- pi_4: -0.2069 (var=0.0176)

### Experiment 4
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7480 (var=0.0085) (Δ vs real +0.0893)
  - iter 2: 0.6840 (var=0.0090) (Δ vs real +0.0253)
  - iter 3: 0.6967 (var=0.0069) (Δ vs real +0.0380)
  - iter 4: 0.6580 (var=0.0097) (Δ vs real -0.0007)
  - iter 5: 0.6460 (var=0.0085) (Δ vs real -0.0127)
  - iter 6: 0.6360 (var=0.0101) (Δ vs real -0.0227)
  - iter 7: 0.6747 (var=0.0118) (Δ vs real +0.0160)
  - iter 8: 0.6567 (var=0.0105) (Δ vs real -0.0020)
  - iter 9: 0.6727 (var=0.0124) (Δ vs real +0.0140)
  - iter 10 (current): 0.6673 (var=0.0106) (Δ vs real +0.0087)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8380 (var=0.0096)
- pi_1: 0.1800 (var=0.0087)
- pi_2: 0.6260 (var=0.0084)
- pi_4: 0.6633 (var=0.0149)

### Experiment 5
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0283 (var=0.0377) (Δ vs real -0.0150)
  - iter 2: 0.0167 (var=0.0396) (Δ vs real +0.0300)
  - iter 3: 0.1033 (var=0.0272) (Δ vs real +0.1167)
  - iter 4: 0.0700 (var=0.0374) (Δ vs real +0.0833)
  - iter 5: 0.0475 (var=0.0192) (Δ vs real +0.0608)
  - iter 6: 0.0483 (var=0.0412) (Δ vs real +0.0617)
  - iter 7: 0.0350 (var=0.0348) (Δ vs real +0.0483)
  - iter 8: 0.0700 (var=0.0263) (Δ vs real +0.0833)
  - iter 9: 0.0517 (var=0.0264) (Δ vs real +0.0650)
  - iter 10 (current): 0.0492 (var=0.0330) (Δ vs real +0.0625)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0150 (var=0.0203)
- pi_4: 0.7083 (var=0.0508)
- pi_2: -0.0175 (var=0.0181)
- pi_3: -0.0750 (var=0.0313)

### Experiment 6
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
**Candidate trajectory (this loop):**
  - iter 1: -0.1425 (var=0.0166) (Δ vs real +0.0075)
  - iter 2: -0.1425 (var=0.0195) (Δ vs real +0.0075)
  - iter 3: -0.1133 (var=0.0156) (Δ vs real +0.0367)
  - iter 4: -0.1133 (var=0.0181) (Δ vs real +0.0367)
  - iter 5: -0.1292 (var=0.0192) (Δ vs real +0.0208)
  - iter 6: -0.1233 (var=0.0174) (Δ vs real +0.0267)
  - iter 7: -0.0950 (var=0.0178) (Δ vs real +0.0550)
  - iter 8: -0.1292 (var=0.0163) (Δ vs real +0.0208)
  - iter 9: -0.1575 (var=0.0195) (Δ vs real -0.0075)
  - iter 10 (current): -0.0967 (var=0.0161) (Δ vs real +0.0533)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4600 (var=0.0383)
- pi_1: -0.0092 (var=0.0076)
- pi_2: 0.1042 (var=0.0121)
- pi_3: 0.0383 (var=0.0077)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: This candidate is by far the best-fitting model seen in this family: it is within ~0.01 of the data on Exp 2 (0.500 vs 0.510), Exp 5 (-0.028 vs -0.013) and Exp 6 (-0.143 vs -0.150), and within ~0.035 on Exp 1 (0.788 vs 0.753). Critically, it simultaneously reproduces the three patterns that split the earlier theories: the near-zero/slightly-negative unanimity contrasts (Exps 5 and 6) that killed pi_4's coherence premium, the moderate positional dominance of Exp 4 that killed pi_1's validity-locked TTB, and the negative agree-minus-conflict of Exp 3 that killed pi_2/pi_3. It is faithful to the arbiter's prescription: compensatory, all-cue integration, position-vs-validity mixture gated by display readability, monotone faint-praise discount \u03b8, no unanimity/coherence term. The one addition (density dilution) is direction-blind and does not smuggle a coherence bonus back in, so it stays inside the family.

Two residuals remain. (i) Exp 3: simulated -0.126 vs observed -0.208 \u2014 the model does not push the sparse 'single weak endorsement vs empty panel' agreement cells far enough sub-chance (or its conflict cells are not high enough), so the contrast is ~40% too shallow. (ii) Exp 4: simulated 0.748 vs observed 0.659 \u2014 the model is *too* positional/too deterministic on scrambled displays with an interior best cue; \u03bc saturates at ~1 and \u03b2 is high enough that positional WADD wins nearly always, whereas humans are noticeably noisier there.

A third, systematic issue: between-subject variance is far too small. Real var is 0.023 (Exp1), 0.014 (Exp2), 0.037 (Exp3), 0.038 (Exp4); the candidate gives 0.006, 0.006, 0.008, 0.009 \u2014 3\u20135x too homogeneous. The parameter boxes are too tight to generate the observed individual spread, and the theory itself claims heterogeneity is continuous in \u03b2, \u03b8, \u03bc, so the ranges should reflect that.
Rationale: The candidate was ACCEPTED and should be the base \u2014 do NOT redesign it. Keep the exact PWA-DD mechanism (weighted-additive, position/validity mixture gated by display readability, monotone \u03b8 discount, density dilution, no coherence term) and make three targeted, in-family knob adjustments:

1. Deepen Exp 3 (currently -0.126 vs -0.208): raise the faint-praise discount slightly, e.g. \u03b8 \u2208 [0.10, 0.18] instead of [0.06, 0.14]. This drives the sparse 'lone weak endorsement vs empty panel' agreement cells further sub-chance while leaving equal-count conflict trials inert, and (because \u0394count = 1 in both U and V cells) it largely cancels in the Exp-6 contrast and moves the two Exp-5 cells together, so the good fits there should survive. Verify the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor stays at ceiling under the upper end of the new \u03b8 box.

2. Soften Exp 4 (currently 0.748 vs 0.659): the readability gate saturates \u03bc \u2192 1 on the scrambled/interior-best display and \u03b2 is too high there. Reduce the saturation slightly \u2014 e.g. lower \u03bc1 to ~[1.0, 1.6] and \u03bc2 to ~[0.15, 0.45], and/or shift the \u03b2 box down to ~[2.2, 5.0] \u2014 so a residual fraction of validity weighting plus more choice noise pulls the positional-vs-TTB hit rate down toward 0.66. Exp 2 (also scrambled) is currently at 0.500 vs 0.510 and a small validity leak nudges it in the right direction, so this is a two-birds change; check Exp 6 does not drift positive.

3. Fix the heterogeneity deficit: widen the boxes so per-subject metric variance approaches the observed 0.014\u20130.038 rather than the current 0.006\u20130.009. Concretely widen \u03b2 (e.g. [2.0, 7.0]), \u03b5 (e.g. [0.03, 0.22]), \u03bc0 (e.g. [0.0, 0.35]) and \u03b8. Since the point estimates are already close, widening with the mid-points held near current values should preserve pooled fit while restoring realistic between-subject spread.

No mechanism-family change is warranted or permitted here \u2014 the prescribed compensatory position-bound WADD with faint-praise discounting is clearly the right family, and this is a calibration pass on top of it.

**Outcome of your advice:** iter 1 candidate loss=0.0557 -> iter 2 candidate loss=0.0484 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted iter-2 candidate is a genuine improvement (0.0557 -> 0.0484) and it remains faithful to the arbiter's prescribed family: all-cue compensatory integration, validity-vs-position mixture gated by display readability, monotone faint-praise discount \u03b8, direction-blind density dilution, no coherence/unanimity premium. It is still the only theory in the comparison set that simultaneously gets Exp 3 negative (-0.17 vs -0.208), Exp 4 moderate (0.684 vs 0.659) and Exp 6 negative (-0.143 vs -0.150) \u2014 the three contrasts that individually killed pi_1, pi_2/pi_3 and pi_4. Between-subject variance also improved (0.008-0.040 vs the 0.006-0.009 of iter 1), now roughly in the observed range on Exps 4-6 though still low on Exps 1-3.

However, the iter-2 calibration traded residuals rather than uniformly shrinking them. Wins: Exp 3 (+0.082 -> +0.037) and Exp 4 (+0.089 -> +0.025), driven by the \u03bc0 reduction and the \u03b2/\u03bb flattening. Losses: Exp 2 (-0.010 -> +0.048), Exp 5 (-0.015 -> +0.030, sign now flipped positive) and Exp 1 (+0.034 -> +0.045). The Exp 2 regression is directly attributable to lowering \u03bc1: the readability gate no longer saturates on that scrambled display, so a validity leak pushed the metric from a near-perfect 0.500 up to 0.558 (my own iter-1 'two-birds' suggestion overshot here). The Exp 5 sign flip is a \u03b8 x dilution interaction: raising \u03b8 while the sparse lo-cell (weak single endorsement vs empty) is strongly amplified by the 1/(c0+\u03ba T) denominator drives that cell far sub-chance, while the dense hi-cell (5/5 vs 4/5) is compressed \u2014 so hi-lo went positive when the data want it slightly negative. Exp 1 remains systematically ~0.045 too high, i.e. the model is globally a bit too deterministic on validity-sorted displays.
Rationale: Keep PWA-DD exactly as is mechanically \u2014 no family change, no new terms. This is a third calibration pass with three targeted, mostly-orthogonal knobs, chosen to undo the two costs my last critique incurred while preserving its two wins.

1. UNDO the \u03bc1 reduction (my iter-1 advice that backfired). Exp 2 was near-perfect at 0.500 when the readability gate saturated (\u03bc\u21921) and regressed to 0.558 once \u03bc1 was lowered to 1.15. Restore \u03bc1 to roughly [1.4, 2.2] (mid ~1.8) so Exp 2's scrambled display re-saturates and returns to ~0.51. Do NOT try to fix Exp 4 through the gate \u2014 the \u03b2/\u03bb route already worked (0.748 -> 0.684) and should be kept.

2. Reduce global determinacy slightly to fix Exp 1 (and mop up Exp 4/Exp 5). Raise the lapse box to about \u03b5 \u2208 [0.06, 0.26] (mid ~0.16) and/or shave the top of \u03b2 to ~[2.0, 5.0]. This is a single lever that moves four residuals in the right direction at once: Exp 1 0.798 -> ~0.78, Exp 2 (post-fix) stays ~0.51 (it is at chance, so \u03b5 is inert there), Exp 4 0.684 -> ~0.66, Exp 5 0.017 -> ~0.010. Its only cost is compressing Exp 3's contrast.

3. Compensate Exp 3 with the lever that already worked, taken one small step further \u2014 not with more \u03b8. Lower \u03bc0 to [0.0, 0.08] (mid ~0.04). \u03bc0 is arithmetically inert wherever the gate saturates (Exps 2, 4, 6) and, in Exp 3's ASCENDING display, it is the residual leak that props up the lowest-validity cue sitting in column 0 \u2014 exactly what blunts the agree-minus-conflict contrast. Hold \u03b8 where it is (~[0.08, 0.16]); pushing \u03b8 higher again is what flipped Exp 5 positive and inflated Exp 1, so that direction is now closed.

4. One small extra knob for Exp 5's sign: shade the dilution gain down, \u03ba \u2208 [0.7, 1.6] (mid ~1.15). Less dilution differential between the sparse lo-cell and the dense hi-cell shrinks the artificial hi-lo gap and lets the observed slightly-negative value (-0.013) come back, without touching the direction-blindness of the mechanism or reintroducing any coherence bonus.

Guardrails to check before submitting: (a) the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor must stay at ceiling under the upper end of the \u03b8 box combined with the higher \u03b5; (b) Exp 6 must not drift positive \u2014 it is currently the best-fit experiment (-0.143 vs -0.150) and depends on strong column-0 positional dominance, so verify \u03bb and the gate still saturate there; (c) keep the widened \u03b2/\u03b5/\u03b8 boxes for between-subject spread, especially on Exps 1-3 where simulated variance (0.008) is still ~2-4x below observed (0.014-0.037).

**Outcome of your advice:** iter 2 candidate loss=0.0484 -> iter 3 candidate loss=0.0691 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate was REJECTED by the gate (0.0691 vs the running-best 0.0484), so the base remains iter 2. It is still faithful to the arbiter's prescribed family — all-cue compensatory weighted-additive integration, validity-vs-position mixture gated by display readability, monotone faint-praise theta, direction-blind density dilution, no coherence/unanimity premium — and no mechanism drift occurred; this was purely a four-knob calibration pass that traded residuals badly.

Decomposing the four edits against the iter-2 base:
(1) mu1 restored to [1.4,2.2]: WORKED as intended. Exp 2 moved 0.558 -> 0.5225 (Δ +0.048 -> +0.013). The readability gate re-saturating on the scrambled display is the right call and should be kept.
(2) mu0 lowered 0.07 -> 0.04: helped Exp 3 a lot (-0.1715 -> -0.2093, essentially exact) but simultaneously blew up Exp 1 (0.798 -> 0.838, Δ +0.045 -> +0.085). This confirms mu0 is a single shared lever pulling Exps 1 and 3 in opposite directions: less positional leak on ascending/sorted displays = purer validity weighting = more TTB-consistent responding, which Exp 1 does not want.
(3) eps raised / beta ceiling shaved: did NOT deliver the promised Exp 1 relief (it was swamped by the mu0 effect) and cost Exp 6, which drifted from a near-perfect -0.1425 to -0.1133 (Δ +0.0075 -> +0.037). Global lapse is a blunt instrument here: Exp 6's contrast is a difference of cell proportions, so extra lapse compresses it toward zero.
(4) kap lowered 1.5 -> 1.15: my own iter-2 advice, and it had the WRONG SIGN. Exp 5 went 0.017 -> 0.103 (Δ +0.030 -> +0.117), the single largest regression. Less dilution means the dense hi-cell (5/5 vs 4/5, T=0.9) is decompressed and p_hi rises, widening hi-lo, when the data want it slightly negative (-0.013). The correct direction is MORE dilution gain, not less.

So the net picture: one validated win (mu1), one genuine but expensive win (mu0 on Exp 3 at Exp 1's expense), and two clear losses (eps on Exp 6, kap on Exp 5).
Rationale: Regenerate on the UNCHANGED iter-2 accepted base, keeping PWA-DD mechanically identical (no family change, no new terms, no re-introduction of a coherence/unanimity premium). Make a MINIMAL, two-knob edit instead of the four-knob sweep that just got rejected — that sweep failed precisely because coupled levers cancelled.

1. KEEP the mu1 restoration to [1.4, 2.2] (mid ~1.8). This is the one edit in iter 3 that is clearly attributable and clearly positive: Exp 2's scrambled display re-saturates the readability gate and the metric returns from 0.558 to ~0.52. It is arithmetically inert on the monotone displays (Exps 1, 3, 5) and on Exps 4/6, which saturate through the interior-best-cue term anyway.

2. REVERSE my iter-2 kap advice — it had the wrong sign and the gate punished it. Do NOT lower the dilution gain; RAISE it modestly relative to the iter-2 box: kap in roughly [1.1, 2.3] (mid ~1.7) rather than iter-2's [1.0, 2.0] (mid 1.5) or iter-3's [0.7, 1.6]. More dilution compresses the dense hi-cell (5/5 vs 4/5) further, pulling Exp 5's hi-minus-lo down from +0.017 toward the observed -0.013. Take a small step: kap is a global denominator and overshooting it will flatten Exp 1 and Exp 4 as well.

3. REVERT the other two iter-3 edits to their iter-2 values, with one small exception. Put eps and the beta ceiling back to the iter-2 boxes (eps ~[0.03, 0.22], beta ~[2.0, 5.6]) — the extra lapse did not fix Exp 1 and did visibly cost Exp 6, which is currently the best-fitting experiment and must not be sacrificed. For mu0, do not go all the way back to 0.07 nor stay at 0.04: I have now pushed mu0 in both directions across iterations, so split the flanking values and use mu0 in [0.0, 0.11] (mid ~0.055). That should land Exp 3 around -0.19 and Exp 1 around 0.81-0.82, sharing the residual between them rather than concentrating it in Exp 1.

4. Leave theta alone at [0.08, 0.16]. Both the up-direction (flips Exp 5 positive, inflates Exp 1) and further tinkering have now been tested; it is not the productive lever.

If, after this, Exp 1 is still ~0.04-0.05 too high, the next in-family knob to try is NOT global lapse but the validity-sensitivity exponent gamma: shading gamma down (e.g. [0.9, 1.6]) flattens the validity weight profile specifically on sorted displays, which is exactly where Exp 1's excess TTB-consistency comes from, while being largely inert wherever the positional gradient dominates (Exps 2, 4, 6). Flag that as the reserve lever rather than changing it in the same pass.

Guardrails: verify the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor stays at ceiling under the higher kap (the denominator at T=0.5 grows, so check D is still comfortably positive), and confirm Exp 6 returns to ~-0.14 once eps is restored.

**Outcome of your advice:** iter 3 candidate loss=0.0691 -> iter 4 candidate loss=0.0475 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is the new running-best (0.0475) and is fully faithful to the arbiter's prescribed family: all-cue compensatory weighted-additive integration, validity-vs-position mixture gated by display readability, monotone faint-praise \u03b8, direction-blind density dilution, and no coherence/unanimity premium anywhere. It remains the only theory in the comparison set that simultaneously gets Exp 3 clearly negative (-0.177 vs -0.208), Exp 4 moderate (0.658 vs 0.659, essentially exact), and Exp 6 negative (-0.113 vs -0.150) \u2014 the three contrasts that individually falsified pi_1, pi_2/pi_3 and pi_4.

Residuals by experiment: Exp1 +0.023 (best yet), Exp2 +0.024, Exp3 +0.031, Exp4 -0.001 (exact), Exp5 +0.083, Exp6 +0.037. Two of the six now dominate the aggregate loss, and both are REGRESSIONS relative to the iter-2 base rather than untouched weak spots: Exp 5 was +0.030 at iter 2 and is now +0.083; Exp 6 was -0.1425 (residual +0.0075) at iters 1 AND 2 and is now -0.1133 (+0.037). This pass changed three things (mu1 up, kap up, c0 down) but the mu1 change is inert on Exps 5/6 (monotone display / gate saturated through the interior term), and kap up should have pushed Exp 5's hi-minus-lo DOWN. So the c0 reduction (0.09 -> 0.07) is the near-certain culprit for both regressions: shrinking the low-density floor of the denominator sharpens exactly the sparse cells that Exps 5 and 6 are built out of. In Exp 5 the lo-cell (weak single endorsement vs empty panel, T=0.1) is driven further sub-chance by \u03b8 once the tiny denominator amplifies it, so p_hi - p_lo widens positive when the data want ~0. In Exp 6 the U1/V1 anchor cells (single endorsement vs empty, T=0.1) and the ladder cells are all sparse, and over-sharpening them compresses the L1-L4 / U4-U1 / V2-V1 differences toward zero, exactly the observed drift from -0.14 to -0.11.

Between-subject variance is now roughly right on Exps 4-6 (sim 0.010-0.037 vs obs 0.013-0.038) but still ~2x low on Exps 1-3 (sim 0.009-0.013 vs obs 0.014-0.037), and Exp 5's simulated spread (0.037) is now above observed (0.014). This is second-order; the point-estimate residuals on Exps 5 and 6 are the priority.
Rationale: Keep PWA-DD mechanically identical \u2014 no family change, no new terms, no re-introduction of a coherence/unanimity premium, no stopping rule. Make a SINGLE-KNOB edit this time; the four-knob sweep at iter 3 was rejected precisely because coupled levers cancelled, and the current pass shows the same problem in miniature (kap and c0 pushed Exp 5 in opposite directions and c0 won).

1. REVERT c0 to roughly [0.05, 0.13] (mid ~0.09), i.e. back to the iter-2 value, and CHANGE NOTHING ELSE. Keep kap at [1.1, 2.3] (mid 1.7) \u2014 raising it was my iter-3 correction and it is directionally right for Exp 5. Keep mu1 at [1.4, 2.2] \u2014 that is the one edit with a clean, attributable win (Exp 2 back near 0.52). Keep beta, eps, theta, mu0, gamma, lam, mu2 exactly as they are; eps and theta have both been probed in both directions and are closed.

Expected effect: raising the low-density floor pulls the sparse lo-cell in Exp 5 back toward 0.5 (it is currently over-driven sub-chance by \u03b8), so hi-minus-lo should fall from +0.070 toward ~+0.02, and it restores the sparse-cell determinacy gradient in Exp 6, which should return to ~-0.14 (residual +0.0075). The cost is small on Exp 3 (~-0.177 -> ~-0.171, residual +0.031 -> +0.037), and c0 up should also shave Exp 1 slightly, which is currently on the high side (+0.023). Net expected: Exp5 -0.05, Exp6 -0.03, Exp3 +0.006 \u2014 a clear aggregate gain if kap=1.7 is genuinely doing its job, and it also cleanly disambiguates the confounded kap/c0 pass you just ran.

2. If and only if step 1 lands and Exp 3 slips further than ~-0.165, use the reserve lever I flagged last round rather than another global one: shade the validity-sensitivity exponent gamma down slightly (e.g. [0.9, 1.7]). Gamma flattens the validity profile specifically on the monotone displays where Exps 1/3/5 live and is largely inert wherever the positional gradient dominates (Exps 2, 4, 6). Do NOT touch mu0 again \u2014 it is a shared lever that trades Exp 1 against Exp 3 one-for-one and I have now pushed it in both directions.

Guardrails: re-check the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor stays comfortably at ceiling with c0 back at 0.09 combined with kap 1.7 (the T=0.5 denominator is now ~0.94, so verify beta*D still gives p >= 0.78); confirm Exp 4 stays near 0.66 (it is currently exact and must not be sacrificed); and confirm Exp 2 stays near 0.52.

**Outcome of your advice:** iter 4 candidate loss=0.0475 -> iter 5 candidate loss=0.0384 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate is the new running-best (0.0384) and is fully faithful to the arbiter's prescribed family: all-cue compensatory weighted-additive integration, a validity-vs-position weight mixture gated by display readability, a strictly monotone faint-praise discount \u03b8, direction-blind density dilution, and no coherence/unanimity premium or stopping rule anywhere. My iter-4 single-knob advice (revert c0 to ~[0.05,0.13], change nothing else) was validated by the gate: it delivered exactly the predicted repairs on the two sparse-cell experiments, Exp 5 (+0.083 -> +0.061) and Exp 6 (+0.037 -> +0.021), and it also fixed Exp 2 (+0.024 -> +0.002, essentially exact) at only a small cost elsewhere.

Current residuals: Exp1 +0.045, Exp2 +0.002, Exp3 +0.026, Exp4 -0.013, Exp5 +0.061, Exp6 +0.021. Two experiments now carry most of the loss and, importantly, they want the SAME correction. Exp 5's hi-minus-lo is +0.048 vs an observed -0.013: the dense hi-cell (5/5 vs 4/5, T=0.9) is still not compressed enough relative to the sparse lo-cell (weak single endorsement vs empty, T=0.1). Exp 1 is +0.045 too TTB-consistent on the medium-density conflict pairs (T\u22480.4-0.5, e.g. [1,0,0,0,0] vs [0,1,1,1,1]); the model is simply too determinate there. Both cells sit at moderate-to-high endorsement density, so a modest increase in the dilution gain \u03ba pushes both in the right direction simultaneously, whereas the sparse cells that anchor Exps 3 and 6 (T=0.1-0.4) are only weakly touched because the c0 floor now dominates their denominator.

The theory continues to be the only one in the comparison set that jointly reproduces the three diagnostic contrasts that individually falsified the alternatives: Exp 3 clearly negative (-0.182 vs -0.208; pi_2/pi_3 are strongly positive), Exp 4 moderate (0.646 vs 0.659; pi_1 collapses to 0.18), and Exps 5/6 near-zero-to-negative (pi_4 gives +0.71 and +0.46). Between-subject variance is now about right on Exps 4-6 but still ~2-3x low on Exps 1-3 (sim 0.005-0.015 vs obs 0.014-0.037) \u2014 second-order relative to the point-estimate residuals.
Rationale: Keep PWA-DD mechanically identical \u2014 no family change, no new terms, no coherence/unanimity premium, no stopping rule. The single-knob discipline has now been validated twice by the gate (iter 4 c0-only, iter 5 confirmed), while the multi-knob sweep at iter 3 was rejected; stay with one knob.

1. THE EDIT: raise the dilution gain \u03ba from [1.1, 2.3] (mid 1.7) to roughly [1.4, 2.6] (mid ~2.0), and change NOTHING else. This is a small step in a direction the gate has already rewarded (my iter-3 correction of raising \u03ba was part of the accepted iter-4 pass), not an oscillation \u2014 I have never advised lowering \u03ba since that one wrong-signed suggestion at iter 2, which is closed. Mechanistically \u03ba only bites where endorsement density is high, so it is targeted: (a) Exp 5's dense hi-cell (T=0.9, denom ~1.62 -> ~1.89) is compressed further toward 0.5 while the sparse lo-cell (T=0.1, denom ~0.26 -> ~0.29) barely moves, so hi-minus-lo should fall from +0.048 toward ~+0.02 or below; (b) Exp 1's conflict pairs sit at T\u22480.4-0.5, so their hit rate should shave from 0.798 toward ~0.77-0.78. Expected net: Exp5 -0.03, Exp1 -0.02, against a small cost on Exp 3 (whose conflict cells at T=0.2-0.4 will lose a little sharpness, contrast ~-0.182 -> ~-0.175, residual +0.026 -> +0.033) and near-inertness on Exps 2/6, which are chance-level and sparse-cell dominated respectively.

2. EXPLICITLY DO NOT TOUCH: c0 (just validated at [0.05,0.13] \u2014 lowering it demonstrably over-sharpens the sparse cells that Exps 5 and 6 are built from), \u03bc1 (the clean Exp-2 win, now essentially exact at 0.512), \u03bc0 (a shared lever that trades Exp 1 against Exp 3 one-for-one; probed in both directions and closed), \u03b8 and \u03b5 (both probed in both directions and closed; extra global lapse cost Exp 6 last time), \u03bb, \u03bc2, \u03b2.

3. RESERVE LEVER, only if step 1 lands and Exp 3 slips past ~-0.165: shade the validity exponent \u03b3 UP slightly (e.g. [1.3, 2.1]) rather than down \u2014 note that my earlier 'shade \u03b3 down' note would deepen the Exp-1 fix but make Exp 3 shallower, which is the wrong direction now that Exp 3 is the more fragile of the two. Do not change \u03b3 in the same pass as \u03ba.

Guardrails to verify before submitting: (a) the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor must stay comfortably at ceiling with \u03ba=2.0 (T=0.5 gives denom ~1.09, so check \u03b2\u00b7D still yields p \u2273 0.75 on the correct side across the box); (b) Exp 4 must stay near 0.66 \u2014 it is currently at -0.013 and its trials are moderately dense, so if \u03ba=2.0 drags it below ~0.63 take a smaller step (\u03ba mid ~1.85); (c) Exp 2 must stay near 0.51 and Exp 6 must not drift above ~-0.12. Secondarily, if it is free, widen \u03b2 modestly at the top only (e.g. [2.0, 6.0]) to lift the still-too-low between-subject spread on Exps 1-3 \u2014 but only if the point estimates above are unharmed.

**Outcome of your advice:** iter 5 candidate loss=0.0384 -> iter 6 candidate loss=0.0428 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate I am grading (iter 6, the single-knob kap raise from mid 1.7 -> 2.0) was REJECTED by the gate (0.0428 vs the running-best 0.0384), so the base stays at iter 5. Mechanically it is still fully faithful to the arbiter's prescribed family 
inc all-cue compensatory weighted-additive integration, validity-vs-position mixture gated by display readability, strictly monotone faint-praise theta, direction-blind density dilution, no coherence/unanimity premium and no stopping rule 
inc so this is purely a calibration failure, not family drift.

Decomposition of the kap edit against the iter-5 base: Exp1 0.798 -> 0.748 (residual +0.045 -> -0.005, a large and genuine win); Exp2 0.512 -> 0.527 (+0.002 -> +0.017); Exp3 -0.182 -> -0.158 (+0.026 -> +0.050); Exp4 0.646 -> 0.636 (-0.013 -> -0.023); Exp5 +0.0475 -> +0.0483 (essentially UNCHANGED, +0.061 -> +0.062); Exp6 -0.129 -> -0.123 (+0.021 -> +0.027). Net: one ~0.04 gain against ~0.055 of spread-out losses.

The crucial diagnostic is that kap did NOTHING for its intended target, Exp 5, and my iter-5 prediction there was sign-wrong. In the hi-cell (5/5 vs 4/5) the numerator is w_weakest - theta, which is NEGATIVE, so p_hi sits just below 0.5. Increasing the denominator on that dense cell pulls p_hi UP toward 0.5, i.e. it WIDENS hi-minus-lo rather than shrinking it (the observed +0.048 barely moved because the two effects nearly cancel). Any lever that compresses a sub-chance dense cell toward 0.5 is the wrong sign for Exp 5. The correct in-family lever for Exp 5 is the floor c0, which acts selectively on the SPARSE lo-cell (weak single endorsement vs empty, T=0.1) that theta drives strongly sub-chance: raising c0 pulls p_lo up toward 0.5 and shrinks hi-minus-lo. This is exactly what the gate already rewarded at iter 4 -> 5, where c0 [0.03,0.11] -> [0.05,0.13] moved Exp5 +0.083 -> +0.061 AND Exp6 +0.037 -> +0.021 AND Exp2 +0.024 -> +0.002, with Exp3 actually getting slightly MORE negative (-0.177 -> -0.182) rather than paying a cost.

The theory itself remains the only one in the comparison set that jointly reproduces the three contrasts that individually falsified the alternatives (Exp3 clearly negative vs pi_2/pi_3's +0.45/+0.62; Exp4 moderate vs pi_1's 0.18; Exps 5/6 near-zero-to-negative vs pi_4's +0.71/+0.46). Between-subject variance is about right on Exps 4-6 and still ~2x low on Exps 1-3, which stays second-order.
Rationale: Rebuild on the UNCHANGED iter-5 accepted base (loss 0.0384), keeping PWA-DD mechanically identical: no family change, no new terms, no coherence/unanimity premium, no stopping rule. Stay with the single-knob discipline that the gate has twice rewarded (iters 4 and 5) and that the four-knob sweep (iter 3) and this last kap step both failed to beat.

1. REVERT kap to the iter-5 box [1.1, 2.3] (mid 1.7). My iter-5 push on kap is now closed: it is arithmetically the wrong sign for Exp 5 (the dense hi-cell numerator w_weakest - theta is negative, so extra dilution moves p_hi UP toward 0.5 and widens hi-minus-lo) and it charged Exp3/Exp2/Exp4/Exp6 for a win it happened to deliver on Exp1. Do not raise kap again, and do not lower it either (that direction was refuted at iter 3).

2. THE EDIT (one knob only): raise the dilution FLOOR c0 from [0.05, 0.13] to roughly [0.08, 0.18] (mid ~0.13), changing nothing else. This is a second, modest step in the one direction the gate has already validated end-to-end. It bites almost exclusively on the sparse cells (T=0.1-0.2), where c0 dominates the denominator, and is nearly inert on dense cells: Exp5's lo-cell denominator goes ~0.26 -> ~0.31 (+19%) so p_lo, currently over-driven sub-chance by theta, rises toward 0.5 and hi-minus-lo should fall from +0.048 toward ~+0.02 or below (observed -0.013), while the hi-cell denominator moves only ~1.62 -> ~1.67. The same mechanism should push Exp6's sparse anchor cells (U1, V1) back toward 0.5, making U4-U1 and V2-V1 more negative and returning the contrast from -0.123/-0.129 toward ~-0.14 (observed -0.150). Exp2 is at chance and stayed unaffected last time; Exp3 empirically did NOT pay a cost when c0 was last raised, so the anticipated 'agree-cell flattening' worry has already been tested and did not materialise at this step size.

3. EXPLICITLY DO NOT TOUCH: mu1 (the clean Exp-2 win, essentially exact at 0.512), mu0 (a shared lever that trades Exp1 against Exp3 one-for-one; probed both ways and closed), theta and eps (both probed both ways; extra global lapse cost Exp6), lam, mu2, beta, gamma.

4. RESERVE, for a LATER pass only if step 2 lands and Exp1 is still ~+0.04 too TTB-consistent: the iter-6 result shows Exp1's excess determinacy is genuinely reachable via kap, so a HALF-step (kap mid ~1.85, i.e. [1.25, 2.45]) may recover part of that 0.04 win at roughly half the collateral cost 
inc but only after c0 has been isolated and scored, never in the same pass.

Guardrails: (a) re-verify the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor stays comfortably at ceiling with c0 = 0.13 and kap = 1.7 (T=0.5 gives denom ~0.98; check beta*D still yields p >~ 0.75 across the box); (b) Exp4 must not slip below ~0.63 
inc its trials are moderately dense so c0 should be near-inert there; (c) Exp3 must not rise above ~-0.165; (d) if it is free, widen the top of beta only (e.g. [2.0, 6.0]) to lift the still-too-low between-subject spread on Exps 1-3, but only if none of the point estimates above degrade.

**Outcome of your advice:** iter 6 candidate loss=0.0428 -> iter 7 candidate loss=0.0457 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The iter-7 candidate (single-knob c0 raise, [0.05,0.13] -> [0.08,0.18]) was REJECTED by the gate (0.0457 vs the running-best 0.0384), so the base stays at iter 5. Mechanically it is still fully faithful to the arbiter's prescribed family \u2014 all-cue compensatory weighted-additive integration, validity-vs-position weight mixture gated by display readability, strictly monotone faint-praise theta, direction-blind density dilution, no coherence/unanimity premium and no stopping rule. This is a calibration failure, not family drift.

Decomposition of the c0 edit against the iter-5 base: Exp1 0.798 -> 0.778 (residual +0.045 -> +0.025, a real win); Exp2 0.512 -> 0.512 (inert, as predicted); Exp5 +0.048 -> +0.035 (small win, as predicted); Exp3 -0.182 -> -0.159 (+0.026 -> +0.049, a real loss); Exp4 0.646 -> 0.675 (-0.013 -> +0.016, small loss); Exp6 -0.129 -> -0.095 (+0.021 -> +0.055, the biggest loss). My iter-6 prediction that raising c0 would push Exp 6 back toward -0.14 was SIGN-WRONG. The arithmetic I missed: Exp 6's negative contrast is carried mainly by V2-V1 (and partly U4-U1), where the SPARSE anchor cell (V1 = single strong endorsement vs empty, T=0.1) is near ceiling precisely because its denominator is tiny. Raising the floor c0 de-sharpens V1 much more than the dense V2, so V2-V1 shrinks toward zero and the whole contrast drifts positive. Exp 3's agree cells (single endorsement vs empty, T=0.1) are hurt by exactly the same de-sharpening. So the sparse-cell sharpness that Exps 3 and 6 need is in direct conflict with the global de-sharpening that Exps 1 and 5 want.

The upshot after seven iterations: BOTH global de-sharpening levers are now closed. kappa up (iter 6) bought Exp1 at the price of Exp2/3/4/6 and did nothing for its intended target Exp5; kappa down was refuted at iter 3; c0 down over-sharpens (iter 4); c0 up de-sharpens the sparse anchors that Exps 3 and 6 live on (iter 7). eps/beta and theta have been probed in both directions. mu0 trades Exp1 against Exp3 one-for-one. The base's remaining loss is concentrated in Exp5 (+0.061) and Exp1 (+0.045), with Exp3 (+0.026) third; Exp2 (+0.002), Exp4 (-0.013) and Exp6 (+0.021) are close and must be protected.

The theory itself remains the only one in the comparison set that jointly reproduces the three contrasts that individually falsified the alternatives (Exp3 clearly negative vs pi_2/pi_3's +0.45/+0.62; Exp4 moderate vs pi_1's 0.18; Exps 5/6 near-zero-to-negative vs pi_4's +0.71/+0.46). Between-subject variance is about right on Exps 4-6 and still ~2x low on Exps 1-3 \u2014 second-order.
Rationale: Rebuild on the UNCHANGED iter-5 accepted base (loss 0.0384): revert c0 to [0.05, 0.13] and kappa to [1.1, 2.3], and keep PWA-DD mechanically identical \u2014 no family change, no new terms, no coherence/unanimity premium, no stopping rule. Keep the single-knob discipline that the gate rewarded at iters 4 and 5.

1. DO NOT touch c0 or kappa again. Both directions of both denominator knobs have now been tested and each is a global de-sharpener: it always buys Exp1/Exp5 with Exp3 and Exp6 currency, and Exp6's contrast (carried by the sparse V1/U1 anchor cells being near ceiling) is the most fragile thing in the model. Those two levers are closed.

2. THE EDIT (one knob only): shade the validity-sensitivity exponent gamma DOWN, from [1.1, 1.9] (mid 1.5) to roughly [0.9, 1.6] (mid ~1.25). This is the reserve lever I flagged at iter 4 and never actually spent, and it is the only remaining knob that is arithmetically INERT on Exp 6 (and Exp 2 and Exp 4), because on those scrambled/interior-best displays the readability gate saturates and the weights are purely positional \u2014 gamma never enters. So it attacks Exp1 without paying the Exp6 tax that killed the last two attempts.

Why it should bite hard on Exp1 and only softly on Exp3 (a favourable exchange rate): Exp1's diagnostic conflict trials are 1-vs-4 coalitions, e.g. [1,0,0,0,0] vs [0,1,1,1,1], where the model's evidence is 2*w0 - 1 + 3*theta. With gamma=1.5 that is comfortably positive (w0 ~0.50 -> +0.36) so the model follows the top cue nearly always; flattening w0 to ~0.45 drops it to ~+0.26 and drives the hit rate down toward the observed 0.753. Exp3's conflict trials, by contrast, are 1-vs-1 and 2-vs-2 ORDER comparisons (w4 vs w0, w3 vs w1, w0+w1 vs w3+w4): flattening shrinks their magnitude but never flips their sign, so those hit rates only creep toward 0.5 rather than crossing over. Moreover the loss on Exp3's conflict side is partly self-cancelling: agree-cell G ((0,0,0,0,1) vs empty) has evidence w4 - theta, and lowering w4 pushes that cell further sub-chance, which makes the agree-minus-conflict contrast MORE negative. Net expected: Exp1 ~0.798 -> ~0.77-0.78 (residual +0.045 -> ~+0.02), Exp3 roughly flat to slightly worse (-0.182 -> ~-0.172), Exps 2/4/6 unchanged by construction. If Exp5's display is validity-monotone, gamma down also raises w4 toward theta, pulling the sparse lo-cell up toward 0.5 and shrinking hi-minus-lo (a bonus); if Exp5's display is scrambled, gamma is simply inert there and nothing is lost.

3. EXPLICITLY DO NOT TOUCH: mu1 (the clean Exp-2 win, essentially exact at 0.512), mu0 (shared Exp1/Exp3 lever, probed both ways and closed), theta and eps (both probed both ways; extra lapse cost Exp6), lam, mu2, beta, c0, kappa. Change gamma and nothing else so the pass is cleanly attributable \u2014 that is what the two ACCEPTED iterations did and what every rejected pass failed to do.

Guardrails before submitting: (a) verify the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor is still comfortably at ceiling at the BOTTOM of the gamma box (flatter weights leave the dominance evidence at 1 - 5*theta, so it should be safe, but check p >= 0.75); (b) Exp3 must not rise above ~-0.170 \u2014 if the hand-check says it will, take a half-step (gamma [1.0, 1.7], mid ~1.35) instead; (c) Exps 2, 4 and 6 must be numerically unchanged \u2014 if they move at all, the gate is not saturating where you think it is and that is a bug worth finding; (d) only if everything above is free, widen the TOP of beta (e.g. [2.0, 6.0]) to lift the still-too-low between-subject spread on Exps 1-3.

**Outcome of your advice:** iter 7 candidate loss=0.0457 -> iter 8 candidate loss=0.0420 -> the gate marked it REJECTED.

### Iteration 8
Verdict: regenerate
Interpretation: The iter-8 candidate (single-knob gamma shade-down, [1.1,1.9] -> [0.9,1.6]) was REJECTED by the gate (0.0420 vs the running-best 0.0384), so the base remains iter 5. Mechanically it is still fully faithful to the arbiter's prescribed family: all-cue compensatory weighted-additive integration, a validity-vs-position weight mixture gated by display readability, a strictly monotone faint-praise theta, direction-blind density dilution, no coherence/unanimity premium and no stopping rule. There is no family drift; this is purely calibration.

Decomposition of the gamma edit against the iter-5 base: Exp1 0.798 -> 0.764 (residual +0.045 -> +0.010, a large and genuine win, exactly as I predicted); Exp2 0.512 -> 0.512 (inert, as predicted \u2014 the readability gate does saturate there); Exp4 0.646 -> 0.657 (-0.013 -> -0.002, essentially free); Exp6 -0.129 -> -0.129 (inert, as predicted). But: Exp3 -0.182 -> -0.133 (+0.026 -> +0.075, the biggest single loss in the pass) and Exp5 +0.048 -> +0.070 (+0.061 -> +0.083, worse, and the OPPOSITE of the direction I predicted). My hand-argument that flattening w4 would push Exp3's agree-cell G further sub-chance and thereby self-cancel was wrong: flattening shrinks the magnitude of Exp3's 1-vs-1 and 2-vs-2 ORDER conflict comparisons (w4 vs w0, w3 vs w1, w0+w1 vs w3+w4) toward zero, so the conflict hit rate collapses toward 0.5 and the agree-minus-conflict contrast becomes much less negative. Similarly in Exp5 the flattening raised the middle/weak cue weights enough that the sparse lo-cell moved less than the numerator-sign arithmetic suggested, and hi-minus-lo widened. So gamma-down buys Exp1 at roughly a 2:1 exchange rate against Exp3+Exp5, and the direction is closed.

After eight iterations the ledger is: c0 (both directions) and kappa (both directions) are closed \u2014 they are global de-sharpeners that always buy Exp1/Exp5 with Exp3/Exp6 currency. mu0 is closed (trades Exp1 against Exp3 one-for-one). eps and beta (global lapse/determinacy) are closed (they cost Exp6, whose contrast is a difference of cell proportions). mu1 is settled and delivering an essentially exact Exp2. gamma-down is now closed too. The base's remaining loss is concentrated in Exp5 (+0.061) and Exp1 (+0.045), with Exp3 (+0.026) third; Exp2 (+0.002), Exp4 (-0.013) and Exp6 (+0.021) are close and must be protected.

The theory itself remains the only one in the comparison set that jointly reproduces the three contrasts that individually falsified the alternatives (Exp3 clearly negative vs pi_2/pi_3's +0.45/+0.62; Exp4 moderate vs pi_1's 0.18; Exps 5/6 near-zero-to-negative vs pi_4's +0.71/+0.46). Between-subject variance is about right on Exps 4-6 and still ~2x low on Exps 1-3 \u2014 second-order.
Rationale: Rebuild on the UNCHANGED iter-5 accepted base (loss 0.0384): revert gamma to [1.1, 1.9] (mid 1.5), c0 to [0.05, 0.13], kappa to [1.1, 2.3], and keep PWA-DD mechanically identical \u2014 no family change, no new terms, no coherence/unanimity premium, no stopping rule, no re-parameterisation of the readability gate. Keep the single-knob discipline that the gate rewarded at iters 4 and 5.

1. DO NOT touch gamma again in either direction. My iter-7 push (gamma down) was rejected: it wins ~0.035 on Exp1 but loses ~0.049 on Exp3 and ~0.022 on Exp5, and even a half-step keeps that unfavourable exchange rate because the loss channel (flattened ORDER comparisons in Exp3's conflict cells) scales the same way as the gain channel. Also do not touch c0, kappa, mu0, mu1, eps, beta, lam or mu2 \u2014 all have now been probed in at least one direction that the gate punished.

2. THE EDIT (one knob only): shade the faint-praise discount theta DOWN, from [0.08, 0.16] (mid 0.12) to roughly [0.06, 0.13] (mid ~0.095). This is a direction I have never advised \u2014 my only previous theta advice was to raise it (iter 1), and that raise is what flipped Exp5 positive in the first place. Note the empirical trace supports the reversal: at iter 1, with theta [0.06,0.14], Exp5 was -0.028 against an observed -0.013 (the best Exp5 fit ever produced in this loop); every subsequent iteration with theta >= [0.08,0.16] has had Exp5 stuck at +0.03 to +0.10.

Why it should attack the two largest residuals at once. (a) Exp5: the sparse lo-cell (weak single endorsement vs empty, T=0.1) has numerator w_j - theta with a tiny denominator, so it is strongly over-driven sub-chance; lowering theta pulls p_lo up toward 0.5 by a large amount, while the dense hi-cell (5/5 vs 4/5, T=0.9, denominator ~1.6) moves in the same direction but ~6x less because of the density divisor. Net: hi-minus-lo should fall from +0.048 toward ~+0.01 or below (observed -0.013). This is exactly the density-selective asymmetry the mechanism already contains \u2014 unlike c0/kappa, theta enters the NUMERATOR, so it does not de-sharpen the sparse anchor cells that Exps 3 and 6 depend on; it only rescales the endorsement-count penalty. (b) Exp1: the diagnostic 1-vs-4 conflict pairs have numerator 2*w0 - 1 + 3*theta, so lowering theta lowers the evidence for the top-cue option and should shave the hit rate from 0.798 toward ~0.78 (observed 0.753) \u2014 a smaller but same-signed win, obtained WITHOUT flattening the weight profile, so Exp3's order comparisons keep their magnitude.

3. The known cost is Exp3's agree cells G and H (single endorsement vs empty, numerator w_j - theta) rising toward 0.5, making agree-minus-conflict less negative. Crucially, this time the conflict side is untouched (gamma and the weight profile are unchanged), so the damage is one-sided and roughly half of what the gamma pass incurred. Hand-check it: if the computed Exp3 contrast rises above ~-0.170, take a half-step instead (theta [0.07, 0.145], mid ~0.108).

4. Guardrails before submitting: (a) the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor gets STRONGER under lower theta (numerator 1 - 5*theta rises from 0.40 to ~0.525), so verify it does not overshoot into implausible near-determinism given beta and eps; (b) Exp6 must stay at ~-0.13 \u2014 theta enters U4-U1 and V2-V1 with delta-count = 1 on both sides so it should largely cancel, but confirm numerically, since Exp6 is the most fragile good fit in the model; (c) Exp2 must stay near 0.51 and Exp4 near 0.65; (d) change ONLY theta so the pass is cleanly attributable \u2014 that is the property both ACCEPTED iterations had and every rejected pass lacked.

**Outcome of your advice:** iter 8 candidate loss=0.0420 -> iter 9 candidate loss=0.0370 -> the gate marked it ACCEPTED.

### Iteration 9 (most recent)
Verdict: regenerate
Interpretation: The iter-9 candidate is the new running-best (0.0370) and remains fully faithful to the arbiter's prescribed family: all-cue compensatory weighted-additive integration, a validity-vs-position weight mixture gated by display readability, a strictly monotone faint-praise theta, direction-blind density dilution, no coherence/unanimity premium and no stopping rule. My iter-8 advice (shade theta down, half-step per the guardrail) was validated by the gate.

Decomposition of the theta half-step against the iter-5 base: Exp1 0.798 -> 0.741 (residual +0.045 -> -0.0125, the biggest single win and, notably, a slight OVERSHOOT past the observed 0.753); Exp6 -0.129 -> -0.1575 (+0.021 -> -0.0075, now essentially exact); Exp2 0.512 -> 0.509 (exact); Exp4 0.646 -> 0.673 (-0.013 -> +0.014, small cost); Exp5 +0.048 -> +0.052 (+0.061 -> +0.065, essentially flat \u2014 my predicted Exp5 gain did NOT materialise, which given Exp5's simulated jitter across iterations (0.035\u20130.103) is mostly cell-level sampling noise on a 2-cell contrast); Exp3 -0.182 -> -0.1365 (+0.026 -> +0.0715, the anticipated one-sided cost, and now the largest residual in the model).

So the loss is now concentrated in exactly two places: Exp3 (+0.0715, model contrast too shallow \u2014 its conflict-family hit rate is too close to chance and/or its sparse agree cells sit too high) and Exp5 (+0.065, hi-minus-lo too positive). Exp1 has flipped to slightly UNDER-shooting, Exp2/Exp6 are essentially exact, Exp4 is +0.014. Between-subject variance remains about right on Exps 4\u20136 and ~2x low on Exps 1\u20133 \u2014 second-order.

The theory continues to be the only one in the comparison set that jointly reproduces the three contrasts that individually falsified the alternatives (Exp3 negative vs pi_2/pi_3's +0.45/+0.62; Exp4 moderate vs pi_1's 0.18; Exps 5/6 near-zero-to-negative vs pi_4's +0.71/+0.46).
Rationale: Keep PWA-DD mechanically identical \u2014 no family change, no new terms, no coherence/unanimity premium, no stopping rule, no re-parameterisation of the readability gate. Keep the single-knob discipline: every ACCEPTED pass in this loop (iters 4, 5, 9) changed exactly one box; every rejected pass either changed several or spent a lever whose exchange rate was unfavourable.

1. THE EDIT (one knob only): shade the validity-sensitivity exponent gamma UP, from [1.1, 1.9] (mid 1.5) to roughly [1.3, 2.1] (mid ~1.7). This is the reserve lever I flagged at iter 5 and never spent; it is NOT a repeat of the rejected iter-7/8 push, which was gamma DOWN, and the iter-8 result gives me a directly measured mirror-image estimate of its exchange rate rather than a hand argument. Measured on the same base, gamma [1.1,1.9] -> [0.9,1.6] produced Exp1 -0.034, Exp3 +0.049 (worse), Exp5 +0.022 (worse), Exps 2/4/6 numerically inert (the gate saturates there, so gamma never enters). Reversing the sign therefore predicts: Exp3 contrast deepens from -0.137 toward ~-0.175/-0.18 (residual +0.0715 -> ~+0.03, by far the largest available gain, driven by the 1-vs-1 and 2-vs-2 ORDER comparisons w4-vs-w0, w3-vs-w1, w0+w1-vs-w3+w4 regaining magnitude), Exp5 tightens somewhat, and Exp1 rises from 0.741 back toward ~0.77 \u2014 which is now a benefit for the first time in this loop, because theta-down overshot Exp1 to -0.0125 below the observed 0.753. Exps 2, 4 and 6 should be numerically unchanged by construction, which protects the two currently-exact fits (Exp2 0.509, Exp6 -0.1575).

2. Guardrails / contingency, hand-checked BEFORE submitting: (a) Exp1 must not exceed ~0.79 \u2014 if the mid-box hand-check puts it above that, take the half-step gamma [1.2, 2.0] (mid ~1.6) instead; (b) confirm Exps 2, 4 and 6 are numerically inert \u2014 if any of them moves, the readability gate is not saturating where the theory says it does and that is a bug worth finding rather than a calibration result; (c) verify the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor stays comfortably at ceiling at the TOP of the gamma box (steeper weights concentrate mass on the best cue but the dominance numerator is 1 - 5*theta and unaffected, so this should be safe); (d) do not treat a small Exp5 movement as signal \u2014 that contrast is built from two 2-trial cells and jitters ~\u00b10.03 across iterations, so judge the pass on Exp3 and Exp1.

3. EXPLICITLY DO NOT TOUCH: theta (just spent, and a further reduction would push Exp1 further below target and Exp3 further shallow), c0 and kappa (both directions of both denominator knobs are closed \u2014 they are global de-sharpeners that always buy Exp1/Exp5 with Exp3/Exp6 currency, and Exp6 is now essentially exact and is the most fragile good fit in the model), mu0 (trades Exp1 against Exp3 one-for-one), mu1 (the clean, exact Exp2 fit), eps and beta (probed both ways; global lapse compresses Exp6's difference-of-proportions contrast), lam and mu2. Change gamma and nothing else so the pass stays cleanly attributable. If it is genuinely free after the point estimates check out, widening the TOP of beta only (e.g. [2.0, 6.0]) would help the still-too-low between-subject spread on Exps 1\u20133 \u2014 but only as a strictly secondary move that leaves the six point estimates unharmed.

**Outcome of your advice:** iter 9 candidate loss=0.0370 -> CURRENT candidate loss=0.0487 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The iter-10 candidate (single-knob mu0 shade-down, [0.0,0.14] -> [0.0,0.08]) was REJECTED (0.0487 vs the running-best 0.0370), so the base stays at iter 9. Mechanically it is fully faithful to the arbiter's prescribed family: all-cue compensatory weighted-additive integration, a validity-vs-position weight mixture gated by display readability, a strictly monotone faint-praise theta, direction-blind density dilution, no coherence/unanimity premium, no stopping rule. No family drift; this is a pure calibration failure.\n\nDecomposition against the iter-9 base: Exp3 -0.1365 -> -0.1497 (+0.0715 -> +0.0583, a real but small gain, and the only intended one); Exp4 0.6727 -> 0.6673 (+0.014 -> +0.0087, small gain); Exp5 0.0517 -> 0.0492 (flat); Exp2 0.5092 -> 0.5142 (tiny cost); Exp1 0.7408 -> 0.7767 (-0.0125 -> +0.0233, i.e. it overshot straight through the target rather than landing on it); Exp6 -0.1575 -> -0.0967 (-0.0075 -> +0.0533, by far the biggest loss). Two things follow. (1) mu0 is now definitively closed: it has been pushed in both directions and both passes were punished; its Exp1/Exp3 exchange rate is unfavourable and its Exp1 step size is much larger than the proposer's hand-check predicted (+0.036 observed vs +0.003 predicted). (2) The Exp6 movement is diagnostically important: the proposer argued (correctly, I believe, given interior-best + mu1*incong >= 1) that mu saturates on Exp6 so mu0 is arithmetically inert there \\u2014 yet the metric moved by 0.06. Either the readability gate is not saturating where the theory says it does (a bug worth a five-line numerical check of mu on the Exp6 display), or Exp6's contrast carries a simulation noise floor of ~\\u00b10.03-0.06 (its history over parameter-inert passes spans -0.095 to -0.1575, which supports the noise reading). Under either reading, iter 9's 'essentially exact' Exp6 fit is partly luck, and the loop is now operating close to its Monte-Carlo noise floor on Exps 5 and 6.\n\nCritically, the knob I actually recommended last round \\u2014 gamma UP \\u2014 was never tested: the proposer explicitly deviated and spent the pass on mu0 instead. So the iter-10 rejection is NOT evidence against gamma up; that lever remains open and is the only remaining lever with a directly measured (mirror-image) exchange rate.\n\nOn the iter-9 base the loss is concentrated in exactly two places: Exp3 (+0.0715, contrast too shallow \\u2014 the robust, low-noise target) and Exp5 (+0.065, hi-minus-lo too positive \\u2014 a 2-cell contrast that jitters ~\\u00b10.03 and is therefore only half signal). Exp1 (-0.0125), Exp2 (-0.0008), Exp4 (+0.014) and Exp6 (-0.0075) are all at or inside the noise floor and must be protected. The theory remains the only one in the comparison set that jointly reproduces the three contrasts that individually falsified the alternatives (Exp3 negative vs pi_2/pi_3's +0.45/+0.62; Exp4 moderate vs pi_1's 0.18; Exps 5/6 near-zero-to-negative vs pi_4's +0.71/+0.46).",
  "verdict": "regenerate",
  "rationale": "Rebuild on the UNCHANGED iter-9 accepted base (loss 0.0370): revert mu0 to [0.0, 0.14] and keep PWA-DD mechanically identical \\u2014 no family change, no new terms, no coherence/unanimity premium, no stopping rule, no re-parameterisation of the readability gate. Keep the single-knob discipline (every ACCEPTED pass \\u2014 iters 4, 5, 9 \\u2014 changed exactly one box).\\n\\n1. THE EDIT (one knob only): shade the validity-sensitivity exponent gamma UP as a HALF-step, [1.1, 1.9] (mid 1.5) -> [1.2, 2.0] (mid ~1.6). This is not an oscillation and not a repeat of a punished direction: the rejected iter-7/8 push was gamma DOWN, and my iter-9 gamma-UP recommendation was never executed (the proposer substituted mu0). The exchange rate is measured, not hand-argued: on the same mechanism, gamma mid 1.5 -> 1.25 produced Exp1 -0.034, Exp3 +0.049 (worse), Exp5 +0.022 (worse), Exps 2/4/6 numerically inert. Mirroring at half that magnitude predicts Exp3 -0.137 -> ~-0.157 (residual +0.0715 -> ~+0.051), Exp1 0.741 -> ~0.755 (residual -0.0125 -> ~0), and Exp5 +0.052 -> ~+0.061 (residual +0.065 -> ~+0.074). In squared-residual terms that is roughly +0.0025 gained on Exp3 and +0.0002 on Exp1 against ~-0.0013 on Exp5 \\u2014 a net gain, and the Exp5 cost is at or below that experiment's own jitter. I explicitly accept the proposer's arithmetic that a FULL gamma step (mid 1.7) roughly cancels the Exp3 gain against Exp5; that is exactly why the recommendation is now a half-step rather than the full one.\\n\\n2. WHY NOT ANOTHER LEVER. mu0 is now closed in both directions (both passes punished). c0 and kappa are closed in both directions (global de-sharpeners: they always buy Exp1/Exp5 with Exp3/Exp6 currency). theta was just spent successfully at a half-step and a further reduction would push Exp1 further below target while making Exp3 shallower. eps and global beta compress Exp6's difference-of-proportions contrast. mu1 delivers an essentially exact Exp2. lam and mu2 only bite where the gate saturates (Exps 2/4/6), all of which are already inside the noise floor. Gamma-up is the only untested lever that attacks the largest, least noisy residual (Exp3) while moving Exp1 back onto \\u2014 rather than through \\u2014 its target.\\n\\n3. GUARDRAILS, hand-checked before submitting: (a) Exp1 must land in [0.75, 0.78]; if the mid-box check puts it above 0.79, use gamma [1.15, 1.95] instead. (b) Verify numerically that mu = 1 on the Exp2, Exp4 and Exp6 displays \\u2014 print it. If mu < 1 anywhere the theory claims saturation, that is a bug, not a calibration result, and it explains the unexpected Exp6 movement in this pass; fix it before spending any further parameter pass. (c) The [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor must stay comfortably at ceiling at the TOP of the gamma box (numerator 1 - 5*theta is unaffected, so this should be safe \\u2014 confirm p >= 0.75). (d) Do NOT read a \\u00b10.03 movement on Exp5 or Exp6 as signal; judge this pass on Exp3 and Exp1 only. (e) Change gamma and nothing else \\u2014 clean attributability is the property every accepted pass in this loop has had, and the last three rejected passes lacked."
}
```

## Usage

```json
{
  "input_tokens": 43365,
  "output_tokens": 8439
}
```
