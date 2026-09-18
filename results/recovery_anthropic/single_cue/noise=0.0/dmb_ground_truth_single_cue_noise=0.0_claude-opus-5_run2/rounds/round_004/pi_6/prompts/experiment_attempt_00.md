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
**Description:** **Capacity-Gated Strongly-Saturating Redundancy Validity-Inversion Tallying (GVIT-CAP-S4).**

Five commitments describe how people combine binary expert ratings whose stated validities they have been told.

**(1) The count comparator is a gate, not a weight.** Before any evaluation of *whose* rating is whose, the subject asks a purely ordinal question: does one product win on more attributes than the other? This is answered with unweighted feature-wise comparisons. If the answer is "no" (equal numbers of winning attributes, including displays where nothing discriminates), the display is construed as *non-diagnostic*, no candidate is generated, and the subject flips a mental coin. Stated validities are **never** used as a tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) When a majority exists, integration uses inverse-informativeness weights.** The subjective news value of a rating is a smooth, panel-relative *decreasing* function of stated validity: s_j = (v_j - v_min)/(v_max - v_min) within the displayed panel, w_j = 1 - lambda_eff*f(s_j). A near-perfect expert is heard as merely restating the obvious consensus quality of the product, so his rating adds almost nothing (and, when lambda_eff*f > 1, is mildly counter-indicative); a barely-diagnostic expert is surprising and is treated as genuine private information carrying full weight. Evidence is the single scalar E = sum_j w_j*sign(a_j - b_j).

**(3) The redundancy construal saturates VERY STRONGLY just above the panel floor.** Redundancy is not judged on a linear scale of stated accuracy: "being one of the accurate ones" is a nearly categorical construal. An expert only has to be slightly above the panel's least-accurate voice before he is already heard as echoing the consensus. Formally the discount is a strongly concave function of panel-relative validity, f(s) = s^omega with omega well below 1 (~0.26-0.42), so the discount rises steeply just above the panel floor and then flattens. Two consequences follow that neither a linear nor a mildly concave inversion produces. (i) A majority carried by *middling* experts (the 2nd-5th most accurate voices) carries almost no subjective evidence: mid-panel support is discounted nearly as heavily as star support, so a mid-backed majority facing a lone star is followed only weakly, close to chance. (ii) Conversely, when the count loser is backed by the panel's genuinely idiosyncratic bottom voices, the anti-count evidence is *strengthened*, sharpening the reversal. So the model separates three displays that a linear inversion renders nearly identical: majority-by-mid-experts (followed barely above chance), majority-by-bottom-voices (followed near ceiling), and star-backed majorities (reversed toward floor).

**(4) Redundancy is comparative; evidence spread over many contradicting cues is compressed.** An expert is discounted only because other experts are heard as echoing him, so the inversion scales with how many voices actually speak on the trial: lambda_eff = lambda_n*(D-1)/((D-1)+kappa_D), where D is the number of discriminating cues; when exactly one expert differentiates the products there is no chorus and his rating is taken at face value. Confidence in E is then compressed by the number of mutually contradicting cues, D^gamma.

**(5) The inversion is a capacity-limited construal: it collapses when the panel exceeds working-memory span.** Discounting the star requires holding the validity ordering of the whole panel in mind, which is possible only up to roughly seven experts. Beyond capacity the ordering blurs, so the redundancy construal washes out toward equal weighting (lambda_n = lambda*(1 - psi*o)) and integration becomes noisier (slope divided by 1 + chi*o), where o = logistic((n - 7.5)/0.2). Finally an attention lapse epsilon replaces the decision by a coin flip.

Cross-design signature: exact chance on tally-tied profiles; strong systematic reversal of star-backed majorities on panels of <=7 experts; near-chance for the same star-backed profiles on panels of 8+ experts, with preserved tally adherence there for broad-mixture majorities; near-ceiling following of a lone discriminating expert; only weakly above-chance following of mid-panel-backed majorities that oppose a lone star; and markedly attenuated confidence whenever the weighted evidence is spread across many mutually contradicting cues.

**Parameters:**
- lam: [1.10, 1.80]
- beta: [5.8, 7.6]
- gamma: [0.74, 1.06]
- kappa_d: [0.01, 0.45]
- psi: [0.32, 0.48]
- chi: [0.75, 1.15]
- epsilon: [0.07, 0.27]
- omega: [0.26, 0.42]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
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


## COMPETING THEORY
**Description:** **Redundancy-Discounted Tiered Integration with Majority Discounting, Panel-Size Capacity Dilution, and Graded Redundancy Relaxation (RDTI-MCR).**

All previous commitments are retained. (i) Stated validities are encoded only *coarsely and ordinally*: a median split into a "star" tier and a "marginal" tier, with no gradations inside a tier. (ii) The star tier is treated as **redundant** — the very accurate experts are construed as merely restating the obvious consensus quality of the product — so their ratings enter integration with only a small residual weight, while the idiosyncratic marginal experts carry full weight. Evidence is the tier-weighted signed sum E = Σ_j w_j (a_j − b_j), and choice is logistic in that scalar plus an attention lapse. Because the weights take only two values, tier-matched profiles cancel *exactly* and give an exact coin flip, with no validity tie-break and no take-the-best fallback. (iii) The same redundancy logic is applied to the majority itself: a feature-count majority is the most "obvious" statistic in the display, so subjects apply a *contrarian consensus discount* proportional to the count margin. (iv) Integration is capacity-limited over the *whole displayed panel*, so the effective slope is divided by a power of the panel size n, in addition to a milder within-trial spread discount over the number of discriminating cues D.

**The new commitment is that the redundancy discount is itself a comparative, graded judgement rather than a fixed property of a cue.** A star expert is discounted precisely *because other experts are seen as restating what he says*; the discount is therefore a function of how much of the panel is actually speaking on this trial. When almost nothing in the display discriminates — in the limit, when a single expert is the only one who differentiates the two products — there is no chorus for that expert to be redundant with, so his rating is taken at (nearly) full face value. The discount relaxes continuously back toward full weight as the number of discriminating cues D falls toward one: eps_eff = eps_w + (1 − eps_w)·ψ·exp(−(D−1)/κ_D), with ψ controlling how completely redundancy can be released and κ_D controlling how fast it releases. Because κ_D is small, the release is essentially confined to genuinely sparse displays. This predicts a sharp, design-level signature that no fixed-weight tiering model produces: **profiles in which exactly one high-validity expert discriminates are followed at high rates (≈0.78–0.85), even though profiles in which that same expert discriminates alongside several others are followed near chance**, since only in the latter case is the star rating construed as redundant restatement. Tally-tied, tier-matched profiles remain at exactly chance (E = 0 and margin = 0 regardless of eps_eff), and all dense profiles (D ≥ 2) are left effectively untouched.

**Parameters:**
- eps_w: [0.03, 0.34]
- beta: [2.2, 3.6]
- delta: [-0.10, 0.25]
- tau: [0.5, 1.3]
- lapse: [0.05, 0.18]
- count_bias: [0.28, 0.62]
- psi: [0.35, 0.85]
- kappa_d: [0.15, 0.45]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with a wide validity spread (0.95, 0.85, 0.75, 0.62, 0.55) make the cue hierarchy salient, so Take The Best (TTB) has an unambiguous search order, while Tallying is indifferent to that order. The 12 unique pairs (repeated 8x each, ~96 trials) fall into three families, all of which dissociate the two models.

(1) Direct-conflict pairs (T1-T6, T11-T12): one option wins the highest discriminating cue while the other wins the majority of the remaining lower-validity cues. TTB predicts choice of the top-cue winner on every one of these; Tallying predicts the opposite option. Critically, the tally margin is parametrically varied: margin 3 (T1/T2: 1 vs 4 cue wins), margin 2 (T5/T6: 1 vs 3; T11/T12: 1 vs 3 with the top cue tied so TTB's deciding cue is expert 2), and margin 1 (T3/T4: 2 vs 3). This yields the key *quantitative* dissociation: TTB's softmax operates on a binary 1/0 score, so its predicted choice probability for the top-cue winner is identical (sigmoid(beta), lapse-corrected) across all conflict trials regardless of tally margin, whereas Tallying's softmax probability grows monotonically with the tally margin (exp(beta*1) vs exp(beta*2) vs exp(beta*3) contrasts). Thus a flat, margin-invariant preference for the top-cue winner is uniquely TTB, while a graded, margin-increasing preference for the majority-cue winner is uniquely Tallying.

(2) Tally-tie pairs (T7-T10): the two options split cue wins evenly (1-1 or 2-2), so Tallying is forced to guess at exactly 50/50 irrespective of its parameters, while TTB still decides confidently via the highest discriminating cue. Systematic above-chance choice of the top-cue winner here cannot be produced by Tallying under any beta/epsilon, making these trials a parameter-free falsification test.

(3) Left/right and direction counterbalancing: every conflict and tie configuration appears in both A-favoring and B-favoring versions, so any side bias cancels and cannot masquerade as evidence for either heuristic.

T11/T12 additionally probe whether the deciding cue must be the very best cue: expert 1 ties, so TTB's decision rests on expert 2 while the tally strongly favors the other option; this separates TTB from any 'only-the-single-best-cue-matters' caricature as well as from Tallying.

[1] Goal: dissociate Tallying (advocated) from Take The Best by factorially crossing (i) which option wins the highest-validity discriminating cue \u2014 the only thing TTB is sensitive to \u2014 with (ii) the tally margin, the only thing Tallying is sensitive to. Six experts with a wide, deliberately scrambled validity spread [0.93, 0.57, 0.79, 0.86, 0.62, 0.71] give the descending cue order f0 > f3 > f2 > f5 > f4 > f1, so validity rank is decoupled from screen position (unlike a monotone left-to-right layout, this prevents a positional-ordering confound from mimicking TTB).

Core manipulation (trials 1,3,5,7,9 and their exact A/B mirrors 2,4,6,8,10): option A always wins expert f0 (validity .93), so TTB's deciding cue and predicted winner are IDENTICAL across all five cells; its predicted P(choose A) is the constant (1-eps)*sigmoid(beta) + eps/2 regardless of the rest of the profile. Meanwhile the tally margin is parametrically swept M = -3, -1, 0, +2, +4 by re-allocating the five lower-validity cues. Tallying's softmax gives P(A) = sigmoid(beta*M) (lapse-corrected), i.e. a strictly monotone, sign-crossing psychometric function of M: strong B-preference at M=-3, weak B at M=-1, exactly 50/50 at M=0, and increasingly strong A at +2 and +4. A flat line across the five margin cells is a unique TTB signature; a monotone increasing line crossing chance at M=0 is a unique Tallying signature. Crucially the design contains both conflict cells (M=-3, -1: TTB says A, Tallying says B) and agreement cells (M=+2, +4: both say A), so the same subject's data pin down beta from the agreement cells and then test whether the conflict cells reverse (Tallying) or not (TTB) \u2014 a within-subject, parameter-anchored test rather than a mere direction contrast.

Tally-tie cells (trials 5, 6): three cue wins each. Tallying MUST be exactly 50/50 for every possible beta and epsilon, while TTB confidently picks the f0 winner. Systematic above-chance choice here parameter-free falsifies Tallying; chance performance parameter-free falsifies TTB.

Low-rank deciding cue (trials 11, 12): the two best experts (f0=.93 and f3=.86) are tied, so TTB must descend to f2 (.79), which favors A, while the tally favors B by 2 (B wins f5, f4, f1). This checks whether the one-reason rule really cascades down the validity hierarchy or whether subjects instead aggregate the three lower experts \u2014 an independent conflict test whose TTB prediction is the same magnitude as in trials 1/3 (binary score) but whose Tallying prediction is intermediate (|M|=2).

Every configuration appears in both an A-favoring and a B-favoring mirror, so side bias cancels exactly and cannot masquerade as support for either heuristic. Tie cues are encoded sometimes as 0-0 and sometimes as 1-1 to prevent a 'count the 1s' surface strategy from being confounded with either model. 12 unique pairs \u00d7 8 repetitions \u2248 96 trials.

[2] GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from softmax Tallying (competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: any two pairs with the same M must produce exactly the same P(choose A) for every possible beta and epsilon. LSNC instead depends on the full composition (nA, nB, nT): tied/uninformative cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so two profiles with identical M but different numbers of tied cues yield different adherence. The design is therefore a 3 (margin: M = 1, 2, 3) x 2 (informative density: SPARSE, all discriminating cues won by one option but many tied cues; DENSE, no/few tied cues but the loser also wins several cues) factorial, fully mirrored for side. Tallying predicts a pure main effect of margin and EXACTLY ZERO density effect (parameter-free); LSNC predicts sparse > dense at every margin.

CELLS (A-favouring version; the mirror swaps A and B):
- S1 sparse M=+1: (nA,nB,nT) = (1,0,5)
- D1 dense  M=+1: (3,2,1)
- S2 sparse M=+2: (2,0,4)
- D2 dense  M=+2: (4,2,0)
- S3 sparse M=+3: (3,0,3)
- D3 dense  M=+3: (4,1,1)

QUANTITATIVE LSNC PREDICTIONS (k=2.5, nu=0.20, epsilon=0, exact enumeration of the hypergeometric sample compositions): M=1: 0.625 (sparse) vs 0.572 (dense); M=2: 0.702 vs 0.630; M=3: 0.754 vs 0.707. So LSNC predicts a ~5-7 point sparse-minus-dense advantage at EVERY margin (and the gap grows with k), while adherence stays capped well below 1 even at M=3. Tallying predicts 0 at every margin. Aggregated over the 3 margins each subject contributes 48 sparse and 48 dense trials, so the parameter-free density contrast is powerfully estimated within subject.

CONFOUND CONTROL (critical): (i) In every A-favouring cell option A wins the highest-validity discriminating cue f0 (0.92), which also discriminates in every cell, so Take-The-Best (or any 'best cue' strategy) predicts a CONSTANT P(A) across all six cells and cannot generate a density effect. (ii) Validity-weighted additive evidence (log-odds weights w = [2.44, 0.41, 1.59, 0.90, 0.20, 1.21]) was deliberately matched between the sparse and dense member of each margin pair: M=1: +2.44 vs +2.13; M=2: +2.64 vs +2.53; M=3: +3.05 vs +3.16. Hence a WADD-type subject also cannot produce the sparse>dense pattern; only sample-limited counting with tie dilution can. (iii) Validity order is scrambled relative to screen position (descending order is f0, f2, f5, f3, f1, f4) so positional heuristics do not mimic validity-ordered search. (iv) Tied cues are encoded sometimes as 1-1 and sometimes as 0-0 so that the number of displayed 1s is not a constant cue to density. (v) Every configuration appears in both an A-favouring and a B-favouring mirror (12 unique pairs x 8 repetitions = 96 trials), so side bias cancels exactly.

INTERPRETATION: a graded, margin-monotone function that is IDENTICAL for sparse and dense cells supports Tallying; a margin-monotone function that is systematically shifted upward in the sparse (tie-padded, proportion-1.0) cells relative to the dense cells with the same margin, together with adherence ceilings around 0.70-0.80 at M=3, uniquely supports LSNC and cannot be produced by Tallying under any beta/epsilon.

[3] GOAL: dissociate softmax Tallying (advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: P(A) = (1-eps)*sigmoid(beta*M) + eps/2. Therefore any two profiles with the same M MUST yield exactly the same P(A) for every beta and epsilon -- a parameter-free invariance. LSNC instead depends on the full composition (nA, nB, nT): because the inspected subset is drawn uniformly from ALL features, tied cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so adherence must fall monotonically as a fixed margin is realized with fewer ties and more loser-wins.

DESIGN: 8 experts, scrambled validity spread [0.93, 0.55, 0.78, 0.86, 0.60, 0.71, 0.66, 0.82] (descending rank f0 > f3 > f7 > f2 > f5 > f6 > f4 > f1, so validity order is decoupled from screen position). 2 (margin: M = 2, 4) x 3 (composition: SPARSE / MID / DENSE) factorial, each cell fully mirrored for side => 12 unique pairs x 8 repetitions ~ 96 trials.

Cells (A-favouring version; mirror swaps A and B):
- S2 sparse M=2: (nA,nB,nT) = (2,0,6)
- M2 mid    M=2: (3,1,4)
- D2 dense  M=2: (5,3,0)
- S4 sparse M=4: (4,0,4)
- M4 mid    M=4: (5,1,2)
- D4 dense  M=4: (6,2,0)

QUANTITATIVE PREDICTIONS. Tallying: a pure margin main effect and EXACTLY ZERO composition effect -- the three M=2 cells must be statistically identical, the three M=4 cells must be statistically identical, and the M=4 level must exceed M=2 by the sigmoid gap sigmoid(4b) - sigmoid(2b). LSNC (exact enumeration, k=2.5, nu=0.20, eps=0): M=2 -> 0.661 (sparse) vs ~0.62 (mid) vs 0.598 (dense); M=4 -> 0.750 (sparse) vs ~0.72 (mid) vs 0.689 (dense). I.e. LSNC predicts a monotone ~6-point decline across the composition ladder at BOTH margins, plus adherence capped near 0.75-0.80 even at M=4. Note that at sample size K=1 all same-margin cells coincide (P = 0.5 + 0.3*M/n), so any composition effect in the data is diagnostic of multi-cue but sub-total sampling; a flat ladder is diagnostic of full integration (Tallying).

CONFOUND CONTROLS. (i) In every A-favouring cell option A wins the highest-validity cue f0 (0.93), which discriminates in all six cells, so Take-The-Best / any 'best cue' rule predicts a CONSTANT P(A) across all cells and cannot generate either the margin effect or a composition ladder. (ii) Validity-weighted additive evidence (log-odds weights) was deliberately matched WITHIN each margin level: M=2 cells give net +2.99 (sparse), +3.04 (mid), +3.12 (dense); M=4 cells give +4.55, +4.78, +4.39. Hence WADD-type integration also predicts no composition effect, so a sparse>mid>dense ordering can only come from sample-limited counting with tie dilution, while a flat ladder plus a clear margin effect uniquely supports Tallying. (iii) Tied cues are encoded sometimes 1-1 and sometimes 0-0 so that display density of 1s is not itself a cue to composition. (iv) Every configuration appears in both an A-favouring and a B-favouring mirror, so side bias cancels exactly.

INTERPRETATION: margin-graded but composition-invariant adherence (with the M=4 cells approaching the ceiling implied by the beta estimated from the M=2 cells) supports Tallying; a systematic sparse > mid > dense decline at each margin, with adherence capped around 0.75 at M=4, supports LSNC.

[4] GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from Count-Direction with Conflict-Graded Confidence (CDCC, competing). Both theories agree on the count-based DIRECTION of choice and both cap adherence below 1, so the classic tally-vs-TTB contrasts are useless here. They differ on exactly two things, and this design isolates both.

KEY DISSOCIATION 1 (parameter-free for LSNC): COHERENCE. LSNC's predicted probability is a function of the composition (nA, nB, nT) ALONE 
cue identity/validity is irrelevant because the inspected subset is drawn uniformly at random. CDCC's confidence is gated by the coherence term C = (W_win - W_lose)/(W_win + W_lose) with convex weights w = (v-0.5)^rho, so it must show a large drop when the highest-validity cue is won by the count LOSER, holding (nA, nB, nT) exactly fixed. Every composition therefore appears in a COHERENT version (count winner also wins the top-validity cues, loser wins only the weakest ones) and a CONFLICT version (count winner wins only the weakest cues, loser wins the top cue(s)), with the identical (nA, nB, nT).

KEY DISSOCIATION 2 (composition ordering reversal): with n = 6 fixed, CDCC's dilution term is S = M/D^gamma (D = nA+nB, ties irrelevant), whereas LSNC's dilution comes from tied cues occupying sampling slots and from the PROPORTION of sampled informative cues won. These give opposite orderings for two of the three cells: LSNC ranks (4,2,0) ABOVE (2,1,3) (0.630 vs 0.589), while CDCC ranks (2,1,3) ABOVE (4,2,0) (0.633 vs 0.602 averaged over coherence; 0.707 vs 0.651 in the coherent versions), because CDCC penalises D = 6 heavily via D^1.5 while LSNC penalises the 3 tied slots.

DESIGN: 6 experts, scrambled validity spread [0.94, 0.57, 0.86, 0.62, 0.70, 0.78] (descending rank f0 > f2 > f5 > f4 > f3 > f1, so validity rank is decoupled from screen position and positional heuristics cannot mimic validity-based coherence). One very high (0.94) and two low (0.57, 0.62) validities make the convex weighting in CDCC bite hard (w = (v-.5)^3 gives f0 = .085 vs f1 = .0003, a 250:1 ratio). 3 compositions x 2 coherence levels x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the mirror swaps A and B):
- T1/T2  C1-COHERENT (nA,nB,nT) = (2,1,3): A wins f0(.94), f2(.86); B wins f1(.57); f3,f4,f5 tied.
- T3/T4  C1-CONFLICT (2,1,3): A wins f1(.57), f3(.62); B wins f0(.94); f2,f4,f5 tied.
- T5/T6  C2-COHERENT (3,1,2): A wins f0,f2,f5; B wins f1; f3,f4 tied.
- T7/T8  C2-CONFLICT (3,1,2): A wins f1,f3,f4; B wins f0; f2,f5 tied.
- T9/T10 C3-COHERENT (4,2,0): A wins f0,f2,f4,f5; B wins f1,f3.
- T11/T12 C3-CONFLICT (4,2,0): A wins f1,f3,f4,f5; B wins f0,f2.

QUANTITATIVE PREDICTIONS. LSNC (exact enumeration, k = 2.5, nu = 0.20, eps = 0): P(count winner) = 0.589 for (2,1,3), 0.656 for (3,1,2), 0.630 for (4,2,0) 
 IDENTICAL for the coherent and conflict member of each pair (a strictly zero coherence effect, for every k, nu, eps). CDCC (beta = 2.6, gamma = 1.5, kappa = 0.7, rho = 3.0, eps = 0.10): coherent 0.707 / 0.757 / 0.651 versus conflict 0.544 / 0.576 / 0.553 
 a 10-18 point coherence effect that cannot be switched off (kappa > 0.4 by hypothesis) and that is largest for the sparse (2,1,3) and (3,1,2) cells where the loser's single win is the top expert.

Each subject contributes 48 coherent and 48 conflict trials, so the coherence contrast is estimated within subject with high power. INTERPRETATION: adherence that depends only on the (nA,nB,nT) composition, is flat across the coherence manipulation, and shows the (3,1,2) > (4,2,0) > (2,1,3) ordering with a ceiling near 0.65, uniquely supports LSNC. A systematic coherent > conflict gap (never dropping below 0.5) together with the (3,1,2) > (2,1,3) > (4,2,0) ordering uniquely supports CDCC. Below-chance adherence in the conflict cells would falsify BOTH theories in favour of validity-weighted / one-reason integration, so the design is also self-diagnostic.

CONTROLS: (i) every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; (ii) tied cues are encoded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not covary with composition; (iii) validity order is scrambled relative to display position; (iv) no pair is degenerate 
 every trial has a strict count winner and every trial requires reading more than one rating.

[5] GOAL: dissociate Count-Direction with Conflict-Graded Confidence (CDCC, advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Both theories agree that the DIRECTION of choice is the count winner and both cap adherence below 1, so tally-vs-TTB style contrasts are useless. They differ on exactly what modulates adherence: LSNC's probability is a function of the composition (nA, nB, nT) ALONE (uniform sampling makes cue identity/validity irrelevant), whereas CDCC's confidence is (i) gated by validity COHERENCE C = (W_win - W_lose)/(W_win + W_lose) with convex weights w = (v-0.5)^rho, and (ii) diluted by S = M/D^gamma where D = nA+nB, so tied cues are completely irrelevant to CDCC but occupy sampling slots for LSNC. This design attacks both differences and, crucially, contains two ORDERING REVERSALS.

DESIGN: 7 experts, scrambled validity spread [0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55] (descending rank f0 > f2 > f5 > f4 > f3 > f1 > f6, so validity rank is decoupled from screen position and positional heuristics cannot mimic validity coherence). One very high (0.94) and two very low (0.58, 0.55) validities make the convex weighting bite: w = (v-.5)^3 gives f0 = .0852 vs f6 = .000125, a ~680:1 ratio. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the even-numbered mirror swaps A and B):
- T1/T2  SPARSE-COHERENT (nA,nB,nT) = (2,1,4): A wins f0(.94), f2(.86); B wins f6(.55); rest tied.
- T3/T4  SPARSE-CONFLICT  (2,1,4): A wins f1(.58), f6(.55); B wins f0(.94); rest tied. IDENTICAL composition to T1/T2.
- T5/T6  DENSE-COHERENT   (4,2,1): A wins f0,f2,f5,f4; B wins f3,f1; f6 tied.
- T7/T8  DENSE-CONFLICT   (4,2,1): A wins f1,f3,f4,f6; B wins f0,f2; f5 tied. IDENTICAL composition to T5/T6.
- T9/T10 UNANIMOUS-SPARSE (2,0,5): A wins f2,f4; five tied cues, no cue opposes.
- T11/T12 UNANIMOUS-DENSE (4,0,3): A wins f0,f3,f5,f6; three tied cues, no cue opposes.

QUANTITATIVE PREDICTIONS.
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): (2,1,4) coherent 0.708 vs conflict 0.541; (4,2,1) coherent 0.651 vs conflict 0.540; (2,0,5) 0.928; (4,0,3) 0.888.
LSNC (exact enumeration, k=2.5, nu=0.20, eps=0): (2,1,4) 0.579 for BOTH coherent and conflict; (4,2,1) 0.621 for BOTH; (2,0,5) 0.680; (4,0,3) 0.770.

THREE ORTHOGONAL DIAGNOSTICS:
(1) COHERENCE (parameter-free for LSNC): holding (nA,nB,nT) exactly fixed, CDCC predicts a 11-17 point coherent-minus-conflict gap that cannot be switched off (kappa >= 0.4 by hypothesis), while LSNC predicts EXACTLY ZERO for every k, nu, epsilon. Each subject contributes 32 coherent and 32 conflict trials on matched compositions, so this contrast is estimated within subject with high power.
(2) SPARSITY REVERSAL among the coherent cells: CDCC's D^gamma dilution makes (2,1,4) [D=2 winners vs 1 loser, D=3] MORE confident than (4,2,1) [D=6] despite the smaller raw margin (0.708 > 0.651); LSNC's tie dilution predicts the OPPOSITE order (0.579 < 0.621) because the sparse cell wastes 4 of 7 sampling slots on tied cues. Same reversal for the unanimous cells: CDCC predicts (2,0,5) > (4,0,3) (0.928 > 0.888, and 0.911 > 0.826 at gamma=1.75), LSNC predicts (4,0,3) > (2,0,5) (0.770 > 0.680). A margin-2 profile beating a margin-4 profile is impossible for LSNC under any parameters.
(3) CEILING: on the clean unanimous-sparse cell CDCC allows 0.90-0.95 adherence, whereas LSNC is capped near 0.68 (and at most ~0.79 even at k=3.5, nu=0.12) because a small noisy sample easily reverses the sign. Adherence above ~0.85 on T9/T10 is outside LSNC's reachable range.

CONTROLS: (i) every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; (ii) tied cues are encoded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not covary with composition or coherence; (iii) validity rank is scrambled relative to display position; (iv) no pair is degenerate: every trial has a strict count winner and requires reading several ratings; (v) systematic BELOW-chance adherence in the conflict cells (i.e. following the single best cue against the tally) would falsify BOTH theories in favour of one-reason/WADD integration, so the design is self-diagnostic.

INTERPRETATION: a large coherent > conflict gap that never dips below 0.5, together with sparse > dense ordering at matched or even smaller margins and near-0.9 adherence on unanimous sparse profiles, uniquely supports CDCC. Flat adherence across the coherence manipulation with a (4,0,3) > (2,0,5) > (4,2,1) > (2,1,4) composition ordering and a ceiling near 0.7 uniquely supports LSNC.

[6] GOAL: dissociate RDTI-MCR (advocated) from Count-Direction with Conflict-Graded Confidence (CDCC, competing). The two models agree on nothing important here: CDCC makes DIRECTION purely count-based (ties => exactly 0.5, parameter-free) and lets validity affect only confidence, never sign, and never drops adherence to the count-winner below 0.5. RDTI-MCR instead (i) median-splits the panel into a discounted 'star' tier and a full-weight 'marginal' tier, (ii) applies a contrarian discount proportional to the count margin, and (iii) releases the star discount only when the discriminating set is essentially a single voice (D=1).

PANEL: 6 experts, validities [0.95, 0.57, 0.88, 0.60, 0.66, 0.83] (screen order deliberately scrambled). Descending rank f0(.95) > f2(.88) > f5(.83) > f4(.66) > f3(.60) > f1(.57), so the RDTI median split is unambiguous: STAR = {f0,f2,f5}, MARGINAL = {f4,f3,f1}, with a wide .83/.66 gap at the split. CDCC's convex weights w=(v-.5)^3 are .091/.066/.036 for the stars versus .0041/.0010/.0003 for the marginals (a >20:1 ratio), so its coherence gate is driven almost entirely by the stars. n=6 makes RDTI's panel-size dilution term exactly 1, removing a nuisance factor. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the following trial is the exact A/B mirror):
C1 STAR-SOLO: only f0(.95) discriminates, A wins it (D=1).
C2 MARGINAL-SOLO: only f1(.57) discriminates, A wins it (D=1).
C3 TIE-1v1 MISMATCHED: A wins f1(.57), B wins f0(.95); tally tied.
C4 TIE-2v2 MISMATCHED: A wins f3(.60) and f4(.66), B wins f0(.95) and f2(.88); tally tied.
C5 DENSE STAR-MAJORITY: A wins f0,f2,f5 (all stars), B wins f3,f4 (marginals); nA=3, nB=2, f1 tied.
C6 DENSE MARGINAL-MAJORITY: A wins f1,f3,f4 (all marginals), B wins f0,f2 (stars); nA=3, nB=2, f5 tied.

QUANTITATIVE PREDICTIONS, P(choose A) at nominal parameters.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1 = 0.79; C2 = 0.87; C3 = 0.86; C4 = 0.93; C5 = 0.07; C6 = 0.94.
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): C1 = 0.945; C2 = 0.945; C3 = 0.500; C4 = 0.500; C5 = 0.600; C6 = 0.521.

THREE ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED, TIER-MISMATCHED CELLS (C3, C4) are parameter-free for CDCC: nA = nB so it MUST be exactly 0.5 for every beta, gamma, kappa, rho, epsilon. RDTI predicts strong, systematic choice of the option backed only by the LOW-validity marginal experts (0.78-0.93 across its whole parameter box for C3, 0.85-0.97 for C4). Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive. Note the sign is also diagnostic against WADD/TTB, which would predict the opposite (below 0.5), so the cell is self-diagnostic three ways.
(2) DIRECTION REVERSAL ON DENSE CONFLICT (C5 vs C6). Both cells have the same count margin (+1 for A) and the same D=5, so CDCC must put BOTH above 0.5 for A (0.52-0.70 depending on gamma/beta; conflict can only erode confidence toward 0.5, never past it). RDTI's redundancy discount plus contrarian majority discount makes C5 collapse to ~0.07 while C6 rises to ~0.94 -- an ~0.87 swing between two profiles CDCC treats as same-direction. Observing sub-chance adherence to the count winner in C5 falsifies CDCC outright; observing C5 and C6 both mildly above 0.5 with C5 > C6 falsifies RDTI.
(3) GRADED REDUNDANCY RELEASE (C1 vs C2). Both are single-discriminating-cue trials with the count winner also winning the only cue, so CDCC's S = 1, C = +1 and its prediction is EXACTLY IDENTICAL for the two cells regardless of parameters (~0.94). RDTI predicts C2 > C1 by 0.08-0.23 (the solo star is only partially released from its redundancy discount: eps_eff = eps_w + (1-eps_w)*psi < 1), and predicts C1 to sit noticeably below CDCC's ~0.94 ceiling. Comparing C1 (0.79-ish, star alone) with C5 (0.07, same star embedded in a dense display) also isolates RDTI's claim that a star is discounted only when there is a chorus to be redundant with.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition; validity rank is decoupled from screen position; no pair is degenerate (every trial requires reading several ratings and no cell has the two models making the same prediction).

[7] GOAL: dissociate Count-Direction with Conflict-Graded Confidence (CDCC, advocated) from Redundancy-Discounted Tiered Integration with Majority Discounting (RDTI-MCR, competing). CDCC makes DIRECTION purely count-based (exact 0.5 on tally ties, parameter-free; never sub-chance adherence to the count winner) and lets validity enter only as a confidence gate, with dilution S = M/D^gamma. RDTI-MCR median-splits the panel into a discounted 'star' tier and a full-weight 'marginal' tier, adds a contrarian discount proportional to the count margin, and releases the star discount only for essentially solo displays (D=1). This design attacks the three points where their signs, not just their magnitudes, diverge.

PANEL: 8 experts, validities in scrambled screen order [0.95, 0.56, 0.85, 0.60, 0.65, 0.90, 0.53, 0.80]. Descending rank f0(.95) > f5(.90) > f2(.85) > f7(.80) >> f4(.65) > f3(.60) > f1(.56) > f6(.53), so RDTI's median split is unambiguous (STAR = {f0,f5,f2,f7}, MARGINAL = {f4,f3,f1,f6}) with a wide .80/.65 gap at the split, and CDCC's convex weights w=(v-.5)^3 are .091/.064/.043/.027 for the stars versus .0034/.0010/.0002/.00003 for the marginals (>25:1), so its coherence gate is star-driven. n=8 (rather than 6) also puts RDTI's capacity term (n/6)^tau above 1, which shrinks its evidence term relative to the count-margin discount and sharpens the predicted sign reversals. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the next trial is its exact A/B mirror):
C1 (T1/T2) TALLY-TIED 3v3, TIER-MISMATCHED: A wins marginals f4,f3,f1; B wins stars f0,f5,f2; f6,f7 tied.
C2 (T3/T4) TALLY-TIED 2v2, TIER-MISMATCHED: A wins the two weakest experts f6(.53), f1(.56); B wins the two best f0(.95), f5(.90); f2,f3,f4,f7 tied.
C3 (T5/T6) DENSE STAR-MAJORITY (D=6, M=+2): A wins all four stars f0,f5,f2,f7; B wins marginals f4,f3; f1,f6 tied.
C4 (T7/T8) DENSE MARGINAL-MAJORITY (D=6, M=+2): A wins all four marginals f4,f3,f1,f6; B wins stars f0,f5; f2,f7 tied.
C5 (T9/T10) SPARSE UNANIMOUS STAR (D=2, M=+2): A wins only f0(.95) and f5(.90); six cues tied.
C6 (T11/T12) MID STAR-MAJORITY (D=4, M=+2): A wins stars f0,f5,f2; B wins marginal f4; f1,f3,f6,f7 tied.

QUANTITATIVE PREDICTIONS, P(choose A).
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): C1 = 0.500, C2 = 0.500, C3 = 0.650, C4 = 0.532, C5 = 0.928, C6 = 0.754.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1 = 0.935, C2 = 0.911, C3 = 0.086, C4 = 0.941, C5 = 0.487, C6 = 0.180.

THREE ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED TIER-MISMATCHED CELLS (C1, C2). CDCC MUST be exactly 0.5 for every beta, gamma, kappa, rho, epsilon (nA = nB returns the coin flip before any validity computation). RDTI predicts strong, systematic choice of the option backed only by the LOW-validity marginal experts (0.85-0.97 across its whole parameter box, since E = k(1 - eps_w) > 0 with eps_w <= 0.34 and the count-margin term is exactly zero). Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive; note the sign also separates RDTI from WADD/TTB, which would predict below 0.5 here, so the cell is self-diagnostic three ways.
(2) SIGN REVERSAL ON STAR-BACKED MAJORITIES (C3 and C6 versus C4). C3, C4 and C6 all have count margin +2 for A. CDCC must put ALL THREE above 0.5 (conflict can only erode confidence toward chance, never past it) and predicts the coherent star-backed cells ABOVE the conflict cell: C3 0.65 and C6 0.75 versus C4 0.53. RDTI predicts C3 = 0.09 and C6 = 0.18 (sub-chance: 4*eps_w <= 1.36 < 2 so the tier-weighted evidence is negative whenever the count winner's wins are the stars, and the contrarian majority discount pushes it further down), while C4 = 0.94 -- an ~0.85 swing between profiles CDCC treats as same-direction and similarly signed. Observing systematic sub-chance adherence to the count winner in C3/C6 falsifies CDCC outright; observing C3 > C4 > 0.5 falsifies RDTI.
(3) DILUTION LADDER AT CONSTANT MARGIN AND CONSTANT COHERENCE (C5, C6, C3: D = 2, 4, 6 with M = +2 and the count winner always holding the stars). CDCC's S = M/D^gamma predicts a steep monotone decline 0.93 -> 0.75 -> 0.65, with near-ceiling (0.90-0.95) adherence on the sparse unanimous star cell C5 -- the classic CDCC signature that sparse clean evidence beats dense larger-D evidence. RDTI predicts the same ordering but hugging or crossing below chance (0.49 -> 0.18 -> 0.09), because its graded redundancy release is essentially confined to D = 1 (relax = psi*exp(-(D-1)/kappa_d) is already <= 0.03 at D = 2) while the count-margin discount of 0.45*2 applies in all three cells. Thus C5 is the cleanest magnitude test: CDCC needs >= 0.85, RDTI at nominal parameters sits at chance and only reaches ~0.8 at the extreme corner of its box.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition or tier; validity rank is scrambled relative to screen position so positional heuristics cannot mimic tiering; no pair is degenerate (every trial requires reading several ratings) and no cell has the two models making the same prediction.

INTERPRETATION: exact chance on C1/C2, above-chance adherence to the count winner in every one of C3-C6 with the coherent cells highest, and a steep sparse-to-dense decline 0.93/0.75/0.65 at fixed margin uniquely supports CDCC. Strong marginal-tier-driven preferences on the tally-tied cells plus sub-chance choice of the star-backed majority in C3/C6 uniquely supports RDTI-MCR.

[8] GOAL: dissociate RDTI-MCR (advocated) from GVIT-CAP-S4 (competing). Both models discount high-validity experts as 'redundant', both release the discount when only one cue discriminates, and both compress evidence by D, so generic conflict trials are useless. They differ on exactly three things, and the panel is engineered so that each difference produces a SIGN reversal or a parameter-free contrast.

PANEL: 6 experts, validities in scrambled screen order [f0=.95, f1=.78, f2=.82, f3=.55, f4=.76, f5=.80]. Descending rank f0>f2>f5>f1>f4>f3. Two structural properties are deliberate. (a) The median split falls INSIDE a tight cluster (.80 vs .78), so RDTI's coarse ordinal tiering (STAR={f0,f2,f5} weight eps_w; MARGINAL={f1,f4,f3} weight 1) cuts sharply between two nearly identical stated validities. (b) There is one lone floor expert (f3=.55) far below a dense upper cluster, so GVIT's strongly saturating inversion f(s)=s^omega (omega .26-.42) gives ALL of f0,f2,f5,f1,f4 nearly the same (and, for lam>1.1, negative) weight while f3 alone keeps weight +1. Hence GVIT ~ 'follow the least accurate expert, mildly opposed by everyone else' whereas RDTI ~ 'follow the bottom-half tier, stars count 0.18'. Validity rank is decoupled from screen position; n=6 keeps RDTI's capacity term exactly 1 and is below GVIT's 7.5 overload threshold, removing nuisance factors. 6 configurations x 2 exact A/B mirrors = 12 unique pairs x 8 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0.

CELLS (A-favouring version listed; the following trial is its exact mirror):
K1 (T1/T2) TALLY-TIED, MEDIAN-BOUNDARY: A wins f5(.80, star tier), B wins f1(.78, marginal tier); 4 cues tied.
K2 (T3/T4) TALLY-TIED 2v2, WIDE MISMATCH: A wins f4(.76)+f3(.55) [marginals], B wins f0(.95)+f2(.82) [stars].
K3 (T5/T6) MID-CLUSTER vs FLOOR: A wins f1(.78)+f4(.76), B wins f3(.55); f0,f2,f5 tied (D=3, margin +1).
K4 (T7/T8) DENSE MIXED (D=6, margin +2): A wins f0,f2,f5 (all stars) AND f3 (floor); B wins f1(.78),f4(.76).
K5 (T9/T10) SOLO STAR: only f0(.95) discriminates, A wins it (D=1).
K6 (T11/T12) SOLO FLOOR: only f3(.55) discriminates, A wins it (D=1).

QUANTITATIVE PREDICTIONS, P(choose A) at nominal parameters.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): K1=0.14, K2=0.93, K3=0.81, K4=0.16, K5=0.77, K6=0.85.
GVIT-CAP-S4 (lam=1.45, beta=6.7, gamma=.90, kappa_d=.10, psi=.40, chi=.95, eps=.17, omega=.34): K1=0.500, K2=0.500, K3=0.12, K4=0.62, K5=0.914, K6=0.914.

FOUR ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED CELLS K1, K2 are parameter-free for GVIT: nA=nB triggers the count gate and returns EXACTLY 0.5 for every lam, beta, gamma, kappa_d, psi, chi, eps, omega. RDTI predicts strong, systematic preference for the marginal-tier-backed option: across its entire parameter box K1 gives P(A)=0.08-0.27 (i.e. 73-92% choice of the f1-backed option) and K2 gives 0.82-0.95. Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive. K1 additionally localises the tier boundary to the .80/.78 crack: a subject who tiers at all but places the cut elsewhere will be at chance on K1 while still strong on K2.
(2) SIGN REVERSAL K3 (mid cluster vs floor expert). RDTI: all three discriminating cues are in the marginal tier, so E=+1 exactly regardless of eps_w; P(A)=0.71-0.93 over the whole box. GVIT: the saturating inversion makes f1 and f4 (.78/.76) almost as 'redundant' as the stars while f3 keeps full weight, so E<0 for every admissible lam/omega; P(A)=0.10-0.25. An ~0.6-0.8 swing on identical stimuli. Observing that subjects follow the two middling experts over the single worst expert falsifies GVIT's saturating-inversion commitment; following the .55 expert against two .76-.78 experts falsifies RDTI.
(3) SIGN REVERSAL K4 (dense, margin +2 for A). RDTI: E=3*eps_w+1-2 <= 0.02, and the contrarian consensus discount subtracts count_bias*2, giving P(A)=0.10-0.40 (always sub-chance) - the star-backed majority is REJECTED. GVIT: the floor expert f3 favours A with weight +1 while the opposing mids carry negative weight, so E>0 and P(A)=0.50-0.85. This is the cleanest test of RDTI's majority-discount plus fixed two-valued tiering against GVIT's graded inverse-validity weights.
(4) SOLO CELLS K5 vs K6 (both D=1). For GVIT, lam_eff = lam*(D-1)/((D-1)+kappa_D) = 0 exactly at D=1, so ALL weights are 1 and the two cells have IDENTICAL predictions (~0.86-0.99) for every parameter setting - a parameter-free equality. RDTI releases the star discount only partially (eps_eff = eps_w+(1-eps_w)*psi < 1), so it predicts solo-floor > solo-star by 0.05-0.25 (0.85 vs 0.77 nominally) and predicts the solo-star cell to sit well below GVIT's >=0.86 floor. Comparing K5 (star alone, ~0.77) with K4 (same star embedded in a chorus, ~0.16) also isolates RDTI's graded-relaxation claim.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; the number of displayed 1s does not track composition; validity rank is scrambled relative to screen position; no pair is degenerate and no cell has the two models predicting the same thing.

INTERPRETATION: exact chance on K1/K2 plus following the .55 expert in K3 and the floor-backed option in K4, with equal near-ceiling adherence on both solo cells, uniquely supports GVIT-CAP-S4. Strong marginal-tier-driven choices on the tally-tied cells, adherence to the two middling experts in K3, sub-chance choice of the star-backed majority in K4, and a lower solo-star than solo-floor adherence uniquely support RDTI-MCR.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate GVIT-CAP-S4 (advocated) from RDTI-MCR (competing). Both models discount high-validity experts as 'redundant', both release the discount when a single cue discriminates, and both compress evidence by the number of discriminating cues, so ordinary conflict trials are useless. They differ on exactly three points, and the panel is engineered so each difference produces either a parameter-free equality that only one model can honour, or a sign reversal.\n\nPANEL: 7 experts, validities in deliberately scrambled screen order [f0=.74, f1=.95, f2=.55, f3=.78, f4=.70, f5=.52, f6=.90]. Descending rank f1(.95) > f6(.90) > f3(.78) > f0(.74) > f4(.70) > f2(.55) > f5(.52). Two structural properties are deliberate. (a) RDTI's median split (k = 7//2 = 3) makes STAR = {f1,f6,f3} (weight eps_w) and MARGINAL = {f0,f4,f2,f5} (weight 1), and CRUCIALLY the marginal tier contains BOTH a middling pair (.74/.70) and a genuine bottom pair (.55/.52); because RDTI's weights take only two values, these two pairs are interchangeable for it. (b) GVIT's panel-relative saturating inversion f(s)=s^omega (omega .26-.42) with vmin=.52, vmax=.95 assigns s^.34 = 1.00/.959/.843/.796/.744/.405/0 to f1/f6/f3/f0/f4/f2/f5, so the middling experts (.74,.70) are discounted almost as heavily as the stars (weights ~ -0.07 and ~ +0.00 at lam_eff=1.34) while the two bottom voices keep large positive weight (+0.46 and +1.00). Hence GVIT sharply separates a mid-backed majority from a bottom-backed one, whereas RDTI cannot. n=7 is below GVIT's 7.5 capacity threshold (o=.076), so the inversion is intact and only mildly noisy. 6 configurations x 2 exact A/B mirrors = 12 unique pairs x 8 repetitions = 96 trials; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition.\n\nCELLS (A-favouring version listed; the following trial is its exact mirror):\nC1 (T1/T2) MID-MAJORITY vs LONE STAR (D=3, margin +1): A wins f0(.74) and f4(.70); B wins f1(.95); f2,f3,f5,f6 tied.\nC2 (T3/T4) BOTTOM-MAJORITY vs LONE STAR (D=3, margin +1): A wins f2(.55) and f5(.52); B wins f1(.95); f0,f3,f4,f6 tied. IDENTICAL tier composition to C1 for RDTI.\nC3 (T5/T6) TALLY-TIED 2v2, TIER-MISMATCHED: A wins f2(.55), f5(.52); B wins f1(.95), f6(.90); f0,f3,f4 tied.\nC4 (T7/T8) MID-PAIR vs FLOOR (D=3, margin +1): A wins f0(.74), f4(.70); B wins f5(.52); f1,f2,f3,f6 tied.\nC5 (T9/T10) SOLO STAR (D=1): only f1(.95) discriminates, A wins it.\nC6 (T11/T12) SOLO FLOOR (D=1): only f5(.52) discriminates, A wins it.\n\nQUANTITATIVE PREDICTIONS, P(choose A) at nominal parameters.\nGVIT-CAP-S4 (lam=1.45, beta=6.7, gamma=.90, kappa_d=.10, psi=.40, chi=.95, eps=.17, omega=.34): C1=0.63, C2=0.90, C3=0.500 (exact), C4=0.15, C5=0.91, C6=0.91.\nRDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1=0.92, C2=0.92 (exactly equal), C3=0.92, C4=0.83, C5=0.75, C6=0.84.\n\nFOUR ORTHOGONAL DIAGNOSTICS:\n(1) MID vs BOTTOM MAJORITY (C1 vs C2) is PARAMETER-FREE for RDTI: both cells have two marginal-tier winners, one star-tier loser, D=3, margin +1, same n, so E = 2 - eps_w and the predicted P(A) is IDENTICAL for every eps_w, beta, delta, tau, lapse, count_bias, psi, kappa_d. GVIT's saturating inversion predicts a large gap: E(C1) = 1 - lam_eff*(s0^om + s4^om - 1) ranges 0.06-0.53 over its whole box (and can go slightly negative at omega=.26, high lam), while E(C2) = 1 + lam_eff*(1 - s2^om) ranges 1.5-2.0, giving C1 ~ 0.52-0.82 versus C2 ~ 0.86-0.96, i.e. a C2-minus-C1 gap of about 0.15-0.40 that RDTI must set to exactly 0. Each subject contributes 16 trials per cell, so this within-subject contrast is decisive.\n(2) TALLY-TIED TIER-MISMATCHED CELL (C3) is parameter-free for GVIT: nA = nB triggers the count gate and returns EXACTLY 0.5 for every lam, beta, gamma, kappa_d, psi, chi, eps, omega, with no validity tie-break. RDTI predicts a strong, systematic preference for the option backed only by the two WORST experts (E = 2(1-eps_w) >= 1.32, margin = 0 so the contrarian discount is inert): 0.85-0.96 across its entire box. Note the sign is also opposite to WADD/TTB (which would go below 0.5 here), so the cell is self-diagnostic three ways.\n(3) SIGN REVERSAL ON C4 (mid pair against the floor expert). RDTI: all three discriminating cues are marginal-tier, so E = +1 EXACTLY regardless of eps_w, and P(A) = 0.72-0.92 over the box. GVIT: E = 1 - lam_eff*(s0^om + s4^om) is negative for every admissible lam and omega (-0.33 to -1.9), so P(A) = 0.10-0.35 - subjects should follow the single .52 expert AGAINST two .70-.74 experts. Comparing C4 with C1, which is the same A-side profile with only the opponent changed from the floor expert to the star, isolates GVIT's claim that a lone .95 expert is heard as counter-indicative while a lone .52 expert is heard as decisive: GVIT predicts a ~0.48 swing (0.63 vs 0.15), RDTI predicts both high and close (0.92 vs 0.83).\n(4) SOLO CELLS C5 vs C6 (both D=1). For GVIT lam_eff = lam_n*(D-1)/((D-1)+kappa_d) = 0 exactly at D=1, so all weights equal 1 and the two cells have IDENTICAL predictions (0.865-0.965) for every parameter setting - a parameter-free equality, and a high floor. RDTI releases the star discount only partially (eps_eff = eps_w + (1-eps_w)*psi < 1), so it predicts solo-floor > solo-star (nominally 0.84 vs 0.75) and allows the solo-star cell to fall as low as ~0.5, below GVIT's reachable floor.\n\nCONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tied cues alternate between 1-1 and 0-0 coding; no pair is degenerate (every trial requires reading several ratings) and no cell has the two models predicting the same thing.\n\nINTERPRETATION: exact chance on C3, a large bottom-backed-minus-mid-backed gap (C2 >> C1) at matched composition, sub-chance choice of the mid-backed majority in C4, and equal near-ceiling adherence on the two solo cells uniquely support GVIT-CAP-S4. Equal adherence on C1 and C2, strong marginal-tier-driven choice on the tally-tied C3, above-chance adherence in C4, and solo-floor exceeding solo-star uniquely support RDTI-MCR.",
  "validities": [
    0.74,
    0.95,
    0.55,
    0.78,
    0.7,
    0.52,
    0.9
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
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
      0,
      1
    ],
    [
      1,
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
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
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
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
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
  "input_tokens": 26644,
  "output_tokens": 16146
}
```
