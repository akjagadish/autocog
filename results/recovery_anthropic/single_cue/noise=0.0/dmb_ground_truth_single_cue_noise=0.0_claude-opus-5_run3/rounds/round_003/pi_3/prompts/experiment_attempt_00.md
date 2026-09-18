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
**Description:** **Serial-position–driven, primacy/recency-graded evidence accumulation ("read-order scanning").**

People do not re-sort the expert ratings by the stated validities before deciding. They inspect the ratings in the order in which they appear on the screen (reading order) and their attention decays sharply across that inspection sequence, so the evidence carried by a cue is discounted geometrically by its *serial position*, not by its validity. Stated validities exert only a weak, partial pull on the inspection order (parameter gamma, small).

Two further claims:

1. **The gradient is steep but graded.** Weights fall off as w_j = exp(-lambda * rank_j) with |lambda| well above zero, so decisions are usually driven by the first cue(s) that discriminate — near-lexicographic but not strictly all-or-none: when the leading cues tie, the next-inspected cues take over, and when several later cues line up they can occasionally overturn a leading cue if the gradient is shallow.

2. **The gradient has a sign: most readers are primacy-oriented, a substantial minority are recency-oriented.** Reading a short list produces either an anchoring-on-the-first-item strategy (primacy, lambda > 0) or a last-impression-counts strategy (recency, lambda < 0). The population is bimodal: about one third of subjects anchor on the *last-read* cue. Near-zero gradients (i.e., genuine equal-weight tallying) are essentially absent — people do not integrate all six/five cues equally.

Evidence is evaluated *relatively*: the signed weighted difference is normalised by the total weight of the cues that actually discriminate, so a decision based on one early cue is as confident as a decision based on many, and only genuinely near-balanced weighted evidence produces hesitation. Choice is a softmax on this relative evidence with inverse temperature beta, plus a lapse epsilon.

Because validity order and screen order coincide in a validity-sorted display but dissociate in a scrambled display, this theory predicts TTB-looking behaviour in the former and tally-looking behaviour in the latter — without either heuristic being the actual mechanism. The causal driver is *where the cue sits on the screen*.

**Parameters:**
- lam: [1.2, 3.5]
- gamma: [0.0, 0.25]
- beta: [4.0, 12.0]
- epsilon: [0.0, 0.10]
- scan_orientation: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---- unpack the trial stimulus -> (2, n_features) ------------------
    if isinstance(state, dict):
        a = np.asarray(list(state.get('option_a_ratings', [])), dtype=float)
        b = np.asarray(list(state.get('option_b_ratings', [])), dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            stim = stim.reshape(2, -1)
        elif stim.ndim > 2:
            stim = stim.reshape(2, -1)
    if stim.shape[0] != 2:
        stim = stim.reshape(2, -1)

    a = stim[0].astype(float)
    b = stim[1].astype(float)
    n = a.shape[0]
    d = a - b

    # ---- inspection ranks: screen position, weakly pulled by validity --
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fallback: assume the display is already validity-sorted
        val = np.linspace(0.95, 0.55, n)

    order = np.argsort(-val, kind='stable')       # 0 = highest validity
    vrank = np.empty(n, dtype=float)
    vrank[order] = np.arange(n, dtype=float)
    pos = np.arange(n, dtype=float)               # reading-order index

    gamma = float(parameters.get('gamma', 0.0))
    gamma = min(max(gamma, 0.0), 1.0)
    rank = gamma * vrank + (1.0 - gamma) * pos

    # ---- signed attention gradient (primacy vs recency reader) --------
    lam_mag = float(parameters.get('lam', 2.0))
    orient = float(parameters.get('scan_orientation', 1.0))
    RECENCY_SHARE = 0.35                          # population constant
    lam = -lam_mag if orient < RECENCY_SHARE else lam_mag

    z = -lam * (rank - float(np.mean(rank)))      # centered for stability
    z = z - float(np.max(z))
    w = np.exp(z)

    # ---- relative (normalised) weighted evidence ----------------------
    num = float(np.sum(w * d))
    den = float(np.sum(w * np.abs(d)))
    if den <= 1e-12:
        S = 0.0                                   # nothing discriminates
    else:
        S = num / den                             # in [-1, 1]

    beta = float(parameters.get('beta', 6.0))
    eps = float(parameters.get('epsilon', 0.0))
    eps = min(max(eps, 0.0), 1.0)

    logits = np.array([beta * S, 0.0], dtype=float)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    p = e / np.sum(e)

    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
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


## COMPETING THEORY
**Description:** **Anchored compensatory integration with load-gated degradation of weighted-additive processing and partial evidence amplification.**

(1) *Diagnosticity is read convexly.* The subjective pull of expert j is d_j = (v_j - 0.5)^rho with rho ~2.3-2.7, so a .95 expert is treated as a near-oracle and .55-.65 experts as near-noise.

(2) *Screen position is a graded attention multiplier, not a search order.* a_j = 1 + alpha*exp(-lam*j) with alpha ~1.5-2.1, lam ~0.9-1.3; the cue weight is the PRODUCT w_j ∝ d_j * a_j, rescaled once to sum to 1 (never renormalised per trial). The first-read cue is rehearsed longest and anchors the running impression, breaking validity ties toward the pos-0 side, but it cannot overturn a genuine diagnosticity gap.

(3) **Processing load is the single switch that degrades weighted-additive integration toward unit-weight counting.** Let L be the number of features on which BOTH products received a positive expert rating (design-general form: the count of non-discriminating positive endorsements; equivalently L/n_features as a density). Under this load two things happen together, both expressions of the same effort–accuracy tradeoff: (a) the diagnosticity exponent flattens, rho_eff = rho / (1 + c*L), so validity differentiation degrades toward equal weighting; (b) the decision maker falls back on the crude 'how many experts back each side' cue, i.e. supra-additive corroboration appears with delta_eff = delta * L/(1+L). Corroboration is silent on sparse displays (L = 0): counting is a load-induced shortcut, not a permanent feature of integration.

(4) **NEW — the margin is read against a partially-attenuated yardstick, not against the raw amount of evidence considered.** Two trials with the same signed weighted margin are not equally decisive: a margin obtained while very little total diagnostic weight was in play feels proportionally larger than the same margin obtained amid a large mass of conflicting weight, but the correction is only PARTIAL. Evidence is E = (Score_A - Score_B) / D^psi, where D = total discriminating weight and psi ~0.2-0.4 is well below 1. At psi = 0 this is the pure absolute margin (all thin-evidence trials near chance); at psi = 1 it would be the fully relative, per-trial-normalised evidence the arbiter rejected (every decision equally confident). Real subjects sit near the low end: thin, near-balanced displays are magnified enough to escape chance, while dense displays keep an essentially absolute margin. This is the theory's sharpest quantitative claim — decisiveness scales sub-linearly with the reciprocal of the evidence mass.

(5) Choice = logistic(beta*E) with a lapse epsilon. No early stopping, no threshold rule, no conclusion-graded commitment, no full per-trial normalisation, and individual differences are unimodal jitter in rho, alpha, lam, delta, c, psi, beta and lapse — no strategy subpopulations, no primacy/recency reader types.

**Parameters:**
- rho: [2.3, 2.7]
- alpha: [1.5, 2.1]
- lam: [0.9, 1.3]
- delta: [0.30, 0.55]
- load_flatten: [0.65, 1.10]
- psi: [0.20, 0.40]
- beta: [6.0, 8.5]
- epsilon: [0.02, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. unpack the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b

    if not np.any(np.abs(d) > 1e-12):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> convex diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fallback: assume the display is validity-sorted, descending
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0)

    rho = float(np.clip(parameters.get('rho', 2.5), 0.1, 8.0))
    alpha = float(np.clip(parameters.get('alpha', 1.8), 0.0, 6.0))
    lam = float(np.clip(parameters.get('lam', 1.1), 0.01, 6.0))
    beta = float(np.clip(parameters.get('beta', 7.0), 0.0, 200.0))
    eps = float(np.clip(parameters.get('epsilon', 0.05), 0.0, 1.0))
    delta = float(np.clip(parameters.get('delta', 0.40), 0.0, 2.0))
    c_load = float(np.clip(parameters.get('load_flatten', 0.85), 0.0, 3.0))
    psi = float(np.clip(parameters.get('psi', 0.30), 0.0, 0.65))

    # --- overlapping positive endorsements = processing load.
    # L = number of features on which BOTH options were endorsed.  These
    # cues carry no discriminating information but they enlarge the
    # comparison problem, and under that load validity differentiation
    # degrades toward unit-weight counting.
    L = float(np.count_nonzero((a > 0.5) & (b > 0.5)))
    rho_eff = rho / (1.0 + c_load * L)
    rho_eff = float(np.clip(rho_eff, 0.6, 8.0))

    diag = np.power(np.clip(val - 0.5, 1e-12, None), rho_eff)   # diagnosticity

    # ------------------------------------------------------------------
    # 3. graded primacy attention multiplier (screen-position anchor)
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    att = 1.0 + alpha * np.exp(-lam * pos)

    w = diag * att
    tot = float(np.sum(w))
    if (not np.isfinite(tot)) or tot <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = w / tot            # fixed rescaling only (NOT per-trial normalisation)

    # ------------------------------------------------------------------
    # 4. LOAD-GATED supra-additive corroboration: the crude 'how many
    #    experts back each side' bonus is a shortcut people fall back on
    #    only when the display is dense (L > 0).  On sparse displays
    #    (L = 0) delta_eff = 0 and integration stays purely
    #    weighted-additive.
    # ------------------------------------------------------------------
    delta_eff = delta * (L / (1.0 + L))

    pos_mask = d > 1e-12
    neg_mask = d < -1e-12
    s_a = float(np.sum(w[pos_mask] * np.abs(d[pos_mask])))
    s_b = float(np.sum(w[neg_mask] * np.abs(d[neg_mask])))
    k_a = float(np.count_nonzero(pos_mask))
    k_b = float(np.count_nonzero(neg_mask))
    f_a = (k_a ** delta_eff) if k_a > 0.0 else 0.0
    f_b = (k_b ** delta_eff) if k_b > 0.0 else 0.0

    # ------------------------------------------------------------------
    # 5. PARTIAL amplification: the raw margin is read against a
    #    sub-linearly attenuated yardstick D^psi (psi well below 1), so
    #    thin displays are magnified somewhat while dense displays keep an
    #    essentially absolute margin.  psi = 0 -> pure absolute margin;
    #    psi = 1 would be the forbidden full per-trial normalisation.
    # ------------------------------------------------------------------
    D = s_a + s_b
    if not np.isfinite(D) or D <= 1e-12:
        return np.array([0.5, 0.5])
    E = (s_a * f_a - s_b * f_b) / (D ** psi)

    z = np.clip(beta * E, -60.0, 60.0)
    p_a = 1.0 / (1.0 + np.exp(-z))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p
```

**`policy source code`:**
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with clearly ordered, well-spread validities (0.95, 0.85, 0.75, 0.65, 0.55) so that Take-The-Best's cue cascade is unambiguous. The 12 unique pairs (each repeated 8 times, ~96 trials) are chosen to create four distinct diagnostic classes:

(1) CONFLICT / non-compensatory pairs (trials 1, 2, 8, 10): the option favored by the single highest-validity discriminating cue is beaten on the tally (e.g. A=[1,0,0,0,0] vs B=[0,1,1,1,1], tally 1-4 for B). TTB predicts a confident choice of A; Tallying predicts a confident choice of B, with margin-graded confidence (4-cue margin in trials 1/8 vs 1-cue margin in trials 2/10).

(2) TALLY-TIE pairs (trials 3, 4): A wins two cues, B wins two cues, so Tallying is exactly at chance (50/50 regardless of beta and epsilon), whereas TTB decides confidently on the top cue. Any systematic deviation from 50% here is impossible for Tallying at any parameter setting and is therefore the sharpest single-trial discriminator.

(3) LOW-CUE-DECIDES pairs (trials 7, 9): the top two cues tie, the third cue (validity .75) discriminates in one direction while the two lowest cues both favor the other option, so TTB and Tallying make opposite predictions in a case where TTB has to descend the cascade rather than read the first cue.

(4) AGREEMENT pairs with different tally margins (trials 5, 6, 11, 12): both theories pick the same option, but Tallying's softmax makes accuracy strongly margin-dependent (4-1 vs 2-1), while TTB predicts identical choice probability sigmoid(beta) irrespective of margin. Comparing accuracy on trials 5/11 versus 6/12 gives a purely quantitative dissociation that does not depend on which option is 'correct'.

All pairs are mirrored across A and B to avoid side bias, and no pair is degenerate (every pair has at least one discriminating cue and the two theories either disagree in direction or in predicted confidence).

[1] Six experts with well-spread but deliberately *non-monotone* validity order across screen positions (0.90, 0.60, 0.80, 0.55, 0.70, 0.95) so that Take-The-Best's cue cascade (f5 > f0 > f2 > f4 > f1 > f3) cannot be confused with a simple left-to-right reading strategy, while Tallying (which ignores order entirely) is unaffected. Twelve unique pairs, each repeated 8 times (96 trials), fall into four diagnostic classes.

(1) NON-COMPENSATORY CONFLICT with graded tally margin (trials 1-4). The single highest-validity cue (f5, 95%) favours one option while the tally favours the other, by margin 4 (trials 1, 2), margin 2 (trial 3) or margin 1 (trial 4). TTB predicts the SAME confidence sigmoid(beta) for all four, always siding with the f5 winner; Tallying predicts the opposite direction with confidence strictly increasing in the margin (sigmoid(4*beta) > sigmoid(2*beta) > sigmoid(beta)). Direction AND the margin gradient jointly identify the model.

(2) TALLY-TIE pairs (trials 5-8). Feature-wise wins are exactly balanced (3-3 in trials 5/6 with no ties; 2-2 with two tied cues in trials 7/8), so Tallying is pinned at exactly 50% for every (beta, epsilon) - no parameter setting can move it. TTB decides confidently: on trials 5/6 the top cue f5 discriminates immediately; on trials 7/8 f5 ties and TTB must descend to f0 (90%). Any systematic departure from chance here is impossible for Tallying, making these the sharpest single-cell discriminators, and trials 7/8 additionally verify that subjects use validity rank rather than screen position.

(3) LOW-CUE-DECIDES CONFLICT (trials 9, 10). The two most valid cues (f5, f0) tie; the third-ranked cue f2 (80%) discriminates one way while the three lower cues (f4, f1, f3) all favour the other option (tally 1-3). TTB must walk the cascade and then choose against the majority; Tallying chooses the majority. Opposite directions in a case where TTB's one-reason logic is hardest to mimic.

(4) AGREEMENT pairs with extreme margin contrast (trials 11, 12). Both theories favour the same option, but with tally margin 5 (trial 11) versus margin 1 (trial 12). TTB predicts identical accuracy on the two; Tallying predicts near-ceiling on trial 11 and markedly lower accuracy on trial 12. This is a purely quantitative dissociation that is independent of choice direction and of any objective 'correct' answer.

All classes are mirrored across sides (TTB predicts A on 6 pairs and B on 6; Tallying's non-tie predictions are likewise 4/4), so side bias cannot masquerade as either model. No pair is degenerate: every pair has at least one discriminating cue, and in every pair the two theories differ either in predicted direction or in predicted confidence.

[2] GOAL: dissociate Take-The-Best (validity-ordered lexicographic search) from the 'read-order scanning' model (geometric weighting by SCREEN position, primacy for ~65% of subjects, recency for ~35%, with only a weak validity pull gamma<=0.25).

KEY DESIGN MOVE: the display order is deliberately NOT validity-ordered, and crucially the top-validity cue is placed in the MIDDLE of the screen (position 3, 95%), while the second-best cue sits at the END (position 5, 85%) and a mid-validity cue sits FIRST (position 0, 70%). Validities by screen position are [.70, .55, .60, .95, .65, .85], so TTB's cue cascade is pos3 > pos5 > pos0 > pos4 > pos2 > pos1. A primacy scanner anchors on pos0, a recency scanner anchors on pos5, and TTB anchors on pos3 -- three mutually distinguishable anchors. I verified numerically that this ordering is robust across the scanning model's whole parameter range: even at the maximum validity pull (gamma=.25) and the shallowest gradient (lam=1.2), pos0 still has the largest primacy weight and pos5 the largest recency weight, so the predictions below do not depend on lucky parameter values. (This is why pos0 was given 0.70 rather than the lowest validity: with a low-validity first cue the gamma=.25 pull would have flipped the primacy anchor onto the top-validity cue and destroyed the dissociation.)

TRIAL CLASSES (12 unique pairs, each repeated 8x = 96 trials; every pair mirrored across sides so side bias cannot mimic either model; TTB predicts A on 6 and B on 6, and so do both scanning orientations).

(1) DOUBLE-DISSOCIATION pairs 1-2 (A=[1,0,0,0,0,1] vs B=[0,0,0,1,0,0] and mirror). Only pos0, pos3, pos5 discriminate; the two screen-EXTREME cues favour one option and the top-validity MIDDLE cue favours the other. TTB must choose the pos3 option; BOTH primacy and recency scanners choose the other option (checked numerically for lam in [1.2,3.5] x gamma in [0,.25]: |S|>0.78 in every case). These are the sharpest cells -- no setting of the scanning model's parameters, of either orientation, can reproduce TTB here.

(2) PRIMACY-vs-TTB pairs 3-4 and 7-8. Pair 3: pos0 favours A, pos3 favours B, nothing else discriminates -> TTB=B, recency=B, primacy=A. Pair 7: pos3 ties so TTB must DESCEND to pos5 (A) while pos0 favours B -> TTB=A, recency=A, primacy=B. These isolate primacy readers and simultaneously verify that subjects use validity rank, not left-to-right position, when the leading screen cue conflicts with the cascade.

(3) RECENCY-vs-TTB pairs 5-6. pos3 (95%) favours B, pos5 (85%) favours A, nothing else discriminates -> TTB=B, primacy=B, recency=A. These isolate recency readers.

(4) COMPENSATORY / GRADED-CONFIDENCE pairs 9-12. Pair 9 (A=[1,0,0,0,0,0] vs B=[0,1,1,0,0,0]): one early cue against two slightly later, lower-validity cues -> TTB=A confidently, primacy=A but with markedly reduced relative evidence (S ~= 0.44-0.59 depending on lam), recency=B. Pair 11 (A=[0,0,0,0,0,1] vs B=[1,1,1,0,0,0]): the single last-position 85% cue against the three leading cues -> TTB=A, recency=A (S ~= 0.57-1.0), primacy=B with near-maximal evidence. Because TTB's choice probability is the SAME sigmoid(beta) on every non-tied pair, any systematic accuracy gradient across pairs 3/7 (near-|S|=1 for the scanner) versus pairs 9/11 (intermediate |S|) is a purely quantitative signature of the scanning model that TTB cannot produce at any (beta, epsilon).

DIAGNOSTIC YIELD: for a primacy subject TTB and the scanning model disagree in DIRECTION on 8 of 12 pairs (1,2,3,4,7,8,11,12); for a recency subject they disagree on 6 of 12 (1,2,5,6,9,10). Under TTB, choices should track pos3 (then pos5) regardless of screen position; under the competitor, choices should track pos0 (or pos5) and show margin-graded confidence. Every pair has at least one discriminating cue, so no cell is degenerate.

[3] GOAL: dissociate Take-The-Best (validity-ordered lexicographic search, one reason, constant confidence sigmoid(beta) on every discriminating pair) from 'read-order scanning' (geometric weighting by SCREEN position, ~65% primacy / ~35% recency readers, only a weak validity pull gamma<=0.25, and relative/normalised evidence that is graded when several later cues oppose the leading one).

DESIGN MOVE (new relative to previously explored designs): five experts, display order NOT validity-ordered, with the top-validity cue at screen position 2 and, crucially, the two screen EXTREMES given adjacent mid-high validities (pos0 = .78, pos4 = .72). Validities by screen position are [.78, .60, .95, .88, .72], so TTB's cascade is pos2 > pos3 > pos0 > pos4 > pos1. Three mutually distinct anchors: TTB anchors on pos2, a primacy scanner on pos0, a recency scanner on pos4. I verified the anchor assignment is robust over the scanning model's whole parameter box: with rank = gamma*vrank + (1-gamma)*pos, at the maximum validity pull gamma = .25 the ranks are [0.50, 1.75, 1.50, 2.50, 3.75], so pos0 is still the minimum (largest primacy weight) and pos4 still the maximum (largest recency weight); at gamma = 0 this holds trivially. Deliberately giving pos0 the third-highest validity (rather than the lowest) prevents the gamma pull from flipping the primacy anchor onto pos2, which would destroy the dissociation.

TRIAL CLASSES (12 unique pairs, each repeated 8x = 96 trials; every class mirrored across sides, so TTB predicts A on 6 and B on 6, and so do BOTH scanning orientations \u2014 side bias cannot mimic any model).

(1) DOUBLE DISSOCIATION, pairs 1-2: A=[1,0,0,0,1] vs B=[0,0,1,0,0]. Only the two screen extremes (favouring A) and the top-validity middle cue (favouring B) discriminate. TTB must choose the pos2 option; BOTH primacy and recency scanners choose the other option (worst-case relative evidence checked numerically: primacy S = .55 at gamma=.25, lam=1.2, rising to .94 at lam=3.5; recency S = .84 at gamma=0, lam=1.2). No parameter setting of either orientation reproduces TTB here \u2014 the sharpest cells.

(2) RECENCY-ISOLATING, pairs 3-4: A=[1,0,0,0,0] vs B=[0,0,0,0,1]. pos2 and pos3 tie, so TTB descends to pos0 (.78) and picks A; primacy also picks A (S ~ 1); recency picks B (S ~ -1). Under TTB the across-subject distribution of P(A) is unimodal near sigmoid(beta); under scanning it must be BIMODAL, with roughly one third of subjects locked on B. Subject-level bimodality on a cell where TTB is uniform is a signature TTB cannot produce at any (beta, epsilon).

(3) PRIMACY-ISOLATING, pairs 5-6: A=[1,0,0,0,0] vs B=[0,0,1,0,1]. TTB (pos2) and recency (pos4) both pick B; primacy picks A (S = .52 worst case at gamma=.25/lam=1.2, .82 at gamma=0). Complementary bimodality with the opposite majority direction.

(4) CASCADE-DESCENT DOUBLE DISSOCIATION, pairs 7-8: A=[0,0,0,1,0] vs B=[1,0,0,0,1]. The top cue pos2 ties, so TTB must walk to pos3 (.88) and choose A, while both scanners are dominated by the extremes and choose B (primacy S = -.84; recency S = -.54 to -.94 across the box). This verifies that any TTB-looking behaviour survives when the deciding validity cue is not the first-listed one, and doubles the direction-disagreement yield.

(5) COMPENSATORY / GRADED-CONFIDENCE, pairs 9-12. Pair 9: A=[1,0,0,0,0] vs B=[0,1,1,1,0] \u2014 one leading screen cue against the three next-read cues. TTB says B with its usual full confidence; a recency reader says B with near-maximal evidence (S = -.96); a primacy reader says A but with markedly REDUCED relative evidence (S = .24-.41 at lam=1.2, near 1 only at lam=3.5). Pair 11: A=[0,0,0,0,1] vs B=[0,1,1,1,0] \u2014 the mirror-image compensatory case for recency readers (recency S = +.41 weak, primacy S = -.96 strong, TTB = B). Because TTB predicts the SAME choice probability on all twelve pairs, any systematic accuracy gradient between the extreme-anchored pairs 1-8 (|S| near 1) and the compensatory pairs 9-12 (|S| intermediate for whichever orientation the subject has) is a purely quantitative signature of the scanning model, independent of direction and of any objective 'correct' answer.

DIAGNOSTIC YIELD: TTB and the scanning model disagree in DIRECTION on 8 of 12 pairs for a primacy subject (1,2,5,6,7,8,9,10) and on 8 of 12 for a recency subject (1,2,3,4,7,8,11,12). Under TTB choices track pos2 (then pos3, then pos0) regardless of screen location and show flat confidence; under the competitor they track pos0 or pos4, split bimodally across subjects, and show margin-graded confidence on the compensatory pairs. Every pair has at least one discriminating cue, so no cell is degenerate.

[4] GOAL: dissociate threshold-terminated compensatory integration with conclusion-graded commitment (weights fixed by validity, UNNORMALISED accumulation, early stop when no single unread expert could reverse the lead, amplified commitment after a stop vs. shallow readout of an unresolved residual) from read-order scanning (geometric weighting by SCREEN position, ~65% primacy / ~35% recency, evidence NORMALISED by the discriminating weight so one early cue is as confident as many).

DESIGN MOVE: five experts with validities by screen position [.78, .74, .70, .58, .90] \u2014 the single most valid expert (90%) is placed LAST, three mid-validity experts occupy the leading positions, and a near-useless expert (58%) sits at position 3. Under the advocated model the compressed weights are w \u2248 [.235, .191, .149, .043, .381] (stable across rho in [1.25,1.45]). Because the largest weight belongs to the last-read expert, the sufficiency bar for every earlier screen position is phi*w4 \u2248 .38-.44: NO single early cue can ever clear it, but a crowd of two-to-three leading mid cues can. The model therefore switches qualitatively as leading cues are added: 1 leading cue -> never stops, reads the 90% expert, shallow residual readout; 2 leading cues -> borderline/likely stop; 3 leading cues -> confident stop that never even reads the best expert (tallying-looking behaviour, choosing AGAINST the 90% expert). The competitor's weights are position-based, so its normalised evidence is near |S| = 1 on all of these, flat.

CORE LADDER (pairs 7/8, 3/4, 11/12, 5/6): the same target option is opposed by the last-listed 90% expert while it is supported by, respectively, (a) only pos0, (b) pos1+pos2, (c) pos0+pos1, (d) pos0+pos1+pos2. Advocated model predicts a steep MONOTONE gradient with a DIRECTION REVERSAL: P(support-side) \u2248 .25 (one cue, unresolved, best expert wins the residual), \u2248 .49 (two mid cues, unresolved and near chance), \u2248 .85 (pos0+pos1, stop typically triggered, amplified commitment despite a total weighted margin of only +.05), \u2248 .97 (three cues, stop certain). Read-order scanning predicts a FLAT ceiling: for a primacy reader |S| = .96-1.00 on all four (choose the leading-cue side every time), for a recency reader |S| = .78-1.00 in the opposite direction on all four. No (lam, gamma, beta, epsilon) can produce a monotone gradient plus a reversal across cells whose normalised evidence is essentially constant, and no accumulator with normalisation can put pair 3/4 near chance while pair 5/6 is at ceiling.

BOTH-ORIENTATIONS-AGREE CELL (pairs 1/2): d = [+1,-1,-1,+1,0] with the 90% expert tied. The earliest AND the latest discriminating cue favour the same option, so primacy AND recency scanners both choose it with S \u2248 +.45 to +.94 (P \u2248 .97-1.00) \u2014 no orientation escape hatch. The advocated model never clears the bar (max partial |E| = .235 < .38), so it exhausts the list and reads out a residual of only -.06, predicting a slight preference for the OPPOSITE option (P \u2248 .35-.43). This is the sharpest single cell: opposite direction, and the competitor is pinned near ceiling for its whole parameter box.

CASCADE-INVERSION CELL (pairs 9/10): d = [0,0,-1,0,+1] \u2014 the 70% expert at position 2 against the 90% expert at position 4. The advocated model cannot stop on the small pos2 cue, reads to the end and picks the 90% side (P \u2248 .76-.91); a primacy scanner picks the pos2 side robustly (pos2 keeps the lower inspection rank even at gamma = .25), a recency scanner picks the pos4 side. Direction disagreement with the 65% majority orientation, and a graded (not ceiling) confidence for the advocated model.

Every class is mirrored across sides (advocated model favours A on 6 pairs and B on 6; primacy scanning likewise 6/6), so side bias cannot mimic either model. 12 unique pairs x 8 repetitions = 96 trials. No pair is degenerate: every pair has at least one discriminating cue and the two theories differ in direction, in confidence level, or in the shape of the within-subject gradient across the ladder.

[5] GOAL: dissociate read-order scanning (position-geometric weights, ~65% primacy / ~35% recency readers, evidence NORMALISED by the discriminating weight so a one-cue decision is as confident as a six-cue decision) from threshold-terminated compensatory integration with conclusion-graded commitment (validity-only weights, UNNORMALISED accumulation, early stop only when no single unread expert can reverse the lead, kappa-amplified commitment after a stop vs. shallow beta readout of an unresolved residual).

DESIGN MOVE (new): six experts whose validities by SCREEN position are [.78, .66, .95, .86, .70, .56]. The two heaviest experts (95%, 86%) sit at positions 2 and 3, a mid expert (78%) leads, and the WEAKEST expert (56%) is last. Two consequences that the two theories read completely differently:
  (a) Competing model's compressed weights (rho in [1.25,1.45]) are w ~ [.18, .08, .34, .25, .11, .02]. Because w2=.34 is the largest, the sufficiency bar for positions 0 and 1 is phi*w2 ~ .33-.39, so NO combination of the two leading cues (max .27) can ever clear it; the model is forced to read the 95% expert, and if that expert opposes the leaders the residual is tiny and the trial ends UNRESOLVED with a shallow beta readout (near chance). Conversely a lone pos2 (or pos2+pos3) cue clears the bar at position 2 (.34/.28 = 1.2-1.4) and is committed with beta*kappa (ceiling).
  (b) Scanning anchors are robust and unambiguous: vrank = [2,4,0,1,3,5], so even at the maximum validity pull gamma=.25 the inspection ranks are [0.5, 1.75, 1.5, 2.5, 3.75, 5.0] - position 0 still has the smallest rank (primacy anchor) and position 5 (the 56% expert, whose low validity pushes it even later) still has the largest (recency anchor, gap 1.25 to pos4). Anchors hold for every lam in [1.2,3.5].

TRIAL CLASSES (12 unique pairs x 8 repetitions = 96 trials; every class mirrored, and the predicted-choice split is exactly 6 A / 6 B for the competing model AND for both scanning orientations, so side bias cannot mimic anything).

(1) SINGLE-CUE WEIGHT LADDER, pairs 1-4. Exactly one expert discriminates: pos5 (w=.02) in pair 1, pos1 (.08) in pair 2, pos0 (.18) in pair 3, pos2 (.34) in pair 4. For scanning, one discriminating cue gives |S| = 1 EXACTLY, for every lam, gamma and orientation, so all four cells must sit at the same flat ceiling sigmoid(beta) ~ .95-1.00. For the competing model a single discriminating cue can never trigger the reversal check (the list is exhausted at that cue), so it is read out with the shallow beta and accuracy is a steep MONOTONE ladder in the cue's validity weight: P ~ .52-.56 (pos5), .59-.74 (pos1), .67-.90 (pos0), .80-.98 (pos2). Flat ceiling vs. a four-step ladder rising from chance is a purely quantitative signature that no (beta, epsilon) of the normalising model can produce.

(2) BOTH-ORIENTATIONS-AGREE DIRECTION REVERSAL, pairs 7-8 (sharpest cells). d = [+1, 0, -1, 0, 0, +1]: the first-read 78% cue and the last-read 56% cue favour A, the mid-list 95% cue favours B. Primacy readers pick A (S = .54 worst case at gamma=.25/lam=1.2, .94 at lam=3.5); recency readers pick A even harder (S = .97-1.00 across the whole box) because the last position dominates. So scanning predicts P(A) ~ .90-1.00 with NO orientation escape hatch. The competing model cannot stop on pos0 (.18 < .33 bar), cannot stop on pos2 (|E| = .16 vs a .25-.29 bar), and exhausts the list with a residual of about -.14, predicting P(A) ~ .18-.35 - the OPPOSITE option. Opposite direction over both full parameter boxes.

(3) UNRESOLVED-RESIDUAL vs NORMALISED CEILING, pairs 5-6. d = [+1, +1, -1, 0, 0, +1]: three cues favour A but the single 95% cue opposes, leaving a final weighted margin of only -.055. Scanning: primacy S = .61-.87, recency S = .97-.99 - both orientations at ceiling for A (P >= .90) because normalisation makes the sign, not the size, of the margin decisive. Competing: the leaders may occasionally clear the pos1 bar (only at the shallowest s), otherwise the trial is unresolved and read out shallowly, giving P(A) ~ .36-.50. Same stimulus family as class (2) but here the dissociation is confidence-level rather than direction, which pins down normalisation independently of anchor identity.

(4) COMPETING-AT-CEILING / SCANNING-BIMODAL, pairs 9-10. d = [0, 0, +1, +1, 0, -1]: the 95% and 86% mid-list experts favour A, the last-read 56% expert favours B. The competing model stops at pos2 (ratio 1.16-1.41, stop prob .87-1.00) or certainly at pos3 (ratio > 4) and commits with beta*kappa: P(A) ~ .97-.99, unimodal across subjects. Scanning splits: primacy S = +.89 to +.98 (A), recency S = -.88 to -.99 (B), so the subject-level distribution of P(A) must be BIMODAL with ~35% of subjects locked on B and an aggregate near .65. Bimodality where the competitor is pinned at ceiling.

(5) MID-CUE STOP WITH A TINY MARGIN, pairs 11-12. d = [-1, 0, 0, +1, 0, 0]: the leading 78% expert favours B, the 86% expert at position 3 favours A. Competing: pos0 cannot stop, and after pos3 the margin (+.073) still fails the .11-.13 bar, so the trial ends unresolved and only weakly tilted, P(A) ~ .59-.67. Scanning: primacy S = -.83 to -.95 (B, ceiling), recency S = +.95 (A, ceiling) - again an extreme bimodal split with an aggregate near .35, i.e. the opposite direction from the competitor's modal prediction, plus within-subject near-determinism where the competitor predicts near-chance hesitation.

DIAGNOSTIC YIELD: direction disagreement with BOTH orientations on pairs 7/8; direction disagreement with the 65% majority on 9/10 and 11/12; confidence-level disagreement (ceiling vs near-chance) on 5/6; and a flat-vs-monotone contrast across the four single-cue cells 1-4 that no normalising model can produce and no unnormalised model can avoid. Every pair has at least one discriminating cue, so no cell is degenerate.

[6] GOAL: dissociate anchored compensatory integration with LOAD-GATED weight flattening + supra-additive corroboration + PARTIAL (psi~0.3) margin normalisation from read-order scanning (screen-position geometric weights, FULL per-trial normalisation, ~65% primacy / ~35% recency readers).

VALIDITIES BY SCREEN POSITION: [.68, .55, .95, .60, .86, .72]. The single best expert (95%) sits in the MIDDLE (pos2), a mid expert leads (pos0 = .68) and a mid expert closes (pos5 = .72). Under the competitor (rank = gamma*vrank + (1-gamma)*pos, gamma<=.25) the ranks are [0.75, 2.0, 1.5, 3.25, 3.25, 4.25], so pos0 is the primacy anchor and pos5 the recency anchor for EVERY (lam, gamma) in its box, while the advocated model's diagnosticity term keeps pos2 dominant at low load. Three distinct anchors, robustly.

THE CENTRAL ADVERSARIAL MOVE: several cells come in PAIRS with an IDENTICAL difference vector d = a - b, differing only in how many features carry a shared positive endorsement (L = both options rated 1). The competitor's evidence S = sum(w*d)/sum(w*|d|) depends only on d and screen position, so it predicts EXACTLY THE SAME choice probability within each such pair u2014 no parameter setting can produce any load effect. The advocated model predicts large, signed load effects (rho_eff = rho/(1+cL) flattens the validity gradient; delta_eff = delta*L/(1+L) turns on counting).

CELLS (8 base designs, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials; advocated favours A on 8 and B on 8 overall, so side bias cannot mimic anything).

(1) SINGLE-CUE LADDER, trials 1-2 (+mirrors 9-10). Exactly one expert discriminates and L = 0: pos2 (95%) in trial 1, pos3 (60%) in trial 2. The competitor is pinned at |S| = 1 in BOTH cells for every lam, gamma and orientation -> a FLAT ceiling (~.95-1.00). The advocated model reads a single cue against a partially attenuated yardstick, E = w_j^(1-psi), giving a steep ladder: P ~ .99 for the 95% cue but only ~.55-.60 for the 60% cue. Flat ceiling vs. a 40-point ladder is a purely quantitative signature the normalising model cannot produce.

(2) LOAD x CUE-STRENGTH CROSSOVER, trials 3-4 (+11-12). Trial 3 repeats trial 2's d (only pos3 discriminates) but adds three shared positive endorsements (L = 3); trial 4 repeats trial 1's d (only pos2 discriminates) with L = 4. Competitor: identical to trials 2 and 1 respectively (zero effect). Advocated: flattening RAISES the weak cue's relative weight (P .55-.60 -> .72-.80, about +18 pts) and LOWERS the strong cue's (P ~.99 -> ~.92-.93, about -6 pts). A crossover interaction of ~25 points produced by ratings that discriminate nothing at all is impossible for any model that only reads d.

(3) LOAD-INDUCED DIRECTION REVERSAL, trials 5-6 (+13-14) u2014 the sharpest cells. Both have d = [+1, 0, -1, 0, 0, +1]: the first-read 68% cue and the last-read 72% cue favour A, the mid-list 95% cue favours B. Trial 5 has L = 0, trial 6 pads three shared positives (L = 3). The competitor predicts A in BOTH and in both orientations (primacy S ~ +.43, recency S ~ +.93; P(A) ~ .85-.99) with no difference between them u2014 no orientation escape hatch, no load sensitivity. The advocated model predicts B at L = 0 (P(A) ~ .07-.10, checked at the extreme primacy corner alpha=2.1/lam=0.9/rho=2.3) and A at L = 3 (P(A) ~ .84-.92, checked across c in [.65,1.10] and rho in [2.3,2.7]) because flattening plus the 2-vs-1 corroboration bonus flips the sign of E. Same d, opposite direction: within-subject reversal across load is diagnostic no matter what the individual's parameters are.

(4) UNIMODAL vs BIMODAL, trial 7 (+15). d = [0,0,+1,0,0,-1]: the middle 95% expert against the last-read 72% expert, L = 0. Advocated: unimodal P(A) ~ .95-.98 for every subject. Competitor: primacy readers S ~ +.93 (P(A) ~ .98) but the ~35% recency readers S ~ -.93 (P(A) ~ .02) -> a strongly BIMODAL subject-level distribution with aggregate ~ .65. Both the aggregate level and the shape of the between-subject distribution differ.

(5) MAJORITY-DIRECTION DISAGREEMENT, trial 8 (+16). d = [+1,0,-1,0,0,0]: leading 68% cue vs. middle 95% cue, L = 0. Advocated: B, confidently (P(A) ~ .04-.09) because diagnosticity beats the primacy multiplier when rho is convex and load is zero. Competitor: the 65% primacy majority chooses A (P(A) ~ .85-.99), the recency minority chooses B; aggregate ~ .60-.65, bimodal. This cell also anchors the interpretation of trials 5/6: if subjects show the primacy-anchored A response here AND no load effect in 5-vs-6, the competitor wins outright.

Every pair has at least one discriminating cue, no cell is degenerate, and each of the three load-matched contrasts (2 vs 3, 1 vs 4, 5 vs 6) is a within-subject test on which the competing theory is mathematically forced to predict a null.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate READ-ORDER SCANNING (screen-position geometric weights, ~65% primacy / ~35% recency readers, FULL per-trial normalisation of evidence S = sum(w*d)/sum(w*|d|)) from ANCHORED COMPENSATORY INTEGRATION (convex diagnosticity (v-.5)^rho, primacy attention MULTIPLIER, load-gated flattening + supra-additive corroboration, and only PARTIAL margin normalisation psi~0.3).\n\nVALIDITIES BY SCREEN POSITION: [.72, .56, .80, .95, .62]. Deliberately non-monotone: the near-oracle expert (95%) sits at position 3, a mid expert leads (pos0 = .72) and a weak expert closes (pos4 = .62). Anchor check for the scanning model (rank = gamma*vrank + (1-gamma)*pos, vrank = [2,4,1,0,3]): at gamma = 0 ranks are [0,1,2,3,4]; at the maximum validity pull gamma = .25 ranks are [0.50, 1.75, 1.75, 2.25, 3.75]. So pos0 is the primacy anchor and pos4 the recency anchor for EVERY (lam, gamma) in the box, while the competitor's weights (rho=2.5, alpha=1.8, lam=1.1, L=0) are w = [.232, .005, .216, .529, .019], i.e. dominated by pos3. Three distinct anchors, robustly, and crucially the competitor's dominant cue is neither screen extreme.\n\nCELLS (12 base designs, each mirrored across sides -> 24 unique pairs, K = 4 -> 96 trials; mirroring makes the predicted-choice split exactly 12 A / 12 B for the competitor AND for both scanning orientations, so side bias cannot mimic anything).\n\n(1) SINGLE-CUE LADDER, cells 1-4 (pairs 1-8, L = 0). Exactly one expert discriminates: pos1 (.56, w=.005), pos4 (.62, w=.019), pos0 (.72, w=.232), pos3 (.95, w=.529). For scanning, one discriminating cue gives |S| = 1 EXACTLY for every lam, gamma and orientation, so all four cells sit at the same flat ceiling sigmoid(beta) ~ .93-1.00. For the competitor E = w^(1-psi) and accuracy is a steep MONOTONE ladder: P ~ .55 (pos1), ~.61 (pos4), ~.94 (pos0), ~.99 (pos3). Flat ceiling vs a 45-point ladder is a purely quantitative signature that no fully normalising model can produce and no partially normalising model can avoid. The pos3 rung also serves as an attention/calibration anchor: it shows the low pos1 value is not general inattention.\n\n(2) LOAD-MATCHED NULL #1, cells 5 & 6 (pairs 9-12). Both have d = [+1,0,0,-1,0] (leading 72% cue vs the 95% cue); cell 5 has L = 0 (a=[1,0,0,0,0] vs b=[0,0,0,1,0]), cell 6 pads three shared positive endorsements at pos1/pos2/pos4 (a=[1,1,1,0,1] vs b=[0,1,1,1,1]). Scanning depends only on d and screen position, so it is MATHEMATICALLY FORCED to predict identical behaviour in both: primacy readers A (S ~ +.95, P(A) ~ .98), recency readers B, aggregate ~ .65 in BOTH cells. The competitor predicts a full DIRECTION REVERSAL driven by ratings that discriminate nothing: rho_eff = rho/(1+cL) flattens (w -> [.380,.087,.203,.240,.091]) so P(A) goes from ~.04-.08 (L=0) to ~.61-.77 (L=3) across its whole (rho, c, alpha, lam, beta) box.\n\n(3) BOTH-ORIENTATIONS-AGREE DIRECTION CELL, cell 7 (pairs 13-14) - the sharpest single cell. d = [+1,0,0,-1,+1]: the first-read 72% cue and the last-read 62% cue favour A, the 95% cue favours B. Primacy readers choose A (S = .64 worst case at gamma=.25/lam=1.2, .95 at gamma=0), recency readers choose A even more strongly (S = .54-.99 across the box) - NO orientation escape hatch, scanning predicts P(A) ~ .90-1.00 for every subject. The competitor chooses B for its entire box (P(A) ~ .10 at nominal parameters, ~.22-.29 even at the extreme primacy corner alpha=2.1/lam=0.9/rho=2.3). Opposite direction over full parameter boxes.\n\n(4) LOAD-MATCHED NULL #2, cell 8 (pairs 15-16). Same d as cell 7 but with L = 2 shared positives (a=[1,1,1,0,1] vs b=[0,1,1,1,0]). Scanning: identical to cell 7 (P(A) ~ .95, both orientations). Competitor: flattening plus the 2-vs-1 corroboration bonus (delta_eff = delta*L/(1+L)) flips it to P(A) ~ .90 - a ~80-point within-subject reversal on a cell where the advocated theory predicts an exact null.\n\n(5) COMPENSATORY / GRADED-EVIDENCE CELL, cell 9 (pairs 17-18). a=[1,0,0,0,0] vs b=[0,1,1,1,0]: the leading cue against the three next-read cues, L = 0. Competitor: B at near-ceiling (P(A) ~ .02). Scanning: primacy readers still choose A but with markedly REDUCED relative evidence (S = .41 at lam=1.2 rising to .94 at lam=3.5), recency readers choose B at S ~ -.96. The predicted within-subject accuracy DROP between cell 3 (|S| = 1) and cell 9 (|S| intermediate) for primacy readers, alongside direction disagreement with the majority, is a scanning-specific gradient.\n\n(6) UNIMODAL vs BIMODAL, cell 10 (pairs 19-20). a=[0,0,0,1,0] vs b=[0,0,0,0,1]: the 95% expert (pos3) against the last-read 62% expert (pos4). Competitor: unimodal ceiling, P(A) ~ .99 for every subject. Scanning: primacy S = +.54 to +.94 (A), recency S = -.54 to -.99 (B) -> a strongly BIMODAL subject-level distribution with ~35% of subjects locked on B and an aggregate near .65. Both the aggregate level and the SHAPE of the between-subject distribution differ.\n\n(7) ANCHOR-vs-DIAGNOSTICITY BALANCE + its load partner, cells 11 & 12 (pairs 21-24). d = [+1,0,-1,0,0]: leading 72% cue vs the 80% cue at pos2. At L = 0 the competitor's primacy multiplier almost exactly cancels the diagnosticity gap (P(A) ~ .54, low between-subject spread); with L = 3 shared positives (cell 12: a=[1,1,0,1,1] vs b=[0,1,1,1,1]) flattening pushes it to P(A) ~ .83. Scanning predicts the same strongly bimodal .97/.03 split (aggregate ~.65) in both, with zero load effect.\n\nDIAGNOSTIC YIELD: (i) flat-vs-ladder across four single-cue cells tests full vs partial normalisation; (ii) three load-matched contrasts (5vs6, 7vs8, 11vs12) are within-subject tests on which the advocated theory is mathematically forced to predict an exact null while the competitor predicts reversals of up to 80 points; (iii) cell 7 gives direction disagreement with BOTH scanning orientations; (iv) cells 9-12 give bimodality where the competitor is unimodal, plus direction disagreement with the 65% primacy majority. Every pair has at least one discriminating cue, so no cell is degenerate.",
  "validities": [
    0.72,
    0.56,
    0.8,
    0.95,
    0.62
  ],
  "trial_a_ratings": [
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
      0
    ],
    [
      1,
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
      0
    ],
    [
      1,
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
      1,
      1,
      1,
      0,
      1
    ],
    [
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
      1,
      1,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
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
      1,
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
      0
    ],
    [
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1,
      1
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
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
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
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
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
      1,
      1,
      1,
      0
    ],
    [
      1,
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
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 19999,
  "output_tokens": 27203
}
```
