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
**Validities (n_features=6):** [0.62, 0.7, 0.95, 0.86, 0.66, 0.58]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  trial 2: A=[1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 1]
  trial 3: A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 7: A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 8: A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 9: A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 10: A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  trial 11: A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 12: A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 13: A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 14: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 15: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Rationale:** GOAL: dissociate READ-ORDER SCANNING (advocated: geometric weights in SCREEN position, ~65% primacy / ~35% recency readers, only a weak validity pull gamma<=.25, evidence FULLY normalised so S = sum(w*d)/sum(w*|d|) and any unanimous discriminating set gives |S| = 1 exactly) from READ-UNTIL-SUFFICIENT INTEGRATION (competitor: mildly compressed validity weights x a position attention multiplier, sufficiency stop, SILENCE CALIBRATION c = 1 - eta*(fraction silent)^q*mean diagnosticity of the speakers, and a CONVEX kappa read-out that crushes small residual imbalances).

VALIDITIES BY SCREEN POSITION: [.62, .70, .95, .86, .66, .58]. New layout relative to everything explored so far: the two heaviest experts sit at the CENTRE-RIGHT of the list (pos2 = .95, pos3 = .86), the two screen EXTREMES are the two WEAKEST experts (pos0 = .62, pos5 = .58), and two mid experts fill pos1 (.70) and pos4 (.66). Anchor check for scanning: vrank = [4,2,0,1,3,5], so rank = gamma*vrank + (1-gamma)*pos gives pos0 = 4*gamma (minimum for every gamma < 1/3, hence the primacy anchor for the whole gamma in [0,.25] box) and pos5 = 5 exactly (maximum, since the largest competitor rank is pos4 = 4-gamma), so pos5 is the recency anchor for every (lam, gamma). Crucially the extremes are the LEAST valid experts, so a scanning reader of either orientation anchors on a cue the competitor treats as nearly negligible: competitor weights at nominal parameters (rho = 1.15, alpha = .9, tau = 1.3, gamma_att = .05) are w = [.37, .50, 1.06, .75, .29, .13], i.e. dominated by the mid-list .95 expert, with the two extremes together (.50) worth less than half of it.

8 base cells, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials. Mirroring makes the predicted-choice split exactly 8 A / 8 B for the competitor AND for both scanning orientations, so side bias cannot mimic anything. Sparse cells are padded with SHARED positive endorsements on non-discriminating features (both theories depend only on d, so the padding changes nothing but makes the displays natural).

(1) INVERTED SINGLE-CUE LADDER + CORROBORATION DILUTION, cells 1-4 (pairs 1-8). In all four the discriminating cues are unanimous, so scanning's normalised evidence is |S| = 1 EXACTLY for every lam, gamma and orientation: it is mathematically pinned to one number, sigmoid(beta) ~ .95-1.00, in all four cells. The competitor's silence term makes them wildly different and INVERTED with respect to cue quality: cell 1 (only pos2, the .95 expert, discriminates; 5/6 silent) gives c ~ .02-.22 -> P ~ .52-.67, a near coin flip; cell 2 (only pos5, the .58 expert) gives c ~ .78-.86 -> P ~ .92-.94; cell 3 (only pos1, .70) is the intermediate rung, P ~ .81-.89; cell 4 adds the .70 and .86 experts AGREEING with the .95 expert (3/6 silent) and restores near-ceiling commitment, P ~ .90-.94. So the competitor predicts (i) the single WORST expert alone to be ~35 points MORE decisive than the single BEST expert alone, and (ii) a ~40-point gain from adding two strictly weaker corroborating experts to cell 1's cue. Scanning predicts an exact null across all four cells. This flat-vs-inverted-ladder contrast is the flagship quantitative signature and no (beta, epsilon, lam, gamma, orientation) of the scanning model can produce any spread here.

(2) BOTH-ORIENTATIONS-AGREE DIRECTION CELL, cell 5 (pairs 9-10), sharpest cell. d = [+1,0,-1,-1,0,+1]: the first-read .62 cue and the last-read .58 cue favour A, the two mid-list heavyweights (.95, .86) favour B. Scanning chooses A for BOTH orientations across its whole box (primacy S = +.17 at the worst corner gamma=.25/lam=1.2, +.70 to +.79 elsewhere; recency S = +.88 to +.93), so P(A) ~ .67-1.00 with no orientation escape hatch. The competitor cannot stop on pos0 (.37 vs a 1.11 bar), reads the two heavyweights, stops at pos3 with R ~ -.66 and, because only 2/6 experts are silent, the silence discount is negligible (c ~ .93): P(A) ~ .18-.25. Opposite direction over both full parameter boxes, in a dense display where no silence machinery is doing the work.

(3) SPARSE EXTREMES DUEL: UNIMODAL vs BIMODAL, cell 6 (pairs 11-12). d = [+1,0,0,0,0,-1]: the leading .62 expert against the last-read .58 expert, 4/6 silent. The competitor clears the sufficiency bar immediately at pos0 (ratio 2.7) and commits on R = +1 with only a mild silence discount (the speakers are both low-validity, so vbar ~ .22 and c ~ .87): P(A) ~ .93-.95, UNIMODAL for every subject. Scanning splits hard: primacy readers S ~ +1 (P(A) ~ .99), recency readers S ~ -1 (P(A) ~ .01), giving a strongly BIMODAL subject-level distribution with aggregate ~ .65. Same ballpark aggregate is impossible to confuse with the competitor's tight unimodal .94, and the between-subject distribution shape is the decisive statistic.

(4) MAJORITY-DIRECTION DISAGREEMENT, cell 7 (pairs 13-14). d = [-1,0,+1,0,0,0]: the leading .62 cue favours B, the mid-list .95 cue favours A, 4/6 silent. The competitor never stops on pos0, ends with R ~ +.48 which the convex read-out and the silence factor (c ~ .56) discount to P(A) ~ .60-.68, unimodal. Scanning: the 65% primacy majority chooses B (P(A) ~ .02-.10), the 35% recency minority chooses A (P(A) ~ .90-.98); aggregate ~ .35, i.e. the OPPOSITE side from the competitor, with maximal between-subject variance.

(5) DENSE 3-vs-3 SPLIT, cell 8 (pairs 15-16). d = [+1,+1,+1,-1,-1,-1]: no expert is silent, so the competitor's silence machinery is exactly off; it accumulates .37+.50+1.06 = 1.93 by pos2, clears the bar set by the strongest unread expert (1.05*.75 = .79) and commits on R = +1: P(A) ~ .95-.97, unimodal. Scanning: primacy readers also choose A at ceiling, but recency readers are dominated by pos5 and choose B at ceiling, so the subject-level distribution must again be BIMODAL with ~35% of subjects locked on B and an aggregate near .65 -- a 30-point aggregate gap plus a categorical shape difference, in the one cell where neither silence calibration nor normalisation-of-sparse-evidence can be blamed.

DIAGNOSTIC YIELD: (i) four cells where scanning is mathematically pinned to a single probability while the competitor predicts a 40-point INVERTED validity ladder plus a large corroboration gain; (ii) one cell (5) with direction disagreement against BOTH scanning orientations over the full boxes; (iii) three cells (6, 7, 8) where scanning must be bimodal at the subject level and the competitor unimodal, one of them (7) with an aggregate direction reversal. Every pair has at least one discriminating cue; no cell is degenerate.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



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
**Description:** **Read-until-sufficient integration with mildly compressed diagnosticity, silence-calibrated evidence, and an S-SHAPED (convex) read-out of relative evidence.**

People do not run a fixed heuristic (TTB, tally, or weighted-additive). They read the expert ratings in the order the screen presents them, accumulate weighted differences, and stop as soon as their running lead is safe from any single expert they have not yet read. Six claims:

1. **Cue weights = MILDLY COMPRESSED stated diagnosticity x a modest reading-position bonus.** w_j = ((v_j-0.5)/0.5)^rho * (1 + alpha*exp(-j/tau) + gamma*exp(-(n-1-j)/tau)), with rho ~1.0-1.3, i.e. subjective weight is essentially linear-to-slightly-convex in (v-0.5). A .95 expert is only ~2x a .72 expert and ~5-8x a .56 expert -- clearly heavier, and roughly *balanced* by a pair of upper-middling experts (.78 + .72), but routinely outvotable by three mid-validity experts ('tally-like without unit-weight tallying'). This exponent sits at the crossover point where the two screen-extreme flankers of a mid-list .95 expert neither dominate it nor are dominated by it, which is what turns double-dissociation displays into genuine coin flips. Position is an attention multiplier, never the causal driver.

2. **Serial reading with a graded sufficiency stop.** After each rating that actually discriminates, the reader asks whether any single unread expert could still overturn the current lead; the bar is theta = phi * max weight among unread positions and stopping is graded, q = sigmoid(s*(|E|/theta - 1)). This is the only place where screen order has structural force, and it is display-contingent. Because the bar is set by the strongest unread expert, a merely salient first cue stops the read only when its weighted lead genuinely exceeds what any single remaining expert could reverse -- and as diagnosticity compression relaxes (higher rho) a salient-but-mediocre first cue loses that race to an unread .95 expert.

3. **Evidence is read FULLY relatively, with no load machinery.** R = sum(w*d)/sum(w*|d|) over the cues read so far; features on which both products are endorsed (or both unendorsed) are simply dropped, so any dense-vs-sparse contrast is exactly zero.

4. **Silence calibration.** Validity is treated as sensitivity: a difference only a top expert notices while the rest of the panel is silent implies a marginal quality gap and is acted on hesitantly. Gain is multiplied by c = 1 - eta*(fraction silent)^q * (weighted mean diagnosticity of the speaking experts), which bites only when silence is overwhelming, producing a *reversed* single-cue validity ladder. This term is a function of raw diagnosticity, not of the compressed weights, so the ladder is invariant to rho.

5. **Decisiveness is a CONVEX (S-shaped) function of the evidence balance, not a linear one.** A near-balanced set of reasons does not merely produce a slightly tilted preference; it produces genuine indifference. Read-out is logistic(beta * c * sign(R)*|R|^kappa) with kappa ~1.4-1.9. A small residual imbalance (|R| ~ 0.1-0.2, e.g. two screen-extreme cues almost exactly offsetting one top-validity cue) is discounted to near-nothing and the choice is a coin flip; a clear weighted majority (|R| ~ 0.5) is still followed at ~0.78-0.82; and a categorical conclusion (|R| = 1, i.e. a sufficiency stop, or a single discriminating cue) is unchanged, so committed reads stay decisive.

6. **Choice** = logistic(beta * c * sign(R)|R|^kappa) with lapse epsilon. Individual differences are unimodal jitter in rho, alpha, phi, s, kappa, beta, eta and lapse -- no strategy subpopulations, no primacy/recency reader types, no learning (there is no feedback).

**Parameters:**
- rho: [1.00, 1.32]
- alpha: [0.7, 1.2]
- gamma: [0.0, 0.15]
- tau: [1.0, 1.6]
- phi: [0.95, 1.20]
- s: [7.0, 14.0]
- beta: [2.80, 3.50]
- kappa: [1.35, 1.85]
- eta: [1.5, 2.1]
- sil_pow: [2.4, 3.0]
- epsilon: [0.0, 0.06]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
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
        else:
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
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-9, 1.0)      # scaled diagnosticity in (0,1]

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    rho = float(np.clip(parameters.get('rho', 1.15), 0.1, 6.0))
    alpha = float(np.clip(parameters.get('alpha', 0.9), 0.0, 4.0))
    gamma = float(np.clip(parameters.get('gamma', 0.05), 0.0, 3.0))
    tau = float(np.clip(parameters.get('tau', 1.3), 0.2, 8.0))
    phi = float(np.clip(parameters.get('phi', 1.05), 0.3, 3.0))
    s = float(np.clip(parameters.get('s', 10.0), 0.5, 80.0))
    beta = float(np.clip(parameters.get('beta', 3.15), 0.05, 40.0))
    kappa = float(np.clip(parameters.get('kappa', 1.6), 0.5, 5.0))
    eta = float(np.clip(parameters.get('eta', 1.8), 0.0, 5.0))
    q_sil = float(np.clip(parameters.get('sil_pow', 2.7), 1.0, 8.0))
    eps = float(np.clip(parameters.get('epsilon', 0.03), 0.0, 0.5))

    # ------------------------------------------------------------------
    # 4. cue weights: compressed diagnosticity x modest reading-position
    #    attention.  rho ~1.0-1.3 places a .95 cue at roughly the combined
    #    strength of two upper-middling cues, so screen-extreme flankers
    #    neither dominate nor are dominated by a mid-list top expert.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    att = 1.0 + alpha * np.exp(-pos / tau) + gamma * np.exp(-(n - 1.0 - pos) / tau)
    w = np.power(vs, rho) * att
    w = np.clip(w, 1e-12, None)

    # ------------------------------------------------------------------
    # 5. silence calibration: validity = sensitivity.  A difference that
    #    only the most sensitive experts can see, while the rest of the
    #    panel sees none, implies a marginal quality gap -> discounted.
    #    A difference an insensitive expert notices implies a large gap.
    # ------------------------------------------------------------------
    m = int(np.count_nonzero(disc))
    f_sil = float(n - m) / float(n)
    wd = w[disc]
    tot_wd = float(np.sum(wd))
    if tot_wd <= 0:
        return np.array([0.5, 0.5])
    vbar = float(np.sum(wd * vs[disc]) / tot_wd)
    c = 1.0 - eta * (f_sil ** q_sil) * vbar
    c = float(np.clip(c, 0.02, 1.0))

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    # convex (S-shaped) read-out of the relative balance: |R| = 1 is
    # untouched, small residual imbalances are discounted toward chance.
    def _drive(R):
        R = float(np.clip(R, -1.0, 1.0))
        return float(np.sign(R) * (abs(R) ** kappa))

    # ------------------------------------------------------------------
    # 6. serial reading in screen order with graded sufficiency stopping;
    #    evidence is read fully relatively (R in [-1, 1]).
    # ------------------------------------------------------------------
    idxs = np.flatnonzero(disc)
    E = 0.0
    M = 0.0
    remain = 1.0
    p_a = 0.0
    last = int(idxs.shape[0]) - 1
    for t in range(idxs.shape[0]):
        j = int(idxs[t])
        E += w[j] * (1.0 if d[j] > 0 else -1.0)
        M += w[j]
        R = (E / M) if M > 1e-12 else 0.0
        p_here = _sig(beta * c * _drive(R))
        if t >= last:
            q = 1.0
        else:
            tail = w[j + 1:]
            bar = phi * float(np.max(tail)) if tail.size > 0 else 0.0
            if bar <= 1e-12:
                q = 1.0
            else:
                q = _sig(s * (abs(E) / bar - 1.0))
        p_a += remain * q * p_here
        remain *= (1.0 - q)
        if remain <= 1e-12:
            break
    if remain > 1e-12:
        R = (E / M) if M > 1e-12 else 0.0
        p_a += remain * _sig(beta * c * _drive(R))

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Cell-5 pairs (trials 9 and 10 of the design), mirrored across sides:\n    #   d = a - b = [+1, 0, -1, -1, 0, +1]   -> extremes (pos0=.62, pos5=.58) favour A\n    #   d = a - b = [-1, 0, +1, +1, 0, -1]   -> extremes favour B\n    # Score = fraction of these trials on which the subject chose the option\n    # endorsed by the two SCREEN-EXTREME (weakest) experts, i.e. against the\n    # two mid-list heavyweight experts (.95 and .86).\n    target_A = (1, 0, -1, -1, 0, 1)\n    target_B = (-1, 0, 1, 1, 0, -1)\n\n    hits = 0\n    n = 0\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'],\n                                  data['option_b_ratings'],\n                                  data['response']):\n        try:\n            a = [int(v) for v in a_raw]\n            b = [int(v) for v in b_raw]\n        except Exception:\n            continue\n        if len(a) != len(b) or len(a) != 6:\n            continue\n        d = tuple(int(x - y) for x, y in zip(a, b))\n        if d == target_A:\n            extreme_side = 0  # option A\n        elif d == target_B:\n            extreme_side = 1  # option B\n        else:\n            continue\n        n += 1\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        if r == extreme_side:\n            hits += 1\n\n    if n == 0:\n        return 0.5\n    return float(hits) / float(n)",
  "rationale": "The design's cell 5 (pairs 9/10, d = [+1,0,-1,-1,0,+1] and its mirror) is the one place where the two theories predict OPPOSITE choices over their entire parameter boxes, and where BOTH read-order orientations (primacy and recency) agree, so the advocated theory stays unimodal and tight rather than splitting into subpopulations. Under read-order scanning, the first-read (.62) and last-read (.58) cues dominate whichever orientation the subject has, giving normalised evidence S strongly positive for the extreme-favoured option (P ~ .67-1.00, typically ~.85+). Under read-until-sufficient integration, the leading .62 cue cannot clear the sufficiency bar set by the unread .95 expert, so the reader continues, the two mid-list heavyweights dominate, and with only 2/6 experts silent the silence discount barely bites: P(extreme-favoured option) ~ .18-.25. The metric is the proportion of these 12 trials per subject on which the extreme-endorsed option was chosen: expected ~0.85 for scanning vs ~0.22 for the competitor, with per-subject variance dominated by binomial noise over 12 trials (~0.01-0.02), so Welch's test at N=25 should be decisively significant. It is not trivially true for my theory: it is a directional prediction about which cues drive choice, which the competitor makes an equally strong opposite prediction about."
}
```

## Usage

```json
{
  "input_tokens": 12814,
  "output_tokens": 2427
}
```
