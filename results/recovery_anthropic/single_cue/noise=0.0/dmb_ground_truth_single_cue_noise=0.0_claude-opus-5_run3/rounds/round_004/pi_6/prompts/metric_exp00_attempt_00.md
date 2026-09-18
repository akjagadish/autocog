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
**Validities (n_features=6):** [0.72, 0.6, 0.95, 0.86, 0.56, 0.7]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  trial 3: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 7: A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 9: A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 10: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 11: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 12: A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 13: A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 14: A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  trial 15: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 16: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]

**Rationale:** GOAL: dissociate the advocated model (mildly compressed diagnosticity weights x reading-position attention, serial read with a sufficiency stop, FULLY relative evidence R in [-1,1], SILENCE CALIBRATION c = 1 - eta*(fraction silent)^q * mean diagnosticity of the speaking experts, and a CONVEX kappa read-out that turns small residual imbalances into coin flips) from read-order scanning (geometric weights in SCREEN position, ~65% primacy / ~35% recency readers, evidence FULLY normalised so any unanimous cue set gives |S| = 1 exactly).

VALIDITIES BY SCREEN POSITION: [.72, .60, .95, .86, .56, .70]. The near-oracle expert (95%) sits mid-list at pos2, a mid expert leads (pos0 = .72) and a mid expert closes (pos5 = .70), with a weak expert (.56) at pos4 and a near-useless one (.60) at pos1. Scanning anchors are robust: vrank = [2,4,0,1,5,3], so with rank = gamma*vrank + (1-gamma)*pos the ranks at the maximum validity pull gamma = .25 are [0.50, 1.75, 1.50, 2.50, 4.25, 4.50] and at gamma = 0 they are [0,1,2,3,4,5]; pos0 is the primacy anchor and pos5 the recency anchor for EVERY (lam, gamma) in the competitor's box. The advocated model's weights at nominal parameters (rho = 1.15, alpha = 0.9, tau = 1.3) are w = [.74, .22, 1.06, .75, .09, .37], i.e. dominated by the mid-list 95% cue but with the two screen extremes summing to almost exactly that value (1.11 vs 1.06) - the crossover the theory advertises.

8 base cells, each mirrored across sides (16 unique pairs, K = 6 -> 96 trials). Mirroring makes the predicted-choice split exactly 8 A / 8 B for the advocated model AND for both scanning orientations, so side bias cannot mimic anything.

(1) REVERSED SINGLE-CUE VALIDITY LADDER + SILENCE DILUTION, cells 1-4 (pairs 1-8). Cell 1: only pos2 (.95) discriminates. Cell 2: only pos4 (.56). Cell 3: only pos0 (.72). Cell 4: pos0, pos2 and pos5 ALL favour the same option. In every one of these four cells the discriminating cues are unanimous, so the competitor's normalised evidence is |S| = 1 EXACTLY for every lam, gamma, orientation - it is mathematically forced to predict the SAME probability, sigmoid(beta) ~ .95-1.00, in all four cells. The advocated model's silence term makes them wildly different and, crucially, INVERTED with respect to cue quality: with 5 of 6 experts silent, c = 1 - eta*(5/6)^q*vbar collapses for a lone .95 cue (c ~ .02-.22 -> P ~ .52-.67) but barely touches a lone .56 cue (c ~ .84-.90 -> P ~ .93-.95), with the lone .72 cue in between (P ~ .75-.90); adding two agreeing cues to the .95 cue (cell 4, silence only 3/6) restores near-ceiling commitment (P ~ .90-.94). A monotone reversed ladder (worst expert alone MORE decisive than the best expert alone) plus a large dilution gain from adding redundant cues, across cells where the competitor is pinned to a single number, is the flagship quantitative signature.

(2) BOTH-ORIENTATIONS-AGREE INDIFFERENCE CELL, cell 5 (pairs 9-10). d = [+1,0,-1,0,0,+1]: the first-read 72% cue and the last-read 70% cue favour A, the mid-list 95% cue favours B. Primacy readers choose A (S = .54-.96 across the box), recency readers choose A even harder (S = .78-1.00): scanning predicts P(A) ~ .90-1.00 with NO orientation escape hatch. The advocated model lands on a residual |R| ~ 0.02-0.09 which the convex kappa read-out crushes to nothing: P(A) ~ .47-.57 across its whole (rho, alpha, tau, kappa) box, i.e. a genuine coin flip where the competitor is at ceiling. Sharpest single cell.

(3) UNIMODAL-NEAR-CHANCE vs BIMODAL-EXTREME, cell 6 (pairs 11-12). d = [+1,0,-1,0,0,0]: leading 72% cue against the mid-list 95% cue, four experts silent. Advocated: unresolved read, |R| ~ .18 discounted by kappa and by the silence factor (c ~ .57) -> P(A) ~ .46-.52 for every subject. Scanning: primacy readers P(A) ~ .90-1.00, recency readers P(A) ~ .00-.10 -> strongly BIMODAL subject-level distribution with aggregate ~ .64. Aggregate level AND between-subject shape differ.

(4) SUFFICIENCY-STOP CELL, cell 7 (pairs 13-14). d = [0,0,+1,0,0,-1]: the 95% expert (pos2) against the last-read 70% expert. The advocated model clears the sufficiency bar at pos2 (|E|/phi*max unread = 1.34) and commits on a categorical R = +1, but with the 4/6-silence discount: P(A) ~ .78-.88, unimodal. Scanning splits again: primacy P(A) ~ .99, recency P(A) ~ .01, aggregate ~ .65 with maximal between-subject variance. Same aggregate ballpark, opposite distributional shape - and, paired with cell 6, it shows the advocated model moving where scanning cannot (cells 6 and 7 differ by 30+ points for the advocated model but are both ~.65 aggregate for scanning).

(5) DENSE CROWD-BEATS-EXTREMES DIRECTION CELL, cell 8 (pairs 15-16). d = [+1,-1,-1,-1,0,+1]: both screen extremes favour A while the three mid-list cues (.60, .95, .86) favour B; only one expert is silent so the silence discount is ~zero. The advocated model accumulates past the extremes, triggers the stop at pos3 with R ~ -.47 and predicts P(A) ~ .28-.35. Scanning predicts A for BOTH orientations (primacy S = .41-.73, recency S = .78-.96 -> P(A) ~ .84-1.00). Direction disagreement with both orientations again, this time in a dense display where no silence machinery is doing the work - so it cannot be explained away as a silence artefact.

DIAGNOSTIC YIELD: (i) four cells where the competitor is mathematically pinned to one identical probability while the advocated model predicts a 35-point INVERTED validity ladder and a large redundancy/dilution gain; (ii) two cells (5 and 8) where scanning predicts near-ceiling A for both primacy and recency readers while the advocated model predicts a coin flip and a reversal respectively; (iii) two cells (6, 7) where scanning must be bimodal at the subject level and the advocated model unimodal. Every pair has at least one discriminating cue; no cell is degenerate.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Cells where BOTH scanning orientations (primacy and recency) predict the\n    # option favoured by the two screen-extreme cues (pos0 = .72, pos5 = .70),\n    # while the advocated model predicts a coin flip (cell 5) or a reversal\n    # toward the mid-list crowd (cell 8).\n    #   cell 5: d = (+1, 0, -1, 0, 0, +1)  and its mirror\n    #   cell 8: d = (+1, -1, -1, -1, 0, +1) and its mirror\n    targets = {\n        (1, 0, -1, 0, 0, 1),\n        (-1, 0, 1, 0, 0, -1),\n        (1, -1, -1, -1, 0, 1),\n        (-1, 1, 1, 1, 0, -1),\n    }\n\n    hits = []\n    for _, row in data.iterrows():\n        try:\n            a = [int(v) for v in row['option_a_ratings']]\n            b = [int(v) for v in row['option_b_ratings']]\n        except Exception:\n            continue\n        if len(a) != 6 or len(b) != 6:\n            continue\n        d = tuple(int(x) - int(y) for x, y in zip(a, b))\n        if d not in targets:\n            continue\n        resp = row['response']\n        try:\n            resp = int(resp)\n        except Exception:\n            continue\n        # option favoured by the screen-extreme cues (positions 0 and 5 agree)\n        fav_a = (d[0] > 0)\n        chose_a = (resp == 0)\n        hits.append(1.0 if (chose_a == fav_a) else 0.0)\n\n    if len(hits) == 0:\n        return float('nan')\n    return float(np.mean(hits))",
  "rationale": "The metric is the proportion of choices consistent with the option favoured by the two screen-extreme experts (position 0, v=.72 and position 5, v=.70) restricted to the two design cells where the competing read-order scanning model makes the SAME prediction for primacy and recency readers (so it is unimodal across subjects, no bimodality inflating between-subject variance): cell 5 (d = +1,0,-1,0,0,+1) and cell 8 (d = +1,-1,-1,-1,0,+1), plus their side mirrors. Under scanning, normalised evidence S is ~.73-.97 for primacy readers and ~.96-.98 for recency readers in both cells, so P(extreme-favoured) is near ceiling (~.90-.98 after lapses) for every subject. Under the advocated read-until-sufficient model with mildly compressed diagnosticity and a convex read-out, the mid-list .95 expert almost exactly offsets the two extremes in cell 5, and the convex kappa read-out crushes the residual |R|~.02-.09 to a coin flip (P~.50-.55); in cell 8 the dense mid-list crowd (.60/.95/.86) outweighs the two extremes and triggers a sufficiency stop at R~-.47, giving P(extreme-favoured)~.28-.35. Pooling 24 trials per subject (12 per cell) yields ~.42 for the advocated model vs ~.94 for scanning, with tight per-subject binomial + parameter noise (SD~.10 and ~.05 respectively), which is easily separable at N=25.  It is not trivially true of my theory: it depends jointly on the compression exponent, the sufficiency stop and the convex read-out, and any weighted-additive or lexicographic account with validity-driven weights could land anywhere on it."
}
```

## Usage

```json
{
  "input_tokens": 12519,
  "output_tokens": 4768
}
```
