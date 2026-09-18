# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
**Read-until-sufficient integration with mildly compressed diagnosticity, silence-calibrated evidence, and an S-SHAPED (convex) read-out of relative evidence.**

People do not run a fixed heuristic (TTB, tally, or weighted-additive). They read the expert ratings in the order the screen presents them, accumulate weighted differences, and stop as soon as their running lead is safe from any single expert they have not yet read. Six claims:

1. **Cue weights = MILDLY COMPRESSED stated diagnosticity x a modest reading-position bonus.** w_j = ((v_j-0.5)/0.5)^rho * (1 + alpha*exp(-j/tau) + gamma*exp(-(n-1-j)/tau)), with rho ~1.0-1.3, i.e. subjective weight is essentially linear-to-slightly-convex in (v-0.5). A .95 expert is only ~2x a .72 expert and ~5-8x a .56 expert -- clearly heavier, and roughly *balanced* by a pair of upper-middling experts (.78 + .72), but routinely outvotable by three mid-validity experts ('tally-like without unit-weight tallying'). This exponent sits at the crossover point where the two screen-extreme flankers of a mid-list .95 expert neither dominate it nor are dominated by it, which is what turns double-dissociation displays into genuine coin flips. Position is an attention multiplier, never the causal driver.

2. **Serial reading with a graded sufficiency stop.** After each rating that actually discriminates, the reader asks whether any single unread expert could still overturn the current lead; the bar is theta = phi * max weight among unread positions and stopping is graded, q = sigmoid(s*(|E|/theta - 1)). This is the only place where screen order has structural force, and it is display-contingent. Because the bar is set by the strongest unread expert, a merely salient first cue stops the read only when its weighted lead genuinely exceeds what any single remaining expert could reverse -- and as diagnosticity compression relaxes (higher rho) a salient-but-mediocre first cue loses that race to an unread .95 expert.

3. **Evidence is read FULLY relatively, with no load machinery.** R = sum(w*d)/sum(w*|d|) over the cues read so far; features on which both products are endorsed (or both unendorsed) are simply dropped, so any dense-vs-sparse contrast is exactly zero.

4. **Silence calibration.** Validity is treated as sensitivity: a difference only a top expert notices while the rest of the panel is silent implies a marginal quality gap and is acted on hesitantly. Gain is multiplied by c = 1 - eta*(fraction silent)^q * (weighted mean diagnosticity of the speaking experts), which bites only when silence is overwhelming, producing a *reversed* single-cue validity ladder. This term is a function of raw diagnosticity, not of the compressed weights, so the ladder is invariant to rho.

5. **Decisiveness is a CONVEX (S-shaped) function of the evidence balance, not a linear one.** A near-balanced set of reasons does not merely produce a slightly tilted preference; it produces genuine indifference. Read-out is logistic(beta * c * sign(R)*|R|^kappa) with kappa ~1.4-1.9. A small residual imbalance (|R| ~ 0.1-0.2, e.g. two screen-extreme cues almost exactly offsetting one top-validity cue) is discounted to near-nothing and the choice is a coin flip; a clear weighted majority (|R| ~ 0.5) is still followed at ~0.78-0.82; and a categorical conclusion (|R| = 1, i.e. a sufficiency stop, or a single discriminating cue) is unchanged, so committed reads stay decisive.

6. **Choice** = logistic(beta * c * sign(R)|R|^kappa) with lapse epsilon. Individual differences are unimodal jitter in rho, alpha, phi, s, kappa, beta, eta and lapse -- no strategy subpopulations, no primacy/recency reader types, no learning (there is no feedback).

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## THEORY 2 — pi_3
**Serial-position–driven, primacy/recency-graded evidence accumulation ("read-order scanning").**

People do not re-sort the expert ratings by the stated validities before deciding. They inspect the ratings in the order in which they appear on the screen (reading order) and their attention decays sharply across that inspection sequence, so the evidence carried by a cue is discounted geometrically by its *serial position*, not by its validity. Stated validities exert only a weak, partial pull on the inspection order (parameter gamma, small).

Two further claims:

1. **The gradient is steep but graded.** Weights fall off as w_j = exp(-lambda * rank_j) with |lambda| well above zero, so decisions are usually driven by the first cue(s) that discriminate — near-lexicographic but not strictly all-or-none: when the leading cues tie, the next-inspected cues take over, and when several later cues line up they can occasionally overturn a leading cue if the gradient is shallow.

2. **The gradient has a sign: most readers are primacy-oriented, a substantial minority are recency-oriented.** Reading a short list produces either an anchoring-on-the-first-item strategy (primacy, lambda > 0) or a last-impression-counts strategy (recency, lambda < 0). The population is bimodal: about one third of subjects anchor on the *last-read* cue. Near-zero gradients (i.e., genuine equal-weight tallying) are essentially absent — people do not integrate all six/five cues equally.

Evidence is evaluated *relatively*: the signed weighted difference is normalised by the total weight of the cues that actually discriminate, so a decision based on one early cue is as confident as a decision based on many, and only genuinely near-balanced weighted evidence produces hesitation. Choice is a softmax on this relative evidence with inverse temperature beta, plus a lapse epsilon.

Because validity order and screen order coincide in a validity-sorted display but dissociate in a scrambled display, this theory predicts TTB-looking behaviour in the former and tally-looking behaviour in the latter — without either heuristic being the actual mechanism. The causal driver is *where the cue sits on the screen*.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
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



### METRIC
Rationale:
The metric is the proportion of choices consistent with the option favoured by the two screen-extreme experts (position 0, v=.72 and position 5, v=.70) restricted to the two design cells where the competing read-order scanning model makes the SAME prediction for primacy and recency readers (so it is unimodal across subjects, no bimodality inflating between-subject variance): cell 5 (d = +1,0,-1,0,0,+1) and cell 8 (d = +1,-1,-1,-1,0,+1), plus their side mirrors. Under scanning, normalised evidence S is ~.73-.97 for primacy readers and ~.96-.98 for recency readers in both cells, so P(extreme-favoured) is near ceiling (~.90-.98 after lapses) for every subject. Under the advocated read-until-sufficient model with mildly compressed diagnosticity and a convex read-out, the mid-list .95 expert almost exactly offsets the two extremes in cell 5, and the convex kappa read-out crushes the residual |R|~.02-.09 to a coin flip (P~.50-.55); in cell 8 the dense mid-list crowd (.60/.95/.86) outweighs the two extremes and triggers a sufficiency stop at R~-.47, giving P(extreme-favoured)~.28-.35. Pooling 24 trials per subject (12 per cell) yields ~.42 for the advocated model vs ~.94 for scanning, with tight per-subject binomial + parameter noise (SD~.10 and ~.05 respectively), which is easily separable at N=25.  It is not trivially true of my theory: it depends jointly on the compression exponent, the sufficiency stop and the convex read-out, and any weighted-additive or lexicographic account with validity-driven weights could land anywhere on it.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cells where BOTH scanning orientations (primacy and recency) predict the
    # option favoured by the two screen-extreme cues (pos0 = .72, pos5 = .70),
    # while the advocated model predicts a coin flip (cell 5) or a reversal
    # toward the mid-list crowd (cell 8).
    #   cell 5: d = (+1, 0, -1, 0, 0, +1)  and its mirror
    #   cell 8: d = (+1, -1, -1, -1, 0, +1) and its mirror
    targets = {
        (1, 0, -1, 0, 0, 1),
        (-1, 0, 1, 0, 0, -1),
        (1, -1, -1, -1, 0, 1),
        (-1, 1, 1, 1, 0, -1),
    }

    hits = []
    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row['option_a_ratings']]
            b = [int(v) for v in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != 6 or len(b) != 6:
            continue
        d = tuple(int(x) - int(y) for x, y in zip(a, b))
        if d not in targets:
            continue
        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        # option favoured by the screen-extreme cues (positions 0 and 5 agree)
        fav_a = (d[0] > 0)
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == fav_a) else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_6 (simulated): 0.4325 (var=0.0102)
- Predicted under pi_3 (simulated): 0.9692 (var=0.0013)
- Observed on real data: 0.4900 (var=0.0105)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
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



### METRIC
Rationale:
The design's cell 5 (pairs 9/10, d = [+1,0,-1,-1,0,+1] and its mirror) is the one place where the two theories predict OPPOSITE choices over their entire parameter boxes, and where BOTH read-order orientations (primacy and recency) agree, so the advocated theory stays unimodal and tight rather than splitting into subpopulations. Under read-order scanning, the first-read (.62) and last-read (.58) cues dominate whichever orientation the subject has, giving normalised evidence S strongly positive for the extreme-favoured option (P ~ .67-1.00, typically ~.85+). Under read-until-sufficient integration, the leading .62 cue cannot clear the sufficiency bar set by the unread .95 expert, so the reader continues, the two mid-list heavyweights dominate, and with only 2/6 experts silent the silence discount barely bites: P(extreme-favoured option) ~ .18-.25. The metric is the proportion of these 12 trials per subject on which the extreme-endorsed option was chosen: expected ~0.85 for scanning vs ~0.22 for the competitor, with per-subject variance dominated by binomial noise over 12 trials (~0.01-0.02), so Welch's test at N=25 should be decisively significant. It is not trivially true for my theory: it is a directional prediction about which cues drive choice, which the competitor makes an equally strong opposite prediction about.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cell-5 pairs (trials 9 and 10 of the design), mirrored across sides:
    #   d = a - b = [+1, 0, -1, -1, 0, +1]   -> extremes (pos0=.62, pos5=.58) favour A
    #   d = a - b = [-1, 0, +1, +1, 0, -1]   -> extremes favour B
    # Score = fraction of these trials on which the subject chose the option
    # endorsed by the two SCREEN-EXTREME (weakest) experts, i.e. against the
    # two mid-list heavyweight experts (.95 and .86).
    target_A = (1, 0, -1, -1, 0, 1)
    target_B = (-1, 0, 1, 1, 0, -1)

    hits = 0
    n = 0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = [int(v) for v in a_raw]
            b = [int(v) for v in b_raw]
        except Exception:
            continue
        if len(a) != len(b) or len(a) != 6:
            continue
        d = tuple(int(x - y) for x, y in zip(a, b))
        if d == target_A:
            extreme_side = 0  # option A
        elif d == target_B:
            extreme_side = 1  # option B
        else:
            continue
        n += 1
        try:
            r = int(resp)
        except Exception:
            continue
        if r == extreme_side:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)

### RESULTS
- Predicted under pi_6 (simulated): 0.2300 (var=0.0174)
- Predicted under pi_3 (simulated): 0.9717 (var=0.0024)
- Observed on real data: 0.8500 (var=0.0111)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    dis_hits = 0.0
    dis_n = 0.0
    agr_hits = 0.0
    agr_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.size == 0:
            continue

        # Take-The-Best winner: features are listed in descending validity
        # order, so scan left to right for the first discriminating cue.
        ttb_winner = None
        for j in range(a.size):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally-tie pairs excluded (uninformative for the contrast)
        tally_winner = 0 if a_wins > b_wins else 1

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hit = 1.0 if r == ttb_winner else 0.0

        if tally_winner != ttb_winner:
            dis_hits += hit
            dis_n += 1.0
        else:
            agr_hits += hit
            agr_n += 1.0

    if dis_n == 0.0 or agr_n == 0.0:
        return float('nan')

    return float(dis_hits / dis_n - agr_hits / agr_n)
```

**Observed (real) value:** 0.0933 (var=0.0064)
**Predicted under pi_6:** 0.0110 (var=0.0029)
**Predicted under pi_3:** 0.1027 (var=0.0215)

### Experiment 4
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.9, 0.6, 0.8, 0.55, 0.7, 0.95], dtype=float)

    n_match = 0.0
    n_total = 0.0

    for _, row in data.iterrows():
        a = np.asarray(list(row['option_a_ratings']), dtype=float)
        b = np.asarray(list(row['option_b_ratings']), dtype=float)
        n = a.shape[0]
        if validities.shape[0] == n:
            val = validities
        else:
            # fallback: assume given order is already descending validity
            val = np.arange(n, 0, -1, dtype=float)
        order = np.argsort(-val, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        margin = abs(a_wins - b_wins)
        if margin < 2:
            continue

        ttb_winner = None
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        if ttb_winner == tally_winner:
            continue  # only conflict trials

        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        n_total += 1.0
        if resp == tally_winner:
            n_match += 1.0

    if n_total == 0:
        return float('nan')
    return float(n_match / n_total)
```

**Observed (real) value:** 0.7750 (var=0.0072)
**Predicted under pi_6:** 0.7530 (var=0.0025)
**Predicted under pi_3:** 0.7780 (var=0.0646)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Rate of choosing the option favoured by the HIGHEST-VALIDITY cue (screen
    # position 3, validity .95) on the 'double-dissociation' pairs, i.e. trials
    # where the two screen-extreme cues (positions 0 and 5) jointly favour one
    # option while the middle top-validity cue favours the other.
    num = 0.0
    den = 0.0
    num_all = 0.0
    den_all = 0.0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        if a.shape[0] < 6 or b.shape[0] < 6:
            continue
        d3 = a[3] - b[3]
        if d3 == 0:
            continue
        top_side = 0 if d3 > 0 else 1
        try:
            resp = int(r)
        except Exception:
            continue
        chose_top = 1.0 if resp == top_side else 0.0
        num_all += chose_top
        den_all += 1.0
        d0 = a[0] - b[0]
        d5 = a[5] - b[5]
        if d0 != 0 and d5 != 0:
            s0 = 1.0 if d0 > 0 else -1.0
            s5 = 1.0 if d5 > 0 else -1.0
            s3 = 1.0 if d3 > 0 else -1.0
            if s0 == s5 and s0 == -s3:
                num += chose_top
                den += 1.0
    if den > 0:
        return float(num / den)
    if den_all > 0:
        return float(num_all / den_all)
    return float('nan')
```

**Observed (real) value:** 0.5175 (var=0.0233)
**Predicted under pi_6:** 0.4450 (var=0.0192)
**Predicted under pi_3:** 0.0375 (var=0.0022)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    VAL = np.array([0.78, 0.60, 0.95, 0.88, 0.72], dtype=float)

    def ttb_winner(a, b, val):
        order = np.argsort(-val, kind='stable')
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    def scan_sign(d, lam):
        n = d.shape[0]
        pos = np.arange(n, dtype=float)
        z = -lam * (pos - pos.mean())
        z = z - z.max()
        w = np.exp(z)
        num = float(np.sum(w * d))
        den = float(np.sum(w * np.abs(d)))
        if den <= 1e-12:
            return 0.0
        return num / den

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
        except Exception:
            continue
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        val = VAL if n == VAL.shape[0] else np.linspace(0.95, 0.55, n)
        w_ttb = ttb_winner(a, b, val)
        if w_ttb is None:
            continue
        d = a - b
        s_prim = scan_sign(d, 2.0)    # primacy reader (lambda > 0)
        s_rec = scan_sign(d, -2.0)    # recency reader (lambda < 0)
        if abs(s_prim) < 0.15 or abs(s_rec) < 0.15:
            continue
        # scanner choice: S > 0 -> A (0), S < 0 -> B (1)
        c_prim = 0 if s_prim > 0 else 1
        c_rec = 0 if s_rec > 0 else 1
        # keep only cells where BOTH scanning orientations oppose TTB
        if c_prim == c_rec and c_prim != w_ttb:
            try:
                r = int(row['response'])
            except Exception:
                continue
            hits.append(1.0 if r == w_ttb else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.5075 (var=0.0075)
**Predicted under pi_6:** 0.3663 (var=0.0121)
**Predicted under pi_3:** 0.0269 (var=0.0009)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != 5 or b.shape[0] != 5:
            continue
        d = a - b
        # target cell: d = +-[1,-1,-1,1,0]  (pairs 1 and 2)
        pattern_ok = (
            abs(d[4]) < 1e-9
            and abs(abs(d[0]) - 1.0) < 1e-9
            and abs(abs(d[1]) - 1.0) < 1e-9
            and abs(abs(d[2]) - 1.0) < 1e-9
            and abs(abs(d[3]) - 1.0) < 1e-9
            and abs(d[0] - d[3]) < 1e-9
            and abs(d[1] - d[2]) < 1e-9
            and abs(d[0] + d[1]) < 1e-9
        )
        if not pattern_ok:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        # side supported by expert positions 0 and 3 (the first- and last-
        # discriminating cues); opposed by positions 1 and 2
        favored_is_a = d[0] > 0
        chose_favored = (resp == 0) if favored_is_a else (resp == 1)
        hits += 1 if chose_favored else 0
        tot += 1

    if tot == 0:
        return float('nan')
    return float(hits) / float(tot)

```

**Observed (real) value:** 0.8200 (var=0.0126)
**Predicted under pi_6:** 0.7913 (var=0.0231)
**Predicted under pi_3:** 0.9513 (var=0.0036)

### Experiment 8
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != b.shape[0] or a.shape[0] < 6:
            continue
        d = a - b
        disc = [int(i) for i in np.nonzero(np.abs(d) > 1e-9)[0]]
        target = None
        # Class (2): d nonzero at {0,2,5}, first and last agree, mid-list top cue opposes
        if disc == [0, 2, 5]:
            if d[0] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        # Class (3): d nonzero at {0,1,2,5}, three cues agree, single top cue opposes
        elif disc == [0, 1, 2, 5]:
            if d[0] == d[1] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        if target is None:
            continue
        n += 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp == target:
            hits += 1
    if n == 0:
        return 0.5
    return float(hits) / float(n)

```

**Observed (real) value:** 0.8413 (var=0.0134)
**Predicted under pi_6:** 0.7238 (var=0.0070)
**Predicted under pi_3:** 0.9694 (var=0.0014)

### Experiment 9
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    hi_load_hits, hi_load_n = 0, 0
    lo_load_hits, lo_load_n = 0, 0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        n = int(min(a.shape[0], b.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b

        pos_idx = np.where(d > 0.5)[0]   # features favouring A
        neg_idx = np.where(d < -0.5)[0]  # features favouring B
        k_a = pos_idx.size
        k_b = neg_idx.size

        # keep only the 3-cue conflict cells: two cues on one side, one on the other
        if not ((k_a == 2 and k_b == 1) or (k_a == 1 and k_b == 2)):
            continue

        # did the subject pick the side backed by the LARGER NUMBER of cues?
        if k_a == 2:
            chose_many = 1 if int(resp) == 0 else 0
        else:
            chose_many = 1 if int(resp) == 1 else 0

        # processing load = shared positive endorsements (non-discriminating 1s)
        L = int(np.count_nonzero((a > 0.5) & (b > 0.5)))

        if L > 0:
            hi_load_hits += chose_many
            hi_load_n += 1
        else:
            lo_load_hits += chose_many
            lo_load_n += 1

    if hi_load_n == 0 or lo_load_n == 0:
        return 0.0

    return float(hi_load_hits / hi_load_n - lo_load_hits / lo_load_n)

```

**Observed (real) value:** 0.0533 (var=0.0494)
**Predicted under pi_6:** -0.0050 (var=0.0446)
**Predicted under pi_3:** 0.0033 (var=0.0067)

### Experiment 10
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    weak_idx = {1, 4}      # validities .56 and .62
    strong_idx = {0, 3}    # validities .72 and .95

    weak_hits = []
    strong_hits = []

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
        except Exception:
            continue
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        d = a - b
        nz = np.flatnonzero(np.abs(d) > 1e-9)
        if nz.size != 1:
            continue  # only single-discriminating-cue trials
        j = int(nz[0])
        # 1 if subject chose the option endorsed by the single discriminating expert
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        endorsed_a = d[j] > 0
        hit = 1.0 if (chose_a == endorsed_a) else 0.0
        if j in weak_idx:
            weak_hits.append(hit)
        elif j in strong_idx:
            strong_hits.append(hit)

    if len(weak_hits) == 0 or len(strong_hits) == 0:
        return float('nan')

    return float(np.mean(strong_hits) - np.mean(weak_hits))

```

**Observed (real) value:** -0.2025 (var=0.0220)
**Predicted under pi_6:** -0.1838 (var=0.0208)
**Predicted under pi_3:** 0.0062 (var=0.0039)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all ten experiments pi_6 is the far more task-invariant account. It matches: the reversed single-cue validity ladder in Exp10 (obs -0.203 vs pi_6 -0.184, pi_3 +0.006), the coin-flip double-dissociation cells in Exp5 (obs 0.518 vs pi_6 0.445, pi_3 0.038) and Exp6 (obs 0.508 vs pi_6 0.366, pi_3 0.027), the pooled extreme-vs-crowd cells in Exp1 (obs 0.490 vs pi_6 0.433, pi_3 0.969), the tally-dominance in Exp4 (obs 0.775 vs 0.753/0.778), the null load effect in Exp9, and it is roughly as close as pi_3 in Exp7 (0.82 vs 0.79/0.95) and Exp8 (0.841 vs 0.724/0.969). pi_3's unique wins are Exp3 (marginal, 0.093 vs 0.011/0.103) and, decisively, Exp2 (obs 0.850 vs pi_6 0.230, pi_3 0.972). But pi_3 pays for that with catastrophic misses wherever screen-extreme cues do NOT drive choice (Exps 1, 5, 6) and it is structurally blind to the strongest, most replicated qualitative signature in the whole corpus \\u2014 the inverted validity ladder on single-cue trials (Exp10). pi_3 is therefore the degenerate member of the pair: it is a one-trick position model that is pinned to ceiling whenever a unanimous or edge-favoured cue set exists, which the data flatly contradict in half the experiments.\n\nThat said, pi_6 is not safe either. Exp2's cell 5 (two weakest screen-extreme experts .62/.58 favouring A against the two heavyweights .95/.86 favouring B) came out at 0.85 for the WEAK pair, a 62-point miss for pi_6 and the single most informative anomaly in the dataset. Combined with Exp10's inverted ladder, Exp4/Exp8's majority-following, and Exp5/Exp6's coin flips in 2-vs-1 contests against a .95 expert, the data look like: (i) validity weights that are heavily compressed \\u2014 close to unit-weight counting, with one .95 expert worth roughly two mid experts; (ii) an inverse-diagnosticity gap inference (a difference flagged only by an imprecise expert implies a bigger quality gap) that operates in DENSE displays too, not only under overwhelming silence; (iii) genuine indifference when counts/weights nearly balance. pi_6 has (i) only mildly, and (ii) only through a silence term that switches off in dense displays \\u2014 exactly why it fails Exp2. Rather than patch pi_3's position gradient again, the productive move is to replace pi_3 with a competitor built around compressed/unit-weight counting plus salience-biased sampling and inverse-validity gap inference, which can in principle explain the entire corpus including the cells where pi_6 fails.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_3 with a 'capacity-limited cue sampling + unit-weight majority vote + inverse-diagnosticity gap inference' theory. Core claims: (1) Subjects do not weight cues by stated validity in a graded additive way; they SAMPLE a subset of the discriminating cues, with sampling probability rising with both stated validity and screen salience (first- and last-read positions get an attention bonus), then (2) vote by near-unit weights over the sampled discriminating cues \\u2014 counting, not weighting \\u2014 so a single .95 expert is worth only ~1-1.5 mid experts and is routinely outvoted by two or three weaker ones (explains Exp4 0.775, Exp8 0.841, Exp1 cell 8, Exp3's small TTB-conflict effect). (3) The evidential IMPACT of each sampled difference is inversely related to that expert's validity ('validity = sensitivity': if only the imprecise expert notices a difference, the true quality gap must be large; if only the near-oracle notices it, the gap is marginal). Crucially this gap-inference operates on every cue in every display, not only when the panel is overwhelmingly silent \\u2014 this is the key departure from pi_6 and the only way to get Exp10's -0.20 ladder AND Exp2's 0.85 preference for the two weakest screen-extreme experts over the .95+.86 pair in a dense display. (4) When the sampled vote is tied or nearly tied, the tie is broken by the inverse-diagnosticity term rather than by raw validity, and residual near-balance produces genuine indifference (convex/threshold read-out), reproducing the coin flips in Exp1 (0.49), Exp5 (0.52) and Exp6 (0.51) for 2-weak-vs-1-strong contests, and the null load effect in Exp9. (5) No primacy/recency subpopulations and no strategy mixtures; position enters only as a sampling-probability bonus for the two list edges, so it can tip 2-vs-2 contests (Exp2) without ever producing pi_3's ceiling predictions in Exps 1/5/6. Parameterise: compression exponent on (v-0.5) near 0 (near-unit weights), sampling floor/edge bonus, gap-inference strength, indifference threshold, softmax beta, lapse. This theory must be checked to reproduce, at minimum: Exp10 \\u2248 -0.20, Exp2 \\u2248 0.85, Exp5/6 \\u2248 0.50, Exp1 \\u2248 0.49, Exp4 \\u2248 0.78, Exp8 \\u2248 0.84, Exp7 \\u2248 0.82."
}
```

## Usage

```json
{
  "input_tokens": 28388,
  "output_tokens": 12244
}
```
