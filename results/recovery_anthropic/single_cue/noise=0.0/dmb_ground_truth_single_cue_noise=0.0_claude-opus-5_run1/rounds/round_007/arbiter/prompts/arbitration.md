# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_9") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_9" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7
**MDSC-NV (sharpened) — Maximum-Doubt Single-Cue selection with Null-Verdict capture.**

Four claims; the mechanism family is unchanged from the accepted base, but one quantitative commitment is sharpened.

**(1) One-reason choice by MAXIMUM DOUBT (no summation).** On each trial the subject settles A-vs-B with the verdict of a SINGLE expert. Which expert is consulted is decided by *doubt*: a disagreement voiced by a near-chance expert (v≈.53) is a strange, newsworthy fact that has to be adjudicated, whereas a disagreement voiced by a near-certain expert (v≈.97) is discounted as a mere echo of the impression the subject already has. Salience is log u_j = delta*log(1-v_j) + lam*j and the consulted row is drawn from softmax(u). **The doubt gain is LARGER than previously assumed (delta ≈ 13-17, not 8-12): the doubt rule is essentially argmax, not a graded share.** This is the sharpened commitment. It matters wherever two moderately-doubtful rows are within ~0.4 log-validity of one another (e.g. v=.55 vs v=.70 rows two columns apart): a share-based rule lets the second-most-doubtful row steal ~18% of the verdict and blunts the choice, whereas the near-argmax rule leaves the display fully decided by the single most doubtful row. Coalitions still buy nothing: five agreeing rows never outvote one more-doubtful row, which is why choice is systematically anti-validity in every design where the printed labels are usable.

**(2) Reading position is a WEAK tie-breaker.** lam = pos_ratio*delta with pos_ratio ≈ 0.09-0.14 — an order of magnitude below the doubt term's dynamic range. Position therefore only arbitrates between rows whose stated validities are (nearly) *equal* — exactly the .955/.950/.945 and .958/.954/.950 designs, where it makes the last low-validity row win. Whenever validities genuinely differ, doubt beats recency: a low-validity EARLY column defeats a late .93/.99 column. Because delta is raised and pos_ratio held fixed, the *ratio* of doubt-range to positional gain is preserved, so all orderings in the corpus are unchanged.

**(3) NULL-VERDICT CAPTURE.** Doubt is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the most doubtful expert **in the whole display**, before the subject checks whether that expert actually discriminates. With probability q (≈ .41, slightly higher than previously estimated) the subject simply goes with whoever captured attention; if that expert gave both products the same rating it returns no verdict, the subject has nothing to adjudicate and guesses. With probability 1-q the subject re-targets deliberately, restricting the doubt competition to the rows that differ. Sharp prediction: *holding the discriminating rows fixed, adding agreement on a MORE doubtful (lower-validity) row pushes choice toward chance*, whereas adding agreement on a high-validity row does nothing. Displays in which only two high/mid-validity rows disagree while the display's most doubtful expert stays silent are therefore near chance — no compensatory or one-reason model predicts that non-discriminating rows matter at all.

**(4) LABEL-DEGENERACY GATE (all-or-none) and LOAD LAPSE.** The doubt rule presupposes that each printed validity picks out one expert. If any label is printed twice (tolerance .002 — it must NOT fire for the .004/.005-gap designs) the experts are interchangeable, the doubt ordering is unavailable, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). Independently, the more rows disagree the more often the subject loses the thread: p(guess) = 1-(1-eps0)*rho^(m-2), which compresses conflict displays toward chance and keeps even unanimous 6-row displays well below ceiling. Individual subjects differ mainly in how leaky (eps0, rho) and how capture-prone (q) they are, not in the doubt rule itself.

No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # MDSC-NV: Maximum-Doubt Single-Cue selection with Null-Verdict capture
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed labels) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9975)

    # ---------- parameters ----------
    delta = float(parameters['doubt_gain'])
    lam = float(parameters['pos_ratio']) * delta        # weak positional gain
    q = float(np.clip(float(parameters['capture']), 0.0, 1.0))
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    kap = float(parameters['tally_kappa'])

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    fav_a = diff[disc] > 0.0
    n_a = float(np.sum(fav_a))
    n_b = float(m) - n_a

    # ---------- label-degeneracy gate (all-or-none, exact repetition) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        # experts cannot be individuated -> unweighted graded row tally
        z = float(np.clip(kap * (n_a - n_b), -60.0, 60.0))
        p_core_a = 1.0 / (1.0 + np.exp(-z))
    else:
        # doubt salience over ALL rows
        logu = delta * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)

        lg = logu - np.max(logu)
        s = np.exp(lg)
        ssum = float(np.sum(s))
        s = s / ssum if (np.isfinite(ssum) and ssum > 0.0) else np.full(n, 1.0 / n)

        ld = logu[disc]
        ld = ld - np.max(ld)
        t = np.exp(ld)
        tsum = float(np.sum(t))
        t = t / tsum if (np.isfinite(tsum) and tsum > 0.0) else np.full(m, 1.0 / m)

        # (i) deliberate re-targeting: doubt competition among disagreeing rows
        p_rest_a = float(np.sum(t[fav_a]))

        # (ii) null-verdict capture: attention grabbed by the most doubtful
        #      expert in the whole display; if it is silent -> guess
        mask = np.zeros(n, dtype=bool)
        mask[disc] = True
        s_nondisc = float(np.sum(s[~mask]))
        s_disc = s[disc]
        p_glob_a = float(np.sum(s_disc[fav_a])) + 0.5 * s_nondisc

        p_core_a = q * p_glob_a + (1.0 - q) * p_rest_a

    # ---------- load-dependent lapse ----------
    expo = max(m - 2, 0)
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_9
**NAVA-HLp — Normalized Anti-Validity Accumulation with a HALF-LOUDNESS load lapse and a PARTIAL unanimity discount.**

Six claims. The evidence rule is unchanged from the accepted base: graded, compensatory, anti-validity, divisively normalized, with no one-reason step, no argmax, no verdict capture and no doubt trigger. Only the *price charged for holding rival voices* is revised, and only partially.

**(1) Every dissenting row speaks, and doubt is what makes it loud.** A printed validity close to chance makes a disagreement *newsworthy* (it demands adjudication); a printed validity close to certainty makes the same disagreement a mere echo of the impression the subject already has. Each row carries weight w_j = (1 − v_j)^γ · exp(λ·j): doubt raised to a moderate power γ ≈ 1.9–2.5, times a modest reading-position gain λ ≈ 0.28–0.45. All discriminating rows contribute simultaneously, so coalitions matter monotonically, while the global direction of choice stays systematically anti-validity in every design whose labels are usable.

**(2) Divisive normalization relative to the loudest dissent.** Weights enter only as ratios to the loudest discriminating voice, ŵ_j = w_j / max_{k∈disc} w_k, and E = Σ_{j∈disc} s_j ŵ_j / (σ + Σ_{j∈disc} ŵ_j), p(A) = σ(βE). Absolute scale is discarded (dominance displays decided only by high-validity experts are still resolved confidently) while the denominator grows with each speaking row, so a coalition erodes a doubtful dissenter's lead toward, but rarely far past, chance.

**(3) The displacement leak is a DOUBT GRADIENT.** Doubt is a property of the expert, not of the disagreement, so the most doubtful expert *in the whole display* competes for attention before the subject checks whether it discriminates. What costs accuracy is the *doubt gradient* between the loudest silent expert and the loudest speaking one: attention is pulled off the adjudication only when the silent expert is appreciably MORE doubtful than the expert currently carrying the argument. Formally d = clip((v_loudest-disc − v_loudest-silent)/Δ, 0, 1) · w_silent,max/(w_silent,max + w_disc,max) with fixed Δ = .05, and the leak multiplies the retention term by (1 − κ·d), κ ≈ .40–.95. Silent rows never contribute evidence; the term is purely subtractive and can only shrink |p − .5|.

**(4) Losing the thread is charged HALF PER ROW and HALF PER LOUD VOICE.** What has to be held in mind while the comparison is run is neither the raw number of disagreeing rows (which over-charges a display in which one towering doubtful voice drowns out five whispers) nor purely the effective number of loud voices (which under-charges a long unanimous list that still has to be scanned). Both costs are real and roughly equally weighted: L = ½·(m − 2) + ½·c·(Σ_{j∈disc} ŵ_j − 1), and p(guess) = 1 − (1−ε₀)·ρ^max(L,0)·(1 − κ·d) with ρ ≈ .64–.80 and floor ε₀ ≈ .07–.15.

**(5) NEW — unanimity is CHEAPER BUT NOT FREE.** The loud-voice component of the load is an *adjudication* cost: it is the price of holding rival loud voices against one another. When every speaking row agrees there is no adjudication, but the rows still have to be scanned and their loudness still occupies the comparison, so the cost is discounted rather than waived: c = 1 when both options are favoured by at least one row, and c = c₀ ≈ .45–.75 when the display is unanimous. The row-scan half, ½·(m − 2), is charged in full regardless, so a 6-vs-0 dominance display remains well below ceiling while a 3-vs-3 display of the same size is markedly noisier. This is the model's sharp prediction about dominance: accuracy on unanimous displays falls with list length but strictly more slowly than on conflict displays of matched length and matched weight concentration.

**(6) Degenerate printed labels collapse the weighting.** If any two printed validities are within .002 the experts cannot be individuated; all weights become equal and the same normalized rule reduces to an unweighted tally, E = (n_A − n_B)/(σ + m) with the same β and σ — exact chance on 1-1 splits, no anti-validity and no recency signature. The displacement leak is switched off, because no row is 'the doubtful one'.

No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # NAVA-HLp: Normalized Anti-Validity Accumulation with doubt-gradient displacement leak,
    # HALF-ROW / HALF-LOUD-VOICE load lapse, and a PARTIAL unanimity discount on the loud-voice half
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed labels) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
    beta = float(parameters['beta'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    kappa = float(np.clip(float(parameters['kappa']), 0.0, 1.0))
    c0 = float(np.clip(float(parameters['c0']), 0.0, 1.0))

    DV = 0.05  # fixed doubt-gradient scale (not fitted)

    def sig(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])

    # ---------- label-degeneracy gate (printed labels not individuable) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    # ---------- doubt x reading-position weights ----------
    if degenerate:
        w = np.ones(n, dtype=float)
    else:
        logw = gamma * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)
        logw = logw - np.max(logw)
        w = np.exp(np.clip(logw, -700.0, 0.0))

    wd = w[disc]
    wmax = float(np.max(wd))
    if (not np.isfinite(wmax)) or wmax <= 0.0:
        what = np.ones(m, dtype=float)
        wmax = 1.0
    else:
        what = wd / wmax

    # ---------- divisive normalization ----------
    num = float(np.sum(signs * what))
    den = sigma + float(np.sum(what))
    if den <= 0.0:
        E = 0.0
    else:
        E = num / den
    p_core_a = sig(beta * E)

    # ---------- doubt-gradient displacement leak (bounded, evidence-free) ----------
    d = 0.0
    if (not degenerate) and m < n:
        mask = np.ones(n, dtype=bool)
        mask[disc] = False
        sil_idx = np.nonzero(mask)[0]
        if sil_idx.size > 0:
            k_s = int(sil_idx[int(np.argmax(w[sil_idx]))])
            ws = float(w[k_s])
            k_l = int(disc[int(np.argmax(wd))])
            gap = float(v[k_l] - v[k_s])          # positive iff silent row is MORE doubtful
            mult = float(np.clip(gap / DV, 0.0, 1.0))
            if np.isfinite(ws) and (ws + wmax) > 0.0:
                d = mult * ws / (ws + wmax)
    d = float(min(max(d, 0.0), 1.0))

    # ---------- HALF-ROW / HALF-LOUD-VOICE effective load, with partial unanimity discount ----------
    sum_what = float(np.sum(what))
    if not np.isfinite(sum_what):
        sum_what = float(m)
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    c = 1.0 if min(n_a, n_b) >= 1.0 else c0   # unanimity: adjudication cost discounted, not waived
    expo = 0.5 * float(m - 2) + 0.5 * c * (sum_what - 1.0)
    expo = float(max(expo, 0.0))

    keep = (1.0 - eps0) * (rho ** expo) * (1.0 - kappa * d)
    keep = float(min(max(keep, 0.0), 1.0))
    g = 1.0 - keep

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=6):** [0.9, 0.97, 0.62, 0.55, 0.7, 0.74]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 2: A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 3: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 5: A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 8: A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 9: A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 10: A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 11: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 12: A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 15: A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 16: A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]

**Rationale:** GOAL: separate MDSC-NV (advocated) from NAVA-HLp (competitor). Both models are anti-validity ('doubt') and both add a reading-position gain, and both let silent rows hurt (MDSC via null-verdict capture, NAVA via its doubt-gradient displacement leak), so capture alone cannot separate them. The decisive axis is SINGLE-CUE NEAR-ARGMAX (delta = 13-17 on log(1-v)) versus DIVISIVELY-NORMALIZED COMPENSATORY SUMMATION (gamma = 1.9-2.5). A validity contrast of only ~0.3 in log(1-v) makes two rows essentially interchangeable for NAVA (weight ratio e^{2.2*0.3} ~ 1.9, i.e. w-hat ~ 0.5-0.6 each) but ~90:1 for MDSC. Therefore a coalition of two or three 'second-tier doubtful' rows must overturn the single most doubtful row for NAVA (sum of w-hat > 1 flips the sign of E) while buying literally nothing for MDSC.

VALIDITIES [0.90, 0.97, 0.62, 0.55, 0.70, 0.74] at columns c0..c5. Engineered properties: (i) c3 (.55) is MDSC's global doubt anchor at EVERY corner of its box (delta 13-17 x pos_ratio .09-.14 gives lam 1.17-2.38; c3's log-salience lead over the next row, c4 or c2, is 3.4-4.4 log units), so MDSC is effectively 'whoever wins c3; if c3 is silent, capture forces a guess with prob q'. (ii) For NAVA the same display has w-hat(c2)=0.49, w-hat(c4)=0.58, w-hat(c5)=0.60 relative to c3, i.e. any two of them sum to >1 and overturn c3. (iii) Validity does NOT track column order, so 'read the first/last row' cannot masquerade as either model. (iv) All label gaps >= .04, far above both degeneracy tolerances (.002/.0025), so neither model can retreat into tally mode. (v) The two high-validity rows (c0 .90, c1 .97) are near-weightless for both models and serve as silent filler and as the capture/leak probe.

8 base structures, each with its exact A/B mirror (16 unique pairs, K=6 -> 96 trials), so both models' predicted responses are perfectly key-balanced; non-discriminating rows carry matched 1s AND matched 0s so displays are dense.

BLOCK 1 - CROWN JEWEL: COALITION GRADIENT AGAINST THE DOUBT ANCHOR (bases 1-4). A always wins c3 (.55) while B's coalition of slightly-less-doubtful rows grows: 1-vs-0 (base 1), 1-vs-1 with c5 (base 2), 1-vs-2 with c4,c5 (base 3), 1-vs-3 with c2,c4,c5 (base 4). MDSC predicts a SHALLOW, load-only, never-crossing-chance decline ~.88, .87, .79, .73 (its core stays at .96-.99 in every cell because c3 is the argmax). NAVA predicts a CRASH THROUGH CHANCE ~.89, .64, .46, .41 as the summed coalition passes w-hat = 1 between the 1-vs-1 and 1-vs-2 displays. Checked at the corners of NAVA's box: base 3 lies in [.42,.53] and base 4 in [.40,.45], while MDSC stays in [.70,.80] - a 25-35 point separation with matched load. Flat-and-above-chance versus crossing-chance is the parameter-robust quantitative signature.

BLOCK 2 - ROW-COUNT TIE AND 1-vs-5 (bases 5, 8). Base 5: A wins c0 (.90, weightless) and c3 (.55); B wins c4,c5. At an EXACT 2-2 row count MDSC predicts ~.74 (the anchor decides) while NAVA predicts ~.47 (.42-.52 across corners) because the summed late coalition beats the anchor. Base 8: A wins c3 alone, B wins the other FIVE rows; MDSC ~.65 (one more-doubtful row outvotes five), NAVA ~.44 (the coalition wins). No tally account can produce either value, and the two models split directionally.

BLOCK 3 - CAPTURE / DISPLACEMENT-LEAK PROBE (base 6 vs base 1). Base 6 is a unanimous 2-0 display carried by c0 and c1 with the .55 anchor SILENT; base 1 is a dominance display in which the anchor SPEAKS, with exactly matched MDSC load (expo = 0 in both). MDSC predicts a 15-point drop (.875 -> .724) purely because a NON-discriminating row is the most doubtful one - the size of the drop is pinned by q and cannot exceed ~.20 of the core. NAVA predicts a drop too (its doubt-gradient leak fires: gap .35 > Delta, weight ratio .99), but its magnitude is proportional to kappa and ranges .52-.73. This cell therefore does not decide the contest; it pins kappa/q and guards against attributing the Block-1 divergence to a general silent-row effect.

BLOCK 4 - SHARED LOAD BASELINE (base 7, unanimous 6-0). MDSC ~.66, NAVA ~.68 (its unanimity discount c0 partially offsets the larger loud-voice load). Together with base 1 (~.88 for both) this pins the lapse parameters of both models free of the contrast, so divergences in Blocks 1-2 cannot be written off as global noise.

INTERPRETATION: if choices track the ROW COUNT / summed loudness - dropping below chance as soon as two moderately doubtful rows oppose the single most doubtful one (bases 3, 4, 5, 8 in the .41-.47 range) - compensatory divisive summation (NAVA-HLp) is confirmed and the near-argmax single-cue rule is falsified. If instead the option holding the .55 row is chosen ~.73-.88 regardless of how many slightly-less-doubtful rows oppose it, and dominance accuracy drops ~15 points purely as a function of whether the .55 expert happens to be silent, MDSC-NV wins.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design's decisive contrast is whether the option backed by the single most doubtful expert (column 3, validity .55) survives when an opposing coalition of slightly-less-doubtful experts (c2 .62, c4 .70, c5 .74) speaks against it. I select exactly those trials in which the .55 expert discriminates AND at least one row favours the other option (base structures 2, 3, 4, 5 and 8: 1-vs-1, 1-vs-2, 1-vs-3, 2-vs-2, 1-vs-5), and score the proportion of choices that follow the .55 anchor. MDSC-NV (near-argmax single-cue on doubt, delta 13-17) says coalitions buy essentially nothing: the anchor's core probability stays .96-.99 in every one of these cells and only the load lapse pulls the observed rate down, giving an expected anchor-follow rate around .72-.80. NAVA-HLp's divisive compensatory summation gives the coalition rows normalized weights of ~.49-.60 each, so two or three of them sum past 1 and flip the sign of E, predicting anchor-follow rates around .41-.50 (i.e. below chance) on most of these cells and ~.48 pooled. The predicted gap is therefore roughly .27-.30 with the two theories straddling chance in opposite directions. Because the metric aggregates ~30 unique-trial repetitions per subject (5 base structures x 2 mirrors x 6 reps = 60 trials) the binomial noise per subject is ~.06 SD and the only additional between-subject spread comes from lapse parameters (a few points), so the contrast should be both large and tight. The metric is well-defined on a single subject's slice and on the pooled frame alike, and it is not trivially true of MDSC-NV: any compensatory or tally-like process (including the competitor) would put it at or below .5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = 0.0
    tot = 0.0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    ANCHOR = 3  # column with the lowest printed validity (0.55)
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = [int(x) for x in list(a_raw)]
        b = [int(x) for x in list(b_raw)]
        k = min(len(a), len(b))
        if k <= ANCHOR:
            continue
        if a[ANCHOR] == b[ANCHOR]:
            continue  # anchor silent -> not a coalition test
        n_a = 0
        n_b = 0
        for i in range(k):
            if a[i] > b[i]:
                n_a += 1
            elif b[i] > a[i]:
                n_b += 1
        anchor_favors_a = a[ANCHOR] > b[ANCHOR]
        opposing = n_b if anchor_favors_a else n_a
        if opposing < 1:
            continue  # pure dominance for the anchor side -> no coalition
        try:
            resp = int(r)
        except Exception:
            continue
        chose_a = (resp == 0)
        if chose_a == anchor_favors_a:
            hits += 1.0
        tot += 1.0
    if tot == 0.0:
        return float('nan')
    return float(hits / tot)

### RESULTS
- Predicted under pi_7 (simulated): 0.7610 (var=0.0035)
- Predicted under pi_9 (simulated): 0.4777 (var=0.0043)
- Observed on real data: 0.8620 (var=0.0155)

## EXPERIMENT 2 (proposed by pi_9)

### DESIGN
**Validities (n_features=6):** [0.52, 0.95, 0.9, 0.78, 0.8, 0.83]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  trial 3: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 7: A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 8: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  trial 9: A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 10: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 11: A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 14: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 16: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 18: A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 19: A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 20: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 21: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 23: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 24: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Rationale:** GOAL: separate NAVA-HLp (advocated: divisively-normalized COMPENSATORY summation of doubt-weighted rows, gamma = 1.9-2.5) from MDSC-NV (competitor: NEAR-ARGMAX single-cue doubt selection, delta = 13-17). Both models are anti-validity, both add a reading-position gain of comparable relative size (MDSC trades one column of recency for 1.725/15 = 0.115 log-(1-v) units, NAVA for 0.36/2.2 = 0.164), and both let silent rows hurt (capture vs. doubt-gradient leak). Position and silent-row effects therefore CANNOT separate them; the only decisive axis is whether the weight of the OPPOSING rows enters the decision at all (NAVA: continuously, through sum(w-hat)) or not (MDSC: only the identity of the doubt argmax matters).

VALIDITIES [0.52, 0.95, 0.90, 0.78, 0.80, 0.83] at columns c0..c5. Engineering: (i) c0 = .52 sits at the FIRST column, i.e. doubt and recency are OPPOSED for it, yet it is MDSC's doubt argmax at every corner of its parameter box (its log-salience lead over the runner-up is 4.1 log units at delta=13/pos_ratio=.14 and 8.8 at delta=17/pos_ratio=.09), so MDSC reduces to 'whoever wins c0; if c0 is silent, capture forces a guess with prob q'. (ii) The three late mid-doubt rows c3 (.78), c4 (.80), c5 (.83) have NAVA w-hat = .53, .62, .62 relative to c0 (at gamma=2.2, lam=.36), because a log-(1-v) gap of only ~0.8-1.0 is huge for delta=15 (ratio ~1e5) but small for gamma=2.2 (ratio ~2). Hence ANY TWO of them sum to w-hat > 1 and must overturn c0 for NAVA, while buying literally nothing for MDSC. (iii) c1 (.95) and c2 (.90) are near-weightless for both models (w-hat = .010 and .065) and serve as filler and as graded-ladder rungs. (iv) All label gaps >= .02, far above both degeneracy tolerances (.002), so neither model can retreat into tally mode; validity does not track column order, so 'read the first/last row' cannot masquerade as either model.

12 base structures, each with its exact A/B mirror (24 unique pairs, K = 4 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense. Predictions below: NAVA (gamma=2.2, lam=.36, sigma=.20, beta=3.3, eps0=.11, rho=.72, kappa=.60, c0=.60) / MDSC (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81).

BLOCK 1 - CROWN JEWEL A: FLAT-vs-GRADED 1-vs-1 LADDER AT CONSTANT LOAD (bases 2-5, pairs 3-10). A always wins c0; B wins exactly one row, whose doubt weight is swept: c1 (w-hat .010), c2 (.065), c3 (.53), c5 (.62). m = 2 and the doubt argmax (c0) speaks in every cell, so MDSC must predict the SAME value on all four (.88, .88, .88, .88) for every parameter setting - its verdict is set by the argmax alone and the opposing row's validity is irrelevant beyond an e^-7 softmax crumb. NAVA must predict a monotone 26-point slide (.89, .87, .67, .63) because the opposing w-hat enters both the numerator and the normalizing denominator. Robustness: at the corners of NAVA's box the c0-vs-c5 cell lies in [.42, .77], i.e. 11-46 points below MDSC's flat .88. Flat-vs-graded within matched m is the cleanest parameter-free read-out in the design.

BLOCK 2 - CROWN JEWEL B: COALITION GRADIENT AGAINST THE DOUBT ANCHOR (bases 5-8, pairs 9-16). A wins c0 while B's coalition of slightly-less-doubtful late rows grows: 1-vs-1 (c5), 1-vs-2 (c4,c5), 1-vs-3 (c3,c4,c5), 1-vs-5 (all others). MDSC predicts a shallow, load-only decline that NEVER crosses chance (.88, .81, .75, .66; its core is pinned at ~1.0 in every cell because c0 is the argmax and also the capture target). NAVA predicts a CRASH THROUGH CHANCE as sum(w-hat) passes 1 between the 1-vs-1 and 1-vs-2 displays (.63, .45, .40, .43). Corner checks: NAVA's 1-vs-2 cell lies in [.34, .61] and its 1-vs-3 cell in [.33, .50], while MDSC stays in [.72, .84] - a 20-45 point separation at matched display size, with a directional split at 1-vs-2 and beyond.

BLOCK 3 - ROW-COUNT TIE AND MATCHED-m CONFLICT (bases 9, 12; pairs 17-18, 23-24). Base 9 is an exact 2-vs-2 row tie: A wins c0 (.52) + c2 (.90, weightless), B wins c4,c5. NAVA .47 (the summed late coalition beats the anchor); MDSC .75 (the anchor decides). Base 12 is a 3-vs-3 tie at m=6: A wins c0,c1,c2, B wins c3,c4,c5. NAVA .44; MDSC .66. No vote-counting account can produce either value in either direction, and the two models split by 22-28 points at exact tallies.

BLOCK 4 - SHARED BASELINES THAT BLOCK THE 'GLOBAL NOISE' AND 'SILENT-ROW' ESCAPES (bases 1, 10, 11; pairs 1-2, 19-22). Base 1 (m=1, A wins c0 alone): NAVA .89 / MDSC .88 - pins eps0 for both. Base 10 (unanimous 2-0 carried by c4,c5 with the .52 anchor SILENT): NAVA .73 (its doubt-gradient leak fires: gap .31 > Delta, d = .62) / MDSC .72 (null-verdict capture) - the two accounts agree here, so the Block-1/2 divergence cannot be re-described as a general silent-row effect. Base 11 (unanimous 6-0): NAVA .68 / MDSC .66 - shared load baseline, and together with base 12 it tests NAVA's PARTIAL unanimity discount (claim 5): NAVA says the 3-3 display is ~24 points worse than the 6-0 display of identical size, MDSC says they are identical (same m, same argmax verdict).

INTERPRETATION: if all four 1-vs-1 displays sit flat at ~.88 regardless of how doubtful the opposing expert is, and if the option holding the .52 row is still chosen ~.72-.84 when two, three or five slightly-less-doubtful rows oppose it, MDSC-NV's near-argmax single-cue rule is confirmed and compensatory divisive summation is falsified. If instead choice probability slides monotonically with the opposing row's doubt weight at constant m (.89 -> .63), and drops through chance as soon as two moderately doubtful rows oppose the single most doubtful one (bases 6-9, 12 in the .40-.47 range), NAVA-HLp is confirmed and coalitions demonstrably buy exactly what their normalized loudness is worth.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The decisive axis between NAVA-HLp (compensatory divisive summation) and MDSC-NV (near-argmax single-cue doubt selection) is whether the WEIGHT of the opposing rows enters the decision at all. I isolate exactly the coalition displays in which the most-doubtful expert (column 0, v=.52 \u2014 MDSC's doubt argmax at every corner of its parameter box) discriminates and is opposed by at least two of the slightly-less-doubtful late rows (c3=.78, c4=.80, c5=.83). This selection picks out bases 6, 7, 8, 9 and 12 of the design (trials 11/12, 13/14, 15/16, 17/18, 23/24), i.e. 10 unique pairs \u00d7 4 reps = 40 trials per subject \u2014 enough to keep binomial noise per subject small (~.006 variance).

The metric is the proportion of those trials on which the subject picks the option endorsed by the .52 anchor. Hand-computing both generative models at mid-box parameters: NAVA gives .452, .404, .427, .472, .439 across the five bases (mean \u2248 .44 \u2014 BELOW chance, because \u03a3\u0175 of the opposing coalition exceeds 1 and normalization lets it overturn the anchor), whereas MDSC gives .806, .747, .662, .747, .662 (mean \u2248 .73 \u2014 always ABOVE chance, because only the identity of the doubt argmax matters and coalitions buy nothing; only the load lapse pulls it toward .5). The gap is ~.29 with a directional split across chance, so it cannot be mimicked by differences in lapse rate, side bias, or general anti-validity tendency: both theories are anti-validity, both charge load, and mirrored A/B versions of every base are included so side bias cancels exactly. Per-subject parameter heterogeneity moves each theory only within its own corner range (NAVA .33\u2013.61, MDSC .66\u2013.84), leaving the two distributions essentially non-overlapping at N=25.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(list(a_raw), dtype=float).ravel()
        b = np.asarray(list(b_raw), dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k < 6:
            continue
        # the near-chance anchor expert (column 0, validity .52) must discriminate
        if a[0] == b[0]:
            continue
        anchor_side = 0 if a[0] > b[0] else 1  # 0 == option A, 1 == option B
        # count late, mid-doubt rows (columns 3,4,5) that oppose the anchor
        opp = 0
        for j in (3, 4, 5):
            if a[j] == b[j]:
                continue
            fav = 0 if a[j] > b[j] else 1
            if fav != anchor_side:
                opp += 1
        if opp < 2:
            continue
        n += 1.0
        try:
            r = int(resp)
        except Exception:
            continue
        if r == anchor_side:
            hits += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)


### RESULTS
- Predicted under pi_7 (simulated): 0.7245 (var=0.0054)
- Predicted under pi_9 (simulated): 0.4215 (var=0.0085)
- Observed on real data: 0.8260 (var=0.0180)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """One-reason adherence on model-disagreement trials.

    For every trial we compute (a) the option favoured by the FIRST
    discriminating cue in validity order (features are listed in
    descending validity: 0.95, 0.85, 0.75, 0.65, 0.55) and (b) the option
    favoured by the unit-weight tally of feature-wise wins.  We keep only
    trials where these two disagree in direction OR where the tally is tied
    (so a counting rule must guess).  The metric is the proportion of such
    trials on which the subject picked the option favoured by the first
    discriminating cue.
    """
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]

        # first discriminating cue (features already in descending validity)
        first_winner = None
        for j in range(n):
            if a[j] > b[j]:
                first_winner = 0
                break
            if b[j] > a[j]:
                first_winner = 1
                break
        if first_winner is None:
            continue  # no cue discriminates: uninformative trial

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None  # counting rule must guess

        if tally_winner is not None and tally_winner == first_winner:
            continue  # agreement control trial: not diagnostic

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hits.append(1.0 if r == first_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2820 (var=0.0035)
**Predicted under pi_7:** 0.2878 (var=0.0023)
**Predicted under pi_9:** 0.3068 (var=0.0031)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.6, 0.93, 0.55, 0.87, 0.7, 0.78], dtype=float)

    def analyze(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
        val = validities[:n] if validities.shape[0] >= n else np.arange(n, 0, -1, dtype=float)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_winner = 0
        elif margin < 0:
            tally_winner = 1
        else:
            tally_winner = -1
        order = np.argsort(-val, kind='stable')
        ttb_winner = -1
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        return margin, tally_winner, ttb_winner

    info = [analyze(r) for _, r in data.iterrows()]
    margins = np.array([x[0] for x in info], dtype=float)
    tally_w = np.array([x[1] for x in info], dtype=int)
    ttb_w = np.array([x[2] for x in info], dtype=int)
    resp = data['response'].to_numpy().astype(int)

    # Group 1: tally-tie trials where TTB is decisive (parameter-free for Tallying)
    tie_mask = (tally_w == -1) & (ttb_w >= 0)
    # Group 2: trials where tally margin is >= 2 AND TTB points the other way
    dis_mask = (np.abs(margins) >= 2) & (tally_w >= 0) & (ttb_w >= 0) & (tally_w != ttb_w)

    if tie_mask.sum() == 0 and dis_mask.sum() == 0:
        return float('nan')

    if dis_mask.sum() > 0:
        p_tally = float(np.mean(resp[dis_mask] == tally_w[dis_mask]))
    else:
        p_tally = 0.5
    if tie_mask.sum() > 0:
        p_ttb_tie = float(np.mean(resp[tie_mask] == ttb_w[tie_mask]))
    else:
        p_ttb_tie = 0.5

    return p_tally - p_ttb_tie

```

**Observed (real) value:** 0.1744 (var=0.0127)
**Predicted under pi_7:** 0.2300 (var=0.0149)
**Predicted under pi_9:** 0.3256 (var=0.0184)

### Experiment 5
**Design**
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.68, 0.96, 0.52, 0.88, 0.58, 0.78], dtype=float)

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        try:
            a = np.asarray([float(x) for x in row['option_a_ratings']], dtype=float)
            b = np.asarray([float(x) for x in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        k = int(min(a.shape[0], b.shape[0]))
        if k == 0:
            continue
        a = a[:k]
        b = b[:k]
        if V.shape[0] >= k:
            v = V[:k]
        else:
            v = np.concatenate([V, np.full(k - V.shape[0], 0.55)])[:k]

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # keep only unit-tally TIES with at least one discriminating cue
        if a_wins != b_wins or a_wins == 0:
            continue

        w = v ** 2.0
        d = float(np.sum(w * (a - b)))
        # discard tie trials whose validity-weighted margin is itself
        # near zero (no clear "higher-validity side" to predict)
        if abs(d) < 0.15:
            continue

        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        if resp not in (0, 1):
            continue

        chose_a = (resp == 0)
        picked_high_validity = chose_a if d > 0 else (not chose_a)
        hits += 1.0 if picked_high_validity else 0.0
        n += 1.0

    if n == 0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.2856 (var=0.0160)
**Predicted under pi_7:** 0.2778 (var=0.0058)
**Predicted under pi_9:** 0.2956 (var=0.0037)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    v = np.array([0.58, 0.97, 0.53, 0.80, 0.66, 0.90], dtype=float)
    w = v ** 2.0  # AVAI-style amplified weights (gamma=2)

    num = 0.0
    den = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0], w.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        ww = w[:n]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # not a tally tie
        d = float(np.sum(ww * (a - b)))
        if abs(d) < 0.30:
            continue  # tie trials whose weighted margin is negligible
        den += 1.0
        if d > 0:
            num += 1.0 if int(resp) == 0 else 0.0
        else:
            num += 1.0 if int(resp) == 1 else 0.0

    if den == 0:
        return 0.5
    return num / den
```

**Observed (real) value:** 0.1500 (var=0.0194)
**Predicted under pi_7:** 0.1500 (var=0.0133)
**Predicted under pi_9:** 0.1067 (var=0.0070)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    correct = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        nz = d[d != 0.0]
        if nz.size == 0:
            continue
        # dominance trial: every discriminating cue favours the same option
        if not (np.all(nz > 0) or np.all(nz < 0)):
            continue
        dominant_is_a = bool(np.all(nz > 0))
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        correct.append(1.0 if (chose_a == dominant_is_a) else 0.0)

    if len(correct) == 0:
        return 0.5
    return float(np.mean(correct))
```

**Observed (real) value:** 0.7427 (var=0.0173)
**Predicted under pi_7:** 0.7467 (var=0.0048)
**Predicted under pi_9:** 0.7473 (var=0.0050)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # design validities (fixed by the experimental design)
    v_design = np.array([0.95, 0.70, 0.55, 0.95, 0.70, 0.55], dtype=float)
    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k == 0:
            continue
        if v_design.size >= k:
            vv = v_design[:k]
        else:
            vv = np.linspace(0.95, 0.55, k)
        d = a[:k] - b[:k]
        idx = np.nonzero(d != 0.0)[0]
        if idx.size != 2:
            continue
        j1 = int(idx[0])
        j2 = int(idx[1])
        # the two discriminating cues must point to OPPOSITE options
        if d[j1] * d[j2] > 0:
            continue
        # and the LATER cue must be the LESS valid one
        if not (vv[j2] < vv[j1]):
            continue
        late_choice = 0 if d[j2] > 0 else 1  # 0 = A, 1 = B
        try:
            resp = int(r)
        except Exception:
            continue
        n += 1
        if resp == late_choice:
            hits += 1
    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5067 (var=0.0112)
**Predicted under pi_7:** 0.4817 (var=0.0130)
**Predicted under pi_9:** 0.5208 (var=0.0068)

### Experiment 9
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        fav_a = d[disc] > 0.0
        # skip unanimous (dominance) displays: both theories agree there
        if fav_a.all() or (~fav_a).all():
            continue
        # option favoured by the LAST-READ (highest-index) discriminating row
        last_idx = disc[-1]
        late_option = 0 if d[last_idx] > 0.0 else 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hits.append(1.0 if resp == late_option else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4383 (var=0.0023)
**Predicted under pi_7:** 0.4508 (var=0.0038)
**Predicted under pi_9:** 0.4628 (var=0.0033)

### Experiment 10
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment); fallback handles other lengths.
    V_DESIGN = np.array([0.955, 0.545, 0.950, 0.540, 0.945, 0.535], dtype=float)
    LAM = 1.6
    DELTA = 0.9

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if V_DESIGN.size >= n:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.545)])[:n]

        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue

        # salience of each discriminating row: recency x (1 - validity)
        j = disc.astype(float)
        logu = LAM * j + DELTA * np.log(np.clip(1.0 - v[disc], 1e-9, None))
        logu = logu - np.max(logu)
        u = np.exp(logu)
        top = disc[int(np.argmax(u))]
        top_owner_a = bool(d[top] > 0.0)  # True -> the salient row favours A

        n_a = int(np.sum(d[disc] > 0.0))
        n_b = m - n_a

        # include only trials where the row-count tally does NOT strictly
        # agree with the most salient (late / low-validity) discriminating row
        if n_a > n_b:
            majority_a = True
            tie = False
        elif n_b > n_a:
            majority_a = False
            tie = False
        else:
            majority_a = None
            tie = True
        if (not tie) and (majority_a == top_owner_a):
            continue

        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == top_owner_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7697 (var=0.0057)
**Predicted under pi_7:** 0.7686 (var=0.0034)
**Predicted under pi_9:** 0.7766 (var=0.0030)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.958, 0.546, 0.954, 0.542, 0.950, 0.538], dtype=float)
    LAM = 1.5      # reading-order recency gain (shared by both accounts)
    GAM = 0.9      # oddness exponent on (1 - validity) (shared by both accounts)

    hits = 0.0
    tot = 0.0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n == V_DESIGN.size:
            v = V_DESIGN.copy()
        else:
            v = np.linspace(0.95, 0.55, n)
        v = np.clip(v, 0.5, 0.995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        signs = np.sign(diff[disc])
        # keep only CONFLICT displays (both options win at least one row)
        if np.all(signs > 0) or np.all(signs < 0):
            continue

        logw = GAM * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + LAM * disc.astype(float)
        k = int(np.argmax(logw))
        favored_a = signs[k] > 0.0

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        hits += 1.0 if (chose_a == favored_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.8189 (var=0.0074)
**Predicted under pi_7:** 0.7714 (var=0.0022)
**Predicted under pi_9:** 0.7783 (var=0.0025)

### Experiment 12
**Design**
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (fixed by the experiment)
    v_default = np.array([0.53, 0.72, 0.86, 0.93, 0.99], dtype=float)

    def weights(n):
        if n <= v_default.size:
            v = v_default[:n]
        else:
            v = np.concatenate([v_default, np.full(n - v_default.size, 0.99)])
        v = np.clip(v, 0.5, 0.9995)
        logw = 0.9 * np.log(1.0 - v) + 1.5 * np.arange(n, dtype=float)
        logw = logw - np.max(logw)
        return np.exp(logw)

    wcache = {}
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'].values,
                                  data['option_b_ratings'].values,
                                  data['response'].values):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        # keep only CONFLICT displays (both options favoured by at least one expert)
        if nA == 0 or nB == 0:
            continue
        if n not in wcache:
            wcache[n] = weights(n)
        w = wcache[n]
        S = float(np.sum(np.sign(d) * w))
        if S == 0.0:
            continue
        fav = 0 if S > 0 else 1  # response coding: 0 = A, 1 = B
        try:
            r = int(resp)
        except Exception:
            continue
        hits.append(1.0 if r == fav else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4178 (var=0.0051)
**Predicted under pi_7:** 0.3878 (var=0.0029)
**Predicted under pi_9:** 0.4247 (var=0.0030)

### Experiment 13
**Design**
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-fixed printed validities for this experiment
    v = np.array([0.55, 0.90, 0.75, 0.95, 0.68, 0.85], dtype=float)
    n = v.size
    j = np.arange(n, dtype=float)

    # MDSC-NV salience: near-argmax doubt rule (delta=15, lam=0.115*15)
    logu = 15.0 * np.log(np.clip(1.0 - v, 1e-12, None)) + 1.725 * j
    # SSI-DS+ compensatory summation weights (gamma=0.9, lam=1.5)
    w_ssi = np.power(np.clip(1.0 - v, 1e-12, None), 0.9) * np.exp(1.5 * j)

    hits = 0
    tot = 0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        if a.size != n or b.size != n:
            continue
        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        if disc.size == 0:
            continue

        # MDSC-NV predicted winner: verdict of the single most doubtful
        # discriminating expert
        k = disc[int(np.argmax(logu[disc]))]
        pred_mdsc = 0 if d[k] > 0 else 1

        # SSI-DS+ predicted winner: sign of the weighted sum over all
        # discriminating rows (coalitions add, late rows dominate)
        E = float(np.sum(np.sign(d[disc]) * w_ssi[disc]))
        if E == 0.0:
            continue
        pred_ssi = 0 if E > 0 else 1

        if pred_ssi == pred_mdsc:
            continue  # non-diagnostic trial type

        tot += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == pred_mdsc:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.5925 (var=0.0026)
**Predicted under pi_7:** 0.6833 (var=0.0038)
**Predicted under pi_9:** 0.6033 (var=0.0045)

### Experiment 14
**Design**
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.51, 0.72, 0.80, 0.90, 0.95, 0.985], dtype=float)

    GAMMA = 0.90
    LAM = 1.50
    DELTA = 15.0
    LAM_M = 1.725

    hits = 0.0
    tot = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n <= V_DESIGN.size:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.985)])
        v = np.clip(v, 0.5, 0.9995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        if disc.size == 0:
            continue
        signs = np.sign(diff[disc])

        # --- SSI-DS+ side: max-normalised weighted SUM, late/high-validity dominated
        logw = GAMMA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM * disc.astype(float)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        E = float(np.sum(signs * w))
        if abs(E) < 1e-9:
            continue
        ssi_side = 1.0 if E > 0 else -1.0

        # --- MDSC-NV side: single most-doubtful DISCRIMINATING row decides
        sal = DELTA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM_M * disc.astype(float)
        k = int(np.argmax(sal))
        mdsc_side = float(signs[k])
        if mdsc_side == 0.0:
            continue

        # keep only trials where the two accounts point to OPPOSITE options
        if mdsc_side == ssi_side:
            continue

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        ssi_favours_a = (ssi_side > 0)
        hits += 1.0 if (chose_a == ssi_favours_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return 0.5
    return float(hits / tot)
```

**Observed (real) value:** 0.2608 (var=0.0079)
**Predicted under pi_7:** 0.2592 (var=0.0042)
**Predicted under pi_9:** 0.2617 (var=0.0055)

### Experiment 15
**Design**
  A=[1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = []
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n < 5:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        disc = [int(i) for i in np.nonzero(d != 0.0)[0]]
        # require feature 0 (validity .68) to discriminate
        if 0 not in disc:
            continue
        # require the global doubt anchor, feature 3 (validity .52), to be SILENT
        if 3 in disc:
            continue
        fav_a = [i for i in disc if d[i] > 0.0]
        fav_b = [i for i in disc if d[i] < 0.0]
        if d[0] > 0.0:
            lone, other = fav_a, fav_b
            lone_is_a = True
        else:
            lone, other = fav_b, fav_a
            lone_is_a = False
        # exactly one dissenter (feature 0) against a coalition of >=2 more valid experts
        if len(lone) != 1 or len(other) < 2:
            continue
        try:
            resp = int(r)
        except Exception:
            continue
        chose_lone = (resp == 0) if lone_is_a else (resp == 1)
        hits.append(1.0 if chose_lone else 0.0)
    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5460 (var=0.0126)
**Predicted under pi_7:** 0.6410 (var=0.0093)
**Predicted under pi_9:** 0.4660 (var=0.0150)

### Experiment 16
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    V = np.array([0.62, 0.78, 0.86, 0.93, 0.53, 0.97], dtype=float)
    ANCHOR = 4  # the .53 expert (below DTV's doubt trigger)
    hits = 0.0
    tot = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'].tolist(),
                                  data['option_b_ratings'].tolist(),
                                  data['response'].tolist()):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n != 6:
            continue
        diff = a[:6] - b[:6]
        disc = np.nonzero(diff != 0.0)[0]
        m = int(disc.size)
        if m < 3:
            continue
        # the globally most doubtful expert (.53) must be SILENT
        if ANCHOR in set(disc.tolist()):
            continue
        d = diff[disc]
        na = int(np.sum(d > 0.0))
        nb = m - na
        if min(na, nb) != 1 or max(na, nb) < 2:
            continue
        if na == 1:
            lone = int(disc[d > 0.0][0])
            fav = 0  # option A favoured by the lone dissenter
        else:
            lone = int(disc[d < 0.0][0])
            fav = 1
        # the lone dissenter must be the LEAST valid discriminating expert
        if abs(V[lone] - float(np.min(V[disc]))) > 1e-9:
            continue
        tot += 1.0
        if int(resp) == fav:
            hits += 1.0
    if tot < 1.0:
        return 0.5
    return float(hits / tot)
```

**Observed (real) value:** 0.4675 (var=0.0144)
**Predicted under pi_7:** 0.6388 (var=0.0149)
**Predicted under pi_9:** 0.5387 (var=0.0195)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The two new experiments were designed as a head-to-head on exactly the axis that divides the theories: does an opposing coalition of slightly-less-doubtful experts overturn the single most doubtful expert? Both experiments give the same, very loud answer: no. Observed anchor-follow rates were .862 (Exp 1, anchor v=.55, coalitions up to 1-vs-5) and .826 (Exp 2, anchor v=.52, coalitions of 2, 3 and 5 opposing rows, plus 3-vs-3 ties). pi_9's divisively-normalized compensatory summation predicted .478 and .422 \u2014 i.e. it predicted the anchor would be overturned and choice would fall BELOW chance, a directional error of ~.40 in both experiments, and far outside the between-subject variance (.015-.018). This is a decisive falsification of the compensatory/coalition-additive claim for near-chance experts. pi_7's near-argmax single-cue doubt rule got the direction and the flatness right, but still underpredicted by ~.10 in both cases: the real subjects follow the sub-chance-ish anchor even more deterministically than MDSC-NV's load lapse allows.\n\nHowever, pi_7 is not uniformly right. On Exps 15 and 16 \u2014 which are precisely the displays where the GLOBAL doubt anchor (.52 / .53) is SILENT and the lone dissenter is merely the least-valid DISCRIMINATING expert (.68 / .62) opposed by a coalition of two or more \u2014 the observed lone-dissenter rates were .546 and .4675, i.e. at or slightly below chance, while pi_7 predicted .641 and .639 (over-predictions of .10 and .17; pi_9 was closer here, .466/.539). The pattern across all sixteen experiments is therefore sharper than either theory: the one-reason 'adjudicate the odd voice' rule appears to be triggered in an essentially ABSOLUTE way by experts whose printed validity is very close to chance (~.52-.55), and when such an expert speaks, coalitions buy literally nothing (Exps 1, 2, 10, 11, 14 all fit); but when no such near-chance expert discriminates, behaviour reverts to a graded, compensatory, still anti-validity and recency-tilted rule that leaves 1-vs-2/3 splits near chance (Exps 15, 16, 13, 12, where pi_9 was consistently the better fit). Neither theory encodes that gate: pi_7 makes the argmax relative (so the least-valid discriminating row always wins, over-predicting 15/16), and pi_9 makes everything compensatory (catastrophically under-predicting 1/2). The shared experiments (3, 5, 6, 7, 9, 11, 14) are fit about equally well by both and do not separate them.\n\nSince pi_9 is the clear loser overall and fails by ~.40 on the two most diagnostic designs, it should be replaced \u2014 but not by a clone of pi_7, which itself has a systematic, replicated failure mode. The replacement should be the theory that reconciles both sets of results.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_9 with a DOUBT-THRESHOLD-GATED MIXTURE theory (working name: DTG-2R, 'Doubt-Triggered One-Reason with compensatory fallback'). Core claims:\n\n(1) ABSOLUTE doubt trigger, not relative argmax. An expert commands one-reason adjudication only if its PRINTED validity falls below an absolute threshold theta (fit range ~.56-.64), i.e. only experts that are essentially coin-flips are 'newsworthy'. Trigger strength should be a steep but smooth function of (theta - v), e.g. g = sigmoid((theta - v)/tau) with tau ~ .01-.04.\n\n(2) When a sub-threshold expert discriminates, choice follows its verdict with very high probability, independent of how many higher-validity rows oppose it: target ~.85-.90 after lapses on 1-vs-1 through 1-vs-5 and on 2-vs-2 / 3-vs-3 ties (this is what Exps 1 and 2 demand: .862 and .826, flat across coalition size). The load lapse must be MILD in these cells \u2014 noticeably milder than pi_7's current eps0/rho, which drag these to .72-.76.\n\n(3) When NO sub-threshold expert discriminates, the trigger is off and evidence is compensatory: a divisively-normalized, anti-validity ((1-v)^gamma) and recency-tilted (exp(lam*j)) weighted sum over discriminating rows, so that a lone mid-doubt dissenter (v~.62-.68) opposed by a coalition of 2-3 sits at .45-.56 (Exps 15, 16) and conflict-display metrics land near .42-.60 (Exps 12, 13).\n\n(4) Silent sub-threshold experts do NOT need a special 'capture' mechanism: the near-chance behaviour in silent-anchor displays follows automatically from (3). Drop null-verdict capture and the doubt-gradient leak, or keep at most a small residual \u2014 they are now redundant and were the source of pi_7's over-prediction in Exps 15/16.\n\n(5) Preserve the two features already validated across the corpus: the label-degeneracy collapse to an unweighted tally when two printed validities are within .002 (Exps 10, 11 fit at ~.77-.82 via recency among near-equal labels), and a load-dependent lapse that keeps unanimous 6-0 dominance at ~.74 (Exp 7) rather than ceiling.\n\nThe new theory differs from pi_7 by making the one-reason rule conditional on an ABSOLUTE near-chance validity rather than on being the relative minimum among discriminating rows, and by being compensatory when no near-chance expert speaks; it differs from pi_9 by denying that coalitions can ever overturn a genuinely near-chance dissenter. Parameter ranges should be chosen so that, at mid-box, the model reproduces: Exp1 ~.86, Exp2 ~.83, Exp15 ~.54, Exp16 ~.47, Exp7 ~.74, Exp13 ~.59, Exp12 ~.42, Exp14 ~.26, Exp6 ~.15."
}
```

## Usage

```json
{
  "input_tokens": 41094,
  "output_tokens": 4083
}
```
