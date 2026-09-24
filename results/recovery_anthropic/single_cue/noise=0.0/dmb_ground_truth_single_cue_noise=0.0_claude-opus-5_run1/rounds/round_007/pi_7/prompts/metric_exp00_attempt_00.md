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



## ADVOCATED THEORY
**Description:** **MDSC-NV (sharpened) — Maximum-Doubt Single-Cue selection with Null-Verdict capture.**

Four claims; the mechanism family is unchanged from the accepted base, but one quantitative commitment is sharpened.

**(1) One-reason choice by MAXIMUM DOUBT (no summation).** On each trial the subject settles A-vs-B with the verdict of a SINGLE expert. Which expert is consulted is decided by *doubt*: a disagreement voiced by a near-chance expert (v≈.53) is a strange, newsworthy fact that has to be adjudicated, whereas a disagreement voiced by a near-certain expert (v≈.97) is discounted as a mere echo of the impression the subject already has. Salience is log u_j = delta*log(1-v_j) + lam*j and the consulted row is drawn from softmax(u). **The doubt gain is LARGER than previously assumed (delta ≈ 13-17, not 8-12): the doubt rule is essentially argmax, not a graded share.** This is the sharpened commitment. It matters wherever two moderately-doubtful rows are within ~0.4 log-validity of one another (e.g. v=.55 vs v=.70 rows two columns apart): a share-based rule lets the second-most-doubtful row steal ~18% of the verdict and blunts the choice, whereas the near-argmax rule leaves the display fully decided by the single most doubtful row. Coalitions still buy nothing: five agreeing rows never outvote one more-doubtful row, which is why choice is systematically anti-validity in every design where the printed labels are usable.

**(2) Reading position is a WEAK tie-breaker.** lam = pos_ratio*delta with pos_ratio ≈ 0.09-0.14 — an order of magnitude below the doubt term's dynamic range. Position therefore only arbitrates between rows whose stated validities are (nearly) *equal* — exactly the .955/.950/.945 and .958/.954/.950 designs, where it makes the last low-validity row win. Whenever validities genuinely differ, doubt beats recency: a low-validity EARLY column defeats a late .93/.99 column. Because delta is raised and pos_ratio held fixed, the *ratio* of doubt-range to positional gain is preserved, so all orderings in the corpus are unchanged.

**(3) NULL-VERDICT CAPTURE.** Doubt is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the most doubtful expert **in the whole display**, before the subject checks whether that expert actually discriminates. With probability q (≈ .41, slightly higher than previously estimated) the subject simply goes with whoever captured attention; if that expert gave both products the same rating it returns no verdict, the subject has nothing to adjudicate and guesses. With probability 1-q the subject re-targets deliberately, restricting the doubt competition to the rows that differ. Sharp prediction: *holding the discriminating rows fixed, adding agreement on a MORE doubtful (lower-validity) row pushes choice toward chance*, whereas adding agreement on a high-validity row does nothing. Displays in which only two high/mid-validity rows disagree while the display's most doubtful expert stays silent are therefore near chance — no compensatory or one-reason model predicts that non-discriminating rows matter at all.

**(4) LABEL-DEGENERACY GATE (all-or-none) and LOAD LAPSE.** The doubt rule presupposes that each printed validity picks out one expert. If any label is printed twice (tolerance .002 — it must NOT fire for the .004/.005-gap designs) the experts are interchangeable, the doubt ordering is unavailable, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). Independently, the more rows disagree the more often the subject loses the thread: p(guess) = 1-(1-eps0)*rho^(m-2), which compresses conflict displays toward chance and keeps even unanimous 6-row displays well below ceiling. Individual subjects differ mainly in how leaky (eps0, rho) and how capture-prone (q) they are, not in the doubt rule itself.

No feedback is given, so the rule is stationary across the block.

**Parameters:**
- doubt_gain: [13.0, 17.0]
- pos_ratio: [0.09, 0.14]
- capture: [0.33, 0.49]
- eps0: [0.19, 0.29]
- rho: [0.77, 0.85]
- tally_kappa: [1.00, 1.40]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **NAVA-HLp — Normalized Anti-Validity Accumulation with a HALF-LOUDNESS load lapse and a PARTIAL unanimity discount.**

Six claims. The evidence rule is unchanged from the accepted base: graded, compensatory, anti-validity, divisively normalized, with no one-reason step, no argmax, no verdict capture and no doubt trigger. Only the *price charged for holding rival voices* is revised, and only partially.

**(1) Every dissenting row speaks, and doubt is what makes it loud.** A printed validity close to chance makes a disagreement *newsworthy* (it demands adjudication); a printed validity close to certainty makes the same disagreement a mere echo of the impression the subject already has. Each row carries weight w_j = (1 − v_j)^γ · exp(λ·j): doubt raised to a moderate power γ ≈ 1.9–2.5, times a modest reading-position gain λ ≈ 0.28–0.45. All discriminating rows contribute simultaneously, so coalitions matter monotonically, while the global direction of choice stays systematically anti-validity in every design whose labels are usable.

**(2) Divisive normalization relative to the loudest dissent.** Weights enter only as ratios to the loudest discriminating voice, ŵ_j = w_j / max_{k∈disc} w_k, and E = Σ_{j∈disc} s_j ŵ_j / (σ + Σ_{j∈disc} ŵ_j), p(A) = σ(βE). Absolute scale is discarded (dominance displays decided only by high-validity experts are still resolved confidently) while the denominator grows with each speaking row, so a coalition erodes a doubtful dissenter's lead toward, but rarely far past, chance.

**(3) The displacement leak is a DOUBT GRADIENT.** Doubt is a property of the expert, not of the disagreement, so the most doubtful expert *in the whole display* competes for attention before the subject checks whether it discriminates. What costs accuracy is the *doubt gradient* between the loudest silent expert and the loudest speaking one: attention is pulled off the adjudication only when the silent expert is appreciably MORE doubtful than the expert currently carrying the argument. Formally d = clip((v_loudest-disc − v_loudest-silent)/Δ, 0, 1) · w_silent,max/(w_silent,max + w_disc,max) with fixed Δ = .05, and the leak multiplies the retention term by (1 − κ·d), κ ≈ .40–.95. Silent rows never contribute evidence; the term is purely subtractive and can only shrink |p − .5|.

**(4) Losing the thread is charged HALF PER ROW and HALF PER LOUD VOICE.** What has to be held in mind while the comparison is run is neither the raw number of disagreeing rows (which over-charges a display in which one towering doubtful voice drowns out five whispers) nor purely the effective number of loud voices (which under-charges a long unanimous list that still has to be scanned). Both costs are real and roughly equally weighted: L = ½·(m − 2) + ½·c·(Σ_{j∈disc} ŵ_j − 1), and p(guess) = 1 − (1−ε₀)·ρ^max(L,0)·(1 − κ·d) with ρ ≈ .64–.80 and floor ε₀ ≈ .07–.15.

**(5) NEW — unanimity is CHEAPER BUT NOT FREE.** The loud-voice component of the load is an *adjudication* cost: it is the price of holding rival loud voices against one another. When every speaking row agrees there is no adjudication, but the rows still have to be scanned and their loudness still occupies the comparison, so the cost is discounted rather than waived: c = 1 when both options are favoured by at least one row, and c = c₀ ≈ .45–.75 when the display is unanimous. The row-scan half, ½·(m − 2), is charged in full regardless, so a 6-vs-0 dominance display remains well below ceiling while a 3-vs-3 display of the same size is markedly noisier. This is the model's sharp prediction about dominance: accuracy on unanimous displays falls with list length but strictly more slowly than on conflict displays of matched length and matched weight concentration.

**(6) Degenerate printed labels collapse the weighting.** If any two printed validities are within .002 the experts cannot be individuated; all weights become equal and the same normalized rule reduces to an unweighted tally, E = (n_A − n_B)/(σ + m) with the same β and σ — exact chance on 1-1 splits, no anti-validity and no recency signature. The displacement leak is switched off, because no row is 'the doubtful one'.

No feedback is given, so the rule is stationary across the block.

**Parameters:**
- gamma: [1.9, 2.5]
- lam: [0.28, 0.45]
- sigma: [0.12, 0.30]
- beta: [2.8, 3.8]
- eps0: [0.07, 0.15]
- rho: [0.64, 0.80]
- kappa: [0.40, 0.95]
- c0: [0.48, 0.75]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    hits = 0.0\n    tot = 0.0\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n    ANCHOR = 3  # column with the lowest printed validity (0.55)\n    for a_raw, b_raw, r in zip(a_col, b_col, r_col):\n        a = [int(x) for x in list(a_raw)]\n        b = [int(x) for x in list(b_raw)]\n        k = min(len(a), len(b))\n        if k <= ANCHOR:\n            continue\n        if a[ANCHOR] == b[ANCHOR]:\n            continue  # anchor silent -> not a coalition test\n        n_a = 0\n        n_b = 0\n        for i in range(k):\n            if a[i] > b[i]:\n                n_a += 1\n            elif b[i] > a[i]:\n                n_b += 1\n        anchor_favors_a = a[ANCHOR] > b[ANCHOR]\n        opposing = n_b if anchor_favors_a else n_a\n        if opposing < 1:\n            continue  # pure dominance for the anchor side -> no coalition\n        try:\n            resp = int(r)\n        except Exception:\n            continue\n        chose_a = (resp == 0)\n        if chose_a == anchor_favors_a:\n            hits += 1.0\n        tot += 1.0\n    if tot == 0.0:\n        return float('nan')\n    return float(hits / tot)",
  "rationale": "The design's decisive contrast is whether the option backed by the single most doubtful expert (column 3, validity .55) survives when an opposing coalition of slightly-less-doubtful experts (c2 .62, c4 .70, c5 .74) speaks against it. I select exactly those trials in which the .55 expert discriminates AND at least one row favours the other option (base structures 2, 3, 4, 5 and 8: 1-vs-1, 1-vs-2, 1-vs-3, 2-vs-2, 1-vs-5), and score the proportion of choices that follow the .55 anchor. MDSC-NV (near-argmax single-cue on doubt, delta 13-17) says coalitions buy essentially nothing: the anchor's core probability stays .96-.99 in every one of these cells and only the load lapse pulls the observed rate down, giving an expected anchor-follow rate around .72-.80. NAVA-HLp's divisive compensatory summation gives the coalition rows normalized weights of ~.49-.60 each, so two or three of them sum past 1 and flip the sign of E, predicting anchor-follow rates around .41-.50 (i.e. below chance) on most of these cells and ~.48 pooled. The predicted gap is therefore roughly .27-.30 with the two theories straddling chance in opposite directions. Because the metric aggregates ~30 unique-trial repetitions per subject (5 base structures x 2 mirrors x 6 reps = 60 trials) the binomial noise per subject is ~.06 SD and the only additional between-subject spread comes from lapse parameters (a few points), so the contrast should be both large and tight. The metric is well-defined on a single subject's slice and on the pooled frame alike, and it is not trivially true of MDSC-NV: any compensatory or tally-like process (including the competitor) would put it at or below .5."
}
```

## Usage

```json
{
  "input_tokens": 13905,
  "output_tokens": 3603
}
```
