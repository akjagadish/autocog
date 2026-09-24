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



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    hits = 0.0\n    n = 0.0\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'],\n                                  data['option_b_ratings'],\n                                  data['response']):\n        a = np.asarray(list(a_raw), dtype=float).ravel()\n        b = np.asarray(list(b_raw), dtype=float).ravel()\n        k = int(min(a.size, b.size))\n        if k < 6:\n            continue\n        # the near-chance anchor expert (column 0, validity .52) must discriminate\n        if a[0] == b[0]:\n            continue\n        anchor_side = 0 if a[0] > b[0] else 1  # 0 == option A, 1 == option B\n        # count late, mid-doubt rows (columns 3,4,5) that oppose the anchor\n        opp = 0\n        for j in (3, 4, 5):\n            if a[j] == b[j]:\n                continue\n            fav = 0 if a[j] > b[j] else 1\n            if fav != anchor_side:\n                opp += 1\n        if opp < 2:\n            continue\n        n += 1.0\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        if r == anchor_side:\n            hits += 1.0\n\n    if n == 0.0:\n        return 0.5\n    return float(hits / n)\n",
  "rationale": "The decisive axis between NAVA-HLp (compensatory divisive summation) and MDSC-NV (near-argmax single-cue doubt selection) is whether the WEIGHT of the opposing rows enters the decision at all. I isolate exactly the coalition displays in which the most-doubtful expert (column 0, v=.52 \\u2014 MDSC's doubt argmax at every corner of its parameter box) discriminates and is opposed by at least two of the slightly-less-doubtful late rows (c3=.78, c4=.80, c5=.83). This selection picks out bases 6, 7, 8, 9 and 12 of the design (trials 11/12, 13/14, 15/16, 17/18, 23/24), i.e. 10 unique pairs \\u00d7 4 reps = 40 trials per subject \\u2014 enough to keep binomial noise per subject small (~.006 variance).\n\nThe metric is the proportion of those trials on which the subject picks the option endorsed by the .52 anchor. Hand-computing both generative models at mid-box parameters: NAVA gives .452, .404, .427, .472, .439 across the five bases (mean \\u2248 .44 \\u2014 BELOW chance, because \\u03a3\\u0175 of the opposing coalition exceeds 1 and normalization lets it overturn the anchor), whereas MDSC gives .806, .747, .662, .747, .662 (mean \\u2248 .73 \\u2014 always ABOVE chance, because only the identity of the doubt argmax matters and coalitions buy nothing; only the load lapse pulls it toward .5). The gap is ~.29 with a directional split across chance, so it cannot be mimicked by differences in lapse rate, side bias, or general anti-validity tendency: both theories are anti-validity, both charge load, and mirrored A/B versions of every base are included so side bias cancels exactly. Per-subject parameter heterogeneity moves each theory only within its own corner range (NAVA .33\\u2013.61, MDSC .66\\u2013.84), leaving the two distributions essentially non-overlapping at N=25."
}
```

## Usage

```json
{
  "input_tokens": 14806,
  "output_tokens": 6906
}
```
