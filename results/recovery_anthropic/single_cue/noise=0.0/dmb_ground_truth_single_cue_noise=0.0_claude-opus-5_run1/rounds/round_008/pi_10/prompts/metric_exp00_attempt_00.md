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
**Validities (n_features=6):** [0.64, 0.96, 0.72, 0.9, 0.52, 0.68]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 2: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1]
  trial 3: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 4: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  trial 7: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 9: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 10: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 13: A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  trial 14: A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  trial 15: A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 16: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 17: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 18: A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 19: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 20: A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 21: A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 22: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 23: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 24: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** GOAL: separate DTG-2R/OC-sharp (advocated) from MDSC-NV (competitor). Both are anti-validity accounts with a weak recency tilt and a load leak, and this design is deliberately built so they AGREE about which row carries the verdict in every display; they differ only on three mechanisms, each of which is isolated by a matched-load contrast.

VALIDITIES [0.64, 0.96, 0.72, 0.90, 0.52, 0.68] at columns c0..c5. Engineering: (i) EXACTLY ONE expert, c4 = .52, lies below DTG's absolute sharp gate (theta .558-.588, tau .006-.013): g(.52) = .95-1.00, whereas g(.64) <= .018, g(.68) <= .001, g(.72) <= 4e-5. So DTG runs its one-reason limb iff c4 discriminates and its compensatory limb otherwise. (ii) c4 is ALSO MDSC's global doubt anchor at every corner of its box: score_j = log(1-v_j) + pos_ratio*j gives c4 (-0.40 at pos_ratio .09, -0.20 at .14) ahead of the runner-up c5 (.68, last column: -0.69 / -0.44) by .24-.29 log-(1-v) units, i.e. 3.2-4.9 delta units (share .96-.99); with c4 silent the secondary MDSC anchor is unambiguously c5 (ahead of c0 by .33-.58 units). MDSC therefore reduces to 'whoever wins c4; if c4 is silent, whoever wins c5, with capture q dragging the trial toward chance'. (iii) Validity does not track column index, so 'read the first/last row' cannot masquerade as either model; the minimum label gap (.04) is 20x both degeneracy tolerances. (iv) DTG's compensatory weights (gamma 1.9, lam .11) over the non-anchor rows are c5=.199, c0=.144, c2=.111, c3=.018, c1=.002, so c0+c2 (.255) OVERTURNS a lone c5 dissenter: coalitions of merely mediocre experts beat the most doubtful discriminating row for DTG but buy nothing for MDSC.

12 base structures, each with its exact A/B mirror (24 unique pairs, K=4 -> 96 trials), perfectly key-balanced. Silent rows carry matched 1s AND matched 0s so displays are dense. Predictions use DTG (theta=.573, tau=.010, mu=.7, gamma=1.9, lam=.11, sigma=.55, beta=2.2, eps_t=.28, rho_t=.99, rho_tc=.74, eps0=.20, rho_m=.885, rho_c=.82) and MDSC (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81).

BLOCK 1 - CROWN JEWEL: FLAT-vs-DECLINING ANCHOR LADDER (bases 1,2,3 = pairs 1-6). A always wins c4 (.52) while B's coalition of MORE valid experts grows 0 -> 2 -> 5 rows (m = 1, 3, 6). DTG stays in the cheap one-reason limb with k_conf <= 1, so rho_tc never fires and rho_t~.99 makes extra rows nearly free: .860, .856, .846 (FLAT within 2 points). MDSC pays rho^(m-2) on every added row: .880, .808, .664 (a 22-point monotone decline). Same decisive row, same verdict, matched direction: flat vs steep is parameter-robust. Base 1 also pins the shared lapse baseline (~.86-.88 for both).

BLOCK 2 - OPPOSITION-COST vs RAW-LOAD LEAK AT MATCHED m (bases 3 vs 4, 6 vs 5). Base 4 is a 3-vs-3 split at m=6 with the anchor on A's side; base 3 is 1-vs-5 at the same m=6. DTG charges rho_tc per minority row beyond the first, so it must predict a ~16-point DROP (.846 -> .689) purely from the balance of the split, whereas MDSC's leak sees only m and must call them equal (.664 vs .653). The same test at m=4: base 6 (1-vs-3, .853) versus base 5 (2-vs-2, .761) for DTG, against .739/.749 (no ordering) for MDSC. This dissociates a coalition-driven from a length-driven leak with display length held exactly constant.

BLOCK 3 - CROWN JEWEL: THE ABSOLUTE THRESHOLD / COMPENSATORY REVERSAL (bases 7,8,9 = pairs 13-18, read against bases 2 and 6). With the .52 anchor SILENT, A wins c5 (.68 - MDSC's secondary argmax, but ORDINARY evidence for DTG) and B's coalition grows: 1-vs-1 (c0), 1-vs-2 (c0,c2), 1-vs-3 (c0,c2,c3). DTG's compensatory limb sums max-normalised anti-validity weights and crashes through chance: .544, .469, .465. MDSC's near-argmax rule keeps the c5-holder above chance throughout, with only its load leak moving: .722, .683, .648. Base 8 versus base 2 is the sharpest single comparison: two structurally IDENTICAL 1-vs-2 displays against the same coalition (c0,c2) in which only the identity of the lone dissenter changes, .52 -> .68. DTG must swing 39 points ACROSS chance (.856 -> .469); MDSC must swing only ~12 points and stay above chance in both cells (.808 -> .683).

BLOCK 4 - COMPENSATORY LOAD vs CHEAP SCAN ON UNANIMOUS DISPLAYS (bases 10, 11, 12 = pairs 19-24). Base 12 is a unanimous 2-0 with the anchor speaking (DTG .860 / MDSC .880, shared baseline). Base 10 is a unanimous 5-0 with the anchor SILENT and base 11 the unanimous 6-0 obtained by letting the anchor speak too - the two displays differ in exactly one row. DTG says ADDING the sixth agreeing row makes the display markedly EASIER (.698 -> .846, +15 points), because the .52 row flips the trial out of the effortful, rho_m-degraded compensatory integration into the near-free one-reason limb; MDSC says the sixth row barely helps (.622 -> .664, +4) because capture is only partially satisfied while load increases. DTG also predicts long unanimous displays sit confidently at ~.85 when the anchor speaks, well above MDSC's ~.66.

INTERPRETATION: if accuracy is essentially FLAT in display length whenever the .52 expert speaks (~.85 from 1-vs-0 through 1-vs-5 and from 2-0 through 6-0), drops by ~15 points when the split is BALANCED at the same m, and collapses to or below chance as soon as the lone dissenter is .68 instead of .52 (bases 8, 9), the absolute sharp doubt gate with an opposition-costed one-reason limb and a load-costed compensatory limb is confirmed and MDSC-NV is falsified. If instead accuracy declines steadily with the raw number of discriminating rows even when the verdict-bearing row is unchanged (.88 -> .66), balance of the split makes no difference at matched m, and the option holding the most doubtful DISCRIMINATING row is still chosen ~.65-.72 when two or three more valid experts oppose it, MDSC-NV's near-argmax relative doubt rule with null-verdict capture wins.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** DTG-2R/OC-sharp (load-graded compensatory limb) — Doubt-Threshold-Gated one-reason choice with compensatory fallback, where the doubt gate is an ABSOLUTE and SHARP near-chance criterion, the one-reason limb is degraded by OPPOSITION rather than by raw display length, and the compensatory limb is degraded by DISPLAY LOAD more steeply than previously assumed. (1) One-reason adjudication is commanded only by an expert whose PRINTED validity is essentially a coin flip. The trigger is g_j = sigmoid((theta - v_j)/tau) evaluated only over rows that actually discriminate (P_trig = max_{j in disc} g_j), with theta ~ .558-.588 and tau ~ .006-.013. An expert at .52-.55 is 'newsworthy' (g >= .9) while an expert at .62-.68 is NOT (g <= .02): a merely mediocre expert is ordinary evidence, not a scandal demanding adjudication. Silent near-chance experts are irrelevant, so no null-verdict capture and no doubt-gradient leak are needed. (2) When a genuinely sub-threshold expert speaks, its verdict decides the trial essentially independently of how MANY higher-validity rows oppose it; among discriminating rows the consulted verdict is drawn from softmax(log g_j + mu*j), so the recency tilt mu only arbitrates BETWEEN near-chance rows. (3) Because a single row is consulted, merely scanning extra rows costs the one-reason limb almost nothing (rho_t ~ .985-1.0), but holding a genuine RIVAL COALITION does cost it: each minority row beyond the first charges rho_tc (~.62-.86). Hence a near-chance dissenter standing ALONE against 1-5 opposing rows is followed at ~.82-.86 flat in coalition size (Exps 15, 16), whereas BALANCED splits drift toward chance. (4) When no near-chance expert discriminates — including displays whose least valid dissenter is only mid-doubt (v ~ .58-.68) — evidence is compensatory: anti-validity, weakly recency-tilted weights w_j = (1-v_j)^gamma * exp(lam*j), divisively normalised to the loudest dissent, E = sum(s_j*what_j)/(sigma + sum(what_j)), p = sigmoid(beta*E). NEW, SHARPENED CLAIM: the compensatory rule is a genuinely effortful integration whose thread is lost quickly as the display lengthens — every discriminating row beyond the second multiplies the retained evidence by rho_m ~ .84-.93, appreciably steeper than the one-reason limb's near-free scan. This asymmetry is the theory's signature: one-reason choice is cheap to run on long displays and expensive only under opposition, whereas compensatory integration is expensive in sheer list length. Consequently long unanimous displays stay confident but clearly below ceiling (~.74-.76), and long conflict displays settle appreciably closer to chance than short ones, while short displays (m=2-3) are essentially unaffected. (5) If any two printed validities are within .002 the experts cannot be individuated, the doubt gate has no referent, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.

**Parameters:**
- theta: [0.558, 0.588]
- tau: [0.006, 0.013]
- mu: [0.45, 0.95]
- gamma: [1.5, 2.3]
- lam: [0.0, 0.22]
- sigma: [0.40, 0.72]
- beta: [1.8, 2.6]
- eps_t: [0.20, 0.36]
- rho_t: [0.985, 1.00]
- rho_tc: [0.62, 0.86]
- eps0: [0.13, 0.27]
- rho_m: [0.84, 0.93]
- rho_c: [0.74, 0.90]
- tally_kappa: [0.6, 1.3]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # DTG-2R/OC: Doubt-Threshold-Gated one-reason choice with compensatory fallback;
    # one-reason limb degraded by OPPOSITION (rho_tc) rather than by raw row count,
    # compensatory limb degraded by DISPLAY LOAD (rho_m)
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

    # ---------- printed validities ----------
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
    theta = float(parameters['theta'])
    tau = float(max(float(parameters['tau']), 1e-4))
    mu = float(parameters['mu'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
    beta = float(parameters['beta'])
    eps_t = float(np.clip(float(parameters['eps_t']), 0.0, 1.0))
    rho_t = float(np.clip(float(parameters['rho_t']), 1e-6, 1.0))
    rho_tc = float(np.clip(float(parameters['rho_tc']), 1e-6, 1.0))
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho_m = float(np.clip(float(parameters['rho_m']), 1e-6, 1.0))
    rho_c = float(np.clip(float(parameters['rho_c']), 1e-6, 1.0))
    kappa = float(parameters['tally_kappa'])

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
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    k_conf = int(min(n_a, n_b))

    # ---------- lapses ----------
    expo = max(m - 2, 0)
    keep_c = (1.0 - eps0) * (rho_m ** expo) * (rho_c ** k_conf)
    keep_c = float(min(max(keep_c, 0.0), 1.0))
    # one-reason limb: scanning extra rows is nearly free, holding a RIVAL
    # coalition is not -> charge rho_tc per minority row beyond the first
    keep_t = (1.0 - eps_t) * (rho_t ** expo) * (rho_tc ** max(k_conf - 1, 0))
    keep_t = float(min(max(keep_t, 0.0), 1.0))

    # ---------- label-degeneracy gate ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        p_core = sig(kappa * (n_a - n_b))
        p_a = 0.5 + keep_c * (p_core - 0.5)
        p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
        p = np.array([p_a, 1.0 - p_a], dtype=float)
        return p / p.sum()

    # ---------- absolute doubt trigger over DISCRIMINATING rows ----------
    zz = np.clip((v - theta) / tau, -60.0, 60.0)
    gtrig = 1.0 / (1.0 + np.exp(zz))          # = sigmoid((theta - v)/tau)
    gd = gtrig[disc]
    P_trig = float(np.clip(np.max(gd), 0.0, 1.0))

    # verdict of the consulted (near-chance) expert; recency tilt among them
    logs = np.log(np.clip(gd, 1e-300, None)) + mu * disc.astype(float)
    logs = logs - np.max(logs)
    s = np.exp(np.clip(logs, -700.0, 0.0))
    ssum = float(np.sum(s))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        s = np.full(m, 1.0 / m)
    else:
        s = s / ssum
    p_or_a = float(np.sum(s[signs > 0.0]))
    p_t = 0.5 + keep_t * (p_or_a - 0.5)

    # ---------- compensatory fallback (divisively normalised anti-validity) ----------
    logw = gamma * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)
    ld = logw[disc]
    ld = ld - np.max(ld)
    w = np.exp(np.clip(ld, -700.0, 0.0))
    wsum = float(np.sum(w))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        w = np.ones(m, dtype=float)
        wsum = float(m)
    E = float(np.sum(signs * w)) / (sigma + wsum)
    p_comp = sig(beta * E)
    p_c = 0.5 + keep_c * (p_comp - 0.5)

    # ---------- doubt-threshold-gated mixture ----------
    p_a = P_trig * p_t + (1.0 - P_trig) * p_c
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    return p / p.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    ANCHOR = 4   # printed validity .52  (the only sub-.558 expert)\n    SECOND = 5   # printed validity .68, last column (secondary doubt row)\n\n    t_num = 0.0; t_den = 0.0   # anchor-follow rate on LONG, low-opposition displays\n    c_num = 0.0; c_den = 0.0   # c5-follow rate when the .52 anchor is SILENT\n\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_, b_, r_ in zip(a_col, b_col, r_col):\n        try:\n            a = np.asarray(a_, dtype=float).ravel()\n            b = np.asarray(b_, dtype=float).ravel()\n            r = int(r_)\n        except Exception:\n            continue\n        if a.size < 6 or b.size < 6:\n            continue\n        d = a[:6] - b[:6]\n        disc = np.nonzero(d != 0.0)[0]\n        m = int(disc.size)\n        if m == 0:\n            continue\n        nA = int(np.sum(d[disc] > 0.0))\n        nB = m - nA\n        kconf = min(nA, nB)\n        chose_a = (r == 0)\n\n        anchor_speaks = bool(np.any(disc == ANCHOR))\n        second_speaks = bool(np.any(disc == SECOND))\n\n        if anchor_speaks and m >= 4 and kconf <= 1:\n            # long display, anchor either alone against a coalition or with everyone\n            fav_a = (d[ANCHOR] > 0.0)\n            t_den += 1.0\n            if chose_a == fav_a:\n                t_num += 1.0\n        elif (not anchor_speaks) and second_speaks and kconf == 1:\n            # anchor silent; is the .68 row the LONE dissenter on its side?\n            fav_a_5 = (d[SECOND] > 0.0)\n            side_count = nA if fav_a_5 else nB\n            if side_count == 1:\n                c_den += 1.0\n                if chose_a == fav_a_5:\n                    c_num += 1.0\n\n    t_rate = (t_num / t_den) if t_den > 0 else 0.5\n    c_rate = (c_num / c_den) if c_den > 0 else 0.5\n    return float(t_rate - c_rate)",
  "rationale": "The two theories agree on WHICH row carries the verdict in every display of this design; they disagree about (i) whether long displays are cheap when the .52 anchor speaks, and (ii) whether a merely mid-doubt lone dissenter (.68) still dictates the choice when the anchor is silent.\n\nThe metric is a within-subject difference of two rates computed from the stimulus structure alone (no theory-specific fitting):\n\nT = P(choose the option favoured by the .52 anchor) on LONG, low-opposition displays (anchor discriminates, m>=4 discriminating rows, at most one row in the minority) -> trials 3, 6, 11, 15, 18, 23 (24 trials/subject).\nC = P(choose the option favoured by the .68 last-column row) when the .52 anchor is SILENT and the .68 row stands ALONE against a coalition of 1-3 more valid rows -> trials 7, 8, 9, 19, 20, 21 (24 trials/subject).\n\nDTG-2R/OC-sharp: the one-reason limb is charged for opposition, not for list length, so T stays high (~.85) even at m=6; and with no sub-threshold expert discriminating, the compensatory limb (anti-validity weights normalised to the loudest dissent, load-degraded) pulls C to or below chance (~.47-.52). Predicted T - C ~ +0.35.\n\nMDSC-NV: the load leak rho^(m-2) drags long displays down (T ~ .66-.74) while near-argmax doubt keeps the .68-holder chosen well above chance (C ~ .65-.72). Predicted T - C ~ 0.00.\n\nBoth components use 24 trials per subject, so binomial noise is modest, and both theories' predicted rates are driven by parameters with narrow ranges, keeping between-subject variance small relative to the ~0.35 mean gap. A near-zero value falsifies the absolute sharp doubt gate + opposition-costed one-reason limb; a large positive value falsifies MDSC-NV."
}
```

## Usage

```json
{
  "input_tokens": 14652,
  "output_tokens": 10775
}
```
