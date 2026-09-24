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
**Validities (n_features=7):** [0.635, 0.93, 0.615, 0.645, 0.97, 0.52, 0.625]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1]
  trial 9: A=[1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  trial 11: A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 12: A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1]
  trial 13: A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 14: A=[1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1, 0]
  trial 15: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0]
  trial 17: A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1]
  trial 18: A=[1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 19: A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1]
  trial 20: A=[0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 21: A=[1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0]
  trial 22: A=[0, 1, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 0]
  trial 23: A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1, 1]
  trial 24: A=[0, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]

**Rationale:** GOAL: separate DTG-2R/OC-sharp (advocated) from SSG-6 (competitor). The two accounts share an absolute, razor-sharp near-chance trigger, an anti-validity divisively-normalised integration limb with a weak recency tilt, and a nearly length-free adjudication limb, so they always AGREE about WHICH row carries the verdict here. They differ on exactly two mechanisms, and the design isolates both while holding the decisive row, its verdict, and (in the key ladders) the conflict count k_conf = 1 exactly constant: (A) SSG-6 adds a RELATIVE salience-share gate (share = sum g_j*o_j / sum o_j with o_j = (1-v_j)^gamma_s, c_share ~ .72, floor ~ .40), so adjudication is only partly available when several mildly-doubtful (but non-scandalous) experts own a chunk of the display's oddness; DTG's gate is purely ABSOLUTE (P_trig = max_j g_j), so the identity of the OPPOSING rows is irrelevant as long as the near-chance expert speaks. (B) SSG-6 sets pi_one = 0 whenever the discriminating rows are UNANIMOUS ('nothing to adjudicate'), forcing dominance displays through its expensive, frequently-abandoned integration limb (eps0 .26-.34), whereas DTG runs its cheap one-reason limb on unanimous displays too.

VALIDITIES [0.635, 0.93, 0.615, 0.645, 0.97, 0.52, 0.625] at columns c0..c6. Engineering: (i) EXACTLY ONE expert, c5 = .52, is sub-threshold for both models (g ~ 1.00 over both boxes); the four mids (.615,.625,.635,.645) are non-scandalous for both (g <= .11 even at DTG's worst corner theta=.588/tau=.013, <= .03 for SSG) yet carry appreciable ODDNESS relative to the anchor ((1-v) ratios .74-.80, i.e. o-shares .10-.19 each at gamma_s = 7.5), while the two highs (.93,.97) have utterly negligible oddness (< 1e-4), so share ~ 1 when they are the opposition. (ii) The anchor sits LATE (c5, j=5) so both models' positional gains (mu) REINFORCE rather than threaten its selection in the adjudication softmax: even at DTG's worst corner (theta=.588, tau=.013, mu=.95) the anchor still owns ~.87 of the softmax against all four mids. (iii) Validity does not track column index (a high sits last), so 'read the first/last row' cannot masquerade as either model; minimum label gap is .01, five times both degeneracy tolerances. (iv) The integration limbs of the two models use nearly identical weights (gamma 1.9 vs gamma_w 1.45), so they never disagree in DIRECTION - the whole contrast is quantitative.

12 base structures, each with its exact A/B mirror (24 unique pairs, K = 4 -> 96 trials), perfectly key-balanced. Silent rows carry matched 1s AND matched 0s so every display is dense and cannot be solved by counting 1s. Predictions use DTG (theta=.573, tau=.010, mu=.70, gamma=1.9, lam=.11, sigma=.55, beta=2.2, eps_t=.28, rho_t=.99, rho_tc=.74, eps0=.20, rho_m=.885, rho_c=.82) and SSG-6 (gamma_s=7.5, v_crit=.581, w_h=.0095, c_share=.72, w_share=.05, floor=.40, mu=.5, eps_t=.25, rho_opp=.86, gamma_w=1.45, lam=.135, sigma=.68, beta=2.4, eps0=.30, rho_m=.90, rho_c=.68).

BLOCK 1 - CROWN JEWEL: OPPONENT-TYPE x OPPONENT-COUNT LADDER WITH THE ANCHOR ALONE (bases 4,6,7,8,9; pairs 4,6,7,8,9 + mirrors). In every cell the .52 anchor favours A alone (k_conf = 1 throughout, so neither model's opposition cost fires) and only the NUMBER and TYPE of opposing rows change:
  * vs 1 mid (.615, m=2): DTG .860 / SSG .861
  * vs 2 mids (.615,.625, m=3): DTG .857 / SSG .781
  * vs 4 mids (all, m=5): DTG .850 / SSG .657
  * vs 1 high (.97, m=2): DTG .860 / SSG .875
  * vs 2 highs (.93,.97, m=3): DTG .857 / SSG .874
  * vs 2 mids + 2 highs (m=5): DTG .850 / SSG .780
DTG must predict a FLAT ~.85-.86 profile across all six cells for every parameter setting (P_trig = 1, rho_t ~ 1, rho_tc never fires at k_conf = 1) - 'a near-chance dissenter standing alone is followed at ~.85 flat in coalition size'. SSG-6 must instead produce a two-way INTERACTION: flat ~.87 when the opposition is high-validity (share stays ~1, gate wide open) but a monotone 20-point slide .861 -> .781 -> .657 as mid-doubt rows accumulate and steal oddness share, plus the matched-m contrast 4 mids (.657) versus 2 mids + 2 highs (.780) which is a pure TYPE effect at identical display length and identical row count. Robustness: across gamma_s = 6.4-8.6 the 4-mid cell lies in [.62,.74] for SSG while DTG stays in [.76,.86] even at its own worst corner. Flat-in-everything versus graded-in-oddness-share is the decisive parameter-light read-out.

BLOCK 2 - UNANIMITY: DOES THE SCANDALOUS ROW HELP A DOMINANCE DISPLAY? (bases 1,2,3 vs 10,11; pairs 1,2,3,10,11 + mirrors). Unanimous displays carried by the anchor: m=1 (anchor alone) DTG .860 / SSG .715; m=3 (anchor + two highs) DTG .857 / SSG .697; m=5 (anchor + four mids) DTG .850 / SSG .693. Matched anchor-SILENT unanimous displays: m=2 (two mids) DTG .773 / SSG .744; m=5 (four mids + .93) DTG .701 / SSG .690. Read at matched m=5, DTG says adding/substituting the anchor row makes the display ~15 points EASIER (.701 -> .850) because it flips the trial out of the load-degraded compensatory limb into the near-free one-reason limb, whereas SSG says it is worth exactly nothing (.690 -> .693) since unanimity kills adjudication. Equally, base 1 versus base 7 (anchor alone unanimous vs anchor contradicted by one mid) is flat for DTG (.860 -> .860) but a +15-point jump for SSG (.715 -> .861): contradiction MAKING a display easier is SSG's signature, invariance is DTG's.

BLOCK 3 - SHARED INTEGRATION BASELINES THAT BLOCK THE 'GLOBAL NOISE' ESCAPE (bases 10,11,12; pairs 10,11,12 + mirrors). With the anchor silent both models integrate and must agree within ~3 points: 2-0 mids .773/.744, 5-0 .701/.690, and the exact 2-vs-2 mid split .467/.473 - both BELOW chance at a row-count tie, which no vote-counting account can produce in either direction. These cells pin eps0, rho_m and rho_c for both models free of the contrast, so any Block-1/2 divergence cannot be re-described as a lapse-rate difference; they also give a mild test of SSG's steeper conflict leak.

INTERPRETATION: if the lone .52 dissenter is followed at ~.85 regardless of whether it is opposed by one high, two highs, one mid, two mids or four mids, and if unanimous displays carried by the .52 expert stay at ~.85 while the same-length displays with the .52 row silenced fall to ~.70, the purely ABSOLUTE sharp gate with a cheap, opposition-only-costed one-reason limb and a load-costed compensatory limb (DTG-2R/OC-sharp) is confirmed and the salience-share gate plus the pi_one = 0 unanimity rule is falsified. If instead follow rates slide monotonically with the number of mid-doubt (but not high-validity) opponents, and unanimous anchor displays sit ~15 points BELOW the conflict displays that contain the very same anchor, SSG-6 wins.

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
**Description:** **SSG-6 — Salience-Share Gating with an opposition-graded adjudication limb and a leakier integration limb.**

The architecture is exactly SSG-5's; one quantitative commitment is revised, none is reversed.

**(1) Two modes: adjudication vs. integration.** On every trial the subject either lets ONE expert adjudicate the choice or integrates all disagreeing rows. Mode is decided by a RELATIVE quantity — the share of the display's total *oddness* owned by genuinely near-chance experts — not by display length nor by an absolute knife edge. Each discriminating row has oddness o_j = (1-v_j)^gamma_s (gamma_s ≈ 6.4-8.6, a contrast-enhanced, nearly ordinal 'who is the strangest expert here' read-out) and a scandal value g_j = sigmoid((v_crit - v_j)/w_h) with a crisp boundary (w_h ≈ .008-.011, v_crit ≈ .576-.586). The gate is share = Σ_j g_j·o_j / Σ_j o_j and pi_one = (max_j g_j)·[f + (1-f)·sigmoid((share - c)/w)]: adjudication requires BOTH that some expert be near-chance AND that near-chance experts own most of the display's oddness.

**(2) Adjudication is faithful but NOT flat in opposition; it is flat only in display length.** When a near-chance expert commands the trial, subjects follow its verdict with high fidelity (eps_t ≈ .19-.29) and scanning extra, silent or agreeing rows costs nothing at all — one-reason choice is free in sheer list length, which is why anchor-follow rates stay high on six-row displays. What it is NOT free in is genuine CONTRADICTION: each additional row that argues the other way beyond the first charges rho_opp ≈ .82-.90. A scandalous dissenter standing alone against one opposing voice is followed almost as often as a scandalous dissenter in a unanimous display, but once two or three experts are lined up the other way the adjudication itself becomes contested and the verdict is dragged appreciably toward chance. Among co-scandalous rows the consulted verdict is drawn from softmax(log g_j + mu·j) with a WEAK positional gain (mu ≈ .35-.62); reading order can never lift an ordinary expert over a near-chance one.

**(3) REVISED — integration is an intrinsically effortful, frequently-abandoned operation, and it is fragile under DISAGREEMENT.** When the gate is shut, evidence is integrated with w_j = (1-v_j)^gamma_w·exp(lam·j) (gamma_w ≈ 1.32-1.58, lam ≈ .10-.17, so doubt, not position, buys weight), divisively normalised to the loudest dissent, E = Σ sign_j·ŵ_j/(sigma + Σ ŵ_j), p = sigmoid(beta·E). The sharpened claim is that the BASELINE abandonment rate of integration is appreciably higher than previously assumed (eps0 ≈ .26-.34, not ≈ .22-.30): weighing several probabilistically-labelled experts against one another is an operation subjects decline to complete on roughly a third of trials even before any conflict or load penalty, so integrated verdicts are systematically less decided than the weights alone imply. On top of this baseline the retained evidence leaks further with display load (rho_m) and much more steeply with every additional row that argues the other way (rho_c ≈ .62-.75 per minority row). Because the baseline lapse applies to EVERY integration trial — including the unanimous/dominance displays where there is nothing to adjudicate — it predicts that even fully consistent multi-row displays stay clearly below ceiling (~.74-.75), while conflict displays are pulled nearer chance still and never reach a near-deterministic anti-validity verdict.

The theory's signature is therefore that BOTH modes are degraded by contradiction and NEITHER is degraded by list length — they differ in how steeply contradiction bites (mild for adjudication, severe for integration) and in their baseline completion rates (adjudication is cheap and usually completed; integration is expensive and often abandoned). When all discriminating experts agree there is nothing to adjudicate (pi_one = 0) and only the load/baseline leak applies. If two printed validities are indistinguishable (gap < .002) oddness has no referent and choice collapses to a graded unweighted row tally. No feedback is given, so the rule is stationary across the block.

**Parameters:**
- gamma_s: [6.4, 8.6]
- v_crit: [0.576, 0.586]
- w_h: [0.008, 0.011]
- c_share: [0.69, 0.75]
- w_share: [0.040, 0.065]
- gate_floor: [0.33, 0.47]
- mu: [0.35, 0.62]
- eps_t: [0.19, 0.29]
- rho_opp: [0.82, 0.90]
- gamma_w: [1.32, 1.58]
- lam: [0.10, 0.17]
- sigma: [0.58, 0.78]
- beta: [2.15, 2.65]
- eps0: [0.26, 0.34]
- rho_m: [0.87, 0.93]
- rho_c: [0.62, 0.75]
- tally_kappa: [0.7, 1.3]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # SSG: Salience-Share Gating
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
    gamma_s = float(parameters['gamma_s'])
    v_crit = float(parameters['v_crit'])
    w_h = float(max(float(parameters['w_h']), 1e-4))
    c_share = float(parameters['c_share'])
    w_share = float(max(float(parameters['w_share']), 1e-4))
    floor = float(np.clip(float(parameters['gate_floor']), 0.0, 1.0))
    mu = float(parameters['mu'])
    eps_t = float(np.clip(float(parameters['eps_t']), 0.0, 1.0))
    rho_opp = float(np.clip(float(parameters['rho_opp']), 1e-6, 1.0))
    gamma_w = float(parameters['gamma_w'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
    beta = float(parameters['beta'])
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

    # ---------- leak on the integration limb ----------
    keep_c = (1.0 - eps0) * (rho_m ** max(m - 2, 0)) * (rho_c ** k_conf)
    keep_c = float(min(max(keep_c, 0.0), 1.0))

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

    vd = v[disc]
    jd = disc.astype(float)

    # ---------- scandal values and sharpened oddness ----------
    zz = np.clip((v_crit - vd) / w_h, -60.0, 60.0)
    g = 1.0 / (1.0 + np.exp(-zz))                       # near-chance "scandal" weight
    log_g = -np.logaddexp(0.0, -zz)                     # stable log sigmoid
    log_o = gamma_s * np.log(np.clip(1.0 - vd, 1e-12, None))
    log_o_shift = log_o - np.max(log_o)
    o = np.exp(np.clip(log_o_shift, -700.0, 0.0))
    o_sum = float(np.sum(o))
    if (not np.isfinite(o_sum)) or o_sum <= 0.0:
        o = np.ones(m, dtype=float)
        o_sum = float(m)

    share = float(np.sum(g * o) / o_sum)
    g_share = sig((share - c_share) / w_share)
    P_trig = float(np.clip(np.max(g), 0.0, 1.0))

    # unanimous displays need no adjudication
    if k_conf == 0:
        pi_one = 0.0
    else:
        pi_one = P_trig * (floor + (1.0 - floor) * g_share)
    pi_one = float(np.clip(pi_one, 0.0, 1.0))

    # ---------- which row adjudicates (recency only breaks scandal ties) ----------
    sc = log_g + mu * jd
    sc = sc - np.max(sc)
    ww = np.exp(np.clip(sc, -700.0, 0.0))
    ws = float(np.sum(ww))
    if (not np.isfinite(ws)) or ws <= 0.0:
        ww = np.full(m, 1.0 / m)
    else:
        ww = ww / ws
    p_or_a = float(np.sum(ww[signs > 0.0]))
    fid = (1.0 - eps_t) * (rho_opp ** max(k_conf - 1, 0))
    fid = float(min(max(fid, 0.0), 1.0))
    p_one_a = 0.5 + fid * (p_or_a - 0.5)

    # ---------- integration limb ----------
    logw = gamma_w * np.log(np.clip(1.0 - vd, 1e-12, None)) + lam * jd
    logw = logw - np.max(logw)
    wi = np.exp(np.clip(logw, -700.0, 0.0))
    wsum = float(np.sum(wi))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        wi = np.ones(m, dtype=float)
        wsum = float(m)
    E = float(np.sum(signs * wi)) / (sigma + wsum)
    p_comp = sig(beta * E)
    p_c_a = 0.5 + keep_c * (p_comp - 0.5)

    # ---------- mixture ----------
    p_a = pi_one * p_one_a + (1.0 - pi_one) * p_c_a
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    # printed validities of this design (fixed at design time)\n    V = np.array([0.635, 0.93, 0.615, 0.645, 0.97, 0.52, 0.625])\n    anchor = int(np.argmin(V))  # the lone near-chance expert, v = .52 (column 5)\n\n    follows = []\n    for a_r, b_r, resp in zip(data['option_a_ratings'],\n                              data['option_b_ratings'],\n                              data['response']):\n        try:\n            a = np.asarray(list(a_r), dtype=float)\n            b = np.asarray(list(b_r), dtype=float)\n        except Exception:\n            continue\n        n = int(min(a.size, b.size, V.size))\n        if n <= anchor:\n            continue\n        d = a[:n] - b[:n]\n        if d[anchor] == 0.0:\n            continue  # anchor silent -> not a test trial\n        s = np.sign(d[anchor])\n        opp = [j for j in range(n) if j != anchor and d[j] != 0.0 and np.sign(d[j]) != s]\n        # exclude displays where a high-validity expert opposes the anchor\n        if any(V[j] >= 0.9 for j in opp):\n            continue\n        # keep only unanimous-with-anchor displays (0 opponents) and displays with\n        # a COALITION of >=2 mid-doubt opponents; drop the single-mid-opponent cell\n        # where the two theories coincide\n        if len(opp) == 1:\n            continue\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        chose_a = (r == 0)\n        follows.append(1.0 if ((s > 0) == chose_a) else 0.0)\n\n    if len(follows) == 0:\n        return 0.5\n    return float(np.mean(follows))",
  "rationale": "The metric is the anchor-follow rate on exactly the trials where the two accounts make maximally divergent commitments: displays in which the lone near-chance expert (v=.52, column 5) discriminates, is never opposed by a high-validity expert (.93/.97), and is either unopposed (unanimous dominance displays: base pairs 1/13, 2/14, 3/15) or opposed by a coalition of two or four mid-doubt experts (pairs 4/16, 8/20). Under DTG-2R/OC-sharp the absolute sharp gate sets P_trig=1 on all of these, the one-reason limb is nearly free in list length, and rho_tc does not compound at k_conf<=1 for the unanimous cells, so the follow rate must be a flat ~.85 in every one of these cells regardless of the number/type of opponents and regardless of unanimity (predicted pooled value ~.855). Under SSG-6 two distinct mechanisms pull the same trials down: (a) pi_one = 0 on unanimous displays forces them through the high-baseline-lapse integration limb (~.69-.72), and (b) the relative salience-share gate is stolen by accumulating mid-doubt opponents, driving the 4-mid cell to ~.66 and the 2-mid cell to ~.78 (predicted pooled value ~.71). The 40 trials per subject entering the average make the per-subject binomial noise small (~.003-.005 variance), and the cells chosen are those whose predictions are stable across each theory's parameter box, so the ~.15 mean gap should be highly reliable at N=25. Trials where the anchor faces a single mid opponent, or where high-validity experts oppose it, are deliberately excluded because both theories agree there (~.86), which would only dilute the contrast; anchor-silent displays are excluded for the same reason."
}
```

## Usage

```json
{
  "input_tokens": 16348,
  "output_tokens": 6927
}
```
