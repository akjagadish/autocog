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
**Validities (n_features=6):** [0.62, 0.78, 0.86, 0.93, 0.53, 0.97]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 9: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 10: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 12: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 13: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 14: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 15: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 16: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 17: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 18: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  trial 19: A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 20: A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1]
  trial 21: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 23: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 24: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]

**Rationale:** GOAL: separate DTV-RG/CL (advocated) from MDSC-NV (competitor). The two share the anti-validity ('doubt') direction and a load leak, so the design is built so that they always AGREE about WHICH row is the decisive/doubt-anchor row, and differ only on the three mechanisms that genuinely distinguish them: (A) NULL-VERDICT CAPTURE (MDSC: silent rows matter, because attention is grabbed pre-attentively by the globally most doubtful expert and a silent capture target forces a guess with prob q~.41; DTV: non-discriminating rows are strictly irrelevant); (B) ABSOLUTE-DOUBT TRIGGER + RANK-GEOMETRIC SUMMATION (DTV: phi jumps from ~.28 to ~.85 as soon as the least valid DISCRIMINATING expert is above v_c~.575, so with the sub-trigger expert silent DTV becomes a near-tally in which coalitions of more valid experts overturn the lone doubtful dissenter, crossing chance; MDSC is near-argmax one-reason always, so coalitions buy nothing and it can never cross chance); (C) DTV's two-component leak (load x opposition, rho_c) versus MDSC's pure-load leak.

VALIDITIES [0.62, 0.78, 0.86, 0.93, 0.53, 0.97] at columns c0..c4,c5. Three engineered properties. (i) Exactly one expert, c4 = .53, sits BELOW DTV's trigger (v_c ~ .575); every other expert (.62, .78, .86, .93, .97) sits well above it, so DTV is in take-the-least mode whenever c4 speaks and in near-tally mode whenever c4 is silent. (ii) c4 = .53 is placed at column 4 so that doubt and reading position are ALIGNED for it: MDSC's log-salience (delta 13-17, lam = pos_ratio*delta) makes c4 the global argmax by >7 log units at every corner of its parameter box, and the second-most-doubtful discriminating row (c0 = .62) beats the next candidate (c1 = .78) by >5 log units, so MDSC is effectively pure argmax and cannot hide behind graded sharing. Crucially MDSC's doubt ordering over the discriminating rows (.53 < .62 < .78 < .86 < .93 < .97) is IDENTICAL to DTV's ascending-validity ordering, so the models never disagree about which row is decisive \u2014 only about what is done with it. (iii) Validity does not track column index (.62,.78,.86,.93,.53,.97), so 'take the first/last row' cannot masquerade as either model, and all label gaps (>= .04) are far above both degeneracy tolerances (.002/.0025), keeping both models out of tally/degenerate mode. Silent rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s.

12 base structures, each with its exact A/B mirror (24 unique pairs, K = 4 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Predictions below use DTV (phi_lo=.25, phi_hi=.88, v_c=.575, w=.015, beta=5, eps0=.24, rho_m=.79, rho_c=.97) and MDSC (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81).

BLOCK 1 \u2014 CROWN JEWEL: NULL-VERDICT CAPTURE ON DOMINANCE DISPLAYS (bases 1-4, pairs 1-8). Four unanimous displays that differ ONLY in WHICH experts are silent, holding m and min(n_A,n_B) exactly constant within each matched pair.
* Base 1 (m=2 unanimous, A wins c4 [.53] and c3; the anchor SPEAKS): DTV .878, MDSC .880.
* Base 2 (m=2 unanimous, A wins c0 [.62] and c3; the .53 anchor is SILENT): DTV .878, MDSC .724.
* Base 4 (m=4 unanimous, A wins c4,c1,c2,c3; anchor speaks): DTV .737, MDSC .749.
* Base 3 (m=4 unanimous, A wins c0,c1,c2,c3; anchor SILENT): DTV .737, MDSC .647.
DTV must call base1 = base2 and base3 = base4 EXACTLY for every parameter setting (E is computed over discriminating rows only and the leak sees only m and min(n_A,n_B)). MDSC must produce a 10-16 point DROP purely because a NON-discriminating row happens to be the most doubtful one. Flat-within-pair versus stepped is a parameter-free read-out of capture, and base 1 simultaneously pins eps0 for both models.

BLOCK 2 \u2014 CROWN JEWEL: COALITION CRASH WITH THE ANCHOR SILENT (bases 2, 5, 6, 7; pairs 3-4, 9-14). A always wins c0 (.62, the least valid DISCRIMINATING row) while B's opposing coalition of more valid experts grows, with the .53 anchor silent throughout, so DTV is locked in near-tally mode (phi~.85):
* 1-vs-0 (base 2): DTV .878 / MDSC .724
* 1-vs-1, B wins c1 (base 5): DTV .632 / MDSC .724
* 1-vs-2, B wins c1,c2 (base 6): DTV .240 / MDSC .682  <- 44-point DIRECTIONAL REVERSAL
* 1-vs-3, B wins c1,c2,c3 (base 7): DTV .271 / MDSC .647  <- 38-point DIRECTIONAL REVERSAL
MDSC predicts a shallow, monotone, always-above-chance decline driven only by load (its core is pinned at ~.795 whatever the coalition size); DTV predicts a CRASH THROUGH CHANCE between the 1-vs-1 and 1-vs-2 displays as the rank-geometric sum flips sign. Checked at the corners of DTV's box (phi_hi .84-.92): E at 1-vs-2 is -.55 to -.77, always negative.

BLOCK 3 \u2014 THE ABSOLUTE-DOUBT TRIGGER (bases 6 vs 8, 9 vs 10). Structurally identical conflict displays in which only the IDENTITY of the lone dissenter changes, from c0 (.62, above the trigger) to c4 (.53, below it):
* 1-vs-2: base 6 (lone .62 dissenter) DTV .240 -> base 8 (A wins c4, B wins c0,c1) DTV .769: a 53-point swing CROSSING chance. MDSC: .682 -> .808, a 13-point swing with both cells above chance.
* 2-vs-2 at an exact row-count tie: base 9 (A wins c0,c3; B wins c1,c2; anchor silent) DTV .523 (essentially CHANCE \u2014 the geometric sum nearly cancels despite a lopsided validity profile) vs base 10 (A wins c4,c3; B wins c1,c2; anchor speaks) DTV .708, a 19-point swing. MDSC: .647 -> .749, and it can never sit at chance on base 9 because its argmax row (c0) decides. Base 9 is the sharpest single cell: DTV ~.52, MDSC ~.65 at a 2-2 tally tie (no tally account can produce either value systematically).

BLOCK 4 \u2014 LOAD VERSUS OPPOSITION LEAK (bases 11, 12; pairs 21-24). Base 11 is a 6-vs-0 unanimous display (DTV .648, MDSC .663); base 12 is a 3-vs-3 display of the same six rows with the .53 anchor favouring A (DTV .626, MDSC .663). DTV's second leak component (rho_c^min(n_A,n_B)) says the balanced-conflict display is measurably noisier than the unanimous display of the SAME size even though the decisive row is equally decisive; MDSC's pure-load leak says they must be identical. Base 12 also shows both models above chance at an exact 3-3 tally, which kills any vote-counting account outright.

INTERPRETATION: if dominance displays are flat within matched pairs (base1 = base2, base3 = base4) regardless of who is silent, and if a single .62 dissenter is beaten by two- and three-row coalitions of more valid experts (bases 6-7 below chance) while a single .53 dissenter beats the same coalitions (base 8 ~.77), the absolute-doubt trigger plus rank-geometric summation is confirmed and null-verdict capture is dead. If instead 1-0/2-0 and 4-0 dominance accuracy drops 10-16 points purely because the .53 expert happens to be silent, and the lone .62 dissenter stays above chance against every coalition, MDSC-NV wins. No pair is degenerate: on 8 of the 12 bases the models disagree in direction or by more than 10 points, and bases 1, 4 and 11 double as the shared lapse/load baseline (models within 2 points there), so divergences elsewhere cannot be written off as global noise.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **DTV-RG/CL — Doubt-Triggered one-reason with Rank-Geometric weighting, under a two-component (load + conflict) leak and a DECISIVE degenerate-label tally.**

The evidence rule is unchanged from the accepted base; two claims about *how the rule degrades* are sharpened.

**(1) Rank-geometric evidence, decided from the LEAST valid dissenting expert.** On each trial the subject lines the *discriminating* rows up in order of printed validity, least valid first, and sums their verdicts with a geometric discount: E = Σ_r s_r · φ^r (r = 0 for the least valid discriminating expert, s_r = ±1 its verdict); p(A) = σ(βE). Only the ORDER of the printed labels enters, never the size of the gaps, which is why the .955/.950/.945 and .958/.954/.950 families are still resolved almost deterministically in favour of the strictly least valid dissenter, and why coalitions of more valid experts normally buy nothing (though a large enough coalition can still overturn one doubtful voice).

**(2) The discount rate is set by the ABSOLUTE doubtfulness of the decisive voice.** One-reason choice is a *reaction to a near-chance expert*: φ = φ_lo + (φ_hi − φ_lo)·σ((v₀ − v_c)/w), with v₀ the validity of the least valid DISCRIMINATING expert, φ_lo ≈ .25 (take-the-least), φ_hi ≈ .88 (near tally), v_c ≈ .575, w ≈ .015. Below the trigger the display is settled by one strange dissent; above it nothing is newsworthy and the rows are weighted nearly equally.

**(3) No null-verdict capture.** Non-discriminating rows are strictly irrelevant whatever their validity; all pull toward chance comes from the leak in (4).

**(4) NEW — the leak has TWO separable sources, and dominance displays leak only through the first.** Losing the thread is driven (i) by sheer display load — how many rows have to be held while the ordering is run — and (ii) independently by *opposition*: every additional row that argues the other way adds a fresh occasion to lose track of which side one was on. p(guess) = 1 − (1−ε₀)·ρ_m^max(m−2,0)·ρ_c^min(n_A,n_B), with ρ_m ≈ .79 (load) and ρ_c ≈ .97 (opposition). The two terms are close to orthogonal in the corpus: unanimous (dominance) displays have min(n_A,n_B)=0 and therefore leak only by load, so they stay well above the balanced-conflict displays of the same size, whereas a 3-vs-3 display of six rows is markedly noisier than a 6-vs-0 display of six rows. This is a direct, testable clash both with a pure-load leak (which charges a 6-0 display exactly what it charges a 3-3 display) and with a pure-opposition leak (which would make unanimous displays near-perfect).

**(5) NEW — when labels are degenerate the fallback tally is DECISIVE, not tepid.** The doubt ordering presupposes that each printed validity picks out one expert. If any label is repeated (gap < .002) the columns are interchangeable and choice collapses to an unweighted row tally — but that tally is used with a steep slope (κ ≈ 1.6, not ≈ 0.85): having nothing else to go on, the subject commits hard to the vote margin. Hence 1-vs-1 splits under duplicated labels are an exact coin flip (no anti-validity, no recency), while 2-vs-0 and 4-vs-0 splits under duplicated labels are chosen at ~.83 and ~.95 before the leak — which is what keeps unanimous displays high and what makes a 2-vs-1 display under duplicated labels clearly favour the majority rather than hovering near chance.

No feedback is given, so the rule is stationary across the block.

**Parameters:**
- phi_lo: [0.21, 0.29]
- phi_hi: [0.84, 0.92]
- v_crit: [0.570, 0.580]
- v_width: [0.013, 0.018]
- beta: [4.5, 5.5]
- eps0: [0.20, 0.28]
- rho_m: [0.76, 0.82]
- rho_c: [0.93, 1.00]
- tally_kappa: [1.30, 1.90]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # DTV-RG/CL: Doubt-Triggered one-reason with Rank-Geometric weighting,
    # two-component (load + conflict) leak, decisive degenerate-label tally
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
    phi_lo = float(np.clip(float(parameters['phi_lo']), 1e-3, 0.999))
    phi_hi = float(np.clip(float(parameters['phi_hi']), 1e-3, 0.999))
    v_crit = float(parameters['v_crit'])
    v_width = float(max(float(parameters['v_width']), 1e-4))
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

    # ---------- label-degeneracy gate (repeated printed labels) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        # experts cannot be individuated -> decisive unweighted row tally
        p_core_a = sig(kappa * (n_a - n_b))
    else:
        # order discriminating rows by printed validity (ascending);
        # ties broken by reading position (later read = more decisive)
        order = np.lexsort((-disc.astype(float), v[disc]))
        idx_sorted = disc[order]
        s_sorted = signs[order]

        v0 = float(v[idx_sorted[0]])
        # doubt trigger: only a near-chance dissenter buys one-reason choice
        phi = phi_lo + (phi_hi - phi_lo) * sig((v0 - v_crit) / v_width)
        phi = float(np.clip(phi, 1e-4, 0.9999))

        r = np.arange(m, dtype=float)
        w = np.power(phi, r)
        E = float(np.sum(s_sorted * w))
        p_core_a = sig(beta * E)

    # ---------- two-component leak: display load x opposition ----------
    expo_m = max(m - 2, 0)
    k_conf = int(min(n_a, n_b))
    g = 1.0 - (1.0 - eps0) * (rho_m ** expo_m) * (rho_c ** k_conf)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    V = np.array([0.62, 0.78, 0.86, 0.93, 0.53, 0.97], dtype=float)\n    ANCHOR = 4  # the .53 expert (below DTV's doubt trigger)\n    hits = 0.0\n    tot = 0.0\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'].tolist(),\n                                  data['option_b_ratings'].tolist(),\n                                  data['response'].tolist()):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        n = int(min(a.size, b.size))\n        if n != 6:\n            continue\n        diff = a[:6] - b[:6]\n        disc = np.nonzero(diff != 0.0)[0]\n        m = int(disc.size)\n        if m < 3:\n            continue\n        # the globally most doubtful expert (.53) must be SILENT\n        if ANCHOR in set(disc.tolist()):\n            continue\n        d = diff[disc]\n        na = int(np.sum(d > 0.0))\n        nb = m - na\n        if min(na, nb) != 1 or max(na, nb) < 2:\n            continue\n        if na == 1:\n            lone = int(disc[d > 0.0][0])\n            fav = 0  # option A favoured by the lone dissenter\n        else:\n            lone = int(disc[d < 0.0][0])\n            fav = 1\n        # the lone dissenter must be the LEAST valid discriminating expert\n        if abs(V[lone] - float(np.min(V[disc]))) > 1e-9:\n            continue\n        tot += 1.0\n        if int(resp) == fav:\n            hits += 1.0\n    if tot < 1.0:\n        return 0.5\n    return float(hits / tot)",
  "rationale": "This isolates Block 2's 'coalition crash' cells (trial pairs 11-14: 1-vs-2 and 1-vs-3 displays in which the lone dissenter is c0 = .62, the least valid DISCRIMINATING expert, while the globally most doubtful expert c4 = .53 stays SILENT). The metric is the proportion of those trials on which the subject sides with the lone least-valid dissenter.\n\nDTV-RG/CL: with the sub-trigger .53 expert silent, v0 = .62 > v_c, so phi jumps to ~.85 and evidence becomes a near-tally rank-geometric sum: E = 1 - phi - phi^2 (1-vs-2) and 1 - phi - phi^2 - phi^3 (1-vs-3), both clearly NEGATIVE at every corner of the parameter box. The coalition of more valid experts overturns the lone dissenter, so choice falls BELOW chance: predicted ~.24-.27.\n\nMDSC-NV: the doubt rule is near-argmax over discriminating rows, and c0 (.62) is by far the most doubtful discriminating row, so the deliberate route almost always sides with the lone dissenter; only the capture-to-silent-row component (q ~ .41 x 0.5) and the load lapse pull it toward chance, leaving it clearly ABOVE chance: predicted ~.65-.68.\n\nHence a ~0.41 directional gap that straddles chance, with 16 trials per subject and parameter ranges that barely move either prediction, so between-subject variance is dominated by (small) binomial noise. The metric uses no theory-specific machinery: it is just 'do people follow the lone least-valid dissenter against a coalition?'"
}
```

## Usage

```json
{
  "input_tokens": 14591,
  "output_tokens": 4965
}
```
