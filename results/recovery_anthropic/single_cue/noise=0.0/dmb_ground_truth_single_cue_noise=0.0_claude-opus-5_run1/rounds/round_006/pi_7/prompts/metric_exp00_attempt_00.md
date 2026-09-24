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
**Validities (n_features=5):** [0.68, 0.95, 0.8, 0.52, 0.88]

**Trial pairs (n=18):**
  trial 1: A=[1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1]
  trial 6: A=[0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 8: A=[0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0]
  trial 9: A=[1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 10: A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  trial 13: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 14: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  trial 15: A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 16: A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 17: A=[0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 18: A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]

**Rationale:** GOAL: separate MDSC-NV (advocated) from DTV-RG/CL (competitor). The two accounts share the anti-validity ('doubt') direction and a load-based leak, so the design is built so that they AGREE about WHICH row is decisive and differ only on the three mechanisms that actually distinguish them: (A) NULL-VERDICT CAPTURE (MDSC: non-discriminating rows matter, because attention is grabbed pre-attentively by the globally most doubtful expert and a silent capture target forces a guess with prob q; DTV: silent rows are strictly irrelevant); (B) SINGLE-CUE vs RANK-GEOMETRIC SUMMATION with DTV's ABSOLUTE-DOUBT TRIGGER (DTV's phi jumps from ~.25 to ~.88 as soon as the least valid DISCRIMINATING expert is above v_c~.575, turning it into a near-tally in which coalitions overturn the doubtful voice; MDSC is near-argmax one-reason always, so coalitions buy nothing); (C) leak structure.

VALIDITIES [0.68, 0.95, 0.80, 0.52, 0.88] at columns c0..c4. Three properties are deliberately engineered. (i) One expert, c3 = .52, sits BELOW DTV's trigger (v_c ~ .575); every other expert (.68, .80, .88, .95) sits well above it. So whenever c3 discriminates DTV is in take-the-least mode (phi~.27), and whenever c3 is silent DTV flips into near-tally mode (phi~.88). (ii) c3 = .52 is also MDSC's global doubt anchor: with delta in [13,17] and lam = pos_ratio*delta, log-salience is c3 >> c0 > c2 > c4 > c1 across the ENTIRE parameter box (margin of c3 over c0 is 8-11 log units), and this ordering is exactly the ascending-validity ordering DTV uses, so the models can never be confused about which row is decisive. (iii) Validity does NOT track column order (.68, .95, .80, .52, .88), so 'use the first/last row' cannot masquerade as either model, and all gaps (>=.07) are far above both degeneracy tolerances (.002/.0025), keeping both models out of tally/degenerate mode.

9 base structures, each with its exact A/B mirror (18 unique pairs, K=5 -> 90 trials), so both models' predicted responses are exactly key-balanced and side bias cannot mimic either. Silent rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions below use MDSC (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81) and DTV (phi_lo=.25, phi_hi=.88, v_c=.575, w=.015, beta=5, eps0=.24, rho_m=.79, rho_c=.97).

BLOCK 1 - CROWN JEWEL: NULL-VERDICT CAPTURE ON DOMINANCE DISPLAYS (bases 1-4, pairs 1-8). Four unanimous displays that differ ONLY in WHICH experts are silent.
* B1 (m=1, A wins c3 = the .52 anchor): MDSC .88, DTV .875.
* B2 (m=1, A wins c0 = .68; the .52 anchor is SILENT): MDSC .724, DTV .875.
* B3 (m=2, A wins c3 and c1; anchor speaks): MDSC .88, DTV .879.
* B4 (m=2, A wins c0 and c1; anchor SILENT): MDSC .724, DTV .880.
DTV (like every compensatory or standard one-reason model) must call B1=B2 and B3=B4 EXACTLY - a flat ~.88 profile for any parameter setting, because E is computed over discriminating rows only and the leak sees only m and min(n_A,n_B), both identical within each pair. MDSC must produce a 13-17 point DROP (robust over its whole box) purely as a function of whether the .52 expert happens to be the one that speaks. A 2-0 (or 1-0) dominance display becoming measurably harder because a NON-discriminating row is the most doubtful one is the signature no competitor can generate.

BLOCK 2 - CROWN JEWEL: COALITIONS vs ONE REASON WITH THE ANCHOR SILENT (bases 2, 5, 6, 7; pairs 3-4, 9-14). A always wins c0 (.68, the least valid DISCRIMINATING row) while B's opposing coalition grows, and the .52 anchor stays silent throughout, so DTV is locked in near-tally mode (phi~.88):
* B2 1-vs-0: MDSC .724 / DTV .875
* B5 1-vs-1 (B wins c2): MDSC .712 / DTV .608
* B6 1-vs-2 (B wins c2,c4): MDSC .669 / DTV .230  <- 44-point DIRECTIONAL REVERSAL
* B7 1-vs-3 (B wins c2,c4,c1): MDSC .637 / DTV .271  <- 37-point DIRECTIONAL REVERSAL
MDSC predicts a shallow, monotone decline driven only by load (p_core pinned at ~.78 whatever the coalition size; never crossing chance). DTV predicts a CRASH THROUGH CHANCE between the 1-vs-1 and 1-vs-2 displays as the geometric sum flips sign. Shallow-and-always-above-chance vs steep-crossing is a parameter-robust quantitative signature (checked at the corners of both parameter boxes: MDSC B6 stays in [.63,.72], DTV B6 stays in [.21,.26]).

BLOCK 3 - THE ABSOLUTE-DOUBT TRIGGER (bases 8-9 vs 6-7; pairs 11-18). Structurally identical conflict displays in which only the IDENTITY of the lone dissenter changes, from c0 (.68, above DTV's trigger) to c3 (.52, below it):
* 1-vs-2: B6 (lone .68) vs B8 (lone .52). DTV: .230 -> .771, a 54-point swing that crosses chance - exactly its claim (2). MDSC: .669 -> .808, a 14-point swing, both above chance (the extra rise is capture being satisfied rather than a change of rule).
* 1-vs-3: B7 (lone .68) vs B9 (lone .52). DTV: .271 -> .713 (44 points). MDSC: .637 -> .749 (11 points).
A large, chance-crossing swing keyed to the absolute validity of the dissenter confirms DTV's trigger; a small swing with both cells above chance confirms MDSC's near-argmax rule plus capture. Bases 1, 3, 8 and 9 also double as the shared lapse/load baseline (both models within 4 points there), so any divergence in Blocks 1-2 cannot be written off as global noise.

INTERPRETATION: if 1-0 and 2-0 dominance accuracy drops ~15 points purely because the .52 expert is silent, and if the single .68 dissenter still beats two- and three-row coalitions of more valid experts, MDSC-NV is confirmed and DTV-RG/CL is falsified. If instead dominance displays are flat at ~.88 regardless of who is silent, and the coalition displays invert (majority chosen ~.75 when the dissenter is .68, but the lone dissenter chosen ~.77 when it is .52), rank-geometric summation with the absolute-doubt trigger wins and null-verdict capture is dead.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    hits = []\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n    for a_raw, b_raw, r in zip(a_col, b_col, r_col):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        n = int(min(a.size, b.size))\n        if n < 5:\n            continue\n        a = a[:n]\n        b = b[:n]\n        d = a - b\n        disc = [int(i) for i in np.nonzero(d != 0.0)[0]]\n        # require feature 0 (validity .68) to discriminate\n        if 0 not in disc:\n            continue\n        # require the global doubt anchor, feature 3 (validity .52), to be SILENT\n        if 3 in disc:\n            continue\n        fav_a = [i for i in disc if d[i] > 0.0]\n        fav_b = [i for i in disc if d[i] < 0.0]\n        if d[0] > 0.0:\n            lone, other = fav_a, fav_b\n            lone_is_a = True\n        else:\n            lone, other = fav_b, fav_a\n            lone_is_a = False\n        # exactly one dissenter (feature 0) against a coalition of >=2 more valid experts\n        if len(lone) != 1 or len(other) < 2:\n            continue\n        try:\n            resp = int(r)\n        except Exception:\n            continue\n        chose_lone = (resp == 0) if lone_is_a else (resp == 1)\n        hits.append(1.0 if chose_lone else 0.0)\n    if len(hits) == 0:\n        return float('nan')\n    return float(np.mean(hits))",
  "rationale": "This isolates the design's Block-2/Block-3 coalition displays in which the single .68 expert (feature 0) dissents against a coalition of two or three MORE valid experts (features 1, 2, 4) while the .52 anchor (feature 3) is silent \\u2014 i.e. base structures 6 and 7 (trial pairs 11-14), 20 trials per subject.\n\nMDSC-NV is a near-argmax maximum-doubt one-reason rule: the least valid DISCRIMINATING expert (.68) decides regardless of how many more valid experts oppose it, and capture by the silent .52 anchor only pulls part of the way toward chance. It therefore predicts the lone .68 dissenter's option is still chosen well ABOVE chance (~.64-.67).\n\nDTV-RG/CL, because the least valid dissenter (.68) is ABOVE its absolute-doubt trigger (v_c \\u2248 .575), flips into near-tally mode (\\u03c6 \\u2248 .88), so a 2- or 3-member coalition of more valid experts overturns the lone dissenter: it predicts this same quantity CRASHES BELOW chance (~.23-.27).\n\nSo the predicted values are ~.65 vs ~.25 \\u2014 a ~40-point gap that straddles chance, with each subject contributing 20 trials, giving tight per-subject estimates (binomial SD \\u2248 .10) and a clean Welch separation at N=25. It is not trivially true of MDSC: a compensatory model, a pure tally, a recency rule, or any ordinary take-the-best (which would favour the most valid expert's side) all give distinct values, and the contrast is purely about whether coalitions of more valid experts can overturn one doubtful dissenter."
}
```

## Usage

```json
{
  "input_tokens": 13611,
  "output_tokens": 3345
}
```
