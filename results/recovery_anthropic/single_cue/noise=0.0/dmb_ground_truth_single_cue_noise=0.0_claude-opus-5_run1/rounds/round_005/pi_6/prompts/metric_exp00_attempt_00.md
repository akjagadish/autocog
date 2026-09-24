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
**Validities (n_features=6):** [0.51, 0.72, 0.8, 0.9, 0.95, 0.985]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 2: A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1]
  trial 10: A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 11: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 14: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 15: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 17: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 19: A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 20: A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 21: A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 22: A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 23: A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 24: A=[1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Rationale:** GOAL: separate SSI-DS+ (advocated) from MDSC-NV (competitor). Both invert validity and both add a reading-position gain, so the separation has to be built on three mechanistic axes: (A) MAGNITUDE OF THE VALIDITY INVERSION \u2014 SSI-DS+ uses (1-v)^gamma with gamma~0.9 so position (exp(lam*j), lam~1.5) dominates and the LATE rows carry the decision, whereas MDSC-NV uses (1-v)^delta with delta=13-17 so the single LOWEST-validity expert carries the decision wherever it sits; (B) SUMMATION vs SINGLE-CUE \u2014 SSI-DS+ sums all discriminating rows and passes the max-normalised sum through a sigmoid (coalitions add, and the sum can be near-zero even at a lopsided row count), MDSC-NV draws ONE row so coalitions buy nothing; (C) NULL-VERDICT CAPTURE \u2014 MDSC-NV says NON-discriminating rows matter (with prob q~.41 attention is captured by the most doubtful expert in the whole display; if that expert is silent the subject guesses), while SSI-DS+ says silent rows are strictly irrelevant.

VALIDITIES [0.51, 0.72, 0.80, 0.90, 0.95, 0.985]: validity RISES monotonically with column index, so anti-validity (doubt) and recency pull in exactly OPPOSITE directions on every trial \u2014 the cleanest possible geometry for axis (A). All pairwise gaps are >= 0.035, far above both models' label tolerances (.0025 / .002), so neither model can retreat into tally/degenerate mode; the contrast is purely mechanistic. SSI weights (gamma=.9, lam=1.5): w = [0.53, 1.43, 4.72, 11.33, 27.21, 41.27] \u2014 late-dominated but with w2+w3+w4 = 43.3 just barely exceeding w5 = 41.3, which makes a genuine coalition-summation test possible. MDSC log-salience (delta=15, lam=1.725): [-10.7, -17.4, -20.7, -29.4, -38.0, -54.4] \u2014 column 0 (v=.51) is the global doubt anchor by ~6.7 log units and stays the argmax over the whole parameter box (delta 13-17 x pos_ratio .09-.14), so MDSC is effectively 'whoever wins column 0; if column 0 is silent, guess with prob q and otherwise follow the lowest-validity discriminating row'.

12 base structures, each with its exact A/B mirror (24 unique pairs, K=4 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions use SSI-DS+ (gamma=.90, lam=1.50, beta=3.6, eps0=.24, rho=.78, eta=.73) and MDSC-NV (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81).

BLOCK 1 \u2014 CROWN JEWEL: NULL-VERDICT CAPTURE ON UNANIMOUS m=2 DISPLAYS (bases A,B,C = pairs 1-6). Three 2-0 dominance displays that differ ONLY in WHICH experts are silent. SSI-DS+ must predict the SAME saturated value on all three (E is computed over discriminating rows only and always lies in (1,2); unanimous m=2 also means zero load): .875 / .877 / .880 \u2014 a flat profile, and the base A minus base B/C difference must be ZERO for every parameter setting. MDSC-NV predicts a step: base A (disc = cols 0,1, so the globally most doubtful expert speaks) ~.88, base B (disc = cols 2,3, col0 silent) ~.72, base C (disc = cols 4,5, col0 silent) ~.72. A ~15-point drop between displays with identical 2-0 dominance structure is producible by no compensatory or standard one-reason account; flatness kills null-verdict capture. Base A also pins eps0 for both models.

BLOCK 2 \u2014 DOUBT vs RECENCY DIRECTIONAL REVERSALS AT m=2 (bases D,E = pairs 7-10).
\u2022 Base D: A wins col0 (.51, first) vs B wins col5 (.985, last). SSI: w5/w0 = 78 -> E ~ -.99 -> p(A) ~ .14. MDSC: the .51 row wins the doubt competition by ~44 log units and it is also the capture target -> p(A) ~ .88. A ~74-POINT REVERSAL, the single largest contrast in the design.
\u2022 Base E: A wins col1 (.72) vs B wins col4 (.95), with col0 SILENT. SSI: p(A) ~ .14 (same late-row logic). MDSC: doubt still picks col1, but capture on the silent col0 drags it back -> p(A) ~ .72. ~58-point reversal, and the D-vs-E comparison within MDSC (.88 vs .72) is a second, independent read-out of capture that SSI must call identical (.141 vs .144).

BLOCK 3 \u2014 COALITIONS ADD vs COALITIONS BUY NOTHING (bases F,G,H,K = pairs 11-16, 21-22).
\u2022 Base F (m=6, 1-vs-5): A wins col0 alone; B wins cols 1-5. SSI: summed weight -2.07 max-normalised -> p(A) ~ .36 (majority/late coalition wins). MDSC: one more-doubtful row outvotes five -> p(A) ~ .66. Clean directional split at a 1-vs-5 row count.
\u2022 Base G (m=3, 2-vs-1): A wins cols 4,5; B wins col0. SSI ~ .80; MDSC ~ .19. ~60-point reversal.
\u2022 Base H (m=4, exact 2-2 tie): A wins cols 0,1; B wins cols 4,5. SSI ~ .27; MDSC ~ .75. ~48-point reversal at a row-count tie, which no tally account can produce in either direction.
\u2022 Base K (m=4, 1-vs-3): A wins col5 alone; B wins cols 1,2,3. SSI ~ .68 (the lone heavy late row outweighs three earlier rows); MDSC ~ .35 (lowest-validity discriminating row is col1, favouring B; capture adds a pull to chance). ~33-point reversal.

BLOCK 4 \u2014 SUM NEAR ZERO AT A LOPSIDED ROW COUNT (base J = pairs 19-20). A wins cols 2,3,4 (summed weight 43.3), B wins col5 (41.3). SSI: E = +.05 -> p(A) ~ .52, i.e. essentially CHANCE despite a 3-vs-1 row count \u2014 the signature 'big count margin, cancelling weighted margin' prediction. MDSC: lowest-validity discriminating row is col2 (favouring A) -> p(A) ~ .65. A 13-point gap in which SSI is at chance and MDSC is clearly above it.

BLOCK 5 \u2014 UNANIMITY DECAY / eta AND CAPTURE-MONOTONICITY (bases I,L = pairs 17-18, 23-24). Base L: unanimous m=4 on cols 2-5 (col0 silent). Base I: unanimous m=6 (all rows, col0 speaks). SSI, whose load is discounted by eta on unanimous displays and which ignores silent rows, must produce a STRICTLY MONOTONE decline with display size: .88 (m=2) > .76 (m=4) > .68 (m=6). MDSC must produce a NON-MONOTONE profile because capture depends on whether col0 speaks: .88 (m=2, base A) > .65 (m=4, base L) ~ .66 (m=6, base I) \u2014 the m=6 display is no worse than the smaller m=4 one, because at m=6 the doubt anchor is guaranteed to discriminate.

INTERPRETATION: if the three unanimous m=2 displays sit flat at ~.87 regardless of which experts are silent, if the option holding the LATE high-validity rows is chosen on bases D, E, G, H (~.14-.27 for the early/low-validity holder), if the lone late row beats three earlier rows (base K ~.68) and the 3-vs-1 base J sits at chance, then compensatory max-normalised summation with a weak validity inversion and competition-scaled interference (SSI-DS+) is confirmed and MDSC-NV's near-argmax doubt rule plus null-verdict capture is falsified. If instead unanimous 2-0 accuracy drops ~15 points purely as a function of whether the .51 expert happens to speak, the early .51 row beats the late .985 row (base D ~.88), and one doubtful row outvotes a five-row coalition (base F above chance), MDSC-NV wins. No pair is degenerate: on 8 of the 12 bases the models disagree in direction or by more than 30 points, and the unanimous bases double as the shared lapse baseline.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** SSI-DS+ (Strategy Selection by Label-Code Injectivity, with Distinctive-Cue Summation and Competition-Scaled Interference).

(1) STRATEGY SWITCH. Before the block the subject reads the printed validity list as a naming code. If all labels are distinct printed symbols (0.955 vs 0.950 counts as distinct) the experts are individuable and the subject adopts a WEIGHTED mode; if any label is repeated the experts are interchangeable and the subject adopts an unweighted MAJORITY TALLY mode. The switch is all-or-none at the design level (only a small per-trial slip probability omega), not a graded function of numeric gap size.

(2) WEIGHTED MODE = COMPENSATORY SUMMATION over all discriminating rows: E = sum_j sign(a_j-b_j) * (1-v_j)^gamma * exp(lam*j), rescaled so the strongest discriminating row has magnitude 1, then p(A) = sigmoid(beta*E). Validity enters with INVERTED sign (near-certain experts are redundant echoes of the general impression; near-chance disagreements are the news that must be adjudicated) and position enters as a reading-order recency gain. Coalitions ADD, so several early distinctive rows can outvote a single later row.

(3) TALLY MODE = graded majority p(A) = sigmoid(kappa*(n_A - n_B)): 1-1 splits give exactly 0.50, 2-1 splits ~0.77 (harder than probability matching, softer than a hard majority rule).

(4) CONFLICT INTERFERENCE IS COMPETITION-SCALED (the refinement). Guessing is driven by the load of holding several discrepancies in mind while ADJUDICATING BETWEEN THEM. The load is therefore not the raw number of discriminating rows but the number of rows that actually have to be pitted against one another: when the display is unanimous (min(n_A,n_B) = 0, i.e. every discrepancy points the same way) there is nothing to adjudicate and the extra rows impose only a discounted reading load eta*(m-2) rather than the full (m-2). Formally g = 1 - (1-eps0) * rho^L with L = (m-2) on opposed displays and L = eta*(m-2) on unanimous displays, and the core probability is mixed toward 0.5 by g. This predicts that dominance accuracy degrades with the number of agreeing experts only mildly, whereas conflict displays degrade steeply with the number of opposed experts - the two are dissociated, not a single monotone function of display size.

No feedback, hence no learning; the rule is stationary across the block.

**Parameters:**
- gamma: [0.86, 0.94]
- lam: [1.45, 1.55]
- beta: [3.2, 4.0]
- kappa: [1.10, 1.30]
- eps0: [0.22, 0.26]
- rho: [0.755, 0.80]
- omega: [0.0, 0.04]
- eta: [0.62, 0.84]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
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
            flat = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = flat[0].ravel(), flat[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed label code) ----------
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
    beta = float(parameters['beta'])
    kappa = float(parameters['kappa'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    omega = float(np.clip(float(parameters['omega']), 0.0, 0.5))
    eta = float(np.clip(float(parameters['eta']), 0.0, 1.0))

    # ---------- STRATEGY SWITCH: injectivity of the printed label code ----------
    LABEL_TOL = 2.5e-3   # two printed labels closer than this are the same symbol
    if n <= 1:
        injective = True
    else:
        vs = np.sort(v)
        gaps = np.diff(vs)
        injective = bool(np.all(gaps > LABEL_TOL))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])

    # ---------- weighted mode: distinctive-cue SUMMATION ----------
    # log weight = gamma*log(1 - v_j) + lam*j ; rescaled so strongest row = 1
    logw = gamma * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + lam * disc.astype(float)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    E = float(np.sum(signs * w))
    z = float(np.clip(beta * E, -60.0, 60.0))
    p_weighted_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- tally mode: graded majority ----------
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    zt = float(np.clip(kappa * (n_a - n_b), -60.0, 60.0))
    p_tally_a = 1.0 / (1.0 + np.exp(-zt))

    if injective:
        p_core_a = (1.0 - omega) * p_weighted_a + omega * p_tally_a
    else:
        p_core_a = (1.0 - omega) * p_tally_a + omega * p_weighted_a

    # ---------- conflict interference / lapse (competition-scaled load) ----------
    # load counts only rows that must be ADJUDICATED against one another:
    # unanimous (dominance) displays impose only a discounted reading load.
    conflict = float(min(n_a, n_b))
    expo = float(max(m - 2, 0))
    if conflict <= 0.0:
        expo = eta * expo
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    V_DESIGN = np.array([0.51, 0.72, 0.80, 0.90, 0.95, 0.985], dtype=float)\n\n    GAMMA = 0.90\n    LAM = 1.50\n    DELTA = 15.0\n    LAM_M = 1.725\n\n    hits = 0.0\n    tot = 0.0\n\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'],\n                                  data['option_b_ratings'],\n                                  data['response']):\n        try:\n            a = np.asarray(list(a_raw), dtype=float).ravel()\n            b = np.asarray(list(b_raw), dtype=float).ravel()\n        except Exception:\n            continue\n        n = int(min(a.size, b.size))\n        if n == 0:\n            continue\n        a = a[:n]\n        b = b[:n]\n\n        if n <= V_DESIGN.size:\n            v = V_DESIGN[:n]\n        else:\n            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.985)])\n        v = np.clip(v, 0.5, 0.9995)\n\n        diff = a - b\n        disc = np.nonzero(diff != 0.0)[0]\n        if disc.size == 0:\n            continue\n        signs = np.sign(diff[disc])\n\n        # --- SSI-DS+ side: max-normalised weighted SUM, late/high-validity dominated\n        logw = GAMMA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM * disc.astype(float)\n        logw = logw - np.max(logw)\n        w = np.exp(logw)\n        E = float(np.sum(signs * w))\n        if abs(E) < 1e-9:\n            continue\n        ssi_side = 1.0 if E > 0 else -1.0\n\n        # --- MDSC-NV side: single most-doubtful DISCRIMINATING row decides\n        sal = DELTA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM_M * disc.astype(float)\n        k = int(np.argmax(sal))\n        mdsc_side = float(signs[k])\n        if mdsc_side == 0.0:\n            continue\n\n        # keep only trials where the two accounts point to OPPOSITE options\n        if mdsc_side == ssi_side:\n            continue\n\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        chose_a = (r == 0)\n        ssi_favours_a = (ssi_side > 0)\n        hits += 1.0 if (chose_a == ssi_favours_a) else 0.0\n        tot += 1.0\n\n    if tot == 0.0:\n        return 0.5\n    return float(hits / tot)",
  "rationale": "The design contains six base structures (D, E, F, G, H, K) on which SSI-DS+ and MDSC-NV point to OPPOSITE options: SSI-DS+ sums max-normalised weights that are dominated by the late, high-validity rows, while MDSC-NV lets the single most doubtful (lowest-validity) discriminating row decide, with coalitions buying nothing. The metric selects exactly those conflict trials online (by recomputing the two decision rules per trial from the stimulus, using each theory's central parameter values) and reports the proportion of choices that went to the SSI-DS+ favoured option. Unanimous displays, base J (where both rules agree), and any trial where the rules coincide are excluded automatically, so the metric is not trivially true for either account: it is a pure directional read-out of 'weighted late-row summation' vs 'anti-validity single cue'. Expected value under SSI-DS+ is roughly 0.75-0.78 (average of .86, .86, .64, .80, .73, .68 across bases D, E, F, G, H, K) and under MDSC-NV roughly 0.25 (average of .12, .28, .34, .19, .25, .35) \\u2014 a ~0.5 gap. Because ~48 trials per subject enter the average, binomial noise is only ~0.06 SD and parameter heterogeneity is modest (both models' lapse terms are bounded), so the per-subject spread should be far below the between-theory gap."
}
```

## Usage

```json
{
  "input_tokens": 14106,
  "output_tokens": 3531
}
```
