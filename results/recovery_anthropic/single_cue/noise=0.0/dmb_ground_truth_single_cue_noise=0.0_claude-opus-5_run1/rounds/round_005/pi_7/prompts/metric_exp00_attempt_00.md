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
**Validities (n_features=6):** [0.55, 0.9, 0.75, 0.95, 0.68, 0.85]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 7: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 9: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1, 0]
  trial 10: A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 12: A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 13: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 14: A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 16: A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: separate MDSC-NV (advocated) from SSI-DS+ (competitor). Both models invert validity (near-chance experts carry the decision) and both add a reading-position gain, so they can only be separated on three axes: (A) DOUBT GAIN MAGNITUDE — MDSC-NV uses (1-v)^delta with delta=13-17, SSI-DS+ uses (1-v)^gamma with gamma~0.9, i.e. ~17x weaker, so with comparable positional gains (lam_MDSC = pos_ratio*delta ~ 1.2-2.4 vs lam_SSI ~ 1.5) the two models put their weight on DIFFERENT rows: MDSC is governed by the lowest-validity row wherever it sits, SSI is governed by the LAST row unless it is extremely non-distinctive; (B) SINGLE-CUE vs SUMMATION — MDSC draws one row and cannot let a coalition add up, SSI sums and passes the max-normalised sum through a sigmoid; (C) NULL-VERDICT CAPTURE — MDSC says non-discriminating rows matter (with prob q~.41 attention is captured by the most doubtful expert in the WHOLE display, and if that expert is silent the subject guesses), whereas SSI (like every compensatory or one-reason model) says silent rows are strictly irrelevant.

VALIDITIES [0.55, 0.90, 0.75, 0.95, 0.68, 0.85]: scrambled so validity does not track column order; the two most doubtful experts sit at column 0 (.55, FIRST) and column 4 (.68, late), the two most certain at columns 1 (.90) and 3 (.95). All pairwise gaps >= .05, far above both models' label tolerances (.002 / .0025), so neither model is in tally/degenerate mode — the contrast is purely mechanistic. MDSC log-salience (delta=15, lam=1.725): col4 -10.2 > col0 -12.0 > col2 -17.3 > col5 -19.8 > col1 -32.8 > col3 -39.8. SSI log-weight (gamma=.9, lam=1.5): col5 5.79 > col4 4.98 > col3 1.80 ~ col2 1.75 >> col1 -0.57 ~ col0 -0.72. Note the near-perfect crossing of the two orderings at the extremes (col0 is MDSC's #2 and SSI's last; col5 is SSI's #1 and MDSC's #4).

8 base structures, each with its exact A/B mirror (16 unique pairs, K=6 -> 96 trials), so both models' predicted responses are perfectly key-balanced; non-discriminating rows carry matched 1s AND matched 0s so displays are dense.

BLOCK 1 — CROWN JEWEL: NULL-VERDICT CAPTURE ON UNANIMOUS m=2 DISPLAYS (bases 1-3). All three are 2-0 dominance displays that differ ONLY in which rows are silent. SSI must predict the same saturated value (~.88) on all three, because E is computed over discriminating rows only and a unanimous 2-row display always gives E in (1,2). MDSC predicts a graded decline set by how much global doubt mass sits on SILENT rows: base 1 (disc = cols 0 & 4, the two most doubtful rows -> s_nondisc ~ 0) ~ .88; base 3 (disc = cols 0 & 2, so the globally most doubtful expert col4 is silent -> s_nondisc ~ .86) ~ .75; base 2 (disc = cols 1 & 3, the two highest-validity rows, BOTH doubtful experts silent -> s_nondisc ~ 1) ~ .72. A flat .88/.88/.88 profile falsifies null-verdict capture; a 15-16 point drop from base 1 to base 2 with identical 2-0 dominance structure is producible by no compensatory or standard one-reason model. Base 1 simultaneously pins eps0 for both models.

BLOCK 2 — DOUBT vs RECENCY DIRECTIONAL REVERSALS (bases 4-5, m=2 splits).
- Base 4: A wins col0 (.55, first column), B wins col5 (.85, last column). MDSC: the .55 row wins the doubt competition by ~8 log units -> p(A) ~ .75. SSI: col5's weight is ~670x col0's -> E ~ -1 -> p(A) ~ .14. ~60-point reversal.
- Base 5: A wins col1 (.90), B wins col3 (.95), with both doubtful rows silent. MDSC: doubt still picks col1 (.90 is 2x as doubtful as .95) but capture drags it toward chance -> p(A) ~ .72. SSI: position dominates -> p(A) ~ .15. ~57-point reversal. Bases 4 and 5 jointly estimate the delta/lam ratio: only a doubt gain above ~10 can make an EARLY .55 (or a mid-display .90) row beat a late row.

BLOCK 3 — COALITIONS BUY NOTHING (bases 6, 8).
- Base 6 (m=6): A wins col4 alone (.68, the globally most doubtful expert); B wins the other five rows. MDSC: one more-doubtful row outvotes five -> p(A) ~ .62 (above chance) despite being outvoted 1-vs-5. SSI: the summed coalition (dominated by col5) wins -> p(A) ~ .39 (below chance). A clean directional split at a 1-vs-5 row count.
- Base 8 (m=4, 2-vs-2): A wins cols 0 & 4 (the two most doubtful), B wins cols 3 & 5 (SSI's heavy late rows). MDSC ~ .75; SSI ~ .32 (its summed weight is dominated by col5). ~43-point reversal at an exact row-count tie, which no tally account can produce either way.

BLOCK 4 — UNANIMITY DECAY / eta TEST (base 7): A wins all six rows. MDSC's raw-m interference gives ~.66; SSI's competition-scaled load (eta discount, nothing to adjudicate) gives ~.69. Together with base 1 (~.88 for both) this pins the lapse parameters free of the contrast and gives a mild test of the eta discount.

INTERPRETATION: if unanimous 2-0 displays vary by ~15 points purely as a function of WHICH experts are silent (base1 > base3 > base2), if the early .55 row and the .90 row beat the late .85/.95 rows (bases 4-5 above chance), and if the lone doubtful row beats a five-row coalition (base 6 above chance) and the doubt pair beats the late pair (base 8 ~ .75), MDSC-NV is confirmed and SSI-DS+ is falsified. If instead all three unanimous m=2 displays sit flat at ~.88, and bases 4, 5, 6, 8 all land in the .14-.39 range (the option holding the LATE rows being chosen), compensatory summation over discriminating rows with an all-or-none injectivity switch wins and the null-verdict-capture claim is dead.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Design-fixed printed validities for this experiment\n    v = np.array([0.55, 0.90, 0.75, 0.95, 0.68, 0.85], dtype=float)\n    n = v.size\n    j = np.arange(n, dtype=float)\n\n    # MDSC-NV salience: near-argmax doubt rule (delta=15, lam=0.115*15)\n    logu = 15.0 * np.log(np.clip(1.0 - v, 1e-12, None)) + 1.725 * j\n    # SSI-DS+ compensatory summation weights (gamma=0.9, lam=1.5)\n    w_ssi = np.power(np.clip(1.0 - v, 1e-12, None), 0.9) * np.exp(1.5 * j)\n\n    hits = 0\n    tot = 0\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_raw, b_raw, r in zip(a_col, b_col, r_col):\n        try:\n            a = np.asarray(a_raw, dtype=float).ravel()\n            b = np.asarray(b_raw, dtype=float).ravel()\n        except Exception:\n            continue\n        if a.size != n or b.size != n:\n            continue\n        d = a - b\n        disc = np.nonzero(d != 0.0)[0]\n        if disc.size == 0:\n            continue\n\n        # MDSC-NV predicted winner: verdict of the single most doubtful\n        # discriminating expert\n        k = disc[int(np.argmax(logu[disc]))]\n        pred_mdsc = 0 if d[k] > 0 else 1\n\n        # SSI-DS+ predicted winner: sign of the weighted sum over all\n        # discriminating rows (coalitions add, late rows dominate)\n        E = float(np.sum(np.sign(d[disc]) * w_ssi[disc]))\n        if E == 0.0:\n            continue\n        pred_ssi = 0 if E > 0 else 1\n\n        if pred_ssi == pred_mdsc:\n            continue  # non-diagnostic trial type\n\n        tot += 1\n        try:\n            resp = int(r)\n        except Exception:\n            continue\n        if resp == pred_mdsc:\n            hits += 1\n\n    if tot == 0:\n        return 0.5\n    return float(hits) / float(tot)",
  "rationale": "The metric is a pure choice-key agreement score restricted to the trial types on which the two theories make OPPOSITE binary predictions. For each trial I recompute (a) MDSC-NV's near-argmax maximum-doubt winner (the verdict of the single most doubtful discriminating expert, log u = 15*log(1-v) + 1.725*j) and (b) SSI-DS+'s compensatory summation winner (sign of sum of sign*(1-v)^0.9*exp(1.5*j)). On this design the two orderings cross at the extremes (col0, v=.55, is MDSC's #2 but SSI's last; col5, v=.85, is SSI's #1 but MDSC's #4), so 8 of the 16 pairs (trials 4,5,6,8 and their exact mirrors 12,13,14,16 => 48 of 96 trials per subject) are diagnostic and key-balanced (4 favour A, 4 favour B), so response bias cannot inflate the score. Hand-computing the generative probabilities: MDSC-NV gives p(its own key) ~ .75 (t4/t12), ~.72 (t5/t13), ~.62 (t6/t14, the lone doubtful row beating a 5-row coalition), ~.75 (t8/t16), i.e. an expected metric value ~.71; SSI-DS+ gives ~.14, ~.15, ~.39, ~.32, i.e. ~.25. Gap ~.46. Because the score averages 48 trials per subject and the free parameters (q, eps0, rho, beta, gamma) only shift these probabilities by a few points, per-subject SD should be ~.07-.08 (var ~.006), giving a Welch t of roughly 15-20 at N=25. The metric also mirrors the two key qualitative claims: doubt beats recency (early .55 / mid .90 rows outvoting late .85/.95 rows) and coalitions buy nothing (1 row beating 5)."
}
```

## Usage

```json
{
  "input_tokens": 12937,
  "output_tokens": 7014
}
```
