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
**Validities (n_features=6):** [0.958, 0.546, 0.954, 0.542, 0.95, 0.538]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 7: A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 8: A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 9: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 10: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 11: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 12: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 13: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 14: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 15: A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: dissociate OCO-LI (advocated) from SSI-DS+ (competitor). The two accounts share a salience form ((1-v)^exponent * exp(lam*j)) and a conflict-interference term, so they can only be separated on three axes: (A) HOW the salience is used u2014 OCO-LI samples ONE anchoring row (choice = normalized salience share) whereas SSI-DS+ SUMS the rows and passes the rescaled sum through a sigmoid (extremizing, coalitions add); (B) THE GATE u2014 OCO-LI's individuation gate is a GRADED function of the numeric validity gap, phi = mean_j min(1, d_j/tau) with tau in [.02,.045], while SSI-DS+ switches all-or-none on printed-label injectivity with a tolerance of only .0025; (C) INTERFERENCE u2014 OCO-LI's guess term grows with the raw number of discriminating rows m, SSI-DS+ discounts unanimous displays by eta.

VALIDITIES: [0.958, 0.546, 0.954, 0.542, 0.950, 0.538] u2014 three near-identical high labels and three near-identical low labels, arranged H,L,H,L,H,L. Every expert's nearest neighbour is EXACTLY 0.004 away, so (i) all six printed labels are numerically distinct and every consecutive gap (0.004) exceeds SSI-DS+'s LABEL_TOL = 0.0025, forcing SSI-DS+ into full WEIGHTED (compensatory summation) mode; (ii) OCO-LI's gate is pinned to phi = 0.004/tau in [0.089, 0.20] (point ~0.13), so OCO-LI must be ~87% probability-matching the raw PROPORTION of discriminating rows. High vs low validity still differ by 0.41, so the anti-validity/oddness factor is fully alive for both models; the alternating layout lets recency and oddness be aligned or opposed. Salience mass (lam~1.6, exponent~0.9) is u ~ [0.06, 2.6, 1.7, 70, 50, 1910]: column 5 (late, low-validity) carries almost all of it.

8 base structures, each with its exact A/B mirror (16 unique pairs, K=6 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions use lam=1.65/1.5, delta=gamma=0.9, eps=0.28/0.24, rho=0.79/0.78, beta=3.6, eta=0.73, phi=0.13.

BLOCK 1 u2014 SHARED-ISH BASELINES / INTERFERENCE (bases 1-2, pairs 1-4). Base 1: unanimous m=2 (A wins cols 4,5) u2192 both ~.86, pinning eps0. Base 2: unanimous m=6 u2192 OCO-LI ~.64 (full load) vs SSI-DS+ ~.68-.70 (load discounted by eta because there is nothing to adjudicate). A steep unanimity decay (.86 -> .64) supports OCO-LI's raw-m interference; a shallow one (.86 -> .70) supports the competition-scaled version.

BLOCK 2 u2014 GATE-DIAGNOSTIC m=2 SPLITS (bases 3-4, pairs 5-8). Proportion is exactly 1/1, so OCO-LI can leave chance only by the factor phi<=0.20.
u2022 Base 3: A wins col0 (.958, first) vs B wins col5 (.538, last); recency and oddness both hand the salience to B. OCO-LI ~.43-.47 (barely below chance); SSI-DS+ ~.14 (E is essentially -1, sigmoid saturates). ~30-point gap.
u2022 Base 4: A wins col3 (.542) vs B wins col4 (.950); oddness and recency OPPOSED, salience narrowly favours A. OCO-LI ~.51 (chance); SSI-DS+ ~.72-.75. ~21-point gap. Compressed-toward-chance vs saturated is the signature of the graded gate.

BLOCK 3 u2014 CROWN JEWEL: PROPORTION vs SUMMATION AT ASYMMETRIC m (bases 5-7, pairs 9-14). Here the row COUNT and the salience mass point in opposite directions.
u2022 Base 5 (m=3): A wins cols 0,2 (both high-validity, early), B wins col5 (low, last). Proportion 2/3 for A; >99% of salience mass on B. OCO-LI ~.55 (ABOVE chance); SSI-DS+ ~.22 (WELL BELOW). ~33-point directional reversal.
u2022 Base 6 (m=4): A wins col5 alone, B wins cols 0,1,2. Proportion 1/4 for A; salience 99.8% for A. OCO-LI ~.43 (below chance, majority wins); SSI-DS+ ~.72 (above chance, the lone late low-validity row wins). Full reversal on one display.
u2022 Base 7 (m=5): A wins cols 0,1,2,3; B wins col5. Proportion 4/5 for A; salience 95% for B. OCO-LI ~.57; SSI-DS+ ~.33.
Across bases 6, 5, 7 A's share of discriminating rows rises .25 -> .67 -> .80: OCO-LI must produce a MONOTONE INCREASING profile (~.43, .55, .57) tracking the tally proportion, whereas SSI-DS+ produces a profile dictated by the summed (last-column-dominated) weight (~.72, .22, .33) that is non-monotone in proportion. Monotone-in-proportion versus governed-by-the-summed-salience is the decisive quantitative signature and needs no free parameters to read off.

BLOCK 4 u2014 BALANCED HIGH-m SPLIT (base 8, pairs 15-16). A wins cols 0,2 (early highs), B wins cols 3,5 (late lows); proportion is exactly 1/2 so OCO-LI's ungated component is pinned to .500 (predicted ~.47 after interference) while SSI-DS+'s summed weight is ~-1.05 rescaled -> ~.28. This isolates the gate at a tally tie with more conflict than block 2 and maps how interference erodes the pull as m grows.

INTERPRETATION: if the m=2 splits hug chance (.43-.51), the 2-vs-1 base 5 sits ABOVE chance while the 1-vs-3 base 6 sits BELOW chance, and the asymmetric bases order themselves monotonically by the proportion of rows favouring A, the graded label-individuation gate is confirmed and the all-or-none injectivity switch plus compensatory summation is falsified. If instead every split saturates toward the option owning the late low-validity row (.14-.28) and base 6 is chosen ~72% of the time despite being outvoted 1-vs-3, SSI-DS+ wins. No pair is degenerate: on 6 of 8 bases the models disagree in direction or by >19 points; the two unanimous bases supply the lapse baseline and simultaneously test the eta discount.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Odd-Cue-Out Anchoring under Label Individuation (OCO-LI).**

Four claims, all about *which single row is allowed to anchor the comparison*, and about *whether the anchoring competition can run at all*.

1. **One discriminating row decides, and which one is stochastic.** On each trial the subject settles the A-vs-B comparison by anchoring on a SINGLE row on which the two products differ (a one-reason rule). No summation of evidence occurs, so a validity-weighted integrator is never the right description. Which row wins the anchoring competition varies from trial to trial (not fixed within a subject), so choices are graded within a subject and per-subject metric variance stays essentially binomial — exactly what the population variances in this corpus show.

2. **Anchor salience = ODDNESS, not diagnosticity.** The probability that row j is the anchor is proportional to u_j = exp(lam * j) * (1 - v_j)^delta.
   * *Odd-cue-out (delta > 0):* a disagreement located on a near-chance expert (v ≈ .53) is experienced as a strange, idiosyncratic fact that must be adjudicated, whereas a disagreement on a near-certain expert (v ≈ .97) is discounted as a mere echo of the general impression the subject already has. Communicated validity is therefore used with INVERTED sign. This is why choice is systematically *sub-chance* with respect to stated validity in every design where the labels are usable.
   * *Reading-order recency (lam > 0):* rows are read left-to-right and the later-read discrepancies are the ones still active in working memory when the response is emitted, so they capture the anchor more often. Position is a genuine second determinant (it is required by the 1-cue-late-versus-3-cue-early coalitions), but it is subordinate to, and — crucially — gated by, claim 3.

3. **Label-ambiguity gating (the new mechanism).** The anchoring competition presupposes that each expert can be *individuated*: the subject must be able to bind a particular stated validity to a particular column. That binding is only available when the stated validities are mutually discriminable. Define, for each expert j, its isolation d_j = min_{k≠j} |v_j - v_k| and its individuation s_j = min(1, d_j / tau); the design-level individuation is phi = mean_j s_j. With probability phi the subject runs the odd-cue-out competition above; with probability 1 - phi the columns are interchangeable in memory, no row is 'the odd one', every discriminating row is equally likely to become the anchor, and the choice probability collapses to the *proportion* of discriminating rows favouring each option (a probability-matched tally). When validity labels are duplicated or tightly clustered (e.g. [.95,.70,.55,.95,.70,.55]) phi → 0, the anti-validity and recency effects switch OFF, and a two-row conflict is decided by a coin flip. When all labels are distinct (gaps ≥ .05) phi → 1 and the anti-validity/recency anchoring runs at full strength.

4. **Conflict interference.** The more rows m on which the products differ, the more competing discrepancies must be juggled and the more often the subject loses the thread and simply guesses: P(guess) = 1 - (1 - eps0) * rho^(m-2). Two-row comparisons are made nearly cleanly; five- and six-row comparisons drift toward chance. Difficulty grows with the *number* of discrepancies, not with their diagnosticity — which is why unanimous (dominance) trials are far from ceiling and get worse as more experts speak.

The rule is stationary: there is no feedback, hence no learning across the block.

*Sharp falsifiable contrasts:* (i) hold cue content fixed and only duplicate versus de-duplicate the stated validity vector — OCO-LI predicts the anti-validity effect switches off and on; any recency-only or validity-integration account predicts no change. (ii) With all-distinct labels, put the low-validity discriminating row EARLY and the high-validity row LATE: OCO-LI predicts a near-even split (oddness and recency now oppose each other), whereas a pure recency account predicts the late high-validity option and a Bayesian account predicts it even more strongly.

**Parameters:**
- lam: [1.50, 1.80]
- delta: [0.82, 0.98]
- eps_base: [0.25, 0.31]
- rho: [0.765, 0.82]
- tau: [0.02, 0.045]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Odd-Cue-Out Anchoring under Label Individuation (OCO-LI)
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
            flat = arr.reshape(2, -1)
            a, b = flat[0].ravel(), flat[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.995)

    # ---------- parameters ----------
    lam = float(parameters['lam'])
    delta = float(parameters['delta'])
    eps0 = float(np.clip(float(parameters['eps_base']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    tau = float(max(float(parameters['tau']), 1e-6))

    # ---------- label individuation (design-level gate) ----------
    if n <= 1:
        phi = 1.0
    else:
        D = np.abs(v.reshape(-1, 1) - v.reshape(1, -1))
        np.fill_diagonal(D, np.inf)
        d_min = np.min(D, axis=1)
        s = np.clip(d_min / tau, 0.0, 1.0)
        phi = float(np.mean(s))
    phi = float(min(max(phi, 0.0), 1.0))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])

    fav_a = diff[disc] > 0.0

    # (i) odd-cue-out anchoring: salience = recency x oddness
    j = disc.astype(float)
    log_u = lam * j + delta * np.log(np.clip(1.0 - v[disc], 1e-9, None))
    log_u = log_u - np.max(log_u)
    u = np.exp(log_u)
    tot = float(np.sum(u))
    if (not np.isfinite(tot)) or tot <= 0.0:
        p_anchor_a = 0.5
    else:
        p_anchor_a = float(np.sum(u[fav_a])) / tot

    # (ii) ungated fallback: all discriminating rows equally likely to anchor
    p_flat_a = float(np.sum(fav_a)) / float(m)

    p_core_a = phi * p_anchor_a + (1.0 - phi) * p_flat_a

    # ---------- conflict interference ----------
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    V_DESIGN = np.array([0.958, 0.546, 0.954, 0.542, 0.950, 0.538], dtype=float)\n    LAM = 1.5      # reading-order recency gain (shared by both accounts)\n    GAM = 0.9      # oddness exponent on (1 - validity) (shared by both accounts)\n\n    hits = 0.0\n    tot = 0.0\n\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        try:\n            a = np.asarray(a_raw, dtype=float).ravel()\n            b = np.asarray(b_raw, dtype=float).ravel()\n        except Exception:\n            continue\n        n = int(min(a.size, b.size))\n        if n == 0:\n            continue\n        a = a[:n]\n        b = b[:n]\n\n        if n == V_DESIGN.size:\n            v = V_DESIGN.copy()\n        else:\n            v = np.linspace(0.95, 0.55, n)\n        v = np.clip(v, 0.5, 0.995)\n\n        diff = a - b\n        disc = np.nonzero(diff != 0.0)[0]\n        m = int(disc.size)\n        if m < 2:\n            continue\n        signs = np.sign(diff[disc])\n        # keep only CONFLICT displays (both options win at least one row)\n        if np.all(signs > 0) or np.all(signs < 0):\n            continue\n\n        logw = GAM * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + LAM * disc.astype(float)\n        k = int(np.argmax(logw))\n        favored_a = signs[k] > 0.0\n\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        chose_a = (r == 0)\n        hits += 1.0 if (chose_a == favored_a) else 0.0\n        tot += 1.0\n\n    if tot == 0.0:\n        return float('nan')\n    return float(hits / tot)",
  "rationale": "Both theories agree on the *form* of row salience (late position x low validity), so the only thing that separates them is whether that salience DECIDES the trial (SSI-DS+ sums the rescaled weights and passes them through a steep sigmoid, so the single dominant late/low-validity row saturates the choice) or merely wins a rarely-run anchoring competition (OCO-LI: the label-individuation gate is pinned near phi ~= 0.13 by the 0.004 label spacing, so ~87% of the time the choice is a probability-matched tally of discriminating rows, which sits near chance on these conflict displays and can even oppose the salient row when it is outvoted).\n\nThe metric restricts to CONFLICT displays (bases 3-8; unanimous dominance trials are excluded because both models predict ~0.86 there and would only dilute the contrast) and scores the proportion of choices agreeing with the single most salient discriminating row under the shared salience form. Hand-computed predictions at the designs' nominal parameters: OCO-LI gives 0.51, 0.55, 0.45, 0.43, 0.43, 0.53 across the six conflict bases -> pooled ~0.48 (i.e. essentially chance, since the tally usually rules); SSI-DS+ gives 0.73, 0.86, 0.78, 0.72, 0.67, 0.72 -> pooled ~0.75 (saturated on the dominant row). A ~0.27 gap over 72 trials per subject means per-subject binomial SD ~0.06, so the contrast is both large in mean and tight within subject.\n\nCrucially this is not trivially true of my theory: OCO-LI *does* contain the same anti-validity/recency salience, and if the gate were all-or-none (or absent) OCO-LI would also score high here. Near-chance salience-consistency on conflict displays is the specific fingerprint of a graded label-individuation gate plus one-reason (non-summing) anchoring."
}
```

## Usage

```json
{
  "input_tokens": 12643,
  "output_tokens": 7547
}
```
