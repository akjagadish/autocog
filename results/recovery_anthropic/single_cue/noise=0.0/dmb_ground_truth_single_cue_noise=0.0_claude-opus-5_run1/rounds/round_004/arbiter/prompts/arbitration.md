# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
**Odd-Cue-Out Anchoring under Label Individuation (OCO-LI).**

Four claims, all about *which single row is allowed to anchor the comparison*, and about *whether the anchoring competition can run at all*.

1. **One discriminating row decides, and which one is stochastic.** On each trial the subject settles the A-vs-B comparison by anchoring on a SINGLE row on which the two products differ (a one-reason rule). No summation of evidence occurs, so a validity-weighted integrator is never the right description. Which row wins the anchoring competition varies from trial to trial (not fixed within a subject), so choices are graded within a subject and per-subject metric variance stays essentially binomial — exactly what the population variances in this corpus show.

2. **Anchor salience = ODDNESS, not diagnosticity.** The probability that row j is the anchor is proportional to u_j = exp(lam * j) * (1 - v_j)^delta.
   * *Odd-cue-out (delta > 0):* a disagreement located on a near-chance expert (v ≈ .53) is experienced as a strange, idiosyncratic fact that must be adjudicated, whereas a disagreement on a near-certain expert (v ≈ .97) is discounted as a mere echo of the general impression the subject already has. Communicated validity is therefore used with INVERTED sign. This is why choice is systematically *sub-chance* with respect to stated validity in every design where the labels are usable.
   * *Reading-order recency (lam > 0):* rows are read left-to-right and the later-read discrepancies are the ones still active in working memory when the response is emitted, so they capture the anchor more often. Position is a genuine second determinant (it is required by the 1-cue-late-versus-3-cue-early coalitions), but it is subordinate to, and — crucially — gated by, claim 3.

3. **Label-ambiguity gating (the new mechanism).** The anchoring competition presupposes that each expert can be *individuated*: the subject must be able to bind a particular stated validity to a particular column. That binding is only available when the stated validities are mutually discriminable. Define, for each expert j, its isolation d_j = min_{k≠j} |v_j - v_k| and its individuation s_j = min(1, d_j / tau); the design-level individuation is phi = mean_j s_j. With probability phi the subject runs the odd-cue-out competition above; with probability 1 - phi the columns are interchangeable in memory, no row is 'the odd one', every discriminating row is equally likely to become the anchor, and the choice probability collapses to the *proportion* of discriminating rows favouring each option (a probability-matched tally). When validity labels are duplicated or tightly clustered (e.g. [.95,.70,.55,.95,.70,.55]) phi → 0, the anti-validity and recency effects switch OFF, and a two-row conflict is decided by a coin flip. When all labels are distinct (gaps ≥ .05) phi → 1 and the anti-validity/recency anchoring runs at full strength.

4. **Conflict interference.** The more rows m on which the products differ, the more competing discrepancies must be juggled and the more often the subject loses the thread and simply guesses: P(guess) = 1 - (1 - eps0) * rho^(m-2). Two-row comparisons are made nearly cleanly; five- and six-row comparisons drift toward chance. Difficulty grows with the *number* of discrepancies, not with their diagnosticity — which is why unanimous (dominance) trials are far from ceiling and get worse as more experts speak.

The rule is stationary: there is no feedback, hence no learning across the block.

*Sharp falsifiable contrasts:* (i) hold cue content fixed and only duplicate versus de-duplicate the stated validity vector — OCO-LI predicts the anti-validity effect switches off and on; any recency-only or validity-integration account predicts no change. (ii) With all-distinct labels, put the low-validity discriminating row EARLY and the high-validity row LATE: OCO-LI predicts a near-even split (oddness and recency now oppose each other), whereas a pure recency account predicts the late high-validity option and a Bayesian account predicts it even more strongly.

`predict(parameters, state, history) -> np.ndarray`:
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

## THEORY 2 — pi_6
SSI-DS+ (Strategy Selection by Label-Code Injectivity, with Distinctive-Cue Summation and Competition-Scaled Interference).

(1) STRATEGY SWITCH. Before the block the subject reads the printed validity list as a naming code. If all labels are distinct printed symbols (0.955 vs 0.950 counts as distinct) the experts are individuable and the subject adopts a WEIGHTED mode; if any label is repeated the experts are interchangeable and the subject adopts an unweighted MAJORITY TALLY mode. The switch is all-or-none at the design level (only a small per-trial slip probability omega), not a graded function of numeric gap size.

(2) WEIGHTED MODE = COMPENSATORY SUMMATION over all discriminating rows: E = sum_j sign(a_j-b_j) * (1-v_j)^gamma * exp(lam*j), rescaled so the strongest discriminating row has magnitude 1, then p(A) = sigmoid(beta*E). Validity enters with INVERTED sign (near-certain experts are redundant echoes of the general impression; near-chance disagreements are the news that must be adjudicated) and position enters as a reading-order recency gain. Coalitions ADD, so several early distinctive rows can outvote a single later row.

(3) TALLY MODE = graded majority p(A) = sigmoid(kappa*(n_A - n_B)): 1-1 splits give exactly 0.50, 2-1 splits ~0.77 (harder than probability matching, softer than a hard majority rule).

(4) CONFLICT INTERFERENCE IS COMPETITION-SCALED (the refinement). Guessing is driven by the load of holding several discrepancies in mind while ADJUDICATING BETWEEN THEM. The load is therefore not the raw number of discriminating rows but the number of rows that actually have to be pitted against one another: when the display is unanimous (min(n_A,n_B) = 0, i.e. every discrepancy points the same way) there is nothing to adjudicate and the extra rows impose only a discounted reading load eta*(m-2) rather than the full (m-2). Formally g = 1 - (1-eps0) * rho^L with L = (m-2) on opposed displays and L = eta*(m-2) on unanimous displays, and the core probability is mixed toward 0.5 by g. This predicts that dominance accuracy degrades with the number of agreeing experts only mildly, whereas conflict displays degrade steeply with the number of opposed experts - the two are dissociated, not a single monotone function of display size.

No feedback, hence no learning; the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
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

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
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



### METRIC
Rationale:
Both theories agree on the *form* of row salience (late position x low validity), so the only thing that separates them is whether that salience DECIDES the trial (SSI-DS+ sums the rescaled weights and passes them through a steep sigmoid, so the single dominant late/low-validity row saturates the choice) or merely wins a rarely-run anchoring competition (OCO-LI: the label-individuation gate is pinned near phi ~= 0.13 by the 0.004 label spacing, so ~87% of the time the choice is a probability-matched tally of discriminating rows, which sits near chance on these conflict displays and can even oppose the salient row when it is outvoted).

The metric restricts to CONFLICT displays (bases 3-8; unanimous dominance trials are excluded because both models predict ~0.86 there and would only dilute the contrast) and scores the proportion of choices agreeing with the single most salient discriminating row under the shared salience form. Hand-computed predictions at the designs' nominal parameters: OCO-LI gives 0.51, 0.55, 0.45, 0.43, 0.43, 0.53 across the six conflict bases -> pooled ~0.48 (i.e. essentially chance, since the tally usually rules); SSI-DS+ gives 0.73, 0.86, 0.78, 0.72, 0.67, 0.72 -> pooled ~0.75 (saturated on the dominant row). A ~0.27 gap over 72 trials per subject means per-subject binomial SD ~0.06, so the contrast is both large in mean and tight within subject.

Crucially this is not trivially true of my theory: OCO-LI *does* contain the same anti-validity/recency salience, and if the gate were all-or-none (or absent) OCO-LI would also score high here. Near-chance salience-consistency on conflict displays is the specific fingerprint of a graded label-individuation gate plus one-reason (non-summing) anchoring.

Source:
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

### RESULTS
- Predicted under pi_5 (simulated): 0.4964 (var=0.0034)
- Predicted under pi_6 (simulated): 0.7364 (var=0.0031)
- Observed on real data: 0.8189 (var=0.0074)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.53, 0.72, 0.86, 0.93, 0.99]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  trial 3: A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 5: A=[1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  trial 9: A=[0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 11: A=[1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 12: A=[1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 13: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0]
  trial 14: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  trial 15: A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 16: A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 17: A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 18: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 19: A=[1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 20: A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 21: A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 22: A=[1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 23: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate SSI-DS+ (advocated) from OCO-LI (competitor) on the two axes that remain once the label gate is neutralised: (A) COMPENSATORY SUMMATION-THROUGH-A-SIGMOID versus ONE-ROW ANCHOR SAMPLING (proportional salience share), and (B) COMPETITION-SCALED versus RAW-m CONFLICT INTERFERENCE (the eta discount on unanimous displays).

VALIDITIES: [0.53, 0.72, 0.86, 0.93, 0.99]. Deliberately five experts (not six) so the exp(lam*j) recency span is small enough that validity can counteract it, and validity RISES with column index so that oddness ((1-v)^exp) and recency pull in opposite directions and the resulting salience weights are all within one order of magnitude. With lam=1.5, exp=0.9 the shared salience weights are w = [0.51, 1.43, 3.42, 8.23, 6.41] \u2014 note the non-monotone peak at column 3, which lets me build coalitions whose SUM exceeds the single largest row without any row dominating. Crucially all pairwise validity gaps are >= 0.06, i.e. above OCO-LI's tau (max .045), so phi = 1 EXACTLY: OCO-LI runs its odd-cue-out anchoring at full strength and cannot be rescued (or blamed) by the individuation gate. All gaps also exceed SSI-DS+'s LABEL_TOL = .0025, so SSI-DS+ is locked into WEIGHTED summation mode. Both models therefore use the same salience form and always agree in DIRECTION; the entire contrast is quantitative, which is what makes it sharp.

Key algebra: with S_A, S_B the salience mass on each side and M the largest single discriminating weight, OCO-LI predicts p_core = S_A/(S_A+S_B) (proportional, bounded, near-linear in the salience share) whereas SSI-DS+ predicts p_core = sigmoid(beta*(S_A-S_B)/M) (saturating, and boosted whenever the winning side is a COALITION of sub-maximal rows, because the sum is normalised by the max row rather than by the total).

24 unique pairs (12 base structures each with its exact A/B mirror), K = 4 -> 96 trials, so both models' predicted responses are perfectly key-balanced. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions below use SSI-DS+ (gamma=.90, lam=1.50, beta=3.6, eps0=.24, rho=.78, eta=.73) and OCO-LI (delta=.90, lam=1.65, eps_base=.28, rho=.79, phi=1).

BLOCK A \u2014 m=2 SALIENCE-RATIO LADDER (pairs 1-10). Exactly one row favours each option; the ratio r = w_small/w_large is swept: A2 (col3 vs col4, r~.78), A5 (col4 vs col2, r~.53), A1 (col2 vs col1, r~.42), A4 (col1 vs col0, r~.36), A3 (col3 vs col1, r~.17). Interference is constant across all five (m=2), so this isolates the response function. OCO-LI must produce a nearly LINEAR climb in the salience share: ~.52, .66, .67, .69, .78. SSI-DS+ must produce a SATURATING profile that is 10-15 points higher at intermediate ratios and then flattens: ~.64, .76, .80, .81, .84. Same ordering, very different curvature and intercept \u2014 regressing p on the salience share yields slope ~.7 for OCO-LI versus a shallow, high-intercept curve for SSI-DS+.

BLOCK B \u2014 CROWN JEWEL: COALITION SUMMATION (pairs 11-18). Here several sub-maximal rows are pitted against one large row.
\u2022 B1 (m=3): A wins cols 2+3 (sum 11.65), B wins col 4 (6.41). SSI-DS+: E = +0.64 of the max row -> p ~ .74. OCO-LI: share .60 -> p ~ .56. ~18-point gap, the largest in the design.
\u2022 B2 (m=3): A wins cols 0+1 (1.93), B wins col 2 (3.42). SSI-DS+ ~ .31, OCO-LI ~ .40 (both below chance, 9-point gap in the mirror-image direction, ruling out a global response-scaling explanation).
\u2022 B3 (m=4): A wins cols 1,2,3, B wins col 4. SSI-DS+ ~ .71 vs OCO-LI ~ .55 (~15 points): three added rows push SSI-DS+ up despite the extra interference, whereas OCO-LI's share barely moves off .6 and its raw-m guessing drags it toward chance.
\u2022 B4 (m=4): A wins cols 3+4, B wins cols 1+2. SSI-DS+ ~ .73, OCO-LI ~ .63.
Across B1/B3 versus their m=2 counterparts, SSI-DS+ predicts that ADDING agreeing rows to the winning coalition keeps or increases decisiveness, while OCO-LI predicts erosion toward chance \u2014 the summation-versus-sampling signature.

BLOCK C \u2014 UNANIMITY DECAY / eta TEST (pairs 19-24). Unanimous displays with m=2 (C1: cols 3,4), m=4 (C3: cols 1-4) and m=5 (C2: all rows). Both models coincide at m=2 (~.86-.88), which pins eps0/eps_base free of the contrast. Then SSI-DS+, whose load is discounted by eta because there is nothing to adjudicate, decays only to ~.77 (m=4) and ~.72 (m=5), while OCO-LI's raw-m interference decays to ~.72 and ~.67. A shallow unanimity decay together with steep conflict decay (Block A/B m=4 trials well below the m=4 unanimous trials) is the competition-scaled interference signature; a single monotone function of display size supports OCO-LI.

INTERPRETATION: if the m=2 ladder is compressed and high (~.64-.84) and the coalition trials B1/B3 sit ~15-18 points above what the raw salience share predicts, compensatory summation with max-normalisation is confirmed and one-row anchor sampling is falsified. If instead choice probabilities track the salience SHARE almost linearly (~.52-.78) and coalitions buy nothing beyond their proportional mass, OCO-LI wins. No pair is degenerate: on 9 of the 12 bases the models differ by 9-18 percentage points, and the unanimous bases double as the shared lapse baseline and the eta test.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
Both theories agree on the DIRECTION of preference on every conflict display in this design (the side carrying the larger salience mass), so a direction-agnostic accuracy score is not trivially true for either model; what differs is the DECISIVENESS with which that side is chosen. SSI-DS+ maps evidence through sigmoid(beta * (S_A - S_B)/max_row), which saturates and is strongly boosted whenever the winning side is a coalition of sub-maximal rows; OCO-LI maps it to the raw salience SHARE S_A/(S_A+S_B), which is bounded and near-linear, hence much closer to chance on exactly these displays. Restricting to conflict displays (min(n_A,n_B) >= 1) excludes the unanimous trials where the two accounts nearly coincide (they differ only via eta), so the metric concentrates all trials where the gap is largest. Trial-by-trial hand computation with central parameter values gives an expected pooled score of ~0.75 under SSI-DS+ versus ~0.63 under OCO-LI (largest per-pair gaps at the coalition trials 11/12: .74 vs .56, and 15/16: .71 vs .55). Because the score aggregates ~72 trials per subject and only mildly depends on the free parameters (eps0/rho/beta shift it by ~0.02-0.03), per-subject variance should be ~0.003-0.004, giving a Welch t of roughly 6-7 at N=25.

Source:
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

### RESULTS
- Predicted under pi_5 (simulated): 0.6297 (var=0.0048)
- Predicted under pi_6 (simulated): 0.7386 (var=0.0031)
- Observed on real data: 0.4178 (var=0.0051)

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
**Predicted under pi_5:** 0.2722 (var=0.0027)
**Predicted under pi_6:** 0.2560 (var=0.0021)

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
**Predicted under pi_5:** 0.1956 (var=0.0184)
**Predicted under pi_6:** 0.1639 (var=0.0160)

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
**Predicted under pi_5:** 0.2767 (var=0.0068)
**Predicted under pi_6:** 0.2928 (var=0.0057)

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
**Predicted under pi_5:** 0.1650 (var=0.0115)
**Predicted under pi_6:** 0.1450 (var=0.0086)

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
**Predicted under pi_5:** 0.7427 (var=0.0074)
**Predicted under pi_6:** 0.7773 (var=0.0049)

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
**Predicted under pi_5:** 0.5083 (var=0.0072)
**Predicted under pi_6:** 0.5033 (var=0.0076)

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
**Predicted under pi_5:** 0.4711 (var=0.0034)
**Predicted under pi_6:** 0.4419 (var=0.0028)

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
**Predicted under pi_5:** 0.4900 (var=0.0045)
**Predicted under pi_6:** 0.7540 (var=0.0026)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the ten experiments the two accounts share almost everything (inverted-validity salience, reading-order gain, a load-dependent lapse) and they are indistinguishable on Experiments 3-8 (one-reason-vs-tally adherence ~.28, tally-tie anti-validity .15-.29, dominance ~.74, duplicated-label null at .507, last-row adherence ~.44) \u2014 both track those observations well. The theories separate only on the three designs built to test HOW the salience is used.\n\nExperiments 1 and 10 (alternating near-identical labels .958/.546..., .955/.545...) are decisive against pi_5. Observed salience-consistency is .819 and .770; pi_6 predicted .736 and .754 (close), while pi_5 predicted .496 and .490 \u2014 a 28-33 point miss in both. The cause is pi_5's GRADED individuation gate: with 0.004 label gaps and tau in [.02,.045] phi collapses to ~.13, so pi_5 is forced to probability-match the raw row tally and is pinned to chance. The data say emphatically that near-identical-but-distinct labels do NOT switch the anti-validity/recency mechanism off \u2014 subjects follow the single late low-validity row even when outvoted 4-to-1. pi_5's central novel mechanism (graded numeric-gap gating) is therefore falsified, while pi_6's all-or-none printed-label injectivity switch survives (and is independently supported by Experiment 8, where exactly repeated labels do produce a clean .507 chance result that a pure anti-validity account cannot explain).\n\nExperiment 2 is a joint failure, but it is informative rather than fatal. With rising validities [.53,.72,.86,.93,.99] the observed score is .418 \u2014 BELOW chance \u2014 versus .630 (pi_5) and .739 (pi_6). The metric's weights assumed lam=1.5, which puts the peak on column 3 and lets late columns dominate. I hand-checked the alternative that recency is much WEAKER than the doubt term: with w = (1-v)*exp(0.5*j) the evidence-favoured side disagrees with the metric on 6 of the 9 conflict bases, giving ~1/3 before lapses and ~.42 after mixing toward chance \u2014 exactly the observed .418. The same weights, applied to Experiment 1, favour the metric side on 6/6 conflict bases (including the 1-vs-3 and 1-vs-4 displays), giving ~.82-.85 after the same lapse \u2014 exactly the observed .819, and likewise .77 for Experiment 10. So the whole corpus is coherent under: strongly inverted validity (the most DOUBTFUL discriminating expert rules), only a weak positional gain, decisive (saturating, not proportional) use of that evidence, an exact-label-duplication collapse to tally, and a lapse that grows with display size. pi_6 has the right architecture but the wrong recency-to-doubt ratio; pi_5 has both the wrong gate and (via one-row proportional sampling) too little decisiveness \u2014 it sits at chance precisely where the data are most extreme.\n\npi_5 is the clear loser and its distinctive mechanism is refuted, so it should be replaced. Rather than patch it, the loop is better served by a genuinely different competitor to pi_6 that can already account for all ten results.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_5 with a NON-COMPENSATORY 'maximum-doubt' account that is a sharp rival to pi_6's compensatory summation, and that is calibrated to the whole corpus.\n\nSketch \u2014 MDSC (Maximum-Doubt Single-Cue selection):\n(1) On each trial the subject scans only the rows on which A and B differ and settles the trial with the SINGLE discriminating expert whose stated validity is LOWEST, i.e. the most doubtful/most 'newsworthy' disagreement. No summation: coalitions of agreeing rows buy nothing, so being outvoted 1-vs-4 is irrelevant (this is what Experiments 1/10 show: .82/.77 in favour of the lone late low-validity row).\n(2) Row position enters only as a WEAK tie-breaker/recency gain, log u_j = delta*log(1-v_j) + lam*j with lam small (0.2-0.7, definitely NOT the 1.5 both current models assume) and delta near 1. This is the critical quantitative commitment: it makes col-5 win among the three near-tied .546/.542/.538 lows in Experiment 1 (hence ~.82) while making the low-validity EARLY columns beat the late .93/.99 columns in Experiment 2 (hence ~.42, below chance). Selection of the winning row should be a soft-max over u with a moderate temperature so the rule is decisive (near-argmax) rather than probability-matching \u2014 proportional/share-based selection produces ~.52 on Experiment 2 and is ruled out.\n(3) LABEL-DEGENERACY GATE, all-or-none and only on EXACT repetition of a printed label (tolerance ~.002, not a graded numeric-gap function): when any validity value is printed twice ([.95,.70,.55,.95,.70,.55]) the experts cannot be individuated, the doubt rule is unavailable, and choice reduces to an unweighted row tally (coin flip on 1-1 splits) \u2014 this is required by Experiment 8's .507 and must NOT fire for the .004-gap designs of Experiments 1/10.\n(4) LOAD-DEPENDENT LAPSE: p(guess) = 1 - (1-eps0)*rho^(m-2) with a fairly large eps0 (~.3) and rho ~.8, so unanimous 6-row displays land near .74 (Experiment 7) and conflict displays are compressed toward chance in the right amount to turn a 6/6 base-level hit rate into the observed .82 and a 3/9 base-level rate into the observed .42.\n\nHow it differs from both: unlike pi_5 it drops the graded numeric-gap gate (falsified) and drops proportional/one-row-sampling near-chance behaviour; unlike pi_6 it denies compensatory summation \u2014 the decisive contrast becomes coalition displays where several sub-maximal doubtful rows oppose one more doubtful row (pi_6: the coalition can win; MDSC: never). Parameter ranges must be set so that lam stays well below the doubt term's dynamic range, and predictions should be sanity-checked against all ten observed values (Exp1 ~.82, Exp2 ~.42, Exp3 ~.28, Exp4 ~.17, Exp5 ~.29, Exp6 ~.15, Exp7 ~.74, Exp8 ~.51, Exp9 ~.44, Exp10 ~.77) before submission."
}
```

## Usage

```json
{
  "input_tokens": 29500,
  "output_tokens": 16494
}
```
