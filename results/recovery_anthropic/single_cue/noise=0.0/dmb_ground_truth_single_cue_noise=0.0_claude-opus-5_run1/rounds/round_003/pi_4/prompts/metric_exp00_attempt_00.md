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
**Validities (n_features=6):** [0.955, 0.545, 0.95, 0.54, 0.945, 0.535]

**Trial pairs (n=18):**
  trial 1: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 6: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 10: A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 11: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 12: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 13: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 14: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 15: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 16: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 18: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]

**Rationale:** GOAL: dissociate RDS (advocated) from OCO-LI (competitor). The two models share their salience form exp(lam*j)*(1-v_j)^delta and their conflict-interference term 1-(1-eps0)*rho^(m-2); the ONLY structural difference is OCO-LI's label-individuation gate phi = mean_j min(1, d_j/tau), where d_j is the distance of expert j's stated validity to its nearest neighbour. With probability 1-phi OCO-LI abandons the anchoring competition and probability-matches the raw PROPORTION of discriminating rows. Any design with all-distinct labels (gaps >= .05) drives phi -> 1 and makes the models identical; a design with EXACT duplicates was already run. This design instead uses NEAR-duplicates: v = [0.955, 0.545, 0.950, 0.540, 0.945, 0.535]. Every expert's nearest neighbour is exactly 0.005 away, so d_j = 0.005 for all j and phi = min(1, 0.005/tau) is pinned to [0.11, 0.25] for tau in [0.02, 0.045] (point estimate phi ~ 0.156). This is not degenerate (all six labels are numerically distinct, and high vs low validities differ by 0.41, so the anti-validity/oddness factor is fully alive) but OCO-LI must be ~85% probability-matching while RDS runs the pure salience competition. Layout alternates high/low validity across columns (H,L,H,L,H,L), so recency and anti-validity can be aligned or opposed. RDS salience with lam=1.6, delta=0.9: u = [0.06, 2.44, 1.65, 60.4, 44.2, 1496]. Note u1 > u2 and u3 > u4: one step of oddness slightly outweighs one column of recency, a within-RDS constraint tested by base 5.

9 base structures, each with its exact A/B mirror (18 unique pairs, K = 5 -> 90 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions below use lam=1.6, delta=0.9, eps0=0.28, rho=0.78 (RDS) and the same plus tau=0.032 -> phi=0.156 (OCO-LI).

BLOCK 1 - SHARED BASELINES (bases 1-2, pairs 1-4). Base 1: unanimous m=2 (A wins cols 2,3). Base 2: unanimous m=6 (A wins every row). All discriminating rows point the same way, so p_flat = p_anchor = 1 and BOTH models predict identically (~.86 at m=2, ~.63 at m=6). These pin eps_base and rho for both models, so the contrast trials cannot be dismissed as global noise, and they test the shared claim that unanimity gets WORSE as more rows speak.

BLOCK 2 - m=2 SPLIT TIES, WHERE OCO-LI IS FORCED TOWARD CHANCE (bases 3-5, pairs 5-10). Exactly one row favours A and one favours B, so the flat tally is 1/1 = .500 and OCO-LI can move away from chance only by the factor phi <= .25.
  - Base 3 (A wins col0 .955, B wins col1 .545; adjacent, recency and oddness ALIGNED against A): RDS ~ .16, OCO-LI ~ .45.
  - Base 4 (A wins col0 .955, B wins col5 .535; maximal column distance): RDS ~ .14 (its floor eps0/2), OCO-LI ~ .45.
  - Base 5 (A wins col1 .545, B wins col2 .950; recency and oddness OPPOSED): RDS ~ .57 (oddness narrowly beats one column of recency), OCO-LI ~ .51. This base has little cross-model leverage but is the sharpest internal test of the lam/delta trade-off inside the shared salience rule, and it rules out pure position accounts.
A ~.14-.16 profile on bases 3-4 with a crossover to ~.57 on base 5 is uniquely RDS; a compressed .45/.45/.51 profile hugging chance on all three is the signature of the gate.

BLOCK 3 - CROWN JEWEL: DIRECTIONAL REVERSALS AT ASYMMETRIC m (bases 6, 7, 9; pairs 11-14, 17-18). Here the proportion of discriminating rows and the salience mass point in OPPOSITE directions.
  - Base 6 (m=3): A wins cols 0 (.955) and 2 (.950), B wins col5 (.535, last column). Proportion favours A (2/3) but 99.7% of the salience mass sits on B's late low-validity row. RDS ~ .22 (well BELOW chance); OCO-LI ~ .54 (ABOVE chance). ~32-point directional disagreement.
  - Base 7 (m=4): A wins col5 alone (.535, last), B wins cols 0,1,2. Proportion favours B (3/4); salience favours A. RDS ~ .72; OCO-LI ~ .44. Full directional reversal on one display.
  - Base 9 (m=5): A wins cols 0,1,2,3, B wins col5 alone. Proportion 4/5 for A; salience 96% for B. RDS ~ .34; OCO-LI ~ .56.
Across bases 7, 6, 9 A's share of discriminating rows rises .25 -> .67 -> .80. OCO-LI must therefore produce a MONOTONE INCREASING profile (~.44, .54, .56) tracking the tally proportion, whereas RDS produces a NON-MONOTONE profile dictated entirely by which option owns the last discriminating column (~.72, .22, .34). Monotone-in-proportion versus governed-by-the-last-column is the decisive quantitative signature and needs no free parameters to read off.

BLOCK 4 - BALANCED HIGH-m SPLITS (base 8, pairs 15-16). A wins cols 0,1 (early), B wins cols 4,5 (late); proportion is exactly 1/2 so OCO-LI's ungated component is pinned to .500, while RDS's salience is 99.8% on B's late rows. RDS ~ .28, OCO-LI ~ .47 at m=4. This isolates the position effect at a tally tie with more conflict than block 2, and jointly with base 4 it maps how far interference erodes the anti-validity/recency pull as m grows.

INTERPRETATION: if the m=2 splits sit at .14-.16 (and their mirrors at .84-.86), the 2-vs-1 base 6 sits below chance while the 1-vs-3 base 7 sits above chance, and choices track the OWNER OF THE LAST DISCRIMINATING COLUMN rather than the row count, RDS is confirmed and the label-individuation gate is falsified (with near-duplicate labels the gate should have suppressed exactly these effects). If instead every split trial collapses toward chance and the three asymmetric bases order themselves monotonically by the proportion of rows favouring A, OCO-LI's gating claim is confirmed. No pair is degenerate: on 6 of the 9 bases the models disagree by >18 points or in direction, and the two unanimous bases supply the shared lapse/interference baseline.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** **Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS).**

Three claims, all about *which single cue gets to decide*, not about integration.

1. **One cue decides, but which one is stochastic.** On every trial the subject settles the comparison on a SINGLE discriminating expert row (a one-reason rule). Which row wins the competition for the decision varies from trial to trial, so choice is graded *within* a subject (this is required by the data: per-subject metric variances are essentially binomial, i.e. subjects are homogeneous but individually stochastic, ruling out fixed per-subject cue orders). The deciding row j is drawn from the discriminating set D with probability proportional to a salience weight u_j.

2. **Salience = serial recency × distinctiveness (redundancy-discounted validity).** u_j = exp(lambda*j) * (1 - v_j)^delta.
   * *Recency (lambda > 0):* rows are read top-to-bottom / left-to-right and the later-read rows are the ones still active in working memory when the choice is made, so they capture the decision. Column position, not stated diagnosticity, is the dominant driver (this is the arbiter's core claim, retained).
   * *Distinctiveness (delta > 0):* subjects treat a near-certain expert (v = .97) as *redundant* — "I already know what he will say, he just echoes the consensus" — whereas a row on which a near-chance expert (v = .53) discriminates feels like idiosyncratic, decision-relevant information that must be adjudicated. Communicated validity is therefore used, but with inverted sign: high-validity rows are discounted as uninformative repeats of the general impression. This is why choices are systematically *sub-chance* with respect to validity in every experiment.

3. **Conflict interference.** The more rows on which the two products differ, the more competing comparisons must be held simultaneously, and the more often the subject loses the thread and simply guesses: P(guess) = 1 - (1-eps0)*rho^(m-2), where m = |D|. Two-row comparisons are made almost cleanly (hence near-deterministic anti-validity choice when only a .97 row and a .53 row differ); five- and six-row comparisons collapse toward coin-flipping. Difficulty grows with the number of differences, not with their diagnosticity.

No learning, no feedback use: the rule is stationary across the block.

*Why the two extra mechanisms are needed on top of pure position-recency (justified deviation from the arbiter's exact form):* pure position models (graded recency Luce, or its Take-The-Last limit) plus a uniform lapse are mathematically unable to reproduce the observed ordering Exp4 (0.15) < Exp1 (0.28) ≈ Exp3 (0.29): Exp4's diagnostic trial is an ADJACENT single-pair contrast (column 1 vs column 2), whose raw position-based prediction (1/(1+e^lambda)) is always *higher* (closer to chance) than the multi-cue Exp1/Exp3 trials, so with a shared lapse floor Exp4 can never come out lowest. Distinctiveness fixes this (a .97-vs-.53 contrast is the most lopsided contrast in the whole corpus). Likewise pure position forces Exp2's tie-trial term to exactly 0.5 and its 1-vs-3/1-vs-5 coalition term to ~2/3, capping the Exp2 metric at 0.167 only in the noiseless limit; distinctiveness lowers the tie term (the .93 row loses to the .60 row that precedes it) so that the observed 0.174 is reachable with realistic noise. Conflict interference then supplies the trial-type-dependent floor that lets Exp1/Exp3 sit at ~0.28 while Exp4 sits at ~0.15.

**Parameters:**
- lam: [1.4, 1.8]
- delta: [0.8, 1.0]
- eps_base: [0.24, 0.32]
- rho: [0.73, 0.83]
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
    eps0 = float(parameters['eps_base'])
    rho = float(parameters['rho'])
    eps0 = float(np.clip(eps0, 0.0, 1.0))
    rho = float(np.clip(rho, 1e-6, 1.0))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])

    # salience of each discriminating row:
    #   recency  exp(lam * column_index)
    #   distinctiveness  (1 - validity) ** delta   (redundancy discounting)
    j = disc.astype(float)
    log_u = lam * j + delta * np.log(np.clip(1.0 - v[disc], 1e-9, None))
    log_u = log_u - np.max(log_u)          # numerically stable
    u = np.exp(log_u)

    tot = float(np.sum(u))
    if not np.isfinite(tot) or tot <= 0.0:
        p_core_a = 0.5
    else:
        s_a = float(np.sum(u[diff[disc] > 0.0]))
        p_core_a = s_a / tot

    # ---------- conflict interference: guess rate grows with |D| ----------
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Design validities (fixed by the experiment); fallback handles other lengths.\n    V_DESIGN = np.array([0.955, 0.545, 0.950, 0.540, 0.945, 0.535], dtype=float)\n    LAM = 1.6\n    DELTA = 0.9\n\n    hits = []\n    for _, row in data.iterrows():\n        try:\n            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()\n            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()\n        except Exception:\n            continue\n        n = int(min(a.size, b.size))\n        if n == 0:\n            continue\n        a = a[:n]\n        b = b[:n]\n\n        if V_DESIGN.size >= n:\n            v = V_DESIGN[:n]\n        else:\n            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.545)])[:n]\n\n        d = a - b\n        disc = np.nonzero(d != 0.0)[0]\n        m = int(disc.size)\n        if m < 2:\n            continue\n\n        # salience of each discriminating row: recency x (1 - validity)\n        j = disc.astype(float)\n        logu = LAM * j + DELTA * np.log(np.clip(1.0 - v[disc], 1e-9, None))\n        logu = logu - np.max(logu)\n        u = np.exp(logu)\n        top = disc[int(np.argmax(u))]\n        top_owner_a = bool(d[top] > 0.0)  # True -> the salient row favours A\n\n        n_a = int(np.sum(d[disc] > 0.0))\n        n_b = m - n_a\n\n        # include only trials where the row-count tally does NOT strictly\n        # agree with the most salient (late / low-validity) discriminating row\n        if n_a > n_b:\n            majority_a = True\n            tie = False\n        elif n_b > n_a:\n            majority_a = False\n            tie = False\n        else:\n            majority_a = None\n            tie = True\n        if (not tie) and (majority_a == top_owner_a):\n            continue\n\n        try:\n            resp = int(row['response'])\n        except Exception:\n            continue\n        chose_a = (resp == 0)\n        hits.append(1.0 if (chose_a == top_owner_a) else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "The two theories share everything except OCO-LI's label-individuation gate, which in this near-duplicate-validity design forces phi ~ 0.16, i.e. ~85% probability matching to the raw proportion of discriminating rows. So the decisive question is: on trials where the row-count tally and the single most salient (late, low-validity) discriminating row point in different directions (or the tally is exactly tied and thus uninformative), do choices follow the salient row or hug the tally?\n\nThe metric selects exactly those conflict trials (tally tied, or tally majority opposite to the highest-salience discriminating row, using the design's known validities with lam=1.6, delta=0.9) and reports the proportion of choices that follow the salient-row owner. This selection automatically drops the two unanimous baseline bases (where both models agree) and keeps bases 3-9, ~70 of 90 trials per subject, so the metric is tight per subject (binomial with ~70 observations).\n\nQuantitative predictions from the two generative models with central parameters: RDS ~ 0.73-0.76 (salience essentially owns the decision; only the conflict-interference lapse pulls it down), OCO-LI ~ 0.50 (its 84% ungated component is pinned to the tally, which by construction opposes or is neutral w.r.t. the salient row). A ~0.24 gap with per-subject SD around 0.05-0.07 gives a very large Welch t at N=25. The metric is also mirror-symmetric (each base appears with its A/B mirror), so side bias cannot inflate or deflate it."
}
```

## Usage

```json
{
  "input_tokens": 12562,
  "output_tokens": 7679
}
```
