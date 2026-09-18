# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
**Amplified-Validity Additive Integration (AVAI).**

People do not stop at one reason (Take The Best) and do not count cues as equal (Tallying). Instead, on every trial they read all expert ratings, convert each communicated validity v_j into a subjective *pull* w_j = v_j^gamma, and form a single continuous evidence difference d = sum_j w_j (a_j - b_j). Choice is a logistic function of that continuous difference, p(A) = (1-eps)*sigmoid(beta*d) + eps/2.

Three claims give the theory its empirical bite:

1. **No stopping rule.** Every cue enters the sum, so a single high-validity cue can be outvoted by a coalition of weaker cues. This is why, when the top cue points one way and three or four lesser cues point the other, people overwhelmingly follow the coalition (unlike TTB).

2. **Unequal but compressed weights.** Because all validities lie in (0.5, 1], raising them to a modest power gamma (~2) yields weights that are all of the same order of magnitude (roughly 0.2-0.9). Integration therefore *looks* like counting most of the time, but with a systematic tilt toward the option holding the more valid cues. Crucially, the exponent gamma is a *subjective amplification* of stated validity, not the normatively correct log-odds transform: people over-weight validity relative to plain counting yet massively under-weight it relative to Bayes (log(v/(1-v)) would make one 0.95 cue beat two mid cues; humans do not do this). gamma is the single psychological parameter that locates a person on the tallying-to-lexicographic continuum.

3. **Magnitude sensitivity.** Because the logistic acts on the raw weighted difference rather than on a binarised winner code, trials differ *gradedly* in confidence. Unit-tally ties are not forced to chance: they resolve toward whichever side owns the higher-validity cues, but only weakly when the weighted margin is small (e.g. 0.87+0.55 vs 0.78+0.70 stays near chance) and strongly when it is large (0.93 vs 0.60 is near-deterministic). Symmetrically, trials with a big count margin but a small weighted margin become noisy rather than deterministic.

Residual attentional lapses are captured by an epsilon-rate uniform guess. No learning occurs across trials (there is no feedback), so the rule is applied stationarily.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Amplified-Validity Additive Integration (AVAI).
    # d = sum_j (v_j ** gamma) * (a_j - b_j); p(A) = sigmoid(beta * d),
    # mixed with an epsilon lapse. All cues are consulted (no stopping
    # rule) and the logistic acts on the CONTINUOUS weighted difference,
    # so both tie-trials and large-margin trials are graded.
    import numpy as np

    # ---- parse the stimulus into two rating vectors -------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            a = np.asarray(vals[0], dtype=float).ravel()
            b = np.asarray(vals[1], dtype=float).ravel()
    else:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            half = arr.shape[0] // 2
            a, b = arr[:half], arr[half:2 * half]
        else:
            flat = arr.reshape(2, -1)
            a, b = flat[0], flat[1]

    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]

    # ---- validities -> subjective weights ----------------------------
    v_raw = parameters.get('validities', None)
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.shape[0] < n:
            pad = np.full(n - v.shape[0], 0.55)
            v = np.concatenate([v, pad])
        v = v[:n]
    # validities are probabilities of being right; keep them in (0.5, 1]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-9)

    gamma = float(parameters['gamma'])
    gamma = max(gamma, 0.0)
    w = np.power(v, gamma)

    # ---- continuous weighted evidence difference ---------------------
    d = float(np.sum(w * (a - b)))

    beta = float(parameters['beta'])
    eps = float(parameters['epsilon'])
    eps = min(max(eps, 0.0), 1.0)

    # numerically stable two-option softmax on [beta*d, 0]
    z = np.array([beta * d, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / e.sum()

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
**Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS).**

Three claims, all about *which single cue gets to decide*, not about integration.

1. **One cue decides, but which one is stochastic.** On every trial the subject settles the comparison on a SINGLE discriminating expert row (a one-reason rule). Which row wins the competition for the decision varies from trial to trial, so choice is graded *within* a subject (this is required by the data: per-subject metric variances are essentially binomial, i.e. subjects are homogeneous but individually stochastic, ruling out fixed per-subject cue orders). The deciding row j is drawn from the discriminating set D with probability proportional to a salience weight u_j.

2. **Salience = serial recency × distinctiveness (redundancy-discounted validity).** u_j = exp(lambda*j) * (1 - v_j)^delta.
   * *Recency (lambda > 0):* rows are read top-to-bottom / left-to-right and the later-read rows are the ones still active in working memory when the choice is made, so they capture the decision. Column position, not stated diagnosticity, is the dominant driver (this is the arbiter's core claim, retained).
   * *Distinctiveness (delta > 0):* subjects treat a near-certain expert (v = .97) as *redundant* — "I already know what he will say, he just echoes the consensus" — whereas a row on which a near-chance expert (v = .53) discriminates feels like idiosyncratic, decision-relevant information that must be adjudicated. Communicated validity is therefore used, but with inverted sign: high-validity rows are discounted as uninformative repeats of the general impression. This is why choices are systematically *sub-chance* with respect to validity in every experiment.

3. **Conflict interference.** The more rows on which the two products differ, the more competing comparisons must be held simultaneously, and the more often the subject loses the thread and simply guesses: P(guess) = 1 - (1-eps0)*rho^(m-2), where m = |D|. Two-row comparisons are made almost cleanly (hence near-deterministic anti-validity choice when only a .97 row and a .53 row differ); five- and six-row comparisons collapse toward coin-flipping. Difficulty grows with the number of differences, not with their diagnosticity.

No learning, no feedback use: the rule is stationary across the block.

*Why the two extra mechanisms are needed on top of pure position-recency (justified deviation from the arbiter's exact form):* pure position models (graded recency Luce, or its Take-The-Last limit) plus a uniform lapse are mathematically unable to reproduce the observed ordering Exp4 (0.15) < Exp1 (0.28) ≈ Exp3 (0.29): Exp4's diagnostic trial is an ADJACENT single-pair contrast (column 1 vs column 2), whose raw position-based prediction (1/(1+e^lambda)) is always *higher* (closer to chance) than the multi-cue Exp1/Exp3 trials, so with a shared lapse floor Exp4 can never come out lowest. Distinctiveness fixes this (a .97-vs-.53 contrast is the most lopsided contrast in the whole corpus). Likewise pure position forces Exp2's tie-trial term to exactly 0.5 and its 1-vs-3/1-vs-5 coalition term to ~2/3, capping the Exp2 metric at 0.167 only in the noiseless limit; distinctiveness lowers the tie term (the .93 row loses to the .60 row that precedes it) so that the observed 0.174 is reachable with realistic noise. Conflict interference then supplies the trial-type-dependent floor that lets Exp1/Exp3 sit at ~0.28 while Exp4 sits at ~0.15.

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

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.93, 0.57, 0.75, 0.75, 0.57, 0.93]

**Trial pairs (n=18):**
  trial 1: A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  trial 5: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 8: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 9: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 12: A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  trial 13: A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  trial 14: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  trial 15: A=[1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 16: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 17: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate Amplified-Validity Additive Integration (AVAI, advocated) from Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS). The two models differ on three orthogonal axes: (i) POSITION-INVARIANCE \u2014 AVAI depends only on the validity-weighted difference, RDS is dominated by exp(lam*j) recency, so column position of the deciding row is everything; (ii) SIGN OF THE VALIDITY EFFECT \u2014 AVAI weights cues by v^gamma (high validity wins), RDS discounts high validity as 'redundant' via (1-v)^delta (low validity wins); (iii) EFFECT OF THE NUMBER OF DISCRIMINATING ROWS \u2014 AVAI's confidence GROWS with more agreeing cues, RDS's conflict-interference term drives choices toward chance as |D| grows.

VALIDITIES: a deliberately PALINDROMIC vector [0.93, 0.57, 0.75, 0.75, 0.57, 0.93]. This is the design's key trick: every validity level appears twice, once early and once late in the display. Validity content can therefore be held EXACTLY constant while column position is manipulated. Under AVAI (weights at gamma=2: .865, .325, .563, .563, .325, .865) any pair in which A wins column j and B wins the mirror column 5-j has weighted difference EXACTLY ZERO, so AVAI must predict exactly 50/50 for ANY gamma, beta and epsilon \u2014 a parameter-free prediction. Under RDS the same trials are strongly lopsided toward the later column (u_j = e^{1.6 j}(1-v_j)^{0.9}: u = [0.09, 2.32, 7.05, 34.9, 281.5, 272.4]).

18 unique pairs (9 base structures, each with its exact A/B mirror so both models' predicted responses are perfectly key-balanced), K = 5 repetitions = 90 trials. Non-discriminating rows are filled with matched 1s (and some matched 0s) so displays are dense and cannot be solved by counting 1s.

BLOCK 1 \u2014 PARAMETER-FREE MIRROR TIES (bases 1, 2, 5; pairs 1-4 and 9-10).
\u2022 Base 1: A wins col0 (.93), B wins col5 (.93). AVAI = .500 exactly. RDS \u2248 .14 (the late .93 row captures the decision, m=2 so interference is minimal).
\u2022 Base 2: A wins col2 (.75), B wins col3 (.75). AVAI = .500 exactly. RDS \u2248 .26 (adjacent columns \u2192 only one factor of e^lam).
\u2022 Base 5: A wins cols 4,5; B wins cols 0,1. AVAI = .500 exactly. RDS \u2248 .72 (both late rows favour A; m=4 pulls it back toward chance).
A flat 50% profile across these three, with the mirrors also at 50%, is uniquely AVAI. RDS is forced to produce an ordered spread (~.14 < ~.26 < ~.72) whose ordering is set purely by column distance and |D|.

BLOCK 2 \u2014 POSITION x VALIDITY INTERACTION (bases 3, 4; pairs 5-8). Both bases contain the identical validity contrast (.93 vs .57), only relocated.
\u2022 Base 3: A wins col0 (.93), B wins col1 (.57). AVAI p(A) \u2248 .96; RDS p(A) \u2248 .17 (DIRECTIONAL REVERSAL, both position and distinctiveness push to B).
\u2022 Base 4: A wins col5 (.93), B wins col4 (.57). AVAI p(A) \u2248 .96 \u2014 EXACTLY the same as base 3; RDS \u2248 .49 (recency of col5 and distinctiveness of col4 nearly cancel).
AVAI predicts base3 = base4; RDS predicts a ~32-point gap between two trials with identical validity content. This is the crown-jewel quantitative dissociation.

BLOCK 3 \u2014 WEIGHTED MARGIN VS LATE-CUE COALITION (base 6; pairs 11-12). A wins cols 0 and 5 (.93,.93), B wins cols 3 and 4 (.75,.57). AVAI d = +0.84 \u2192 p(A) \u2248 .99. RDS \u2248 .48 (the two late B rows roughly balance the late A row, and m=4 interference flattens it). Near-determinism vs chance on the same display.

BLOCK 4 \u2014 CONFLICT-INTERFERENCE GRADIENT (bases 7, 8, 9; pairs 13-18). All discriminating rows favour the same option, with |D| = 2, 4, 6.
\u2022 |D|=2 (cols 0,1): AVAI \u2248 1.00, RDS \u2248 .86.
\u2022 |D|=4 (cols 0,2,3,5): AVAI \u2248 1.00, RDS \u2248 .72.
\u2022 |D|=6 (all rows): AVAI \u2248 1.00, RDS \u2248 .63.
AVAI predicts accuracy that is flat-at-ceiling or weakly INCREASING with the number of agreeing cues (the weighted difference grows monotonically); RDS predicts a monotone DECREASE, because difficulty is driven by the count of differences rather than their diagnosticity. Observing >95% correct on a unanimous 6-0 trial is essentially impossible for RDS (its ceiling there is ~.63 given eps_base and rho ranges), while a drop to ~.63 on unanimous trials is impossible for AVAI. These trials also double as the lapse/beta baseline for AVAI.

No pair is degenerate: on 6 of the 9 bases the models disagree in direction or by more than 30 percentage points, and the three mirror-tie bases constitute a parameter-free test that AVAI can only pass by sitting exactly at chance.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
This is the dominance-accuracy metric: the proportion of trials on which the subject picks the option favoured by EVERY discriminating expert row (trials 13-18 here: |D| = 2, 4, and 6, all cues agreeing). Under AVAI the weighted evidence difference on such trials is large and grows with the number of agreeing cues, so choice is at ceiling (p ~ 0.98-1.00 for any admissible gamma/beta, limited only by the small epsilon lapse). Under RDS the choice on a dominance trial is correct only when the subject does not lose the thread: the conflict-interference term forces guessing at rate 1-(1-eps0)*rho^(m-2), giving ~0.86 at |D|=2, ~0.72 at |D|=4 and ~0.63 at |D|=6, i.e. a pooled value near 0.74. The metric therefore has a ~0.25 mean gap. It is also tight per subject: 30 trials contribute per subject, and both theories' parameter ranges move the expected value by only a few points, so per-subject variance is essentially binomial (~0.007 for RDS, ~0.0003 for AVAI), which comfortably clears Welch's test at N=25. It is not trivially true for AVAI because unanimous-cue accuracy is a plain behavioural quantity that any one-reason or non-interfering model could equally produce at ceiling; only RDS's specific claim that difficulty grows with the COUNT of differing rows (rather than their diagnosticity) predicts a substantial sub-ceiling value here.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.9847 (var=0.0005)
- Predicted under pi_4 (simulated): 0.7440 (var=0.0064)
- Observed on real data: 0.7427 (var=0.0173)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.95, 0.7, 0.55, 0.95, 0.7, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 2: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 7: A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 8: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 9: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 10: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 11: A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 12: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 13: A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 14: A=[0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 15: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS, advocated) from Amplified-Validity Additive Integration (AVAI, competitor). The two models differ on three orthogonal axes: (i) DIRECTION OF THE VALIDITY EFFECT \u2014 AVAI weights each row by v^gamma so the option holding the more valid experts wins, whereas RDS discounts high validity as redundant via (1-v)^delta so the option holding the LESS valid discriminating expert wins; (ii) COLUMN POSITION \u2014 AVAI is completely permutation-invariant, RDS is dominated by exp(lam*j) recency, so the later of two discriminating rows almost always captures the decision; (iii) NUMBER OF DISCRIMINATING ROWS \u2014 AVAI's confidence grows monotonically as more cues agree, RDS's conflict-interference term 1-(1-eps0)rho^(m-2) drags choices toward chance as |D| grows.

VALIDITIES: a REPEATED-BLOCK vector [0.95, 0.70, 0.55, 0.95, 0.70, 0.55] (deliberately not the palindrome used in an earlier design). Every validity level occurs exactly twice \u2014 once in the early block (cols 0-2) and once in the late block (cols 3-5) \u2014 so validity content can be held EXACTLY constant while the position of the high vs. low expert is swapped. Under AVAI (weights at gamma=2: .903, .490, .303, .903, .490, .303) any pair in which A wins column j and B wins column j+3 has weighted difference EXACTLY ZERO, i.e. p(A)=.500 for ANY gamma, beta, epsilon \u2014 a parameter-free prediction. Under RDS the same displays are strongly lopsided toward the LATER column (u_j = e^{1.6j}(1-v_j)^{0.9}: u = [0.067, 1.68, 12.0, 8.19, 204, 1460]).

8 base structures, each accompanied by its exact A/B mirror (16 unique pairs, K=6 \u2192 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows are filled with matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s.

BLOCK 1 \u2014 PARAMETER-FREE CROSS-BLOCK TIES (pairs 1-4).
\u2022 Base 1: A wins col0 (.95), B wins col3 (.95). AVAI = .500 exactly. RDS \u2248 .14 (the late .95 row wins the salience competition, m=2 so interference is minimal).
\u2022 Base 2: A wins col2 (.55), B wins col5 (.55). AVAI = .500 exactly. RDS \u2248 .15.
A flat 50% profile (with mirrors also at 50%) is uniquely AVAI; a strong pull toward the option holding the LATER identical-validity row is uniquely RDS and cannot be produced by any AVAI parameterisation.

BLOCK 2 \u2014 CROWN JEWEL: HIGH-EARLY vs LOW-LATE (pairs 5-8). Both bases contain the identical validity contrast (.95 vs .55), only relocated across the two blocks.
\u2022 Base 3: A wins col0 (.95), B wins col5 (.55). AVAI d=+0.60 \u2192 p(A) \u2248 .98. RDS \u2248 .14 \u2014 a near-total DIRECTIONAL REVERSAL (recency and distinctiveness both hand the decision to the late .55 row).
\u2022 Base 4: A wins col5 (.55), B wins col3 (.95). AVAI d=-0.60 \u2192 p(A) \u2248 .02. RDS \u2248 .86 \u2014 the mirror-image reversal.
AVAI says the .95-holder is chosen ~98% in both bases regardless of where the rows sit; RDS says the option holding the LATE row is chosen ~86% in both bases regardless of its validity. These two bases alone separate the models by ~85 percentage points in opposite directions.

BLOCK 3 \u2014 WEIGHTED MARGIN vs LATE-CUE COALITION (pairs 9-10). Base 5: A wins cols 0 and 3 (both .95), B wins cols 4 (.70) and 5 (.55). AVAI d=+1.01 \u2192 p(A) \u2248 1.00 (its largest margin outside the unanimous trials). RDS \u2248 .28: the two late B rows dominate salience and m=4 interference flattens the rest. Near-determinism vs. sub-chance on the same display.

BLOCK 4 \u2014 CONFLICT-INTERFERENCE GRADIENT (pairs 11-16). All discriminating rows favour the same option, with |D| = 2 (cols 4,5), 4 (cols 0,2,3,5) and 6 (all rows).
\u2022 |D|=2: AVAI \u2248 .99, RDS \u2248 .86.
\u2022 |D|=4: AVAI \u2248 1.00, RDS \u2248 .72.
\u2022 |D|=6: AVAI \u2248 1.00, RDS \u2248 .63.
AVAI predicts accuracy flat at ceiling (weakly increasing with the number of agreeing cues, since d grows monotonically); RDS predicts a monotone DECREASE, because difficulty tracks the count of differences rather than their diagnosticity. Observing >95% correct on the unanimous 6-0 trial is essentially impossible for RDS (ceiling ~.63 given eps_base/rho ranges), while a drop to ~.63 on a unanimous display is impossible for AVAI. These trials also serve as AVAI's lapse/beta baseline.

No pair is degenerate: on 6 of the 8 bases the models disagree in direction or by more than 30 percentage points, and the two cross-block tie bases constitute a parameter-free test that AVAI can only pass by sitting exactly at chance.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Isolates the 'crown jewel' trial family (design bases 3 and 4 and their mirrors: trials 5-8) via a purely structural filter: exactly two discriminating columns, pointing to opposite options, with the LATER column carrying the LOWER validity (0.95 early vs 0.55 late). On these displays the two theories make opposite, near-deterministic predictions.

AVAI integrates weighted validity, so it must pick the option holding the 0.95 cue regardless of column position: weighted difference |d| = 0.95^gamma - 0.55^gamma ~ 0.60, beta*|d| ~ 3-5, so p(later-cue option) ~ 0.02-0.05, essentially independent of where the rows sit. RDS instead lets recency x distinctiveness decide: the later, lower-validity row has salience ratio exp(lambda*5)*(0.45)^delta vs exp(0)*(0.05)^delta, i.e. thousands to one, so the later-cue option is picked with probability ~ (1-eps0) + eps0/2 ~ 0.85 (m=2 means minimal conflict interference).

Predicted values: ~0.85 under RDS vs ~0.03 under AVAI, a gap of ~0.8. Because 24 trials per subject enter the average and both theories are near-deterministic on them, per-subject variance is tiny (binomial ~0.005 for RDS, ~0.001 for AVAI, plus negligible parameter spread), so the contrast is both large in mean and tight across subjects. Crucially it is not trivially true for RDS: any position-blind integrator (including tallying, Bayesian weighting, or plain TTB) would land far below 0.5 here, so a value near 0.85 in humans is diagnostic of position-driven, redundancy-discounted one-reason choice.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.0300 (var=0.0018)
- Predicted under pi_4 (simulated): 0.8367 (var=0.0052)
- Observed on real data: 0.5067 (var=0.0112)

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
**Predicted under pi_3:** 0.2870 (var=0.0019)
**Predicted under pi_4:** 0.2843 (var=0.0034)

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
**Predicted under pi_3:** 0.1722 (var=0.0132)
**Predicted under pi_4:** 0.1856 (var=0.0146)

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
**Predicted under pi_3:** 0.9389 (var=0.0016)
**Predicted under pi_4:** 0.3056 (var=0.0052)

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
**Predicted under pi_3:** 0.9850 (var=0.0010)
**Predicted under pi_4:** 0.1583 (var=0.0128)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments the record is lopsided. pi_3 (AVAI, validity-weighted additive integration) matches only Exp3 (0.287 vs 0.282) and Exp4 (0.172 vs 0.174), and fails catastrophically everywhere the metric is diagnostic of its core claims: Exp1 dominance accuracy (predicted 0.985, observed 0.743 \\u2014 people err on ~26% of trials where every discriminating cue agrees, which no ceiling-integration model can produce), Exp5 tally-ties (predicted 0.939, observed 0.286), Exp6 tally-ties (predicted 0.985, observed 0.150), and Exp2 crown-jewel (predicted 0.030, observed 0.507). The direction of the validity effect in AVAI is simply wrong: on weighted-tie/conflict trials subjects reliably choose the option that LOSES the highest-validity expert (0.15\\u20130.29 across three independent designs), and their accuracy on unanimous displays degrades with the number of differing rows. pi_4 (RDS) captures Exp1 (0.744 vs 0.743, including the |D|=2/4/6 conflict gradient), Exp3 (0.284), Exp4 (0.186), Exp5 (0.306 vs 0.286) and Exp6 (0.158 vs 0.150) \\u2014 five of six. Its single failure is Exp2 (predicted 0.837, observed 0.507), and it is the mechanism RDS advertises as its crown jewel: recency \\u00d7 distinctiveness on two-cue opposite-direction displays. Importantly, the Exp2 between-subject variance (0.0112) is essentially binomial for 24 trials at p=0.5, so subjects are homogeneously AT CHANCE there \\u2014 it is not a mixture of validity-followers and recency-followers. So the corpus is not 'each theory wins some'; pi_3 is dominated almost everywhere and should go. But there is a real structural puzzle worth encoding in the replacement: the anti-validity one-reason effect is near-deterministic in Exp6 (0.97-vs-0.53 rows, all six stated validities distinct) yet vanishes to exact chance in Exp2 (0.95-vs-0.55 rows) \\u2014 and Exp2 (and Exp1) are precisely the designs whose validity vectors are duplicated/palindromic ([0.95,0.70,0.55,0.95,0.70,0.55], [0.93,0.57,0.75,0.75,0.57,0.93]), while Exps 3\\u20136 all use unique validity values. Notably, in Exp1 the only measurable metric was dominance trials, where the deciding-cue rule is irrelevant and only conflict noise matters \\u2014 exactly the value RDS/any noisy one-reason model gives (0.74). This suggests the missing ingredient is not recency but a gate on whether the stated validity labels are individuating at all.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_3 (AVAI) entirely. Do NOT propose another validity-weighted integrator: the sign of the validity effect on conflict trials is empirically negative in three independent designs (Exp3 0.28, Exp5 0.29, Exp6 0.15) and unanimous-cue accuracy is far below ceiling (Exp1 0.74), both of which any additive-integration-with-lapse model must miss.\n\nProposed replacement \\u2014 'Odd-Cue-Out Anchoring with Label-Ambiguity Gating' (a sharper competitor to RDS that keeps its two successful mechanisms and fixes its one failure):\n(1) One discriminating row decides the trial (one-reason, stochastic across trials, so per-subject variance stays binomial). \n(2) Salience of a discriminating row is driven by LOW stated validity \\u2014 the near-chance expert's dissent is treated as idiosyncratic, decision-relevant information, while a near-certain expert is discounted as an echo of the general impression: u_j \\u221d (1 - v_j)^delta. Position/recency should be OMITTED or given only a small secondary weight \\u2014 it is not needed for any observed value and it is what breaks RDS on Exp2.\n(3) NEW, and the theory's empirical bite: the salience competition only operates to the extent that the communicated validity labels are mutually individuating. Define a design-level (or trial-level) discriminability of the validity set (e.g. how isolated each v_j is from the other stated validities, or an explicit penalty when a validity value is repeated in the vector). When validities are duplicated or clustered \\u2014 as in Exp2's [0.95,0.70,0.55,0.95,0.70,0.55] and Exp1's palindrome \\u2014 the subject cannot bind a validity to a particular column, salience differences wash out, and choice on conflict trials collapses to chance (\\u2248 0.507 observed in Exp2). When all validities are distinct (Exps 3\\u20136), the low-validity row wins near-deterministically on 2-cue conflicts (Exp6 = 0.15).\n(4) Retain conflict interference: guessing grows with the number of discriminating rows m, P(guess) = 1 - (1-eps0)*rho^(m-2). This is what produces Exp1's dominance profile (\\u22480.86/0.72/0.63 for m=2/4/6, pooled 0.74) and softens Exp5/Exp3 (m=4-5 conflict trials at ~0.28) relative to Exp6's near-deterministic m=2 trials (0.15).\n\nThis theory should reproduce all six observed values (Exp1 0.74, Exp2 ~0.50, Exp3 ~0.28, Exp4 ~0.17, Exp5 ~0.29, Exp6 ~0.15) and is cleanly falsifiable against pi_4: with all-distinct validities, place the LOW-validity discriminating row EARLY and the HIGH-validity row LATE. RDS (recency \\u00d7 distinctiveness) predicts the late high-validity option is chosen; the new theory predicts the early low-validity option is chosen. A second dissociation: hold cue content fixed and only duplicate versus de-duplicate the stated validity vector \\u2014 the new theory predicts the anti-validity effect switches on and off, RDS predicts no change."
}
```

## Usage

```json
{
  "input_tokens": 21410,
  "output_tokens": 14919
}
```
