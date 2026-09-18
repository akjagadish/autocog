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



## ADVOCATED THEORY
**Description:** **Amplified-Validity Additive Integration (AVAI).**

People do not stop at one reason (Take The Best) and do not count cues as equal (Tallying). Instead, on every trial they read all expert ratings, convert each communicated validity v_j into a subjective *pull* w_j = v_j^gamma, and form a single continuous evidence difference d = sum_j w_j (a_j - b_j). Choice is a logistic function of that continuous difference, p(A) = (1-eps)*sigmoid(beta*d) + eps/2.

Three claims give the theory its empirical bite:

1. **No stopping rule.** Every cue enters the sum, so a single high-validity cue can be outvoted by a coalition of weaker cues. This is why, when the top cue points one way and three or four lesser cues point the other, people overwhelmingly follow the coalition (unlike TTB).

2. **Unequal but compressed weights.** Because all validities lie in (0.5, 1], raising them to a modest power gamma (~2) yields weights that are all of the same order of magnitude (roughly 0.2-0.9). Integration therefore *looks* like counting most of the time, but with a systematic tilt toward the option holding the more valid cues. Crucially, the exponent gamma is a *subjective amplification* of stated validity, not the normatively correct log-odds transform: people over-weight validity relative to plain counting yet massively under-weight it relative to Bayes (log(v/(1-v)) would make one 0.95 cue beat two mid cues; humans do not do this). gamma is the single psychological parameter that locates a person on the tallying-to-lexicographic continuum.

3. **Magnitude sensitivity.** Because the logistic acts on the raw weighted difference rather than on a binarised winner code, trials differ *gradedly* in confidence. Unit-tally ties are not forced to chance: they resolve toward whichever side owns the higher-validity cues, but only weakly when the weighted margin is small (e.g. 0.87+0.55 vs 0.78+0.70 stays near chance) and strongly when it is large (0.93 vs 0.60 is near-deterministic). Symmetrically, trials with a big count margin but a small weighted margin become noisy rather than deterministic.

Residual attentional lapses are captured by an epsilon-rate uniform guess. No learning occurs across trials (there is no feedback), so the rule is applied stationarily.

**Parameters:**
- gamma: [1.8, 2.7]
- beta: [4.5, 8.5]
- epsilon: [0.0, 0.04]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    correct = []\n    for _, row in data.iterrows():\n        try:\n            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()\n            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()\n        except Exception:\n            continue\n        n = int(min(a.size, b.size))\n        if n == 0:\n            continue\n        d = a[:n] - b[:n]\n        nz = d[d != 0.0]\n        if nz.size == 0:\n            continue\n        # dominance trial: every discriminating cue favours the same option\n        if not (np.all(nz > 0) or np.all(nz < 0)):\n            continue\n        dominant_is_a = bool(np.all(nz > 0))\n        try:\n            resp = int(row['response'])\n        except Exception:\n            continue\n        chose_a = (resp == 0)\n        correct.append(1.0 if (chose_a == dominant_is_a) else 0.0)\n\n    if len(correct) == 0:\n        return 0.5\n    return float(np.mean(correct))",
  "rationale": "This is the dominance-accuracy metric: the proportion of trials on which the subject picks the option favoured by EVERY discriminating expert row (trials 13-18 here: |D| = 2, 4, and 6, all cues agreeing). Under AVAI the weighted evidence difference on such trials is large and grows with the number of agreeing cues, so choice is at ceiling (p ~ 0.98-1.00 for any admissible gamma/beta, limited only by the small epsilon lapse). Under RDS the choice on a dominance trial is correct only when the subject does not lose the thread: the conflict-interference term forces guessing at rate 1-(1-eps0)*rho^(m-2), giving ~0.86 at |D|=2, ~0.72 at |D|=4 and ~0.63 at |D|=6, i.e. a pooled value near 0.74. The metric therefore has a ~0.25 mean gap. It is also tight per subject: 30 trials contribute per subject, and both theories' parameter ranges move the expected value by only a few points, so per-subject variance is essentially binomial (~0.007 for RDS, ~0.0003 for AVAI), which comfortably clears Welch's test at N=25. It is not trivially true for AVAI because unanimous-cue accuracy is a plain behavioural quantity that any one-reason or non-interfering model could equally produce at ceiling; only RDS's specific claim that difficulty grows with the COUNT of differing rows (rather than their diagnosticity) predicts a substantial sub-ceiling value here."
}
```

## Usage

```json
{
  "input_tokens": 11000,
  "output_tokens": 2596
}
```
