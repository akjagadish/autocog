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
**Validities (n_features=6):** [0.85, 0.95, 0.65, 0.75, 0.6, 0.9]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 11: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 15: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 16: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 17: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 18: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 19: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 20: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 21: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 22: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. SWC's weights are dominated by POSITION (convex recency, coefficient lam*rho >= 0.64, plus a weak POSITIVE validity component <= 0.20); IVC's weights are a function of the validity VALUES ONLY through the distrust kernel (1-v_j)^kappa, with zero position sensitivity. Prior designs anti-correlated the two profiles via strictly DECREASING validities ([4], vs NVW) or broke them apart with non-monotone duplicated validities ([6]). This design takes the unexplored route: STRICTLY INCREASING validities v = [0.55, 0.60, 0.68, 0.80, 0.90, 0.96]. Now SWC's recency gradient and its weak validity component ALIGN (both favor late cues), so SWC's weight profile is late-heavy at EVERY corner of its box and its predictions are maximally extreme; IVC's inverted kernel is early-heavy (the 55% expert alone carries ~42% of total weight at kappa=2.4). The two weight profiles are therefore near-perfectly ANTI-ALIGNED on every conflict trial. All boxes below were verified at the corners of BOTH parameter spaces; the sign arguments are parameter-free (IVC's d for any 1-vs-1 duel is tau*(inv_A - inv_B)/Z, strictly decreasing in the early cue's validity since the kernel is monotone and flats cancel; SWC's duel d = w_j - w_6 is strictly increasing in j because BOTH vnorm_j and pos_j increase with j for every admissible parameter combination).

AXIS 1 - VALIDITY-GAP LADDER, OPPOSITE-SIGNED SLOPES (L1-L5 + mirrors). A wins exactly one early cue (c1..c5) against B's win of the last cue c6 (96%). IVC predicts a monotone DECREASING series in P(A): 0.64-0.75 (c1, 55%), 0.59-0.68 (c2), 0.55-0.60 (c3), 0.52-0.54 (c4), 0.50-0.51 (c5) - as the early opponent becomes more valid, its inverted weight shrinks toward c6's. SWC predicts a monotone INCREASING series: 0.15-0.35, 0.15-0.34, 0.15-0.36, 0.18-0.40, 0.20-0.44 - as the early opponent becomes later, its recency weight grows. Every adjacent rung has non-overlapping boxes, and the two theories demand opposite-signed slopes across the whole ladder - a parameter-free quantitative signature. L5 is the sharpest single trial: the two BEST experts (90% vs 96%) duel; IVC predicts exactly chance (both inverted weights are tiny), SWC predicts P(later expert's product) in [0.56, 0.80].

AXIS 2 - CROSSOVER COUNT AGAINST THE LAST CUE (L1, M3, M4 + mirrors). How many early-cue wins does it take to flip choice toward the early option? IVC's answer: k* = 1 (P(A) already >= 0.64 with the single 55% cue, L1). SWC's answer: k* >= 5 or never - with FOUR early cues (M3: A wins c1-c4 vs c6) SWC still predicts P(A) in [0.18, 0.44] while IVC predicts [0.82, 0.90]; with ALL FIVE early cues (M4) SWC's box only reaches [0.29, 0.61] (it straddles chance because the last cue's convex recency weight rivals five early cues combined) while IVC predicts [0.82, 0.92]. The measured crossover point separates the theories by 3-4 cues.

AXIS 3 - DOUBLE DISSOCIATION IN CONFIDENCE (W1 vs M4 + mirrors). W1: the product endorsed by the five LATER, MORE valid experts (incl. 90% and 96%) vs the single WORST expert (55%, first position). M4: the product endorsed by the five EARLIER, LESS valid experts vs the single BEST expert (96%, last position). IVC: W1 in [0.52, 0.60] (the lone 55% expert soaks up ~42% of the weight) but M4 in [0.82, 0.92] -> P(W1) - P(M4) is NEGATIVE everywhere (<= -0.21). SWC: W1 in [0.79, 0.97] but M4 in [0.29, 0.61] -> the difference is POSITIVE everywhere (>= +0.18). Opposite-signed differences with disjoint boxes on both trials; note M4 is deliberately SWC-risky (observed P(A) >= 0.65 on M4 falsifies SWC's entire space), making the design genuinely adversarial rather than one-sided.

AXIS 4 - BLOCK FLIP (B1 + mirror). A wins the three early low-validity experts, B the three late high-validity experts. IVC: [0.77, 0.90] for the early block; SWC: [0.05, 0.28]. Non-overlapping.

AXIS 5 - ADJACENT MICRO-DUELS (A1, A2 + mirrors). c3(68%) vs c4(80%) and c4(80%) vs c5(90%): small but sign-clean opposite tilts. IVC: P(earlier/lower-validity product) in [0.55, 0.58] (A1) and [0.50, 0.53] (A2); SWC: [0.42, 0.48] and [0.39, 0.45]. Parameter-free opposite signs on every adjacent duel - a micro-sign series that neither theory can reverse anywhere in its box.

E1/E1m are total-split agreement anchors establishing that subjects can perform the task. Aggregate signature: regressing P(A) on inverted-validity evidence sum((1-v_j)^2.7 * s_j) versus recency evidence sum((j/5)^3.5 * s_j) yields opposite-loading profiles, since the two evidence axes are anti-correlated by construction here. IVC predicts a validity-allegiance-determined probability profile (~8 distinct levels, invariant to which positions carry which allegiances); SWC predicts a position-ordered profile that breaks every such invariance. Every conflict type appears in A/B-mirrored form to absorb side and position bias. 24 unique pairs, K = 96 // 24 = 4 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Subjective-Weighted Counting with convex cue-order recency (SWC), deterministic-attention variant. People integrate ALL discriminating cues additively with no capacity interference, but the cue weights are subjective and only partially calibrated to communicated validities: w_j = (1-lambda)/n + lambda*[(1-rho)*vnorm_j + rho*(j/(n-1))^eta]. Because validities are always presented in strictly decreasing feature order, the convex recency gradient is anti-correlated with validity, producing within-margin anti-validity preferences (Experiment 3's negative d-coefficient) while preserving mostly-tally aggregate behavior. Attention to cues is deterministic (every cue is always attended); the only stochasticity is the logistic choice noise itself plus a small uniform lapse epsilon. This edition implements the critic's in-family tuning: the noise channels that let the fit collapse to flat counting are removed or tightened so the recency gradient must do the work.

**Parameters:**
- beta: [0.8, 1.7]
- epsilon: [0.0, 0.03]
- lambda: [0.8, 1.0]
- rho: [0.8, 1.0]
- eta: [2.5, 4.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # SWC: Subjective-Weighted Counting with convex cue-order recency
    # (deterministic-attention variant).
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SWC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: linearly decreasing validities.
        v = np.linspace(0.95, 0.55, n)

    lam = float(parameters["lambda"])
    rho = float(parameters["rho"])
    eta = float(parameters["eta"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Subjective validity component: min-max normalized validities.
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)

    # (2) Recency component: position in presentation order, convexly
    # transformed so that late cues can be strongly overweighted.
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)

    # Subjective weights: mixture of flat counting, validity-following,
    # and recency-following. Because validities are presented in strictly
    # decreasing feature order, the recency part is anti-correlated with
    # validity -- the source of within-margin anti-validity preferences.
    g = (1.0 - rho) * vnorm + rho * pos
    w = (1.0 - lam) / n + lam * g

    # (3) Deterministic attention: every cue is attended on every trial
    # (per-cue attention lapse removed -- real subjects are highly
    # consistent, and stochastic lapses were the dominant noise source
    # inflating between-subject variance).
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    # NO capacity interference: beta is design-independent.
    d = float(np.sum(w * s))

    # Numerically stable logistic.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** INVERTED-VALIDITY COUNTING (IVC) — distrust-weighted tallying. Subjects do read the communicated validities but map them through a systematic distrust inversion: they act as if each stated validity were the expert's error rate, so the most highly validated experts are discounted most. Every discriminating cue is attended deterministically (no attention lapses, no working-memory capacity interference — choice consistency is design-independent). Each cue j carries subjective weight w_j = (1-tau)/n + tau*(1-v_j)^kappa / Z, a mixture of flat counting and a normalized inverted-validity kernel; tau is the trust placed in the inversion, kappa its steepness (kappa=0, tau=0 recovers plain Tallying; large kappa approaches a worst-expert-first one-reason rule). Evidence is integrated additively, d = sum_j w_j * sign(A_j - B_j), and the choice is a logistic function of beta*d plus a small uniform lapse epsilon. Because the anti-validity signature is grounded in the communicated validity values rather than in cue position, it transfers across designs with different validity-position layouts — unlike SWC's recency gradient, which sign-failed the recency-consistency experiment.

**Parameters:**
- tau: [0.85, 1.0]
- kappa: [2.4, 3.0]
- beta: [1.8, 2.5]
- epsilon: [0.03, 0.07]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # IVC: Inverted-Validity Counting (distrust-weighted tallying).
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"IVC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])
    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Distrust inversion: treat the communicated validity as if it
    # were the expert's error rate. (1 - v_j)^kappa is small for highly
    # validated experts and large for weakly validated ones; Z normalizes
    # the kernel so the mixture weights sum to one.
    inv = np.power(1.0 - v, kappa)
    Z = float(inv.sum())
    if Z > 1e-12:
        w_inv = inv / Z
    else:
        w_inv = np.full(n, 1.0 / n)

    # Subjective weights: mixture of flat counting and inverted validity.
    # tau = 0 recovers plain Tallying; tau = 1 with large kappa approaches
    # a 'worst-expert-first' one-reason rule.
    w = (1.0 - tau) / n + tau * w_inv

    # (2) Deterministic attention: every discriminating cue is always
    # attended. NO capacity interference: beta is design-independent
    # (anchor consistency stays high even at n = 8 features).
    s = np.sign(a - b)

    # (3) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Small independent uniform lapse.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"Final-Expert Dominance Index (FEDI).\n\n    Among trials where the LAST-presented expert (feature position n-1) is the\n    ONLY cue favoring one option (i.e., that option's rating vector is all\n    zeros except a 1 in the final position), compute the proportion of\n    choices that follow the final expert's recommendation.\n\n    A convex-recency weighting (SWC) predicts adherence well above 0.5\n    (the last cue carries the largest subjective weight at every corner of\n    its parameter box). A distrust-inverted validity weighting (IVC)\n    predicts adherence below 0.5, because the final expert here has a high\n    stated validity (0.90) and is therefore heavily discounted.\n    \"\"\"\n    df = data.reset_index(drop=True)\n\n    A = np.array(\n        [list(map(int, x)) for x in df['option_a_ratings']], dtype=int\n    )\n    B = np.array(\n        [list(map(int, x)) for x in df['option_b_ratings']], dtype=int\n    )\n    r = df['response'].to_numpy(dtype=int)\n\n    if A.ndim != 2 or A.shape[0] == 0:\n        return float('nan')\n\n    n_feat = A.shape[1]\n    last = n_feat - 1\n\n    a_last = A[:, last]\n    b_last = B[:, last]\n\n    # Last expert discriminates between the options.\n    disc = a_last != b_last\n\n    # The option endorsed by the final expert wins ONLY that final cue\n    # (its rating vector sums to 1). This selects the single-cue and\n    # multi-cue 'ladder' duels against the last expert (trials 1-14 and\n    # their mirrors) and excludes trials where the last-cue option also\n    # wins other cues.\n    a_sel = disc & (a_last == 1) & (A.sum(axis=1) == 1)\n    b_sel = disc & (b_last == 1) & (B.sum(axis=1) == 1)\n    sel = a_sel | b_sel\n\n    if not np.any(sel):\n        return float('nan')\n\n    # response: 0 = chose A, 1 = chose B.\n    # Did the subject follow the final expert's endorsed option?\n    followed = np.where(a_sel[sel], r[sel] == 0, r[sel] == 1)\n\n    return float(np.mean(followed))",
  "rationale": "FEDI = P(follow the last-presented expert) over the 14 unique 'last-expert duels' (trials 1-14 plus mirrors; 56 trials per subject), where the option endorsed by the final expert wins only that cue (selection rule: last feature discriminates and that option's rating vector sums to 1).\n\nWhy this contrast is sharp and sign-clean across the FULL parameter boxes:\n\nSWC weights: w_j = (1-lam)/6 + lam*[(1-rho)*vnorm_j + rho*(j/5)^eta]. At every corner (lam in [0.8,1], rho in [0.8,1], eta in [2.5,4.5]) the last cue's weight w_6 strictly dominates every other single weight and even the sum of the first four (e.g., corner lam=1,rho=1,eta=4.5: w=[0, .001, .016, .106, .366, 1]; corner lam=.8,rho=.8,eta=2.5: w=[.148, .204, .121, .281, .400, .810]). So SWC's P(follow last expert) is above 0.5 on every selected trial type: single-cue duels ~0.58-0.85, M3 (four early cues vs c6) ~0.51-0.74, M4 (five early cues vs c6) ~0.42-0.71. Pooled SWC FEDI \u2248 0.68 (box range ~0.57-0.79).\n\nIVC weights: w_j = (1-tau)/6 + tau*(1-v_j)^kappa/Z with v_6 = 0.90, so the last expert is nearly discarded (w_6 \u2248 0.01-0.04 of ~0.17-0.25 total), while the early low-validity experts (v=0.65, 0.60) soak up most weight. IVC's P(follow last expert) is below 0.5 on every selected trial type: single-cue duels ~0.30-0.51, M3 ~0.23-0.29, M4 ~0.08-0.16. Pooled IVC FEDI \u2248 0.36 (box range ~0.32-0.39, very tight because the inverted kernel is steep and tau is near 1).\n\nStatistical properties at N=25 humans: between-theory mean gap \u2248 0.32; per-subject the metric averages 56 Bernoulli trials (binomial sd \u2248 0.063), plus modest parameter heterogeneity (SWC sd \u2248 0.07, IVC sd \u2248 0.02), giving per-subject sd \u2248 0.09 under SWC and \u2248 0.07 under IVC. Welch t \u2248 13, p << 0.001 \u2014 comfortably below alpha = 0.01 even at the most adversarial corners of either box (worst-case gap \u2248 0.18-0.25 still yields t \u2248 7-9). The metric works identically on the pooled DataFrame and on each single subject's 96-trial slice (each subject contributes exactly 56 selected trials), so between_subject_variance is well defined.\n\nInterpretation: FEDI is a direct behavioral measure of how much the final, most-recently-presented expert drives choices when it is opposed by earlier experts. SWC (recency-weighted counting) demands high adherence; IVC (distrust inversion of the communicated 90% validity) demands low adherence. The two theories' predictions are non-overlapping on essentially every selected trial type."
}
```

## Usage

```json
{
  "input_tokens": 6066,
  "output_tokens": 22321,
  "reasoning_tokens": 20898,
  "provider": "Sail Research"
}
```
