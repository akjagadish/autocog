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
**Validities (n_features=6):** [0.6, 0.95, 0.6, 0.85, 0.65, 0.9]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 2: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 13: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 15: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Rationale:** GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) from Tallying via a validity arrangement whose MISBOUND hierarchy consults the three WEAKEST experts before any of the three strongest — a 'weak-expert dictatorship' schedule no prior design used. SETUP: 6 features, validities [0.60, 0.95, 0.55, 0.85, 0.65, 0.90] (f0..f5). The misbound weights w = val reversed = [0.90, 0.65, 0.85, 0.55, 0.95, 0.60], so the misbinding cascade consults f4 (advertised 65%) -> f0 (60%) -> f2 (55%) -> f1 (95%) -> f5 (90%) -> f3 (85%): the three lowest-validity experts occupy the top three rungs of the subjective hierarchy, and the 95% expert is not consulted until depth 4. Tallying ignores validities entirely: p(A) = (1-eps)*sigmoid(beta*m) + eps/2, depending only on the tally margin m. The misbinding model emits a constant p(decider winner) = (1-eps)*sigmoid(beta)+eps/2, which over its parameter ranges (beta in [0.6,1.2], eps in [0.08,0.18]) lies in [0.62, 0.80] — a hard ceiling. SIGNATURE 1 — WEAK-EXPERT DICTATORSHIP SWEEP (T1-T5): the misbound top cue f4 (a 65% expert) always decides for A while the margin sweeps -3, -1, 0, +1, +3. Misbinding TTB predicts a FLAT p(A) in [0.62, 0.80] on all five; Tallying predicts a strictly graded monotone curve passing through EXACTLY 0.5 at T3 (m=0, parameter-free knife-edge). Single-cell non-overlap at T1 (m=-3): Tallying is mathematically <= 0.463 for ANY (beta, eps) in its ranges (max at beta=0.1, eps=0.5), while misbinding is >= 0.62 — disjoint prediction intervals. T6-T8 are the A/B mirrors (decider f4 -> B at m=+3, 0, -3): at T6 the four STRONGEST experts (95%, 90%, 85%, 60%) all favor A with margin +3, yet misbinding follows the lone 65% expert to B (p(A) in [0.20, 0.38]) while Tallying is >= 0.537 — again disjoint. T5 provides the ceiling test: unanimous-ish margin +3 where Tallying can reach ~0.97 but misbinding is capped at 0.80; human consistency above ~0.82 falsifies the advocated theory at its parameter boundary. SIGNATURE 2 — ISO-MARGIN OPPOSITE-DECIDER PAIRS (parameter-free for Tallying): (T4, T12) at m=+1, (T3, T7) at m=0, (T5, T6) at m=+3 — identical margins, opposite f4 deciders. Tallying predicts IDENTICAL p(A) within each pair for ANY parameters; misbinding predicts a ~0.70 vs ~0.30 split within each pair, straddling 0.5. SIGNATURE 3 — KNIFE-EDGE LADDER AT m=0 WITH VARYING CASCADE DEPTH (T3, T7, T9, T10, T11): five margin-0 cells pinning Tallying to EXACTLY 0.50 regardless of (beta, epsilon) — any consistent preference on any rung falsifies Tallying — while the misbound decider sits at depth 1 (f4, T3/T7), depth 2 (f0, T9), depth 3 (f2, T10), and depth 4 (f1, T11), in both directions. Misbinding predicts the same ~0.70/0.30 split at every depth: depth invariance. SIGNATURE 4 — DECIDER-VALIDITY INVARIANCE AT m=+1 (T4, T13, T14, T15): four margin-+1 cells whose misbound deciders are advertised at 65% (f4), 60% (f0), 55% (f2), and 95% (f1). Misbinding predicts the SAME follow rate on all four — the follow probability cannot depend on the decider's advertised accuracy; any monotone effect of decider validity (e.g., subjects following the 95% expert in T15 more than the 55% expert in T14) falsifies the kappa=1 misbinding claim. These are agreement cells for Tallying (identical margin), so they isolate the misbinding model's internal structure. GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~[0.62,0.80] toward the misbound decider on all 15 discriminating trials, exactly 0.5 on the full tie), invariant to margin, cascade depth, and decider validity; Tallying emits a monotone psychometric function of margin with exact 0.5 on all five margin-0 cells and exact equality within margin levels (T4=T12=T13=T14=T15; T5=T6; T1=T8). The profiles are structurally disjoint — e.g., misbinding says p(T1)=p(T5) (both decider f4->A) while Tallying says p(T1) <= 0.463 < 0.537 <= p(T5); misbinding says p(T3)=p(T4) while Tallying says p(T3)=0.5 < p(T4) — so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. T16 (full tie) anchors lapse rate and side bias. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Reversed-Hierarchy Frugality (misbinding TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude information. However, the subjective cue hierarchy is ANTI-correlated with the instructed validities: subjects misbind the communicated validity list to feature positions (binding the highest validity to the last-presented feature), so the cascade effectively runs in ascending true validity (kappa = 1, full misbinding). When no cue discriminates, the model guesses uniformly. Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice. Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope), in direct opposition to Tallying's graded curve. The noise regime is moderate: the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 sits near 0.60, matching the human follow rate implied by both experiments.

**Parameters:**
- beta: [0.6, 1.2]
- epsilon: [0.08, 0.18]
- kappa: {1.0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Reversed-Hierarchy Frugality (misbinding TTB). One-reason
    # decision making: cues are consulted one at a time and the first
    # strictly discriminating cue decides. The subjective hierarchy is
    # ANTI-correlated with the instructed validities (kappa = 1: the
    # communicated validity list is misbound to feature positions,
    # highest validity bound to the last-presented feature), so the
    # cascade runs in ascending true validity. No integration across
    # cues occurs -> flat psychometric profile across tally margins.
    # Noise: softmax (inverse temperature beta) over the binary winner
    # score, plus an independent lapse epsilon mixing in a uniform
    # choice. History is ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # Subjective validities: mixture of the instructed vector and its
    # end-to-end reversal. kappa = 1 -> fully reversed hierarchy
    # (full positional misbinding), per the theory's fixed claim.
    kappa = float(parameters["kappa"])
    w = (1.0 - kappa) * val + kappa * val[::-1]

    # Descending SUBJECTIVE validity; stable argsort so ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-w, kind="stable")

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue — pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges averages ~0.68 with per-subject
    # values spanning ~[0.62, 0.75].
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Knife-edge decider-direction contrast at balanced tally margins.\n    #\n    # For every trial we compute (a) the tally margin m = (#features A wins)\n    # - (#features B wins), and (b) the winner implied by the advocated\n    # misbinding-TTB cascade (subjective hierarchy = instructed validities\n    # reversed end-to-end: cue order f4 -> f0 -> f2 -> f1 -> f3 -> f5).\n    #\n    # We keep only trials with |m| in {0, 3} (excluding the |m|=1 cells,\n    # whose decider directions are unbalanced across margins) and where the\n    # cascade has a discriminating cue. In this subset the misbinding-A\n    # cells are {T1 (m=-3), T3 (m=0), T5 (m=+3), T9 (m=0), T10 (m=0)} and\n    # the misbinding-B cells are {T6 (m=+3), T7 (m=0), T8 (m=-3), T11 (m=0),\n    # T12 (m=0)} -- perfectly mirror-balanced in margin.\n    #\n    # Metric = P(choose A | cascade says A) - P(choose A | cascade says B).\n    #\n    # Under Tallying, p(A) = (1-eps)*sigmoid(beta*m) + eps/2, which is\n    # exactly symmetric in m. Within each group the margins are balanced\n    # (-3/+3 pairs and 0s), so BOTH group means are exactly 0.5 for ANY\n    # (beta, epsilon) -- the metric is pinned to 0 parameter-free.\n    #\n    # Under misbinding TTB, p(A) = p_follow on cascade-A cells and\n    # 1 - p_follow on cascade-B cells, so the metric equals 2*p_follow - 1,\n    # which over the advocated parameter ranges lies in ~[0.27, 0.44]\n    # (mean ~0.36) regardless of margin.\n\n    df = data.reset_index(drop=True)\n    a_lists = df['option_a_ratings'].tolist()\n    b_lists = df['option_b_ratings'].tolist()\n    resp = df['response'].to_numpy()\n\n    # Design validities and the kappa=1 reversed subjective hierarchy.\n    val = np.array([0.6, 0.95, 0.6, 0.85, 0.65, 0.9])\n    w = val[::-1]\n    cue_order = np.argsort(-w, kind='stable').tolist()  # -> [4, 0, 2, 1, 3, 5]\n\n    a_side = []  # trials where the misbound cascade favors A\n    b_side = []  # trials where the misbound cascade favors B\n\n    for i in range(len(df)):\n        av = [int(v) for v in a_lists[i]]\n        bv = [int(v) for v in b_lists[i]]\n        n = len(av)\n\n        margin = (sum(1 for j in range(n) if av[j] > bv[j])\n                  - sum(1 for j in range(n) if bv[j] > av[j]))\n        if abs(margin) == 1:\n            # Unbalanced decider directions at |m|=1; excluded so the\n            # Tallying expectation stays exactly 0.\n            continue\n\n        winner = -1\n        for j in cue_order:\n            if j >= n:\n                continue\n            if av[j] > bv[j]:\n                winner = 0\n                break\n            if bv[j] > av[j]:\n                winner = 1\n                break\n        if winner == -1:\n            continue  # full-tie trial (T16): no discriminating cue\n\n        chose_a = 1.0 if resp[i] == 0 else 0.0\n        if winner == 0:\n            a_side.append(chose_a)\n        else:\n            b_side.append(chose_a)\n\n    pa = float(np.mean(a_side)) if len(a_side) > 0 else 0.5\n    pb = float(np.mean(b_side)) if len(b_side) > 0 else 0.5\n    return float(pa - pb)",
  "rationale": "The metric exploits a parameter-free knife-edge of Tallying: its choice probability depends only on the tally margin m via a symmetric sigmoid, p(A|-m) = 1 - p(A|m). I select the 10 trial cells whose tally margins are balanced across the misbinding cascade's decider direction: cascade-A cells {T1 (m=-3), T3, T9, T10 (m=0), T5 (m=+3)} and cascade-B cells {T8 (m=-3), T7, T11, T12 (m=0), T6 (m=+3)}. Within each group the margins are exactly mirrored, so under Tallying both P(choose A) group means equal 0.5 for ANY (beta, epsilon) in its ranges \u2014 the metric is identically 0 in expectation, with only binomial sampling noise (sd ~0.13 per subject over 30 trials per side). Under the advocated misbinding TTB, every one of these cells is decided by the first cue of the reversed hierarchy, so P(choose A | cascade-A) = p_follow and P(choose A | cascade-B) = 1 - p_follow, giving metric = 2*p_follow - 1, which lies in ~[0.27, 0.44] across the advocated beta/epsilon ranges (mean ~0.36) and is flat in margin. The between-theory mean gap (~0.36) is large relative to per-subject noise (~0.12-0.13 sd), so with N=25 subjects Welch's t is expected around 8-11, far below alpha=0.01. The metric works identically on a single subject's slice (each subject completes all 16 cells x 6 reps, so both groups are always populated), giving a well-defined between-subject variance. Unlike an overall agreement rate (which extreme-beta Tallying could push toward 0.6, eroding the gap) or a single-cell contrast (noisy), this contrast cannot be attenuated by any Tallying parameterization because Tallying's prediction is exactly 0 by symmetry."
}
```

## Usage

```json
{
  "input_tokens": 5251,
  "output_tokens": 10331,
  "reasoning_tokens": 8875,
  "provider": "Sail Research"
}
```
