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
**Validities (n_features=7):** [0.6, 0.9, 0.8, 0.95, 0.65, 0.75, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  trial 2: A=[1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  trial 6: A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  trial 7: A=[0, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 1]
  trial 9: A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  trial 10: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  trial 11: A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  trial 12: A=[0, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1]
  trial 13: A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  trial 14: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 16: A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Tallying (advocated) from Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) via a design organized around a NOVEL structural centerpiece: a balanced, orthogonal 2x5 FACTORIAL GRID crossing tally margin with misbound-decider direction, plus a 'sandwich-scrambled' validity arrangement whose misbound hierarchy is non-monotone in advertised accuracy. SETUP: 7 features (no prior design used more than 6, giving a wider margin range up to +/-5) with validities [0.60, 0.90, 0.55, 0.80, 0.65, 0.75, 0.95] (f0..f6). The misbound weights w = val reversed = [0.95, 0.75, 0.65, 0.80, 0.55, 0.90, 0.60], so the misbinding cascade consults f0 -> f5 -> f3 -> f1 -> f2 -> f6 -> f4, i.e., advertised accuracies along the subjective hierarchy run 60% -> 75% -> 80% -> 90% -> 55% -> 95% -> 65%: a NON-MONOTONE sandwich in which the first-listed 60% expert is trusted most, the 95% expert is consulted SIXTH (next to last), and the 55% expert sits above it at depth 5. Tallying ignores validities entirely: p(A) = (1-eps)*sigmoid(beta*m) + eps/2, depending only on margin m. Misbinding TTB emits a constant follow probability p = (1-eps)*sigmoid(beta)+eps/2, which over its parameter ranges (beta in [0.6,1.2], eps in [0.08,0.18]) lies in [0.62, 0.75] toward the first discriminating cue in the order f0,f5,f3,f1,f2,f6,f4. SIGNATURE 1 - THE ORTHOGONAL FACTORIAL GRID (T1-T10): the misbound TOP cue f0 (advertised only 60%) always decides, toward A in T1-T5 and toward B in T6-T10 (exact A/B mirrors), while the margin sweeps -3, -1, 0, +1, +3. This is a fully balanced 2 (decider direction) x 5 (margin) design, and the two theories predict TRANSPOSED main-effect structures in the marginal means: Tallying predicts a NULL decider-row effect (both row means exactly 0.5, since margins are symmetric within each row) with a strictly graded margin-column effect passing through EXACTLY 0.5 at m=0; misbinding TTB predicts a strong decider-row effect (~0.70 vs ~0.30) with a NULL margin-column effect (every column mean exactly 0.5). A 2x5 ANOVA or model comparison on the grid cleanly awards the data to one theory: any significant margin effect with null decider effect falsifies misbinding; any significant decider effect with null margin effect falsifies Tallying. SIGNATURE 2 - SINGLE-CELL DISJOINT PREDICTION INTERVALS AT EXTREME MARGINS (T14, T15): on T14, ALL six other experts - including the 95% expert - favor A (margin +5) and only the first-listed 60% expert favors B. Misbinding follows the 60% expert to B, predicting p(A) <= 0.38; Tallying predicts p(A) = (1-eps)*sigmoid(5*beta)+eps/2 >= 0.56 even at its WORST parameters (beta=0.1, eps=0.5) - the intervals are disjoint, so this one cell alone falsifies one theory. T15 is the exact mirror (margin -5, lone 60% expert favors A): Tallying <= 0.44, misbinding >= 0.62. These cells also separate misbinding from STANDARD TTB (which follows the 95% expert to A on T14), making the design adversarial to the whole TTB family. SIGNATURE 3 - PARAMETER-FREE KNIFE EDGES WITH COMPOSITION CONTROL (T3, T8, T11): three margin-0 cells - a 2-2 tally with decider f0->A (T3), its mirror with decider f0->B (T8), and a sparse 1-1 tally with decider f0->A (T11). Tallying is pinned to EXACTLY 0.50 on all three regardless of (beta, eps) and regardless of tally composition; misbinding predicts ~0.70/~0.30 splits on all three, invariant to whether 2 or 6 cues sit on each side. SIGNATURE 4 - DECIDER-VALIDITY AND DEPTH INVARIANCE AT FIXED MARGIN (T2, T12, T13): three margin-+1 cells whose misbound deciders are the 60% first-listed expert at depth 1 (T2), the 90% expert at depth 4 (T12), and the 95% expert at depth 6 (T13, where it is the ONLY discriminating cue). Misbinding predicts the SAME follow rate on all three - the follow probability cannot depend on the decider's advertised accuracy or cascade depth; any monotone validity gradient (e.g., following the lone 95% expert more than the 60% expert) falsifies the kappa=1 misbinding claim. Tallying also predicts identical values here (same margin), so these cells isolate the competitor's internal structure. GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~[0.62,0.75] toward the misbound decider on all 15 discriminating trials, exactly 0.5 on the full tie), invariant to margin, tally composition, cascade depth, and decider validity; Tallying emits a monotone psychometric function of margin with exact 0.5 on the three margin-0 cells, exact equality within margin levels (T1=T10-margin... specifically T1=T10? No: T1 (m=+3, dec->A) = T10 (m=+3, dec->B) in p(A); T2=T9=T12=T13; T5=T6; T14 highest, T15 lowest), and values far outside the misbinding cap on T14/T15. The profiles are structurally disjoint - e.g., misbinding says p(T15)=p(T1) (both decider->A) while Tallying says p(T15) <= 0.44 < 0.56 <= p(T1); misbinding says p(T3)=p(T2) while Tallying says p(T3)=0.5 < p(T2) - so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. T16 (full tie) anchors lapse rate and side bias (both theories: exactly 0.5). With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n# Design-time constants for this experiment.\nVALIDITIES = np.array([0.6, 0.9, 0.8, 0.95, 0.65, 0.75, 0.6])\n\n\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"Cue-following deviation from chance on tally-tied (margin-0) trials.\n\n    For every trial whose tally margin is zero (A and B win on equal numbers\n    of features), find the first strictly discriminating cue under the\n    reversed-hierarchy cue order (the competing theory's subjective\n    hierarchy). Record whether the subject's choice followed that cue.\n    Tallying predicts exactly chance (0 deviation) on every tied trial,\n    for ANY (beta, epsilon). One-reason misbinding predicts a systematic\n    follow rate of p = (1-eps)*sigmoid(beta) + eps/2, i.e. a deviation of\n    roughly +0.12 to +0.25 above chance.\n    \"\"\"\n    # Reversed-hierarchy subjective weights and cue order (stable ties).\n    w = VALIDITIES[::-1].astype(float)\n    cue_order = np.argsort(-w, kind=\"stable\")\n\n    a_col = data[\"option_a_ratings\"].tolist()\n    b_col = data[\"option_b_ratings\"].tolist()\n    resp_col = data[\"response\"].tolist()\n\n    follows = []\n    for a_raw, b_raw, r_raw in zip(a_col, b_col, resp_col):\n        a = np.asarray(a_raw, dtype=float)\n        b = np.asarray(b_raw, dtype=float)\n        if a.shape != b.shape or a.ndim != 1:\n            continue\n\n        # Tally margin: number of features A wins minus number B wins.\n        margin = int(np.sum(a > b) - np.sum(b > a))\n        if margin != 0:\n            # Only tied tallies carry the parameter-free signature.\n            continue\n\n        # First discriminating cue under the reversed hierarchy.\n        decider = None\n        for j in cue_order:\n            if a[j] > b[j]:\n                decider = 0  # cue favors A\n                break\n            if b[j] > a[j]:\n                decider = 1  # cue favors B\n                break\n        if decider is None:\n            # Full tie (T16): both theories say 0.5; no information.\n            continue\n\n        resp = int(r_raw)\n        follows.append(1.0 if resp == decider else 0.0)\n\n    if len(follows) == 0:\n        return 0.0\n\n    # Deviation of the tied-trial cue-follow rate from chance.\n    return float(np.mean(follows)) - 0.5",
  "rationale": "The design contains exactly four tally-tied (margin-0) cells: T3, T8, T11, and the full tie T16. Tallying's softmax over equal counts is EXACTLY uniform for any (beta, epsilon), so its probability of choosing in the direction of any particular cue on a tied trial is exactly 0.5 \u2014 a parameter-free knife-edge. The reversed-hierarchy one-reason model instead always follows the first discriminating cue in its misbound order (f3, f5, f4, f1, f2, f0, f6): on T3 and T11 the decider points to B, on T8 to A, so its tied-trial follow rate is the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2, which lies in [0.62, 0.75] over its parameter ranges. The metric pools all tied trials that possess a discriminating cue (3 cells x 6 reps = 18 trials per subject), codes each choice as 'followed the misbound decider' or not, and returns the follow rate minus 0.5. Expected values: Tallying ~ 0.00 (per-subject variance only binomial, ~0.25/18 = 0.014); misbinding ~ +0.18 (variance ~0.013). With N=25 the Welch t-statistic is ~5.5 (p ~ 1e-6) at typical misbinding parameters, and still ~3.6 (p < 0.001) at its worst-case corner (beta=0.6, eps=0.18), safely below alpha=0.01. The metric avoids all parameter-dependent contrasts (e.g., psychometric slope over margins), because Tallying's beta range [0.1, 20] is wide and a slope-based metric could flip sign or collapse; the tied-trial contrast is the unique signature that is exactly zero under Tallying for every parameter setting, while strictly positive under one-reason deciding. It works identically on the pooled DataFrame and on each single subject's 96-trial slice, so the between-subject variance is well defined."
}
```

## Usage

```json
{
  "input_tokens": 5556,
  "output_tokens": 14995,
  "reasoning_tokens": 13870,
  "provider": "Sail Research"
}
```
