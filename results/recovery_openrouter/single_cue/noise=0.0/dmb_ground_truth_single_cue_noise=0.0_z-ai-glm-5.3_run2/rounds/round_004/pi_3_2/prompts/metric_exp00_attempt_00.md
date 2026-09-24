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
**Validities (n_features=6):** [0.9, 0.8, 0.95, 0.7, 0.6, 0.65]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 13: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 14: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 15: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (advocated; positional misbinding, w = val[::-1]) from Weakest-Expert-First Frugality (competing; value-based anti-validity sort, ascending validity). These two one-reason models are behaviorally IDENTICAL whenever the validity list is monotone in position (both then consult cues in ascending validity), so the design's foundation is a SCRAMBLED validity arrangement that makes the two hierarchies genuinely different permutations — the one structural axis that separates them. SETUP: 6 features with all-distinct validities [0.90, 0.55, 0.70, 0.60, 0.80, 0.95] (f0..f5). Distinctness is deliberate: it makes the competing model's free tie_break parameter irrelevant, stripping it of extra flexibility. The two hierarchies are: ADVOCATED (w = val reversed = [0.95, 0.80, 0.60, 0.70, 0.55, 0.90], descending): f0 -> f5 -> f1 -> f3 -> f2 -> f4, i.e., the misbinding model's most trusted cue is the FIRST-listed 90% expert and its second is the LAST-listed 95% expert. COMPETING (ascending validity): f1 (55%) -> f3 (60%) -> f2 (70%) -> f4 (80%) -> f0 (90%) -> f5 (95%), i.e., the weakest expert first and the two strongest experts consulted LAST. The hierarchies disagree on 12 of 15 cue pairs; crucially, whenever f0 or f5 discriminates against any of f1..f4, the advocated model follows the 90%/95% (STRONG) expert while the competing model follows the WEAKER discriminator — a head-to-head 'strong-expert vs weak-expert dictatorship' test of the two anti-validity claims. SIGNATURE 1 — STRONG-VS-WEAK DISSOCIATION LADDER (T1-T13): on T1-T8 the first-listed 90% expert (f0, the advocated model's top cue, consulted 5th by the competitor) favors one option while a single weaker expert (55%, 70%, 60%, or 80%) favors the other; on T9-T13 the last-listed 95% expert (f5, the advocated model's second cue, consulted LAST by the competitor) plays the same role. Advocated predicts p(A) = (1-eps)*sigmoid(beta)+eps/2 in [0.62, 0.75] toward the strong expert on every rung; competing predicts the same quantity toward the weak expert, i.e., p(A) in [0.20, 0.40]. THE PREDICTION INTERVALS ARE DISJOINT ON EVERY ONE OF THE 13 DISSOCIATING CELLS: any cell with p(A) > 0.55 falsifies the weak-expert-first theory; any cell with p(A) < 0.45 falsifies the misbinding theory. T1-T12 are exact A/B mirror pairs, ruling out side bias. SIGNATURE 2 — DECIDER-VALIDITY GRADIENT (parameter-free direction test): across T1, T3, T5, T7, T9, T11, T13 the advocated decider is always the highest-validity discriminator (90% or 95%) while the competing decider sweeps 55-80%. If subjects' choices track the STRONG expert's direction, the competing theory's core claim (subjective weights = -validity) is falsified regardless of (beta, eps, tie_break); if they track the WEAK expert's direction, the kappa=1 positional-misbinding claim is falsified. Both models additionally predict follow-rate INVARIANCE across which weak/strong expert decides (flat follow rate across rungs) — a shared internal check. SIGNATURE 3 — COALITION CONFLICT (T14): the 90% expert alone favors A while the 55%, 60%, and 70% experts favor B (tally margin -2). Advocated follows f0 -> A; competing follows f1 -> B; Tallying also says B. This cell simultaneously dissociates the advocated model from the competing theory AND from any margin-integrating strategy, and tests the shared one-reason claim p(T14) = p(T1) (same deciders, different margin and cascade context). SIGNATURE 4 — CONTROLS: T15 (only the 60% expert discriminates) is an agreement cell where both theories follow the same decider at different cascade depths (advocated depth 4, competing depth 2) — it anchors the follow rate and tests both models' depth-invariance claim; any difference between p(T15) and the dissociation-cell follow rates falsifies the flat-follow assumption of whichever model mispredicts it. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse rate and side bias. GLOBAL PROFILES: the advocated model emits p(A) ≈ [0.62, 0.75] on T1,T3,T5,T7,T9,T11,T13,T14,T15 and ≈ [0.25, 0.38] on T2,T4,T6,T8,T10,T12, with 0.50 on T16; the competing model emits the EXACT TRANSPOSE (near-perfect anti-correlation on 14 of 16 cells, agreement only on T15 and T16). Because the intervals are disjoint cell-by-cell, model comparison on the full 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each — and even the AGGREGATE choice rate across the 12 mirror cells (advocated: strong-expert direction ~70%; competing: weak-expert direction ~70%) is diagnostic. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

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
**Description:** Weakest-Expert-First Frugality (value-based anti-validity TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude/tally information. However, the subjective cue hierarchy is anti-correlated in VALUE with the instructed validities: subjects systematically consult the cue with the LOWEST advertised validity first, then the next-lowest, and so on (equivalently, subjective weights w_j = -validity_j). Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope). When no cue discriminates, the model guesses uniformly. Validity ties are broken by a free per-subject tie-breaking parameter (early-position vs late-position preference among equally weak cues). Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice, giving an implied cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 that spans roughly 0.54 to 0.86 across subjects with a mean near 0.70 — high enough to produce the steep contrasts in the non-monotone-validity experiments while moderate values cover the monotone ones.

**Parameters:**
- beta: [0.5, 1.5]
- epsilon: [0.06, 0.18]
- tie_break: {0, 1}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Weakest-Expert-First Frugality (value-based anti-validity TTB).
    # One-reason decision making: cues are consulted one at a time and
    # the first strictly discriminating cue decides. The subjective
    # hierarchy is anti-correlated in VALUE with the instructed
    # validities: the cue with the LOWEST advertised validity is
    # consulted FIRST, then the next-lowest, etc. (subjective weights
    # w_j = -validity_j). This is a VALUE-based sort (ascending
    # validity), not a positional reversal of the validity list, so in
    # experiments where the validity list is not monotone in feature
    # position the two hierarchies genuinely differ. No integration
    # across cues occurs -> flat psychometric profile across tally
    # margins. Noise: softmax (inverse temperature beta) over the
    # binary winner score, plus an independent lapse epsilon mixing in
    # a uniform choice. History is ignored (no feedback in this task).
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

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    tie_break = float(parameters["tie_break"])

    # Weakest-expert-first hierarchy: sort cues by ASCENDING validity
    # value. Ties in validity are broken by the free tie_break
    # parameter: 0 -> earlier feature position first (stable),
    # 1 -> later feature position first among equally weak cues.
    # np.lexsort uses the LAST key as primary: primary = val ascending,
    # secondary = position (forward or reversed).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, val))

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

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges spans ~[0.54, 0.86] across
    # subjects with a mean near 0.70.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Dissociation metric: fraction of choices matching the advocated\n    # (Reversed-Hierarchy Frugality) cascade winner, computed ONLY on\n    # trials where the two theories' cue hierarchies yield OPPOSITE\n    # predictions. On those trials the advocated theory predicts a\n    # match rate of p = (1-eps)*sigmoid(beta) + eps/2 ~ 0.62-0.80,\n    # while the competing theory predicts the complementary rate\n    # ~ 1-p ~ 0.20-0.40, so the two theories produce visibly\n    # different numbers on this scalar.\n    val = np.array([0.9, 0.8, 0.95, 0.7, 0.6, 0.65])\n    n = len(val)\n\n    # Advocated hierarchy: subjective weights = reversed validity list.\n    w_adv = val[::-1]  # [0.65, 0.6, 0.7, 0.95, 0.8, 0.9]\n    order_adv = np.argsort(-w_adv, kind=\"stable\")  # f3, f5, f4, f2, f0, f1\n\n    # Competing hierarchy: ascending advertised validity.\n    order_comp = np.lexsort((np.arange(n), val))  # f4, f5, f3, f1, f0, f2\n\n    def cascade_winner(a, b, order):\n        for j in order:\n            if a[j] > b[j]:\n                return 0\n            if b[j] > a[j]:\n                return 1\n        return -1  # no discriminating cue\n\n    matches = []\n    for ra, rb, resp in zip(\n        data[\"option_a_ratings\"], data[\"option_b_ratings\"], data[\"response\"]\n    ):\n        a = [int(v) for v in ra]\n        b = [int(v) for v in rb]\n        r = int(resp)\n\n        wa = cascade_winner(a, b, order_adv)\n        wc = cascade_winner(a, b, order_comp)\n\n        # Keep only trials that genuinely dissociate the two theories:\n        # both cascades terminate and they pick different options.\n        if wa < 0 or wc < 0 or wa == wc:\n            continue\n\n        matches.append(1.0 if r == wa else 0.0)\n\n    if len(matches) == 0:\n        return 0.5\n    return float(np.mean(matches))",
  "rationale": "The two theories share identical one-reason mechanics and identical noise (softmax(beta) + lapse epsilon); they differ ONLY in cue order. With validities [0.9, 0.8, 0.95, 0.7, 0.6, 0.65], the advocated (positional reversal) hierarchy is f3, f5, f4, f2, f0, f1 while the competing (ascending-validity) hierarchy is f4, f5, f3, f1, f0, f2. Working through all 16 trial types, the two cascades issue OPPOSITE predictions exactly on trials T1, T2, T3, T4 and T13 (30 of 96 trials per subject): e.g. T1 pits f0 (advocated's 5th cue, competing's 5th cue in reverse) against f1, T3 pits f2 against f0, T13 pits f5 against f4. On every other trial type (T5-T8, T9-T12, T14, T15) both cascades terminate on the same cue and thus make the SAME prediction, and T16 is a full tie (both guess). The metric therefore recomputes both cascades directly from the ratings columns, keeps only the dissociating rows, and returns the fraction of responses that match the advocated cascade's winner. Expected values: under the advocated theory this equals its cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2, which over its parameter ranges (beta in [0.6,1.2], eps in [0.08,0.18]) lies in roughly [0.62, 0.80] (mean ~0.68). Under the competing theory the advocated winner is the REJECTED option, so the metric equals 1-p, which over ITS ranges (beta in [0.5,1.5], eps in [0.06,0.18]) lies in roughly [0.14, 0.46] (mean ~0.30). The mean gap (~0.38) is large relative to per-subject variability: each subject contributes 30 dissociating trials (binomial SD ~0.085), and subject-level heterogeneity in p adds only ~0.05, giving a total between-subject SD of ~0.10 and a Welch t at N=25 of roughly 12 \u2014 far beyond the alpha=0.01 threshold. Because 4 of the 5 dissociating cells come in exact A/B mirror pairs (T1/T2, T3/T4), any side bias cancels. The metric is a simple proportion, so it is equally well-defined on the pooled DataFrame and on a single subject's slice, satisfying the per-subject variance requirement. If human data are generated by the advocated misbinding process, this metric will be near 0.68 and the competing theory would have to predict ~0.30 \u2014 a falsifiably wrong forecast."
}
```

## Usage

```json
{
  "input_tokens": 5706,
  "output_tokens": 8074,
  "reasoning_tokens": 6853,
  "provider": "Sail Research"
}
```
