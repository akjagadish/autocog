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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 2: A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 3: A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 5: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 7: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 9: A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 11: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 13: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 16: A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Rationale:** This design dissociates Reversed-Hierarchy Frugality (misbinding TTB) from Tallying by exploiting their two orthogonal trial-dependent quantities: (i) Tallying's choice depends ONLY on the tally margin m = a_wins - b_wins, giving p(A) = (1-eps)*sigmoid(beta*m) + eps/2, a strictly graded psychometric curve; (ii) misbinding TTB's choice depends ONLY on which cue first discriminates when the cascade runs in ASCENDING true validity (kappa=1 binds the highest subjective validity to the LAST feature), giving a CONSTANT p(winner) = (1-eps)*sigmoid(beta) + eps/2 ~ 0.60 on every discriminating trial. I use 5 features with validities [0.90, 0.80, 0.70, 0.60, 0.55], so the reversed cascade consults the 0.55 expert first, then 0.60, 0.70, 0.80, 0.90. CENTRAL SIGNATURE 1 - margin sweep with the lowest-validity cue as fixed decider (trials 1-4 and 5-8): the 0.55 expert always favors one option while the tally margin sweeps across -3, -2, -1, 0 (and mirrored +3, +2, +1, 0). Misbinding TTB predicts a FLAT p ~ 0.60 toward the lowest-cue winner on all eight trials; Tallying predicts a strictly monotone graded curve that passes through EXACTLY 0.5 at margin 0 (trials 4, 8) - a knife-edge prediction independent of beta - and reverses sign at negative margins (trial 1: Tallying says B strongly, misbinding TTB says A). CENTRAL SIGNATURE 2 - matched-margin opposite-decider pairs, a parameter-free falsification for Tallying: pairs (2, 10), (4, 8), (6, 9), (7, 14) have IDENTICAL tally margins (-2, 0, +2, +1) but opposite lowest-cue winners. Tallying predicts IDENTICAL p(A) within each pair for ANY (beta, eps); misbinding TTB predicts a ~0.60 vs ~0.40 split within each pair. Trial 14 is especially diagnostic: it is the deepest cascade (only the 0.90 expert discriminates, all lower cues tie), and misbinding TTB predicts the SAME constant p there as on trial 1 (where the 0.55 expert decides) - cascade-depth invariance - while Tallying treats trials 7 and 14 identically (both margin +1) though misbinding TTB gives opposite choices. CENTRAL SIGNATURE 3 - reversed-hierarchy depth probes at margin 0 (trials 11, 12, 13): the decider is the 0.60, 0.70, or 0.80 expert while the tally is exactly 1-1, so Tallying is pinned to exactly 0.50 regardless of parameters while misbinding TTB stays at ~0.60/~0.40. Trial 15 is a depth-3 conflict (0.70 expert favors A, tally margin -1 favors B). Trial 5 additionally separates misbinding TTB from STANDARD TTB: the top-validity 0.90 expert and the tally both favor A, but the reversed cascade follows the 0.55 expert to B - if subjects choose A here, the misbinding claim fails. Trial 16 is a full-tie control anchoring lapse and side bias (both theories: exactly 0.5). The overall profile is non-monotone in margin (margin +3 -> p(A) ~ 0.40; margin -3 -> p(A) ~ 0.60), which no monotone psychometric function of margin - i.e., no parameterization of Tallying - can reproduce, whereas misbinding TTB predicts one constant across all 15 discriminating trials with variation only in direction. 16 unique pairs, K = 6 (96 trials exactly), give 6 repetitions per cell for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Reversed-Hierarchy Frugality (misbinding TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude information. However, the subjective cue hierarchy is ANTI-correlated with the instructed validities: subjects misbind the communicated validity list to feature positions (binding the highest validity to the last-presented feature), so the cascade effectively runs in ascending true validity (kappa = 1, full misbinding). When no cue discriminates, the model guesses uniformly. Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice. Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope), in direct opposition to Tallying's graded curve. The noise regime is moderate: the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 sits near 0.60, matching the human follow rate implied by both experiments.

**Parameters:**
- beta: [0.45, 0.55]
- epsilon: [0.11, 0.15]
- kappa: {1.0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Reversed-Hierarchy Frugality: one-reason decision making (TTB
    # cascade) run over a SUBJECTIVE cue hierarchy that is
    # anti-correlated with the instructed validities. Subjects
    # misbind the communicated validity list to feature positions
    # (highest validity bound to the last-presented feature), so the
    # cascade effectively consults cues in ascending true validity.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. History is ignored (no feedback in this task).
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
    # end-to-end reversal. kappa = 0 -> instructed hierarchy;
    # kappa = 1 -> fully reversed hierarchy (near-complete misbinding).
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

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta), mapping beta=0 to 50/50
    # guessing and large beta to deterministic choice.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    # Reversed-cascade follow rate (centered).\n    #\n    # For every trial we determine the winner implied by a one-reason\n    # cascade that consults cues in ASCENDING true validity (i.e., the\n    # 0.5-validity expert first, ..., the 0.9-validity expert last) --\n    # exactly the subjective hierarchy implied by full misbinding\n    # (kappa = 1). Full-tie trials carry no directional prediction and\n    # are dropped. The metric is the signed follow rate:\n    #   +1  -> subject always picks the reversed-cascade winner\n    #    0  -> chance\n    #   -1  -> subject always picks the opposite option\n    #\n    # Predicted values:\n    #   Misbinding TTB: p(follow) = (1-eps)*sigmoid(beta) + eps/2 ~ 0.60\n    #                   on EVERY discriminating trial  ->  metric ~ +0.21\n    #   Tallying:       on the 7 trials where the tally winner OPPOSES\n    #                   the lowest-validity cue (trials 1,2,3,5,6,7,15)\n    #                   p(follow) < 0.5 for ANY beta > 0; on the 5\n    #                   zero-margin trials (4,8,11,12,13) p(follow)\n    #                   = 0.5 EXACTLY for any (beta, eps); only the 3\n    #                   agreeing trials (9,10,14) push above 0.5.\n    #                   Hence the Tallying metric is strictly NEGATIVE\n    #                   for every parameterization in its ranges\n    #                   (approaching 0 only as beta->0 or eps->0.5).\n    validities = np.array([0.90, 0.80, 0.70, 0.60, 0.50])\n    # Ascending true validity == descending subjective validity under\n    # full misbinding: consult index 4 first, then 3, 2, 1, 0.\n    cue_order = np.argsort(validities, kind=\"stable\")\n\n    n_used = 0\n    follow = 0\n    for a_raw, b_raw, r in zip(data[\"option_a_ratings\"],\n                               data[\"option_b_ratings\"],\n                               data[\"response\"]):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        if a.shape != b.shape or a.size != validities.size:\n            continue\n        winner = None\n        for j in cue_order:\n            if a[j] > b[j]:\n                winner = 0\n                break\n            if b[j] > a[j]:\n                winner = 1\n                break\n        if winner is None:\n            # Full tie (trial 16): no directional prediction.\n            continue\n        n_used += 1\n        if int(r) == winner:\n            follow += 1\n\n    if n_used == 0:\n        return 0.0\n    return 2.0 * (follow / n_used) - 1.0",
  "rationale": "The metric is the signed rate at which subjects follow the winner implied by a cue cascade running in ASCENDING true validity (lowest-validity expert decides first) \u2014 the exact behavioral signature of Reversed-Hierarchy Frugality. It is computed purely from choices, per subject and pooled, and uses 90 of the 96 trials (all discriminating pairs; the full-tie trial 16 is dropped).\n\nPredicted separation:\n\n(1) Under misbinding TTB, every discriminating trial has the same follow probability p = (1-eps)*sigmoid(beta) + eps/2. With beta in [0.45, 0.55] and eps in [0.11, 0.15], p is tightly pinned near 0.60, so the pooled metric is ~+0.21 and each subject's value lies in a narrow band (parameter spread contributes <0.001 variance; binomial noise over 90 trials gives per-subject sd ~0.10).\n\n(2) Under Tallying, the follow rate on each trial is governed by the tally margin m = a_wins - b_wins relative to the reversed-cascade direction d. The design makes this contrast one-sided for ANY Tallying parameters: on 7 trials (1,2,3,5,6,7,15) the tally winner OPPOSES the lowest-validity cue, so p(follow) < 0.5 for any beta > 0 (deeply below 0.5 for the large beta allowed by its range, up to 20); on 5 zero-margin trials (4,8,11,12,13) Tallying is pinned to EXACTLY 0.5 for any (beta, eps) \u2014 the softmax over equal tallies is uniform and the lapse keeps it uniform; only 3 trials (9,10,14) favor the reversed-cascade winner. The Tallying metric is therefore strictly negative across its entire parameter space: about -0.27 at large beta / small eps, about -0.07 at beta = 0.1 / eps = 0, and approaching 0 only in the degenerate beta->0 or eps->0.5 limit.\n\nWorst-case gap ~0.21 (degenerate Tallying) and typical gap 0.4-0.5, against per-subject sd ~0.10 under both theories. At N = 25 subjects per group the Welch t-statistic is at least ~5-7 (p << 0.01) even in the worst-case parameter corner, and an order of magnitude larger for moderate/large Tallying beta. The metric also works unchanged on a single subject's slice (90 usable trials), so between-subject variance is well defined."
}
```

## Usage

```json
{
  "input_tokens": 4594,
  "output_tokens": 14050,
  "reasoning_tokens": 12703,
  "provider": "Sail Research"
}
```
