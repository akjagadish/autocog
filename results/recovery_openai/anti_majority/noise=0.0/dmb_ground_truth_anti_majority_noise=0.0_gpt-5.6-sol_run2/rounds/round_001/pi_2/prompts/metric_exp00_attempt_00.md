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
**Validities (n_features=7):** [0.95, 0.86, 0.77, 0.68, 0.61, 0.56, 0.51]

**Trial pairs (n=24):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 13: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 17: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  trial 18: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0]
  trial 19: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 20: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0]
  trial 21: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]
  trial 22: A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 23: A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  trial 24: A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Rationale:** This design targets the competitor's count-gating discontinuity rather than merely contrasting positive versus negative interpretations of 1. There are four matched triplets, indexed by the validity rank (1 through 4) of the highest-validity discriminating expert. Within every base triplet, that expert gives B a 1 and A a 0. Lower-validity A-favoring ratings are then added one at a time, producing tally/count margins for A of 0, +1, and +2 while leaving the highest-validity discriminating rating unchanged. Tallying predicts P(A)=0.5 at margin 0 and a monotonic increase above 0.5 at margins +1 and +2. Count-Gated Markedness Minimization instead favors A at margin 0 because its validity tiebreak treats B's mark on the best discriminating expert as adverse; as soon as one additional low-validity A mark creates a nonzero count difference, the count gate activates and the theory reverses to favoring B, increasingly strongly at margin +2. Thus, adding a nominally favorable low-validity rating causes no reversal under Tallying but a qualitative A-to-B reversal under the competitor. The same construction at four validity ranks shows that the effect is tied to the proposed gate rather than one expert's screen location. Every pair is also presented with A and B exchanged, balancing response side. The 24 unique pairs are repeated four times for 96 trials.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



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
**Description:** Count-Gated Markedness Minimization theory: decision makers do not automatically interpret a binary 1 as a benefit. Instead, they can adopt a stable markedness schema in which 1 denotes a defect, burden, or potentially misleading positive claim. They first compare the number of marked features and prefer the option with fewer marks, with larger count differences producing stronger preferences. Validity is deliberately ignored at this primary stage. Only when marked counts tie do they consult the highest-validity discriminating feature, treating the option marked on that feature as worse. A subject-level defect-schema-strength parameter represents stable conviction that the task's marked state is adverse rather than an arbitrary free sign; residual activation of the conventional benefit schema produces structured polarity uncertainty. Choice precision and occasional lapses add response variability. The theory predicts that complement-coding the stimuli or explicitly defining 1 as beneficial rather than defective should systematically reverse its choices.

**Parameters:**
- validities: validities
- count_precision: [0.8, 2.4]
- validity_tiebreak_precision: [0.8, 2.4]
- defect_schema_strength: [0.6, 1.0]
- lapse_rate: [0.0, 0.05]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Count-Gated Markedness Minimization expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    count_delta = float(np.sum(a) - np.sum(b))

    if count_delta != 0.0:
        # Positive evidence means A carries greater marked-state burden,
        # and therefore supports choosing B. Validity cannot override a
        # difference in the primary marked-feature count.
        evidence_for_b = float(parameters["count_precision"]) * count_delta
    else:
        # On a count tie, inspect the most valid discriminating feature.
        # A mark on that cue is treated as a defect, reversing the usual
        # positive-polarity Take-The-Best interpretation.
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            return np.array([0.5, 0.5], dtype=float)
        order = differing[np.argsort(-validities[differing], kind="stable")]
        j = int(order[0])
        marked_delta = float(a[j] - b[j])
        evidence_for_b = (
            float(parameters["validity_tiebreak_precision"]) * marked_delta
        )

    # Symmetric logits make positive evidence favor B under the defect
    # schema. Stable schema strength mixes this representation with the
    # residual conventional representation in which a mark is beneficial.
    defect_logits = np.array(
        [-0.5 * evidence_for_b, 0.5 * evidence_for_b], dtype=float
    )
    defect_logits = defect_logits - np.max(defect_logits)
    p_defect = np.exp(defect_logits)
    p_defect /= p_defect.sum()

    schema_strength = float(parameters["defect_schema_strength"])
    p_benefit = p_defect[::-1]
    p_core = schema_strength * p_defect + (1.0 - schema_strength) * p_benefit

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Weighted choice alignment with the feature-count majority.\"\"\"\n    if data is None or len(data) == 0:\n        return 0.0\n\n    weighted_scores = []\n    weights = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        count_delta = float(np.sum(a) - np.sum(b))\n\n        # Count ties do not identify a tallying preference and are omitted.\n        if count_delta == 0.0:\n            continue\n\n        response = int(row[\"response\"])\n        # +1 when the option with more positive ratings was selected,\n        # -1 when the option with fewer positive ratings was selected.\n        choice_sign = 1.0 if response == 0 else -1.0\n        alignment = choice_sign * np.sign(count_delta)\n\n        # Margin-two trials receive more weight because both theories make\n        # more precise, oppositely directed predictions on them.\n        weight = abs(count_delta)\n        weighted_scores.append(weight * alignment)\n        weights.append(weight)\n\n    if not weights:\n        return 0.0\n    return float(np.sum(weighted_scores) / np.sum(weights))\n",
  "rationale": "This is a side-balanced, weighted count-majority alignment score. It is +1 when choices consistently favor the option winning on more binary features and -1 when they consistently favor the option carrying fewer marked features. Tallying predicts a positive value on every non-tied count comparison, whereas Count-Gated Markedness Minimization predicts a negative value because its count gate favors the lower-count option. Weighting by the absolute count margin emphasizes the margin-two conditions, where both models' choice probabilities are furthest from chance in opposite directions, while retaining margin-one observations to reduce trial noise. Count-tie trials are excluded because tallying necessarily predicts chance there and their inclusion would add substantial within-subject noise. Exchanging A and B across matched trials makes the statistic insensitive to a general response-side preference, and averaging 64 informative trials per subject should yield a stable subject-level estimate."
}
```

## Usage

```json
{
  "input_tokens": 4771,
  "output_tokens": 3283
}
```
