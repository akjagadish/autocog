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
**Validities (n_features=6):** [0.68, 0.66, 0.64, 0.62, 0.95, 0.52]

**Trial pairs (n=12):**
  trial 1: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 2: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 5: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  trial 7: A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 9: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 10: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  trial 11: A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]

**Rationale:** The design exploits a sharp conflict-gated crossover unique to CGRDA while holding Take The Best's decisive cue constant. Expert 5 has the highest validity (0.95) and always discriminates in favor of the designated TTB winner. Each family contains an exactly balanced 2-versus-2 trial and a matched 3-versus-1 trial differing only in Expert 1's direction. On balanced trials, the final discriminating expert (Expert 6) opposes Expert 5. TTB therefore chooses Expert 5's option with the same probability q = epsilon/2 + (1-epsilon)*sigmoid(beta) on every trial. CGRDA instead predicts the Expert-6 option on every balanced trial throughout its parameter ranges: tally balance fully opens the recency gate, and recency_strength is at least 1.15 while normalized core evidence is bounded in magnitude by 1. On matched imbalanced trials, reversing Expert 1 creates three endorsements for the TTB winner versus one final endorsement for its opponent. The imbalance of two suppresses recency by at least exp(-5), while the early Expert-1 endorsement and two additional concordant endorsements make CGRDA favor the TTB winner. Thus, the critical statistic is the probability of choosing the Expert-5 winner: TTB predicts invariance between balanced and imbalanced conditions, whereas CGRDA predicts a crossover from below chance on balanced trials to above chance on imbalanced trials. Moving the second pro-TTB endorsement across Experts 2, 3, and 4 provides an additional serial-position manipulation: TTB remains invariant, while CGRDA's integrated evidence changes with primacy. Exact A/B reversals control response-side bias. The 12 unique pairs are each repeated eight times, yielding 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

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

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
**Description:** Conflict-Gated Redundancy-Discounted Attention (CGRDA) proposes that people integrate all discriminating cues, but cue influence depends jointly on instructed validity, serial position, directional redundancy, and the current conflict configuration. Instructed validity and early presentation increase a cue's baseline attentional weight. Successive endorsements of an option have diminishing marginal impact because repeated evidence is treated as partially redundant. When the numbers of cues favoring the two options are closely balanced, attention is reoriented toward the final discriminating cue, producing a conflict-gated recency contribution. Thus, an early compact set of reliable cues can defeat a larger but redundant set, while the last cue can reverse the decision in balanced conflicts. Primacy, redundancy discounting, recency strength, and conflict sensitivity are stable subject-level traits.

**Parameters:**
- validities: validities
- primacy: [0.8, 1.4]
- redundancy_discount: [0.25, 1.25]
- validity_sensitivity: [0.3, 1.5]
- recency_strength: [1.15, 1.65]
- balance_sensitivity: [2.5, 5.0]
- beta: [0.8, 1.8]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CGRDA expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive differences favor B and negative differences favor A.
    diff = stim[1] - stim[0]
    signs = np.sign(diff)
    discriminating = np.flatnonzero(signs != 0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    primacy = float(parameters["primacy"])
    redundancy_discount = float(parameters["redundancy_discount"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    recency_strength = float(parameters["recency_strength"])
    balance_sensitivity = float(parameters["balance_sensitivity"])
    beta = float(parameters["beta"])

    # Convert communicated validities to reliability evidence. Normalizing
    # preserves validity rank without making the score depend arbitrarily
    # on the absolute log-odds scale selected by an experiment.
    clipped = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(clipped / (1.0 - clipped))
    reliability /= max(float(np.max(reliability)), 1e-12)
    reliability = np.maximum(reliability, 0.05)

    count_a = 0
    count_b = 0
    integrated = 0.0
    total_weight = 0.0

    for j in discriminating:
        direction = float(signs[j])
        if direction > 0:
            count_b += 1
            repetition = count_b
        else:
            count_a += 1
            repetition = count_a

        validity_weight = reliability[j] ** validity_sensitivity
        position_weight = np.exp(-primacy * float(j))
        novelty_weight = float(repetition) ** (-redundancy_discount)
        weight = validity_weight * position_weight * novelty_weight

        integrated += direction * weight
        total_weight += weight

    # Put integrated evidence on a common bounded scale across feature
    # counts and experiments.
    core_evidence = integrated / max(total_weight, 1e-12)

    # Recency is strongest at exact tally balance and falls rapidly as one
    # side acquires more endorsements. The final discriminating cue, not
    # necessarily the least-valid cue, determines the direction of this
    # conflict-resolution contribution.
    tally_imbalance = abs(count_b - count_a)
    conflict_gate = np.exp(-balance_sensitivity * float(tally_imbalance))
    last_direction = float(signs[int(discriminating[-1])])
    recency_evidence = recency_strength * conflict_gate * last_direction

    choice_evidence = core_evidence + recency_evidence
    utilities = np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])

    x = beta * utilities
    x = x - np.max(x)
    probs = np.exp(x)
    probs /= probs.sum()
    return probs.astype(np.float64)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Balanced-minus-imbalanced change in choosing Expert 5's winner.\"\"\"\n    if data is None or len(data) == 0:\n        return float(\"nan\")\n\n    interaction_scores = []\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.size < 5 or b.size != a.size:\n            continue\n\n        diff = b - a\n        signs = np.sign(diff)\n        n_b = int(np.sum(signs > 0))\n        n_a = int(np.sum(signs < 0))\n        imbalance = abs(n_b - n_a)\n\n        # The design's critical conditions are exact 2-vs-2 balance and\n        # 3-vs-1 imbalance. Ignore any unexpected rows.\n        if imbalance == 0:\n            condition_sign = 1.0\n        elif imbalance == 2:\n            condition_sign = -1.0\n        else:\n            continue\n\n        # Expert 5 is feature index 4 and always identifies the TTB winner.\n        if diff[4] > 0:\n            expert5_winner = 1\n        elif diff[4] < 0:\n            expert5_winner = 0\n        else:\n            continue\n\n        chose_expert5_winner = float(int(row[\"response\"]) == expert5_winner)\n        # Averaging this score gives P(E5 winner | balanced) minus\n        # P(E5 winner | imbalanced), since the schedule is balanced.\n        interaction_scores.append(condition_sign * chose_expert5_winner)\n\n    if len(interaction_scores) == 0:\n        return float(\"nan\")\n    return float(np.mean(interaction_scores))",
  "rationale": "This is the conflict-by-choice interaction targeted by the design: the probability of choosing the Expert-5 winner on exactly balanced trials minus that probability on 3-versus-1 trials. TTB uses Expert 5 first on every stimulus, so its subject-level beta and lapse parameters affect both conditions identically and cancel in the expected contrast, keeping between-subject parameter variance low. CGRDA predicts a negative contrast: on balanced trials its open recency gate favors the final Expert-6 endorsement, whereas on imbalanced trials the gate closes and integrated evidence favors Expert 5's option. The metric aligns A/B reversals to the cue-defined winner, uses all 96 observations, and is computed identically on pooled and single-subject data."
}
```

## Usage

```json
{
  "input_tokens": 4632,
  "output_tokens": 1630
}
```
