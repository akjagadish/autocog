# metric_exp00_attempt_01

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
**Validities (n_features=5):** [0.95, 0.85, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design provides a granular dissociation of the WADD (Weighted Additive) and Tallying models by varying the trade-off between the number of winning features and their validities. Tallying purely counts the number of winning features, ignoring validities, leading to ties when counts are equal and strict preferences when one option wins on more features. In contrast, WADD scales features by a non-linear function of their validities. By distributing five features with linearly spaced validities, we create trials where Tallying prefers an option with a higher count of low-validity features, while WADD prefers the option with fewer, but higher-validity features. We also include trials where Tallying predicts a tie but WADD predicts a strong preference, and trials where both models agree on the direction but differ in the magnitude of preference, allowing for precise parameter estimation of WADD's gamma scaling and choice noise.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make decisions by computing a weighted sum of the features for each option, where the weights correspond to a non-linear scaling of the subjective validities of the cues. This Weighted Additive (WADD) strategy allows for compensatory decision-making, where multiple weak cues can override a single strong cue, but the non-linear scaling (gamma parameter) flexibly tunes how strongly higher-validity cues dominate over weaker ones. A softmax choice rule on these weighted sums, along with a lapse rate, introduces response noise and accounts for the intermediate choice patterns observed in human data.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    # Calculate weighted sum for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.0, 1.0]
- epsilon: [0.4, 0.9]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
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
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))
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
[0] rationale: Tallying decides based purely on the unweighted count of features. Therefore, it strongly predicts choosing Option B when B has more positive features than A, and choosing Option A when B has fewer. The difference in the probability of choosing B between these two sets of trials should be highly positive under Tallying. WADD, on the other hand, weights features by their subjective validity (which is non-linearly scaled). Because the experimental design pits the number of features against their validities (e.g., giving A a single highly valid feature while B gets several less valid ones), WADD will frequently choose contrary to the raw feature count. Thus, the difference in choice probabilities based on raw feature counts will be much smaller (or even negative) for WADD compared to Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of features present in A and B
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    # Tallying predicts choosing B when sum_b > sum_a, and choosing A when sum_b < sum_a
    # We measure the difference in the probability of choosing B between these two conditions
    mask_b_greater = sum_b > sum_a
    mask_a_greater = sum_b < sum_a
    
    prob_b_when_b_greater = data.loc[mask_b_greater, 'response'].mean()
    prob_b_when_a_greater = data.loc[mask_a_greater, 'response'].mean()
    
    # If a subject has no trials in one of the conditions (unlikely given the design), return 0
    if np.isnan(prob_b_when_b_greater) or np.isnan(prob_b_when_a_greater):
        return 0.0
        
    return float(prob_b_when_b_greater - prob_b_when_a_greater)
outcome: self_sim=-0.0010 (var=0.1652) adversary_sim=0.1215 (var=0.0216) welch_t=-1.417 p=0.1667 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # Project lists to strings for hashable matching\n    a_str = data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    \n    # Trial 1: A=[1,0,0,0,0], B=[0,1,1,1,0]. Tallying tally diff = 3 - 1 = 2.\n    t1_mask = (a_str == \"10000\") & (b_str == \"01110\")\n    # Trial 2: A=[1,0,0,0,0], B=[0,0,1,1,1]. Tallying tally diff = 3 - 1 = 2.\n    t2_mask = (a_str == \"10000\") & (b_str == \"00111\")\n    \n    # Trial 7: A=[1,0,1,0,0], B=[0,1,0,1,1]. Tallying tally diff = 3 - 2 = 1.\n    t7_mask = (a_str == \"10100\") & (b_str == \"01011\")\n    # Trial 8: A=[1,1,0,0,0], B=[0,0,1,1,1]. Tallying tally diff = 3 - 2 = 1.\n    t8_mask = (a_str == \"11000\") & (b_str == \"00111\")\n    \n    p_b_t1 = data.loc[t1_mask, 'response'].mean()\n    p_b_t2 = data.loc[t2_mask, 'response'].mean()\n    p_b_t7 = data.loc[t7_mask, 'response'].mean()\n    p_b_t8 = data.loc[t8_mask, 'response'].mean()\n    \n    # Handle missing trial types gracefully\n    p_b_t1 = 0.5 if pd.isna(p_b_t1) else p_b_t1\n    p_b_t2 = 0.5 if pd.isna(p_b_t2) else p_b_t2\n    p_b_t7 = 0.5 if pd.isna(p_b_t7) else p_b_t7\n    p_b_t8 = 0.5 if pd.isna(p_b_t8) else p_b_t8\n    \n    # Tallying predicts exactly 0 difference for both pairs. \n    # WADD predicts positive differences because B has higher validity features in T1 vs T2, and T7 vs T8.\n    return float((p_b_t1 - p_b_t2) + (p_b_t7 - p_b_t8))",
  "rationale": "Tallying decides based purely on the unweighted count of features. In Trial 1 and Trial 2, Option B has exactly 2 more winning features than Option A, so Tallying predicts the exact same probability of choosing B for both trials. Similarly, in Trial 7 and Trial 8, Option B has exactly 1 more winning feature than A, leading to identical choice probabilities under Tallying. Therefore, the metric (P(B|T1) - P(B|T2)) + (P(B|T7) - P(B|T8)) has an expected value of exactly 0 for Tallying. Conversely, WADD weights features by their subjective validities. Because Option B's features in Trial 1 have higher validities than in Trial 2, and similarly for Trial 7 versus Trial 8, WADD strongly predicts a higher probability of choosing B in the first trial of each pair. This produces a strictly positive expected value for WADD, effectively isolating the difference in how the two models treat feature validities while controlling for raw feature counts."
}
```

## Usage

```json
{
  "prompt_token_count": 3942,
  "candidates_token_count": 998,
  "total_token_count": 9738
}
```
