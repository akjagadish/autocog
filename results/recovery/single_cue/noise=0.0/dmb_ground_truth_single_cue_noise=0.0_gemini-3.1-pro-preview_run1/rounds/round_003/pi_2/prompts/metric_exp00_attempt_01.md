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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate Tallying from the Weighted Additive (WADD) model, we exploit their fundamentally different treatment of feature validities. Tallying completely ignores validities, relying only on the count of features where one option dominates another. WADD, conversely, weights features by their validities (exponentiated by a free parameter gamma). We use 5 features with a large spread in validities: two high-validity experts (0.95, 0.90) and three low-validity experts (0.60, 0.55, 0.50). The trial set includes 'compensatory' trials where one option wins on more features (favored by Tallying) but the competing option wins on fewer, much higher-validity features (favored by WADD). We also include 'tie' trials where both options win on the same number of features—causing Tallying to predict a uniform guess—but where the weighted sums differ, allowing WADD to predict a deterministic preference. Variations in these combinations target the varying gamma parameter of the WADD model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



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
**Description:** People make decisions by computing a weighted sum of the features for each option, but the weights are non-linear transformations of the provided cue validities. Specifically, subjects exponentiate the validities by a parameter gamma, which allows the strategy to continuously interpolate between Tallying (gamma = 0) and Take-The-Best (gamma -> infinity). The option with the higher total weighted score is preferred, subject to decision noise (softmax) and occasional lapses.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.2]
- gamma: [-1.0, 2.0]
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
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    a_score = np.dot(a, weights)
    b_score = np.dot(b, weights)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    probs = np.asarray(probabilities, dtype=float)
    probs /= np.sum(probs)
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
[0] rationale: This metric focuses specifically on Trial 1, where Option A has fewer positive features (2 vs 3) but those features have much higher validities (0.95 and 0.90) compared to Option B's features (0.60, 0.55, 0.50). The Tallying heuristic completely ignores the validities and only counts the number of winning features, leading it to reliably prefer Option B. In contrast, the Weighted Additive (WADD) model weights the features by their exponentiated validities, causing it to strongly prefer Option A. By calculating the proportion of times Option A is chosen on this specific trial type, we obtain a metric that yields a stark reversal between the two theories (low for Tallying, high for WADD).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
    mask = (a_keys == (1, 1, 0, 0, 0)) & (b_keys == (0, 0, 1, 1, 1))
    
    if mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())

outcome: self_sim=0.1600 (var=0.0200) adversary_sim=0.3617 (var=0.1524) welch_t=-2.429 p=0.02131 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    a_keys = data['option_a_ratings'].apply(tuple)\n    b_keys = data['option_b_ratings'].apply(tuple)\n    \n    # Trial 2: A wins on 2 features, B wins on 2 features. WADD prefers A.\n    t2_mask = (a_keys == (1, 0, 1, 0, 0)) & (b_keys == (0, 1, 0, 1, 0))\n    # Trial 6: A wins on 2 features, B wins on 2 features. WADD prefers B.\n    t6_mask = (a_keys == (0, 1, 0, 0, 1)) & (b_keys == (1, 0, 0, 1, 0))\n    # Trial 8: A wins on 2 features, B wins on 2 features. WADD prefers A.\n    t8_mask = (a_keys == (0, 1, 1, 0, 0)) & (b_keys == (1, 0, 0, 0, 1))\n    \n    t2_score = (data.loc[t2_mask, 'response'] == 0).sum()\n    t6_score = (data.loc[t6_mask, 'response'] == 1).sum()\n    t8_score = (data.loc[t8_mask, 'response'] == 0).sum()\n    \n    total_trials = t2_mask.sum() + t6_mask.sum() + t8_mask.sum()\n    \n    if total_trials == 0:\n        return 0.5\n        \n    return float((t2_score + t6_score + t8_score) / total_trials)",
  "rationale": "This metric focuses on 'tie' trials (Trials 2, 6, and 8) where both Option A and Option B have exactly two features that dominate the other. Because Tallying only counts the number of winning features and ignores validities, it considers these options perfectly equivalent and predicts exactly a 50% uniform guess, resulting in a mean metric of 0.5 with extremely low variance. In contrast, the Weighted Additive (WADD) model uses the feature validities to break the ties (preferring A, B, and A respectively), leading to a proportion of WADD-consistent choices significantly higher than 0.5."
}
```

## Usage

```json
{
  "prompt_token_count": 3842,
  "candidates_token_count": 595,
  "total_token_count": 7018
}
```
