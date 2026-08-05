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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Unique Features Weakest-Link) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their fundamentally different integration rules for unique features after shared features are canceled. The Advocated Theory averages unique features and applies a weakest-link penalty, strictly preferring a single high-validity feature over a coalition of multiple moderately high-validity features because the coalition yields a lower mean and a higher range penalty. The Competing Theory sums unique features relative to a subjective threshold (rho). This additive integration allows a coalition of moderate features to easily overcome a single high-validity feature, provided they exceed the threshold. Furthermore, the Competing Theory's threshold implies that low-validity features can have negative utility, causing it to potentially prefer an option with no unique features over one with a single low-validity unique feature. The Advocated Theory, however, always assigns positive utility to any feature and will strictly prefer the option with the low-validity feature. By contrasting single high-validity features against coalitions of lower-validity features, and single low-validity features against empty feature sets, we can robustly separate the averaging/weakest-link mechanism from the thresholded additive mechanism.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Unique Features Weakest-Link: Decision-makers simplify choices by first cancelling out features shared by both options. They then evaluate each option based solely on its unique features, computing the average subjective validity of these features but applying a disproportionate penalty based on the option's 'weakest link' (the gap between its best and worst unique features).

**Parameters:**
- gamma: [0.1, 10.0]
- lambda_weak: [0.0, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    lambda_weak = float(parameters["lambda_weak"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    def get_score(unique_x):
        active_w = w[unique_x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        min_w = np.min(active_w)
        max_w = np.max(active_w)
        
        # Averaging baseline with a penalty based on the weakest link's distance from the best feature
        return mean_w - lambda_weak * (max_w - min_w)
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Thresholded Unique Features with Spread Penalty: Decision-makers simplify choices by cancelling out shared features, then evaluate the unique features relative to a subjective validity threshold. Features above the threshold provide positive evidence, while those below act as penalties. These values are integrated additively, but options with multiple unique features suffer a conflict penalty proportional to the spread (max - min) of their thresholded validities. This penalizes options with a wide variance in their unique features while strictly preserving shared-feature cancellation.

**Parameters:**
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_penalty: [0.0, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Transform validities and apply subjective threshold
    v_trans = val ** gamma
    w = v_trans - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # Additive integration of thresholded evidence
        base_score = np.sum(active_w)
        
        # Spread penalty applied if there are multiple unique features
        if len(active_w) > 1:
            conflict_penalty = lambda_penalty * (np.max(active_w) - np.min(active_w))
            return base_score - conflict_penalty
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    target_trials_a_chosen = 0\n    target_trials_total = 0\n    \n    for _, row in data.iterrows():\n        a_ratings = row['option_a_ratings']\n        b_ratings = row['option_b_ratings']\n        \n        unique_a = [i for i, (a, b) in enumerate(zip(a_ratings, b_ratings)) if a == 1 and b == 0]\n        unique_b = [i for i, (a, b) in enumerate(zip(a_ratings, b_ratings)) if b == 1 and a == 0]\n        \n        if len(unique_a) == 1:\n            if len(unique_b) == 0 or unique_a[0] < min(unique_b):\n                target_trials_total += 1\n                if row['response'] == 0:\n                    target_trials_a_chosen += 1\n                    \n    if target_trials_total == 0:\n        return 0.5\n    return target_trials_a_chosen / target_trials_total",
  "rationale": "This metric calculates the proportion of times Option A is chosen in trials where A has exactly one unique feature that is strictly superior to all of B's unique features (or B has no unique features). Under the Advocated Theory (Unique Features Weakest-Link), the evaluation of B is an average of its unique features minus a weakest-link penalty, meaning B's score can never exceed the value of its best unique feature. Since A's single unique feature is better than B's best, the Advocated Theory strictly prefers A in all these trials. Under the Competing Theory (Thresholded Unique Features), however, B can easily accumulate a higher score if the subjective threshold is low (allowing B's multiple features to additively overpower A's single feature) or A can receive a negative score if the threshold is high. Thus, the Competing Theory will predict a significantly lower rate of choosing A in these specific trials compared to the Advocated Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3743,
  "candidates_token_count": 492,
  "total_token_count": 8740
}
```
