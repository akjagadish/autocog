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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.85, 0.7, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 9: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Strict Diminishing Returns on Absolute Subjective Utility) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their divergent integration mechanisms for unique features. Both models cancel shared features and evaluate the remaining unique features relative to a subjective threshold. However, the Advocated Theory ranks these features by their absolute utility and heavily discounts subsequent features (diminishing returns). The Competing Theory integrates features additively but applies a penalty proportional to the spread (max - min) of their validities. We design trials pitting a single high-validity feature against a coalition of multiple moderately high-validity features with zero or small spread. The Competing Theory, being additive, will strongly prefer the coalition. The Advocated Theory, due to strict diminishing returns, will heavily discount the coalition and prefer the single high-validity feature. We also include trials with large spreads to trigger the Competing Theory's spread penalty, causing it to penalize options that the Advocated Theory evaluates more favorably.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Strict Diminishing Returns on Absolute Subjective Utility: Decision-makers evaluate options by first cancelling out shared features. They then assess the unique features relative to a subjective validity threshold, where features below the threshold act as negative evidence (penalties). To reflect limited attention capacity, these features are ranked by their absolute utility (magnitude of impact, whether positive or negative), and their contribution is heavily discounted by a strict diminishing returns multiplier based on their rank.

**Parameters:**
- gamma: [0.1, 20.0]
- rho: [0.0, 2.0]
- delta: [0.0, 1.0]
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
    delta = float(parameters["delta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Subjective utility: validities transformed and shifted by a threshold
    w = (val ** gamma) - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # Rank active features by absolute thresholded utility descending
        order = np.argsort(np.abs(active_w))[::-1]
        sorted_w = active_w[order]
        
        # Apply rank-dependent scaling (delta <= 1 for strict diminishing returns)
        ranks = np.arange(len(sorted_w))
        discounted_w = sorted_w * (delta ** ranks)
        
        return np.sum(discounted_w)
        
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
[0] rationale: This metric calculates the proportion of times a subject chooses an option with a single feature over an option with multiple features. The Advocated Theory (Strict Diminishing Returns) strongly discounts additional features, often leading to a preference for a single high-validity feature. The Competing Theory evaluates features additively (with a spread penalty), which typically leads to a strong preference for the coalition of multiple features, as the additive value generally outweighs the single feature. Thus, the Advocated Theory will predict a significantly higher value for this metric than the Competing Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify the number of active features for options A and B
    n_features_a = data['option_a_ratings'].apply(lambda x: sum(x))
    n_features_b = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Filter for trials where one option has exactly 1 feature and the other has >1 features
    mask = ((n_features_a == 1) & (n_features_b > 1)) | ((n_features_b == 1) & (n_features_a > 1))
    
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    n_a = n_features_a[mask]
    n_b = n_features_b[mask]
    
    # Calculate if the subject chose the option with exactly 1 feature
    chose_1_feature = ((subset['response'] == 0) & (n_a == 1)) | ((subset['response'] == 1) & (n_b == 1))
    
    return float(chose_1_feature.mean())
outcome: self_sim=0.8382 (var=0.0165) adversary_sim=0.7596 (var=0.0439) welch_t=+1.601 p=0.1173 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_keys = data['option_a_ratings'].apply(tuple)\n    b_keys = data['option_b_ratings'].apply(tuple)\n    \n    # Trial 4: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0]\n    mask_4 = (a_keys == (0, 1, 1, 0, 0, 0)) & (b_keys == (1, 0, 0, 0, 1, 0))\n    # Trial 5: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]\n    mask_5 = (a_keys == (1, 0, 0, 0, 1, 0)) & (b_keys == (0, 1, 1, 0, 0, 0))\n    # Trial 8: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]\n    mask_8 = (a_keys == (1, 0, 0, 0, 0, 1)) & (b_keys == (0, 1, 1, 0, 0, 0))\n    \n    choices_high_max = 0\n    total = 0\n    \n    if mask_4.any():\n        choices_high_max += (data.loc[mask_4, 'response'] == 1).sum()\n        total += mask_4.sum()\n        \n    if mask_5.any():\n        choices_high_max += (data.loc[mask_5, 'response'] == 0).sum()\n        total += mask_5.sum()\n        \n    if mask_8.any():\n        choices_high_max += (data.loc[mask_8, 'response'] == 0).sum()\n        total += mask_8.sum()\n        \n    if total == 0:\n        return 0.5\n        \n    return float(choices_high_max / total)",
  "rationale": "This metric calculates the proportion of choices for the option containing the single highest-validity feature (feature 0) when it is paired with a low-validity feature (creating a large spread). The Advocated Theory (Strict Diminishing Returns) will strongly prefer this option because the highest-validity feature dominates the evaluation, and the secondary feature is heavily discounted. The Competing Theory (Thresholded Unique Features with Spread Penalty) will avoid this option because the large difference in validities between the high and low features triggers a massive spread penalty, making the alternative (a coalition of moderately high features with low spread) much more attractive."
}
```

## Usage

```json
{
  "prompt_token_count": 4392,
  "candidates_token_count": 706,
  "total_token_count": 8921
}
```
