# metric_exp00_attempt_02

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

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Feature Coherence and Anchoring) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their differing penalties for feature dispersion. While both models apply a spread penalty (max - min) to unique features, the Advocated Theory applies an additional 'gap penalty' based on the index distance between the best unique feature (the anchor) and the next best unique feature. Crucially, to avoid penalizing naturally sparse fallback options, the Advocated Theory only applies this gap penalty if the option has more than 2 unique features OR if it has exactly 2 unique features but claims the absolute best feature in the environment (index 0). We design trials that contrast 2-feature options starting at index 0 (large gap penalty) versus 2-feature options starting at index 1 (zero gap penalty), and 3-feature options with varying gaps. The Competing Theory, relying solely on additive values and spread penalties, will evaluate these shifts smoothly, whereas the Advocated Theory will show sharp preference reversals due to the conditional activation of the gap penalty.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Feature Coherence and Anchoring (Unique Features Only): Decision-makers simplify choices by first cancelling out shared features, then evaluating the remaining unique features. They expect high-quality options to be coherent. A gap penalty is applied if an option's best unique feature (the anchor) is disconnected from its next best unique feature. However, to avoid unfairly penalizing naturally sparse fallback options, this gap penalty is only applied if the option has a dense profile (>2 unique features) OR if it boasts the absolute highest-validity feature in the environment (index 0) but fails to back it up. A spread penalty is also applied to unique features to penalize internal conflict.

**Parameters:**
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_iso: [0.0, 5.0]
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
    lambda_iso = float(parameters["lambda_iso"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities
    w = (val ** gamma) - rho
    
    # Identify shared and unique features
    shared = (a > 0) & (b > 0)
    unique_a = np.where((a > 0) & ~shared)[0]
    unique_b = np.where((b > 0) & ~shared)[0]
    
    def get_score(unique_idx):
        if len(unique_idx) == 0:
            return 0.0
            
        base_score = np.sum(w[unique_idx])
        
        # Isolation / Gap penalty for the anchor calculated strictly on unique features.
        # Applied if the profile is dense (>2 unique features) OR 
        # if it's a 2-feature profile that claims the absolute best feature (index 0).
        gap_penalty = 0.0
        if len(unique_idx) > 2 or (len(unique_idx) == 2 and unique_idx[0] == 0):
            anchor = unique_idx[0]
            next_best = unique_idx[1]
            gap = next_best - anchor - 1
            if gap > 0:
                gap_penalty = lambda_iso * gap
                
        # Spread penalty on unique features
        conflict = 0.0
        if len(unique_idx) > 1:
            conflict = lambda_penalty * (np.max(w[unique_idx]) - np.min(w[unique_idx]))
            
        return base_score - gap_penalty - conflict
        
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
[0] rationale: Measures the difference in preference for Option A between Trial 2 and Trial 1. In Trial 1, Option A has 2 unique features starting at the absolute best feature (index 0), which triggers a severe gap penalty under the Advocated theory. In Trial 2, Option A has 2 unique features but starts at index 1, avoiding the gap penalty entirely. The Competing theory applies a spread penalty to both smoothly based on validities, predicting a smaller difference in preference shifts compared to the sharp discontinuity predicted by the Advocated theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 0, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 0))
    p_a_t1 = (data[t1_mask]['response'] == 0).mean()
    p_a_t2 = (data[t2_mask]['response'] == 0).mean()
    return float(p_a_t2 - p_a_t1)
outcome: self_sim=0.0525 (var=0.0469) adversary_sim=-0.0038 (var=0.0194) welch_t=+1.092 p=0.2813 (N=25, alpha=0.01) -> reject

[1] rationale: Contrasts Trial 5 and Trial 1 to isolate the massive gap penalty. In Trial 5, after cancelling shared features, Option A has only the best unique feature (index 0), which triggers no gap penalty under the Advocated Theory. In Trial 1, Option A has unique features at index 0 and index 4, which triggers a huge gap penalty (gap=3) under the Advocated Theory, causing a sharp drop in its choice probability. The Competing Theory evaluates both smoothly based on additive values and spread penalties, predicting a much smaller difference between the two trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    t5_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    p_a_t1 = (data[t1_mask]['response'] == 0).mean()
    p_a_t5 = (data[t5_mask]['response'] == 0).mean()
    
    return float(p_a_t5 - p_a_t1)
outcome: self_sim=0.7025 (var=0.0481) adversary_sim=0.6638 (var=0.0842) welch_t=+0.533 p=0.5968 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))\n    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 0, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 0))\n    t3_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 1, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 1))\n    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 0))\n    t5_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))\n    t6_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 0, 1))\n    \n    p_a_t1 = (data[t1_mask]['response'] == 0).mean()\n    p_a_t2 = (data[t2_mask]['response'] == 0).mean()\n    p_a_t3 = (data[t3_mask]['response'] == 0).mean()\n    p_a_t4 = (data[t4_mask]['response'] == 0).mean()\n    p_a_t5 = (data[t5_mask]['response'] == 0).mean()\n    p_a_t6 = (data[t6_mask]['response'] == 0).mean()\n    \n    gap_trials = (p_a_t1 + p_a_t3 + p_a_t4) / 3.0\n    no_gap_trials = (p_a_t2 + p_a_t5 + p_a_t6) / 3.0\n    \n    return float(no_gap_trials - gap_trials)",
  "rationale": "By pooling the choice probabilities of Option A across trials where it incurs the Advocated Theory's unique gap penalty (Trials 1, 3, 4) versus trials where it safely avoids it (Trials 2, 5, 6), we create a stable, low-variance contrast. The Advocated Theory will predict a substantial difference due to the conditional activation of the gap penalty, whereas the Competing Theory relies on smooth additive integration and spread penalties, thus predicting a markedly different aggregate shift."
}
```

## Usage

```json
{
  "prompt_token_count": 4770,
  "candidates_token_count": 818,
  "total_token_count": 9033
}
```
