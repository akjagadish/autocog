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
  trial 1: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 3: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 8: A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Relative Threshold Heuristic) from the Competing Theory (Weakest Link), we exploit their fundamentally different treatment of shared features. The Advocated Theory compares options cue-by-cue, meaning any feature possessed by both options is immediately canceled out and has zero effect on the choice. In contrast, the Competing Theory evaluates each option holistically by computing the mean and range (weakest link penalty) of its active features; thus, adding a shared feature alters both the mean and the range penalty of both options, often asymmetrically. We design base trials comparing unique advantages, and then create derivative trials by adding shared high-validity or shared low-validity features. The Advocated Theory predicts identical choice probabilities across these variations, whereas the Competing Theory predicts massive shifts in preference due to the changing weakest-link penalties.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Relative Threshold Heuristic with Bounded Regret: Decision-makers evaluate options by directly comparing them on a cue-by-cue basis, focusing only on distinguishing features (unique advantages and disadvantages). To capture the dilution effect without relying on a holistic average, decision-makers apply a threshold (`theta`) to their unique advantages. Low-validity unique advantages that fall below this threshold actually penalize the option. Furthermore, the regret penalty for unique disadvantages is also thresholded (`theta_regret`), ensuring that decision-makers only penalize the absence of truly critical (high-validity) features. This prevents options with multiple medium-validity advantages from being overly punished for missing a single high-validity feature.

**Parameters:**
- gamma: [0.01, 5.0]
- delta: [0.01, 5.0]
- lambda_regret: [0.0, 2.0]
- theta: [0.0, 1.0]
- theta_regret: [0.0, 1.0]
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
    delta = float(parameters["delta"])
    lambda_regret = float(parameters["lambda_regret"])
    theta = float(parameters["theta"])
    theta_regret = float(parameters["theta_regret"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights for presence and absence
    w_pos = val ** gamma
    w_neg = val ** delta
    
    def get_score(x, other_x):
        unique_adv = (x > 0) & (other_x == 0)
        unique_disadv = (x == 0) & (other_x > 0)
        
        # Sum of unique advantages, penalized by a threshold (theta)
        # This allows low-validity features to have a net negative impact
        adv_score = np.sum(w_pos[unique_adv] - theta) if np.any(unique_adv) else 0.0
        
        # Regret penalty for unique disadvantages, also thresholded
        disadv_score = lambda_regret * np.sum(np.maximum(0.0, w_neg[unique_disadv] - theta_regret)) if np.any(unique_disadv) else 0.0
        
        return adv_score - disadv_score
        
    score_a = get_score(a, b)
    score_b = get_score(b, a)
    
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
**Description:** Weakest Link (Worst-Case Focus): Decision-makers evaluate an option primarily by averaging the subjective validities of its present features, but they apply a disproportionate penalty based on the option's 'weakest link' (the lowest validity active feature). Specifically, the penalty is proportional to the gap between the option's best feature and its weakest feature. This cognitive mechanism reflects a worst-case focus where attention is drawn to the most inferior attribute, naturally predicting both the dilution effect (adding weak features reduces the mean and increases the weakest-link penalty) and variance aversion (options with tighter feature clusters have smaller weakest-link penalties).

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
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        min_w = np.min(active_w)
        max_w = np.max(active_w)
        
        # Averaging baseline with a penalty based on the weakest link's distance from the best feature
        return mean_w - lambda_weak * (max_w - min_w)
        
    score_a = get_score(a)
    score_b = get_score(b)
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    # Project lists to strings for hashable grouping\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    \n    # response=0 means A was chosen, response=1 means B was chosen.\n    # So 1 - mean(response) is the empirical probability of choosing A.\n    p_A = 1.0 - data.groupby(['A_str', 'B_str'])['response'].mean()\n    \n    def get_p(a, b):\n        if (a, b) in p_A.index:\n            return p_A.loc[(a, b)]\n        return 0.5\n        \n    # Trial pairs that differ only by a shared feature added to both options\n    # Advocated theory predicts 0 difference; Weakest Link predicts large shifts.\n    \n    # Trial 2 vs Trial 1 (Shared feature 1 added)\n    d1 = get_p('11000', '10110') - get_p('01000', '00110')\n    \n    # Trial 3 vs Trial 1 (Shared feature 5 added)\n    d2 = get_p('01001', '00111') - get_p('01000', '00110')\n    \n    # Trial 5 vs Trial 4 (Shared feature 1 added)\n    d3 = get_p('11100', '10011') - get_p('01100', '00011')\n    \n    # Trial 8 vs Trial 7 (Shared feature 1 added)\n    d4 = get_p('10100', '10010') - get_p('00100', '00010')\n    \n    return float(np.abs(d1) + np.abs(d2) + np.abs(d3) + np.abs(d4))\n",
  "rationale": "The Advocated Theory (Relative Threshold Heuristic) evaluates options purely on their unique advantages and disadvantages, meaning that adding a shared feature to both options will exactly cancel out and leave the choice probabilities unchanged. The Competing Theory (Weakest Link), however, evaluates options holistically by looking at the mean and range (weakest link penalty) of their active features. Adding a shared feature fundamentally alters both the mean and the penalty term for both options, leading to divergent shifts in preference. This metric computes the sum of absolute differences in the probability of choosing Option A between base trials and derivative trials (where shared features are added). It should be near zero for the Advocated Theory and substantially positive for the Competing Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3752,
  "candidates_token_count": 725,
  "total_token_count": 6625
}
```
