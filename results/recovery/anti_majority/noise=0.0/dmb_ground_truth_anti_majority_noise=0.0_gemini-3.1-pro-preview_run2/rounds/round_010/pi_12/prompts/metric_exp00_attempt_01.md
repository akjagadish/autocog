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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 5: A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates 'Threshold-Gated Dilution Theory' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by equating the sum of validities while varying the raw count of active cues. The Competing theory dilutes evidence based on the sum of validities; therefore, when two options have the same validity sum, they suffer identical dilution, and the choice is driven purely by the non-linearly scaled numerators. In contrast, the Advocated theory dilutes based on the raw count of active cues, but only after a threshold (tau) is crossed. By pairing options with equal validity sums but different cue counts (e.g., n=2 vs n=3, or n=3 vs n=4), we force the options to potentially straddle the threshold in the Advocated theory. This results in one option being evaluated additively while the other is heavily diluted, predicting massive preference shifts that the Competing theory cannot replicate.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Threshold-Gated Dilution Theory: Decision-makers integrate cues differently depending on the number of active cues. When an option has only a few active cues (below a threshold), they are integrated additively (acting like a non-compensatory strategy if weights are steep). However, when a coalition of many cues is present, it triggers a shift to an averaging process, where the total evidence is diluted by the number of active cues. This explains why adding many weak cues to an option can paradoxically reduce its subjective value.

**Parameters:**
- lambda_val: [1.0, 20.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- tau: {1, 2, 3, 4}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    tau = int(parameters["tau"])
    
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    if n_a <= tau:
        v_a = np.sum(w * a)
    else:
        v_a = np.sum(w * a) / (n_a ** gamma)
        
    if n_b <= tau:
        v_b = np.sum(w * b)
    else:
        v_b = np.sum(w * b) / (n_b ** gamma)
        
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Evidence Dilution and Non-linear Weighting Theory (Validity-based Dilution with Amplified Penalty): Decision-makers evaluate options by integrating the validities of present features. However, instead of purely adding evidence, they partially average it. The presence of many low-validity features can paradoxically dilute the overall subjective value of an option (Evidence Dilution). This dilution is proportional to the sum of the validities of the present cues, and subjects apply a non-linear scaling to feature validities, amplifying the impact of the most valid cues. A potentially strong dilution penalty allows for severe subjective devaluation of options burdened with numerous weak features.

**Parameters:**
- lambda_val: [1.0, 20.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting to capture TTB-like reliance on top cues
    w = val ** lambda_val
    
    # Dilute by the sum of validities of the present cues
    sum_val_a = np.sum(val * a)
    sum_val_b = np.sum(val * b)
    
    # Calculate subjective values with a dilution factor (gamma)
    v_a = np.sum(w * a) / (sum_val_a ** gamma) if sum_val_a > 0 else 0.0
    v_b = np.sum(w * b) / (sum_val_b ** gamma) if sum_val_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
[0] rationale: Measures the choice probability of the option [0,0,0,1,1,1] over [0,0,1,1,1,0] (Trial 5). Under the Advocated theory, both options have exactly 3 active cues, so they suffer identical dilution. The option [0,0,1,1,1,0] has strictly better validities and is therefore always preferred. Under the Competing theory, dilution is based on the sum of validities. Option [0,0,1,1,1,0] has a higher validity sum and thus suffers a much heavier dilution penalty, meaning that for sufficiently high gamma, the Competing theory predicts a paradoxical preference for the objectively weaker option [0,0,0,1,1,1].
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    mask1 = (a_str == '001110') & (b_str == '000111')
    mask2 = (a_str == '000111') & (b_str == '001110')
    
    valid_trials = mask1 | mask2
    if not valid_trials.any():
        return 0.0
        
    subset = data[valid_trials]
    m1 = mask1[valid_trials]
    
    chose_target = np.where(m1, subset['response'] == 1, subset['response'] == 0)
    
    return float(np.mean(chose_target))
outcome: self_sim=0.4695 (var=0.0279) adversary_sim=0.4653 (var=0.0188) welch_t=+0.097 p=0.9228 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    mask1 = (a_str == '110000') & (b_str == '001100')\n    mask2 = (a_str == '001100') & (b_str == '110000')\n    \n    valid = mask1 | mask2\n    if not valid.any():\n        return 0.5\n        \n    subset = data[valid]\n    m1 = mask1[valid]\n    \n    chose_A = np.where(m1, subset['response'] == 0, subset['response'] == 1)\n    \n    return float(np.mean(chose_A))",
  "rationale": "Measures the choice probability of the option [1,1,0,0,0,0] over [0,0,1,1,0,0] (Trial 1). Under the Advocated theory, both options have exactly 2 active cues, so they suffer identical dilution, and the option with the strongest validities ([1,1,0,0,0,0]) is overwhelmingly preferred. Under the Competing theory, dilution is based on the sum of validities. The option [1,1,0,0,0,0] has a much higher validity sum (1.80 vs 1.40) and thus suffers a much heavier dilution penalty. This heavy penalty severely reduces the preference for [1,1,0,0,0,0] under the Competing theory relative to the Advocated theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3839,
  "candidates_token_count": 461,
  "total_token_count": 6938
}
```
