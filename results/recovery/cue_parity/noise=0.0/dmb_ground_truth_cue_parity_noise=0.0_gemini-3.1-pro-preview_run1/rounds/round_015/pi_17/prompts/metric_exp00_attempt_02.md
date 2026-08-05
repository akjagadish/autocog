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
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.8, 0.75]

**Trial pairs (n=15):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 11: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 13: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 15: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Strategy Selection mixing TTB and Tallying) from the Competing Theory (Normalized Weighted Sequential Evidence Accumulation), this design exploits the Advocated Theory's insensitivity to the exact magnitudes of validities once ordinal rank and tally counts are established. We construct a critical set of trials where Option A always wins the highest valid discriminating feature (so TTB strongly favors A), and Option B always wins exactly one more feature than A (so Tallying consistently favors B with a difference of exactly 1). Because TTB outputs a binary preference (1.0 for the winner) regardless of whether the winning feature is the 1st, 2nd, or 3rd most valid, and Tallying outputs a probability based strictly on the integer difference in feature wins, the Advocated Theory predicts the EXACT SAME choice probability for Option A across all these critical trials. In contrast, the Competing Theory weights evidence by a non-linear scaling of the exact validities. Because the validities strictly decrease, the accumulated evidence varies widely across these trials depending on exactly which features comprise the tallies (e.g., F1 vs F4+F5 yields different evidence than F2 vs F3+F4). Even if the Competing Theory uses a low threshold and stops early, it will stop on different features (F1, F2, or F3) and thus accumulate different amounts of evidence. Consequently, the Competing Theory structurally must predict a graded, variable response profile across these trials, whereas the Advocated Theory predicts a perfectly flat, discrete profile.

**Computed schedule:** 15 unique pairs × 6 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Selection based on Environment Classification

**Parameters:**
- dispersion_threshold: [0.0, 0.3]
- slope: [1.0, 100.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Environment classification based on validity dispersion
    dispersion = np.std(val)
    threshold = float(parameters["dispersion_threshold"])
    slope = float(parameters["slope"])
    
    # Probability of selecting Take-The-Best over Tallying
    w_ttb = 1.0 / (1.0 + np.exp(-slope * (dispersion - threshold)))
    
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    diff = a - b
    ttb_a, ttb_b = 0.0, 0.0
    for idx in order:
        if diff[idx] > 0:
            ttb_a = 1.0
            break
        elif diff[idx] < 0:
            ttb_b = 1.0
            break
            
    # Tallying prediction
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    
    beta = float(parameters["beta"])
    
    z_ttb = beta * np.array([ttb_a, ttb_b])
    p_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb /= np.sum(p_ttb)
    
    z_tally = beta * np.array([tally_a, tally_b])
    p_tally = np.exp(z_tally - np.max(z_tally))
    p_tally /= np.sum(p_tally)
    
    # Mixture of strategies
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Decision-makers use a Normalized Weighted Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity. They accumulate the differences between the options weighted by the features' validities, which are non-linearly scaled and normalized to sum to 1. This normalization ensures the evidence scale is consistent across experiments with varying numbers of features. If the absolute accumulated evidence reaches or exceeds a specific threshold (bounded between 0 and 1), they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally.

**Parameters:**
- threshold: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    # Normalize validities raised to gamma so they sum to 1
    weights = (val ** gamma) / np.sum(val ** gamma)
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx] * weights[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: The Advocated Theory (Strategy Selection) predicts the exact same choice probability for the TTB-favored option across all critical trials where TTB and Tallying are in conflict and the Tally difference is exactly 1. In contrast, the Competing Theory (Normalized Weighted Sequential Evidence Accumulation) accumulates different amounts of evidence across these trials depending on exactly which features are active, structurally predicting a variable response profile. By computing the variance of the proportion of TTB choices across these critical trial types, we obtain a metric that is near 0 for the Advocated Theory and significantly greater than 0 for the Competing Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ttb_choices = []
    trial_ids = []
    
    for idx, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        diff = a - b
        ttb_winner = -1
        for i in range(len(diff)):
            if diff[i] > 0:
                ttb_winner = 0
                break
            elif diff[i] < 0:
                ttb_winner = 1
                break
                
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        tally_winner = 0 if sum_a > sum_b else (1 if sum_b > sum_a else -1)
        tally_diff = abs(sum_a - sum_b)
        
        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner and tally_diff == 1:
            chose_ttb = 1 if row['response'] == ttb_winner else 0
            ttb_choices.append(chose_ttb)
            a_tup = tuple(a.tolist())
            b_tup = tuple(b.tolist())
            trial_id = tuple(sorted([a_tup, b_tup]))
            trial_ids.append(trial_id)
            
    if len(ttb_choices) == 0:
        return 0.0
        
    df_critical = pd.DataFrame({'trial_id': trial_ids, 'chose_ttb': ttb_choices})
    means = df_critical.groupby('trial_id')['chose_ttb'].mean()
    
    if len(means) < 2:
        return 0.0
        
    return float(np.var(means, ddof=0))

outcome: self_sim=0.0010 (var=0.0001) adversary_sim=0.0207 (var=0.0019) welch_t=-2.180 p=0.03798 (N=25, alpha=0.01) -> reject

[1] rationale: To maximize the discrimination between the two theories, we focus on a direct, targeted contrast between Trial 3 and Trial 1. In both trials, Option A is favored by Take-The-Best (winning the most valid feature, F1), while Option B is favored by Tallying (winning two features vs. one). Because the Advocated Theory mixes TTB and Tallying based merely on ordinal ranks and tally differences, it predicts the exact same choice probability for Option A in both trials, yielding a difference of 0. In contrast, the Competing Theory accumulates validity-weighted evidence. In Trial 1, Option B's features are F2 and F3 (high validity), making it a strong competitor. In Trial 3, Option B's features are F4 and F5 (lower validity), making it a weaker competitor. Thus, the Competing Theory predicts a significantly higher probability of choosing Option A in Trial 3 than in Trial 1. Taking the difference in choice proportions directly isolates this structural divergence.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t1_a = (1, 0, 0, 0, 0)
    t1_b = (0, 1, 1, 0, 0)
    
    t3_a = (1, 0, 0, 0, 0)
    t3_b = (0, 0, 0, 1, 1)
    
    chose_a_t1 = []
    chose_a_t3 = []
    
    for idx, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        chose_a = (row['response'] == 0)
        
        if (a == t1_a and b == t1_b):
            chose_a_t1.append(1 if chose_a else 0)
        elif (a == t1_b and b == t1_a):
            chose_a_t1.append(0 if chose_a else 1)
            
        elif (a == t3_a and b == t3_b):
            chose_a_t3.append(1 if chose_a else 0)
        elif (a == t3_b and b == t3_a):
            chose_a_t3.append(0 if chose_a else 1)
            
    m1 = np.mean(chose_a_t1) if chose_a_t1 else 0.0
    m3 = np.mean(chose_a_t3) if chose_a_t3 else 0.0
    
    return float(m3 - m1)
outcome: self_sim=0.0633 (var=0.0424) adversary_sim=0.1800 (var=0.0762) welch_t=-1.694 p=0.09729 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    val = np.array([0.95, 0.9, 0.85, 0.8, 0.75])\n    \n    val_diffs = []\n    chose_ttbs = []\n    \n    for idx, row in data.iterrows():\n        a = np.array(row['option_a_ratings'])\n        b = np.array(row['option_b_ratings'])\n        \n        diff = a - b\n        ttb_winner = -1\n        for i in range(len(diff)):\n            if diff[i] > 0:\n                ttb_winner = 0\n                break\n            elif diff[i] < 0:\n                ttb_winner = 1\n                break\n                \n        sum_a = np.sum(a)\n        sum_b = np.sum(b)\n        tally_diff = sum_a - sum_b\n        tally_winner = 0 if tally_diff > 0 else (1 if tally_diff < 0 else -1)\n        \n        # Filter for critical trials where TTB and Tally conflict by exactly 1\n        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner and abs(tally_diff) == 1:\n            chose_ttb = 1 if row['response'] == ttb_winner else 0\n            \n            v_a = np.sum(val * a)\n            v_b = np.sum(val * b)\n            \n            v_diff = (v_a - v_b) if ttb_winner == 0 else (v_b - v_a)\n                \n            val_diffs.append(v_diff)\n            chose_ttbs.append(chose_ttb)\n            \n    if len(val_diffs) < 2:\n        return 0.0\n        \n    cov_matrix = np.cov(val_diffs, chose_ttbs)\n    return float(cov_matrix[0, 1] * 100)",
  "rationale": "To cleanly dissociate the two theories, this metric computes the covariance between the linear evidence difference (based on validities) and the binary choice to follow Take-The-Best (TTB) across all critical trials where TTB and Tallying conflict by exactly a tally difference of 1. According to the Advocated Theory, the probability of choosing the TTB-favored option is a constant mixture probability across all these trials because it relies solely on ordinal ranks and the tally difference of 1; thus, the covariance should be zero. In contrast, the Competing Theory accumulates evidence based on non-linearly scaled validities, meaning the actual probability of choosing the TTB-favored option will vary systematically with the actual validity difference. This yields a strong positive covariance for the Competing Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 5357,
  "candidates_token_count": 713,
  "total_token_count": 11132
}
```
