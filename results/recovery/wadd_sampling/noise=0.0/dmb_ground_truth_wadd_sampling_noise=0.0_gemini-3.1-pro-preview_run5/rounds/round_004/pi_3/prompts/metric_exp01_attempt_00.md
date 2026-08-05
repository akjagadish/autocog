# metric_exp01_attempt_00

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
**Validities (n_features=6):** [0.9, 0.9, 0.7, 0.7, 0.5, 0.5]

**Trial pairs (n=11):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 8: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate pure Weighted Additive (WADD) from Evidence-Dependent Noise, we must orthogonalize the net evidence difference between the options and the total feature conflict. Pure WADD predicts that choice probabilities are determined solely by the difference in subjectively weighted evidence. The Competing Theory posits that the choice temperature scales with total conflict (the sum of weights of features that differ between the options). By using pairs of features with identical validities (e.g., 0.9 and 0.9), we can add 'canceling' evidence to both options. This leaves the net evidence difference perfectly unchanged for ANY subjective validity transformation (gamma), but strictly increases the total conflict. Pure WADD predicts identical choice probabilities across these conflict levels, whereas Evidence-Dependent Noise predicts probabilities will regress toward 0.5 on high-conflict trials.

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
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
**Description:** Evidence-Dependent Noise: Decision-makers evaluate options using a single compensatory mechanism (Weighted Additive) where features are weighted by a subjective transformation of their validities. However, the decision process is subject to evidence-dependent noise: the temperature of the softmax choice rule scales with the total conflict between the options (defined as the total weighted evidence of features that differ between the two options). This ensures that trials with higher evidence magnitudes or greater feature conflict naturally generate higher decision noise. This single-mechanism approach preserves log-odds linearity while explaining why extremeness flattens across trials with varying evidence magnitudes.

**Parameters:**
- gamma: [0.0, 5.0]
- beta: [0.1, 20.0]
- theta: [0.0, 10.0]
- epsilon: [0.0, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus expects shape (2, n_features); got {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Subjective feature weighting
    weights = val ** gamma
    
    # Calculate weighted sums of evidence for each option
    scores = np.dot(stim, weights)
    
    # Calculate conflict: total weight of features where the options differ
    diff = np.abs(stim[0] - stim[1])
    conflict = np.dot(diff, weights)
    
    # Effective beta scales inversely with conflict (higher conflict = more noise)
    beta_eff = beta / (1.0 + theta * conflict)
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta_eff * (scores - np.max(scores))
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
[0] rationale: The experimental design consists of pairs of trials where the net evidence is perfectly matched, but the total conflict differs because 'cancelling' features (with identical validities) are added to both options. Under the pure Weighted Additive (WADD) theory, the subjective weighting transformation maintains this exact cancellation, meaning choice probabilities should be identical across the low and high conflict pairs. Thus, the difference in accuracy should be zero. In contrast, the Evidence-Dependent Noise theory posits that choice temperature scales with total conflict. Therefore, the high conflict trials will exhibit more noise (probabilities regressing toward 0.5), resulting in a positive difference in accuracy (low conflict accuracy > high conflict accuracy).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['A'] = data['option_a_ratings'].apply(tuple)
    data['B'] = data['option_b_ratings'].apply(tuple)
    
    # Map of (A, B) tuples to the 'correct' (higher net evidence) response
    low_conflict = {
        ((1,0,0,0,0,0), (0,0,0,0,0,0)): 0,
        ((1,0,0,1,0,0), (0,0,0,0,1,0)): 0,
        ((0,0,0,0,0,0), (1,0,0,0,0,0)): 1,
        ((0,0,0,1,0,0), (1,0,0,0,1,0)): 1,
        ((0,0,0,0,0,1), (0,0,0,0,0,0)): 0,
        ((0,0,0,1,0,1), (0,0,0,0,1,0)): 0
    }
    
    # High conflict versions of the above trials (added cancelling features)
    high_conflict = {
        ((1,1,0,0,0,0), (0,0,1,0,0,0)): 0,
        ((1,1,0,1,0,0), (0,0,1,0,1,0)): 0,
        ((0,1,0,0,0,0), (1,0,1,0,0,0)): 1,
        ((0,1,0,1,0,0), (1,0,1,0,1,0)): 1,
        ((0,1,0,0,0,1), (0,0,1,0,0,0)): 0,
        ((0,1,0,1,0,1), (0,0,1,0,1,0)): 0
    }
    
    low_acc = []
    high_acc = []
    
    for _, row in data.iterrows():
        pair = (row['A'], row['B'])
        resp = row['response']
        
        if pair in low_conflict:
            low_acc.append(1 if resp == low_conflict[pair] else 0)
        elif pair in high_conflict:
            high_acc.append(1 if resp == high_conflict[pair] else 0)
            
    if len(low_acc) == 0 or len(high_acc) == 0:
        return 0.0
        
    return float(np.mean(low_acc) - np.mean(high_acc))
outcome: self_sim=0.0278 (var=0.0088) adversary_sim=0.0850 (var=0.0133) welch_t=-1.924 p=0.06054 (N=25, alpha=0.01) -> reject

[1] rationale: Instead of looking at raw accuracy differences, which can be noisy across subjects, this metric directly measures the variance of choices for each trial type. The variance of a binary choice is p*(1-p). Pure WADD predicts identical choice probabilities across matched low/high conflict pairs, meaning the difference in choice variance should be exactly zero. The Evidence-Dependent Noise theory predicts that high conflict increases noise, pushing choice probabilities closer to 0.5. Since p*(1-p) is maximized at p=0.5, this regression to the mean strictly increases the choice variance. By calculating the difference in choice variance between matched high and low conflict trials, we obtain a robust, bounded measure of how much conflict increases decision noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['A'] = data['option_a_ratings'].apply(tuple)
    data['B'] = data['option_b_ratings'].apply(tuple)
    data['pair'] = list(zip(data['A'], data['B']))
    
    low_conflict = {
        ((1,0,0,0,0,0), (0,0,0,0,0,0)),
        ((1,0,0,1,0,0), (0,0,0,0,1,0)),
        ((0,0,0,0,0,0), (1,0,0,0,0,0)),
        ((0,0,0,1,0,0), (1,0,0,0,1,0)),
        ((0,0,0,0,0,1), (0,0,0,0,0,0)),
        ((0,0,0,1,0,1), (0,0,0,0,1,0))
    }
    
    high_conflict = {
        ((1,1,0,0,0,0), (0,0,1,0,0,0)),
        ((1,1,0,1,0,0), (0,0,1,0,1,0)),
        ((0,1,0,0,0,0), (1,0,1,0,0,0)),
        ((0,1,0,1,0,0), (1,0,1,0,1,0)),
        ((0,1,0,0,0,1), (0,0,1,0,0,0)),
        ((0,1,0,1,0,1), (0,0,1,0,1,0))
    }
    
    # Calculate the variance of choices (p * (1 - p)) for each trial pair
    # We group by pair and subject to get per-subject, per-pair choice probability
    grouped = data.groupby(['subject_id', 'pair'])['response'].mean().reset_index()
    grouped['variance'] = grouped['response'] * (1.0 - grouped['response'])
    
    # Separate into low and high conflict
    low_mask = grouped['pair'].isin(low_conflict)
    high_mask = grouped['pair'].isin(high_conflict)
    
    # Average variance across subjects and relevant pairs
    var_low = grouped[low_mask].groupby('subject_id')['variance'].mean()
    var_high = grouped[high_mask].groupby('subject_id')['variance'].mean()
    
    # Return the mean difference in choice variance (High conflict - Low conflict)
    # If the competing theory is correct, High conflict trials regress to p=0.5,
    # which strictly increases the variance (p*(1-p)) of the responses.
    diffs = var_high - var_low
    if len(diffs) == 0:
        return 0.0
    
    return float(diffs.mean())

outcome: self_sim=0.0104 (var=0.0022) adversary_sim=0.0344 (var=0.0020) welch_t=-1.859 p=0.06915 (N=25, alpha=0.01) -> reject

[2] rationale: By transforming the raw accuracy into log-odds space, we normalize for the subject's baseline decision noise (beta). Pure WADD predicts that the net evidence is perfectly matched between the low and high conflict pairs, so the log-odds of choosing the target should be identical (difference = 0). The Evidence-Dependent Noise theory predicts that the effective beta scales inversely with conflict, meaning the log-odds of choosing the target will be strictly lower for high conflict trials. Computing the difference in log-odds amplifies the signal for subjects with high baseline accuracy (where EDN predicts a large raw accuracy drop but WADD predicts none) and provides a more robust, lower-variance contrast than raw accuracy differences.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['A'] = data['option_a_ratings'].apply(tuple)
    data['B'] = data['option_b_ratings'].apply(tuple)
    data['pair'] = list(zip(data['A'], data['B']))
    
    low_conflict = {
        ((1,0,0,0,0,0), (0,0,0,0,0,0)): 0,
        ((1,0,0,1,0,0), (0,0,0,0,1,0)): 0,
        ((0,0,0,0,0,0), (1,0,0,0,0,0)): 1,
        ((0,0,0,1,0,0), (1,0,0,0,1,0)): 1,
        ((0,0,0,0,0,1), (0,0,0,0,0,0)): 0,
        ((0,0,0,1,0,1), (0,0,0,0,1,0)): 0
    }
    
    high_conflict = {
        ((1,1,0,0,0,0), (0,0,1,0,0,0)): 0,
        ((1,1,0,1,0,0), (0,0,1,0,1,0)): 0,
        ((0,1,0,0,0,0), (1,0,1,0,0,0)): 1,
        ((0,1,0,1,0,0), (1,0,1,0,1,0)): 1,
        ((0,1,0,0,0,1), (0,0,1,0,0,0)): 0,
        ((0,1,0,1,0,1), (0,0,1,0,1,0)): 0
    }
    
    targets = {**low_conflict, **high_conflict}
    df = data[data['pair'].isin(targets.keys())].copy()
    df['target'] = df['pair'].map(targets)
    df['is_correct'] = (df['response'] == df['target']).astype(float)
    df['is_high'] = df['pair'].isin(high_conflict)
    
    agg = df.groupby(['subject_id', 'is_high'])['is_correct'].mean().unstack('is_high')
    
    if False not in agg.columns or True not in agg.columns:
        return 0.0
        
    low_acc = agg[False].values
    high_acc = agg[True].values
    
    low_acc = np.clip(low_acc, 0.05, 0.95)
    high_acc = np.clip(high_acc, 0.05, 0.95)
    
    lo_low = np.log(low_acc / (1.0 - low_acc))
    lo_high = np.log(high_acc / (1.0 - high_acc))
    
    return float(np.mean(lo_low - lo_high))
outcome: self_sim=0.1706 (var=0.3527) adversary_sim=0.5063 (var=0.4050) welch_t=-1.928 p=0.05979 (N=25, alpha=0.01) -> reject

[3] rationale: Previous attempts to measure the accuracy drop included all matched trial pairs, which diluted the effect because some 'high conflict' pairs had relatively small increases in conflict (e.g., only one pair of canceling features). By restricting the analysis strictly to the trials with the absolute minimum conflict (0 canceling features) and the absolute maximum conflict (2 canceling features), we double the expected difference in total conflict between the sets. Under the Evidence-Dependent Noise theory, this maximizes the difference in effective temperature, leading to a much larger predicted accuracy drop, while Pure WADD still rigorously predicts a difference of exactly zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['A'] = data['option_a_ratings'].apply(tuple)
    data['B'] = data['option_b_ratings'].apply(tuple)
    
    # Lowest conflict trials (0 canceling feature pairs)
    low_pairs = {
        ((1,0,0,0,0,0), (0,0,0,0,0,0)): 0,
        ((0,0,0,0,0,0), (1,0,0,0,0,0)): 1,
        ((0,0,0,0,0,1), (0,0,0,0,0,0)): 0,
    }
    
    # Highest conflict trials (2 canceling feature pairs)
    high_pairs = {
        ((1,1,0,1,0,0), (0,0,1,0,1,0)): 0,
        ((0,1,0,1,0,0), (1,0,1,0,1,0)): 1,
        ((0,1,0,1,0,1), (0,0,1,0,1,0)): 0,
    }
    
    low_accs = []
    high_accs = []
    
    for subj, subj_df in data.groupby('subject_id'):
        low_correct = 0
        low_total = 0
        high_correct = 0
        high_total = 0
        
        for _, row in subj_df.iterrows():
            pair = (row['A'], row['B'])
            resp = row['response']
            if pair in low_pairs:
                if resp == low_pairs[pair]:
                    low_correct += 1
                low_total += 1
            elif pair in high_pairs:
                if resp == high_pairs[pair]:
                    high_correct += 1
                high_total += 1
                
        if low_total > 0 and high_total > 0:
            low_accs.append(low_correct / low_total)
            high_accs.append(high_correct / high_total)
            
    if not low_accs:
        return 0.0
        
    return float(np.mean(low_accs) - np.mean(high_accs))
outcome: self_sim=0.0311 (var=0.0180) adversary_sim=0.1278 (var=0.0231) welch_t=-2.382 p=0.02128 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    data = data.copy()\n    data['A'] = data['option_a_ratings'].apply(tuple)\n    data['B'] = data['option_b_ratings'].apply(tuple)\n    data['pair'] = list(zip(data['A'], data['B']))\n    \n    # Using the exact trial pairs from the experimental design\n    # These pairs have identical net evidence but differ greatly in total conflict\n    low_conflict = {\n        ((1,0,0,0,0,0), (0,0,0,0,0,0)), # Trial 1\n        ((0,0,1,0,0,0), (0,0,0,0,1,0)), # Trial 4\n        ((0,0,1,0,1,0), (0,0,0,0,0,0))  # Trial 10\n    }\n    \n    high_conflict = {\n        ((1,0,1,0,1,0), (0,0,0,1,0,1)), # Trial 3\n        ((1,0,1,0,0,0), (0,1,0,0,1,0)), # Trial 5\n        ((1,0,1,0,1,0), (0,1,0,0,0,0))  # Trial 11\n    }\n    \n    # For all these trials, Option A is the higher-evidence choice\n    # Response == 0 means chosen A\n    data['correct'] = (data['response'] == 0).astype(float)\n    \n    diffs = []\n    for subj, subj_df in data.groupby('subject_id'):\n        low_mask = subj_df['pair'].isin(low_conflict)\n        high_mask = subj_df['pair'].isin(high_conflict)\n        \n        if low_mask.sum() > 0 and high_mask.sum() > 0:\n            low_acc = subj_df.loc[low_mask, 'correct'].mean()\n            high_acc = subj_df.loc[high_mask, 'correct'].mean()\n            diffs.append(low_acc - high_acc)\n            \n    if not diffs:\n        return 0.0\n        \n    return float(np.mean(diffs))",
  "rationale": "Previous attempts mapped trial pairs that were not actually present in the experimental design, severely reducing the available data and statistical power. By explicitly targeting the exact low-conflict (Trials 1, 4, 10) and high-conflict (Trials 3, 5, 11) pairs from the experimental schedule, we can perfectly isolate the effect of total feature conflict while holding net evidence constant. Pure WADD predicts exactly zero difference in accuracy between these sets, whereas Evidence-Dependent Noise predicts a strictly positive difference (accuracy drop on high conflict trials due to increased decision noise). Averaging this paired difference within each subject provides a highly robust, low-variance contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 7175,
  "candidates_token_count": 736,
  "total_token_count": 10680
}
```
