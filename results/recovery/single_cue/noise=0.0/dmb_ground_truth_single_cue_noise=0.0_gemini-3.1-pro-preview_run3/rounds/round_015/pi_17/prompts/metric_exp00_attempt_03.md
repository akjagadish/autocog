# metric_exp00_attempt_03

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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the advocated Decisiveness-Dependent Strategy Selection model from the competing Feature Cancellation model, we manipulate the number of surviving (unshared) features while holding the tally difference constant. The advocated theory's strategy selection and resulting choice determinism depend solely on the absolute difference in tallies (delta_tally), completely ignoring shared features. In contrast, the competing theory scales its choice determinism (beta) inversely by the number of surviving features. We designed tied-tally trials (delta_tally = 0) with either 2 or 4 surviving features, and unequal-tally trials (delta_tally = 1) with either 3 or 5 surviving features. The advocated theory predicts identical determinism across trials with the same tally difference, whereas the competing theory is structurally forced to predict significantly noisier choices on trials with more surviving features.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decisiveness-Dependent Strategy Selection with Sharp Transition: Decision-makers probabilistically select between a compensatory Tallying strategy and a non-compensatory Take-The-Best (TTB) strategy on a trial-by-trial basis. The probability of using Tallying is a logistic function of the absolute difference in tally scores between the two options. By strictly constraining the sensitivity (theta) to be positive and the threshold to [0.1, 0.9], the model naturally transitions to a sharp step function where Tallying heavily dominates for decisive tally differences (delta >= 1), while TTB is strictly reserved as a tie-breaker for complex/tied stimuli (delta == 0).

**Parameters:**
- theta: [1.0, 20.0]
- threshold: [0.1, 0.9]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    delta_tally = abs(a_wins - b_wins)
    
    if a_wins > b_wins:
        p_a_tally = 1.0
    elif b_wins > a_wins:
        p_a_tally = 0.0
    else:
        p_a_tally = 0.5
        
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    p_a_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_a_ttb = 1.0
            break
        elif b[idx] > a[idx]:
            p_a_ttb = 0.0
            break
            
    # Strategy selection probability
    # Probability of using Tallying depends on the decisiveness of the tally
    exponent = -theta * (delta_tally - threshold)
    exponent = np.clip(exponent, -500.0, 500.0) # Prevent overflow
    p_use_tally = 1.0 / (1.0 + np.exp(exponent))
    
    p_a_core = p_use_tally * p_a_tally + (1.0 - p_use_tally) * p_a_ttb
    p_b_core = 1.0 - p_a_core
    
    p_core = np.array([p_a_core, p_b_core])
    
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
**Description:** Feature Cancellation then Tally/TTB with Cancellation-Scaled Determinism

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Feature cancellation: identify surviving distinct features
    mask = a != b
    
    # Tallying on distinct features
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    score_a = tally_a
    score_b = tally_b
    
    # If tallies are tied, break the tie using TTB on surviving features
    if tally_a == tally_b and np.sum(mask) > 0:
        order = np.argsort(val)[::-1]
        for idx in order:
            if mask[idx]:
                if a[idx] > b[idx]:
                    score_a += 1.0
                elif b[idx] > a[idx]:
                    score_b += 1.0
                break
            
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with cancellation-scaled determinism
    n_surviving = np.sum(mask)
    effective_beta = beta / n_surviving if n_surviving > 0 else beta
    
    z = effective_beta * (scores - np.max(scores))
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
[0] rationale: This metric calculates the difference in choice determinism (absolute deviation of the response rate from 0.5) between trials with few surviving features (2 or 3) and trials with many surviving features (4 or 5). The Advocated theory bases its strategy selection solely on the tally difference, which is matched across these sets, thus predicting near-zero difference in determinism. The Competing theory structurally scales its determinism inversely by the number of surviving features, forcing a higher determinism on trials with fewer surviving features.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    n_surviving = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        n_surviving.append(sum(1 for x, y in zip(a, b) if x != y))
        
    df = data.copy()
    df['n_surv'] = n_surviving
    
    subj_diffs = []
    for subj, subj_df in df.groupby('subject_id'):
        det = {}
        for n in [2, 3, 4, 5]:
            subset = subj_df[subj_df['n_surv'] == n]
            if len(subset) > 0:
                p = subset['response'].mean()
                det[n] = abs(p - 0.5)
            else:
                det[n] = np.nan
                
        low_surv_det = np.nanmean([det.get(2, np.nan), det.get(3, np.nan)])
        high_surv_det = np.nanmean([det.get(4, np.nan), det.get(5, np.nan)])
        
        subj_diffs.append(low_surv_det - high_surv_det)
        
    return float(np.nanmean(subj_diffs))
outcome: self_sim=-0.0067 (var=0.0042) adversary_sim=0.0221 (var=0.0054) welch_t=-1.469 p=0.1486 (N=25, alpha=0.01) -> reject

[1] rationale: By comparing the probability of choosing Option A on Trial 1 (2 surviving features) vs Trial 4 (5 surviving features), we can perfectly dissociate the two theories. In the Advocated theory, Trial 1 leads to strict TTB (due to a tally difference of 0) and Trial 4 leads to strict Tallying (due to a tally difference of 1); both strategies deterministically favor Option A, yielding a difference in P(Choose A) of ~0. In the Competing theory, both trials yield a final score difference of 1 favoring Option A, but the determinism is scaled inversely by the number of surviving features. Thus, the Competing theory predicts a significantly higher P(Choose A) on Trial 1 (beta/2) than on Trial 4 (beta/5), yielding a large positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Convert option A ratings to string to uniquely identify trials
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[1, 0, 1, 1, 1] (2 surviving features)
    t1_str = '10111'
    # Trial 4: A=[0, 1, 1, 1, 0] (5 surviving features)
    t4_str = '01110'
    
    subj_diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        t1_df = subj_df[subj_df['A_str'] == t1_str]
        t4_df = subj_df[subj_df['A_str'] == t4_str]
        
        if len(t1_df) > 0 and len(t4_df) > 0:
            # response == 0 means choice A
            p_A_T1 = (t1_df['response'] == 0).mean()
            p_A_T4 = (t4_df['response'] == 0).mean()
            subj_diffs.append(p_A_T1 - p_A_T4)
            
    if not subj_diffs:
        return 0.0
        
    return float(np.nanmean(subj_diffs))
outcome: self_sim=0.0217 (var=0.0143) adversary_sim=0.0725 (var=0.0091) welch_t=-1.664 p=0.103 (N=25, alpha=0.01) -> reject

[2] rationale: Instead of raw probability differences which squash to 0 when determinism is high, we compute the empirical smoothed log-odds of choosing Option A for each trial type. The Advocated theory bases its strategy choice strictly on the tally difference, which is identical for T1 vs T2 (both 0) and T3 vs T4 (both 1). Thus, its predicted log-odds are perfectly matched within these pairs, yielding an expected difference of exactly 0. In contrast, the Competing theory scales the log-odds inversely by the number of surviving features. Therefore, the log-odds for T1 (2 surviving) are strictly greater than T2 (4 surviving), and T3 (3 surviving) strictly greater than T4 (5 surviving). This metric avoids the saturation problem of raw probabilities and yields a large positive mean for the Competing theory while remaining strictly centered at 0 for the Advocated theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    n_surv_list = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        n_surv_list.append(sum(1 for x, y in zip(a, b) if x != y))
        
    df = data.copy()
    df['n_surv'] = n_surv_list
    
    subj_diffs = []
    for subj, subj_df in df.groupby('subject_id'):
        counts = {}
        totals = {}
        for n in [2, 3, 4, 5]:
            subset = subj_df[subj_df['n_surv'] == n]
            counts[n] = (subset['response'] == 0).sum()
            totals[n] = len(subset)
            
        L = {}
        for n in [2, 3, 4, 5]:
            A = counts[n]
            B = totals[n] - A
            L[n] = np.log((A + 0.5) / (B + 0.5))
            
        # T1 (n_surv=2) vs T2 (n_surv=4), both have tally diff = 0
        # T3 (n_surv=3) vs T4 (n_surv=5), both have tally diff = 1
        diff = (L[2] - L[4]) + (L[3] - L[5])
        subj_diffs.append(diff)
        
    return float(np.mean(subj_diffs))
outcome: self_sim=0.0045 (var=1.0647) adversary_sim=0.3813 (var=1.5271) welch_t=-1.170 p=0.2478 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    # Convert option A ratings to a string to uniquely identify the 4 trial types\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # Define the string identifiers for the 4 trials\n    t1_str = '10111' # Trial 1: 2 surviving features, delta_tally = 0\n    t2_str = '10101' # Trial 2: 4 surviving features, delta_tally = 0\n    t3_str = '01111' # Trial 3: 3 surviving features, delta_tally = 1\n    t4_str = '01110' # Trial 4: 5 surviving features, delta_tally = 1\n    \n    subj_scores = []\n    for subj, subj_df in data.groupby('subject_id'):\n        # Count the number of times Option B (response == 1) was chosen for each trial type\n        b1 = subj_df[subj_df['A_str'] == t1_str]['response'].sum()\n        b2 = subj_df[subj_df['A_str'] == t2_str]['response'].sum()\n        b3 = subj_df[subj_df['A_str'] == t3_str]['response'].sum()\n        b4 = subj_df[subj_df['A_str'] == t4_str]['response'].sum()\n        \n        # Calculate the normalized relative difference in B choices.\n        # The denominator (B_x + B_y + 1) stabilizes variance by down-weighting \n        # subjects with low determinism (high B counts, which are dominated by noise).\n        score1 = (b2 - b1) / (b1 + b2 + 1.0)\n        score2 = (b4 - b3) / (b3 + b4 + 1.0)\n        \n        subj_scores.append(score1 + score2)\n        \n    if len(subj_scores) == 0:\n        return 0.0\n        \n    return float(np.mean(subj_scores))",
  "rationale": "Advocated theory bases strategy selection solely on the absolute tally difference. Thus, it predicts identical choice probabilities for Trial 1 (2 surviving features) and Trial 2 (4 surviving features), as both have a tally difference of 0. Similarly, it predicts identical probabilities for Trial 3 and Trial 4 (tally difference of 1). Therefore, the expected difference in the number of B choices (B2 - B1 and B4 - B3) is exactly zero under the Advocated theory. The Competing theory structurally scales choice determinism inversely by the number of surviving features. Thus, it predicts a higher rate of B choices (the non-favored option) for trials with more surviving features (B2 > B1 and B4 > B3). To detect this difference robustly, we compute the normalized relative difference (B2 - B1) / (B1 + B2 + 1) + (B4 - B3) / (B3 + B4 + 1). This specific functional form is crucial: it strictly bounds the metric and heavily down-weights subjects with low determinism (who contribute massive binomial noise but almost zero signal), thereby dramatically increasing the signal-to-noise ratio compared to a simple linear difference in probabilities which failed previously."
}
```

## Usage

```json
{
  "prompt_token_count": 5330,
  "candidates_token_count": 836,
  "total_token_count": 32355
}
```
