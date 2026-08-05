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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate Stochastic Take-The-Best (Search Error) from Strategy Selection (Mixture of TTB and WADD), we vary the number of lower-validity 'backup' features favoring Option B while keeping the top discriminating feature favoring Option A. In Stochastic TTB, the probability of choosing Option A is predominantly determined by the first discriminating feature (yielding 1 - p_skip probability for A). Adding more features that favor B only captures the small remaining probability mass (p_skip) that leaks past the first feature, resulting in very minor decreases in the probability of choosing A. In contrast, the Strategy Selection model incorporates a Weighted Additive (WADD) component that evaluates the sum of all features. When Option B has many lower-validity features, WADD strongly favors B over A, causing a massive shift in the Strategy Selection model's predicted choice probabilities. By comparing trials where B has one backup feature versus many, we can observe either a tiny shift (supporting Stochastic TTB) or a massive shift (supporting Strategy Selection).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Stochastic Take-The-Best (Search Error)

**Parameters:**
- p_skip: [0.0, 0.5]
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
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Search through features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_skip = float(parameters["p_skip"])
    epsilon = float(parameters["epsilon"])
    
    p_A_core = 0.0
    p_B_core = 0.0
    prob_reach = 1.0
    
    # For each feature, there is a chance (1 - p_skip) to evaluate it correctly.
    # If it discriminates, we stop. Otherwise, or if skipped, we continue to the next.
    for idx in order:
        if a[idx] > b[idx]:
            p_A_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
        elif b[idx] > a[idx]:
            p_B_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
            
    # If all features are skipped or none discriminate, guess randomly
    p_A_core += prob_reach * 0.5
    p_B_core += prob_reach * 0.5
    
    p_core = np.array([p_A_core, p_B_core])
    
    # Apply general response lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Strategy Selection (Mixture of TTB and WADD): Decision-makers are not strictly bound to a single decision strategy. Instead, there is a mixture of strategies used either across the population or within individuals on a trial-by-trial basis. Specifically, individuals choose between a non-compensatory heuristic (Take-The-Best) and a compensatory strategy (Weighted Additive Model). TTB relies solely on the highest-validity discriminating feature, while WADD computes a weighted sum of all features using their validities. The parameter 'p_wadd' dictates the probability of using WADD over TTB, allowing the model to capture both strict one-reason decision making and sensitivity to lower-validity cues when they strongly favor one option. The baseline probability of using WADD is restricted to reflect that non-compensatory heuristics are predominant.

**Parameters:**
- p_wadd: [0.1, 0.5]
- beta: [0.1, 20.0]
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
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take-The-Best (TTB) Strategy ---
    order = np.argsort(validities)[::-1]
    a_wins = 0.0
    b_wins = 0.0
    for idx in order:
        if a[idx] > b[idx]:
            a_wins = 1.0
            break
        elif b[idx] > a[idx]:
            b_wins = 1.0
            break
            
    if a_wins == 0.0 and b_wins == 0.0:
        p_ttb = np.array([0.5, 0.5])
    else:
        p_ttb = np.array([a_wins, b_wins])
        
    # --- Weighted Additive (WADD) Strategy ---
    scores = stim @ validities
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd_dist = e / e.sum()
    
    # --- Mixture Model ---
    p_wadd = float(parameters["p_wadd"])
    p_core = p_wadd * p_wadd_dist + (1.0 - p_wadd) * p_ttb
    
    # --- Lapse Rate ---
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
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
[0] rationale: The experimental design constructs matched pairs of trials (e.g., Trial 1 vs Trial 2) where the Weighted Additive (WADD) difference between Option A and Option B is mathematically identical, and standard Take-The-Best (TTB) always favors Option A. The Strategy Selection model (Mixture of TTB and WADD) therefore predicts the exact same probability of choosing Option A for both trials in a pair. However, Stochastic TTB evaluates features sequentially and can skip them; it predicts a higher probability of choosing Option A when A's secondary favorable features appear earlier in the validity ranking (which happens in the 'odd' trials: 1, 3, 5). This metric calculates the difference in the probability of choosing Option A between these 'odd' trials and the matched 'even' trials (2, 4, 6). The Strategy Selection model predicts a difference of 0, whereas Stochastic TTB predicts a strictly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    odd_A = {(1, 0, 1, 0, 0, 1), (1, 1, 0, 0, 0, 1), (1, 1, 0, 0, 1, 0)}
    even_A = {(1, 0, 0, 1, 1, 0), (1, 0, 1, 0, 1, 0), (1, 0, 1, 1, 0, 0)}
    
    # The 'A' option in the design always has a 1 for the highest validity feature (index 0)
    # We identify if the subject chose this 'A' option, robust to potential left/right swapping
    chose_A = ((a_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 0)) | \
              ((b_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 1))
              
    is_odd = a_tup.isin(odd_A) | b_tup.isin(odd_A)
    is_even = a_tup.isin(even_A) | b_tup.isin(even_A)
    
    p_a_odd = chose_A[is_odd].mean()
    p_a_even = chose_A[is_even].mean()
    
    if np.isnan(p_a_odd) or np.isnan(p_a_even):
        return 0.0
        
    return float(p_a_odd - p_a_even)
outcome: self_sim=0.0379 (var=0.0082) adversary_sim=0.0083 (var=0.0069) welch_t=+1.204 p=0.2346 (N=25, alpha=0.01) -> reject

[1] rationale: In the previous iteration, the metric averaged across all three trial pairs. However, the theoretical gap between Stochastic TTB and Strategy Selection is not uniform across pairs. Specifically, for Trials 3/5 vs 4/6, Stochastic TTB predicts a large difference in choice probabilities because the secondary winning feature is shifted from rank 2 (index 1) to rank 3 (index 2), causing a first-order drop in sequential reach probability. Strategy Selection, however, predicts a difference of exactly zero because the WADD differences and standard TTB predictions are mathematically identical across these matched pairs. By isolating only these high-leverage trial pairs, we maximize the expected mean difference while filtering out the noise from Trial 1/2 which has a much smaller theoretical effect size.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    # Group 1 (Trials 3 & 5): Target option wins on features 0 and 1.
    # Group 2 (Trials 4 & 6): Target option wins on features 0 and 2, and perfectly matches Group 1 on WADD difference.
    group1_targets = {(1, 1, 0, 0, 0, 1), (1, 1, 0, 0, 1, 0)}
    group2_targets = {(1, 0, 1, 0, 1, 0), (1, 0, 1, 1, 0, 0)}
    
    # Identify if the subject chose the 'Target' option (which always has a 1 for the highest validity feature)
    chose_target = ((a_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 0)) | \
                   ((b_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 1))
                   
    is_group1 = a_tup.isin(group1_targets) | b_tup.isin(group1_targets)
    is_group2 = a_tup.isin(group2_targets) | b_tup.isin(group2_targets)
    
    p_target_g1 = chose_target[is_group1].mean()
    p_target_g2 = chose_target[is_group2].mean()
    
    if np.isnan(p_target_g1) or np.isnan(p_target_g2):
        return 0.0
        
    return float(p_target_g1 - p_target_g2)
outcome: self_sim=0.0463 (var=0.0109) adversary_sim=0.0019 (var=0.0074) welch_t=+1.643 p=0.1071 (N=25, alpha=0.01) -> reject

[2] rationale: Previous attempts using all pairs or Trials 3-6 failed because the high binomial variance in trials with smaller WADD differences drowned out the mean gap. By isolating only Trial 5 vs Trial 6, we maintain the exact same theoretical signal for Stochastic TTB (difference at feature 1, proportional to p_skip * (1 - p_skip)), while minimizing the variance for the Strategy Selection model. Strategy Selection predicts a choice probability very close to 1 for both these trials due to the maximal WADD difference (0.30), which severely restricts the binomial variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    # Trial 5 and Trial 6 have the largest WADD difference (0.30), meaning the Strategy Selection 
    # model predicts a choice probability for A very close to 1 (subject to epsilon lapse). 
    # This minimizes the binomial variance of the responses.
    # Trial 5: Option A wins on feature 1.
    # Trial 6: Option B wins on feature 1.
    t5_target = (1, 1, 0, 0, 1, 0)
    t6_target = (1, 0, 1, 1, 0, 0)
    
    chose_target = ((a_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 0)) | \
                   ((b_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 1))
                   
    is_t5 = a_tup.apply(lambda x: x == t5_target) | b_tup.apply(lambda x: x == t5_target)
    is_t6 = a_tup.apply(lambda x: x == t6_target) | b_tup.apply(lambda x: x == t6_target)
    
    p_t5 = chose_target[is_t5].mean()
    p_t6 = chose_target[is_t6].mean()
    
    if np.isnan(p_t5) or np.isnan(p_t6):
        return 0.0
        
    return float(p_t5 - p_t6)
outcome: self_sim=0.0713 (var=0.0156) adversary_sim=-0.0137 (var=0.0135) welch_t=+2.491 p=0.01627 (N=25, alpha=0.01) -> reject

[3] rationale: Stochastic TTB predicts a higher probability of choosing Option A in trials where A's secondary favorable features appear earlier in the search order. This difference is largest for Trial 5 vs Trial 6, and similarly large for Trial 3 vs Trial 4. Strategy Selection, however, predicts exactly zero difference between 5 and 6, and zero difference between 3 and 4, because the WADD differences and standard TTB predictions are identical within these matched pairs. Previous attempts showed that Trial 5 vs Trial 6 provides a strong signal but borders on the significance threshold due to sample size. By combining the difference from Trial 5 vs 6 with the difference from Trial 3 vs 4, weighted by 0.5 to account for the higher binomial variance in Trials 3/4 (due to their smaller WADD difference), we maximize the signal-to-noise ratio. This linear combination effectively increases the theoretical mean gap for Stochastic TTB while keeping the Strategy Selection variance strictly controlled, pushing the t-statistic well past the significance threshold.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    # Option A in the design always has a 1 for the highest validity feature (index 0)
    chose_target = ((a_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 0)) | \
                   ((b_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 1))
                   
    t3 = (1, 1, 0, 0, 0, 1)
    t4 = (1, 0, 1, 0, 1, 0)
    t5 = (1, 1, 0, 0, 1, 0)
    t6 = (1, 0, 1, 1, 0, 0)
    
    is_t3 = (a_tup == t3) | (b_tup == t3)
    is_t4 = (a_tup == t4) | (b_tup == t4)
    is_t5 = (a_tup == t5) | (b_tup == t5)
    is_t6 = (a_tup == t6) | (b_tup == t6)
    
    p_t3 = chose_target[is_t3].mean()
    p_t4 = chose_target[is_t4].mean()
    p_t5 = chose_target[is_t5].mean()
    p_t6 = chose_target[is_t6].mean()
    
    if np.isnan(p_t5) or np.isnan(p_t6) or np.isnan(p_t3) or np.isnan(p_t4):
        return 0.0
        
    # Combine the two high-leverage pairs. We weight Trial 3 vs 4 by 0.5 because 
    # Strategy Selection has a smaller WADD difference for this pair, leading to 
    # choice probabilities closer to 0.5 and thus higher binomial variance. 
    return float((p_t5 - p_t6) + 0.5 * (p_t3 - p_t4))
outcome: self_sim=0.0819 (var=0.0235) adversary_sim=-0.0050 (var=0.0176) welch_t=+2.144 p=0.03727 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    a_ratings = data['option_a_ratings'].tolist()\n    b_ratings = data['option_b_ratings'].tolist()\n    responses = data['response'].tolist()\n    \n    chose_A_one_backup = []\n    chose_A_many_backup = []\n    \n    for a, b, resp in zip(a_ratings, b_ratings, responses):\n        # Identify the first discriminating feature to determine which option is 'A' (the TTB favorite)\n        for i in range(5):\n            if a[i] != b[i]:\n                a_is_better = (a[i] > b[i])\n                break\n        else:\n            continue\n            \n        if a_is_better:\n            b_option = b\n            chose_A = (resp == 0)\n        else:\n            b_option = a\n            chose_A = (resp == 1)\n            \n        # Count how many features favor option B\n        b_sum = sum(b_option)\n        \n        if b_sum >= 3:\n            chose_A_many_backup.append(int(chose_A))\n        elif b_sum == 1:\n            chose_A_one_backup.append(int(chose_A))\n            \n    if not chose_A_one_backup or not chose_A_many_backup:\n        return 0.0\n        \n    return float(np.mean(chose_A_one_backup) - np.mean(chose_A_many_backup))",
  "rationale": "By directly comparing trials where Option B has 1 backup feature (Trials 2, 4, 6) against trials where Option B has 3 or 4 backup features (Trials 1, 3, 5), we heavily exploit the WADD component in the Strategy Selection model. Strategy Selection predicts a massive drop in the probability of choosing Option A when B has many backup features because the compensatory WADD strategy strongly flips to favor B. In contrast, Stochastic TTB evaluates sequentially and predicts only a very tiny difference between these trial types, as the choice is still dominantly driven by the first discriminating feature."
}
```

## Usage

```json
{
  "prompt_token_count": 6458,
  "candidates_token_count": 555,
  "total_token_count": 9898
}
```
