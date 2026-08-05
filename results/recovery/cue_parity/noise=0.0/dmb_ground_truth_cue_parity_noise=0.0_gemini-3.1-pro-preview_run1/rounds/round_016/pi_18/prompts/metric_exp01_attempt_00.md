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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 8: A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** This design quantitatively dissociates the Advocated Theory (Parallel Similarity and Feature-Difference Model) from the Competing Theory (Environment Classification Strategy Selection) by exploiting the trial-by-trial similarity metric unique to the Advocated Theory. The Competing Theory evaluates the environment's validity dispersion once, establishing a fixed mixture weight between Take-The-Best (TTB) and Tallying for all trials. Consequently, if two trials have identical feature differences (Option A - Option B), the Competing Theory must predict the exact same choice probability for both, regardless of the options' baseline features. In contrast, the Advocated Theory computes the Jaccard similarity of the options' positive features on every trial, using this similarity to dynamically interpolate between Tallying and a Weighted Additive (WADD) strategy. By constructing pairs of trials with perfectly identical feature differences but vastly different numbers of shared positive features, we create a stark double dissociation: the Competing Theory predicts a flat, identical response profile across these matched pairs, whereas the Advocated Theory predicts significant shifts in choice probability as the varying similarity triggers a trial-by-trial transition between compensatory and non-compensatory processing.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Parallel Similarity and Feature-Difference Model: Decision-makers evaluate options in parallel by computing a similarity-weighted score. Instead of a discrete stopping rule, the strategy smoothly transitions between Tallying (equal weighting) and a heavily weighted linear model (WADD) based on the overall similarity of the options. A Jaccard-like similarity metric is used. To ensure stability, the feature weights for both strategies are normalized to sum to 1.0, and the softmax temperature parameter is expanded to accommodate the normalized scale, allowing for highly deterministic choice behavior.

**Parameters:**
- sim_threshold: [0.0, 1.0]
- slope: [-50.0, 50.0]
- gamma: [0.1, 10.0]
- beta: [0.1, 50.0]
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
    
    # Calculate similarity using Jaccard index (matching presences)
    a_bool = a > 0.5
    b_bool = b > 0.5
    intersection = np.sum(a_bool & b_bool)
    union = np.sum(a_bool | b_bool)
    sim = float(intersection) / float(union) if union > 0 else 1.0
    
    threshold = float(parameters["sim_threshold"])
    slope = float(parameters["slope"])
    
    # Logistic transition function:
    # With slope in [-50, 50], optimization can determine if high similarity 
    # leads to Tallying (alpha->1) or WADD (alpha->0).
    z_alpha = -slope * (sim - threshold)
    z_alpha = np.clip(z_alpha, -100, 100)
    alpha = 1.0 / (1.0 + np.exp(z_alpha))
    
    gamma = float(parameters["gamma"])
    # Non-linear scaling of validities for the WADD component, normalized to sum to 1
    w_wadd_raw = val ** gamma
    w_wadd = w_wadd_raw / np.sum(w_wadd_raw)
    
    # Tallying weights, normalized to sum to 1
    w_tally = np.ones_like(val) / len(val)
    
    # Interpolate feature weights
    weights = alpha * w_tally + (1.0 - alpha) * w_wadd
    
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
    beta = float(parameters["beta"])
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
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
[0] rationale: The experimental design contains pairs of trials with perfectly identical feature differences (Option A - Option B) but different absolute numbers of positive features (which alters Jaccard similarity). The Competing Theory evaluates options based purely on feature differences and an environment-level strategy mixture, predicting identical choice probabilities for trials with the same difference vector. The Advocated Theory, however, dynamically shifts its decision weights trial-by-trial based on the similarity of the options, predicting divergent choice probabilities. This metric calculates the average absolute difference in choice proportions between these matched trial pairs. It will be near zero for the Competing Theory and significantly positive for the Advocated Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    # Create a key based on the feature differences (Option A - Option B)
    data['diff_key'] = data.apply(lambda row: tuple(np.array(row['option_a_ratings']) - np.array(row['option_b_ratings'])), axis=1)
    
    # Create a key based on the total number of positive features in A to distinguish matched pairs
    data['sum_a'] = data.apply(lambda row: int(np.sum(row['option_a_ratings'])), axis=1)
    
    # Calculate the mean response (probability of choosing B) for each trial type
    grouped = data.groupby(['diff_key', 'sum_a'])['response'].mean().reset_index()
    
    # For each matched pair (identical diff_key), calculate the absolute difference in choice probability
    diffs = grouped.groupby('diff_key')['response'].agg(lambda x: x.max() - x.min())
    
    # Return the average absolute difference across the 4 pairs
    return float(diffs.mean())

outcome: self_sim=0.0358 (var=0.0115) adversary_sim=0.0150 (var=0.0028) welch_t=+0.871 p=0.3896 (N=25, alpha=0.01) -> reject

[1] rationale: The Competing Theory maintains a fixed strategy mixture across the entire experiment, meaning it predicts exactly the same choice probability for trials with identical feature differences (e.g., Trial 1 vs Trial 2). Thus, the shifts in P(B) within pairs (d1, d2, d3) are purely independent binomial noise with an expected value of 0, making the product of any two shifts reliably 0. In contrast, the Advocated Theory dynamically shifts its strategy trial-by-trial based on option similarity. When the similarity increases (from Trial 1 to 2, 3 to 4, etc.), the shift in probability is systematically coupled. Due to the carefully constructed validities, the shifts d1 and d3 will always have the same sign, while d2 will have the opposite sign. Therefore, the composite cross-product `d1*d3 - d1*d2 - d2*d3` will be strictly positive for the Advocated Theory, effectively canceling out parameter-specific directions (like positive vs negative slope) while maintaining a tight distribution.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_trial_num(a, b):
        a = tuple(a)
        b = tuple(b)
        if a == (1, 0, 0, 0, 0, 0, 0) and b == (0, 1, 1, 1, 0, 0, 0): return 1
        if a == (1, 0, 0, 0, 1, 1, 1) and b == (0, 1, 1, 1, 1, 1, 1): return 2
        if a == (0, 1, 1, 0, 0, 0, 0) and b == (1, 0, 0, 0, 0, 0, 0): return 3
        if a == (0, 1, 1, 0, 1, 1, 1) and b == (1, 0, 0, 0, 1, 1, 1): return 4
        if a == (1, 0, 0, 1, 0, 0, 0) and b == (0, 1, 1, 0, 0, 0, 0): return 5
        if a == (1, 0, 0, 1, 1, 1, 1) and b == (0, 1, 1, 0, 1, 1, 1): return 6
        if a == (0, 1, 0, 0, 0, 0, 0) and b == (1, 0, 1, 1, 0, 0, 0): return 7
        if a == (0, 1, 0, 0, 1, 1, 1) and b == (1, 0, 1, 1, 1, 1, 1): return 8
        return 0
        
    data = data.copy()
    data['trial_num'] = data.apply(lambda row: get_trial_num(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    
    m = data.groupby('trial_num')['response'].mean()
    
    def get_m(t):
        return float(m[t]) if t in m else 0.5
        
    # Calculate the shift in P(Choose B) between matched pairs of identical feature differences
    d1 = get_m(2) - get_m(1)
    d2 = get_m(4) - get_m(3)
    d3 = get_m(6) - get_m(5)
    
    # Cross-multiply the shifts to extract the systematic signature of dynamic strategy transitioning
    return float(d1 * d3 - d1 * d2 - d2 * d3)
outcome: self_sim=0.0005 (var=0.0606) adversary_sim=0.0001 (var=0.0022) welch_t=+0.007 p=0.9942 (N=25, alpha=0.01) -> reject

[2] rationale: This metric measures the difference in the probability of choosing Option A between two specific sets of trials: D1 (Trials 1 and 2) and D4 (Trials 7 and 8). In D1, Option A possesses the most valid feature (v1=0.95), while Option B possesses features 2, 3, and 4. In D4, the features are identical except that v1 and v2 are swapped between the options. 

For the Competing Theory, the Take-The-Best (TTB) strategy heavily favors A in D1 and B in D4. Meanwhile, the Tallying strategy predicts exactly the same (favoring B) for both sets because the count difference is identical. Since the Competing Theory is a fixed mixture of TTB and Tallying, subtracting the D4 probability from the D1 probability completely cancels out the Tallying component, isolating the TTB component. This results in a very large, strictly positive difference (typically ~0.60).

For the Advocated Theory, the decision is based on a mixture of WADD and Tallying. Since the validities of v1 (0.95) and v2 (0.85) are very close, swapping them between the options has a minimal effect on the WADD score for most parameter values. Consequently, the Advocated Theory predicts nearly identical choice probabilities for D1 and D4, resulting in a metric value close to zero. This creates a massive mean gap between the theories with relatively low between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # D1 (Trials 1 and 2): Option A has feature 1, Option B has features 2, 3, 4
    # D4 (Trials 7 and 8): Option A has feature 2, Option B has features 1, 3, 4
    # Response is 0 if A is chosen, 1 if B is chosen. 
    # Probability of choosing A is 1 - response.
    
    def is_d1(a):
        a = tuple(a)
        return a == (1, 0, 0, 0, 0, 0, 0) or a == (1, 0, 0, 0, 1, 1, 1)
        
    def is_d4(a):
        a = tuple(a)
        return a == (0, 1, 0, 0, 0, 0, 0) or a == (0, 1, 0, 0, 1, 1, 1)
        
    data = data.copy()
    data['is_d1'] = data['option_a_ratings'].apply(is_d1)
    data['is_d4'] = data['option_a_ratings'].apply(is_d4)
    
    d1_data = data[data['is_d1']]
    d4_data = data[data['is_d4']]
    
    if len(d1_data) == 0 or len(d4_data) == 0:
        return 0.0
        
    p_a_d1 = 1.0 - d1_data['response'].mean()
    p_a_d4 = 1.0 - d4_data['response'].mean()
    
    return float(p_a_d1 - p_a_d4)
outcome: self_sim=0.1708 (var=0.0781) adversary_sim=0.2908 (var=0.0982) welch_t=-1.429 p=0.1595 (N=25, alpha=0.01) -> reject

[3] rationale: The experimental design pairs trials (e.g., 1 & 2, 3 & 4) that have identical feature differences between Option A and Option B, but different overall counts of shared positive features (altering Jaccard similarity). The Competing Theory processes only the feature differences, and thus predicts identical underlying choice probabilities for both trials in a pair. The Advocated Theory dynamically adjusts its strategy based on similarity, leading to diverging underlying probabilities. A simple squared difference of observed choice proportions between paired trials is biased upwards by binomial noise, which previously caused failure. This metric computes an unbiased estimator of the squared difference in underlying choice probabilities by subtracting the unbiased variance estimate of the sample proportions. For the Competing Theory, the expectation of this metric is exactly 0. For the Advocated Theory, it will be strictly positive, providing a clean separation immune to binomial noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_trial_num(a):
        a = tuple(a)
        if a == (1, 0, 0, 0, 0, 0, 0): return 1
        if a == (1, 0, 0, 0, 1, 1, 1): return 2
        if a == (0, 1, 1, 0, 0, 0, 0): return 3
        if a == (0, 1, 1, 0, 1, 1, 1): return 4
        if a == (1, 0, 0, 1, 0, 0, 0): return 5
        if a == (1, 0, 0, 1, 1, 1, 1): return 6
        if a == (0, 1, 0, 0, 0, 0, 0): return 7
        if a == (0, 1, 0, 0, 1, 1, 1): return 8
        return 0
        
    data = data.copy()
    data['trial_num'] = data['option_a_ratings'].apply(get_trial_num)
    
    agg = data.groupby('trial_num')['response'].agg(['mean', 'count'])
    
    total_unbiased_sq = 0.0
    pairs = [(1,2), (3,4), (5,6), (7,8)]
    valid_pairs = 0
    
    for low, high in pairs:
        if low in agg.index and high in agg.index:
            p1 = float(agg.loc[low, 'mean'])
            n1 = float(agg.loc[low, 'count'])
            p2 = float(agg.loc[high, 'mean'])
            n2 = float(agg.loc[high, 'count'])
            
            if n1 > 1 and n2 > 1:
                diff2 = (p1 - p2)**2
                var_est = (p1 * (1.0 - p1)) / (n1 - 1.0) + (p2 * (1.0 - p2)) / (n2 - 1.0)
                total_unbiased_sq += (diff2 - var_est)
                valid_pairs += 1
                
    if valid_pairs == 0:
        return 0.0
        
    return float(total_unbiased_sq / valid_pairs)
outcome: self_sim=0.0010 (var=0.0064) adversary_sim=-0.0004 (var=0.0004) welch_t=+0.088 p=0.9302 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def is_trial_1(a, b):\n        a_tup = tuple(int(x) for x in a)\n        b_tup = tuple(int(x) for x in b)\n        return a_tup == (1, 0, 0, 0, 0, 0) and b_tup == (0, 1, 1, 0, 0, 0)\n    \n    def is_trial_7(a, b):\n        a_tup = tuple(int(x) for x in a)\n        b_tup = tuple(int(x) for x in b)\n        return a_tup == (0, 1, 0, 0, 0, 0) and b_tup == (0, 0, 1, 1, 0, 0)\n        \n    data = data.copy()\n    data['is_t1'] = data.apply(lambda row: is_trial_1(row['option_a_ratings'], row['option_b_ratings']), axis=1)\n    data['is_t7'] = data.apply(lambda row: is_trial_7(row['option_a_ratings'], row['option_b_ratings']), axis=1)\n    \n    t1_data = data[data['is_t1']]\n    t7_data = data[data['is_t7']]\n    \n    if len(t1_data) == 0 or len(t7_data) == 0:\n        return 0.0\n        \n    # Return difference in P(Choose B). \n    # A positive value means A was chosen more often in Trial 7 than in Trial 1.\n    return float(t1_data['response'].mean() - t7_data['response'].mean())",
  "rationale": "This metric isolates the difference in the probability of choosing Option A between Trial 1 and Trial 7. In the Competing Theory, both Trial 1 and Trial 7 produce identical predictions. For both trials, Take-The-Best (TTB) favors Option A (due to the highest differing validity being in favor of A), and Tallying favors Option B by exactly the same margin (1 positive feature for A vs 2 for B). Because the Competing Theory uses a fixed mixture of TTB and Tallying for each subject, it predicts the exact same choice probability for both trials. Thus, the expected difference P(A|Trial 7) - P(A|Trial 1) is exactly 0. \n\nIn the Advocated Theory, Trial 1 and Trial 7 have the same Jaccard similarity (sim = 0), meaning the dynamic strategy weight (alpha) is identical. The Tallying score is also identical. However, the Weighted Additive (WADD) score differs. For Trial 1, the WADD score difference is w1 - w2 - w3. For Trial 7, it is w2 - w3 - w4. Because of the validities (0.95, 0.9, 0.85, 0.6), the WADD score for Trial 7 is mathematically guaranteed to be strictly greater than for Trial 1 across all valid parameter ranges (i.e., w2 - w3 - w4 > w1 - w2 - w3). Consequently, the Advocated Theory systematically predicts a higher probability of choosing A in Trial 7 than in Trial 1. \n\nBy calculating `mean(response_Trial_1) - mean(response_Trial_7)`, we obtain a metric that is near 0 for the Competing Theory but significantly positive for the Advocated Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 7084,
  "candidates_token_count": 856,
  "total_token_count": 26876
}
```
