# metric_exp01_attempt_01

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
**Validities (n_features=5):** [0.98, 0.92, 0.58, 0.54, 0.51]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Soft Threshold Tallying from Tallying (Equal Weight), we use a 5-feature design with two highly valid features and three low-validity features. Tallying simply counts the number of positive features, blind to their validities. Soft Threshold Tallying applies a sigmoid threshold that effectively masks out the low-validity features, tallying only the high-validity ones. By pitting options with fewer but high-validity features against options with more but low-validity features, we force the two models to make opposing predictions.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Soft Threshold Tallying posits that decision-makers use a compensatory tallying heuristic, but down-weight cues that are less reliable. They evaluate each feature's subjective validity against a personal threshold using a smooth sigmoid function, creating a soft inclusion mask. Features that fall well below the threshold are effectively ignored, while those above are tallied with near-equal weight. This smooth transition provides a bridge between pure equal-weight tallying and selective attention to high-validity information, while allowing for continuous optimization.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- threshold: [-5.0, 1.0]
- steepness: [1.0, 50.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Threshold Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    steepness = float(parameters["steepness"])
    
    # Soft threshold using a logistic sigmoid to create a smooth inclusion mask
    weights = 1.0 / (1.0 + np.exp(-steepness * (validities - threshold)))
    
    # Tally the features weighted by the soft threshold mask
    scores = np.sum(stim * weights, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Tallying (Equal Weight) theory posits that decision-makers simply count the number of positive features (or cues) for each option and choose the option with the higher tally, ignoring cue validities completely. This is a compensatory heuristic that treats all pieces of evidence equally.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Stimulus is the pair of option feature vectors for the current trial
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
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
[0] rationale: This metric isolates trials where the two theories make strictly opposing predictions. By filtering for trials where Option B has a higher total count of positive features, we isolate the cases where pure Equal Weight Tallying strongly predicts choosing B. However, the experimental design ensures that in these exact trials, Option A possesses more high-validity features. Thus, Soft Threshold Tallying, which down-weights low-validity features, will strongly predict choosing Option A. The metric simply calculates the proportion of Option A choices in this subset. Tallying will yield a value close to 0, whereas Soft Threshold Tallying will yield a value close to 1.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Filter to trials where Option B has strictly more positive features than Option A
    mask = sum_a < sum_b
    if not mask.any():
        return 0.0
        
    subset = data[mask]
    
    # Calculate the proportion of times the subject chose Option A (response == 0)
    # Pure Tallying predicts B (metric ~ 0), while Soft Threshold Tallying predicts A (metric ~ 1)
    # because Option A was designed to have fewer total features but more high-validity features.
    return float(np.mean(subset['response'] == 0))
outcome: self_sim=0.1748 (var=0.0326) adversary_sim=0.1383 (var=0.0078) welch_t=+0.906 p=0.3713 (N=25, alpha=0.01) -> reject

[1] rationale: To perfectly isolate the effect of down-weighting low-validity features, we control for the total number of features (the only variable Equal Weight Tallying cares about). We compare pairs of trials that have the exact same difference in total features (diff_total) but differ in the number of high-validity features (diff_high). Equal Weight Tallying predicts a difference of exactly zero for this metric, as it only responds to diff_total. Soft Threshold Tallying, however, will show a positive difference because it relies more heavily on diff_high when the threshold parameter acts to silence low-validity features.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate total feature counts and high-validity feature counts (first 3 features)
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    diff_total = sum_a - sum_b
    
    high_a = data['option_a_ratings'].apply(lambda x: sum(x[:3]))
    high_b = data['option_b_ratings'].apply(lambda x: sum(x[:3]))
    diff_high = high_a - high_b
    
    # Group 1: Trials where diff_total == -2
    # A1: diff_high > 0 (Trials 1, 2)
    # B1: diff_high == 0 (Trial 3)
    mask_a1 = (diff_total == -2) & (diff_high > 0)
    mask_b1 = (diff_total == -2) & (diff_high == 0)
    
    # Group 2: Trials where diff_total == -1
    # A2: diff_high == 2 (Trials 4, 7)
    # B2: diff_high == 1 (Trials 6, 8)
    mask_a2 = (diff_total == -1) & (diff_high == 2)
    mask_b2 = (diff_total == -1) & (diff_high == 1)
    
    # Calculate probability of choosing A (response == 0)
    p_a_a1 = 1.0 - data[mask_a1]['response'].mean() if mask_a1.any() else 0.0
    p_a_b1 = 1.0 - data[mask_b1]['response'].mean() if mask_b1.any() else 0.0
    
    p_a_a2 = 1.0 - data[mask_a2]['response'].mean() if mask_a2.any() else 0.0
    p_a_b2 = 1.0 - data[mask_b2]['response'].mean() if mask_b2.any() else 0.0
    
    # The metric is the sum of differences in P(Choice A) between high and low diff_high trials,
    # controlling exactly for diff_total. 
    return float((p_a_a1 - p_a_b1) + (p_a_a2 - p_a_b2))
outcome: self_sim=0.0158 (var=0.0393) adversary_sim=0.0183 (var=0.0270) welch_t=-0.049 p=0.9615 (N=25, alpha=0.01) -> reject

[2] rationale: To maximize the discrimination between Tallying and Soft Threshold Tallying, we contrast two specific trials that push the models in opposite directions. On Trial 7, Option A has fewer total features but they are high-validity, while Option B has more total features but they are low-validity. Tallying strongly prefers B, whereas Soft Threshold Tallying often prefers A. On Trial 5, both options only have low-validity features, with A having more than B. Tallying strongly prefers A, whereas Soft Threshold Tallying is often indifferent (P(A) ~ 0.5) because those features fall below the threshold. The difference P(A | Trial 7) - P(A | Trial 5) will be highly negative for Tallying, but significantly higher (less negative or positive) for Soft Threshold Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Create string representations to easily identify trials
    a_str = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Trial 7: A has 2 high-validity features, B has 3 low-validity features.
    # Tallying prefers B (sum 2 vs 3). STT often prefers A (ignores B's low-validity features).
    mask_t7 = (a_str == "011000") & (b_str == "000111")
    
    # Trial 5: A has 3 low-validity features, B has 2 low-validity features.
    # Tallying prefers A (sum 3 vs 2). STT is often indifferent (ignores both).
    mask_t5 = (a_str == "000111") & (b_str == "000110")
    
    p_a_t7 = float(np.mean(data.loc[mask_t7, 'response'] == 0)) if mask_t7.any() else 0.0
    p_a_t5 = float(np.mean(data.loc[mask_t5, 'response'] == 0)) if mask_t5.any() else 0.0
    
    # Tallying will yield a strongly negative value (low P(A|T7) - high P(A|T5)).
    # Soft Threshold Tallying will yield a significantly higher value because it boosts P(A|T7)
    # and reduces P(A|T5) towards 0.5.
    return p_a_t7 - p_a_t5

outcome: self_sim=-0.6550 (var=0.1136) adversary_sim=-0.7283 (var=0.0399) welch_t=+0.936 p=0.3552 (N=25, alpha=0.01) -> reject

[3] rationale: Trial 7 pits Option A (2 high-validity features) against Option B (3 low-validity features). Tallying evaluates purely on feature count and strongly predicts choosing Option B. Soft Threshold Tallying down-weights the low-validity features of B, systematically reducing the probability of choosing B compared to pure Tallying. By isolating the single most discriminative trial, we minimize the averaging effect of trials where the models agree, producing a tight point estimate that consistently separates the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Trial 7 is specifically designed to pit 2 high-validity features (Option A) 
    # against 3 low-validity features (Option B).
    # Identify Trial 7 by matching the exact feature strings.
    a_str = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    mask_t7 = (a_str == "011000") & (b_str == "000111")
    
    if not mask_t7.any():
        return 0.0
        
    # Return the proportion of times the subject chose Option B (response == 1)
    # Tallying strongly prefers B (3 vs 2 features), so this value will be close to 1.
    # Soft Threshold Tallying often down-weights B's low-validity features, 
    # shifting the preference toward A and thus yielding a significantly lower value.
    return float(data.loc[mask_t7, 'response'].mean())
outcome: self_sim=0.8100 (var=0.0461) adversary_sim=0.8450 (var=0.0178) welch_t=-0.692 p=0.4928 (N=25, alpha=0.01) -> reject

[4] rationale: To perfectly isolate the effect of continuous cue weighting and eliminate variance from the beta and epsilon parameters, we compare pairs of trials where Tallying predicts the exact same choice probabilities. In Trials 4 and 2, Option A has exactly 1 fewer feature than Option B. For Tallying, P(A|T4) - P(A|T2) is exactly 0. However, for Soft Threshold Tallying, the difference in evidence between T4 and T2 reduces to w2 - w5 (the weight of the 2nd most valid cue minus the weight of the least valid cue). Because STT assigns higher weights to more valid cues, this difference is strictly positive. The same logic applies to Trials 6 and 1, where Option A has exactly 1 more feature than Option B. Summing these two differences yields a metric that is exactly 0 in expectation for Tallying (with very low variance purely due to binomial noise), but strictly positive for Soft Threshold Tallying, maximizing the statistical contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Create string representations to easily identify trials
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Trial 1: A=[0,0,1,1,1] (sum=3), B=[1,1,0,0,0] (sum=2) -> Diff = +1
    mask_t1 = (a_str == "00111") & (b_str == "11000")
    # Trial 6: A=[0,0,1,1,0] (sum=2), B=[1,0,0,0,0] (sum=1) -> Diff = +1
    mask_t6 = (a_str == "00110") & (b_str == "10000")
    
    # Trial 2: A=[1,0,0,0,0] (sum=1), B=[0,0,1,1,0] (sum=2) -> Diff = -1
    mask_t2 = (a_str == "10000") & (b_str == "00110")
    # Trial 4: A=[1,1,0,0,0] (sum=2), B=[0,0,1,1,1] (sum=3) -> Diff = -1
    mask_t4 = (a_str == "11000") & (b_str == "00111")
    
    p_a_t1 = data.loc[mask_t1, 'response'].eq(0).mean() if mask_t1.any() else 0.5
    p_a_t6 = data.loc[mask_t6, 'response'].eq(0).mean() if mask_t6.any() else 0.5
    p_a_t2 = data.loc[mask_t2, 'response'].eq(0).mean() if mask_t2.any() else 0.5
    p_a_t4 = data.loc[mask_t4, 'response'].eq(0).mean() if mask_t4.any() else 0.5
    
    # The metric contrasts pairs of trials that have the EXACT SAME difference in feature counts.
    # Tallying predicts a difference of exactly 0 in expectation.
    return float((p_a_t4 - p_a_t2) + (p_a_t6 - p_a_t1))

outcome: self_sim=0.0075 (var=0.0312) adversary_sim=0.0325 (var=0.0222) welch_t=-0.541 p=0.5912 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Convert feature lists to strings for exact trial matching\n    a_str = data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    \n    # Identify the three trials where Option A has exactly 1 fewer feature than Option B.\n    # Tallying predicts the exact same choice probability for all three trials.\n    mask_t2 = (a_str == \"10000\") & (b_str == \"00110\")\n    mask_t4 = (a_str == \"11000\") & (b_str == \"00111\")\n    mask_t5 = (a_str == \"01100\") & (b_str == \"10011\")\n    \n    # Calculate probability of choosing Option A (response == 0)\n    p_a_t2 = data.loc[mask_t2, 'response'].eq(0).mean() if mask_t2.any() else 0.5\n    p_a_t4 = data.loc[mask_t4, 'response'].eq(0).mean() if mask_t4.any() else 0.5\n    p_a_t5 = data.loc[mask_t5, 'response'].eq(0).mean() if mask_t5.any() else 0.5\n    \n    # Tallying expectation: ((P_A + P_A) / 2) - P_A = 0\n    # Soft Threshold Tallying expectation: T2 and T4 strongly favor A (P_A near 1) because A has the highest validity cue,\n    # while T5 is balanced (P_A near 0.5) because the top validities are split. Result is significantly > 0.\n    return float((p_a_t2 + p_a_t4) / 2.0 - p_a_t5)",
  "rationale": "Tallying depends purely on the difference in the number of positive features. For Trials 2, 4, and 5, Option A has exactly 1 fewer positive feature than Option B (diff = -1). Thus, Tallying predicts the exact same probability of choosing Option A for all three trials, making `(P(A|T2) + P(A|T4))/2 - P(A|T5)` exactly 0 in expectation. Soft Threshold Tallying, however, cares about validities. In T2 and T4, Option A contains the highest validity feature, while Option B contains only low validity features, causing STT to strongly prefer Option A (P(A) near 1). In T5, the high validity features are split between A and B, making STT roughly indifferent (P(A) near 0.5). Therefore, for STT, this metric will be significantly positive. By comparing a near-1 probability to a near-0.5 probability, we avoid the saturation issue where subtracting two near-1 probabilities collapses the difference to 0."
}
```

## Usage

```json
{
  "prompt_token_count": 6400,
  "candidates_token_count": 762,
  "total_token_count": 13882
}
```
