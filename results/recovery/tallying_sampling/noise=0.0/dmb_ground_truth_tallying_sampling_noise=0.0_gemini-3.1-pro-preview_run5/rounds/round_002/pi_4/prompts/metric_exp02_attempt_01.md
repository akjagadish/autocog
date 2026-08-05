# metric_exp02_attempt_01

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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design uses a 5-feature environment with two highly valid features and three low-validity features to create a sharp divergence between Tallying and Soft Threshold Tallying. Pure Tallying weights all features equally, choosing the option with the most positive features. Soft Threshold Tallying applies a sigmoid threshold that effectively masks out the low-validity features, causing choices to be driven predominantly by the high-validity features. The trial pairs systematically pit options with fewer high-validity features against options with more low-validity features, leading to strictly opposing predictions. We also include trials where Tallying is indifferent but Soft Threshold Tallying has a strong preference.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



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

[5] rationale: Tallying depends purely on the difference in the number of positive features. For Trials 2, 4, and 5, Option A has exactly 1 fewer positive feature than Option B (diff = -1). Thus, Tallying predicts the exact same probability of choosing Option A for all three trials, making `(P(A|T2) + P(A|T4))/2 - P(A|T5)` exactly 0 in expectation. Soft Threshold Tallying, however, cares about validities. In T2 and T4, Option A contains the highest validity feature, while Option B contains only low validity features, causing STT to strongly prefer Option A (P(A) near 1). In T5, the high validity features are split between A and B, making STT roughly indifferent (P(A) near 0.5). Therefore, for STT, this metric will be significantly positive. By comparing a near-1 probability to a near-0.5 probability, we avoid the saturation issue where subtracting two near-1 probabilities collapses the difference to 0.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert feature lists to strings for exact trial matching
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Identify the three trials where Option A has exactly 1 fewer feature than Option B.
    # Tallying predicts the exact same choice probability for all three trials.
    mask_t2 = (a_str == "10000") & (b_str == "00110")
    mask_t4 = (a_str == "11000") & (b_str == "00111")
    mask_t5 = (a_str == "01100") & (b_str == "10011")
    
    # Calculate probability of choosing Option A (response == 0)
    p_a_t2 = data.loc[mask_t2, 'response'].eq(0).mean() if mask_t2.any() else 0.5
    p_a_t4 = data.loc[mask_t4, 'response'].eq(0).mean() if mask_t4.any() else 0.5
    p_a_t5 = data.loc[mask_t5, 'response'].eq(0).mean() if mask_t5.any() else 0.5
    
    # Tallying expectation: ((P_A + P_A) / 2) - P_A = 0
    # Soft Threshold Tallying expectation: T2 and T4 strongly favor A (P_A near 1) because A has the highest validity cue,
    # while T5 is balanced (P_A near 0.5) because the top validities are split. Result is significantly > 0.
    return float((p_a_t2 + p_a_t4) / 2.0 - p_a_t5)
outcome: self_sim=0.0306 (var=0.0263) adversary_sim=0.0200 (var=0.0089) welch_t=+0.283 p=0.7786 (N=25, alpha=0.01) -> reject

[6] rationale: This metric calculates the proportion of choices that align with Soft Threshold Tallying's predictions on the exact 5 trials where Soft Threshold Tallying and Equal Weight Tallying make strictly opposing predictions. Tallying purely counts features, while Soft Threshold Tallying prioritizes the high-validity features. By aggregating across all 5 opposing trials, we maximize the signal-to-noise ratio. Tallying will yield a metric near 0 (modulo lapse rate), while Soft Threshold Tallying will yield a metric near 1, guaranteeing a massive and highly significant gap.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Identify the 5 trials where Tallying and Soft Threshold Tallying make completely opposite predictions.
    # T1: A has more features, B has the higher validity ones. STT prefers B (1), Tallying prefers A (0).
    t1 = (a_str == "00111") & (b_str == "11000")
    # T2: B has more features, A has the higher validity one. STT prefers A (0), Tallying prefers B (1).
    t2 = (a_str == "10000") & (b_str == "00110")
    # T3: B has more features, A has the higher validity one. STT prefers A (0), Tallying prefers B (1).
    t3 = (a_str == "01000") & (b_str == "00111")
    # T4: B has more features, A has the higher validity ones. STT prefers A (0), Tallying prefers B (1).
    t4 = (a_str == "11000") & (b_str == "00111")
    # T6: A has more features, B has the higher validity one. STT prefers B (1), Tallying prefers A (0).
    t6 = (a_str == "00110") & (b_str == "10000")
    
    stt_aligned = 0
    total = 0
    
    if t1.any():
        stt_aligned += (data.loc[t1, 'response'] == 1).sum()
        total += t1.sum()
    if t2.any():
        stt_aligned += (data.loc[t2, 'response'] == 0).sum()
        total += t2.sum()
    if t3.any():
        stt_aligned += (data.loc[t3, 'response'] == 0).sum()
        total += t3.sum()
    if t4.any():
        stt_aligned += (data.loc[t4, 'response'] == 0).sum()
        total += t4.sum()
    if t6.any():
        stt_aligned += (data.loc[t6, 'response'] == 1).sum()
        total += t6.sum()
        
    if total == 0:
        return 0.5
        
    return float(stt_aligned / total)
outcome: self_sim=0.1745 (var=0.0286) adversary_sim=0.1427 (var=0.0089) welch_t=+0.820 p=0.4176 (N=25, alpha=0.01) -> reject

[7] rationale: Because the Soft Threshold Tallying parameter space draws `threshold` from [-5.0, 1.0], it behaves identically to Equal Weight Tallying for the majority of the distribution (when threshold is low, all features are tallied). To statistically discriminate the two models, we must maximize the penalty on the subset of subjects where the threshold is high enough to silence low-validity features. This metric computes a composite score: `P(A|T1, T6) - P(A|T2, T3, T4)`. Tallying strongly predicts A for the first group and B for the second, yielding a metric near +2.0. However, when STT's threshold is active, it predicts the exact opposite (yielding -3.0). Averaging this divergence over the parameter space guarantees a massive and highly significant downward shift in the mean metric for STT compared to Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert feature arrays to strings for exact trial matching
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Trials where Tallying strongly predicts Option A, but STT (with high threshold) predicts Option B
    t1 = (a_str == "00111") & (b_str == "11000")
    t6 = (a_str == "00110") & (b_str == "10000")
    
    # Trials where Tallying strongly predicts Option B, but STT (with high threshold) predicts Option A
    t2 = (a_str == "10000") & (b_str == "00110")
    t3 = (a_str == "01000") & (b_str == "00111")
    t4 = (a_str == "11000") & (b_str == "00111")
    
    # Calculate probability of choosing Option A (response == 0) for each trial type
    p_a_t1 = data.loc[t1, 'response'].eq(0).mean() if t1.any() else 0.5
    p_a_t6 = data.loc[t6, 'response'].eq(0).mean() if t6.any() else 0.5
    
    p_a_t2 = data.loc[t2, 'response'].eq(0).mean() if t2.any() else 0.5
    p_a_t3 = data.loc[t3, 'response'].eq(0).mean() if t3.any() else 0.5
    p_a_t4 = data.loc[t4, 'response'].eq(0).mean() if t4.any() else 0.5
    
    # Contrast the two groups of trials
    # Tallying expectation: ~ (1 + 1) - (0 + 0 + 0) = 2.0
    # Soft Threshold Tallying expectation: significantly lower than 2.0, because for the subset of 
    # subjects with high thresholds, the metric evaluates to ~ (0 + 0) - (1 + 1 + 1) = -3.0.
    return float((p_a_t1 + p_a_t6) - (p_a_t2 + p_a_t3 + p_a_t4))
outcome: self_sim=1.1275 (var=0.7159) adversary_sim=1.2863 (var=0.2217) welch_t=-0.820 p=0.4176 (N=25, alpha=0.01) -> reject

[8] rationale: This metric leverages the fundamental mathematical invariance of Equal Weight Tallying: it depends solely on the difference in the number of positive features. By pairing trials that have the exact same feature count difference, we create contrasts that evaluate to exactly 0 in expectation for Tallying, regardless of the subject's beta or epsilon parameters. 

Specifically:
1) Trials 4 and 8 both have a feature difference of 0. Tallying predicts P(A) = 0.5 for both, so `P(A|T4) - P(A|T8) = 0`.
2) Trials 1 and 2 both have a feature difference of -1. Tallying predicts identical probabilities, so `P(A|T1) - P(A|T2) = 0`.
3) Trials 6 and 5 both have a feature difference of +1. Tallying predicts identical probabilities, so `P(A|T6) - P(A|T5) = 0`.

However, for Soft Threshold Tallying, the subjective weights of the features strictly follow their validities (w1 > w2 > w3 > w4 > w5). When we compute the score differences for these pairs under STT, the contrasts algebraically reduce to positive sums of weight differences (e.g., `w2 - w5`). Because higher validities always receive equal or higher weights in STT, every single one of these three differences is strictly positive. Summing them yields a metric that is tightly centered at 0 for Tallying (with minimal variance due to subtracting matched binomials) but significantly strictly positive for Soft Threshold Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert feature arrays to string for exact matching
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1] (Diff = 0)
    t4 = (a_str == "11100") & (b_str == "00111")
    # Trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0] (Diff = 0)
    t8 = (a_str == "00111") & (b_str == "11100")
    
    # Trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1] (Diff = -1)
    t1 = (a_str == "11000") & (b_str == "00111")
    # Trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0] (Diff = -1)
    t2 = (a_str == "10000") & (b_str == "00110")
    
    # Trial 6: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0] (Diff = +1)
    t6 = (a_str == "00110") & (b_str == "10000")
    # Trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0] (Diff = +1)
    t5 = (a_str == "00111") & (b_str == "11000")
    
    p_a_t4 = data.loc[t4, 'response'].eq(0).mean() if t4.any() else 0.5
    p_a_t8 = data.loc[t8, 'response'].eq(0).mean() if t8.any() else 0.5
    
    p_a_t1 = data.loc[t1, 'response'].eq(0).mean() if t1.any() else 0.5
    p_a_t2 = data.loc[t2, 'response'].eq(0).mean() if t2.any() else 0.5
    
    p_a_t6 = data.loc[t6, 'response'].eq(0).mean() if t6.any() else 0.5
    p_a_t5 = data.loc[t5, 'response'].eq(0).mean() if t5.any() else 0.5
    
    # For Equal Weight Tallying, these differences are exactly 0 in expectation.
    diff_48 = p_a_t4 - p_a_t8
    diff_12 = p_a_t1 - p_a_t2
    diff_65 = p_a_t6 - p_a_t5
    
    return float(diff_48 + diff_12 + diff_65)
outcome: self_sim=0.0550 (var=0.1310) adversary_sim=-0.0200 (var=0.0593) welch_t=+0.860 p=0.3949 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    subj_scores = []\n    for subj, df_subj in data.groupby('subject_id'):\n        a_s = df_subj['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n        b_s = df_subj['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n        \n        # Trials where Tallying strongly predicts B (A has fewer features), \n        # but high-threshold STT strongly predicts A (A has high-validity features)\n        m123 = ((a_s == \"11000\") & (b_s == \"00111\")) | \\\n               ((a_s == \"10000\") & (b_s == \"00110\")) | \\\n               ((a_s == \"01000\") & (b_s == \"00111\"))\n               \n        # Trials where Tallying strongly predicts A (A has more features),\n        # but high-threshold STT strongly predicts B (B has high-validity features)\n        m567 = ((a_s == \"00111\") & (b_s == \"11000\")) | \\\n               ((a_s == \"00110\") & (b_s == \"10000\")) | \\\n               ((a_s == \"00011\") & (b_s == \"01000\"))\n               \n        p_a_123 = df_subj.loc[m123, 'response'].eq(0).mean() if m123.any() else 0.5\n        p_a_567 = df_subj.loc[m567, 'response'].eq(0).mean() if m567.any() else 0.5\n        \n        # Tallying score: strongly negative (~ -1.0)\n        # High-threshold STT score: strongly positive (~ +1.0)\n        score = p_a_123 - p_a_567\n        \n        # ReLU-like rectification isolates the sub-population of STT subjects \n        # who have a high enough threshold to diverge from Tallying.\n        subj_scores.append(max(0.0, float(score)))\n        \n    return float(np.mean(subj_scores))",
  "rationale": "Soft Threshold Tallying (STT) has a parameter space where for ~80-90% of the distribution (threshold < 0.5), it behaves exactly like Equal Weight Tallying because all features pass the threshold. Linear contrasts averaged across the entire parameter space fail to discriminate the theories because the Tallying-like majority washes out the signal. To fix this, we calculate a per-subject divergence score that is strongly negative for pure Tallying and strongly positive for STT with a high threshold. By applying a non-linear rectification `max(0, score)` before averaging across subjects, we strictly zero out the Tallying subjects (and the Tallying-like STT subjects) while preserving the massive positive spike from the high-threshold STT subjects. This guarantees a near-zero mean and variance for Tallying, but a significantly positive mean for STT."
}
```

## Usage

```json
{
  "prompt_token_count": 10436,
  "candidates_token_count": 814,
  "total_token_count": 17302
}
```
