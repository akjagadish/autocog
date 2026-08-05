# metric_exp01_attempt_03

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

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the 'Diminishing Returns and Cue Count Dilution Theory' (Advocated) from the 'Evidence Averaging Heuristic Theory' (Competing) by exploiting the mathematical bounds on how shared cues dilute evidence differences. Under the Competing theory, the difference between two options is divided by `n + prior_count`. When moving from 1 active cue to 3 active cues (by adding 2 shared cues), the difference shrinks by a factor of `(1 + prior_count) / (3 + prior_count)`, which is strictly bounded to be greater than 1/3. Consequently, if there is a strong preference in the 1-cue baseline, the Competing theory mathematically guarantees a relatively strong preference remains in the 3-cue trial. In contrast, the Advocated theory divides by `n^gamma`. When moving from 1 to 3 cues, the difference shrinks by `1 / 3^gamma`. Since gamma can be up to 3.0, this shrinkage factor can be as extreme as 1/27. Thus, the Advocated theory can predict a massive, near-complete collapse to chance (50%) when shared cues are added, which the Competing theory's linear denominator mathematically cannot replicate.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Diminishing Returns and Cue Count Dilution Theory: Decision-makers evaluate options by integrating the validities of active cues, but they perceive validity through a concave transformation that reflects diminishing marginal returns for extremely high-validity cues. Furthermore, the total integrated evidence for an option is diluted by the raw count of its active features. This captures the heuristic that presenting 'more arguments' can reduce the perceived average strength of the evidence, penalizing options that rely on many weak cues without imposing a strict capacity limit or a hard zero-bound.

**Parameters:**
- lambda_val: [0.1, 1.0]
- gamma: [0.0, 3.0]
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
    
    # Concave transformation of validities for diminishing returns
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    # Integrate evidence and dilute by the count of active features
    v_a = np.sum(w * a) / (n_a ** gamma) if n_a > 0 else 0.0
    v_b = np.sum(w * b) / (n_b ** gamma) if n_b > 0 else 0.0
    
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
**Description:** Decision-makers evaluate options based on the relative average validity of their active cues, rather than the simple sum. A base prior is included to prevent zero-division and establish a baseline expectation. This average-based integration naturally predicts the 'less-is-more' effect, as adding weak cues dilutes the overall average evidence of an option, while shared cues have a context-dependent impact by simultaneously altering the numerator and denominator.

**Parameters:**
- lambda_val: [0.1, 10.0]
- prior_count: [0.01, 10.0]
- prior_sum: [0.0, 10.0]
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
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    # Calculate average validity with a base prior to prevent zero-division
    v_a = (np.sum(w * a) + prior_sum) / (n_a + prior_count)
    v_b = (np.sum(w * b) + prior_sum) / (n_b + prior_count)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)
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
[0] rationale: The metric computes the difference in the probability of choosing Option A between Trial 1 (where A has 1 top cue) and Trial 2 (where A has 2 top cues), against a constant Option B (1 weak cue). The Advocated theory predicts a massive drop in preference for A (positive metric value) due to exponential cue count dilution (n^gamma). The Competing theory computes an average validity, which for Option A remains relatively constant (or increases slightly depending on the prior) because all active cues have the same high validity. This leads to a zero or negative metric value for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10000') & (b_str == '00001')
    t2_mask = (a_str == '11000') & (b_str == '00001')
    
    p_a_t1 = 1.0 - data[t1_mask]['response'].mean() if t1_mask.sum() > 0 else 0.5
    p_a_t2 = 1.0 - data[t2_mask]['response'].mean() if t2_mask.sum() > 0 else 0.5
    
    return float(p_a_t1 - p_a_t2)
outcome: self_sim=0.2767 (var=0.1104) adversary_sim=0.1642 (var=0.0585) welch_t=+1.369 p=0.1781 (N=25, alpha=0.01) -> reject

[1] rationale: In Trial 4, options A (1 cue) and B (2 cues) both consist entirely of high-validity cues. The Competing Theory (Evidence Averaging) evaluates options by averaging cues with a prior. Since the active cues are highly valid, adding a second identical cue pulls the average up (away from the lower prior), making B strictly better than A. Thus, it predicts a high probability of choosing B. The Advocated Theory (Cue Count Dilution) divides the sum of evidence by n^gamma. For gamma > 1, the denominator grows exponentially, penalizing B so heavily that its value drops below A's. Thus, the Advocated Theory predicts a much lower probability of choosing B. We simply measure the proportion of times B is chosen in Trial 4.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t4_mask = (a_str == '10000') & (b_str == '11000')
    
    if t4_mask.sum() == 0:
        return 0.5
        
    # Response is 1 if subject chose B (11000), 0 if A (10000)
    return float(data[t4_mask]['response'].mean())
outcome: self_sim=0.3650 (var=0.1007) adversary_sim=0.4025 (var=0.0407) welch_t=-0.499 p=0.6207 (N=25, alpha=0.01) -> reject

[2] rationale: The metric computes the difference in the probability of choosing Option A between Trial 1 (where A has 1 top cue) and Trial 3 (where A has 4 top cues), against a constant Option B (1 weak cue). The Advocated theory predicts a massive drop in preference for A (positive metric value) due to exponential cue count dilution (n^gamma). In Trial 3, dividing by 4^gamma heavily penalizes Option A, potentially dropping its value below Option B's. The Competing theory computes an average validity, which for Option A remains relatively constant and high across both trials because all active cues have the same high validity. This leads to a near-zero metric value for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10000') & (b_str == '00001')
    t3_mask = (a_str == '11110') & (b_str == '00001')
    
    p_a_t1 = 1.0 - data[t1_mask]['response'].mean() if t1_mask.sum() > 0 else 0.5
    p_a_t3 = 1.0 - data[t3_mask]['response'].mean() if t3_mask.sum() > 0 else 0.5
    
    return float(p_a_t1 - p_a_t3)
outcome: self_sim=0.3458 (var=0.1324) adversary_sim=0.2092 (var=0.1139) welch_t=+1.377 p=0.1749 (N=25, alpha=0.01) -> reject

[3] rationale: To maximize the discrimination between the theories, we measure the overall tendency to choose the option with FEWER cues across all trials where cue counts differ (Trials 2, 3, and 4). The Advocated theory heavily penalizes options with more cues due to exponential cue count dilution (n^gamma), making the 1-cue option highly attractive. The Competing theory averages evidence, so adding more top-validity cues generally increases or maintains the option's value, making the 1-cue option less attractive. By averaging the preference for the 1-cue option across these three distinct choice contexts, we aggregate the signal and reduce per-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t2_mask = (a_str == '11000') & (b_str == '00001')
    t3_mask = (a_str == '11110') & (b_str == '00001')
    t4_mask = (a_str == '10000') & (b_str == '11000')
    
    p_1cue_t2 = data[t2_mask]['response'].mean() if t2_mask.sum() > 0 else 0.5
    p_1cue_t3 = data[t3_mask]['response'].mean() if t3_mask.sum() > 0 else 0.5
    p_1cue_t4 = 1.0 - data[t4_mask]['response'].mean() if t4_mask.sum() > 0 else 0.5
    
    return float((p_1cue_t2 + p_1cue_t3 + p_1cue_t4) / 3.0)
outcome: self_sim=0.5931 (var=0.0931) adversary_sim=0.5403 (var=0.0407) welch_t=+0.722 p=0.4746 (N=25, alpha=0.01) -> reject

[4] rationale: The metric computes the difference in the probability of choosing Option A between Trial 1 (where A has 1 cue and B has 1 cue) and Trial 4 (where both A and B have 4 cues, adding 3 shared cues). The previous failed metrics used incorrect binary masks that did not match the dataset, leading to null results. With the correct masks, the Advocated theory (Cue Count Dilution) predicts a massive drop in preference for A (positive metric value) because the evidence difference is divided by 4^gamma, which can be up to 64. This causes the choice probability to crash to 50%. The Competing theory (Evidence Averaging) divides the difference by (4 + prior_count). Since prior_count is bounded, the difference shrinks by at most a factor of 4, meaning a significant preference for A should remain in Trial 4, leading to a much smaller drop.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10000') & (b_str == '01000')
    t4_mask = (a_str == '10111') & (b_str == '01111')
    
    p_a_t1 = 1.0 - data[t1_mask]['response'].mean() if t1_mask.sum() > 0 else 0.5
    p_a_t4 = 1.0 - data[t4_mask]['response'].mean() if t4_mask.sum() > 0 else 0.5
    
    return float(p_a_t1 - p_a_t4)
outcome: self_sim=0.0658 (var=0.0282) adversary_sim=0.0125 (var=0.0197) welch_t=+1.219 p=0.2291 (N=25, alpha=0.01) -> reject

[5] rationale: Instead of raw probability differences (which suffer from saturation effects if beta is high), we compute the empirical log-odds of choosing Option A in Trial 1 versus Trial 4. The log-odds directly approximate the underlying evidence difference scaled by beta. For the Advocated theory, the evidence difference is divided by 4^gamma (which can be up to 64), causing a massive drop in the log-odds toward zero. For the Competing theory, the evidence difference is divided by (4 + prior_count), which bounds the shrinkage factor to at most 4. Thus, the drop in log-odds (L1 - L4) will be substantially larger under the Advocated theory than the Competing theory, providing a much stronger contrast than simple probability differences.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10000') & (b_str == '01000')
    t4_mask = (a_str == '10111') & (b_str == '01111')
    
    c1 = (data[t1_mask]['response'] == 0).sum()
    n1 = t1_mask.sum()
    
    c4 = (data[t4_mask]['response'] == 0).sum()
    n4 = t4_mask.sum()
    
    if n1 == 0 or n4 == 0:
        return 0.0
        
    p1 = (c1 + 0.5) / (n1 + 1.0)
    p4 = (c4 + 0.5) / (n4 + 1.0)
    
    l1 = np.log(p1 / (1.0 - p1))
    l4 = np.log(p4 / (1.0 - p4))
    
    return float(l1 - l4)
outcome: self_sim=0.2697 (var=0.4853) adversary_sim=0.0511 (var=0.3955) welch_t=+1.164 p=0.2501 (N=25, alpha=0.01) -> reject

[6] rationale: The Advocated theory (Cue Count Dilution) predicts that the subjective difference between options is severely compressed due to the concave validity transformation (lambda <= 1.0) and exponential dilution (n^gamma). This suppresses the evidence difference, leading to choice probabilities that hover near 0.5 across all trials. In contrast, the Competing theory (Evidence Averaging) allows for convex transformations (lambda up to 10.0) and only dilutes linearly, predicting much stronger, confident preferences for the superior Option A. Therefore, the overall probability of choosing Option A across all trials will be substantially higher under the Competing theory. By averaging over all 96 trials, we maximize the signal-to-noise ratio and ensure minimal per-subject variance, creating a highly reliable discriminator.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Option A is strictly superior to Option B in all 4 trial types.
    # Response is 0 if A is chosen, 1 if B is chosen.
    # We calculate the overall proportion of times the superior Option A is chosen.
    return float(1.0 - data['response'].mean())
outcome: self_sim=0.5571 (var=0.0037) adversary_sim=0.5860 (var=0.0090) welch_t=-1.284 p=0.2065 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    n_cues = data['option_a_ratings'].apply(sum)\n    high_n_prob_b = data[n_cues >= 3]['response'].mean()\n    low_n_prob_b = data[n_cues <= 2]['response'].mean()\n    return float(high_n_prob_b - low_n_prob_b)",
  "rationale": "To maximize signal-to-noise ratio and minimize per-subject variance, we split the trials into two halves: 'low cue count' (Trials 1 & 2) and 'high cue count' (Trials 3 & 4). Since Option A always features the highest validity cue (0.95) versus Option B's second highest (0.85), subjects generally prefer Option A, meaning the probability of choosing B is below 0.5. The Advocated theory (Cue Count Dilution) predicts that as the number of shared cues increases, the perceived evidence difference shrinks exponentially, driving the choice probability toward 50%. Thus, the probability of choosing B should noticeably increase (move closer to 0.5) in the high cue count condition. The Competing theory (Evidence Averaging) divides by a linear denominator, preserving the evidence difference much more strongly, predicting a relatively constant preference for Option A across all trials. By comparing the mean response (which encodes the probability of choosing B) between the high and low cue count halves, we capture the exponential dilution effect while utilizing every trial to keep the variance extremely tight."
}
```

## Usage

```json
{
  "prompt_token_count": 6633,
  "candidates_token_count": 350,
  "total_token_count": 9964
}
```
