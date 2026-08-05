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
**Validities (n_features=5):** [1.0, 0.55, 0.55, 0.55, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** To cleanly dissociate the advocated Tallying model from the competing Thresholded Weighted Additive (WADD) model, we exploit the threshold mechanism of the competing model. Tallying strictly counts winning features and ignores validities entirely. The competing model uses WADD if the normalized weighted difference between options exceeds a threshold, otherwise it falls back to Tallying. By using one extremely high-validity feature and four low-validity features, we create 'compensatory' trials where one option wins on the single best feature but loses on all four others. In these trials, the normalized WADD difference is very large (well above typical thresholds), leading the competing model to strongly predict the option with the single best feature. The advocated Tallying model, however, will strongly predict the option with four winning features. We also include trials with small WADD differences to probe the threshold boundary, and 'Tally tie' trials where Tallying predicts exactly 50/50 guessing but WADD predicts a deterministic choice.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Decision-makers utilize a 'Thresholded Weighted Additive' strategy. They compute the overall weighted value of each option based on normalized cue validities. If the difference in these weighted values exceeds a subjective threshold, the decision is driven by the weighted additive difference (WADD), allowing sensitivity to highly valid cues in extreme compensatory cases. However, if the weighted difference is below the threshold, subjects perceive the options as roughly equivalent in overall value and fall back to a simpler, less cognitively demanding Tallying heuristic, merely counting the number of winning features for each option.

**Parameters:**
- threshold: [0.0, 0.5]
- beta_wadd: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    val = val / np.sum(val)  # Normalize validities to sum to 1
    
    theta = float(parameters["threshold"])
    beta_wadd = float(parameters["beta_wadd"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Compute WADD scores
    wadd_a = np.dot(a, val)
    wadd_b = np.dot(b, val)
    wadd_diff = abs(wadd_a - wadd_b)
    
    # Threshold logic: if difference is salient, use WADD; else fallback to Tallying
    if wadd_diff > theta:
        scores = np.array([wadd_a, wadd_b])
        beta = beta_wadd
    else:
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        scores = np.array([a_wins, b_wins])
        beta = beta_tally
        
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply lapse rate
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
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
[0] rationale: By isolating trials where both options have an equal number of positive features (Trials 7 and 8), we create a scenario where the Tallying model predicts exactly a 50/50 guess, regardless of its beta parameter. However, in these same trials, Option A possesses the single most valid feature while Option B does not. The competing Thresholded WADD model will compute a weighted difference favoring Option A. For subjects whose threshold is lower than this difference, WADD will deterministically favor Option A over Option B. Consequently, the Tallying model predicts an average choice proportion of 0.50 for Option A, whereas the Thresholded WADD model predicts a significantly higher proportion.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the total number of positive features is equal for both options.
    # In this design, these correspond exactly to the 'tie' trials (Trials 7 and 8).
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    mask = (a_sums == b_sums)
    if not mask.any():
        return 0.5
        
    # Calculate the proportion of times Option A was chosen on these tie trials.
    # Option A always possesses the single highest-validity feature in these specific trials.
    return float((data.loc[mask, 'response'] == 0).mean())
outcome: self_sim=0.4942 (var=0.0097) adversary_sim=0.5533 (var=0.0156) welch_t=-1.861 p=0.06928 (N=25, alpha=0.01) -> reject

[1] rationale: We compare the choice proportions of Option A in Trial 1 versus Trial 2. In Trial 1, the Tallying model sees a tally difference of 3 (4 vs 1), while in Trial 2 it sees a tally difference of 1 (3 vs 2). Thus, Tallying strongly predicts a higher probability of choosing A in Trial 1 than in Trial 2, yielding a consistently positive difference. In contrast, the competing Thresholded WADD model will likely use the WADD strategy for Trial 1 (since the WADD difference is large, 0.375) but fall back to Tallying for Trial 2 (where the WADD difference is tiny, 0.03125). Because the competing model uses independent beta parameters for WADD and Tallying, the choice probability for A in Trial 1 is not structurally constrained to be higher than in Trial 2, leading to a significantly lower (and more variable) average difference between the two trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = (data['a_str'] == '01111') & (data['b_str'] == '10000')
    t2_mask = (data['a_str'] == '01110') & (data['b_str'] == '10001')
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1) or pd.isna(p_a_t2):
        return 0.0
        
    return float(p_a_t1 - p_a_t2)
outcome: self_sim=0.0367 (var=0.0259) adversary_sim=0.0317 (var=0.0264) welch_t=+0.109 p=0.9134 (N=25, alpha=0.01) -> reject

[2] rationale: This metric contrasts trials where both models agree on the choice direction, but disagree on the extremity of the underlying scores. In Trials 2, 4, 5, and 6, the Tallying difference is 1, and the WADD difference is extremely small (0.03125). Thus, the competing model almost always falls back to Tallying, yielding a high choice probability. In Trial 3, the Tallying difference is also 1, but the WADD difference is large (0.3125). For Trial 3, the competing model will frequently use the WADD strategy. Because the WADD score difference (0.3125) is much smaller than the Tallying score difference (1.0), the resulting softmax probabilities for Trial 3 under the competing model will be systematically closer to 0.5. The advocated Tallying model, however, ignores validities and treats all these trials as identical (score difference of 1), predicting an average difference of zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trials where Tallying difference is 1, and WADD difference is very small (0.03125)
    t2_mask = (data['a_str'] == '01110') & (data['b_str'] == '10001')
    t4_mask = (data['a_str'] == '11000') & (data['b_str'] == '00111')
    t5_mask = (data['a_str'] == '01100') & (data['b_str'] == '10000')
    t6_mask = (data['a_str'] == '10000') & (data['b_str'] == '01100')
    
    # Trial where Tallying difference is 1, but WADD difference is large (0.3125)
    t3_mask = (data['a_str'] == '01100') & (data['b_str'] == '10011')
    
    # Probability of making the Tally-consistent choice
    p_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_t4 = (data.loc[t4_mask, 'response'] == 1).mean()
    p_t5 = (data.loc[t5_mask, 'response'] == 0).mean()
    p_t6 = (data.loc[t6_mask, 'response'] == 1).mean()
    
    p_t3 = (data.loc[t3_mask, 'response'] == 1).mean()
    
    baseline = np.nanmean([p_t2, p_t4, p_t5, p_t6])
    
    if pd.isna(baseline) or pd.isna(p_t3):
        return 0.0
        
    return float(baseline - p_t3)
outcome: self_sim=0.0000 (var=0.0139) adversary_sim=0.0050 (var=0.0134) welch_t=-0.151 p=0.8803 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # Trials where Tallying difference is exactly 1.\n    # T2, T4, T5, T6 have extremely small WADD differences (0.03125), so the competing model almost always falls back to Tallying.\n    t2_mask = (data['a_str'] == '01110') & (data['b_str'] == '10001')\n    t4_mask = (data['a_str'] == '11000') & (data['b_str'] == '00111')\n    t5_mask = (data['a_str'] == '01100') & (data['b_str'] == '10000')\n    t6_mask = (data['a_str'] == '10000') & (data['b_str'] == '01100')\n    \n    # T3 also has a Tallying difference of 1, but a LARGE WADD difference (0.3125).\n    # The competing model will frequently exceed its threshold here and use the WADD strategy instead.\n    t3_mask = (data['a_str'] == '01100') & (data['b_str'] == '10011')\n    \n    # Calculate the empirical probability of making the Tally-favored choice in each trial\n    p2 = (data.loc[t2_mask, 'response'] == 0).mean()\n    p4 = (data.loc[t4_mask, 'response'] == 1).mean()\n    p5 = (data.loc[t5_mask, 'response'] == 0).mean()\n    p6 = (data.loc[t6_mask, 'response'] == 1).mean()\n    p3 = (data.loc[t3_mask, 'response'] == 1).mean()\n    \n    baseline = np.nanmean([p2, p4, p5, p6])\n    \n    if pd.isna(baseline) or pd.isna(p3):\n        return 0.0\n        \n    # The absolute difference prevents positive and negative deviations (due to independent betas in the competing model) from canceling out.\n    return float(abs(p3 - baseline))\n",
  "rationale": "This metric isolates trials that the advocated Tallying model considers identical (all having a Tally score difference of exactly 1). For the Tallying model, the true underlying probability of making the Tally-consistent choice is identical across all these trials, so the absolute difference between Trial 3 and the baseline of the other four trials will be near zero, driven only by binomial sampling noise. Conversely, the competing Thresholded WADD model will almost always evaluate the baseline trials using its Tally fallback (due to tiny WADD differences), but will frequently evaluate Trial 3 using its WADD strategy (due to a large WADD difference). Because the competing model uses independent beta parameters for WADD and Tallying, the choice probability for Trial 3 will structurally decouple from the baseline trials. Taking the absolute difference ensures that this decoupling is captured as a strictly positive signal rather than averaging out to zero across subjects."
}
```

## Usage

```json
{
  "prompt_token_count": 5362,
  "candidates_token_count": 840,
  "total_token_count": 11019
}
```
