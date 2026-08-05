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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** This design dissociates the Weighted Additive (WADD) theory from Tallying by manipulating the distribution of cue validities across the options. WADD integrates all features by weighting them according to a subjective transformation of their validities, whereas Tallying ignores validities entirely and simply counts the number of features favoring each option. The trial set includes pairs where one option is supported by a few high-validity cues while the other is supported by a larger number of low-validity cues, leading to opposing predictions. Additionally, several trials present an equal number of winning features for both options (a tie for Tallying), where WADD strongly prefers the option with higher-validity features.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



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
[0] rationale: This metric isolates trials where Tallying considers the two options to be perfectly tied (i.e., an equal number of features favor Option A and Option B). On these specific trials, Tallying predicts that the subject will guess uniformly at random (50% probability for either option). However, WADD integrates the unequal feature validities and maintains a strict preference for one of the options. By calculating the proportion of choices that align with WADD's preference strictly on these Tallying-tie trials, we expect a value of exactly 0.5 under the Tallying data generation process, and a value significantly greater than 0.5 under the WADD data generation process.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Calculate Tallying wins for each option
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    # Identify trials where Tallying predicts a tie
    tie_mask = (a_wins == b_wins)
    
    if not np.any(tie_mask):
        return 0.5
        
    # Calculate WADD scores
    a_wadd = np.dot(a_mat, val)
    b_wadd = np.dot(b_mat, val)
    
    # WADD predicted response (0 for A, 1 for B)
    wadd_pred = (b_wadd > a_wadd).astype(int)
    
    responses = data['response'].values
    
    # Calculate the proportion of choices that match the WADD prediction on Tallying-tie trials
    matches = (responses[tie_mask] == wadd_pred[tie_mask])
    
    return float(np.mean(matches))
outcome: self_sim=0.5428 (var=0.0078) adversary_sim=0.5050 (var=0.0063) welch_t=+1.592 p=0.1179 (N=25, alpha=0.01) -> reject

[1] rationale: By isolating trials where Tallying computes exactly a 1-point win difference between the options, we can compare choice probabilities across items that Tallying treats as identically difficult. Within this set, there are trials where WADD sees a very small value difference (0.15) and trials where WADD sees a much larger value difference (0.55 to 0.75). Under Tallying, the rate of choosing the favored option should be identical across these two sets, yielding a difference of 0. Under WADD, subjects should be significantly more accurate on the trials with the larger WADD difference. This produces a robust positive contrast for WADD and an expectation of exactly 0 for Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Calculate Tallying wins for each option
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_diff = np.abs(a_wins - b_wins)
    # 0 if A is favored, 1 if B is favored
    tally_favored = (b_wins > a_wins).astype(int)
    
    # Calculate raw WADD scores to identify trial types
    wadd_a = np.dot(a_mat, val)
    wadd_b = np.dot(b_mat, val)
    wadd_diff = np.abs(wadd_a - wadd_b)
    
    # Isolate trials where Tallying sees exactly a 1-point difference
    mask_tally_1 = (tally_diff == 1)
    
    # Split these into trials where WADD sees a very small vs. large difference
    mask_low = mask_tally_1 & (wadd_diff < 0.3)
    mask_high = mask_tally_1 & (wadd_diff > 0.5)
    
    responses = data['response'].values
    
    subjects = data['subject_id'].unique()
    diffs = []
    for subj in subjects:
        subj_mask = (data['subject_id'] == subj).values
        
        idx_low = subj_mask & mask_low
        idx_high = subj_mask & mask_high
        
        if np.sum(idx_low) == 0 or np.sum(idx_high) == 0:
            continue
            
        # Accuracy defined as choosing the option favored by both Tallying and WADD
        acc_low = np.mean(responses[idx_low] == tally_favored[idx_low])
        acc_high = np.mean(responses[idx_high] == tally_favored[idx_high])
        
        diffs.append(acc_high - acc_low)
        
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.1300 (var=0.0428) adversary_sim=0.0056 (var=0.0151) welch_t=+2.586 p=0.01355 (N=25, alpha=0.01) -> reject

[2] rationale: To overcome the high variance of isolating small subsets of trials, this metric leverages the full dataset of 96 trials per subject. It computes the point-biserial correlation between the subject's binary choices and the continuous score difference predicted by each theory (WADD vs. Tallying). Because the theories make subtly different predictions about the relative difficulty/strength of preference across the 8 trial types, WADD agents will show a higher correlation with the WADD difference than with the Tallying difference, yielding a robust positive value. Conversely, Tallying agents will correlate more strongly with the Tallying difference, yielding a robust negative value. Aggregating across all trials minimizes per-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # WADD score difference (B - A)
    a_wadd = np.dot(a_mat, val)
    b_wadd = np.dot(b_mat, val)
    wadd_diff = b_wadd - a_wadd
    
    # Tallying score difference (B wins - A wins)
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_diff = b_wins - a_wins
    
    resp_B = data['response'].values
    
    subjects = data['subject_id'].unique()
    diffs = []
    for subj in subjects:
        mask = (data['subject_id'] == subj).values
        y = resp_B[mask]
        x_wadd = wadd_diff[mask]
        x_tally = tally_diff[mask]
        
        # Skip if subject gave the same response on all trials
        if np.std(y) == 0 or np.std(x_wadd) == 0 or np.std(x_tally) == 0:
            continue
            
        r_wadd = np.corrcoef(x_wadd, y)[0, 1]
        r_tally = np.corrcoef(x_tally, y)[0, 1]
        
        diffs.append(r_wadd - r_tally)
        
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.0069 (var=0.0012) adversary_sim=-0.0093 (var=0.0007) welch_t=+1.834 p=0.07325 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Project the list of Option A ratings to a tuple so we can uniquely identify trial types\n    A_tuples = data['option_a_ratings'].apply(tuple)\n    \n    # The 8 unique trial types identified by their Option A configuration\n    T1 = (1, 1, 0, 0, 0)\n    T2 = (1, 0, 1, 0, 0)\n    T3 = (0, 1, 1, 1, 0)\n    T4 = (1, 0, 0, 0, 1)\n    T5 = (1, 0, 0, 0, 0)\n    T6 = (0, 1, 1, 0, 1)\n    T7 = (0, 0, 1, 1, 1)\n    T8 = (1, 0, 1, 0, 1)\n    \n    logits = {}\n    for T in [T1, T2, T3, T4, T5, T6, T7, T8]:\n        mask = (A_tuples == T)\n        if not mask.any():\n            logits[T] = 0.0\n            continue\n            \n        # Count how many times Option A was chosen (response == 0)\n        count_A = np.sum(data.loc[mask, 'response'] == 0)\n        N = mask.sum()\n        \n        # Compute the smoothed logit of P(A)\n        p_A_smoothed = (count_A + 0.5) / (N + 1.0)\n        logits[T] = np.log(p_A_smoothed / (1.0 - p_A_smoothed))\n        \n    # We construct a linear contrast of the logits across 7 trial types.\n    # Under Tallying, the expected logit is strictly proportional to the difference in Tally wins.\n    # The Tallying differences (A - B) for these trials are:\n    # T3 (+1), T6 (+1), T8 (+1), T1 (-1), T2 (0), T4 (0), T5 (0)\n    # The contrast weights perfectly cancel out the Tallying differences:\n    # (1 + 1 + 1) + 3*(-1) + 0 - 0 + 0 = 0.\n    # Therefore, Tallying predicts an expected contrast of exactly 0, regardless of beta or epsilon.\n    \n    # Under WADD, the differences (A - B) are:\n    # T3 (+0.75), T6 (+0.55), T8 (+0.75), T1 (-0.15), T2 (+0.20), T4 (-0.10), T5 (+0.10)\n    # Applying the same contrast weights yields a strongly positive expected value:\n    # (0.75 + 0.55 + 0.75) + 3*(-0.15) + 0.20 - (-0.10) + 0.10 = +2.00.\n    \n    C = logits[T3] + logits[T6] + logits[T8] + 3 * logits[T1] + logits[T2] - logits[T4] + logits[T5]\n    \n    return float(C)",
  "rationale": "By transforming the choice proportions into smoothed logits, we linearize the relationship with the underlying subjective value differences, making the metric robust to the inverse temperature parameter (beta). We then construct a specific linear contrast of these logits across 7 trial types. The weights of this contrast are mathematically designed to sum to exactly 0 when multiplied by the Tallying score differences, meaning Tallying predicts an expected value of exactly 0 for this metric, completely eliminating between-subject variance caused by varying beta or epsilon parameters. Conversely, when the same weights are applied to the validity-weighted WADD score differences, they yield a strongly positive sum (+2.00 * beta), creating a massive, reliable gap between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 5507,
  "candidates_token_count": 979,
  "total_token_count": 36417
}
```
