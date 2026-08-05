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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 8: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 9: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 10: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 11: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 12: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** This design exploits the fundamental difference between linear integration (WADD) and softmax-based probabilistic sampling (PCS). We use five cues with linearly spaced validities: [0.9, 0.8, 0.7, 0.6, 0.5]. We construct pairs where the difference in the sum of validities is identical (e.g., +0.2, -0.2, or exactly 0). Because WADD is a linear model, it predicts identical choice probabilities across trials with the same validity difference. In contrast, PCS uses a softmax function over validities to determine sampling probabilities. Due to the strict convexity of the exponential function, the difference in sampling probabilities for two cues with a fixed validity difference (e.g., 0.9 vs 0.7 compared to 0.7 vs 0.5) is much larger at the high end of the validity scale. Moreover, for trials where the sum of validities is perfectly tied (e.g., 0.9 + 0.5 vs 0.8 + 0.6), WADD predicts exactly 50/50 guessing, whereas the convexity of the PCS softmax assigns a higher sampling probability to the extreme cues, leading PCS to systematically prefer the option with the single highest validity cue. This provides a crisp, parameter-free quantitative dissociation between the two theories.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
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
**Description:** Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

**Parameters:**
- gamma: [0.0, 2.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: This metric calculates the proportion of times Option A is chosen on trials where the sum of objective validities is perfectly tied between Option A and Option B (trials 7, 8, and 9). In these trials, Option A always possesses the single highest validity cue. Under WADD, the linear integration of features (even with random subjective weights drawn uniformly) yields a symmetric preference, resulting in a choice proportion for A around 0.5. In contrast, PCS uses a softmax over validities to sample a single feature. The convexity of the softmax assigns disproportionately high sampling probabilities to the most valid cues, leading PCS to systematically prefer Option A over Option B.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    A_mat = np.array(data['option_a_ratings'].tolist())
    B_mat = np.array(data['option_b_ratings'].tolist())
    
    score_A = A_mat @ validities
    score_B = B_mat @ validities
    
    tied_trials = np.abs(score_A - score_B) < 1e-5
    
    if not np.any(tied_trials):
        return 0.5
        
    return float(np.mean(data.loc[tied_trials, 'response'] == 0))
outcome: self_sim=0.4908 (var=0.0337) adversary_sim=0.5275 (var=0.0049) welch_t=-0.933 p=0.358 (N=25, alpha=0.01) -> reject

[1] rationale: This metric computes the difference in choice probability for the better option between two types of trials: 0.9 vs 0.7 validity cues, and 0.7 vs 0.5 validity cues. In PCS, because sampling probabilities are derived from a softmax over validities, the convexity of the exponential function ensures that the difference in sampling probability is larger for higher validities (e^{0.9g} - e^{0.7g} > e^{0.7g} - e^{0.5g}), resulting in a higher accuracy for 0.9 vs 0.7 compared to 0.7 vs 0.5. In contrast, WADD draws subjective weights uniformly from (0, 1), so the probability that w1*0.9 > w2*0.7 is 1 - 0.7/1.8 ≈ 0.611, while the probability that w1*0.7 > w2*0.5 is 1 - 0.5/1.4 ≈ 0.642. Thus, WADD predicts a negative difference, whereas PCS predicts a positive difference. This opposing ordinal prediction provides a robust, parameter-free dissociation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_mat = np.array(data['option_a_ratings'].tolist())
    B_mat = np.array(data['option_b_ratings'].tolist())
    
    # Identify trials comparing 0.9 vs 0.7
    t1 = (A_mat[:, 0] == 1) & (B_mat[:, 2] == 1) & (A_mat.sum(axis=1) == 1) & (B_mat.sum(axis=1) == 1)
    t4 = (A_mat[:, 2] == 1) & (B_mat[:, 0] == 1) & (A_mat.sum(axis=1) == 1) & (B_mat.sum(axis=1) == 1)
    
    # Identify trials comparing 0.7 vs 0.5
    t3 = (A_mat[:, 2] == 1) & (B_mat[:, 4] == 1) & (A_mat.sum(axis=1) == 1) & (B_mat.sum(axis=1) == 1)
    t6 = (A_mat[:, 4] == 1) & (B_mat[:, 2] == 1) & (A_mat.sum(axis=1) == 1) & (B_mat.sum(axis=1) == 1)
    
    responses = data['response'].values
    
    # Proportion choosing 0.9 over 0.7
    choose_09 = np.sum((t1) & (responses == 0)) + np.sum((t4) & (responses == 1))
    n_09 = np.sum(t1) + np.sum(t4)
    p_09 = choose_09 / n_09 if n_09 > 0 else 0.5
    
    # Proportion choosing 0.7 over 0.5
    choose_07 = np.sum((t3) & (responses == 0)) + np.sum((t6) & (responses == 1))
    n_07 = np.sum(t3) + np.sum(t6)
    p_07 = choose_07 / n_07 if n_07 > 0 else 0.5
    
    return float(p_09 - p_07)
outcome: self_sim=-0.0638 (var=0.2399) adversary_sim=-0.0200 (var=0.0260) welch_t=-0.424 p=0.6745 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    df = data.copy()\n    # Create hashable trial identifiers safely using list comprehensions\n    df['trial_id'] = df['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x])) + \\\n                     \"_\" + \\\n                     df['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    \n    subj_vars = []\n    for subj, subj_df in df.groupby('subject_id'):\n        # Proportion of choosing A (response == 0) for each trial type\n        p_A = subj_df.groupby('trial_id')['response'].apply(lambda x: np.mean(x == 0))\n        # Variance of these proportions across the 12 trial types\n        subj_vars.append(p_A.var(ddof=1))\n        \n    return float(np.mean(subj_vars))",
  "rationale": "PCS uses a softmax over validities to determine the probability of sampling each feature. Because the validities in this design are closely spaced [0.9, 0.8, 0.7, 0.6, 0.5] and the sensitivity parameter gamma is capped at 2.0, the sampling probabilities remain relatively flat. As a result, PCS predicts that choice probabilities will be heavily compressed toward 0.5 across all trial types, making choices highly stochastic. In contrast, WADD computes a weighted sum of features. With the inverse temperature beta drawn from [0.1, 20.0], WADD frequently produces large subjective score differences, leading to highly deterministic choice probabilities (near 0 or 1) for many trial types. We can discriminate the theories by computing the variance of each subject's empirical choice proportions across the 12 trial types. PCS predicts a low variance (dominated by binomial noise around p=0.5), while WADD predicts a significantly higher variance due to its capacity for deterministic responding."
}
```

## Usage

```json
{
  "prompt_token_count": 5190,
  "candidates_token_count": 488,
  "total_token_count": 36812
}
```
