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
**Validities (n_features=6):** [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 5: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 8: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 10: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 11: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 12: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]

**Rationale:** The Advocated theory (WADD) and Competing theory (Mixture of Subjective WADD and Tallying) structurally overlap, but they are subject to different parameter bounds that enforce distinct behavioral predictions. Crucially, the Advocated model allows for much higher inverse temperature (beta up to 20) compared to the Competing model (beta up to 5). By employing a fine-grained hierarchy of 6 features with linearly decreasing validities, we can test choices between adjacent features (e.g., Feature 1 vs. 2, Feature 2 vs. 3, etc.). The Advocated model can predict consistently strong, deterministic preferences (>85%) across all adjacent feature comparisons because its high beta can magnify small differences in objective validities. To mimic this consistent determinism, the Competing model would need its subjective weights to drop dramatically at each step in the hierarchy. However, because its beta is capped at 5, maintaining an 85% choice probability requires a subjective weight difference of at least 0.4 between adjacent features. Across 6 features, this would require a total weight drop of 2.0, which is impossible since subjective weights are bounded in [0, 1]. Thus, the Competing model is forced to predict highly noisy choices for fine-grained trade-offs, while the Advocated model uniquely predicts sharp, consistent preferences.

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
**Description:** Mixture of Subjective WADD and Tallying: Decision-makers probabilistically mix between a simple equal-weight heuristic (Tallying) and a weighted additive rule that uses subjective, free-varying feature weights rather than objective validities. The subjective weights allow the WADD component to capture non-compensatory, Take-The-Best-like behavior (by assigning heavily skewed weights to features), while the Tallying component accounts for the strong equal-weighting pull observed when individuals fall back on simply counting positive features. Choice probabilities are a mixture of the softmax probabilities derived from each strategy, further blended with a uniform lapse rate to account for response errors. Increased choice noise bounds allow the model to better match human sub-optimal choice frequencies.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 1.0]
- w_tally: [0.0, 1.0]
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Subjective WADD ---
    # Uses free subjective weights instead of objective validities
    scores_wadd = stim @ w
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    n_opts = p_core.shape[0]
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
[0] rationale: This metric calculates the proportion of times a subject chooses the option with the objectively higher-validity feature on trials where both options have exactly one positive feature, and those features are adjacent in the validity hierarchy (Trials 1-5). The Advocated model (WADD) can predict highly deterministic choices for the better option across all these pairs because its large maximum beta (20) can amplify the small objective validity differences. The Competing model, however, is constrained by a smaller beta (5) and subjective weights bounded in [0, 1]. To maintain a high probability of choosing the better option across all 5 adjacent pairs, the Competing model's subjective weights would need to drop by an impossible total amount (> 1.0). Therefore, the Competing model is forced to predict much noisier choices on these adjacent single-feature tradeoffs, resulting in a systematically lower metric value compared to the Advocated model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def score_trial(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        if sum(a) == 1 and sum(b) == 1:
            idx_a = a.index(1)
            idx_b = b.index(1)
            if abs(idx_a - idx_b) == 1:
                chose_better = (idx_a < idx_b and row['response'] == 0) or (idx_b < idx_a and row['response'] == 1)
                return 1.0 if chose_better else 0.0
        return np.nan
        
    scores = data.apply(score_trial, axis=1)
    return float(scores.dropna().mean())
outcome: self_sim=0.5210 (var=0.0075) adversary_sim=0.4955 (var=0.0052) welch_t=+1.132 p=0.2633 (N=25, alpha=0.01) -> reject

[1] rationale: In the chosen experimental design, there are several trials where Option A and Option B have an equal number of positive features (Trials 1-5, 11, 12). In every one of these trials, Option A is constructed to possess the features with the higher objective validities. Under the Competing theory, features are a priori exchangeable (subjective weights are drawn randomly without regard to objective validities) and the Tallying component is perfectly tied, meaning the expected probability of choosing Option A is exactly 0.5. Under the Advocated theory, the objective validities systematically scale the subjective weights, biasing the scores in favor of Option A, resulting in an expected probability > 0.5. This metric isolated this structural divergence by measuring the choice proportion for Option A on all equal-sum trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_equal_sum(row):
        return sum(row['option_a_ratings']) == sum(row['option_b_ratings'])
        
    equal_sum_trials = data[data.apply(is_equal_sum, axis=1)]
    
    if len(equal_sum_trials) == 0:
        return np.nan
        
    # In all equal-sum trials in this design, Option A contains the features 
    # with higher objective validities. 
    # We calculate the proportion of times Option A was chosen.
    return float((equal_sum_trials['response'] == 0).mean())
outcome: self_sim=0.5246 (var=0.0091) adversary_sim=0.4904 (var=0.0044) welch_t=+1.481 p=0.1461 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    # Identify trials where both options have the same number of positive features\n    a_sums = data['option_a_ratings'].apply(sum)\n    b_sums = data['option_b_ratings'].apply(sum)\n    equal_sum_mask = a_sums == b_sums\n    \n    eq_data = data[equal_sum_mask].copy()\n    if eq_data.empty:\n        return np.nan\n        \n    # Create a hashable key for the trial pair\n    eq_data['pair_key'] = eq_data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    \n    # Calculate the mean response for each subject on each unique equal-sum pair\n    subject_pair_means = eq_data.groupby(['subject_id', 'pair_key'])['response'].mean()\n    \n    # Calculate determinism as the absolute deviation from random guessing (0.5)\n    determinism = (subject_pair_means - 0.5).abs().mean()\n    \n    return float(determinism)",
  "rationale": "This metric calculates the within-subject determinism of choices specifically on equal-sum trials (where Option A and Option B have the same total number of positive features). Under the Competing theory, the Tallying component perfectly ties on these trials, outputting a 50/50 choice probability. Because the model mixes this 50/50 Tallying probability with the Subjective WADD probability, its final choice probabilities are structurally pulled toward 0.5, resulting in noisier, less deterministic behavior. Under the Advocated theory (WADD), there is no Tallying component to pull probabilities to 0.5; the model relies entirely on weighted sums, which, combined with a high allowed inverse temperature (beta up to 20), allows it to make highly deterministic choices. Therefore, the Advocated theory predicts significantly higher absolute deviations from 0.5 (greater determinism) on equal-sum trials compared to the Competing theory."
}
```

## Usage

```json
{
  "prompt_token_count": 4853,
  "candidates_token_count": 506,
  "total_token_count": 9741
}
```
