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
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 11: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive rule (WADD) from Probabilistic Cue Selection (Random Cue), we exploit the functional form of their choice probabilities. Random Cue's probability of choosing an option is strictly a linear function of the difference in weighted feature sums (because P(A) = 0.5 + 0.5 * ScoreDiff / TotalValidity). In contrast, WADD passes this same score difference through a softmax function, predicting a sigmoidal (S-shaped) relationship. By presenting a series of trials that systematically vary the score difference from zero to large values, we can evaluate whether the choice proportions follow a linear mixture (Random Cue) or a deterministic integration with softmax noise (WADD).

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
**Description:** Probabilistic Cue Selection (Random Cue) posits that decision-makers do not deterministically use the most valid cue (like Take-The-Best) nor do they integrate all cues simultaneously (like WADD). Instead, on each trial, they sample a single cue with a probability proportional to its subjective validity. They then choose the option favored by that sampled cue, guessing uniformly if the sampled cue ties. This single-cue sampling process naturally generates probabilistic choices across trials, producing choice shares near 0.50 for conflict trials where different cues favor different options, without relying on extreme softmax noise.

**Parameters:**
- epsilon: [0.0, 0.5]
- weights: [(0.0, 1.0)] * n_features
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Subjective validities used as sampling weights
    subj_weights = val * w
    sum_w = np.sum(subj_weights)
    
    if sum_w <= 1e-9:
        p_core = np.array([0.5, 0.5])
    else:
        p_cue = subj_weights / sum_w
        a, b = stim[0], stim[1]
        
        p_a = 0.0
        for j in range(n_features):
            if a[j] > b[j]:
                p_a += p_cue[j]
            elif a[j] == b[j]:
                p_a += p_cue[j] * 0.5
                
        p_core = np.array([p_a, 1.0 - p_a])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: This metric calculates the variance of the choice proportions for Option A across all 12 unique trial pairs. The Weighted Additive (WADD) rule relies on a softmax function over score differences, which pushes choice probabilities toward 0 or 1 (a sigmoidal function), resulting in a high variance of choice proportions across different trial types. In contrast, the Probabilistic Cue Selection (Random Cue) model produces choice probabilities that are a linear function of the subjective validities of the cues, compressing the probabilities closer to 0.5 and resulting in a significantly lower variance across trial types.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Create a hashable trial identifier based on the feature vectors of options A and B
    trial_keys = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + '_' + \
                 data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the proportion of times Option A was chosen (where response == 0)
    # for each unique trial type.
    p_a = (1.0 - data['response']).groupby(trial_keys).mean()
    
    # Return the variance of these choice proportions across the 12 trial types.
    return float(p_a.var())

outcome: self_sim=0.0164 (var=0.0033) adversary_sim=0.0033 (var=0.0003) welch_t=+1.104 p=0.2789 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in 'objective accuracy' between easy trials (large score difference) and hard trials (small score difference). Probabilistic Cue Selection (Random Cue) predicts that choice probability is a strictly linear mixture of the cues, resulting in a shallow, linear increase in accuracy as the score difference grows. In contrast, the Weighted Additive (WADD) rule passes the score difference through a softmax function, producing a sigmoidal curve where accuracy steeply increases from hard to easy trials. Measuring this difference captures the steepness of the psychometric function, providing a strong contrast between the linear and sigmoidal predictions while subtracting out baseline subject noise to keep between-subject variance low.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Objective validities used in the experiment
    v = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    # Reconstruct the feature matrices
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Calculate objective scores and differences
    score_a = a_ratings @ v
    score_b = b_ratings @ v
    diff = score_a - score_b
    abs_diff = np.abs(diff)
    
    # Determine if the subject chose the option with the higher objective score
    is_correct = ((diff > 0) & (data['response'] == 0)) | ((diff < 0) & (data['response'] == 1))
    
    # Define "easy" trials (large score difference) and "hard" trials (small score difference)
    easy_mask = abs_diff >= 0.55
    hard_mask = (abs_diff > 0.05) & (abs_diff <= 0.35)
    
    if not np.any(easy_mask) or not np.any(hard_mask):
        return 0.0
        
    # Calculate accuracy on both subsets
    acc_easy = is_correct[easy_mask].mean()
    acc_hard = is_correct[hard_mask].mean()
    
    # Return the difference in accuracy
    return float(acc_easy - acc_hard)
outcome: self_sim=0.1738 (var=0.0279) adversary_sim=0.0790 (var=0.0120) welch_t=+2.373 p=0.02237 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Objective validities used in the experiment\n    v = np.array([0.9, 0.8, 0.7, 0.6, 0.5])\n    \n    # Reconstruct the feature matrices\n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    # Calculate objective scores and differences\n    score_a = a_ratings @ v\n    score_b = b_ratings @ v\n    diff = score_a - score_b\n    \n    # Select \"easy\" trials with a large score difference (abs(diff) >= 0.65)\n    # In this design, this cleanly isolates Trials 7, 9, and 10.\n    mask = np.abs(diff) >= 0.65\n    \n    if not np.any(mask):\n        return 0.5\n        \n    # Determine if the subject chose the option with the higher objective score\n    is_correct = ((diff > 0) & (data['response'] == 0)) | ((diff < 0) & (data['response'] == 1))\n    \n    # Return the simple accuracy on these easy trials\n    return float(is_correct[mask].mean())",
  "rationale": "Rather than subtracting hard trials (which inflates the between-subject variance of the metric due to the sum of variances), we isolate the absolute accuracy strictly on the 'easiest' trials (Trials 7, 9, and 10) where the objective score difference is large (>= 0.65). Random Cue strictly limits choice probabilities to a linear mixture of the cues, meaning that even on these 'easy' trials, the accuracy remains relatively low (around ~63%). In contrast, WADD uses a softmax function over the score differences, heavily amplifying the choice probability for the better option on these specific trials (predicting ~90% accuracy). This produces a massive gap between the theories' mean predictions while keeping the between-subject variance tightly bounded."
}
```

## Usage

```json
{
  "prompt_token_count": 4655,
  "candidates_token_count": 515,
  "total_token_count": 9328
}
```
