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
**Validities (n_features=8):** [0.95, 0.7, 0.65, 0.6, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Parallel Similarity and Feature-Difference Model) from the Competing Theory (Sequential Evidence Accumulation), we exploit the Advocated Theory's unique reliance on trial-by-trial Jaccard similarity to interpolate between Tallying and WADD. The Competing Theory evaluates options strictly based on sequential feature differences (Option A - Option B) in validity order. Consequently, if two trials have perfectly identical feature differences, the Competing Theory must predict the exact same choice probabilities, regardless of the options' shared baseline features. In contrast, the Advocated Theory computes the Jaccard similarity of the options' positive features. By constructing matched pairs of trials with identical feature differences but vastly different numbers of shared positive features (e.g., all remaining features are 0s vs. all remaining features are 1s), we create a sharp double dissociation: the Competing Theory predicts a flat, identical response profile across these matched pairs, whereas the Advocated Theory predicts significant shifts in choice probability as varying similarity triggers a trial-by-trial transition between compensatory (WADD) and equal-weighting (Tallying) processing.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



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
**Description:** Decision-makers use a dynamic sequential evidence accumulation process with a fixed absolute threshold. They evaluate features in decreasing order of validity, maintaining a running sum of the validity-weighted differences between options. After each feature, they compare the accumulated absolute evidence to a fixed threshold. If the evidence meets or exceeds this threshold, they stop and make a choice. This dynamically adapts the amount of information processed: highly conflicting early evidence keeps the accumulated difference near zero, prompting the integration of more features (similar to WADD or Tallying), whereas decisive early evidence leads to quick termination (similar to Take-The-Best).

**Parameters:**
- threshold: [0.0, 5.0]
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
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    e = 0.0
    for idx in order:
        e += val[idx] * (a[idx] - b[idx])
        
        # Dynamic stopping rule based on fixed absolute evidence threshold
        if abs(e) >= threshold and abs(e) > 0:
            break
            
    scores = np.array([e, -e])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
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
[0] rationale: This metric calculates the total absolute difference in choice probabilities between matched pairs of trials. Pairs (1, 2), (3, 4), and (5, 6) feature identical within-pair option differences but vary substantially in the number of shared positive features, which drives Jaccard similarity. The Competing Theory evaluates features sequentially by difference, hence it predicts identically zero true difference between these pairs. In contrast, the Advocated Theory uses Jaccard similarity to interpolate between choice strategies, predicting a large shift in choice probability across the pairs. Thus, the metric should be near zero (subject to sampling noise) for the Competing Theory and significantly greater than zero for the Advocated Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data = data.copy()
    data['trial_type'] = data['option_a_ratings'].apply(tuple)
    
    t1 = (1, 0, 0, 0, 0, 0, 0, 0)
    t2 = (1, 0, 0, 0, 1, 1, 1, 1)
    t3 = (0, 1, 0, 0, 0, 0, 0, 0)
    t4 = (0, 1, 0, 0, 1, 1, 1, 1)
    t5 = (1, 1, 0, 0, 0, 0, 0, 0)
    t6 = (1, 1, 0, 0, 1, 1, 1, 1)
    
    means = data.groupby('trial_type')['response'].mean()
    
    diff_12 = abs(means.get(t1, 0.5) - means.get(t2, 0.5))
    diff_34 = abs(means.get(t3, 0.5) - means.get(t4, 0.5))
    diff_56 = abs(means.get(t5, 0.5) - means.get(t6, 0.5))
    
    return float(diff_12 + diff_34 + diff_56)

outcome: self_sim=0.1075 (var=0.3123) adversary_sim=0.0250 (var=0.0220) welch_t=+0.713 p=0.4817 (N=25, alpha=0.01) -> reject

[1] rationale: To reduce the high variance associated with absolute differences, this metric calculates the product of the mean response shifts between two sets of matched pairs: (Pair 2 vs Pair 1) and (Pair 4 vs Pair 3). Under the Competing Theory, the true difference for both is exactly zero, so their product is zero (with very small binomial variance). Under the Advocated Theory, manipulating the similarity parameter shifts the weight between Tallying and WADD. Because Tallying favors Option B in both matched pairs, the shifts (m2 - m1) and (m4 - m3) will always share the same sign, regardless of whether the similarity slope is positive or negative. Thus, their product yields a stable, strictly positive value for the Advocated Theory, providing a sharp and low-variance discrimination between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data = data.copy()
    data['trial_type'] = data['option_a_ratings'].apply(tuple)
    
    t1 = (1, 0, 0, 0, 0, 0, 0, 0)
    t2 = (1, 0, 0, 0, 1, 1, 1, 1)
    t3 = (0, 1, 0, 0, 0, 0, 0, 0)
    t4 = (0, 1, 0, 0, 1, 1, 1, 1)
    
    means = data.groupby('trial_type')['response'].mean()
    
    m1 = means.get(t1, 0.5)
    m2 = means.get(t2, 0.5)
    m3 = means.get(t3, 0.5)
    m4 = means.get(t4, 0.5)
    
    d1 = m2 - m1
    d2 = m4 - m3
    
    return float(d1 * d2)
outcome: self_sim=-0.0000 (var=0.0266) adversary_sim=0.0000 (var=0.0001) welch_t=-0.001 p=0.9989 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    data = data.copy()\n    data['trial_type'] = data['option_a_ratings'].apply(tuple)\n    \n    t1 = (1, 0, 0, 0, 0, 0, 0, 0)\n    t2 = (1, 0, 0, 0, 1, 1, 1, 1)\n    t3 = (0, 1, 0, 0, 0, 0, 0, 0)\n    t4 = (0, 1, 0, 0, 1, 1, 1, 1)\n    t5 = (1, 1, 0, 0, 0, 0, 0, 0)\n    t6 = (1, 1, 0, 0, 1, 1, 1, 1)\n    \n    grouped = data.groupby('trial_type')['response']\n    means = grouped.mean()\n    counts = grouped.count()\n    \n    def unbiased_sq_diff(a, b):\n        if a not in means or b not in means:\n            return 0.0\n        ma = means[a]\n        mb = means[b]\n        na = counts[a]\n        nb = counts[b]\n        \n        var_a = ma * (1.0 - ma) / (na - 1.0) if na > 1 else 0.0\n        var_b = mb * (1.0 - mb) / (nb - 1.0) if nb > 1 else 0.0\n        \n        return (ma - mb)**2 - var_a - var_b\n\n    val = unbiased_sq_diff(t1, t2) + unbiased_sq_diff(t3, t4) + unbiased_sq_diff(t5, t6)\n    return float(val)",
  "rationale": "This metric evaluates the sum of squared differences in choice probabilities between matched pairs of trials (1 vs 2, 3 vs 4, 5 vs 6) that share identical feature differences but vary in baseline similarity. To eliminate the high variance associated with simple absolute differences, it computes an unbiased estimator of the true squared difference in probabilities by subtracting the expected binomial sampling variance from the observed squared difference of the means. Under the Competing Theory, the true choice probabilities for each pair are identical, so the expected value of this unbiased estimator is exactly zero. Under the Advocated Theory, Jaccard similarity shifts the choice probabilities between pairs, yielding a strictly positive expected value. This provides a sharp, low-variance dissociation between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 4824,
  "candidates_token_count": 643,
  "total_token_count": 9819
}
```
