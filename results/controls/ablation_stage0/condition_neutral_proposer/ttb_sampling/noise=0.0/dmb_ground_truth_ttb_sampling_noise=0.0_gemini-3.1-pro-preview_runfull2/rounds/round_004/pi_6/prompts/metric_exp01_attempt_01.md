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
**Validities (n_features=5):** [0.99, 0.88, 0.77, 0.66, 0.55]

**Trial pairs (n=11):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 11: A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** Both Theory 1 (Exponentially Weighted Compensatory Model) and Theory 2 (Take The Best) share the same deterministic choice direction, as Theory 1's base >= 2.0 guarantees strict lexicographic dominance matching TTB's sequential search. The crucial difference lies in their probabilistic choice generation. Theory 2 (TTB) predicts a constant choice probability (determined entirely by the lapse rate) whenever the top discriminating cue is the same, completely ignoring the states of all lower-ranked cues. In contrast, Theory 1 applies a softmax function over the exponentially weighted sum of all cues, making its choice probability sensitive to the margin of the value difference. By holding the top discriminating cue constant and systematically varying the number of lower-ranked cues that either oppose or support the top cue, we can quantitatively dissociate the models: Theory 2 predicts a flat choice probability across these variations, whereas Theory 1 predicts a graded choice probability that monotonically increases as more lower-ranked cues align with the top cue.

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



## ADVOCATED THEORY
**Description:** Exponentially Weighted Compensatory Model (Rank-based with Strict Dominance Base): Decision-makers evaluate options using a single compensatory process where the weight of each feature grows exponentially with its subjective validity rank, using a base >= 2.0. This guarantees strict lexicographic dominance, ensuring that a single higher-ranked cue always outweighs all lower-ranked cues combined. By enforcing this strict dominance, the model acts identically to Take-The-Best across all conflict scenarios, relying on an independent lapse rate for probabilistic errors rather than softening the decision temperature.

**Parameters:**
- base: [2.0, 10.0]
- tau: [0.0, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    base = float(parameters["base"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Compute weights using exponential growth based on rank
    # A base >= 2.0 ensures strict TTB behavior (lexicographic dominance)
    order = np.argsort(validities, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(validities))
    
    w = base ** ranks
    
    # Compute overall value for each option
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
    # Compute choice probabilities using softmax over values
    logits = tau * np.array([v_a, v_b])
    logits = logits - np.max(logits)
    p = np.exp(logits)
    p = p / np.sum(p)
    
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Take The Best (TTB): People compare two options by ordering features by their subjective validity and searching through them sequentially. The search stops at the first feature that discriminates between the two options (i.e., one option has a higher value than the other), and the decision is based entirely on that single feature. This non-compensatory strategy ignores all other features, preventing any compensatory trade-offs. If no feature discriminates, the learner guesses. Response noise is modeled via an independent lapse rate epsilon, which replaces the deterministic TTB choice with a uniform random pick.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(validities, kind='stable')[::-1]
    
    # Default to guessing if no cue discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
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
[0] rationale: In every trial of the experimental design, the first discriminating cue favors Option A. Take-The-Best (TTB) therefore predicts a constant probability of choosing A across all trials, driven solely by the lapse rate. In contrast, the Exponentially Weighted Compensatory model evaluates the margin of difference between the two options. By comparing the choice rate of A when Option B has no supporting features (maximizing the value difference) against trials where Option B has supporting features (minimizing the value difference), we can isolate the compensatory nature of the exponential model. TTB predicts a difference of 0, while the Exponential model predicts a positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # response == 0 means option A was chosen, response == 1 means option B
    chose_a = 1 - data['response']
    
    # In all 12 trials, the highest-validity discriminating cue favors Option A.
    # TTB will therefore choose Option A with a constant probability across all trials.
    # The Exponential Compensatory model, however, computes the difference in values.
    # When Option B has 0 features, the value difference is maximal (strong A trials).
    # When Option B has features, it reduces the value difference (weak A trials).
    
    sum_b = data['option_b_ratings'].apply(sum)
    
    strong_a_rate = chose_a[sum_b == 0].mean()
    weak_a_rate = chose_a[sum_b > 0].mean()
    
    return float(strong_a_rate - weak_a_rate)
outcome: self_sim=-0.0056 (var=0.0051) adversary_sim=-0.0015 (var=0.0037) welch_t=-0.217 p=0.8289 (N=25, alpha=0.01) -> reject

[1] rationale: By isolating trials where Option A has exactly one positive feature, we control for the baseline strength of Option A. We then compare 'low conflict' trials where Option B has 0 features against 'high conflict' trials where Option B has at least 1 feature. Take-The-Best predicts no difference in choice probabilities between these conditions because the highest validity cue always determines the choice deterministically. The Exponentially Weighted Compensatory Model predicts a higher probability of choosing A in the low conflict condition because the margin of value difference is larger.
metric_source:
def metric(data: pd.DataFrame) -> float:
    chose_a = 1.0 - data['response']
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    mask_a1 = a_sums == 1
    low_conflict = mask_a1 & (b_sums == 0)
    high_conflict = mask_a1 & (b_sums > 0)
    if low_conflict.sum() == 0 or high_conflict.sum() == 0:
        return 0.0
    return float(chose_a[low_conflict].mean() - chose_a[high_conflict].mean())
outcome: self_sim=0.0064 (var=0.0057) adversary_sim=0.0001 (var=0.0050) welch_t=+0.303 p=0.7634 (N=25, alpha=0.01) -> reject

[2] rationale: Following the experimental rationale, the Exponentially Weighted Compensatory Model (Theory 1) predicts a graded probability of choosing the dominant option that depends on the margin of value difference. By holding the top discriminating cue constant and looking at the net support of lower-ranked cues, we can split the trials into two perfectly balanced halves: trials where lower cues align with or are neutral to the top cue (net support >= 0), and trials where lower cues oppose the top cue (net support < 0). Theory 2 (TTB) predicts a completely flat choice probability across both sets (difference = 0), whereas Theory 1 predicts a higher choice probability for the aligned set. Balancing the split to 48 vs 48 trials minimizes the irreducible binomial noise per subject, maximizing the statistical power to detect Theory 1's compensatory margin effect.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    net_supports = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        top_idx = -1
        for i in range(len(a)):
            if a[i] != b[i]:
                top_idx = i
                break
        if top_idx != -1:
            # Calculate net support from lower-ranked cues
            ns = sum(a[top_idx+1:]) - sum(b[top_idx+1:])
        else:
            ns = 0
        net_supports.append(ns)
        
    ns_array = np.array(net_supports, dtype=float)
    chose_a = 1.0 - data['response'].values
    
    # Split trials into those where lower cues generally support or are neutral towards A (ns >= 0)
    # vs those where lower cues oppose A (ns < 0).
    # This balances the dataset into exactly 6 trial types (48 trials) each, minimizing binomial variance.
    pos_mask = ns_array >= 0
    neg_mask = ns_array < 0
    
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return 0.0
        
    return float(chose_a[pos_mask].mean() - chose_a[neg_mask].mean())
outcome: self_sim=0.0017 (var=0.0048) adversary_sim=-0.0004 (var=0.0040) welch_t=+0.111 p=0.9118 (N=25, alpha=0.01) -> reject

[3] rationale: Theory 1 (Exponentially Weighted Compensatory Model) predicts a graded probability of choosing A depending on the margin of value difference. Trial 3 provides the maximum possible value difference in favor of A, while Trial 9 provides the minimum possible value difference. By comparing the choice rate of A between these two extremes, we isolate the compensatory margin effect. Theory 2 (Take-The-Best) predicts a completely flat choice probability across both trials, resulting in an expected difference of 0.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 3 (A=[1,1,1,1,1], B=[0,0,0,0,0]) and Trial 9 (A=[0,0,0,1,0], B=[0,0,0,0,1])
    # Trial 3 has the maximum possible value difference, Trial 9 has the minimum.
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    trial_3_mask = (a_sums == 5) & (b_sums == 0)
    trial_9_mask = (a_sums == 1) & (b_sums == 1)
    
    chose_a = 1.0 - data['response']
    
    if trial_3_mask.sum() == 0 or trial_9_mask.sum() == 0:
        return 0.0
        
    return float(chose_a[trial_3_mask].mean() - chose_a[trial_9_mask].mean())
outcome: self_sim=-0.0237 (var=0.0208) adversary_sim=-0.0112 (var=0.0203) welch_t=-0.309 p=0.759 (N=25, alpha=0.01) -> reject

[4] rationale: Theory 1 (Exponentially Weighted Compensatory Model) and Theory 2 (Take-The-Best) are virtually identical in choice direction due to strict lexicographic dominance (base >= 2.0). The only divergence lies in Theory 1's softmax choice probability, which is sensitive to the absolute margin of value difference. By analytically determining the trials with the absolute smallest margins (Trials 1 and 10) and the absolute largest margins (Trials 8 and 9), we can maximize the theoretical difference in choice probabilities for Theory 1. TTB predicts a completely flat choice probability across all these trials (expected difference = 0). Pooling two trials for each condition halves the irreducible binomial variance compared to previous 1-vs-1 trial comparisons, maximizing the statistical power to detect Theory 1's compensatory margin effect.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate sums and first cue to uniquely identify trials
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    
    # Hard trials (minimum margin of value difference for Theory 1):
    # Trial 1: A=[1,0,0,0,0], B=[0,1,1,1,1] -> a_sums=1, b_sums=4
    # Trial 10: A=[0,1,0,0,0], B=[0,0,1,1,1] -> a_first=0, b_sums=3
    hard_mask = ((a_sums == 1) & (b_sums == 4)) | ((a_first == 0) & (b_sums == 3))
    
    # Easy trials (maximum margin of value difference for Theory 1):
    # Trial 8: A=[1,1,1,1,0], B=[0,0,0,0,0] -> a_first=1, a_sums=4, b_sums=0
    # Trial 9: A=[1,1,1,1,1], B=[0,0,0,0,0] -> a_sums=5, b_sums=0
    easy_mask = ((a_first == 1) & (a_sums == 4) & (b_sums == 0)) | ((a_sums == 5) & (b_sums == 0))
    
    # Response 0 means A was chosen, 1 means B was chosen
    chose_a = 1.0 - data['response']
    
    if hard_mask.sum() == 0 or easy_mask.sum() == 0:
        return 0.0
        
    # Return difference in probability of choosing A (Easy - Hard)
    return float(chose_a[easy_mask].mean() - chose_a[hard_mask].mean())
outcome: self_sim=-0.0050 (var=0.0115) adversary_sim=0.0400 (var=0.0172) welch_t=-1.328 p=0.1906 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Calculate a proxy for value difference assuming base=2\n    # The cues are ordered by validity, so we assign exponentially decreasing weights\n    weights = np.array([16.0, 8.0, 4.0, 2.0, 1.0])\n    \n    # Convert lists to 2D numpy arrays for fast computation\n    a_mat = np.stack(data['option_a_ratings'].values)\n    b_mat = np.stack(data['option_b_ratings'].values)\n    \n    v_a = np.dot(a_mat, weights)\n    v_b = np.dot(b_mat, weights)\n    v_diff = v_a - v_b\n    \n    chose_a = 1.0 - data['response'].values\n    \n    # Return the covariance between the proxy value difference and the choice of A\n    return float(np.cov(v_diff, chose_a)[0, 1])",
  "rationale": "In all 11 trials of the experimental design, the first discriminating cue favors Option A. Take-The-Best (Theory 2) ignores all other cues, predicting a constant probability of choosing A across all trials (determined only by the lapse rate). Consequently, the covariance between any measure of value difference and the choice of A will be exactly 0 on average for TTB. In contrast, the Exponentially Weighted Compensatory Model (Theory 1) translates the margin of value difference into choice probabilities via a softmax function. By calculating a proxy value difference using base=2 weights, we can compute the covariance between this margin and the probability of choosing A. Theory 1 predicts a strictly positive covariance because larger value differences lead to higher probabilities of choosing A. This metric leverages all 11 trials simultaneously, maximizing statistical power and minimizing binomial noise."
}
```

## Usage

```json
{
  "prompt_token_count": 6040,
  "candidates_token_count": 446,
  "total_token_count": 10227
}
```
