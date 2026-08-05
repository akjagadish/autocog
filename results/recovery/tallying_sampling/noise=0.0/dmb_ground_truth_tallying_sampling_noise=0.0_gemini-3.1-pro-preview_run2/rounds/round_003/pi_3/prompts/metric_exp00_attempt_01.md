# metric_exp00_attempt_01

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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=11):**
  trial 1: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 9: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** The Advocated Theory (Softmax Tallying) predicts that choice probability depends exclusively on the difference in the number of positive features between the two options. For example, a 2 vs 1 trial yields the exact same choice probability as a 3 vs 2 trial, since the difference is 1 in both cases. The Competing Theory (Random Subset Tallying) predicts that as the total number of distinguishing features increases (e.g., moving from 1 vs 0 to 2 vs 1 to 3 vs 2), the choice probability will regress towards 0.5. This is because a larger pool of distinguishing features increases the probability that the stochastically sampled subset will favor the option with fewer overall features or result in a tie. By including trials with constant tally differences (e.g., diff=1 and diff=2) but varying total distinguishing features, we quantitatively dissociate the two models.

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
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
**Description:** Random Subset Tallying: Decision-makers use an equal-weight heuristic but are bounded by working memory, preventing them from processing all features simultaneously. Instead of calculating a complete tally and applying post-decision softmax noise, they stochastically sample a subset of the available features on each trial (each feature included independently with some probability) and perform pure tallying strictly on that subset. This provides a mechanistic, cognitive origin for choice variability while preserving the validity-agnostic, compensatory nature of the Tallying heuristic.

**Parameters:**
- sample_prob: [0.7, 1.0]
- epsilon: [0.0, 0.1]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    import itertools
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    n_features = stim.shape[1]
    p = float(parameters["sample_prob"])
    epsilon = float(parameters["epsilon"])
    
    prob_A = 0.0
    
    # Iterate over all possible subsets of features (2^n_features)
    for seq in itertools.product([0, 1], repeat=n_features):
        mask = np.array(seq)
        # Probability of sampling this specific subset
        subset_prob = np.prod(np.where(mask == 1, p, 1.0 - p))
        
        if subset_prob == 0:
            continue
            
        score_A = np.sum(stim[0] * mask)
        score_B = np.sum(stim[1] * mask)
        
        # Pure tallying on the sampled subset
        if score_A > score_B:
            prob_A += subset_prob
        elif score_A == score_B:
            prob_A += 0.5 * subset_prob
            
    prob_B = 1.0 - prob_A
    p_core = np.array([prob_A, prob_B])
    
    # Blend with uniform lapse distribution
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: Softmax Tallying predicts that choice probability is determined solely by the difference in the number of positive features between the options. Therefore, a trial with a 1 vs 0 feature difference should yield the exact same choice probability as a 3 vs 2 trial (both have a difference of 1). Similarly, a 2 vs 0 trial should have the same choice probability as a 4 vs 2 trial. In contrast, Random Subset Tallying predicts that as the total number of features increases, the probability of sampling a subset that results in a tie or favors the option with fewer features increases, pulling the choice probability towards 0.5. This metric measures the drop in the probability of choosing Option A when the total number of features increases (from 1 to 5, and from 2 to 6) while keeping the feature difference constant. It will be near zero for Softmax Tallying, but significantly positive for Random Subset Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    score_A = data['option_a_ratings'].apply(sum)
    score_B = data['option_b_ratings'].apply(sum)
    
    diff = score_A - score_B
    total = score_A + score_B
    
    chose_A = (data['response'] == 0).astype(float)
    
    m11 = (diff == 1) & (total == 1)
    m15 = (diff == 1) & (total == 5)
    m22 = (diff == 2) & (total == 2)
    m26 = (diff == 2) & (total == 6)
    
    v11 = chose_A[m11].mean() if m11.any() else 0.5
    v15 = chose_A[m15].mean() if m15.any() else 0.5
    v22 = chose_A[m22].mean() if m22.any() else 0.5
    v26 = chose_A[m26].mean() if m26.any() else 0.5
    
    return (v11 - v15) + (v22 - v26)
outcome: self_sim=0.0100 (var=0.0361) adversary_sim=0.1200 (var=0.0481) welch_t=-1.895 p=0.06426 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    score_A = data['option_a_ratings'].apply(sum)\n    score_B = data['option_b_ratings'].apply(sum)\n    \n    diff = score_A - score_B\n    total = score_A + score_B\n    \n    chose_A = (data['response'] == 0).astype(float)\n    \n    m1_low = (diff == 1) & (total <= 3)\n    m1_high = (diff == 1) & (total >= 5)\n    \n    m2_low = (diff == 2) & (total <= 4)\n    m2_high = (diff == 2) & (total >= 6)\n    \n    v1_low = chose_A[m1_low].mean() if m1_low.any() else 0.5\n    v1_high = chose_A[m1_high].mean() if m1_high.any() else 0.5\n    v2_low = chose_A[m2_low].mean() if m2_low.any() else 0.5\n    v2_high = chose_A[m2_high].mean() if m2_high.any() else 0.5\n    \n    return (v1_low - v1_high) + (v2_low - v2_high)",
  "rationale": "Softmax Tallying predicts that choice probability is determined solely by the difference in the number of positive features between the options; the total number of features is irrelevant. Random Subset Tallying predicts that as the total number of features increases (for a fixed difference), the choice probability regresses towards 0.5 due to a higher chance of sampling a subset that ties or favors the inferior option. By comparing the choice probabilities for low-total vs high-total trials within the diff=1 and diff=2 conditions, we can distinguish the models. This version pools multiple trials per condition (e.g., all trials with diff=1 and total<=3) to drastically reduce per-subject variance compared to using single-trial point estimates."
}
```

## Usage

```json
{
  "prompt_token_count": 4001,
  "candidates_token_count": 520,
  "total_token_count": 8023
}
```
