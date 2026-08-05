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

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate Tally-then-TTB (advocated) from the Feature Cancellation model (competing), we exploit how each theory scales choice determinism. In Tally-then-TTB, choice determinism is driven purely by the final score difference. A 1-point tally advantage and a tie broken by the highest-validity cue both yield a score difference of exactly 1.0, leading to identical predicted determinism across such trials regardless of how many features the options share. In contrast, the competing model scales its determinism parameter (beta) inversely by the number of 'surviving' (non-canceling) features. We designed pairs of trials that have identical score differences under Tally-then-TTB (either a 1-point tally win or a tied tally broken by the top cue) but drastically different numbers of surviving features (e.g., 5 vs. 1, or 4 vs. 2). The advocated model predicts equal determinism across these pairs, whereas the competing model is structurally forced to predict much noisier choices on trials with more surviving features.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Tally-then-TTB (Tally with Validity Tie-Breaker): Decision-makers first compare options by tallying the total number of winning features for each option, ignoring cue validities. If one option has strictly more wins, it is chosen. This captures the compensatory nature of decision-making when there is a clear majority of supporting cues. However, if the tally results in a tie, the decision-maker falls back to a non-compensatory 'Take-The-Best' strategy to break the tie, choosing the option that wins on the single most valid cue. This hybrid model preserves robust Tallying performance on unequal-tally trials while capturing deterministic validity-based tie-breaking on ambiguous trials.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying phase: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    score_a = a_wins
    score_b = b_wins
    
    # Tie-breaking phase: if tallies are equal, use TTB
    if a_wins == b_wins:
        cue_order = np.argsort(-val, kind="stable")
        for idx in cue_order:
            if a[idx] > b[idx]:
                score_a += 1.0
                break
            elif b[idx] > a[idx]:
                score_b += 1.0
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Feature Cancellation then Tally/TTB with Cancellation-Scaled Determinism

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Feature cancellation: identify surviving distinct features
    mask = a != b
    
    # Tallying on distinct features
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    score_a = tally_a
    score_b = tally_b
    
    # If tallies are tied, break the tie using TTB on surviving features
    if tally_a == tally_b and np.sum(mask) > 0:
        order = np.argsort(val)[::-1]
        for idx in order:
            if mask[idx]:
                if a[idx] > b[idx]:
                    score_a += 1.0
                elif b[idx] > a[idx]:
                    score_b += 1.0
                break
            
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with cancellation-scaled determinism
    n_surviving = np.sum(mask)
    effective_beta = beta / n_surviving if n_surviving > 0 else beta
    
    z = effective_beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] rationale: The experimental design pairs trials that have identical score differences under the advocated model but different numbers of surviving (uncanceled) features. The advocated model predicts equal determinism (and thus equal probability of choosing option A) across all trials, yielding a difference of 0. The competing model scales its determinism parameter inversely by the number of surviving features, predicting a higher probability of choosing option A on trials with fewer surviving features. This metric calculates the difference in the proportion of choosing option A between low-surviving (<=2) and high-surviving (>=4) trials, perfectly discriminating the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_n_surv(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return sum(x != y for x, y in zip(a, b))
        
    n_surv = data.apply(get_n_surv, axis=1)
    chose_a = (data['response'] == 0).astype(float)
    
    low_surv_mask = n_surv <= 2
    high_surv_mask = n_surv >= 4
    
    p_a_low = chose_a[low_surv_mask].mean()
    p_a_high = chose_a[high_surv_mask].mean()
    
    if np.isnan(p_a_low) or np.isnan(p_a_high):
        return 0.0
        
    return float(p_a_low - p_a_high)
outcome: self_sim=-0.0046 (var=0.0031) adversary_sim=0.0554 (var=0.0103) welch_t=-2.588 p=0.01367 (N=25, alpha=0.01) -> reject

[1] rationale: To maximize the discrimination between the two models, we compare the extreme cases in the experimental design: trials with 1 surviving feature versus trials with 5 surviving features. The advocated model predicts an identical probability of choosing Option A for both, as the score difference is exactly 1 in both cases. The competing model scales its determinism parameter by the number of surviving features, predicting a significantly higher probability of choosing Option A on the 1-surviving-feature trial compared to the 5-surviving-feature trial. This metric computes the exact difference in choice proportions between these two extremes to maximize the effect size.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_n_surv(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return sum(x != y for x, y in zip(a, b))
        
    n_surv = data.apply(get_n_surv, axis=1)
    chose_a = (data['response'] == 0).astype(float)
    
    p_1 = chose_a[n_surv == 1].mean()
    p_5 = chose_a[n_surv == 5].mean()
    
    if np.isnan(p_1) or np.isnan(p_5):
        return 0.0
        
    return float(p_1 - p_5)
outcome: self_sim=-0.0067 (var=0.0074) adversary_sim=0.0583 (var=0.0140) welch_t=-2.225 p=0.03131 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    def get_n_surv(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        return sum(x != y for x, y in zip(a, b))\n        \n    n_surv = data.apply(get_n_surv, axis=1)\n    \n    subj_metrics = []\n    for subj, subj_df in data.groupby('subject_id'):\n        subj_n_surv = n_surv.loc[subj_df.index]\n        chose_a_subj = (subj_df['response'] == 0)\n        \n        lo = {}\n        for n in [1, 2, 4, 5]:\n            mask = (subj_n_surv == n)\n            if mask.sum() == 0:\n                lo[n] = 0.0\n                continue\n            n_a = chose_a_subj[mask].sum()\n            n_b = mask.sum() - n_a\n            # Smoothed empirical log odds\n            lo[n] = np.log((n_a + 0.5) / (n_b + 0.5))\n            \n        # Contrast log odds of low-surviving vs high-surviving feature trials\n        val = lo[1] + lo[2] - lo[4] - lo[5]\n        subj_metrics.append(val)\n        \n    if not subj_metrics:\n        return 0.0\n        \n    return float(np.mean(subj_metrics))",
  "rationale": "The experimental design pairs trials that have identical score differences under the advocated model but different numbers of surviving features. The advocated model predicts equal determinism across all trials, yielding an expected difference in log odds of 0. The competing model scales its determinism parameter inversely by the number of surviving features, predicting much higher log odds of choosing the favored option on trials with fewer surviving features. By computing the contrast in smoothed log odds of choosing Option A between low-surviving (1 and 2) and high-surviving (4 and 5) trials, this metric magnifies the differences between probabilities, avoiding the compression issues of raw probability differences and providing a highly discriminative signal."
}
```

## Usage

```json
{
  "prompt_token_count": 4232,
  "candidates_token_count": 557,
  "total_token_count": 8419
}
```
