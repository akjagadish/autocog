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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 8: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** This design leverages an 'invariance' property of the Strategy Mixture (TTB + Tallying) model to quantitatively dissociate it from the WADD model with Non-Linear Cue Scaling. In the Strategy Mixture model, as long as the TTB prediction (determined by the highest discriminating cue) and the Tallying prediction (the count of winning cues) remain constant, the predicted choice probabilities are identical across trials. By creating a sequence of trials where Option A always wins the highest validity cue and Option A and B always win the exact same number of total cues, the Strategy Mixture model predicts the exact same choice probabilities. However, the WADD model integrates the actual validities of the specific lower-tier cues. By shifting which specific lower-validity cues Option A vs. B wins across these trials, WADD predicts systematically shifting choice probabilities, providing a strict quantitative dissociation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) theory with Non-Linear Cue Scaling posits that decision makers compute a total score for each option by summing the validities of the features it possesses, but they may non-linearly amplify or dampen the differences between cue validities. By exponentiating cue validities with a free parameter, the model can naturally interpolate between Tallying (where all cues are weighted equally) and Take The Best (where the highest-validity cue dominates).

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [1.0, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Scale validities non-linearly to allow amplification of the best cues
    val_scaled = val ** gamma
    
    # Compute WADD scores: sum of scaled validities for features possessed by the option
    score_a = np.sum(val_scaled * a)
    score_b = np.sum(val_scaled * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
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
**Description:** Strategy Mixture (TTB + Tallying): Decision makers do not universally adhere to a single strategy. Instead, they use a probabilistic mixture of a non-compensatory strategy (Take The Best) and a compensatory strategy (Tallying). A parameter P_TTB dictates the probability of using TTB on any given trial, while 1 - P_TTB is the probability of using Tallying. This accounts for intermediate levels of TTB-consistency and Tallying-consistency observed in empirical data across subjects and trials. The mixture captures a balance between TTB and Tallying, avoiding over-reliance on uniform guessing.

**Parameters:**
- beta: [0.1, 20.0]
- p_ttb: [0.35, 0.95]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    # Normalize by n_features to keep scale comparable to TTB for the shared beta
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta = float(parameters["beta"])
    
    # TTB Probabilities
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_ttb_weight * p_ttb_dist + (1.0 - p_ttb_weight) * p_tally_dist
    
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
[0] rationale: This metric leverages the invariance property of the Strategy Mixture (TTB + Tallying) model. Trials are grouped into equivalence classes where both the TTB prediction (based on the highest-validity cue) and the Tallying prediction (based on the total number of winning cues) are identical. According to the Strategy Mixture theory, the underlying choice probabilities should be exactly the same for all trials within a group, making the variance of empirical choice proportions within these groups very low (arising only from binomial sampling noise). In contrast, the WADD model with non-linear cue scaling integrates the exact validities of the lower-tier cues, leading to systematically different choice probabilities across trials within the same group. Thus, WADD predicts a significantly higher within-group variance of choice proportions compared to the Strategy Mixture model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = data['option_a_ratings'].tolist()
    b_ratings = data['option_b_ratings'].tolist()
    
    trial_ids = [ (tuple(a), tuple(b)) for a, b in zip(a_ratings, b_ratings) ]
    # Group trials by the TTB winner (cue 0) and the Tallying winner (sum of cues)
    groups = [ (1 if a[0] > b[0] else 0, sum(a)) for a, b in zip(a_ratings, b_ratings) ]
    
    df = pd.DataFrame({
        'subject_id': data['subject_id'].values,
        'trial_id': trial_ids,
        'group': groups,
        'response': data['response'].values
    })
    
    # Compute the proportion of choosing Option B for each specific trial type
    trial_means = df.groupby(['subject_id', 'group', 'trial_id'])['response'].mean().reset_index()
    
    # Count number of unique trial types per subject and group
    counts = trial_means.groupby(['subject_id', 'group'])['trial_id'].count().reset_index(name='n_trials')
    
    # Only consider groups that contain more than 1 trial type to measure within-group variance
    merged = pd.merge(trial_means, counts, on=['subject_id', 'group'])
    merged = merged[merged['n_trials'] > 1]
    
    if len(merged) == 0:
        return 0.0
        
    # Compute variance of choice proportions within each subject and group
    variances = merged.groupby(['subject_id', 'group'])['response'].apply(lambda x: np.var(x.values, ddof=0))
    
    return float(variances.mean())

outcome: self_sim=0.0144 (var=0.0002) adversary_sim=0.0084 (var=0.0000) welch_t=+2.022 p=0.05157 (N=25, alpha=0.01) -> reject

[1] rationale: The Strategy Mixture model predicts identical choice probabilities for trials where the TTB and Tallying predictions match. Trials 1 and 3 both have TTB predicting A and Tallying predicting A, so P(B) should be identical. Trials 4 and 6 both have TTB predicting B and Tallying predicting A, so P(B) should be identical. In contrast, WADD integrates the exact validities of all cues. In Trial 1, B wins higher validity lower-tier cues than in Trial 3, making B more attractive in Trial 1. Similarly, in Trial 6, B wins higher validity cues than in Trial 4, making B more attractive in Trial 6. Thus, the contrast (P(B|Trial 1) - P(B|Trial 3)) + (P(B|Trial 6) - P(B|Trial 4)) will be around 0 for the Strategy Mixture model, but significantly greater than 0 for WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Create string representation of option A ratings to identify trials
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'A_str': a_str,
        'response': data['response']
    })
    
    # Calculate mean response (P(choose B)) for each trial type per subject
    means = df.groupby(['subject_id', 'A_str'])['response'].mean().unstack()
    
    # Extract the trial types of interest. 
    # Trial 1: A='10011', Trial 3: A='11001'
    # Trial 4: A='01110', Trial 6: A='01011'
    t1 = means.get('10011', pd.Series(0, index=means.index))
    t3 = means.get('11001', pd.Series(0, index=means.index))
    t4 = means.get('01110', pd.Series(0, index=means.index))
    t6 = means.get('01011', pd.Series(0, index=means.index))
    
    # Compute the targeted contrast
    contrast = (t1 - t3) + (t6 - t4)
    
    return float(contrast.mean())
outcome: self_sim=0.1467 (var=0.1054) adversary_sim=-0.0283 (var=0.0685) welch_t=+2.098 p=0.04143 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    a_str = data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    \n    df = pd.DataFrame({\n        'subject_id': data['subject_id'],\n        'A_str': a_str,\n        'response': data['response']\n    })\n    \n    means = df.groupby(['subject_id', 'A_str'])['response'].mean().unstack()\n    \n    t1 = means.get('10011', pd.Series(0, index=means.index))\n    t3 = means.get('11001', pd.Series(0, index=means.index))\n    t6 = means.get('01011', pd.Series(0, index=means.index))\n    t4 = means.get('01110', pd.Series(0, index=means.index))\n    t7 = means.get('10001', pd.Series(0, index=means.index))\n    t8 = means.get('10010', pd.Series(0, index=means.index))\n    \n    contrast = (t1 - t3) + (t6 - t4) + (t7 - t8)\n    \n    return float(contrast.mean())",
  "rationale": "Strategy Mixture (TTB + Tallying) groups trials into equivalence classes where the predictions from TTB and Tallying are identical. Trials 1 and 3 are in the same class, as are Trials 6 and 4, and Trials 7 and 8. The Strategy Mixture model consequently predicts identical choice probabilities within these pairs, meaning the difference in P(Choose B) should be zero. In contrast, the WADD model integrates the varying specific validities of the lower-tier cues, making Option B systematically more attractive in Trial 1 vs 3, Trial 6 vs 4, and Trial 7 vs 8. By aggregating these three pairwise differences, we significantly amplify the signal of the WADD model's cue scaling while the Strategy Mixture model strictly predicts an expected sum of zero, improving the signal-to-noise ratio over a single pair."
}
```

## Usage

```json
{
  "prompt_token_count": 4997,
  "candidates_token_count": 537,
  "total_token_count": 8903
}
```
