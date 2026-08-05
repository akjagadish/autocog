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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Weighted Additive (WADD) theory from the Strategy Mixture (TTB + Tallying) model by exploiting how secondary cues are integrated. Across a set of trials, we hold two things strictly constant: the highest discriminating cue (which fixes the TTB choice) and the total number of cues favoring each option (which fixes the Tallying difference at zero). Because the Strategy Mixture model's TTB component only cares about the top cue and its Tallying component only counts the sheer number of wins, it predicts identical choice probabilities across all these trials. WADD, however, integrates the scaled validities of all cues. By systematically swapping which secondary cues belong to Option A versus Option B, WADD predicts a steep, systematic change in choice probabilities as the secondary support for the TTB winner increases or decreases in validity.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) theory with non-linear scaling posits that decision-makers integrate all available information by weighting each feature by its validity, but subjectively scale these validities to adjust the degree of compensation. By exponentiating validities by a parameter gamma and then normalizing them, the model flexibly interpolates between equal weighting (Tallying), proportional weighting, and highly non-compensatory weighting (Take The Best) while keeping the integrated scores on a stable scale. Choices are made probabilistically using a softmax over the options' scores.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 30.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities with normalization to prevent score shrinkage
    weights = val ** gamma
    weights = weights / np.sum(weights)
    
    # Calculate overall scores by weighting features by their scaled validity
    a_score = np.dot(a, weights)
    b_score = np.dot(b, weights)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
**Description:** Decision-makers do not exclusively rely on a single decision rule; rather, choices reflect a probabilistic mixture of strategies on a trial-by-trial basis (or heterogeneity across individuals). The Strategy Mixture theory posits that a decision is made by selecting either a non-compensatory strategy (Take The Best) with probability alpha, or a compensatory strategy (Tallying) with probability 1 - alpha. Take The Best relies solely on the most valid discriminating cue, making it highly frugal and non-compensatory, whereas Tallying integrates information by counting the total number of cues favoring each option. By averaging across these two distinct decision rules, the model seamlessly bridges the non-compensatory invariance observed in Experiment 2 and the compensatory sensitivity observed in Experiment 1.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture model expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) component
    cue_order = np.argsort(-val, kind="stable").tolist()
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
        
    # Tallying component
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for TTB
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # Softmax for Tallying
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_tally
    
    # Lapse rate
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A when facing the 'weakest' B options (those relying on the lowest validity cues) versus the 'strongest' B options (those relying on the highest validity cues available to B), holding the sheer number of cues favoring B constant. Under the Strategy Mixture theory, the Tallying component only counts cues and the Take The Best component only looks at the highest validity cue (which A always wins). Thus, the Mixture model predicts no difference in choice probability (metric ≈ 0). In contrast, the Weighted Additive (WADD) theory integrates the exact validities, predicting a higher probability of choosing A when B's cues are weaker (metric > 0).
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert option_b_ratings to tuples to make them hashable and comparable
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Weakest opponent B options: cues 4 and 5
    weak_b = [(0, 0, 0, 0, 1), (0, 0, 0, 1, 1)]
    # Strongest opponent B options: cues 2 and 3
    strong_b = [(0, 1, 0, 0, 0), (0, 1, 1, 0, 0)]
    
    is_weak = b_tuples.isin(weak_b)
    is_strong = b_tuples.isin(strong_b)
    
    # Calculate probability of choosing Option A (response == 0)
    p_a_weak = (data.loc[is_weak, 'response'] == 0).mean()
    p_a_strong = (data.loc[is_strong, 'response'] == 0).mean()
    
    if pd.isna(p_a_weak) or pd.isna(p_a_strong):
        return 0.0
        
    return float(p_a_weak - p_a_strong)
outcome: self_sim=0.0478 (var=0.0204) adversary_sim=-0.0144 (var=0.0170) welch_t=+1.609 p=0.1142 (N=25, alpha=0.01) -> reject

[1] rationale: Under the Strategy Mixture theory, the probability of choosing Option B is perfectly identical across all 6 trials where B has exactly 2 features (trials 5-10). This is because TTB always favors A (since A has the most valid cue) and Tallying always favors B (since B has 2 cues vs A's 1 cue). The mixture model is blind to the specific validities of B's cues, so any variation in choice probability across these 6 trials is purely due to binomial noise. In contrast, the Weighted Additive (WADD) theory integrates the specific validities of B's cues. It predicts that the probability of choosing B will vary systematically across these 6 trials (e.g., B is much more likely to be chosen when its cues are the 2nd and 3rd most valid compared to when they are the 4th and 5th). Therefore, the variance of the mean choice probabilities across these 6 specific trial types should be significantly higher for data generated by WADD than for data generated by the Strategy Mixture model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Project option_b_ratings to tuple to make it hashable
    data['b_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Focus only on trials where B has exactly 2 features (trials 5-10)
    # These are the trials where Tallying always favors B and TTB always favors A.
    b_2_features = [
        (0, 1, 1, 0, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 0, 1),
        (0, 0, 0, 1, 1)
    ]
    
    subset = data[data['b_tuple'].isin(b_2_features)]
    
    # If not enough data, return 0
    if len(subset) == 0:
        return 0.0
        
    # Calculate the mean response (probability of choosing B) for each of these 6 trial types
    means = subset.groupby('b_tuple')['response'].mean()
    
    # If we don't have all 6, just use what we have
    if len(means) < 2:
        return 0.0
        
    # Return the variance of these choice probabilities across the 6 trial types
    return float(np.var(means, ddof=0))
outcome: self_sim=0.0008 (var=0.0003) adversary_sim=0.0003 (var=0.0002) welch_t=+0.108 p=0.9142 (N=25, alpha=0.01) -> reject

[2] rationale: This metric uses a set of zero-sum linear contrasts applied to the probability of choosing Option B. The trials are split into two groups: those where Option B has exactly 1 cue, and those where Option B has exactly 2 cues. Within each group, the Strategy Mixture model (TTB + Tallying) predicts identical choice probabilities because the Tallying difference is constant and TTB always favors Option A. Therefore, the expected value of any zero-sum contrast within these groups is exactly zero under the Mixture model. In contrast, the Weighted Additive (WADD) theory integrates the specific validities of B's cues, predicting a strictly positive relationship between B's validity rank and its choice probability. By assigning positive weights to the strongest B options and negative weights to the weakest B options within each group, and averaging over all trials, we maximize the signal-to-noise ratio for WADD while keeping the Mixture model's expected value anchored at zero. Averaging across all 10 trial types substantially reduces the within-subject binomial noise compared to using only a subset of extreme trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Project option_b_ratings to a tuple to use as a dictionary key
    data['b_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Define contrast weights for the 10 trial types based on the validity rank of B's cues
    # within their respective tally-groups (1-cue B options and 2-cue B options).
    # Under the Strategy Mixture model, the probability of choosing B is constant within 
    # the 1-cue group and constant within the 2-cue group. Thus, any zero-sum contrast 
    # within these groups will have an expected value of 0.
    # Under WADD, B options with higher validities are chosen more often, yielding a positive sum.
    weights = {
        # 1-cue B options (Trials 1-4)
        (0, 1, 0, 0, 0): 3.0,
        (0, 0, 1, 0, 0): 1.0,
        (0, 0, 0, 1, 0): -1.0,
        (0, 0, 0, 0, 1): -3.0,
        # 2-cue B options (Trials 5-10)
        (0, 1, 1, 0, 0): 5.0,
        (0, 1, 0, 1, 0): 3.0,
        (0, 1, 0, 0, 1): 1.0,
        (0, 0, 1, 1, 0): -1.0,
        (0, 0, 1, 0, 1): -3.0,
        (0, 0, 0, 1, 1): -5.0
    }
    
    # Map each trial to its corresponding contrast weight (default 0 if not found)
    data['weight'] = data['b_tuple'].map(weights).fillna(0.0)
    
    # Calculate the mean weighted response (where response == 1 indicates choosing Option B)
    # We group by subject to ensure balanced weighting if calculating on the pooled DataFrame,
    # then take the mean across subjects.
    subj_scores = data.groupby('subject_id').apply(lambda df: (df['response'] * df['weight']).mean())
    
    return float(subj_scores.mean())

outcome: self_sim=0.0642 (var=0.0235) adversary_sim=-0.0040 (var=0.0124) welch_t=+1.799 p=0.07894 (N=25, alpha=0.01) -> reject

[3] rationale: This metric calculates the difference in the probability of choosing Option B between two perfectly balanced halves of the experimental trials. Group 1 contains the 5 trial types where Option B is supported by higher-validity cues, while Group 2 contains the 5 trial types where Option B is supported by lower-validity cues. Crucially, both groups contain exactly two 1-cue trial types and three 2-cue trial types. Under the Strategy Mixture model, the choice probability depends only on the number of cues favoring each option, making the expected difference between these two groups exactly zero. By contrasting all 90 trials per subject in a balanced way, we minimize within-subject binomial noise while isolating the Weighted Additive (WADD) theory's sensitivity to specific cue validities, which predicts a strictly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    df['b_tuple'] = df['option_b_ratings'].apply(tuple)
    
    # Group 1: B options with higher-validity cues (two 1-cue, three 2-cue)
    group1 = {
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 1, 1, 0, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1)
    }
    
    # Group 2: B options with lower-validity cues (two 1-cue, three 2-cue)
    group2 = {
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 0, 1),
        (0, 0, 0, 1, 1)
    }
    
    def get_diff(sub_df):
        p1 = sub_df.loc[sub_df['b_tuple'].isin(group1), 'response'].mean()
        p2 = sub_df.loc[sub_df['b_tuple'].isin(group2), 'response'].mean()
        if pd.isna(p1) or pd.isna(p2):
            return 0.0
        return float(p1 - p2)
        
    return float(df.groupby('subject_id').apply(get_diff).mean())
outcome: self_sim=0.0387 (var=0.0073) adversary_sim=-0.0071 (var=0.0062) welch_t=+1.969 p=0.0548 (N=25, alpha=0.01) -> reject

[4] rationale: This metric contrasts trials where the Weighted Additive (WADD) theory predicts a large difference in value between the options vs. trials where it predicts a near-zero difference. Across all trials, the Tallying difference is strictly tied (2 cues vs 2 cues), and the Take The Best (TTB) model definitively favors one option. Therefore, the Strategy Mixture model predicts a constant probability of choosing the TTB winner across all trials (expected difference = 0). WADD, however, integrates the exact validities of the secondary cues, predicting that the TTB winner will be chosen significantly more often in the 'High WADD difference' trials than in the 'Low WADD difference' trials. By taking this within-subject difference, we cleanly dissociate the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    df = data.copy()
    df['a_tuple'] = df['option_a_ratings'].apply(tuple)
    df['a_first'] = df['a_tuple'].apply(lambda x: x[0])
    
    # Identify if the subject chose the option favored by Take The Best (TTB)
    # TTB always favors the option with cue 1 (since it has the highest validity)
    df['ttb_chosen'] = ((df['a_first'] == 1) & (df['response'] == 0)) | ((df['a_first'] == 0) & (df['response'] == 1))
    
    # High WADD difference trials: The secondary cues strongly favor the TTB winner
    high_wadd_a = {
        (1, 0, 1, 0, 0),
        (1, 1, 0, 0, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 1, 1, 0)
    }
    
    df['is_high'] = df['a_tuple'].isin(high_wadd_a)
    
    def get_diff(sub_df):
        high_mean = sub_df.loc[sub_df['is_high'], 'ttb_chosen'].mean()
        low_mean = sub_df.loc[~sub_df['is_high'], 'ttb_chosen'].mean()
        if pd.isna(high_mean) or pd.isna(low_mean):
            return 0.0
        return float(high_mean - low_mean)
        
    return float(df.groupby('subject_id').apply(get_diff).mean())
outcome: self_sim=0.0333 (var=0.0128) adversary_sim=-0.0096 (var=0.0077) welch_t=+1.501 p=0.1403 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    df = data.copy()\n    \n    # Identify if the subject chose the option favored by Take The Best (TTB)\n    # The highest validity cue is always feature 0 (validity 0.95).\n    # Thus, TTB always favors the option that has a 1 for feature 0.\n    df['a_has_f0'] = df['option_a_ratings'].apply(lambda x: x[0] == 1)\n    df['ttb_chosen'] = ((df['a_has_f0']) & (df['response'] == 0)) | ((~df['a_has_f0']) & (df['response'] == 1))\n    \n    # Define zero-sum contrast weights based on the WADD value difference.\n    # Under WADD (with gamma=1), the value differences between the TTB winner and loser are:\n    # Pair 1 (TTB winner has cue 4): diff = 0.0\n    # Pair 2 (TTB winner has cue 3): diff = 0.0\n    # Pair 3 (TTB winner has cue 2): diff = 0.3\n    # Pair 4 (TTB winner has cue 1): diff = 0.4\n    # Centering these differences yields proportional weights: -7, -7, +5, +9.\n    def get_weight(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        ttb_opt = a if a[0] == 1 else b\n        \n        if ttb_opt[1] == 1:\n            return 9.0\n        elif ttb_opt[2] == 1:\n            return 5.0\n        elif ttb_opt[3] == 1:\n            return -7.0\n        elif ttb_opt[4] == 1:\n            return -7.0\n        return 0.0\n        \n    df['weight'] = df.apply(get_weight, axis=1)\n    \n    # Calculate the mean weighted response per subject.\n    # Because the Strategy Mixture model predicts a constant P(TTB chosen) across all these trials \n    # (since Tallying is always tied 2 vs 2), its expected score is exactly 0.\n    subj_scores = df.groupby('subject_id').apply(lambda d: (d['ttb_chosen'] * d['weight']).mean())\n    \n    return float(subj_scores.mean())",
  "rationale": "Across all 8 trial types, the Strategy Mixture model predicts a perfectly constant probability of choosing the Take The Best (TTB) winner. This is because TTB always favors the option with the highest validity cue (feature 0), and Tallying always sees a 2-vs-2 tie, rendering the secondary cues irrelevant to the Mixture model. Therefore, any zero-sum contrast applied to the probability of choosing the TTB winner has an expected value of exactly zero under the Mixture model. In contrast, the Weighted Additive (WADD) theory integrates the secondary cues. It predicts that the TTB winner will be chosen much more frequently when it is supported by strong secondary cues (Pairs 3 and 4) than when it is supported by weak secondary cues (Pairs 1 and 2). By applying optimized zero-sum contrast weights (-7, -7, +5, +9) proportional to the WADD value differences, we maximize the signal-to-noise ratio for WADD while keeping the Mixture model strictly anchored at zero."
}
```

## Usage

```json
{
  "prompt_token_count": 7044,
  "candidates_token_count": 840,
  "total_token_count": 14656
}
```
