# metric_exp02_attempt_00

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
**Validities (n_features=7):** [0.96, 0.94, 0.92, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 1]
  trial 4: A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  trial 6: A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  trial 7: A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1]

**Rationale:** This design achieves a strict, parameter-free qualitative dissociation between the Strategy Mixture (TTB + Tallying) model and the WADD model with Non-Linear Cue Scaling. By carefully setting the validities and cue distributions, we create 'Adversarial' trials (e.g., Trial 1) where Option A possesses the single highest-validity cue (winning TTB) AND a larger total number of cues (winning Tallying). Because both sub-strategies of the Mixture model prefer Option A, the Mixture model is mathematically forced to predict a preference for A. However, Option B possesses a concentrated coalition of slightly lower but still highly valid cues. Across the entire allowed parameter space for WADD (gamma in [1.0, 20.0]), the exponentiated sum of Option B's coalition strictly outweighs Option A's single best cue plus its low-validity 'garbage' cues. Thus, WADD deterministically predicts a preference for Option B, while the Strategy Mixture deterministically predicts a preference for Option A. We include inverted versions and crossover trials to fully map the parameter space and rule out alternative heuristics.

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

[2] rationale: Strategy Mixture (TTB + Tallying) groups trials into equivalence classes where the predictions from TTB and Tallying are identical. Trials 1 and 3 are in the same class, as are Trials 6 and 4, and Trials 7 and 8. The Strategy Mixture model consequently predicts identical choice probabilities within these pairs, meaning the difference in P(Choose B) should be zero. In contrast, the WADD model integrates the varying specific validities of the lower-tier cues, making Option B systematically more attractive in Trial 1 vs 3, Trial 6 vs 4, and Trial 7 vs 8. By aggregating these three pairwise differences, we significantly amplify the signal of the WADD model's cue scaling while the Strategy Mixture model strictly predicts an expected sum of zero, improving the signal-to-noise ratio over a single pair.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'A_str': a_str,
        'response': data['response']
    })
    
    means = df.groupby(['subject_id', 'A_str'])['response'].mean().unstack()
    
    t1 = means.get('10011', pd.Series(0, index=means.index))
    t3 = means.get('11001', pd.Series(0, index=means.index))
    t6 = means.get('01011', pd.Series(0, index=means.index))
    t4 = means.get('01110', pd.Series(0, index=means.index))
    t7 = means.get('10001', pd.Series(0, index=means.index))
    t8 = means.get('10010', pd.Series(0, index=means.index))
    
    contrast = (t1 - t3) + (t6 - t4) + (t7 - t8)
    
    return float(contrast.mean())
outcome: self_sim=0.1933 (var=0.2110) adversary_sim=-0.0200 (var=0.1090) welch_t=+1.886 p=0.06602 (N=25, alpha=0.01) -> reject

[3] rationale: To overcome the high between-subject variance from simple P(B) contrasts, this metric uses a split-half cross-validation approach to compute an unbiased estimator of the squared difference in choice probabilities between trial pairs. Under the Strategy Mixture model, trials 1 & 3, 6 & 4, and 7 & 8 form equivalence classes with identical predicted choice probabilities. Thus, their expected difference is exactly zero, and the split-half cross-product will tightly cluster around zero (as the binomial noise in the two halves is independent). Under the WADD model, non-linear cue scaling produces systematically different probabilities within these pairs, so the cross-product will have a strictly positive expected value. This cleanly dissociates the two models with minimal noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'A_str': a_str,
        'response': data['response']
    })
    
    # Trial pairs that belong to the same equivalence class under Strategy Mixture
    # but have different predicted probabilities under WADD due to cue validities.
    pairs = [
        ('10011', '11001'), # t1 vs t3
        ('01011', '01110'), # t6 vs t4
        ('10001', '10010')  # t7 vs t8
    ]
    
    scores = []
    for subj, grp in df.groupby('subject_id'):
        subj_score = 0
        for s_a, s_b in pairs:
            ra = grp[grp['A_str'] == s_a]['response'].values
            rb = grp[grp['A_str'] == s_b]['response'].values
            if len(ra) >= 2 and len(rb) >= 2:
                # Split-half cross-product to get an unbiased estimator of (P(A) - P(B))^2
                # This removes the binomial noise bias that inflates variance.
                ra_even, ra_odd = ra[::2].mean(), ra[1::2].mean()
                rb_even, rb_odd = rb[::2].mean(), rb[1::2].mean()
                subj_score += (ra_even - rb_even) * (ra_odd - rb_odd)
        scores.append(subj_score)
        
    return float(np.mean(scores))
outcome: self_sim=0.0967 (var=0.0348) adversary_sim=0.0039 (var=0.0041) welch_t=+2.353 p=0.02547 (N=25, alpha=0.01) -> reject

[4] rationale: This metric evaluates the within-equivalence-class correlation between the choice response and the relative difference in linear cue validities. By grouping trials into sets where the Strategy Mixture model predicts identical probabilities (e.g., Trials 1, 2, 3, and 4 all have TTB favoring A and Tallying favoring B), we can assign zero-sum contrast weights proportional to the remaining WADD value difference. Under Strategy Mixture, the expected value of this contrast is exactly zero. Under WADD with non-linear cue scaling, Option B becomes systematically more attractive as it wins higher-validity lower-tier cues, leading to a strongly positive expected covariance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Project option A ratings to string to identify trial types
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # We define contrast weights based on the within-group centered WADD difference (B - A).
    # Trials are grouped into equivalence classes under the Strategy Mixture model (where TTB
    # and Tallying make identical predictions). Within each class, the weights sum to 0.
    # Therefore, Strategy Mixture strictly predicts an expected score of 0.
    # WADD predicts a positive score, as P(Choose B) scales with the WADD difference.
    weights = {
        '11000': -3.0, # Trial 1 (Group 1)
        '10100': -1.0, # Trial 2 (Group 1)
        '10010':  1.0, # Trial 3 (Group 1)
        '10001':  3.0, # Trial 4 (Group 1)
        '10011':  1.0, # Trial 5 (Group 2)
        '10101': -1.0, # Trial 6 (Group 2)
        '01100': -1.0, # Trial 7 (Group 3)
        '01010':  1.0  # Trial 8 (Group 3)
    }
    
    w = a_str.map(weights).fillna(0.0)
    
    # Compute the mean covariance between the response and the WADD-derived contrast weight
    subj_scores = (data['response'] * w).groupby(data['subject_id']).mean()
    
    return float(subj_scores.mean())

outcome: self_sim=0.0802 (var=0.0160) adversary_sim=0.0231 (var=0.0055) welch_t=+1.948 p=0.05867 (N=25, alpha=0.01) -> reject

[5] rationale: This metric isolates the first equivalence class of the Strategy Mixture model (Trials 1 through 4), where TTB predicts Option A and Tallying predicts Option B. Because the predictions of both sub-strategies are identical across these four trials, the Strategy Mixture model strictly predicts that the choice probabilities are uniform across them. Consequently, any zero-sum linear contrast within this group has an expected value of exactly zero. In contrast, the WADD model integrates the varying validities of the lower-tier cues. From Trial 1 to Trial 4, Option B wins increasingly higher-validity cues, making it systematically more attractive. By using a flat contrast `(T4 + T3) - (T2 + T1)`, we capture this monotonic shift predicted by WADD while minimizing the variance inflation that comes from using larger linear weights, yielding a high signal-to-noise ratio.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Project option A ratings to a string to reliably identify trial types
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'A_str': a_str,
        'response': data['response']
    })
    
    # Calculate mean response (P(choose B)) for each trial type per subject
    means = df.groupby(['subject_id', 'A_str'])['response'].mean().unstack()
    
    # Under Strategy Mixture, Trials 1, 2, 3, and 4 belong to the same equivalence class
    # (TTB favors A, Tallying favors B). Thus, P(Choose B) should be identical.
    # Under WADD, Option B becomes systematically more attractive from Trial 1 to Trial 4
    # because it wins higher-validity lower-tier cues.
    t1 = means.get('11000', pd.Series(0, index=means.index))
    t2 = means.get('10100', pd.Series(0, index=means.index))
    t3 = means.get('10010', pd.Series(0, index=means.index))
    t4 = means.get('10001', pd.Series(0, index=means.index))
    
    # We use a flat unweighted contrast to prevent amplifying binomial noise.
    # Expected value is exactly 0 under Strategy Mixture, but positive under WADD.
    contrast = (t4 + t3) - (t2 + t1)
    
    return float(contrast.mean())
outcome: self_sim=0.2550 (var=0.1984) adversary_sim=0.0817 (var=0.0554) welch_t=+1.720 p=0.0939 (N=25, alpha=0.01) -> reject

[6] rationale: To drastically reduce the between-subject variance that caused previous metrics to fail, this metric binarizes the within-equivalence-class contrasts using the sign function. Under the Strategy Mixture model, the true choice probabilities within the specified pairs (e.g., T4 vs T1) are identical, so the empirical differences are driven entirely by symmetric binomial noise; the expected sign of the difference is strictly 0. Under the WADD model, the ranking of cue validities guarantees that Option B is systematically favored in one specific direction regardless of the non-linear scaling parameter (gamma). By taking the sign of the empirical difference, WADD will consistently score near +1 (collapsing parameter-driven variance in the magnitude of the difference), while Strategy Mixture will consistently average to 0, producing an enormous signal-to-noise ratio.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'A_str': a_str,
        'response': data['response']
    })
    
    means = df.groupby(['subject_id', 'A_str'])['response'].mean().unstack()
    
    # Fallback to 0.5 if a trial type is somehow missing for a subject
    t1 = means.get('11000', pd.Series(0.5, index=means.index))
    t2 = means.get('10100', pd.Series(0.5, index=means.index))
    t3 = means.get('10010', pd.Series(0.5, index=means.index))
    t4 = means.get('10001', pd.Series(0.5, index=means.index))
    t5 = means.get('10011', pd.Series(0.5, index=means.index))
    t6 = means.get('10101', pd.Series(0.5, index=means.index))
    t7 = means.get('01100', pd.Series(0.5, index=means.index))
    t8 = means.get('01010', pd.Series(0.5, index=means.index))
    
    # WADD consistently predicts P(B) is higher in the first term of each pair
    # Strategy Mixture predicts identical probabilities for both terms in each pair
    c1 = np.sign(t4 - t1)
    c2 = np.sign(t3 - t2)
    c3 = np.sign(t5 - t6)
    c4 = np.sign(t8 - t7)
    
    # Average the signs per subject
    score = (c1 + c2 + c3 + c4) / 4.0
    
    return float(score.mean())
outcome: self_sim=0.2000 (var=0.1500) adversary_sim=0.0550 (var=0.2182) welch_t=+1.195 p=0.2383 (N=25, alpha=0.01) -> reject

[7] rationale: Previous metrics failed because the WADD model's free parameters (like beta and gamma) create massive between-subject variance in the absolute magnitude of the probability differences, destroying the t-test's power. To solve this, this metric computes the within-subject Pearson correlation between the empirical choice proportions and the theoretical linear contrast for the first equivalence class. Under the Strategy Mixture model, the true probabilities are identical, so the empirical correlation is driven purely by binomial noise and has an expected value of exactly 0. Under the WADD model, the probabilities systematically increase from Trial 1 to Trial 4. By taking the correlation coefficient, we standardize the scale to [-1, 1], completely removing the variance caused by individual differences in parameter magnitude while preserving the highly consistent positive linear trend predicted by WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Project option A ratings to a string to identify trial types
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'A_str': a_str,
        'response': data['response']
    })
    
    # Compute the proportion of choosing Option B for each trial type per subject
    means = df.groupby(['subject_id', 'A_str'])['response'].mean().unstack()
    
    # Under Strategy Mixture, Trials 1, 2, 3, and 4 belong to the same equivalence class
    # (TTB favors A, Tallying favors B). Thus, P(Choose B) should be identical.
    # Under WADD, Option B becomes systematically more attractive from Trial 1 to Trial 4
    # because it wins higher-validity lower-tier cues.
    t1 = means.get('11000', pd.Series(0.5, index=means.index))
    t2 = means.get('10100', pd.Series(0.5, index=means.index))
    t3 = means.get('10010', pd.Series(0.5, index=means.index))
    t4 = means.get('10001', pd.Series(0.5, index=means.index))
    
    # The expected theoretical ranking of P(Choose B) under WADD is t1 < t2 < t3 < t4.
    weights = np.array([-3, -1, 1, 3])
    
    corrs = []
    for subj in means.index:
        p_vals = np.array([t1[subj], t2[subj], t3[subj], t4[subj]])
        # Compute Pearson correlation to standardize the magnitude of the effect.
        # This heavily reduces between-subject variance in the WADD group caused by 
        # parameter differences, while maintaining an expected value of 0 for Strategy Mixture.
        if np.std(p_vals) > 1e-5:
            r = np.corrcoef(weights, p_vals)[0, 1]
            corrs.append(r)
        else:
            corrs.append(0.0)
            
    return float(np.mean(corrs))
outcome: self_sim=0.3417 (var=0.2454) adversary_sim=0.1551 (var=0.2735) welch_t=+1.296 p=0.2013 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    a_str = data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    df = pd.DataFrame({\n        'subject_id': data['subject_id'],\n        'A_str': a_str,\n        'response': data['response']\n    })\n    \n    # Strategy Mixture strictly predicts identical probabilities for Trials 1 & 7, and Trials 2 & 8.\n    # T1 & T7: TTB predicts Option A, Tallying predicts Option A.\n    # T2 & T8: TTB predicts Option B, Tallying predicts Option B.\n    # WADD with non-linear scaling strongly differentiates these pairs based on specific cue validities.\n    pairs = [\n        ('1000111', '1010101'), # T1 vs T7\n        ('0101010', '0111000')  # T8 vs T2\n    ]\n    \n    scores = []\n    for subj, grp in df.groupby('subject_id'):\n        subj_score = 0\n        for s_a, s_b in pairs:\n            ra = grp[grp['A_str'] == s_a]['response'].values\n            rb = grp[grp['A_str'] == s_b]['response'].values\n            if len(ra) >= 2 and len(rb) >= 2:\n                # Split-half cross-product provides an unbiased estimator of the squared difference\n                # in true choice probabilities. Under Strategy Mixture, expected value is exactly 0.\n                # Under WADD, the expected value is strictly positive.\n                ra_even, ra_odd = ra[::2].mean(), ra[1::2].mean()\n                rb_even, rb_odd = rb[::2].mean(), rb[1::2].mean()\n                subj_score += (ra_even - rb_even) * (ra_odd - rb_odd)\n        scores.append(subj_score)\n        \n    return float(np.mean(scores))",
  "rationale": "Previous metrics failed because they incorrectly grouped trials that did not actually form equivalence classes under the Strategy Mixture model (e.g., Trial 3 is a tie under Tallying, while Trial 1 is a win for Option A, meaning Mixture does NOT predict identical probabilities for them). By restricting the analysis to the TRUE equivalence classes (Trial 1 vs Trial 7, and Trial 2 vs Trial 8), the Strategy Mixture model strictly predicts an expected difference of exactly zero. We then use a split-half cross-product to compute an unbiased estimator of the squared probability difference. This completely eliminates binomial noise bias, resulting in a tight distribution around 0 for the Strategy Mixture model, while WADD yields a robustly positive score due to its sensitivity to specific lower-tier cue validities."
}
```

## Usage

```json
{
  "prompt_token_count": 9430,
  "candidates_token_count": 705,
  "total_token_count": 17329
}
```
