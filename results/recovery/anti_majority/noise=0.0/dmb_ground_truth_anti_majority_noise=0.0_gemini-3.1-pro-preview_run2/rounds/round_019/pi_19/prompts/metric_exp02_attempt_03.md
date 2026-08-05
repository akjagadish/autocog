# metric_exp02_attempt_03

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
**Validities (n_features=5):** [0.95, 0.7, 0.65, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the 'Evidence Averaging Heuristic Theory' (Advocated) from the 'Shared-Feature Cancellation and Evidence Averaging' theory (Competing) by exploiting a paradoxical preference reversal driven by a strong shared cue. The Competing theory assumes that decision-makers first cancel out any features shared by both options before averaging the remaining unique features. Therefore, adding a shared strong cue to both options leaves their relative evaluations completely unchanged. In contrast, the Advocated theory averages ALL active features. Because the two options have a different number of unique features, adding a shared strong cue pulls the average of the option with fewer unique features up much faster than the option with more unique features. In Trials 2 and 4 (no shared cues), both theories agree on preferring Option A because its unique features have a higher average validity. However, in Trials 1 and 3, when a highly valid shared cue (0.95) is added to both options, the Competing theory continues to strictly prefer Option A (since the shared cue cancels out), whereas the Advocated theory predicts a strict preference reversal to Option B, because Option A's average is weighed down by its larger number of unique weaker cues.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers evaluate options based on the relative average validity of their active cues, rather than the simple sum. A base prior is included to prevent zero-division and establish a baseline expectation. This average-based integration naturally predicts the 'less-is-more' effect, as adding weak cues dilutes the overall average evidence of an option, while shared cues have a context-dependent impact by simultaneously altering the numerator and denominator.

**Parameters:**
- lambda_val: [0.1, 10.0]
- prior_count: [0.01, 10.0]
- prior_sum: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    # Calculate average validity with a base prior to prevent zero-division
    v_a = (np.sum(w * a) + prior_sum) / (n_a + prior_count)
    v_b = (np.sum(w * b) + prior_sum) / (n_b + prior_count)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Shared-Feature Cancellation and Evidence Averaging: Decision-makers evaluate options by first identifying and canceling out features that are shared (active) in both options. After removing these shared elements to simplify the task, they evaluate each option by averaging the validities of its remaining unique active cues. A base prior is included in the average to establish a baseline expectation, but its lower bounds are extended to exactly zero to allow subjects to completely turn off the prior if they rely purely on the raw average of unique cues.

**Parameters:**
- lambda_val: [0.1, 10.0]
- prior_count: [0.0, 10.0]
- prior_sum: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    # Shared-Feature Cancellation
    shared = a * b
    a_unique = a - shared
    b_unique = b - shared
    
    n_a = np.sum(a_unique)
    n_b = np.sum(b_unique)
    
    # Calculate average validity of unique features with a base prior (safeguarded against div-by-zero)
    v_a = (np.sum(w * a_unique) + prior_sum) / max(n_a + prior_count, 1e-6)
    v_b = (np.sum(w * b_unique) + prior_sum) / max(n_b + prior_count, 1e-6)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
[0] rationale: The Competing theory (Shared-Feature Cancellation) predicts that choices in Trial 1 and Trial 4 should be identical, because Trial 4 simply adds shared features to both options which are immediately canceled out, leaving the exact same unique features as Trial 1. Therefore, the difference in choice proportions between these trials should be zero. In contrast, the Advocated theory (Evidence Averaging) predicts that adding shared features inherently changes the averages of both options, typically compressing the difference between them. Thus, the choice probability for the preferred option should regress toward 0.5 in Trial 4 compared to Trial 1, leading to a strictly positive difference in the mean response.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Count the number of active features in option A to identify the trial type
    a_sums = data['option_a_ratings'].apply(sum)
    
    # Trial 1 has 1 active feature in A; Trial 4 has 3 active features in A
    t1_mask = a_sums == 1
    t4_mask = a_sums == 3
    
    # Calculate the difference in the proportion of choosing B (response == 1)
    # between Trial 4 and Trial 1
    return float(data.loc[t4_mask, 'response'].mean() - data.loc[t1_mask, 'response'].mean())
outcome: self_sim=0.0617 (var=0.0136) adversary_sim=0.0350 (var=0.0144) welch_t=+0.796 p=0.4297 (N=25, alpha=0.01) -> reject

[1] rationale: Under the Competing theory (Shared-Feature Cancellation), when Feature 5 is shared by both options, it is immediately canceled out. Therefore, the choice probabilities should be identical whether Feature 5 is present (Trials 2 and 4) or absent (Trials 1 and 3). The expected difference is exactly 0. Under the Advocated theory (Evidence Averaging), adding a weak shared feature dilutes the average validity of the options. Because Option A generally relies on fewer but stronger features, adding a weak feature dilutes Option A's evidence more severely than Option B's, reducing Option A's advantage and increasing the probability of choosing Option B. By comparing the main effect of this shared feature across all trials, we maximize statistical power (using 96 trials per subject) while isolating the exact mechanism of divergence.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Feature 5 (index 4) is the weakest feature.
    # In Trials 2 and 4, it is present in both options (shared).
    # In Trials 1 and 3, it is absent from both options.
    has_f5 = data['option_a_ratings'].apply(lambda x: x[4] == 1)
    
    # Calculate the proportion of times Option B was chosen when F5 is present vs absent.
    p_b_with_f5 = data.loc[has_f5, 'response'].mean()
    p_b_without_f5 = data.loc[~has_f5, 'response'].mean()
    
    return float(p_b_with_f5 - p_b_without_f5)
outcome: self_sim=0.0262 (var=0.0084) adversary_sim=0.0075 (var=0.0082) welch_t=+0.727 p=0.4707 (N=25, alpha=0.01) -> reject

[2] rationale: Under the Competing theory (Shared-Feature Cancellation), the unique features evaluated by the subject are completely identical across all 4 trial types. Consequently, the true probability of choosing Option B is exactly the same across all 96 trials for a given subject. Any variation in the observed proportion of B choices across the 4 trial types is purely due to binomial sampling noise. Under the Advocated theory (Evidence Averaging), however, the true choice probabilities systematically vary across the 4 trial types because adding shared features alters the average evidence of each option differently.

We can elegantly distinguish these theories by computing the variance of the 4 trial-type means for each subject, and subtracting the analytically expected variance due to binomial noise (`p * (1 - p) * 3 / 95`). Under the Competing theory, this 'excess variance' has an expected value of exactly 0, and its variance across subjects is extremely tight. Under the Advocated theory, the excess variance will be systematically positive due to the true underlying variation in choice probabilities. This metric perfectly isolates the theoretical divergence while structurally eliminating baseline preference noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def calc_excess_var(subj_df):
        # Create a unique string key for each trial type based on Option A's features
        keys = subj_df['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
        
        # Calculate the mean response for each of the 4 trial types
        trial_means = subj_df.groupby(keys)['response'].mean()
        
        # Overall proportion of choosing B for this subject
        p = subj_df['response'].mean()
        
        # Observed variance of the 4 trial means (ddof=0)
        obs_var = np.var(trial_means, ddof=0)
        
        # Expected variance purely due to binomial noise.
        # With 96 trials total and 4 groups of 24, the exact expected sample variance 
        # of the 4 means under the null hypothesis (constant p) is exactly:
        exp_var = p * (1.0 - p) * 3.0 / 95.0
        
        # Return the excess variance above binomial noise
        return float(obs_var - exp_var)

    # Average the excess variance across all subjects
    return float(data.groupby('subject_id').apply(calc_excess_var).mean())
outcome: self_sim=-0.0000 (var=0.0000) adversary_sim=-0.0004 (var=0.0000) welch_t=+0.279 p=0.7817 (N=25, alpha=0.01) -> reject

[3] rationale: Under the Competing theory (Shared-Feature Cancellation), shared features are completely ignored. Since Option A's unique feature is much stronger than Option B's unique features across all trials, the choice is consistently easy, leading to a very low overall probability of choosing B. Under the Advocated theory (Evidence Averaging), the shared features present in Trials 2, 3, and 4 are averaged into the options' values. Because these shared features are weaker than A's unique feature, they systematically dilute Option A's strong advantage, pulling the subjective values of A and B closer together. This makes the choice more difficult and systematically increases the overall probability of choosing B. By using the overall mean response across all 96 trials, we capture this massive one-sided main effect while strictly minimizing the binomial sampling noise that plagued trial-by-trial contrasts.
metric_source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
outcome: self_sim=0.2681 (var=0.0210) adversary_sim=0.2971 (var=0.0263) welch_t=-0.666 p=0.5088 (N=25, alpha=0.01) -> reject

[4] rationale: Under the Competing theory (Shared-Feature Cancellation), Trial 1 and Trial 4 are mathematically identical because the shared features in Trial 4 are completely ignored. Thus, the true choice probability is exactly the same, and the absolute difference |p4 - p1| purely reflects binomial sampling noise. Under the Advocated theory (Evidence Averaging), the shared features in Trial 4 are averaged in, which systematically alters the choice probabilities compared to Trial 1. Depending on the subject's specific parameters (e.g., prior vs. lambda), this can either increase or decrease the probability of choosing B. By taking the *absolute* difference per subject before averaging, we prevent these positive and negative shifts from canceling each other out, ensuring a strictly larger expected value under the Advocated theory than the binomial noise baseline predicted by the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subj_metric(subj_df):
        # Option A has 1 active feature in Trial 1, and 3 active features in Trial 4
        a_sums = subj_df['option_a_ratings'].apply(sum)
        
        # Extract the proportion of choosing Option B for Trial 1 and Trial 4
        p1 = subj_df.loc[a_sums == 1, 'response'].mean()
        p4 = subj_df.loc[a_sums == 3, 'response'].mean()
        
        # Return the absolute difference to prevent cancellation across subjects with different parameters
        return abs(p4 - p1)
        
    return float(data.groupby('subject_id').apply(subj_metric).mean())
outcome: self_sim=0.1225 (var=0.0066) adversary_sim=0.0750 (var=0.0046) welch_t=+2.243 p=0.02969 (N=25, alpha=0.01) -> reject

[5] rationale: Under the Competing theory (Shared-Feature Cancellation), the unique features evaluated are mathematically identical across all 4 trial types. Thus, the true choice probability is exactly the same, and any difference between trials is purely binomial sampling noise. Under the Advocated theory (Evidence Averaging), adding shared features in Trials 2, 3, and 4 systematically alters the choice probabilities by diluting Option A's average more heavily than Option B's. By pooling the three trials with shared features (72 trials) and comparing them to the isolated trial (24 trials), we capture this systematic shift while reducing the variance of the binomial noise baseline by 33% compared to a single-trial contrast (Trial 4 vs Trial 1). Taking the absolute difference per subject ensures that bidirectional shifts (depending on specific parameter values like priors) do not cancel out, resulting in a strictly higher expected value under the Advocated theory than the tightly bounded noise baseline of the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subj_metric(subj_df):
        # Trial 1 has 1 active feature for A. Trials 2, 3, 4 have >1 active features.
        a_sums = subj_df['option_a_ratings'].apply(sum)
        
        # Proportion of B choices for trials with shared features vs no shared features
        p_shared = subj_df.loc[a_sums > 1, 'response'].mean()
        p_isolated = subj_df.loc[a_sums == 1, 'response'].mean()
        
        # Return absolute difference to prevent cancellation across subjects
        return abs(p_shared - p_isolated)
        
    return float(data.groupby('subject_id').apply(subj_metric).mean())
outcome: self_sim=0.0783 (var=0.0039) adversary_sim=0.0669 (var=0.0029) welch_t=+0.691 p=0.4928 (N=25, alpha=0.01) -> reject

[6] rationale: Under the Competing theory (Shared-Feature Cancellation), the true choice probability is identical across all 4 trial types because shared features are ignored. Any observed difference between trials is purely binomial sampling noise. Under the Advocated theory (Evidence Averaging), adding shared features dilutes the options' values, pulling them closer together and causing the choice probability to regress toward 0.5. Crucially, the direction of this shift depends on the subject's baseline preference: if they prefer A, the probability of choosing B increases; if they prefer B, it decreases.

Previous attempts using `p4 - p1` suffered from cancellation across subjects, while `abs(p4 - p1)` introduced a large positive bias under the Competing theory due to rectified binomial noise. We solve this by multiplying the shift `(p4 - p1)` by an independent measure of baseline preference: `(0.5 - p23)`. Under the Competing theory, `p23` is strictly independent of `p4` and `p1`, so the expected value of this product is exactly 0, and its variance is heavily suppressed. Under the Advocated theory, `(0.5 - p23)` perfectly aligns with the sign of `(p4 - p1)`, resulting in a strictly positive metric. This isolates the theoretical divergence with zero baseline bias and exceptionally high signal-to-noise ratio.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subj_metric(subj_df):
        # Option A has 1 active feature in T1, 2 in T2/T3, and 3 in T4
        a_sums = subj_df['option_a_ratings'].apply(sum)
        
        # Calculate the proportion of B choices for each dilution level
        p1 = subj_df.loc[a_sums == 1, 'response'].mean()
        p23 = subj_df.loc[a_sums == 2, 'response'].mean()
        p4 = subj_df.loc[a_sums == 3, 'response'].mean()
        
        # Weight by baseline preference using the independent T2/T3 trials
        w = 0.5 - p23
        
        # Return the preference-aligned shift
        return float(w * (p4 - p1))
        
    return float(data.groupby('subject_id').apply(subj_metric).mean())
outcome: self_sim=0.0137 (var=0.0008) adversary_sim=-0.0006 (var=0.0005) welch_t=+1.962 p=0.05592 (N=25, alpha=0.01) -> reject

[7] rationale: Under the Competing theory (Cancellation), Trial 1 and Trial 4 evaluate identical unique features, so the true choice probability is exactly the same and the expected difference is zero. Under the Advocated theory (Averaging), adding shared weak features in Trial 4 systematically dilutes Option A's average more than Option B's, causing a shift in choice probability. Previous linear contrasts (p4 - p1) suffered because baseline probabilities were often pushed near 0 or 1 by extreme parameters, heavily compressing the observable shift. By applying the arcsine square root transformation before taking the difference, we decompress the signal at the extremes and perfectly stabilize the binomial noise variance across subjects. This maximizes the signal-to-noise ratio of the systematic preference shift predicted by Evidence Averaging.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    def subj_metric(subj_df):
        a_sums = subj_df['option_a_ratings'].apply(sum)
        # T1 has 1 active feature for A, T4 has 3 active features
        p1 = subj_df.loc[a_sums == 1, 'response'].mean()
        p4 = subj_df.loc[a_sums == 3, 'response'].mean()
        # Apply arcsine square root transformation to stabilize variance 
        # and stretch probabilities near 0/1 where the signal is compressed.
        return float(np.arcsin(np.sqrt(p4)) - np.arcsin(np.sqrt(p1)))
    return float(data.groupby('subject_id').apply(subj_metric).mean())
outcome: self_sim=0.0655 (var=0.0252) adversary_sim=0.0131 (var=0.0145) welch_t=+1.314 p=0.1954 (N=25, alpha=0.01) -> reject

[8] rationale: Under the Competing theory (Shared-Feature Cancellation), the strong shared cue in Trials 1 and 3 is completely canceled out, leaving the exact same unique features as in Trials 2 and 4. Therefore, the choice probabilities should be identical across these pairs, and the expected difference is exactly 0. Under the Advocated theory (Evidence Averaging), adding the highly valid shared cue (0.95) to both options increases the average validity of Option B more than Option A, because Option B has fewer unique features diluting the average. This leads to a systematic shift toward choosing Option B in Trials 1 and 3 compared to Trials 2 and 4. By computing the mean difference in choice proportions within each subject, we perfectly isolate this systematic shift predicted by the Advocated theory while maintaining an expected value of 0 and very tight variance under the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subj_metric(subj_df):
        # Trials 1 and 3 have 3 active features in A (including the strong shared feature 0)
        # Trials 2 and 4 have 2 active features in A (no shared features)
        a_sums = subj_df['option_a_ratings'].apply(sum)
        
        p_b_shared = subj_df.loc[a_sums == 3, 'response'].mean()
        p_b_noshared = subj_df.loc[a_sums == 2, 'response'].mean()
        
        # Under Competing theory, the shared feature cancels out, so the difference is 0.
        # Under Advocated theory, the shared strong feature pulls B's average up faster than A's,
        # leading to a higher probability of choosing B in the shared feature trials.
        return float(p_b_shared - p_b_noshared)
        
    return float(data.groupby('subject_id').apply(subj_metric).mean())
outcome: self_sim=0.0233 (var=0.0098) adversary_sim=-0.0025 (var=0.0076) welch_t=+0.978 p=0.333 (N=25, alpha=0.01) -> reject

[9] rationale: Under the Competing theory (Shared-Feature Cancellation), Trial 3 and Trial 4 are mathematically identical because the strong shared feature in Trial 3 is canceled out, leaving the exact same unique features as Trial 4. Thus, the expected difference in choice probabilities is exactly 0. Under the Advocated theory (Evidence Averaging), however, Trial 3 represents a critical tipping point: the addition of the highly valid shared feature (0.95) pulls Option B's average up much faster than Option A's (since B has fewer unique features), predicting a robust reversal toward Option B in Trial 3 compared to Trial 4. By isolating this specific pair of trials rather than averaging across all shared-feature trials (some of which do not produce a reversal), we maximize the theoretical divergence.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subj_metric(subj_df):
        # Trial 3: A=[1, 0, 1, 1, 0] (x[3]==1 and x[0]==1)
        # Trial 4: A=[0, 0, 1, 1, 0] (x[3]==1 and x[0]==0)
        t3_mask = subj_df['option_a_ratings'].apply(lambda x: x[3] == 1 and x[0] == 1)
        t4_mask = subj_df['option_a_ratings'].apply(lambda x: x[3] == 1 and x[0] == 0)
        
        p3 = subj_df.loc[t3_mask, 'response'].mean()
        p4 = subj_df.loc[t4_mask, 'response'].mean()
        
        return float(p3 - p4)
        
    return float(data.groupby('subject_id').apply(subj_metric).mean())
outcome: self_sim=0.0117 (var=0.0155) adversary_sim=0.0092 (var=0.0165) welch_t=+0.070 p=0.9446 (N=25, alpha=0.01) -> reject

[10] rationale: Under the Competing theory (Shared-Feature Cancellation), Trial 1 and Trial 2 evaluate identically because their unique features are exactly the same; the same is true for Trial 3 and Trial 4. Therefore, the true underlying choice probabilities should be identical within these pairs (p1=p2 and p3=p4). A previous attempt incorrectly assumed that all four trials had the same choice probability under Cancellation, which failed because the unique features in T1/T2 are different from T3/T4. By computing the squared difference within the mathematically identical pairs and subtracting the exact unbiased estimator of the binomial sampling variance, we create a metric whose expected value under the Competing theory is strictly zero. Under the Advocated theory (Evidence Averaging), the shared features systematically alter the choice probabilities, leading to a strictly positive expected excess variance. This cleanly isolates the theoretical divergence with zero baseline bias.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subj_metric(subj_df):
        a_ratings = subj_df['option_a_ratings'].tolist()
        responses = subj_df['response'].tolist()
        
        r1, r2, r3, r4 = [], [], [], []
        for a, r in zip(a_ratings, responses):
            if a[0] == 1 and a[1] == 1:
                r1.append(r)
            elif a[0] == 0 and a[1] == 1:
                r2.append(r)
            elif a[0] == 1 and a[3] == 1:
                r3.append(r)
            elif a[0] == 0 and a[3] == 1:
                r4.append(r)
                
        def get_excess(ra, rb):
            if len(ra) < 2 or len(rb) < 2:
                return 0.0
            pa = sum(ra) / len(ra)
            pb = sum(rb) / len(rb)
            vara = pa * (1.0 - pa) / (len(ra) - 1)
            varb = pb * (1.0 - pb) / (len(rb) - 1)
            return (pa - pb)**2 - vara - varb
            
        return get_excess(r1, r2) + get_excess(r3, r4)
        
    return float(data.groupby('subject_id').apply(subj_metric).mean())
outcome: self_sim=0.0003 (var=0.0017) adversary_sim=-0.0015 (var=0.0014) welch_t=+0.165 p=0.8693 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def subj_metric(subj_df):\n        a_ratings = subj_df['option_a_ratings'].tolist()\n        responses = subj_df['response'].tolist()\n        \n        r1, r2, r3, r4 = [], [], [], []\n        for a, r in zip(a_ratings, responses):\n            if a[0] == 1 and a[1] == 1:\n                r1.append(r)\n            elif a[0] == 0 and a[1] == 1:\n                r2.append(r)\n            elif a[0] == 1 and a[3] == 1:\n                r3.append(r)\n            elif a[0] == 0 and a[3] == 1:\n                r4.append(r)\n                \n        def get_chi2(ra, rb):\n            if len(ra) == 0 or len(rb) == 0:\n                return 0.0\n            pa = sum(ra) / len(ra)\n            pb = sum(rb) / len(rb)\n            pooled_p = (sum(ra) + sum(rb)) / (len(ra) + len(rb))\n            if pooled_p <= 0.0 or pooled_p >= 1.0:\n                return 0.0\n            \n            multiplier = (len(ra) * len(rb)) / (len(ra) + len(rb))\n            return multiplier * ((pa - pb)**2) / (pooled_p * (1.0 - pooled_p))\n            \n        return get_chi2(r1, r2) + get_chi2(r3, r4)\n\n    return float(data.groupby('subject_id').apply(subj_metric).mean())",
  "rationale": "Under the Competing theory (Shared-Feature Cancellation), Trial 1 and Trial 2 evaluate identically because their unique features are exactly the same; the same applies to Trial 3 and Trial 4. Therefore, any observed differences in choice proportions within these pairs are purely due to binomial sampling noise. By computing the standard Pearson Chi-square statistic for independence on these two pairs, the expected value under the Competing theory is strictly defined by the degrees of freedom (expected value of 2, with tight variance). Under the Advocated theory (Evidence Averaging), the presence of the shared feature in Trials 1 and 3 systematically alters the true choice probabilities compared to Trials 2 and 4. This introduces a non-centrality parameter to the Chi-square distribution. Because the Chi-square statistic standardizes the squared difference by the binomial variance, it optimally accumulates the signal across subjects regardless of whether their specific parameters cause the choice probability to increase or decrease, rigorously distinguishing the theories while perfectly controlling for baseline noise."
}
```

## Usage

```json
{
  "prompt_token_count": 8741,
  "candidates_token_count": 655,
  "total_token_count": 15847
}
```
