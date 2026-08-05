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
**Validities (n_features=6):** [0.98, 0.75, 0.7, 0.65, 0.6, 0.55]

**Trial pairs (n=14):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 12: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 13: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** This design quantitatively and qualitatively dissociates the advocated LCA model with Configural Weighting from the competing Tally-Difference Modulated Feature Differencing theory by exploiting a paradoxical non-monotonicity in the competing theory. The competing theory posits that a large tally difference triggers a shift toward exponential validity weighting, while a small tally difference triggers equal-weighting (tallying). By pitting a single extremely high-validity cue for Option A against an increasing number of lower-validity cues for Option B, we systematically increase the tally difference in favor of B. The competing theory predicts a bizarre reversal: when B has slightly more cues (e.g., 2 vs 1), the small tally difference triggers tallying, causing B to be strongly preferred. But as B gains even MORE cues (e.g., 5 vs 1), the large tally difference triggers validity weighting, and the exponential scaling of A's top cue causes the model to suddenly reverse its preference back to Option A. The advocated LCA model, which integrates evidence using configural weighting and a non-linear transducer, predicts that adding more supportive cues to Option B will monotonically increase its accumulated evidence, strictly predicting no such reversal back to Option A.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



## ADVOCATED THEORY
**Description:** Leaky Competing Accumulator with Non-linear Configural Weighting and Evidence Transduction: Decision-makers evaluate cues simultaneously, with evidence for each option dynamically inhibiting the other in a leaky competing accumulator (LCA). Cues have a configural impact, scaled non-linearly by the total number of supporting cues. Additionally, the integrated evidence for each option is passed through a non-linear transducer (alpha) before entering the accumulation process, allowing the model to amplify the differences driven by high-validity cues and capture strong non-compensatory reversals.

**Parameters:**
- gamma: [-10.0, 10.0]
- leak: [0.1, 2.0]
- inhibition: [0.0, 5.0]
- theta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- alpha: [0.1, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    # Configural weighting: effective validity of a cue is non-linearly modulated by the total number of cues
    # Using max(1e-6, sum) to avoid 0^negative_gamma undefined errors
    sum_a_safe = max(1e-6, sum_a)
    sum_b_safe = max(1e-6, sum_b)
    
    w_a = val * (sum_a_safe ** gamma)
    w_b = val * (sum_b_safe ** gamma)
    
    # Make sure inputs are non-negative and apply non-linear transducer alpha
    I_A = max(0.0, np.sum(w_a * a)) ** alpha
    I_B = max(0.0, np.sum(w_b * b)) ** alpha
    
    # Leaky Competing Accumulator (LCA) simulation
    x_a, x_b = 0.0, 0.0
    dt = 0.1
    steps = 100
    
    for _ in range(steps):
        dx_a = (I_A - leak * x_a - inhibition * x_b) * dt
        dx_b = (I_B - leak * x_b - inhibition * x_a) * dt
        
        x_a = max(0.0, x_a + dx_a)
        x_b = max(0.0, x_b + dx_b)
        
    # Softmax choice based on final activations
    z = theta * np.array([x_a, x_b])
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
**Description:** Tally-Difference Modulated Feature Differencing with Exponential Validity Scaling: Decision-makers evaluate options in parallel by computing feature differences, but the weights applied to these differences are dynamically modulated by the aggregate conflict between the options. When the tally difference is small (high conflict), decision-makers shift toward simple equal-weighting (tallying). When the tally difference is large (low conflict), they rely on validity weights. To capture non-compensatory (Take-The-Best) behavior during validity-weighted decisions, the validities are scaled exponentially, allowing the single best cue to mathematically dominate the sum of lesser cues without squashing the evidence scores.

**Parameters:**
- gamma: [0.0, 10.0]
- threshold: [0.0, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- kappa: [0.0, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    kappa = float(parameters["kappa"])
    
    # Calculate conflict based on the absolute difference in simple tallies
    tally_diff = abs(np.sum(a) - np.sum(b))
    
    # Dynamic weight interpolation:
    # High tally difference -> validity weighting (p_val close to 1)
    # Low tally difference -> equal weighting / tallying (p_val close to 0)
    p_val = 1.0 / (1.0 + np.exp(-gamma * (tally_diff - threshold)))
    
    # Exponential scaling of validities to allow non-compensatory (lexicographic) dominance
    transformed_val = np.exp(kappa * val)
    
    # Tallying uses equal weights (1.0 for each feature)
    w = p_val * transformed_val + (1.0 - p_val) * 1.0
    
    # Parallel evaluation of options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    
    # Softmax choice
    z = beta * np.array([score_a, score_b])
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
[0] rationale: The experimental design groups trials such that the feature differences between Option A and Option B remain exactly identical while the number of tied cues systematically increases. Under the competing theory (Tally-Difference Modulated Feature Differencing), choice probabilities are strictly determined by tally differences and feature differences. Because tied cues alter neither, the competing theory predicts that choice probabilities will remain perfectly invariant within these trial groups; any observed variance should be exclusively due to binomial noise. In contrast, the advocated theory (LCA with Configural Weighting) predicts that adding tied cues non-linearly modulates the effective weights of the discriminating cues. Thus, choice probabilities will systematically shift as tied cues are added. By computing the variance of choice probabilities across trials that share the same feature differences but differ in tied cues, this metric will be significantly higher for the advocated theory than for the competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    df = data.copy()
    
    # Identify trial groups by their feature difference
    df['diff_key'] = df.apply(
        lambda row: tuple(a - b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), 
        axis=1
    )
    
    # Uniquely identify each trial by its exact feature patterns
    df['trial_key'] = df.apply(
        lambda row: (tuple(row['option_a_ratings']), tuple(row['option_b_ratings'])),
        axis=1
    )
    
    # Calculate the choice probability for Option B per trial
    trial_pb = df.groupby(['diff_key', 'trial_key'])['response'].mean().reset_index()
    
    # Calculate the variance of choice probabilities within each feature-difference group
    var_list = []
    for diff, group in trial_pb.groupby('diff_key'):
        if len(group) > 1:
            var_list.append(np.var(group['response'], ddof=0))
            
    return float(np.mean(var_list)) if var_list else 0.0
outcome: self_sim=0.0041 (var=0.0005) adversary_sim=0.0004 (var=0.0001) welch_t=+0.796 p=0.4321 (N=25, alpha=0.01) -> reject

[1] rationale: The experimental design groups trials such that the feature differences between Option A and Option B remain exactly identical while the number of tied cues systematically increases. Under the competing theory, choice probabilities are strictly determined by tally differences and feature differences, both of which remain constant within these groups. Thus, the competing theory predicts perfectly invariant choice probabilities within these groups (a range of 0, plus binomial noise). In contrast, the advocated theory predicts that adding tied cues non-linearly scales the effective weights of the discriminating cues, producing systematic shifts in choice probabilities. By computing the mean range (max - min) of choice probabilities within each feature-difference group, we capture the full extent of this shift, which will be significantly larger for the advocated theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify trial groups by their exact feature difference
    data['diff_key'] = data.apply(
        lambda row: tuple(a - b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), 
        axis=1
    )
    
    # Uniquely identify each trial type
    data['trial_key'] = data.apply(
        lambda row: (tuple(row['option_a_ratings']), tuple(row['option_b_ratings'])),
        axis=1
    )
    
    # Calculate the choice probability for Option B per trial type
    trial_pb = data.groupby(['diff_key', 'trial_key'])['response'].mean().reset_index()
    
    # Calculate the maximum shift (range) of choice probabilities within each feature-difference group
    ranges = []
    for diff, group in trial_pb.groupby('diff_key'):
        if len(group) > 1:
            ranges.append(group['response'].max() - group['response'].min())
            
    return float(np.mean(ranges)) if ranges else 0.0
outcome: self_sim=0.1542 (var=0.0380) adversary_sim=0.0433 (var=0.0088) welch_t=+2.563 p=0.01488 (N=25, alpha=0.01) -> reject

[2] rationale: The experimental design groups trials by identical feature differences while systematically increasing the number of tied cues. The competing theory (Tally-Difference Modulated Feature Differencing) predicts choice probabilities are perfectly invariant to tied cues, meaning the true shift in probability is exactly zero. The advocated theory (LCA with Configural Weighting) predicts systematic shifts because tied cues non-linearly scale the effective weights. Previous attempts using variance or range of probabilities were heavily biased by binomial noise, leading to small effect sizes. This metric computes the shift between 'low' (<2) and 'high' (>=2) tied cues, but evaluates it using a cross-product of independent halves of the trials. This acts as an unbiased estimator of the squared shift: its expected value is exactly 0 under the competing theory (canceling out noise), but strongly positive under the advocated theory, providing a much cleaner, tighter distribution per subject.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    df = data.copy()
    
    # Identify trial groups by their exact feature difference
    df['diff_key'] = df.apply(
        lambda row: tuple(a - b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), 
        axis=1
    )
    
    # Count the number of tied cues (where both options have a '1')
    df['tied_cues'] = df.apply(
        lambda row: sum(a == 1 and b == 1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])),
        axis=1
    )
    
    # Split into low and high tied cues to measure the shift
    df['is_high'] = df['tied_cues'] >= 2
    
    # To avoid positive bias from noise when squaring the shift, we use a cross-product 
    # of independent halves of the data. This creates an unbiased estimator of the squared shift.
    df['half'] = df.groupby(['subject_id', 'diff_key', 'is_high']).cumcount() % 2
    
    # Compute mean response for each condition and half
    means = df.groupby(['subject_id', 'diff_key', 'is_high', 'half'])['response'].mean().unstack(['is_high', 'half'])
    means = means.dropna()
    
    if len(means) == 0:
        return 0.0
        
    # Calculate the shift in choice probability between high and low tied cues for each half
    shift_0 = means[(True, 0)] - means[(False, 0)]
    shift_1 = means[(True, 1)] - means[(False, 1)]
    
    # The cross-product of independent estimates is an unbiased estimator of the true squared shift
    squared_shifts = shift_0 * shift_1
    
    return float(squared_shifts.mean())
outcome: self_sim=0.0084 (var=0.0014) adversary_sim=0.0023 (var=0.0008) welch_t=+0.640 p=0.5255 (N=25, alpha=0.01) -> reject

[3] rationale: The experimental design groups trials such that the feature differences between Option A and Option B remain exactly identical while the number of tied cues systematically increases. The competing theory (Tally-Difference Modulated Feature Differencing) predicts choice probabilities are perfectly invariant to tied cues. The advocated theory (LCA with Configural Weighting) predicts systematic shifts because tied cues non-linearly scale the effective weights. Because the direction of the shift depends on the subject's parameters (e.g., whether the configural parameter gamma is positive or negative), simple averaging cancels out the effect. Previous attempts using variances or squared shifts suffered from high noise and very small effect sizes. This metric uses a cross-validated estimator for the absolute shift: it computes the shift on two independent halves of the trials, and uses the sign of the first half to orient the second half. This estimator scales linearly with the true effect size (yielding a much larger signal than squared shifts) while mathematically guaranteeing an expected value of exactly 0 under the competing theory, completely eliminating the positive bias from binomial noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    df = data.copy()
    
    # Identify trial groups by their exact feature difference
    df['diff_key'] = df.apply(
        lambda row: tuple(a - b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), 
        axis=1
    )
    
    # Count the number of tied cues (where both options have a '1')
    df['tied_cues'] = df.apply(
        lambda row: sum(a == 1 and b == 1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])),
        axis=1
    )
    
    # Split into low and high tied cues
    df['is_high'] = df['tied_cues'] >= 2
    
    # Create two independent halves of the data to avoid noise-induced positive bias
    df['half'] = df.groupby(['subject_id', 'diff_key', 'is_high']).cumcount() % 2
    
    # Compute mean response for each condition and half
    means = df.groupby(['subject_id', 'diff_key', 'is_high', 'half'])['response'].mean().unstack(['is_high', 'half'])
    means = means.dropna()
    
    if len(means) == 0:
        return 0.0
        
    # Calculate the shift in choice probability between high and low tied cues for each half
    shift_0 = means[(True, 0)] - means[(False, 0)]
    shift_1 = means[(True, 1)] - means[(False, 1)]
    
    # Use the sign of the shift from the first half to orient the shift in the second half.
    # This acts as a cross-validated estimator of the absolute shift.
    # Under the competing theory, shift_true = 0, so shift_0 and shift_1 are independent noise,
    # making the expected value of this product EXACTLY 0.
    # Under the advocated theory, shift_true != 0, so the sign will align with shift_1,
    # yielding a strong positive expected value that scales linearly with the effect size.
    cv_abs_shift = np.sign(shift_0) * shift_1
    
    return float(cv_abs_shift.mean())
outcome: self_sim=0.0156 (var=0.0170) adversary_sim=0.0067 (var=0.0095) welch_t=+0.273 p=0.7859 (N=25, alpha=0.01) -> reject

[4] rationale: The experimental design uses pairs of trials where the tally difference and the effective score difference remain exactly constant under the competing theory, because the added cues to Option A and Option B have exactly identical validities (0.75 or 0.65). Consequently, the competing theory predicts that choice probabilities will be perfectly invariant between Trial 1 and Trial 3, and between Trial 4 and Trial 6. In contrast, the advocated theory applies a configural non-linear weighting based on the total number of cues, meaning the effective weights scale differently as cues are added, predicting a systematic shift in choice probabilities. By calculating the shift in choice probabilities between the endpoints of these matched trial sequences on independent halves of the data, we create an unbiased estimator of the squared shift. This estimator has an expected value of exactly 0 under the competing theory (canceling out binomial noise), but is strongly positive under the advocated theory, providing a highly discriminative and low-variance metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify trials by the sum of features in A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    df = pd.DataFrame({
        'sum_a': sum_a,
        'sum_b': sum_b,
        'response': data['response']
    })
    
    # Split into two independent halves to compute an unbiased estimator of the squared shift
    if 'subject_id' in data.columns:
        df['subject_id'] = data['subject_id']
        df['half'] = df.groupby(['subject_id', 'sum_a', 'sum_b']).cumcount() % 2
    else:
        df['half'] = df.groupby(['sum_a', 'sum_b']).cumcount() % 2
        
    means = df.groupby(['sum_a', 'sum_b', 'half'])['response'].mean().to_dict()
    
    def get_p(sa, sb, h):
        return means.get((sa, sb, h), 0.5)
        
    # Group 1: Tally diff = 0
    # Shift between Trial 3 (sum A=3, sum B=3) and Trial 1 (sum A=1, sum B=1)
    shift_g1_h0 = get_p(3, 3, 0) - get_p(1, 1, 0)
    shift_g1_h1 = get_p(3, 3, 1) - get_p(1, 1, 1)
    
    # Group 2: Tally diff = 1
    # Shift between Trial 6 (sum A=3, sum B=4) and Trial 4 (sum A=1, sum B=2)
    shift_g2_h0 = get_p(3, 4, 0) - get_p(1, 2, 0)
    shift_g2_h1 = get_p(3, 4, 1) - get_p(1, 2, 1)
    
    # Unbiased estimator of the sum of squared shifts
    return float(shift_g1_h0 * shift_g1_h1 + shift_g2_h0 * shift_g2_h1)
outcome: self_sim=0.0355 (var=0.0274) adversary_sim=0.0005 (var=0.0034) welch_t=+0.996 p=0.3271 (N=25, alpha=0.01) -> reject

[5] rationale: Under the competing theory (Tally-Difference Modulated Feature Differencing), the choice probabilities are mathematically invariant within trials that share the same tally difference, because the added cues have exactly matched validities that cancel out linearly. Therefore, within a tally difference group, the response is perfectly independent of the total number of cues (`sum_a`). The G-statistic (likelihood ratio test for independence) for these trials will strictly follow a central chi-squared distribution, representing only baseline binomial noise (mean ≈ 4, low variance).

Under the advocated theory (LCA with Configural Weighting), the effective weights and integrated evidence scale non-linearly with the total number of cues (`sum_a ** gamma`). This causes systematic, graded shifts in choice probabilities as cues are added. Because the G-statistic accumulates any deviation from independence as a strictly positive value, it perfectly captures these shifts regardless of whether the subject's parameters cause the probabilities to increase or decrease. This provides a highly discriminative metric with a known, stable baseline noise floor, overcoming the high between-subject variance that plagued linear or squared-shift estimators.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Ensure we can process both the pooled DataFrame and individual subject slices
    if 'subject_id' not in data.columns:
        subjects = [data]
    else:
        subjects = [group for _, group in data.groupby('subject_id')]
        
    g_totals = []
    
    for df in subjects:
        g = 0.0
        df = df.copy()
        
        # The competing theory evaluates options by their tally difference.
        df['tally_diff'] = df.apply(
            lambda row: abs(sum(row['option_a_ratings']) - sum(row['option_b_ratings'])), 
            axis=1
        )
        
        # Within a constant tally difference, the number of tied cues systematically increases
        # which we can track by the sum of features in Option A.
        df['sum_a'] = df['option_a_ratings'].apply(sum)
        
        # For each tally difference group, test for independence between 'sum_a' and 'response'.
        # The competing theory predicts perfect independence (true probability is invariant).
        for td, group in df.groupby('tally_diff'):
            # Create contingency table of sum_a (rows) vs response (columns)
            O = pd.crosstab(group['sum_a'], group['response'])
            
            row_totals = O.sum(axis=1)
            col_totals = O.sum(axis=0)
            grand_total = O.values.sum()
            
            if grand_total == 0: 
                continue
                
            # Compute the G-test statistic (Likelihood Ratio test for independence)
            for i in O.index:
                for j in O.columns:
                    o_val = O.loc[i, j]
                    e_val = row_totals[i] * col_totals[j] / grand_total
                    if o_val > 0 and e_val > 0:
                        g += 2 * o_val * np.log(o_val / e_val)
                        
        g_totals.append(g)
        
    # Return the mean G-statistic across subjects
    return float(np.mean(g_totals))
outcome: self_sim=8.6729 (var=73.5860) adversary_sim=4.4350 (var=11.7072) welch_t=+2.294 p=0.0286 (N=25, alpha=0.01) -> reject

[6] rationale: Under the competing theory, choice probabilities are perfectly invariant across trials that share the same tally difference, meaning the true variance of these probabilities is exactly zero. Any observed variance is purely due to binomial sampling noise. Under the advocated theory, configural weighting systematically shifts the true choice probabilities as the total number of cues increases, resulting in a strictly positive true variance. By computing the sample variance of the empirical probabilities and subtracting the exact unbiased estimator of the binomial noise variance, we obtain a metric whose expected value is exactly 0 under the competing theory (completely eliminating noise-induced positive bias) and strictly positive under the advocated theory. This provides a highly stable, low-variance discriminator.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify trial groups by their tally difference
    data['tally_a'] = data['option_a_ratings'].apply(sum)
    data['tally_b'] = data['option_b_ratings'].apply(sum)
    data['tally_diff'] = abs(data['tally_a'] - data['tally_b'])
    
    if 'subject_id' not in data.columns:
        data['subject_id'] = 'dummy'
        
    true_variances = []
    
    for subj, subj_df in data.groupby('subject_id'):
        for diff_val, group in subj_df.groupby('tally_diff'):
            # Group by specific trial type (tracked by tally_a since tally_diff is constant)
            trial_stats = group.groupby('tally_a')['response'].agg(['count', 'sum'])
            
            # Filter out trial types with <= 1 observation to allow variance correction
            trial_stats = trial_stats[trial_stats['count'] > 1]
            
            k = len(trial_stats)
            if k < 2:
                continue
                
            n = trial_stats['count'].values
            x = trial_stats['sum'].values
            p_hat = x / n
            
            # Sample variance of the empirical probabilities (ddof=1)
            p_mean = np.mean(p_hat)
            s2 = np.sum((p_hat - p_mean)**2) / (k - 1)
            
            # Compute the exact unbiased correction for binomial noise
            # E[x(n-x)/(n^2(n-1))] = p(1-p)/n
            correction = np.mean(x * (n - x) / (n**2 * (n - 1)))
            
            # The difference is a mathematically unbiased estimator of the true variance of P(B)
            true_var = s2 - correction
            true_variances.append(true_var)
            
    if not true_variances:
        return 0.0
        
    return float(np.mean(true_variances))
outcome: self_sim=0.0137 (var=0.0006) adversary_sim=0.0008 (var=0.0001) welch_t=+2.490 p=0.01885 (N=25, alpha=0.01) -> reject

[7] rationale: Under the competing theory (Tally-Difference Modulated Feature Differencing), choice probabilities are perfectly invariant across trials that share the same tally difference (e.g., Trial 1 vs Trial 3, and Trial 4 vs Trial 6). Any observed difference is purely due to binomial sampling noise. Under the advocated theory (LCA with Configural Weighting), adding matched cues non-linearly scales the integrated evidence, causing systematic shifts in choice probability between the endpoints of these sequences (1 cue vs 3 cues). By computing the simple absolute difference in choice probabilities between the endpoints of each sequence, we capture the maximum predicted shift. This approach avoids the high noise variance associated with cross-validated squared estimators or multi-trial range estimators, providing a robust, stable metric that will be significantly larger for the advocated theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    if 'subject_id' not in data.columns:
        data = data.copy()
        data['subject_id'] = 'dummy'
        
    abs_diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        subj_df = subj_df.copy()
        # Calculate tally sum for Option A to identify the number of cues
        subj_df['sum_a'] = subj_df['option_a_ratings'].apply(sum)
        # Calculate tally difference to separate the two matched groups
        subj_df['tally_diff'] = subj_df.apply(
            lambda row: abs(sum(row['option_a_ratings']) - sum(row['option_b_ratings'])), 
            axis=1
        )
        
        # Group 1: Trials 1, 2, 3 (Tally difference = 0)
        g1 = subj_df[subj_df['tally_diff'] == 0]
        p1 = g1[g1['sum_a'] == 1]['response'].mean() # Trial 1
        p3 = g1[g1['sum_a'] == 3]['response'].mean() # Trial 3
        
        # Group 2: Trials 4, 5, 6 (Tally difference = 1)
        g2 = subj_df[subj_df['tally_diff'] == 1]
        p4 = g2[g2['sum_a'] == 1]['response'].mean() # Trial 4
        p6 = g2[g2['sum_a'] == 3]['response'].mean() # Trial 6
        
        total_diff = 0.0
        if pd.notna(p1) and pd.notna(p3):
            total_diff += abs(p3 - p1)
        if pd.notna(p4) and pd.notna(p6):
            total_diff += abs(p6 - p4)
            
        abs_diffs.append(total_diff)
        
    return float(np.mean(abs_diffs))
outcome: self_sim=0.3812 (var=0.0897) adversary_sim=0.2338 (var=0.0198) welch_t=+2.228 p=0.03256 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    if 'subject_id' not in data.columns:\n        data = data.copy()\n        data['subject_id'] = 'dummy'\n        \n    scores = []\n    for subj, subj_df in data.groupby('subject_id'):\n        subj_df = subj_df.copy()\n        subj_df['sum_a'] = subj_df['option_a_ratings'].apply(sum)\n        subj_df['sum_b'] = subj_df['option_b_ratings'].apply(sum)\n        subj_df['a0'] = subj_df['option_a_ratings'].apply(lambda x: x[0])\n        subj_df['b0'] = subj_df['option_b_ratings'].apply(lambda x: x[0])\n        \n        # Trials where A has top cue only\n        t_A1_B5 = subj_df[(subj_df['sum_a'] == 1) & (subj_df['sum_b'] == 5) & (subj_df['a0'] == 1)]\n        t_A1_B2 = subj_df[(subj_df['sum_a'] == 1) & (subj_df['sum_b'] == 2) & (subj_df['a0'] == 1)]\n        \n        # Trials where B has top cue only\n        t_A5_B1 = subj_df[(subj_df['sum_a'] == 5) & (subj_df['sum_b'] == 1) & (subj_df['b0'] == 1)]\n        t_A2_B1 = subj_df[(subj_df['sum_a'] == 2) & (subj_df['sum_b'] == 1) & (subj_df['b0'] == 1)]\n        \n        p_B_5 = t_A1_B5['response'].mean() if len(t_A1_B5) > 0 else 0.5\n        p_B_2 = t_A1_B2['response'].mean() if len(t_A1_B2) > 0 else 0.5\n        \n        p_A_5 = 1.0 - t_A5_B1['response'].mean() if len(t_A5_B1) > 0 else 0.5\n        p_A_2 = 1.0 - t_A2_B1['response'].mean() if len(t_A2_B1) > 0 else 0.5\n        \n        score = (p_B_5 - p_B_2) + (p_A_5 - p_A_2)\n        scores.append(score)\n        \n    return float(np.mean(scores))",
  "rationale": "The experimental design pits a single top-validity cue against an increasing number of lower-validity cues to expose a paradoxical non-monotonicity in the competing theory. The competing theory posits that small tally differences trigger tallying (favoring the option with more cues), but large tally differences trigger validity weighting (causing preference to abruptly revert back to the option with the single top cue). Thus, it predicts that as an option gains even more cues (e.g., 5 vs 2), its choice probability will paradoxically drop. The advocated LCA model uses configural weighting and strictly predicts monotonic accumulation, meaning the option's choice probability will continue to rise. This metric computes the difference in choice probability between the 5-cue and 2-cue conditions; it will be strongly positive for the advocated theory and negative for the competing theory, providing a robust and directly interpretable dissociation."
}
```

## Usage

```json
{
  "prompt_token_count": 10103,
  "candidates_token_count": 885,
  "total_token_count": 14653
}
```
