# metric_exp01_attempt_00

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

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 8: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 11: A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 12: A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated LCA model with Configural Weighting from the competing Dual-Process Dynamic Attention Model by exploiting a mathematical invariance in the competing theory regarding tied cues. The competing theory computes a dynamic mixture of unweighted tallying and validity-weighted integration. Because the softmax function is translation-invariant, adding a tied cue (a '1' for both options) increases the raw tally of both options equally and increases the validity sum of both options equally, leaving the differences completely unchanged. Furthermore, the dynamic attention shift parameter depends only on the absolute tally difference divided by the total number of cues, which also remains strictly constant. Thus, the competing theory predicts perfectly flat, invariant choice probabilities as tied cues are added. The advocated theory, however, utilizes a configural weighting mechanism where the effective weight of every cue is non-linearly scaled by the total number of positive cues favoring that option (sum ** gamma). As tied cues are added, the total sum increases, systematically modulating the configural weights and altering the absolute magnitudes of evidence fed into the non-linear LCA process. Thus, the advocated theory predicts graded, systematic shifts in choice probabilities, whereas the competing theory strictly predicts flat invariance.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



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
**Description:** Dual-Process Dynamic Attention Model: Decision-makers start by evaluating options using a fast, unweighted tallying process. If the relative tally difference (normalized by the number of cues) is large, this simple cue count drives the choice. However, when the initial relative tally difference is small or cues are conflicting, attention dynamically shifts toward the validities of the cues. In this later phase, cues are integrated proportionally to their reliability (validity). The decision-maker integrates the evidence (logits) from both processes before making a final choice, allowing for smooth compensatory behavior in high-conflict trials that scales consistently across environments with varying numbers of cues.

**Parameters:**
- beta_tally: [0.1, 20.0]
- beta_val: [0.1, 20.0]
- gamma: [0.0, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_tally = float(parameters["beta_tally"])
    beta_val = float(parameters["beta_val"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying process (unweighted)
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    z_tally = beta_tally * np.array([tally_a, tally_b])
    
    # Validity-weighted process
    val_a = np.sum(a * val)
    val_b = np.sum(b * val)
    
    z_val = beta_val * np.array([val_a, val_b])
    
    # Dynamic attention shift based on relative tally difference
    n_cues = len(a)
    tally_diff = abs(tally_a - tally_b) / max(1, n_cues)
    p_shift = np.exp(-gamma * tally_diff)
    
    # Mixture of evidence (logits) rather than probabilities
    z_mix = (1.0 - p_shift) * z_tally + p_shift * z_val
    z_mix = z_mix - np.max(z_mix)
    p_mix = np.exp(z_mix) / np.sum(np.exp(z_mix))
    
    # Lapse rate
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
[0] rationale: The experimental design systematically varies the number of tied cues while keeping the raw validity difference between Option A and Option B perfectly constant. The competing Dual-Process Dynamic Attention Model relies purely on the validity-weighted process because the tally difference is zero, predicting completely flat, invariant choice probabilities across all trial types. In contrast, the advocated LCA model with Configural Weighting predicts that the total number of cues non-linearly modulates the effective weights, causing systematic, graded shifts in choice probabilities. By computing the variance of the choice proportions across the different trial types for each subject, we can distinctly capture this difference: the competing model will yield a low variance (only binomial noise from the fixed choice probability), whereas the advocated model will produce a significantly higher variance due to the systematic shifts.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['trial_key'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['chose_a'] = (data['response'] == 0).astype(float)
    subj_trial_prop = data.groupby(['subject_id', 'trial_key'])['chose_a'].mean()
    subj_vars = subj_trial_prop.groupby('subject_id').var(ddof=0)
    return float(subj_vars.mean())
outcome: self_sim=0.0232 (var=0.0004) adversary_sim=0.0236 (var=0.0001) welch_t=-0.089 p=0.9297 (N=25, alpha=0.01) -> reject

[1] rationale: The experimental design keeps the raw validity difference between Option A and Option B constant across all trials, while systematically varying the total number of positive cues ('S'). Because the tally difference is always zero, the competing Dual-Process Dynamic Attention Model relies purely on validity-weighted integration, predicting a completely flat, invariant probability of choosing A across all trial types (yielding differences driven only by binomial sampling noise). In contrast, the advocated LCA model with Configural Weighting non-linearly modulates the effective weights based on the total number of cues, predicting systematic, graded shifts in the choice probability of A as S increases. By computing the squared difference in the choice proportion of A between High-S (S>=4) and Low-S (S<=2) trials for each subject, this metric cleanly isolates the systematic configural shift predicted by the advocated theory from the flat invariance predicted by the competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['chose_a'] = (data['response'] == 0).astype(float)
    data['S'] = data['option_a_ratings'].apply(lambda x: sum(float(v) for v in x))
    
    high_s = data[data['S'] >= 4].groupby('subject_id')['chose_a'].mean()
    low_s = data[data['S'] <= 2].groupby('subject_id')['chose_a'].mean()
    
    sq_diff = (high_s - low_s) ** 2
    return float(sq_diff.mean())
outcome: self_sim=0.0224 (var=0.0009) adversary_sim=0.0105 (var=0.0002) welch_t=+1.759 p=0.08728 (N=25, alpha=0.01) -> reject

[2] rationale: The competing theory integrates validities linearly, predicting that choice probability depends solely on the validity difference. By pairing trials with identical validity differences (0.10) and identical number of cues (S), but shifted base validities (e.g., Trial 1 vs Trial 7), the competing model predicts a difference of exactly zero. In contrast, the advocated LCA model applies a non-linear transducer (alpha) to the integrated evidence. Because alpha is >1 for ~90% of its prior range, this convex transduction amplifies evidence differences more when the absolute base validities are higher. Thus, the advocated theory predicts a strictly positive shift in choice probability for High Base trials compared to matched Low Base trials. This metric isolates this exact difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['S'] = data['option_a_ratings'].apply(lambda x: sum(float(v) for v in x))
    data['is_high_base'] = data['option_a_ratings'].apply(lambda x: float(x[0]) == 1.0)
    data['chose_a'] = (data['response'] == 0).astype(float)
    df_matched = data[data['S'] <= 4]
    p_high = df_matched[df_matched['is_high_base']].groupby('subject_id')['chose_a'].mean()
    p_low = df_matched[~df_matched['is_high_base']].groupby('subject_id')['chose_a'].mean()
    return float((p_high - p_low).mean())
outcome: self_sim=-0.0089 (var=0.0088) adversary_sim=-0.0672 (var=0.0124) welch_t=+2.001 p=0.05126 (N=25, alpha=0.01) -> reject

[3] rationale: The experimental design includes a subset of trials (Trials 1-6) where the raw validity difference between Option A and Option B is perfectly constant (0.10) and the tally difference is zero. For these trials, the competing Dual-Process Dynamic Attention Model predicts an identical, invariant probability of choosing A, because its validity-weighted integration depends only on the constant validity difference. Thus, the true squared difference in choice probabilities between High-S (Trials 5-6) and Low-S (Trials 1-2) is exactly zero. In contrast, the advocated LCA model with Configural Weighting predicts that the increasing number of cues (S) non-linearly modulates the effective weights, causing systematic divergence in choice probabilities between High-S and Low-S trials. To cleanly isolate this divergence without being confounded by binomial sampling noise, this metric computes an unbiased estimator of the squared difference in choice probabilities between High-S and Low-S conditions for each subject. For the competing model, the expected value of this estimator is exactly zero, whereas for the advocated model, it is strictly positive, yielding a highly discriminative contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['S'] = data['option_a_ratings'].apply(lambda x: sum(float(v) for v in x))
    # Trials 1-6 have A's first cue as 1.0
    data['is_base_high'] = data['option_a_ratings'].apply(lambda x: float(x[0]) == 1.0)
    data['chose_a'] = (data['response'] == 0).astype(float)
    
    # Restrict to trials 1-6 where validity difference is perfectly constant
    df = data[data['is_base_high']]
    
    def subject_unbiased_sq_diff(subj_df):
        # High S: Trials 5 and 6
        high_df = subj_df[subj_df['S'] >= 5]
        # Low S: Trials 1 and 2
        low_df = subj_df[subj_df['S'] <= 2]
        
        if len(high_df) == 0 or len(low_df) == 0:
            return np.nan
            
        n_h = len(high_df)
        n_l = len(low_df)
        
        p_h = high_df['chose_a'].mean()
        p_l = low_df['chose_a'].mean()
        
        # Unbiased estimator correction for variance
        var_h = (p_h * (1.0 - p_h)) / (n_h - 1.0) if n_h > 1 else 0.0
        var_l = (p_l * (1.0 - p_l)) / (n_l - 1.0) if n_l > 1 else 0.0
        
        # Unbiased estimator of (pi_h - pi_l)^2
        return float((p_h - p_l)**2 - var_h - var_l)

    res = df.groupby('subject_id').apply(subject_unbiased_sq_diff)
    return float(res.mean())
outcome: self_sim=0.0098 (var=0.0017) adversary_sim=-0.0093 (var=0.0004) welch_t=+2.095 p=0.04327 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    data = data.copy()\n    data['tally_a'] = data['option_a_ratings'].apply(lambda x: sum(float(v) for v in x))\n    data['tally_b'] = data['option_b_ratings'].apply(lambda x: sum(float(v) for v in x))\n    data['chose_a'] = (data['response'] == 0).astype(float)\n    \n    # Filter strictly to trials where the tally difference is zero (Trials 5-12)\n    df = data[data['tally_a'] == data['tally_b']].copy()\n    \n    # Create a unique string identifier for each trial type\n    df['trial_key'] = df['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x)) + '_' + \\\n                      df['option_b_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))\n                      \n    def subject_unbiased_variance(subj_df):\n        stats = subj_df.groupby('trial_key')['chose_a'].agg(['mean', 'count'])\n        stats = stats[stats['count'] > 1]\n        if len(stats) < 2:\n            return np.nan\n            \n        K = len(stats)\n        p_j = stats['mean'].values\n        n_j = stats['count'].values\n        \n        # Sample variance of the observed choice proportions across the K trial types\n        p_bar = np.mean(p_j)\n        S2 = np.sum((p_j - p_bar)**2) / (K - 1)\n        \n        # Unbiased estimator of the binomial sampling noise for each trial type\n        noise_j = (p_j * (1.0 - p_j)) / (n_j - 1.0)\n        mean_noise = np.mean(noise_j)\n        \n        # Subtracting the expected binomial noise yields an unbiased estimator of the TRUE variance in P(A)\n        return S2 - mean_noise\n\n    res = df.groupby('subject_id').apply(subject_unbiased_variance).dropna()\n    return float(res.mean())",
  "rationale": "In the experimental design, Trials 5-12 all feature exactly zero tally difference and an identical raw validity difference (+0.10 favoring A). The competing Dual-Process Dynamic Attention Model is mathematically constrained to predict an identical, invariant probability of choosing A across all of these trials, because its dynamic attention shift parameter strictly resolves to 1.0 (pure validity weighting) and the validity difference is constant. Thus, for the competing theory, the true variance of P(A) across these trial types is exactly zero. In contrast, the advocated LCA model with Configural Weighting predicts that the varying total number of cues across these trials non-linearly modulates the effective weights, causing systematic, graded variations in P(A). This metric isolates this divergence by computing an unbiased estimator of the true variance in P(A) across these specific trial types for each subject. It perfectly subtracts the expected binomial sampling noise, yielding an expected value of exactly 0 for the competing theory and a strictly positive value for the advocated theory."
}
```

## Usage

```json
{
  "prompt_token_count": 6142,
  "candidates_token_count": 788,
  "total_token_count": 13459
}
```
