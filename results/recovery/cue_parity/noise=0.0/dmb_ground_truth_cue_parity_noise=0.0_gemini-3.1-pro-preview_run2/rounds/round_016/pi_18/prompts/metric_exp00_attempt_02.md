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
**Validities (n_features=6):** [0.9, 0.8, 0.7, 0.6, 0.5, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 2: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 8: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the advocated Dual-Process Dynamic Attention Model from the competing LCA theory by exploiting their divergent sensitivity to absolute versus relative validity sums. The advocated theory posits a linear integration of validities (WADD) mixed with a linear tally difference, where the mixture weight depends solely on the absolute tally difference. Consequently, it predicts perfectly invariant choice probabilities across any set of trials where the tally difference and the linear validity difference (V_A - V_B) are both held constant, regardless of the absolute magnitude of the validities. LCA, conversely, applies a non-linear transducer (alpha) and configural weighting (gamma) to the absolute sum of validities for each option. It predicts that adding a constant to the validities of both options (shifting V_A and V_B up while keeping V_A - V_B constant) will non-linearly scale the accumulated evidence, systematically changing the choice probabilities. By presenting sets of trials with constant tally differences and constant validity differences, but increasing absolute validities, the advocated theory strictly predicts flat, invariant lines, whereas LCA predicts systematic curves.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the difference in choice accuracy (choosing the option with the higher validity sum) between trials with a high absolute sum of validities and trials with a low absolute sum of validities. By design, the relative tally difference and relative validity differences are held constant across these trial subsets. The advocated Dual-Process Dynamic Attention Model strictly predicts this difference to be zero, as its choice probabilities rely on relative differences. The competing LCA model, however, utilizes non-linear absolute evidence accumulation, predicting a systematic shift in accuracy as the absolute validities scale up. Thus, this metric strongly discriminates the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.5])
    
    def get_val_sum(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * validities) + np.sum(b * validities)
        
    def get_val_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * validities) - np.sum(b * validities)
        
    val_sums = data.apply(get_val_sum, axis=1)
    val_diffs = data.apply(get_val_diff, axis=1)
    
    # The option with the higher validity sum is the 'correct' choice
    correct_choice = (val_diffs < 0).astype(int)
    acc = (data['response'] == correct_choice).astype(float)
    
    high_mask = val_sums >= 3.6
    low_mask = val_sums <= 3.2
    
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0
        
    return acc[high_mask].mean() - acc[low_mask].mean()

outcome: self_sim=-0.0102 (var=0.0065) adversary_sim=0.0119 (var=0.0113) welch_t=-0.831 p=0.4106 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the total variance in choice probabilities within specific subsets of trials. By design, the advocated Dual-Process Dynamic Attention Model predicts perfectly invariant choice probabilities within each of these subsets, because the relative tally difference and relative linear validity difference are held constant. Therefore, for the advocated model, the variance within these groups will only reflect binomial sampling noise and will be very close to zero. In contrast, the competing Leaky Competing Accumulator (LCA) model applies non-linear scaling to the absolute evidence for each option. As the absolute validities scale up within these trial subsets, the LCA predicts systematic shifts in choice probabilities, leading to a significantly higher variance. Summing the variances across subsets ensures that the metric captures the absolute magnitude of the non-linear divergence regardless of its direction.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Create string representation for matching trial pairs
    data['pair_str'] = data.apply(lambda row: ''.join([str(int(x)) for x in row['option_a_ratings']]) + '_' + ''.join([str(int(x)) for x in row['option_b_ratings']]), axis=1)
    
    # Define the groups of trials that should have identical P(A) under the advocated model
    # In these groups, the tally difference and relative linear validity difference are constant.
    g1 = ['001110_100001', '010110_100100', '011010_101000', '101010_110000']
    g2 = ['100001_001110', '100100_010110', '101000_011010', '110000_101010']
    g3 = ['000111_010100', '001011_011000', '010011_101000', '100011_110000']
    
    # Calculate P(Choose A) for each unique pair
    p_a = data.groupby('pair_str')['response'].apply(lambda x: (x == 0).mean()).to_dict()
    
    total_var = 0.0
    for g in [g1, g2, g3]:
        probs = [p_a.get(pair, np.nan) for pair in g]
        probs = [p for p in probs if not np.isnan(p)]
        if len(probs) > 1:
            total_var += np.var(probs)
            
    return float(total_var)
outcome: self_sim=0.0006 (var=0.0007) adversary_sim=0.0007 (var=0.0015) welch_t=-0.011 p=0.9915 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    data = data.copy()\n    data['pair_str'] = data.apply(\n        lambda row: ''.join(map(str, map(int, row['option_a_ratings']))) + '_' + \n                    ''.join(map(str, map(int, row['option_b_ratings']))), axis=1)\n    \n    p_a = data.groupby('pair_str')['response'].apply(lambda x: (x == 0).mean()).to_dict()\n    \n    # Group 1: A has tally 3, B has tally 2 (Absolute validities increase T1 -> T4)\n    t1 = '001110_100001'\n    t2 = '010110_100100'\n    t3 = '011010_101000'\n    t4 = '101010_110000'\n    \n    # Group 2: A has tally 2, B has tally 3 (Absolute validities increase T5 -> T8)\n    t5 = '100001_001110'\n    t6 = '100100_010110'\n    t7 = '101000_011010'\n    t8 = '110000_101010'\n    \n    # Group 3: A has tally 3, B has tally 2 (Absolute validities increase T9 -> T12)\n    t9 = '000111_010100'\n    t10 = '001011_011000'\n    t11 = '010011_101000'\n    t12 = '100011_110000'\n    \n    def get_p(t):\n        return p_a.get(t, 0.5)\n        \n    # Calculate cross-multiplied differences to avoid binomial variance bias\n    d1_g1 = get_p(t4) - get_p(t1)\n    d2_g1 = get_p(t3) - get_p(t2)\n    prod1 = d1_g1 * d2_g1\n    \n    # For Group 2, the High Tally Option is B, so we track P(B) = 1 - P(A)\n    d1_g2 = get_p(t5) - get_p(t8) \n    d2_g2 = get_p(t6) - get_p(t7)\n    prod2 = d1_g2 * d2_g2\n    \n    d1_g3 = get_p(t12) - get_p(t9)\n    d2_g3 = get_p(t11) - get_p(t10)\n    prod3 = d1_g3 * d2_g3\n    \n    return float(prod1 + prod2 + prod3)",
  "rationale": "This metric calculates the product of independent differences in choice probabilities across monotonically increasing absolute validities. The advocated Dual-Process model predicts perfectly flat probabilities across these trials, so the expected value of each difference is 0, and their product is 0, effectively stripping away binomial noise. The LCA model predicts systematic non-linear scaling, meaning the differences will consistently share the same sign (both positive or both negative depending on the alpha parameter), resulting in a strictly positive expected product. This creates a strong, unbiased discriminator."
}
```

## Usage

```json
{
  "prompt_token_count": 5336,
  "candidates_token_count": 886,
  "total_token_count": 12495
}
```
