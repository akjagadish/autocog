# metric_exp00_attempt_03

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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates Attention-Gated Integration from Evidence Dilution Theory by exploiting their fundamentally different dilution denominators. Attention-Gated Integration penalizes options based strictly on the *number* of cues that pass the attention gate (len(kept_vals)^gamma). Evidence Dilution Theory, however, penalizes options based on the *sum of the validities* of all present cues (sum_val^gamma). We contrast trials where options have the same number of features but vastly different validity sums. For instance, an option with one strong and two very weak cues has a small validity sum, while an option with three moderately strong cues has a large validity sum. Attention-Gated Integration applies the identical dilution penalty (n=3) to both, allowing the option with moderately strong cues to easily dominate. In contrast, Evidence Dilution Theory applies a much harsher dilution penalty to the option with the larger sum of validities, which can paradoxically cause the option with weaker features to be preferred when the dilution parameter (gamma) is high.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Attention-Gated Integration Theory: Decision-makers do not integrate all available information. Instead, they anchor on the most valid feature present in an option and only integrate additional features if their validity is sufficiently close to this maximum (attention gate). Cues that pass this threshold are integrated using a fractional averaging process, which can cause 'dilution' where adding moderately valid cues to a highly valid cue actually decreases the option's subjective value. This explains why dilution occurs in some contexts (when weak cues pass the threshold of a weak top cue) but is absent in others (when a strong top cue filters out the weak cues entirely).

**Parameters:**
- theta: [0.0, 0.5]
- gamma: [0.0, 2.0]
- lambda_val: [0.1, 10.0]
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
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    lambda_val = float(parameters["lambda_val"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    def get_value(features):
        present_cues = np.where(features > 0)[0]
        if len(present_cues) == 0:
            return 0.0
        
        present_vals = val[present_cues]
        max_v = np.max(present_vals)
        
        # Attention gate: keep cues within theta of the max validity
        kept_mask = present_vals >= (max_v - theta)
        kept_vals = present_vals[kept_mask]
        
        # Integration
        w = kept_vals ** lambda_val
        v = np.sum(w) / (len(kept_vals) ** gamma)
        return v
        
    v_a = get_value(a)
    v_b = get_value(b)
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
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
**Description:** Evidence Dilution and Non-linear Weighting Theory (Validity-based Dilution with Amplified Penalty): Decision-makers evaluate options by integrating the validities of present features. However, instead of purely adding evidence, they partially average it. The presence of many low-validity features can paradoxically dilute the overall subjective value of an option (Evidence Dilution). This dilution is proportional to the sum of the validities of the present cues, and subjects apply a non-linear scaling to feature validities, amplifying the impact of the most valid cues. A potentially strong dilution penalty allows for severe subjective devaluation of options burdened with numerous weak features.

**Parameters:**
- lambda_val: [1.0, 20.0]
- gamma: [0.0, 10.0]
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
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting to capture TTB-like reliance on top cues
    w = val ** lambda_val
    
    # Dilute by the sum of validities of the present cues
    sum_val_a = np.sum(val * a)
    sum_val_b = np.sum(val * b)
    
    # Calculate subjective values with a dilution factor (gamma)
    v_a = np.sum(w * a) / (sum_val_a ** gamma) if sum_val_a > 0 else 0.0
    v_b = np.sum(w * b) / (sum_val_b ** gamma) if sum_val_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
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
[0] rationale: This metric calculates the difference in the choice probability of Option A between Trial 3 and Trial 4. In Trial 3, Option A has cues [0.95, 0.55, 0.5] (sum = 2.0) and Option B has [0.9, 0.85, 0.6] (sum = 2.35). In Trial 4, Option A adds the 0.6 cue (sum = 2.6) while B remains the same. Under Evidence Dilution Theory, adding the 0.6 cue significantly increases the dilution denominator (from 2.0 to 2.6), penalizing A heavily and causing a large drop in A's preference from Trial 3 to Trial 4. Under Attention-Gated Integration, if the attention gate filters out the weak cues, A's subjective value remains largely unchanged across the two trials, resulting in a difference near zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    trial_3_mask = data['A_str'] == '100011'
    trial_4_mask = data['A_str'] == '100111'
    
    p_a_3 = 1.0 - data.loc[trial_3_mask, 'response'].mean() if trial_3_mask.sum() > 0 else 0.5
    p_a_4 = 1.0 - data.loc[trial_4_mask, 'response'].mean() if trial_4_mask.sum() > 0 else 0.5
    
    return float(p_a_3 - p_a_4)
outcome: self_sim=0.0058 (var=0.0147) adversary_sim=0.0375 (var=0.0226) welch_t=-0.820 p=0.4162 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in the choice probability of Option A between Trial 2 and Trial 4. In Trial 2, Option A has a smaller validity sum (1.5) compared to Option B (1.75). In Trial 4, Option A has a larger validity sum (2.6) compared to Option B (2.35). Under Evidence Dilution Theory, which penalizes options based on the sum of validities, Option A should be strongly preferred in Trial 2 and heavily penalized in Trial 4, leading to a large positive difference. Under Attention-Gated Integration, weak cues are likely filtered out by the strong top cues (0.95 for A, 0.9 for B), keeping Option A's subjective value relatively stable across both trials and resulting in a difference near zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    trial_2_mask = data['A_str'] == '100010'
    trial_4_mask = data['A_str'] == '100111'
    
    p_a_2 = 1.0 - data.loc[trial_2_mask, 'response'].mean() if trial_2_mask.sum() > 0 else 0.5
    p_a_4 = 1.0 - data.loc[trial_4_mask, 'response'].mean() if trial_4_mask.sum() > 0 else 0.5
    
    return float(p_a_2 - p_a_4)
outcome: self_sim=-0.0042 (var=0.0169) adversary_sim=0.0633 (var=0.0195) welch_t=-1.768 p=0.08338 (N=25, alpha=0.01) -> reject

[2] rationale: This metric contrasts the preference for Option A between Trial 1 and Trial 3. In Trial 1, both options have a single cue (A: 0.95, B: 0.90), making A slightly preferred. In Trial 3, Option A adds two weak cues (0.55, 0.50) while Option B adds two stronger cues (0.85, 0.60). Under Attention-Gated Integration, A's weak cues are likely filtered out by the attention gate (0.95 - theta), while B's strong cue (0.85) is integrated, boosting B's value and causing a drop in A's preference (positive difference). Under Evidence Dilution Theory, however, Option B's large sum of validities (2.35) incurs a severe dilution penalty compared to A's smaller sum (2.00). This heavy penalty on B can paradoxically increase A's relative preference in Trial 3 compared to Trial 1, leading to a negative or near-zero difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    trial_1_mask = data['A_str'] == '100000'
    trial_3_mask = data['A_str'] == '100011'
    
    p_a_1 = 1.0 - data.loc[trial_1_mask, 'response'].mean() if trial_1_mask.sum() > 0 else 0.5
    p_a_3 = 1.0 - data.loc[trial_3_mask, 'response'].mean() if trial_3_mask.sum() > 0 else 0.5
    
    return float(p_a_1 - p_a_3)
outcome: self_sim=0.0750 (var=0.0770) adversary_sim=0.0150 (var=0.0892) welch_t=+0.736 p=0.4654 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))\n    \n    trial_1_mask = data['A_str'] == '100000'\n    trial_4_mask = data['A_str'] == '100111'\n    \n    p_b_1 = data.loc[trial_1_mask, 'response'].mean() if trial_1_mask.sum() > 0 else 0.5\n    p_b_4 = data.loc[trial_4_mask, 'response'].mean() if trial_4_mask.sum() > 0 else 0.5\n    \n    return float(p_b_4 - p_b_1)",
  "rationale": "This metric measures the shift in preference for Option B between Trial 1 and Trial 4. In Trial 1, both options have a single cue, and Option A's cue is stronger (0.95 vs 0.9), so both theories predict a low baseline preference for B. In Trial 4, Option A is burdened with three additional weak cues (summing to 2.6), while Option B adds two cues (summing to 2.35). Under Evidence Dilution Theory, Option A's much larger validity sum incurs a severe dilution penalty, causing a large shift in preference towards Option B. Under Attention-Gated Integration, Option A's weak cues are largely filtered out by the attention gate anchored on its strong 0.95 cue, protecting it from dilution and keeping the preference for B low. Thus, the difference P(B|Trial 4) - P(B|Trial 1) will be large and positive for Evidence Dilution, but near zero for Attention-Gated Integration."
}
```

## Usage

```json
{
  "prompt_token_count": 4845,
  "candidates_token_count": 433,
  "total_token_count": 8961
}
```
