# metric_exp00_attempt_00

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
**Validities (n_features=6):** [0.95, 0.9, 0.8, 0.75, 0.6, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 4: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 5: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 10: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 12: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the advocated LCA model with Configural Weighting from the competing Dual-Process Dynamic Attention Model by exploiting their divergent handling of tied cues when the overall tally is matched. The competing theory posits that when the tally difference is zero, the model relies 100% on a linear validity-weighted sum (WADD); thus, adding cues that are tied (present for both options) adds equal validity to both sides, keeping the validity difference constant and predicting perfectly invariant choice probabilities across trials. The advocated theory, however, applies a non-linear configural weight to all cues based on the total number of cues favoring the option (sum ** gamma). By starting with a single discriminating cue pair and systematically adding tied cues to both options, the total sum increases, which non-linearly scales the effective evidence in the LCA. The advocated theory predicts a systematic shift in choice probabilities as the background tally increases, while the competing theory strictly predicts a flat line. Additional trials with large tally differences are included to exercise the competing theory's unweighted tallying component.

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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    sum_a = data['option_a_ratings'].apply(sum)\n    sum_b = data['option_b_ratings'].apply(sum)\n    \n    # Focus on trials where the tallies are perfectly matched\n    mask = sum_a == sum_b\n    \n    # Compare choice rates between trials with high background tally vs low background tally\n    high_mask = mask & (sum_a >= 4)\n    low_mask = mask & (sum_a <= 2)\n    \n    # P(Choose A) = 1.0 - mean(response)\n    p_a_high = 1.0 - data.loc[high_mask, 'response'].mean()\n    p_a_low = 1.0 - data.loc[low_mask, 'response'].mean()\n    \n    if pd.isna(p_a_high) or pd.isna(p_a_low):\n        return 0.0\n        \n    return float(p_a_high - p_a_low)",
  "rationale": "This metric isolates trials where the unweighted tally difference is exactly zero (trials 1-5). Under the competing Dual-Process Dynamic Attention Model, a zero tally difference forces the model to rely entirely on the validity-weighted sum, which is mathematically identical across these trials. Thus, the competing theory predicts a flat line (difference = 0). The advocated LCA model, however, non-linearly scales the effective evidence of each cue by the total number of supporting cues (configural weighting). By subtracting the choice probability of A in low-tally trials from high-tally trials, we expose this non-linear shift predicted exclusively by the advocated theory."
}
```

## Usage

```json
{
  "prompt_token_count": 4012,
  "candidates_token_count": 407,
  "total_token_count": 5985
}
```
