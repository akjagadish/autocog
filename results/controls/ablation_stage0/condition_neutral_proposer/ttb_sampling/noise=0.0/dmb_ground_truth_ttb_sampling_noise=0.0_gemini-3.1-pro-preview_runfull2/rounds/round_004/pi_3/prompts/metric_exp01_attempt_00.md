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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 8: A=[0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 10: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 11: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** Theory 1 (Take The Best) relies solely on the first discriminating cue, meaning its choice probability is constant (determined only by the lapse rate) regardless of whether lower-ranked cues support or oppose the top cue. Theory 2 (Exponentially Weighted Compensatory Model) also ensures the top cue determines the favored option (due to the strict dominance of the base >= 2.0), but computes a softmax over the total weighted value. This means Theory 2 predicts that choice probability will vary systematically depending on the margin of value difference, which is modulated by the lower-ranked cues. By presenting trials where the top discriminating cue is held constant but lower-ranked cues range from full opposition to full support, we can quantitatively dissociate the constant-probability prediction of Theory 1 from the graded-probability prediction of Theory 2.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Take The Best (TTB): People compare two options by ordering features by their subjective validity and searching through them sequentially. The search stops at the first feature that discriminates between the two options (i.e., one option has a higher value than the other), and the decision is based entirely on that single feature. This non-compensatory strategy ignores all other features, preventing any compensatory trade-offs. If no feature discriminates, the learner guesses. Response noise is modeled via an independent lapse rate epsilon, which replaces the deterministic TTB choice with a uniform random pick.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(validities, kind='stable')[::-1]
    
    # Default to guessing if no cue discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Exponentially Weighted Compensatory Model (Rank-based with Strict Dominance Base): Decision-makers evaluate options using a single compensatory process where the weight of each feature grows exponentially with its subjective validity rank, using a base >= 2.0. This guarantees strict lexicographic dominance, ensuring that a single higher-ranked cue always outweighs all lower-ranked cues combined. By enforcing this strict dominance, the model acts identically to Take-The-Best across all conflict scenarios, relying on an independent lapse rate for probabilistic errors rather than softening the decision temperature.

**Parameters:**
- base: [2.0, 10.0]
- tau: [0.0, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    base = float(parameters["base"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Compute weights using exponential growth based on rank
    # A base >= 2.0 ensures strict TTB behavior (lexicographic dominance)
    order = np.argsort(validities, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(validities))
    
    w = base ** ranks
    
    # Compute overall value for each option
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
    # Compute choice probabilities using softmax over values
    logits = tau * np.array([v_a, v_b])
    logits = logits - np.max(logits)
    p = np.exp(logits)
    p = p / np.sum(p)
    
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
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
[0] rationale: In all 10 trial pairs of the design, the highest-ranked discriminating cue always favors Option A. Therefore, Take The Best (TTB) predicts that Option A will be chosen with a constant probability (1 - epsilon/2) across all trials, regardless of the lower-ranked cues. Consequently, the difference in the choice rate of A between trials where the simple sum of cues favors A (High Margin) and trials where the simple sum of cues opposes A (Low Margin) will be zero under TTB.

Conversely, the Exponentially Weighted Compensatory Model computes a weighted sum of all cues and applies a softmax decision rule. Because the lower-ranked cues heavily modulate the value difference between the options, the softmax choice probability for A will be significantly higher in the High Margin trials than in the Low Margin trials. Thus, this metric (P(A|High Margin) - P(A|Low Margin)) will be tightly centered at 0 for TTB, but strictly positive for the Compensatory Model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    is_a_chosen = (data['response'] == 0)
    
    high_mask = sum_a > sum_b
    low_mask = sum_a < sum_b
    
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0
        
    p_high = is_a_chosen[high_mask].mean()
    p_low = is_a_chosen[low_mask].mean()
    
    return float(p_high - p_low)
outcome: self_sim=-0.0007 (var=0.0047) adversary_sim=-0.0013 (var=0.0096) welch_t=+0.025 p=0.9804 (N=25, alpha=0.01) -> reject

[1] rationale: Under Take The Best (TTB), the decision is determined entirely by the first discriminating cue. In both Trial 1 and Trial 4, the highest-validity cue favors Option A. Thus, TTB predicts that Option B will be chosen at exactly the same rate (epsilon / 2) across both trials, yielding an expected difference of 0. Under the Exponentially Weighted Compensatory Model, the value difference (v_a - v_b) is strictly positive for both, but it is much smaller in Trial 1 (where lower cues heavily oppose A) than in Trial 4. Because of the softmax decision rule, smaller value differences result in choices closer to 50/50 when the temperature parameter (tau) is small. Therefore, the Competing Theory predicts a strictly positive difference (P(B|Trial 1) > P(B|Trial 4)). By isolating the single most extreme conflict trial and the single most extreme alignment trial, we maximize the theoretical contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the sum of features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trial 1: Option A has 1 positive feature, Option B has 4 positive features.
    # This trial represents the maximum conflict where the highest validity cue favors A
    # but all lower validity cues favor B. The value difference (v_a - v_b) is minimal.
    mask_1 = (sum_a == 1) & (sum_b == 4)
    
    # Trial 4: Option A has 5 positive features, Option B has 0.
    # This trial represents maximum alignment. The value difference (v_a - v_b) is maximal.
    mask_4 = (sum_a == 5) & (sum_b == 0)
    
    if mask_1.sum() == 0 or mask_4.sum() == 0:
        return 0.0
        
    # Calculate the proportion of times Option B was chosen (response == 1)
    p_b_1 = data.loc[mask_1, 'response'].mean()
    p_b_4 = data.loc[mask_4, 'response'].mean()
    
    # Take The Best predicts identical choice probabilities for both trials 
    # (driven entirely by the first cue and the lapse rate epsilon), so the expected difference is 0.
    # The Exponentially Weighted Compensatory Model predicts a slightly higher probability 
    # of choosing B in Trial 1 due to the lower value difference when the softmax temperature (tau) is small.
    return float(p_b_1 - p_b_4)
outcome: self_sim=-0.0022 (var=0.0230) adversary_sim=-0.0222 (var=0.0237) welch_t=+0.463 p=0.6455 (N=25, alpha=0.01) -> reject

[2] rationale: Under TTB, Option A is favored by the first discriminating cue in every single trial of this design. With epsilon in [0, 0.5], TTB's probability of choosing A is always between 0.75 and 1.0. Under the Exponentially Weighted Compensatory Model, the choice probability depends on the softmax temperature tau. When tau is very small, EWCM predicts choice probabilities much closer to 0.5, pulling the overall expected choice rate of A slightly lower than TTB across the population.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Overall probability of choosing Option A
    return float((data['response'] == 0).mean())
outcome: self_sim=0.8716 (var=0.0062) adversary_sim=0.8522 (var=0.0066) welch_t=+0.855 p=0.3971 (N=25, alpha=0.01) -> reject

[3] rationale: Under Take The Best (TTB), the probability of choosing Option A is entirely determined by the first discriminating cue and the lapse rate (epsilon). Because the first discriminating cue favors A in every single trial of this design, TTB predicts that the underlying probability of choosing A is identical across all 10 trial types. Consequently, the observed variance of the choice rates across trial types within a subject will simply be the expected binomial noise.

Under the Exponentially Weighted Compensatory Model (EWCM), the underlying choice probabilities vary across trial types because the value difference between A and B heavily depends on the lower-ranked cues. This introduces true variance in the choice probabilities across trial types.

This metric computes the 'excess variance'—the observed variance of choice rates minus the expected binomial variance—for each subject. For TTB, this metric is mathematically expected to be exactly 0, with very tight variance across subjects because it normalizes out individual differences in epsilon. For EWCM, the varying choice probabilities will inflate the observed variance, resulting in a strictly positive excess variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    # Create a hashable trial identifier
    data['trial_id'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, x))) + "_" + data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    def subject_metric(sub_df):
        # Calculate proportion of A choices for each trial type
        p_a_by_trial = sub_df.groupby('trial_id')['response'].apply(lambda x: (x == 0).mean())
        
        # Observed variance of these proportions across the 10 trial types
        obs_var = p_a_by_trial.var(ddof=1)
        if pd.isna(obs_var):
            return 0.0
            
        # Overall proportion of A choices for this subject
        overall_p = (sub_df['response'] == 0).mean()
        N = len(sub_df)
        if N <= 1:
            return 0.0
            
        # Expected binomial variance if all trial types had the exact same underlying probability
        # Using the exact unbiased estimator for p(1-p) to prevent any systematic bias
        n_reps = N / len(p_a_by_trial)
        exp_var = (overall_p * (1.0 - overall_p) * (N / (N - 1))) / n_reps
        
        # The difference will be exactly 0 in expectation for TTB, but positive for EWCM
        return obs_var - exp_var

    # Return the mean of this excess variance across all subjects
    return float(data.groupby('subject_id').apply(subject_metric).mean())
outcome: self_sim=0.0002 (var=0.0000) adversary_sim=-0.0003 (var=0.0000) welch_t=+0.302 p=0.7637 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    sum_a = data['option_a_ratings'].apply(sum)\n    sum_b = data['option_b_ratings'].apply(sum)\n    \n    # Trial 2: A=[1, 1, 1, 1, 1], B=[0, 0, 0, 0, 0]\n    # Represents maximum value difference (Delta) for EWCM.\n    t2_mask = (sum_a == 5) & (sum_b == 0)\n    \n    # Trial 9: A=[0, 0, 0, 1, 0], B=[0, 0, 0, 0, 1]\n    # Represents the absolute minimum value difference for EWCM, \n    # because the only active features are the lowest two validities.\n    # Delta = base^1 - base^0 = base - 1. \n    # Even for the maximum base=10, Delta is only 9, maximizing softmax softening.\n    t9_mask = (sum_a == 1) & (sum_b == 1)\n    \n    if t2_mask.sum() == 0 or t9_mask.sum() == 0:\n        return 0.0\n        \n    p_t2 = (data.loc[t2_mask, 'response'] == 0).mean()\n    p_t9 = (data.loc[t9_mask, 'response'] == 0).mean()\n    \n    return float(p_t2 - p_t9)",
  "rationale": "Under Take The Best (TTB), the choice is determined entirely by the first discriminating cue. In both Trial 2 and Trial 9, the first discriminating cue favors Option A. Thus, TTB predicts identical choice probabilities (1 - epsilon/2) for both trials, yielding an expected difference of exactly 0. Under the Exponentially Weighted Compensatory Model (EWCM), the value difference (Delta) strongly dictates the choice probability via a softmax function. Trial 2 has the maximal possible Delta, whereas Trial 9 has the minimal possible Delta (base - 1) across the entire experiment, making Trial 9 uniquely vulnerable to softmax softening. Thus, EWCM predicts a strictly positive difference (P(A|Trial 2) > P(A|Trial 9))."
}
```

## Usage

```json
{
  "prompt_token_count": 5549,
  "candidates_token_count": 549,
  "total_token_count": 12067
}
```
