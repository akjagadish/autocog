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
**Validities (n_features=7):** [0.98, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Strategy Mixture: TTB + Compensatory WADD/Tallying). The dissociation exploits the Advocated Theory's unweighted sequential tally and early stopping mechanism. By using 7 features with a specific validity profile, we construct 'trap' trials where Option A wins features 2, 3, and 4, while Option B wins feature 1 and features 5, 6, and 7. For the Advocated Theory, the running unweighted tally reaches +2 in favor of A at feature 4, triggering an early stop (if threshold = 2) and a robust choice for A. However, for the Competing Theory, Option B is structurally forced to win across all its sub-models: TTB chooses B (wins feature 1), Tallying chooses B (4 wins vs 3), and WADD chooses B (the massive weight of feature 1 plus the tail outscales features 2-4 for all gamma >= 0). Thus, the Competing Theory mathematically must predict B, while the Advocated Theory uniquely predicts A.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers use a Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity, maintaining a running sum of the differences between the options. If the absolute accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally. This allows the model to smoothly transition between Take-The-Best (low threshold) and Tallying (high threshold) behaviors.

**Parameters:**
- threshold: [0.0, 3.0]
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Strategy Mixture Theory (Environment-Adaptive Toolbox with Tallying Blend)

**Parameters:**
- theta: [0.0, 30.0]
- disp_threshold: [0.0, 0.5]
- gamma: [0.0, 10.0]
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Strategy 1: Take-The-Best (TTB)
    # Inspect features in descending order of validity; stop at the first discriminating feature.
    order = np.argsort(val)[::-1]
    ttb_prob = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_prob = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_prob = np.array([0.0, 1.0])
            break
            
    # Strategy 2: Compensatory (Blend of WADD and Tallying)
    # Evaluate all features, weighting them by a blend of pure Tallying (equal weights) and WADD.
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    centered_val = np.clip(val - 0.5, 1e-6, 0.5)
    
    # Scale validities to [0, 1] based on absolute magnitude, rather than normalizing by max
    scaled_val = centered_val * 2.0
        
    w_wadd = scaled_val ** gamma
    w = alpha * w_wadd + (1.0 - alpha) * 1.0
    
    comp_scores = np.array([np.sum(a * w), np.sum(b * w)])
    
    beta = float(parameters["beta"])
    z = beta * comp_scores
    z -= np.max(z)  # numerical stability
    e = np.exp(z)
    comp_prob = e / np.sum(e)
    
    # Strategy Mixture: Probabilistic selection between TTB and Compensatory based on environment
    # The probability of selecting TTB depends on the dispersion (std dev) of the validities.
    dispersion = float(np.std(val))
    theta = float(parameters["theta"])
    disp_threshold = float(parameters["disp_threshold"])
    
    z_disp = theta * (dispersion - disp_threshold)
    p_ttb = 1.0 / (1.0 + np.exp(-np.clip(z_disp, -20.0, 20.0)))
    
    p_core = p_ttb * ttb_prob + (1.0 - p_ttb) * comp_prob
    
    # Trembling hand error
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
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
[0] rationale: This metric calculates the proportion of choices that align with the Compensatory strategy (WADD) on the specific 'trap' trials identified in the experimental design. On these trials (e.g., A=[1, 0, 0, 1, 1] vs B=[0, 1, 1, 0, 0]), the Advocated Theory (Sequential Evidence Accumulation) is structurally constrained to always choose A, because the unweighted tally never drops below 0 and finishes at +1. In contrast, the Competing Theory's compensatory component strongly favors B because the 2nd and 3rd features carry significantly more weight than the 4th and 5th. By measuring the rate of choosing the compensatory option on these specific trials, we isolate the behavior that is mathematically impossible under the Advocated Theory but highly probable under the Competing Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify the 'trap' trials
    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t2_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    
    # Measure proportion of choices aligning with the Compensatory strategy
    # On T1, Compensatory favors B (response 1)
    # On T2, Compensatory favors A (response 0)
    t1_choices = data.loc[t1_mask, 'response'] == 1
    t2_choices = data.loc[t2_mask, 'response'] == 0
    
    valid_choices = pd.concat([t1_choices, t2_choices])
    
    if len(valid_choices) == 0:
        return 0.0
        
    return float(valid_choices.mean())

outcome: self_sim=0.1433 (var=0.0166) adversary_sim=0.1958 (var=0.0214) welch_t=-1.346 p=0.1846 (N=25, alpha=0.01) -> reject

[1] rationale: This metric exploits a structural constraint in the Advocated Theory (Sequential Evidence Accumulation) to create a pure mathematical dissociation. For Trials 1, 2, and 5, the Advocated Theory's unweighted cumulative evidence path reaches a maximum absolute value of exactly 1. Because of this, it is impossible for the model to stop at any threshold >1, meaning it will always evaluate all features and end with a final score of +1, -1, and +1 respectively. Even if the threshold is <=1, it stops at the first feature with the exact same scores. Therefore, the Advocated Theory predicts that the probability of choosing the Take-The-Best (TTB) favored option is EXACTLY identical across Trials 1, 2, and 5. The metric subtracts the average TTB choice rate of Trials 1 and 2 from Trial 5, which will mathematically average to 0.0 for the Advocated Theory, regardless of the subject's beta or epsilon parameters. In contrast, the Competing Theory's Compensatory sub-model strongly favors the TTB option on Trial 5, but often opposes it on Trials 1 and 2 (due to the compensatory weights of features 2 and 3). Thus, the Competing Theory predicts a significantly higher TTB choice rate on Trial 5 than on Trials 1 and 2, yielding a strictly positive value for this metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t2_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    t5_mask = (a_tuples == (1, 0, 1, 0, 1)) & (b_tuples == (0, 1, 0, 1, 0))
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_b_t2 = (data.loc[t2_mask, 'response'] == 1).mean()
    p_a_t5 = (data.loc[t5_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1): p_a_t1 = 0.0
    if pd.isna(p_b_t2): p_b_t2 = 0.0
    if pd.isna(p_a_t5): p_a_t5 = 0.0
    
    return float(p_a_t5 - (p_a_t1 + p_b_t2) / 2.0)
outcome: self_sim=0.0100 (var=0.0187) adversary_sim=0.0392 (var=0.0184) welch_t=-0.757 p=0.4526 (N=25, alpha=0.01) -> reject

[2] rationale: This metric measures the difference in the rate of choosing the Take-The-Best (TTB) favored option between two sets of trials: {Trial 3, Trial 4} and {Trial 1, Trial 2}. Under the Advocated Theory (Sequential Evidence Accumulation), the unweighted tally for Trials 1 and 2 structurally guarantees choosing the TTB option regardless of threshold. However, for Trials 3 and 4, if the threshold is high (e.g., >2), the tally evaluates all features and ends up favoring the non-TTB option. Thus, the Advocated Theory predicts P(TTB | T3, T4) < P(TTB | T1, T2), yielding a negative metric value. Conversely, under the Competing Theory (Strategy Mixture), both TTB and the Compensatory (WADD) sub-models strongly agree on the TTB option for Trials 3 and 4. But for Trials 1 and 2, the Compensatory sub-model heavily favors the non-TTB option, pulling the choice rate down. Thus, the Competing Theory predicts P(TTB | T3, T4) > P(TTB | T1, T2), yielding a strictly positive metric value.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t2_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    t3_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    t4_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    p_ttb_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_ttb_t2 = (data.loc[t2_mask, 'response'] == 1).mean()
    p_ttb_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    p_ttb_t4 = (data.loc[t4_mask, 'response'] == 1).mean()
    
    if pd.isna(p_ttb_t1): p_ttb_t1 = 0.5
    if pd.isna(p_ttb_t2): p_ttb_t2 = 0.5
    if pd.isna(p_ttb_t3): p_ttb_t3 = 0.5
    if pd.isna(p_ttb_t4): p_ttb_t4 = 0.5
    
    p_ttb_t34 = (p_ttb_t3 + p_ttb_t4) / 2.0
    p_ttb_t12 = (p_ttb_t1 + p_ttb_t2) / 2.0
    
    return float(p_ttb_t34 - p_ttb_t12)
outcome: self_sim=-0.2508 (var=0.1400) adversary_sim=-0.2317 (var=0.0721) welch_t=-0.208 p=0.8361 (N=25, alpha=0.01) -> reject

[3] rationale: This metric computes a structural invariant of the Competing Theory. The metric evaluates the sum of specific choice probabilities: `P(A|T1) + P(A|T3) + P(A|T6)` (where T1 and T3 are averaged with their symmetric counterparts T2 and T4 to halve binomial noise). Under the Competing Theory, the three pure sub-strategies (TTB, WADD, and Tallying) each yield a sum of exactly 2.0 for this specific triplet. Because the Competing Theory is a convex mixture of these strategies, its predicted sum is strictly bounded above by 2.0 (and is typically lower due to epsilon error and softmax flattening). In stark contrast, the Advocated Theory's sequential accumulation structurally predicts a sum of 3.0 for subjects with a threshold between 1 and 2, and 2.0 otherwise, pushing its population mean significantly above 2.0. By exploiting this exact algebraic invariance in the Competing Theory, we achieve a massive mean separation while driving the Competing Theory's between-subject variance to near zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t2_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    t3_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    t4_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    t6_mask = (a_tuples == (0, 1, 1, 1, 0)) & (b_tuples == (1, 0, 0, 0, 1))
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_b_t2 = (data.loc[t2_mask, 'response'] == 1).mean()
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    p_b_t4 = (data.loc[t4_mask, 'response'] == 1).mean()
    p_a_t6 = (data.loc[t6_mask, 'response'] == 0).mean()
    
    # Handle NaNs if a subject missed some trials
    p_a_t1 = 0.5 if pd.isna(p_a_t1) else p_a_t1
    p_b_t2 = 0.5 if pd.isna(p_b_t2) else p_b_t2
    p_a_t3 = 0.5 if pd.isna(p_a_t3) else p_a_t3
    p_b_t4 = 0.5 if pd.isna(p_b_t4) else p_b_t4
    p_a_t6 = 0.5 if pd.isna(p_a_t6) else p_a_t6
    
    p1 = (p_a_t1 + p_b_t2) / 2.0
    p3 = (p_a_t3 + p_b_t4) / 2.0
    p6 = p_a_t6
    
    return float(p1 + p3 + p6)
outcome: self_sim=2.0992 (var=0.1898) adversary_sim=1.8433 (var=0.0576) welch_t=+2.572 p=0.01423 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    # In this design, all trials consist of Option X vs Option Y\n    # Option X: [0, 1, 1, 1, 0, 0, 0] (Feature 1 is 0)\n    # Option Y: [1, 0, 0, 0, 1, 1, 1] (Feature 1 is 1)\n    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])\n    \n    # Check if the subject chose Option X\n    x_chosen = ((a_f1 == 0) & (data['response'] == 0)) | ((a_f1 == 1) & (data['response'] == 1))\n    \n    return float(x_chosen.mean())",
  "rationale": "This metric calculates the overall proportion of trials where the subject chooses Option X ([0, 1, 1, 1, 0, 0, 0]) over Option Y ([1, 0, 0, 0, 1, 1, 1]). Under the Competing Theory (Strategy Mixture), Option Y is structurally forced to win across all sub-models: Take-The-Best favors Y (wins feature 1), Tallying favors Y (4 wins vs 3), and WADD favors Y (feature 1's massive weight plus the tail outscales features 2-4 for all gamma >= 0). Thus, the Competing Theory strictly predicts Option Y, yielding a metric value near 0 (driven only by trembling-hand epsilon). In contrast, the Advocated Theory (Sequential Evidence Accumulation) uniquely predicts Option X for a substantial subset of subjects (those with a threshold between 1 and 2), because the unweighted tally reaches +2 in favor of Option X at feature 4, triggering an early stop. This creates a massive mean separation between the theories with highly stable variance."
}
```

## Usage

```json
{
  "prompt_token_count": 6986,
  "candidates_token_count": 459,
  "total_token_count": 10641
}
```
