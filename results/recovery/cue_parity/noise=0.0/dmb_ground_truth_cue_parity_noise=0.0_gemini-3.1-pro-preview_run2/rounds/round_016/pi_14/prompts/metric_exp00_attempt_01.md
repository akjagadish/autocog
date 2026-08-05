# metric_exp00_attempt_01

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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 12: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated Sequential Evidence Accumulation theory from the competing Leaky Competing Accumulator (LCA) theory by probing for early stopping behavior versus continuous parallel integration. The advocated theory posits that cues are evaluated sequentially and search stops completely if an accumulated evidence threshold is reached. This predicts that if a highly valid top cue provides enough evidence to cross the threshold, all remaining lower-validity cues are entirely ignored, resulting in identical choice probabilities regardless of how much opposing evidence they contain. The competing LCA theory, however, evaluates all cues in parallel. Its accumulators are continuously updated and mutually inhibit each other over time, meaning lower-validity opposing cues will always exert some inhibitory influence on the leading option's accumulator. By fixing the top cue to favor Option A and systematically increasing the number of lower-validity cues favoring Option B, the advocated theory predicts a perfectly flat choice probability curve (due to early stopping), whereas the competing LCA theory predicts a systematic, downward-sloping curve as the parallel opposing evidence mounts.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

**Parameters:**
- theta: [0.0, 10.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Decision-makers evaluate all cues in parallel using a Leaky Competing Accumulator (LCA) model. Evidence from each cue feeds into an option's accumulator proportional to a non-linear function of its validity above chance. The accumulators for the two options mutually inhibit each other and leak over time. This continuous, parallel competition naturally produces graded compensatory behavior: chance-level cues provide zero net evidence, while multiple weak opposing cues can slowly overcome a single strong cue's advantage through lateral inhibition.

**Parameters:**
- gamma: [0.1, 15.0]
- lambda_leak: [0.0, 5.0]
- beta_inhibit: [0.0, 5.0]
- tau: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    leak = float(parameters["lambda_leak"])
    inhibit = float(parameters["beta_inhibit"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear transformation of validity above chance
    weights = np.maximum(val - 0.5, 0.0) ** gamma
    
    # Total input evidence for each option
    I_A = np.sum(weights * a)
    I_B = np.sum(weights * b)
    
    # Simulate the Leaky Competing Accumulator
    x_A, x_B = 0.0, 0.0
    dt = 0.1
    for _ in range(50):
        dx_A = I_A - leak * x_A - inhibit * x_B
        dx_B = I_B - leak * x_B - inhibit * x_A
        
        x_A = max(0.0, x_A + dt * dx_A)
        x_B = max(0.0, x_B + dt * dx_B)
        
    scores = np.array([x_A, x_B])
    
    # Softmax choice rule
    z = tau * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
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
[0] rationale: This metric calculates the sensitivity of the decision-maker to the opposing evidence from lower-validity cues. By fixing the top cue favoring Option A and comparing the choice probability when Option B has no cues versus when Option B has all remaining cues, we can distinguish the two models. The Sequential Evidence Accumulation model predicts early stopping when the top cue is processed, leading to a flat choice probability curve (difference near 0). The Leaky Competing Accumulator evaluates all cues in parallel, so the addition of opposing cues for Option B will systematically shift choices toward B, leading to a large positive difference in response rates.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_sum'] = data['option_b_ratings'].apply(sum)
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        subj_diff = 0
        valid_a = 0
        for a_str in ['10000', '01000', '00100']:
            df_a = subj_df[subj_df['A_str'] == a_str]
            if df_a.empty: continue
            
            min_b = df_a['B_sum'].min()
            max_b = df_a['B_sum'].max()
            
            if min_b == max_b: continue
            
            resp_min = df_a[df_a['B_sum'] == min_b]['response'].mean()
            resp_max = df_a[df_a['B_sum'] == max_b]['response'].mean()
            
            subj_diff += (resp_max - resp_min)
            valid_a += 1
            
        if valid_a > 0:
            diffs.append(subj_diff / valid_a)
            
    if not diffs:
        return 0.0
    return float(np.mean(diffs))
outcome: self_sim=0.0700 (var=0.0422) adversary_sim=0.0200 (var=0.0432) welch_t=+0.856 p=0.3965 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    data = data.copy()\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    mask = (\n        ((data['A_str'] == '10000') & (data['B_str'] == '01111')) |\n        ((data['A_str'] == '01000') & (data['B_str'] == '00111')) |\n        ((data['A_str'] == '00100') & (data['B_str'] == '00011'))\n    )\n    \n    if mask.sum() == 0:\n        return 0.0\n        \n    subj_means = data[mask].groupby('subject_id')['response'].mean()\n    return float(subj_means.mean())\n",
  "rationale": "Instead of looking at the difference between minimum and maximum opposing evidence, this metric directly measures the absolute propensity to choose the compensatory option (Option B) when it possesses all available lower-validity cues against a single top cue (Option A). The Sequential Evidence Accumulation model, depending on the threshold and gamma, can completely flip its decision to Option B if the threshold is high enough to allow the sum of weaker cues to overcome the top cue. In contrast, the LCA model's lateral inhibition tends to suppress the lower-validity cues' accumulators once the top cue gives Option A an early lead, keeping the response rate for Option B consistently lower and more stable."
}
```

## Usage

```json
{
  "prompt_token_count": 4190,
  "candidates_token_count": 414,
  "total_token_count": 6802
}
```
