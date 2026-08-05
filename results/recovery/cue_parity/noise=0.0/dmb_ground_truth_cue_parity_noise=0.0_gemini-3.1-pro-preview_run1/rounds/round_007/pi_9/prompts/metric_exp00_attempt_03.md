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
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.7, 0.6]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Additive Utility with power-law validities), we exploit how the Advocated Theory accumulates *unweighted* evidence differences. Because the Advocated Theory merely counts +1 or -1 in validity order, it can never differentiate between the magnitudes of features once the order is set, except by stopping early. By constructing trials where the running tally of differences fluctuates but never reaches a high threshold, we force the Advocated Theory to either guess (if it reaches the end with a tie) or choose the Take-The-Best option (if the threshold is low). In contrast, the Competing Theory weights features by their chance-centered validities, allowing combinations of slightly less valid features to overpower the single best feature, predicting a strong preference for an option that the Advocated Theory can literally never choose under any parameterization.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



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
**Description:** People use a single-stage Additive Utility evaluation where features are weighted by a power-law transformation of their chance-centered objective validities. By avoiding artificial weight normalization and allowing for sufficiently high softmax temperatures and power parameters, the decision-making process can smoothly and stably interpolate between Tallying (equal weights), proportional weighting, and highly deterministic Take-The-Best behavior (where the most valid cues dominate completely).

**Parameters:**
- beta: [0.1, 50.0]
- gamma: [0.0, 10.0]
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
    gamma = float(parameters["gamma"])
    
    # Transform validities using a simple power law on chance-centered values
    # gamma=0 -> Tallying, gamma=1 -> WADD, gamma>1 -> TTB
    centered_val = val - 0.5
    weights = (centered_val + 1e-6) ** gamma
    
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
[0] rationale: Focuses on Trial 3 where A=[1, 0, 1, 0, 0] and B=[0, 1, 0, 1, 1]. For the Advocated Theory, the running tally never exceeds 1, and the final tally favors B. Thus, it only chooses A if the stopping threshold is <= 1 (approx 33% of the parameter space). For the Competing Theory, the power-law weighting of the validities strongly favors A for almost all gamma values > 0.75 (approx 92% of the parameter space). This creates a stark, stable contrast in the probability of choosing Option A.
metric_source:
def metric(data: pd.DataFrame) -> float:
    is_t3 = data['option_a_ratings'].apply(tuple) == (1, 0, 1, 0, 0)
    return float((data.loc[is_t3, 'response'] == 0).mean())
outcome: self_sim=0.4525 (var=0.1360) adversary_sim=0.5867 (var=0.0467) welch_t=-1.569 p=0.1247 (N=25, alpha=0.01) -> reject

[1] rationale: Focuses on Trial 4 where A=[0, 0, 1, 1, 1] and B=[1, 1, 0, 0, 0]. The Competing Theory almost universally prefers Option B because it possesses the two most highly valid cues, which easily outweigh the three least valid cues under the power-law parameterization for almost all gamma values. In contrast, the Advocated Theory's sequential accumulation process reaches a final tally in favor of A if the stopping threshold is greater than 2. Since the threshold parameter spans [0, 3], the Advocated Theory will choose Option A roughly one-third of the time (whenever threshold > 2). This structural difference predicts a significantly higher probability of choosing B under the Competing Theory compared to the Advocated Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    is_t4 = data['option_a_ratings'].apply(tuple) == (0, 0, 1, 1, 1)
    return float((data.loc[is_t4, 'response'] == 1).mean())
outcome: self_sim=0.6292 (var=0.1336) adversary_sim=0.6142 (var=0.0517) welch_t=+0.174 p=0.8625 (N=25, alpha=0.01) -> reject

[2] rationale: We calculate the difference in the probability of choosing Option A between Trial 2 and Trial 1. Under the Advocated Theory (Sequential Evidence Accumulation), the threshold parameter determines the stopping point. In Trial 1, the model chooses A if threshold <= 1, and guesses if threshold > 1, yielding an expected P(A) of ~66%. In Trial 2, the model chooses B if threshold <= 1, and A if threshold > 1, also yielding an expected P(A) of ~66%. Thus, the difference P(A|T2) - P(A|T1) perfectly cancels out to 0 on average. Conversely, under the Competing Theory (Additive Utility), Option A is generally disfavored in Trial 1 (negative weighted sum for most gamma values) and favored in Trial 2 (positive weighted sum for most gamma values). This leads to a strictly positive difference, creating a highly stable and structural contrast between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 1)
    is_t2 = data['option_a_ratings'].apply(tuple) == (0, 1, 1, 1, 0)
    
    p_a_t1 = (data.loc[is_t1, 'response'] == 0).mean()
    p_a_t2 = (data.loc[is_t2, 'response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t1)
outcome: self_sim=-0.0875 (var=0.3336) adversary_sim=0.2533 (var=0.1426) welch_t=-2.470 p=0.01774 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    is_t2 = data['option_a_ratings'].apply(tuple) == (0, 1, 1, 1, 0)\n    is_t3 = data['option_a_ratings'].apply(tuple) == (1, 0, 1, 0, 0)\n    \n    p_a_t2 = (data.loc[is_t2, 'response'] == 0).mean()\n    p_a_t3 = (data.loc[is_t3, 'response'] == 0).mean()\n    \n    return float(p_a_t2 + p_a_t3)",
  "rationale": "Calculates the sum of the probability of choosing Option A in Trial 2 and Trial 3. Under the Advocated Theory (Sequential Evidence Accumulation), a threshold <= 1 leads to choosing B in T2 and A in T3, while a threshold > 1 leads to choosing A in T2 and B in T3. Thus, for any threshold, the expected sum of P(A|T2) + P(A|T3) is exactly 1.0, resulting in extremely low between-subject variance. Under the Competing Theory (Additive Utility), moderate values of gamma (e.g., gamma around 1 to 3) lead to choosing A in both T2 and T3 due to the power-law weighting of cues, pushing the sum significantly higher than 1.0. This creates a large mean difference and exploits the structural invariance in the Advocated Theory to guarantee a highly significant Welch's t-test."
}
```

## Usage

```json
{
  "prompt_token_count": 4170,
  "candidates_token_count": 374,
  "total_token_count": 8981
}
```
