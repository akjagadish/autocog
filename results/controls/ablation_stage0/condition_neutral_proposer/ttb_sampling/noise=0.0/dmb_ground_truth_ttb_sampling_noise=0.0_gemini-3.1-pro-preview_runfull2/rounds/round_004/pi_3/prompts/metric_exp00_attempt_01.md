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

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 9: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** Theory 1 (Take The Best) stops searching at the first discriminating cue and ignores all lower-ranked cues. Thus, its choice probability is constant (1 - epsilon/2) regardless of how many lower-ranked cues support or oppose the decision. Theory 2 (Exponentially Weighted Compensatory Model) computes a weighted sum of all cues and applies a softmax function. Because the weights grow exponentially, the highest-ranked cue still dictates the deterministic preference, but the magnitude of the value difference (and thus the softmax choice probability) is heavily modulated by the lower-ranked cues. By presenting trials where the top discriminating cue is held constant while the lower-ranked cues either heavily oppose or heavily support it, Theory 1 will predict a flat choice probability across these conditions, whereas Theory 2 will predict a graded choice probability.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Calculate the sum of features for each option\n    sum_a = data['option_a_ratings'].apply(sum)\n    sum_b = data['option_b_ratings'].apply(sum)\n    \n    # Trial 1: Option A has 1 positive feature, Option B has 4 positive features.\n    # This trial represents the maximum conflict where the highest validity cue favors A\n    # but all lower validity cues favor B. The value difference (v_a - v_b) is minimal.\n    mask_1 = (sum_a == 1) & (sum_b == 4)\n    \n    # Trial 4: Option A has 5 positive features, Option B has 0.\n    # This trial represents maximum alignment. The value difference (v_a - v_b) is maximal.\n    mask_4 = (sum_a == 5) & (sum_b == 0)\n    \n    if mask_1.sum() == 0 or mask_4.sum() == 0:\n        return 0.0\n        \n    # Calculate the proportion of times Option B was chosen (response == 1)\n    p_b_1 = data.loc[mask_1, 'response'].mean()\n    p_b_4 = data.loc[mask_4, 'response'].mean()\n    \n    # Take The Best predicts identical choice probabilities for both trials \n    # (driven entirely by the first cue and the lapse rate epsilon), so the expected difference is 0.\n    # The Exponentially Weighted Compensatory Model predicts a slightly higher probability \n    # of choosing B in Trial 1 due to the lower value difference when the softmax temperature (tau) is small.\n    return float(p_b_1 - p_b_4)",
  "rationale": "Under Take The Best (TTB), the decision is determined entirely by the first discriminating cue. In both Trial 1 and Trial 4, the highest-validity cue favors Option A. Thus, TTB predicts that Option B will be chosen at exactly the same rate (epsilon / 2) across both trials, yielding an expected difference of 0. Under the Exponentially Weighted Compensatory Model, the value difference (v_a - v_b) is strictly positive for both, but it is much smaller in Trial 1 (where lower cues heavily oppose A) than in Trial 4. Because of the softmax decision rule, smaller value differences result in choices closer to 50/50 when the temperature parameter (tau) is small. Therefore, the Competing Theory predicts a strictly positive difference (P(B|Trial 1) > P(B|Trial 4)). By isolating the single most extreme conflict trial and the single most extreme alignment trial, we maximize the theoretical contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 3809,
  "candidates_token_count": 641,
  "total_token_count": 27181
}
```
