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
**Validities (n_features=5):** [0.92, 0.85, 0.81, 0.65, 0.55]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Rationale:** To dissociate Feature Cancellation then Tally/TTB (advocated) from Similarity-Scaled WADD (competing), we exploit how each model integrates cues and resolves conflicting information across varying levels of stimulus complexity. Both models scale determinism by the number of differing features, but rely on fundamentally different core decision rules. Trial 1 pits a 2-cue WADD winner against a 3-cue Tally winner (5 differing features). Trial 2 presents a perfectly tied tally where the TTB tie-breaker favors Option A, but the compensatory WADD sum favors Option B (4 differing features). Trial 3 reduces the number of differing features to 3 by introducing shared features, again pitting a 1-cue WADD winner against a 2-cue Tally winner. This design forces the models to predict opposite choices across varying complexity penalties, providing a robust quantitative dissociation.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Feature Cancellation then Tally/TTB with Cancellation-Scaled Determinism

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Feature cancellation: identify surviving distinct features
    mask = a != b
    
    # Tallying on distinct features
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    score_a = tally_a
    score_b = tally_b
    
    # If tallies are tied, break the tie using TTB on surviving features
    if tally_a == tally_b and np.sum(mask) > 0:
        order = np.argsort(val)[::-1]
        for idx in order:
            if mask[idx]:
                if a[idx] > b[idx]:
                    score_a += 1.0
                elif b[idx] > a[idx]:
                    score_b += 1.0
                break
            
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with cancellation-scaled determinism
    n_surviving = np.sum(mask)
    effective_beta = beta / n_surviving if n_surviving > 0 else beta
    
    z = effective_beta * (scores - np.max(scores))
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


## COMPETING THEORY
**Description:** Similarity-Scaled Weighted Additive (WADD) Theory: Decision-makers evaluate options using a fully compensatory weighted additive process, where cue weights are derived from a non-linear transformation of their stated validities (parameterized log-odds). Crucially, the choice determinism is not constant; it scales inversely with stimulus complexity (the number of differing features between the options). When options differ on many features, cognitive load increases and preferences become noisier. This complexity-penalized determinism explains variance in choice consistency and nuanced preference reversals across experiments. Normalizing the weights to sum to 1 ensures that the WADD scores remain on a consistent, bounded scale regardless of theta or the number of features, allowing the beta parameter to apply a stable level of determinism across all experiments before the complexity penalty is applied.

**Parameters:**
- beta: [0.1, 20.0]
- gamma: [0.0, 3.0]
- theta: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear transformation of validities (log-odds raised to theta)
    # Clipped to avoid log(0) or division by zero
    v_clipped = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(v_clipped / (1.0 - v_clipped))
    
    # Apply non-linear scaling
    w = log_odds ** theta
    
    # Normalize weights so the sum of weights is 1.0
    # This keeps the scores on a predictable bounded scale for beta across different n_features
    if np.sum(w) > 0:
        w = w / np.sum(w)
        
    # Compensatory WADD score calculation
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    # Similarity-scaled determinism: penalize beta based on number of differing features
    n_diff = np.sum(a != b)
    if n_diff > 0:
        effective_beta = beta / (n_diff ** gamma)
    else:
        effective_beta = beta
        
    # Softmax choice rule with numerical stability
    z = effective_beta * (scores - np.max(scores))
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    def get_sig(lst):\n        return \"\".join(str(int(x)) for x in lst)\n        \n    data['A_str'] = data['option_a_ratings'].apply(get_sig)\n    \n    t1_sig = \"11000\"\n    t2_sig = \"10001\"\n    t3_sig = \"11100\"\n    \n    score = 0.0\n    \n    t1_data = data[data['A_str'] == t1_sig]\n    if len(t1_data) > 0:\n        score += np.mean(t1_data['response'] == 1)\n        \n    t2_data = data[data['A_str'] == t2_sig]\n    if len(t2_data) > 0:\n        score += np.mean(t2_data['response'] == 0)\n        \n    t3_data = data[data['A_str'] == t3_sig]\n    if len(t3_data) > 0:\n        score += np.mean(t3_data['response'] == 1)\n        \n    return float(score)",
  "rationale": "The metric computes a composite score based on the predicted choices for Trials 1, 2, and 3. The Advocated model (Feature Cancellation then Tally/TTB) consistently predicts B for Trial 1, A for Trial 2, and B for Trial 3, yielding a score near 3.0. The Competing model (Similarity-Scaled WADD) generally predicts the exact opposite (A, B, A) across most parameter regimes, yielding a score near 0.0. Even under extreme parameter values for WADD, its maximum possible expected score is 2.0, ensuring a robust discriminative gap."
}
```

## Usage

```json
{
  "prompt_token_count": 3478,
  "candidates_token_count": 460,
  "total_token_count": 7145
}
```
