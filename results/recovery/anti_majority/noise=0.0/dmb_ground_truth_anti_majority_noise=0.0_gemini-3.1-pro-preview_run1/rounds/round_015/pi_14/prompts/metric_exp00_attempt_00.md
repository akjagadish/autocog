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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  trial 7: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Thresholded Unique Features with Spread Penalty) from the Competing Theory (Diminishing Marginal Utility with Baseline Cost and Loss Aversion), we exploit their fundamentally different treatments of shared features and feature dispersion. The Advocated Theory strictly cancels shared features and evaluates the remaining unique features additively, applying a penalty based on their spread (max - min). The Competing Theory evaluates all features (no cancellation) and applies exponential discounting to features based on their rank, while also amplifying negative features via a loss aversion multiplier. We design base trials pitting options with different spreads and single features against coalitions. We then introduce shared features (both high and low validity). For the Advocated Theory, shared features cancel out, leaving choice probabilities completely unchanged. For the Competing Theory, adding a shared high-validity feature alters the rank of the unique features (increasing the discount on them), and adding a shared low-validity feature introduces a potential negative utility that is amplified by loss aversion, leading to strong preference shifts.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Thresholded Unique Features with Spread Penalty: Decision-makers simplify choices by cancelling out shared features, then evaluate the unique features relative to a subjective validity threshold. Features above the threshold provide positive evidence, while those below act as penalties. These values are integrated additively, but options with multiple unique features suffer a conflict penalty proportional to the spread (max - min) of their thresholded validities. This penalizes options with a wide variance in their unique features while strictly preserving shared-feature cancellation.

**Parameters:**
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_penalty: [0.0, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Transform validities and apply subjective threshold
    v_trans = val ** gamma
    w = v_trans - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # Additive integration of thresholded evidence
        base_score = np.sum(active_w)
        
        # Spread penalty applied if there are multiple unique features
        if len(active_w) > 1:
            conflict_penalty = lambda_penalty * (np.max(active_w) - np.min(active_w))
            return base_score - conflict_penalty
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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


## COMPETING THEORY
**Description:** Diminishing Marginal Utility with Baseline Cost, Independent Discounting, and Loss Aversion: Decision-makers evaluate all features of an option. Each feature's validity is transformed and reduced by a baseline cost 'C'. Features above 'C' provide positive utility, while those below act as negative utility (penalties). To reflect limited attention, positive and negative features are sorted independently by their magnitude and discounted exponentially based on their rank. Furthermore, a loss aversion multiplier amplifies the impact of negative features, explaining why options with multiple weak features are strongly penalized and consistently lose to strict subsets of strong features.

**Parameters:**
- gamma: [0.1, 10.0]
- delta: [0.0, 1.0]
- c: [0.0, 1.0]
- weight_neg: [0.1, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    delta = float(parameters["delta"])
    c = float(parameters["c"])
    weight_neg = float(parameters["weight_neg"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # DO NOT cancel shared features; evaluate the full feature set
    
    # Transform validities and subtract baseline cost C
    v_trans = (val ** gamma) - c
    
    def get_score(x):
        active_v = v_trans[x > 0]
        if len(active_v) == 0:
            return 0.0
        
        # Separate positive and negative evidence
        pos_v = active_v[active_v > 0]
        neg_v = active_v[active_v < 0]
        
        score = 0.0
        
        if len(pos_v) > 0:
            # Sort positive features descending by magnitude
            pos_v_sorted = np.sort(pos_v)[::-1]
            ranks = np.arange(len(pos_v_sorted))
            score += np.sum(pos_v_sorted * (delta ** ranks))
            
        if len(neg_v) > 0:
            # Sort negative features ascending (most negative first)
            neg_v_sorted = np.sort(neg_v)
            ranks = np.arange(len(neg_v_sorted))
            score += weight_neg * np.sum(neg_v_sorted * (delta ** ranks))
            
        return score
        
    score_a = get_score(a)
    score_b = get_score(b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n\n    data = data.copy()\n    \n    # Create a string representation of option A's features to uniquely identify the trial type\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: \"\".join(str(int(v)) for v in x))\n    \n    # Calculate P(choose A) which corresponds to response == 0\n    data['chose_A'] = (data['response'] == 0).astype(float)\n    \n    # Group by A_str to get mean P(A)\n    p_A = data.groupby('A_str')['chose_A'].mean().to_dict()\n    \n    # Safely get probabilities, defaulting to 0.5 if missing (e.g. in small samples)\n    p1 = p_A.get(\"01001\", 0.5)\n    p2 = p_A.get(\"11001\", 0.5)\n    p3 = p_A.get(\"10010\", 0.5)\n    p4 = p_A.get(\"10011\", 0.5)\n    p5 = p_A.get(\"10000\", 0.5)\n    p6 = p_A.get(\"10001\", 0.5)\n    p7 = p_A.get(\"01010\", 0.5)\n    p8 = p_A.get(\"11010\", 0.5)\n    \n    # Calculate the differences in P(A) between trial pairs that differ only by a shared feature\n    diff1 = p1 - p2\n    diff2 = p3 - p4\n    diff3 = p5 - p6\n    diff4 = p7 - p8\n    \n    # The metric is the sum of absolute differences\n    return float(abs(diff1) + abs(diff2) + abs(diff3) + abs(diff4))\n",
  "rationale": "The Advocated Theory posits that decision-makers cancel out shared features before evaluating options. Thus, adding the exact same feature to both Option A and Option B should result in no change to the choice probabilities. The Competing Theory evaluates all features and applies rank-based discounting; adding a shared feature alters the rank of the unique features, leading to shifts in choice probabilities. By pairing trials that are identical except for the addition of a shared feature (e.g., Trial 1 vs Trial 2) and summing the absolute differences in the probability of choosing Option A, we expect the Advocated Theory to yield a value near 0, whereas the Competing Theory will yield a significantly higher value."
}
```

## Usage

```json
{
  "prompt_token_count": 3880,
  "candidates_token_count": 664,
  "total_token_count": 7086
}
```
