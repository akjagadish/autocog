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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Rationale:** This design quantitatively dissociates Weighted Additive (WADD) with exponentiated validities from Rank-Based Weighting (RBW) with Normalized Directional Cues by exploiting the mathematical properties of their weighting functions. RBW uses an exponentially decaying weight based on normalized ordinal rank (alpha^rank). Because exponential decay is strictly convex, RBW will ALWAYS assign a higher combined weight to the most extreme ranks (e.g., the 1st and 5th cues) than to intermediate ranks (e.g., the 2nd and 4th cues), regardless of the alpha parameter. In contrast, WADD scales the raw validities by an exponent (gamma). When gamma < 1, the WADD weighting function becomes concave with respect to the raw validities, assigning higher relative weight to intermediate validities compared to the extremes. By constructing trials where Option A has the highest and lowest validities and Option B has the intermediate validities, RBW is mathematically forced to predict Option A, while WADD will predict Option B for gamma < 1. We also include trials where both models can flip their preferences depending on their respective parameters, ensuring full identifiability of the scaling behaviors.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People integrate information from multiple cues by computing a weighted sum of the feature values, where the weights are a non-linear transformation of the cues' validities. This Weighted Additive (WADD) strategy is compensatory but incorporates a scaling parameter (gamma) that can exponentiate the validities. This allows the decision-maker to dynamically re-balance attention—either steepening the weights to behave more like Take The Best, or flattening them to behave more like Tallying—to match the empirical balance of conflict resolution.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 5.0]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    # Weighted Additive (WADD) heuristic with exponentiated validities.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Exponentiate validities to allow non-linear scaling of weights
    weights = val ** gamma
    
    # Compute the weighted sum of features for each option
    # using the scaled cue validities as weights.
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Rank-Based Weighting with Normalized Directional Cues: Decision-makers evaluate cues based on their informational value (distance from 0.5 validity) rather than raw validity. They sort cues by this importance and assign weights that decay exponentially based on normalized ordinal rank (w = direction * alpha ^ normalized_rank). Normalizing ranks to the [0, 1] interval ensures that the decay profile is invariant to the total number of features in the environment, allowing the same cognitive strategy to scale robustly across contexts with varying dimensionality.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute cue importance (distance from 0.5) and direction
    importance = np.abs(val - 0.5)
    direction = np.sign(val - 0.5)
    
    # Calculate ranks (0 is highest importance, 1 is second highest, etc.)
    order = np.argsort(-importance, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(val), dtype=float)
    
    # Normalize ranks to [0, 1] so that decay is invariant to the number of features
    if len(val) > 1:
        normalized_ranks = ranks / (len(val) - 1)
    else:
        normalized_ranks = ranks
    
    alpha = float(parameters["alpha"])
    # Compute weights: direction * alpha^normalized_rank.
    weights = direction * (alpha ** normalized_ranks)
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
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
[0] rationale: In the experimental design, there are three specific trials where Option A's cues have more extreme validities than Option B's: Trial 1 (A={1,5}, B={2,4}), Trial 2 (A={1,4}, B={2,3}), and Trial 6 (A={2,5}, B={3,4}). Because the Rank-Based Weighting (RBW) theory computes weights using exponential decay based on normalized ranks, its weighting function is strictly convex. This strict convexity mathematically guarantees that RBW will always assign a higher overall value to Option A in these trials, regardless of its alpha parameter, leading to a near-zero rate of choosing Option B (only non-zero due to epsilon noise). In contrast, the Weighted Additive (WADD) theory exponentiates raw validities. When gamma < 1, the weighting function becomes concave, which assigns relatively higher weight to intermediate cues, causing WADD to predict Option B. This metric calculates the proportion of times subjects choose Option B on these three specific trials, effectively exploiting the mathematical dissociation between the convex (RBW) and concave (WADD) weighting functions.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    targets = {'10001', '10010', '01001'}
    mask = a_str.isin(targets)
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
outcome: self_sim=0.3625 (var=0.0233) adversary_sim=0.3717 (var=0.0224) welch_t=-0.214 p=0.8311 (N=25, alpha=0.01) -> reject

[1] rationale: This metric exploits the second derivative (convexity) of the weighting functions to perfectly dissociate the models. It compares the rate of choosing Option B on Trial 6 vs. Trial 1. In Trial 1, the cues are spaced as (1st, 5th) vs (2nd, 4th). In Trial 6, they are (2nd, 5th) vs (3rd, 4th). For Rank-Based Weighting (RBW), the difference in score differences between these two trials is exactly (1-x)^2, where x is the exponential decay factor. Because exponential decay is highly convex, this difference is strictly positive and large, meaning RBW will consistently choose Option A more often in Trial 1 than in Trial 6 (yielding a positive difference in Option B rates). For Weighted Additive (WADD), the difference in score differences is the second difference of the exponentiated validities: 0.95^g - 2*(0.85^g) + 0.75^g. Because the raw validities are tightly clustered, this second difference is mathematically forced to be vanishingly small (near zero) for the entire plausible range of gamma. Consequently, WADD treats Trial 1 and Trial 6 almost identically, yielding a metric value near 0 with extremely low between-subject variance. This guarantees a highly significant Welch's t-test.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['a_tuple'] = data['option_a_ratings'].apply(tuple)
    t6_mask = data['a_tuple'] == (0, 1, 0, 0, 1)
    t1_mask = data['a_tuple'] == (1, 0, 0, 0, 1)
    
    if not t6_mask.any() or not t1_mask.any():
        return 0.0
        
    t6_resp = data.loc[t6_mask, 'response'].mean()
    t1_resp = data.loc[t1_mask, 'response'].mean()
    
    return float(t6_resp - t1_resp)
outcome: self_sim=0.0338 (var=0.0244) adversary_sim=0.0100 (var=0.0290) welch_t=+0.514 p=0.6096 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['a_tuple'] = data['option_a_ratings'].apply(tuple)\n    t4_mask = data['a_tuple'] == (1, 0, 1, 0, 0)\n    t2_mask = data['a_tuple'] == (1, 0, 0, 1, 0)\n    \n    if not t4_mask.any() or not t2_mask.any():\n        return 0.0\n        \n    t4_b_rate = data.loc[t4_mask, 'response'].mean()\n    t2_b_rate = data.loc[t2_mask, 'response'].mean()\n    \n    return float(t4_b_rate - t2_b_rate)",
  "rationale": "WADD and RBW differ in their preference for Option B across trials due to their weighting functions. In Trial 4 (A=[1,0,1,0,0] vs B=[0,1,0,1,1]), WADD prefers B only when gamma < 1.9 (approx 37% of its parameter space), whereas RBW prefers B when alpha > 0.35 (approx 65% of its parameter space). In Trial 2 (A=[1,0,0,1,0] vs B=[0,1,1,0,0]), WADD prefers B when gamma < 1.0 (approx 20% of the space), while RBW mathematically never prefers B regardless of alpha, because the exponential decay of ranks makes the 1st and 4th cues always outweigh the 2nd and 3rd. By subtracting the rate of choosing B in Trial 2 from the rate of choosing B in Trial 4, we isolate this difference. RBW is expected to show a large positive difference because it switches from strongly preferring A in Trial 2 to often preferring B in Trial 4. WADD is expected to show a much smaller difference because its preferences shift less dramatically between these two trials. This metric effectively discriminates the theories while remaining robust to random choice noise."
}
```

## Usage

```json
{
  "prompt_token_count": 4503,
  "candidates_token_count": 490,
  "total_token_count": 15754
}
```
