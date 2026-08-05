# metric_exp02_attempt_02

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

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design dissociates Rank-Based Weighting (RBW) from Weighted Additive (WADD) using a 5-feature environment. RBW evaluates cues based on their ordinal rank of importance and applies an exponentially decaying weight. Because this decay is strictly convex, RBW will consistently favor options possessing a combination of high-ranked and low-ranked cues over options with intermediate-ranked cues (e.g., ranks 0 and 3 vs. ranks 1 and 2), regardless of its alpha parameter. In contrast, WADD scales raw validities by an exponent gamma. For gamma < 1, WADD's weighting function becomes concave over raw validities, causing it to assign higher relative weight to intermediate validities and thus predicting the opposite choice. By including trials that pit extreme-ranked cues against intermediate ones, we force a qualitative divergence where WADD predicts preference reversals that RBW mathematically cannot produce.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the sum of choice proportions for the options mathematically guaranteed to be preferred by Rank-Based Weighting (RBW) due to the strict convexity of its exponential decay function. Across trials 1, 2, 3, and 4, RBW consistently prefers the extreme-ranked combinations over intermediate ones. In contrast, Weighted Additive (WADD) behavior depends on its gamma parameter: for gamma < 1, WADD prefers the exact opposite choices (the intermediate-ranked combinations) due to concavity, while for gamma > 1, it aligns with RBW. Averaged across a uniform prior over their respective parameter spaces, RBW will score near the maximum (4.0, minus epsilon noise), whereas WADD will score around half that (since it splits between 0 and 4 depending on gamma). This creates a massive and stable gap between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    score = 0.0
    
    t1 = data[data['a_str'] == '10010']
    if len(t1) > 0:
        score += (1.0 - t1['response']).mean()
        
    t2 = data[data['a_str'] == '01001']
    if len(t2) > 0:
        score += (1.0 - t2['response']).mean()
        
    t3 = data[data['a_str'] == '11001']
    if len(t3) > 0:
        score += (1.0 - t3['response']).mean()
        
    t4 = data[data['a_str'] == '01100']
    if len(t4) > 0:
        score += t4['response'].mean()
        
    return score
outcome: self_sim=2.4900 (var=0.3654) adversary_sim=2.3188 (var=0.3158) welch_t=+1.037 p=0.3048 (N=25, alpha=0.01) -> reject

[1] rationale: This metric contrasts the choice probabilities on trials 2 and 3 against trial 5 to maximally separate the theories. Trials 2 and 3 test for convexity: RBW is mathematically always convex and thus always prefers Option A. WADD is convex for gamma > 1 (which is ~82% of its prior) and concave for gamma < 1 (where it prefers B). Trial 5 tests the tail drop-off of the weights. Because RBW uses geometric decay, its weights drop off slower than WADD's exponentiated validities for matched intermediate weights. Consequently, RBW prefers Option B on trial 5 for a much wider range of its parameter space (alpha > 0.36, or 64% of the time) compared to WADD (gamma < 1.93, or 37% of the time). By computing `P(A|t2) + P(A|t3) - 2 * P(A|t5)`, we isolate this divergence. For RBW, this value is highly positive (~1.28 on average) because it consistently chooses A in t2/t3 and frequently chooses B in t5. For WADD, the value is much lower (~0.38 on average) because when it chooses A in t2/t3 (gamma > 1), it also frequently chooses A in t5 (gamma > 1.93), causing the terms to cancel out. This structural difference creates a strong, stable gap between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    # Convert list of ratings to a string for safe filtering
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trial 2: A=[0, 1, 0, 0, 1]
    t2 = data[data['a_str'] == '01001']
    p_a_t2 = (1.0 - t2['response']).mean() if len(t2) > 0 else 0.0
    
    # Trial 3: A=[1, 1, 0, 0, 1]
    t3 = data[data['a_str'] == '11001']
    p_a_t3 = (1.0 - t3['response']).mean() if len(t3) > 0 else 0.0
    
    # Trial 5: A=[1, 0, 1, 0, 0]
    t5 = data[data['a_str'] == '10100']
    p_a_t5 = (1.0 - t5['response']).mean() if len(t5) > 0 else 0.0
    
    # Contrast the consistent preference for A in t2 and t3 against the divergence in t5
    return float(p_a_t2 + p_a_t3 - 2.0 * p_a_t5)
outcome: self_sim=0.2612 (var=0.2239) adversary_sim=0.1975 (var=0.2785) welch_t=+0.450 p=0.655 (N=25, alpha=0.01) -> reject

[2] rationale: This metric computes the difference between the probability of choosing Option A on the 'convexity' trials (T1, T2, T3) and the 'tail-dropoff' trial (T5). For RBW, the exponential decay is strictly convex, mathematically guaranteeing a preference for A in T1, T2, and T3 across the entire parameter space. However, in T5, RBW prefers A only for steep decay (alpha < 0.36), which represents 36% of its prior. Thus, RBW's expected metric value is roughly 1.0 - 0.36 = 0.64. For WADD, convexity only holds for gamma > 1 (82% of the prior), so it prefers A in T1, T2, and T3 82% of the time. In T5, WADD prefers A for gamma > 1.93 (63% of the prior). Because gamma > 1.93 is a strict subset of gamma > 1, the difference perfectly isolates the narrow parameter region 1.0 < gamma < 1.93. The expected metric value for WADD is roughly 0.82 - 0.63 = 0.19. This creates a large and highly stable gap (0.64 vs 0.19) by exploiting the structural differences in convexity and tail-dropoff between the models, while averaging across T1-T3 to minimize within-subject noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1 = data[data['a_str'] == '10010']
    t2 = data[data['a_str'] == '01001']
    t3 = data[data['a_str'] == '11001']
    
    p_a_t1 = (1.0 - t1['response']).mean() if len(t1) > 0 else 0.0
    p_a_t2 = (1.0 - t2['response']).mean() if len(t2) > 0 else 0.0
    p_a_t3 = (1.0 - t3['response']).mean() if len(t3) > 0 else 0.0
    
    p_a_convex = (p_a_t1 + p_a_t2 + p_a_t3) / 3.0
    
    t5 = data[data['a_str'] == '10100']
    p_a_t5 = (1.0 - t5['response']).mean() if len(t5) > 0 else 0.0
    
    return float(p_a_convex - p_a_t5)
outcome: self_sim=0.1408 (var=0.0513) adversary_sim=0.0971 (var=0.0616) welch_t=+0.651 p=0.5181 (N=25, alpha=0.01) -> reject

[3] rationale: This metric evaluates the probability that a subject's response on the 'convexity' trials (T1, T2, T3) matches their response on the 'tail-dropoff' trial (T5). For RBW, the exponential decay is strictly convex, guaranteeing a preference for A in T1-T3 across the entire parameter space. However, in T5, RBW prefers A only for steep decay (alpha < 0.36, representing 36% of its prior). Thus, RBW only matches responses (A and A) 36% of the time. For WADD, convexity only holds for gamma > 1 (82% of the prior), so it prefers A in T1-T3 82% of the time. In T5, WADD prefers A for gamma > 1.93 (63% of the prior). Crucially, WADD will ALSO match responses when gamma < 1 (18% of the prior), where it prefers B in both the convexity trials and T5. Because gamma > 1.93 is a strict subset of gamma > 1, WADD's total match rate is roughly 63% (A and A) + 18% (B and B) = 81%. This creates a massive gap (81% for WADD vs 36% for RBW) by looking at the joint distribution of choices across trials, effectively neutralizing the noise that hurt simple subtractive metrics.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1 = data[data['a_str'] == '10010']
    t2 = data[data['a_str'] == '01001']
    t3 = data[data['a_str'] == '11001']
    t5 = data[data['a_str'] == '10100']
    
    if len(t5) == 0:
        return 0.0
        
    p_a_t1 = (1.0 - t1['response']).mean() if len(t1) > 0 else 0.5
    p_a_t2 = (1.0 - t2['response']).mean() if len(t2) > 0 else 0.5
    p_a_t3 = (1.0 - t3['response']).mean() if len(t3) > 0 else 0.5
    
    # Average probability of choosing A on the 'convexity' trials
    p_a_convex = (p_a_t1 + p_a_t2 + p_a_t3) / 3.0
    p_a_t5 = (1.0 - t5['response']).mean()
    
    # Calculate the rate of matching preferences between the convexity trials and the tail-dropoff trial (T5)
    return float(p_a_convex * p_a_t5 + (1.0 - p_a_convex) * (1.0 - p_a_t5))
outcome: self_sim=0.4997 (var=0.0169) adversary_sim=0.5065 (var=0.0138) welch_t=-0.195 p=0.8465 (N=25, alpha=0.01) -> reject

[4] rationale: This metric evaluates the difference in choice proportions between Trial 2 (A=[0, 1, 1, 1]) and Trial 1 (A=[1, 0, 0, 1]). WADD structurally guarantees that Option A is always preferred in Trial 2 across its entire parameter space, meaning `resp_t2` is essentially 0. In Trial 1, WADD occasionally prefers B (when gamma < 1), making `resp_t1` higher than `resp_t2`, so the difference is negative. Conversely, RBW mathematically guarantees that Option A is always preferred in Trial 1 due to strict convexity, making `resp_t1` 0. However, in Trial 2, RBW prefers Option B when decay is steep (alpha < 0.16), making `resp_t2` higher than `resp_t1`, so the difference is positive. This structural inversion perfectly separates the models with very low between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1 = data[data['a_str'] == '1001']
    t2 = data[data['a_str'] == '0111']
    
    resp_t1 = t1['response'].mean() if len(t1) > 0 else 0.5
    resp_t2 = t2['response'].mean() if len(t2) > 0 else 0.5
    
    return float(resp_t2 - resp_t1)
outcome: self_sim=-0.0692 (var=0.1035) adversary_sim=-0.1992 (var=0.0471) welch_t=+1.675 p=0.1014 (N=25, alpha=0.01) -> reject

[5] rationale: This metric measures the difference in the probability of choosing Option B between Trial 4 (A=[0,1,0,0], B=[0,0,1,1]) and Trial 2 (A=[0,1,1,1], B=[1,0,0,0]). For WADD, Option A is strongly preferred in Trial 2 across the entire parameter space, while Option B is strongly preferred in Trial 4 for the vast majority of the gamma prior (<4.5). Thus, P(B|T4) is high and P(B|T2) is low, resulting in a large positive difference. For RBW, the preferences depend on the decay parameter alpha. For alpha > 0.2, RBW behaves similarly to WADD. However, for steep decay (alpha < 0.2), RBW structurally flips its preferences: it prefers Option B in Trial 2 and Option A in Trial 4, producing a negative difference. This subset of the parameter space significantly pulls down RBW's expected value for this metric compared to WADD, creating a reliable and robust discrimination between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t2 = data[data['a_str'] == '0111']
    t4 = data[data['a_str'] == '0100']
    
    p_b_t2 = t2['response'].mean() if len(t2) > 0 else 0.5
    p_b_t4 = t4['response'].mean() if len(t4) > 0 else 0.5
    
    return float(p_b_t4 - p_b_t2)
outcome: self_sim=0.4075 (var=0.1875) adversary_sim=0.4833 (var=0.0833) welch_t=-0.729 p=0.4703 (N=25, alpha=0.01) -> reject

[6] rationale: While the two models have priors that perfectly align their rank-order preferences across the four trial types (making mean choice probabilities nearly indistinguishable), they differ significantly in the MAGNITUDE of their internal score differences. Specifically on Trial 1 (A=[1,0,0,1], B=[0,1,1,0]), WADD's score difference maxes out at a very small 0.21 (and frequently crosses near zero for gamma < 1), causing it to behave largely like random guessing. In contrast, RBW's score difference on Trial 1 can reach up to 1.0, and its mathematically expected score difference over the prior is nearly double that of WADD. By measuring the within-subject determinism (squared distance from 50/50 guessing) on Trial 1, we can cleanly separate the models: RBW will show significantly higher determinism because its larger score differences push softmax probabilities closer to the extremes, whereas WADD remains trapped near 0.5.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1 = data[data['a_str'] == '1001']
    if len(t1) == 0:
        return 0.0
        
    means = t1.groupby('subject_id')['response'].mean()
    determinism = 4.0 * ((means - 0.5) ** 2)
    
    return float(determinism.mean())
outcome: self_sim=0.2131 (var=0.0562) adversary_sim=0.1851 (var=0.0573) welch_t=+0.414 p=0.6805 (N=25, alpha=0.01) -> reject

[7] rationale: This metric exploits a deep structural invariant in the discrete derivatives of the weighting functions. By mapping the empirical choice probabilities back to log-odds (using a clipped logit function to avoid infinities), we recover a proxy for the internal score differences (D_i). For WADD, the weights are given by v^gamma. Because the validities are linearly spaced, the discrete second derivative of the score differences (which corresponds to the linear combination D3 - D2 - 2*D4) is mathematically forced to be extremely close to zero across its entire parameter space. In contrast, RBW's weights follow a geometric progression (alpha^(k/3)), making this same discrete derivative strictly positive and significantly larger. By evaluating this specific linear combination of log-odds, we effectively factor out the noise and the softmax temperature, yielding a metric that evaluates to ~0 for WADD and a large positive number for RBW.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # T2: A=[0, 1, 1, 1], B=[1, 0, 0, 0]
    t2 = data[data['a_str'] == '0111']
    # T3: A=[1, 0, 1, 0], B=[0, 1, 0, 1]
    t3 = data[data['a_str'] == '1010']
    # T4: A=[0, 1, 0, 0], B=[0, 0, 1, 1]
    t4 = data[data['a_str'] == '0100']
    
    if len(t2) == 0 or len(t3) == 0 or len(t4) == 0:
        return 0.0
        
    # Calculate empirical probabilities of choosing A for each subject
    p_a_t2 = 1.0 - t2.groupby('subject_id')['response'].mean()
    p_a_t3 = 1.0 - t3.groupby('subject_id')['response'].mean()
    p_a_t4 = 1.0 - t4.groupby('subject_id')['response'].mean()
    
    # To recover the structural linear score differences, we map probabilities back to log-odds.
    # We clip probabilities to [0.05, 0.95] to prevent infinite log-odds from deterministic subjects.
    def logit(p):
        p_clipped = np.clip(p, 0.05, 0.95)
        return np.log(p_clipped / (1.0 - p_clipped))
        
    l2 = logit(p_a_t2)
    l3 = logit(p_a_t3)
    l4 = logit(p_a_t4)
    
    # Structural invariant E = D3 - D2 - 2*D4
    # For WADD, this linear combination of log-odds is mathematically near 0 for all gamma.
    # For RBW, this combination is strictly positive and significantly larger.
    e_subj = l3 - l2 - 2.0 * l4
    
    return float(e_subj.mean())

outcome: self_sim=1.8718 (var=3.1899) adversary_sim=1.4379 (var=3.8101) welch_t=+0.820 p=0.4164 (N=25, alpha=0.01) -> reject

[8] rationale: This metric exploits a strict ordinal divergence in the internal score differences between Rank-Based Weighting (RBW) and Weighted Additive (WADD) models across Trials 2 and 3. For WADD, the score difference on T3 minus the score difference on T2 is mathematically forced to be nearly zero across its entire prior because it corresponds to the discrete second derivative of its exponentiated validities (w1 - 2*w2 + w3 ~ 0). Consequently, WADD predicts virtually identical choice probabilities for Option A on T2 and T3. In contrast, for RBW, the exponential decay function dictates that the score difference on T3 is strictly and significantly larger than on T2 by a factor proportional to (1 - alpha^0.25)^2. This structural guarantee means RBW will consistently exhibit a higher probability of choosing Option A in T3 compared to T2. By measuring the within-subject difference `P(A|T3) - P(A|T2)`, we perfectly isolate this invariant: the metric will be tightly centered near 0 for WADD and strictly positive for RBW, providing a robust discrimination that is highly resistant to noise from the softmax temperature parameter.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t2 = data[data['a_str'] == '01001']
    t3 = data[data['a_str'] == '10001']
    
    if len(t2) == 0 or len(t3) == 0:
        return 0.0
        
    p_a_t2 = 1.0 - t2.groupby('subject_id')['response'].mean()
    p_a_t3 = 1.0 - t3.groupby('subject_id')['response'].mean()
    
    # Calculate the within-subject difference in preference for Option A
    diff = p_a_t3 - p_a_t2
    
    return float(diff.mean())
outcome: self_sim=0.0474 (var=0.0180) adversary_sim=0.0400 (var=0.0192) welch_t=+0.191 p=0.8493 (N=25, alpha=0.01) -> reject

[9] rationale: This metric exploits a clean structural divergence between the two models on Trials 1, 2, and 3. For Rank-Based Weighting (RBW), the exponential decay function guarantees that Option A is strictly preferred across its entire parameter space; thus, any choices for Option B are purely due to softmax noise. In contrast, for Weighted Additive (WADD), the preference depends on the convexity of its weighting function. When gamma < 1 (which constitutes ~18% of its prior space), WADD structurally prefers Option B across all three of these trials. By calculating each subject's empirical rate of choosing Option B on these trials and raising it to the 4th power, we effectively suppress the background noise (which yields small B-rates for both models) and amplify the signal from the 18% of WADD subjects who consistently choose B. This creates a highly stable, non-linear metric where WADD is expected to score significantly higher than RBW.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Target trials where WADD occasionally prefers B (when gamma < 1), but RBW strictly prefers A.
    target_trials = ['10010', '01001', '10001']
    df_targets = data[data['a_str'].isin(target_trials)]
    
    if len(df_targets) == 0:
        return 0.0
        
    # Calculate the empirical probability of choosing Option B for each subject on these trials
    # response == 1 indicates Option B was chosen
    b_rates = df_targets.groupby('subject_id')['response'].mean()
    
    # Raise the B-rate to the 4th power to heavily penalize noise and isolate the subpopulation
    # of WADD subjects who have a true structural preference for Option B.
    metric_val = (b_rates ** 4).mean()
    
    return float(metric_val)
outcome: self_sim=0.0358 (var=0.0013) adversary_sim=0.0383 (var=0.0016) welch_t=-0.234 p=0.8162 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # Calculate the mean response for each trial type for each subject\n    means = data.groupby(['subject_id', 'a_str'])['response'].mean().unstack()\n    \n    # Calculate the variance of these means across the 5 trial types for each subject,\n    # then average across subjects.\n    # WADD produces smaller internal score differences on these trials, keeping its \n    # probabilities closer to 0.5 (lower variance across trial types).\n    # RBW produces larger internal score differences, pushing its probabilities \n    # closer to 0 or 1 (higher variance across trial types).\n    return float(means.var(axis=1).mean())",
  "rationale": "Rather than relying on the specific direction of preferences (which are largely aligned between the two theories), this metric measures the within-subject dispersion of choice probabilities across the 5 trial types. Because Weighted Additive (WADD) exponentiates linearly-spaced validities, its internal score differences on these specific trial pairs are mathematically constrained to be very small, resulting in choice probabilities that regress toward 0.5. In contrast, Rank-Based Weighting (RBW) uses exponential decay, which naturally produces much larger internal score differences and pushes choice probabilities closer to the extremes (0 or 1). Consequently, the variance of a subject's mean responses across the different trial types will be significantly higher for RBW than for WADD, providing a robust, noise-resistant discrimination."
}
```

## Usage

```json
{
  "prompt_token_count": 9700,
  "candidates_token_count": 386,
  "total_token_count": 19801
}
```
