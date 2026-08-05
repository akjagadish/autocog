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
**Validities (n_features=6):** [0.92, 0.82, 0.72, 0.62, 0.52, 0.52]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 5: A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate WADD-Gamma from Probabilistic Search Take-The-Best (PS-TTB), we use a 6-feature design with a carefully chosen validity gradient. WADD-Gamma integrates all features using a weighted sum where weights are log-odds raised to a power (gamma). PS-TTB, on the other hand, probabilistically samples cue orders based on their validities and stops at the first discriminating cue. We include 'compensatory' trials where the single highest-validity cue points to Option B, but the sum of the remaining lower-validity cues strongly points to Option A. Because PS-TTB stops at the first discriminating cue, it will frequently choose Option B on these trials. WADD-Gamma will integrate the multiple lower-validity cues and, depending on the gamma parameter, can strongly favor Option A. We also include trials where the top cue is tied (0 vs 0 or 1 vs 1) to test how the models handle secondary cues: PS-TTB's probabilistic search will distribute choices among the remaining cues, whereas WADD-Gamma deterministically sums them.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive Model with Power-Scaled Log-Odds (WADD-Gamma). Decision-makers compute a weighted sum of features for each option. The weights are derived from the log-odds of the cue validities, raised to a power gamma. This parameterization allows the model to smoothly interpolate between Tallying/Equal-Weighting (gamma = 0) and standard log-odds WADD (gamma = 1). Choices are then made via a softmax over the weighted sums, incorporating an independent lapse rate for noise.

**Parameters:**
- gamma: [0.0, 0.75]
- beta: [0.0, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to ensure log-odds are strictly positive and well-defined
    v_clipped = np.clip(validities, 0.5001, 0.9999)
    log_odds = np.log(v_clipped / (1.0 - v_clipped))
    
    gamma = float(parameters["gamma"])
    weights = log_odds ** gamma
    
    # Weighted sum for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Probabilistic Search Take-The-Best (PS-TTB)

**Parameters:**
- tau: [0.01, 100.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    diff = stim[0] - stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    tau = float(parameters['tau'])
    epsilon = float(parameters['epsilon'])
    
    n_features = len(validities)
    n_samples = 1000
    
    # Gumbel-max trick to sample permutations without replacement
    # probabilities proportional to softmax(validities / tau)
    logits = validities / (tau + 1e-6)
    gumbels = np.random.gumbel(size=(n_samples, n_features))
    orders = np.argsort(-(logits + gumbels), axis=1)
    
    diff_sign = np.sign(diff)
    ordered_diffs = diff_sign[orders]
    
    # Find the first discriminating cue in each sampled search order
    abs_diffs = np.abs(ordered_diffs)
    first_non_zero_idx = np.argmax(abs_diffs, axis=1)
    has_non_zero = np.any(abs_diffs > 0, axis=1)
    
    first_non_zero_vals = ordered_diffs[np.arange(n_samples), first_non_zero_idx]
    
    wins_a = np.sum((first_non_zero_vals == 1) & has_non_zero)
    wins_b = np.sum((first_non_zero_vals == -1) & has_non_zero)
    
    total = wins_a + wins_b
    if total > 0:
        p = np.array([wins_a / total, wins_b / total])
    else:
        p = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p + epsilon * (np.ones(2) / 2.0)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
[0] rationale: The metric computes the proportion of times the subject chooses Option B in trials where the most valid cue (Cue 1) points to Option B, but a larger number of less valid cues point to Option A. Specifically, it filters for trials where Option A has 0 on Cue 1 and Option B has 1 on Cue 1. PS-TTB, being a lexicographic heuristic, will frequently stop search at Cue 1 and choose Option B. WADD-Gamma, being a compensatory model, will sum the evidence from the remaining cues and favor Option A. Thus, PS-TTB predicts a high value for this metric (close to 1), while WADD-Gamma predicts a low value (close to 0).
metric_source:
def metric(data: pd.DataFrame) -> float:
    mask = data.apply(lambda row: row['option_a_ratings'][0] == 0 and row['option_b_ratings'][0] == 1, axis=1)
    if not mask.any():
        return 0.5
    return float(data[mask]['response'].mean())
outcome: self_sim=0.3323 (var=0.0185) adversary_sim=0.4092 (var=0.0070) welch_t=-2.411 p=0.02062 (N=25, alpha=0.01) -> reject

[1] rationale: This metric isolates a specific, extreme compensatory trial (Trial 1) where Option A has all four lower-validity cues in its favor, while Option B has only the single highest-validity cue. Under WADD-Gamma, the combined weight of the four lower cues (regardless of the gamma parameter) strongly outweighs the single highest cue, leading to a strong preference for Option A (response = 0). Conversely, PS-TTB searches cues sequentially based on their validities; since the first cue is the most valid, PS-TTB will very frequently evaluate it first and immediately choose Option B (response = 1). By focusing strictly on this single, most strongly dissociating trial, we maximize the mean difference between the theories while minimizing within-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    mask = data.apply(lambda row: list(row['option_a_ratings']) == [0, 1, 1, 1, 1] and list(row['option_b_ratings']) == [1, 0, 0, 0, 0], axis=1)
    if not mask.any():
        return 0.5
    return float(data[mask]['response'].mean())
outcome: self_sim=0.3108 (var=0.0336) adversary_sim=0.3523 (var=0.0237) welch_t=-0.868 p=0.39 (N=25, alpha=0.01) -> reject

[2] rationale: This metric isolates Trial 4, where Option A is supported by Cue 2 and Cue 3, while Option B is supported only by the most valid Cue 1. Under WADD-Gamma, the combined weight of Cues 2 and 3 consistently outweighs Cue 1 for all valid gamma values, leading to a strong preference for Option A (response = 0). Under PS-TTB, the most valid cue is heavily prioritized, leading to a frequent choice of Option B (response = 1). Because there are fewer cues favoring Option A in this trial compared to Trial 1, PS-TTB with high noise (large tau) is also less likely to accidentally stumble upon a cue favoring Option A, maximizing the contrast between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Isolate Trial 4 where A has cues 2 and 3, and B has cue 1.
    mask = data.apply(lambda row: list(row['option_a_ratings']) == [0, 1, 1, 0, 0] and list(row['option_b_ratings']) == [1, 0, 0, 0, 0], axis=1)
    if not mask.any():
        return 0.5
    return float(data[mask]['response'].mean())
outcome: self_sim=0.3477 (var=0.0263) adversary_sim=0.4277 (var=0.0197) welch_t=-1.865 p=0.06846 (N=25, alpha=0.01) -> reject

[3] rationale: By computing the difference in choice probabilities between trials where the most valid cue (Cue 1) strongly favors Option B versus when it strongly favors Option A, we isolate the causal impact of the most valid cue. PS-TTB is highly sensitive to the top cue, predicting a large positive difference. In contrast, WADD-Gamma integrates all cues; because the lower-validity cues are designed to compensate and generally favor Option A across these specific trials, WADD-Gamma predicts a much smaller difference (near zero or even negative). This difference-in-means approach also naturally subtracts out baseline noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    mask_x = data.apply(lambda row: row['option_a_ratings'][0] == 0 and row['option_b_ratings'][0] == 1, axis=1)
    mask_y = data.apply(lambda row: row['option_a_ratings'][0] == 1 and row['option_b_ratings'][0] == 0, axis=1)
    val_x = float(data[mask_x]['response'].mean()) if mask_x.any() else 0.5
    val_y = float(data[mask_y]['response'].mean()) if mask_y.any() else 0.5
    return val_x - val_y
outcome: self_sim=-0.1162 (var=0.0328) adversary_sim=-0.0823 (var=0.0142) welch_t=-0.781 p=0.4393 (N=25, alpha=0.01) -> reject

[4] rationale: This metric computes the difference in the proportion of Option B choices between Trial 3 and Trial 6. In Trial 3, Option A is supported by Cues 2 and 3, while Option B is supported solely by the most valid Cue 1. WADD-Gamma consistently integrates all cues and favors Option A across its parameter space. In contrast, PS-TTB often stops at Cue 1 and strongly favors Option B. In Trial 6, Option A is supported by Cues 1, 4, and 5, while B is supported by 2 and 3; here, BOTH theories consistently favor Option A. By subtracting the mean response of Trial 6 from Trial 3, we subtract out baseline noise and isolate the lexicographic vs compensatory mechanism. PS-TTB predicts a large positive difference (switching from B to A), whereas WADD-Gamma predicts a difference near zero or slightly negative (consistently favoring A in both).
metric_source:
def metric(data: pd.DataFrame) -> float:
    def is_t3(row):
        return tuple(row['option_a_ratings']) == (0, 1, 1, 0, 0) and tuple(row['option_b_ratings']) == (1, 0, 0, 0, 0)

    def is_t6(row):
        return tuple(row['option_a_ratings']) == (1, 0, 0, 1, 1) and tuple(row['option_b_ratings']) == (0, 1, 1, 0, 0)

    mask_t3 = data.apply(is_t3, axis=1)
    mask_t6 = data.apply(is_t6, axis=1)
    
    val_t3 = float(data[mask_t3]['response'].mean()) if mask_t3.any() else 0.5
    val_t6 = float(data[mask_t6]['response'].mean()) if mask_t6.any() else 0.5
    
    return val_t3 - val_t6
outcome: self_sim=-0.0500 (var=0.0297) adversary_sim=0.0017 (var=0.0407) welch_t=-0.974 p=0.3353 (N=25, alpha=0.01) -> reject

[5] rationale: To cleanly dissociate WADD-Gamma from PS-TTB across their entire parameter spaces, we exploit a structural difference in how they evaluate specific cue patterns. We compute the metric M = P(B|T1) - P(B|T6) - P(B|T3). In PS-TTB, Trial 1 (A has Cue 1, B has 2-5) and Trial 6 (A has 1,4,5, B has 2,3) both strongly favor A when search is lexicographic (tau -> 0), driving P(B|T1) and P(B|T6) to 0. Meanwhile, Trial 3 (A has 2,3, B has 1) strongly favors B, driving P(B|T3) to 1. This yields M = -1. When PS-TTB is random (tau -> inf), it acts like tallying, yielding M = 0.8 - 0.4 - 0.33 = +0.07. Thus, PS-TTB is strictly bounded between [-1, 0.07]. In contrast, WADD-Gamma always favors B heavily in Trial 1, and A heavily in Trials 6 and 3. This drives P(B|T1) high, while keeping P(B|T6) and P(B|T3) low, resulting in a metric value that is systematically more positive than PS-TTB across all valid gamma and beta parameters.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def is_t1(row):
        return tuple(row['option_a_ratings']) == (1, 0, 0, 0, 0) and tuple(row['option_b_ratings']) == (0, 1, 1, 1, 1)
    def is_t6(row):
        return tuple(row['option_a_ratings']) == (1, 0, 0, 1, 1) and tuple(row['option_b_ratings']) == (0, 1, 1, 0, 0)
    def is_t3(row):
        return tuple(row['option_a_ratings']) == (0, 1, 1, 0, 0) and tuple(row['option_b_ratings']) == (1, 0, 0, 0, 0)

    mask_t1 = data.apply(is_t1, axis=1)
    mask_t6 = data.apply(is_t6, axis=1)
    mask_t3 = data.apply(is_t3, axis=1)

    p_b_t1 = float(data[mask_t1]['response'].mean()) if mask_t1.any() else 0.5
    p_b_t6 = float(data[mask_t6]['response'].mean()) if mask_t6.any() else 0.5
    p_b_t3 = float(data[mask_t3]['response'].mean()) if mask_t3.any() else 0.5

    return p_b_t1 - p_b_t6 - p_b_t3
outcome: self_sim=-0.0233 (var=0.2320) adversary_sim=-0.2183 (var=0.1097) welch_t=+1.668 p=0.1026 (N=25, alpha=0.01) -> reject

[6] rationale: This metric computes the proportion of a subject's choices that match the deterministic profile of WADD-Gamma when gamma=0.75 across the 6 dissociating trials. PS-TTB is a probabilistic mixture of lexicographic search (small tau) and tallying (large tau). Lexicographic search matches exactly 3 of these 6 targets (50%), while probabilistic tallying matches ~3.5 of them (~58%). Thus, PS-TTB is mathematically bounded to an expected match rate below 0.60 across its entire parameter space. In contrast, WADD-Gamma interpolates between deterministic tallying (50% match) and WADD-0.75 (100% match), yielding a significantly higher expected match rate (around 0.75) and providing a robust, guaranteed separation between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert option_a_ratings to tuple of ints for fast, safe matching
    a_tuples = [tuple(int(v) for v in x) for x in data['option_a_ratings']]
    
    # The target profile represents the deterministic choices of WADD-Gamma at gamma=0.75
    targets = {
        (1, 0, 0, 0, 0): 1,  # T1
        (1, 0, 1, 0, 0): 0,  # T2
        (0, 1, 1, 0, 0): 0,  # T3
        (1, 1, 0, 0, 0): 0,  # T4
        (0, 1, 1, 1, 1): 0,  # T5
        (0, 0, 1, 1, 1): 1   # T7
    }
    
    matches = []
    responses = data['response'].values
    for a_tup, resp in zip(a_tuples, responses):
        if a_tup in targets:
            matches.append(1 if resp == targets[a_tup] else 0)
            
    if not matches:
        return 0.5
        
    return float(np.mean(matches))
outcome: self_sim=0.5953 (var=0.0118) adversary_sim=0.5286 (var=0.0044) welch_t=+2.619 p=0.01243 (N=25, alpha=0.01) -> reject

[7] rationale: This metric focuses strictly on Trials 1, 3, and 5. In these three trials, WADD-Gamma consistently favors the exact same option across its entire parameter space (gamma from 0 to 0.75). Thus, WADD-Gamma predicts a very high match rate to these targets (bounded only by epsilon noise). In contrast, PS-TTB heavily depends on its tau parameter for these specific trials: at low tau (lexicographic search), it predicts the exact opposite of WADD-Gamma (0% match), and only at high tau (tallying) does it align with WADD-Gamma (100% match). Averaged over the PS-TTB parameter space, its expected match rate is substantially lower than WADD-Gamma's. By pooling these three trials into a single match proportion, we maximize the mean difference between the theories while drastically reducing between-subject variance compared to computing differences of means.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert option_a_ratings to tuple of ints for fast, safe matching
    a_tuples = [tuple(int(v) for v in x) for x in data['option_a_ratings']]
    
    # We isolate the three trials where WADD-Gamma strongly and consistently 
    # favors the same option regardless of its gamma parameter, but where 
    # PS-TTB transitions from the opposite choice (at low tau) to the same 
    # choice (at high tau). 
    targets = {
        (1, 0, 0, 0, 0): 1,  # Trial 1: WADD always prefers B
        (0, 1, 1, 0, 0): 0,  # Trial 3: WADD always prefers A
        (0, 1, 1, 1, 1): 0   # Trial 5: WADD always prefers A
    }
    
    matches = []
    responses = data['response'].values
    for a_tup, resp in zip(a_tuples, responses):
        if a_tup in targets:
            matches.append(1 if resp == targets[a_tup] else 0)
            
    if not matches:
        return 0.5
        
    return float(np.mean(matches))
outcome: self_sim=0.6894 (var=0.0259) adversary_sim=0.6228 (var=0.0146) welch_t=+1.657 p=0.1045 (N=25, alpha=0.01) -> reject

[8] rationale: This metric computes P(A|T1) + P(A|T8) - P(A|T6). For PS-TTB, this value is bounded between 0.0 (at low tau) and 0.5 (at high tau). For WADD-Gamma, this value is bounded between 0.5 (at gamma=0) and 2.0 (at gamma=0.75). Because the ranges barely overlap and move in opposite directions across their parameter spaces, WADD-Gamma will systematically produce a much higher value than PS-TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def is_t1(row):
        return tuple(int(v) for v in row['option_a_ratings']) == (0, 1, 1, 1, 1, 1) and \
               tuple(int(v) for v in row['option_b_ratings']) == (1, 0, 0, 0, 0, 0)
    def is_t6(row):
        return tuple(int(v) for v in row['option_a_ratings']) == (0, 1, 0, 1, 1, 1) and \
               tuple(int(v) for v in row['option_b_ratings']) == (1, 0, 1, 0, 0, 0)
    def is_t8(row):
        return tuple(int(v) for v in row['option_a_ratings']) == (0, 1, 1, 1, 0, 0) and \
               tuple(int(v) for v in row['option_b_ratings']) == (1, 0, 0, 0, 1, 1)

    mask_t1 = data.apply(is_t1, axis=1)
    mask_t6 = data.apply(is_t6, axis=1)
    mask_t8 = data.apply(is_t8, axis=1)

    p_a_t1 = float((1 - data[mask_t1]['response']).mean()) if mask_t1.any() else 0.5
    p_a_t6 = float((1 - data[mask_t6]['response']).mean()) if mask_t6.any() else 0.5
    p_a_t8 = float((1 - data[mask_t8]['response']).mean()) if mask_t8.any() else 0.5

    return p_a_t1 + p_a_t8 - p_a_t6
outcome: self_sim=0.7317 (var=0.1176) adversary_sim=0.6017 (var=0.0681) welch_t=+1.508 p=0.1385 (N=25, alpha=0.01) -> reject

[9] rationale: We isolate 5 specific trials (T1, T5, T6, T7, T8) and compute the subject's match rate to the deterministic profile of WADD-Gamma at low-to-moderate gamma (A, A, A, B, A). Because WADD-Gamma is highly compensatory, it strongly predicts these exact choices across the vast majority of its parameter space, only diverging at extremely high gamma values. Thus, WADD-Gamma guarantees a very high expected match rate (~80%). In stark contrast, PS-TTB at low tau (lexicographic search) makes the exact opposite choice on all 5 of these trials (0% match), and only partially approaches this profile as tau increases toward random tallying. Averaged over its parameter space, PS-TTB has a drastically lower expected match rate (~30%). By pooling these specific 5 trials into a single bounded proportion, we maximize the mean separation between the theories while inherently restricting between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = [tuple(int(v) for v in x) for x in data['option_a_ratings']]
    
    targets = {
        (0, 1, 1, 1, 1, 1): 0,
        (0, 0, 1, 1, 1, 1): 0,
        (0, 1, 0, 1, 1, 1): 0,
        (1, 1, 0, 0, 0, 0): 1,
        (0, 1, 1, 1, 0, 0): 0
    }
    
    matches = []
    for a_tup, resp in zip(a_tuples, data['response']):
        if a_tup in targets:
            matches.append(1 if resp == targets[a_tup] else 0)
            
    if not matches:
        return 0.5
        
    return float(np.mean(matches))
outcome: self_sim=0.6140 (var=0.0232) adversary_sim=0.5940 (var=0.0045) welch_t=+0.601 p=0.5519 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    t1_a = (0, 1, 1, 1, 1, 1)\n    t1_b = (1, 0, 0, 0, 0, 0)\n    t8_a = (0, 1, 1, 1, 0, 0)\n    t8_b = (1, 0, 0, 0, 1, 1)\n    \n    is_target = []\n    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):\n        a_tup = tuple(int(x) for x in a)\n        b_tup = tuple(int(x) for x in b)\n        if (a_tup == t1_a and b_tup == t1_b) or (a_tup == t8_a and b_tup == t8_b):\n            is_target.append(True)\n        else:\n            is_target.append(False)\n            \n    if not any(is_target):\n        return 0.5\n        \n    return float(data[is_target]['response'].mean())",
  "rationale": "To discriminate the models, we must find trials where they consistently diverge across their entire parameter spaces. WADD-Gamma interpolates between tallying (gamma=0) and weighted addition (gamma=0.75). PS-TTB interpolates between lexicographic search (low tau) and tallying (high tau). In Trial 1 and Trial 8, the most valid cue (Cue 1) strongly favors Option B, but the remaining cues collectively favor Option A. For WADD-Gamma, the combined weight of the numerous lower-validity cues consistently outweighs Cue 1 for all gamma in [0, 0.75], leading to a robust preference for Option A (response ~ 0). For PS-TTB, at low tau, it almost always searches Cue 1 first and chooses Option B. At high tau, it acts like tallying and chooses A on T1 and ties on T8. Thus, across its parameter space, PS-TTB chooses Option B significantly more often than WADD-Gamma. By strictly pooling only Trial 1 and Trial 8, we avoid trials where WADD-Gamma's choice flips depending on gamma (which would inflate its variance), securing a large mean difference with tight within-model variance."
}
```

## Usage

```json
{
  "prompt_token_count": 8612,
  "candidates_token_count": 570,
  "total_token_count": 30953
}
```
