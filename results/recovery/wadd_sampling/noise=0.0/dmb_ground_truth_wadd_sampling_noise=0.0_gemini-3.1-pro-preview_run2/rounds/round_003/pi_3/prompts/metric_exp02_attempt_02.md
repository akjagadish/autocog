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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.75, 0.75]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates WADD (with exponentiated validities) from Bayesian Cue Integration (with belief dampening) by exploiting their divergent limiting behaviors. As Bayesian dampening increases (gamma -> 0), the model does NOT collapse to Equal Weight; instead, it becomes a linear model on the raw probability excess (val - 0.5). In contrast, WADD with low gamma flattens validities and collapses to Tallying (Equal Weight). We construct trials where Bayesian Integration ALWAYS prefers the option with higher sum of (val - 0.5) and log-odds, while WADD with a low gamma reverses preference due to Tallying. Conversely, we include trials where Bayesian Integration ALWAYS prefers the option with more cues, but WADD with a high gamma reverses preference due to the exponentiation of a single dominant cue.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



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
**Description:** Bayesian Cue Integration with Belief Dampening: Decision-makers process cues as conditionally independent pieces of evidence, translating each cue's validity into a log-odds weight. However, instead of taking provided probabilities at face value, humans dampen extreme probabilities, contracting them toward 0.5 (ignorance) before converting them to log-odds. The overall evidence for an option is the sum of the log-odds of its positive features.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 1.0]
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
    
    # Dampen validities toward 0.5 based on gamma
    gamma = float(parameters["gamma"])
    val_adj = 0.5 + (val - 0.5) * gamma
    
    # Clip validities to avoid division by zero or log of zero
    val_adj = np.clip(val_adj, 0.5001, 0.9999)
    
    # Calculate log-odds weights for each cue
    weights = np.log(val_adj / (1.0 - val_adj))
    
    # Compute the evidence for each option as the sum of log-odds
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
[0] rationale: This metric calculates the proportion of times Option B is chosen on Trial 1. Bayesian Cue Integration reliably favors Option B on this trial because the sum of log-odds for the three moderately valid cues outweighs the log-odds of the single highly valid cue, particularly when validities are dampened. WADD, however, can exponentiate validities (gamma > 1), allowing the single most valid cue to dominate and thus leading to a higher likelihood of choosing Option A. The difference in choice proportions on this specific trial provides a strong contrast between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Option A relies solely on the most valid cue [1, 0, 0, 0, 0]
    # and Option B relies on several moderately valid cues [0, 1, 1, 1, 0].
    is_t1 = data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 0, 0, 0])
    if is_t1.sum() == 0:
        return 0.0
    # Return the proportion of times Option B was chosen (response == 1)
    return float(data.loc[is_t1, 'response'].mean())
outcome: self_sim=0.8000 (var=0.0252) adversary_sim=0.7905 (var=0.0202) welch_t=+0.222 p=0.825 (N=25, alpha=0.01) -> reject

[1] rationale: In Trial 4, Option A relies on the highest validity cue and the third highest, while Option B relies on all other cues. Because of the different non-linear transformations applied to validities, the Bayesian model and WADD transition from preferring B to preferring A at different rates across their parameter spaces. Measuring the proportion of times Option A is chosen on this specific trial provides a contrast between the models' weighting extremes.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 4 where A=[1, 0, 1, 0, 0] and B=[0, 1, 1, 1, 1]
    is_t4 = data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 1, 0, 0])
    if is_t4.sum() == 0:
        return np.nan
    return float((data.loc[is_t4, 'response'] == 0).mean())
outcome: self_sim=0.3400 (var=0.0669) adversary_sim=0.3474 (var=0.0311) welch_t=-0.118 p=0.9069 (N=25, alpha=0.01) -> reject

[2] rationale: In Trial 5, Option A relies on the 1st and 4th most valid cues, while Option B relies on the 2nd, 3rd, and 5th. Due to the properties of log-odds, Bayesian Cue Integration with little dampening actually favors Option A on this trial, as the extreme log-odds of the 0.95 validity cue heavily outweighs the others. By contrast, WADD generally favors Option B on this trial unless its gamma parameter is extremely high (gamma > 4). This structural difference in how the moderate cues sum against the best cue across the two models' non-linear transformations provides a strong discriminative signal.
metric_source:
def metric(data: pd.DataFrame) -> float:
    is_t5 = data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 0, 1, 0])
    if is_t5.sum() == 0:
        return np.nan
    return float((data.loc[is_t5, 'response'] == 0).mean())
outcome: self_sim=0.4011 (var=0.0660) adversary_sim=0.4747 (var=0.0284) welch_t=-1.199 p=0.2374 (N=25, alpha=0.01) -> reject

[3] rationale: In Trial 2, Option A is supported only by the second most valid cue (0.85), while Option B is supported by the third, fourth, and fifth (0.75, 0.65, 0.55). For the Bayesian model, the sum of the log-odds for B's cues always strictly outweighs A's cue across the entire valid dampening parameter space, meaning Bayesian Integration will consistently favor Option B. However, the WADD model with a high gamma parameter (gamma > ~3.5) will exponentiate the weights such that the 0.85 cue alone outweighs the sum of the three lesser cues, leading to a strong preference for Option A. Measuring the rate at which subjects choose Option A on this specific trial isolates this regime where WADD can flip its preference but the Bayesian model mathematically cannot.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 2 where A=[0, 1, 0, 0, 0] and B=[0, 0, 1, 1, 1]
    is_t2 = data['option_a_ratings'].apply(lambda x: list(x) == [0, 1, 0, 0, 0])
    if is_t2.sum() == 0:
        return np.nan
    return float((data.loc[is_t2, 'response'] == 0).mean())
outcome: self_sim=0.2832 (var=0.0312) adversary_sim=0.2747 (var=0.0179) welch_t=+0.190 p=0.8502 (N=25, alpha=0.01) -> reject

[4] rationale: This metric calculates the proportion of times the subject chooses the option with the top two valid cues ([1, 1, 0, 0, 0]) over the option with the bottom three valid cues ([0, 0, 1, 1, 1]). Under the Bayesian Cue Integration model, the sum of the log-odds of the top two cues (even when dampened) is mathematically always greater than the sum of the log-odds of the bottom three cues, meaning the Bayesian model strictly prefers the [1, 1, 0, 0, 0] option across its entire parameter space. However, under WADD, the weights are exponentiated validities. For gamma < ~1.5, the sum of the three moderate cues outweighs the two top cues, leading WADD to prefer the [0, 0, 1, 1, 1] option. Because WADD's gamma parameter space covers this regime extensively, WADD will show a significantly lower rate of choosing the [1, 1, 0, 0, 0] option compared to the Bayesian model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the choice is between [1, 1, 0, 0, 0] and [0, 0, 1, 1, 1]
    # This corresponds to Trial 8 (A=[1, 1, 0, 0, 0]) and Trial 7 (B=[1, 1, 0, 0, 0])
    is_t8 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_t7 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    chose_target = 0
    total = 0
    
    if is_t8.any():
        chose_target += (data.loc[is_t8, 'response'] == 0).sum()
        total += is_t8.sum()
        
    if is_t7.any():
        chose_target += (data.loc[is_t7, 'response'] == 1).sum()
        total += is_t7.sum()
        
    if total == 0:
        return 0.5
        
    return float(chose_target / total)
outcome: self_sim=0.7358 (var=0.0639) adversary_sim=0.8350 (var=0.0192) welch_t=-1.721 p=0.09359 (N=25, alpha=0.01) -> reject

[5] rationale: Bayesian Integration with dampening shrinks validities toward 0.5. However, because it converts these dampened probabilities to log-odds, the transformation remains strictly convex. This ensures that the most valid cue (0.98) maintains a dominant log-odds weight over moderate cues across the entire parameter space. Thus, on Trial 6, Bayesian Integration consistently prefers the option with the 0.98 cue, and on Trial 3, it strongly prefers the opposing option for almost all dampening values.
In contrast, WADD exponentiates validities. For gamma < 1, this transformation is concave, meaning moderate cues are given disproportionately high weight. Consequently, WADD prefers the opposing option on Trial 6 for a large portion of its parameter space (gamma < 1.8), and continues to prefer it on Trial 3 until gamma > 4.0.
By taking the difference in the rate of choosing the 0.98-cue option between Trial 6 and Trial 3, we isolate this structural difference. The Bayesian model will show a large positive difference (consistently ~1 on Trial 6 and ~0 on Trial 3), yielding a metric near 0.7 (after epsilon noise). WADD, because it often prefers the opposing option on both trials, will yield a much lower difference (around 0.3). This large mean gap, combined with the low variance of the Bayesian model's mathematically consistent preferences, provides a highly discriminative signal.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 6: target option has cues [1, 0, 1, 0, 0], opposing has [0, 1, 0, 1, 1]
    t6_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    t6_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 1))
    
    # Identify Trial 3: target option has cues [1, 0, 0, 0, 1], opposing has [0, 1, 1, 1, 0]
    t3_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1))
    t3_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 0))
    
    # Calculate proportion of times the target option was chosen for Trial 6
    t6_target = (data.loc[t6_fwd, 'response'] == 0).sum() + (data.loc[t6_rev, 'response'] == 1).sum()
    t6_total = t6_fwd.sum() + t6_rev.sum()
    p_t6 = float(t6_target / t6_total) if t6_total > 0 else 0.5
    
    # Calculate proportion of times the target option was chosen for Trial 3
    t3_target = (data.loc[t3_fwd, 'response'] == 0).sum() + (data.loc[t3_rev, 'response'] == 1).sum()
    t3_total = t3_fwd.sum() + t3_rev.sum()
    p_t3 = float(t3_target / t3_total) if t3_total > 0 else 0.5
    
    # Return the difference in choice rates
    return p_t6 - p_t3
outcome: self_sim=0.3275 (var=0.0781) adversary_sim=0.4767 (var=0.0924) welch_t=-1.806 p=0.07719 (N=25, alpha=0.01) -> reject

[6] rationale: We contrast the choice rate of the 'tallying' option (the option with more moderately valid cues but lacking the most valid cue) between two groups of trials. Group 1 (Trials 6, 7, 8) pits the tallying option against a strong coalition (Cue 1 + Cue 2 or 3). Group 2 (Trials 2, 3, 4) pits the tallying option against a weaker target (Cue 1 alone or with Cue 5). For Bayesian Cue Integration, the target in Group 1 mathematically dominates the tallying option regardless of dampening, dropping the tallying choice rate to near zero. However, in Group 2, high dampening allows the tallying option to win, creating a massive drop in tallying choice between the two groups. In contrast, WADD's exponentiation (when gamma < 1.5) compresses validities so much that the tallying option remains competitive even in Group 1. Thus, WADD predicts a much smaller drop in the tallying choice rate between the two groups. Measuring this within-subject difference isolates the distinct non-linear transformations while canceling out baseline noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert lists to tuples to allow hashing and equality checks
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    # Represent each trial as a frozenset of its two options
    pairs = [frozenset([a, b]) for a, b in zip(a_tup, b_tup)]
    
    # Group 1: The 'tallying' option faces a strong target (Cue 1 + Cue 2 or 3)
    pair_6 = frozenset([(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)])
    pair_78 = frozenset([(1, 1, 0, 0, 0), (0, 0, 1, 1, 1)])
    
    # Group 2: The 'tallying' option faces a weaker target (Cue 1 alone or with Cue 5)
    pair_2 = frozenset([(1, 0, 0, 0, 0), (0, 1, 1, 1, 0)])
    pair_34 = frozenset([(1, 0, 0, 0, 1), (0, 1, 1, 1, 0)])
    
    # Create boolean masks for the two groups of trials
    group1_mask = np.array([p in [pair_6, pair_78] for p in pairs])
    group2_mask = np.array([p in [pair_2, pair_34] for p in pairs])
    
    # Identify the 'tallying' (opposing) option as the one lacking Cue 1
    a_is_opposing = data['option_a_ratings'].apply(lambda x: x[0] == 0)
    b_is_opposing = data['option_b_ratings'].apply(lambda x: x[0] == 0)
    
    # Did the subject choose the 'tallying' option?
    chose_opposing = (a_is_opposing & (data['response'] == 0)) | (b_is_opposing & (data['response'] == 1))
    
    # Calculate the proportion of times the tallying option was chosen in each group
    p_group1 = chose_opposing[group1_mask].mean() if group1_mask.sum() > 0 else 0.5
    p_group2 = chose_opposing[group2_mask].mean() if group2_mask.sum() > 0 else 0.5
    
    # Return the within-subject difference in choice rates
    return float(p_group1 - p_group2)
outcome: self_sim=-0.4317 (var=0.0616) adversary_sim=-0.5061 (var=0.0846) welch_t=+0.973 p=0.3353 (N=25, alpha=0.01) -> reject

[7] rationale: Bayesian Cue Integration with Belief Dampening processes validities by shrinking them toward 0.5 and then applying a log-odds transformation. Because the log-odds function is strictly convex, the most valid cue (0.98) maintains a dominant weight over moderate cues across almost the entire parameter space. Consequently, the Bayesian model acts similarly to 'Take The Best' (TTB) and frequently chooses the option with the 0.98 cue.
In contrast, WADD exponentiates validities. For a significant portion of its parameter space (gamma < 1), this transformation is concave, which flattens the weights and boosts the relative influence of moderate cues. This allows a coalition of moderate cues to overpower the 0.98 cue much more often than in the Bayesian model.
By measuring the overall proportion of trials where the subject chooses the option containing the 0.98 cue across the entire 96-trial session, we capture this fundamental structural difference. The Bayesian model predicts a significantly higher overall choice rate for the best cue compared to WADD. Averaging over all trials minimizes trial-specific binomial noise, yielding a robust metric with low within-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # The first cue has the highest validity (0.98).
    # We compute the proportion of trials where the subject chose the option containing this cue.
    a_has_best_cue = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_best_cue = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    chose_a = data['response'] == 0
    chose_b = data['response'] == 1
    
    chose_best = (a_has_best_cue & chose_a) | (b_has_best_cue & chose_b)
    
    return float(chose_best.mean())
outcome: self_sim=0.4806 (var=0.0177) adversary_sim=0.5498 (var=0.0135) welch_t=-1.957 p=0.0563 (N=25, alpha=0.01) -> reject

[8] rationale: We calculate the difference in the preference for the top-cues option between Trial 1 (where the target has the top TWO cues) and Trial 3 (where the target has only the top ONE cue), both pitted against the same opposing option (the bottom three cues). For the Bayesian model, the sum of log-odds for the top two cues is mathematically always higher than the bottom three (even as dampening approaches 1, the limit of log-odds is proportional to validity - 0.5, where 0.45 + 0.35 > 3 * 0.25). Thus, Bayesian Integration strictly prefers the target in Trial 1 but strictly prefers the opposing option in Trial 3, yielding a difference near 1.0. In contrast, WADD's exponentiation means that for low gamma it prefers the opposing option in BOTH trials (tallying), and for high gamma it prefers the target in BOTH trials (take-the-best). This leads WADD to predict a much smaller difference in choice rates between the two trials, providing a highly discriminative and stable signal.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: Target is (1, 1, 0, 0, 0), opposing is (0, 0, 1, 1, 1)
    is_t1_fwd = (a_tup == (1, 1, 0, 0, 0)) & (b_tup == (0, 0, 1, 1, 1))
    is_t1_rev = (a_tup == (0, 0, 1, 1, 1)) & (b_tup == (1, 1, 0, 0, 0))
    
    # Trial 3: Target is (1, 0, 0, 0, 0), opposing is (0, 0, 1, 1, 1)
    is_t3_fwd = (a_tup == (1, 0, 0, 0, 0)) & (b_tup == (0, 0, 1, 1, 1))
    is_t3_rev = (a_tup == (0, 0, 1, 1, 1)) & (b_tup == (1, 0, 0, 0, 0))
    
    # Choice rates for the target option in each trial type
    t1_chose_target = (is_t1_fwd & (data['response'] == 0)).sum() + (is_t1_rev & (data['response'] == 1)).sum()
    t1_total = is_t1_fwd.sum() + is_t1_rev.sum()
    p1 = float(t1_chose_target / t1_total) if t1_total > 0 else 0.5
    
    t3_chose_target = (is_t3_fwd & (data['response'] == 0)).sum() + (is_t3_rev & (data['response'] == 1)).sum()
    t3_total = is_t3_fwd.sum() + is_t3_rev.sum()
    p3 = float(t3_chose_target / t3_total) if t3_total > 0 else 0.5
    
    return p1 - p3

outcome: self_sim=0.3211 (var=0.0861) adversary_sim=0.5337 (var=0.0902) welch_t=-2.532 p=0.01467 (N=25, alpha=0.01) -> reject

[9] rationale: This metric leverages the fact that Bayesian Cue Integration is mathematically equivalent to WADD restricted to a narrow, intermediate range of its gamma parameter space (roughly 2.5 < gamma < 4.2). Because Bayes is strictly bounded within this 'middle' regime, it ALWAYS prefers the Target in T1 (w1 + w2 > 3*w3) and ALWAYS prefers the Opposing option in T3 (3*w3 > w1). Consequently, both P(Target|T1) and P(Opposing|T3) are strictly > 0.5 for Bayes, meaning the product of their deviations from 0.5 is robustly POSITIVE regardless of noise parameters.

In contrast, WADD's parameter space extends into Tallying (gamma < 2.2) where it prefers the Opposing option in T1, and extreme Take-The-Best (gamma > 4.65) where it prefers the Target in T3. In both of these outer regions (which comprise ~50% of WADD's prior space), exactly one of the two probabilities drops below 0.5, making the product NEGATIVE. By multiplying the deviations from 0.5, we naturally suppress the variance introduced by high-noise subjects (whose deviations approach 0) while perfectly capturing the structural double-dissociation between the models. Bayes will yield a consistently positive point estimate, while WADD will average near zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1: Target has cues [1, 1, 0, 0, 0], Opposing has [0, 0, 1, 1, 1]
    is_t1_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_t1_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Identify Trial 3: Target has cue [1, 0, 0, 0, 0], Opposing has [0, 0, 1, 1, 1]
    is_t3_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    is_t3_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Calculate P(Target | T1)
    t1_target_choices = (is_t1_fwd & (data['response'] == 0)).sum() + (is_t1_rev & (data['response'] == 1)).sum()
    t1_total = is_t1_fwd.sum() + is_t1_rev.sum()
    p_t1 = float(t1_target_choices / t1_total) if t1_total > 0 else 0.5
    
    # Calculate P(Opposing | T3)
    t3_opposing_choices = (is_t3_fwd & (data['response'] == 1)).sum() + (is_t3_rev & (data['response'] == 0)).sum()
    t3_total = is_t3_fwd.sum() + is_t3_rev.sum()
    p_t3 = float(t3_opposing_choices / t3_total) if t3_total > 0 else 0.5
    
    # Return the product of their deviations from 0.5
    return (p_t1 - 0.5) * (p_t3 - 0.5)

outcome: self_sim=0.0088 (var=0.0097) adversary_sim=0.0602 (var=0.0055) welch_t=-2.084 p=0.04292 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify Trial 1: Target has cues [1, 1, 0, 0, 0], Opposing has [0, 0, 1, 1, 1]\n    is_t1_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))\n    is_t1_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))\n    \n    # Identify Trial 2: Target has cue [1, 0, 0, 0, 0], Opposing has [0, 1, 1, 0, 0]\n    is_t2_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))\n    is_t2_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))\n    \n    # Calculate P(Target | T1)\n    t1_target_choices = (is_t1_fwd & (data['response'] == 0)).sum() + (is_t1_rev & (data['response'] == 1)).sum()\n    t1_total = is_t1_fwd.sum() + is_t1_rev.sum()\n    p_t1 = float(t1_target_choices / t1_total) if t1_total > 0 else 0.5\n    \n    # Calculate P(Target | T2)\n    t2_target_choices = (is_t2_fwd & (data['response'] == 0)).sum() + (is_t2_rev & (data['response'] == 1)).sum()\n    t2_total = is_t2_fwd.sum() + is_t2_rev.sum()\n    p_t2 = float(t2_target_choices / t2_total) if t2_total > 0 else 0.5\n    \n    # Return the simple difference between the two choice probabilities\n    return p_t1 - p_t2\n",
  "rationale": "This metric evaluates the within-subject difference in the probability of choosing the option with the strongest cue(s) between Trial 1 and Trial 2. For Bayesian Cue Integration, the evidence gap between the target and opposing options shrinks significantly from T1 to T2 (mathematically, the difference in scores drops by a large, strictly positive margin across all dampening values), leading to a massive drop in the choice probability for the target option (a large positive P(T1) - P(T2)). In contrast, the Weighted Additive (WADD) model's non-linear exponentiation creates a much smaller gap between these two trials across its parameter space, often resulting in similar choice rates for both trials. The within-subject subtraction cancels out individual baseline noise (such as overall temperature and lapse rates), providing a stable signal with low between-subject variance that cleanly discriminates the models."
}
```

## Usage

```json
{
  "prompt_token_count": 9648,
  "candidates_token_count": 752,
  "total_token_count": 41854
}
```
