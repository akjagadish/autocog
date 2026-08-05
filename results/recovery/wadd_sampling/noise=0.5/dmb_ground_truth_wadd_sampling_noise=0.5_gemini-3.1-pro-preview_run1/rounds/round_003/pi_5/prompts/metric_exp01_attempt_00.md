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
**Validities (n_features=5):** [0.95, 0.85, 0.8, 0.75, 0.7]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated model (TTB + Tallying) from the Competing model (WADD + Tallying), we use a 5-feature design with a specific spread of validities. TTB makes decisions based solely on the single most valid discriminating feature, completely ignoring the rest. WADD, on the other hand, integrates all features compensatorily. By creating trials where the single most valid feature favors Option A, but the sum of several moderately valid features strongly favors Option B, TTB and WADD will diverge. In the Advocated model, choices will be a mixture of pure single-feature reliance (TTB) and simple feature counting (Tallying). In the Competing model, choices will be a mixture of compensatory weighting (WADD) and feature counting (Tallying). We include trials where TTB and WADD strongly oppose each other, as well as trials where they agree but Tallying disagrees, ensuring a robust quantitative dissociation across the parameter spaces of both models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Heuristic Toolbox: Subjects make decisions by probabilistically sampling from a repertoire of fast-and-frugal heuristics rather than computing compensatory weighted sums. Specifically, on any given trial, a subject either uses 'Take-The-Best' (TTB) - a lexicographic strategy that bases the choice entirely on the single most valid discriminating feature - or 'Tallying' - an equal-weighting strategy that simply counts the number of winning features for each option. A mixture parameter alpha governs the probability of selecting TTB over Tallying, and an independent lapse rate epsilon accounts for execution noise or random guessing.

**Parameters:**
- alpha: [0.0, 1.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Take-The-Best (TTB) Component
    # Sort features by descending validity. Find the first feature that discriminates.
    order = np.argsort(val)[::-1]
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Tallying Component
    # Count strict feature-wise wins for each option.
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # Mixture of heuristics
    p_mixed = alpha * p_ttb + (1.0 - alpha) * p_tally
    
    # Incorporate shared response noise (lapse rate)
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
**Description:** Decision makers employ a dual-process or strategy mixture approach when evaluating multi-attribute options. Rather than relying entirely on a single strategy, choices are generated by a probabilistic mixture of a simple, unweighted Tallying heuristic (which counts the number of strictly winning features) and a compensatory Weighted Additive (WADD) strategy (which integrates all features weighted by their subjective validities). To ensure equitable application of choice determinism, the evidence scores for both strategies are normalized to a common [0, 1] scale before applying a shared inverse temperature parameter. The mixture parameter 'alpha' dictates the reliance on Tallying versus WADD, allowing the model to capture exact chance-level responding in scenarios where features tie while maintaining sensitivity to cue validities in general.

**Parameters:**
- beta: [0.01, 10.0]
- gamma: [0.0, 5.0]
- alpha: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # WADD Component: Weighted sum using non-linearly scaled validities, normalized to [0, 1]
    subjective_weights = val ** gamma
    sum_weights = np.sum(subjective_weights)
    score_a_wadd = np.sum(a * subjective_weights) / sum_weights
    score_b_wadd = np.sum(b * subjective_weights) / sum_weights
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Tallying Component: Count of strict feature-wise wins, normalized to [0, 1]
    a_wins = float(np.sum(a > b)) / n_features
    b_wins = float(np.sum(b > a)) / n_features
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of the two strategies
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    # Incorporate response noise (lapse rate)
    return (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
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
[0] rationale: This metric isolates trials 1, 3, and 5. In these specific trials, the single most valid discriminating feature always favors Option A, meaning the Take-The-Best (TTB) heuristic will universally choose A. However, the sheer number of remaining moderately valid features strongly favors Option B, meaning the Tallying heuristic will universally choose B. 

Under the Advocated theory (TTB + Tallying), subjects will choose Option A at a rate roughly equal to their mixture parameter 'alpha' (averaging around 50%). 

Under the Competing theory (WADD + Tallying), the Weighted Additive (WADD) strategy will also strongly favor Option B for most realistic values of the non-linear scaling parameter 'gamma', because the sum of the subjective weights for B's features outweighs A's single advantage. Since both WADD and Tallying favor B, the Competing model will predict a much lower rate of choosing Option A. Measuring the choice proportion on these specific trials provides a robust, large-margin dissociation between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    target_A = {
        (1, 0, 0, 0, 0),
        (1, 0, 1, 0, 0),
        (0, 1, 0, 0, 0)
    }
    
    # Convert option lists to tuples to make them hashable for comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    mask = a_tuples.isin(target_A)
    
    subset = data[mask]
    if len(subset) == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((subset['response'] == 0).mean())
outcome: self_sim=0.4428 (var=0.0415) adversary_sim=0.3867 (var=0.0231) welch_t=+1.105 p=0.2753 (N=25, alpha=0.01) -> reject

[1] rationale: Under the Advocated theory (TTB + Tallying), both Trial 2 and Trial 3 are identical from the perspective of the heuristics: TTB favors Option A in both (because the most valid feature f0 is 1 for A and 0 for B), and Tallying favors Option B in both (because B wins 3 features to 2). Therefore, the choice probabilities are determined by the exact same mixture of heuristics, and the expected difference in P(A) between these trials is exactly 0. Under the Competing theory (WADD + Tallying), however, Trial 2 and Trial 3 are treated very differently by the compensatory WADD strategy. In Trial 2, Option A has the top two features (0.95, 0.75), which generally outweighs B's three weaker features. In Trial 3, Option A has the 1st and 3rd features (0.95, 0.6), while Option B has the 2nd, 4th, and 5th, which makes WADD favor B (or favor A much less strongly than in Trial 2). Thus, the Competing theory predicts a significantly higher P(A) in Trial 2 than in Trial 3, creating a robust, zero-centered dissociation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert ratings to tuples for hashable comparison
    a_tups = data['option_a_ratings'].apply(tuple)
    
    # Trial 2: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    mask2 = a_tups == (1, 1, 0, 0, 0)
    # Trial 3: A=[1, 0, 1, 0, 0], B=[0, 1, 0, 1, 1]
    mask3 = a_tups == (1, 0, 1, 0, 0)
    
    p_a_t2 = (data.loc[mask2, 'response'] == 0).mean() if mask2.sum() > 0 else 0.0
    p_a_t3 = (data.loc[mask3, 'response'] == 0).mean() if mask3.sum() > 0 else 0.0
    
    # We measure the difference in preference for Option A between Trial 2 and Trial 3.
    return float(p_a_t2 - p_a_t3)
outcome: self_sim=0.0283 (var=0.0257) adversary_sim=0.0700 (var=0.0520) welch_t=-0.747 p=0.4591 (N=25, alpha=0.01) -> reject

[2] rationale: This metric measures the variance in the probability of choosing Option A across four specific trials (Trials 1, 2, 3, and 5). In all four of these trials, the single most valid discriminating feature favors Option A (so TTB chooses A), while the sheer number of remaining features favors Option B (so Tallying chooses B). Under the Advocated theory (TTB + Tallying), the heuristic conflict is identical across all four trials, meaning the expected probability of choosing A is exactly the same for a given subject (determined by their mixture parameter alpha and lapse rate). Thus, the variance across these four trial types should be near zero (only reflecting binomial noise). Under the Competing theory (WADD + Tallying), the compensatory WADD strategy integrates the specific weights of the features, which vary significantly across these four trials. Therefore, WADD's preference for A will shift dramatically between them, leading to a much higher variance in the choice probabilities.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_tups = data['option_a_ratings'].apply(tuple)
    
    # Trials where TTB chooses A and Tallying chooses B
    t1 = (1, 0, 0, 0, 0)
    t2 = (1, 1, 0, 0, 0)
    t3 = (1, 0, 1, 0, 0)
    t5 = (0, 1, 0, 0, 0)
    
    p1 = (data.loc[a_tups == t1, 'response'] == 0).mean()
    p2 = (data.loc[a_tups == t2, 'response'] == 0).mean()
    p3 = (data.loc[a_tups == t3, 'response'] == 0).mean()
    p5 = (data.loc[a_tups == t5, 'response'] == 0).mean()
    
    ps = [p for p in [p1, p2, p3, p5] if not pd.isna(p)]
    if len(ps) < 2:
        return 0.0
        
    return float(np.var(ps))
outcome: self_sim=0.0002 (var=0.0001) adversary_sim=0.0038 (var=0.0003) welch_t=-0.874 p=0.3878 (N=25, alpha=0.01) -> reject

[3] rationale: This metric isolates Trial 2 and compares it against a baseline of Trials 1, 3, and 5. In all four of these trials, the single most valid discriminating feature favors Option A (so Take-The-Best universally chooses A), while the sheer number of remaining features favors Option B (so Tallying universally chooses B). Under the Advocated theory (TTB + Tallying), the heuristic conflict is identical across all four trial types, meaning the expected probability of choosing A is exactly the same for a given subject (determined by their mixture parameter alpha). Thus, the expected difference is exactly 0. Under the Competing theory (WADD + Tallying), the compensatory WADD strategy integrates the specific weights of the features. In Trial 2, WADD slightly favors Option A. In Trials 1, 3, and 5, WADD strongly favors Option B. Therefore, the Competing theory predicts a significantly higher P(A) in Trial 2 compared to the other three. By averaging three baseline trials, we reduce binomial noise, creating a tight, highly discriminative zero-centered contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert option lists to tuples to make them hashable for comparison
    a_tups = data['option_a_ratings'].apply(tuple)
    
    # Trial 2: TTB favors A, Tally favors B. WADD slightly favors A.
    mask2 = a_tups == (1, 1, 0, 0, 0)
    
    # Trials 1, 3, and 5: TTB favors A, Tally favors B. WADD strongly favors B.
    mask_others = a_tups.isin({
        (1, 0, 0, 0, 0),  # Trial 1
        (1, 0, 1, 0, 0),  # Trial 3
        (0, 1, 0, 0, 0)   # Trial 5
    })
    
    p2 = (data.loc[mask2, 'response'] == 0).mean()
    p_others = (data.loc[mask_others, 'response'] == 0).mean()
    
    if pd.isna(p2) or pd.isna(p_others):
        return 0.0
        
    # Return the difference in preference for Option A
    return float(p2 - p_others)
outcome: self_sim=0.0339 (var=0.0188) adversary_sim=0.1167 (var=0.0432) welch_t=-1.662 p=0.104 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Convert option lists to tuples to make them hashable for comparison\n    a_tups = data['option_a_ratings'].apply(tuple)\n    \n    # Trial Type 1: The option with the most valid feature (f0) also has the least valid feature (f4).\n    # A = [1, 0, 0, 0, 1], B = [0, 1, 1, 1, 0] (and its symmetric counterpart)\n    m_t1_a = a_tups == (1, 0, 0, 0, 1)\n    m_t1_b = a_tups == (0, 1, 1, 1, 0)\n    \n    # Trial Type 3: The option with the most valid feature (f0) has NO other features.\n    # A = [1, 0, 0, 0, 0], B = [0, 1, 1, 1, 1] (and its symmetric counterpart)\n    m_t3_a = a_tups == (1, 0, 0, 0, 0)\n    m_t3_b = a_tups == (0, 1, 1, 1, 1)\n    \n    # Count how often the option possessing the most valid feature (f0=1) was chosen\n    t1_chosen = data.loc[m_t1_a, 'response'].eq(0).sum() + data.loc[m_t1_b, 'response'].eq(1).sum()\n    t1_total = m_t1_a.sum() + m_t1_b.sum()\n    \n    t3_chosen = data.loc[m_t3_a, 'response'].eq(0).sum() + data.loc[m_t3_b, 'response'].eq(1).sum()\n    t3_total = m_t3_a.sum() + m_t3_b.sum()\n    \n    p_t1 = t1_chosen / t1_total if t1_total > 0 else 0.0\n    p_t3 = t3_chosen / t3_total if t3_total > 0 else 0.0\n    \n    # Return the difference in preference for the f0=1 option between Trial Type 1 and Trial Type 3\n    return float(p_t1 - p_t3)",
  "rationale": "Under the Advocated theory (TTB + Tallying), Trial Type 1 and Trial Type 3 present the exact same heuristic conflict: TTB chooses the option with f0=1, while Tallying chooses the alternative (since 2<3 in T1 and 1<4 in T3). Because the heuristic mixture is identical, the expected difference in the probability of choosing the f0=1 option is exactly zero. Under the Competing theory (WADD + Tallying), however, the compensatory WADD strategy and the Tallying strategy both register much stronger evidence against the f0=1 option in Trial Type 3 (where it has only 1 feature vs 4) than in Trial Type 1 (where it has 2 features vs 3). Thus, the Competing model predicts a significantly positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 6159,
  "candidates_token_count": 781,
  "total_token_count": 12322
}
```
