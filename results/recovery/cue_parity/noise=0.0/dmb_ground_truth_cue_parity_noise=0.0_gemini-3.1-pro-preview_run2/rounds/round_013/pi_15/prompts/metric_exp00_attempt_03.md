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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 3: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated 'Adaptive Environment-Structured Strategy Selection' (Strategy Mixture) model from the competing 'Sequential Evidence Accumulation' (SCA) theory by exploiting their divergent handling of secondary cues. The advocated theory posits a mixture of Take-The-Best (which relies exclusively on the first discriminating cue) and Tallying (which relies exclusively on the unweighted sum of cues). Across a sequence of trials, we hold two factors strictly constant: the highest discriminating cue (which fixes the TTB choice) and the total number of cues favoring each option (which fixes the Tallying difference). Because neither component of the advocated theory is sensitive to the specific validities of the secondary cues, it predicts identically flat choice probabilities across all these trials. SCA, in contrast, accumulates evidence based on the non-linearly scaled validities of all evaluated cues. By systematically shifting the secondary cues supporting the top-cue winner from high validity to low validity, SCA predicts a systematic, graded change in choice probabilities, directly contradicting the flat-line prediction of the advocated theory.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Adaptive Environment-Structured Strategy Selection: Decision-makers adapt their choice strategy based on the global statistical structure of the environment. Specifically, they evaluate the dispersion (e.g., standard deviation) of cue validities. In environments with high dispersion, they probabilistically favor a non-compensatory heuristic (Take-The-Best). In environments with low dispersion, they fall back to a simpler compensatory heuristic (Equal-weights Tallying). The mixture between these strategies is dynamically governed by a logistic function of the environment's validity dispersion, with independent confidence (temperature) parameters for each strategy.

**Parameters:**
- theta: [0.0, 0.3]
- k: [1.0, 100.0]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
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
    k = float(parameters["k"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate dispersion of validities in the environment
    dispersion = float(np.std(val))
    
    # Logistic function for strategy weight
    # Higher dispersion -> higher probability of using Take-The-Best
    w_ttb = 1.0 / (1.0 + np.exp(-k * (dispersion - theta)))
    
    # Take-The-Best (TTB) component
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_diff = 0.0
    for j in cue_order:
        if a[j] > b[j]:
            ttb_diff = 1.0
            break
        elif b[j] > a[j]:
            ttb_diff = -1.0
            break
            
    scores_ttb = np.array([ttb_diff, -ttb_diff])
    
    # Tallying component
    # Normalized by the number of cues to ensure it's on a comparable scale to TTB (-1 to 1)
    tally_diff = (np.sum(a) - np.sum(b)) / len(val)
    scores_tally = np.array([tally_diff, -tally_diff])
    
    # Convert scores to probabilities
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    p_ttb = np.exp(z_ttb) / np.sum(np.exp(z_ttb))
    
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # Mixture prediction
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
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
[0] rationale: The metric computes the difference in the probability of choosing Option A between Trial 1 and Trial 6. In both trials, Option A is favored by the most valid cue (cue 0) and has the same total number of positive cues (3 vs 2). The advocated Strategy Mixture theory relies entirely on a mixture of Take-The-Best (which looks only at the first cue) and Tallying (which looks only at the unweighted sum of cues), so it predicts identically flat choice probabilities across these two trials (difference = 0). Conversely, the competing Sequential Evidence Accumulation theory evaluates cues in order with non-linear weights. In Trial 1, the early cues strongly favor A, leading to a high probability of choosing A. In Trial 6, the secondary cues favor B, which pulls the accumulated evidence down and reduces the probability of choosing A. Thus, SCA predicts a strongly positive difference, cleanly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert ratings to tuples for hashable comparison
    a_tup = data['option_a_ratings'].apply(tuple)
    
    # Identify Trial 1 and Trial 6 based on Option A's features
    is_t1 = a_tup == (1, 1, 1, 0, 0)
    is_t6 = a_tup == (1, 0, 0, 1, 1)
    
    # Calculate the probability of choosing Option A (response == 0)
    # For both trials, Option A is the one favored by the most valid cue (cue 0)
    p_a_t1 = 1.0 - data[is_t1]['response'].mean()
    p_a_t6 = 1.0 - data[is_t6]['response'].mean()
    
    # Return the difference in choice probability
    return float(p_a_t1 - p_a_t6)
outcome: self_sim=-0.0167 (var=0.0167) adversary_sim=0.0600 (var=0.0489) welch_t=-1.497 p=0.1425 (N=25, alpha=0.01) -> reject

[1] rationale: The metric computes the difference in the probability of choosing the option favored by the top cue (Cue 0) between trials where the second cue (Cue 1) aligns with the top cue versus trials where it opposes the top cue. The advocated Strategy Mixture theory relies on a mixture of Take-The-Best (which only looks at Cue 0) and Tallying (which looks at the unweighted sum of cues). Since the total number of supporting cues is held constant across these trial groupings, the advocated theory predicts identically flat choice probabilities (difference = 0). The competing Sequential Evidence Accumulation theory evaluates cues in order with non-linear weights. When Cue 1 aligns with Cue 0, accumulated evidence is stronger, leading to a higher choice probability than when Cue 1 opposes Cue 0. By aggregating across 48 trials per condition, this metric heavily suppresses binomial noise, cleanly and reliably dissociating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Determine which option is favored by the most valid cue (Cue 0)
    a_favored = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    
    # Check if the subject chose the option favored by Cue 0
    favored_chosen = ((a_favored) & (data['response'] == 0)) | ((~a_favored) & (data['response'] == 1))
    
    # Determine if the second most valid cue (Cue 1) also supports the favored option
    a_has_cue1 = data['option_a_ratings'].apply(lambda x: x[1] == 1)
    b_has_cue1 = data['option_b_ratings'].apply(lambda x: x[1] == 1)
    cue1_supports_favored = ((a_favored) & a_has_cue1) | ((~a_favored) & b_has_cue1)
    
    # Calculate the mean probability of choosing the favored option in both conditions
    p_support = favored_chosen[cue1_supports_favored].mean()
    p_oppose = favored_chosen[~cue1_supports_favored].mean()
    
    return float(p_support - p_oppose)
outcome: self_sim=0.0008 (var=0.0054) adversary_sim=0.0392 (var=0.0140) welch_t=-1.377 p=0.1762 (N=25, alpha=0.01) -> reject

[2] rationale: This metric computes a linear contrast across the 8 trial types to test for a monotonic effect of secondary cue validity. In all trials, the Option favored by Cue 0 also has the same total number of supporting cues (3 vs 2). Because the advocated Strategy Mixture theory is a mixture of Take-The-Best (which looks only at Cue 0) and Tallying (which looks only at the unweighted sum), it predicts identically flat choice probabilities across all trials, yielding an expected contrast of exactly 0. The competing Sequential Evidence Accumulation theory evaluates cues in order with non-linear weights, predicting a monotonic decrease in choice probability as the secondary cues shift from high validity (early indices) to low validity (late indices). By applying a continuous linear weight (5 - S) and averaging over all 96 trials, this metric extracts the maximal signal of the monotonic trend while heavily suppressing binomial noise, thereby minimizing between-subject variance and maximizing discriminability.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify if option A is the one favored by the most valid cue (Cue 0)
    a_c0 = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    
    # Did the subject choose the Cue 0 winner?
    chose_winner = ((a_c0) & (data['response'] == 0)) | ((~a_c0) & (data['response'] == 1))
    
    # Calculate S: sum of indices (1 to 4) where the winner has a 1
    # This quantifies how "early" the secondary evidence appears.
    def get_S(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        winner = a if a[0] == 1 else b
        return sum(i for i in range(1, 5) if winner[i] == 1)
        
    S = data.apply(get_S, axis=1)
    
    # The expected sum of indices for the secondary cues in this design is 5.
    # Weights: +2 for strongest secondary evidence (T1, T7), -2 for weakest (T6, T8).
    weights = 5 - S
    
    # Return the weighted average, which acts as a linear contrast for the monotonic trend.
    return float((chose_winner * weights).mean())
outcome: self_sim=0.0060 (var=0.0027) adversary_sim=0.0342 (var=0.0075) welch_t=-1.396 p=0.1707 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    c1_chosen = []\n    c4_chosen = []\n    for _, row in data.iterrows():\n        chosen_ratings = row['option_a_ratings'] if row['response'] == 0 else row['option_b_ratings']\n        c1_chosen.append(chosen_ratings[1])\n        c4_chosen.append(chosen_ratings[4])\n    return float(np.mean(c1_chosen) - np.mean(c4_chosen))",
  "rationale": "This metric calculates the difference in the proportion of trials where the subject chose the option supported by the most valid secondary cue (Cue 1) versus the least valid secondary cue (Cue 4). The advocated Strategy Mixture theory relies on Take-The-Best (which only looks at Cue 0) and Tallying (which weights all cues equally). Because the total number of supporting cues is held constant across options in this design, the advocated theory is completely blind to the difference between Cue 1 and Cue 4, predicting an expected difference of exactly 0 regardless of subject parameters. The competing Sequential Evidence Accumulation theory, however, evaluates cues with non-linear weights based on their validities. It gives much more weight to Cue 1 than Cue 4, predicting that choices will align with Cue 1 significantly more often than with Cue 4. By computing this simple marginal difference over all 96 trials, the metric captures the maximal theoretical contrast while heavily suppressing binomial noise, ensuring minimal between-subject variance and a robust dissociation."
}
```

## Usage

```json
{
  "prompt_token_count": 5401,
  "candidates_token_count": 356,
  "total_token_count": 35498
}
```
