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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate deterministic Take The Best (TTB) from Probabilistic Take The Best (Plackett-Luce Cue Search), we use a 5-feature design with spaced validities. TTB assumes a strict lexicographic stopping rule where the choice is entirely determined by the single highest-validity discriminating cue, meaning the number of lower-validity cues favoring the opposite option should have zero effect on choice probabilities. In contrast, Probabilistic TTB models cue search as a Plackett-Luce process where the probability of encountering a cue first is proportional to its exponentiated validity. Consequently, Probabilistic TTB predicts that adding more lower-validity cues favoring the opposing option will cumulatively draw choice probability away from the option favored by the best cue. By comparing trials where the best cue faces off against a single lower-validity cue versus trials where it faces off against multiple lower-validity cues, we can sharply distinguish TTB's invariant choice probabilities from Probabilistic TTB's sensitivity to the number of opposing cues.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Probabilistic Take The Best (Plackett-Luce Cue Search) with High Determinism

**Parameters:**
- gamma: [0.0, 100.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues discriminate and in which direction
    diff = a - b
    favor_a = (diff > 0)
    favor_b = (diff < 0)
    discriminating = favor_a | favor_b
    
    if not np.any(discriminating):
        # No cues discriminate, guess uniformly
        p_a = 0.5
    else:
        # The probability that a given discriminating cue is encountered first 
        # under a Plackett-Luce search order depends only on the relative weights 
        # of the discriminating cues.
        # Subtract max validity for numerical stability before exponentiation.
        max_val = np.max(val[discriminating])
        w = np.exp(gamma * (val - max_val))
        
        weight_a = np.sum(w[favor_a])
        weight_b = np.sum(w[favor_b])
        
        total_weight = weight_a + weight_b
        p_a = weight_a / total_weight
        
    p_core = np.array([p_a, 1.0 - p_a])
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
[0] rationale: Calculates the overall proportion of times Option B is chosen. The experimental design is constructed such that the highest-validity discriminating cue always favors Option A, while lower-validity cues frequently favor Option B. Deterministic Take The Best ignores lower-validity cues and will almost never choose Option B (mean response near 0). Probabilistic Take The Best samples from all discriminating cues, so the cumulative weight of the lower-validity cues will lead to a substantially higher proportion of Option B choices.
metric_source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
outcome: self_sim=0.1215 (var=0.0084) adversary_sim=0.1156 (var=0.0173) welch_t=+0.182 p=0.8566 (N=25, alpha=0.01) -> reject

[1] rationale: In every trial of this experimental design, the highest-validity discriminating cue always favors Option A. Deterministic Take The Best (TTB) uses only this single best cue and ignores all others, so its probability of choosing B (due to noise/lapses) is identical across all trials. Therefore, the difference in B choices between any two subsets of trials will be exactly 0 in expectation for TTB. In contrast, Probabilistic Take The Best samples from all discriminating cues. When Option B is supported by a larger number of lower-validity cues (e.g., 3 or 4), its cumulative probability weight increases compared to trials where it is supported by fewer cues. Thus, Probabilistic TTB predicts a strictly positive difference in the proportion of B choices between the 'high B-cue' and 'low B-cue' trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_matrix = np.vstack(data['option_a_ratings'].values)
    b_matrix = np.vstack(data['option_b_ratings'].values)
    
    # Count how many cues strictly favor Option B over Option A
    favor_b = np.sum(b_matrix > a_matrix, axis=1)
    
    # Group trials by whether B has many (>=3) or few (<=2) favoring cues
    high_mask = favor_b >= 3
    low_mask = favor_b <= 2
    
    if not np.any(high_mask) or not np.any(low_mask):
        return 0.0
        
    # Return the difference in the proportion of B choices
    return float(data['response'].values[high_mask].mean() - data['response'].values[low_mask].mean())
outcome: self_sim=0.0096 (var=0.0034) adversary_sim=0.0196 (var=0.0074) welch_t=-0.482 p=0.6325 (N=25, alpha=0.01) -> reject

[2] rationale: To maximize the contrast between deterministic Take The Best (TTB) and Probabilistic TTB, we compare the two extreme trials in the design. In Trial 1, Option B is supported by four lower-validity cues, exerting the maximum possible 'pull' on Probabilistic TTB to choose B. In Trial 8, Option B is supported by only two cues, while Option A is supported by three. Deterministic TTB ignores all of these lower-validity cues and relies solely on the highest-validity cue (which favors A in both trials), thus predicting an identical rate of B choices (driven purely by noise) in both trials. Probabilistic TTB, however, integrates all cues and predicts a higher rate of B choices in Trial 1 compared to Trial 8. By taking the direct difference P(B | Trial 1) - P(B | Trial 8), we isolate the effect of cue integration while perfectly controlling for subject-specific baseline noise levels.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_matrix = np.vstack(data['option_a_ratings'].values)
    b_matrix = np.vstack(data['option_b_ratings'].values)
    
    # Trial 1: A=[1,0,0,0,0], B=[0,1,1,1,1]
    t1_mask = (a_matrix[:, 0] == 1) & (a_matrix[:, 1] == 0) & (b_matrix[:, 1] == 1) & (b_matrix[:, 4] == 1)
    
    # Trial 8: A=[1,0,0,1,1], B=[0,1,1,0,0]
    t8_mask = (a_matrix[:, 0] == 1) & (a_matrix[:, 3] == 1) & (b_matrix[:, 1] == 1) & (b_matrix[:, 4] == 0)
    
    if not np.any(t1_mask) or not np.any(t8_mask):
        return 0.0
        
    return float(data['response'].values[t1_mask].mean() - data['response'].values[t8_mask].mean())
outcome: self_sim=0.0308 (var=0.0097) adversary_sim=0.0292 (var=0.0160) welch_t=+0.052 p=0.9588 (N=25, alpha=0.01) -> reject

[3] rationale: In all trials of this design, the highest-validity cue (index 0) favors Option A. The advocated theory (deterministic Take The Best) only ever consults this top cue, meaning its probability of choosing B is driven entirely by noise and is mathematically identical across all trials. Therefore, the difference in B choices between any two sets of trials is exactly 0 in expectation. The competing theory (Probabilistic Take The Best), however, weights all discriminating cues exponentially by their validity. When Option B has the second-highest validity cue (Trials 1, 6, 8), its relative weight can still be substantial, leading to a higher rate of B choices. When Option B lacks the second and third highest cues (Trials 4, 5), its highest available cue is much weaker, exponentially decreasing its probability of being chosen under the Plackett-Luce rule. By contrasting these specific trial subsets, we isolate the exponential decay of cue weights in the competing theory, predicting a strictly positive difference, while maintaining a firm prediction of zero difference for the advocated theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    b_matrix = np.vstack(data['option_b_ratings'].values)
    
    # Trials where Option B possesses the second-highest validity cue (index 1)
    mask_high = b_matrix[:, 1] == 1
    
    # Trials where Option B lacks both the second and third highest validity cues (indices 1 and 2)
    mask_low = (b_matrix[:, 1] == 0) & (b_matrix[:, 2] == 0)
    
    if not np.any(mask_high) or not np.any(mask_low):
        return 0.0
        
    return float(data['response'].values[mask_high].mean() - data['response'].values[mask_low].mean())
outcome: self_sim=-0.0011 (var=0.0039) adversary_sim=0.0294 (var=0.0065) welch_t=-1.498 p=0.1412 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Create a hashable representation of the options to uniquely identify each trial type\n    data['trial_key'] = data['option_a_ratings'].apply(tuple)\n    \n    # Calculate the mean response (probability of choosing B) for each trial type within the subject\n    trial_means = data.groupby('trial_key')['response'].mean()\n    \n    # Return the variance of these trial means\n    # For deterministic Take The Best, the true probability of choosing B is identical across all trials \n    # (driven entirely by constant lapse and noise rates), so the variance of trial means is purely binomial noise.\n    # For Probabilistic Take The Best, the true probability of choosing B varies across trials depending on the \n    # cumulative weight of cues favoring B, introducing additional structural variance between the trial means.\n    return float(trial_means.var())",
  "rationale": "In this experimental design, the highest-validity discriminating cue always favors Option A. The advocated theory (deterministic Take The Best) exclusively consults this single best cue and ignores all others. Consequently, its true probability of choosing Option B is identical across all six trial types, driven entirely by subject-specific baseline noise (beta) and lapses (epsilon). For a TTB subject, the variance of their mean response across the six trial types is purely the result of finite-sample binomial sampling noise.\n\nIn contrast, the competing theory (Probabilistic Take The Best) integrates all discriminating cues using a Plackett-Luce process. Because the number of lower-validity cues favoring Option B varies across the six trial types (from 1 to 4), the true probability of choosing Option B fluctuates systematically from trial to trial for any subject with a finite gamma parameter. This introduces structural, within-subject variance to the trial means, over and above binomial noise. By computing the variance of the trial means for each subject, we directly capture this excess structural variability predicted by the competing theory, while maintaining a firm, noise-only baseline for the advocated theory."
}
```

## Usage

```json
{
  "prompt_token_count": 5357,
  "candidates_token_count": 458,
  "total_token_count": 35464
}
```
