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

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 9: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 10: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate Probabilistic Cue Selection (Stochastic TTB) from standard Take-The-Best (TTB), we exploit their fundamentally different treatments of subordinate discriminating features. Standard TTB is strictly lexicographic: the choice and its predicted probability (before uniform lapse) depend entirely on the single highest-validity discriminating feature, with all other features strictly ignored. In contrast, Stochastic TTB samples among ALL discriminating features with probabilities proportional to exp(gamma * validity). By systematically varying the number and validities of the subordinate cues that oppose the highest-validity cue, we can test whether choice probabilities remain constant (as predicted by pure TTB) or shift in proportion to the opposing cues' validities (as predicted by Stochastic TTB). We include trials where the best cue is opposed by multiple highly valid cues, potentially causing a preference reversal under Stochastic TTB but not TTB.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Cue Selection (Stochastic TTB): Human decision-makers use a non-compensatory, one-reason heuristic but are stochastic in their cue retrieval. Instead of strictly ranking features by validity, subjects sample a feature to evaluate with a probability proportional to its validity (via a softmax). If the sampled feature discriminates between the options, they base their choice entirely on that feature. If it does not discriminate, they sample again. This maintains the non-compensatory nature of the decision while naturally introducing variability in which cue is selected, offering a mechanistic explanation for choice noise without relying on compensatory tallying.

**Parameters:**
- gamma: [0.0, 50.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which features discriminate between A and B
    discrim = (a != b)
    
    if not np.any(discrim):
        # If no features discriminate, the decision is a random guess
        p_core = np.array([0.5, 0.5])
    else:
        # The subject repeatedly samples features proportional to exp(gamma * validity)
        # until a discriminating feature is found. The probability that the first
        # discriminating feature found is feature i is equivalent to a softmax over
        # the validities restricted to the set of discriminating features.
        z = gamma * validities[discrim]
        z = z - np.max(z)  # For numerical stability
        w = np.exp(z)
        w = w / np.sum(w)
        
        # The chosen discriminating feature dictates the choice entirely.
        # Sum the probabilities of sampling a feature that favors A vs B.
        favor_a = (a[discrim] > b[discrim])
        favor_b = (b[discrim] > a[discrim])
        
        p_a = np.sum(w[favor_a])
        p_b = np.sum(w[favor_b])
        
        p_core = np.array([p_a, p_b])
        
    # Blend with a uniform lapse rate for general response noise/inattention
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Take-The-Best (TTB) heuristic: People make decisions by ranking features according to their validities and choosing the option that is favored by the single most valid discriminating feature. If no feature discriminates, they guess. This is a lexicographic, non-compensatory strategy. However, human execution of this strategy is highly noisy, so choice probabilities are heavily tempered by response noise (low beta) and random guessing lapses (high epsilon).

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to preserve the original feature order in case of ties.
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    
    scores = np.array([0.0, 0.0])
    # Find the first feature that discriminates between the two options
    for idx in ranked_features:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    # If no feature discriminates, the core preference is uniform
    if scores[0] == 0.0 and scores[1] == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        # Softmax over the scores to introduce response noise
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between trials with a single opposing cue and trials with multiple opposing cues. According to standard Take-The-Best (TTB), the decision is based strictly on the single highest-validity discriminating feature, ignoring all subordinate cues. Therefore, TTB predicts this difference will be zero. In contrast, Probabilistic Cue Selection (Stochastic TTB) predicts that the presence of multiple opposing cues will draw probability mass away from the highest-validity cue, making Option A less likely to be chosen when opposed by multiple valid cues. Thus, Stochastic TTB predicts a significantly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert ratings to string for easy matching
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Response == 0 means Option A was chosen
    choose_A = (data['response'] == 0).astype(float)
    
    # Group 1: Single opposing cue
    # T1: Best cue favors A, 1 opposing cue favors B
    # T5: Best cue favors A, 1 opposing cue favors B
    t1_mask = (a_str == '10000') & (b_str == '01000')
    t5_mask = (a_str == '01000') & (b_str == '00100')
    
    # Group 2: Multiple opposing cues
    # T3: Best cue favors A, 4 opposing cues favor B
    # T6: Best cue favors A, 3 opposing cues favor B
    t3_mask = (a_str == '10000') & (b_str == '01111')
    t6_mask = (a_str == '01000') & (b_str == '00111')
    
    p_a_single = choose_A[t1_mask | t5_mask].mean()
    p_a_multiple = choose_A[t3_mask | t6_mask].mean()
    
    # Handle cases where subjects might not have these trials (though experimental design says they do)
    if pd.isna(p_a_single) or pd.isna(p_a_multiple):
        return 0.0
        
    return float(p_a_single - p_a_multiple)
outcome: self_sim=-0.0111 (var=0.0200) adversary_sim=0.0256 (var=0.0277) welch_t=-0.840 p=0.4054 (N=25, alpha=0.01) -> reject

[1] rationale: Standard Take-The-Best (TTB) incorporates a noise parameter (beta) bounded between 0 and 2, imposing a strict mathematical ceiling on its choice consistency. Even with zero lapse rate (epsilon=0), TTB's probability of choosing the favored option cannot exceed exp(2)/(exp(2)+1) ≈ 0.88. In contrast, Stochastic TTB's inverse temperature (gamma) ranges up to 50, allowing it to behave almost deterministically (consistency ≈ 1.0) when epsilon is low. By computing the overall proportion of TTB-favored choices and raising it to the 4th power, we non-linearly amplify the upper tail of the accuracy distribution. This heavily rewards the near-perfect consistency achievable only by Stochastic TTB, while severely penalizing the inherently noisy choices of TTB, resulting in a large and statistically significant difference in means.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Extract ratings as 2D numpy arrays for vectorized operations
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Find the highest-validity discriminating feature for each trial.
    # Since validities are strictly decreasing [0.95, 0.85, ...], 
    # this is simply the first index where the features differ.
    diff = a_ratings - b_ratings
    non_zero = diff != 0
    first_idx = np.argmax(non_zero, axis=1)
    
    # Determine which option is favored by this feature
    first_diff = diff[np.arange(len(diff)), first_idx]
    
    # If first_diff == 1, A is favored (response 0)
    # If first_diff == -1, B is favored (response 1)
    ttb_favored = (first_diff == -1).astype(int)
    
    # Calculate the subject's overall consistency with the TTB-favored option
    is_favored = (data['response'] == ttb_favored).astype(float)
    
    # Return the 4th power of the mean accuracy to heavily penalize noisy strategies
    return float(np.mean(is_favored) ** 4)
outcome: self_sim=0.1970 (var=0.0410) adversary_sim=0.1490 (var=0.0166) welch_t=+0.999 p=0.3236 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Extract ratings as 2D numpy arrays\n    a_ratings = np.array(data['option_a_ratings'].tolist())\n    b_ratings = np.array(data['option_b_ratings'].tolist())\n    \n    # Find the highest-validity discriminating feature for each trial.\n    # Since validities are strictly decreasing, this is the first non-zero difference.\n    diff = a_ratings - b_ratings\n    non_zero = diff != 0\n    first_idx = np.argmax(non_zero, axis=1)\n    first_diff = diff[np.arange(len(diff)), first_idx]\n    \n    # Determine the TTB-favored option (0 for A, 1 for B)\n    ttb_favored = (first_diff == -1).astype(int)\n    is_favored = (data['response'] == ttb_favored).astype(float)\n    \n    # Count the number of discriminating features favoring each option\n    a_discrim_count = np.sum((diff > 0), axis=1)\n    b_discrim_count = np.sum((diff < 0), axis=1)\n    \n    favored_count = np.where(ttb_favored == 0, a_discrim_count, b_discrim_count)\n    unfavored_count = np.where(ttb_favored == 0, b_discrim_count, a_discrim_count)\n    \n    # Isolate \"Aligned Trials\": trials where the TTB-favored option is opposed \n    # by an equal or fewer number of discriminating features.\n    aligned_mask = favored_count >= unfavored_count\n    \n    if np.sum(aligned_mask) == 0:\n        return 0.5\n        \n    # Return the mean accuracy on these aligned trials\n    return float(np.mean(is_favored[aligned_mask]))",
  "rationale": "To cleanly discriminate the models, we isolate 'aligned trials' where the TTB-favored option has at least as many discriminating features as the unfavored option (e.g., Trial 1, 5, 9, 10). On these trials, Stochastic TTB's choice probability rapidly approaches 1.0 (before lapse) because the sum of weights for the favored option heavily dominates. In contrast, standard TTB's choice probability is mathematically capped at ~0.88 (before lapse) due to its bounded softmax noise parameter (beta <= 2.0), regardless of how many subordinate cues align. By computing the subject's accuracy exclusively on these aligned trials, we maximize the measurable gap between STTB's near-perfect consistency and TTB's inherently capped consistency, while preserving enough trials to keep binomial variance low."
}
```

## Usage

```json
{
  "prompt_token_count": 4899,
  "candidates_token_count": 685,
  "total_token_count": 32843
}
```
