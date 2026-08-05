# metric_exp01_attempt_03

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

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1]
  trial 8: A=[1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1]
  trial 9: A=[1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1]
  trial 10: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 11: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 12: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate Probabilistic Cue Selection (Stochastic TTB) from standard Take-The-Best (TTB), this design strictly controls the number of discriminating features while systematically varying the validity of the opposing feature. Standard TTB's predicted choice probability depends only on the single highest-validity discriminating feature; it predicts perfectly constant choice probabilities across any trials where the best discriminating feature is the same, regardless of the validity of the single opposing feature. In contrast, Stochastic TTB samples among all discriminating features with probabilities proportional to exp(gamma * validity). By creating pairs where exactly two features discriminate (one favoring A, one favoring B) and holding the best feature constant while lowering the validity of the opposing feature, Stochastic TTB predicts a graded increase in the choice probability for A, whereas standard TTB predicts a flat, constant probability.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



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

[2] rationale: To cleanly discriminate the models, we isolate 'aligned trials' where the TTB-favored option has at least as many discriminating features as the unfavored option (e.g., Trial 1, 5, 9, 10). On these trials, Stochastic TTB's choice probability rapidly approaches 1.0 (before lapse) because the sum of weights for the favored option heavily dominates. In contrast, standard TTB's choice probability is mathematically capped at ~0.88 (before lapse) due to its bounded softmax noise parameter (beta <= 2.0), regardless of how many subordinate cues align. By computing the subject's accuracy exclusively on these aligned trials, we maximize the measurable gap between STTB's near-perfect consistency and TTB's inherently capped consistency, while preserving enough trials to keep binomial variance low.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Extract ratings as 2D numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Find the highest-validity discriminating feature for each trial.
    # Since validities are strictly decreasing, this is the first non-zero difference.
    diff = a_ratings - b_ratings
    non_zero = diff != 0
    first_idx = np.argmax(non_zero, axis=1)
    first_diff = diff[np.arange(len(diff)), first_idx]
    
    # Determine the TTB-favored option (0 for A, 1 for B)
    ttb_favored = (first_diff == -1).astype(int)
    is_favored = (data['response'] == ttb_favored).astype(float)
    
    # Count the number of discriminating features favoring each option
    a_discrim_count = np.sum((diff > 0), axis=1)
    b_discrim_count = np.sum((diff < 0), axis=1)
    
    favored_count = np.where(ttb_favored == 0, a_discrim_count, b_discrim_count)
    unfavored_count = np.where(ttb_favored == 0, b_discrim_count, a_discrim_count)
    
    # Isolate "Aligned Trials": trials where the TTB-favored option is opposed 
    # by an equal or fewer number of discriminating features.
    aligned_mask = favored_count >= unfavored_count
    
    if np.sum(aligned_mask) == 0:
        return 0.5
        
    # Return the mean accuracy on these aligned trials
    return float(np.mean(is_favored[aligned_mask]))
outcome: self_sim=0.6800 (var=0.0177) adversary_sim=0.6289 (var=0.0133) welch_t=+1.451 p=0.1534 (N=25, alpha=0.01) -> reject

[3] rationale: Standard Take-The-Best (TTB) makes decisions based EXCLUSIVELY on the single highest-validity discriminating feature. The presence and direction of any subordinate discriminating features are completely ignored. Therefore, TTB mathematically guarantees that the probability of choosing the favored option is identical regardless of whether the second-best discriminating feature supports or opposes the first. In contrast, Probabilistic Cue Selection (Stochastic TTB) samples among all discriminating features. The second-best feature has the second-highest probability of being sampled. If it supports the best feature, the total probability mass favoring that option increases significantly; if it opposes, the mass decreases. By computing the difference in the rate of choosing the TTB-favored option between 'Support' trials (where the top 2 discriminating features agree) and 'Oppose' trials (where they disagree), we isolate this mechanistic difference. TTB guarantees an expected difference of exactly 0.0, while Stochastic TTB predicts a strictly positive difference. Using all available trials minimizes within-subject binomial variance, ensuring a highly significant statistical contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Extract ratings as 2D numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Difference between A and B
    diff = a_ratings - b_ratings
    
    n_trials = len(diff)
    ttb_favored = np.zeros(n_trials, dtype=int)
    is_support = np.zeros(n_trials, dtype=bool)
    
    for i in range(n_trials):
        d = diff[i]
        discrim_indices = np.where(d != 0)[0]
        
        if len(discrim_indices) >= 2:
            first_idx = discrim_indices[0]
            second_idx = discrim_indices[1]
            
            first_favors_A = (d[first_idx] == 1)
            second_favors_A = (d[second_idx] == 1)
            
            ttb_favored[i] = 0 if first_favors_A else 1
            is_support[i] = (first_favors_A == second_favors_A)
        elif len(discrim_indices) == 1:
            first_idx = discrim_indices[0]
            ttb_favored[i] = 0 if d[first_idx] == 1 else 1
            is_support[i] = False
        else:
            ttb_favored[i] = 0
            is_support[i] = False
            
    # Calculate whether the subject chose the option favored by the best cue
    chose_ttb_favored = (data['response'] == ttb_favored).astype(float)
    
    # Return the difference in choice rates between Support and Oppose trials
    if np.sum(is_support) > 0 and np.sum(~is_support) > 0:
        return float(chose_ttb_favored[is_support].mean() - chose_ttb_favored[~is_support].mean())
    return 0.0
outcome: self_sim=0.0075 (var=0.0169) adversary_sim=-0.0128 (var=0.0144) welch_t=+0.573 p=0.5692 (N=25, alpha=0.01) -> reject

[4] rationale: By design, in Trials 1-10 exactly two features discriminate: the more valid one always favors Option A, and the less valid one favors Option B. Standard Take-The-Best (TTB) dictates that the decision is based exclusively on the single best discriminating feature, completely ignoring the opposing feature. Because TTB's noise model only depends on the fact that a discriminating feature was found (assigning a fixed score of 1.0 to the favored option), its predicted probability of choosing Option A is mathematically identical across all these trials, yielding an expected difference of 0.0. In contrast, Stochastic TTB samples between the two discriminating features proportionally to their validities. When the opposing feature is 'weak' (lower validity), it is sampled less often, resulting in a higher probability of choosing Option A compared to when the opposing feature is 'strong' (higher validity). Therefore, Stochastic TTB predicts a strictly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    diff = a_ratings - b_ratings
    
    # Isolate trials where exactly 2 features discriminate (Trials 1-10 in the design)
    n_discrim = np.sum(diff != 0, axis=1)
    mask_2 = (n_discrim == 2)
    
    if np.sum(mask_2) == 0:
        return 0.0
        
    diff_2 = diff[mask_2]
    resp_2 = data['response'].values[mask_2]
    
    # In trials 1-10, the first discriminating feature always favors Option A (diff = 1),
    # and the single opposing feature favors Option B (diff = -1).
    # Find the index of the opposing feature (0-indexed).
    idx_oppose = np.argmax(diff_2 == -1, axis=1)
    
    # Calculate whether the subject chose Option A (response == 0)
    chose_a = 1.0 - resp_2
    
    # Group into Weak Opposition (opposing feature is 4th or 5th, low validity)
    # and Strong Opposition (opposing feature is 2nd or 3rd, high validity)
    mask_weak = idx_oppose >= 3
    mask_strong = idx_oppose <= 2
    
    if np.sum(mask_weak) == 0 or np.sum(mask_strong) == 0:
        return 0.0
        
    p_a_weak = np.mean(chose_a[mask_weak])
    p_a_strong = np.mean(chose_a[mask_strong])
    
    # Return the difference in choice probability for Option A
    return float(p_a_weak - p_a_strong)

outcome: self_sim=0.0290 (var=0.0138) adversary_sim=0.0017 (var=0.0126) welch_t=+0.843 p=0.4032 (N=25, alpha=0.01) -> reject

[5] rationale: In this experimental design, the single highest-validity discriminating feature favors Option A in every single trial. Because standard Take-The-Best (TTB) completely ignores all subordinate features, it predicts that the probability of choosing Option A is identical across all trials, yielding an expected difference of exactly 0.0 between any two subsets of trials. Probabilistic Cue Selection (Stochastic TTB), however, samples among all discriminating features. When the opposing features are weak or few, the probability of choosing A is high; when the opposing features are strong or numerous, the probability of choosing A is heavily diluted. By calculating the theoretical STTB choice probability to perfectly split the trials into a 'Strong A' half and a 'Weak A' half, we compute the difference in the actual choice rates between these two halves. TTB predicts 0.0, while Stochastic TTB predicts a large, strictly positive difference. Using all 96 trials optimally minimizes within-subject binomial variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Use a fixed moderate gamma to compute the theoretical STTB probability of choosing A
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    gamma = 5.0
    weights = np.exp(gamma * validities)
    
    diff = a_ratings - b_ratings
    
    w_A = np.sum((diff == 1) * weights, axis=1)
    w_B = np.sum((diff == -1) * weights, axis=1)
    
    sum_w = w_A + w_B
    p_A_sttb = np.where(sum_w > 0, w_A / sum_w, 0.5)
    
    # In this specific experimental design, the single highest-validity discriminating 
    # feature ALWAYS favors Option A across all 12 unique trials.
    # Therefore, standard TTB predicts the exact same probability of choosing A for ALL trials.
    # We split the trials into two halves based on the theoretical STTB probability:
    # 'Strong A' (where subordinate cues provide weak opposition) and 
    # 'Weak A' (where subordinate cues provide strong/multiple opposition).
    strong_mask = p_A_sttb > 0.7
    weak_mask = p_A_sttb <= 0.7
    
    chose_A = (data['response'] == 0).astype(float).values
    
    if np.sum(strong_mask) == 0 or np.sum(weak_mask) == 0:
        return 0.0
        
    # Return the difference in the rate of choosing Option A
    return float(np.mean(chose_A[strong_mask]) - np.mean(chose_A[weak_mask]))
outcome: self_sim=0.0508 (var=0.0149) adversary_sim=-0.0033 (var=0.0095) welch_t=+1.733 p=0.08979 (N=25, alpha=0.01) -> reject

[6] rationale: In this experimental design, the single highest-validity discriminating feature favors Option A in every single trial. Because standard Take-The-Best (TTB) completely ignores all subordinate features, it predicts that the probability of choosing Option A is identical across all trials, yielding an expected difference of exactly 0.0 between any two subsets of trials. Probabilistic Cue Selection (Stochastic TTB), however, samples among all discriminating features. In Trials 1-10 (where exactly 2 features discriminate), A is opposed by only a single weaker cue, so Stochastic TTB predicts a high probability of choosing A. In Trials 11-12 (where all 5 features discriminate), A is opposed by 3 or 4 subordinate cues, which heavily dilutes the probability mass favoring A. By computing the difference in the choice rate of Option A between these two sets of trials, we establish a metric where TTB predicts exactly 0.0, while Stochastic TTB predicts a large, strictly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Extract ratings as 2D numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Calculate the difference to find discriminating features
    diff = a_ratings - b_ratings
    n_discrim = np.sum(diff != 0, axis=1)
    
    # Response == 0 means Option A was chosen
    chose_A = (data['response'] == 0).astype(float).values
    
    # Trials 1-10 have exactly 2 discriminating features (1 favoring A, 1 favoring B)
    mask_2 = (n_discrim == 2)
    
    # Trials 11-12 have exactly 5 discriminating features (1-2 favoring A, 3-4 favoring B)
    mask_5 = (n_discrim == 5)
    
    if np.sum(mask_2) == 0 or np.sum(mask_5) == 0:
        return 0.0
        
    # Return the difference in the rate of choosing Option A
    return float(np.mean(chose_A[mask_2]) - np.mean(chose_A[mask_5]))
outcome: self_sim=0.0295 (var=0.0206) adversary_sim=0.0120 (var=0.0134) welch_t=+0.475 p=0.6374 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Create a hashable string key for each trial type based on the options\n    data['trial_key'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x])) + '_' + data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # We analyze the rate of choosing Option A\n    data['chose_A'] = (data['response'] == 0).astype(float)\n    \n    def subject_excess_variance(sub_df):\n        # Calculate mean choice of A for each trial type\n        trial_stats = sub_df.groupby('trial_key')['chose_A'].agg(['mean', 'count'])\n        trial_stats = trial_stats[trial_stats['count'] > 1]\n        \n        if len(trial_stats) < 2:\n            return 0.0\n            \n        k = len(trial_stats)\n        n_j = trial_stats['count'].values\n        p_hat_j = trial_stats['mean'].values\n        \n        # Overall mean choice rate\n        p_hat = np.sum(n_j * p_hat_j) / np.sum(n_j)\n        \n        # Between-trial Mean Square (MS_B)\n        SS_B = np.sum(n_j * (p_hat_j - p_hat)**2)\n        MS_B = SS_B / (k - 1)\n        \n        # Within-trial Mean Square (MS_W)\n        SS_W = np.sum(n_j * p_hat_j * (1 - p_hat_j))\n        MS_W = SS_W / np.sum(n_j - 1)\n        \n        # MS_B - MS_W is an unbiased estimator of the excess variance in true probabilities\n        return MS_B - MS_W\n\n    if 'subject_id' in data.columns:\n        excess_vars = data.groupby('subject_id').apply(subject_excess_variance)\n        return float(excess_vars.mean())\n    else:\n        return float(subject_excess_variance(data))\n",
  "rationale": "In this specific experimental design, the single highest-validity discriminating feature favors Option A in every single trial. Because standard Take-The-Best (TTB) completely ignores all subordinate features, it predicts that the underlying true probability of choosing Option A is exactly identical across all 12 trial types for any given subject. Therefore, the variance in observed choice rates across trial types should be purely due to binomial sampling noise. By computing the ANOVA Between-Trial Mean Square (MS_B) and Within-Trial Mean Square (MS_W), the difference (MS_B - MS_W) serves as an unbiased estimator of the true variance in underlying choice probabilities. For TTB, this metric mathematically guarantees an expected value of exactly 0.0. In contrast, Stochastic TTB samples among all discriminating features, so the true probability of choosing A varies significantly depending on the strength and number of opposing subordinate cues. Thus, Stochastic TTB predicts a strictly positive excess variance."
}
```

## Usage

```json
{
  "prompt_token_count": 8724,
  "candidates_token_count": 778,
  "total_token_count": 13865
}
```
