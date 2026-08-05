# metric_exp02_attempt_03

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
**Validities (n_features=10):** [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** To quantitatively and qualitatively dissociate Continuous Weighted Parity from Cue-Parity, we exploit the fact that Continuous Weighted Parity modulates the parity signal based on the validity of the winning features, while Cue-Parity relies strictly on the unweighted integer count of wins. By using 10 features (5 high validity, 5 low validity), we can design trials where option A wins on 5 high-validity or 5 low-validity features. For Cue-Parity, 5 wins is always odd, leading to a deterministic preference for A. However, for Continuous Weighted Parity, the validity weighting shifts the effective sum of wins. For example, 5 low-validity wins can produce a weighted sum of ~4.375, and 5 high-validity wins can produce ~5.625. Both of these weighted sums cross a half-integer boundary (4.5 and 5.5, respectively), causing the continuous cosine parity function to undergo a phase shift and actually predict option B, creating a strict qualitative reversal. Additionally, trials with exactly 4 high or 4 low validity wins push the weighted sum exactly to half-integers (e.g., 4.5 or 3.5), where Continuous Weighted Parity predicts complete 50/50 indifference, whereas Cue-Parity deterministically predicts B (since 4 is even).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Continuous Weighted Parity (Restricted Parameter): Decision makers compute a parity-like signal, but the contribution of each cue is weighted by its validity. The evidence is integrated into a continuous sum of validity-weighted wins, which is then passed through a periodic parity function (negative cosine). To avoid chaotic phase shifts that destroy the parity signal, the influence of validity weights is kept small, allowing them to gently perturb the parity score without pushing the sum into the 'dead zones' (half-integers) of the cosine function.

**Parameters:**
- validities: validities
- w_validity: [0.0, 0.5]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Continuous Weighted Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    w_validity = float(parameters["w_validity"])
    
    # Calculate weights for each cue based on validity
    mean_val = np.mean(validities)
    weights = 1.0 + w_validity * (validities - mean_val)
    
    # Compute weighted sum of strict A wins
    a_wins_mask = (a > b).astype(float)
    weighted_wins = np.sum(weights * a_wins_mask)
    
    # Continuous parity score: maps even integers to -1 (favoring B) and odd integers to +1 (favoring A)
    parity_score = -np.cos(np.pi * weighted_wins)
    
    # Scores for A and B
    score_a = parity_score
    score_b = -parity_score
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax conversion with numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People decide by the PARITY of the evidence rather than its weight or direction. They count the number of cues on which option A strictly beats option B, and prefer A when that count is ODD and B when it is EVEN (zero counts as even). This is a deliberately non-monotone, XOR-like rule: adding one more cue in A's favor flips the preference rather than strengthening it, so option dominance does NOT imply choice — an option that wins on every cue (an even count, when the cue number is even) is rejected. The rule uses no validities and no magnitudes, only the parity of feature-wise wins, which makes it an adversarially hard recovery target: it is uncorrelated with any single cue and with the validity-weighted sum, yet perfectly deterministic, much like the anti-majority ensemble. Response noise enters through a softmax over the binary score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Cue-Parity: prefer A iff the number of cues where A strictly beats B is
    # ODD, else prefer B (zero A-wins is even -> B). Non-monotone in the
    # evidence; uses neither validities nor magnitudes. Stimulus is the pair
    # of option feature vectors: (2, n_features), row 0 = A, row 1 = B.
    # History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Cue-Parity expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    a_wins = int(np.sum(a > b))
    # Odd number of A-favoring cues -> A (0); even (including zero) -> B (1).
    winner = 0 if (a_wins % 2 == 1) else 1

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # score this collapses to sigmoid(beta) for the winner (beta=0 ↔ 50/50;
    # beta ≫ 1 ↔ deterministic).
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
[0] rationale: Continuous Weighted Parity predicts that the choice probabilities for A will be systematically attenuated based on the distance of the weighted sum to the nearest odd integer. For trials in Group 1, the deviation is 0.25 * w_validity, leading to a mild attenuation. For trials in Group 2, the deviation is 0.75 * w_validity, pushing the sum much closer to the 'dead zones' (half-integers) and causing severe attenuation. Thus, CWP predicts P(A | Group 1) > P(A | Group 2). Cue-Parity, however, only cares about the unweighted count of A-wins. Since both groups only contain trials with an odd number of A-wins (1 or 3), Cue-Parity produces identical, deterministic max-score differences for both groups, predicting a difference of roughly 0.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # response == 0 means A was chosen
    chose_a = (data['response'] == 0).astype(float)
    
    # Extract A and B ratings as 2D arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Calculate strict A wins and high-validity A wins (first 3 features)
    a_wins_mask = (a_ratings > b_ratings).astype(int)
    a_wins = a_wins_mask.sum(axis=1)
    high_wins = a_wins_mask[:, :3].sum(axis=1)
    
    # Group 1: Trials where CWP score is slightly attenuated (magnitude = cos(0.25*pi*w))
    # These are trials with 1 win total OR 3 wins with mixed validities.
    g1_mask = (a_wins == 1) | ((a_wins == 3) & ((high_wins == 1) | (high_wins == 2)))
    
    # Group 2: Trials where CWP score is heavily attenuated (magnitude = cos(0.75*pi*w))
    # These are trials with 3 wins all of the same validity type.
    g2_mask = (a_wins == 3) & ((high_wins == 0) | (high_wins == 3))
    
    if not np.any(g1_mask) or not np.any(g2_mask):
        return 0.0
        
    p_a_g1 = chose_a[g1_mask].mean()
    p_a_g2 = chose_a[g2_mask].mean()
    
    return float(p_a_g1 - p_a_g2)
outcome: self_sim=-0.0212 (var=0.0044) adversary_sim=0.0104 (var=0.0051) welch_t=-1.625 p=0.1107 (N=25, alpha=0.01) -> reject

[1] rationale: CWP predicts that choice probabilities for A will be attenuated on trials where the weighted sum of wins is closer to a half-integer. For Group 1 (trials 1, 2, 7, 8), the distance to the nearest half-integer is large, leading to strong preferences. For Group 2 (trials 5, 6), the distance is smaller, leading to attenuated preferences and thus a higher error rate. Cue-Parity, however, predicts identical error rates for both groups because it relies purely on the unweighted integer count of wins, which is odd for both groups. By computing the log odds ratio of errors between Group 2 and Group 1 with dynamically scaled Laplace smoothing, we obtain a metric that is mathematically expected to be exactly 0 for Cue-Parity but strictly positive for CWP. The log odds transformation amplifies the signal even when absolute error rates are small (e.g., due to large beta), ensuring robust discrimination.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins_mask = (a_ratings > b_ratings).astype(int)
    a_wins = a_wins_mask.sum(axis=1)
    high_wins = a_wins_mask[:, :3].sum(axis=1)
    
    # Group 1: Trials 1, 2, 7, 8 (a_wins=1 or (a_wins=3 and high_wins in [1,2]))
    g1_mask = (a_wins == 1) | ((a_wins == 3) & ((high_wins == 1) | (high_wins == 2)))
    
    # Group 2: Trials 5, 6 (a_wins=3 and high_wins in [0,3])
    g2_mask = (a_wins == 3) & ((high_wins == 0) | (high_wins == 3))
    
    # Expected winner for all trials in g1 and g2 is Option A (response == 0)
    # Error is choosing Option B (response == 1)
    errors = (data['response'] == 1).values.astype(float)
    
    err_g1 = errors[g1_mask].sum()
    N_g1 = g1_mask.sum()
    
    err_g2 = errors[g2_mask].sum()
    N_g2 = g2_mask.sum()
    
    if N_g1 == 0 or N_g2 == 0:
        return 0.0
        
    # Dynamically scaled Laplace smoothing ensures exactly 0 bias when empirical error rates are equal
    alpha_g2 = 0.5
    alpha_g1 = 0.5 * (N_g1 / N_g2)
    
    log_odds_g1 = np.log((err_g1 + alpha_g1) / (N_g1 - err_g1 + alpha_g1))
    log_odds_g2 = np.log((err_g2 + alpha_g2) / (N_g2 - err_g2 + alpha_g2))
    
    return float(log_odds_g2 - log_odds_g1)
outcome: self_sim=-0.1840 (var=0.3711) adversary_sim=0.0925 (var=0.3637) welch_t=-1.613 p=0.1134 (N=25, alpha=0.01) -> reject

[2] rationale: Continuous Weighted Parity produces parity scores whose absolute magnitude systematically exceeds the binary 1/0 scores generated by Cue-Parity. Specifically, CWP's score differences between options average around 1.85 across these specific trial types, compared to exactly 1.0 for Cue-Parity. This larger score differential means that, for any given inverse temperature (beta) drawn from the shared parameter distribution, CWP predicts a more deterministic choice process. Therefore, CWP predicts a systematically higher overall accuracy (probability of choosing the parity-favored option) than Cue-Parity. By measuring the overall empirical accuracy across all trials, we robustly discriminate the models: CWP will exhibit a higher mean accuracy than Cue-Parity, and using all trials minimizes the binomial noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    b_wins = (b_ratings > a_ratings).sum(axis=1)
    
    # Identify the target (the option with more 1s)
    target_is_a = a_wins > b_wins
    target_ratings = np.where(target_is_a[:, None], a_ratings, b_ratings)
    total_wins = target_ratings.sum(axis=1)
    
    # The favored option is the target if total wins is odd, else it is the all-zeros option
    favored_is_target = (total_wins % 2 == 1)
    
    # Determine if the subject chose the target
    chose_target = np.where(target_is_a, data['response'] == 0, data['response'] == 1)
    
    # Determine if the subject chose the favored option
    chose_favored = (chose_target == favored_is_target).astype(float)
    
    return float(chose_favored.mean())
outcome: self_sim=0.8650 (var=0.0073) adversary_sim=0.8735 (var=0.0092) welch_t=-0.332 p=0.7413 (N=25, alpha=0.01) -> reject

[3] rationale: To bypass the confounding effects of potential feature-order randomization (which breaks metrics relying on specific validity indices), we use a purely structural contrast: the total number of wins. For Cue-Parity, the absolute difference in evidence scores is always exactly 1.0 regardless of the number of wins, predicting identical accuracy across 1-win and 2-win trials. For Continuous Weighted Parity (CWP), the continuous parity score is derived from a negative cosine wave. In 1-win trials, the weighted sum is close to 1.0 (deviation ~0.25w), keeping the score near the peak magnitude. In 2-win trials, the weighted sum is close to 2.0 but with a larger deviation (~0.5w), pushing it closer to the 'dead zones' (half-integers) and systematically attenuating the score magnitude. Consequently, CWP predicts a structurally higher choice accuracy on 1-win trials than on 2-win trials. By computing Acc(1-win) - Acc(2-win), we obtain a metric that is mathematically expected to be 0 for Cue-Parity but strictly positive for CWP, entirely robust to feature shuffling.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_ones = a_ratings.sum(axis=1)
    b_ones = b_ratings.sum(axis=1)
    total_ones = a_ones + b_ones
    
    # Mask for trials with exactly 1 win or 2 wins total
    mask_1 = (total_ones == 1)
    mask_2 = (total_ones == 2)
    
    if not np.any(mask_1) or not np.any(mask_2):
        return 0.0
        
    # For 1-win trials (odd), the correct choice is the non-zero option.
    # If A has the 1 win, correct is 0 (A). If B has it, correct is 1 (B).
    correct_1 = np.where(a_ones[mask_1] == 1, 0, 1)
    acc_1 = (data['response'].values[mask_1] == correct_1).mean()
    
    # For 2-win trials (even), the correct choice is the ALL-ZEROS option.
    # If A has the 2 wins, A is non-zero, so correct is 1 (B).
    # If B has the 2 wins, B is non-zero, so correct is 0 (A).
    correct_2 = np.where(a_ones[mask_2] == 2, 1, 0)
    acc_2 = (data['response'].values[mask_2] == correct_2).mean()
    
    # We return the difference in accuracy between 1-win and 2-win trials.
    return float(acc_1 - acc_2)
outcome: self_sim=-0.0125 (var=0.0099) adversary_sim=0.0067 (var=0.0075) welch_t=-0.726 p=0.4712 (N=25, alpha=0.01) -> reject

[4] rationale: This metric leverages the exact phase shift described in the experimental rationale. For trials where A wins on exactly 5 cues (Trials 1 and 2), the validity-weighted sum in Continuous Weighted Parity (CWP) is pushed so far from 5 that it frequently crosses the half-integer boundary (4.5 or 5.5). This causes CWP's parity score to flip or severely attenuate, leading to a reduced preference for Option A. However, for trials where A wins on exactly 3 cues (Trials 5 and 6), the weighted sum remains safely within the bounds of predicting Option A. Thus, CWP predicts P(A | 5 wins) < P(A | 3 wins). In stark contrast, Cue-Parity strictly computes the unweighted integer parity. Since both 5 and 3 are odd numbers, Cue-Parity deterministically predicts Option A with the exact same maximum score for both trial types. By calculating the within-subject difference P(A | 5 wins) - P(A | 3 wins), we obtain a metric that is mathematically expected to be exactly 0 for Cue-Parity (canceling out temperature and lapse rate noise) but significantly negative for CWP.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Count the number of cues where A strictly beats B
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    
    # Isolate trials where A wins on exactly 5 cues and exactly 3 cues
    mask_5 = (a_wins == 5)
    mask_3 = (a_wins == 3)
    
    if not np.any(mask_5) or not np.any(mask_3):
        return 0.0
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_5 = (data['response'].values[mask_5] == 0).astype(float).mean()
    p_a_3 = (data['response'].values[mask_3] == 0).astype(float).mean()
    
    # Return the within-subject difference in preference for A
    return float(p_a_5 - p_a_3)
outcome: self_sim=-0.0842 (var=0.0555) adversary_sim=0.0042 (var=0.0143) welch_t=-1.671 p=0.1034 (N=25, alpha=0.01) -> reject

[5] rationale: By pooling trials into 'Strong' (2 and 3 A-wins) and 'Weak' (4 and 5 A-wins) groups, we achieve a high-powered, low-variance contrast. For Continuous Weighted Parity (CWP), the validity-weighted sum for 4- and 5-win trials frequently approaches or crosses the half-integer boundary, leading to severely attenuated or even flipped parity scores. Conversely, 2- and 3-win trials remain safely within the bounds of their respective integer targets, preserving high score magnitudes. Thus, CWP predicts a significantly higher alignment with the parity rule on Strong trials than on Weak trials. Cue-Parity, however, is entirely blind to validities and magnitudes, producing identical deterministic max-score differences for all trials. Consequently, Cue-Parity predicts an accuracy difference of exactly 0 on average. The pooling of 48 trials per group per subject tightly bounds the variance, ensuring robust statistical discrimination at N=25.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Count the number of cues where A strictly beats B
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    
    # Group trials into 'Strong' and 'Weak' based on CWP's predicted score magnitude
    # Strong trials (2 or 3 wins): CWP weighted sum stays far from half-integers
    mask_strong = (a_wins == 2) | (a_wins == 3)
    # Weak trials (4 or 5 wins): CWP weighted sum is pushed towards or across half-integers
    mask_weak = (a_wins == 4) | (a_wins == 5)
    
    # Cue-Parity predicts Option A (response == 0) if a_wins is odd, else Option B (response == 1)
    is_odd = (a_wins % 2 == 1)
    cp_correct = ((is_odd) & (data['response'] == 0)) | ((~is_odd) & (data['response'] == 1))
    
    if mask_strong.sum() == 0 or mask_weak.sum() == 0:
        return 0.0
        
    acc_strong = cp_correct[mask_strong].mean()
    acc_weak = cp_correct[mask_weak].mean()
    
    # Return the within-subject difference in accuracy
    return float(acc_strong - acc_weak)
outcome: self_sim=0.0504 (var=0.0194) adversary_sim=-0.0088 (var=0.0065) welch_t=+1.837 p=0.07388 (N=25, alpha=0.01) -> reject

[6] rationale: This metric leverages a double difference to strictly isolate the attenuation predicted by Continuous Weighted Parity (CWP). Cue-Parity produces identical binary scores for any odd number of wins (predicting A) and any even number of wins (predicting B). Thus, for Cue-Parity, the difference in preference P(A|3 wins) - P(A|2 wins) is mathematically expected to be identical to P(A|5 wins) - P(A|4 wins), making the double difference zero on average. CWP, however, weights the wins by validity. For 4 and 5 win trials, the cumulative validity weight shifts the sum significantly closer to the half-integer 'dead zones' compared to 2 and 3 win trials. Consequently, CWP predicts a severe attenuation in choice determinism for 4 and 5 win trials, meaning P(A|5) - P(A|4) will be substantially smaller than P(A|3) - P(A|2). The metric will yield a robust positive value for CWP and approximately zero for Cue-Parity.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Count the number of cues where A strictly beats B
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    
    mask_3 = (a_wins == 3)
    mask_2 = (a_wins == 2)
    mask_5 = (a_wins == 5)
    mask_4 = (a_wins == 4)
    
    if not (np.any(mask_3) and np.any(mask_2) and np.any(mask_5) and np.any(mask_4)):
        return 0.0
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_3 = (data['response'].values[mask_3] == 0).astype(float).mean()
    p_a_2 = (data['response'].values[mask_2] == 0).astype(float).mean()
    p_a_5 = (data['response'].values[mask_5] == 0).astype(float).mean()
    p_a_4 = (data['response'].values[mask_4] == 0).astype(float).mean()
    
    # Diff_small: Sensitivity to parity for 2 and 3 win trials (small validity shift)
    diff_small = p_a_3 - p_a_2
    # Diff_large: Sensitivity to parity for 4 and 5 win trials (large validity shift)
    diff_large = p_a_5 - p_a_4
    
    # Return the double difference
    return float(diff_small - diff_large)
outcome: self_sim=0.1008 (var=0.0776) adversary_sim=-0.0175 (var=0.0261) welch_t=+1.837 p=0.07388 (N=25, alpha=0.01) -> reject

[7] rationale: In previous attempts, the raw difference in accuracy between 'Strong' (2 and 3 wins) and 'Weak' (4 and 5 wins) trials yielded a promising positive mean for CWP and zero for Cue-Parity, but failed to reach significance due to high between-subject variance. This variance is driven by the fact that subjects with high noise (low beta, high epsilon) show small absolute accuracy drops, while subjects with low noise show large absolute drops. 

To solve this, we compute the relative drop in accuracy by normalizing the difference (Acc_strong - Acc_weak) by the subject's baseline sensitivity above chance (abs(Acc_strong - 0.5) + 0.05). For Cue-Parity, the expected difference remains exactly 0, and the normalization merely scales the binomial noise. For Continuous Weighted Parity, the metric isolates the proportional attenuation effect, which is structurally invariant to the subject's specific noise parameters. This tightens the distribution of the CWP metric, allowing for robust statistical discrimination even at small sample sizes.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Count the number of cues where A strictly beats B
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    
    # Group trials into 'Strong' and 'Weak' based on CWP's predicted score magnitude
    # Strong trials (2 or 3 wins): CWP weighted sum stays far from half-integers
    mask_strong = (a_wins == 2) | (a_wins == 3)
    # Weak trials (4 or 5 wins): CWP weighted sum is pushed towards or across half-integers
    mask_weak = (a_wins == 4) | (a_wins == 5)
    
    if mask_strong.sum() == 0 or mask_weak.sum() == 0:
        return 0.0
        
    # Cue-Parity predicts Option A (response == 0) if a_wins is odd, else Option B (response == 1)
    is_odd = (a_wins % 2 == 1)
    cp_correct = ((is_odd) & (data['response'] == 0)) | ((~is_odd) & (data['response'] == 1))
    
    acc_strong = cp_correct[mask_strong].mean()
    acc_weak = cp_correct[mask_weak].mean()
    
    # Calculate the unscaled difference in accuracy
    numerator = acc_strong - acc_weak
    
    # Normalize by the subject's baseline sensitivity above chance (with a small constant to prevent blowup)
    # This drastically reduces between-subject variance caused by different beta/epsilon parameters
    denominator = np.abs(acc_strong - 0.5) + 0.05
    
    return float(numerator / denominator)
outcome: self_sim=0.1200 (var=0.1205) adversary_sim=-0.0232 (var=0.0667) welch_t=+1.655 p=0.1049 (N=25, alpha=0.01) -> reject

[8] rationale: By strictly isolating trials where Option A wins on exactly 5 cues, we perfectly control for the unweighted integer parity. Cue-Parity treats all 5-win trials identically (since 5 is odd, it deterministically predicts A with the exact same maximum score) and thus predicts a difference of exactly 0 on average. However, Continuous Weighted Parity (CWP) is highly sensitive to the distribution of validities among those 5 wins. For 'Mixed' trials, the high and low validities cancel each other out, keeping the weighted sum close to the target integer 5 and preserving a strong preference for A. For 'Uniform' trials, the validities compound, pushing the weighted sum significantly away from 5, severely attenuating or even reversing the preference for A. By comparing P(A | 5 wins, mixed) - P(A | 5 wins, uniform), we yield a metric that is mathematically zero for Cue-Parity but significantly positive for CWP, with minimal variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins_mask = (a_ratings > b_ratings).astype(int)
    a_wins = a_wins_mask.sum(axis=1)
    
    # The first 5 features have high validity (1.0), the last 5 have low validity (0.5)
    high_wins = a_wins_mask[:, :5].sum(axis=1)
    
    # 'Mixed' 5-win trials (Trials 5 and 6): A wins on 3 high/2 low or 2 high/3 low cues.
    # The validity-weighted sum remains close to 5 (deviation of +/- 0.25 * w_validity).
    # CWP predicts a strong parity score favoring A.
    mask_mixed = (a_wins == 5) & ((high_wins == 2) | (high_wins == 3))
    
    # 'Uniform' 5-win trials (Trials 1 and 2): A wins on 5 high or 5 low cues.
    # The validity-weighted sum is pushed far from 5 (deviation of +/- 1.25 * w_validity),
    # causing the parity score to severely attenuate or even flip to preferring B.
    mask_uniform = (a_wins == 5) & ((high_wins == 0) | (high_wins == 5))
    
    if mask_mixed.sum() == 0 or mask_uniform.sum() == 0:
        return 0.0
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_mixed = (data['response'].values[mask_mixed] == 0).astype(float).mean()
    p_a_uniform = (data['response'].values[mask_uniform] == 0).astype(float).mean()
    
    # Return the within-subject difference in preference for A
    return float(p_a_mixed - p_a_uniform)
outcome: self_sim=0.0983 (var=0.0659) adversary_sim=0.0025 (var=0.0134) welch_t=+1.702 p=0.09804 (N=25, alpha=0.01) -> reject

[9] rationale: To maximize the signal-to-noise ratio, we partition the entire 96-trial dataset into two equal halves (48 trials each) based exactly on Continuous Weighted Parity's theoretical score magnitudes. 'Strong' trials (Trials 5, 6, 7, 8) have validity-weighted sums that remain close to their target integers, yielding high parity scores (magnitudes ~cos(0.25*pi*w) and ~cos(0.75*pi*w)). 'Weak' trials (Trials 1, 2, 3, 4) have validity-weighted sums that are pushed near or across half-integer boundaries, severely attenuating or flipping the scores (magnitudes ~cos(pi*w) and ~cos(1.25*pi*w)). CWP thus predicts significantly higher choice determinism (accuracy relative to the base parity rule) on Strong trials than on Weak trials. For Cue-Parity, however, all 96 trials represent identical binary parity decisions, predicting an accuracy difference of exactly 0. By utilizing all trials, we cut binomial variance in half compared to earlier subset-based metrics, yielding robust discrimination.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins_mask = (a_ratings > b_ratings).astype(int)
    a_wins = a_wins_mask.sum(axis=1)
    
    # The first 5 features have high validity (1.0), the last 5 have low validity (0.5)
    high_wins = a_wins_mask[:, :5].sum(axis=1)
    
    # Strong trials: CWP scores are close to 1.0 (minimal attenuation)
    # T5, T6: 5 wins, mixed validities (high_wins in {2,3}). Score ~ cos(0.25 * pi * w)
    # T7, T8: 3 wins, uniform validities. Score ~ cos(0.75 * pi * w)
    mask_56 = (a_wins == 5) & ((high_wins == 2) | (high_wins == 3))
    mask_78 = (a_wins == 3)
    mask_strong = mask_56 | mask_78
    
    # Weak trials: CWP scores are heavily attenuated or even flipped
    # T3, T4: 4 wins, uniform validities. Score ~ cos(1.0 * pi * w)
    # T1, T2: 5 wins, uniform validities. Score ~ cos(1.25 * pi * w)
    mask_12 = (a_wins == 5) & ((high_wins == 0) | (high_wins == 5))
    mask_34 = (a_wins == 4)
    mask_weak = mask_12 | mask_34
    
    if mask_strong.sum() == 0 or mask_weak.sum() == 0:
        return 0.0
        
    # Cue-Parity predicts Option A (response == 0) if a_wins is odd, else Option B (response == 1)
    is_odd = (a_wins % 2 == 1)
    cp_correct = ((is_odd) & (data['response'] == 0)) | ((~is_odd) & (data['response'] == 1))
    
    acc_strong = cp_correct[mask_strong].mean()
    acc_weak = cp_correct[mask_weak].mean()
    
    # Return the within-subject difference in accuracy
    return float(acc_strong - acc_weak)
outcome: self_sim=0.0467 (var=0.0186) adversary_sim=0.0000 (var=0.0050) welch_t=+1.522 p=0.1369 (N=25, alpha=0.01) -> reject

[10] rationale: Continuous Weighted Parity (CWP) and Cue-Parity make identical directional predictions for the majority of trials when validity weights are small, but they differ fundamentally in the *magnitude* of their internal scores. Cue-Parity produces identical binary score differences for all trials, meaning the probability of choosing the parity-favored option is theoretically identical across all trial types. In contrast, CWP's continuous negative cosine function systematically attenuates the parity signal depending on how far the validity-weighted sum deviates from the nearest odd integer. By grouping trials into four categories based on CWP's expected score magnitudes (G3: 5 wins mixed > G4: 3 wins uniform > G2: 4 wins uniform > G1: 5 wins uniform), CWP predicts a strictly decreasing sequence in choice determinism. We compute the log-odds of choosing the parity-favored option for each group with Laplace smoothing. The metric applies an optimal linear contrast (3*G3 + G4 - G2 - 3*G1) to these log-odds. For Cue-Parity, the expected contrast is exactly 0, and the variance is purely binomial. For CWP, the contrast is strongly positive. The log-odds transformation prevents ceiling effects from squashing the signal when response noise is low, yielding a highly robust, low-variance discriminator.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Identify the target option (the one with the ones, since the other is all zeros)
    a_wins_mask = (a_ratings > b_ratings).astype(int)
    b_wins_mask = (b_ratings > a_ratings).astype(int)
    a_wins = a_wins_mask.sum(axis=1)
    b_wins = b_wins_mask.sum(axis=1)
    
    target_is_a = a_wins > b_wins
    
    target_wins = np.where(target_is_a, a_wins, b_wins)
    target_high_wins = np.where(
        target_is_a, 
        a_wins_mask[:, :5].sum(axis=1), 
        b_wins_mask[:, :5].sum(axis=1)
    )
    
    # Group trials strictly based on CWP's expected score magnitudes (G3 > G4 > G2 > G1)
    # G1: 5 wins uniform (CWP score is heavily attenuated or flipped)
    mask_g1 = (target_wins == 5) & ((target_high_wins == 5) | (target_high_wins == 0))
    # G2: 4 wins uniform (CWP score is attenuated)
    mask_g2 = (target_wins == 4)
    # G3: 5 wins mixed (CWP score is strongly preserved)
    mask_g3 = (target_wins == 5) & ((target_high_wins == 2) | (target_high_wins == 3))
    # G4: 3 wins uniform (CWP score is moderately preserved)
    mask_g4 = (target_wins == 3)
    
    # Cue-Parity predicts the target option if its wins are odd, else the non-target option
    target_odd = (target_wins % 2 == 1)
    cp_predicts_a = (target_is_a & target_odd) | (~target_is_a & ~target_odd)
    cp_predicts = np.where(cp_predicts_a, 0, 1)
    
    # 'Hits' are trials where the subject chose the Cue-Parity prediction
    hits = (data['response'].values == cp_predicts).astype(int)
    
    # Compute log-odds of a hit for each group with fixed Laplace smoothing
    def log_odds(mask):
        if mask.sum() == 0:
            return 0.0
        h = hits[mask].sum()
        m = mask.sum() - h
        return float(np.log((h + 1.0) / (m + 1.0)))
        
    lo_1 = log_odds(mask_g1)
    lo_2 = log_odds(mask_g2)
    lo_3 = log_odds(mask_g3)
    lo_4 = log_odds(mask_g4)
    
    # Optimal linear contrast for the strictly decreasing sequence predicted by CWP
    return float(3*lo_3 + lo_4 - lo_2 - 3*lo_1)
outcome: self_sim=1.8146 (var=14.8086) adversary_sim=0.0387 (var=6.6668) welch_t=+1.916 p=0.06219 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    a_wins_mask = (a_ratings > b_ratings).astype(int)\n    a_wins = a_wins_mask.sum(axis=1)\n    high_wins = a_wins_mask[:, :5].sum(axis=1)\n    \n    # T1, T2: 5 wins, strictly uniform validities\n    mask_12 = (a_wins == 5) & ((high_wins == 0) | (high_wins == 5))\n    \n    # T3, T4: 4 wins, strictly uniform validities\n    mask_34 = (a_wins == 4) & ((high_wins == 0) | (high_wins == 4))\n    \n    if mask_12.sum() == 0 or mask_34.sum() == 0:\n        return 0.0\n        \n    # h is the number of times Option A was chosen\n    h_12 = (data['response'].values[mask_12] == 0).sum()\n    m_12 = mask_12.sum() - h_12\n    \n    h_34 = (data['response'].values[mask_34] == 0).sum()\n    m_34 = mask_34.sum() - h_34\n    \n    # Compute log-odds of choosing A with Laplace smoothing\n    lo_12 = np.log((h_12 + 0.5) / (m_12 + 0.5))\n    lo_34 = np.log((h_34 + 0.5) / (m_34 + 0.5))\n    \n    # We return the negative sum of these log-odds\n    return float(-(lo_12 + lo_34))",
  "rationale": "This metric leverages a mathematically perfect invariant of Cue-Parity to achieve zero expected bias and extremely low variance. For Cue-Parity, T1/T2 (5 wins) are odd, so it predicts A with probability P. T3/T4 (4 wins) are even, so it predicts B with the exact same probability P, meaning it chooses A with probability 1-P. The sum of the log-odds of choosing A for these two groups is strictly log(P/(1-P)) + log((1-P)/P) = 0, regardless of the subject's temperature or lapse rate. However, for Continuous Weighted Parity, the validity weights shift the sum. For T1/T2, the score is cos(1.25*pi*w), which is severely attenuated or even negative. For T3/T4, the score is -cos(pi*w), which is also attenuated but structurally less so than T1/T2. Because cos(1.25*pi*w) < cos(pi*w) for all w > 0, the push towards A on T1/T2 is strictly weaker than the push towards B on T3/T4. Thus, the sum of log-odds is systematically negative for CWP. By returning the negative sum, we get a metric that is robustly positive for CWP and exactly zero for Cue-Parity."
}
```

## Usage

```json
{
  "prompt_token_count": 12688,
  "candidates_token_count": 810,
  "total_token_count": 18607
}
```
