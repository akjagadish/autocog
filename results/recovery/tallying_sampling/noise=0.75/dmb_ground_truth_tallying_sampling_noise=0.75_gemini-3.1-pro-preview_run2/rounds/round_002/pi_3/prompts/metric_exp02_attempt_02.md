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

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate Tallying from Probabilistic Search Take-The-Best (PS-TTB), we employ a 5-feature design with a steep gradient in cue validities. Tallying simply counts the number of positive features and ignores validities, whereas PS-TTB probabilistically searches through cues weighted by their validities and decides based on the first discriminating cue. The design includes 'conflict' trials where one option has more positive features (favored by Tallying) but the competing option is favored by the highest-validity cue (favored by PS-TTB). We also include 'tie' trials where both options have the same number of positive features (Tallying predicts guessing) but the features favoring one option have much higher validities (PS-TTB strongly favors this option), providing a robust quantitative dissociation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying (Equal Weighting) posits that decision-makers simply count the total number of positive features for each option, ignoring the differential validities of the cues. The option with the higher count of positive features is preferred. This represents a compensatory but highly frugal heuristic, where evidence is accumulated equally across all available cues. If the counts are equal, the decision-maker guesses. Response noise is modeled via a softmax over these counts with an independent lapse rate. To account for empirical choices that often deviate from pure tallying on conflict trials, the decision process incorporates substantial choice noise.

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
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
[0] rationale: This metric isolates 'conflict trials' where the option favored by the highest-validity cue (cue 0) is different from the option with the highest total number of positive features. Tallying strictly predicts choices based on the total feature count, while Probabilistic Search Take-The-Best (PS-TTB) heavily favors the option with the highest-validity cue. By calculating the proportion of choices that align with the Tallying prediction on these specific conflict trials, we obtain a metric that will be close to 1 for Tallying and close to 0 for PS-TTB, yielding a maximal and highly reliable contrast between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_ratings.sum(axis=1)
    sum_B = B_ratings.sum(axis=1)
    
    A_cue0 = A_ratings[:, 0]
    B_cue0 = B_ratings[:, 0]
    
    conflict_mask = ((sum_A > sum_B) & (A_cue0 == 0) & (B_cue0 == 1)) | ((sum_B > sum_A) & (B_cue0 == 0) & (A_cue0 == 1))
                    
    if not np.any(conflict_mask):
        return 0.5
        
    conflict_data = data[conflict_mask]
    sum_A_c = sum_A[conflict_mask]
    sum_B_c = sum_B[conflict_mask]
    responses = conflict_data['response'].values
    
    tallying_choices = np.where(sum_A_c > sum_B_c, 0, 1)
    
    return float(np.mean(responses == tallying_choices))
outcome: self_sim=0.6442 (var=0.0169) adversary_sim=0.5944 (var=0.0060) welch_t=+1.648 p=0.1073 (N=25, alpha=0.01) -> reject

[1] rationale: This metric contrasts the probability of choosing Option A on two structurally opposed conflict trials: T5 and T3. On T5, Tallying predicts choosing A (A has more total features), while PS-TTB predicts choosing B (the highest validity cue favors B). On T3, the situation is perfectly reversed: Tallying predicts choosing B, and PS-TTB predicts choosing A. By calculating the difference in the proportion of A choices between T5 and T3, Tallying yields a positive value, whereas PS-TTB yields a strongly negative value. This double-dissociation provides a massive and reliable gap between the two theories, effectively bypassing confounding effects of varying noise parameters.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_ratings.sum(axis=1)
    sum_B = B_ratings.sum(axis=1)
    
    # T5: A has 3 positive features, B has 2, but Cue 0 favors B (A[0]==0)
    mask_T5 = (sum_A == 3) & (sum_B == 2) & (A_ratings[:, 0] == 0)
    
    # T3: A has 2 positive features, B has 3, but Cue 0 favors A (A[0]==1)
    mask_T3 = (sum_A == 2) & (sum_B == 3) & (A_ratings[:, 0] == 1)
    
    # Calculate proportion of times Option A was chosen (response == 0)
    p_A_T5 = np.mean(data['response'].values[mask_T5] == 0) if np.any(mask_T5) else 0.5
    p_A_T3 = np.mean(data['response'].values[mask_T3] == 0) if np.any(mask_T3) else 0.5
    
    return float(p_A_T5 - p_A_T3)
outcome: self_sim=0.2263 (var=0.0627) adversary_sim=0.1105 (var=0.0214) welch_t=+1.996 p=0.05304 (N=25, alpha=0.01) -> reject

[2] rationale: This metric combines three structurally critical trials to maximize the contrast between Tallying and PS-TTB. On Trial 2, Tallying predicts perfect guessing (0.5) while PS-TTB predicts choosing A. On Trial 3, Tallying favors B while PS-TTB favors A. On Trial 5, Tallying favors A while PS-TTB favors B. By calculating P(A|T2) + P(A|T3) - P(A|T5), Tallying is expected to yield a low value (below 0.5) because P(A|T3) is low and P(A|T5) is high. Conversely, PS-TTB will yield a very high value (well above 1.0) because P(A|T2) and P(A|T3) are high while P(A|T5) is low. This composite metric amplifies the directional predictions of both models into a massive mean gap, ensuring high discriminability even in the presence of between-subject noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_ratings.sum(axis=1)
    sum_B = B_ratings.sum(axis=1)
    
    # Trial 2: A has 2, B has 2. Tallying predicts 0.5; PS-TTB strongly favors A (Cue 0 favors A)
    mask_T2 = (sum_A == 2) & (sum_B == 2)
    
    # Trial 3: A has 2, B has 3. Tallying favors B; PS-TTB strongly favors A (Cue 0 favors A)
    mask_T3 = (sum_A == 2) & (sum_B == 3)
    
    # Trial 5: A has 3, B has 2, A[0]==0. Tallying favors A; PS-TTB strongly favors B (Cue 0 favors B)
    mask_T5 = (sum_A == 3) & (sum_B == 2) & (A_ratings[:, 0] == 0)
    
    p_A_T2 = np.mean(data['response'].values[mask_T2] == 0) if np.any(mask_T2) else 0.5
    p_A_T3 = np.mean(data['response'].values[mask_T3] == 0) if np.any(mask_T3) else 0.5
    p_A_T5 = np.mean(data['response'].values[mask_T5] == 0) if np.any(mask_T5) else 0.5
    
    return float(p_A_T2 + p_A_T3 - p_A_T5)
outcome: self_sim=0.2632 (var=0.0721) adversary_sim=0.4032 (var=0.0339) welch_t=-2.150 p=0.03729 (N=25, alpha=0.01) -> reject

[3] rationale: This metric contrasts the probability of choosing Option A on Trial 4 versus Trial 5. On both trials, Option A has exactly 3 positive features and Option B has exactly 2. Therefore, Tallying predicts identical choice probabilities for Option A on both trials, yielding an expected difference of exactly 0. In contrast, PS-TTB is sensitive to cue validities: on Trial 4, the highest-validity cue (Cue 0) favors A, while on Trial 5, Cue 0 favors B. Thus, PS-TTB predicts a higher probability of choosing A on Trial 4 than on Trial 5, resulting in a positive difference. This isolates the core structural difference between the models while perfectly controlling for the tally count.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    
    # Trial 4: A=[1, 1, 1, 0, 0]
    mask_T4 = (A_ratings[:, 0] == 1) & (A_ratings[:, 1] == 1) & (A_ratings[:, 2] == 1) & (A_ratings[:, 3] == 0) & (A_ratings[:, 4] == 0)
    
    # Trial 5: A=[0, 1, 1, 1, 0]
    mask_T5 = (A_ratings[:, 0] == 0) & (A_ratings[:, 1] == 1) & (A_ratings[:, 2] == 1) & (A_ratings[:, 3] == 1) & (A_ratings[:, 4] == 0)
    
    # Calculate proportion of times Option A was chosen (response == 0)
    p_A_T4 = np.mean(data['response'].values[mask_T4] == 0) if np.any(mask_T4) else 0.5
    p_A_T5 = np.mean(data['response'].values[mask_T5] == 0) if np.any(mask_T5) else 0.5
    
    return float(p_A_T4 - p_A_T5)
outcome: self_sim=0.0389 (var=0.0205) adversary_sim=0.0211 (var=0.0235) welch_t=+0.427 p=0.6715 (N=25, alpha=0.01) -> reject

[4] rationale: This metric exploits the core structural invariants of Tallying. Tallying predicts exactly 0.5 when the sum of features is equal (Trials 2 and 6), and identical choice probabilities when the difference in sums is the same (Trials 1, 3, and 4 all have sum_A - sum_B == 1). By calculating (P(A|T6) - P(A|T2)) + (P(A|T4) - P(A|T1,T3)), Tallying yields an expected value of exactly 0.0, with variance solely due to finite-trial binomial noise. In contrast, PS-TTB is highly sensitive to the highest-validity cue (Cue 0), which favors Option A on T6 and T4, but Option B on T2, T1, and T3. Thus, PS-TTB will yield a strictly positive value. Averaging across multiple trials significantly reduces the within-subject variance, ensuring a highly reliable divergence between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_ratings.sum(axis=1)
    sum_B = B_ratings.sum(axis=1)
    
    # Trials where Tallying predicts guessing (sum_A == sum_B)
    # T6: Cue 0 favors A
    mask_eq_A = (sum_A == sum_B) & (A_ratings[:, 0] == 1)
    # T2: Cue 0 favors B
    mask_eq_B = (sum_A == sum_B) & (A_ratings[:, 0] == 0)
    
    # Trials where Tallying predicts A with identical probability (sum_A - sum_B == 1)
    # T4: Cue 0 favors A
    mask_diff1_A = (sum_A - sum_B == 1) & (A_ratings[:, 0] == 1)
    # T1, T3: Cue 0 favors B
    mask_diff1_B = (sum_A - sum_B == 1) & (A_ratings[:, 0] == 0)
    
    p_eq_A = np.mean(data['response'].values[mask_eq_A] == 0) if np.any(mask_eq_A) else 0.5
    p_eq_B = np.mean(data['response'].values[mask_eq_B] == 0) if np.any(mask_eq_B) else 0.5
    
    p_diff1_A = np.mean(data['response'].values[mask_diff1_A] == 0) if np.any(mask_diff1_A) else 0.5
    p_diff1_B = np.mean(data['response'].values[mask_diff1_B] == 0) if np.any(mask_diff1_B) else 0.5
    
    return float((p_eq_A - p_eq_B) + (p_diff1_A - p_diff1_B))
outcome: self_sim=-0.0342 (var=0.0612) adversary_sim=-0.0175 (var=0.1002) welch_t=-0.207 p=0.8366 (N=25, alpha=0.01) -> reject

[5] rationale: This metric evaluates the impact of the highest-validity cue (Cue 0) on choice behavior by calculating the difference in the probability of choosing Option A when Cue 0 favors A versus when it favors B. Under the chosen experimental design, the trials where Cue 0 favors A (T4, T5, T6, T8) actually have a negative average tally difference (meaning Tallying slightly favors B on average), whereas the trials where Cue 0 favors B (T1, T2, T3, T7) have a strongly positive average tally difference (meaning Tallying strongly favors A). Consequently, Tallying predicts a negative value for this metric. In stark contrast, Probabilistic Search Take-The-Best (PS-TTB) relies heavily on the highest-validity cue, predicting a very high probability of choosing A when Cue 0 favors A and a very low probability when Cue 0 favors B, leading to a strongly positive metric. Because it averages over half the trials in each condition, within-subject variance is minimized, resulting in an exceptionally robust discriminative signal.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    cue0_A = A_ratings[:, 0]
    cue0_B = B_ratings[:, 0]
    
    # Trials where the highest validity cue (Cue 0) favors Option A
    mask_A = (cue0_A == 1) & (cue0_B == 0)
    
    # Trials where the highest validity cue (Cue 0) favors Option B
    mask_B = (cue0_A == 0) & (cue0_B == 1)
    
    # Proportion of times Option A was chosen in each subset
    p_A_given_cue0_A = np.mean(data['response'].values[mask_A] == 0) if np.any(mask_A) else 0.5
    p_A_given_cue0_B = np.mean(data['response'].values[mask_B] == 0) if np.any(mask_B) else 0.5
    
    return float(p_A_given_cue0_A - p_A_given_cue0_B)
outcome: self_sim=-0.1396 (var=0.0171) adversary_sim=-0.0788 (var=0.0110) welch_t=-1.812 p=0.0765 (N=25, alpha=0.01) -> reject

[6] rationale: This metric contrasts the probability of choosing Option A on two sets of perfectly opposed conflict trials. On Trials 1 and 3, Option A has more positive features (favored by Tallying) but the highest-validity cue favors Option B (favored by PS-TTB). On Trials 5 and 8, Option B has more positive features (favored by Tallying) but the highest-validity cue favors Option A (favored by PS-TTB). By calculating the difference P(A | T1, T3) - P(A | T5, T8), Tallying expects a strongly positive value, whereas PS-TTB expects a strongly negative value. Grouping two trials per condition reduces within-subject variance, creating a massive and highly reliable gap between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_ratings.sum(axis=1)
    sum_B = B_ratings.sum(axis=1)
    
    # Trials 1 and 3: Tallying favors A (sum_A=3, sum_B=2), but PS-TTB favors B (Cue 0 favors B)
    mask_T1_T3 = (sum_A == 3) & (sum_B == 2) & (A_ratings[:, 0] == 0)
    
    # Trials 5 and 8: Tallying favors B (sum_A < sum_B), but PS-TTB favors A (Cue 0 favors A)
    mask_T5_T8 = ((sum_A == 1) & (sum_B == 2) & (A_ratings[:, 0] == 1)) | \
                 ((sum_A == 2) & (sum_B == 3) & (A_ratings[:, 0] == 1))
                 
    p_A_13 = np.mean(data['response'].values[mask_T1_T3] == 0) if np.any(mask_T1_T3) else 0.5
    p_A_58 = np.mean(data['response'].values[mask_T5_T8] == 0) if np.any(mask_T5_T8) else 0.5
    
    return float(p_A_13 - p_A_58)
outcome: self_sim=0.2250 (var=0.0510) adversary_sim=0.0875 (var=0.0195) welch_t=+2.589 p=0.01335 (N=25, alpha=0.01) -> reject

[7] rationale: Previous attempts failed because they assumed fixed A/B presentation order, which washed out the metric into random noise when options were swapped. By making the trial identification invariant to order (using absolute differences) and defining the response relative to the highest-validity cue (Cue 0), we perfectly isolate the theoretical difference. On Tie trials (T2 and T6), Tallying always predicts exactly 0.5 because the feature counts are identical. In contrast, PS-TTB relies heavily on Cue 0, predicting a probability strictly greater than 0.5 for the Cue 0 option. Summing these probabilities yields exactly 1.0 for Tallying (with zero variance due to beta), while yielding a significantly higher value for PS-TTB, creating a massive and highly reliable gap.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    # The options A and B can be presented in either order. 
    # We identify the trial types by their absolute differences, which are invariant to order.
    diff = np.abs(A_ratings - B_ratings)
    
    # T2: Diff is [1, 1, 0, 1, 1]
    mask_T2 = (diff[:, 0] == 1) & (diff[:, 1] == 1) & (diff[:, 2] == 0) & (diff[:, 3] == 1) & (diff[:, 4] == 1)
    
    # T6: Diff is [1, 1, 1, 1, 0]
    mask_T6 = (diff[:, 0] == 1) & (diff[:, 1] == 1) & (diff[:, 2] == 1) & (diff[:, 3] == 1) & (diff[:, 4] == 0)
    
    # To make the metric directional, we redefine the 'response' 
    # as 1 if the subject chose the option with Cue 0 == 1, and 0 otherwise.
    # This perfectly aligns with the Take-The-Best prediction.
    chose_cue0 = np.where(
        (A_ratings[:, 0] == 1) & (data['response'].values == 0) | \
        (B_ratings[:, 0] == 1) & (data['response'].values == 1), 
        1.0, 0.0
    )
    
    p_cue0_T6 = np.mean(chose_cue0[mask_T6]) if np.any(mask_T6) else 0.5
    p_cue0_T2 = np.mean(chose_cue0[mask_T2]) if np.any(mask_T2) else 0.5
    
    # For both T2 and T6, Tallying predicts exactly 0.5 (tie trials) regardless of beta.
    # PS-TTB predicts picking the Cue 0 option, so p_cue0 will be > 0.5.
    return float(p_cue0_T6 + p_cue0_T2)
outcome: self_sim=0.9883 (var=0.0508) adversary_sim=0.9950 (var=0.0523) welch_t=-0.104 p=0.9178 (N=25, alpha=0.01) -> reject

[8] rationale: This metric perfectly isolates the theoretical divergence by controlling for the total feature count difference. By comparing trials where the feature sum is exactly 3 vs 2, Tallying predicts the exact same probability of choosing the 3-feature option regardless of which specific cues are active, yielding an expected difference of exactly 0. In contrast, PS-TTB is highly sensitive to the highest-validity cue (Cue 0). When Cue 0 favors the 3-feature option (Condition 1), PS-TTB predicts a much higher probability of choosing it than when Cue 0 favors the 2-feature option (Condition 2). Thus, PS-TTB yields a strictly positive value. Because Tallying's expectation is exactly 0 regardless of beta and epsilon, this metric provides a highly reliable and unconfounded contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_ratings.sum(axis=1)
    sum_B = B_ratings.sum(axis=1)
    
    # Identify trials where one option has exactly 3 positive features and the other has 2
    is_3_2 = (sum_A == 3) & (sum_B == 2)
    is_2_3 = (sum_A == 2) & (sum_B == 3)
    
    # Condition 1: The option with 3 features is favored by the highest-validity cue (Cue 0)
    cond1_A = is_3_2 & (A_ratings[:, 0] == 1)
    cond1_B = is_2_3 & (B_ratings[:, 0] == 1)
    cond1_mask = cond1_A | cond1_B
    
    # Condition 2: The option with 3 features is NOT favored by the highest-validity cue (Cue 0)
    cond2_A = is_3_2 & (A_ratings[:, 0] == 0)
    cond2_B = is_2_3 & (B_ratings[:, 0] == 0)
    cond2_mask = cond2_A | cond2_B
    
    # Probability of choosing the 3-feature option in Condition 1
    chose_3_cond1 = np.where(cond1_A, data['response'].values == 0, data['response'].values == 1)
    p1 = np.mean(chose_3_cond1[cond1_mask]) if np.any(cond1_mask) else 0.5
    
    # Probability of choosing the 3-feature option in Condition 2
    chose_3_cond2 = np.where(cond2_A, data['response'].values == 0, data['response'].values == 1)
    p2 = np.mean(chose_3_cond2[cond2_mask]) if np.any(cond2_mask) else 0.5
    
    # Return the difference in probability of picking the 3-feature option
    return float(p1 - p2)
outcome: self_sim=-0.0289 (var=0.0273) adversary_sim=-0.0133 (var=0.0296) welch_t=-0.326 p=0.7458 (N=25, alpha=0.01) -> reject

[9] rationale: This metric exploits a strict mathematical property of the Softmax choice rule used by Tallying. For Tallying, the probability of choosing the higher-sum option on diff=2 trials (p2) is strictly bounded by a concave function of the probability on diff=1 trials (p1), specifically f(p1) = p1^2 / (p1^2 + (1-p1)^2). Because of the lapse rate (epsilon), the observed probabilities are compressed towards 0.5. Since f(p1) is strictly concave for p1 > 0.5, Jensen's inequality guarantees that f(p1_observed) will always be greater than or equal to p2_observed for Tallying, yielding a positive value. In contrast, PS-TTB operates via proportional cue sampling (large tau) and Take-The-Best (small tau). Both regimes of PS-TTB violate this concave relationship heavily, predicting a much lower p2 relative to f(p1), which drives the metric to be negative or strictly lower, providing a robust, parameter-free dissociation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_ratings.sum(axis=1)
    sum_B = B_ratings.sum(axis=1)
    
    diff = np.abs(sum_A - sum_B)
    
    # Identify trials where the sum difference is exactly 1
    mask_diff1 = (diff == 1)
    # Identify trials where the sum difference is exactly 2
    mask_diff2 = (diff == 2)
    
    responses = data['response'].values
    
    # 1 if subject chose the option with the higher sum of positive features, 0 otherwise
    chose_higher = np.where(sum_A > sum_B, responses == 0, responses == 1)
    
    p1 = np.mean(chose_higher[mask_diff1]) if np.any(mask_diff1) else 0.5
    p2 = np.mean(chose_higher[mask_diff2]) if np.any(mask_diff2) else 0.5
    
    # The theoretical Softmax relationship between diff=1 and diff=2 probabilities
    f_p1 = (p1**2) / (p1**2 + (1.0 - p1)**2 + 1e-6)
    
    # We return the deviation from the Softmax expectation.
    return float(f_p1 - p2)
outcome: self_sim=0.0292 (var=0.0340) adversary_sim=-0.0468 (var=0.0260) welch_t=+1.552 p=0.1273 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    A_ratings = np.stack(data['option_a_ratings'].values)\n    B_ratings = np.stack(data['option_b_ratings'].values)\n    \n    sum_A = A_ratings.sum(axis=1)\n    sum_B = B_ratings.sum(axis=1)\n    \n    # Isolate \"Tie\" trials where the total feature counts are identical,\n    # but the highest-validity cue (Cue 0) discriminates between the options.\n    # In this design, this perfectly isolates Trials 2 and 4.\n    mask = (sum_A == sum_B) & (A_ratings[:, 0] != B_ratings[:, 0])\n    \n    if not np.any(mask):\n        return 0.5\n        \n    data_masked = data[mask]\n    A_masked = A_ratings[mask]\n    responses = data_masked['response'].values\n    \n    # Calculate the proportion of times the subject chose the option favored by Cue 0\n    chose_cue0 = np.where(A_masked[:, 0] == 1, responses == 0, responses == 1)\n    \n    return float(np.mean(chose_cue0))",
  "rationale": "This metric isolates 'Tie' trials (Trials 2 and 4) where both options have the exact same number of positive features, but the highest-validity cue (Cue 0) favors one over the other. By structural definition, Tallying predicts perfect guessing (0.5) on these trials regardless of its noise parameters (beta and epsilon), yielding a theoretical expectation of exactly 0.5 with minimal variance. In stark contrast, Probabilistic Search Take-The-Best (PS-TTB) is highly sensitive to the validities. Across all its parameter regimes (small tau mimicking strict Take-The-Best, or large tau mimicking proportional weighting), PS-TTB strongly favors the option with Cue 0. This guarantees a strictly positive deviation from 0.5 for PS-TTB, providing an unconfounded, parameter-free dissociation."
}
```

## Usage

```json
{
  "prompt_token_count": 10201,
  "candidates_token_count": 511,
  "total_token_count": 19001
}
```
