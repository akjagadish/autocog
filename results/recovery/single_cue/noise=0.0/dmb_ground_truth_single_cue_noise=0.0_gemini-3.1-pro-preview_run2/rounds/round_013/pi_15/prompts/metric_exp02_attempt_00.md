# metric_exp02_attempt_00

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

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Tallying with Probabilistic Least-Valid Cue Dropping) from Pure Tallying, we exploit their differing sensitivity to the single least valid feature in a 5-feature design. Pure Tallying treats all features equally, predicting choices based strictly on the overall tally difference, and predicting a 50/50 guess whenever the tally is tied. The Advocated Theory assumes decision-makers occasionally drop the least valid cue and tally the remainder. We include 'tally tie' trials where the least valid cue favors one option; here, the Advocated Theory predicts a systematic preference for the opposing option (since dropping the least valid cue breaks the tie in its favor), whereas Pure Tallying predicts a strict guess. We also include 'tally win' trials where the least valid cue either supports or opposes the tally winner, leading the Advocated Theory to respectively attenuate or amplify the choice probability compared to Pure Tallying. Control trials where the least valid cue is tied ensure both models make identical baseline predictions.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying with Probabilistic Least-Valid Cue Dropping: To save cognitive effort while maintaining robust integration of information, decision-makers predominantly evaluate all available features (Pure Tallying). However, on a small fraction of evaluations, they boundedly drop the single least valid feature and tally exclusively on the remaining subset. This probabilistic dropping naturally produces subtle Take-The-Best or WADD-like biases in edge cases by occasionally ignoring the lowest-validity cue that might otherwise balance the tally, without destroying the overall pure-tallying majority.

**Parameters:**
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- p_drop: [0.0, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    p_drop = float(parameters["p_drop"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(-val, kind="stable")
    n_features = len(val)
    
    # State 0: Pure Tallying (drop 0 cues)
    a_wins_0 = 0.0
    b_wins_0 = 0.0
    for idx in order:
        if a[idx] > b[idx]:
            a_wins_0 += 1.0
        elif b[idx] > a[idx]:
            b_wins_0 += 1.0
            
    scores_0 = np.array([a_wins_0, b_wins_0])
    z_0 = beta * (scores_0 - np.max(scores_0))
    e_0 = np.exp(z_0)
    p_0 = e_0 / np.sum(e_0)
    
    # State 1: Drop 1 least-valid cue
    a_wins_1 = 0.0
    b_wins_1 = 0.0
    K = max(1, n_features - 1)
    top_cues = order[:K]
    for idx in top_cues:
        if a[idx] > b[idx]:
            a_wins_1 += 1.0
        elif b[idx] > a[idx]:
            b_wins_1 += 1.0
            
    scores_1 = np.array([a_wins_1, b_wins_1])
    z_1 = beta * (scores_1 - np.max(scores_1))
    e_1 = np.exp(z_1)
    p_1 = e_1 / np.sum(e_1)
    
    # Expected choice probabilities (mixture)
    p_core = (1.0 - p_drop) * p_0 + p_drop * p_1
    
    n_opts = len(p_core)
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
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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
[0] rationale: Focuses exclusively on 'tally tie' trials where the least valid cue breaks the tie. Under Pure Tallying, subjects should guess uniformly (50%). Under the Advocated Theory (Probabilistic Least-Valid Cue Dropping), subjects will systematically prefer the option that wins when the least valid cue is dropped, resulting in a choice proportion strictly greater than 50%.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    target_choices = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: A is [1, 1, 0, 0, 0], B is [0, 0, 1, 0, 1]
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 0, 1):
            if resp == 0:
                target_choices += 1
            total += 1
        # Trial 2: A is [0, 0, 1, 0, 1], B is [1, 1, 0, 0, 0]
        elif a == (0, 0, 1, 0, 1) and b == (1, 1, 0, 0, 0):
            if resp == 1:
                target_choices += 1
            total += 1
            
    if total == 0:
        return 0.5
    return target_choices / total
outcome: self_sim=0.5485 (var=0.0124) adversary_sim=0.4885 (var=0.0095) welch_t=+2.030 p=0.04801 (N=25, alpha=0.01) -> reject

[1] rationale: By combining four specific trial types, we create a composite metric with a mathematical constant expectation under Pure Tallying. In Trials 1 and 2, Pure Tallying predicts a 50/50 guess, while the Drop model predicts a preference for the option that wins when the least valid cue is dropped. In Trials 3 and 4, Pure Tallying predicts a 2-1 win for Option A; thus, P(choose B in T3) + P(choose A in T4) exactly equals 1.0 for any beta and epsilon. Summing these four target choice probabilities yields exactly 2.0 under Pure Tallying, making the average 0.5. The Advocated Theory systematically increases the probability of the target choice in all four trials, pushing the expected metric strictly above 0.5.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    score = 0.0
    count = 0

    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']

        # Trial 1: Pure Tally ties. Drop model prefers A.
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 0, 1):
            if resp == 0:
                score += 1
            count += 1
        # Trial 2: Pure Tally ties. Drop model prefers B.
        elif a == (0, 0, 1, 0, 1) and b == (1, 1, 0, 0, 0):
            if resp == 1:
                score += 1
            count += 1
        # Trial 3: Pure Tally prefers A (2-1). Drop model ties (1-1).
        # We score choices for B, which should increase under Drop model.
        elif a == (1, 0, 0, 0, 1) and b == (0, 1, 0, 0, 0):
            if resp == 1:
                score += 1
            count += 1
        # Trial 4: Pure Tally prefers A (2-1). Drop model prefers A even more (2-0).
        # We score choices for A, which should increase under Drop model.
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 0, 0, 1):
            if resp == 0:
                score += 1
            count += 1

    if count == 0:
        return 0.5

    return score / count
outcome: self_sim=0.5362 (var=0.0052) adversary_sim=0.4992 (var=0.0033) welch_t=+2.009 p=0.05043 (N=25, alpha=0.01) -> reject

[2] rationale: This metric isolates the exact causal mechanism of the Advocated Theory while completely zeroing out under Pure Tallying. Under Pure Tallying, Trials 3 and 4 present identical 2-vs-1 tally advantages for Option A, making P(A|T4) = P(A|T3) for any beta and epsilon. Similarly, Trials 1 and 2 are 2-vs-2 tally ties, meaning P(A|T1) = 0.5 and P(B|T2) = 0.5. Thus, the metric's expected value under Pure Tallying is precisely 0. Under the Advocated Theory's cue-dropping mechanism, however, dropping the least valid cue removes B's only advantage in T4 (increasing P(A)), removes one of A's advantages in T3 (decreasing P(A)), and breaks the ties in T1 and T2 in favor of the target options (increasing P(A|T1) and P(B|T2)). The metric aggregates all these systematic deviations into a single robust contrast with an expected value strictly greater than 0.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t4_a = 0; t4_n = 0
    t3_a = 0; t3_n = 0
    t1_a = 0; t1_n = 0
    t2_b = 0; t2_n = 0

    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']

        # T4: Pure Tally 2-1 for A. Drop model 2-0 for A.
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 0, 0, 1):
            t4_a += (resp == 0)
            t4_n += 1
        # T3: Pure Tally 2-1 for A. Drop model 1-1 tie.
        elif a == (1, 0, 0, 0, 1) and b == (0, 1, 0, 0, 0):
            t3_a += (resp == 0)
            t3_n += 1
        # T1: Pure Tally 2-2 tie. Drop model 2-1 for A.
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 0, 1):
            t1_a += (resp == 0)
            t1_n += 1
        # T2: Pure Tally 2-2 tie. Drop model 2-1 for B.
        elif a == (0, 0, 1, 0, 1) and b == (1, 1, 0, 0, 0):
            t2_b += (resp == 1)
            t2_n += 1

    p_t4_a = t4_a / t4_n if t4_n > 0 else 0.5
    p_t3_a = t3_a / t3_n if t3_n > 0 else 0.5
    p_t1_a = t1_a / t1_n if t1_n > 0 else 0.5
    p_t2_b = t2_b / t2_n if t2_n > 0 else 0.5

    return (p_t4_a - p_t3_a) + (p_t1_a + p_t2_b - 1.0)
outcome: self_sim=0.1446 (var=0.0830) adversary_sim=-0.0031 (var=0.0521) welch_t=+2.009 p=0.05043 (N=25, alpha=0.01) -> reject

[3] rationale: Focuses on trials where Pure Tallying predicts a 1-point win for one of the options (T3, T4, T6, T7). Under Pure Tallying, the probability of choosing the tally winner is identical across all four of these trial types, so the difference in win rates between {T4, T6, T7} and {T3} is exactly zero. Under the Advocated Theory (Probabilistic Least-Valid Cue Dropping), dropping the least valid cue increases the tally margin from 1 to 2 in T4, T6, and T7 (increasing the probability of choosing the winner), while it decreases the tally margin from 1 to 0 in T3 (decreasing the probability of choosing the winner). This contrast isolates the divergence between the theories while pooling across 39 trials for the positive term to minimize per-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t467_win = 0
    t467_n = 0
    t3_win = 0
    t3_n = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # T3: A wins by 1 (2-1). Drop least valid -> Tie (1-1)
        if a == (1, 0, 0, 0, 1) and b == (0, 1, 0, 0, 0):
            t3_win += (resp == 0)
            t3_n += 1
        # T4: A wins by 1 (2-1). Drop least valid -> A wins by 2 (2-0)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 0, 0, 1):
            t467_win += (resp == 0)
            t467_n += 1
        # T6: A wins by 1 (3-2). Drop least valid -> A wins by 2 (3-1)
        elif a == (1, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1):
            t467_win += (resp == 0)
            t467_n += 1
        # T7: B wins by 1 (3-2). Drop least valid -> B wins by 2 (3-1)
        elif a == (0, 0, 0, 1, 1) and b == (1, 1, 1, 0, 0):
            t467_win += (resp == 1)
            t467_n += 1
            
    p_467 = t467_win / t467_n if t467_n > 0 else 0.5
    p_3 = t3_win / t3_n if t3_n > 0 else 0.5
    
    return float(p_467 - p_3)
outcome: self_sim=0.0462 (var=0.0182) adversary_sim=0.0292 (var=0.0131) welch_t=+0.478 p=0.6349 (N=25, alpha=0.01) -> reject

[4] rationale: This metric specifically targets Trials 1 and 2, where Pure Tallying perceives a perfect 2-2 tie between the options and therefore predicts exactly a 50% chance of choosing either option, regardless of the choice stochasticity parameters (beta and epsilon). However, the Advocated Theory (Probabilistic Least-Valid Cue Dropping) predicts that when the least valid cue is occasionally dropped, the tie is systematically broken in favor of Option B in Trial 1 and Option A in Trial 2. By computing the proportion of choices that align with this tie-breaking direction, the expected value of this metric is mathematically fixed at exactly 0.5 under Pure Tallying, while it will be systematically strictly greater than 0.5 under the Advocated Theory. This provides a clean, zero-expectation baseline contrast with minimal variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    target_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a = tuple(int(x) for x in row['option_a_ratings'])
        b = tuple(int(x) for x in row['option_b_ratings'])
        resp = int(row['response'])
        
        # Trial 1: Pure Tally ties (2-2). Drop least valid -> B wins (2-1)
        if a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 0, 0):
            if resp == 1:
                target_count += 1
            total_count += 1
            
        # Trial 2: Pure Tally ties (2-2). Drop least valid -> A wins (2-1)
        elif a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 0, 1):
            if resp == 0:
                target_count += 1
            total_count += 1
            
    if total_count == 0:
        return 0.5
        
    return float(target_count) / total_count

outcome: self_sim=0.5658 (var=0.0162) adversary_sim=0.5058 (var=0.0083) welch_t=+1.919 p=0.06163 (N=25, alpha=0.01) -> reject

[5] rationale: This metric contrasts trials where dropping the least valid cue amplifies the Tally winner's advantage against trials where it attenuates the advantage to a tie. Under Pure Tallying, the Tally winner has the exact same 1-point advantage in all four of these trial types, so the expected difference in choice probabilities is precisely 0. Under the Advocated Theory, occasionally dropping the least valid cue increases the probability of choosing the Tally winner in the 'Amplify' trials while decreasing it in the 'Attenuate' trials, leading to a strictly positive expected difference. By pooling 24 trials for each proportion, we minimize per-subject variance while maintaining a zero-expectation baseline for the competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    amplify_wins = 0
    amplify_n = 0
    attenuate_wins = 0
    attenuate_n = 0
    
    for _, row in data.iterrows():
        a = tuple(int(x) for x in row['option_a_ratings'])
        b = tuple(int(x) for x in row['option_b_ratings'])
        resp = int(row['response'])
        
        # T4: Tally winner is A (3-2). Drop 5th -> A wins 3-1 (Amplify)
        if a == (1, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1):
            amplify_wins += (resp == 0)
            amplify_n += 1
        # T7: Tally winner is B (3-2). Drop 5th -> B wins 3-1 (Amplify)
        elif a == (0, 0, 0, 1, 1) and b == (1, 1, 1, 0, 0):
            amplify_wins += (resp == 1)
            amplify_n += 1
        # T3: Tally winner is A (3-2). Drop 5th -> Tie 2-2 (Attenuate)
        elif a == (1, 1, 0, 0, 1) and b == (0, 0, 1, 1, 0):
            attenuate_wins += (resp == 0)
            attenuate_n += 1
        # T6: Tally winner is B (3-2). Drop 5th -> Tie 2-2 (Attenuate)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            attenuate_wins += (resp == 1)
            attenuate_n += 1
            
    p_amp = amplify_wins / amplify_n if amplify_n > 0 else 0.5
    p_att = attenuate_wins / attenuate_n if attenuate_n > 0 else 0.5
    
    return float(p_amp - p_att)
outcome: self_sim=0.0017 (var=0.0133) adversary_sim=0.0025 (var=0.0101) welch_t=-0.027 p=0.9784 (N=25, alpha=0.01) -> reject

[6] rationale: This composite metric aggregates over 6 of the 8 trial types (72 trials per subject) to create a highly robust contrast with a mathematically exact expectation under Pure Tallying. Under Pure Tallying, T1 and T2 are 50/50 guesses. In T3/T4, Option A wins the tally by 1 point, so P(B|T3) + P(A|T4) = p_loss + p_win = 1.0. In T6/T7, Option B wins the tally by 1 point, so P(A|T6) + P(B|T7) = p_loss + p_win = 1.0. Summing the target choices across these 6 trials yields an expected sum of 3.0, making the average exactly 0.5 regardless of beta and epsilon. Under the Advocated Theory, dropping the least valid cue systematically breaks the ties in T1/T2 in favor of the target options, attenuates the tally winner's advantage in T3/T6 (increasing the target option's probability), and amplifies the tally winner's advantage in T4/T7 (also increasing the target option's probability). Thus, the expected value strictly exceeds 0.5. Averaging over 72 trials minimizes per-subject variance, ensuring the contrast is highly significant.
metric_source:
def metric(data: pd.DataFrame) -> float:
    score = 0
    count = 0
    
    for _, row in data.iterrows():
        a = tuple(int(x) for x in row['option_a_ratings'])
        b = tuple(int(x) for x in row['option_b_ratings'])
        resp = int(row['response'])
        
        # T1: Pure Tally ties (2-2). Drop 5th -> B wins (1-2). Target: B
        if a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 0, 0):
            score += (resp == 1)
            count += 1
        # T2: Pure Tally ties (2-2). Drop 5th -> A wins (2-1). Target: A
        elif a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 0, 1):
            score += (resp == 0)
            count += 1
        # T3: Pure Tally A wins (3-2). Drop 5th -> Tie (2-2). Target: B (attenuated A advantage)
        elif a == (1, 1, 0, 0, 1) and b == (0, 0, 1, 1, 0):
            score += (resp == 1)
            count += 1
        # T4: Pure Tally A wins (3-2). Drop 5th -> A wins (3-1). Target: A (amplified A advantage)
        elif a == (1, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1):
            score += (resp == 0)
            count += 1
        # T6: Pure Tally B wins (3-2). Drop 5th -> Tie (2-2). Target: A (attenuated B advantage)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            score += (resp == 0)
            count += 1
        # T7: Pure Tally B wins (3-2). Drop 5th -> B wins (3-1). Target: B (amplified B advantage)
        elif a == (0, 0, 0, 1, 1) and b == (1, 1, 1, 0, 0):
            score += (resp == 1)
            count += 1
            
    if count == 0:
        return 0.5
    return float(score) / count
outcome: self_sim=0.5225 (var=0.0037) adversary_sim=0.5028 (var=0.0016) welch_t=+1.351 p=0.1841 (N=25, alpha=0.01) -> reject

[7] rationale: This metric contrasts two sets of trials (X and Y) that have mathematically identical target choice expectations under Pure Tallying, but maximally divergent expectations under the Advocated Theory. Under Pure Tallying, T1 and T2 are both 2-vs-2 ties, so P(A) = 0.5 for both. T3 and T4 are 3-vs-2 tally wins for A, so P(A) is identical. T6 and T7 are 3-vs-2 tally wins for B, so P(B) is identical. Therefore, the expected number of target choices in Set X exactly equals Set Y under Pure Tallying, meaning X / (X + Y) has a rigid expectation of 0.5 regardless of beta and epsilon. Under the Advocated Theory's cue-dropping mechanism, Set X choices are systematically amplified (ties broken in favor of target, or 1-point leads widened to 2-points), while Set Y choices are systematically attenuated (ties broken against target, or 1-point leads reduced to ties). This drives X up and Y down, pushing the metric robustly above 0.5. By computing a single pooled ratio over 72 trials per subject, we massively reduce the per-subject variance compared to taking differences of independent proportions.
metric_source:
def metric(data: pd.DataFrame) -> float:
    x_count = 0
    y_count = 0
    
    for _, row in data.iterrows():
        a = tuple(int(v) for v in row['option_a_ratings'])
        b = tuple(int(v) for v in row['option_b_ratings'])
        resp = int(row['response'])
        
        # Set X: Trials where Advocated Theory predicts an INCREASE in target probability compared to Pure Tallying
        # T2: Pure Tally Tie (2-2). Drop 5th -> A wins (2-1). Target: A
        if a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 0, 1):
            if resp == 0: x_count += 1
        # T4: Pure Tally A wins (3-2). Drop 5th -> A wins (3-1). Target: A
        elif a == (1, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1):
            if resp == 0: x_count += 1
        # T7: Pure Tally B wins (3-2). Drop 5th -> B wins (3-1). Target: B
        elif a == (0, 0, 0, 1, 1) and b == (1, 1, 1, 0, 0):
            if resp == 1: x_count += 1
            
        # Set Y: Trials where Advocated Theory predicts a DECREASE in target probability compared to Pure Tallying
        # T1: Pure Tally Tie (2-2). Drop 5th -> B wins (1-2). Target: A
        elif a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 0, 0):
            if resp == 0: y_count += 1
        # T3: Pure Tally A wins (3-2). Drop 5th -> Tie (2-2). Target: A
        elif a == (1, 1, 0, 0, 1) and b == (0, 0, 1, 1, 0):
            if resp == 0: y_count += 1
        # T6: Pure Tally B wins (3-2). Drop 5th -> Tie (2-2). Target: B
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1: y_count += 1
            
    if x_count + y_count == 0:
        return 0.5
    return float(x_count) / (x_count + y_count)
outcome: self_sim=0.5156 (var=0.0018) adversary_sim=0.5019 (var=0.0008) welch_t=+1.361 p=0.1808 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    positives = 0\n    negatives = 0\n    \n    for _, row in data.iterrows():\n        a = tuple(int(v) for v in row['option_a_ratings'])\n        b = tuple(int(v) for v in row['option_b_ratings'])\n        resp = int(row['response'])\n        \n        # T4: Pure Tally A wins (2-1). Drop 5th -> A wins (2-0). Target: A (amplified)\n        if a == (1, 1, 0, 0, 0) and b == (0, 0, 0, 0, 1):\n            if resp == 0: positives += 1\n        # T6: Pure Tally B wins (2-1). Drop 5th -> B wins (2-0). Target: B (amplified)\n        elif a == (0, 0, 0, 0, 1) and b == (1, 1, 0, 0, 0):\n            if resp == 1: positives += 1\n        # T1: Pure Tally Tie (2-2). Drop 5th -> B wins (2-1). Target: B (tie broken for B)\n        elif a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 0, 0):\n            if resp == 1: positives += 1\n            else: negatives += 1\n        # T2: Pure Tally Tie (2-2). Drop 5th -> A wins (2-1). Target: A (tie broken for A)\n        elif a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 0, 1):\n            if resp == 0: positives += 1\n            else: negatives += 1\n        # T3: Pure Tally A wins (2-1). Drop 5th -> Tie (1-1). Target: A (attenuated)\n        elif a == (1, 0, 0, 0, 1) and b == (0, 0, 0, 1, 0):\n            if resp == 0: negatives += 1\n        # T5: Pure Tally B wins (2-1). Drop 5th -> Tie (1-1). Target: B (attenuated)\n        elif a == (0, 0, 0, 1, 0) and b == (1, 0, 0, 0, 1):\n            if resp == 1: negatives += 1\n            \n    if positives + negatives == 0:\n        return 0.5\n    return float(positives) / (positives + negatives)",
  "rationale": "This metric contrasts two sets of choices that have mathematically identical expectations under Pure Tallying but maximally divergent expectations under the Advocated Theory. Under Pure Tallying, T1 and T2 are 2-vs-2 ties, so P(A) = P(B) = 0.5. T3, T4, T5, and T6 are all 2-vs-1 tally wins, meaning the probability of choosing the tally winner is identical across all four. Thus, the expected number of 'positives' exactly equals the expected number of 'negatives', locking the ratio at 0.5 regardless of beta and epsilon. Under the Advocated Theory's cue-dropping mechanism, 'positives' are systematically amplified (ties broken in favor of the target, or 1-point leads widened to 2-points), while 'negatives' are systematically attenuated (ties broken against the target, or 1-point leads reduced to ties). This drives the 'positives' count up and the 'negatives' count down, pushing the metric robustly above 0.5. By pooling 72 trials per subject into a single ratio, we massively reduce per-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 10110,
  "candidates_token_count": 936,
  "total_token_count": 15319
}
```
