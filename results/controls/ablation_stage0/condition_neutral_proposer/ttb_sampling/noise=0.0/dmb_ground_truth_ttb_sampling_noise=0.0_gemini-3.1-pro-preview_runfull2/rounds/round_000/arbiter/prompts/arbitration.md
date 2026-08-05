# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_2
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 9: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 11: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 12: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To maximally distinguish Tallying (Theory 1) from Weighted Additive (Theory 2), we need a set of trials where the two models make conflicting predictions. Tallying decides solely based on the count of features favoring each option, treating all features as equally important. In contrast, WADD computes a weighted sum using the feature validities. By using a skewed distribution of validities (e.g., [0.95, 0.85, 0.65, 0.55, 0.50]), we can construct trials where one option wins on a smaller number of highly valid features (favored by WADD) while the other option wins on a larger number of less valid features (favored by Tallying). The design also includes trials where Tallying predicts a tie but WADD predicts a clear winner, as well as baseline agreement trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Tallying decides solely based on the difference in the number of feature wins ('tally_diff'). Thus, for any given subject, the true probability of choosing Option A is mathematically identical for all trial types that share the same 'tally_diff'. Any observed variance in choice proportions among trials with the same 'tally_diff' is purely due to binomial sampling noise (which for 8 repetitions is strictly bounded at 0.5 * 0.5 / 8 = 0.03125). In contrast, WADD computes a weighted sum using differing feature validities and subject-specific random weights, meaning its true preference for Option A varies substantially even among trials with the exact same 'tally_diff'. By computing the pooled within-group variance of the choice proportions (grouped by 'tally_diff'), we perfectly isolate this difference. Tallying will yield a strictly small variance (<= 0.03125), while WADD will yield a much larger variance due to the dispersion of its true underlying choice probabilities across the different trial types.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # Create a safe, hashable string identifier for each trial type
    data['trial_id'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x])) + '_' + \
                       data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Calculate the Tallying difference (wins for A - wins for B)
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    data['tally_diff'] = np.sum(a_ratings > b_ratings, axis=1) - np.sum(b_ratings > a_ratings, axis=1)
    
    # Calculate choice for A (response == 0 means A was chosen)
    data['choice_A'] = 1 - data['response']
    
    # Calculate the proportion of times A was chosen for each trial type, per subject
    trial_means = data.groupby(['subject_id', 'tally_diff', 'trial_id'])['choice_A'].mean().reset_index()
    
    # Compute the pooled within-group variance of choice proportions for each subject
    def pooled_variance(df):
        var_sum = 0.0
        df_sum = 0.0
        for t_diff, group in df.groupby('tally_diff'):
            n = len(group)
            if n > 1:
                v = group['choice_A'].var(ddof=1)
                var_sum += v * (n - 1)
                df_sum += (n - 1)
        if df_sum == 0:
            return 0.0
        return float(var_sum / df_sum)
        
    subj_vars = []
    for subj, subj_df in trial_means.groupby('subject_id'):
        subj_vars.append(pooled_variance(subj_df))
        
    return float(np.mean(subj_vars))

### RESULTS
- Predicted under pi_1 (simulated): 0.0216 (var=0.0001)
- Predicted under pi_2 (simulated): 0.0866 (var=0.0030)
- Observed on real data: 0.1080 (var=0.0018)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.98, 0.85, 0.65, 0.55, 0.51]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To maximally distinguish WADD (Theory 1) from Tallying (Theory 2), we need a set of validities that are highly skewed, allowing a small number of high-validity features to outweigh a larger number of low-validity features. The trials are designed to include strong dissociations where WADD prefers the option with fewer, but highly valid features, while Tallying prefers the option with more, but less valid features. We also include trials where Tallying predicts a tie but WADD has a clear preference, and baseline trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option with fewer but higher-validity features (e.g., A=[1, 1, 0, 0, 0]) over the option with more but lower-validity features (e.g., B=[0, 0, 1, 1, 1]). Tallying strictly counts the number of positive features and thus strongly prefers the option with more features, yielding a value close to 0 (modulo noise). WADD, on the other hand, weighs features by their validities. Because the validities are heavily skewed towards the first two features (0.98 + 0.85 = 1.83 vs 0.65 + 0.55 + 0.51 = 1.71), WADD will choose the fewer-but-better option much more frequently, yielding a significantly higher metric value.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_fewer_but_better(a, b):
        return tuple(a) == (1, 1, 0, 0, 0) and tuple(b) == (0, 0, 1, 1, 1)
        
    def is_worse_but_more(a, b):
        return tuple(a) == (0, 0, 1, 1, 1) and tuple(b) == (1, 1, 0, 0, 0)

    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        if is_fewer_but_better(a, b):
            matches.append(1.0 if resp == 0 else 0.0)
        elif is_worse_but_more(a, b):
            matches.append(1.0 if resp == 1 else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.1467 (var=0.0159)
- Predicted under pi_2 (simulated): 0.5692 (var=0.0859)
- Observed on real data: 0.8267 (var=0.0120)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
(no other experiments)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both experiments, Theory 2 (Weighted Additive / WADD) captures the observed data much better than Theory 1 (Tallying). In Experiment 1, subjects show high variance in choice proportions for trials with the same 'tally difference', which Tallying cannot explain (predicting ~0.02 vs observed 0.108). In Experiment 2, subjects overwhelmingly choose the option with fewer but higher-validity features (observed 0.827), whereas Tallying predicts they should choose the option with more features (predicting 0.147). However, even WADD underpredicts the extent to which subjects rely on the highest validity cues in Experiment 2 (predicting 0.569 vs observed 0.827). This suggests subjects are using a much more non-compensatory strategy than WADD, such as Take The Best. Theory 1 is fundamentally degenerate and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Take The Best' (TTB) theory. Unlike Tallying (which weights all cues equally) and WADD (which integrates all cues compensatorily), TTB orders cues by their subjective validity and searches through them sequentially. It stops at the first cue that discriminates between the two options and bases the decision entirely on that single cue. This non-compensatory heuristic will naturally explain the high variance in Experiment 1 (since trials with the same tally difference can be decided by different highest-validity cues) and will correctly predict the extreme preference for the 'fewer but better' option in Experiment 2 (since the highest validity cue alone dictates the choice)."
}
```

## Usage

```json
{
  "prompt_token_count": 5011,
  "candidates_token_count": 379,
  "total_token_count": 5977
}
```
