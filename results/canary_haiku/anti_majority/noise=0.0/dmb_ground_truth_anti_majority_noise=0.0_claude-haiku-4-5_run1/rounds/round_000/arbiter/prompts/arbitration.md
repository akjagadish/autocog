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
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_2
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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.9, 0.7, 0.6, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 9: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 11: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 12: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Rationale:** This design targets a fundamental dissociation between TTB and Tallying. TTB consults cues in order of validity and stops at the first discriminating cue, making it sensitive to validity ordering and ignoring lower-validity information. Tallying counts wins across all features equally, ignoring validity entirely. The key insight: construct trials where a HIGH-validity cue favors one option but MULTIPLE LOWER-validity cues favor the other. TTB will choose based on the high-validity cue alone; Tallying will tally all cues equally and often choose the opposite. We use 4 features with validities [0.9, 0.7, 0.6, 0.55] to create a clear hierarchy. Trial types: (1) High-validity cue discriminates for A; lower-validity cues discriminate for B (3 of them). TTB picks A; Tallying picks B. (2) High-validity cue discriminates for B; lower-validity cues discriminate for A (3 of them). TTB picks B; Tallying picks A. (3) Control trials where TTB and Tallying agree (e.g., high-validity cue and majority of lower cues favor same option). This creates quantitative dissociation: we predict systematic preference reversals between the two theories on conflict trials, with effect size proportional to the validity gap and the number of conflicting lower-validity cues.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric targets the core dissociation: on trials where the highest-validity cue (feature 0, validity 0.9) discriminates but lower-validity cues (features 1-3, validities 0.7, 0.6, 0.55) favor the opposite option, TTB will choose based on feature 0 alone, while Tallying will count all four cues equally and often choose based on the majority of the three lower cues. The metric measures the fraction of such conflict trials where the subject chose the feature-0 winner. TTB theory predicts this fraction should be high (~0.8-0.9, depending on beta and epsilon); Tallying predicts it should be near chance or even inverted (~0.1-0.4), since Tallying often picks the opposite of feature 0. The metric is stable per subject (computed as a simple proportion on a fixed subset of trials) and produces a large, tight mean separation between theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Metric: Preference for high-validity-cue-favoring choice on conflict trials.
    # TTB should strongly prefer choices where the highest-validity cue (feature 0)
    # discriminates, even when lower-validity cues favor the opposite. Tallying
    # treats all cues equally, so it should show much weaker preference for the
    # high-validity-cue winner on these conflict trials.
    #
    # Strategy: identify trials where feature 0 discriminates (A vs B differ on
    # feature 0) AND the majority of features 1-3 favor the opposite option.
    # On such trials, TTB picks the feature-0 winner; Tallying often picks the
    # opposite (majority of 3 lower cues). We measure the fraction of conflict
    # trials where the subject chose the feature-0 winner.
    #
    # This metric is tight per subject (binary choice on a fixed trial set)
    # and has a large mean gap between theories.
    
    conflict_trials = []
    
    for idx, row in data.iterrows():
        a_ratings = np.asarray(row['option_a_ratings'], dtype=int)
        b_ratings = np.asarray(row['option_b_ratings'], dtype=int)
        response = int(row['response'])
        
        # Check if feature 0 discriminates
        if a_ratings[0] == b_ratings[0]:
            continue
        
        # Determine which option wins on feature 0
        feature_0_winner = 0 if a_ratings[0] > b_ratings[0] else 1
        
        # Count wins on features 1-3
        a_wins_lower = np.sum(a_ratings[1:] > b_ratings[1:])
        b_wins_lower = np.sum(b_ratings[1:] > a_ratings[1:])
        
        # Conflict trial: feature 0 favors one option, majority of features 1-3 favor the other
        # We need strict majority (at least 2 out of 3) of lower cues to favor the opposite
        if a_wins_lower > b_wins_lower and feature_0_winner == 1:
            # Feature 0 favors B, but A wins on majority of lower cues
            conflict_trials.append((1, response))  # TTB predicts B (1), Tallying often predicts A (0)
        elif b_wins_lower > a_wins_lower and feature_0_winner == 0:
            # Feature 0 favors A, but B wins on majority of lower cues
            conflict_trials.append((0, response))  # TTB predicts A (0), Tallying often predicts B (1)
    
    if len(conflict_trials) == 0:
        return 0.5  # No conflict trials; neutral
    
    # Compute fraction of conflict trials where subject chose the feature-0 winner
    # (i.e., the TTB prediction)
    ttb_correct = sum(1 for predicted, actual in conflict_trials if predicted == actual)
    return float(ttb_correct) / len(conflict_trials)

### RESULTS
- Predicted under pi_1 (simulated): 0.8472 (var=0.0098)
- Predicted under pi_2 (simulated): 0.4122 (var=0.0053)
- Observed on real data: 0.4012 (var=0.0172)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.65, 0.6, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 2: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 4: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** This design extends the core TTB vs. Tallying dissociation by introducing a NEW dimension: feature-validity ordering effects under TIED high-validity cues. The key insight is that when the highest-validity cue ties between options, TTB must cascade to the next-highest cue, whereas Tallying ignores the hierarchy entirely and counts all features equally. We propose 5 features with validities [0.95, 0.90, 0.65, 0.60, 0.55] to create a sharp validity cliff after the first two cues. Trial design: (1) TIED high-validity cues (features 0 and 1 both tie), forcing TTB to consult feature 2 (validity 0.65). Feature 2 discriminates for option A. Meanwhile, features 3 and 4 (lower validity) discriminate for B. TTB picks A (follows feature 2 after tie-breaking). Tallying: A wins on feature 2 only; B wins on features 3 and 4. Tallying picks B. (2) Mirror: features 0 and 1 tie; feature 2 discriminates for B; features 3 and 4 discriminate for A. TTB picks B; Tallying picks A. (3) Control trials where all theories agree (e.g., feature 0 discriminates for A; no other discriminations, or all lower cues also favor A). This design isolates the validity-ordering cascade effect in TTB versus the egalitarian aggregation in Tallying, creating a quantitative dissociation that is orthogonal to the existing experiment [0]. The tie at the top forces TTB into a specific cascade behavior that Tallying's equal-weight rule directly contradicts.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric directly operationalizes the core theoretical dissociation: TTB's validity-ordered cascade versus Tallying's equal-weight aggregation. On tie-breaking trials (1-4, 7-8), the two theories make divergent predictions. TTB cascades to feature 2 after features 0-1 tie, while Tallying counts all features equally. By measuring the subject's agreement rate with each theory's predictions and computing the difference, we obtain a signed metric that is large and stable when subjects follow one theory consistently. Subjects who follow TTB will show positive values; subjects who follow Tallying will show negative values. The metric is robust to individual differences in beta and epsilon because it measures choice consistency with the underlying decision process, not the parameters themselves. Critical trials (1-4, 7-8) dominate the signal, while control and agreement trials provide stability. This design should yield a large between-theory gap with tight within-subject variance, satisfying both the mean and variance acceptance criteria.

Source:
def metric(data: pd.DataFrame) -> float:
    """
    Measure the degree to which subjects employ a validity-ordered cascade
    (TTB) versus equal-weight tallying (Tallying).
    
    Key insight: On trials where the two highest-validity features (0, 1) tie
    between options, TTB must consult feature 2 (validity 0.65) to break the tie,
    while Tallying counts all features equally. This creates a sharp dissociation:
    - TTB predicts choice based on feature 2 after tie-breaking at features 0-1.
    - Tallying predicts choice based on the majority vote across all features.
    
    We compute a "cascade fidelity" score: the proportion of trials where the
    subject's choice aligns with TTB's prediction (given the validity cascade)
    minus the proportion where it aligns with Tallying's prediction.
    
    On the critical tie-breaking trials (1-4, 7-8), TTB and Tallying diverge.
    On control trials (9-12), both theories largely agree. On low-validity trials
    (5-6), both theories agree. The metric isolates the cascade effect.
    
    Validities: [0.95, 0.90, 0.65, 0.60, 0.55]
    """
    validities = np.array([0.95, 0.90, 0.65, 0.60, 0.55])
    
    # Define the 12 trial pairs (from the experimental design)
    trial_pairs = [
        ([1, 1, 1, 0, 0], [1, 1, 0, 1, 1]),  # trial 1
        ([1, 1, 0, 1, 1], [1, 1, 1, 0, 0]),  # trial 2
        ([1, 1, 1, 0, 0], [1, 1, 0, 1, 1]),  # trial 3
        ([1, 1, 0, 1, 1], [1, 1, 1, 0, 0]),  # trial 4
        ([0, 0, 1, 0, 0], [0, 0, 0, 1, 1]),  # trial 5
        ([0, 0, 0, 1, 1], [0, 0, 1, 0, 0]),  # trial 6
        ([0, 0, 1, 0, 0], [0, 0, 0, 1, 1]),  # trial 7
        ([0, 0, 0, 1, 1], [0, 0, 1, 0, 0]),  # trial 8
        ([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]),  # trial 9
        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),  # trial 10
        ([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]),  # trial 11
        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),  # trial 12
    ]
    
    # Helper: compute TTB prediction (0 = choose A, 1 = choose B, None = tie)
    def ttb_predict(a, b, vals):
        cue_order = np.argsort(-vals, kind='stable')
        for j in cue_order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None  # All tied
    
    # Helper: compute Tallying prediction (0 = choose A, 1 = choose B, None = tie)
    def tallying_predict(a, b):
        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])
        b_wins = sum(1 for i in range(len(b)) if b[i] > a[i])
        if a_wins > b_wins:
            return 0
        elif b_wins > a_wins:
            return 1
        else:
            return None  # Tied
    
    # Count agreement with each theory, excluding ties
    ttb_matches = 0
    tallying_matches = 0
    valid_trials = 0
    
    for idx, row in data.iterrows():
        a = np.array(row['option_a_ratings'], dtype=int)
        b = np.array(row['option_b_ratings'], dtype=int)
        response = int(row['response'])
        
        ttb_pred = ttb_predict(a, b, validities)
        tallying_pred = tallying_predict(a, b)
        
        # Only count trials where both theories make a definite prediction
        if ttb_pred is not None and tallying_pred is not None:
            valid_trials += 1
            if response == ttb_pred:
                ttb_matches += 1
            if response == tallying_pred:
                tallying_matches += 1
    
    if valid_trials == 0:
        return 0.0
    
    # Cascade fidelity: TTB agreement rate minus Tallying agreement rate
    # Positive values favor TTB; negative values favor Tallying.
    ttb_rate = ttb_matches / valid_trials
    tallying_rate = tallying_matches / valid_trials
    
    return ttb_rate - tallying_rate

### RESULTS
- Predicted under pi_1 (simulated): 0.4587 (var=0.0220)
- Predicted under pi_2 (simulated): -0.4417 (var=0.0226)
- Observed on real data: 0.2300 (var=0.2070)

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
  "interpretation": "Analyzing the two experiments reveals a critical pattern that neither theory alone captures well across both tasks.\n\nExperiment 1 (TTB-favoring design): The observed metric (0.4012, var=0.0172) aligns much more closely with Tallying's prediction (0.4122, var=0.0053) than with TTB's prediction (0.8472, var=0.0098). This is a striking failure for pi_1: the experiment was explicitly designed to favor TTB by pitting a high-validity cue against multiple lower-validity cues, yet subjects behaved like Tallying theorists, treating all features roughly equally. The tight variance on the observed data (0.0172) suggests consistent subject behavior, but it contradicts TTB's core premise that validity ordering drives choice.\n\nExperiment 2 (Tallying-favoring design): The observed metric (0.2300, var=0.2070) lies between the two theories' predictions (pi_1: 0.4587; pi_2: -0.4417), but critically, it is much closer to pi_1's prediction and has substantially higher variance (0.2070 vs. 0.0220 and 0.0226 for the predictions). This suggests subject heterogeneity or that neither theory's mechanism is being deployed consistently. The high variance is concerning and indicates instability.\n\nCross-experiment analysis: \n- pi_1 (TTB) fails decisively on Experiment 1, where it predicts 0.85 but observes 0.40\u2014a 0.45-point error. It performs better on Experiment 2 (predicts 0.46, observes 0.23), but this is likely accidental alignment rather than theoretical success.\n- pi_2 (Tallying) succeeds on Experiment 1 (predicts 0.41, observes 0.40\u2014a 0.01-point error, nearly perfect). However, it fails dramatically on Experiment 2 (predicts -0.44, observes 0.23\u2014a 0.67-point error, and in the opposite direction).\n\nNeither theory is task-invariant. TTB's validity-ordering mechanism appears not to be operating in subjects' cognition\u2014subjects in Experiment 1 ignored the validity hierarchy entirely. Tallying's equal-weight rule works well when features are balanced, but fails when the task structure (as in Experiment 2) creates a situation where validity hierarchy becomes relevant.\n\nThe high variance in Experiment 2 (0.2070) is a red flag suggesting that subjects are switching strategies or that the underlying decision process is more complex than either pure TTB or pure Tallying. This could indicate a hybrid or context-dependent strategy: subjects may use Tallying by default (explaining Experiment 1), but when features are structured with high-validity ties (Experiment 2), they partially shift toward cascade-like reasoning, though inconsistently.\n\nA new theory is warranted because: (1) neither existing theory explains both experiments well; (2) the pattern suggests a conditional or adaptive mechanism rather than a fixed single strategy; (3) proposing a new model for either theory would only patch one experiment without addressing the fundamental incompatibility.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a theory of adaptive or conditional decision-making that blends validity-ordered cascade with equal-weight tallying based on task structure. The key insight from the data: subjects default to Tallying (equal-weight feature counting) when the validity structure is ambiguous or when features are roughly balanced in discriminative power (Experiment 1). However, when the task explicitly creates a sharp validity cliff (Experiment 2: validities [0.95, 0.90, ...] with a large drop to 0.65), subjects partially engage in validity-aware cascade reasoning, though with substantial heterogeneity. The new theory should specify: (1) A primary mechanism: Tallying with equal-weight feature counting as the baseline heuristic. (2) A triggering condition for validity-ordered cascade: When the ratio or gap between top-validity cues and lower cues exceeds a threshold (e.g., >0.25 validity gap), subjects shift partially toward TTB-like cascade. (3) A mixing parameter \u03b1 that interpolates between Tallying (\u03b1=0) and TTB (\u03b1=1), modulated by the validity structure. (4) Uncertainty or noise in the validity perception: subjects may not perfectly learn or apply stated validities, introducing heterogeneity. This theory would predict: Experiment 1 (validities [0.9, 0.7, 0.6, 0.55], gaps \u22640.3): \u03b1 \u2248 0, behavior \u2248 Tallying (observed 0.40 \u2713). Experiment 2 (validities [0.95, 0.90, 0.65, ...], sharp 0.25 gap): \u03b1 partially elevated, behavior intermediate between Tallying and TTB (observed 0.23, between -0.44 and 0.46 \u2713, with higher variance due to heterogeneous \u03b1). The new theory should be called something like 'Adaptive Validity-Aware Tallying' or 'Conditional Cascade-Tallying Hybrid' and should include a principled mechanism for when and how much subjects weight validity ordering versus equal aggregation."
}
```

## Usage

```json
{
  "input_tokens": 7961,
  "output_tokens": 1255
}
```
