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
**Validities (n_features=4):** [0.9, 0.7, 0.55, 0.52]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 6: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 7: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 8: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 9: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 11: A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  trial 12: A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** This design targets a critical dissociation between Take The Best (TTB) and Tallying. TTB consults features in order of validity and stops at the first discriminating cue, making it entirely dependent on which high-validity feature first discriminates. Tallying counts wins across ALL features equally, regardless of validity order. We construct trials where: (1) a high-validity feature discriminates in favor of one option, but (2) multiple lower-validity features unanimously favor the opposite option. TTB will choose based solely on the high-validity cue and ignore the lower-validity consensus. Tallying will count all features equally and choose the option with more feature wins. This creates a quantitative dissociation in choice probability: TTB should strongly prefer the high-validity-favored option, while Tallying should prefer the option with more feature wins. We use 4 features with validities [0.9, 0.7, 0.55, 0.52] to create a clear hierarchy. Trial pairs are designed so that on some trials the highest-validity feature discriminates one way while the three lower-validity features unanimously discriminate the opposite way, creating maximum conflict between the theories.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
TTB is a cue-hierarchy model: it consults features in order of validity and stops at the first discriminating cue. On this design, when the top cue (feature 0, validity 0.9) discriminates, TTB will choose based on that cue alone and ignore all others. Tallying counts feature wins equally across all features, so it has no principled reason to preferentially follow the top cue — when the top cue favors one option but the three lower-validity features unanimously favor the other (as in trials 1–4, 9–10, 11–12), Tallying will often choose the option with more total wins (3 vs 1), even though the top cue disagrees. This metric computes the proportion of trials where the subject's choice agrees with the top cue's prediction (when the top cue discriminates). TTB should show a high proportion (≈0.9+, modulated by beta and epsilon), while Tallying should show a lower proportion because it frequently chooses based on the multi-feature consensus. The metric is robust: it only counts trials where the top cue discriminates (removing noise from ties), and it works per-subject (enabling between-subject variance estimation) because each subject sees all trial types multiple times across the 8 repetitions.

Source:
def metric(data: pd.DataFrame) -> float:
    # Metric: Proportion of trials where subject chooses the option that wins on the HIGHEST-VALIDITY feature.
    # TTB predicts subjects will almost always choose based on the top cue (validity 0.9),
    # so they should choose the option that wins on feature 0 with high probability.
    # Tallying treats all features equally, so it has no reason to preferentially choose
    # based on feature 0 alone — it will sometimes choose the option that loses on feature 0
    # but wins on the lower-validity features (e.g., 3 vs 1 on the remaining features).
    # This metric isolates the signature of TTB's cue hierarchy.
    
    def choose_based_on_top_cue(row):
        a_ratings = np.array(row['option_a_ratings'])
        b_ratings = np.array(row['option_b_ratings'])
        # Top cue is feature 0 (validity 0.9)
        if a_ratings[0] > b_ratings[0]:
            # Top cue favors A
            return row['response'] == 0
        elif b_ratings[0] > a_ratings[0]:
            # Top cue favors B
            return row['response'] == 1
        else:
            # Top cue ties; this trial does not constrain the metric
            # (neither theory predicts a strong preference)
            return np.nan
    
    # Apply to all trials
    matches = data.apply(choose_based_on_top_cue, axis=1)
    # Count trials where response matches top-cue prediction, excluding ties
    valid_matches = matches[~matches.isna()]
    if len(valid_matches) == 0:
        return np.nan
    return float(valid_matches.sum()) / float(len(valid_matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.8248 (var=0.0097)
- Predicted under pi_2 (simulated): 0.3202 (var=0.0034)
- Observed on real data: 0.3842 (var=0.0029)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.8, 0.75, 0.7, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 9: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design targets a complementary dissociation to the earlier experiment by exploiting a different structural vulnerability: the temporal ordering of validity-based decisions. We construct trials where MULTIPLE high-validity features discriminate in favor of one option, but a SINGLE very-high-validity feature discriminates in the opposite direction. TTB will stop at the first (highest-validity) cue and choose based solely on it, ignoring all subsequent features regardless of their count or validity. Tallying will weight all features equally and choose the option that wins on more features overall. Critically, we engineer trials where: (1) the single highest-validity feature (0.95) discriminates for option A, (2) the next three features (0.80, 0.75, 0.70) all discriminate for option B, and (3) a lower-validity feature (0.55) also favors B. TTB will deterministically choose A (stopping at the highest-validity cue). Tallying will count 1 win for A vs. 3 wins for B and choose B. This creates a stark quantitative dissociation in the opposite direction from Experiment 0: here Tallying should strongly prefer the option with more mid-validity feature wins, while TTB should strongly prefer the option favored by the single highest-validity cue. We include control trials where the highest-validity cue and the tally agree to establish baseline choice rates, and trials with feature ties to probe noise parameters. This design isolates whether subjects use a sequential validity-ordered stopping rule (TTB) or an equal-weight tallying rule (Tallying).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric directly operationalizes the core dissociation between the two theories on this experimental design. The design is specifically engineered so that TTB (consulting features in validity order, stopping at the first discriminator) will often choose the option favored by the 0.95-validity feature, while Tallying (counting feature-wise wins) will choose the option that wins on more of the lower-validity features. On trials like 1, 2, 5, 6 where A has the highest-validity feature but B wins on features 2, 3, 4, TTB will choose A (stopping at feature 1) while Tallying will choose B (3 wins vs. 1 win). The metric measures the proportion of choices that align with the tally winner (ignoring trials where all features tie, since neither theory has a basis for preference). Tallying subjects should show high rates (approaching 1.0 if beta is high and epsilon is low), while TTB subjects should show much lower rates because they are swayed by the single highest-validity feature. The metric is stable per-subject because each subject completes 90 trials (~9 per unique pair type), providing sufficient data to estimate their tally-consistency rate with low noise. This creates a large, robust mean difference between the two theories while maintaining low between-subject variance within each theory.

Source:
def metric(data: pd.DataFrame) -> float:
    # Metric: Proportion of trials where subject chose the option with MORE feature wins (tally-consistent choice)
    # TTB will often choose the option favored by the highest-validity feature (0.95),
    # even when that option loses on the remaining features (1 win vs. 3+ wins).
    # Tallying will choose the option with more feature wins.
    # This metric measures the rate at which subjects exhibit tally-consistent behavior.
    
    def compute_tally_winner(row):
        # Returns 0 if A wins more features, 1 if B wins more, None if tie
        a_ratings = np.asarray(row['option_a_ratings'], dtype=float)
        b_ratings = np.asarray(row['option_b_ratings'], dtype=float)
        a_wins = float(np.sum(a_ratings > b_ratings))
        b_wins = float(np.sum(b_ratings > a_ratings))
        if a_wins > b_wins:
            return 0
        elif b_wins > a_wins:
            return 1
        else:
            return None
    
    # Compute tally winner for each trial
    data_copy = data.copy()
    data_copy['tally_winner'] = data_copy.apply(compute_tally_winner, axis=1)
    
    # Filter out tie trials (where tally provides no basis for preference)
    non_tie_trials = data_copy[data_copy['tally_winner'].notna()]
    
    if len(non_tie_trials) == 0:
        return 0.5  # No informative trials
    
    # Count how often subject's response matches the tally winner
    tally_consistent = (non_tie_trials['response'] == non_tie_trials['tally_winner']).sum()
    
    # Return proportion of tally-consistent choices
    return float(tally_consistent) / len(non_tie_trials)


### RESULTS
- Predicted under pi_1 (simulated): 0.1371 (var=0.0100)
- Predicted under pi_2 (simulated): 0.8753 (var=0.0054)
- Observed on real data: 0.7204 (var=0.0038)

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
  "interpretation": "Across both experiments, the observed data reveals a striking pattern that neither pure theory captures well. In Experiment 1, the observed metric (0.3842) aligns closely with pi_2's prediction (0.3202) but deviates dramatically from pi_1's prediction (0.8248). This suggests subjects are NOT using a strict validity-ordered hierarchical stopping rule; instead, they appear to weight multiple features more equally. In Experiment 2, the observed metric (0.7204) falls between pi_1's prediction (0.1371) and pi_2's prediction (0.8753), but notably closer to pi_2. This intermediate performance on Experiment 2 is particularly revealing: it suggests subjects are not purely following the highest-validity cue (as TTB predicts) nor are they deterministically choosing based on feature counts (as pure Tallying predicts). The data pattern across both experiments indicates subjects employ a HYBRID strategy: they do consult features in some validity-ordered fashion (as pi_1 proposes), but they do not stop at the first discriminator as strictly as TTB predicts. Instead, subjects appear to integrate information across multiple features with some weighting by validity, but with substantial influence from the overall feature consensus. Pi_1 systematically overestimates adherence to the top cue in Experiment 1 (predicts 0.82 vs. observed 0.38) and dramatically underestimates tally-consistency in Experiment 2 (predicts 0.14 vs. observed 0.72). Pi_2 better captures the central tendency in both experiments, predicting 0.32 vs. 0.38 in Experiment 1 and 0.88 vs. 0.72 in Experiment 2. However, pi_2's overestimation in Experiment 2 suggests it assumes too much deterministic feature-counting behavior. The core issue is that both theories are too extreme: TTB is too rigid in its stopping rule, and pure Tallying is too indifferent to validity structure. The observed data suggests a more nuanced decision process that respects validity differences but does not stop at the first discriminator.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory: 'Validity-Weighted Accumulation' or 'Soft Threshold Decision Rule.' This theory should: (1) Accumulate evidence across features weighted by their validities (not equal weights as in Tallying, not binary hierarchy as in TTB); (2) Use a soft decision threshold rather than a hard stopping rule\u2014features are consulted in descending validity order, but each feature contributes to a running accumulation score rather than triggering a deterministic stop; (3) The accumulation rule might be: for each feature in validity order, add its contribution (scaled by validity) to a running score for each option; (4) At each step, compute a soft threshold or decision boundary: if the accumulated evidence for one option exceeds a threshold (possibly relative to the remaining features' potential contribution), choose that option; otherwise, continue to the next feature; (5) Alternatively, accumulate all features weighted by validity and choose based on the weighted sum (similar to a linear model or weighted Tallying). This hybrid approach would explain why subjects in Experiment 1 do not rigidly follow the top cue (observed 0.38 vs. TTB's 0.82) while still respecting validity structure enough to outperform pure Tallying in cases where validity differences are large. It would also explain the intermediate performance in Experiment 2 (observed 0.72, between TTB's 0.14 and Tallying's 0.88): subjects integrate the high-validity feature's signal but do not ignore the consensus of lower-validity features entirely. The new theory should retain the softmax noise and lapse parameters from both existing theories but replace the core decision logic with a validity-weighted accumulation process."
}
```

## Usage

```json
{
  "input_tokens": 6690,
  "output_tokens": 890
}
```
