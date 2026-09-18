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
**Validities (n_features=8):** [0.95, 0.86, 0.79, 0.72, 0.66, 0.61, 0.56, 0.51]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  trial 7: A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Rationale:** Eight experts have strictly ordered, widely separated validities. Every pair creates a conflict between Take The Best and Tallying: the highest-validity discriminating expert favors one option, while the majority of discriminating experts favors the other. The design orthogonally balances whether A or B is favored by the decisive expert and manipulates the opposing tally margin. In trials where Expert 1 discriminates, the tally margins are 2, 4, and 6; where Expert 1 ties and Expert 2 is the first discriminating expert, the margins are 1, 3, and 5. Take The Best therefore predicts the same choice probability for the option favored by the first discriminating expert on every trial: epsilon/2 + (1-epsilon)*sigmoid(beta), invariant to both cue rank and tally margin. Tallying predicts that this option is chosen below chance, with probability epsilon/2 + (1-epsilon)*sigmoid(-beta*m), and predicts a systematic decrease as the opposing margin m increases. Thus, the critical dissociation is both directional and quantitative: invariant support for the top-cue winner under TTB versus increasingly strong rejection of it under Tallying. Swapping A and B for each evidence pattern controls response-side preferences. The 12 unique pairs yield exactly 96 trials with eight repetitions per pair.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
For each subject, the metric asks whether the option favored by the highest-validity discriminating expert was selected on more than half of trials, assigning 0.5 for an exact tie. It then reports the prevalence of subjects showing that directional pattern. TTB predicts above-chance agreement because the first discriminating cue determines every choice, whereas tallying predicts below-chance agreement because every designed trial has the opposing option winning the cue tally. Thresholding at the subject level exploits this sign reversal while reducing sensitivity to individual beta and epsilon values and to binomial trial noise. The A/B-swapped pairs ensure that the result cannot be generated merely by a response-side preference.

Source:
def metric(data: pd.DataFrame) -> float:
    """Prevalence of subjects choosing the first-discriminating-cue winner above chance."""
    if data is None or len(data) == 0:
        return float("nan")

    def subject_score(subj):
        agreements = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            diff = a - b
            discriminating = np.flatnonzero(diff != 0)
            if discriminating.size == 0:
                continue
            first = int(discriminating[0])
            ttb_response = 0 if diff[first] > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

        if len(agreements) == 0:
            return float("nan")
        rate = float(np.mean(agreements))
        if rate > 0.5:
            return 1.0
        if rate < 0.5:
            return 0.0
        return 0.5

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        score = subject_score(subj)
        if np.isfinite(score):
            scores.append(score)

    return float(np.mean(scores)) if len(scores) else float("nan")

### RESULTS
- Predicted under pi_1 (simulated): 0.9900 (var=0.0049)
- Predicted under pi_2 (simulated): 0.0000 (var=0.0000)
- Observed on real data: 1.0000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=8):** [0.94, 0.87, 0.8, 0.73, 0.66, 0.6, 0.55, 0.51]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  trial 7: A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  trial 10: A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  trial 11: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 1]
  trial 12: A=[1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]

**Rationale:** This design targets the parameter-free prediction of Tallying on exactly balanced evidence. On every trial, A and B win on the same number of discriminating features, so Tallying predicts P(A)=P(B)=0.5 for every beta and epsilon. Nevertheless, each pair has a uniquely defined first discriminating expert in descending-validity order. Take The Best therefore predicts choice of that expert's favored option with probability epsilon/2 + (1-epsilon)*sigmoid(beta), which is strictly above chance throughout the stated parameter ranges. The six underlying evidence patterns vary both the rank of the first discriminating expert (1, 3, or 5) and the total number of discriminating experts (2, 4, 6, or 8), while preserving an exactly tied tally. Take The Best predicts the same above-chance preference across these manipulations, whereas Tallying remains exactly at chance. Each pattern is paired with an A/B reversal, controlling response-side bias. Thus the critical statistic is the probability of choosing the option favored by the first discriminating expert: exactly 0.5 under Tallying versus a common value above 0.5 under Take The Best. The 12 unique pairs are repeated eight times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is first-discriminating-cue choice accuracy. Every designed pair has an exactly tied tally, so Tallying predicts a tightly estimated value of 0.5. Take The Best instead favors the option identified by the first differing feature and predicts substantially greater than 0.5 alignment except at its highest noise levels. Pooling all 96 trials keeps the per-subject sampling variance low, while the A/B-reversed pairs balance the favored response side and therefore prevent a generic A/B response bias from creating the contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices consistent with the highest-validity discriminating cue."""
    if len(data) == 0:
        return float("nan")

    aligned = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        favored = None

        # Features are already ordered from highest to lowest validity.
        for j in range(min(len(a), len(b))):
            if a[j] > b[j]:
                favored = 0
                break
            if b[j] > a[j]:
                favored = 1
                break

        # Omit completely nondiscriminating pairs, although none occur in
        # the specified design.
        if favored is not None:
            aligned.append(float(int(response) == favored))

    if not aligned:
        return float("nan")
    return float(np.mean(aligned))

### RESULTS
- Predicted under pi_1 (simulated): 0.8492 (var=0.0121)
- Predicted under pi_2 (simulated): 0.4848 (var=0.0022)
- Observed on real data: 0.3242 (var=0.0174)

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
  "interpretation": "Across both experiments, pi_1 is the stronger theory, but neither theory is task-invariant. Experiment 1 is decisive: the observed prevalence of subjects favoring the highest-validity discriminating cue is 1.0000, essentially matching pi_1's 0.9900 prediction and maximally contradicting pi_2's 0.0000 prediction. The zero observed between-subject variance further indicates that this directional result is highly consistent. Experiment 2, however, qualitatively rejects both theories. The observed first-cue alignment is 0.3242: not merely lower than pi_1's 0.8492, but reliably below chance. Pi_2 is numerically closer at 0.4848, yet its exactly balanced tallies impose a structural prediction of 0.5 for every beta and epsilon; no regeneration of ordinary tallying parameters can produce systematic below-chance alignment. The elevated observed variance in Experiment 2 (0.0174 versus 0.0022 predicted by pi_2) also suggests meaningful heterogeneity in attention or tie resolution. Thus pi_1 captures the first experiment extremely well but fails the second, while pi_2 fails the first maximally and cannot explain the direction of the second. Pi_1 should remain as the current benchmark, whereas pi_2 is the appropriate theory to replace.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace equal-weight Tallying with a conflict-contingent attentional evidence theory. The central claim should be that cue use depends jointly on validity rank, redundancy, and the configuration of conflict rather than on a fixed tally or a fixed first-cue cascade. One promising account is bounded, redundancy-discounted integration with serial-position-dependent attention: early/high-validity evidence receives strong initial weight, repeated same-direction endorsements have diminishing marginal influence, and the final discriminating evidence receives a recency boost when the evidence counts are balanced. This can favor the compact high-validity minority in Experiment 1 while allowing the lower-ranked, oppositely directed endpoint evidence to dominate tie resolution in Experiment 2, producing below-chance first-cue alignment. The theory should also permit stable subject differences in primacy, recency, and redundancy discounting to address Experiment 2's larger variance. It must make trial-level predictions\u2014not merely fit the two pooled metrics\u2014and should be tested with new trials that independently cross tally imbalance, first-versus-last cue agreement, number of redundant cues, and presentation order. Reversing cue display order is especially diagnostic: a genuine validity mechanism should follow validity, whereas a recency-based tie-resolution mechanism should follow presentation order. This replacement differs from pi_1 by integrating multiple cues and allowing conflict-dependent attention, and from pi_2 by abandoning equal, context-invariant cue contributions."
}
```

## Usage

```json
{
  "input_tokens": 5790,
  "output_tokens": 2958
}
```
