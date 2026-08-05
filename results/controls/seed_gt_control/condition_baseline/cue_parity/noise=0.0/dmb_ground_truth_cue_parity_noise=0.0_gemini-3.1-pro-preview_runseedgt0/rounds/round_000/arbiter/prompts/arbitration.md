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
People decide by the PARITY of the evidence rather than its weight or direction. They count the number of cues on which option A strictly beats option B, and prefer A when that count is ODD and B when it is EVEN (zero counts as even). This is a deliberately non-monotone, XOR-like rule: adding one more cue in A's favor flips the preference rather than strengthening it, so option dominance does NOT imply choice — an option that wins on every cue (an even count, when the cue number is even) is rejected. The rule uses no validities and no magnitudes, only the parity of feature-wise wins, which makes it an adversarially hard recovery target: it is uncorrelated with any single cue and with the validity-weighted sum, yet perfectly deterministic, much like the anti-majority ensemble. Response noise enters through a softmax over the binary score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
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
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 7: A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** To dissociate Cue-Parity from Tallying, we manipulate the number of cues on which option A strictly beats option B. Tallying predicts a monotonic preference for A as the number of A-wins increases relative to B-wins. Cue-Parity, in contrast, predicts a non-monotone, oscillating preference based solely on whether the number of A-wins is odd or even. For example, when A dominates B on 2 or 4 features, Tallying strongly predicts A, but Cue-Parity predicts B (because 2 and 4 are even). Conversely, when A wins on 1 feature and B wins on 2 or 3, Tallying predicts B, but Cue-Parity predicts A (because 1 is odd). We include these crucial dissociations as well as baseline trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
We isolate the trials where Cue-Parity and Tallying make strictly opposite predictions. On trials where A wins on 2 or 4 cues and B wins on 0 or 1 (mask_123), Cue-Parity predicts B (because 2 and 4 are even) while Tallying predicts A (because A has more wins). Conversely, on trials where A wins on 1 cue and B wins on 2 or 3 (mask_45), Cue-Parity predicts A (because 1 is odd) while Tallying predicts B. By taking the difference in the proportion of B choices between these two sets of trials, we get a metric that is approximately +1 for Cue-Parity and -1 for Tallying, providing maximal discriminability.

Source:
def metric(data: pd.DataFrame) -> float:
    a_wins = data.apply(lambda row: sum(a > b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    b_wins = data.apply(lambda row: sum(b > a for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    
    mask_123 = (a_wins.isin([2, 4])) & (b_wins.isin([0, 1]))
    mask_45 = (a_wins == 1) & (b_wins.isin([2, 3]))
    
    val_123 = data.loc[mask_123, 'response'].mean() if mask_123.any() else 0.5
    val_45 = data.loc[mask_45, 'response'].mean() if mask_45.any() else 0.5
    
    return float(val_123 - val_45)

### RESULTS
- Predicted under pi_1 (simulated): 0.7083 (var=0.0555)
- Predicted under pi_2 (simulated): -0.7050 (var=0.0284)
- Observed on real data: 0.7539 (var=0.0403)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 9: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Rationale:** To strongly dissociate Tallying from Cue-Parity, this design focuses on trials where option A and option B win on an EQUAL number of features (i.e., a_wins == b_wins). Tallying predicts complete indifference (50/50 guessing) whenever the tally is tied, because it merely compares the counts. Cue-Parity, however, makes completely deterministic predictions even when the counts are tied, because it only checks whether a_wins is odd or even. For example, if both options win on exactly 1 feature, Tallying predicts a tie, whereas Cue-Parity deterministically predicts A (since 1 is odd). If both options win on exactly 2 features, Tallying predicts a tie, whereas Cue-Parity deterministically predicts B (since 2 is even). We include these tied-tally trials alongside asymmetric trials (a_wins != b_wins) to ensure a robust quantitative dissociation across the full parameter space.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of responses that align with Cue-Parity's deterministic predictions specifically on trials where the tally is tied (i.e., option A and option B win on an equal number of features). On these trials, Tallying predicts complete indifference (50% chance for A or B), yielding an expected metric value of 0.5. Cue-Parity, however, makes deterministic choices based on the parity of the wins (predicting A if odd, B if even), yielding an expected metric value close to 1.0 (subject only to the epsilon lapse rate).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_wins = data.apply(lambda row: sum(1 for x, y in zip(row['option_a_ratings'], row['option_b_ratings']) if x > y), axis=1)
    b_wins = data.apply(lambda row: sum(1 for x, y in zip(row['option_a_ratings'], row['option_b_ratings']) if y > x), axis=1)
    
    mask = a_wins == b_wins
    if not mask.any():
        return 0.5
        
    tied_data = data[mask]
    tied_a_wins = a_wins[mask]
    
    # Cue-Parity predicts A (0) if a_wins is odd, B (1) if a_wins is even
    cp_preds = (tied_a_wins % 2 == 0).astype(int)
    
    matches = (tied_data['response'] == cp_preds).astype(float)
    return float(matches.mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.8596 (var=0.0092)
- Predicted under pi_2 (simulated): 0.5018 (var=0.0052)
- Observed on real data: 0.8178 (var=0.0117)

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
  "interpretation": "Across both experiments, Theory 1 (Cue-Parity) flawlessly captures the observed data, while Theory 2 (Tallying) completely fails. In Experiment 1, the observed metric is strongly positive (+0.75), aligning perfectly with Cue-Parity's non-monotone predictions and contradicting Tallying's monotonic predictions (-0.70). In Experiment 2, the observed metric on tied-tally trials is highly deterministic (+0.81), again matching Cue-Parity's predictions while Tallying predicted indifference (+0.50). Given that the data clearly follows a parity-based rule, Theory 2 is degenerate and needs to be completely replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory that builds on the success of the parity mechanism but explores a different nuance. Since Cue-Parity (unweighted) perfectly explains the current data, a strong competitor could be 'Validity-Weighted Parity', where the parity is calculated not just on the raw count of features, but is influenced by the validities of the cues, or perhaps a 'Lexicographic Parity' where parity is only computed on the top K most valid cues. This would maintain the core parity mechanism that fits the data but test if subjects are sensitive to cue validities."
}
```

## Usage

```json
{
  "prompt_token_count": 4480,
  "candidates_token_count": 301,
  "total_token_count": 5354
}
```
