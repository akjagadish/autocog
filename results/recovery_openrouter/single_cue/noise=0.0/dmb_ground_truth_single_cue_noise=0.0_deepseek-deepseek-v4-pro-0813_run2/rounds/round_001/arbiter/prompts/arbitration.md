# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
On each choice, a decision maker probabilistically selects between two comparison strategies. With probability q, centered near 0.25, the choice is made by a one-reason stopping rule: cues are inspected in descending validity order and the first discriminating cue determines the preferred option, with no further cues consulted. With probability 1 - q, the choice is made by full feature-win tallying: each option receives one point for every feature on which it strictly beats the other option, ties contribute nothing, and the option with the larger tally is preferred. Each strategy's preference is passed through its own softmax choice rule, giving nearly deterministic but not perfectly errorless responding, and there is no additional uniform lapse. The mixture probability q is a stable subject-level parameter applied independently on every trial, so the model recovers pure Take The Best when q = 1, pure Tallying when q = 0, and an intermediate strategy mixture when q is near 0.25.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    def _softmax(scores, beta):
        z = beta * (scores - np.max(scores))
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    n_features = stim.shape[1]
    a = stim[0]
    b = stim[1]

    validities = np.asarray(parameters['validities'], dtype=float)
    if validities.shape[0] == n_features:
        cue_order = np.argsort(-validities, kind='stable').tolist()
    else:
        cue_order = list(range(n_features))

    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        p_ttb = np.ones(2) / 2.0
    else:
        ttb_scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        p_ttb = _softmax(ttb_scores, float(parameters['beta_ttb']))

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    p_tally = _softmax(
        np.array([a_wins, b_wins]),
        float(parameters['beta_tally'])
    )

    q = float(parameters['q'])
    p = q * p_ttb + (1.0 - q) * p_tally

    epsilon = float(parameters['epsilon'])
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2) / 2.0

    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.9, 0.8, 0.7, 0.6, 0.55, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 13: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 14: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 15: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 17: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 21: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 23: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 24: A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 25: A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 26: A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 27: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 28: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 29: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 30: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 31: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 32: A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]

**Rationale:** Six features with a spread of validities. Feature 0 (validity 0.90) is tied on every trial, so Take The Best skips it and stops at feature 1 (validity 0.80), which always discriminates. The four lower features are arranged to set the tally difference d = A_wins - B_wins to -2, -1, 0, +1, or +2. Critically, for each d there are trials where feature 1 favors A and trials where feature 1 favors B. Pure Tallying depends only on d and therefore must predict identical choice probabilities for the two feature-1 directions within each d. The TTB/Tallying mixture instead predicts a within-d split of about q≈0.25: e.g. at d=0, pure Tallying predicts 0.5 for all trials, while the mixture predicts ≈0.62 when feature 1 favors A and ≈0.38 when it favors B; at d=+1, pure Tallying predicts near 1.0, while the mixture predicts ≈1.0 vs ≈0.75. This within-d split cannot be produced by any beta or epsilon in pure Tallying. The 32 unique pairs are repeated K=3 for 96 total trials.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
This is the observed within-tally-difference split by the first discriminating TTB cue. Feature 0 is tied on every trial, so TTB drops to feature 1 (validity 0.80), which always discriminates. I classify every trial by whether feature 1 favors A or B, and subtract the proportion of A choices in the feature-1-favors-B class from the proportion in the feature-1-favors-A class. The trial set has equal feature-1 direction counts within each tally difference d = A_wins - B_wins, so pure Tallying predicts the same A-choice probability in the two classes and the metric has expectation 0. The advocated TTB/Tallying mixture instead shifts the favored class by q times the near-deterministic TTB preference gap; with q near 0.25 and beta_ttb near 5.25 the expected metric is approximately 0.247. Each subject contributes 48 trials per class, keeping between-subject noise low.

Source:
def metric(data):
    a1 = data["option_a_ratings"].apply(lambda x: x[1])
    b1 = data["option_b_ratings"].apply(lambda x: x[1])
    f1_a = a1 > b1
    f1_b = b1 > a1
    chose_a = data["response"].astype(int) == 0
    if f1_a.sum() == 0 or f1_b.sum() == 0:
        return 0.0
    return float(chose_a[f1_a].mean() - chose_a[f1_b].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.2375 (var=0.0045)
- Predicted under pi_2 (simulated): -0.0158 (var=0.0048)
- Observed on real data: -0.1300 (var=0.0159)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=8):** [0.92, 0.88, 0.8, 0.72, 0.65, 0.6, 0.55, 0.52]

**Trial pairs (n=32):**
  trial 1: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  trial 6: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  trial 8: A=[1, 1, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 1]
  trial 9: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 15: A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 16: A=[1, 1, 0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0]
  trial 17: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 18: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  trial 19: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 20: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 21: A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  trial 22: A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  trial 23: A=[1, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  trial 24: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  trial 25: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 26: A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 27: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 28: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 29: A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  trial 30: A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  trial 31: A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1]
  trial 32: A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Rationale:** This design isolates a signature of pure Tallying that the TTB/Tallying mixture cannot reproduce. Pure Tallying depends only on the tally difference d = A_wins - B_wins. When d = 0, its softmax over equal tallies is exactly uniform and the lapse term also preserves uniformity, so it must predict P(A) = 0.5 for every such pair, regardless of which individual cue discriminates. The mixture model instead adds a TTB component (q ≈ 0.25) that stops at the first discriminating cue in descending-validity order. By tying the top two or top three cues on every trial, the first discriminator is pushed to feature 2 or feature 3. All 32 unique pairs are constructed to have d = 0, so the Tallying component of the mixture also gives 0.5, but the TTB component shifts choice probabilities to about 0.62 when the first discriminator favors A and about 0.38 when it favors B (using q ≈ 0.25 and beta_ttb ≈ 5.25). The critical contrast is therefore first-discriminator direction: pure Tallying predicts 0.5 vs 0.5 (difference 0), while the mixture predicts approximately 0.62 vs 0.38 (difference about 0.25). The design balances first-discriminator direction, top-tie value (0/0 vs 1/1), and discriminator position (feature 2 vs feature 3) across the 32 unique pairs, which are repeated K = 3 times for 96 trials.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
All 32 unique pairs in this design have equal feature-win tallies for A and B, so pure Tallying predicts an A-choice probability of 0.5 on every trial. The TTB mixture violates this only when the first validity-ordered discriminating feature favors one option. The metric classifies each trial by that first discriminator using the validity order 0.92, 0.88, 0.80, 0.72, 0.65, 0.60, 0.55, 0.52, then returns the difference in A-choice rate between first-discriminator-favors-A and first-discriminator-favors-B trials. Pure Tallying gives approximately 0.0, while the mixture gives approximately 0.62 minus 0.38, about 0.25, providing a large and stable contrast.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.92, 0.88, 0.80, 0.72, 0.65, 0.60, 0.55, 0.52])
    order = np.argsort(-validities)
    pos = []
    neg = []
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        first = 'A'
        found = False
        for j in order:
            if a[j] > b[j]:
                first = 'A'
                found = True
                break
            if b[j] > a[j]:
                first = 'B'
                found = True
                break
        if not found:
            continue
        chose_a = 1.0 if row.response == 0 else 0.0
        if first == 'A':
            pos.append(chose_a)
        else:
            neg.append(chose_a)
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(np.mean(pos) - np.mean(neg))


### RESULTS
- Predicted under pi_3 (simulated): 0.2375 (var=0.0114)
- Predicted under pi_2 (simulated): 0.0075 (var=0.0109)
- Observed on real data: -0.2742 (var=0.0069)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = None
        # Cues are in descending-validity order, so the first discriminating
        # feature is the Take The Best cue.
        for j in range(len(a)):
            if a[j] > b[j]:
                pred = 0
                break
            if b[j] > a[j]:
                pred = 1
                break
        if pred is not None:
            matches.append(1.0 if r == pred else 0.0)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2487 (var=0.0065)
**Predicted under pi_3:** 0.2483 (var=0.0023)
**Predicted under pi_2:** 0.1175 (var=0.0068)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        response_a = 1.0 if int(row["response"]) == 0 else 0.0

        # Take The Best style second-cue prediction.
        # By design, feature 0 is tied on every trial, so feature 1 is
        # the first discriminating cue.
        if a[1] > b[1]:
            second_wins_a = True
        elif b[1] > a[1]:
            second_wins_a = False
        else:
            second_wins_a = True

        second_score = response_a if second_wins_a else (1.0 - response_a)

        # Tallying total score. Feature-specific ties contribute zero.
        wins_a = float(np.sum(a > b))
        wins_b = float(np.sum(b > a))
        tally_d = wins_a - wins_b

        if tally_d > 0.0:
            tally_score = response_a
        elif tally_d < 0.0:
            tally_score = 1.0 - response_a
        else:
            # Tallying has no preference on a tied tally.
            tally_score = 0.5

        scores.append(tally_score - second_score)

    return float(np.mean(scores)) if scores else 0.0
```

**Observed (real) value:** 0.4196 (var=0.0128)
**Predicted under pi_3:** 0.4169 (var=0.0101)
**Predicted under pi_2:** 0.6304 (var=0.0462)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (pi_3) is the stronger account overall, but neither theory is task-invariant. Theory 1 matches Experiments 3 and 4 almost exactly: its predicted values are 0.2483 and 0.4169 against observed 0.2487 and 0.4196. This confirms that some subjects use a high-validity-first Take The Best route mixed with feature-win tallying. However, in Experiments 1 and 2 Theory 1 predicts a positive first-discriminator-direction effect of about 0.2375, while the observed effects are negative: -0.1300 and -0.2742. That sign reversal cannot be produced by changing beta, epsilon, or q, because the TTB route always scans validities from high to low and therefore always pushes the metric positive. Theory 2 (pi_2) predicts approximately zero on both Experiments 1 and 2, which is closer in sign but still misses the substantial negative observed effects, especially -0.2742 in Experiment 2. Theory 2 also badly mismatches Experiments 3 and 4: it predicts 0.1175 versus observed 0.2487, and 0.6304 versus observed 0.4196. Its core claim, that only the tally difference A_wins - B_wins matters and cue validity order is irrelevant, is directly contradicted by the large first-discriminator-direction effects in Experiments 1 and 2. Thus pi_2 is the degenerate account here, while pi_3 is a strong but incomplete competitor because it lacks a mechanism for reversed or low-validity-first cue scanning. The negative effects in Experiments 1 and 2 are diagnostic: they suggest that a portion of choices follows the last discriminating cue, or a reversed validity order, instead of the first discriminating cue. In those designs the option favored by the highest-validity first cue tends to be disfavored by the last discriminating cue, so a backward lexicographic component would generate exactly the observed negative contrasts. A new theory should therefore mix feature-win tallying with frontward and backward lexicographic scanning in a task-invariant way.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory to replace pi_2: a Tallying plus bi-directional Take The Best mixture. On each trial, with stable subject-level mixture probabilities, the decision maker uses one of three routes: (1) forward lexicographic scanning in descending validity order, (2) backward lexicographic scanning in ascending validity order, stopping at the first cue that discriminates in that reversed order, and (3) feature-win tallying, where ties contribute nothing. Each selected route produces a preference that is passed through a softmax choice rule, with an optional uniform lapse. The crucial new feature relative to pi_2 is that cue order is no longer irrelevant, so the model can produce non-zero first-discriminator-direction effects. The crucial new feature relative to pi_3 is that the lexicographic route has a polarity parameter, allowing the first-validity-cue effect to be positive or negative across subjects or trials. Parameter ranges should include a forward TTB weight near 0.2 to 0.3, because Experiments 3 and 4 support roughly a 25 percent high-to-low TTB contribution, a backward TTB weight able to reach roughly 0.2 to 0.4 to explain the negative contrasts in Experiments 1 and 2, and a residual tallying weight. The next model should treat the forward and backward route probabilities as identifiable subject-level parameters rather than forcing one fixed direction, and should be tested on experiments that orthogonalize forward-cue direction, backward-cue direction, and tally difference."
}
```

## Usage

```json
{
  "input_tokens": 10987,
  "output_tokens": 22049,
  "reasoning_tokens": 21251,
  "provider": "Baidu"
}
```
