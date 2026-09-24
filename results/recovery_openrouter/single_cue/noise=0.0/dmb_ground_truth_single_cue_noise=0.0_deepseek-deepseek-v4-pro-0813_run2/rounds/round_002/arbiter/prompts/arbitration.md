# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_4
People choose between two options through a stable, subject-level mixture of three comparison routes: (1) forward lexicographic scanning, inspecting cues in descending validity order and stopping at the first discriminating cue; (2) backward lexicographic scanning, inspecting cues in ascending validity order and stopping at the first discriminating cue in that reversed order; and (3) feature-win tallying, where each option receives one point per feature on which it strictly beats the other and ties contribute nothing. The forward and backward lexicographic weights jointly act as a subject-level polarity parameter for first-validity-cue effects: forward weight supports a positive first-discriminator effect, while backward weight supports a negative one. Each route's preferred option is passed through its own softmax choice rule, so responding is noisy but not controlled by a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Preferred validity vector comes from the experiment.  If the
    # parameter resolver ever leaves it unresolved, fall back to a
    # descending validity vector so the model is still scorable.
    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.ones(2, dtype=float) / 2.0

        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.9, 0.82, 0.74, 0.66, 0.58, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 4: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 8: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 13: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 14: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 15: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 16: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 17: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 18: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 19: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 20: A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 21: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 22: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 23: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 24: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 25: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 26: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 27: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 28: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 29: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 30: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 31: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 32: A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Rationale:** This design isolates a direction-of-lexicographic-scanning dissociation. All 32 unique pairs hold the feature-win tally difference at d = 0, so both theories' tallying routes are neutral and cannot drive the choice. On every trial the highest-validity cue (feature 0, validity 0.90) discriminates, and the lowest-validity cue (feature 5, validity 0.50) discriminates in the opposite direction. The advocated TTB/Tally mixture has only a descending lexicographic route, so its TTB component stops at feature 0; with q near 0.25 it predicts P(A) ≈ 0.62 when feature 0 favors A and ≈ 0.38 when feature 0 favors B. The competing model instead has a strong backward (ascending) lexicographic route (weight ≈ 0.42-0.54, beta ≈ 1.3-1.7) plus a weak forward route and neutral tallying, so on the same conflict trials it predicts the mirror pattern: P(A) ≈ 0.32-0.40 when feature 0 favors A/feature 5 favors B, and ≈ 0.60-0.68 when feature 0 favors B/feature 5 favors A. The middle four features are varied with equal A/B wins or ties to keep d = 0 while generating 32 unique pairs, 16 in each top-cue direction, repeated K = 3 for 96 trials.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
The metric measures the signed effect of the highest-validity cue: mean P(choose A) on trials where feature 0 favors A minus mean P(choose A) on trials where feature 0 favors B. All pairs in this design are feature-win tally neutral and feature 0 always discriminates, so tallying contributes 0.5 in both directions. The advocated forward TTB mixture therefore predicts a positive value near 0.25, while the competing backward-heavy mixture predicts a negative value near -0.35. Each subject has 48 trials in each top-cue direction, so this contrast has a large between-theory gap and only modest per-subject binomial variability.

Source:
def metric(data):
    import numpy as np
    a0 = np.array([int(r[0]) for r in data['option_a_ratings']])
    b0 = np.array([int(r[0]) for r in data['option_b_ratings']])
    choose_a = (np.asarray(data['response'], dtype=int) == 0).astype(float)
    top_a = a0 > b0
    top_b = ~top_a
    if not np.any(top_a) or not np.any(top_b):
        return 0.0
    return float(choose_a[top_a].mean() - choose_a[top_b].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.2613 (var=0.0080)
- Predicted under pi_4 (simulated): -0.2833 (var=0.0124)
- Observed on real data: -0.6583 (var=0.0598)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.9, 0.82, 0.74, 0.66, 0.58, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 7: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 8: A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 11: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 12: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 14: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 15: A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 16: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 17: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 18: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 19: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 20: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 21: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 22: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 24: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 25: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 26: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 27: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 28: A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 29: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 30: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 31: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 32: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Rationale:** All 32 unique pairs are top-vs-bottom conflict trials: feature 0 (validity 0.90) and feature 5 (validity 0.50) always discriminate in opposite directions. Therefore the descending lexicographic route (TTB) stops immediately at feature 0, while the ascending lexicographic route (backward) stops immediately at feature 5. The four middle cues are varied only to set the feature-win tally difference d = A_wins - B_wins. In the 16 top-A/low-B trials d ranges 0..+4; in the 16 top-B/low-A trials d ranges -3..0. The competing model (q*TTB + tally, no backward route) predicts a step-like curve: for top-A/low-B it gives P(A) near 0.62 at d=0 and near 1.0 for d>=1; for top-B/low-A it gives P(A) near 0.38 at d=0 and near 0.0 for d<=-1. The advocated model has a strong backward route (w_backward about 0.48, beta about 1.5) that cancels the top-cue signal: for top-A/low-B P(A) stays in a compressed band near 0.36-0.48 across d, and for top-B/low-A P(A) stays near 0.52-0.64. The key quantitative dissociations are therefore top-A/low-B d=+1 (competing about 1.0 vs advocated about 0.46) and top-B/low-A d=-1 (competing about 0.005 vs advocated about 0.54). This d-response pattern cannot be produced by any q or beta in the competing model because the competing model lacks the backward lexicographic route.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
This contrast uses the top-vs-bottom conflict trials where the first validity cue and the full tally point in the same direction. In the competing model, the TTB route and tallying both select the top-cued option on these trials, so choice of A is near 1 for top-A trials and near 0 for top-B trials, giving a pooled difference near +1. In the advocated model, the backward lexicographic route partially cancels the top cue: top-A trials stay near 0.47 and top-B trials stay near 0.53, so the same difference is near -0.06. The metric uses 36 top-A and 36 top-B trials per subject, making per-subject estimates reasonably tight.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    if len(data) == 0:
        return 0.0

    a = np.array([list(x) for x in data['option_a_ratings']], dtype=int)
    b = np.array([list(x) for x in data['option_b_ratings']], dtype=int)
    diff = a - b

    top_cue = diff[:, 0]
    tally_diff = (diff > 0).sum(axis=1) - (diff < 0).sum(axis=1)

    chose_a = (data['response'].to_numpy() == 0).astype(float)

    top_a_plus = (top_cue > 0) & (tally_diff >= 1)
    top_b_minus = (top_cue < 0) & (tally_diff <= -1)

    mean_top_a = np.mean(chose_a[top_a_plus]) if np.any(top_a_plus) else 0.0
    mean_top_b = np.mean(chose_a[top_b_minus]) if np.any(top_b_minus) else 0.0

    return float(mean_top_a - mean_top_b)

### RESULTS
- Predicted under pi_3 (simulated): 0.9950 (var=0.0001)
- Predicted under pi_4 (simulated): -0.0722 (var=0.0187)
- Observed on real data: -0.6711 (var=0.0489)

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
**Predicted under pi_4:** 0.2306 (var=0.0024)

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
**Predicted under pi_4:** 0.4335 (var=0.0093)

### Experiment 5
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    a1 = data["option_a_ratings"].apply(lambda x: x[1])
    b1 = data["option_b_ratings"].apply(lambda x: x[1])
    f1_a = a1 > b1
    f1_b = b1 > a1
    chose_a = data["response"].astype(int) == 0
    if f1_a.sum() == 0 or f1_b.sum() == 0:
        return 0.0
    return float(chose_a[f1_a].mean() - chose_a[f1_b].mean())
```

**Observed (real) value:** -0.1300 (var=0.0159)
**Predicted under pi_3:** 0.2375 (var=0.0045)
**Predicted under pi_4:** -0.1325 (var=0.0113)

### Experiment 6
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
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

```

**Observed (real) value:** -0.2742 (var=0.0069)
**Predicted under pi_3:** 0.2375 (var=0.0114)
**Predicted under pi_4:** -0.2879 (var=0.0081)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, pi_4 is clearly the stronger theory. It gets the direction of the reversal effects correct: Experiment 1 observed -0.6583 and pi_4 predicted -0.2833; Experiment 2 observed -0.6711 and pi_4 predicted -0.0722; Experiment 5 observed -0.1300 and pi_4 predicted -0.1325; Experiment 6 observed -0.2742 and pi_4 predicted -0.2879; Experiment 4 observed 0.4196 and pi_4 predicted 0.4335. In contrast, pi_3 gets the sign wrong on four decisive probes: +0.2613 vs -0.6583 in Experiment 1, +0.9950 vs -0.6711 in Experiment 2, +0.2375 vs -0.1300 in Experiment 5, and +0.2375 vs -0.2742 in Experiment 6. pi_3 is essentially a forward TTB plus tally model and cannot accommodate the persistent negative first-cue effects. However pi_4 is not yet a fully task-invariant account. Its unconditional, stable mixture assigns the backward lexicographic route only moderate weight, so it substantially underestimates the extreme reversal in Experiments 1 and 2. Simply increasing the backward weight enough to fit those two experiments would likely make the Experiment 5 effect much too negative. The observed pattern therefore suggests a stimulus-contingent route selection mechanism rather than a fixed trial-independent mixture, so a new theory is justified.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_3 with a conflict-gated reverse-lexicographic and tallying theory. The new theory should retain pi_4's backward lexicographic route and tallying route, but make route activation depend on the trial's cue structure instead of using one fixed mixture across all trials. Specifically: (1) detect extreme-cue conflict, defined as the highest-validity cue and the lowest-validity cue discriminating in opposite directions while the absolute tally difference abs(d) is small, roughly 0 or 1; (2) on such conflict trials use the backward lexicographic route with high probability, with subject-level reversal weight around 0.75 to 0.95 and a strong beta around 3 to 6 so the top-cue split approaches the observed -0.66 rather than -0.28; (3) on non-conflict trials use a mostly tallying route with only weak forward lexicographic weight, preserving the large tally advantage in Experiment 4, the low forward-TTB accuracy in Experiment 3, and the moderate negative second-cue effects in Experiments 5 and 6; (4) allow substantial between-subject variance in the reversal weight and beta to match the high observed variances around 0.049 to 0.060 in Experiments 1 and 2. This replaces pi_3's fixed q*TTB plus tally mixture with a backward route plus a conflict trigger, and differs from pi_4 by making strong backward scanning conditional on extreme-cue conflict rather than a constant moderate mixture weight."
}
```

## Usage

```json
{
  "input_tokens": 14487,
  "output_tokens": 14094,
  "reasoning_tokens": 13414,
  "provider": "Baidu"
}
```
