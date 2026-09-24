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
**Validities (n_features=6):** [0.92, 0.84, 0.76, 0.68, 0.6, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 9: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 11: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]

**Rationale:** This design uses six binary cues with widely spread validities. It places TTB's first discriminating cue at position 0, 1, or 2 in the validity-sorted order, while packing the later cues to oppose or tie the tallying score. In half the unique pairs the first discriminating cue favors A and in half it favors B. TTB therefore predicts p(A) is uniformly high for all six TTB-A trials and uniformly low for all six TTB-B trials, because later cues are never consulted after the first discriminating cue. Tallying instead ignores validity and counts all features: in the six oppose pairs the tally winner is opposite to TTB (count differences of 1 vs 5, 1 vs 4, and 1 vs 3), producing a crossover; in the six tie pairs tallying predicts 0.5 regardless of beta. The 12 unique pairs are each repeated 8 times for roughly 96 trials per subject. This gives a quantitative dissociation: TTB choice probability should be invariant within first-cue winner type, whereas tallying choice probability should move from near 0 to 0.5 to near 1 across the tally-difference levels.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
On each trial I first find the validity-ordered cue that discriminates between A and B. I sign that top discriminating cue +1 when it favors A and -1 when it favors B, and I also compute the tally margin as the number of A-winning features minus the number of B-winning features. The metric is a weighted contrast of A choices: each trial is weighted by the top-cue sign times (abs(tally margin) + 1). The +1 keeps the zero-margin trials in the contrast because TTB still treats the top discriminating cue as decisive there, but assigns them lower weight because an indifferent tally cannot separate the theories. Under TTB the expected weighted contrast is positive and invariant to lower-cue conflict; under Tallying the same contrast is negative because choices follow the feature-count margin, with the largest tally-conflict trials weighted most heavily. Because every unique pair repeats eight times within each subject, this averages many binary responses and holds per-subject variance relatively low.

Source:
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.92, 0.84, 0.76, 0.68, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind='stable')
    A = np.array([np.asarray(row, dtype=float) for row in data['option_a_ratings']])
    B = np.array([np.asarray(row, dtype=float) for row in data['option_b_ratings']])
    resp_is_a = (data['response'].to_numpy(dtype=float) == 0.0).astype(float)

    winner = np.full(A.shape[0], -1, dtype=int)
    unresolved = np.ones(A.shape[0], dtype=bool)
    for j in cue_order:
        diff = A[:, j] - B[:, j]
        first = unresolved & (diff != 0)
        winner[first] = np.where(diff[first] > 0, 0, 1)
        unresolved[first] = False

    tally = (A > B).sum(axis=1) - (B > A).sum(axis=1)
    side = np.where(winner == 0, 1.0, np.where(winner == 1, -1.0, 0.0))
    signed_weight = side * (np.abs(tally) + 1.0)

    numerator = float(np.sum(signed_weight * resp_is_a))
    denom = float(np.sum(signed_weight[signed_weight > 0]))
    if denom == 0.0:
        return 0.0
    return numerator / denom


### RESULTS
- Predicted under pi_1 (simulated): 0.7347 (var=0.0373)
- Predicted under pi_2 (simulated): -0.6093 (var=0.0194)
- Observed on real data: -0.6137 (var=0.0420)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=8):** [0.57, 0.7, 0.5, 0.77, 0.92, 0.84, 0.63, 0.52]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  trial 8: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  trial 12: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 13: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 14: A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  trial 15: A=[1, 1, 1, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1]
  trial 16: A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]

**Rationale:** Eight experts with validities spread from .50 to .92; the highest validity is placed at display index 4 so TTB must re-sort and cannot be mimicked by left-to-right scanning. In every unique pair the top cue discriminates: trials 1-8 have top cue A, trials 9-16 top cue B. TTB therefore predicts a constant p(A) within each block (sigmoid(beta) for A-top, sigmoid(-beta) for B-top, plus lapse), because lower cues are never consulted. Tallying instead counts all eight cues, so p(A) = sigmoid(beta*d) with d = #A wins - #B wins. The d sequence is +8,+6,+4,+2,0,-2,-4,-6 for A-top and -8,-6,-4,-2,0,+2,+4,+6 for B-top. This yields a full psychometric curve for Tallying within each TTB-constant block, including d=0 pairs where Tallying must guess but TTB remains confident, and crossover pairs where the top cue and the majority of cues point in opposite directions. Mirrored blocks control for A/B response bias. 16 unique pairs x 6 repeats = 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric estimates the within-subject linear sensitivity of choosing the top cue's favored option to how strongly the full tally of all features agrees with that top cue. For each trial, it takes the highest-validity cue at display index 4, codes whether the response followed that cue, and constructs a signed tally advantage for that same option (positive when the majority of all features also favor the top cue's option, negative when they oppose it). Trials with a tally tie are excluded because they do not specify a tallying direction. The returned OLS slope pools evidence across all informative trials. Take The Best predicts this slope is zero: whenever the top cue discriminates, the top-cue choice probability is constant regardless of how lower features split. Tallying predicts a positive slope: as the signed tally advantage for the top option increases, the probability of choosing that option rises monotonically. Using a regression coefficient across all informative trials rather than a small subset contrast keeps between-subject variance low while preserving a large theory separation.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    resp = data['response'].astype(int).tolist()

    # Highest validity is at display index 4: [0.57, 0.7, 0.5, 0.77, 0.92, 0.84, 0.63, 0.52]
    top_idx = 4

    xs = []
    ys = []

    for a, b, r in zip(a_col, b_col, resp):
        if a[top_idx] == b[top_idx]:
            continue

        top_is_a = a[top_idx] > b[top_idx]

        a_wins = 0
        b_wins = 0
        for fa, fb in zip(a, b):
            if fa > fb:
                a_wins += 1
            elif fb > fa:
                b_wins += 1

        d = a_wins - b_wins
        if d == 0:
            continue

        # Signed tally advantage for the option favored by the top cue.
        signed_tally_for_top = d if top_is_a else -d

        # 1 if the subject chose the option favored by the top cue, else 0.
        top_chosen = 1.0 if ((top_is_a and r == 0) or ((not top_is_a) and r == 1)) else 0.0

        xs.append(float(signed_tally_for_top))
        ys.append(top_chosen)

    if len(xs) < 2:
        return 0.0

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    xm = x.mean()
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0:
        return 0.0

    slope = float(np.sum((x - xm) * (y - y.mean())) / sxx)
    return slope

### RESULTS
- Predicted under pi_1 (simulated): -0.0002 (var=0.0000)
- Predicted under pi_2 (simulated): 0.0679 (var=0.0003)
- Observed on real data: 0.0659 (var=0.0002)

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
  "interpretation": "pi_2 (Tallying) is clearly the task-invariant theory across the two experiments. In Experiment 1 the observed weighted contrast was -0.6137, almost exactly matching Tallying's predicted -0.6093, while TTB predicted +0.7347. TTB fails even directionally because its one-reason stopping rule follows the top discriminating cue even when the remaining cues strongly oppose it. Experiment 2 gives the same picture: the observed signed-tally slope was 0.0659, matching Tallying's predicted 0.0679, while TTB predicted a flat -0.0002. Those failures are structural: adjusting beta, epsilon, or lapse within TTB cannot make lower-cue majorities drive choice in opposition trials or produce a positive relation between choice and the tallying evidence. The problem is TTB's core stopping-rule claim, not its parameterization, so pi_1 should be replaced entirely.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 with a validity-weighted compensatory integration theory. The new theory should hold that people inspect all features and build a signed advantage score by summing the contributions of every discriminating feature, but each feature contributes according to its perceived validity rather than all features necessarily being equally weighted. Features that tie contribute nothing. Choice then follows a softmax over the weighted advantage plus a lapse term. The weight vector should have a range or prior centered near equal weights because both observed experiments sit close to plain Tallying: the five lower-validity cues must be able to outweigh the single high-validity cue in Experiment 1, and signed tally evidence must drive choice in Experiment 2. The new theory differs from pi_1 by abandoning one-reason stopping and differs from pi_2 by allowing validity-graded cue weights, so it is a substantive competitor rather than a reparameterization."
}
```

## Usage

```json
{
  "input_tokens": 6628,
  "output_tokens": 5831,
  "reasoning_tokens": 5419,
  "provider": "Baidu"
}
```
