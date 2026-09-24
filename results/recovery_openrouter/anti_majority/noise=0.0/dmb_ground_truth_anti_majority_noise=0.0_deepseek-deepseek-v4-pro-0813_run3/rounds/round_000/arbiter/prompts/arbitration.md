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
**Validities (n_features=6):** [0.9, 0.8, 0.72, 0.65, 0.62, 0.78]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 10: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 11: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 12: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** Six features are used. The highest-validity expert (0.90) is always the first cue TTB tests. I hold that cue's discriminating direction constant across blocks and vary the lower five cues so Tallying's net evidence changes while TTB sees the same first discriminating cue. In the top-A block the top cue is 1 vs 0 and lower-cue net wins take values +5, +3, +1, -1, -3, -5; in the top-B block the top cue is reversed. TTB predicts identical p(A) for all six top-A pairs and identical p(A) for all six top-B pairs because the first discriminating cue is unchanged; Tallying predicts a graded logistic change because its tally advantage is 1 + lower_net for top-A pairs and lower_net - 1 for top-B pairs. At lower_net = -1 in the top-A block and lower_net = +1 in the top-B block, Tallying is exactly tied while TTB commits to A or B. At lower_net = -3/-5 in the top-A block and +3/+5 in the top-B block, the two theories make opposite directional predictions. The critical signature is flat TTB choice probabilities across the six lower-net levels versus a monotone logistic gradient for Tallying. Twelve unique pairs repeated K = 8 gives 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Response is recoded so y=1 means chose A and y=0 means chose B. The highest-validity cue is feature 0. For trials where feature 0 favors A, TTB says p(A) is the same no matter what the five lower-validity cues say; similarly, for trials where feature 0 favors B, TTB says p(A) is the same across lower-cue configurations. The metric therefore takes, within each top-cue block, the proportion of A choices on trials where the lower five cues favor A on net minus the proportion on trials where they favor B on net. TTB's expected value is 0 in both blocks, up to binomial noise. Tallying counts every cue, so it predicts more A choices when lower cues favor A and fewer when lower cues favor B; hence both differences are positive, and their average is near +1 when choices are decisive. Averaging the two top-cue blocks controls for TTB's fixed bias toward A or B and yields a scalar with small per-subject variance under TTB and a large positive value under Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    A = np.array(data['option_a_ratings'].tolist(), dtype=int)
    B = np.array(data['option_b_ratings'].tolist(), dtype=int)
    y = 1.0 - data['response'].to_numpy(dtype=float)

    top_a = A[:, 0] > B[:, 0]
    top_b = A[:, 0] < B[:, 0]

    lower_a_wins = (A[:, 1:] > B[:, 1:]).sum(axis=1)
    lower_b_wins = (B[:, 1:] > A[:, 1:]).sum(axis=1)
    lower_net = lower_a_wins - lower_b_wins

    high = lower_net >= 1
    low = lower_net <= -1

    def diff(top_mask):
        hi = y[top_mask & high]
        lo = y[top_mask & low]
        if len(hi) == 0 or len(lo) == 0:
            return 0.0
        return float(hi.mean() - lo.mean())

    return (diff(top_a) + diff(top_b)) / 2.0

### RESULTS
- Predicted under pi_1 (simulated): 0.0092 (var=0.0043)
- Predicted under pi_2 (simulated): 0.6258 (var=0.0234)
- Observed on real data: -0.5117 (var=0.0161)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.92, 0.84, 0.76, 0.68, 0.6, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 7: A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 10: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 12: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Rationale:** Twelve unique pairs form a 4 × 3 factorial. Six features have validities [0.92, 0.84, 0.76, 0.68, 0.60, 0.55], so TTB's cue cascade is feature 0 → 1 → 2 → 3 → 4 → 5. The first factor is the direction of the first discriminating cue: feature 0 favors A; feature 0 favors B; feature 0 ties and feature 1 favors A; feature 0 ties and feature 1 favors B. The second factor is the tally difference (+2, 0, -2), implemented by the remaining lower features. Tallying ignores which features discriminate, so it predicts three choice-probability levels that depend only on tally difference and are identical across the four first-cue conditions. TTB ignores all lower features once a discriminating cue is found, so it predicts only two choice-probability levels (high for first-cue A, low for first-cue B) and no effect of tally difference. This yields opposite directional predictions on conflict trials: top A with tally -2 (pair 3), top B with tally +2 (pair 4), top tie/second A with tally -2 (pair 9), and top tie/second B with tally +2 (pair 10). It is the complement of the already-explored design: instead of holding the top cue constant and varying lower net, it holds tally difference constant across cue-order conditions and varies the first discriminating cue. With 12 unique pairs repeated K=8, total trials = 96.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric measures the signed behavioral sensitivity of choices to the objective tally difference: trials where option A beats option B on strictly more features (tally difference positive) versus trials where option B beats option A on strictly more features (tally difference negative). In this design, the positive and negative tally groups each contain four repeated pairs, two of which have a first discriminating cue favoring A and two favoring B. Therefore a Take The Best agent is expected to choose A at 50% in both groups, making this contrast zero on average, while a Tallying agent should choose A at a high rate in the positive group and a low rate in the negative group, producing a large positive value. Averaging across four pairs per group per subject uses 32 trials per group, stabilizing the per-subject estimate and keeping between-subject variance small.

Source:
def metric(data: pd.DataFrame) -> float:
    a_rows = data['option_a_ratings'].tolist()
    b_rows = data['option_b_ratings'].tolist()
    tally = []
    for ra, rb in zip(a_rows, b_rows):
        a_wins = sum(1 for x, y in zip(ra, rb) if x > y)
        b_wins = sum(1 for x, y in zip(ra, rb) if y > x)
        tally.append(a_wins - b_wins)
    tally = np.asarray(tally, dtype=float)
    choose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    pos_rate = choose_a[tally > 0].mean()
    neg_rate = choose_a[tally < 0].mean()
    return float(pos_rate - neg_rate)

### RESULTS
- Predicted under pi_1 (simulated): -0.0013 (var=0.0059)
- Predicted under pi_2 (simulated): 0.7112 (var=0.0379)
- Observed on real data: -0.6438 (var=0.0430)

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
  "interpretation": "pi_1 is numerically closer to the observed data than pi_2 in both experiments. pi_1 predicted 0.0092 and -0.0013, whereas the observed values were -0.5117 and -0.6438, giving absolute errors of about 0.52 and 0.64. pi_2 predicted large positive values, 0.6258 and 0.7112, giving much larger absolute errors of about 1.14 and 1.36. However, pi_1 is only closer because its flat null prediction happens to sit near zero; it still misses the main empirical fact that choices depend strongly on the lower features. The consistent negative sign across two differently designed experiments is the decisive pattern: subjects choose the option that has more 1-valued ratings less often, i.e. their sensitivity to the tally is reversed relative to standard Tallying. The observed values are close to a sign-reversal of pi_2's predictions, which points to a valence reversal rather than just high noise. Neither current theory can capture that pattern without violating its own description, so a new theory is needed.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_2 with a defect-counting or inverse-tallying theory. The theory claims that people treat a binary rating of 1 as a negative or costly signal, or equivalently reverse the polarity of the cues, and then tally across all features choosing the option with the larger number of 0-valued cues. Ties lead to guessing. Response noise should remain a softmax over the two defect tallies with inverse temperature beta plus an independent lapse epsilon. The predict function should compute a_wins_defects = sum over features of B_j > A_j and b_wins_defects = sum over features of A_j > B_j, then apply the same softmax and lapse machinery currently used by pi_2. This differs from pi_1 by using all features instead of stopping at the first discriminating cue, and differs from pi_2 by reversing the feature valence. Deterministic defect-counting predicts about -0.83 for experiment 1 and about -1.00 for experiment 2; to match the observed -0.51 and -0.64, parameter ranges should be centered near beta around 1 to 5 and epsilon around 0.30 to 0.40, rather than near-deterministic settings. A validity-weighted defect tally can be considered later, but the equal-weight version is the clearest first instantiation."
}
```

## Usage

```json
{
  "input_tokens": 5993,
  "output_tokens": 13222,
  "reasoning_tokens": 12678,
  "provider": "Baidu"
}
```
