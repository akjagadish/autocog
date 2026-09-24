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
Negative Validity-Weighted Additive (NWA): people combine all binary feature differences linearly, but assign negative weight to a feature advantage, and the absolute weight magnitude increases with the feature's stated validity. Formally, choice is based on S = sum_j w_j * (A_j - B_j), with w_j = -v_j^gamma (gamma > 0), so an option's cue advantage makes that option less likely to be chosen. Choice probability uses a softmax/sigmoid over this score with inverse temperature beta, plus a small uniform lapse epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # NWA: negative validity-weighted additive integration of all cues.
    # state is shape (2, n_features): row 0 = option A, row 1 = option B.
    # history is ignored because there is no trial-by-trial feedback.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"NWA expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Negative validity-scaled weights: an advantage on a cue is evidence
    # against that option, with stronger validity cues carrying more weight.
    weights = -(v ** gamma)

    score = float(np.dot(weights, a - b))

    # Stable sigmoid for p(A | score).
    z = beta * score
    if z >= 0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    return np.array([p_a, 1.0 - p_a], dtype=float)


`policy(probs) -> int`:
def policy(probs):
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


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
**Validities (n_features=6):** [0.9, 0.85, 0.8, 0.55, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Rationale:** Six experts are split into three high-validity cues (0.90, 0.85, 0.80) and three low-validity cues (0.55, 0.50, 0.50). NWA assigns negative validity-scaled weights, so an option's advantage on a high-validity cue is strong evidence against that option, whereas Tallying only counts the number of feature-wise wins and ignores validity. The eight pairs are organized by tally difference d = #A wins - #B wins. Trials 1-2 have d=0: Tallying must predict 0.50 for both, but NWA predicts B on trial 1 and A on trial 2 because the high-validity wins switch sides. Trials 3-4 both have d=+1, so Tallying predicts the same A-choice probability for both; NWA predicts B on trial 3 (A wins the three high cues) and A on trial 4 (A wins only the three low cues while B wins two high cues). Trials 5-6 mirror this at d=-1. Trials 7-8 are d=+2 and d=-2, where Tallying predicts strong A and strong B respectively, but NWA predicts the opposite. With beta=2.7, gamma=2.25, NWA p_A is approximately .02, .98, .01, .90, .99, .10, .006, .995 for trials 1-8; Tallying p_A is .50, .50, sigmoid(beta), sigmoid(beta), sigmoid(-beta), sigmoid(-beta), sigmoid(2beta), sigmoid(-2beta). The within-d reversals are the critical quantitative dissociation: Tallying cannot produce different p_A for trials with the same tally difference, regardless of beta or epsilon.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric averages three signed contrasts between trial types that have the same tally difference (0, +1, and -1) but opposite validity configuration. Within each matched pair, Tallying must assign identical choice probabilities regardless of beta or epsilon, so its expected signed contrast is exactly zero. NWA instead predicts a large positive contrast in each pair because a high-validity advantage makes that option less likely. Averaging three independent contrasts, each estimated from 12 repeated presentations, yields an expected NWA score near 0.90 with small per-subject variance.

Source:
def metric(data):
    import numpy as np
    import pandas as pd

    df = data.copy()
    df["pair_key"] = df.apply(
        lambda r: (tuple(r["option_a_ratings"]), tuple(r["option_b_ratings"])),
        axis=1,
    )

    nwa_negative_pairs = [
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0)),
        ((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
    ]
    nwa_positive_pairs = [
        ((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 0), (1, 1, 1, 0, 0, 0)),
    ]

    def p_a_for_key(d, key):
        mask = d["pair_key"] == key
        return 1.0 - float(d.loc[mask, "response"].mean())

    def subject_score(d):
        neg = [p_a_for_key(d, k) for k in nwa_negative_pairs]
        pos = [p_a_for_key(d, k) for k in nwa_positive_pairs]
        return sum([pos[i] - neg[i] for i in range(3)]) / 3.0

    scores = df.groupby("subject_id", sort=False).apply(subject_score)
    return float(scores.mean())


### RESULTS
- Predicted under pi_3 (simulated): 0.8928 (var=0.0032)
- Predicted under pi_2 (simulated): 0.0017 (var=0.0075)
- Observed on real data: 0.3622 (var=0.0979)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 2: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 3: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 6: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 7: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]

**Rationale:** This design uses four experts with a wide validity spread (0.95, 0.85, 0.65, 0.55). The first six unique pairs are all 2-vs-2 splits, so Tallying's net feature-win difference is zero for every one of them; Tallying must predict p(A)=0.50 exactly regardless of beta or epsilon because the softmax over equal tallies is uniform and the lapse mixture remains uniform. NWA, by contrast, assigns negative validity-scaled weights, so the same zero tally produces a graded range of p(A) as the two A-won features move from the two highest-validity experts (NWA p_A approximately 0.07) to the two lowest-validity experts (NWA p_A approximately 0.93). The last two pairs are 3-vs-1 splits with tally differences +2 and -2. Tallying therefore predicts a strong A preference for the +2 pair and a strong B preference for the -2 pair, while NWA predicts the opposite direction on both because the majority wins are low-validity and the single minority win is the highest-validity expert (NWA p_A approximately 0.23 and 0.77, assuming beta=2.70, gamma=2.25, epsilon=0). This creates a flat-vs-graded signature on the d=0 trials and a sign reversal on the d=+/-2 trials, which no Tallying beta or epsilon can reproduce.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
On all 2-vs-2 zero-net-tally pairs Tallying predicts p(A)=0.50 exactly for every beta and epsilon, because the softmax over equal tallies is uniform and the lapse mixture remains uniform. NWA instead predicts a graded ordering: trials where A wins the two highest-validity features get strongly negative scores and low p(A), while their mirror-image trials get strongly positive scores and high p(A). The metric contrasts empirical A-choice rates on the two high-NWA mirror pairs against the two low-NWA mirror pairs: mean choose A given diff in {(-1,-1,1,1),(-1,1,-1,1)} minus mean choose A given diff in {(1,1,-1,-1),(1,-1,1,-1)}. Its expected value is 0 under Tallying and approximately 0.62 under NWA at the stated parameters. It deliberately ignores the 3-vs-1 trials, whose Tallying predictions vary with beta and would add Tallying-side variance.

Source:
def metric(data: pd.DataFrame) -> float:
    diffs = [tuple(int(x) - int(y) for x, y in zip(a, b)) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    choose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    high_keys = {(-1, -1, 1, 1), (-1, 1, -1, 1)}
    low_keys = {(1, 1, -1, -1), (1, -1, 1, -1)}
    high_mask = np.array([d in high_keys for d in diffs], dtype=bool)
    low_mask = np.array([d in low_keys for d in diffs], dtype=bool)
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0
    return float(choose_a[high_mask].mean() - choose_a[low_mask].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.6125 (var=0.0127)
- Predicted under pi_2 (simulated): 0.0025 (var=0.0212)
- Observed on real data: 0.5300 (var=0.0883)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data):
    import numpy as np
    min_class = 12
    groups = {0: [], 1: []}
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)
        if a[0] > b[0]:
            grp = 0
        elif b[0] > a[0]:
            grp = 1
        else:
            continue
        tally = int(np.sum(a > b) - np.sum(b > a))
        chose_a = 1.0 if int(row['response']) == 0 else 0.0
        groups[grp].append((tally, chose_a))
    parts = []
    for grp in (0, 1):
        vals = groups[grp]
        if not vals:
            parts.append(0.0)
            continue
        d_a = [d for d, y in vals if y == 1.0]
        d_b = [d for d, y in vals if y == 0.0]
        if len(d_a) < min_class or len(d_b) < min_class:
            parts.append(0.0)
        else:
            parts.append((np.mean(d_a) - np.mean(d_b)) / 6.0)
    return float(np.mean(parts))
```

**Observed (real) value:** -0.4323 (var=0.0402)
**Predicted under pi_3:** -0.5184 (var=0.0087)
**Predicted under pi_2:** 0.4517 (var=0.0090)

### Experiment 4
**Design**
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    validities = np.array([0.9, 0.8, 0.65, 0.55, 0.5])
    cue_order = np.argsort(-validities, kind='stable')
    a_when_ttb_a = []
    a_when_ttb_b = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner == -1:
            continue
        chose_a = 1 if int(row['response']) == 0 else 0
        if ttb_winner == 0:
            a_when_ttb_a.append(chose_a)
        else:
            a_when_ttb_b.append(chose_a)
    if not a_when_ttb_a or not a_when_ttb_b:
        return 0.0
    return float(np.mean(a_when_ttb_a) - np.mean(a_when_ttb_b))
```

**Observed (real) value:** -0.2650 (var=0.0713)
**Predicted under pi_3:** -0.2804 (var=0.0079)
**Predicted under pi_2:** -0.0083 (var=0.0072)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments pi_3 is the only theory that gets the valence and approximate size right: Exp3 observed -0.4323 vs pi_3 -0.5184 (pi_2 +0.4517, wrong sign); Exp4 observed -0.2650 vs pi_3 -0.2804 (pi_2 -0.0083); Exp2 observed 0.5300 vs pi_3 0.6125 (pi_2 0.0025); Exp1 observed 0.3622 vs pi_3 0.8928 (pi_2 0.0017). Pi_2's null or opposite predictions are structural: tallying treats all cues equally, ignores communicated validities, and on zero-net-tally pairs is forced to 0.50 regardless of beta or epsilon, so no parameter regeneration can repair it. Observed between-subject variance is much larger than either simulation, suggesting heterogeneity, but that should be captured by a new competitor to pi_3 rather than by recycling pi_2. Pi_2 is therefore degenerate as a theory, not merely miscalibrated.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_2 with a heterogeneous signed-validity weighting theory. The new theory should state that people weight binary feature differences by communicated validity, but the sign of the weighting is not uniform: a majority use the negative-valence/contra-indicator rule captured by pi_3, while a minority use positive-valence trust or near-equal tallying. Its predict function should compute score_i = sum_j s_i * v_j^gamma * (A_j - B_j) with softmax noise and lapse, where s_i is a subject-level valence sign or a cue-specific sign drawn from a latent strategy distribution, and gamma/beta vary across subjects. This should preserve the aggregate anti-tally/anti-TTB signs and close magnitudes seen in Exp3 and Exp4, while attenuating the overly sharp Exp1 prediction toward the observed 0.36 and reproducing the large per-subject variance. It differs from pi_2 because validities are used explicitly and equal tallying is only a special boundary; it differs from pi_3 by adding valence and strategy heterogeneity rather than a single fixed negative-weight rule."
}
```

## Usage

```json
{
  "input_tokens": 6909,
  "output_tokens": 7654,
  "reasoning_tokens": 7156,
  "provider": "NextBit"
}
```
