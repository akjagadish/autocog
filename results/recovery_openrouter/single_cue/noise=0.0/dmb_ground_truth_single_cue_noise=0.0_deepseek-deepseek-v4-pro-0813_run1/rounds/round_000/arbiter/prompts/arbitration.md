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
**Validities (n_features=8):** [0.9, 0.75, 0.7, 0.65, 0.6, 0.55, 0.52, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 1]
  trial 5: A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  trial 10: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  trial 11: A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 1, 0, 1]
  trial 12: A=[0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1, 0]
  trial 13: A=[0, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0]
  trial 14: A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  trial 15: A=[0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** Eight experts are used. Expert 0 is advertised as 90% accurate; the remaining seven are clearly lower (75% down to 50%). Every critical trial makes Expert 0 discriminate: it favors one option. The seven lower experts are arranged so that k = 0,1,...,7 of them favor the opposite option. Take The Best stops at Expert 0 and therefore predicts a constant probability of choosing the top-cued option across all k. Tallying counts all eight experts; the tally margin for the top-cued option is 8 - 2k, so its predicted choice probability is a decreasing sigmoid in k, crossing chance at k=4 and reversing for k>4. Two mirrored blocks are included (Expert 0 favors A in one block, B in the other) to double observations and control for response bias. This yields a strong quantitative dissociation: a flat psychometric function over k under Take The Best versus a negatively sloped sigmoid under Tallying. The 16 unique pairs are repeated 6 times for roughly 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
For each trial, identify the option favored by the highest-validity cue (feature 0) and count k, the number of lower-validity cues favoring the opposite option. Define choose_top = 1 if the subject chose the top-cued option. Under Take The Best, the first cue always decides, so P(choose_top) is flat across k and the within-subject slope of choose_top against k is approximately zero. Under Tallying, the tally margin for the top-cued option is 8 - 2k, so P(choose_top) falls as k increases, producing a reliably negative slope. The metric averages each subject's least-squares slope, giving a tight per-subject estimate because every k has 12 observations per subject.

Source:
def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    def subject_slope(sdf):
        a = np.array(sdf['option_a_ratings'].tolist(), dtype=float)
        b = np.array(sdf['option_b_ratings'].tolist(), dtype=float)
        resp = sdf['response'].to_numpy(dtype=float)
        n = a.shape[0]

        top = np.full(n, -1, dtype=int)
        top[a[:, 0] > b[:, 0]] = 0
        top[b[:, 0] > a[:, 0]] = 1
        valid = top >= 0

        opp_wins = np.zeros(n, dtype=int)
        for j in range(1, a.shape[1]):
            opp_wins[(top == 0) & (b[:, j] > a[:, j])] += 1
            opp_wins[(top == 1) & (a[:, j] > b[:, j])] += 1

        choose_top = np.where(resp == top, 1.0, 0.0)

        x = opp_wins[valid].astype(float)
        y = choose_top[valid]
        if x.size < 2 or np.ptp(x) == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    slopes = data.groupby('subject_id', sort=False).apply(subject_slope)
    vals = np.atleast_1d(np.asarray(slopes, dtype=float))
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))

### RESULTS
- Predicted under pi_1 (simulated): 0.0006 (var=0.0002)
- Predicted under pi_2 (simulated): -0.1360 (var=0.0009)
- Observed on real data: -0.0959 (var=0.0010)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.9, 0.8, 0.7, 0.6, 0.6, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 4: A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 10: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 13: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 14: A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 15: A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  trial 16: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Rationale:** Six experts with validities 0.90, 0.80, 0.70, 0.60, 0.55, 0.50 define a fixed TTB cue order. The 16 unique pairs form a cue-locus × tally-margin design. In A-first trials (A0-A7) the first discriminating cue in validity order favors A; in B-first trials (B0-B7) it favors B. Higher-validity cues are tied on r=1 and r=2 trials so the first discriminator is not always the top expert, extending the already-explored design. Lower cues are arranged to produce tally margins (#A wins - #B wins) of -4, -3, -2, 0, +2, +3, +4. TTB stops at the first discriminator, so it predicts a single p(A) for all A-first trials and a single p(A) for all B-first trials, independent of the lower-cue margin. Tallying counts all six cues, so it predicts p(A) = sigmoid(beta * margin), increasing monotonically with the margin and equal to chance at margin 0. Thus TTB predicts a flat step function in margin within each first-discriminator direction, whereas Tallying predicts a graded psychometric function; zero-margin trials additionally oppose TTB's directional prediction to Tallying's chance prediction. The 16 unique pairs are repeated K=6 times for 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric extracts the two orthogonal signatures built into the 16 unique pairs. It labels every trial by the TTB first-discriminator direction, discovered by scanning features in descending validity order, and by the tally margin defined as the number of features A wins minus the number B wins. The first component, mean B choice on B-first trials minus mean B choice on A-first trials, isolates the constant directional bias predicted by TTB; Tallying predicts zero because each direction contains exactly matched tally counts. The second component, mean B choice on negative-margin trials minus mean B choice on positive-margin trials, isolates the psychometric margin effect predicted by Tallying; TTB predicts zero because positive and negative margin groups are balanced in first-discriminator direction. Subtracting the margin component from the directional component makes the metric positive under TTB and negative under Tallying. Using all 96 trials keeps the between-subject variance small.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.6], dtype=float)
    cue_order = np.argsort(-validities, kind='stable')

    y = data['response'].astype(float).to_numpy()
    dirs = np.zeros(len(data), dtype=float)
    margins = np.zeros(len(data), dtype=float)

    for i, (_, row) in enumerate(data.iterrows()):
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)

        margins[i] = float(np.sum(a > b) - np.sum(b > a))

        direction = 0.0
        for j in cue_order:
            if a[j] > b[j]:
                direction = 1.0
                break
            if b[j] > a[j]:
                direction = -1.0
                break
        dirs[i] = direction

    d_pos = dirs == 1.0
    d_neg = dirs == -1.0
    m_pos = margins > 0.0
    m_neg = margins < 0.0

    directional_contrast = y[d_neg].mean() - y[d_pos].mean()
    margin_contrast = y[m_neg].mean() - y[m_pos].mean()

    return float(directional_contrast - margin_contrast)


### RESULTS
- Predicted under pi_1 (simulated): 0.7279 (var=0.0420)
- Predicted under pi_2 (simulated): -0.7597 (var=0.0423)
- Observed on real data: -0.8592 (var=0.0589)

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
  "interpretation": "Across both experiments, pi_2 (Tallying) is clearly the better task-invariant account. In Experiment 1 the observed within-subject psychometric slope is -0.0959, close in sign and rough scale to Tallying's -0.1360, while pi_1 (Take The Best) predicts 0.0006, essentially flat, because its first discriminating cue determines choice regardless of how many lower-validity cues oppose it. In Experiment 2 the observed signed metric is -0.8592, again aligned with Tallying's -0.7597, but opposite in sign to pi_1's +0.7279. This is a decisive falsification of pi_1: the directional term under TTB is positive and the margin term is near zero, so no plausible beta or epsilon rescaling can flip TTB's prediction to the observed negative value. pi_2 also has some quantitative miscalibration: it is steeper than observed in Experiment 1 and slightly less extreme than observed in Experiment 2, which suggests a fixed equal-weight tally with one shared beta is an approximation rather than a fully specified account. Nonetheless, pi_2 captures the qualitative and nearly quantitative pattern in both experiments, so it should be retained. pi_1 is degenerate for these data and should be replaced by a new competitor rather than reparameterized.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 with a validity-weighted integration theory. On each trial, the learner should compare the two options on all cues, assign each cue comparison a weight that increases with the advertised or inferred cue validity, and integrate the signed feature-wise evidence into a decision value; ties contribute zero. Choice then follows a softmax over that weighted evidence with inverse temperature beta plus a lapse term. This new theory differs from TTB because it does not stop at the first discriminating cue and therefore predicts graded psychometric effects; it differs from pure equal-weight Tallying because low-validity cues are discounted, which can reduce the Exp1 slope relative to the pure tally while preserving the graded margin effect needed for Exp2. The initial model should use monotone validity weights, e.g. weights proportional to validity or rank-transformed validity, with beta and epsilon calibrated to match the observed values of approximately -0.096 in Experiment 1 and -0.859 in Experiment 2, and should include validation experiments that distinguish equal-weight tallying from validity-weighted integration by manipulating the spread of cue validities independently of raw tally margins."
}
```

## Usage

```json
{
  "input_tokens": 6651,
  "output_tokens": 19729,
  "reasoning_tokens": 19185,
  "provider": "Baidu"
}
```
