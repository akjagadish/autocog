# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_3
People invert the meaning of binary expert ratings: a rating of 1 is treated as a defect or costly negative signal, and a rating of 0 is treated as a clean or favorable signal. Across all features, they tally how often each option is defect-free relative to the other option, and choose the option with the larger defect-free tally. Ties lead to guessing. Choice probability is a noisy softmax over the two defect tallies with inverse temperature beta, plus an independent lapse process that produces uniform guessing with probability epsilon. The heuristic uses all features equally, ignores validities and past trials, and differs from Take The Best by aggregating rather than stopping at the first discriminating cue, and from standard Tallying by reversing the valence of the binary ratings.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Inverse-tallying expects a (2, n_features) stimulus; got shape {stim.shape}.')

    a, b = stim[0], stim[1]

    # Defect-free wins: A has a 0 where B has a 1.
    a_wins_defects = float(np.sum(b > a))
    # Defect-free wins: B has a 0 where A has a 1.
    b_wins_defects = float(np.sum(a > b))

    scores = np.array([a_wins_defects, b_wins_defects])

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    # Numerically stable softmax over inverse-cue tallies.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=7):** [0.9, 0.84, 0.78, 0.72, 0.66, 0.6, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0]

**Rationale:** Take The Best stops at the first discriminating cue in validity order and ignores all lower cues. Inverse tallying aggregates all cues with reversed valence (0 = favorable) and chooses the option with more zeros. This design holds the inverse-tallying score at exactly zero for every trial: each option has exactly one 1 and one 0 in the two discriminating positions, and all other positions tie. Thus inverse tallying predicts p(A)=0.5 for every pair, regardless of its beta and epsilon. TTB, however, consults cues in validity order and stops at the first discriminating cue. By shifting the position of that first discriminating cue from feature 0 to feature 5 and varying its direction (A=1/B=0 vs A=0/B=1), TTB predicts p(A) to be high when the first cue favors A and low when it favors B, regardless of the lower balancing cue. This yields a strong quantitative dissociation: a flat 0.5 psychometric function across all 12 pairs for inverse tallying versus a step function of first-cue direction for TTB. Unlike the already-explored designs, which held the top cue constant while varying lower-cue net (experiment 0) or varied first-cue direction and tally difference with standard Tallying (experiment 1), this design uses the inverse-tallying tie prediction on every trial and varies the first-cue position across all six possible positions.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric scores each choice as +1 when it matches the Take The Best prediction and -1 when it does not, then averages the scores. TTB processes cues in descending validity and selects the first discriminating cue. For all 12 pair types in this design, the first discriminating cue has a fixed direction, so TTB predicts each pair consistently. Inverse tallying, by contrast, always has exactly one defect-free win for A and one for B on every trial, producing tied tallies and a 0.5 choice probability throughout. Therefore inverse tallying has expected signed contrast 0, while TTB has expected contrast (1 - epsilon)*(2*sigmoid(beta) - 1), which is positive across the beta and epsilon ranges. Averaging over 96 trials per subject makes this contrast both large in mean and stable across subjects.

Source:
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.84, 0.78, 0.72, 0.66, 0.6, 0.55])
    cue_order = np.argsort(-validities, kind='stable')

    a = np.array([list(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([list(r) for r in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int)

    n = a.shape[0]
    if n == 0:
        return 0.0

    predicted = np.full(n, -1, dtype=int)
    for cue in cue_order:
        undecided = predicted == -1
        disc = np.logical_and(undecided, a[:, cue] != b[:, cue])
        if not np.any(disc):
            continue
        predicted[disc] = np.where(a[disc, cue] > b[disc, cue], 0, 1)

    decided = predicted != -1
    if not np.any(decided):
        return 0.0

    match = (resp[decided] == predicted[decided]).astype(float)
    return float(np.mean(2.0 * match - 1.0))

### RESULTS
- Predicted under pi_1 (simulated): 0.7113 (var=0.0398)
- Predicted under pi_3 (simulated): 0.0175 (var=0.0062)
- Observed on real data: -0.4300 (var=0.0221)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.9, 0.8, 0.7, 0.6, 0.55, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 9: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 10: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 11: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Rationale:** Six features with validities [0.90, 0.80, 0.70, 0.60, 0.55, 0.50] fix TTB's cue order as feature 0 through 5. Every trial has all six features discriminating, so TTB always stops at feature 0; its direction is manipulated across trials. The remaining five features are arranged so that the inverse-tally score s_A = #(A=0,B=1) - #(A=1,B=0) takes values +2, 0, or -2. This creates a 2 (feature-0 direction: A=1/B=0 vs A=0/B=1) x 3 (inverse tally score) factorial with two replications per cell. Inverse tallying ignores which feature discriminates and uses only s_A, so it predicts p(A) ordered by s (+2 > 0 > -2) and no effect of feature-0 direction. TTB stops at feature 0 and ignores all lower features, so it predicts p(A) high when feature 0 favors A and low when feature 0 favors B, with no effect of s. The conflict cells (feature-0 A with s=-2; feature-0 B with s=+2) yield opposite directional predictions, and the s=0 cells yield p(A)=0.5 for inverse tallying while TTB commits to the top cue. This double dissociation quantitatively separates the two theories.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The metric contrasts choice proportions when the inverse-tally score s_A = #(A=0,B=1) - #(A=1,B=0) is +2 versus -2. Inverse tallying predicts that larger s_A leads to a much higher probability of choosing A, so this contrast is strongly positive on average. TTB, however, stops at feature 0 and ignores s_A entirely. By design, both the s=+2 and s=-2 trial cells contain two trials with feature 0 favoring A and two trials with feature 0 favoring B, so TTB's expected p(A) in each of these cells is exactly 0.5 regardless of beta or epsilon. Thus the metric is near zero for TTB and clearly positive for inverse tallying, and each cell has 32 observations per subject, keeping within-subject variance small.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a = np.array(data['option_a_ratings'].tolist(), dtype=int)
    b = np.array(data['option_b_ratings'].tolist(), dtype=int)
    # inverse-tally advantage for A: #(A=0,B=1) - #(A=1,B=0)
    s = (b > a).sum(axis=1) - (a > b).sum(axis=1)
    mask_pos = s == 2
    mask_neg = s == -2
    pos = data.loc[mask_pos, 'response']
    neg = data.loc[mask_neg, 'response']
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    p_pos = (pos == 0).mean()
    p_neg = (neg == 0).mean()
    return float(p_pos - p_neg)

### RESULTS
- Predicted under pi_1 (simulated): -0.0244 (var=0.0050)
- Predicted under pi_3 (simulated): 0.6381 (var=0.0147)
- Observed on real data: 0.4925 (var=0.0658)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
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
```

**Observed (real) value:** -0.5117 (var=0.0161)
**Predicted under pi_1:** 0.0092 (var=0.0043)
**Predicted under pi_3:** -0.5379 (var=0.0084)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Metric**
```python
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
```

**Observed (real) value:** -0.6438 (var=0.0430)
**Predicted under pi_1:** -0.0013 (var=0.0059)
**Predicted under pi_3:** -0.6069 (var=0.0076)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments pi_3, the inverse-tallying theory, is clearly the better account. It predicts the observed negative lower-cue effects in Experiment 3 (-0.5379 predicted vs -0.5117 observed) and Experiment 4 (-0.6069 predicted vs -0.6438 observed) almost exactly, and it correctly predicts the positive direction of the inverse-tally contrast in Experiment 2 (0.6381 predicted vs 0.4925 observed), although it somewhat overpredicts the size of that effect. pi_1, by contrast, is effectively falsified: TTB predicts no effect of lower-validity cues, so it is near zero on Experiments 2-4 (-0.0244, 0.0092, -0.0013) while the observed values are +0.4925, -0.5117, and -0.6438. Experiment 1 is decisive for why neither current theory is sufficient. Every trial has inverse-tally scores exactly tied, so pi_3 must guess and predicts near-zero signed TTB contrast (0.0175), but the observed value is strongly negative (-0.4300). That means participants do not guess when the inverse tally is tied. Instead they appear to fall back on cue order with inverted valence: they choose the option that is 0 on the highest-validity discriminating cue, i.e., they prefer the option whose only defect/1 is on the lower-validity cue. pi_1 predicts the opposite sign in Experiment 1 and is therefore doubly wrong. So the best move is to keep pi_3 as the useful incumbent but replace the clearly degenerate pi_1 with a new theory that adds a validity-ordered inverse tie-break to inverse tallying.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory replacing pi_1, something like 'Inverse tallying with lexicographic inverse tie-break'. Operational sketch: treat 1 as a defect and 0 as a clean signal. First compute the inverse tally advantage s_A = number of features where A=0 and B=1 minus number of features where A=1 and B=0. When s_A is nonzero, choose the option with the larger inverse tally, using a softmax over s_A with inverse temperature beta_tally plus an epsilon lapse. When s_A is zero, do not guess uniformly. Instead, break the tie by consulting features in descending validity order, stopping at the first discriminating feature, and choosing the option with a 0 on that feature. That tie-break should also be implemented with a noisy softmax and lapse, so the pooled prediction is not deterministically -1 but can land near the observed -0.43 in Experiment 1. Validities should mainly govern the tie-break, while the main tally can remain unweighted or only weakly validity-weighted, because equal-weight inverse tallying already fits Experiments 3 and 4 well. Tune beta_tally, beta_tiebreak, epsilon, and possibly an individual-difference mixture so the predictions approximate Experiment 1 around -0.43, Experiment 2 around +0.49, Experiment 3 around -0.51, and Experiment 4 around -0.64. This theory differs from pi_1 because it aggregates all cues rather than stopping at the first discriminating cue, and it differs from pi_3 because ties in the inverse tally are resolved by validity-ordered inverse cue use rather than uniform guessing."
}
```

## Usage

```json
{
  "input_tokens": 7506,
  "output_tokens": 14687,
  "reasoning_tokens": 13940,
  "provider": "Baidu"
}
```
