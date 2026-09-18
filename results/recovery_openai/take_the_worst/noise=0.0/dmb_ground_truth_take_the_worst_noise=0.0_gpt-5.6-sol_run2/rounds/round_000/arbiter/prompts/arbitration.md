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
**Validities (n_features=7):** [0.95, 0.86, 0.78, 0.7, 0.64, 0.57, 0.52]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  trial 10: A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  trial 11: A=[0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1]
  trial 12: A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 1]
  trial 13: A=[1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  trial 14: A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  trial 15: A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 16: A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0]

**Rationale:** Seven experts with sharply separated advertised validities establish an unambiguous cue hierarchy. The eight structural contrasts are each presented in both A/B orientations, controlling side bias and yielding 16 unique pairs (six repetitions each, for 96 trials). For the first orientation of each contrast, TTB always selects A; for the reversed orientation it always selects B. Critically, TTB predicts the same choice probability on every trial because its binary winner score is invariant to cue rank and to all evidence below the first discriminating cue. Tallying instead predicts distinct patterns. Contrast 1 has one win per option and therefore forces exact 50/50 tallying despite a clear TTB choice. Contrasts 2-4 oppose the highest-validity cue with lower-validity majorities whose tally margins favor the other option by 1, 3, and 5, producing an increasingly strong reversal under tallying but no change under TTB. Contrasts 5-6 make the highest one or two cues tie, so TTB proceeds to the second- or third-ranked cue and still chooses A, whereas tallying favors B by margins 1 and 2. Contrasts 7-8 make both theories choose A but vary the tally margin from 1 to 4; tallying predicts a quantitative confidence gradient while TTB again predicts a constant choice probability. Thus the design tests both categorical conflict and the central quantitative dissociation: tally-margin sensitivity versus TTB invariance.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This is the within-subject OLS slope of TTB-consistent choice on whether tallying opposes, ties, or supports the TTB winner. TTB predicts the same probability of choosing its winner in all three categories, so its expected slope is zero regardless of beta or lapse rate. Tallying predicts a strong positive gradient: low TTB agreement when the tally opposes TTB, exactly 0.5 at tally ties, and high agreement when tallying supports TTB. Collapsing tally margins to their sign is deliberate: it preserves the robust categorical disagreement while avoiding sensitivity to softmax saturation at large beta. The slope uses nearly all 96 trials and removes each subject's overall tendency to follow TTB through centering, yielding substantially lower between-subject variance than a raw accuracy score or a contrast based on only one trial pair.

Source:
def metric(data: pd.DataFrame) -> float:
    # Recode each response as agreement with Take-The-Best (TTB), then
    # estimate the within-data slope of that agreement on the direction
    # of the equal-weight tally relative to the TTB winner.
    x = []
    y = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b

        # Features are ordered from highest to lowest advertised validity.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_winner = 0 if diff[first] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        tally_margin_for_ttb = (
            a_wins - b_wins if ttb_winner == 0 else b_wins - a_wins
        )

        # -1: tally opposes TTB; 0: tally ties; +1: tally supports TTB.
        x.append(float(np.sign(tally_margin_for_ttb)))
        y.append(float(int(response) == ttb_winner))

    if len(x) < 2:
        return float("nan")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - np.mean(x)
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return float("nan")

    return float(np.dot(xc, y - np.mean(y)) / denom)

### RESULTS
- Predicted under pi_1 (simulated): -0.0095 (var=0.0016)
- Predicted under pi_2 (simulated): 0.3604 (var=0.0118)
- Observed on real data: 0.0011 (var=0.0014)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=10):** [0.96, 0.9, 0.84, 0.78, 0.72, 0.67, 0.62, 0.58, 0.54, 0.51]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  trial 3: A=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 6: A=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  trial 9: A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 11: A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
  trial 12: A=[0, 0, 0, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
  trial 13: A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 14: A=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
  trial 15: A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
  trial 16: A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]

**Rationale:** This design tests the advocated theory's permutation invariance rather than repeating a tally-margin gradient. Every trial in the first block contains exactly five feature wins for A, three for B, and two ties, so Tallying assigns all eight trials the identical score difference of +2 and predicts the same A-choice probability: (1-epsilon)*sigmoid(2*beta)+epsilon/2. However, the same win/loss/tie multiset is permuted across validity ranks. On trials 1-4 the first discriminating cue favors A, whereas on trials 5-8 it favors B; the first discriminator occurs at ranks 1, 2, or 3 across exemplars. TTB therefore predicts two sharply separated probability clusters within a set that Tallying predicts to be homogeneous. Trials 9-16 reverse A and B for exact side balance: Tallying gives every reversed trial the same score difference of -2, while TTB's winner also reverses. Thus the key quantitative diagnostic is within-block heterogeneity: systematic choice changes based on the highest-ranked discriminating cue support TTB, whereas stable choice probabilities under validity-rank permutations support Tallying. Six repetitions of each of the 16 unique pairs yield 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This is signed choice alignment with Tallying, restricted to the maximally diagnostic trials where Tallying and Take The Best predict opposite winners. It ranges from -1 to +1: positive values indicate choices following the tally winner, while negative values indicate choices following the highest-validity discriminating cue. In this design, half of the unique pairs—and their exact reversals—are such conflict trials, yielding 48 observations per subject. Tallying predicts a positive expectation approximately (1-epsilon)*tanh(beta), whereas TTB predicts a negative expectation approximately -(1-epsilon)*tanh(beta/2). Thus the theories lie on opposite sides of zero rather than differing only in degree. Exact A/B reversals remove side bias, and averaging all conflict-trial repetitions keeps the per-subject variance low. The metric is linear across observations, so its pooled value remains compatible with its single-subject interpretation.

Source:
def metric(data: pd.DataFrame) -> float:
    contrasts = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        cue_directions = np.sign(a - b)

        # +1 denotes an A winner and -1 a B winner.
        tally_winner = float(np.sign(cue_directions.sum()))

        discriminating = np.flatnonzero(cue_directions != 0)
        if tally_winner == 0 or discriminating.size == 0:
            continue
        ttb_winner = float(cue_directions[discriminating[0]])

        # Retain only trials on which the theories make opposite predictions.
        if tally_winner != ttb_winner:
            chosen_side = 1.0 if int(row["response"]) == 0 else -1.0
            contrasts.append(chosen_side * tally_winner)

    if len(contrasts) == 0:
        return float("nan")
    return float(np.mean(contrasts))

### RESULTS
- Predicted under pi_1 (simulated): -0.6767 (var=0.0457)
- Predicted under pi_2 (simulated): 0.7633 (var=0.0349)
- Observed on real data: 0.3433 (var=0.0123)

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
  "interpretation": "The experiments produce a crossover rather than a single-theory victory. In Experiment 1, the observed slope (0.0011) is essentially the invariance predicted by Take The Best (-0.0095) and is far from Tallying's positive gradient (0.3604). The observed between-subject variance (0.0014) also closely matches pi_1 (0.0016) and is much lower than pi_2 (0.0118), suggesting a highly consistent absence of tally-direction effects. In Experiment 2, however, the observed conflict alignment is positive (0.3433): subjects favor the tally winner, directly contradicting pi_1's negative prediction (-0.6767) and qualitatively supporting pi_2's positive prediction (0.7633), although pure Tallying substantially overstates the effect. The low observed variance (0.0123) suggests a stable intermediate tendency rather than highly heterogeneous, all-or-none strategy use. Thus pi_1 captures Experiment 1 exceptionally well but is structurally incapable of producing the positive Experiment 2 result; pi_2 captures the direction of Experiment 2 but fails decisively in Experiment 1. A fixed mixture also fails: interpolation between the predictions implies almost no Tallying in Experiment 1 but roughly 71% Tallying in Experiment 2. Adjusting only beta or epsilon cannot explain why evidence integration changes across tasks, because those parameters alter response strength rather than the underlying cue-use rule. The data therefore require a task-sensitive theory of how a decision rule is selected, not merely a new noisy instantiation of either fixed heuristic.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_2 with a capacity-sensitive strategy-arbitration theory. The new theory should hold that people attempt validity-ordered, lexicographic processing when the cue hierarchy is manageable, but increasingly lose reliable access to cue ranks as the number of features and comparison load rise. Under low load, choices therefore approximate TTB and remain insensitive to lower-cue tally direction, explaining the seven-feature result. Once cue-set complexity exceeds an individual cognitive-capacity threshold, rank information is compressed and multiple cues are pooled with approximately equal or coarsened weights, producing tally-consistent choices in the ten-feature task. Arbitration should occur probabilistically within subjects, with a smooth load-dependent gate rather than assigning each subject permanently to one heuristic; this can generate the stable, attenuated positive alignment observed in Experiment 2. Include parameters for capacity or rank-retrieval precision, the steepness of the load gate, evidence sensitivity, and lapse, with the gate driven by observable task complexity rather than experiment identity. The theory should predict near-lexicographic behavior around seven well-separated cues, predominantly but not purely integrative behavior around ten cues, and testable transitions at intermediate feature counts. This is a genuinely new theoretical claim about adaptive representation and strategy selection, while retaining pure TTB and pure Tallying only as limiting regimes."
}
```

## Usage

```json
{
  "input_tokens": 6582,
  "output_tokens": 2747
}
```
