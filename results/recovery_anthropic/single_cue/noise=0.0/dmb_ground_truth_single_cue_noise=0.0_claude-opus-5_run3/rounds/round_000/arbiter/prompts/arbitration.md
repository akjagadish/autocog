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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 11: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 12: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Rationale:** Five experts with clearly ordered, well-spread validities (0.95, 0.85, 0.75, 0.65, 0.55) so that Take-The-Best's cue cascade is unambiguous. The 12 unique pairs (each repeated 8 times, ~96 trials) are chosen to create four distinct diagnostic classes:

(1) CONFLICT / non-compensatory pairs (trials 1, 2, 8, 10): the option favored by the single highest-validity discriminating cue is beaten on the tally (e.g. A=[1,0,0,0,0] vs B=[0,1,1,1,1], tally 1-4 for B). TTB predicts a confident choice of A; Tallying predicts a confident choice of B, with margin-graded confidence (4-cue margin in trials 1/8 vs 1-cue margin in trials 2/10).

(2) TALLY-TIE pairs (trials 3, 4): A wins two cues, B wins two cues, so Tallying is exactly at chance (50/50 regardless of beta and epsilon), whereas TTB decides confidently on the top cue. Any systematic deviation from 50% here is impossible for Tallying at any parameter setting and is therefore the sharpest single-trial discriminator.

(3) LOW-CUE-DECIDES pairs (trials 7, 9): the top two cues tie, the third cue (validity .75) discriminates in one direction while the two lowest cues both favor the other option, so TTB and Tallying make opposite predictions in a case where TTB has to descend the cascade rather than read the first cue.

(4) AGREEMENT pairs with different tally margins (trials 5, 6, 11, 12): both theories pick the same option, but Tallying's softmax makes accuracy strongly margin-dependent (4-1 vs 2-1), while TTB predicts identical choice probability sigmoid(beta) irrespective of margin. Comparing accuracy on trials 5/11 versus 6/12 gives a purely quantitative dissociation that does not depend on which option is 'correct'.

All pairs are mirrored across A and B to avoid side bias, and no pair is degenerate (every pair has at least one discriminating cue and the two theories either disagree in direction or in predicted confidence).

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is a within-subject, parameter-free contrast that isolates the one-reason signature of Take-The-Best while cancelling out each subject's overall noise level.

For every trial I recompute (a) the TTB winner (first discriminating cue in descending-validity feature order) and (b) the tally winner. Trials split into CONFLICT trials, where the two heuristics point to different options (design trials 1, 2, 7, 8, 9, 10), and AGREEMENT trials, where they point to the same option (trials 5, 6, 11, 12). Tally-tie pairs (3, 4) are dropped because they are uninformative for this particular contrast. The metric is P(choose the TTB-favoured option | conflict) minus P(choose the TTB-favoured option | agreement).

Predictions:
- Take-The-Best: choice probability for the top-cue winner is sigmoid(beta) softened by the lapse, identically on every decisive pair regardless of tally margin or conflict status. So the two rates are the same and the metric is exactly 0 in expectation, for ALL values of beta and epsilon. Only binomial sampling noise (48 vs 32 trials per class) contributes, giving a tight per-subject variance of roughly 0.013.
- Tallying: on conflict trials the subject systematically chooses AGAINST the TTB option (rate 1 - q), while on agreement trials they choose WITH it (rate q). The metric is therefore strongly negative, around -0.6 to -0.9 for any moderately decisive beta, with modest across-subject spread driven mostly by the lapse rate.

Because the TTB prediction is a hard zero that does not depend on the fitted nuisance parameters, and the Tallying prediction is a large negative number, the two theories should separate by roughly 0.7-0.8 with small per-subject variance on both sides. It is not trivially true for TTB in the sense of being a restatement of the model: any strategy that weights evidence compensatorily (tallying, weighted-additive, or even partial cue integration) will push the score negative, while pure one-reason decision making pins it at zero.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    dis_hits = 0.0
    dis_n = 0.0
    agr_hits = 0.0
    agr_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.size == 0:
            continue

        # Take-The-Best winner: features are listed in descending validity
        # order, so scan left to right for the first discriminating cue.
        ttb_winner = None
        for j in range(a.size):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally-tie pairs excluded (uninformative for the contrast)
        tally_winner = 0 if a_wins > b_wins else 1

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hit = 1.0 if r == ttb_winner else 0.0

        if tally_winner != ttb_winner:
            dis_hits += hit
            dis_n += 1.0
        else:
            agr_hits += hit
            agr_n += 1.0

    if dis_n == 0.0 or agr_n == 0.0:
        return float('nan')

    return float(dis_hits / dis_n - agr_hits / agr_n)

### RESULTS
- Predicted under pi_1 (simulated): 0.0038 (var=0.0080)
- Predicted under pi_2 (simulated): -0.6931 (var=0.0366)
- Observed on real data: 0.0933 (var=0.0064)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.9, 0.6, 0.8, 0.55, 0.7, 0.95]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 2: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 9: A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  trial 10: A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  trial 11: A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Rationale:** Six experts with well-spread but deliberately *non-monotone* validity order across screen positions (0.90, 0.60, 0.80, 0.55, 0.70, 0.95) so that Take-The-Best's cue cascade (f5 > f0 > f2 > f4 > f1 > f3) cannot be confused with a simple left-to-right reading strategy, while Tallying (which ignores order entirely) is unaffected. Twelve unique pairs, each repeated 8 times (96 trials), fall into four diagnostic classes.

(1) NON-COMPENSATORY CONFLICT with graded tally margin (trials 1-4). The single highest-validity cue (f5, 95%) favours one option while the tally favours the other, by margin 4 (trials 1, 2), margin 2 (trial 3) or margin 1 (trial 4). TTB predicts the SAME confidence sigmoid(beta) for all four, always siding with the f5 winner; Tallying predicts the opposite direction with confidence strictly increasing in the margin (sigmoid(4*beta) > sigmoid(2*beta) > sigmoid(beta)). Direction AND the margin gradient jointly identify the model.

(2) TALLY-TIE pairs (trials 5-8). Feature-wise wins are exactly balanced (3-3 in trials 5/6 with no ties; 2-2 with two tied cues in trials 7/8), so Tallying is pinned at exactly 50% for every (beta, epsilon) - no parameter setting can move it. TTB decides confidently: on trials 5/6 the top cue f5 discriminates immediately; on trials 7/8 f5 ties and TTB must descend to f0 (90%). Any systematic departure from chance here is impossible for Tallying, making these the sharpest single-cell discriminators, and trials 7/8 additionally verify that subjects use validity rank rather than screen position.

(3) LOW-CUE-DECIDES CONFLICT (trials 9, 10). The two most valid cues (f5, f0) tie; the third-ranked cue f2 (80%) discriminates one way while the three lower cues (f4, f1, f3) all favour the other option (tally 1-3). TTB must walk the cascade and then choose against the majority; Tallying chooses the majority. Opposite directions in a case where TTB's one-reason logic is hardest to mimic.

(4) AGREEMENT pairs with extreme margin contrast (trials 11, 12). Both theories favour the same option, but with tally margin 5 (trial 11) versus margin 1 (trial 12). TTB predicts identical accuracy on the two; Tallying predicts near-ceiling on trial 11 and markedly lower accuracy on trial 12. This is a purely quantitative dissociation that is independent of choice direction and of any objective 'correct' answer.

All classes are mirrored across sides (TTB predicts A on 6 pairs and B on 6; Tallying's non-tie predictions are likewise 4/4), so side bias cannot masquerade as either model. No pair is degenerate: every pair has at least one discriminating cue, and in every pair the two theories differ either in predicted direction or in predicted confidence.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains pairs where the highest-validity cue (f5, or f2 after f5/f0 tie) points one way while the feature tally points the other way with a margin of at least 2 (trials 1, 2, 3, 9, 10). Restricting to margin>=2 conflicts guarantees that Tallying is confident (softmax on a tally difference of 2-4 saturates for almost all beta in the allowed range), so Tallying predicts choice of the tally winner at rate (1-eps)*sigmoid(margin*beta)+eps/2, typically ~0.85-1.0. Take-The-Best predicts exactly the opposite option at rate (1-eps)*sigmoid(beta)+eps/2, so its tally-consistency rate is typically ~0.0-0.15. The metric is a simple proportion over 40 pooled trials per subject, so binomial noise is small, and the only between-subject spread comes from the lapse parameter eps (bounded contribution <= 0.25). This yields a mean gap of roughly 0.7-0.75 with per-subject variance on the order of 0.01, which is far beyond what N=25 needs. It is not trivially true for Tallying: it is a directional prediction on genuinely conflicting evidence, and any subject who consults more than one reason but weights cues (e.g., weighted-additive) would also land high, while a strict one-reason decider lands low.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.9, 0.6, 0.8, 0.55, 0.7, 0.95], dtype=float)

    n_match = 0.0
    n_total = 0.0

    for _, row in data.iterrows():
        a = np.asarray(list(row['option_a_ratings']), dtype=float)
        b = np.asarray(list(row['option_b_ratings']), dtype=float)
        n = a.shape[0]
        if validities.shape[0] == n:
            val = validities
        else:
            # fallback: assume given order is already descending validity
            val = np.arange(n, 0, -1, dtype=float)
        order = np.argsort(-val, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        margin = abs(a_wins - b_wins)
        if margin < 2:
            continue

        ttb_winner = None
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        if ttb_winner == tally_winner:
            continue  # only conflict trials

        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        n_total += 1.0
        if resp == tally_winner:
            n_match += 1.0

    if n_total == 0:
        return float('nan')
    return float(n_match / n_total)

### RESULTS
- Predicted under pi_1 (simulated): 0.1530 (var=0.0129)
- Predicted under pi_2 (simulated): 0.8660 (var=0.0083)
- Observed on real data: 0.7750 (var=0.0072)

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
  "interpretation": "The two experiments are jointly inconsistent with either heuristic as currently written.\n\nExperiment 1 (features presented in descending-validity order, position = validity rank): observed metric +0.093 (var 0.006). Take-The-Best predicts ~0 (0.004); Tallying predicts -0.69. The data land essentially on the TTB prediction and are ~0.79 away from Tallying. Note the metric is a *difference* of rates, so it cannot by itself confirm that subjects were confident, but any compensatory integration (tallying, or even validity-weighted WADD, which on 4 of the 6 conflict pairs sides against the top cue) would have driven this strongly negative. It did not go negative. So Exp 1 rules out compensatory integration in the direction Tallying requires.\n\nExperiment 2 (validities deliberately scrambled relative to screen position: f5=0.95, f0=0.90, ...): observed 0.775 tally-consistency on margin>=2 conflict trials. Validity-TTB predicts 0.153, Tallying 0.866. Here the data look Tallying-like and are far from TTB.\n\nNeither theory is task-invariant. But the discrepancy has an obvious structural signature: the only thing that changed between experiments is whether presentation position tracks validity. Consider a lexicographic rule that scans cues in *screen position* order rather than stated-validity order. In Exp 1 position == validity rank, so it reproduces TTB exactly (metric ~0, matching +0.09). In Exp 2 I checked all five metric-relevant conflict trials (1, 2, 3, 9, 10): in every one the leftmost discriminating cue points to the *tally* winner, so a positional lexicographic rule scores 1.0 on that metric \\u2014 and 0.775 is exactly what noisy/mixed positional scanning looks like. A single mechanism (order-of-inspection driven by presentation position, with steeply decaying weight on later-inspected cues) reproduces BOTH observations, whereas validity-TTB explains only Exp 1 and Tallying only Exp 2. This is a classic case where both incumbents fit one experiment each and a new, nested theory should be introduced.\n\nTallying is the more decisively falsified of the two (its Exp 1 miss is 0.79 and it makes a parameter-free wrong-direction prediction there), so it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 (Tallying) with a POSITION-DRIVEN, PRIMACY-WEIGHTED EVIDENCE ACCUMULATION theory, which nests both incumbents and can explain both experiments.\n\nCore claim: subjects inspect the expert ratings in the order they are displayed (left to right / reading order) rather than reordering them by the stated validities. Evidence from earlier-inspected cues is weighted far more heavily than later cues, so decisions are effectively driven by the first cue(s) that discriminate, but weighting is graded rather than strictly lexicographic. Stated validities exert only partial influence on the inspection order/weights.\n\nSuggested model instantiation:\n  score(A) - score(B) = sum_j w_j * (a_j - b_j),\n  w_j = exp(-lambda * rank_j),\n  rank_j = gamma * (rank of feature j by descending stated validity) + (1 - gamma) * (screen position index j),\n  choice via softmax with inverse temperature beta plus lapse epsilon.\n\nParameter ranges: lambda in [0, 3] (0 = pure Tallying/equal weights; large = one-reason lexicographic), gamma in [0, 1] (1 = validity-ordered TTB, 0 = pure position/reading-order scanning), beta in [0.5, 8], epsilon in [0, 0.4]. The Exp 1 + Exp 2 data jointly imply moderate-to-large lambda with gamma near 0 (or a mixture), which yields metric ~0 in Exp 1 and ~0.8-1.0 in Exp 2.\n\nThis differs from Theory 1 (validity-ordered, strictly one-reason, ignores position) and from the discarded Theory 2 (order-blind, equal-weight counting) by making the *presentation order* the causal driver of which reason is used, and by allowing graded rather than all-or-none reliance on the first discriminating cue.\n\nA good next experiment for this contrast: hold the pair structure fixed and manipulate only the on-screen ordering of the same experts (validity-sorted vs. scrambled vs. reverse-sorted); Theory 1 predicts invariance, the new theory predicts choice reversals on top-cue-vs-tally conflict pairs."
}
```

## Usage

```json
{
  "input_tokens": 10265,
  "output_tokens": 5884
}
```
