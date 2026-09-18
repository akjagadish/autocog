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
**Validities (n_features=5):** [0.92, 0.8, 0.7, 0.6, 0.53]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 10: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 13: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 14: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** Goal: dissociate Take-The-Best (one-reason, cue-order-driven, magnitude-blind) from Tallying (equal-weight vote count, margin-graded) both in *direction* and in *graded magnitude* of choice probabilities.

Five experts with clearly spread validities (0.92, 0.80, 0.70, 0.60, 0.53) so the cue cascade is unambiguous and subjects are told expert 1 is much better than experts 3-5.

The 16 unique pairs (each repeated 6x, 96 trials) fall into five diagnostic families, each mirrored so that neither response key is systematically correct:

(1) DIRECTIONAL CONFLICT WITH GRADED TALLY MARGIN. Pairs 1-6: the option winning the top (92%) cue loses on all remaining cues, with the tally margin varied at 3 (1 vs 4 wins), 2 (1 vs 3 wins, one tie) and 1 (2 vs 3 wins). TTB predicts the SAME high probability of choosing the top-cue option in all three, independent of margin; Tallying predicts the OPPOSITE option with a probability that increases monotonically with margin (sigmoid(beta*margin)). This is the core quantitative dissociation: a flat-versus-graded profile across margin, plus opposite signs.

(2) TALLY-TIE / TTB-DISCRIMINATES. Pairs 7-8: A=[1,0,1,0,0] vs B=[0,1,0,1,0]. Feature wins are 2-2, so Tallying must guess (exactly 50/50 regardless of beta), while TTB is driven deterministically by the 92% cue. Any systematic departure from 50% here is fatal to Tallying.

(3) TTB-STRONG / TALLY-WEAK ON A SINGLE LOW CUE. Pairs 9-10: the options tie on the four top cues and differ only on the 53% cue. TTB is fully committed (its stopping rule gives the winner probability sigmoid(beta), the same as for a top-cue win), whereas Tallying only has a margin of 1, its weakest signal. Comparing accuracy here against family (5) below separates the two models' beta scalings.

(4) DEEP-CUE CASCADE CONFLICT. Pairs 13-14: options tie on the 92% cue, the 80% cue favors one option, and the three lower cues all favor the other. TTB stops at cue 2 and chooses against a 3-vs-1 tally. This checks that the cascade (not just the single best cue) governs choice.

(5) AGREEMENT ANCHORS. Pairs 11-12 (4 wins vs 1 win, same direction for both models) and 15-16 (only the top cue discriminates, all else tied) provide low-conflict baselines that pin down epsilon and beta for both models so the conflict trials are interpreted against a calibrated noise level.

Predicted pattern if TTB is right: near-ceiling choice of the top-cue option in families 1-4 with NO modulation by tally margin, and above-chance choice in the tally-tie pairs. If Tallying is right: choice reverses in family 1, scales with margin (m=1 < m=2 < m=3), is at chance in family 7-8, and is weak in pairs 9-10.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains 8 unique pairs (trials 1-6, 13-14; 48 of the 96 trials) in which the option favored by the first discriminating cue in the validity cascade is exactly the option that LOSES the equal-weight tally. On these conflict trials the two theories make opposite directional predictions with no overlap: TTB chooses the cue-cascade winner with probability (1-eps)*sigmoid(beta)+eps/2 (expected ~0.84 across the parameter range, always >= 0.5), while Tallying chooses that same option with probability 1 - [(1-eps)*sigmoid(beta*margin)+eps/2] (expected ~0.16, always <= 0.5). The metric is the proportion of conflict trials on which the subject picked the TTB-favored (tally-losing) option. It is computed from the stimuli themselves, not from a model fit, and averages over 48 trials per subject, so per-subject binomial noise is tiny; the only across-subject spread comes from individual noise levels (beta, epsilon), which shifts both theories symmetrically toward 0.5 but can never make them cross it. The expected between-theory gap (~0.68) dwarfs the per-subject SD (~0.1), giving a very large Welch t at N=25. Crucially, the metric is not trivially true of TTB: any subject who weights cues equally, or who is swayed by the 3-4 vs 1 feature majority, drives it below 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # Take-The-Best winner: first discriminating cue in validity order.
        # Feature columns are already ordered by descending validity in this design.
        ttb_winner = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        # Tallying winner: majority of feature-wise wins (ties excluded).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        if tally_winner == ttb_winner:
            continue  # not a conflict trial
        resp = row['response']
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        hits += 1.0 if int(resp) == ttb_winner else 0.0
        n += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)

### RESULTS
- Predicted under pi_1 (simulated): 0.8375 (var=0.0108)
- Predicted under pi_2 (simulated): 0.1425 (var=0.0084)
- Observed on real data: 0.7533 (var=0.0228)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.93, 0.58, 0.86, 0.51, 0.74, 0.65]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 2: A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  trial 3: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0]
  trial 5: A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  trial 7: A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  trial 9: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 11: A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  trial 12: A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 13: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 15: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 16: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Rationale:** GOAL: dissociate Tallying (advocated) from Take-The-Best (competing) by exploiting the fact that TTB's choice probability is *constant* (sigmoid(beta)) whenever ANY cue discriminates, whereas Tallying's is a graded function of the vote margin, sigmoid(beta*m). The previously explored design put the gradation manipulation only inside directional-conflict trials; here the core diagnostic is a *no-conflict* margin ladder, which cleanly separates the two models on magnitude alone (no direction confound), and it is cross-validated against conflict and tally-tie families that separate them on direction.

Six experts with widely spread, deliberately non-monotonic validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65. TTB's cue cascade is therefore f0 > f2 > f4 > f5 > f1 > f3, which is NOT the display order, so cue-identity effects cannot be mimicked by a simple left-to-right reading strategy. 16 unique pairs, each repeated 6x (96 trials), fully key-balanced (every pair appears with A and B swapped).

FAMILY C 	rairairs 1-8): MARGIN LADDER WITHOUT CONFLICT. The 93% cue always favors the same option that wins the tally, and the tally margin takes the values m = 1, 2, 3, 4 (each in both key directions). TTB predicts EXACTLY the same accuracy at m=1 as at m=4 (flat line, sigmoid(beta) after lapse). Tallying predicts a steeply rising curve, e.g. at beta=0.6 roughly .65 / .77 / .86 / .92. This is the central quantitative signature: a flat-versus-monotone accuracy profile over margin with the direction of choice held constant, so no directional confound can explain it.

FAMILY D (pairs 9-12): TALLY TIES, TTB DISCRIMINATES. Pairs 9-10 are 2-2 splits where the 93% cue discriminates; pairs 11-12 are 2-2 splits where f0 and f2 TIE and the first discriminating cue is the 74% expert (with the 65%, 58% and 51% experts split against it). Tallying must be exactly 50/50 in all four, irrespective of beta and of which cue is on top. TTB predicts systematic above-chance choice of the top-discriminating-cue option, and — importantly — the SAME magnitude for the f0-driven and f4-driven pairs. Any deviation from chance here falsifies Tallying; equality of the f0- and f4-driven deviations is a further TTB-specific prediction.

FAMILY E (pairs 13-16): GRADED DIRECTIONAL CONFLICT. The 93% cue favors one option while the tally favors the other, with tally margin 4 (pair 13/14: 5 wins vs 1) or 2 (pair 15/16: 3 wins vs 1). TTB predicts the same probability of choosing the f0-winner in both, and this probability must equal its Family-C and Family-D accuracy (all are sigmoid(beta)). Tallying predicts the opposite option, with a much larger effect at m=4 than at m=2, and with beta already pinned down by the Family-C ladder — so the conflict trials are a genuine out-of-sample prediction rather than a free fit.

DECISION RULE: If Tallying is correct, accuracy in Family C rises monotonically with margin, Family D is at chance, and Family E reverses against the 93% expert more strongly at m=4 than m=2, with all effects captured by a single (beta, epsilon). If TTB is correct, one flat probability level sigmoid(beta) accounts for Families C, D (top-cue side) and E (top-cue side) simultaneously, with zero margin modulation anywhere.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the proportion of choices that go to the option favoured by the first (highest-validity) discriminating cue, restricted to exactly those trials where the tally does NOT back that option: the 2-2 tally ties (Family D, pairs 9-12) and the graded directional-conflict pairs (Family E, pairs 13-16). This isolates the only place the two heuristics make qualitatively different, non-overlapping commitments, and it deliberately excludes the Family-C margin ladder, whose contrast vanishes whenever beta is large (both models then predict the same flat near-ceiling accuracy), which would only dilute the signal and add noise.

Predicted values: under Take-The-Best every included trial has a discriminating top cue, so the model chooses that option with probability (1-eps)*sigmoid(beta)+eps/2 on ALL of them, giving a metric near 0.85-0.9 for typical parameters and only mildly modulated by the lapse rate. Under Tallying the same trials split into two blocks with predictions that are both far below that: the tally-tie trials force exactly 0.5 regardless of beta and epsilon, and the conflict trials force the OPPOSITE choice (probability eps/2, i.e. near 0.1), so the pooled value sits around 0.3. The expected between-theory gap is roughly 0.55 on a 0-1 scale.

Variance is kept tight because (a) 48 of the 96 trials qualify, so binomial noise per subject is small; (b) under Tallying the tie block is parameter-free (exactly chance for every subject) and the conflict block saturates near zero, so subject-to-subject spread is driven only by the lapse rate; (c) under TTB the value is a single sigmoid-plus-lapse number, again with limited spread. Crucially, the metric is not trivially true for Tallying: Tallying could in principle score anywhere from 0 to 1 here, and any systematic above-chance preference for the top-cue option on the 2-2 splits (which Tallying must call at exactly 50/50) pushes the metric toward the TTB value.

Source:
def metric(data: pd.DataFrame) -> float:
    # Fixed design validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65
    val = np.array([0.93, 0.58, 0.86, 0.51, 0.74, 0.65], dtype=float)

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape[0] != b.shape[0]:
            continue
        if a.shape[0] == val.shape[0]:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(a.shape[0])

        tally = int(np.sum(a > b)) - int(np.sum(b > a))

        winner = None
        for j in order:
            j = int(j)
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no cue discriminates: uninformative for both models

        ttb_sign = 1 if winner == 0 else -1
        # Keep only trials where the tally does NOT favour the TTB winner:
        # either the tally is exactly tied (Family D) or it points the
        # other way (Family E, directional conflict).
        if tally * ttb_sign > 0:
            continue

        n += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == winner:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)

### RESULTS
- Predicted under pi_1 (simulated): 0.8204 (var=0.0123)
- Predicted under pi_2 (simulated): 0.3058 (var=0.0050)
- Observed on real data: 0.5100 (var=0.0136)

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
  "interpretation": "Neither heuristic is task-invariant. In Experiment 1 the observed conflict-trial metric was 0.753, very close to TTB's 0.838 and catastrophically far from Tallying's 0.143 \\u2014 subjects overwhelmingly followed the highest-validity discriminating cue when the opposing tally margin was small-to-moderate (1\\u20133 net votes among 5 cues). In Experiment 2 the observed value was 0.510, which is 0.31 away from TTB's 0.820 and 0.20 away from Tallying's 0.306; here neither theory is right, but pure TTB is clearly disconfirmed: with 6 cues and conflict margins of 5-vs-1 and 3-vs-1, subjects abandoned the top cue about half the time. Decomposing the Exp-2 metric makes the picture coherent: it pools Family D (2-2 tally ties, where Tallying must be exactly 0.5 and TTB must be ~0.85) with Family E (large-margin directional conflicts, where TTB must be ~0.85 and Tallying ~0.1). A pooled 0.51 is most naturally produced by subjects who are above chance on the tie trials (killing pure Tallying) but who defect to the many-cue majority on the big-margin conflicts (killing pure TTB). That is exactly the signature of a compensatory, validity-weighted integration rule: the top cue carries a big but finite weight that can be outvoted when enough moderately valid cues line up against it. A quick check confirms this: a weighted-additive score with weights that are a convex (superlinear) function of validity predicts near-ceiling TTB-consistent choice on all Exp-1 conflict pairs (all of which the top cue's weight still dominates) yet predicts a reversal on Exp-2 pairs 13/14 (five cues, including the .86 and .74 experts, against the .93 expert) and near-chance on pairs 11/12 (a virtual weight tie), yielding a pooled Exp-2 value in the 0.5\\u20130.65 range. That single mechanism, with one weighting parameter, reconciles both experiments; neither pure one-reason nor pure equal-weight can. Between the two current theories, Tallying (pi_2) is the worse performer overall (absolute errors 0.61 and 0.20 vs TTB's 0.08 and 0.31), and its parameter-free 0.5 prediction on tally ties is contradicted in both designs. It should be replaced rather than merely re-fit \\u2014 no choice of beta/epsilon can move Tallying off 0.143 in Experiment 1.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Tallying with a compensatory VALIDITY-WEIGHTED ADDITIVE model (WADD / Franklin's rule with a validity-sensitivity exponent), which nests and strictly generalises both current competitors. Description sketch: people evaluate each option by summing the values of its cues, each weighted by its (subjectively transformed) validity, and choose the option with the higher weighted sum; only the difference in weighted sums matters, so a single highly valid cue can be outvoted if enough moderately valid cues oppose it, but small numbers of weak cues cannot overturn a strong cue. Concretely: w_j \\u221d (2*v_j - 1)^gamma (or equivalently logit(v_j)^gamma), normalised; score_i = sum_j w_j * x_ij; p = softmax(beta * (score_A - score_B)) mixed with lapse epsilon. Parameter ranges: gamma in [0.5, 4] (gamma -> large reproduces lexicographic/TTB behaviour, gamma -> 0 reproduces equal-weight Tallying, intermediate gamma is the genuinely new regime), beta in [1, 15] scaled to the normalised weight difference, epsilon in [0, 0.2]. Key falsifiable commitments that distinguish it from BOTH incumbents: (a) on tally-tie trials where the weighted difference is non-zero (Exp-2 pairs 9/10) it predicts strongly above chance, unlike Tallying; (b) on trials where the weighted sums nearly cancel (Exp-2 pairs 11/12, 13/14) it predicts near-chance or reversal against the top cue, unlike TTB; (c) it predicts graded, weight-margin-dependent accuracy rather than TTB's flat sigmoid(beta) across all discriminating trials. When testing it, use a metric that is per-family rather than pooled (e.g. accuracy on TTB-winner as a function of the opposing weighted margin) so its graded prediction is separable from TTB's flat one. If future data show bimodality across subjects (large between-subject variance with individuals at ceiling or at floor), a strategy-mixture variant (a fraction of subjects lexicographic, the rest compensatory) should be considered as the next challenger."
}
```

## Usage

```json
{
  "input_tokens": 11193,
  "output_tokens": 5165
}
```
