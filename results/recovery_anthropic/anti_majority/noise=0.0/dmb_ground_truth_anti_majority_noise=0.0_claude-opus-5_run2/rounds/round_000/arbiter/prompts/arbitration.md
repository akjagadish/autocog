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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=14):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 11: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 14: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** Five experts with well-spread validities (0.90, 0.80, 0.70, 0.60, 0.55) so that the validity ordering is unambiguous and TTB's cue cascade is well defined. The design contains three diagnostic trial classes plus controls, each presented in both A/B mirror forms to remove side bias.

(1) CONFLICT trials (pairs 1-6): the highest discriminating cue favors one option while the majority of remaining cues favors the other. E.g. A=[1,0,0,0,0] vs B=[0,1,1,1,1]: TTB stops on expert 1 and picks A with probability ~sigmoid(beta), whereas Tallying counts 1 vs 4 and picks B with probability ~sigmoid(3*beta) \u2014 the two theories predict opposite responses, and Tallying's tally margin (3) is large so its prediction is nearly deterministic for any moderate beta. Pairs 3-4 use a smaller tally margin (1 vs 2), so Tallying predicts a *weaker* (margin-1 softmax) preference for the option TTB rejects; the graded difference across margin sizes is a quantitative signature that separates the models beyond mere direction.

(2) TALLY-TIE trials (pairs 7-10): the feature-wise wins are exactly balanced, so Tallying is forced to 50/50 regardless of beta, while TTB still discriminates decisively. Pairs 7-8 tie 2-2 with the top cue discriminating (TTB confident); pairs 9-10 tie 1-1 with all three high-validity cues equal so TTB is driven by the low-validity expert 4. If subjects show systematic (non-chance) preferences on these trials, Tallying cannot account for it at any parameter setting.

(3) CONTROL trials (pairs 11-14) where both theories agree, providing clean leverage on beta and epsilon so that the diagnostic trials are interpreted with well-identified noise parameters.

Pair 5-6 additionally makes the top cue non-discriminating (both options rated 1 by expert 1), forcing TTB down to expert 2 while Tallying reads a 1-vs-3 margin the other way \u2014 testing whether people truly stop at the first *discriminating* cue rather than integrating.

Advocated theory (TTB) predicts: choice of the option favored by the highest discriminating cue on ALL of pairs 1-10, with roughly constant accuracy independent of how many lower cues oppose it, and above-chance systematic choice on the tally-tie pairs. Tallying predicts the opposite direction on pairs 1-6 with strength graded by tally margin, and exact chance on pairs 7-10.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



### METRIC
Rationale:
The design deliberately contains conflict pairs (1-6) in which the first discriminating cue in validity order favors one option while the feature-wise tally favors the other. Restricting to exactly those trials where the two heuristics make opposite, well-defined predictions, the metric reports the proportion of choices consistent with the top discriminating cue (Take The Best).

Under TTB this proportion is (1-eps)*sigmoid(beta) + eps/2, i.e. near 0.85-1.0 for most parameter draws; under Tallying it is the complement, 1 - [(1-eps)*sigmoid(m*beta) + eps/2], i.e. near 0.0-0.15, since the tally margins on these pairs (3, 1, 2) all point the other way. The expected mean gap is roughly 0.75, while the per-subject spread comes only from lapse/temperature draws and binomial noise over ~36 conflict trials per subject, so between-subject variance should be small (on the order of 0.005-0.01). This makes the contrast large in mean and tight per subject.

I deliberately exclude the tally-tie trials (pairs 7-10), where Tallying is at chance rather than opposite, because including them would pull Tallying's value toward 0.5 and shrink the gap. I also exclude control trials where the heuristics agree, since they carry no discriminative information. The metric is well defined on a single subject's slice and on the pooled frame alike.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def ttb_winner(a, b):
        # features are ordered by descending validity (index 0 = 0.9, ... index 4 = 0.55)
        for j in range(len(a)):
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    def tally_winner(a, b):
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        if a_wins > b_wins:
            return 0
        if b_wins > a_wins:
            return 1
        return -1

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row["option_a_ratings"]]
            b = [int(v) for v in row["option_b_ratings"]]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        w_ttb = ttb_winner(a, b)
        w_tal = tally_winner(a, b)
        # keep only trials where the two heuristics make opposite, well-defined predictions
        if w_ttb < 0 or w_tal < 0 or w_ttb == w_tal:
            continue
        resp = row["response"]
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        n += 1
        if int(resp) == w_ttb:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)

### RESULTS
- Predicted under pi_1 (simulated): 0.8172 (var=0.0109)
- Predicted under pi_2 (simulated): 0.1522 (var=0.0104)
- Observed on real data: 0.7744 (var=0.0262)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.9, 0.82, 0.74, 0.66, 0.58, 0.53]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 8: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 10: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 14: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 15: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]

**Rationale:** I advocate TALLYING and design for a quantitative dissociation that the previous experiment (which emphasized direction reversals only) did not isolate: the *graded* dependence of choice probability on the tally margin versus TTB's flat, margin-independent confidence.

Six experts with clearly ordered, well-spread validities (0.90, 0.82, 0.74, 0.66, 0.58, 0.53) make TTB's cue cascade unambiguous (cue order = feature order). Every pair appears in both A/B mirror forms (16 unique pairs, K=6 -> 96 trials) so side bias cancels.

(1) GRADED-AGREEMENT trials (pairs 1-6). Both models favour the same option, but the tally margin is +1 (3 wins vs 2), +3 (4 vs 1), and +5 (5 vs 0). TTB always stops at expert 1 and therefore predicts EXACTLY the same accuracy, (1-eps)*sigmoid(beta)+eps/2, in all three conditions; its binary score never changes. Tallying predicts monotonically increasing accuracy, sigmoid(1*beta) < sigmoid(3*beta) < sigmoid(5*beta). Because these conditions also pin down beta and epsilon, a reliable accuracy gradient across margin 1 -> 3 -> 5 cannot be absorbed by any TTB parameter setting: TTB is forced to attribute all three to one sigmoid(beta). This is the core quantitative signature, and it is measured on trials where the two theories agree in DIRECTION, so it is not contaminated by any direction-reversal strategy shift.

(2) CONFLICT trials (pairs 7-12). Expert 1 favours one option while the remaining experts favour the other, with tally margins of -1 (2 vs 3), -3 (1 vs 4) and -4 (1 vs 5). TTB predicts choice of the expert-1 option with the SAME probability in all three; Tallying predicts choice of the opposite option with strength graded by margin. Direction and gradient are both diagnostic here, and the -1 case is the crucial one: Tallying predicts a weak reversal (sigmoid(beta)) exactly matching in magnitude what it predicts for the +1 agreement pairs.

(3) DESCENT trial (pairs 13-14): experts 1 and 2 both rate the two products 1, so TTB must descend to expert 3 and choose that option, while the tally is 1 vs 3 the other way. This tests whether people truly stop at the first discriminating cue or integrate across the three low-validity experts.

(4) TALLY-TIE trial (pairs 15-16): A wins on experts 1-2, B wins on experts 3-4 (2-2 tie). Tallying is forced to exactly 50/50 for any parameters; TTB predicts a decisive expert-1-based preference. This is the competing theory's unique prediction and provides a hard falsification test of my advocated model; including it makes the design fair rather than stacked.

Predictions if Tallying is right: accuracy rises steeply with |tally margin| in both the agreement and conflict blocks, choices reverse against expert 1 on pairs 7-12 and 13-14, and choices sit at chance on pairs 15-16. If TTB is right: flat accuracy across all margins, expert-1 option chosen throughout pairs 1-12, expert-3 option on 13-14, and clearly above-chance systematic choice on the tally-tie pairs.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This is a within-subject DIFFERENCE score that cancels the nuisance parameters (beta, epsilon) for the competing theory while remaining maximally large for mine.

Two trial sets are extracted purely from the stimuli:
(1) AGREEMENT set: trials where the top discriminating cue and the tally point to the same option, with a large tally margin (|margin| >= 3) -- pairs 3/4 (4 vs 1) and 5/6 (5 vs 0).
(2) CONFLICT set: trials where the first discriminating cue points one way and the tally the other, with |margin| >= 2 -- pairs 9/10 (1 vs 4), 11/12 (1 vs 5), and 13/14 (TTB must descend to expert 3 while the tally is 3 vs 1 the other way).

The metric is P(choose the TTB-favoured option | agreement) - P(choose the TTB-favoured option | conflict).

Under Take The Best, the choice probability for the first-discriminating-cue option is exactly (1-eps)*sigmoid(beta) + eps/2 in BOTH sets, independent of tally margin, so the difference is ~0 for every parameter setting; per-subject deviations come only from binomial sampling over 24 + 36 trials (sd ~0.08), so it is tight around zero regardless of how beta and epsilon are drawn.

Under Tallying, the same quantity is (1-eps)*sigmoid(k*beta)+eps/2 in the agreement set and its complement in the conflict set, giving a difference approaching (1-eps) -- typically 0.6-1.0 and never negative.

Thus the two theories are separated by a large mean gap with modest, mostly lapse-driven between-subject variance, and the contrast is not trivially true for Tallying: if humans stop at the first valid cue the score sits at zero.

Source:
def metric(data: pd.DataFrame) -> float:
    agree_hits = []
    conflict_hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        diff = a - b
        nz = np.nonzero(diff)[0]
        if nz.size == 0:
            continue
        # features are listed in descending validity order in this design
        ttb_choice = 0 if diff[nz[0]] > 0 else 1
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin == 0:
            continue
        tally_choice = 0 if margin > 0 else 1
        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        chose_ttb = 1.0 if resp == ttb_choice else 0.0
        if tally_choice == ttb_choice and abs(margin) >= 3:
            agree_hits.append(chose_ttb)
        elif tally_choice != ttb_choice and abs(margin) >= 2:
            conflict_hits.append(chose_ttb)
    if len(agree_hits) == 0 or len(conflict_hits) == 0:
        return 0.0
    return float(np.mean(agree_hits) - np.mean(conflict_hits))

### RESULTS
- Predicted under pi_1 (simulated): -0.0008 (var=0.0076)
- Predicted under pi_2 (simulated): 0.7178 (var=0.0313)
- Observed on real data: -0.6339 (var=0.0508)

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
  "interpretation": "Across the two experiments the observed data are internally consistent but neither theory captures both.\n\nExperiment 1 (conflict pairs, proportion of choices matching the top discriminating cue): observed 0.774. TTB (pi_1) predicted 0.817 \u2014 an excellent match. Tallying (pi_2) predicted 0.152 \u2014 the exact opposite pole, a ~0.62 error. Clear win for pi_1 here.\n\nExperiment 2 (P(TTB option | agreement, |margin|>=3) \u2212 P(TTB option | conflict, |margin|>=2)): observed \u22120.634 with modest between-subject variance (var=0.051, sd\u22480.23), so it is a robust, subject-consistent effect. TTB predicted \u22480 (its score is margin-independent, so the difference must vanish); Tallying predicted +0.718. The observed value is large and NEGATIVE \u2014 i.e. it lies outside the range either theory can produce for ANY parameter setting. TTB is off by 0.63 (\u22487 SDs of its own between-subject spread); Tallying is off by 1.35 and has the wrong sign too.\n\nWhat generates a negative score? Decomposing: the conflict set (pairs 9-14) shows high choice of the top-cue option (consistent with Exp 1's 0.77), while the agreement set (pairs 3/4: 4-vs-1; pairs 5/6: 5-vs-0 dominance) shows LOW choice of the option favoured by both heuristics \u2014 subjects preferred the option with fewer positive expert ratings even when it was dominated. That is exactly the pattern produced by an integrative rule with reversed polarity: 'choose the option with LESS total positive evidence'. Check this against Exp 1: in every conflict pair the top-cue winner is by construction the tally loser, so an anti-tally (or anti-WADD) rule chooses the top-cue option on all of pairs 1-6, yielding \u22480.85 before lapses \u2014 matching the observed 0.774. Applied to Exp 2, the same rule yields \u2248\u22120.7 to \u22121.0 \u2014 matching the observed \u22120.634. In other words, the data are quantitatively reproduced by 'Tallying (or validity-weighted additive integration) with the sign flipped', which is why pi_1 looks right on Exp 1 (where anti-tally and TTB coincide by design) and fails on Exp 2 (where they diverge, on the agreement/dominance trials that pi_1 predicts should be identical to conflict trials).\n\nSo pi_1 is the better of the two overall (mean absolute error 0.33 vs 0.99), but it is fundamentally unable to produce the margin/agreement-dependent reversal in Exp 2, because its binary one-reason score is flat across margins by construction. pi_2 has the right structural machinery (compensatory, margin-graded) but the wrong direction on every diagnostic trial in both experiments, so it is the degenerate one and should be replaced by a theory that keeps compensatory integration but allows the evidence polarity to be negative.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_2 (Tallying) with a SIGNED COMPENSATORY INTEGRATION theory \u2014 call it 'Polarised Weighted Evidence Accumulation'.\n\nCore claim: subjects integrate ALL cues into a single scalar for each option, score_i = sum_j w_j * s * x_ij, where x_ij is the binary rating, w_j is a validity-derived weight, and s is a POLARITY parameter that expresses how the subject maps an expert endorsement onto desirability. Crucially s is allowed to be negative: in this task population the endorsements are treated as counting AGAINST the option (a '1' is read as a red flag / warning / the response mapping is systematically inverted), so the option with the smaller weighted sum of 1s is chosen. Choice is a softmax over the two scores with inverse temperature beta plus a lapse epsilon.\n\nStructural details to implement:\n- weights w_j = ((v_j - 0.5)/0.5)^gamma normalised, with gamma a steepness parameter: gamma\u22480 gives equal weighting (pure tallying), large gamma gives near-lexicographic (TTB-like) behaviour. This lets the SAME family nest both incumbent heuristics as special cases and makes the theory a genuinely stronger competitor to pi_1 rather than an antipode of it.\n- polarity s in [-1, +1] (or a signed scalar); the default/prior mass should sit on s < 0, since both experiments demand it.\n- Softmax over the *difference* of weighted sums, so choice probability is graded by evidence margin (this is what Exp 2's design isolates and what TTB cannot produce).\n\nWhy this fits everything: with s<0 and small-to-moderate gamma (compensatory), Exp 1's conflict pairs 1-6 all have the top-cue winner as the tally loser, so the model chooses the top-cue option \u224885% of the time (observed 0.774 with a lapse of ~0.15-0.25); Exp 2's agreement/dominance pairs 3-6 are chosen AGAINST the dominant option (low P(TTB)) while conflict pairs 9-14 still favour the single-1 option (high P(TTB)), giving a difference of roughly \u22120.6 to \u22120.8 (observed \u22120.634). TTB gives 0 there and Tallying gives +0.72; only signed compensatory integration reaches the observed value.\n\nSuggested parameter ranges for the first model instantiation: s \u2208 [-1.0, -0.4] (mostly negative, small tail near 0 to keep falsifiability), gamma \u2208 [0, 1.5], beta \u2208 [0.5, 3.0] (per unit of weighted margin), epsilon \u2208 [0.05, 0.30]. The predict function must return exactly 0.5/0.5 when the weighted scores tie, and must remain margin-sensitive (do NOT collapse to a binary winner score), because the margin gradient is the key discriminator against pi_1 in future experiments."
}
```

## Usage

```json
{
  "input_tokens": 10582,
  "output_tokens": 9059
}
```
