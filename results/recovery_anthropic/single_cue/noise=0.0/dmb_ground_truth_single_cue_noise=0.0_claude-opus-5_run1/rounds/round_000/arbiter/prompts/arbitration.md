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
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 7: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 9: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 10: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Rationale:** Five experts with clearly spread, strictly ordered validities (0.95, 0.85, 0.75, 0.65, 0.55) so the TTB cue cascade is unambiguous and matches feature index order. The 12 unique pairs (repeated 8x each = 96 trials) are built to dissociate one-reason decision making from unit-weight tallying along three quantitative dimensions:

(1) CONFLICT MAGNITUDE. Pairs 1-4 pit the single highest-validity cue against a coalition of lower cues, varying the tally margin from 4 (A=[1,0,0,0,0] vs B=[0,1,1,1,1]) to 1 (A=[1,0,0,0,0] vs B=[0,1,1,0,0]). TTB predicts identical choice probabilities across these pairs (always the option winning cue 1, with p fixed by beta and epsilon), whereas Tallying predicts an ordered gradient: near-deterministic choice of the coalition option at margin 4 and a shallower, beta-scaled preference at margin 1. This is a quantitative signature, not merely a directional one: TTB is flat over margins, Tallying is monotone increasing.

(2) TALLY-TIE TRIALS. Pairs 5-6 (A=[1,0,1,0,0] vs B=[0,1,0,1,0] and its mirror) give each option exactly two feature wins, so Tallying must guess at exactly 50/50 regardless of its parameters, while TTB is fully decisive in favour of the option winning cue 1. Any systematic deviation from chance here cannot be produced by Tallying at all, making these trials a parameter-free falsification test.

(3) CASCADE POSITION. Pairs 9-11 place the first discriminating cue lower in the cascade (cue 2 after a tie on cue 1; cue 3 after ties on cues 1-2) while the remaining lower cues favour the other option. TTB still predicts the same high, constant probability for the option winning the first discriminating cue; Tallying predicts the opposite choice. If subjects' accuracy on the discriminating-cue option degrades as that cue moves down the cascade, that also constrains the subjective-validity assumption inside TTB.

Pairs 7-8 are agreement controls where both heuristics select the same option, giving a baseline for lapse rate (epsilon) and beta that is uncontaminated by the model contrast. Every pair and its mirror image appear so that the correct response is balanced across the A and B keys for both models, ruling out side bias as an explanation. No pair is degenerate: on every trial at least one model is decisive, and on 10 of 12 the models disagree either in direction or in predicted confidence.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains three families of trials on which a lexicographic one-reason rule and a unit-weight counting rule make opposite (or differently determinate) commitments: (i) pairs 1-4, where the single top cue is pitted against a coalition of 1-4 lower cues; (ii) pairs 9-12, where the first discriminating cue sits lower in the cascade and all remaining cues favour the other option; and (iii) pairs 5-6, where the tally is exactly even so a counting rule must guess at 50/50 whatever its parameters, while the cue cascade is fully decisive. The metric pools exactly these trials (agreement controls 7-8 are excluded, since both rules pick the same option there and they only dilute the contrast) and reports the proportion of choices going to the option favoured by the first discriminating cue. Under Take The Best this is (1-eps)*sigmoid(beta)+eps/2 on every included trial, i.e. high (~0.85 averaged over the parameter ranges) and tight, because each subject contributes 80 trials. Under Tallying it is the complement of the tally-consistent probability on the directional-conflict trials (~0.09) and exactly 0.5 on the two tie pairs, giving ~0.17 overall. Both theories' per-subject spread is driven mainly by the lapse rate, which shifts the two values toward 0.5 by comparable, modest amounts, so the ~0.68 mean gap dwarfs the between-subject variance. Mirror-image pairs are included in equal numbers, so any A/B key bias cancels and cannot inflate or deflate the score for either theory.

Source:
def metric(data: pd.DataFrame) -> float:
    """One-reason adherence on model-disagreement trials.

    For every trial we compute (a) the option favoured by the FIRST
    discriminating cue in validity order (features are listed in
    descending validity: 0.95, 0.85, 0.75, 0.65, 0.55) and (b) the option
    favoured by the unit-weight tally of feature-wise wins.  We keep only
    trials where these two disagree in direction OR where the tally is tied
    (so a counting rule must guess).  The metric is the proportion of such
    trials on which the subject picked the option favoured by the first
    discriminating cue.
    """
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]

        # first discriminating cue (features already in descending validity)
        first_winner = None
        for j in range(n):
            if a[j] > b[j]:
                first_winner = 0
                break
            if b[j] > a[j]:
                first_winner = 1
                break
        if first_winner is None:
            continue  # no cue discriminates: uninformative trial

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None  # counting rule must guess

        if tally_winner is not None and tally_winner == first_winner:
            continue  # agreement control trial: not diagnostic

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hits.append(1.0 if r == first_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_1 (simulated): 0.8515 (var=0.0078)
- Predicted under pi_2 (simulated): 0.1903 (var=0.0042)
- Observed on real data: 0.2820 (var=0.0035)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.6, 0.93, 0.55, 0.87, 0.7, 0.78]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 12: A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 13: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 14: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 15: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 16: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]

**Rationale:** GOAL: dissociate one-reason decision making (TTB) from unit-weight vote counting (Tallying) using (i) a validity order that is deliberately NOT the display order, (ii) a tally-margin gradient over which TTB is flat, (iii) parameter-free tally-tie trials, and (iv) cascade-depth manipulations.

VALIDITIES: six experts shown in display order with validities [0.60, 0.93, 0.55, 0.87, 0.70, 0.78]. The descending cue cascade is therefore feature index 1 (.93) > 3 (.87) > 5 (.78) > 4 (.70) > 0 (.60) > 2 (.55). Because the top cue is Expert 2 rather than Expert 1, any subject who follows a genuine validity cascade must reorder the display, whereas Tallying is completely invariant to the permutation. This also guards against a confound present when validities descend with index: 'use the leftmost cue' can no longer masquerade as TTB.

16 unique pairs, each repeated 6x (96 trials), every pair present together with its A/B mirror so the model-predicted response is exactly balanced across keys (no side-bias explanation).

(1) TALLY-MARGIN GRADIENT (pairs 1-6). One option wins only the top cue (index 1); the other wins a coalition of lower-validity cues, with the tally margin escalating 1 -> 2 -> 4: [0,1,0,0,0,0] vs [1,0,1,0,0,0], vs [1,0,1,0,1,0], vs [1,0,1,1,1,1]. TTB is decisive for the top-cue option with the SAME probability sigmoid(beta)-scaled value on all three, i.e. flat across margins. Tallying reverses direction (choosing the coalition option) and its confidence increases monotonically with margin, from a shallow beta-scaled preference at margin 1 to near-determinism at margin 4. Flat-vs-monotone over margin is a quantitative, not merely directional, signature.

(2) PARAMETER-FREE TALLY TIES (pairs 7-10). Pair 7: [0,1,0,0,0,0] vs [1,0,0,0,0,0] — each option wins exactly one cue, so Tallying MUST be exactly 50/50 for any beta and epsilon, while TTB is decisive for A (cue 1, .93, beats cue 0, .60). Note the naive 'read left to right' strategy predicts the opposite option here, so this pair simultaneously falsifies Tallying and index-order pseudo-TTB. Pair 9: [0,0,1,1,0,0] vs [0,0,0,0,1,1] — a 2-2 tally tie in which TTB's winner is set by a mid-cascade comparison (cue 3, .87, vs cues 5/4), again forcing Tallying to chance while TTB is decisive for A.

(3) CASCADE DEPTH WITH CONSTANT TALLY (pairs 11-14). Pair 11 ([0,0,0,0,0,1] vs [1,0,1,0,1,0]) has the identical tally structure to pair 3 (1 vs 3) but the first discriminating cue sits third in the cascade (index 5) after two ties; pair 13 ([0,0,0,0,1,0] vs [1,0,1,0,0,0]) matches pair 1's margin of 1 but the discriminator is fourth in the cascade (index 4). Tallying predicts exactly the same probabilities as its margin-matched counterparts, while TTB predicts the same high probability for the discriminating-cue option regardless of depth. Any systematic decay in choosing the discriminating-cue option as it moves down the cascade would violate both models as specified and constrain subjective validities inside TTB.

(4) AGREEMENT BASELINE (pairs 15-16). [1,1,1,0,1,0] vs [0,0,0,1,0,1]: the top cue and the tally (4-2) point the same way, so both models predict the same option. These trials identify epsilon and beta free of the model contrast, so the contrast trials can be read as genuine model divergence rather than overall noise.

No pair is degenerate: on 14 of 16 pairs the models disagree in direction, and on the remaining agreement pairs they differ in nothing but supply the noise baseline.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Two diagnostic trial families are contrasted within subject, which cancels overall lapse/noise level and sharpens the model split.

(1) Tally-tie trials (pairs 7/8 and 9/10): each option wins exactly the same number of cues, so Tallying is forced to exactly 50/50 for ANY beta and epsilon, while TTB is decisive (cue .93 beats cue .60; cue .87 beats cues .78/.70). I score the proportion of choices for the TTB-designated winner here: 0.5 under Tallying (only binomial jitter over 24 trials), ~0.85 under TTB.

(2) Large-margin disagreement trials (pairs 3/4, 5/6, 11/12): the tally favours the coalition option by 2-4 cues while TTB's top discriminating cue favours the other option. I score the proportion of choices for the tally winner: high (~0.85-0.95) under Tallying because the softmax is driven by beta*margin, near 0.15 under TTB.

The metric is the difference (2) - (1). Tallying predicts a clearly positive value (~+0.35); TTB predicts a strongly negative value (~-0.70), and because under TTB both terms are driven by the same beta and epsilon they move in opposite directions, amplifying rather than blurring the gap. The tie term is parameter-free for Tallying, so my side's variance is dominated by binomial noise alone, keeping the per-subject estimate tight at N=25.

Source:
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.6, 0.93, 0.55, 0.87, 0.7, 0.78], dtype=float)

    def analyze(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
        val = validities[:n] if validities.shape[0] >= n else np.arange(n, 0, -1, dtype=float)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_winner = 0
        elif margin < 0:
            tally_winner = 1
        else:
            tally_winner = -1
        order = np.argsort(-val, kind='stable')
        ttb_winner = -1
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        return margin, tally_winner, ttb_winner

    info = [analyze(r) for _, r in data.iterrows()]
    margins = np.array([x[0] for x in info], dtype=float)
    tally_w = np.array([x[1] for x in info], dtype=int)
    ttb_w = np.array([x[2] for x in info], dtype=int)
    resp = data['response'].to_numpy().astype(int)

    # Group 1: tally-tie trials where TTB is decisive (parameter-free for Tallying)
    tie_mask = (tally_w == -1) & (ttb_w >= 0)
    # Group 2: trials where tally margin is >= 2 AND TTB points the other way
    dis_mask = (np.abs(margins) >= 2) & (tally_w >= 0) & (ttb_w >= 0) & (tally_w != ttb_w)

    if tie_mask.sum() == 0 and dis_mask.sum() == 0:
        return float('nan')

    if dis_mask.sum() > 0:
        p_tally = float(np.mean(resp[dis_mask] == tally_w[dis_mask]))
    else:
        p_tally = 0.5
    if tie_mask.sum() > 0:
        p_ttb_tie = float(np.mean(resp[tie_mask] == ttb_w[tie_mask]))
    else:
        p_ttb_tie = 0.5

    return p_tally - p_ttb_tie


### RESULTS
- Predicted under pi_1 (simulated): -0.7044 (var=0.0395)
- Predicted under pi_2 (simulated): 0.3828 (var=0.0220)
- Observed on real data: 0.1744 (var=0.0127)

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
  "interpretation": "Across both experiments the observed data sit on the Tallying side of the contrast but are systematically attenuated toward the middle, and neither theory as instantiated captures that pattern.\n\nExperiment 1: TTB predicted 0.85, Tallying 0.19, observed 0.282. Subjects overwhelmingly chose the coalition option over the top-cue option, so pure one-reason decision making is decisively falsified. But the observation is reliably above Tallying's prediction (and the between-subject variance is small, 0.0035, so this is not sampling noise). The excess is exactly where it should be if cue validities are weighted: the metric pools 8 directional-conflict pairs (where a weighted-additive rule agrees with the tally, since 0.95 < 0.85+0.75+...) with 2 tally-tie pairs (5/6: A = .95+.75 = 1.70 vs B = .85+.65 = 1.50), on which a validity-weighted rule is not at chance but leans toward the TTB winner. That lifts the metric from ~0.17 to ~0.25-0.30 \n\nExperiment 2: TTB predicted -0.70, Tallying +0.38, observed +0.174. Again the direction is Tallying's, but the magnitude is roughly half. Decomposing: on the large-margin disagreement pairs (3/4, 5/6, 11/12) a validity-weighted rule agrees with the tally (0.93 vs 1.85; 0.78 vs 1.85), so p_tally is high (~0.85), matching Tallying. But on the tally-tie family a weighted rule is NOT at 0.5: pair 7/8 (.93 vs .60) strongly favours the TTB winner, while pair 9/10 (.55+.87=1.42 vs .70+.78=1.48) is essentially a coin flip that in fact tips slightly against the TTB winner. Averaged, p_ttb_tie \\u2248 0.65-0.70 rather than Tallying's 0.5, which pulls the difference metric down from ~0.38 to ~0.15-0.20 \\u2014 precisely the observed 0.174.\n\nSo the same single mechanism \\u2014 an additive integration of ALL cues weighted by their stated validities \\u2014 explains both deviations quantitatively and in the same direction, whereas Tallying is systematically biased (too extreme on tie trials, where it is parameter-free at 0.5) and TTB is wrong in sign on both experiments. TTB is the clear loser and is degenerate here; Tallying is directionally right but its parameter-free 0.5 prediction on tie trials is falsified in both experiments (Exp1 ties pull the metric up, Exp2 ties pull it down), which no re-parameterisation of Tallying can fix. The productive move is to replace TTB with the intermediate model that dominates both.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace THEORY 1 (Take The Best) with a validity-weighted additive integration model (WADD / Franklin's rule), which subsumes the strengths of both current theories while fixing their systematic errors.\n\nCore claim: on each trial the decision maker computes a weighted evidence difference d = sum_j w_j * (a_j - b_j), where w_j is a monotone function of the communicated validity of expert j (e.g. w_j = v_j, or w_j = v_j - 0.5, or w_j = (v_j)^gamma with a free sensitivity exponent gamma >= 0 that interpolates between unit-weight tallying at gamma = 0 and near-lexicographic weighting at large gamma). Choice probability is a logistic/softmax on the SCALED difference: p(A) = (1-eps) * sigmoid(beta * d) + eps/2. Unlike TTB, every cue is consulted (no stopping rule); unlike Tallying, cues are not equally weighted, so tally-tie trials are NOT forced to 0.5 \\u2014 they resolve toward whichever option holds the higher-validity cues, and only near-exact weighted ties produce chance responding.\n\nParameter ranges should be chosen so the model reproduces both observations: (i) in Exp1 the coalition option is chosen on ~85-90% of directional-conflict trials while the tie pairs 5/6 (weighted diff 1.70 vs 1.50, a small margin) yield only a mild ~0.55-0.65 lean toward the higher-validity option, giving a pooled metric near 0.28; (ii) in Exp2 the large-margin trials give p_tally ~0.85 while the tie family averages p_ttb_tie ~0.65-0.70 (near-deterministic for A on pairs 7/8, near-chance or slightly against A on pairs 9/10), giving a metric near 0.17. Concretely: gamma in [0, 3] centred low-to-moderate (weights close to raw validities), beta in [1, 6] on the raw weighted-difference scale, eps in [0, 0.15]. Make sure the softmax operates on the CONTINUOUS weighted difference (magnitude-sensitive), not on a binarised winner code \\u2014 that magnitude sensitivity is what produces the graded tie-trial behaviour that distinguishes this theory from Tallying and the graded margin behaviour that distinguishes it from TTB.\n\nFuture experiments should target the region where WADD and Tallying diverge most: pairs whose unit tally is tied but whose weighted sums differ by a large amount (WADD decisive, Tallying at exactly 0.5), and pairs whose unit tally strongly favours one option but whose weighted sum is near zero or reversed (e.g. one very-high-validity cue vs two barely-above-chance cues)."
}
```

## Usage

```json
{
  "input_tokens": 11263,
  "output_tokens": 3954
}
```
