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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 10: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 11: A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 12: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 14: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 15: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 16: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design exploits the core structural difference between Take-The-Best (TTB) and Tallying: TTB's choice (and its predicted choice probability, sigmoid(beta)) is determined solely by the single highest-validity discriminating cue and is therefore INVARIANT to how many lower-validity cues oppose it, whereas Tallying's predicted probability is a softmax over win counts and is therefore GRADED in the tally margin. I use 5 features with a wide, strictly ordered validity spread [0.95, 0.85, 0.75, 0.65, 0.55] so the cue hierarchy is unambiguous. The critical trials (T1-T4 and their mirrors T5-T8) hold the TTB-relevant structure fixed — the top cue (validity 0.95) always discriminates in favor of the same option — while the tally margin against that option grows from 0 (tie) to 1, 2, and 3. TTB predicts a FLAT probability of choosing the top-cue winner across all four margin levels (identical sigmoid(beta) - epsilon mixture), while Tallying predicts a monotonically DECLINING curve: 0.5, sigmoid(-beta), sigmoid(-2beta), sigmoid(-3beta). This is a quantitative, parametric signature: the slope of P(top-cue winner) vs. tally margin is exactly zero under TTB and strongly negative under Tallying. Trials T9-T12 repeat the same logic with the top cue tied and the second cue (0.85) discriminating, and T15-T16 with the third cue (0.75) discriminating, testing whether the dissociation persists deeper in the cue hierarchy (TTB predicts the same flat sigmoid(beta) at every cue level; Tallying's predictions depend only on counts, not on which cue discriminates). Trials T13-T14 are agreement trials (both heuristics favor the same option) serving as a baseline. Every conflict type appears in both A-favored and B-favored versions to absorb side/position bias. With 16 unique pairs, K = 96 // 16 = 6 repetitions yields exactly 96 trials per subject. Predicted dissociation summary: (1) on tally-tie trials with a discriminating top cue (T1, T5, T9, T11), TTB predicts above-chance preference for the cue winner while Tallying predicts exactly 50/50 regardless of beta; (2) across the margin gradient (T1->T4), TTB predicts zero slope, Tallying predicts a steep negative slope; (3) TTB is indifferent to which cue in the hierarchy discriminates (same probability at f0, f1, f2 levels for matched tally structure), while any validity-sensitive weighting would break this invariance.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric is the proportion of trials on which subjects choose TAKE-THE-BEST's favored option, restricted to the 'pure conflict' trials where the two heuristics disagree (the tally majority favors the opposite option to the top discriminating cue). In this design those are trials T2,T3,T4,T6,T7,T8,T10,T12,T15,T16 — 10 of 16 trial types, i.e. 60 of the 96 trials per subject, so the per-subject estimate is based on a large sample and is very stable. Predicted values: under TTB, every conflict trial has identical choice probability p = (1-eps)*sigmoid(beta) + eps/2 > 0.5 for any beta>0, eps<0.5 (e.g. ~0.73 at beta=1, ~0.87 averaged over the parameter prior, approaching 1 at large beta). Under Tallying, the same trials have probability (1-eps)*sigmoid(-m*beta)+eps/2 with tally margins against the TTB option of 1, 2, or 3, which is strictly BELOW 0.5 for any beta>0 (e.g. ~0.17 at beta=1, ~0.13 averaged over the prior, approaching 0 at large beta). So the two theories sit on opposite sides of 0.5 with a large signed gap (~0.5-0.75) that persists across the whole plausible parameter range, while per-subject binomial variance is tiny (sd ~0.06-0.12 per subject with 60 conflict trials). I deliberately excluded tie trials (where Tallying guesses at exactly 0.5) and agreement trials, because including them only dilutes the mean gap without reducing variance enough to compensate. The metric is a simple proportion, so it is well-defined both on the pooled DataFrame and on each single subject's slice, and it involves no free parameters, no fitting, and no hashing of the list-valued rating columns.

Source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Design validities (fixed by experiment): strictly decreasing,
    # so the TTB cue order is simply feature 0,1,2,3,4.
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_vec(x):
        if isinstance(x, str):
            return np.array(
                [int(v) for v in x.strip("[]()").replace(",", " ").split()]
            )
        return np.asarray(x, dtype=float)

    hits = []
    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = _to_vec(a_cell)
        b = _to_vec(b_cell)
        if a.size == 0 or a.shape != b.shape:
            continue

        # --- TTB winner: first discriminating cue in validity order ---
        ttb_winner = None
        for j in cue_order:
            if j >= a.size:
                continue
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue  # no discriminating cue: TTB guesses, uninformative

        # --- Tallying winner: strict feature-wise win counts ---
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally ties -> Tallying guesses at 0.5, dilutes contrast
        tally_winner = 0 if a_wins > b_wins else 1

        if tally_winner == ttb_winner:
            continue  # heuristics agree: uninformative

        # Pure conflict trial: TTB's option vs the tally majority's option.
        hits.append(1.0 if int(resp) == ttb_winner else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_1 (simulated): 0.8443 (var=0.0128)
- Predicted under pi_2 (simulated): 0.1357 (var=0.0112)
- Observed on real data: 0.2080 (var=0.0060)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 2: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 5: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 9: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 12: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 13: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 14: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 15: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 16: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** This design's core novelty is a cue-allegiance flip at FIXED tally counts — the mirror image of the classic margin-gradient design. Tallying's predicted probability depends ONLY on the win counts through a softmax, which is shift-invariant (softmax over [3,2] equals softmax over [1,0]); it is therefore EXACTLY invariant to which features each option wins on and to cue depth. TTB's choice, in contrast, is determined solely by the allegiance of the highest-validity discriminating cue. The factorial core (trials 1-8) crosses tally margin {1, 2} x top-cue allegiance {pro-tally (agree), anti-tally (conflict)}: (i) C+ vs C- (and H+ vs H-) hold the win counts and even the tie placement identical (f0 tied, counts 3-2) while permuting cue allegiances so the top discriminating cue flips from favoring the tally winner to opposing it. Tallying predicts a precisely zero change in P(A) (identical sigma(beta)); TTB predicts a full flip from sigma(beta) to sigma(-beta). (ii) C vs H holds allegiance fixed and doubles the margin: TTB predicts exactly zero change (its probability is sigma(beta) for the top-cue winner regardless of how many cues oppose it), while Tallying predicts sigma(beta) -> sigma(2beta). The two theories thus occupy orthogonal cells of the 2x2: TTB shows a main effect of allegiance only; Tallying a main effect of margin only. The depth series (trials 9-14) places the anti-tally win at f0 (v=.95), f2 (v=.85), or f3 (v=.65), all at margin 1 with counts (3,2) or (2,1): both theories predict FLAT behavior here (TTB's probability is cue-depth invariant; Tallying's softmax is shift-invariant across (3,2)/(2,1)), so any validity-weighted hybrid would break this double invariance — a sharp signature that neither target theory but only mixtures would produce. The extreme anchors (trials 15-16) are 1-vs-5 splits where the top cue alone opposes five cues: TTB predicts P(cue winner) = sigma(beta) — numerically IDENTICAL to its prediction on the margin-1 conflict trials (invariance to 4 extra opposing cues), while Tallying predicts sigma(4beta) for the tally winner, near-deterministic at any reasonable beta. Quantitative dissociation summary: (1) on the allegiance flip (C+/C-, H+/H-), Tallying's predicted slope is exactly 0 and TTB's is -tanh(beta/2); (2) on the margin manipulation (C vs H), TTB's predicted slope is exactly 0 and Tallying's is positive; (3) across the depth series both are flat, jointly excluding weighted hybrids; (4) E1 vs C- isolates TTB's cue-count invariance against Tallying's steep margin gradient. Mirrors of every trial type absorb side/position bias. With 16 unique pairs, K = 96 // 16 = 6, yielding exactly 96 trials per subject.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design's 16 trial types split into 4 agree types (tally winner = TTB winner: trials 1, 3, 5, 7) and 12 conflict types (tally winner = TTB loser: trials 2, 4, 6, 8-16). The metric is the difference between overall choice-consistency with Tallying and overall choice-consistency with TTB. On agree trials the two consistencies are identical and cancel; on conflict trials they are exact complements, so the metric equals 0.75 * (2*P(choose tally winner | conflict trials) - 1). Under simulated Tallying data, P(tally winner | conflict) = softmax over tallies (e.g. ~0.77 at beta=1, ~1.0 at large beta), giving a strongly POSITIVE value (~+0.4 at beta=1, up to +0.75). Under simulated TTB data, P(tally winner | conflict) = 1 - sigmoid(beta) (TTB always follows the top cue regardless of how many cues oppose it, including the 1-vs-5 anchors), giving a strongly NEGATIVE value (~-0.35 at beta=1, down to -0.75). The between-theory gap in the mean is therefore large at every plausible parameter setting (roughly 0.09 even at beta=0.1, ~0.27 at beta=0.3, ~0.75 at beta>=2), while per-subject variance stays tight because each subject contributes 96 trials (72 conflict) of simple binary indicators: per-subject SD ~0.08, so with N=25 subjects the Welch t-statistic ranges from ~4 (worst case, near-random parameters) to ~35 (moderate/large beta), comfortably below alpha=0.01. The metric works identically on the pooled DataFrame and on any single subject's slice (it is just a difference of two proportions), and it is side-balanced (the tally winner is option A on exactly 8 of the 16 trial types), so position/label bias cancels in expectation.

Source:
def metric(data: pd.DataFrame) -> float:
    """Tally-vs-TTB choice-allegiance contrast.

    For every trial we determine (i) the TALLYING winner (option with more
    strict feature-wise wins) and (ii) the TAKE-THE-BEST winner (winner on
    the highest-validity discriminating cue, validities fixed by design as
    [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]).  The metric is

        mean( response == tally winner ) - mean( response == TTB winner )

    On the 4 trial types where both heuristics pick the same option
    (trials 1, 3, 5, 7 and mirrors) the two indicators are identical and
    cancel.  On the 12 conflict trial types they are exact complements, so
    the metric reduces to 0.75 * (2*P(tally winner | conflict) - 1): strongly
    positive under Tallying, strongly negative under TTB.
    """
    validities = np.array([0.95, 0.9, 0.85, 0.65, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    tally_match = []
    ttb_match = []

    for ca, cb, cr in zip(a_col, b_col, r_col):
        a = np.asarray(ca, dtype=float).ravel()
        b = np.asarray(cb, dtype=float).ravel()
        r = int(cr)

        # Tallying winner: more strict feature-wise wins (ties on a feature
        # contribute to neither side).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1  # undiscriminating tally -> excluded

        # TTB winner: first discriminating cue in descending-validity order.
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break

        if tally_winner >= 0:
            tally_match.append(1.0 if r == tally_winner else 0.0)
        if ttb_winner >= 0:
            ttb_match.append(1.0 if r == ttb_winner else 0.0)

    if len(tally_match) == 0 or len(ttb_match) == 0:
        return 0.0

    return float(np.mean(tally_match) - np.mean(ttb_match))

### RESULTS
- Predicted under pi_1 (simulated): -0.4837 (var=0.0228)
- Predicted under pi_2 (simulated): 0.5479 (var=0.0187)
- Observed on real data: 0.1583 (var=0.0053)

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
  "interpretation": "Converting both metrics to a common currency \u2014 P(choose the tally-majority option on TTB-vs-Tallying conflict trials) \u2014 gives: Experiment 1 observed = 1 - 0.208 = 0.792 (pi_1 predicted 0.156, pi_2 predicted 0.864); Experiment 2 observed = 0.5 + 0.1583/1.5 = 0.605 (pi_1 predicted 0.177, pi_2 predicted 0.865). Two conclusions follow. (1) Take-The-Best (pi_1) is decisively refuted in both experiments: its central signature \u2014 flat, above-chance adherence to the single top-validity discriminating cue regardless of how many cues oppose it \u2014 is violated. Subjects overwhelmingly follow the cue majority, landing on the opposite side of 0.5 from TTB in Experiment 1 and giving a positive tally-minus-TTB contrast in Experiment 2. TTB is degenerate for this domain. (2) Tallying (pi_2) gets the direction right in both experiments, but it is NOT task-invariant in its current form. Its own simulations predict essentially identical conflict-trial consistency in the two designs (0.864 vs 0.865), yet the observed consistency differs sharply (0.792 vs 0.605). No single (beta, epsilon) setting of pure Tallying can reproduce both: Experiment 1 requires an effective beta near 1 (softmax over tally margins 1-3 gives ~0.82), while Experiment 2 requires beta near 0.25 \u2014 and because Experiment 2's conflict trials actually have LARGER tallied/validity-weighted margins (mean weighted margin ~1.05 vs ~0.89 in Experiment 1), every fixed-weight tally variant (unweighted, v-weighted, v^2-weighted, log-odds, rank-weighted, capacity-limited sampling) predicts Experiment 2 should be MORE consistent than Experiment 1, the opposite of what is observed. The low between-subject variances (0.005-0.006) show these are reliable population-level patterns, not noise. The data therefore demand a theory that (a) is majority-driven like Tallying (explaining Experiment 1's 0.79), but (b) shows reduced, experiment-dependent consistency driven by validity-sensitive attraction to a dominant top cue precisely when a highly valid cue opposes a majority of weaker cues (Experiment 2's drop to 0.61, e.g., the 1-vs-5 anchor trials where a 0.95-validity cue faces five cues including 0.5). Neither existing theory satisfies both; the winning theory's invariance prediction is itself falsified. Per the arbitration logic, the degenerate theory (pi_1, TTB) should be replaced by a new theory that is a stronger competitor to Tallying \u2014 one that continuously spans the TTB-Tallying space and can explain the experiment-dependent attenuation.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 (Take-The-Best) with a 'Noisy Validity-Weighted Evidence Integration with one-reason lapses' (NVW) theory: (1) Evidence weighting: each feature-wise win for an option contributes evidence weighted by w_j proportional to v_j^gamma, where gamma is a validity-steepness parameter; gamma = 0 recovers Tallying (equal weights) and gamma -> infinity recovers TTB (top cue only), so the theory nests both existing theories as limiting cases. (2) One-reason mode: on each trial, with probability omega the agent abandons integration and relies solely on the single most valid discriminating cue (a TTB-like one-reason decision); omega should increase with the dominance of the top cue's validity over the opposing majority's cues \u2014 this captures the minority of top-cue choices that drag Experiment 2's tally-consistency down to 0.605 while leaving Experiment 1 (where observed consistency is high, 0.79) mostly integration-driven. (3) Capacity/attention: each cue is independently attended with probability alpha (or a fixed number m of the n cues is sampled), so effective decision noise grows with feature count and with majorities resting on few, weak cues \u2014 producing attenuated, design-dependent consistency rather than Tallying's invariant ~0.86. (4) Response rule: softmax over the (weighted, attended) evidence difference with inverse temperature beta, plus an independent lapse epsilon. Parameter ranges to propose: gamma in [0, 4], omega in [0, 0.4] (with the validity-dominance coupling above), alpha in [0.5, 1.0], beta in [0.5, 5], epsilon in [0, 0.2]. Target behavior: Experiment 1 metric (P choose TTB winner | conflict) approximately 0.20-0.28 and Experiment 2 metric (tally-minus-TTB consistency) approximately +0.10 to +0.22, with the between-experiment gap reproduced by the same fixed parameters. This theory keeps Tallying (pi_2) as the incumbent to beat, is falsifiable against it (it predicts validity-sensitive deviations from tally invariance that pure Tallying forbids), and can be validated on new designs that manipulate the validity gap between the top cue and the opposing majority at fixed tally margins."
}
```

## Usage

```json
{
  "input_tokens": 7280,
  "output_tokens": 26194,
  "reasoning_tokens": 25021,
  "provider": "Sail Research"
}
```
