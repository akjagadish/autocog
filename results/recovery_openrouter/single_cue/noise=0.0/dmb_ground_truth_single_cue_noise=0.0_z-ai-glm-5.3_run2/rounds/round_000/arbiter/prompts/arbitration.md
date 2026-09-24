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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=13):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 11: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 12: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 13: A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Rationale:** The design exploits the core structural difference between Take-The-Best (TTB) and Tallying: TTB's choice is determined solely by the single highest-validity discriminating cue, whereas Tallying integrates the sign of every feature comparison. I use 5 features with a clear validity hierarchy [0.90, 0.80, 0.70, 0.60, 0.55] so the cue cascade is unambiguous. The trial set is built to produce a QUANTITATIVE dissociation, not just a qualitative one. (1) Conflict trials T1-T3: option A wins only on the top cue (validity 0.90) while B wins on progressively fewer lower cues (tally margins -3, -2, -1 for A). TTB predicts p(A) is IDENTICAL across T1, T2, T3 (a constant (1-eps)*sigmoid(beta)+eps/2, since the TTB score is always [1,0]); Tallying predicts a strictly graded p(A) that increases with the tally margin (sigmoid(-3*beta), sigmoid(-2*beta), sigmoid(-beta)). A flat vs. graded profile across these three trial types is the central quantitative signature. (2) Mirror conflict trials T4-T6 test the same dissociation in the opposite direction (TTB->B, Tallying->A), ruling out side-bias explanations. (3) Tally-tie trials T7-T8: the tally is exactly 1-1, so Tallying is forced to 50/50 regardless of beta, while TTB is near-deterministic (top cue decides). These trials alone can falsify Tallying at the aggregate level: any consistent preference is TTB-only. (4) Cascade-depth trials T11-T12: the top cue ties and the SECOND cue (0.80) discriminates against the tally majority — TTB follows cue 2, Tallying follows the 3-vs-2 count. This confirms subjects respect the validity ordering, not just 'any high cue wins'. (5) Agreement trials T9-T10 (both theories favor the same option, margin +1) and a full-tie trial T13 (both predict exactly 50/50) serve as controls: they anchor the lapse rate and verify that neither theory is needed where both agree. Crucially, TTB predicts a single constant choice probability across all five TTB->A trial types (T1, T2, T3, T7, T9), whereas Tallying predicts five distinct probabilities spanning from well below 0.5 to well above 0.5 — model comparison on the response profile across these cells cleanly separates the theories even after fitting beta and epsilon to each. With 13 unique pairs and K=7 repetitions (91 trials), each cell is sampled enough to estimate choice proportions per subject.

**Computed schedule:** 13 unique pairs × 7 reps = 91 trials per subject.



### METRIC
Rationale:
The metric is a theory-opposed choice-profile contrast built from two TTB-diagnostic signatures in this design, combined into one scalar (level minus slope) so that both push in the same direction.

Construction: for every trial I reconstruct, from the rating vectors and the known validities [0.9, 0.8, 0.7, 0.6, 0.5], (i) the TTB-favored option (first discriminating cue in validity order) and (ii) the tally margin of that option (its feature-win count minus the opponent's). Trials with no discriminating cue (T13) are dropped. Then:

(1) LEVEL = P(choose the TTB-favored option | tally does not favor it), i.e. over the 10 conflict/tie trial types (margins -3, -2, -1, 0). Under TTB every one of these trials has the same TTB score, so the level equals q = (1-eps)*sigmoid(beta) + eps/2, typically 0.6-0.95. Under Tallying these cells have probabilities sigmoid(-3b), sigmoid(-2b), sigmoid(-b) and 0.5, so the level is dragged well below 0.5 (e.g. 0.32 at beta=0.5, 0.13 at beta=2).

(2) SLOPE = OLS slope of the match indicator on the tally margin across all discriminating trials (margins -3..+1). Under TTB the choice probability is flat in the margin, so the slope is exactly 0 in expectation for ANY (beta, epsilon) — this term is parameter-free for my theory. Under Tallying the probability strictly increases with the margin, giving a positive slope (e.g. 0.11 at beta=0.5, 0.22 at beta=2).

Subtracting the slope from the level makes both deviations add: TTB yields approximately q - 0 (>= 0.5), while Tallying yields level - slope, which is pushed further down (0.21 at beta=0.5, -0.09 at beta=2, 0.44 even at beta=0.1). The between-theory gap is therefore large across the whole plausible parameter space (0.09 at beta=0.1/eps=0, 0.41 at beta=0.5, >0.9 at beta=2), while per-subject sampling noise is small: the statistic is a fixed linear contrast over ~84 trials per subject, with binomial standard error around 0.05-0.08. At N=25 this gives t > 3.9 even in the near-degenerate beta=0.1, eps=0 case, and t > 10 for any beta >= 0.3. The mirror trials (T4-T6, T8, T10, T12) are included with their signs flipped, so pure A/B side biases cancel. The slope term also protects against parameter heterogeneity across subjects: it is exactly zero for TTB regardless of each subject's beta and epsilon, so it cannot inflate my theory's between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=float)
    cue_order = np.argsort(-validities, kind="stable")

    margins = []
    matches = []
    for a_cell, b_cell, resp in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float).ravel()
        b = np.asarray(b_cell, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] != validities.shape[0]:
            continue
        diff = a - b

        # Take-The-Best cascade: first discriminating cue in validity order.
        ttb_winner = -1
        for j in cue_order:
            if diff[j] > 0:
                ttb_winner = 0
                break
            if diff[j] < 0:
                ttb_winner = 1
                break
        if ttb_winner < 0:
            # Full tie (trial 13): no TTB winner, uninformative for both theories.
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # Tally margin of the TTB-favored option (positive = tally agrees with TTB).
        if ttb_winner == 0:
            margin = a_wins - b_wins
        else:
            margin = b_wins - a_wins

        r = int(resp)
        match = 1.0 if r == ttb_winner else 0.0
        margins.append(margin)
        matches.append(match)

    if not margins:
        return 0.0

    x = np.asarray(margins, dtype=float)
    y = np.asarray(matches, dtype=float)

    # (1) LEVEL: P(choosing the TTB-favored option) on trials where the
    # tally does NOT favor it (tally majority against, or exact tally tie).
    low = x <= 0
    level = float(np.mean(y[low])) if np.any(low) else 0.0

    # (2) SLOPE: OLS slope of the match indicator on the tally margin,
    # across all trials with a TTB winner (margins -3, -2, -1, 0, +1).
    vx = float(np.var(x))
    if len(x) > 1 and vx > 0:
        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
    else:
        slope = 0.0

    return level - slope

### RESULTS
- Predicted under pi_1 (simulated): 0.8479 (var=0.0111)
- Predicted under pi_2 (simulated): 0.0323 (var=0.0195)
- Observed on real data: 0.3633 (var=0.0060)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.6, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 11: A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 12: A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 13: A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 14: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 15: A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** This design dissociates Tallying from Take-The-Best (TTB) along two orthogonal quantitative dimensions simultaneously: (i) the tally MARGIN (a_wins - b_wins), which is the ONLY trial-dependent quantity in Tallying's softmax (p(A) = (1-eps)*sigmoid(beta*margin) + eps/2), and (ii) the identity/validity of the first discriminating cue, which is the ONLY trial-dependent quantity in TTB (p(A) = (1-eps)*sigmoid(beta) + eps/2 toward the top-cue winner, constant across ALL discriminating trials). I use 6 features with an unambiguous validity hierarchy [0.95, 0.85, 0.75, 0.65, 0.60, 0.55]. CENTRAL SIGNATURE 1 — the full crossing psychometric function (T1-T5 and mirror T6-T10): the top cue (0.95) always favors one option while the tally margin sweeps across its full range (-4, -2, 0, +2, +4). TTB predicts a FLAT p(A) = (1-eps)*sigmoid(beta)+eps/2 across all five trials in each direction; Tallying predicts a strictly monotone graded curve that passes through EXACTLY 0.5 at margin 0 (T3/T8) — a knife-edge prediction independent of beta that any consistent preference falsifies — and reverses sign at negative margins (T1/T2: TTB says A with high confidence, Tallying says B; T6/T7: the mirror sign reversal, ruling out side bias). T1 vs T10 is especially clean: identical tally margins (-4) but opposite TTB winners, so Tallying predicts identical p(A) on both while TTB predicts opposite extremes. CENTRAL SIGNATURE 2 — cascade-depth invariance (T11-T15): the top cue TIES and the TTB decision comes from cue 2 (0.85), cue 3 (0.75), or even cue 5 (0.60, T14). TTB predicts the SAME constant p(A) on T1, T2, T3, T11, T12, T13, T14 (all TTB->A regardless of which cue decided or how deep the cascade ran); Tallying predicts six distinct values spanning sigmoid(-2beta) to sigmoid(+1beta), including exact 0.5 on T3/T13/T14. T15 is a depth-2 sign reversal (cue 2 favors B, tally margin +2 favors A). T14 probes the bottom of the hierarchy: when cues 1-4 all tie, TTB commits to the single 0.60-validity cue while Tallying sits at exactly 50/50 — if subjects treat a lone low-validity cue as uninformative, TTB fails where Tallying succeeds. T16 (full tie) is the shared 50/50 control anchoring lapse rate and side bias. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 6 repetitions per subject for stable choice-proportion estimates. Model comparison on the full 16-cell response profile — flat vs. graded across margins, constant vs. varied across cascade depths, and exact-0.5 vs. deterministic on margin-0 trials — cleanly separates the theories even after fitting beta and epsilon to each.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
CORE IDEA: the signed choice-tally covariance, mean over trials of s*margin where s=+1 for choosing A and margin = (#features A wins) - (#features B wins). This is the locally most powerful (weak-signal optimal) linear discriminator between the two theories on this stimulus set, and it works on both the pooled data and any single subject's 96-trial slice.

WHY IT SEPARATES THE THEORIES:
(1) Under Tallying, p(A) = (1-eps)*sigmoid(beta*margin) + eps/2, so E[s] = (1-eps)*tanh(beta*margin/2). Because tanh is odd, E[s*margin] = (1-eps)*|margin|*tanh(beta*|margin|/2) >= 0 on EVERY trial with nonzero margin — every trial type pushes the statistic in the same (positive) direction. The statistic equals (1-eps)*mean(|m|*tanh(beta|m|/2)), which for this design ranges from ~beta*E[m^2]/2 = 2.69*beta (small beta) up to (1-eps)*28/16 = 1.75*(1-eps) (large beta). So for any nondegenerate (beta, eps) the Tallying value is clearly positive.
(2) Under TTB, E[s] = (1-eps)*tanh(beta/2)*ttb_sign, independent of margin. Summing ttb_sign*margin over the 16 trial types gives exactly -2 (T1..T5 contribute -4,-2,0,+2,+4 = 0; T6..T10 contribute -4,-2,0,+2,+4 = 0; T11..T14 contribute -1,+1,0,0 = 0; T15 contributes -2; T16 contributes 0), so the TTB value is -0.125*(1-eps)*tanh(beta/2) — slightly NEGATIVE for all beta>0. The sign of the statistic itself flips between theories.

WHY THE CONTRAST IS TIGHT ACROSS SUBJECTS: each subject contributes 96 trials with 6 repetitions of each of the 16 cells; the statistic is an average of bounded per-trial values (|s*margin| <= 4). At moderate/large beta the Tallying choices on the high-margin cells (|m|=4: T1,T5,T6,T10; |m|=2: five more cells) are near-deterministic, so per-subject SD is tiny (~0.04 at beta=2), giving t > 50 against TTB's small negative value. Even at small beta the m-weighting is exactly the Neyman-Pearson-optimal weighting (signal proportional to margin), maximizing the mean-to-SD ratio; e.g. at beta=0.1, eps=0 the gap is ~0.27 with per-subject SD ~0.24, still yielding t ~ 4 at N=25. Margin-0 trials (T3, T8, T13, T14, T16) contribute exactly 0 under both theories, so they add no variance while the design's mirror structure (T1 vs T10, T6 vs T5) cancels any side bias.

ROBUSTNESS: the statistic requires no parameter fitting, no knowledge of beta/epsilon, uses only the schema columns, is invariant to trial order, and is well-defined on a single subject's slice (mean over their own trials), so between-subject variance is computable.

Source:
import numpy as np
import pandas as pd


def _to_matrix(col):
    # col: pandas Series whose cells are lists/tuples/arrays of binary ratings
    return np.array([list(r) for r in col], dtype=float)


def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    A = _to_matrix(data['option_a_ratings'])
    B = _to_matrix(data['option_b_ratings'])
    if A.ndim != 2 or A.shape != B.shape or A.shape[0] == 0:
        return 0.0

    # Tally margin: number of features where A beats B minus number where B beats A.
    margin = (A > B).sum(axis=1) - (B > A).sum(axis=1)

    # Signed choice: +1 if the subject chose A (response == 0), -1 if B (response == 1).
    resp = pd.to_numeric(data['response'], errors='coerce').to_numpy(dtype=float)
    s = 1.0 - 2.0 * resp

    # Alignment between the signed choice and the tally margin.
    # Tallying: E[s | margin] = (1-eps) * tanh(beta*margin/2)  =>  E[s*margin] > 0,
    # scaling like beta * E[margin^2] / 2 for small beta and saturating at mean(|margin|)
    # for large beta (max value 28/16 = 1.75 on this design).
    # TTB: E[s | trial] = (1-eps)*tanh(beta/2)*ttb_sign, and on this stimulus set
    # sum over the 16 trial types of ttb_sign * margin = -2 (mean -0.125), so the
    # TTB value is slightly NEGATIVE for any beta > 0.
    return float(np.mean(s * margin))

### RESULTS
- Predicted under pi_1 (simulated): -0.1621 (var=0.0229)
- Predicted under pi_2 (simulated): 1.3208 (var=0.1532)
- Observed on real data: 0.0358 (var=0.0181)

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
  "interpretation": "Neither theory is task-invariant \u2014 in fact, each theory fails most dramatically on the experiment designed to favor IT.\n\n**Experiment 1 (TTB-favoring metric: level \u2212 slope).** TTB's predicted value is structurally bounded below by 0.5: its 'level' term is q = (1\u2212\u03b5)\u00b7\u03c3(\u03b2) + \u03b5/2 \u2265 0.5 for ANY (\u03b2, \u03b5), and its slope on tally margin is exactly 0 in expectation. The observed value of 0.363 (var = 0.006, so a tight between-subject estimate) lies far BELOW TTB's entire achievable range \u2014 TTB is strictly falsified, not just quantitatively off (its simulation said 0.848). Tallying (predicted 0.032) can hit 0.363, but only at a moderately sharp \u03b2 \u2248 0.3\u20130.4.\n\n**Experiment 2 (tally-margin covariance).** The observed value is 0.036 \u2248 0. Tallying at the \u03b2 \u2248 0.3\u20130.4 needed for Experiment 1 predicts \u2248 0.7\u20130.9 here (its simulation at sharper \u03b2 said 1.32) \u2014 a massive miss, so Tallying cannot fit the two experiments jointly with any single parameter setting. TTB's range on this metric is [\u22120.125, 0]; the observed slightly-positive value again falls outside it. So TTB is outside its predictable interval on BOTH experiments; Tallying covers each value in isolation but no single (\u03b2, \u03b5) fits both.\n\n**The joint constraint is the key.** For any margin-monotone compensatory rule (Tallying, weighted-additive, random-cue sampling), the Experiment 2 statistic is approximately 2\u00b7(Experiment 1 slope)\u00b7E[m\u00b2] \u2248 10.75 \u00d7 slope. Since Experiment 2 \u2248 0.04, the Experiment 1 slope must be \u2248 0; therefore Experiment 1's deviation from 0.5 is a pure LEVEL effect: subjects chose the TTB-favored option only ~36% of the time when the tally opposed it \u2014 a systematic ANTI-alignment with the top cue, with no compensatory margin integration at all. I verified that weighted-additive, equal-weight tallying, single-random-cue, and mixture-of-strategies accounts all fail this joint constraint (any of them that reproduces 0.363 in Exp 1 overpredicts Exp 2 by an order of magnitude).\n\n**What fits both with ONE noise setting:** a one-reason (frugal) rule run over a REVERSED cue hierarchy \u2014 i.e., consulting cues in ascending validity order (equivalently, misbinding the communicated validity list to feature positions in reverse, or flipping the rating-scale meaning). On Experiment 1 this rule anti-matches TTB on every discriminating trial regardless of margin, giving level \u2248 (1\u2212\u03b5)(1\u2212\u03c3(\u03b2)) + \u03b5/2, slope = 0; at \u03b2 = 0.6, \u03b5 = 0.1 the metric is \u2248 0.37 (observed 0.363). On Experiment 2 the reversed cascade yields a mean signed choice\u00d7margin of +0.125\u00b7(1\u2212\u03b5)\u00b7tanh(\u03b2/2) \u2248 0.033 at the same parameters (observed 0.036). One parameter setting nails both experiments, including the small observed variances. This is a coherent psychological story: subjects remain frugal single-cue users but misbind the instructed validities to the features (e.g., binding the first-listed validity to the last-listed feature, or treating the final/most recent feature as most diagnostic), plus ordinary response noise.\n\n**Verdict rationale.** Both theories are degenerate with respect to task-invariance \u2014 each loses on its own home-turf diagnostic (TTB by 0.49 in Exp 1, Tallying by 1.29 in Exp 2), and TTB's predicted interval excludes the observed value in both experiments. Per the arbitration logic, I propose a brand-new theory rather than re-parameterizing an existing one (no parameter range can rescue TTB, whose Exp 1 metric is bounded below by 0.5, and Tallying's failure is structural, not parametric). The new theory replaces pi_1 (TTB), the more degenerate of the pair; it also preserves a sharp contrast with the surviving Tallying theory (single-reason anti-hierarchy vs. full integration), which future experiments can arbitrate.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 with 'Reversed-Hierarchy Frugality' (misbinding TTB). CORE CLAIM: people are still one-reason decision makers \u2014 they consult cues one at a time and let the first discriminating cue decide \u2014 but the subjective cue hierarchy is ANTI-correlated with the instructed validities. Mechanistically, subjects misbind the communicated validity list to feature positions (e.g., binding the highest validity to the last-presented feature, a presentation-order/validity misalignment), or equivalently consult cues in ascending subjective validity. No magnitude information and no integration across cues is used; ties on the deciding cascade force a guess. Response noise: softmax with inverse temperature \u03b2 over the binary winner score, plus lapse \u03b5.\n\nIMPLEMENTATION: predict(parameters, stimulus, history): (1) stim is (2, n_features); (2) build subjective validities w = parameters['validities'] reversed end-to-end (w_j = v_{n\u22121\u2212j}), or more generally w_j = (1\u2212\u03ba)\u00b7v_j + \u03ba\u00b7v_{n\u22121\u2212j} with a binding-confusion parameter \u03ba \u2208 [0,1]; (3) run the standard TTB cascade in descending w order (first strict discriminator wins; full tie \u2192 uniform guess); (4) softmax with \u03b2 and lapse \u03b5 exactly as in the current pi_1 code. Policy unchanged (sample from the probability vector).\n\nPARAMETER RANGES: \u03b2 \u2208 [0.1, 3.0], \u03b5 \u2208 [0.0, 0.3], \u03ba \u2208 [0.5, 1.0] (the data indicate near-complete reversal: \u03ba \u2248 1 with \u03b2 \u2248 0.6, \u03b5 \u2248 0.1 reproduces Exp 1 \u2248 0.37 and Exp 2 \u2248 0.03 simultaneously). Keep validities as an experiment-supplied parameter.\n\nHOW IT DIFFERS FROM BOTH CURRENT THEORIES: vs. TTB (pi_1), it keeps the frugal one-reason architecture but inverts the cue hierarchy \u2014 this is the only single change that lets a frugal model produce Exp 1's below-0.5 level while keeping Exp 2's margin covariance near zero; vs. Tallying (pi_2), it uses no integration at all, so it predicts a FLAT choice profile across tally margins (slope 0) where Tallying predicts a graded psychometric curve \u2014 a cleanly opposing, falsifiable signature. The theory is further distinguishable from a mere 'anti-TTB' flip by the \u03ba parameter: designs that decorrelate feature position from validity (shuffled validity-to-position assignment, or probing which specific cue drives choices) can separately identify misbinding, reversal, and noise, and should be used as the next generation of experiments."
}
```

## Usage

```json
{
  "input_tokens": 7621,
  "output_tokens": 28383,
  "reasoning_tokens": 26766,
  "provider": "Sail Research"
}
```
