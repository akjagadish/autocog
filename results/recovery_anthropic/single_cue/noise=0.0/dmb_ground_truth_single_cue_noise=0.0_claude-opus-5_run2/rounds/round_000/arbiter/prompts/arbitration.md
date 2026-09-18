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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.62, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 10: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 12: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** Five experts with a wide validity spread (0.95, 0.85, 0.75, 0.62, 0.55) make the cue hierarchy salient, so Take The Best (TTB) has an unambiguous search order, while Tallying is indifferent to that order. The 12 unique pairs (repeated 8x each, ~96 trials) fall into three families, all of which dissociate the two models.

(1) Direct-conflict pairs (T1-T6, T11-T12): one option wins the highest discriminating cue while the other wins the majority of the remaining lower-validity cues. TTB predicts choice of the top-cue winner on every one of these; Tallying predicts the opposite option. Critically, the tally margin is parametrically varied: margin 3 (T1/T2: 1 vs 4 cue wins), margin 2 (T5/T6: 1 vs 3; T11/T12: 1 vs 3 with the top cue tied so TTB's deciding cue is expert 2), and margin 1 (T3/T4: 2 vs 3). This yields the key *quantitative* dissociation: TTB's softmax operates on a binary 1/0 score, so its predicted choice probability for the top-cue winner is identical (sigmoid(beta), lapse-corrected) across all conflict trials regardless of tally margin, whereas Tallying's softmax probability grows monotonically with the tally margin (exp(beta*1) vs exp(beta*2) vs exp(beta*3) contrasts). Thus a flat, margin-invariant preference for the top-cue winner is uniquely TTB, while a graded, margin-increasing preference for the majority-cue winner is uniquely Tallying.

(2) Tally-tie pairs (T7-T10): the two options split cue wins evenly (1-1 or 2-2), so Tallying is forced to guess at exactly 50/50 irrespective of its parameters, while TTB still decides confidently via the highest discriminating cue. Systematic above-chance choice of the top-cue winner here cannot be produced by Tallying under any beta/epsilon, making these trials a parameter-free falsification test.

(3) Left/right and direction counterbalancing: every conflict and tie configuration appears in both A-favoring and B-favoring versions, so any side bias cancels and cannot masquerade as evidence for either heuristic.

T11/T12 additionally probe whether the deciding cue must be the very best cue: expert 1 ties, so TTB's decision rests on expert 2 while the tally strongly favors the other option; this separates TTB from any 'only-the-single-best-cue-matters' caricature as well as from Tallying.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains four trial types (T7-T10) on which the two options split feature-wise wins exactly evenly (1-1 or 2-2), so the Tallying softmax is mathematically uniform for ANY beta and epsilon: Tallying must choose the highest-validity-cue winner exactly 50% of the time, with only binomial noise (32 such trials per subject => between-subject SD ~0.09). Take The Best, in contrast, resolves these pairs on the first discriminating cue and therefore chooses the top-cue winner with probability (1-eps)*sigmoid(beta)+eps/2, which for essentially the whole plausible parameter range sits well above chance (~0.80-0.95). The metric is the proportion of tally-tie trials on which the option favored by the highest-validity discriminating cue was chosen. It is parameter-free for Tallying (fixed at 0.5, tight across subjects) and clearly above 0.5 for TTB, giving a large mean gap with modest per-subject variability. It also works identically on a single subject's slice and on the pooled data, and side/direction counterbalancing in the design ensures response bias cannot inflate it.

Source:
def metric(data: pd.DataFrame) -> float:
    validities = [0.95, 0.85, 0.75, 0.62, 0.55]
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fall back: assume features already ordered by descending validity
            v = np.arange(n, 0, -1).astype(float)
        order = np.argsort(-v, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # only tally-tie trials

        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no discriminating cue at all

        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_1 (simulated): 0.8988 (var=0.0090)
- Predicted under pi_2 (simulated): 0.4988 (var=0.0068)
- Observed on real data: 0.4738 (var=0.0073)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.93, 0.57, 0.79, 0.86, 0.62, 0.71]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 7: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 9: A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 10: A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 11: A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 12: A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]

**Rationale:** Goal: dissociate Tallying (advocated) from Take The Best by factorially crossing (i) which option wins the highest-validity discriminating cue \u2014 the only thing TTB is sensitive to \u2014 with (ii) the tally margin, the only thing Tallying is sensitive to. Six experts with a wide, deliberately scrambled validity spread [0.93, 0.57, 0.79, 0.86, 0.62, 0.71] give the descending cue order f0 > f3 > f2 > f5 > f4 > f1, so validity rank is decoupled from screen position (unlike a monotone left-to-right layout, this prevents a positional-ordering confound from mimicking TTB).

Core manipulation (trials 1,3,5,7,9 and their exact A/B mirrors 2,4,6,8,10): option A always wins expert f0 (validity .93), so TTB's deciding cue and predicted winner are IDENTICAL across all five cells; its predicted P(choose A) is the constant (1-eps)*sigmoid(beta) + eps/2 regardless of the rest of the profile. Meanwhile the tally margin is parametrically swept M = -3, -1, 0, +2, +4 by re-allocating the five lower-validity cues. Tallying's softmax gives P(A) = sigmoid(beta*M) (lapse-corrected), i.e. a strictly monotone, sign-crossing psychometric function of M: strong B-preference at M=-3, weak B at M=-1, exactly 50/50 at M=0, and increasingly strong A at +2 and +4. A flat line across the five margin cells is a unique TTB signature; a monotone increasing line crossing chance at M=0 is a unique Tallying signature. Crucially the design contains both conflict cells (M=-3, -1: TTB says A, Tallying says B) and agreement cells (M=+2, +4: both say A), so the same subject's data pin down beta from the agreement cells and then test whether the conflict cells reverse (Tallying) or not (TTB) \u2014 a within-subject, parameter-anchored test rather than a mere direction contrast.

Tally-tie cells (trials 5, 6): three cue wins each. Tallying MUST be exactly 50/50 for every possible beta and epsilon, while TTB confidently picks the f0 winner. Systematic above-chance choice here parameter-free falsifies Tallying; chance performance parameter-free falsifies TTB.

Low-rank deciding cue (trials 11, 12): the two best experts (f0=.93 and f3=.86) are tied, so TTB must descend to f2 (.79), which favors A, while the tally favors B by 2 (B wins f5, f4, f1). This checks whether the one-reason rule really cascades down the validity hierarchy or whether subjects instead aggregate the three lower experts \u2014 an independent conflict test whose TTB prediction is the same magnitude as in trials 1/3 (binary score) but whose Tallying prediction is intermediate (|M|=2).

Every configuration appears in both an A-favoring and a B-favoring mirror, so side bias cancels exactly and cannot masquerade as support for either heuristic. Tie cues are encoded sometimes as 0-0 and sometimes as 1-1 to prevent a 'count the 1s' surface strategy from being confounded with either model. 12 unique pairs \u00d7 8 repetitions \u2248 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains conflict trials in which the tally winner and the one-reason (TTB) winner are opposite options, and where the tally margin is large (|M| >= 2): trials 1/2 (A wins the .93 expert but loses the tally 1-4) and trials 11/12 (the two best experts tie so TTB descends to f2 and picks one option, while the tally favors the other by 2). This metric is the proportion of choices on exactly those trials that agree with the tally winner.

Prediction under Tallying: with margin magnitude 2-3, softmax choice is near-deterministic for the tally winner for almost the whole beta range, so the score is (1-eps)*sigmoid(beta*|M|)+eps/2, i.e. near 0.85-1.0. Prediction under TTB: exactly the complement, eps/2 + (1-eps)*(1-sigmoid(beta)), i.e. near 0-0.15. The expected mean gap is ~0.7-0.75 while per-subject variance is small (only lapse-rate spread plus binomial noise over 32 trials at extreme p), which is what Welch's test at N=25 needs. Restricting to |M| >= 2 avoids the weak M=1 cells where Tallying's own prediction is soft; excluding agreement and tie cells removes trials where the two theories predict the same thing and which would only dilute the contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    val = np.array([0.93, 0.57, 0.79, 0.86, 0.62, 0.71], dtype=float)
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if val.shape[0] == n:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(n)
        # tally margin (A wins minus B wins)
        margin = int(np.sum(a > b) - np.sum(b > a))
        if abs(margin) < 2:
            continue
        tally_winner = 0 if margin > 0 else 1
        # take-the-best winner
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
            continue  # keep only conflict trials
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == tally_winner else 0.0)
    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_1 (simulated): 0.1431 (var=0.0111)
- Predicted under pi_2 (simulated): 0.8525 (var=0.0139)
- Observed on real data: 0.6538 (var=0.0092)

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
  "interpretation": "Across the two experiments the evidence is decisively against Take The Best (pi_1) and broadly, though imperfectly, in favour of feature counting (pi_2).\n\nExperiment 1 was pi_1's own designed falsification test: on tally-tie pairs (1-1 and 2-2 splits) TTB must pick the top-validity-cue winner with p ~ 0.90, while Tallying is mathematically pinned at 0.50 for every beta and epsilon. The observed value is 0.4738 (var=0.0073), essentially indistinguishable from chance and from the binomial-only variance expected under a parameter-free 0.5 prediction (32 trials, p=.5 gives var ~ .0078). There is no evidence of a subgroup of TTB users; subjects are homogeneously at chance. This is a clean, parameter-free refutation of pi_1. Notably it also refutes any validity-weighted additive account: on T7 (A wins the .95 cue, B wins the .85 cue) and T9 (A wins .95 and .75, B wins .85 and .62) a WADD/log-odds integrator would strongly favour A, yet behaviour is at chance. Subjects appear to be counting cue wins with genuinely equal weights.\n\nExperiment 2 confirms the direction: on high-margin conflict trials where TTB and the tally point in opposite directions, subjects followed the tally 65.4% of the time (TTB predicted 0.143, Tallying 0.853). pi_1 is off by ~0.51; pi_2 is off by ~0.20 and on the correct side of chance. So pi_2 is the only theory whose sign is right in both experiments, and it is exactly right where it makes a parameter-free prediction (Exp 1).\n\nHowever, pi_2 is not fully correct either: its simulated beta range was too high, producing near-deterministic tally following (0.85) where the data show substantially attenuated adherence (0.65), despite margins of |M| = 2 and 3. Two readings are possible: (a) Tallying with much lower beta / higher lapse, which would reproduce both numbers; or (b) the adherence is not a pure function of the raw margin but of how many features were actually inspected \u2014 i.e. counting over a limited, noisily sampled subset of cues, which caps adherence well below 1 even at large margins and predicts dilution by uninformative (tied) features. The current data cannot separate (a) from (b), and that is exactly the contrast worth putting on the table next.\n\nSince pi_1 (TTB) is falsified parameter-free in Experiment 1 and badly wrong in Experiment 2, it should be discarded rather than re-parameterised (no beta/epsilon can move it off 0.90 on tally-tie trials without simultaneously destroying it elsewhere; only epsilon -> 1, a degenerate all-guessing model, would fit). Replacing it with a sharper competitor to Tallying is the productive move.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 (Take The Best) entirely. It is parameter-free falsified: on Experiment 1's tally-tie trials it must predict ~0.90 for the top-cue winner and the data sit at 0.474, and it predicts 0.14 where the data show 0.65 in Experiment 2. No re-parameterisation rescues it except degenerate all-lapse.\n\nPropose instead a 'limited-sample / noisy feature counting' theory that is a genuine rival to full Tallying while remaining consistent with BOTH observations:\n\nCore claim: subjects do not integrate all cues, nor a single best cue. On each trial they inspect a limited random subset of the features (subset size drawn from a distribution with mean k < n; features sampled with EQUAL probability, not validity-weighted, since Exp 1's chance performance on tally-ties rules out validity weighting), count feature-wise wins within that sample, and choose the option ahead within the sample; a within-sample tie forces a guess. Optionally each inspected comparison is registered with error probability nu (attention/encoding noise), plus a lapse epsilon.\n\nWhy it fits both experiments: (i) On globally tally-tied, side-counterbalanced pairs, uniform subset sampling is symmetric, so choice is exactly 50/50 \u2014 reproducing Exp 1's 0.474 as parameter-freely as Tallying does. (ii) On large-margin conflict trials, adherence to the global tally winner is capped well below 1 because a small sample can reverse the sign \u2014 naturally yielding the attenuated ~0.65 observed in Exp 2 without needing an implausibly small beta.\n\nCritical dissociation from pi_2 for the next experiment: softmax Tallying's choice probability depends only on the RAW margin M, whereas the sampling account depends on the PROPORTION of informative cues won (and on the number of tied/uninformative cues present). So design pairs holding M constant while varying the informative-cue base rate \u2014 e.g. a 2-0 split (M=2, 2 informative cues, 4 tied) versus a 4-2 split (M=2, 6 informative cues). Tallying predicts identical choice probabilities; the sampling model predicts markedly higher adherence for 2-0 and dilution when many tied cues pad the profile. Ensure the new theory's parameters (mean sample size k, encoding noise nu, epsilon) are ranged so that its Exp 1 prediction stays at 0.5 and its Exp 2 prediction lands near 0.60-0.70."
}
```

## Usage

```json
{
  "input_tokens": 9993,
  "output_tokens": 4319
}
```
