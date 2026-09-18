# feedback_iter_00

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Replace equal-weight tallying with a validity-gated sequential evidence theory. Under this account, cues are inspected in descending validity, but stopping is controlled by an absolute confidence threshold derived from communicated cue validity. A sufficiently diagnostic cue terminates search and determines choice; if no inspected cue reaches threshold, evidence from multiple remaining cues is integrated using validity-dependent weights. In the present tasks, the high validities of the first available discriminating cues should usually exceed threshold, producing top-cue dominance in Experiment 1 and reliability that is largely invariant to additional agreeing cues in Experiment 2. Unlike strict TTB, however, the new theory predicts compensatory integration when all available cues are weak, validity differences are small, or early cues fall below threshold. Include parameters for the stopping threshold, sensitivity to stated validity, integration weights below threshold, response temperature, and lapse rate, with subject-level heterogeneity. Future tests should hold cue rank and comparison signs fixed while manipulating absolute validities and validity gaps: strict TTB predicts no change, whereas validity-gated processing predicts a transition from one-cue dominance to multi-cue integration.


## CANDIDATE THEORY
Confidence-Capped Cascade Integration proposes that people translate each communicated cue validity into an absolute diagnosticity signal and inspect cues from most to least valid. A discriminating cue controls the decision immediately when its diagnosticity exceeds a subject-specific confidence threshold. Earlier weak cues do not force premature stopping: search continues until a sufficiently diagnostic cue is encountered. If no discriminating cue reaches threshold, the decision maker switches to compensatory integration, summing all discriminating cues with weights determined by their stated validities. Thus, the theory produces one-cue dominance when reliable cues are available but graded multi-cue sensitivity when every cue is weak. Subject heterogeneity arises from differences in confidence threshold, sensitivity to stated validity, integration weighting, response temperature, and lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Confidence-Capped Cascade Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got shape {validities.shape}."
        )

    threshold = float(parameters["confidence_threshold"])
    sensitivity = float(parameters["validity_sensitivity"])
    integration_exponent = float(parameters["integration_exponent"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # A validity is the communicated probability that a cue points toward
    # the superior option. Its log odds are therefore an absolute measure
    # of the cue's diagnosticity. Clipping handles validity exactly equal
    # to 0.5 or 1 without numerical singularities.
    v = np.clip(validities, 0.5, 1.0 - 1e-9)
    log_diagnosticity = np.log(v / (1.0 - v))
    subjective_confidence = sensitivity * log_diagnosticity

    # Stable sorting preserves the displayed feature order when two cues
    # have equal communicated validity.
    cue_order = np.argsort(-validities, kind="stable")
    differences = stim[0] - stim[1]

    decision_evidence = None
    for j in cue_order:
        if differences[j] == 0.0:
            continue
        if subjective_confidence[j] >= threshold:
            # The first sufficiently diagnostic discriminating cue caps
            # search. Positive evidence favors A and negative favors B.
            decision_evidence = (
                np.sign(differences[j]) * subjective_confidence[j]
            )
            break

    if decision_evidence is None:
        # When no cue is individually convincing, combine all available
        # weak evidence. The exponent controls how sharply integration
        # privileges validity differences.
        weights = np.power(
            np.maximum(log_diagnosticity, 1e-12), integration_exponent
        )
        decision_evidence = float(
            np.sum(weights * np.sign(differences))
        )

    # Symmetric utilities ensure exact indifference when integrated evidence
    # is zero. Positive evidence favors action 0 (A), negative favors 1 (B).
    logits = beta * np.array(
        [0.5 * decision_evidence, -0.5 * decision_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- confidence_threshold: [0.05, 0.2]
- validity_sensitivity: [0.75, 1.5]
- integration_exponent: [0.5, 2.5]
- beta: [4.0, 16.0]
- epsilon: [0.0, 0.15]

`rationale`:
The model replaces equal-weight tallying with a genuine validity-gated cascade while avoiding the rigidity of strict Take The Best. In both reported experiments, the highly valid early discriminating cues will generally cross the low-to-moderate absolute confidence threshold, so later agreeing or conflicting cues are ignored. This yields top-cue control in Experiment 1 and approximately margin-invariant response reliability in Experiment 2, matching the observed metrics. The mechanism remains distinguishable from strict TTB across experiments: lowering absolute validities toward 0.5 causes cues to fall below threshold, activating validity-weighted integration and producing graded effects of additional evidence. Because the gate depends on absolute log diagnosticity rather than cue rank alone, holding ranks and signs fixed while changing absolute validities predicts a qualitative transition from one-cue to multi-cue processing. The narrow lapse range and moderately high response temperatures preserve robust majority-level dominance without making behavior deterministic, while all cognitive parameters vary at the subject level.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0000 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0000.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Classify whether choices are predominantly controlled by the highest-ranked
    # discriminating cue on trials where at least three net lower-cue votes point
    # in the opposite direction.
    agreements = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        nz = np.flatnonzero(d != 0)
        if nz.size == 0:
            continue

        # Features are already ordered from highest to lowest validity.
        first = int(nz[0])
        ttb_sign = 1.0 if d[first] > 0 else -1.0  # +1 means A, -1 means B
        tally_margin = float(np.sum(np.sign(d)))  # positive means tally favors A

        # Retain only strong conflicts: tally favors the other option by at
        # least three net cue votes. This selects patterns 2, 3, 5, and 6 and
        # their A/B reversals in the specified design.
        if ttb_sign * tally_margin < 0 and abs(tally_margin) >= 3.0:
            ttb_response = 0 if ttb_sign > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

    if len(agreements) == 0:
        return 0.5

    # A categorical dominance index: 1 if the top discriminating cue controls
    # a strict majority of these high-conflict choices, otherwise 0.
    return float(np.mean(agreements) > 0.5)

```

**Observed (real) value:** 1.0000 (var=0.0000)
**Candidate (simulated) value:** 1.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0196)

### Experiment 2
**Design**
  A=[1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """AIC model-selection indicator for graded tally-margin sensitivity."""
    if data is None or len(data) == 0:
        return 0.0

    margins = []
    successes = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signed_margin = int(np.sum(a > b) - np.sum(b > a))
        if signed_margin == 0:
            continue

        winner = 0 if signed_margin > 0 else 1
        margins.append(abs(signed_margin))
        successes.append(float(int(row["response"]) == winner))

    if len(successes) == 0:
        return 0.0

    margins = np.asarray(margins, dtype=int)
    successes = np.asarray(successes, dtype=float)
    levels = np.arange(1, 7, dtype=float)
    n_by_margin = np.asarray([np.sum(margins == m) for m in range(1, 7)], dtype=float)
    k_by_margin = np.asarray(
        [np.sum(successes[margins == m]) for m in range(1, 7)], dtype=float
    )
    keep = n_by_margin > 0
    levels = levels[keep]
    n_by_margin = n_by_margin[keep]
    k_by_margin = k_by_margin[keep]

    if len(levels) < 2:
        return 0.0

    # Best margin-invariant winner probability: the observable implication
    # of TTB for this design. Its parameterization permits probabilities
    # arbitrarily close to the full interval [0.5, 1].
    n = float(np.sum(n_by_margin))
    k = float(np.sum(k_by_margin))
    q_const = float(np.clip(k / n, 0.5 + 1e-12, 1.0 - 1e-12))
    ll_ttb = k * np.log(q_const) + (n - k) * np.log1p(-q_const)

    # Maximize the exact Tallying likelihood over its stated beta and lapse
    # ranges. The grid is denser where sigmoid(beta*m) changes most rapidly.
    beta = np.concatenate(
        [np.linspace(0.1, 3.0, 241), np.linspace(3.1, 20.0, 170)]
    )
    epsilon = np.linspace(0.0, 0.5, 251)

    core = 1.0 / (1.0 + np.exp(-beta[:, None] * levels[None, :]))
    q = (
        (1.0 - epsilon[None, :, None]) * core[:, None, :]
        + epsilon[None, :, None] * 0.5
    )
    q = np.clip(q, 1e-12, 1.0 - 1e-12)

    ll = np.sum(
        k_by_margin[None, None, :] * np.log(q)
        + (n_by_margin - k_by_margin)[None, None, :] * np.log1p(-q),
        axis=2,
    )
    ll_tally = float(np.max(ll))

    # Tallying has one additional identifiable probability-shape parameter.
    # Thus AIC favors it exactly when the likelihood-ratio deviance exceeds 2.
    deviance = 2.0 * (ll_tally - ll_ttb)
    return float(deviance > 2.0)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Candidate (simulated) value:** 0.0000 (var=0.1204)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.1204)
- pi_1: 0.0000 (var=0.0736)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate matches both canonical empirical signatures exactly. In Experiment 1, it reproduces strict majority control by the highest-ranked discriminating cue even under strong conflict from lower-ranked cues (1.0 observed and simulated), with complete subject-level consistency. In Experiment 2, it reproduces the absence of detectable graded tally-margin sensitivity (0.0 observed and simulated), as expected when an early high-validity cue usually crosses the confidence gate. The nonzero per-subject variance in Experiment 2 (0.1204) indicates some heterogeneous subjects would individually trigger the binary AIC indicator, whereas the observed per-subject variance is zero. However, this metric is a noisy categorical model-selection statistic when applied to subject slices, and the pooled canonical estimate\u2014the primary target\u2014is exactly correct. The implementation is faithful to the prescribed validity-gated sequential family: descending-validity inspection, absolute diagnosticity-based stopping, and validity-weighted integration when no cue crosses threshold are all present.",
  "verdict": "continue",
  "rationale": "The candidate was accepted with aggregate loss 0.0000 and captures the cross-experiment pattern without drifting from the arbiter-prescribed mechanism. Experiment 1 establishes top-cue dominance, while Experiment 2 establishes pooled invariance to additional agreeing cues; both are fit exactly. Although Experiment 2's simulated between-subject variance is higher than observed, that secondary mismatch is insufficient to justify further tuning given the perfect canonical fit and the brittleness of a binary per-subject AIC indicator. Keep this candidate for future experiments that manipulate absolute validity and validity gaps, where its gated integration component can be distinguished from strict TTB."
}
```

## Usage

```json
{
  "input_tokens": 5861,
  "output_tokens": 342
}
```
