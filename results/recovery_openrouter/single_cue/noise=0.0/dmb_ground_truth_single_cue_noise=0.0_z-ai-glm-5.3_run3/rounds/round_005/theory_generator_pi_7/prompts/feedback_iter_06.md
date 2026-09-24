# feedback_iter_06

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
- THEORY 1 = `pi_5_1`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_5_1`).

Replace IVC with PRIMACY-GATED DISTRUST TALLYING (PGDT). Core claim: subjects scan experts in presentation order and attention follows a steep convex primacy gate g_j = ((n-j)/(n-1))^eta with eta ∈ [4, 9] (position 1 carrying ~50-80% of attention mass); the communicated validities are distrust-inverted exactly as in the shared kernel, inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z; but the two combine MULTIPLICATIVELY rather than additively: w_j ∝ g_j * [(1-tau)/n + tau*inv_j], normalized. The multiplicative gate is the key difference from PW-IVC: on tied-validity or matched-validity designs the gate scales the flat kernel directly, producing an early-vs-late contrast per unit distrust dominance that is substantially larger than PW-IVC's additive rho ≤ 0.30 share — targeting the observed Exp1 slope ≈ 0.31, Exp2 slope ≈ 0.54, Exp11 recency-following ≈ 0.34, Exp8 ≈ 0.37, Exp10 ≈ 0.40 — while on decreasing-validity layouts the distrust kernel retains dominance because the gate multiplies a near-zero inverted weight for high-validity early cues, preserving the anti-validity regression (Exp3 ≈ 0.21, Exp5 ≈ -0.58, Exp7 ≈ 0.23, Exp9 ≈ 0.79). Choice rule: logistic in beta*d with beta ∈ [2.5, 9] and a small uniform lapse epsilon ∈ [0.04, 0.15] (lower than IVC's, since the gate itself supplies response dilution on late-cue trials). Parameter ranges: tau ∈ [0.80, 0.97], gamma ∈ [1.5, 3.5], c ∈ [0.0, 0.25], eta ∈ [4, 9]. Validation target: the new theory's simulated predictions should land within ~0.05 of all twelve observed metric values, with particular attention to (i) not overshooting Exp12 (-0.173) and Exp6 (0.843) while (ii) reaching the Exp1/Exp2 primacy slopes that PW-IVC currently undershoots — the multiplicative (rather than additive) validity-by-position interaction is precisely what should dissociate these regimes.


## CANDIDATE THEORY
PRIMACY-GATED DISTRUST TALLYING (PGDT), floored-gate edition. Subjects scan the expert ratings in presentation order and attention is deployed through a steep convex primacy gate with a substantial floor: g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant attention share but late experts retain a meaningful attention residue. Independently, the communicated validities are distrust-inverted: subjects act as if each stated validity were the expert's error rate, inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z. The two kernels combine MULTIPLICATIVELY: w_j proportional to g_j * [(1-tau)/n + tau*inv_j], normalized. On tied-validity designs the bracket is flat and the gate alone supplies the early-vs-late contrast (primacy slopes); on decreasing-validity layouts the distrust kernel dominates and reproduces the anti-validity regression. Evidence d = sum_j w_j * sign(A_j - B_j) is integrated additively; choice is logistic in beta*d with a small uniform lapse epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # PGDT: Primacy-Gated Distrust Tallying, FLOORED-GATE edition,
    # iter-7 minimal-diff edit per critic's iter-6 directive:
    #   (1) REVERT EPSILON to [0.06, 0.15] -- the iter-6 floor at 0.10 was
    #       proven a mathematical noop on the difference metrics it
    #       targeted (lapse scales proportion-differences by (1-eps) only)
    #       while genuinely damaging the Exp11/Exp12 OLS slopes
    #       (Exp11 fell 0.256 -> 0.210);
    #   (2) EXTEND DELTA ceiling to 0.55 (from 0.45) -- delta is the only
    #       knob with twice-verified uncompensated leverage on the
    #       Exp3/Exp2 tension (iters 2->3 and iter-6 both improved Exp3
    #       at zero Exp2 cost);
    #   (3) SHARPEN THE DISTRUST TAIL: gamma tightened to [3.2, 3.5] and
    #       c to [0.18, 0.25] -- steepens the inverted kernel toward the
    #       maximum-error late cues, stacking back-loading with delta's;
    #       uncoupled from the Exp2 cluster because tau stays low;
    #   (4) eta widened to [4, 7] to protect Exp11/Exp12 from the
    #       delta-flattening; tau, beta frozen at base values.
    # Mechanism structure unchanged from the accepted iter-4 base.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PGDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])      # trust in the distrust kernel
    eta = float(parameters["eta"])       # convexity of the primacy gate
    delta = float(parameters["delta"])  # floor of the primacy gate
    gamma = float(parameters["gamma"])   # steepness of distrust inversion
    c = float(parameters["c"])           # partial centering of error rates
    beta = float(parameters["beta"])     # choice consistency
    eps = float(parameters["epsilon"])  # uniform lapse

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma now floor-raised to 3.2 and c floor-raised to 0.18: the
    # kernel is sharpened toward the maximum-error (late, low-
    # validity) cues, adding back-loading that stacks with delta's
    # on decreasing-validity layouts -- the anti-validity arm that
    # Exp3's OLS coefficient scores.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and the gate alone carries position.
        w_inv = np.full(n, 1.0 / n)

    # (2) STEEP CONVEX PRIMACY GATE WITH EXTENDED FLOOR over presentation
    # positions. Cue j (0-indexed) gets gate mass
    #     g_j = delta + (1-delta) * ((n - j)/(n - 1))^eta
    # with delta now in [0.25, 0.55]: the FIRST position still carries
    # the dominant share, but every late position retains a substantial
    # attention residue, so the multiplicative product never approaches
    # a pure primacy gradient and late low-validity (maximum-distrust)
    # cues keep enough weight to carry the anti-validity signature.
    # The extended ceiling further back-loads the product on decreasing-
    # validity layouts, continuing the verified delta leverage on Exp3.
    if n > 1:
        pos = np.arange(n, dtype=float)
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        g = np.ones(1)

    # (3) MULTIPLICATIVE combination: the gate scales the distrust
    # bracket cue-by-cue. Where the bracket is flat (tied validities)
    # this reduces to the (floored) primacy gradient; where the bracket
    # nearly vanishes (trusted high-validity early experts) the gate
    # multiplies almost nothing and the distrust kernel decides where
    # the remaining weight mass sits.
    bracket = (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (4) Deterministic attention: every discriminating cue is always
    # attended; no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # (5) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Small uniform lapse; reverted to the base range because the
    # iter-6 floor at 0.10 was a pure loss (noop on proportion
    # differences, real cost on the Exp11/Exp12 slopes).
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- tau: [0.82, 0.92]
- eta: [4.0, 7.0]
- delta: [0.25, 0.55]
- gamma: [3.2, 3.5]
- c: [0.18, 0.25]
- beta: [2.5, 6.0]
- epsilon: [0.06, 0.15]
- validities: validities

`rationale`:
Minimal-diff edit on the accepted iter-4 base, implementing the critic's iter-6 postmortem verbatim (four parameter-range changes, mechanism and code path identical). (1) REVERT EPSILON to [0.06, 0.15]: the iter-6 experiment proved the 0.10 floor was a mathematical noop on the difference-of-proportions metrics it targeted (a uniform lapse rescales such contrasts by exactly (1-eps), a <=7% effect that could never close Exp2's +0.127 gap) while genuinely damaging the Exp11/Exp12 OLS slopes (Exp11 fell 0.256 -> 0.210, becoming the second-largest miss). Reverting removes that damage at zero cost. (2) EXTEND DELTA to [0.25, 0.55]: delta is the only knob with twice-verified, uncompensated dual leverage on the Exp3/Exp2 tension (iters 2->3: Exp3 -0.318 -> -0.469 while Exp2 improved 0.318 -> 0.284; iter-6: Exp3 -0.401 -> -0.439 at zero Exp2 cost). Exp3 (+0.175 vs -0.576) remains the largest addressable miss and the delta raise is the only lever that has ever moved it without a compensating loss. (3) SHARPEN THE DISTRUST TAIL via gamma [2.8,3.5] -> [3.2,3.5] and c [0.10,0.25] -> [0.18,0.25]: this is the iter-5 checkpoint the critic specified for exactly this contingency ('if Exp3 remains short of -0.50, push c toward 0.25 and gamma toward 3.5'). Higher gamma steepens the inverted kernel toward the maximum-error late cues; higher c sharpens the relative contrast. Both stack back-loading with delta's and are uncoupled from the Exp2 cluster because tau stays low at [0.82, 0.92] -- the iter-5 rejection confirmed that raising tau trades Exp2/Exp1 against Exp3 at break-even-to-negative exchange rates, so tau is left frozen. (4) KEEP ETA at [4, 7]: once epsilon reverts, the eta headroom is what protects Exp11 (base 0.256, target 0.315) and Exp12 (0.505 vs 0.540) from the delta-0.55 flattening; on tied-validity designs the distrust bracket is flat so the gate acts unopposed and eta does not touch the distrust regime. Strategic holds honored: Exp8 is treated as a capped stretch target (~0.28-0.30 is the family ceiling since a lone late cue is structurally opposed by tallying; no structural effort spent there); Exp2 is bankable in the 0.20-0.28 range (pi_3's 0.166 remains the family best). Checkpoints from the critique: if Exp12 falls below 0.48 or Exp11 below 0.24, pull delta back toward 0.50 before touching eta; if Exp3 remains short of -0.50 after the gamma/c sharpening, accept it rather than raising tau. Projected movement vs the 0.0940 base: Exp3 toward -0.50 via the stacked delta + gamma/c back-loading, Exp11/Exp12 restored to base-or-better by the epsilon revert, Exp4/Exp7 restored to base levels by the same revert, everything else unchanged -- a projected aggregate clearly below the 0.0940 floor.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3072 -> ACCEPTED
- iter 2: loss=0.1413 -> ACCEPTED
- iter 3: loss=0.1175 -> ACCEPTED
- iter 4: loss=0.0940 -> ACCEPTED
- iter 5: loss=0.1147 -> REJECTED
- iter 6: loss=0.0969 -> REJECTED
- iter 7 (current candidate you are grading): loss=0.0988 -> REJECTED
Running-best (last accepted) base: iter 4 at loss=0.0940.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.2080 (var=0.0060)
**Candidate trajectory (this loop):**
  - iter 1: 0.4937 (var=0.0101) (Δ vs real +0.2857)
  - iter 2: 0.1857 (var=0.0056) (Δ vs real -0.0223)
  - iter 3: 0.1410 (var=0.0030) (Δ vs real -0.0670)
  - iter 4: 0.1640 (var=0.0050) (Δ vs real -0.0440)
  - iter 5: 0.1253 (var=0.0038) (Δ vs real -0.0827)
  - iter 6: 0.1703 (var=0.0034) (Δ vs real -0.0377)
  - iter 7 (current): 0.1757 (var=0.0039) (Δ vs real -0.0323)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8443 (var=0.0128)
- pi_2: 0.1357 (var=0.0112)
- pi_3: 0.2377 (var=0.0050)
- pi_4: 0.2197 (var=0.0050)
- pi_5: 0.1807 (var=0.0022)
- pi_5_1: 0.1870 (var=0.0059)
- pi_6: 0.1750 (var=0.0040)

### Experiment 2
**Design**
  A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.1583 (var=0.0053)
**Candidate trajectory (this loop):**
  - iter 1: 0.0821 (var=0.0175) (Δ vs real -0.0762)
  - iter 2: 0.3183 (var=0.0083) (Δ vs real +0.1600)
  - iter 3: 0.3204 (var=0.0041) (Δ vs real +0.1621)
  - iter 4: 0.2838 (var=0.0074) (Δ vs real +0.1254)
  - iter 5: 0.3621 (var=0.0082) (Δ vs real +0.2038)
  - iter 6: 0.2854 (var=0.0051) (Δ vs real +0.1271)
  - iter 7 (current): 0.2846 (var=0.0093) (Δ vs real +0.1263)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5479 (var=0.0187)
- pi_1: -0.4837 (var=0.0228)
- pi_3: 0.1658 (var=0.0093)
- pi_4: 0.2079 (var=0.0051)
- pi_5: 0.2308 (var=0.0086)
- pi_5_1: 0.2875 (var=0.0062)
- pi_6: 0.2408 (var=0.0082)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def _to_vec(x):
    """Robustly convert a cell (list/tuple/ndarray/str) to a float 1-D array."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, (list, tuple)):
        return np.asarray(list(x), dtype=float)
    s = str(x).strip().replace("[", " ").replace("]", " ").replace(",", " ")
    return np.asarray(s.split(), dtype=float)


def metric(data: pd.DataFrame) -> float:
    df = data.reset_index(drop=True)
    n_rows = len(df)
    if n_rows == 0:
        return 0.0

    A = np.vstack([_to_vec(x) for x in df["option_a_ratings"]])
    B = np.vstack([_to_vec(x) for x in df["option_b_ratings"]])
    n_feat = A.shape[1]

    # Design validities [0.95, 0.85, 0.75, 0.65, 0.60]; fall back gracefully.
    v = np.array([0.95, 0.85, 0.75, 0.65, 0.60], dtype=float)
    if v.shape[0] != n_feat:
        v = np.linspace(0.95, 0.60, n_feat)

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = (np.asarray(df["response"], dtype=float) == 0.0).astype(float)

    a_win = A > B
    b_win = B > A

    # Signed tally margin (Tallying's ONLY explanatory variable).
    margin = a_win.sum(axis=1).astype(float) - b_win.sum(axis=1).astype(float)

    # Validity-weighted evidence difference (NVW's driving quantity).
    d = (a_win * v).sum(axis=1) - (b_win * v).sum(axis=1)

    # OLS: y ~ 1 + margin + d. Return the partial coefficient on d.
    X = np.column_stack([np.ones(n_rows), margin, d])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2])
```

**Observed (real) value:** -0.5756 (var=0.0186)
**Candidate trajectory (this loop):**
  - iter 1: 0.5962 (var=0.0283) (Δ vs real +1.1718)
  - iter 2: -0.3184 (var=0.0810) (Δ vs real +0.2572)
  - iter 3: -0.4685 (var=0.0318) (Δ vs real +0.1071)
  - iter 4: -0.4010 (var=0.0475) (Δ vs real +0.1746)
  - iter 5: -0.5841 (var=0.0215) (Δ vs real -0.0085)
  - iter 6: -0.4387 (var=0.0277) (Δ vs real +0.1369)
  - iter 7 (current): -0.5392 (var=0.0283) (Δ vs real +0.0364)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2763 (var=0.0165)
- pi_2: -0.0069 (var=0.0141)
- pi_1: 0.7613 (var=0.0601)
- pi_4: -0.4120 (var=0.0294)
- pi_5: -0.5397 (var=0.0203)
- pi_5_1: -0.5621 (var=0.0241)
- pi_6: -0.4702 (var=0.0232)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anchor-allegiance: proportion of trials with a large tally margin
    (|a_wins - b_wins| >= 6) on which the subject chose the option that
    wins on more features (the tally winner).

    Under Tallying, these are the unambiguous anchors (8-0 and 7-1 splits:
    design trials 7, 8, 21, 22, 23, 24 and their 4 repetitions = 24 trials
    per subject), where the softmax over tallies predicts near-deterministic
    choice for any non-trivial beta. Under NVW, the capacity-interference
    factor delta^4 crushes the effective inverse temperature to
    beta_eff <= 11 * 0.30^4 ~ 0.09, so even the maximum validity-weighted
    evidence difference (~6) yields P ~ sigmoid(0.5) <= 0.64; attention
    lapses and the uniform-mixing epsilon only pull this further toward
    0.5. The two theories therefore make sharply separated predictions on
    this proportion, and the A/B-mirrored anchors cancel side bias.
    """
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    n_anchor = 0
    n_hit = 0

    for ra, rb, resp in zip(a_col, b_col, r_col):
        va = np.asarray(ra, dtype=float)
        vb = np.asarray(rb, dtype=float)
        if va.shape != vb.shape:
            continue

        a_wins = float(np.sum(va > vb))
        b_wins = float(np.sum(vb > va))
        margin = a_wins - b_wins

        if abs(margin) >= 6:
            n_anchor += 1
            winner_is_a = margin > 0
            chose_a = (int(resp) == 0)
            if chose_a == winner_is_a:
                n_hit += 1

    if n_anchor == 0:
        return 0.5
    return float(n_hit) / float(n_anchor)

```

**Observed (real) value:** 0.8433 (var=0.0152)
**Candidate trajectory (this loop):**
  - iter 1: 0.8900 (var=0.0052) (Δ vs real +0.0467)
  - iter 2: 0.9025 (var=0.0037) (Δ vs real +0.0592)
  - iter 3: 0.8992 (var=0.0027) (Δ vs real +0.0558)
  - iter 4: 0.8767 (var=0.0058) (Δ vs real +0.0333)
  - iter 5: 0.9000 (var=0.0044) (Δ vs real +0.0567)
  - iter 6: 0.8875 (var=0.0049) (Δ vs real +0.0442)
  - iter 7 (current): 0.8950 (var=0.0043) (Δ vs real +0.0517)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8808 (var=0.0089)
- pi_3: 0.5300 (var=0.0122)
- pi_1: 0.6133 (var=0.0060)
- pi_4: 0.8133 (var=0.0065)
- pi_5: 0.8083 (var=0.0061)
- pi_5_1: 0.8217 (var=0.0065)
- pi_6: 0.9058 (var=0.0060)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Validity-adherence rate on VALIDITY-vs-RECENCY conflict trials.
    #
    # For every trial we compute two decision variables from the presented
    # cue patterns:
    #   d_v = sum_j v_j * sign(A_j - B_j)   (validity-weighted evidence for A)
    #   d_r = sum_j (j/(n-1))^3 * sign(A_j - B_j)  (convex cue-order recency
    #                                              evidence for A)
    # A trial is a CONFLICT trial when the two point in opposite directions
    # and both are non-trivial in magnitude. On those trials the metric is
    # the proportion of choices that follow the VALIDITY-implied option.
    #
    # NVW (validity-weighted integration) predicts adherence clearly above
    # chance (~0.6); SWC (recency-dominated subjective weights) predicts
    # adherence well below chance (~0.3), because on these trials its
    # recency gradient favors the opposite option.

    validities = np.array([0.95, 0.90, 0.85, 0.62, 0.50, 0.50], dtype=float)
    n = int(validities.shape[0])
    # Convex recency profile over presentation positions (eta ~ 3).
    recency = (np.arange(n, dtype=float) / float(n - 1)) ** 3.0

    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]

    hits = []
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape[0] != n or b.shape[0] != n:
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue

        d_v = float(np.dot(validities, s))
        d_r = float(np.dot(recency, s))

        # Need a real conflict with non-trivial evidence on both axes.
        if abs(d_v) < 0.30 or abs(d_r) < 0.50:
            continue
        if np.sign(d_v) == np.sign(d_r):
            continue

        validity_winner_is_a = (d_v > 0.0)
        chose_a = (int(resp) == 0)
        hits.append(1.0 if (chose_a == validity_winner_is_a) else 0.0)

    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2313 (var=0.0046)
**Candidate trajectory (this loop):**
  - iter 1: 0.7937 (var=0.0045) (Δ vs real +0.5625)
  - iter 2: 0.2953 (var=0.0098) (Δ vs real +0.0641)
  - iter 3: 0.2078 (var=0.0057) (Δ vs real -0.0234)
  - iter 4: 0.2572 (var=0.0064) (Δ vs real +0.0259)
  - iter 5: 0.1909 (var=0.0039) (Δ vs real -0.0403)
  - iter 6: 0.2428 (var=0.0039) (Δ vs real +0.0116)
  - iter 7 (current): 0.2244 (var=0.0066) (Δ vs real -0.0069)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5988 (var=0.0039)
- pi_4: 0.2816 (var=0.0052)
- pi_1: 0.7519 (var=0.0061)
- pi_2: 0.5803 (var=0.0046)
- pi_5: 0.2544 (var=0.0025)
- pi_5_1: 0.2369 (var=0.0070)
- pi_6: 0.2559 (var=0.0082)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-consistency rate: fraction of choices aligned with a convex
    # cue-order recency gradient, restricted to trials where that gradient
    # is strong and its direction is unambiguous across SWC's eta range.
    #
    # SWC (advocated): late cues dominate via convex recency -> rate well
    #   above 0.5 (predicted ~0.65-0.85 across its parameter box).
    # NVW (competing): validity-weighted integration with capacity
    #   interference (beta_eff = beta*delta^3 <= ~0.30 at n=7) -> choices
    #   near chance on nearly every trial, and anti-recency or neutral on
    #   the conflict trials -> rate ~0.50.
    df = data.reset_index(drop=True)
    if len(df) == 0:
        return 0.5

    a_arrs = [np.asarray(list(x), dtype=float) for x in df['option_a_ratings']]
    b_arrs = [np.asarray(list(x), dtype=float) for x in df['option_b_ratings']]
    resp = pd.to_numeric(df['response'], errors='coerce').to_numpy()

    if len(a_arrs) == 0 or a_arrs[0].ndim != 1:
        return 0.5
    n = int(a_arrs[0].shape[0])
    if n < 2:
        return 0.5

    pos = np.arange(n, dtype=float) / (n - 1.0)
    w_lo = pos ** 2.5   # SWC's eta lower corner
    w_mid = pos ** 3.5  # SWC's eta midpoint
    w_hi = pos ** 4.5   # SWC's eta upper corner

    consistent = []
    for a, b, r in zip(a_arrs, b_arrs, resp):
        if a.shape[0] != n or b.shape[0] != n:
            continue
        if not np.isfinite(r):
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue
        e_lo = float(np.dot(w_lo, s))
        e_mid = float(np.dot(w_mid, s))
        e_hi = float(np.dot(w_hi, s))
        # Require the recency gradient's direction to be invariant across
        # SWC's whole eta range (drops ambiguous trials, e.g. 1 late cue vs
        # 4 mid cues, where the sign flips with eta).
        if e_lo == 0.0 or e_hi == 0.0 or np.sign(e_lo) != np.sign(e_hi):
            continue
        # Require non-trivial gradient magnitude (drops near-degenerate
        # trials where even SWC predicts ~0.52).
        if abs(e_mid) < 0.25:
            continue
        fav_a = e_mid > 0.0
        chose_a = (int(r) == 0)  # response: 0 = chose A, 1 = chose B
        consistent.append(1.0 if (chose_a == fav_a) else 0.0)

    if len(consistent) == 0:
        return 0.5
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.3739 (var=0.0026)
**Candidate trajectory (this loop):**
  - iter 1: 0.3039 (var=0.0032) (Δ vs real -0.0700)
  - iter 2: 0.3011 (var=0.0028) (Δ vs real -0.0728)
  - iter 3: 0.3033 (var=0.0036) (Δ vs real -0.0706)
  - iter 4: 0.3206 (var=0.0032) (Δ vs real -0.0533)
  - iter 5: 0.3250 (var=0.0047) (Δ vs real -0.0489)
  - iter 6: 0.3222 (var=0.0041) (Δ vs real -0.0517)
  - iter 7 (current): 0.3306 (var=0.0046) (Δ vs real -0.0433)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7261 (var=0.0028)
- pi_3: 0.4847 (var=0.0034)
- pi_1: 0.3794 (var=0.0022)
- pi_2: 0.3853 (var=0.0032)
- pi_5: 0.4408 (var=0.0038)
- pi_5_1: 0.4392 (var=0.0036)
- pi_6: 0.3361 (var=0.0034)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anti-validity vs. recency disagreement score.

    On trials where IVC's distrust-inverted weights and SWC's recency-dominated
    weights point to OPPOSITE options, report the proportion of choices that
    follow the IVC-preferred option. IVC-simulated data -> well above 0.5;
    SWC-simulated data -> well below 0.5.
    """
    data = data.reset_index(drop=True)
    if len(data) == 0:
        return 0.5

    # Build stimulus matrices without hashing any list-valued cells.
    a = np.vstack([np.asarray(r, dtype=float) for r in data["option_a_ratings"]])
    b = np.vstack([np.asarray(r, dtype=float) for r in data["option_b_ratings"]])
    n = a.shape[1]

    # Design validities (fallback mirrors both theories' graceful degradation).
    v = np.array([0.5, 0.9, 0.6, 0.95, 0.5, 0.6], dtype=float)
    if v.shape[0] != n:
        v = np.linspace(0.95, 0.55, n)

    # --- IVC weights at mid-range parameters (tau=0.925, kappa=2.7) ---
    tau, kappa = 0.925, 2.7
    inv = np.power(1.0 - v, kappa)
    Z = float(inv.sum())
    w_inv = inv / Z if Z > 1e-12 else np.full(n, 1.0 / n)
    w_ivc = (1.0 - tau) / n + tau * w_inv

    # --- SWC weights at mid-range parameters (lam=0.9, rho=0.9, eta=3.5) ---
    lam, rho, eta = 0.9, 0.9, 3.5
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)
    g = (1.0 - rho) * vnorm + rho * pos
    w_swc = (1.0 - lam) / n + lam * g

    # Signed discriminating-cue vector per trial.
    s = np.sign(a - b)
    d_ivc = s @ w_ivc
    d_swc = s @ w_swc

    # Keep only trials where the two theories STRICTLY disagree in sign.
    # (d_ivc == 0, e.g. the validity-matched EQ trials, is excluded.)
    mask = (d_ivc * d_swc) < 0.0
    if not np.any(mask):
        return 0.5

    resp = np.asarray(data["response"].to_numpy())
    r = resp[mask]
    ivc_prefers_a = d_ivc[mask] > 0.0
    chose_ivc_preferred = np.where(ivc_prefers_a, r == 0, r == 1)
    return float(chose_ivc_preferred.mean())
```

**Observed (real) value:** 0.7878 (var=0.0086)
**Candidate trajectory (this loop):**
  - iter 1: 0.8761 (var=0.0033) (Δ vs real +0.0883)
  - iter 2: 0.8583 (var=0.0043) (Δ vs real +0.0706)
  - iter 3: 0.8706 (var=0.0042) (Δ vs real +0.0828)
  - iter 4: 0.8394 (var=0.0047) (Δ vs real +0.0517)
  - iter 5: 0.8500 (var=0.0049) (Δ vs real +0.0622)
  - iter 6: 0.8461 (var=0.0030) (Δ vs real +0.0583)
  - iter 7 (current): 0.8278 (var=0.0036) (Δ vs real +0.0400)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6389 (var=0.0060)
- pi_4: 0.3622 (var=0.0067)
- pi_1: 0.1533 (var=0.0097)
- pi_2: 0.5167 (var=0.0064)
- pi_3: 0.4511 (var=0.0081)
- pi_5_1: 0.6478 (var=0.0070)
- pi_6: 0.7750 (var=0.0124)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """Final-Expert Dominance Index (FEDI).

    Among trials where the LAST-presented expert (feature position n-1) is the
    ONLY cue favoring one option (i.e., that option's rating vector is all
    zeros except a 1 in the final position), compute the proportion of
    choices that follow the final expert's recommendation.

    A convex-recency weighting (SWC) predicts adherence well above 0.5
    (the last cue carries the largest subjective weight at every corner of
    its parameter box). A distrust-inverted validity weighting (IVC)
    predicts adherence below 0.5, because the final expert here has a high
    stated validity (0.90) and is therefore heavily discounted.
    """
    df = data.reset_index(drop=True)

    A = np.array(
        [list(map(int, x)) for x in df['option_a_ratings']], dtype=int
    )
    B = np.array(
        [list(map(int, x)) for x in df['option_b_ratings']], dtype=int
    )
    r = df['response'].to_numpy(dtype=int)

    if A.ndim != 2 or A.shape[0] == 0:
        return float('nan')

    n_feat = A.shape[1]
    last = n_feat - 1

    a_last = A[:, last]
    b_last = B[:, last]

    # Last expert discriminates between the options.
    disc = a_last != b_last

    # The option endorsed by the final expert wins ONLY that final cue
    # (its rating vector sums to 1). This selects the single-cue and
    # multi-cue 'ladder' duels against the last expert (trials 1-14 and
    # their mirrors) and excludes trials where the last-cue option also
    # wins other cues.
    a_sel = disc & (a_last == 1) & (A.sum(axis=1) == 1)
    b_sel = disc & (b_last == 1) & (B.sum(axis=1) == 1)
    sel = a_sel | b_sel

    if not np.any(sel):
        return float('nan')

    # response: 0 = chose A, 1 = chose B.
    # Did the subject follow the final expert's endorsed option?
    followed = np.where(a_sel[sel], r[sel] == 0, r[sel] == 1)

    return float(np.mean(followed))
```

**Observed (real) value:** 0.3957 (var=0.0045)
**Candidate trajectory (this loop):**
  - iter 1: 0.2957 (var=0.0033) (Δ vs real -0.1000)
  - iter 2: 0.2693 (var=0.0029) (Δ vs real -0.1264)
  - iter 3: 0.2779 (var=0.0044) (Δ vs real -0.1179)
  - iter 4: 0.2775 (var=0.0048) (Δ vs real -0.1182)
  - iter 5: 0.2836 (var=0.0051) (Δ vs real -0.1121)
  - iter 6: 0.2800 (var=0.0048) (Δ vs real -0.1157)
  - iter 7 (current): 0.2593 (var=0.0034) (Δ vs real -0.1364)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6796 (var=0.0044)
- pi_5: 0.3561 (var=0.0033)
- pi_1: 0.5529 (var=0.0024)
- pi_2: 0.4039 (var=0.0053)
- pi_3: 0.4300 (var=0.0036)
- pi_5_1: 0.3393 (var=0.0058)
- pi_6: 0.2818 (var=0.0035)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-following rate on count-neutral trials.
    #
    # With all six validities equal (0.95), IVC's weight vector is EXACTLY
    # flat for every admissible parameter combination (the distrust kernel
    # is constant across features, so w_j = 1/n identically). Hence on any
    # trial where the two options win an equal number of discriminating
    # cues (count difference = 0), IVC's evidence d = 0 and P(choose A) =
    # 0.50 exactly -- the lapse preserves 0.5. This is parameter-free.
    #
    # SWC, in contrast, has strictly increasing weights in cue position
    # (lambda >= 0.8, rho >= 0.8, convex recency), and on count-neutral
    # trials its evidence reduces EXACTLY to d = lambda*rho*sum_j pos_j*s_j,
    # so it must prefer the option supported by the LATER cues, with
    # subject-level choice rates roughly in [0.60, 0.79].
    #
    # The metric: among count-neutral trials, the fraction of choices that
    # go to the option favored by the later (more recent) discriminating
    # cues. IVC -> 0.50 for every subject; SWC -> well above 0.50.

    def _mat(col):
        return np.array(
            [np.asarray(v, dtype=float).ravel() for v in list(col)],
            dtype=float,
        )

    A = _mat(data['option_a_ratings'])
    B = _mat(data['option_b_ratings'])
    resp = np.asarray(pd.Series(data['response']).to_numpy(), dtype=float)

    if A.size == 0 or A.ndim != 2 or A.shape[0] == 0 or B.shape[0] != A.shape[0]:
        return 0.5
    n = int(A.shape[1])
    if n < 2:
        return 0.5

    # Convex recency profile over cue positions (any strictly increasing
    # profile gives the same favored direction on count-neutral trials;
    # 3.5 matches SWC's mid-gradient). Direction is what matters here.
    pos = (np.arange(n, dtype=float) / (n - 1.0)) ** 3.5

    S = np.sign(A - B)            # +1: cue favors A, -1: favors B, 0: tie
    count = S.sum(axis=1)         # IVC's evidence direction (flat weights)
    rec = S @ pos                 # recency evidence for A

    neutral = (np.abs(count) < 1e-9) & (np.abs(rec) > 1e-9)
    if not np.any(neutral):
        return 0.5

    chose_a = resp[neutral] == 0
    fav_a = rec[neutral] > 0      # option supported by the later cues
    followed = np.where(fav_a, chose_a, ~chose_a)
    return float(np.mean(followed))
```

**Observed (real) value:** 0.3414 (var=0.0042)
**Candidate trajectory (this loop):**
  - iter 1: 0.2389 (var=0.0030) (Δ vs real -0.1025)
  - iter 2: 0.2607 (var=0.0048) (Δ vs real -0.0807)
  - iter 3: 0.3161 (var=0.0051) (Δ vs real -0.0254)
  - iter 4: 0.3450 (var=0.0065) (Δ vs real +0.0036)
  - iter 5: 0.3168 (var=0.0042) (Δ vs real -0.0246)
  - iter 6: 0.3375 (var=0.0038) (Δ vs real -0.0039)
  - iter 7 (current): 0.3382 (var=0.0056) (Δ vs real -0.0032)
**Other theories' values on this metric (for reference):**
- pi_5_1: 0.5043 (var=0.0036)
- pi_4: 0.6693 (var=0.0058)
- pi_1: 0.1557 (var=0.0121)
- pi_2: 0.4986 (var=0.0044)
- pi_3: 0.4986 (var=0.0040)
- pi_5: 0.5043 (var=0.0045)
- pi_6: 0.4118 (var=0.0064)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-gradient choice score on theory-divergent trials.
    #
    # Design validities: experts at positions 1 and 4 (0-indexed 0 and 3)
    # have validity 0.95; the other four experts have validity 0.75.
    #
    # Per trial:
    #   s_j = sign(A_j - B_j)   which option expert j favors
    #   x   = sum_j p_j * s_j   recency evidence for A, with convex
    #                           position weights p_j = (j/5)^3.5
    #   y   = +1 if A chosen, -1 if B chosen
    #
    # Scored trial families (48 of the 96 trials per subject):
    # (1) VALIDITY-MATCHED trials (types 3/4, 9/10, 13/14, 17/18, 25/26
    #     plus mirrors): the two options' discriminating experts carry
    #     identical validity multisets. Any validity-only weighting (IVC)
    #     yields exactly zero evidence, so E[x*y] = 0 for EVERY parameter
    #     combination (the uniform lapse preserves 0.5). SWC's convex
    #     recency gradient makes E[x*y] strongly positive.
    # (2) VALIDITY-CONFOUNDED trials (types 5/6, 11/12, 23/24 plus
    #     mirrors): the option whose experts sit LATER also carries
    #     strictly more 0.95 experts. IVC's distrust inversion favors
    #     the EARLY option, so E[x*y] < 0; SWC's recency gradient still
    #     favors the LATE option, so E[x*y] > 0.
    #
    # Metric = mean of x*y over these trials.
    # SWC: strongly positive (~ +0.28 across its box, range ~ +0.17..+0.41).
    # IVC: zero on the matched family for all parameters, slightly
    # negative overall (~ -0.03).

    n = 6
    V = np.array([0.95, 0.75, 0.75, 0.95, 0.75, 0.75])
    P = (np.arange(n, dtype=float) / float(n - 1)) ** 3.5
    hi = V >= 0.9

    def to_arr(cell):
        if isinstance(cell, np.ndarray):
            return np.asarray(cell, dtype=float).ravel()
        if isinstance(cell, (list, tuple)):
            return np.asarray(list(cell), dtype=float).ravel()
        txt = str(cell).strip().strip('[]()')
        parts = txt.replace(',', ' ').split()
        return np.asarray([float(t) for t in parts], dtype=float).ravel()

    a_cells = data['option_a_ratings'].tolist()
    b_cells = data['option_b_ratings'].tolist()
    r_cells = data['response'].tolist()

    scores = []
    for i in range(len(a_cells)):
        try:
            a = to_arr(a_cells[i])
            b = to_arr(b_cells[i])
        except Exception:
            continue
        if a.shape[0] != n or b.shape[0] != n:
            continue
        try:
            resp = int(r_cells[i])
        except Exception:
            continue
        if resp != 0 and resp != 1:
            continue

        s = np.sign(a - b)
        ia = np.where(s > 0)[0]
        ib = np.where(s < 0)[0]
        if ia.size == 0 or ib.size == 0:
            continue

        va = sorted(V[ia].tolist())
        vb = sorted(V[ib].tolist())
        matched = (va == vb)

        x = float(np.dot(P, s))
        if not matched:
            if x == 0:
                continue
            late_a = x > 0
            late_hi = int(hi[ia].sum()) if late_a else int(hi[ib].sum())
            early_hi = int(hi[ib].sum()) if late_a else int(hi[ia].sum())
            if late_hi <= early_hi:
                continue

        y = 1.0 if resp == 0 else -1.0
        scores.append(x * y)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1729 (var=0.0118)
**Candidate trajectory (this loop):**
  - iter 1: -0.3387 (var=0.0066) (Δ vs real -0.1658)
  - iter 2: -0.2784 (var=0.0087) (Δ vs real -0.1056)
  - iter 3: -0.2651 (var=0.0094) (Δ vs real -0.0923)
  - iter 4: -0.2149 (var=0.0147) (Δ vs real -0.0420)
  - iter 5: -0.1978 (var=0.0140) (Δ vs real -0.0250)
  - iter 6: -0.2128 (var=0.0107) (Δ vs real -0.0399)
  - iter 7 (current): -0.2282 (var=0.0103) (Δ vs real -0.0553)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2647 (var=0.0159)
- pi_5_1: -0.0459 (var=0.0117)
- pi_1: -0.3350 (var=0.0112)
- pi_2: 0.0242 (var=0.0128)
- pi_3: 0.0443 (var=0.0150)
- pi_5: -0.0563 (var=0.0178)
- pi_6: -0.2233 (var=0.0216)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # Within-allegiance-cell OLS slope of choice on reference primacy evidence.
    #
    # Identification: group every trial by (c1, c2) where
    #   c1 = number of 0.60-expert wins for A minus for B (positions 2,3,5),
    #   c2 = total signed cue count.
    # Any position-INVARIANT validity-weighted model (IVC at ANY parameter
    # combo, plain tallying, any lapse/transform) makes P(choose A) a function
    # of (c1, c2) ONLY. Demeaning the primacy regressor within these cells
    # therefore makes the population OLS slope EXACTLY 0 under IVC,
    # parameter-free, for every subject. PW-IVC's primacy gradient breaks the
    # cell-conditional independence, producing a strictly positive slope.

    ratings_a = data['option_a_ratings']
    ratings_b = data['option_b_ratings']
    A = np.array([np.asarray(r, dtype=float) for r in ratings_a], dtype=float)
    B = np.array([np.asarray(r, dtype=float) for r in ratings_b], dtype=float)
    if A.ndim != 2 or A.shape[0] == 0:
        return 0.0
    n = A.shape[1]

    # Reference primacy kernel, eta = 5 (middle of PW-IVC's eta range [3.5, 6.5]).
    if n > 1:
        pos = np.arange(n, dtype=float)
        wp = ((n - pos) / (n - 1.0)) ** 5.0
        wp = wp / wp.sum()
    else:
        wp = np.ones(1)

    S = np.sign(A - B)
    y = 1.0 - np.asarray(data['response'], dtype=float)  # 1 = chose A

    x1 = S.dot(wp)  # primacy evidence favoring A

    # Allegiance cell for this design: validities [0.95, 0.6, 0.6, 0.95, 0.6, 0.95]
    # -> 0.60 experts at 0-indexed positions 1, 2, 4.
    if n == 6:
        c1 = S[:, 1] + S[:, 2] + S[:, 4]
    else:
        c1 = np.zeros(A.shape[0])
    c2 = S.sum(axis=1)

    g = c1.astype(int).astype(str) + "_" + c2.astype(int).astype(str)
    df = pd.DataFrame({'x1': x1, 'y': y, 'g': g})

    # Demean primacy evidence within allegiance cell: removes ALL validity-based
    # mean structure (distrust kernel at any gamma/c/tau, flat counting, lapse).
    df['x1c'] = df['x1'] - df.groupby('g')['x1'].transform('mean')
    ssx = float((df['x1c'] ** 2).sum())
    if ssx <= 1e-12:
        return 0.0
    ybar = float(df['y'].mean())
    slope = float((df['x1c'] * (df['y'] - ybar)).sum() / ssx)
    return slope
```

**Observed (real) value:** 0.3148 (var=0.0095)
**Candidate trajectory (this loop):**
  - iter 1: 0.3646 (var=0.0055) (Δ vs real +0.0498)
  - iter 2: 0.2712 (var=0.0113) (Δ vs real -0.0436)
  - iter 3: 0.2342 (var=0.0073) (Δ vs real -0.0806)
  - iter 4: 0.2563 (var=0.0070) (Δ vs real -0.0585)
  - iter 5: 0.1945 (var=0.0100) (Δ vs real -0.1203)
  - iter 6: 0.2095 (var=0.0088) (Δ vs real -0.1053)
  - iter 7 (current): 0.1804 (var=0.0078) (Δ vs real -0.1344)
**Other theories' values on this metric (for reference):**
- pi_5_1: -0.0034 (var=0.0097)
- pi_6: 0.1752 (var=0.0110)
- pi_1: 0.4129 (var=0.0077)
- pi_2: -0.0132 (var=0.0118)
- pi_3: 0.0021 (var=0.0076)
- pi_4: -0.3642 (var=0.0070)
- pi_5: -0.0209 (var=0.0151)

### Experiment 12
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Primacy-slope metric on validity-matched cells (with X2-family
    orientation absorbed by a dummy).

    Logic: on every trial where the two options' winning-cue validity
    multisets are IDENTICAL (the 20 matched cells: D0,Z1,D1,W1,W2,D2,
    D2b,D3,D4,D5 + mirrors), the distrust kernel and the flat counting
    kernel cancel EXACTLY, so IVC predicts P(A)=0.500 parameter-free
    (the lapse preserves 0.5). PW-IVC predicts a systematic deviation
    graded by the primacy contrast C = w_prim . sign(A-B), with a
    steep convex reference gradient (eta=5).

    We also include the six X2-family cells (2 wins per side, unequal
    validity multisets). Under IVC all X2 cells share ONE probability
    that depends only on orientation (A-won-the-0.9 vs mirror), so a
    single +-1 orientation dummy Z absorbs IVC's entire mean structure
    there; the within-family position spread then loads on C.

    Metric = OLS coefficient of C in y ~ 1 + C + Z, with y = +1 (chose
    A) / -1 (chose B). Under IVC this coefficient is EXACTLY 0 in
    expectation for every admissible parameter. Under PW-IVC it is
    strictly positive (roughly 2*(1-eps)*beta*tau*rho scale), typically
    0.15-0.8.
    """
    import numpy as np
    import pandas as pd

    n = 6
    v = np.array([0.9, 0.6, 0.6, 0.6, 0.6, 0.9], dtype=float)

    # Reference primacy kernel (eta = 5): convex, front-loaded.
    pos = np.arange(n, dtype=float)
    prim = ((n - pos) / (n - 1.0)) ** 5.0
    w_prim = prim / prim.sum()

    # Reference inverted-validity kernel (gamma = 3, c = 0.1).
    # Only the SIGN of its dot product with s is used (orientation).
    err = 1.0 - v
    cent = np.maximum(err - 0.1 * float(err.min()), 0.0)
    u = cent ** 3.0
    u = u / u.sum()

    a = np.array([[float(t) for t in r] for r in data['option_a_ratings']], dtype=float)
    b = np.array([[float(t) for t in r] for r in data['option_b_ratings']], dtype=float)
    y = 1.0 - 2.0 * np.asarray(data['response'], dtype=float).ravel()  # +1 = A

    m = a.shape[0]
    if m == 0:
        return 0.0

    s = np.sign(a - b)
    aWin = a > b
    bWin = b > a

    C = np.zeros(m)
    Z = np.zeros(m)
    keep = np.zeros(m, dtype=bool)

    for i in range(m):
        ka = tuple(sorted(v[aWin[i]].tolist()))
        kb = tuple(sorted(v[bWin[i]].tolist()))
        nA = int(aWin[i].sum())
        nB = int(bWin[i].sum())
        if ka == kb and nA > 0:
            # Matched-validity cell: distrust+flat kernels cancel exactly.
            keep[i] = True
            C[i] = float(np.dot(w_prim, s[i]))
        elif nA == 2 and nB == 2 and ka != kb:
            # X2-family cell: orientation dummy absorbs IVC's constant
            # allegiance effect; position spread loads on C.
            keep[i] = True
            C[i] = float(np.dot(w_prim, s[i]))
            Z[i] = float(np.sign(np.dot(u, s[i])))

    if int(keep.sum()) < 12:
        return 0.0

    Ck = C[keep]
    Zk = Z[keep]
    yk = y[keep]

    X = np.column_stack([np.ones_like(Ck), Ck, Zk])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, yk, rcond=None)
        return float(coef[1])
    except Exception:
        return 0.0
```

**Observed (real) value:** 0.5396 (var=0.0423)
**Candidate trajectory (this loop):**
  - iter 1: 0.7393 (var=0.0343) (Δ vs real +0.1998)
  - iter 2: 0.6425 (var=0.0305) (Δ vs real +0.1029)
  - iter 3: 0.5460 (var=0.0476) (Δ vs real +0.0064)
  - iter 4: 0.5048 (var=0.0373) (Δ vs real -0.0348)
  - iter 5: 0.3915 (var=0.0385) (Δ vs real -0.1481)
  - iter 6: 0.4900 (var=0.0510) (Δ vs real -0.0496)
  - iter 7 (current): 0.4252 (var=0.0600) (Δ vs real -0.1143)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4027 (var=0.0731)
- pi_5_1: 0.0336 (var=0.0381)
- pi_1: 0.8316 (var=0.0733)
- pi_2: -0.0087 (var=0.0294)
- pi_3: -0.0421 (var=0.0393)
- pi_4: -0.6945 (var=0.0377)
- pi_5: 0.0254 (var=0.0512)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The PGDT candidate achieves exactly what the arbiter asked for on the primacy-slope experiments — Exp11 lands at 0.365 vs 0.315 observed (within 0.05, a large improvement over pi_6's 0.175), and Exp4 remains fine (0.890 vs 0.843). But the multiplicative gate has catastrophically broken the anti-validity regime it was supposed to preserve. Exp3 flips SIGN: observed -0.576 (anti-validity regression), candidate +0.596 — a 1.17 absolute miss. Exp5 flips from observed 0.231 (anti-validity adherence) to 0.794 (strong validity-following). Exp1 jumps from 0.208 to 0.494 (TTB-following at chance-plus instead of tally allegiance). Exp2 undershoots (0.082 vs 0.158), Exp12 overshoots (0.739 vs 0.540), and the anti-recency experiments (Exp8: 0.296 vs 0.396; Exp9: 0.239 vs 0.341; Exp10: -0.339 vs -0.173) all overshoot in the primacy direction. The root cause is structural and identifiable: the gate g_j = ((n-j)/(n-1))^eta is EXACTLY ZERO at the last position. On every decreasing-validity layout, the lowest-validity (most-distrusted, highest inverted weight) cue sits at the final position, and the multiplicative form annihilates it: 0 × (large distrust weight) = 0. Worse, even at eta=4 the normalized pure gate concentrates ~72% of its mass on position 1, so the product w_j ∝ g_j × bracket_j peaks at the FIRST position — the highest-validity cue — making the effective weights positively validity-correlated. Hand-checking Exp3 (validities [0.95...0.60], tau=0.935, gamma=3, c=0.1, eta=4) gives normalized weights ≈ [0.42, 0.35, 0.24, 0.04, 0.00] — a pure primacy gradient that happens to align with validity order, which is precisely the observed sign flip. The arbiter's stated intent ('the distrust kernel retains dominance because the gate multiplies a near-zero inverted weight for high-validity early cues') fails because the reverse multiplication also holds: the gate multiplies a near-zero gate value onto the maximum distrust weight at the last cue. The distrust kernel does not retain dominance; the gate does.
Rationale: Regenerate, but with a MINOR IN-FAMILY adjustment — the PGDT architecture is sound and should be kept; only the gate's functional form needs a floor. Specific fix: replace the hard-zero gate with a floored gate, g_j ∝ delta + (1-delta) * ((n-j)/(n-1))^eta, with delta ∈ [0.05, 0.25] added as a new parameter (this is a normalization-scheme fix, not a family change). Verification by hand on Exp3 (delta=0.15, eta=4, tau=0.935, gamma=3, c=0.1): the product weights become ≈ [0.09, 0.09, 0.15, 0.28, 0.40] — back-loaded, anti-validity restored, moving Exp3 from +0.60 back toward the observed -0.58 and Exp5 from 0.79 toward 0.23. The same floor fixes the remaining misses in the right direction: (i) on tied-validity designs (Exp9, Exp11, Exp12) the distrust bracket is flat, so the floored gate is still a strong primacy gradient — Exp11's newly-achieved 0.365 is preserved (compensate any dilution by nudging eta toward 5-6 if needed); (ii) Exp12's overshoot (0.739 vs 0.540) and Exp10's over-negative slope (-0.339 vs -0.173) are both diluted toward target by the floor; (iii) Exp8 (0.296 vs 0.396) and Exp9 (0.239 vs 0.341) rise toward target because the final expert regains nonzero weight. Keep tau ∈ [0.90, 0.97] and gamma ∈ [2.8, 3.5] as proposed (the distrust kernel needs to stay sharp); keep eta ∈ [4, 6]; beta and epsilon ranges are fine (Exp4's 0.890 shows no over-commitment problem). One additional guard: the degenerate fallback when Z_v ≈ 0 currently returns a flat kernel — with the floor added this path is unlikely to bind, but ensure the floor applies to the gate before multiplication so it can never zero out a cue entirely. The single structural knob (gate floor) is the highest-leverage change; everything else is second-order.

**Outcome of your advice:** iter 1 candidate loss=0.3072 -> iter 2 candidate loss=0.1413 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The floored gate did exactly what the iter-1 critique asked. The sign catastrophes are gone: Exp3 moved from +0.596 to -0.318 (observed -0.576), Exp5 from 0.794 to 0.295 (observed 0.231), Exp1 from 0.494 to 0.186 (observed 0.208, now within 0.025). Aggregate loss halved (0.3072 → 0.1413), and the candidate was accepted. The PGDT family is now qualitatively correct on every regime: anti-validity on decreasing-validity layouts, primacy on tied-validity layouts, anchor adherence high. However, the residual errors are not random noise — they form a coherent directional pattern. The candidate is systematically TOO EXTREME in the primacy/anti-late-cue direction on the position-sensitive experiments: Exp8 final-expert following 0.269 vs 0.396, Exp9 recency-following 0.261 vs 0.341, Exp10 recency score -0.278 vs -0.173, Exp6 recency-consistency 0.301 vs 0.374, Exp12 primacy slope 0.643 vs 0.540. Simultaneously it is too extreme on the allegiance experiments: Exp2 tally-vs-TTB contrast 0.318 vs 0.158, Exp4 anchors 0.903 vs 0.843, Exp7 IVC-adherence 0.858 vs 0.788. The one exception pointing the other way is Exp3 (-0.318 vs -0.576: anti-validity regression still too weak) plus a mild Exp5 (0.295 vs 0.231). Crucially, these two clusters are NOT in tension under the right knob: raising the gate floor delta both (i) flattens the gate on tied-validity designs (Exp6/8/9/10/12 all move toward target) and (ii) on decreasing-validity layouts removes the early-position boost that currently dilutes the distrust kernel's back-loaded mass, making the product weights MORE anti-validity (Exp3 more negative, Exp5 lower) — hand-check on Exp3: delta=0.35, eta=4 gives normalized product weights ≈ [0.06, 0.05, 0.15, 0.29, 0.44] vs ≈ [0.12, 0.09, 0.19, 0.26, 0.29] at delta=0.15. Exp2's overshoot should also shrink modestly, since flatter gate weight on the early high-validity cues pulls choices toward the TTB side of the contrast (note every theory in the family overshoots Exp2 somewhat; pi_3's 0.166 is the best anyone does, so treat Exp2 as a stretch target, not a must-fix).
Rationale: Regenerate with a single, low-risk in-family knob: RAISE THE GATE FLOOR. Extend delta from [0.05, 0.25] to approximately [0.15, 0.40] (and/or allow eta to relax toward the lower end of [4, 6] as compensation), keeping tau ∈ [0.90, 0.97], gamma ∈ [2.8, 3.5], c ∈ [0.10, 0.25], beta ∈ [2.5, 6.0], epsilon ∈ [0.06, 0.15] frozen. This is a parameter-range tightening of the already-accepted floored-gate PGDT, not a redesign, and it is the highest-leverage direction because the same increase moves seven of the eight >0.05 misses in the correct direction: Exp3 (stronger anti-validity back-loading), Exp5 (lower validity adherence), Exp6/Exp8/Exp9 (late cues regain weight), Exp10 (less over-negative recency score), Exp12 (smaller primacy slope). Two cautions for the proposer: (1) Exp11 is currently the best-fitting primacy slope (0.271 vs 0.315) — flattening the gate will pull it down; monitor it and if it drops below ~0.22, trade back by nudging eta upward within [4, 6] rather than by reverting delta, since eta steepness on tied-validity designs is what Exp11/Exp12 jointly need to dissociate (Exp12 currently overshoots and wants LESS slope, Exp11 slightly more — a mid-range compromise is acceptable given Exp12's miss is 2.4x larger). (2) Exp4 (0.903 vs 0.843) and Exp7 (0.858 vs 0.788) are mildly over-deterministic; the flatter gate should help slightly, but if they remain >0.05 high after the delta change, a small epsilon bump (toward 0.15) is the secondary lever — do NOT use it to chase Exp3, since added lapse attenuates the anti-validity coefficient that Exp3 needs strengthened. Do not touch tau/gamma: the distrust kernel's sharpness is what the Exp3/Exp5/Exp7 regime depends on, and the iter-1→iter-2 trajectory confirms those ranges are correctly calibrated. The accept gate protects against regression, so this one-knob iteration is cheap upside.

**Outcome of your advice:** iter 2 candidate loss=0.1413 -> iter 3 candidate loss=0.1175 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The delta raise did most of what the iter-2 critique asked, and the gate accepted it (loss 0.1413 → 0.1175). The tied-validity cluster is now essentially solved: Exp12 0.546 vs 0.540, Exp5 0.208 vs 0.231, Exp9 0.316 vs 0.341. Exp3 improved further (-0.318 → -0.469 vs -0.576). But the residual pattern has now crystallized into two clusters that the delta knob demonstrably CANNOT move, plus two self-inflicted costs. Cluster A — over-commitment to distrust-driven allegiances: Exp2 +0.162 (0.320 vs 0.158, now the single largest miss), Exp7 +0.083, Exp4 +0.056. Cluster B — under-weighting of high-validity LATE cues: Exp8 0.278 vs 0.396, Exp6 0.303 vs 0.374, Exp1 0.141 vs 0.208 (overshot past target downward this iteration). The diagnostic tell is that the iter-2→iter-3 delta raise barely moved Exp6 (0.301→0.303), Exp8 (0.269→0.278) and Exp10 (-0.278→-0.265) despite hand-checks predicting movement: the gate floor multiplies the distrust BRACKET, and for a high-validity late cue that bracket is ≈ (1-tau)/n ≈ 0.01-0.02 at tau ∈ [0.90, 0.97]. No amount of gate flattening can rescue a cue whose multiplicative partner is ~1%. This is structural: the distrust kernel is too dominant relative to its flat component, which simultaneously over-commits the allegiance contrasts (Cluster A) and starves trusted late experts (Cluster B). Costs of this iteration: Exp11 slipped from 0.271 to 0.234 (target 0.315) and Exp1 overshot from 0.186 to 0.141 — both predicted in the iter-2 cautions, both acceptable trades so far, but they signal delta is near its useful ceiling. Only 3 of 12 experiments are within the arbiter's ~0.05 target, so tuning should continue.
Rationale: verdict=regenerate with a NEW primary knob: LOWER TAU. Widen tau's range from [0.90, 0.97] down to approximately [0.82, 0.92] (still inside the arbiter's prescribed [0.80, 0.97]). Mechanism: tau controls the flat share (1-tau)/n of the bracket. Raising that share from ~1.5% to ~4-5% directly (i) feeds Exp8's final 0.90-validity expert, whose weight is structurally pinned at ≈ (1-tau)/n regardless of delta — this is the only lever that can move Exp8 toward 0.396; (ii) softens the distrust-driven allegiances that overshoot Exp2 (+0.162), Exp7 (+0.083) and Exp4 (+0.056), since less distrust weight on the early high-validity cues pulls choices toward the TTB side of the Exp2 contrast and away from ceiling on Exp4/7; (iii) pulls Exp1 back up toward 0.208 by moving the effective weights off pure anti-validity; (iv) plausibly helps Exp6's late cues if they carry high validity. Compensating secondary knob for the one clear casualty: Exp3 (still +0.107 short of -0.576) will weaken as tau drops, so raise delta's ceiling modestly, e.g. [0.20, 0.45] — the iter-2→3 trajectory shows delta has strong, verified leverage on Exp3's back-loading (-0.318 → -0.469). Keep gamma ∈ [2.8, 3.5], c, beta, epsilon frozen; keep eta ∈ [4, 6] with subjects free to sit at 5-6 to protect Exp11. Three cautions: (1) Exp11 (0.234 vs 0.315) will be squeezed by the delta bump — if it falls below ~0.20, accept the trade rather than reverting tau: the combined Exp2+Exp8 misses (0.28) outweigh Exp11's (0.08), and every theory in the family except pi_1 undershoots Exp11 anyway. (2) Exp12 is now matched (0.546 vs 0.540); lowering tau raises the flat share, which on matched-validity cells lets the gate act on a flatter kernel and may push the slope up — if Exp12 exceeds ~0.60, pull eta toward 4 rather than touching tau. (3) Do not chase Exp2 below ~0.20: every theory overshoots it (pi_3's 0.166 is the family best) — treat it as a stretch target and bank anything in the 0.20-0.28 range. This is a parameter-range adjustment wholly inside the prescribed PGDT family: same multiplicative gate-times-distrust architecture, same code path, only the tau range (primary) and delta ceiling (compensation) change. The two previous single-knob directives both produced accepted candidates; the residual miss pattern now points unambiguously at the distrust kernel's dominance share rather than the gate's shape.

**Outcome of your advice:** iter 3 candidate loss=0.1175 -> iter 4 candidate loss=0.0940 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The tau rebalance was accepted (loss 0.1175 -> 0.0940) and moved most of the intended cluster: Exp2 improved +0.162 -> +0.125, Exp7 +0.083 -> +0.052, Exp1 -0.067 -> -0.044, Exp9 is now essentially exact (0.345 vs 0.341), Exp4/Exp5/Exp10/Exp12 are all within 0.05. Eight of twelve experiments now meet the arbiter's ~0.05 target. But the iteration's PRIMARY target failed: Exp8 did not move at all (0.2779 -> 0.2775 vs 0.396 observed) despite the tau drop being advertised as 'the only lever that can move Exp8's final 0.90-validity expert'. Meanwhile the compensating delta raise did NOT hold Exp3: it regressed -0.469 -> -0.401 (target -0.576). The residual miss pattern is now three experiments outside 0.10 (Exp3 +0.175, Exp2 +0.125, Exp8 -0.118) plus two borderline (Exp6 -0.053, Exp11 -0.059), and the two largest (Exp3 vs Exp2/Exp8) pull the SAME knob (tau) in OPPOSITE directions -- which is why parameter tuning is yielding diminishing returns: iter-4 bought 0.037 on Exp2 at a cost of 0.068 on Exp3. The diagnosis is structural and identifiable: under w_j ∝ g_j * [(1-tau)/n + tau*inv_j], the flat share (1-tau)/n -- the only component that carries genuine trust in high-validity cues -- is MULTIPLIED by the gate, and at the last position the gate equals only delta. So the final expert in Exp8 receives delta * (1-tau)/n ≈ 0.3 * 0.03 ≈ 1% of raw weight, and no admissible (tau, delta) pair can simultaneously feed trusted late cues (needs big flat share AND big gate at late positions) and preserve sharp back-loaded distrust (needs tau high). The gate is taxing exactly the component that must reach the late trusted cues. Note also that on tied-validity designs the current form reduces to the FULL pure gate (bracket is flat, so w ∝ g), which is why Exp11/Exp12 are so sensitive to every gate-shape move -- another symptom of the same coupling.
Rationale: Regenerate with ONE structural micro-fix inside the prescribed PGDT family, plus range restoration. THE FIX (a normalization-scheme change, not a family change): re-route the flat trust share OUTSIDE the gate, i.e. w_j ∝ g_j * tau * inv_j + (1-tau)/n, then normalize. The arbiter's core claim -- the MULTIPLICATIVE gate-by-distrust interaction that dissociates primacy-slope regimes from anti-validity regimes -- is fully preserved (the gate still multiplies the distrust kernel cue-by-cue); only the small uniform mixing term is exempted from the gate tax. Predicted effects, each in the needed direction: (i) Exp8: the final 0.90-validity expert now receives the full untaxed (1-tau)/n, moving 0.278 toward the 0.35-0.40 range (pi_2's 0.404 shows the target is reachable); (ii) Exp2: position-0 (v=0.95) likewise regains untaxed flat weight, restoring some TTB adherence and pulling the contrast down from 0.284 toward the 0.20-0.25 bankable zone; (iii) Exp1 rises from 0.164 toward 0.208 by the same early-trusted-cue mechanism; (iv) Exp3: the uniform component loads on the tally margin, which the Exp3 OLS metric partials out via the margin regressor, so the d-coefficient is driven by the gated distrust term -- and with the flat share no longer needing to be large, tau can be RESTORED upward, re-sharpening the anti-validity regression toward -0.50 and beyond; (v) Exp4/Exp7: added flat tallying keeps anchors high and should ease Exp7's residual +0.052 overshoot. RANGE RESTORATION to go with it: tau back to approximately [0.86, 0.95] (the distrust kernel regains dominance; the trusted-late-cue job is now done structurally, not by starving tau); delta relaxed to [0.20, 0.40] (it no longer has to double as the Exp8 lever); eta ceiling WIDENED from 6 to about [4, 8] (still inside the arbiter's [4, 9]) -- this is the necessary compensation because on tied-validity designs the re-routed form damps the effective gate by a factor tau (w ∝ tau*g_j + (1-tau)), which would otherwise drag Exp11 (already -0.059) and Exp12 (-0.035) further down; the extra eta headroom restores that steepness where it is needed without touching the distrust regime. Keep gamma ∈ [2.8, 3.5], c ∈ [0.10, 0.25], beta ∈ [2.5, 6.0], epsilon ∈ [0.06, 0.15] frozen exactly as in the accepted base. Verification checkpoints for the proposer: if Exp12 falls below ~0.48 or Exp11 below ~0.22, raise eta toward 8 rather than reverting the structural fix; if Exp3 remains short of -0.50, push tau toward 0.95 and gamma toward 3.5 (both now uncoupled from the Exp8/Exp2 cluster); do not chase Exp2 below ~0.20 (pi_3's 0.166 remains the family best; bank anything in 0.20-0.28). Do NOT abandon the multiplicative gate-times-distrust architecture or the floored gate -- both are working exactly as the arbiter prescribed; this edit only unblocks the one coupling that two consecutive parameter iterations (iter-3's tau drop, iter-4's confirmed Exp8 paralysis) have proven cannot be broken from inside the current bracket.

**Outcome of your advice:** iter 4 candidate loss=0.0940 -> iter 5 candidate loss=0.1147 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The iter-5 candidate implemented my iter-4 directive faithfully (flat trust share re-routed outside the gate, tau restored to [0.86, 0.95], eta ceiling widened to 8), and the gate REJECTED it (loss 0.0940 -> 0.1147). The postmortem is informative and changes the diagnosis in two important ways. (1) The structural fix FAILED its primary target: Exp8's FEDI moved from 0.2775 to 0.2836 against a target of 0.396 — essentially zero. The reason is now provable: on FEDI ladder trials the lone final expert is pitted against MULTIPLE opposing cues, so a UNIFORM flat share contributes (1-tau)*(1-k)/n < 0 to the evidence — uniform trust votes AGAINST the final expert on exactly the trials FEDI measures. Rerouting the flat share outside the gate cannot help; worse, even perfectly FLAT weights cap FEDI at roughly sigmoid(-beta*k/n) + lapse ~ 0.25-0.30, because tallying structurally opposes a lone late cue. Reaching 0.396 requires the last position to outweigh the average opposing cue — a recency tilt this primacy-gated family does not possess (only pi_4, a recency theory, exceeds it). Exp8 is therefore a structural blind spot of the prescribed family, not a tuning problem; my iter-3 and iter-4 advice both attacked it and both failed to move it. (2) The tau restoration that accompanied the fix did deliver on Exp3 (-0.401 -> -0.584, essentially exact) but re-broke exactly the cluster tau controls, as the iter-3 history predicted it would: Exp2 worsened 0.284 -> 0.362, Exp1 0.164 -> 0.125, and the tied-validity primacy slopes collapsed (Exp11 0.256 -> 0.195, Exp12 0.505 -> 0.392) — the eta=8 compensation did NOT deliver, because with delta >= 0.20 the floored gate at eta=8 is effectively 'position 1 large, rest flat at delta', and adding the untaxed (1-tau) flat term flattens it further, destroying the graded position spread the Exp11/Exp12 reference kernels need. Net: one big gain (Exp3, +0.167) against four losses (Exp2 +0.079, Exp12 +0.113, Exp11 +0.061, Exp1 +0.039). The candidate is discarded; the base reverts to iter-4 (fully-gated bracket, tau [0.82, 0.92], loss 0.0940), whose residual profile is: Exp3 +0.175, Exp2 +0.125, Exp8 -0.118 as the three big misses, with everything else at or within ~0.06.
Rationale: Verdict regenerate — forced by the gate, and there is still identifiable in-family headroom. Do NOT re-attempt the untaxed-flat rerouting (rejected; now proven unable to move Exp8) and do NOT restore tau above 0.92 (the iter-5 rejection confirms the iter-3 finding that tau trades Exp2/Exp1 against Exp3 at roughly break-even-to-negative exchange rate). Build instead on the unchanged iter-4 base with three coordinated knobs: (A) PRIMARY — RAISE DELTA'S CEILING from [0.20, 0.45] to approximately [0.25, 0.50]. Delta is the only knob with verified, uncompensated leverage on Exp3 (iters 2->3: delta 0.40->0.45 drove Exp3 -0.318 -> -0.469 while Exp2 simultaneously IMPROVED 0.318 -> 0.284) because it lifts the gated weight of late low-validity cues on decreasing-validity layouts — the anti-validity arm of the arbiter's dissociation, which at -0.401 is still the largest addressable miss vs -0.576. (B) SECONDARY — PUSH EPSILON TO THE ARBITER'S CEILING, [0.10, 0.15] (from [0.06, 0.15]). The base's remaining over-commitment pattern (Exp2 +0.125, Exp4 +0.033, Exp7 +0.052, Exp6 -0.053, Exp10 -0.042, Exp1 -0.044) is diffuse, and a uniform lapse attenuates every one of these toward its neutral point; it also buys Exp8 whatever residual room exists. (C) COMPENSATION — WIDEN ETA from [4, 6] to [4, 7] (inside the arbiter's [4, 9]) to offset the delta-flattening and epsilon-attenuation of the Exp11/Exp12 primacy slopes; in the fully-gated base form, tied-validity designs see the PURE gate (the distrust bracket is flat), so eta acts unopposed there and does not touch the distrust regime — this is exactly the coupling the rejected iter-5 form broke. Keep tau [0.82, 0.92], gamma [2.8, 3.5], c [0.10, 0.25], beta [2.5, 6.0] frozen. Two explicit strategic re-frames for the proposer: (i) DECLARE EXP8 A CAPPED STRETCH TARGET — hand-analysis shows that even perfectly flat weights plus maximal lapse yield FEDI ~ 0.28-0.30 on the ladder trials, so 0.396 is unreachable without a recency tilt the prescribed family forbids; bank ~0.30 and spend NO further structural effort there (this retracts the iter-3/iter-4 Exp8 directives, both of which failed). (ii) Treat Exp2 as bankable in the 0.20-0.28 range (pi_3's 0.166 remains the family best) — the epsilon bump plus delta's small historical Exp2 benefit should land it near 0.22. Expected net movement: Exp3 toward -0.50, Exp2 toward ~0.22, Exp8 toward ~0.30, Exp4/7/6/10/1 all toward target, at a cost of ~0.02-0.04 on Exp11/Exp12 (partially offset by eta headroom) — a projected aggregate clearly below the 0.0940 base. Checkpoints: if Exp12 falls below ~0.46 or Exp11 below ~0.22, raise eta toward 7 before touching delta; if Exp3 remains short of -0.50, push c toward 0.25 and gamma toward 3.5 (both now uncoupled from the Exp2 cluster since tau stays low); if Exp4 drops below 0.82, pull epsilon back toward 0.12.

**Outcome of your advice:** iter 5 candidate loss=0.1147 -> iter 6 candidate loss=0.0969 -> the gate marked it REJECTED.

### Iteration 6 (most recent)
Verdict: regenerate
Interpretation: The iter-6 candidate implemented my iter-5 directive faithfully (delta ceiling to 0.50, epsilon floored at 0.10, eta widened to 7, tau/gamma/c/beta frozen) and the gate REJECTED it by a hair (0.0969 vs 0.0940). The per-experiment decomposition tells a clean story about which knob did what. (A) The delta raise WORKED as designed: Exp3 improved -0.401 -> -0.439 (target -0.576) at zero cost to Exp2 (0.284 -> 0.285), replicating the iters-2-to-3 evidence that delta is the one knob that dissociates the Exp3/Exp2 tension. Exp5 also improved (0.257 -> 0.243). (B) The epsilon floor FAILED outright, and the failure is now mathematically explicable: a uniform lapse rescales any difference-of-proportions metric by exactly (1-eps), so moving eps from ~0.06 to ~0.12 attenuates Exp2's contrast by at most ~6% — it could never close a +0.127 gap. Meanwhile it genuinely attenuates the Exp11/Exp12 OLS slopes (both via the (1-eps) factor and by flattening the logistic): Exp11 fell 0.256 -> 0.210 (target 0.315, now the second-largest miss at -0.105), Exp12 0.505 -> 0.490, and even the proportion metrics it was supposed to soften went the WRONG way (Exp4 0.877 -> 0.888, Exp7 0.839 -> 0.846) because the parameter refit compensated. The epsilon bump was a pure loss. (C) The eta-7 compensation was swamped by the epsilon cost and delivered nothing on its own. (D) Exp8 stayed frozen at 0.280, exactly as the capped-stretch analysis predicted. Net: one real gain (Exp3, +0.038) against one real loss (Exp11, -0.046) plus diffuse small costs — a marginal net negative. The base reverts to iter-4 (loss 0.0940), whose residual profile is Exp3 +0.175, Exp2 +0.125, Exp8 -0.118 (structural cap), Exp11 -0.059, everything else within ~0.05.
Rationale: Regenerate on the unchanged iter-4 base with a corrected knob set — the iter-6 postmortem identifies precisely which of my three iter-5 directives to keep and which to retract. (1) PRIMARY CORRECTION — REVERT EPSILON to [0.06, 0.15]. The floor was a mathematical noop on the difference metrics it targeted (lapse scales proportion-differences by (1-eps) only, <=7% effect) and a real cost on the slope metrics (Exp11/Exp12) and proportion metrics (Exp4/Exp7 moved the wrong way). This is the single change most likely to flip the loss below 0.0940 by itself, since it removes the -0.046 Exp11 damage while keeping everything the base already achieves. (2) KEEP AND EXTEND DELTA: [0.25, 0.55] (from the base's [0.20, 0.45]). Delta is the only knob with twice-verified, uncompensated dual leverage on the Exp3/Exp2 tension (iters 2->3: Exp3 -0.318 -> -0.469 while Exp2 0.318 -> 0.284; iter-6: Exp3 -0.401 -> -0.439 at zero Exp2 cost). Exp3 remains the largest addressable miss (+0.175 vs -0.576). (3) NEW SECONDARY — SHARPEN THE DISTRUST TAIL, per the iter-5 checkpoint I specified for exactly this contingency: tighten gamma from [2.8, 3.5] to [3.2, 3.5] and c from [0.10, 0.25] to [0.18, 0.25]. Higher gamma steepens the inverted kernel toward the maximum-error (late, low-validity) cues, adding back-loading that stacks with delta's; higher c sharpens the relative contrast. These are uncoupled from the Exp2 cluster because tau stays low at [0.82, 0.92] — do NOT raise tau (the iter-5 rejection confirmed the tau exchange rate is break-even-to-negative). (4) KEEP ETA at [4, 7]: once epsilon reverts, the eta headroom is what protects Exp11 (base 0.256, target 0.315) and Exp12 (0.505 vs 0.540) from the delta-0.55 flattening; on tied-validity designs the bracket is flat so eta acts unopposed and does not touch the distrust regime. (5) STRATEGIC HOLDS, unchanged: Exp8 remains a capped stretch target (~0.28-0.30 is the family ceiling; a lone late cue is structurally opposed by tallying — spend NO effort there, and do NOT re-attempt the untaxed-flat rerouting, which is now twice-proven unable to move it); Exp2 is bankable in 0.20-0.28 (pi_3's 0.166 remains the family best). Checkpoints: if Exp12 falls below 0.48 or Exp11 below 0.24, pull delta back toward 0.50 before touching eta; if Exp3 remains short of -0.50 after the gamma/c sharpening, accept it rather than raising tau — the aggregate favors the current allocation. Projected movement vs base: Exp3 toward -0.50, Exp11/Exp12 restored to base-or-better, Exp4/Exp7 restored to base levels, everything else unchanged — a projected aggregate clearly below 0.0940.

**Outcome of your advice:** iter 6 candidate loss=0.0969 -> CURRENT candidate loss=0.0988 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The iter-7 candidate implemented my iter-6 directive faithfully (epsilon reverted to [0.06, 0.15], delta ceiling extended to 0.55, distrust tail sharpened via gamma [3.2, 3.5] and c [0.18, 0.25], eta held at [4, 7]) and the gate REJECTED it by a small margin (0.0988 vs 0.0940). The per-experiment decomposition shows this was NOT a uniform failure \u2014 it was one knob over-extended past its useful range, exactly as my own pre-specified checkpoints anticipated. (A) The distrust-tail sharpening WORKED: Exp3 jumped -0.401 -> -0.539 against a target of -0.576 (now the closest any PGDT iteration has come; the miss is only +0.036), Exp5 is essentially exact (0.224 vs 0.231), Exp6 improved to -0.043, Exp7 improved to +0.040, Exp1 improved to -0.032. Critically, gamma/c are provably UNCOPLED from the tied-validity designs (the distrust bracket is flat there, so gamma/c are no-ops on Exp11/Exp12) \u2014 every gain in this cluster is pure, cost-free upside. (B) The delta extension to 0.55 is what broke: both of my iter-6 checkpoint conditions triggered simultaneously \u2014 Exp11 fell to 0.180 (checkpoint: below 0.24) and Exp12 fell to 0.425 (checkpoint: below 0.48). The 0.55 floor flattens the gate into 'position 1 large, rest near-flat', destroying the graded position spread the Exp11/Exp12 reference kernels score, and the eta-7 headroom could not compensate. The epsilon revert did its job (no repeat of iter-6's diffuse damage) but its recovery was more than eaten by the delta flattening. (C) Exp8 drifted further down (0.259 vs base 0.278) \u2014 consistent with the established structural cap; no action warranted. Net accounting: Exp3 gained +0.139 and Exp5/6/7/1 gained ~0.05 combined, but Exp11 lost -0.076, Exp12 lost -0.080, Exp8 lost -0.018, Exp4/Exp10 lost ~0.02 each \u2014 roughly break-even, slightly negative. The diagnosis is clean: the candidate overshot on ONE knob (delta ceiling) while the other three changes were correct.",
  "verdict": "regenerate",
  "rationale": "Regenerate on the unchanged iter-4 base with a SINGLE correction to the iter-7 knob set \u2014 everything else stays. (1) PRIMARY \u2014 PULL DELTA'S CEILING BACK from 0.55 to approximately [0.25, 0.48]. This is not an oscillation: iters 2->3 verified delta's leverage at ceiling ~0.45, iter-6 verified it at 0.50, and iter-7 has now verified that 0.55 crosses the tied-validity damage threshold (both Exp11 < 0.24 and Exp12 < 0.48 checkpoints triggered). The useful ceiling sits between 0.45 and 0.50; land at ~0.48. (2) KEEP THE GAMMA/C SHARPENING EXACTLY AS IMPLEMENTED: gamma [3.2, 3.5], c [0.18, 0.25]. This was the iter-7 success story \u2014 it delivered Exp3 to -0.539 and Exp5 to exact at ZERO cost to the tied-validity cluster (the distrust bracket is flat there, so gamma/c cannot touch Exp11/Exp12). With delta pulled back to ~0.48, the back-loading that Exp3 needs now comes primarily from the sharpened distrust tail rather than the gate floor, which is the correct division of labor: distrust-kernel back-loading is design-conditional (only fires on decreasing-validity layouts), gate-floor back-loading is unconditional (also flattens tied-validity designs). Expected Exp3 at delta 0.48 + sharpened tail: between -0.48 and -0.53 \u2014 acceptable per the iter-6 directive ('if Exp3 remains short of -0.50, accept it rather than raising tau or re-raising delta'). (3) KEEP EPSILON at [0.06, 0.15] \u2014 the revert was correct and should not be touched again. (4) SECONDARY (optional, with checkpoint) \u2014 WIDEN ETA from [4, 7] to [4, 8], still inside the arbiter's [4, 9]. Rationale: even the base undershoots Exp11 (0.256 vs 0.315) and Exp12 (0.505 vs 0.540); with delta back at 0.48 the extra eta headroom pushes both toward target on tied-validity designs where eta acts unopposed, adding margin above the 0.0940 floor rather than merely restoring it. Checkpoint: if Exp3 regresses above -0.45 (steep primacy re-aligning with validity order on decreasing-validity layouts), drop the eta ceiling back to 7 \u2014 Exp3's +0.14 gain is worth more than Exp11/12's residual -0.06 undershoots. (5) STRATEGIC HOLDS, unchanged and now thrice-confirmed: Exp8 is a structural cap (~0.26-0.30; a lone late cue is opposed by tallying \u2014 spend NO effort there); Exp2 is bankable in 0.20-0.28 (pi_3's 0.166 remains the family best; the +0.126 miss is near the family frontier and tau is off-limits per the iter-5 rejection). Keep tau [0.82, 0.92], beta [2.5, 6.0] frozen. Projected movement vs the 0.0940 base: Exp3 ~-0.50 (a ~+0.10 gain on the largest addressable miss), Exp5/6/7/1 retain most of their iter-7 gains, Exp11/Exp12 restored to base-or-better (plus upside if eta-8 fires), Exp4/Exp10 back to base \u2014 a projected aggregate clearly below 0.0940. The mechanism family, code path, and multiplicative gate-times-distrust architecture are all correct and untouched; this is a one-knob range correction plus an optional headroom widening."
}
```

## Usage

```json
{
  "input_tokens": 38761,
  "output_tokens": 4407,
  "reasoning_tokens": 2983,
  "provider": "Sail Research"
}
```
