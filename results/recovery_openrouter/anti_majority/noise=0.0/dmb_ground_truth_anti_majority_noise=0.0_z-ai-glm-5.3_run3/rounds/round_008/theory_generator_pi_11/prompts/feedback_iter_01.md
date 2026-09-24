# feedback_iter_01

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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 2 (= `pi_10`).

Replace pi_10 with a new theory, 'Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI)'. Per-trial core is unchanged from the validated shared kernel: endorsements are evidence AGAINST the carrying option, E = sum_j v_j^gamma (b_j - a_j) with RAW (unnormalized) validity-power weights, p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, A_eff = (1-eps)(2m-1) concentrated near 0.69, ties exactly 50%. The population departs from both predecessors in three calibrated ways. (1) GAMMA SUPPORT: a single continuous uniform distribution on [0, 2.2] — no empty gaps and no narrow tail segment (both directly refuted by Exps 2 and 16: the gap must be filled at constant density, and any cliff at 2.10 or 2.0 is falsified), with the slight extension past 2.0 providing the modest true tail mass that Exp 18's flip rate (0.4335 < RSD's 0.5335) and Exp 17's intermediate Bayes factor require, while keeping Exp 2's crossing statistic near its observed -0.21. (2) BETA: log-uniform on [1.8, 6.0], independent of gamma — no segment-coupled beta boxes (the direct cause of pi_10's Exp 15 catastrophe). (3) THE NEW MECHANISM: per-trial stochastic validity-attention jitter — with probability delta ~ 0.10-0.13 per trial, the effective gamma for that trial is redrawn uniformly from [gamma_habit - 0.35, gamma_habit + 0.35] clipped to [0, 2.4]. This is the key competitor to pi_7: it explains why fixed-gamma measurement models (Exp 16's tail estimator, Exp 18's flip rungs, Exp 17's BF) see apparent mass above 2.0 and softened per-subject likelihoods in real data, while population-level density probes (Exp 2) still recover an essentially uniform [0, 2] shape, and why Exps 13/14's Bayes factors are smaller in real data than pi_7's noise-free simulation predicts. Falsifiable signatures: near-uniform recovered gamma density with no plateau on rung ladders; apparent tail-mass estimates that INFLATE on shallow-slope designs (Exp 16 ~ 0.25-0.35) while crossing-point statistics stay near -0.2 (Exp 2); flip-trial rates pulled 0.05-0.10 below RSD's predictions on gamma* ~ 0.7-1.0 ladders (Exp 18 ~ 0.43-0.47); per-subject log BFs against fixed-gamma competitors attenuated ~30-50% relative to RSD's predictions (Exps 13, 17); steep-slope posterior mass still high (Exp 12 ~ 0.85-0.92) since jitter is modest. Keep delta small enough to preserve the excellent Exps 3-12 fits that both theories currently achieve through the shared core.


## CANDIDATE THEORY
Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI, calibration pass 2). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity-power weights: E = sum_j v_j^gamma (b_j - a_j), identical vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with A_eff = (1-eps)(2m-1) near 0.69 (steep normalized slope x moderate asymptote). The population: gamma_habit ~ U[0, 2.2] (single continuous uniform — no gaps, no cliffs, support held fixed at 2.2), beta log-uniform on [1.8, 6.0] independent of gamma. The novel mechanism — per-trial stochastic validity-attention jitter — is now calibrated to deliver its own falsifiable signatures: with per-trial probability delta ~ 0.18-0.24, the subject's effective gamma is redrawn uniformly from a WIDER, slightly down-tilted window [gamma_habit - 0.55, gamma_habit + 0.45] (clipped to [0, 2.5]), and on the same attention event the log inverse temperature is jittered by U[-0.2, +0.2]. The wider, down-tilted gamma window manufactures the APPARENT tail mass that fixed-gamma measurement models recover (Exp 14 target ~0.28-0.32) while the true support stays at [0, 2.2] so the crossing-point statistic (Exp 18, real -0.208) and the population density probes stay near their observed values; the correlated log-beta jitter softens per-subject likelihoods (attenuating Exps 11/12/15 BFs toward their real values) WITHOUT touching the asymptote anchors (Exps 7/8), since eps and m are unchanged. Falsifiable signatures: apparent tail-mass estimates ~0.25-0.35 on shallow-slope designs; per-subject log BFs against fixed-gamma competitors attenuated 30-50%; steep-slope posterior mass still >= 0.85; flip rates in [0.43, 0.47]; no plateau in recovered gamma rung ladders.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI),
    # calibration pass 2: stronger, slightly down-tilted gamma jitter plus
    # a correlated per-trial log-beta jitter on the same attention event.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"JU-AEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- (1) GAMMA SUPPORT: single continuous uniform on [0, 2.2] ----
    # UNCHANGED from the accepted base: no gaps, no cliffs, support held
    # exactly at [0, 2.2] (per critic: do NOT extend past 2.2; the Exp 17
    # sign flip and Exp 18 upward drift were driven by true mass above
    # 2.0 combining with up-side jitter).
    gamma_habit = 2.2 * float(parameters["gamma_u"])

    # ---- (2) PRECISION: log-uniform raw inverse temperature, ----
    # ---- independent of gamma (no segment-coupled beta boxes)    ----
    log_beta_habit = float(parameters["log_beta"])
    beta = float(np.exp(log_beta_habit))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # ---- (3) PER-TRIAL STOCHASTIC VALIDITY-ATTENTION JITTER ----
    # STRENGTHENED per critic: delta raised to ~0.18-0.24, the redraw
    # window widened to a slightly DOWN-TILTED [gamma - 0.55, gamma + 0.45]
    # clipped to [0, 2.5], and on the SAME attention event the log inverse
    # temperature is jittered by U[-0.2, +0.2] (correlated beta softening).
    # The down-tilt manufactures apparent tail mass for fixed-gamma
    # measurement models while pulling the crossing statistic back toward
    # its observed -0.21; the beta jitter softens per-subject likelihoods
    # (attenuating the Exps 11/12/15 Bayes factors) without touching the
    # asymptote anchors (eps, m unchanged).
    delta = float(parameters["delta"])
    log_beta_t = log_beta_habit
    if np.random.random() < delta:
        lo = max(0.0, gamma_habit - 0.55)
        hi = min(2.5, gamma_habit + 0.45)
        gamma_t = lo + (hi - lo) * np.random.random()
        # Correlated precision jitter on the same attention event:
        # log_beta shifts by U[-0.2, +0.2].
        log_beta_t = log_beta_habit + 0.4 * (np.random.random() - 0.5)
    else:
        gamma_t = gamma_habit
    beta = float(np.exp(log_beta_t))

    # RAW validity weighting: w_j = v_j^gamma_t, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma_t)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster. UNCHANGED (Exps 7/8 anchors are spot-on).
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)

`parameters`:
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- delta: [0.18, 0.24]
- validities: validities

`rationale`:
This is a minimal-diff calibration pass on the accepted JU-AEI base, applying exactly the critic's four in-family adjustments. (a) STRENGTHEN THE JITTER: delta raised from [0.10, 0.13] to [0.18, 0.24] and the redraw window widened from +/-0.35 to a slightly down-tilted [gamma - 0.55, gamma + 0.45], clipped to [0, 2.5]. This is the primary lever for the two decisive under-deliveries: Exp 14's apparent tail mass (0.186 vs real 0.298, var ~0.0003 — the most decisive diagnostic in the battery) and the per-subject likelihood softness demanded by Exps 11/12/15. The down-tilt (wider below the habitual gamma than above) manufactures apparent tail mass for fixed-gamma measurement models while simultaneously pulling the Exp 18 crossing statistic back down toward its observed -0.208 (the base's -0.149 was drifting high), addressing the critic's pre-registered asymmetry contingency without a redesign. (b) CORRELATED LOG-BETA JITTER: on the same attention event, log_beta shifts by U[-0.2, +0.2]. This softens per-subject likelihoods and attenuates the Exps 11/12/15 Bayes factors toward their real values (89.9 -> ~52.6; 30.8 -> ~12.9) WITHOUT lowering the asymptote anchors — epsilon and m are untouched, so Exps 7/8 (0.68/0.34, currently spot-on) are preserved, exactly as the critic prescribed. (c) GAMMA SUPPORT HELD AT [0, 2.2]: the Exp 17 sign flip (-0.07 vs real +0.84) and Exp 18 upward drift were diagnosed as true-mass-above-2.0 effects; the support is unchanged so the apparent tail is manufactured purely by the jitter, leaving the population density probes and crossing statistics intact. (d) GUARDRAILS: because delta stays <= 0.24 and the beta jitter is +/-0.2 in log space (a subject at beta=3 can dip to ~2.5, still leaving normalized slopes beta*sum(v^gamma) far above the 9.5 threshold), Exp 10 should stay >= 0.85; the down-tilted window keeps Exp 16's flip rate inside [0.43, 0.47] (down-side jitter pulls it toward the real 0.4335 from the base's 0.4397 rather than pushing it up); and epsilon/m are unchanged so Exp 2's slight low bias is not compounded. All other code — the shared anti-endorsement raw-weight kernel, graded polarity, lapse, tie handling, policy — is re-emitted verbatim from the accepted base.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0855 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0712 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.0712.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dissociation score: on trials where Take-The-Best (top-validity
    # cascade) and Tallying (cue-count majority) make DIFFERENT modal
    # predictions, how often does the subject side with the TTB option?
    # TTB predicts > 0.5 on every such trial (constant sigmoid(beta)
    # confidence, including on tally-tie trials 7/8 where Tallying must
    # guess at exactly 50%). Tallying predicts < 0.5 on conflict trials
    # (majority option wins) and exactly 0.5 on tally ties. So the metric
    # is positive under TTB and negative under Tallying.
    V = np.array([0.95, 0.85, 0.65, 0.60, 0.50])
    order = np.argsort(-V, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    matches = []
    for i in range(len(a_col)):
        a = np.asarray(a_col[i], dtype=float)
        b = np.asarray(b_col[i], dtype=float)

        # TTB winner: first discriminating cue in validity order.
        ttb = -1
        for j in order:
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb == -1:
            continue  # TTB guesses; not diagnostic

        # Tallying winner: strict cue-count majority.
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins > b_wins:
            tal = 0
        elif b_wins > a_wins:
            tal = 1
        else:
            tal = -1  # tally tie -> Tallying guesses

        # Keep only dissociation trials: the two theories' modal
        # predictions differ (tally tie, or tally majority against TTB).
        if tal == ttb:
            continue

        matches.append(1.0 if resp[i] == ttb else 0.0)

    if len(matches) == 0:
        return 0.0
    # Center at 0.5: positive => TTB-like, negative => Tallying-like.
    return float(np.mean(matches)) - 0.5
```

**Observed (real) value:** 0.1580 (var=0.0120)
**Candidate trajectory (this loop):**
  - iter 1: 0.1660 (var=0.0096) (Δ vs real +0.0080)
  - iter 2 (current): 0.1417 (var=0.0052) (Δ vs real -0.0163)
**Other theories' values on this metric (for reference):**
- pi_1: 0.3480 (var=0.0087)
- pi_2: -0.2963 (var=0.0068)
- pi_3: 0.2473 (var=0.0026)
- pi_4: 0.1530 (var=0.0069)
- pi_5: 0.2150 (var=0.0067)
- pi_6: 0.1693 (var=0.0082)
- pi_7: 0.1660 (var=0.0056)
- pi_8: 0.1537 (var=0.0082)
- pi_9: 0.1513 (var=0.0063)
- pi_10: 0.1513 (var=0.0077)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(choose the TALLY winner | Tallying and Take-The-Best disagree).

    On every trial we compute (a) the tally winner (option winning more
    strict feature-wise comparisons) and (b) the TTB winner (winner on
    the first discriminating cue in descending-validity order, cues here
    are already ordered by validity). We keep only CONFLICT trials where
    both heuristics have a definite winner and they point to opposite
    options, and return the proportion of choices falling on the tally
    winner's side.

    Under Tallying this proportion is high (sigmoid(2*beta) on the
    margin-2 grid conflicts, sigmoid(6*beta) on the extreme-conflict
    trials where one option wins 7 of 8 cues); under TTB it is low
    (1 - sigmoid(beta)). Works on the pooled DataFrame and on any
    single subject's slice (each subject has 7 unique conflict trials
    x 4 reps = 28 conflict observations).
    """
    validities = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60])
    cue_order = np.argsort(-validities, kind="stable")

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))

        # Tallying needs a strict majority; skip tied tallies.
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Take-The-Best winner: first discriminating cue in validity order.
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        # Keep only genuine conflicts between the two heuristics.
        if ttb_winner == tally_winner:
            continue

        resp = int(row["response"])
        hits.append(1.0 if resp == tally_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.1814 (var=0.0118)
**Candidate trajectory (this loop):**
  - iter 1: 0.1536 (var=0.0101) (Δ vs real -0.0279)
  - iter 2 (current): 0.1814 (var=0.0089) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8664 (var=0.0110)
- pi_1: 0.1536 (var=0.0134)
- pi_3: 0.2557 (var=0.0053)
- pi_4: 0.2579 (var=0.0137)
- pi_5: 0.1479 (var=0.0061)
- pi_6: 0.1793 (var=0.0080)
- pi_7: 0.1764 (var=0.0097)
- pi_8: 0.1571 (var=0.0041)
- pi_9: 0.1714 (var=0.0079)
- pi_10: 0.1714 (var=0.0079)

### Experiment 3
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Depth/conflict invariance contrast for TTB vs PWEI.

    Three structurally identified trial classes (validity order is fixed:
    [0.90, 0.89, 0.88, 0.87, 0.70, 0.60]):

    HIGH  - the top cue (0.90) discriminates and every other discriminating
            cue agrees with it (design trials 1, 2, 9, 10; 24 obs/subject).
            PWEI evidence margin |D| is maximal (~1 or w_top).
    DEPTH5- cues 1-4 all tie; only the 0.70-validity cue discriminates
            (design trials 11, 12; 12 obs/subject). PWEI's weight for
            0.70^gamma is ~1e-5 of total, so PWEI predicts ~50% choices.
    REV   - the top cue discriminates one way, the three sub-top cues
            (0.89, 0.88, 0.87) discriminate the other way, tail cues tie
            (design trials 3, 4; 12 obs/subject). In PWEI's steep-gamma
            regime the sub-top block outweighs the top cue, so PWEI
            chooses AGAINST the TTB winner.

    For each class compute p = P(choose the TTB winner). TTB's structural
    invariant is a single constant confidence p* on every discriminating
    trial, so 2*p_HIGH - p_DEPTH5 - p_REV = 0 exactly, at ANY (beta, eps).
    PWEI predicts p_HIGH ~ 0.9, p_DEPTH5 ~ 0.5, p_REV ~ 0.1-0.37, giving a
    value ~0.9-1.2 across its entire parameter box.
    """
    VALS = np.array([0.90, 0.89, 0.88, 0.87, 0.70, 0.60])
    order = np.argsort(-VALS, kind="stable")  # validity-descending cue indices

    n_high, k_high = 0, 0
    n_d5, k_d5 = 0, 0
    n_rev, k_rev = 0, 0

    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        s = np.sign(a - b)  # +1 favors A, -1 favors B, 0 tie, per cue

        # First discriminating cue in validity order (TTB's search).
        first = None
        for j in order:
            if s[j] != 0.0:
                first = j
                break
        if first is None:
            continue  # identical options: both theories say 50%

        ttb_winner_is_a = s[first] > 0
        chose_ttb = (int(resp) == 0) == ttb_winner_is_a

        if first == 4:
            # DEPTH5: only the 0.70-validity cue discriminates.
            n_d5 += 1
            k_d5 += int(chose_ttb)
        elif first == 0 and s[1] == -s[0] and s[2] == -s[0] and s[3] == -s[0] \
                and s[4] == 0.0 and s[5] == 0.0:
            # REV: top cue vs the three sub-top cues, tail ties.
            n_rev += 1
            k_rev += int(chose_ttb)
        elif first == 0:
            # HIGH candidate: all other discriminating cues agree with top.
            others = [s[j] for j in range(1, 6) if s[j] != 0.0]
            if all(o == s[0] for o in others):
                n_high += 1
                k_high += int(chose_ttb)
        # All other trial types (mixed conflicts, cue-2-led trials, ties)
        # are deliberately excluded to keep the classes pure.

    p_high = k_high / n_high if n_high > 0 else 0.5
    p_d5 = k_d5 / n_d5 if n_d5 > 0 else 0.5
    p_rev = k_rev / n_rev if n_rev > 0 else 0.5

    return 2.0 * p_high - p_d5 - p_rev
```

**Observed (real) value:** -0.7167 (var=0.0939)
**Candidate trajectory (this loop):**
  - iter 1: -0.6400 (var=0.0629) (Δ vs real +0.0767)
  - iter 2 (current): -0.7467 (var=0.0744) (Δ vs real -0.0300)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0033 (var=0.0461)
- pi_3: 1.0483 (var=0.0759)
- pi_2: 0.7083 (var=0.0512)
- pi_4: -0.5050 (var=0.1068)
- pi_5: -0.7033 (var=0.1239)
- pi_6: -0.6950 (var=0.0763)
- pi_7: -0.7067 (var=0.0731)
- pi_8: -0.7200 (var=0.0585)
- pi_9: -0.7233 (var=0.0493)
- pi_10: -0.7750 (var=0.0581)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # OLS slope of "chose the TTB (first-discriminating-cue) winner" on the
    # PWEI power-weighted evidence margin measured TOWARD that winner.
    # TTB: choice prob is one constant p* on every discriminating trial
    #      (any beta, eps)  =>  slope exactly 0.
    # PWEI: prob is a psychometric function of the weighted margin D
    #      =>  strongly positive slope (~0.5 over the design's margin range).
    val = np.array([0.95, 0.92, 0.91, 0.90, 0.89, 0.88], dtype=float)
    gamma = 40.0  # mid of PWEI's calibrated regime [35, 50]
    w = np.power(val, gamma)
    w = w / w.sum()
    cue_order = np.argsort(-val, kind="stable")

    xs = []
    ys = []
    for a_cell, b_cell, r in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        s = np.sign(a - b)
        if not np.any(s != 0):
            # Identical options: pure guessing for both theories; excluded.
            continue
        # TTB winner: first discriminating cue in descending-validity order.
        winner = None
        for j in cue_order:
            if s[j] != 0:
                winner = 0 if s[j] > 0 else 1
                break
        if winner is None:
            continue
        # Power-weighted evidence for option A.
        D = float(np.sum(w * s))
        # Margin re-signed so that positive = evidence favors the TTB winner.
        x = D if winner == 0 else -D
        y = 1.0 if int(r) == winner else 0.0
        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return 0.0
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    if sxx <= 1e-12:
        return 0.0
    sxy = float(np.sum((xs - xs.mean()) * (ys - ys.mean())))
    return float(sxy / sxx)
```

**Observed (real) value:** -0.5230 (var=0.0540)
**Candidate trajectory (this loop):**
  - iter 1: -0.5538 (var=0.0264) (Δ vs real -0.0308)
  - iter 2 (current): -0.5133 (var=0.0355) (Δ vs real +0.0098)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5548 (var=0.0148)
- pi_1: 0.0081 (var=0.0061)
- pi_2: 0.5897 (var=0.0315)
- pi_4: -0.4252 (var=0.0349)
- pi_5: -0.5389 (var=0.0255)
- pi_6: -0.5237 (var=0.0265)
- pi_7: -0.5575 (var=0.0274)
- pi_8: -0.5635 (var=0.0198)
- pi_9: -0.5423 (var=0.0159)
- pi_10: -0.5479 (var=0.0211)

### Experiment 5
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Lower-cue evidence-usage slope.

    For every trial with a discriminating cue, determine the Take-The-Best
    winner (first cue in descending-validity order that differs). Pool trials
    into cells keyed by the inverted-polarity integrated evidence favoring
    that winner, E_w = sum_j w_j * (loser_j - winner_j), w_j proportional to
    validity. Then compute the OLS slope of P(choose the TTB winner) on E_w
    across cells.

    TTB: the winner's choice probability is the SAME constant p* on every
    discriminating trial (only the top discriminating cue is used), so the
    psychometric function over E_w is exactly flat -> slope = 0 for ANY
    (beta, epsilon).

    SPI: choice probability is a strictly increasing sigmoid of E_w
    (rho < 0.5 preserves monotonicity, lapse is affine), and the design's
    cells sweep E_w from -1 (unanimous endorsement) to +0.58 -> large
    positive slope.
    """
    validities = np.array([0.95, 0.85, 0.75, 0.70, 0.65, 0.60], dtype=float)
    w = validities / validities.sum()
    cue_order = np.argsort(-validities, kind="stable")

    a_cells = data["option_a_ratings"].tolist()
    b_cells = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    n = len(data)
    E_list = []
    Y_list = []

    for i in range(n):
        a = np.asarray(a_cells[i], dtype=float).ravel()
        b = np.asarray(b_cells[i], dtype=float).ravel()
        if a.shape[0] != validities.shape[0] or b.shape[0] != validities.shape[0]:
            continue

        # TTB cascade: first discriminating cue in validity order.
        winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner < 0:
            continue  # no discriminating cue: uninformative for the contrast

        # Inverted-polarity integrated evidence favoring the TTB winner:
        # the loser's endorsements (1s) count FOR the winner (sparse option).
        if winner == 0:
            E = float(np.sum(w * (b - a)))
        else:
            E = float(np.sum(w * (a - b)))

        E_list.append(E)
        Y_list.append(1.0 if int(resp[i]) == winner else 0.0)

    if len(E_list) < 4:
        return 0.0

    E = np.asarray(E_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)

    # Pool trials into cells by (rounded) evidence value; mirrored twins and
    # repetitions of the same pair share the same E_w and are averaged.
    keys = np.round(E, 4)
    uniq = np.unique(keys)
    if uniq.shape[0] < 2:
        return 0.0

    cell_x = np.empty(uniq.shape[0], dtype=float)
    cell_y = np.empty(uniq.shape[0], dtype=float)
    for idx in range(uniq.shape[0]):
        mask = keys == uniq[idx]
        cell_x[idx] = E[mask].mean()
        cell_y[idx] = Y[mask].mean()

    x = cell_x - cell_x.mean()
    y = cell_y - cell_y.mean()
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0

    return float(np.sum(x * y) / denom)
```

**Observed (real) value:** 0.5944 (var=0.0376)
**Candidate trajectory (this loop):**
  - iter 1: 0.5896 (var=0.0187) (Δ vs real -0.0048)
  - iter 2 (current): 0.5949 (var=0.0251) (Δ vs real +0.0005)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0009 (var=0.0089)
- pi_4: 0.4379 (var=0.0326)
- pi_2: -0.6876 (var=0.0238)
- pi_3: 0.0089 (var=0.0056)
- pi_5: 0.5894 (var=0.0647)
- pi_6: 0.5414 (var=0.0227)
- pi_7: 0.5915 (var=0.0172)
- pi_8: 0.5704 (var=0.0179)
- pi_9: 0.5813 (var=0.0162)
- pi_10: 0.5701 (var=0.0179)

### Experiment 6
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Fixed design constants: 8 cues with the experiment's validities.
    validities = np.array([0.95, 0.92, 0.62, 0.60, 0.62, 0.60, 0.62, 0.60])

    # Parse per-trial rating vectors (cells are lists -> project safely).
    a_lists = [list(r) for r in data["option_a_ratings"]]
    b_lists = [list(r) for r in data["option_b_ratings"]]
    A = np.asarray(a_lists, dtype=float)
    B = np.asarray(b_lists, dtype=float)
    resp = np.asarray(data["response"]).astype(int)

    n_feat = A.shape[1]
    if n_feat == len(validities):
        cue_order = np.argsort(-validities, kind="stable")
    else:
        cue_order = np.arange(n_feat)

    n_a = A.sum(axis=1)
    n_b = B.sum(axis=1)

    # TTB winner per trial: first discriminating cue in validity order.
    winner = np.full(A.shape[0], -1, dtype=int)
    for j in cue_order:
        open_rows = winner == -1
        if not open_rows.any():
            break
        wa = open_rows & (A[:, j] > B[:, j])
        wb = open_rows & (B[:, j] > A[:, j])
        winner[wa] = 0
        winner[wb] = 1

    abs_diff = np.abs(n_a - n_b)
    denser = np.where(n_a > n_b, 0, 1)  # option carrying MORE 1-ratings

    # Extreme-count trials only (|n_A - n_B| >= 5): the sign of SPI's
    # evidence is robust across its whole parameter box on these cells.
    valid = (winner >= 0) & (abs_diff >= 5)
    ttb_picks_denser = (winner == denser)

    chose_winner = (resp == winner)

    grp_d = valid & ttb_picks_denser      # TTB winner is the DENSER option
    grp_s = valid & (~ttb_picks_denser)    # TTB winner is the SPARSER option

    nd = int(grp_d.sum())
    ns = int(grp_s.sum())
    if nd == 0 or ns == 0:
        return 0.0

    p_denser = float(chose_winner[grp_d].mean())
    p_sparser = float(chose_winner[grp_s].mean())

    return p_denser - p_sparser
```

**Observed (real) value:** -0.6758 (var=0.0393)
**Candidate trajectory (this loop):**
  - iter 1: -0.6946 (var=0.0421) (Δ vs real -0.0188)
  - iter 2 (current): -0.6837 (var=0.0359) (Δ vs real -0.0079)
**Other theories' values on this metric (for reference):**
- pi_4: -0.5012 (var=0.0594)
- pi_1: 0.0138 (var=0.0135)
- pi_2: 0.7354 (var=0.0335)
- pi_3: -0.0125 (var=0.0078)
- pi_5: -0.6912 (var=0.1205)
- pi_6: -0.6900 (var=0.0330)
- pi_7: -0.6850 (var=0.0317)
- pi_8: -0.6883 (var=0.0128)
- pi_9: -0.6725 (var=0.0255)
- pi_10: -0.6917 (var=0.0411)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Per-subject choice CONSISTENCY (extremity) toward the sparse option on
    # near-unanimous trials (one option all-zeros, the other endorsed on at
    # least n_features-1 cues), averaged over subjects.
    #
    # RFI-delta: polarity is a stable per-subject trait with only a small
    # attentional slip (delta <= 0.12) and small lapse (eps <= 0.08), so EVERY
    # subject -- majority (anti-endorsement) and minority (conventional)
    # alike -- is strongly and stably consistent on these trials:
    # per-subject p(sparse) is either ~[0.80, 0.92] or its mirror ~[0.08, 0.20],
    # i.e. |2p-1| ~ [0.6, 0.85] for essentially all subjects, with tight
    # between-subject spread.
    #
    # SPI: the wide lapse/polarity-mixing boxes (eps in [0.05, 0.35],
    # rho in [0, 0.35], beta in [0.5, 20]) mean per-subject attenuation varies
    # enormously; the average subject sits much closer to chance (mean
    # |2p-1| far below RFI's), and the population spread is wide.
    # The absolute value makes the statistic polarity-symmetric, so RFI's
    # small conventional minority does NOT inflate the between-subject
    # variance (both clusters have the same extremity).

    a_ratings = data['option_a_ratings'].tolist()
    b_ratings = data['option_b_ratings'].tolist()
    resp = data['response'].astype(float).values
    subj = data['subject_id'].values

    n = len(data)
    a_sum = np.empty(n, dtype=float)
    b_sum = np.empty(n, dtype=float)
    n_feat = np.empty(n, dtype=float)
    for i in range(n):
        av = list(a_ratings[i])
        bv = list(b_ratings[i])
        a_sum[i] = float(np.sum(av))
        b_sum[i] = float(np.sum(bv))
        n_feat[i] = float(len(av))

    # Near-unanimous trials: sparse side has zero endorsements, dense side
    # is endorsed on all or all-but-one cue (E in ~[0.8, 1.0] for any
    # weighting gamma in [0, 2]).
    thresh = n_feat - 1.0
    sparse_is_a = (a_sum == 0.0) & (b_sum >= thresh)
    sparse_is_b = (b_sum == 0.0) & (a_sum >= thresh)
    extreme = sparse_is_a | sparse_is_b

    # 1 if the subject chose the all-zeros (sparse) option, else 0.
    chose_sparse = np.where(sparse_is_a, 1.0 - resp, resp)

    vals = []
    for s in np.unique(subj[extreme]):
        m = extreme & (subj == s)
        if m.sum() == 0:
            continue
        p = float(np.mean(chose_sparse[m]))
        vals.append(abs(2.0 * p - 1.0))

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.7044 (var=0.0382)
**Candidate trajectory (this loop):**
  - iter 1: 0.6800 (var=0.0402) (Δ vs real -0.0244)
  - iter 2 (current): 0.7044 (var=0.0331) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7611 (var=0.0118)
- pi_4: 0.5167 (var=0.0365)
- pi_1: 0.6944 (var=0.0505)
- pi_2: 0.7267 (var=0.0418)
- pi_3: 0.6067 (var=0.0143)
- pi_6: 0.7544 (var=0.0264)
- pi_7: 0.7111 (var=0.0259)
- pi_8: 0.7178 (var=0.0239)
- pi_9: 0.6922 (var=0.0227)
- pi_10: 0.7067 (var=0.0235)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Polarity-agnostic preference EXTREMITY on high-contrast trials.
    #
    # Select trials where the two options' endorsement counts differ by >= 3
    # (design trials R3..R8 and mirrors: |E| in [~0.35, 1.0] for every gamma
    # in [0,2]; the count-sparser option is the inverted-evidence-favored
    # option on all of them). Per subject, compute p_hat = rate of choosing
    # the count-sparser option on those 48 trials, then take |p_hat - 0.5|.
    # The metric is invariant to a subject's polarity (a conventional-polarity
    # subject and a red-flag subject both yield large extremity), which
    # neutralizes RFI's ~5% minority cluster, and directly measures the
    # ATTENUATION away from a pure deterministic preference: SPI's large
    # lapse (eps in [0.05,0.35], mean ~0.20) plus polarity mixture
    # (rho in [0,0.35], mean ~0.175) pulls per-subject p toward 0.5
    # (predicted extremity ~0.22), whereas RFI-delta's small slip
    # (delta ~0.10) and tiny lapse (eps <= 0.08) keep every subject
    # near saturation (predicted extremity ~0.35, per-subject range
    # [0.75, 0.92] in p). Applied to the pooled DataFrame the function
    # returns the mean of the per-subject values, so the pooled point
    # estimate is exactly the population mean of the per-subject metric.
    sids = []
    chose_sparse = []
    for r in data.itertuples(index=False):
        a = np.asarray(r.option_a_ratings, dtype=float)
        b = np.asarray(r.option_b_ratings, dtype=float)
        sa = int(a.sum())
        sb = int(b.sum())
        # Keep only high-contrast trials (count difference >= 3).
        if abs(sa - sb) < 3:
            continue
        resp = int(r.response)  # 0 = chose A, 1 = chose B
        sparse_is_a = sa < sb
        picked_sparse = (resp == 0) if sparse_is_a else (resp == 1)
        sids.append(r.subject_id)
        chose_sparse.append(1.0 if picked_sparse else 0.0)
    if len(sids) == 0:
        return 0.0
    tmp = pd.DataFrame({"sid": sids, "cs": chose_sparse})
    per_subj = tmp.groupby("sid")["cs"].mean()
    extremity = (per_subj - 0.5).abs()
    return float(extremity.mean())
```

**Observed (real) value:** 0.3467 (var=0.0108)
**Candidate trajectory (this loop):**
  - iter 1: 0.3433 (var=0.0068) (Δ vs real -0.0033)
  - iter 2 (current): 0.3571 (var=0.0063) (Δ vs real +0.0104)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2367 (var=0.0084)
- pi_5: 0.3758 (var=0.0028)
- pi_1: 0.3283 (var=0.0115)
- pi_2: 0.3421 (var=0.0094)
- pi_3: 0.1929 (var=0.0036)
- pi_6: 0.3467 (var=0.0078)
- pi_7: 0.3521 (var=0.0065)
- pi_8: 0.3438 (var=0.0047)
- pi_9: 0.3658 (var=0.0051)
- pi_10: 0.3629 (var=0.0063)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.98, 0.95, 0.88, 0.85, 0.82, 0.80,
                  0.78, 0.76, 0.74, 0.72, 0.62, 0.62], dtype=float)
    WTIE = np.power(V, 0.75)

    df = data.reset_index(drop=True)

    # ---------- 1. classify trials into evidence cells ----------
    # Cell = the dense-minus-sparse endorsement pattern; its evidence under
    # any gamma is x(gamma) = sum_j v_j^gamma * dvec_j  (>= 0 on this design).
    subj_cells = {}
    for i in range(len(df)):
        a = np.asarray(df.at[i, "option_a_ratings"], dtype=float)
        b = np.asarray(df.at[i, "option_b_ratings"], dtype=float)
        if a.shape != b.shape or a.shape[0] != V.shape[0]:
            continue
        if np.array_equal(a, b):
            continue  # identical vectors: uninformative
        ca = float(np.sum(a))
        cb = float(np.sum(b))
        if ca < cb:
            sparse, dvec = 0, b - a
        elif cb < ca:
            sparse, dvec = 1, a - b
        else:
            wa = float(np.dot(WTIE, a))
            wb = float(np.dot(WTIE, b))
            if wa < wb:
                sparse, dvec = 0, b - a
            else:
                sparse, dvec = 1, a - b
        y = 1.0 if int(df.at[i, "response"]) == sparse else 0.0
        key = tuple(int(round(v)) for v in dvec)
        sid = df.at[i, "subject_id"]
        cells = subj_cells.setdefault(sid, {})
        ent = cells.setdefault(key, [0, 0.0])
        ent[0] += 1
        ent[1] += y

    if not subj_cells:
        return 0.0

    # ---------- 2. per-subject polarity orientation ----------
    # RFI-delta has a ~5.5% conventional-polarity minority; orienting each
    # subject by their overall sparse-choice rate maps that minority onto
    # the majority cluster (removing the huge mixture variance a signed
    # statistic would otherwise suffer). Under GSP every subject's overall
    # sparse rate stays well above 0.5 (min ~0.65), so orientation never
    # misfires there.
    for cells in subj_cells.values():
        tot_n = sum(e[0] for e in cells.values())
        tot_k = sum(e[1] for e in cells.values())
        if tot_n > 0 and tot_k < 0.5 * tot_n:
            for e in cells.values():
                e[1] = e[0] - e[1]

    # ---------- 3. reference mixtures ----------
    # Both theories reduce to the SAME per-subject family
    #     p_i = q*sigmoid(s*x_i(gamma)) + (1-q)*(1 - sigmoid(s*x_i(gamma)))
    # with theory-specific priors:
    #   RFI: s = 2*beta/sum(v^gamma), beta~U[3,8]  -> s in [0.5, 1.69];
    #        q = (1-eps)(1-delta)+eps/2 in [0.85, 0.92] (TIGHT); gamma in [0,1]
    #   GSP: s = beta_raw in [1.1, 3.2] (log-uniform);
    #        q = (1-eps)*m+eps/2 in [0.72, 0.98] (WIDE); gamma in [0, 1.5]
    def sigmoid(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    keys = sorted({k for cells in subj_cells.values() for k in cells})
    ncell = len(keys)
    if ncell == 0:
        return 0.0
    key_mat = np.array([list(k) for k in keys], dtype=float)  # (C, 12)

    GAMMAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    XG = np.array([key_mat @ np.power(V, g) for g in GAMMAS])  # (7, C)

    # RFI mixture components
    BETAS = np.array([3.0, 4.25, 5.5, 6.75, 8.0])
    Q_RFI = np.array([0.86, 0.885, 0.91])
    QW_RFI = np.array([0.25, 0.5, 0.25])
    rg, rs, rq, rw = [], [], [], []
    for gi in range(5):  # gamma in [0, 1]
        Sg = float(np.sum(np.power(V, GAMMAS[gi])))
        for be in BETAS:
            s = 2.0 * be / Sg
            for qi in range(3):
                rg.append(gi)
                rs.append(s)
                rq.append(Q_RFI[qi])
                rw.append(QW_RFI[qi] / 25.0)
    rg = np.array(rg); rs = np.array(rs); rq = np.array(rq); rw = np.array(rw)

    # GSP mixture components
    SG = np.exp(np.linspace(np.log(1.1), np.log(3.2), 7))
    Q_GSP = np.array([0.735, 0.79, 0.845, 0.90, 0.955])
    gg, gs, gq, gw = [], [], [], []
    w_unit = 1.0 / (7 * 7 * 5)
    for gi in range(7):  # gamma in [0, 1.5]
        for s in SG:
            for q in Q_GSP:
                gg.append(gi); gs.append(s); gq.append(q); gw.append(w_unit)
    gg = np.array(gg); gs = np.array(gs); gq = np.array(gq); gw = np.array(gw)

    def comp_logps(gidx, slopes, qs):
        z = slopes[:, None] * XG[gidx, :]
        sig = sigmoid(z)
        p = qs[:, None] * sig + (1.0 - qs[:, None]) * (1.0 - sig)
        p = np.clip(p, 0.01, 0.99)
        return np.log(p), np.log(1.0 - p)

    LP_r, L1_r = comp_logps(rg, rs, rq)
    LP_g, L1_g = comp_logps(gg, gs, gq)
    LW_r = np.log(rw)
    LW_g = np.log(gw)

    def logmix(LP, L1, LW, k_vec, n_vec):
        ll = (k_vec[None, :] * LP + (n_vec - k_vec)[None, :] * L1).sum(axis=1) + LW
        m = np.max(ll)
        return float(m + np.log(np.sum(np.exp(ll - m))))

    # ---------- 4. per-subject marginal log-likelihood ratio ----------
    vals = []
    for sid in sorted(subj_cells, key=str):
        cells = subj_cells[sid]
        k_vec = np.zeros(ncell)
        n_vec = np.zeros(ncell)
        for ci, key in enumerate(keys):
            ent = cells.get(key)
            if ent is not None:
                n_vec[ci] = float(ent[0])
                k_vec[ci] = float(ent[1])
        if n_vec.sum() <= 0:
            continue
        vals.append(logmix(LP_r, L1_r, LW_r, k_vec, n_vec)
                    - logmix(LP_g, L1_g, LW_g, k_vec, n_vec))

    if not vals:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** -1.7633 (var=3.8102)
**Candidate trajectory (this loop):**
  - iter 1: -1.2241 (var=2.7895) (Δ vs real +0.5392)
  - iter 2 (current): -1.1244 (var=2.3586) (Δ vs real +0.6388)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6498 (var=1.2048)
- pi_6: -0.6480 (var=2.0074)
- pi_1: -2.7859 (var=6.5552)
- pi_2: -1.5565 (var=3.7090)
- pi_3: -1.7169 (var=0.8671)
- pi_4: -1.4521 (var=2.5200)
- pi_7: -1.3031 (var=2.4039)
- pi_8: -0.6950 (var=1.2398)
- pi_9: -0.7021 (var=1.9618)
- pi_10: -1.1568 (var=1.5167)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # ------------------------------------------------------------------
    # Theory-neutral psychometric slope test.
    #
    # Both theories are EXACTLY re-parameterizable, on every trial, as
    #   p(choose A) = 0.5 + A * ( sigmoid(s * E_norm(gamma)) - 0.5 ),
    #     E_norm(gamma) = sum_j v_j^gamma (b_j - a_j) / sum_j v_j^gamma,
    # where
    #   GSP  (raw weights):    s = beta_raw * sum(v^gamma) in [9.1, 38.4],
    #                          A = (1-eps)(2m-1) in [0.44, 0.98];
    #   RFI  (normalized wts): s = beta in [3, 8],
    #                          A = +-(1-eps)(1-2 delta), |A| in [0.70, 0.84].
    # The two slope supports are DISJOINT: every GSP subject's effective
    # normalized evidence slope lies above ~9, every RFI subject's below 8.
    # The attenuation (m, eps, delta) and the polarity sign are absorbed
    # by the free coefficient A in [-1, 1], which removes exactly the
    # level/attenuation heterogeneity that destroyed the level-based
    # statistics tried before.
    #
    # Per subject we fit (gamma, s, A) by binomial maximum likelihood on
    # ALL 96 trials (grid search; exact likelihood, no approximation),
    # and report the posterior probability that s > 9.5 under a
    # log-uniform prior on s in [1.2, 48] (profiled over gamma and A).
    #   GSP subjects -> posterior mass almost entirely above 9.5
    #                  (expected value ~0.75-0.85);
    #   RFI subjects -> posterior mass below 9.5
    #                  (expected value ~0.15-0.45).
    # The pooled metric is the mean of the per-subject posterior
    # probabilities; a single subject's slice returns that subject's
    # posterior probability.
    # ------------------------------------------------------------------

    VALS = np.array([0.99, 0.95, 0.88, 0.85, 0.82, 0.80, 0.78,
                     0.76, 0.74, 0.72, 0.50, 0.50], dtype=float)

    GAMMA_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    S_GRID = np.logspace(np.log10(1.2), np.log10(48.0), 40)
    A_GRID = np.linspace(-0.98, 0.98, 41)
    S_THRESH = 9.5

    n_g, n_s, n_a = len(GAMMA_GRID), len(S_GRID), len(A_GRID)

    # per-gamma cue weights and their totals
    W = np.power(VALS[None, :], GAMMA_GRID[:, None])      # (n_g, F)
    WSUM = W.sum(axis=1)                                  # (n_g,)
    ABOVE = S_GRID > S_THRESH                            # (n_s,)

    def _vec(x):
        if isinstance(x, np.ndarray):
            return x.astype(float)
        return np.asarray(list(x), dtype=float)

    def _subject_value(sub):
        a_list = sub["option_a_ratings"].tolist()
        b_list = sub["option_b_ratings"].tolist()
        r = sub["response"].to_numpy()

        # aggregate identical trials: key -> [n_chose_A, n_total]
        agg = {}
        for i in range(len(sub)):
            a = _vec(a_list[i])
            b = _vec(b_list[i])
            key = (tuple(int(round(v)) for v in a),
                   tuple(int(round(v)) for v in b))
            y = 1 if int(r[i]) == 0 else 0   # response 0 = chose A
            if key in agg:
                agg[key][0] += y
                agg[key][1] += 1
            else:
                agg[key] = [y, 1]

        nll = np.zeros((n_g, n_s, n_a), dtype=float)
        for (ta, tb), (k, n) in agg.items():
            da = np.asarray(ta, dtype=float)
            db = np.asarray(tb, dtype=float)
            d = db - da                     # endorsement difference (B - A)
            if not np.any(d != 0.0):
                continue                    # tie trials: no information
            e = (W @ d) / WSUM              # (n_g,) normalized evidence for A
            for gi in range(n_g):
                x = S_GRID * e[gi]           # (n_s,)
                ax = np.abs(x)
                ex = np.exp(-ax)
                sg = np.where(x >= 0.0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
                dev = sg - 0.5                               # (n_s,)
                p = 0.5 + A_GRID[None, :] * dev[:, None]     # (n_s, n_a)
                p = np.clip(p, 1e-9, 1.0 - 1e-9)
                nll[gi] -= k * np.log(p) + (n - k) * np.log1p(-p)

        # profile the negative log-likelihood over gamma and A
        prof = nll.min(axis=(0, 2))         # (n_s,)
        w = np.exp(-(prof - prof.min()))   # likelihood weights (log-uniform s prior)
        return float(w[ABOVE].sum() / w.sum())

    subj = data["subject_id"].to_numpy()
    vals = []
    for s in np.unique(subj):
        sub = data[subj == s]
        vals.append(_subject_value(sub))
    if len(vals) == 0:
        return 0.5
    return float(np.mean(vals))
```

**Observed (real) value:** 0.9113 (var=0.0227)
**Candidate trajectory (this loop):**
  - iter 1: 0.8852 (var=0.0329) (Δ vs real -0.0262)
  - iter 2 (current): 0.9092 (var=0.0229) (Δ vs real -0.0021)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7849 (var=0.0519)
- pi_5: 0.5780 (var=0.0892)
- pi_1: 0.9526 (var=0.0112)
- pi_2: 0.9697 (var=0.0075)
- pi_3: 0.5167 (var=0.0397)
- pi_4: 0.6639 (var=0.0486)
- pi_7: 0.9161 (var=0.0235)
- pi_8: 0.8983 (var=0.0297)
- pi_9: 0.8413 (var=0.0356)
- pi_10: 0.8791 (var=0.0271)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd
from collections import defaultdict


def metric(data: pd.DataFrame) -> float:
    """Total log marginal-likelihood ratio (log Bayes factor) of RSD-vgp over GSP.

    Both theories share the identical choice core
        p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
        E(gamma) = sum_j v_j**gamma * (b_j - a_j),
    and differ ONLY in the population supports of (gamma, log beta):
        RSD-vgp: gamma ~ U[0,2], log beta ~ U[ln1.8, ln6]
        GSP    : gamma ~ U[0,1.5], log beta ~ U[ln1.1, ln3.2]
    For every subject the exact binomial likelihood of their choices over the
    distinct trial types is integrated (grid) against each theory's prior; the
    attenuation A_eff gets a shared neutral prior (the two theories' A_eff
    ranges nearly coincide, so it must not drive the comparison). Subjects are
    independent, so on the pooled DataFrame the value is the sum of the
    per-subject log Bayes factors (the joint log BF of the whole dataset); on a
    single-subject slice it is that subject's individual log Bayes factor.
    Positive values favor RSD-vgp, negative values favor GSP.
    """
    VAL = np.array([1.0, 0.99, 0.92, 0.85, 0.75, 0.65,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    df = data
    n_rows = len(df)
    if n_rows == 0:
        return 0.0

    a_list = df['option_a_ratings'].tolist()
    b_list = df['option_b_ratings'].tolist()
    resp = np.asarray(df['response'].values, dtype=int)
    subj = df['subject_id'].values

    # Hashable string keys for the trial types (avoids unhashable list cells).
    abmap = {}
    key_list = []
    for i in range(n_rows):
        a = a_list[i]
        b = b_list[i]
        ka = ''.join(['1' if int(x) else '0' for x in a])
        kb = ''.join(['1' if int(x) else '0' for x in b])
        key = ka + '#' + kb
        key_list.append(key)
        if key not in abmap:
            abmap[key] = (np.array([int(x) for x in a], dtype=float),
                          np.array([int(x) for x in b], dtype=float))

    tkeys = sorted(abmap.keys())
    tidx = {k: i for i, k in enumerate(tkeys)}
    T = len(tkeys)

    K = defaultdict(lambda: np.zeros(T))  # A-choices per trial type
    N = defaultdict(lambda: np.zeros(T))  # presentations per trial type
    for i in range(n_rows):
        s = subj[i]
        ti = tidx[key_list[i]]
        N[s][ti] += 1.0
        if resp[i] == 0:
            K[s][ti] += 1.0

    # Parameter grids (cover both theories' supports).
    gam = np.arange(0.02, 2.0001, 0.02)      # validity exponent gamma
    lbg = np.arange(0.02, 1.9401, 0.02)      # log raw inverse temperature
    Agr = np.arange(0.42, 0.9901, 0.03)     # effective attenuation A_eff
    G, L, NA = len(gam), len(lbg), len(Agr)

    # E_t(gamma) = sum_j v_j^gamma (b_j - a_j), shape (G, T)
    W = VAL[None, :] ** gam[:, None]
    E = np.empty((G, T))
    for key in tkeys:
        a, b = abmap[key]
        E[:, tidx[key]] = W @ (b - a)

    # s = sigmoid(beta * E) - 0.5, shape (L, G, T)
    beta = np.exp(lbg)
    u = beta[:, None, None] * E[None, :, :]
    au = np.abs(u)
    ex = np.exp(-au)
    sig = np.where(u >= 0.0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
    s = sig - 0.5

    # p(A) = 0.5 + A_eff * s, shape (NA, L, G, T)
    P = 0.5 + Agr[:, None, None, None] * s[None, :, :, :]
    P = np.clip(P, 1e-9, 1.0 - 1e-9)
    logP = np.log(P)
    log1P = np.log1p(-P)
    del P, s, sig, u, au, ex, E, W

    # Log priors.
    logwA = -np.log(NA)  # shared neutral attenuation prior
    logpR_g = np.full(G, np.log(0.5))                          # RSD: U[0,2]
    logpG_g = np.where(gam <= 1.5, np.log(1.0 / 1.5), -np.inf)  # GSP: U[0,1.5]
    loR, hiR = np.log(1.8), np.log(6.0)
    loG, hiG = np.log(1.1), np.log(3.2)
    logpR_b = np.where((lbg >= loR) & (lbg <= hiR),
                       -np.log(hiR - loR), -np.inf)              # RSD: U[ln1.8, ln6]
    logpG_b = np.where((lbg >= loG) & (lbg <= hiG),
                       -np.log(hiG - loG), -np.inf)             # GSP: U[ln1.1, ln3.2]

    PR = logpR_b[None, :, None] + logpR_g[None, None, :] + logwA  # (L, G)
    PG = logpG_b[None, :, None] + logpG_g[None, None, :] + logwA  # (L, G)

    def lse(x):
        m = np.max(x)
        if not np.isfinite(m):
            return -np.inf
        return m + np.log(np.sum(np.exp(x - m)))

    total_llr = 0.0
    for sid in list(N.keys()):
        kk = K[sid]
        nn = N[sid]
        if nn.sum() <= 0:
            continue
        # binomial log-likelihood on the (NA, L, G) parameter grid
        LL = (np.einsum('t,ijkt->ijk', kk, logP)
              + np.einsum('t,ijkt->ijk', nn - kk, log1P))
        total_llr += lse(LL + PR[None, :, :]) - lse(LL + PG[None, :, :])

    return float(total_llr)
```

**Observed (real) value:** 52.5984 (var=6.4849)
**Candidate trajectory (this loop):**
  - iter 1: 89.8509 (var=16.8172) (Δ vs real +37.2525)
  - iter 2 (current): 123.6290 (var=16.2951) (Δ vs real +71.0306)
**Other theories' values on this metric (for reference):**
- pi_7: 96.6930 (var=15.9176)
- pi_6: -21.1257 (var=0.4647)
- pi_1: -150.7700 (var=0.2496)
- pi_2: 220.1524 (var=4.1508)
- pi_3: -87.8944 (var=0.2065)
- pi_4: 18.4709 (var=6.0537)
- pi_5: -41.9588 (var=1.0440)
- pi_8: 67.3084 (var=8.4566)
- pi_9: 113.8881 (var=19.4047)
- pi_10: 114.0736 (var=12.7298)

### Experiment 12
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # ------------------------------------------------------------------
    # Per-subject BOUNDED log Bayes factor (GSP vs RSD-vgp), averaged
    # over subjects:
    #     score_i = tanh( log m_GSP(data_i) - log m_RSD(data_i) )
    #
    # Both theories share the EXACT per-trial likelihood
    #     p(A) = 0.5 + A_eff * ( sigmoid(beta * E) - 0.5 ),
    #     E    = sum_j v_j^gamma * (b_j - a_j)   (raw anti-endorsement),
    #     A_eff = (1 - eps) * (2m - 1),
    # so eps and m enter ONLY through A_eff and each subject's 96 choices
    # identify just (gamma, ln beta, A_eff). The theories differ ONLY in
    # the population prior over (gamma, ln beta):
    #     GSP : gamma ~ U[0, 1.5],  ln beta ~ U[ln 1.1, ln 3.2]
    #     RSD : gamma ~ U[0, 2.0],  ln beta ~ U[ln 1.8, ln 6.0]
    # (RSD's two-component gamma mixture is algebraically uniform.)
    # Each marginal is computed by grid quadrature over the theory's exact
    # box, using THIS experiment's true validities
    #     [1, 1, 1, .5, .5, .5, .5, .5, .5, .6, .6, .7]
    # (verified by reproducing the design's own mid-box predictions:
    # B1 = .77, F0 = .42, C2 = .74 at gamma=.75, beta=2, A_eff=.70).
    # A_eff receives a shared uniform prior on [0.42, 0.98], which covers
    # both theories' induced ranges and cancels in the ratio.
    # tanh() bounds each subject's contribution to (-1, 1), capping the
    # heavy log-BF tails that destroyed the between-subject variance
    # budget of the unbounded version.
    # ------------------------------------------------------------------

    V = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7])

    gam = np.linspace(0.0, 2.0, 81)                    # gamma grid (union)
    lnb = np.linspace(np.log(1.1), np.log(6.0), 61)    # ln beta grid (union)
    ave = np.linspace(0.42, 0.98, 12)                   # A_eff grid (shared)
    betas = np.exp(lnb)
    ng, nb, na = len(gam), len(lnb), len(ave)

    gsp_g = gam <= 1.5 + 1e-9
    gsp_b = lnb <= np.log(3.2) + 1e-9
    rsd_b = lnb >= np.log(1.8) - 1e-9
    all_g = np.ones(ng, dtype=bool)

    def subject_score(sub: pd.DataFrame) -> float:
        # ---- pool mirrored twins into canonical (X, Y) cells, X <= Y ----
        agg = {}
        for a, b, r in zip(sub["option_a_ratings"].tolist(),
                           sub["option_b_ratings"].tolist(),
                           sub["response"].tolist()):
            ta = tuple((int(v) for v in a))
            tb = tuple((int(v) for v in b))
            if ta <= tb:
                X, Y, choseX = ta, tb, (int(r) == 0)
            else:
                X, Y, choseX = tb, ta, (int(r) == 1)
            key = (X, Y)
            cnt = agg.get(key)
            if cnt is None:
                cnt = agg[key] = [0, 0]
            cnt[0] += 1
            if choseX:
                cnt[1] += 1

        D, kX, nT = [], [], []
        for (X, Y), cnt in agg.items():
            d = np.asarray(Y, dtype=float) - np.asarray(X, dtype=float)
            if not np.any(d != 0.0):
                continue            # exact tie: p = 0.5, cancels in the BF
            D.append(d)
            kX.append(float(cnt[1]))
            nT.append(float(cnt[0]))
        if not D:
            return 0.0
        D = np.vstack(D)                                # (C, F)
        kX = np.asarray(kX)
        kY = np.asarray(nT) - kX

        nf = D.shape[1]
        if nf <= len(V):
            v = V[:nf]
        else:
            v = np.concatenate([V, np.full(nf - len(V), 0.6)])

        W = np.power(v[None, :], gam[:, None])          # (ng, F)
        E = D @ W.T                                     # (C, ng)
        x = E[:, :, None] * betas[None, None, :]         # (C, ng, nb)
        sig = 0.5 * (1.0 + np.tanh(0.5 * x))             # stable logistic
        C = sig.shape[0]
        sig = sig.reshape(C, ng * nb)

        LL = np.empty((na, ng * nb))
        for ia in range(na):
            p = 0.5 + ave[ia] * (sig - 0.5)
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            ll = kX[:, None] * np.log(p) + kY[:, None] * np.log1p(-p)
            LL[ia] = ll.sum(axis=0)
        LL = LL.reshape(na, ng, nb)

        def logm(gmask, bmask):
            # grid-average of exp(LL) over the theory's box = marginal
            # likelihood under the uniform prior on that box
            sub_ = LL[:, gmask, :][:, :, bmask]
            m = float(sub_.max())
            return m + float(np.log(np.mean(np.exp(sub_ - m))))

        lg = logm(gsp_g, gsp_b)
        lr = logm(all_g, rsd_b)
        return float(np.tanh(lg - lr))

    data = data.reset_index(drop=True)
    if len(data) == 0:
        return 0.0
    if "subject_id" not in data.columns:
        return subject_score(data)
    scores = [subject_score(sub)
              for _, sub in data.groupby("subject_id", sort=False)]
    if not scores:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** -0.4144 (var=0.4154)
**Candidate trajectory (this loop):**
  - iter 1: -0.1967 (var=0.3747) (Δ vs real +0.2178)
  - iter 2 (current): -0.2917 (var=0.3583) (Δ vs real +0.1227)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2650 (var=0.2313)
- pi_7: -0.2520 (var=0.3854)
- pi_1: 0.7899 (var=0.0150)
- pi_2: -0.9161 (var=0.0663)
- pi_3: 0.7753 (var=0.0098)
- pi_4: 0.0191 (var=0.3421)
- pi_5: 0.5030 (var=0.2038)
- pi_8: -0.3417 (var=0.3534)
- pi_9: -0.2905 (var=0.4581)
- pi_10: -0.2310 (var=0.5023)

### Experiment 13
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Joint log Bayes factor between the two population models.
    # Both theories share the SAME per-trial likelihood
    #   p(A) = 0.5 + A_eff * (sigmoid(beta * E_gamma) - 0.5),
    #   E_gamma = sum_j v_j^gamma * (b_j - a_j),  A_eff = (1-eps)(2m-1).
    # They differ ONLY in the population priors over (gamma, beta, A_eff):
    #   RSD-vgp: gamma ~ U[0,2]; beta ~ logU[1.8,6]; m~U[0.74,1], eps~U[0.03,0.11]
    #   EPAI-xp: gamma ~ 0.55*U[0,1.2] + 0.45*U[1.2,2.4]; beta ~ logU[2,7];
    #            m~U[0.78,0.96], eps~U[0.03,0.10]
    # The metric marginalizes the likelihood over each theory's prior by
    # deterministic quadrature and returns sum_i ln[ m_RSD(data_i)/m_EPAI(data_i) ]
    # over subjects (subjects are independent, so the joint log Bayes factor
    # is exactly the sum of the per-subject log Bayes factors).
    val = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    if data is None or len(data) == 0:
        return 0.0

    # ---------- map trials to unique stimulus types ----------
    a_lists = data['option_a_ratings'].tolist()
    b_lists = data['option_b_ratings'].tolist()
    keys = [(tuple(x), tuple(y)) for x, y in zip(a_lists, b_lists)]

    type_index = {}
    diff_rows = []
    for kk in keys:
        if kk not in type_index:
            type_index[kk] = len(diff_rows)
            diff_rows.append(np.asarray(kk[1], dtype=float) - np.asarray(kk[0], dtype=float))
    D = np.vstack(diff_rows)                      # [T, F]  b - a per type
    T = D.shape[0]
    tidx = np.array([type_index[kk] for kk in keys], dtype=int)

    resp_ser = pd.to_numeric(pd.Series(data['response'].values), errors='coerce')
    resp = resp_ser.to_numpy(dtype=float)
    if np.isnan(resp).any():
        vals = [str(r).strip().upper() for r in data['response'].values]
        resp = np.array([0.0 if v in ('0', 'A') else 1.0 for v in vals])

    uq, sidx = np.unique(np.asarray(data['subject_id'].values), return_inverse=True)
    n_sub = int(len(uq))

    K = np.zeros((n_sub, T))                     # A-choice counts per type
    N = np.zeros((n_sub, T))                     # total trials per type
    isA = (resp == 0).astype(float)
    np.add.at(K, (sidx, tidx), isA)
    np.add.at(N, (sidx, tidx), 1.0)

    # ---------- gamma quadrature: 96 midpoint bins on [0, 2.4] ----------
    ge = np.linspace(0.0, 2.4, 97)
    gm = 0.5 * (ge[:-1] + ge[1:])
    dg = float(ge[1] - ge[0])
    wR_g = np.where(gm <= 2.0, 0.5, 0.0) * dg                 # RSD: U[0,2]
    wE_g = np.where(gm <= 1.2, 0.55 / 1.2, 0.45 / 1.2) * dg    # EPAI mixture
    wR_g = wR_g / wR_g.sum()
    wE_g = wE_g / wE_g.sum()
    G = len(gm)

    # ---------- beta quadrature: 12 midpoint bins in log space ----------
    le = np.linspace(np.log(1.8), np.log(7.0), 13)
    bm = 0.5 * (le[:-1] + le[1:])
    beta = np.exp(bm)

    def _log_overlap(lo, hi):
        lo_c = np.clip(le[:-1], lo, hi)
        hi_c = np.clip(le[1:], lo, hi)
        w = np.maximum(hi_c - lo_c, 0.0)
        s = w.sum()
        return w / s if s > 0 else np.full_like(w, 1.0 / len(w))

    wR_b = _log_overlap(np.log(1.8), np.log(6.0))   # logU[1.8, 6]
    wE_b = _log_overlap(np.log(2.0), np.log(7.0))   # logU[2, 7]
    B = len(beta)

    # ---------- evidence and sigmoid - 0.5 ----------
    Wg = np.power(val[None, :], gm[:, None])         # [G, F] v^gamma
    Ev = Wg @ D.T                                    # [G, T]
    x = beta[:, None, None] * Ev[None, :, :]         # [B, G, T]
    x = np.clip(x, -60.0, 60.0)
    u = 1.0 / (1.0 + np.exp(-x)) - 0.5               # sigmoid(beta*E) - 0.5
    C = B * G
    u_flat = u.reshape(C, T)                         # c = b*G + g

    w_cR = (wR_b[:, None] * wR_g[None, :]).ravel()
    w_cE = (wE_b[:, None] * wE_g[None, :]).ravel()

    # ---------- A_eff quadrature from each theory's (m, eps) uniforms ----------
    def _a_grid(m_lo, m_hi, e_lo, e_hi):
        mm = m_lo + (m_hi - m_lo) * (np.arange(7) + 0.5) / 7.0
        ee = e_lo + (e_hi - e_lo) * (np.arange(4) + 0.5) / 4.0
        A = (2.0 * mm - 1.0)[:, None] * (1.0 - ee)[None, :]
        return A.ravel()

    def _prep(A_vals, w_c):
        pu = 0.5 + A_vals[:, None, None] * u_flat[None, :, :]   # p(A)
        D1 = np.log(pu) - np.log1p(-pu)                        # log p - log(1-p)
        L0 = np.log1p(-pu)                                     # log(1-p)
        nA = A_vals.size
        D1m = D1.reshape(nA * C, T)
        L0m = L0.reshape(nA * C, T)
        logw = (np.full(nA, -np.log(nA))[:, None] + np.log(w_c)[None, :]).ravel()
        return D1m, L0m, logw

    A_R = _a_grid(0.74, 1.00, 0.03, 0.11)
    A_E = _a_grid(0.78, 0.96, 0.03, 0.10)
    D1R, L0R, lwR = _prep(A_R, w_cR)
    D1E, L0E, lwE = _prep(A_E, w_cE)

    def _logsumexp(z):
        m = np.max(z)
        return float(m + np.log(np.sum(np.exp(z - m))))

    total = 0.0
    for i in range(n_sub):
        # log likelihood of subject i's counts at every grid point:
        # sum_t [ K_t*log p_t + (N_t-K_t)*log(1-p_t) ]
        lr = _logsumexp(D1R @ K[i] + L0R @ N[i] + lwR)
        le_i = _logsumexp(D1E @ K[i] + L0E @ N[i] + lwE)
        total += lr - le_i
    return float(total)
```

**Observed (real) value:** 6.0286 (var=0.5984)
**Candidate trajectory (this loop):**
  - iter 1: 6.9418 (var=0.4087) (Δ vs real +0.9132)
  - iter 2 (current): -3.8083 (var=0.9632) (Δ vs real -9.8369)
**Other theories' values on this metric (for reference):**
- pi_7: 5.3165 (var=0.3027)
- pi_8: -30.9335 (var=3.3944)
- pi_1: 136.1063 (var=0.3030)
- pi_2: -106.1209 (var=4.0607)
- pi_3: 135.2209 (var=0.2267)
- pi_4: 38.8942 (var=0.7056)
- pi_5: 7.2875 (var=0.2574)
- pi_6: 15.9085 (var=0.2273)
- pi_9: 6.8983 (var=0.2368)
- pi_10: -9.6424 (var=0.6044)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """Estimated population fraction of subjects whose validity exponent
    gamma lies above 2.0 (the hard support edge of RSD-vgp's gamma box),
    measured with the anti-endorsement raw-weight measurement model that
    BOTH theories share per-trial, and stabilized by a weak empirical-Bayes
    smoothing prior so the statistic is well-behaved on single subjects."""
    VAL = np.array([1.0, 1.0, 1.0, 0.74, 0.72, 0.68, 0.63,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    KAPPA = 15.0      # smoothing prior strength (in subjects)
    P0 = 0.075        # smoothing prior mean tail mass
    TAIL = 2.0        # gamma support edge of the competing theory

    if data is None or len(data) == 0:
        return P0

    # ---- tidy parse (list cells -> hashable tuples) ----
    ka = [tuple(int(v) for v in x) for x in data['option_a_ratings']]
    kb = [tuple(int(v) for v in x) for x in data['option_b_ratings']]
    resp = np.asarray(data['response'].values).astype(int)
    choseA = (resp == 0).astype(int)

    df = pd.DataFrame({'s': np.asarray(data['subject_id'].values),
                       'ka': ka, 'kb': kb, 'cA': choseA})
    cnt = df.groupby(['s', 'ka', 'kb']).agg(
        n=('cA', 'size'), k=('cA', 'sum')).reset_index()

    uniq = sorted(set(zip(cnt['ka'].values, cnt['kb'].values)))
    tidx = {t: i for i, t in enumerate(uniq)}
    subjects = list(pd.unique(cnt['s']))
    sidx = {s: i for i, s in enumerate(subjects)}
    S = len(subjects)
    T = len(uniq)
    if S == 0 or T == 0:
        return P0

    K = np.zeros((S, T))   # A-choices per (subject, trial type)
    N = np.zeros((S, T))   # trials per (subject, trial type)
    for row in cnt.itertuples(index=False):
        K[sidx[row.s], tidx[(row.ka, row.kb)]] = float(row.k)
        N[sidx[row.s], tidx[(row.ka, row.kb)]] = float(row.n)

    # ---- diffuse parameter grids (theory-neutral, covers both supports) ----
    gam = np.linspace(0.0, 2.6, 53)                       # validity exponent
    Agrid = np.linspace(0.35, 0.95, 7)                    # effective attenuation
    Bgrid = np.exp(np.linspace(np.log(1.8), np.log(7.0), 7))  # inverse temp
    Wg = VAL[None, :] ** gam[:, None]                     # (53, 13)

    # per-trial-type choice probabilities under the shared core:
    # p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
    # E(gamma) = sum_j v_j^gamma * (b_j - a_j)
    LP = np.empty((T, 53, 7, 7))
    L1P = np.empty((T, 53, 7, 7))
    for t, (a, b) in enumerate(uniq):
        av = np.array(a, dtype=float)
        bv = np.array(b, dtype=float)
        E = Wg @ (bv - av)                                # (53,)
        x = Bgrid[None, :] * E[:, None]                   # (53, 7)
        ex = np.exp(-np.abs(x))
        sig = np.where(x >= 0.0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
        P = 0.5 + Agrid[None, :, None] * (sig[:, None, :] - 0.5)  # (53,7,7)
        P = np.clip(P, 1e-6, 1.0 - 1e-6)
        LP[t] = np.log(P)
        L1P[t] = np.log1p(-P)

    tail_mask = gam > (TAIL + 1e-9)

    # ---- per-subject posterior P(gamma > 2 | own 96 trials) ----
    s_vals = np.empty(S)
    chunk = 64
    for i0 in range(0, S, chunk):
        Kc = K[i0:i0 + chunk]
        Nc = N[i0:i0 + chunk]
        logL = (np.tensordot(Kc, LP, axes=(1, 0)) +
                np.tensordot(Nc - Kc, L1P, axes=(1, 0)))   # (s,53,7,7)
        m = logL.max(axis=(1, 2, 3), keepdims=True)
        w = np.exp(logL - m)
        post = w.sum(axis=(2, 3))                          # marginal over A,beta
        tot = post.sum(axis=1)
        tot = np.where(tot <= 0.0, 1.0, tot)
        s_vals[i0:i0 + chunk] = post[:, tail_mask].sum(axis=1) / tot

    raw = float(np.mean(s_vals))
    # empirical-Bayes shrinkage toward the prior mean; with one subject the
    # posterior mean is prior-dominated (tight across subjects), with many
    # subjects the data dominate (population tail mass).
    return (KAPPA * P0 + S * raw) / (KAPPA + S)
```

**Observed (real) value:** 0.2983 (var=0.0006)
**Candidate trajectory (this loop):**
  - iter 1: 0.1864 (var=0.0003) (Δ vs real -0.1120)
  - iter 2 (current): 0.1069 (var=0.0002) (Δ vs real -0.1914)
**Other theories' values on this metric (for reference):**
- pi_8: 0.1396 (var=0.0003)
- pi_7: 0.0990 (var=0.0001)
- pi_1: 0.0175 (var=0.0000)
- pi_2: 0.7299 (var=0.0001)
- pi_3: 0.0173 (var=0.0000)
- pi_4: 0.1168 (var=0.0002)
- pi_5: 0.0871 (var=0.0002)
- pi_6: 0.0638 (var=0.0001)
- pi_9: 0.0951 (var=0.0001)
- pi_10: 0.1215 (var=0.0002)

### Experiment 15
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # ==================================================================
    # Population log Bayes factor: log P(data | RSD-vgp) - log P(data | SGAI-r7).
    # Both theories share the identical per-trial choice kernel
    #     p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
    #     E = sum_j v_j^gamma * (b_j - a_j),  A_eff = (1-eps)(2m-1),
    # and differ ONLY in their population priors over (gamma, beta, eps, m)
    # plus SGAI's per-trial gamma jitter (prob ~0.11, +/-0.4 around the
    # habitual gamma, capped at [0, 2.2]).  So the marginal likelihood
    # under each theory's stated population prior isolates exactly the
    # claims that separate them:
    #   RSD-vgp: gamma ~ U[0,2] exactly; beta log-U[1.8,6]; eps U[.03,.11];
            #   m U[.74,1]; no jitter.
    #   SGAI-r7: 80% majority (gamma 20%[.8,1.3]/48%[1.3,1.7]/24%[1.7,2.0]/
    #   8%(2.0,2.2]; beta log-U[1.8,6]; eps U[.03,.10]; m U[.8,.98]) plus
    #   20% low-precision cluster (gamma U[0,1]; beta log-U[1.2,1.8];
    #   eps U[.15,.25]); per-trial gamma jitter.
    # The metric integrates the shared kernel over these priors on a grid
    # (SGAI's jitter is folded into the per-trial kernel as a 0.89/0.11
    # mixture over the +/-0.4 gamma window) and returns the total log
    # likelihood ratio.  On a pooled DataFrame this is the dataset log Bayes
    # factor (sum of independent per-subject contributions); on a single
    # subject's slice it is that subject's log-BF contribution.
    # ==================================================================
    df = data.reset_index(drop=True)
    if len(df) == 0:
        return 0.0

    v = np.array([1.0, 1.0, 1.0,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                  0.6])
    nf = len(v)

    def _to_arr(x):
        if isinstance(x, np.ndarray):
            a = x.astype(float)
            if a.size == nf:
                return a
        try:
            a = np.asarray(list(x), dtype=float)
            if a.size == nf:
                return a
        except (TypeError, ValueError):
            pass
        s = str(x).strip().strip("[]()")
        return np.array([float(tok) for tok in s.replace(",", " ").split()])

    A = np.vstack([_to_arr(r) for r in df["option_a_ratings"]])
    B = np.vstack([_to_arr(r) for r in df["option_b_ratings"]])
    resp = df["response"].to_numpy(dtype=float)  # 0 = chose A, 1 = chose B

    # ---- unique trial types ----------------------------------------
    keys = [(tuple(A[i]), tuple(B[i])) for i in range(len(df))]
    index_of = {}
    type_idx = np.empty(len(keys), dtype=int)
    for i, k in enumerate(keys):
        if k not in index_of:
            index_of[k] = len(index_of)
        type_idx[i] = index_of[k]
    types = list(index_of.keys())
    Amat = np.array([np.array(t[0], dtype=float) for t in types])
    Bmat = np.array([np.array(t[1], dtype=float) for t in types])
    T = len(types)

    subj_codes = pd.factorize(df["subject_id"])[0]
    S = int(subj_codes.max()) + 1
    nA = np.zeros((S, T))
    nB = np.zeros((S, T))
    np.add.at(nA, (subj_codes, type_idx), (resp == 0).astype(float))
    np.add.at(nB, (subj_codes, type_idx), (resp == 1).astype(float))

    # ---- master parameter grids ------------------------------------
    gmaster = np.round(np.arange(0.0, 2.2 + 1e-9, 0.05), 6)       # 45 gammas
    bmaster = np.exp(np.linspace(np.log(1.2), np.log(6.0), 48))  # 48 betas

    # E[g, t] = sum_j v_j^g * (B_tj - A_tj)
    Vg = v[None, :] ** gmaster[:, None]
    E = Vg @ (Bmat - Amat).T

    def _sigmoid(x):
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    # s[g, b, t] = sigmoid(b * E) - 0.5   (kernel without jitter)
    X = bmaster[None, :, None] * E[:, None, :]
    s_master = _sigmoid(X) - 0.5

    # SGAI per-trial gamma jitter folded into the kernel:
    # s_jit(g) = 0.89*s(g) + 0.11*mean_{gt in [g-0.4, g+0.4] cap [0,2.2]} s(gt)
    delta = 0.11
    s_jit = np.empty_like(s_master)
    for gi in range(len(gmaster)):
        g = gmaster[gi]
        lo = int(np.searchsorted(gmaster, max(0.0, g - 0.4) - 1e-9))
        hi = int(np.searchsorted(gmaster, min(2.2, g + 0.4) + 1e-9))
        if hi <= lo:
            hi = lo + 1
        s_jit[gi] = ((1.0 - delta) * s_master[gi]
                     + delta * s_master[lo:hi].mean(axis=0))

    def _logml(smat, g_idx, g_w, b_idx, b_w, alphas, a_w):
        # marginal log-likelihood for every subject under one population
        # component given by (gamma grid/weights, beta grid/weights,
        # A_eff grid/weights); smat supplies the (possibly jittered) kernel.
        s_sel = smat[np.ix_(g_idx, b_idx)]                 # (ng, nb, T)
        p = 0.5 + alphas[None, None, :, None] * s_sel[:, :, None, :]
        p = np.clip(p, 1e-12, 1.0 - 1e-12)                 # (ng, nb, na, T)
        LP = np.log(p).reshape(-1, T)
        LQ = np.log1p(-p).reshape(-1, T)
        ll = nA @ LP.T + nB @ LQ.T                         # (S, ncombo)
        w = (g_w[:, None, None] * b_w[None, :, None]
             * a_w[None, None, :]).reshape(-1)
        ll = ll + np.log(np.maximum(w, 1e-300))[None, :]
        m = np.max(ll, axis=1)
        return m + np.log(np.sum(np.exp(ll - m[:, None]), axis=1))

    # ---- RSD-vgp population ----------------------------------------
    g_idx = np.arange(0, 41)                    # gamma ~ U[0, 2]
    g_w = np.full(41, 1.0 / 41)
    b_idx = np.arange(12, 48)                   # beta log-U[1.8, 6]
    b_w = np.full(36, 1.0 / 36)
    eps = np.linspace(0.03, 0.11, 4)
    mm = np.linspace(0.74, 1.00, 7)
    al = ((1.0 - eps)[None, :] * (2.0 * mm[:, None] - 1.0)).reshape(-1)
    lml_R = _logml(s_master, g_idx, g_w, b_idx, b_w, al,
                   np.full(al.size, 1.0 / al.size))

    # ---- SGAI-r7 low-precision cluster (weight 0.2) -----------------
    g_idx = np.arange(0, 21)                    # gamma ~ U[0, 1]
    g_w = np.full(21, 1.0 / 21)
    b_idx = np.arange(0, 12)                    # beta log-U[1.2, 1.8]
    b_w = np.full(12, 1.0 / 12)
    eps = np.linspace(0.15, 0.25, 4)
    mm = np.linspace(0.80, 0.98, 5)
    al = ((1.0 - eps)[None, :] * (2.0 * mm[:, None] - 1.0)).reshape(-1)
    lml_L = _logml(s_jit, g_idx, g_w, b_idx, b_w, al,
                   np.full(al.size, 1.0 / al.size))

    # ---- SGAI-r7 majority cluster (weight 0.8) ----------------------
    g_idx = np.concatenate([np.arange(16, 26), np.arange(26, 34),
                            np.arange(34, 40), np.arange(40, 45)])
    g_w = np.concatenate([np.full(10, 0.20 / 10), np.full(8, 0.48 / 8),
                          np.full(6, 0.24 / 6), np.full(5, 0.08 / 5)])
    b_idx = np.arange(12, 48)                   # beta log-U[1.8, 6]
    b_w = np.full(36, 1.0 / 36)
    eps = np.linspace(0.03, 0.10, 3)
    mm = np.linspace(0.80, 0.98, 5)
    al = ((1.0 - eps)[None, :] * (2.0 * mm[:, None] - 1.0)).reshape(-1)
    lml_M = _logml(s_jit, g_idx, g_w, b_idx, b_w, al,
                   np.full(al.size, 1.0 / al.size))

    lml_S = np.logaddexp(np.log(0.2) + lml_L, np.log(0.8) + lml_M)

    llr = lml_R - lml_S   # per-subject log Bayes factor (RSD vs SGAI)
    return float(np.sum(llr))
```

**Observed (real) value:** 12.9184 (var=1.0896)
**Candidate trajectory (this loop):**
  - iter 1: 30.7524 (var=2.9313) (Δ vs real +17.8340)
  - iter 2 (current): 11.3814 (var=0.8739) (Δ vs real -1.5370)
**Other theories' values on this metric (for reference):**
- pi_7: 23.7874 (var=2.2777)
- pi_9: -14.1580 (var=0.3187)
- pi_1: -19.5136 (var=0.1437)
- pi_2: 11.8045 (var=1.9732)
- pi_3: -21.6258 (var=0.1020)
- pi_4: 6.5010 (var=0.5862)
- pi_5: -8.3945 (var=0.2766)
- pi_6: 28.0330 (var=1.5827)
- pi_8: 17.1418 (var=1.6894)
- pi_10: 7.5133 (var=1.5781)

### Experiment 16
**Design**
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Fixed design validities (15 experts, design-time constants).
    V = np.array([1.0, 1.0, 1.0, 0.8, 0.75, 0.65, 0.6,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def to_vec(x):
        if isinstance(x, str):
            s = x.strip().lstrip('[').rstrip(']')
            return np.array([float(t) for t in s.replace(',', ' ').split()])
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            arr = arr.ravel()
        return arr

    # Raw (unnormalized) validity weights at a low-gamma and a high-gamma
    # regime. Under the shared anti-endorsement core, a trial's evidence is
    # E(gamma) = sum_j v_j^gamma (b_j - a_j); the two critical 'flip' rungs
    # of this design are exactly the trials where sign(E) changes between
    # these two regimes (flip points gamma* ~ 0.73 and ~ 1.0).
    w_lo = np.power(V, 0.5)
    w_hi = np.power(V, 1.5)

    # Bayesian (Beta) regularization of the sparse-side rate: 96 pseudo-
    # trials centered at 0.5. Per subject only 16 critical trials exist
    # (binomial SE ~ 0.12), so the raw per-subject rate is unstable; the
    # shrinkage stabilizes individual-subject estimates while leaving the
    # pooled estimate (16 x 25 = 400 critical trials) essentially
    # unchanged.
    KAPPA = 96.0
    M0 = 0.5

    k = 0  # sparse-side choices on flip trials
    n = 0  # flip trials seen

    a_col = data['option_a_ratings'].values
    b_col = data['option_b_ratings'].values
    r_col = data['response'].values

    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if a.size != V.size or b.size != V.size:
            continue
        e_lo = float(np.sum(w_lo * (b - a)))
        e_hi = float(np.sum(w_hi * (b - a)))
        if e_lo == 0.0 or e_hi == 0.0:
            continue  # tie trial, uninformative
        if (e_lo > 0.0) == (e_hi > 0.0):
            continue  # not a flip trial (sign of evidence is gamma-invariant)
        # Flip trial: low-gamma subjects prefer one side, high-gamma
        # subjects the other. The low-gamma-preferred side is the one with
        # the smaller raw validity weight (the 'count-sparser' side).
        n += 1
        sparse_is_a = (e_lo > 0.0)
        chose_a = (int(float(r_raw)) == 0)
        if chose_a == sparse_is_a:
            k += 1

    if n + KAPPA <= 0:
        return float(M0)
    return float((k + KAPPA * M0) / (n + KAPPA))
```

**Observed (real) value:** 0.4335 (var=0.0024)
**Candidate trajectory (this loop):**
  - iter 1: 0.4397 (var=0.0010) (Δ vs real +0.0063)
  - iter 2 (current): 0.4275 (var=0.0009) (Δ vs real -0.0060)
**Other theories' values on this metric (for reference):**
- pi_9: 0.3750 (var=0.0007)
- pi_7: 0.5335 (var=0.0013)
- pi_1: 0.8058 (var=0.0004)
- pi_2: 0.1931 (var=0.0005)
- pi_3: 0.8549 (var=0.0001)
- pi_4: 0.5201 (var=0.0006)
- pi_5: 0.5212 (var=0.0004)
- pi_6: 0.5759 (var=0.0005)
- pi_8: 0.4018 (var=0.0010)
- pi_10: 0.4754 (var=0.0011)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

_CACHE = {}

_VALIDITIES = np.array([1.0, 1.0, 1.0,
                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                        0.72, 0.63])


def _to_array(x):
    """Coerce a rating cell (list/tuple/ndarray/str) to a float ndarray."""
    if isinstance(x, str):
        x = [t for t in x.strip('[]() ').split(',') if t.strip() != '']
    return np.asarray(list(x), dtype=float)


def metric(data: pd.DataFrame) -> float:
    """
    Mean per-subject log Bayes factor, RSD-vgp vs HT-AEI-r6.

    Both theories share the exact per-trial likelihood
        p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
        E    = sum_j v_j^gamma * (b_j - a_j),
    so the ONLY thing that distinguishes them is the population
    distribution of (gamma, beta). This metric integrates the exact
    binomial likelihood of each subject's 96 choices over each theory's
    stated population prior:
      RSD-vgp : gamma ~ U[0, 2] exactly; beta log-U[1.8, 6.0].
      HT-AEI  : 40% gamma U[0,1]  & beta log-U[2.2, 4.0];
                25% gamma U[1.05,1.35] & beta log-U[2.2, 3.2];
                35% gamma U[2.0,2.1]  & beta log-U[2.2, 2.8].
    with a common neutral uniform prior on A_eff in [0.42, 0.97]
    (the shared range of (1-eps)(2m-1) under both theories).

    Returns the mean over subjects of log m_RSD - log m_HT.
    Positive = data favor RSD-vgp; negative = data favor HT-AEI-r6.
    On a single subject's slice it returns that subject's log Bayes
    factor, so the pooled value is exactly the mean of the per-subject
    values and the between-subject variance is well defined.
    """
    df = data.reset_index(drop=True).copy()

    def _key(x):
        return ''.join([str(int(round(v))) for v in _to_array(x)])

    df['stim'] = (df['option_a_ratings'].apply(_key) + '|' +
                  df['option_b_ratings'].apply(_key))

    stims = sorted(set(df['stim'].tolist()))
    ckey = tuple(stims)

    if ckey not in _CACHE:
        # ---------------- parameter grids ----------------
        g_grid = np.round(np.arange(0.0, 2.1001, 0.01), 10)      # 211 pts
        b_grid = np.exp(np.linspace(np.log(1.8), np.log(6.0), 41))
        A_grid = np.linspace(0.42, 0.97, 15)

        dg = float(g_grid[1] - g_grid[0])
        dlb = float(np.log(b_grid[1]) - np.log(b_grid[0]))
        dA = float(A_grid[1] - A_grid[0])

        # ---------------- population priors ----------------
        # gamma densities (per unit gamma)
        pr_rsd = np.where(g_grid <= 2.0 + 1e-9, 0.5, 0.0)
        pr_ht = (0.40 * np.where(g_grid <= 1.0 + 1e-9, 1.0, 0.0)
                 + 0.25 * np.where((g_grid >= 1.05 - 1e-9) &
                                   (g_grid <= 1.35 + 1e-9),
                                   1.0 / 0.30, 0.0)
                 + 0.35 * np.where((g_grid >= 2.0 - 1e-9) &
                                   (g_grid <= 2.10 + 1e-9), 10.0, 0.0))

        def _logu(lo, hi):
            # log-uniform density on the beta grid
            return np.where((b_grid >= lo - 1e-9) & (b_grid <= hi + 1e-9),
                            1.0 / np.log(hi / lo), 0.0)

        # joint (gamma, beta) grid log-masses for RSD
        dens_rsd = pr_rsd[:, None] * _logu(1.8, 6.0)[None, :]
        lw_rsd = np.full(dens_rsd.shape, -np.inf)
        msk = dens_rsd > 0
        lw_rsd[msk] = np.log(dens_rsd[msk]) + np.log(dg) + np.log(dlb)

        # joint (gamma, beta) grid log-masses for HT (segment-specific beta)
        bw_ht = np.zeros((len(g_grid), len(b_grid)))
        bw_ht[g_grid <= 1.0 + 1e-9] = _logu(2.2, 4.0)
        bw_ht[(g_grid >= 1.05 - 1e-9) & (g_grid <= 1.35 + 1e-9)] = _logu(2.2, 3.2)
        bw_ht[(g_grid >= 2.0 - 1e-9) & (g_grid <= 2.10 + 1e-9)] = _logu(2.2, 2.8)
        dens_ht = pr_ht[:, None] * bw_ht
        lw_ht = np.full(dens_ht.shape, -np.inf)
        msk = dens_ht > 0
        lw_ht[msk] = np.log(dens_ht[msk]) + np.log(dg) + np.log(dlb)

        # common neutral A_eff prior (uniform on the shared range)
        log_a_w = float(np.log(dA / (0.97 - 0.42)))

        # ---------------- per-stimulus likelihood tables ----------------
        first = df.drop_duplicates(subset='stim').set_index('stim')
        a_mat = np.vstack([_to_array(first.loc[s, 'option_a_ratings'])
                           for s in stims])
        b_mat = np.vstack([_to_array(first.loc[s, 'option_b_ratings'])
                           for s in stims])
        D = b_mat - a_mat                                  # (n_u, 14)
        Wg = np.power(_VALIDITIES[:, None], g_grid[None, :])  # (14, n_g)
        E = D @ Wg                                        # (n_u, n_g)

        X = E[:, :, None] * b_grid[None, None, :]          # (n_u, n_g, n_b)
        S = 1.0 / (1.0 + np.exp(-np.clip(X, -60.0, 60.0)))
        core = S - 0.5
        P = 0.5 + A_grid[None, None, None, :] * core[:, :, :, None]
        P = np.clip(P, 1e-9, 1.0 - 1e-9)
        logP = np.log(P)
        log1mP = np.log1p(-P)

        _CACHE[ckey] = (logP, log1mP, lw_rsd, lw_ht, log_a_w)
    else:
        logP, log1mP, lw_rsd, lw_ht, log_a_w = _CACHE[ckey]

    stim_idx = {s: i for i, s in enumerate(stims)}
    n_u = len(stims)

    vals = []
    for sid, sdf in df.groupby('subject_id', sort=True):
        nA = np.zeros(n_u)
        nB = np.zeros(n_u)
        for s, v in sdf.groupby('stim')['response']:
            v = np.asarray(v, dtype=float)
            i = stim_idx[s]
            nb = float(np.sum(v))          # response 1 = chose B
            nB[i] = nb
            nA[i] = float(len(v)) - nb    # response 0 = chose A

        # exact binomial log-likelihood on the (gamma, beta, A_eff) grid
        LL = (np.tensordot(nA, logP, axes=([0], [0])) +
              np.tensordot(nB, log1mP, axes=([0], [0])))   # (n_g, n_b, n_A)

        def _marg(lw):
            tot = LL + lw[:, :, None] + log_a_w
            mx = float(np.max(tot))
            return mx + float(np.log(np.sum(np.exp(tot - mx))))

        vals.append(_marg(lw_rsd) - _marg(lw_ht))

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.8355 (var=2.6525)
**Candidate trajectory (this loop):**
  - iter 1: -0.0726 (var=0.9273) (Δ vs real -0.9081)
  - iter 2 (current): 0.1581 (var=0.8823) (Δ vs real -0.6774)
**Other theories' values on this metric (for reference):**
- pi_7: 0.3381 (var=0.3081)
- pi_10: -0.2872 (var=0.6994)
- pi_1: 0.2008 (var=0.0004)
- pi_2: -2.3152 (var=0.1061)
- pi_3: 0.2408 (var=0.0009)
- pi_4: 0.1507 (var=0.3046)
- pi_5: 0.0018 (var=0.2637)
- pi_6: 0.2901 (var=0.3029)
- pi_8: -0.0356 (var=1.2437)
- pi_9: 0.2908 (var=1.3375)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Crossing-point population statistic.
    # Per subject: fit the shared core model p(B) = 0.5 - A*(sigmoid(beta*E(gamma)) - 0.5),
    # E(gamma) = sum_j v_j^gamma * (b_j - a_j), by grid MLE over (gamma, beta, A),
    # then score s = P(gamma > 1.95) - P(1.40 < gamma <= 1.95) under the flat-prior
    # profile-likelihood posterior over gamma. Metric = mean of s over subjects.
    a_list = data["option_a_ratings"].to_numpy()
    b_list = data["option_b_ratings"].to_numpy()
    A_mat = np.vstack([np.asarray(r, dtype=float) for r in a_list])
    B_mat = np.vstack([np.asarray(r, dtype=float) for r in b_list])
    y = data["response"].to_numpy(dtype=float)
    subj = data["subject_id"].to_numpy()

    n_feat = A_mat.shape[1]
    if n_feat >= 8:
        # design validities: six 1.00 cues, twenty-four 0.50 cues, 0.64, 0.76
        V = np.array([1.0] * 6 + [0.5] * (n_feat - 8) + [0.64, 0.76], dtype=float)
    else:
        V = np.ones(n_feat, dtype=float)

    gam = np.linspace(0.0, 2.5, 126)
    bet = np.geomspace(1.6, 7.5, 8)
    amps = np.linspace(0.25, 0.97, 9)

    Wg = V[None, :] ** gam[:, None]  # (G, F) validity-power weights per gamma

    scores = []
    for s_id in pd.unique(data["subject_id"]):
        m = subj == s_id
        D = (B_mat[m] - A_mat[m]).astype(float)  # (n, F)
        yy = y[m]  # 1 = chose B
        E = D @ Wg.T  # (n, G) raw evidence for A over B
        X = E[:, :, None] * bet[None, None, :]  # (n, G, B)
        d = 0.5 * np.tanh(0.5 * X)  # sigmoid(X) - 0.5, stable, in [-0.5, 0.5]
        ll_prof = np.full(gam.shape[0], -np.inf)  # profile log-likelihood over gamma
        for a in amps:
            P = np.clip(0.5 - a * d, 1e-9, 1.0 - 1e-9)  # p(choose B)
            ll = np.sum(
                yy[:, None, None] * np.log(P)
                + (1.0 - yy)[:, None, None] * np.log(1.0 - P),
                axis=0,
            )  # (G, B)
            ll_prof = np.maximum(ll_prof, ll.max(axis=1))
        w = np.exp(ll_prof - ll_prof.max())
        w = w / w.sum()  # posterior over gamma (flat prior on [0, 2.5])
        p_hi = float(w[gam > 1.95].sum())          # tail mass above RSD's hard cap
        p_gap = float(w[(gam > 1.40) & (gam <= 1.95)].sum())  # HT's empty-gap mass
        scores.append(p_hi - p_gap)
    return float(np.mean(scores))
```

**Observed (real) value:** -0.2077 (var=0.1170)
**Candidate trajectory (this loop):**
  - iter 1: -0.1488 (var=0.2517) (Δ vs real +0.0590)
  - iter 2 (current): -0.0012 (var=0.1996) (Δ vs real +0.2065)
**Other theories' values on this metric (for reference):**
- pi_10: 0.1782 (var=0.1651)
- pi_7: -0.2078 (var=0.1272)
- pi_1: -0.0043 (var=0.0003)
- pi_2: 0.9966 (var=0.0002)
- pi_3: -0.0004 (var=0.0000)
- pi_4: -0.2504 (var=0.1229)
- pi_5: 0.0592 (var=0.0565)
- pi_6: -0.0842 (var=0.0441)
- pi_8: -0.0112 (var=0.3413)
- pi_9: -0.3784 (var=0.2838)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate is a substantial, genuinely mechanism-level improvement over both predecessors and was correctly accepted by the gate (loss 0.0855). The shared anti-endorsement raw-weight kernel is now essentially locked in: Exps 1, 3–10 all land within noise of the real values (dissociation 0.166 vs 0.158; depth contrast -0.64 vs -0.72; evidence slopes -0.55/0.59 vs -0.52/0.59; density contrast -0.69 vs -0.68; extremity anchors 0.68/0.34 vs 0.70/0.35; steep-slope posterior 0.885 vs 0.911), and the two headline repair targets were hit: Exp 16's flip rate is now 0.4397 vs real 0.4335 (pi_7 overshot at 0.5335) and Exp 18's crossing statistic is -0.149 vs real -0.208 (pi_10 catastrophically overshot at +0.178). However, the NEW mechanism — per-trial gamma jitter — is miscalibrated on exactly the diagnostics it was invented to explain. (1) Exp 14 (tail-mass estimator, tiny variance, so highly decisive): candidate 0.186 vs real 0.298; the arbiter's own falsifiable signature specified 0.25–0.35, and the jitter delivered only ~60% of the required apparent tail inflation. (2) Exp 15: candidate 30.75 vs real 12.9 — the jitter was supposed to ATTENUATE the RSD-vs-SGAI log BF 30–50% below pi_7's 23.8, but it actually INCREASED it to 30.8. The simulated subjects are still 'too clean' relative to real data. (3) Exp 11 shows the same failure in the same direction: 89.9 vs real 52.6 (pi_7: 96.7) — only ~7% attenuation where ~45% is needed. (4) Exp 12 (real -0.41 vs cand -0.20) and Exp 9 (real -1.76 vs cand -1.22) independently corroborate that real per-subject likelihoods are softer/lower-precision than the candidate simulates. (5) One counter-signal: Exp 17 went slightly negative (-0.07 vs real +0.84, where even no-jitter pi_7 only reaches +0.34) — the [2.0, 2.2] tail mass plus up-side jitter lets HT-AEI's edge-hug cluster partially fit the simulated subjects. This is a genuine tension: Exps 11/12/14/15 all demand noisier, higher-apparent-tail subjects, while Exps 17/18 penalize excessive true high-gamma mass. The resolution is to generate the apparent tail and the likelihood softness through the JITTER (which leaves the population density probe and crossing statistics intact) rather than through true support mass — i.e., strengthen the jitter while holding the gamma support fixed at [0, 2.2].
Rationale: The candidate was accepted and the base will not regress, so one more calibration pass is cheap and clearly warranted: the theory's single novel mechanism under-delivers on its own falsifiable signatures, and in two cases (Exps 11, 15) moved in the WRONG direction relative to pi_7. Concrete in-family adjustments, all within the arbiter-prescribed JU-AEI family: (a) STRENGTHEN THE JITTER: raise delta from 0.10–0.13 to roughly 0.18–0.25 and/or widen the redraw window from ±0.35 to ±0.50 (clip to [0, 2.5]). This is the primary lever for Exp 14's apparent tail mass (0.186 → target ~0.28–0.32) and for the per-subject likelihood softening that Exps 11, 12, and 15 require (real subjects behave as if their effective gamma and precision fluctuate more trial-to-trial than any static parameter can absorb). (b) Optionally add a small correlated per-trial log-beta jitter (±0.15–0.25 on the same delta event) to soften BFs WITHOUT lowering the asymptote anchors — do NOT raise epsilon or lower m, since Exps 7/8 extremity (0.68/0.34) are currently spot-on and would degrade. (c) DO NOT extend the gamma support past 2.2 — the Exp 17 sign flip (-0.07 vs real +0.84) and the Exp 18 upward drift (-0.149 vs real -0.208) are both driven by true mass above 2.0 combining with up-side jitter; keep the support exactly [0, 2.2] and let the jitter alone manufacture the apparent tail. If Exp 18 rises above ~-0.10 under the stronger jitter, make the window asymmetric (e.g., [gamma - 0.55, gamma + 0.35]) — a minor adjustment, not a redesign. (d) Guardrails: after re-tuning, verify Exp 10 stays ≥ 0.85, Exp 16 stays in [0.43, 0.47], and Exp 2 (currently 0.154 vs real 0.181, already slightly low) does not drop further; if the stronger jitter pushes Exp 2 down, compensate with a slightly tighter epsilon range at the low end. Priority ordering if tensions remain: Exp 14 (var ~0.0003, decisive) and Exp 15 (the pi_10 catastrophe site) outweigh Exp 17, whose real between-subject variance (2.65) makes it the noisiest diagnostic in the battery.

**Outcome of your advice:** iter 1 candidate loss=0.0855 -> CURRENT candidate loss=0.0712 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The gate accepted this pass (loss 0.0855 -> 0.0712), and the shared core remains excellent: Exps 2, 5, 7 are now essentially exact (0.1814/0.5949/0.7044 vs real 0.1814/0.5944/0.7044), Exps 1, 3, 4, 6, 8, 10, 16 all sit within noise, and Exp 15 \u2014 the pi_10 catastrophe site \u2014 improved dramatically (30.75 -> 11.38 vs real 12.92). So the strengthening direction was partially right. However, two of the four adjustments in this calibration pass backfired on exactly the diagnostics they were aimed at, and one produced a new sign flip. (1) EXP 14 (tail-mass estimator, var ~0.0002, the most decisive diagnostic in the battery): 0.107 vs real 0.298 \u2014 WORSE than iter 1's 0.186, moving away from the arbiter-specified 0.25-0.35 signature. The down-tilted window [gamma-0.55, gamma+0.45] was applied on the theory it would 'manufacture apparent tail mass', but a fixed-gamma per-subject MLE recovers apparent high gamma only from UP-side excursions; biasing the redraw downward pulls the static fit down, and the tail estimate collapsed. The pre-registered down-tilt contingency in my prior critique was conditioned on Exp 18 rising above -0.10; it was applied unconditionally and demonstrably hurt Exp 14 without fixing Exp 18. (2) EXP 18: -0.001 vs real -0.208 \u2014 also WORSE than iter 1 (-0.149), so the down-tilt failed on its own target. (3) EXP 13 flipped sign: -3.81 vs real +6.03 (iter 1: +6.94, near-perfect). The likely culprit is the UP-side of the correlated log-beta jitter: a subject at habit beta ~3.3 drawing +0.2 in log space reaches beta ~7.3, which lies OUTSIDE RSD's box (beta <= 6) but INSIDE EPAI's (beta <= 7) \u2014 the jitter manufactures exactly the evidence pattern EPAI rewards. (4) EXP 11 shows the same pathology: 123.6 vs real 52.6, WORSE than iter 1's 89.9 \u2014 the up-side beta excursions make simulated subjects look super-steep, inflating RSD's advantage instead of attenuating it. (5) EXP 9 drifted slightly further (-1.12 vs real -1.76). On the plus side, Exp 12 improved (-0.29 vs -0.20, real -0.41) and Exp 17 improved (-0.07 -> 0.16, real 0.84), confirming that likelihood softening per se is the right medicine \u2014 the problem is the asymmetry direction of the two jitter windows, not the jitter mechanism.",
  "verdict": "regenerate",
  "rationale": "verdict: regenerate"
}
```

## Usage

```json
{
  "input_tokens": 62698,
  "output_tokens": 4924,
  "reasoning_tokens": 4225,
  "provider": "Sail Research"
}
```
