# feedback_iter_03

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
- THEORY 2 = `pi_8`
- The recommendation below acts on THEORY 2 (= `pi_8`).

Replace pi_8 (EPAI-xp) with a new theory — Stochastic-Gate Anti-Endorsement Integration (SGAI) — that keeps the empirically confirmed core (endorsements read as evidence AGAINST the endorsed option; E = sum_j v_j^gamma (b_j - a_j) with RAW unnormalized validity weights; p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2) but revises the population and adds within-subject noise, targeting exactly the three failures: (i) TRIAL-LEVEL ATTENTION GATE: on each trial, with probability delta in [0.10, 0.25], the subject re-samples cue attention, drawing an effective per-trial gamma_t from a wide distribution around their mean gamma (or falling back to raw cue counting); this within-subject jitter flattens per-subject gamma identification, which reproduces Experiment 2's inflated tail-mass estimate (~0.25-0.32 vs observed 0.298) WITHOUT placing genuine mass above gamma = 2.2, so Experiment 1's RSD-vs-EPAI log Bayes factor stays mildly positive (~+3 to +7, matching observed +6.03 and avoiding EPAI's -31). (ii) POPULATION PRIORS: gamma from a mixture with mean ~1.0 and support [0, 2.2] — predominantly below 2 with only a 5-10% sliver in (2.0, 2.2] (no mass above 2.2, so Exp 1 stays safe); beta log-uniform on [1.6, 6.0] (floor slightly BELOW RSD's 1.8, since Exps 13/14 hint at lower-precision subjects than either theory admits); m in [0.76, 0.98] and epsilon in [0.03, 0.12], keeping mean A_eff ~0.65-0.70 to preserve the good fits on Exps 9 and 10. (iii) LOW-PRECISION SUBPOPULATION: 10-15% of subjects with beta near the floor and elevated epsilon (0.15-0.25), absorbing the shallow-slope / GSP-ward drift seen in Experiments 13 (predicted ~45-60 vs observed 52.6, fixing RSD's 96.7 over-prediction) and 14 (predicted ~-0.38 to -0.45 vs observed -0.414). Because the per-trial core is unchanged, Experiments 3-10 and 12 should remain well fit; the new degrees of freedom (delta, gamma jitter width, low-precision mixture weight) are each pinned by a currently-missed experiment (2, 13/14, 11 respectively), giving the new theory falsifiable signatures distinct from both pi_7 (hard gamma <= 2, noise-free) and pi_8 (gamma mass in (2.0, 2.4], beta floor 2.0): per-subject psychometric functions with shallower mid-slope than their saturated asymptote implies, elevated trial-to-trial inconsistency on repeated identical trials beyond binomial noise, and a positive-but-attenuated Exp 1 log Bayes factor.


## CANDIDATE THEORY
Stochastic-Gate Anti-Endorsement Integration, revision 4 (SGAI-r4). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma_t (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2. The population is bimodal in precision: a dominant high-precision cluster (75%; beta log-uniform on [2.0, 6.0]) whose habitual validity exponent sits at a calibrated midpoint between the iter-2 and iter-3 distributions — 20% in [0.8, 1.3], 48% in [1.3, 1.7], 24% in [1.7, 2.0], and an 8% sliver in (2.0, 2.2], none above 2.2 — plus a 25% low-precision cluster (beta log-uniform on [1.2, 1.8], habitual gamma uniform on [0, 1.0], elevated lapse 0.15-0.25) that lands genuinely inside the GSP box. On each trial, with probability delta in [0.10, 0.13], the subject re-samples cue attention with a per-trial exponent drawn from a symmetric distribution of half-width 0.4 around their OWN habitual gamma (gamma_t ~ U[max(0, gamma-0.4), min(2.2, gamma+0.4)]). Falsifiable signatures: psychometric mid-slopes shallower than saturated asymptotes imply, supra-binomial trial-to-trial inconsistency on repeated identical trials, a bimodal precision population straddling the RSD/GSP beta boundary, and per-subject gamma posteriors that spill above 2.0 without any habitual mass above 2.2.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Stochastic-Gate Anti-Endorsement Integration, revision 4 (SGAI-r4).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SGAI-r4 expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Population: habitual validity exponent gamma (majority) ----
    # gamma_u ~ U[0, 1]. MIDPOINT correction between iter-2 and iter-3
    # distributions (critic iter 3, advice (a)+(c)): 20% in [0.8, 1.3],
    # 48% in [1.3, 1.7], 24% in [1.7, 2.0], and an 8% sliver in
    # (2.0, 2.2] (raised from 5% toward the arbiter's 10% ceiling for
    # direct Exp-14 tail mass). NO habitual mass above 2.2 (guardrail
    # for Exp 13). Iter 2's distribution gave Exp 11 = 34.5, iter 3's
    # gave 109.3; the observed 52.6 sits between, closer to iter 2's
    # side, so this midpoint pulls the high-gamma mass back while the
    # enlarged sliver keeps feeding Exp 14.
    u = float(parameters["gamma_u"])
    if u < 0.20:
        gamma = 0.8 + 0.5 * (u / 0.20)
    elif u < 0.68:
        gamma = 1.3 + 0.4 * ((u - 0.20) / 0.48)
    elif u < 0.92:
        gamma = 1.7 + 0.3 * ((u - 0.68) / 0.24)
    else:
        gamma = 2.0 + 0.2 * ((u - 0.92) / 0.08)

    # ---- Precision subpopulation (bimodal) ----
    # 25% of subjects are low-precision (enlarged from 20% per the
    # critic): beta log-uniform on [1.2, 1.8] (DEEP inside the GSP box,
    # below the RSD floor of 1.8), elevated lapse epsilon in
    # [0.15, 0.25], habitual gamma forced LOW (uniform on [0, 1.0]).
    # The remaining 75% draw beta log-uniformly over [2.0, 6.0] with
    # a small lapse.
    if float(parameters["prec_u"]) < 0.25:
        beta = float(np.exp(float(parameters["lp_log_beta"])))
        epsilon = float(parameters["lp_epsilon"])
        gamma = 1.0 * u  # low-precision subjects keep gamma <= 1.0
    else:
        beta = float(np.exp(float(parameters["log_beta"])))
        epsilon = float(parameters["epsilon"])

    m = float(parameters["m"])
    delta = float(parameters["delta"])

    # ---- Trial-level stochastic attention gate (RELATIVE draw) ----
    # With probability delta (bottom of the range, [0.10, 0.13], per
    # the critic's Exp-10 guardrail) the subject re-samples cue
    # attention for THIS trial, drawing an effective exponent from a
    # SYMMETRIC distribution of half-width 0.4 (narrowed from 0.5)
    # AROUND THEIR OWN habitual gamma:
    #     gamma_t ~ U[max(0, gamma - 0.4), min(2.2, gamma + 0.4)].
    # The draw is relative, so the low-precision cluster's shallow
    # gamma is never globally overridden, while steep-gamma subjects
    # still jitter across the 2.0 boundary on a minority of trials.
    if np.random.random() < delta:
        lo = max(0.0, gamma - 0.4)
        hi = min(2.2, gamma + 0.4)
        gamma_t = lo + (hi - lo) * np.random.random()
    else:
        gamma_t = gamma

    # RAW validity weighting: w_j = v_j^gamma_t, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features.
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
    # (1 - m) with the conventional polarity.
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
- log_beta: [0.693, 1.792]
- epsilon: [0.03, 0.10]
- m: [0.78, 0.98]
- delta: [0.10, 0.13]
- prec_u: [0, 1]
- lp_log_beta: [0.182, 0.588]
- lp_epsilon: [0.15, 0.25]
- validities: validities

`rationale`:
Minimal-diff edit of the ACCEPTED SGAI-r3 base, applying the critic's four midpoint recalibrations while leaving the per-trial likelihood core (anti-endorsement reading, raw v^gamma_t weights, m/eps polarity-lapse shell) untouched. (a) MAJORITY HABITUAL GAMMA PULLED BACK HALFWAY: the mixture is now 20% [0.8, 1.3], 48% [1.3, 1.7], 24% [1.7, 2.0], 8% (2.0, 2.2], none above 2.2. Iter 2's shallower distribution gave Exp 11 = 34.5; iter 3's steeper one gave 109.3; the observed 52.6 sits between, closer to iter 2, so this midpoint should land Exp 11 in the critic's [45, 75] success band. Pulling the [1.7, 2.0] mass from 30% to 24% should also partially recover Exp 5 (0.544 -> ~0.57+) and stop the Exp 1/2 undershoot from deepening. (b) LOW-PRECISION CLUSTER ENLARGED FROM 20% TO 25%, kept DEEP (beta log-uniform [1.2, 1.8], habitual gamma <= 1.0, epsilon 0.15-0.25). This is the critic's single lever moving Exps 9, 11, 12, and 14 in their observed directions simultaneously: each genuinely GSP-box subject contributes a negative per-subject log BF to Exp 11 (pulling 109 toward ~50-70 without touching the majority), pushes Exp 12 from -0.31 toward the observed -0.414, pushes Exp 9 from -0.97 toward -1.5 or lower, and — because their flat psychometrics yield diffuse per-subject gamma posteriors that spread mass across the whole grid including above 2.0 — adds Exp 14 tail mass. (c) THE (2.0, 2.2] SLIVER RAISED FROM 5% TO 8% (within the arbiter's stated 10% ceiling) for direct Exp 14 tail mass; combined with (b) this targets Exp 14 >= 0.20 while Exp 13 (currently 4.64, band +3..+7, observed 6.03) has room to absorb a small movement. (d) EXP 10 GUARDRAIL: delta narrowed to the bottom of its range [0.10, 0.13] and the relative gate jitter half-width narrowed from 0.5 to 0.4, reducing the within-subject psychometric flattening that dropped Exp 10 from 0.892 to 0.842; the residual ~0.05 gap is accepted as the price of the low-precision cluster, which the observed data demands (high slope-identification 0.911 AND a GSP-leaning subgroup, Exp 12 = -0.414). Mean A_eff is unchanged (m [0.78, 0.98], epsilon [0.03, 0.10]) since Exp 7 at 0.681 vs 0.704 is within tolerance. All guardrails respected: no habitual mass above gamma 2.2, Exp 13 expected to stay in +3..+7, Exp 10 expected >= 0.85.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0945 -> ACCEPTED
- iter 2: loss=0.0893 -> ACCEPTED
- iter 3: loss=0.0802 -> ACCEPTED
- iter 4 (current candidate you are grading): loss=0.0808 -> REJECTED
Running-best (last accepted) base: iter 3 at loss=0.0802.

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
  - iter 1: 0.1743 (var=0.0074) (Δ vs real +0.0163)
  - iter 2: 0.1363 (var=0.0062) (Δ vs real -0.0217)
  - iter 3: 0.1420 (var=0.0048) (Δ vs real -0.0160)
  - iter 4 (current): 0.1297 (var=0.0067) (Δ vs real -0.0283)
**Other theories' values on this metric (for reference):**
- pi_1: 0.3480 (var=0.0087)
- pi_2: -0.2963 (var=0.0068)
- pi_3: 0.2473 (var=0.0026)
- pi_4: 0.1530 (var=0.0069)
- pi_5: 0.2150 (var=0.0067)
- pi_6: 0.1693 (var=0.0082)
- pi_7: 0.1660 (var=0.0056)
- pi_8: 0.1537 (var=0.0082)

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
  - iter 1: 0.1757 (var=0.0103) (Δ vs real -0.0057)
  - iter 2: 0.1550 (var=0.0084) (Δ vs real -0.0264)
  - iter 3: 0.1579 (var=0.0091) (Δ vs real -0.0236)
  - iter 4 (current): 0.1750 (var=0.0079) (Δ vs real -0.0064)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8664 (var=0.0110)
- pi_1: 0.1536 (var=0.0134)
- pi_3: 0.2557 (var=0.0053)
- pi_4: 0.2579 (var=0.0137)
- pi_5: 0.1479 (var=0.0061)
- pi_6: 0.1793 (var=0.0080)
- pi_7: 0.1764 (var=0.0097)
- pi_8: 0.1571 (var=0.0041)

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
  - iter 1: -0.6417 (var=0.0973) (Δ vs real +0.0750)
  - iter 2: -0.6500 (var=0.0639) (Δ vs real +0.0667)
  - iter 3: -0.7183 (var=0.0700) (Δ vs real -0.0017)
  - iter 4 (current): -0.7267 (var=0.0372) (Δ vs real -0.0100)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0033 (var=0.0461)
- pi_3: 1.0483 (var=0.0759)
- pi_2: 0.7083 (var=0.0512)
- pi_4: -0.5050 (var=0.1068)
- pi_5: -0.7033 (var=0.1239)
- pi_6: -0.6950 (var=0.0763)
- pi_7: -0.7067 (var=0.0731)
- pi_8: -0.7200 (var=0.0585)

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
  - iter 1: -0.5176 (var=0.0245) (Δ vs real +0.0055)
  - iter 2: -0.5486 (var=0.0230) (Δ vs real -0.0256)
  - iter 3: -0.5549 (var=0.0288) (Δ vs real -0.0318)
  - iter 4 (current): -0.5053 (var=0.0282) (Δ vs real +0.0178)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5548 (var=0.0148)
- pi_1: 0.0081 (var=0.0061)
- pi_2: 0.5897 (var=0.0315)
- pi_4: -0.4252 (var=0.0349)
- pi_5: -0.5389 (var=0.0255)
- pi_6: -0.5237 (var=0.0265)
- pi_7: -0.5575 (var=0.0274)
- pi_8: -0.5635 (var=0.0198)

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
  - iter 1: 0.5826 (var=0.0166) (Δ vs real -0.0118)
  - iter 2: 0.5759 (var=0.0190) (Δ vs real -0.0185)
  - iter 3: 0.5444 (var=0.0276) (Δ vs real -0.0500)
  - iter 4 (current): 0.5577 (var=0.0199) (Δ vs real -0.0367)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0009 (var=0.0089)
- pi_4: 0.4379 (var=0.0326)
- pi_2: -0.6876 (var=0.0238)
- pi_3: 0.0089 (var=0.0056)
- pi_5: 0.5894 (var=0.0647)
- pi_6: 0.5414 (var=0.0227)
- pi_7: 0.5915 (var=0.0172)
- pi_8: 0.5704 (var=0.0179)

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
  - iter 1: -0.6983 (var=0.0346) (Δ vs real -0.0225)
  - iter 2: -0.6946 (var=0.0207) (Δ vs real -0.0188)
  - iter 3: -0.7200 (var=0.0202) (Δ vs real -0.0442)
  - iter 4 (current): -0.7004 (var=0.0233) (Δ vs real -0.0246)
**Other theories' values on this metric (for reference):**
- pi_4: -0.5012 (var=0.0594)
- pi_1: 0.0138 (var=0.0135)
- pi_2: 0.7354 (var=0.0335)
- pi_3: -0.0125 (var=0.0078)
- pi_5: -0.6912 (var=0.1205)
- pi_6: -0.6900 (var=0.0330)
- pi_7: -0.6850 (var=0.0317)
- pi_8: -0.6883 (var=0.0128)

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
  - iter 1: 0.6444 (var=0.0238) (Δ vs real -0.0600)
  - iter 2: 0.7222 (var=0.0235) (Δ vs real +0.0178)
  - iter 3: 0.6811 (var=0.0253) (Δ vs real -0.0233)
  - iter 4 (current): 0.7089 (var=0.0274) (Δ vs real +0.0044)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7611 (var=0.0118)
- pi_4: 0.5167 (var=0.0365)
- pi_1: 0.6944 (var=0.0505)
- pi_2: 0.7267 (var=0.0418)
- pi_3: 0.6067 (var=0.0143)
- pi_6: 0.7544 (var=0.0264)
- pi_7: 0.7111 (var=0.0259)
- pi_8: 0.7178 (var=0.0239)

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
  - iter 1: 0.3300 (var=0.0047) (Δ vs real -0.0167)
  - iter 2: 0.3308 (var=0.0065) (Δ vs real -0.0158)
  - iter 3: 0.3542 (var=0.0062) (Δ vs real +0.0075)
  - iter 4 (current): 0.3279 (var=0.0068) (Δ vs real -0.0187)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2367 (var=0.0084)
- pi_5: 0.3758 (var=0.0028)
- pi_1: 0.3283 (var=0.0115)
- pi_2: 0.3421 (var=0.0094)
- pi_3: 0.1929 (var=0.0036)
- pi_6: 0.3467 (var=0.0078)
- pi_7: 0.3521 (var=0.0065)
- pi_8: 0.3438 (var=0.0047)

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
  - iter 1: -0.7799 (var=1.9369) (Δ vs real +0.9833)
  - iter 2: -0.5997 (var=1.4966) (Δ vs real +1.1636)
  - iter 3: -0.9729 (var=2.3275) (Δ vs real +0.7903)
  - iter 4 (current): -0.9791 (var=1.8786) (Δ vs real +0.7841)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6498 (var=1.2048)
- pi_6: -0.6480 (var=2.0074)
- pi_1: -2.7859 (var=6.5552)
- pi_2: -1.5565 (var=3.7090)
- pi_3: -1.7169 (var=0.8671)
- pi_4: -1.4521 (var=2.5200)
- pi_7: -1.3031 (var=2.4039)
- pi_8: -0.6950 (var=1.2398)

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
  - iter 1: 0.8920 (var=0.0248) (Δ vs real -0.0193)
  - iter 2: 0.8539 (var=0.0408) (Δ vs real -0.0574)
  - iter 3: 0.8422 (var=0.0362) (Δ vs real -0.0691)
  - iter 4 (current): 0.8139 (var=0.0549) (Δ vs real -0.0974)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7849 (var=0.0519)
- pi_5: 0.5780 (var=0.0892)
- pi_1: 0.9526 (var=0.0112)
- pi_2: 0.9697 (var=0.0075)
- pi_3: 0.5167 (var=0.0397)
- pi_4: 0.6639 (var=0.0486)
- pi_7: 0.9161 (var=0.0235)
- pi_8: 0.8983 (var=0.0297)

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
  - iter 1: 30.7611 (var=3.5302) (Δ vs real -21.8373)
  - iter 2: 34.5122 (var=5.6399) (Δ vs real -18.0862)
  - iter 3: 109.3492 (var=7.6061) (Δ vs real +56.7508)
  - iter 4 (current): 58.8717 (var=7.7056) (Δ vs real +6.2733)
**Other theories' values on this metric (for reference):**
- pi_7: 96.6930 (var=15.9176)
- pi_6: -21.1257 (var=0.4647)
- pi_1: -150.7700 (var=0.2496)
- pi_2: 220.1524 (var=4.1508)
- pi_3: -87.8944 (var=0.2065)
- pi_4: 18.4709 (var=6.0537)
- pi_5: -41.9588 (var=1.0440)
- pi_8: 67.3084 (var=8.4566)

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
  - iter 1: -0.0564 (var=0.3984) (Δ vs real +0.3580)
  - iter 2: -0.2105 (var=0.4711) (Δ vs real +0.2039)
  - iter 3: -0.3105 (var=0.4005) (Δ vs real +0.1040)
  - iter 4 (current): -0.1558 (var=0.4879) (Δ vs real +0.2586)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2650 (var=0.2313)
- pi_7: -0.2520 (var=0.3854)
- pi_1: 0.7899 (var=0.0150)
- pi_2: -0.9161 (var=0.0663)
- pi_3: 0.7753 (var=0.0098)
- pi_4: 0.0191 (var=0.3421)
- pi_5: 0.5030 (var=0.2038)
- pi_8: -0.3417 (var=0.3534)

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
  - iter 1: 4.1711 (var=0.5507) (Δ vs real -1.8576)
  - iter 2: 10.4280 (var=0.3365) (Δ vs real +4.3994)
  - iter 3: 4.6351 (var=0.3723) (Δ vs real -1.3935)
  - iter 4 (current): 4.8981 (var=0.4358) (Δ vs real -1.1305)
**Other theories' values on this metric (for reference):**
- pi_7: 5.3165 (var=0.3027)
- pi_8: -30.9335 (var=3.3944)
- pi_1: 136.1063 (var=0.3030)
- pi_2: -106.1209 (var=4.0607)
- pi_3: 135.2209 (var=0.2267)
- pi_4: 38.8942 (var=0.7056)
- pi_5: 7.2875 (var=0.2574)
- pi_6: 15.9085 (var=0.2273)

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
  - iter 1: 0.1015 (var=0.0001) (Δ vs real -0.1968)
  - iter 2: 0.0983 (var=0.0001) (Δ vs real -0.2000)
  - iter 3: 0.1360 (var=0.0002) (Δ vs real -0.1623)
  - iter 4 (current): 0.1030 (var=0.0001) (Δ vs real -0.1953)
**Other theories' values on this metric (for reference):**
- pi_8: 0.1396 (var=0.0003)
- pi_7: 0.0990 (var=0.0001)
- pi_1: 0.0175 (var=0.0000)
- pi_2: 0.7299 (var=0.0001)
- pi_3: 0.0173 (var=0.0000)
- pi_4: 0.1168 (var=0.0002)
- pi_5: 0.0871 (var=0.0002)
- pi_6: 0.0638 (var=0.0001)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: SGAI was accepted by the loss gate, and it preserves the core anti-endorsement/raw-weight fits well: Experiments 1-6, 8, 10 are all essentially on top of the observed values (e.g., Exp 1: 0.174 vs 0.158; Exp 4: -0.518 vs -0.523; Exp 5: 0.583 vs 0.594; Exp 6: -0.698 vs -0.676; Exp 10: 0.892 vs 0.911). Exp 13 also landed in the arbiter's prescribed band (4.17 vs observed 6.03, target +3 to +7), repairing pi_8's -31 catastrophe. However, the candidate fails on exactly the three experiments the SGAI revisions were built to fix, and the failures are diagnostic. (1) Exp 14 (tail mass): 0.1015 vs observed 0.2983 — statistically indistinguishable from pi_7's 0.0990. The trial-level gate as implemented (delta ~0.17, gamma_t redrawn UNIFORMLY over [0, 2.2]) does not inflate the per-subject posterior mass above gamma = 2. The reason is mechanistic: a uniform redraw mostly dilutes the habitual signal toward a shallow, near-linear component, and the Exp 14 measurement model absorbs that dilution by lowering its free A_eff (grid down to 0.35) — not by shifting posterior gamma upward. To make subjects LOOK high-gamma, the gated trials must mimic steep-validity-weighting behavior, i.e., the re-draw must be biased toward the upper end of the support, not uniform. (2) Exp 12 (tanh GSP-vs-RSD BF): -0.056 vs observed -0.414 (pi_8 achieved -0.342). The 13% low-precision subpopulation with beta in [1.6, 2.0] is too small and sits too close to the RSD beta floor (1.8) to pull the bounded BF meaningfully GSP-ward; most of its beta support [1.8, 2.0] is INSIDE the RSD box, so these subjects are scored as weak RSD, not GSP. (3) Exp 11 (RSD-vs-GSP joint log BF): 30.8 vs observed 52.6 — the majority population is not confidently RSD-like enough. Note the resolution of the apparent Exp 11 / Exp 12 tension is bimodality: a dominant cluster firmly inside the RSD box (beta well above 1.8) plus a somewhat larger, DEEPER low-precision cluster (beta down to ~1.3-1.5, gamma <= 1.2) that lands inside the GSP box. The current 13% at [1.6, 2.0] straddles the boundary and captures neither. Secondary misses: Exp 9 (-0.78 vs -1.76; the added noise pulls the per-subject RFI-vs-GSP BF toward 0, whereas enlarging/deepening the GSP-like cluster would push it negative as observed) and Exp 7 (0.644 vs 0.704; mean A_eff ~0.67 is slightly too attenuated — both Exp 7 and Exp 8 are under, so a small increase in effective attenuation helps both).
Rationale: Regenerate, staying strictly inside the SGAI family (anti-endorsement core, raw v^gamma_t weights, m/eps polarity-lapse shell, stochastic gate, low-precision subpopulation) with these in-family recalibrations: (a) RESTRUCTURE THE GATE DRAW, not just its rate: on gated trials draw gamma_t from an upper-biased distribution over the support — e.g., gamma_t ~ U[1.2, 2.2] or a Beta-shaped density concentrated near the top of [0, 2.2] (optionally with a small point mass at gamma_t = 0 for raw cue counting). Uniform redraws are absorbed as A_eff attenuation; upper-biased redraws produce the steep-validity choice patterns that the Exp 14 measurement model attributes to gamma > 2, which is what the observed 0.2983 requires. Consider also raising delta toward [0.25, 0.40] — the sign-invariant contrast metrics (Exps 1-6) and the saturated-extremity metrics (Exps 7/8) are largely insensitive to gamma-jitter on high-margin trials, and Exp 10 has headroom (0.892 vs 0.911). (b) DEEPEN AND SLIGHTLY ENLARGE the low-precision cluster: 15-18% of subjects with beta log-uniform on roughly [1.3, 2.0] (below the RSD floor so they genuinely land in the GSP box) and epsilon 0.15-0.25; keep their gamma low (<= 1.2). This pushes Exp 12 toward -0.35 to -0.45 and Exp 9 toward -1.5 or lower, matching observation. (c) STRENGTHEN THE MAJORITY CLUSTER'S RSD CREDENTIALS to lift Exp 11 from ~31 toward ~50 and protect Exp 13 as the gate draw moves upward: shift the main beta log-uniform range up (e.g., [2.0, 6.0]) and concentrate habitual gamma mass in [0.8, 1.8] with the (2.0, 2.2] sliver reduced to ~5% or kept only via the gate. (d) NUDGE mean A_eff from ~0.67 to ~0.70 (raise the m floor slightly or lower the eps ceiling) to close the small undershoots on Exps 7 and 8. Monitor Exp 13 (keep the log BF positive, ideally 4-7) and Exp 10 (keep the slope posterior ~0.89-0.92) as the guardrails; if the upper-biased gate drags Exp 13 below ~+3, trade off by lowering delta rather than re-flattening the draw. Do not place habitual mass above gamma = 2.2 — that is what flipped pi_8 to -31 on Exp 13.

**Outcome of your advice:** iter 1 candidate loss=0.0945 -> iter 2 candidate loss=0.0893 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: SGAI-r2 was accepted by the loss gate (0.0945 -> 0.0893) and several of the in-family recalibrations worked exactly as intended: Exp 7 improved to 0.722 vs 0.704 (the A_eff nudge), Exp 11 rose from 30.8 to 34.5, and Exp 12 moved from -0.056 to -0.21 — both in the right direction. However, the iteration's PRIMARY objective failed, and two guardrails slipped. (1) EXP 14 DID NOT MOVE: 0.0983 vs observed 0.2983, essentially unchanged from iter 1's 0.1015 and indistinguishable from pi_7's 0.0990. The upper-biased gate draw (gamma_t ~ U[1.2, 2.2] at delta 0.25-0.40) was implemented faithfully and had NO effect on the estimated tail mass. This is now the second confirmation of a mechanistic fact: per-trial gamma jitter — whatever its shape — cannot inflate the Exp 14 posterior mass above gamma = 2, because the measurement model's free A_eff (grid down to 0.35) absorbs mixture noise as attenuation rather than as steepness. The gate is a dead lever for Exp 14. (2) THE GATE IS NOW ACTIVELY COSTLY: Exp 9 went the WRONG way (-0.78 -> -0.60 vs observed -1.76) and Exp 10 dropped (0.892 -> 0.854 vs 0.911) — within-subject jitter flattens exactly the per-subject psychometric identification these two metrics rely on. (3) A STRUCTURAL BUG-LIKE INTERACTION: the gate draw U[1.2, 2.2] applies to LOW-PRECISION subjects too, overriding their habitual gamma <= 1.2 on 25-40% of trials. This makes the supposedly GSP-box cluster behave steeply on a third of its trials, diluting the bimodal population structure — likely why Exp 12 only reached -0.21 instead of the predicted -0.35..-0.45, and why Exp 9 did not go more negative. The arbiter's original spec was a draw 'around their mean gamma', not a global upper-biased draw. (4) Exp 13 overshot the guardrail band (+10.4 vs observed 6.03, target +3..+7): the majority cluster is now slightly TOO RSD-like. (5) Remaining misses: Exp 11 (34.5 vs 52.6), Exp 12 (-0.21 vs -0.414), Exp 9 (-0.60 vs -1.76), Exp 14 (0.098 vs 0.298) — all population-level. Exps 1-6, 8 remain essentially on top of the data.
Rationale: Regenerate, staying strictly inside the SGAI family (anti-endorsement core, raw v^gamma_t weights, m/eps shell, stochastic gate, bimodal precision population), with these recalibrations: (a) MAKE THE GATE DRAW RELATIVE TO THE SUBJECT'S OWN HABITUAL GAMMA, per the arbiter's original spec ('a wide distribution around their mean gamma'): e.g., gamma_t ~ U[gamma, min(gamma + 1.0, 2.2)] or a symmetric wide draw centered on gamma. The current global U[1.2, 2.2] draw overrides the low-precision cluster's shallow gamma on a third of its trials, destroying the bimodality that Exps 9 and 12 measure; a relative draw restores the GSP cluster's per-subject credentials at zero cost elsewhere. (b) LOWER DELTA back toward the arbiter's original [0.10, 0.20]. Two iterations have now shown the gate cannot deliver Exp 14's tail mass (free A_eff absorbs the jitter) while it demonstrably degrades Exp 9 (-0.60 vs -1.76) and Exp 10 (0.854 vs 0.911); a gentler gate recovers both. (c) REDISTRIBUTE HABITUAL GAMMA UPWARD IN THE MAJORITY CLUSTER as the NEW primary lever for Exp 14: concentrate mass in roughly [1.3, 2.0] with mean ~1.6-1.7 (e.g., 15% in [0.8, 1.3], 50% in [1.3, 1.7], 30% in [1.7, 2.0], keep the 5% sliver in (2.0, 2.2], none above 2.2). Subjects with true gamma near 1.8-2.0 have per-subject posteriors that SPILL above 2.0, which is the only in-family route to the observed 0.298 tail mass — and it simultaneously pushes Exp 11 up toward 52.6, pulls Exp 13 back from +10.4 into the +3..+7 band, steepens Exp 10 back toward 0.91, pushes Exp 9 negative, and nudges Exps 1/2 up toward 0.158/0.181 (both currently slightly under). (d) DEEPEN THE LOW-PRECISION CLUSTER modestly (to ~20%, beta log-uniform on [1.2, 1.8], gamma <= 1.0, epsilon 0.15-0.25): with fix (a) in place these subjects now stay genuinely shallow on gated trials, so each one contributes much more GSP pull; this should move Exp 12 from -0.21 toward -0.41 and Exp 9 toward -1.5 without needing a large cluster that would sink Exp 11. Guardrails: keep Exp 13 in +3..+7 (the gamma shift should reduce it — verify it does not undershoot), keep Exp 10 in 0.89-0.92, and do NOT raise mean A_eff further (Exp 7 at 0.722 is already slightly over the observed 0.704). Do not place habitual mass above gamma = 2.2.

**Outcome of your advice:** iter 2 candidate loss=0.0893 -> iter 3 candidate loss=0.0802 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent)
Verdict: regenerate
Interpretation: SGAI-r3 was accepted by the loss gate (0.0893 -> 0.0802) and the recalibrations that worked are clear: Exp 13 is back inside the guardrail band (4.64 vs observed 6.03, target +3..+7), Exp 12 continues its monotone march toward the target (-0.056 -> -0.21 -> -0.31 vs -0.414), Exp 9 improved (-0.60 -> -0.97 vs -1.76), Exp 3 is essentially exact (-0.718 vs -0.717), and Exps 4, 6, 8 remain close. The relative gate draw also fixed the bimodality-destroying interaction diagnosed in iter 2. However, the iteration's PRIMARY lever overshot badly on its main target while barely moving its intended one. (1) EXP 11 OVERSHOT MASSIVELY: 109.3 vs observed 52.6 (var 6.5) — worse than pi_7's 96.7 and roughly triple the iter-2 value of 34.5. The habitual-gamma redistribution (35% of the majority at 1.7-2.2) made the population too strongly RSD-like on the joint log Bayes factor. My iter-2 advice to push gamma up was directionally right but the step size was roughly twice too large; the correct setting sits between the iter-2 and iter-3 distributions. (2) EXP 14 BARELY MOVED: 0.136 vs observed 0.298 (var 0.0006 — the miss is ~40 between-subject SDs). The 'posterior spill' from gamma 1.7-2.0 subjects plus the 5% sliver delivered only +0.038 over iter 2. Combined with iter 2's null result on the gate, this confirms that neither per-trial jitter nor modest spill can close the gap; the remaining in-family mass must come from (a) a larger (2.0, 2.2] sliver (the arbiter explicitly allows up to 10%; currently only 5%), and (b) more low-precision subjects whose diffuse per-subject posteriors spread mass across the gamma grid including above 2.0. (3) EXP 10 SLIPPED BELOW GUARDRAIL: 0.892 -> 0.854 -> 0.842 vs observed 0.911 — the fingerprint of the enlarged low-precision cluster (their per-subject slope posteriors sit well below the majority's). (4) EXP 5 DEGRADED (0.583 -> 0.544 vs 0.594), likely also from the upward gamma shift. (5) Small persistent undershoots on Exps 1/2 (0.142/0.158 vs 0.158/0.181). Net: the candidate is the best base so far, but the population layer is now mis-calibrated in a well-identified direction: too much high-gamma majority mass (Exp 11 +57) and too little tail-generating mass (Exp 14 -0.16).
Rationale: Regenerate, staying strictly inside the SGAI family (anti-endorsement core, raw v^gamma_t weights, m/eps shell, relative stochastic gate, bimodal precision population), with a MIDPOINT correction — explicitly not an oscillation, but settings between the iter-2 and iter-3 values: (a) PULL THE MAJORITY HABITUAL GAMMA BACK HALFWAY: redistribute to roughly 20% [0.8, 1.3], 50% [1.3, 1.7], 25% [1.7, 2.0], 5% (2.0, 2.2]. Iter 2's distribution gave Exp 11 = 34.5, iter 3's gave 109.3; the observed 52.6 sits between, closer to iter 2's side. This should also partially recover Exp 5 (0.544 -> ~0.57+) and stop the Exp 1/2 undershoot from deepening. (b) ENLARGE THE LOW-PRECISION CLUSTER FROM 20% TO ~24-25%, keeping it DEEP (beta log-uniform [1.2, 1.8], habitual gamma <= 1.0, epsilon 0.15-0.25). Each genuinely GSP-box subject contributes a negative per-subject log BF to Exp 11 (pulling 109 toward ~50-70 without touching the majority), pushes Exp 12 from -0.31 toward -0.41, pushes Exp 9 from -0.97 toward -1.5 or lower, and — because their flat psychometrics yield diffuse gamma posteriors — adds tail mass above 2.0 for Exp 14. This is the single lever that moves Exps 9, 11, 12, and 14 in their observed directions simultaneously. (c) RAISE THE (2.0, 2.2] SLIVER FROM 5% TOWARD 8-10% (the arbiter's stated ceiling) for direct Exp 14 tail mass; monitor Exp 13, which has room to absorb a small decrease (currently 4.64, band +3..+7, observed 6.03). (d) EXP 10 GUARDRAIL: set delta to the bottom of its range ([0.10, 0.13]) and optionally narrow the relative gate jitter to +/-0.4; expect ~0.85, and accept a small residual gap (~0.05) as the price of the low-precision cluster — do not shrink the cluster to rescue Exp 10, since the observed data demands BOTH high slope-identification (0.911) AND a GSP-leaning subgroup (Exp 12 = -0.414), and the cluster is what delivers the latter. Guardrails: keep Exp 13 in +3..+7, keep Exp 10 >= 0.85, keep mean A_eff unchanged (Exp 7 at 0.681 vs 0.704 is fine), and place NO habitual mass above gamma = 2.2. Success criteria for the next candidate: Exp 11 in [45, 75], Exp 14 >= 0.20, Exp 12 <= -0.35, Exp 9 <= -1.3.

**Outcome of your advice:** iter 3 candidate loss=0.0802 -> CURRENT candidate loss=0.0808 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "SGAI-r4 was REJECTED by the gate (0.0808 vs 0.0802), and the per-experiment pattern shows a clean split between one validated recalibration and one failed one. WHAT WORKED: the majority-gamma midpoint pullback (20/48/24/8) landed Experiment 11 squarely in the target band (58.9 vs observed 52.6, my [45, 75] criterion) \u2014 this calibration is now confirmed and should be treated as the anchor. Exps 2, 4, 5, 6, 7 also improved or held (Exp 2: 0.175 vs 0.181; Exp 7: 0.709 vs 0.704; Exp 13 stayed in band at 4.90). WHAT FAILED: (1) Enlarging the low-precision cluster from 20% to 25% with elevated epsilon (0.15-0.25) was supposed to move Exps 12 and 14 and instead moved NEITHER \u2014 Exp 12 regressed from -0.31 to -0.156 and Exp 14 dropped from 0.136 to 0.103 \u2014 while it DID degrade Exp 10 further (0.842 -> 0.814, now below the 0.85 guardrail). The mechanism is now clear from two iterations of evidence: high-lapse GSP-box subjects supply almost no per-subject likelihood evidence, so they neither contribute sharp negative tanh to Exp 12 nor negative log BF to Exp 11; they only flatten the slope posteriors that Exp 10 measures. The Bayes-factor metrics (9, 11, 12) reward SHARP GSP-box subjects, not numerous diffuse ones. (2) Exp 12's regression is additionally explained by the majority pullback: with habitual gamma in [1.3, 1.7] and beta log-uniform on [2.0, 6.0] (median ~3.5), a large fraction of majority subjects now sit inside the GSP/RSD overlap (gamma <= 1.5, beta in [1.8, 3.2]) and contribute tanh ~ 0 instead of positive RSD evidence \u2014 the negative pull must therefore come from sharper GSP-box subjects, not from a bigger diffuse cluster. (3) Exp 14 remains the dominant miss (0.103 vs 0.298, var 0.0006 \u2014 the miss is many between-subject SDs). Three iterations have now confirmed the trial-level gate is a dead lever for tail mass (free A_eff down to 0.35 absorbs jitter as attenuation); the only lever that ever moved it is steep-majority spill (iter 3's 0.136). Note also that the observed value implies roughly 40% of real subjects show genuine gamma > 2 choice signatures (raw posterior mass ~0.41, above even a flat-diffuse 0.23) \u2014 a magnitude the arbiter's 2.2-cap / 10%-sliver guardrails structurally cannot deliver; the residual should be flagged to the arbiter rather than fixed by breaking the family. (4) Secondary: Exp 1 undershot further (0.130 vs 0.158) \u2014 steeper majority gamma raises TTB-siding and would help; Exp 9 is unchanged at -0.98 vs -1.76.",
  "verdict": "regenerate",
  "rationale": "Regenerate on the iter-3 accepted base, re-applying the validated iter-4 majority calibration plus one NEW lever \u2014 converting the low-precision cluster from diffuse to sharp. Stay strictly inside the SGAI family (anti-endorsement core, raw v^gamma_t weights, m/eps shell, relative stochastic gate, bimodal precision population). (a) MAJORITY HABITUAL GAMMA: keep the iter-4 distribution as the anchor but nudge the top bins up slightly \u2014 e.g., 18% [0.8, 1.3], 45% [1.3, 1.7], 28% [1.7, 2.0], 9% (2.0, 2.2], none above 2.2. This preserves Exp 11 near 60-75 while feeding Exp 14 spill, Exp 12, and Exp 1 (all of which improved under the steeper iter-3 majority). (b) PRIMARY NEW LEVER \u2014 SHARPEN THE LOW-PRECISION CLUSTER: shrink it to 20-22%, keep beta log-uniform [1.2, 1.8] and habitual gamma <= 1.0, but CUT epsilon from [0.15, 0.25] to roughly [0.05, 0.12]. A clean low-beta subject contributes tanh near -0.8/-0.9 on Exp 12 (vs ~-0.2 when lapse-heavy), a strongly negative per-subject log BF on Exp 11 (buying headroom for the steeper majority), pushes Exp 9 toward -1.3 or lower, and \u2014 because their effective normalized slope s = beta * sum(v^gamma) ~ 10-15 stays above the 9.5 threshold with clean data \u2014 no longer drags Exp 10 down (target >= 0.86). This one change moves Exps 9, 10, 11, and 12 in their observed directions simultaneously, unlike the diffuse-cluster enlargement which moved nothing and cost Exp 10. (c) EXP 14: raise the sliver to the arbiter's 10% ceiling and concentrate the third bin at [1.8, 2.0] to maximize spill; keep delta at the bottom of [0.10, 0.13] with the +/-0.4 relative jitter. Accept a residual gap (~0.13-0.17 vs 0.298) and state it explicitly in the rationale as a structural limit of the prescribed gamma <= 2.2 support \u2014 do NOT place habitual mass above 2.2 and do not re-widen the gate (both are confirmed dead or harmful levers). (d) GUARDRAILS: Exp 11 in [45, 75] \u2014 if the sharpened LP cluster pulls it below 45, rebalance by trimming LP to 18% or raising the sliver, not by re-flattening the majority; Exp 13 in [+3, +7] (monitor: sharp out-of-box subjects can shift it either way); Exp 10 >= 0.85; mean A_eff unchanged (Exp 7/8 are fine). Success criteria for the next candidate: Exp 12 <= -0.35, Exp 10 >= 0.86, Exp 9 <= -1.2, Exp 11 in [45, 75], Exp 14 >= 0.12, aggregate loss < 0.0802."
}
```

## Usage

```json
{
  "input_tokens": 47917,
  "output_tokens": 19489,
  "reasoning_tokens": 18010,
  "provider": "Sail Research"
}
```
