# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_3
**Absolute-Evidence Weighted Integration (log-odds^gamma diagnosticity weights, no divisive normalization).**

People integrate ALL stated expert ratings, but with diagnosticity weights derived from the *magnitude* of each expert's stated validity: d_i = log(v_i/(1-v_i)) (the normative evidence a binary cue carries), passed through a non-compensatoriness exponent gamma, w_i = d_i^gamma. Evidence for option A is E = sum_i w_i (a_i - b_i), converted to choice by a softmax with inverse temperature beta plus a lapse epsilon. The family nests both classical heuristics as limits: gamma -> 0 gives equal weights (Tallying); gamma = 1 gives naive-Bayes / Franklin's rule; gamma -> infinity collapses weight onto the single most valid discriminating expert (Take-The-Best).

The crucial claim is that evidence is accumulated in ABSOLUTE log-odds-derived units: the weight vector is NOT renormalised to sum to one, and decision noise (beta) is constant in those absolute units. Consequently the *total diagnosticity* of an environment, D = sum_i d_i^gamma, governs how decisively choices are made. Three consequences follow that no rank-based rule and no scale-invariant (normalised) weighted-additive rule can produce:

(1) Environment-dependent non-compensatoriness. A structurally identical conflict - top cue versus 4-5 opposing lower cues - is resolved MORE top-cue-consistently in an environment containing a dominant, highly valid expert (e.g. .93 alongside .80) than in a smoothly graded environment topping out at .90, because the former yields larger absolute margins. Humans show exactly this asymmetry (~81% top-cue agreement in the 6-cue .93 environment vs ~74% in the graded 5-cue one).

(2) Graded, sub-ceiling conflict agreement. In the graded environment the absolute margin for 'top cue vs the four lower cues' is small, so agreement is only ~0.62-0.69, while 'top two cues vs the bottom three' is near-deterministic (~1.0) - a within-family gradient in the weighted margin that TTB forbids and that produces sub-ceiling pooled agreement without a large lapse rate.

(3) Environment-dependent heterogeneity. Because the graded environment's pivotal margins sit near zero, individual differences in gamma translate into large between-subject differences in conflict agreement there, whereas the dominant-cue environment keeps every subject on the top-cue side - predicting larger between-subject variance in the graded environment specifically.

It further predicts near-deterministic choice on unanimous (all-cue) pairs, and near-chance responding on pairs decided only by a .55-validity expert, since those margins are small in absolute units.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Absolute-Evidence Weighted Integration.
    # Weights come from the *magnitudes* of the stated validities:
    #     d_i = log(v_i / (1 - v_i))          (log-odds diagnosticity)
    #     w_i  =  d_i ** gamma                 (gamma = non-compensatoriness)
    # gamma -> 0  : equal weights            == Tallying
    # gamma  = 1  : log-odds weights         == naive Bayes / Franklin's rule
    # gamma -> inf: top cue dominates        == Take-The-Best
    # Evidence  E = sum_i w_i * (a_i - b_i)  -> softmax(beta * E) -> lapse.
    # The weight vector is NOT renormalised to sum to 1.  Evidence lives in
    # absolute log-odds units, so environments with greater total
    # diagnosticity (sum_i d_i**gamma) yield larger margins and hence more
    # decisive, more top-cue-consistent choices.
    # History is ignored (no feedback is given in this domain).
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        # Flat vector: assume [A features..., B features...]
        if stim.shape[0] % 2 != 0:
            return np.ones(2) / 2.0
        half = stim.shape[0] // 2
        stim = np.vstack([stim[:half], stim[half:]])
    if stim.ndim != 2 or stim.shape[0] != 2:
        return np.ones(2) / 2.0

    n_features = int(stim.shape[1])
    if n_features == 0:
        return np.ones(2) / 2.0

    # --- validities -------------------------------------------------
    val = parameters.get("validities", None)
    try:
        v = np.asarray(val, dtype=float).ravel()
    except Exception:
        v = np.array([])
    if v.size != n_features or not np.all(np.isfinite(v)):
        v = np.linspace(0.90, 0.55, n_features)
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # --- diagnosticity weights (ABSOLUTE, un-normalised) ------------
    gamma = float(parameters["gamma"])
    d = np.log(v / (1.0 - v))              # >= 0, larger = more diagnostic
    d = np.maximum(d, 1e-12)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        w = np.power(d, gamma)
    w = np.where(np.isfinite(w), w, 0.0)
    tot = float(np.sum(w))
    if tot <= 0.0 or not np.isfinite(tot):
        w = np.ones(n_features)
    # NOTE: deliberately no `w = w / tot` here.  The absolute scale of
    # the weights is the theoretical content of this model.

    # --- weighted-additive evidence ---------------------------------
    a = stim[0]
    b = stim[1]
    E = float(np.dot(w, a - b))
    if not np.isfinite(E):
        E = 0.0

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over [beta*E, 0]  (== sigmoid(beta*E) for A)
    z = np.array([beta * E, 0.0], dtype=float)
    z = np.clip(z, -700.0, 700.0)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        p = np.ones(p.shape[0]) / p.shape[0]
    else:
        p = p / s
    return int(np.random.choice(len(p), p=p))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=6):** [0.72, 0.85, 0.55, 0.78, 0.6, 0.66]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 4: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 10: A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 11: A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 12: A=[1, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  trial 13: A=[0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  trial 14: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 15: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate Take-The-Best (one-reason, validity-ordered, all-or-none confidence) from Absolute-Evidence Weighted Integration (AEWI: unnormalised log-odds^gamma weights, softmax in absolute evidence units). Both are *weight-order* consistent, so a Tallying-style design will not separate them; what separates them is (a) the SIGN flip AEWI produces when several mid/low cues jointly outweigh the top cue in absolute log-odds units, and (b) the GRADED magnitude of choice consistency that AEWI predicts across pairs where TTB necessarily predicts one constant probability sigmoid(beta).

ENVIRONMENT: six experts with a deliberately COMPRESSED top (0.85) and a long tail (0.78, 0.72, 0.66, 0.60, 0.55), displayed in scrambled order [0.72, 0.85, 0.55, 0.78, 0.60, 0.66] so that no 'use the first-listed expert' strategy can mimic TTB and genuine validity re-sorting is required. Log-odds diagnosticities are d = (0.944, 1.735, 0.201, 1.266, 0.406, 0.663); at the theory's own gamma range (1.355-1.475) the AEWI weights are approx w(.85)=2.11-2.25, w(.78)=1.38-1.42, w(.72)=0.92-0.93, w(.66)=0.55-0.57, w(.60)=0.26-0.29, w(.55)=0.09-0.11. Crucially the top cue's weight is LESS than the sum of any two-or-more of the next cues, so AEWI reverses TTB's choice on most conflict trials, while TTB (which only reads validity RANK) is unaffected.

TRIAL FAMILIES (16 unique pairs, each mirrored across A/B so response bias cannot mimic either model; K = 6 repetitions, 96 trials):
(1) GRADED TOP-CUE CONFLICTS (pairs 1-8). The 0.85 expert favors one option while 1, 2, 3 or all 5 lower experts favor the other. TTB predicts the top-cue option with a CONSTANT probability sigmoid(beta) in every case. AEWI predicts a steep monotone collapse and a sign reversal: p(top-cue option) approx 0.83 (1 opposing), 0.43 (2 opposing), 0.20 (3 opposing), 0.10 (5 opposing). This single curve is the primary discriminator: it contains both an ordinal reversal and a within-family gradient that TTB forbids. The 2-opposing case sits essentially at the AEWI zero-margin point (0.41-0.46 across the gamma range), which is also where AEWI uniquely predicts maximal between-subject heterogeneity.
(2) TTB-DESCENT CONFLICT (pairs 9-10). The 0.85 cue ties, so TTB descends to 0.78 and picks that option; the remaining four cues (0.72, 0.66, 0.60, 0.55) all oppose. AEWI margin is -0.41 to -0.53, i.e. p(TTB option) approx 0.26-0.31. Opposite-sign prediction with the deciding cue no longer being the visually salient best expert.
(3) ABSOLUTE-MARGIN / CONFIDENCE SCALING (pairs 11-16), direction-free. Pairs 11-12: ONLY the 0.55 expert discriminates (all other cues tie) - TTB must be as decisive here as anywhere (sigmoid(beta)), AEWI predicts near chance (approx 0.55). Pairs 13-14: ONLY the 0.85 expert discriminates - AEWI approx 0.99. Pairs 15-16: unanimous 6-0 - AEWI approx 1.00. Because pairs 13-16 pin TTB's beta at a high value, TTB is then forced to predict that same high value on the 0.55-only pairs and on all conflict pairs; any observed monotone dependence of choice consistency on the absolute weighted margin is fatal for TTB and is exactly AEWI's core claim.

DIAGNOSTIC SUMMARY: TTB predicts one number (roughly constant agreement with the top discriminating cue) across pairs 1,3,5,7,9,11,13,15; AEWI predicts the ordered sequence approx (0.83, 0.43, 0.20, 0.10, 0.28, 0.55, 0.99, 1.00). No parameter setting of either model can imitate the other's pattern on this set.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Both models are compared on the SAME quantity: agreement with the Take-The-Best choice (the option favoured by the highest-validity discriminating expert). Trials are split, using the design's own structure, into (i) pairs where an absolute-evidence integrator's margin also strongly favours the TTB option (single-cue-discriminating .85 pairs, unanimous pairs, and the 1-opposing-cue conflicts) and (ii) pairs where an integrator's summed log-odds margin REVERSES the TTB choice (3-opposing, 5-opposing, and the TTB-descent pairs). Ambiguous near-zero-margin pairs are dropped. The metric is the within-subject difference in TTB-agreement between these two trial sets. Under Take-The-Best only the RANK of the first discriminating cue matters, so agreement is the same constant sigmoid(beta)-based value in both sets for every subject regardless of beta and epsilon: the difference is 0 up to binomial sampling noise (36 trials per set, sd ~0.10). Under AEWI the difference is large and positive (~0.94 vs ~0.19, i.e. ~0.75) and is insensitive to its gamma/beta ranges. Because the contrast is taken within subject, individual differences in noise/lapse cancel for TTB, giving a huge mean gap with tight per-subject variance in both theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    if len(data) == 0:
        return 0.0

    A = np.array([[int(x) for x in r] for r in data['option_a_ratings']], dtype=float)
    B = np.array([[int(x) for x in r] for r in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int)
    n_feat = A.shape[1]

    val = np.array([0.72, 0.85, 0.55, 0.78, 0.60, 0.66], dtype=float)
    if val.shape[0] != n_feat:
        val = np.linspace(0.90, 0.55, n_feat)
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-6)

    order = np.argsort(-val, kind='stable')

    # log-odds diagnosticity weights with a mid-range gamma (reference AEWI weights)
    d = np.log(val / (1.0 - val))
    w = np.power(np.maximum(d, 1e-12), 1.4)
    beta_ref = 2.0

    diff = A - B  # +1 favours A on that cue, -1 favours B

    # Take-The-Best winner per trial: first discriminating cue in validity order
    n = A.shape[0]
    ttb = np.full(n, -1, dtype=int)  # 0 = A, 1 = B, -1 = no discrimination
    undecided = np.ones(n, dtype=bool)
    for j in order:
        dj = diff[:, j]
        pickA = undecided & (dj > 0)
        pickB = undecided & (dj < 0)
        ttb[pickA] = 0
        ttb[pickB] = 1
        undecided = undecided & (dj == 0)

    valid = ttb >= 0
    if not np.any(valid):
        return 0.0

    # Signed absolute-evidence margin IN FAVOUR OF the TTB-chosen option
    E = diff.dot(w)
    sign_ttb = np.where(ttb == 0, 1.0, -1.0)
    margin = E * sign_ttb
    p_aewi = 1.0 / (1.0 + np.exp(-beta_ref * margin))

    agree = (resp == ttb).astype(float)

    high = valid & (p_aewi > 0.65)   # integration and one-reason agree strongly
    low = valid & (p_aewi < 0.35)    # integration reverses the one-reason choice

    if high.sum() == 0 or low.sum() == 0:
        return 0.0

    return float(agree[high].mean() - agree[low].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.0061 (var=0.0056)
- Predicted under pi_3 (simulated): 0.7472 (var=0.0055)
- Observed on real data: -0.6733 (var=0.0205)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=7):** [0.74, 0.9, 0.55, 0.7, 0.82, 0.62, 0.78]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0]
  trial 2: A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  trial 9: A=[0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  trial 10: A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  trial 14: A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate Take-The-Best (one-reason, validity-RANK only, all-or-none confidence) from Absolute-Evidence Weighted Integration (AEWI: unnormalised log-odds^gamma weights, softmax in absolute evidence units). Both models are weight-order consistent, so the separation must come from (a) SIGN reversals when two or three mid-validity cues jointly outweigh the top cue in absolute log-odds units, and (b) a GRADED confidence ladder across pairs where TTB is forced to predict a single constant probability sigmoid(beta).

ENVIRONMENT: seven experts, one clearly dominant cue (0.90) plus a dense mid band (0.82, 0.78, 0.74, 0.70) and two weak cues (0.62, 0.55), displayed in SCRAMBLED order [0.74, 0.90, 0.55, 0.70, 0.82, 0.62, 0.78] so no 'use the first-listed expert' rule can imitate TTB and genuine validity re-sorting is required. Log-odds diagnosticities d = (1.046, 2.197, 0.201, 0.847, 1.516, 0.489, 1.266). At the theory's gamma range (1.355-1.475) the AEWI weights are approx w(.90)=2.91-3.19, w(.82)=1.76-1.85, w(.78)=1.38-1.42, w(.74)=1.06-1.07, w(.70)=0.78-0.80, w(.62)=0.36-0.37, w(.55)=0.10-0.11. The band is tuned so that the top cue BEATS any single mid cue and both weak cues combined, is roughly TIED with {0.82+0.74}, is slightly BEATEN by {0.82+0.78}, and is decisively beaten by three mid cues. TTB, reading only rank, is blind to all of these distinctions.

TRIAL FAMILIES (16 unique pairs, every pair mirrored across A/B so response bias cannot mimic either model; K = 6, 96 trials). In every odd-numbered pair TTB chooses A, in every even-numbered pair TTB chooses B.
(1) GRADED TOP-CUE CONFLICTS, spanning the AEWI zero-margin crossing. Pairs 1-2: 0.90 vs {0.62, 0.55} -> AEWI p(top option) ~ 0.99 (agrees with TTB). Pairs 3-4: 0.90 vs {0.82, 0.74} -> AEWI ~ 0.54-0.63 (top side, but barely). Pairs 5-6: 0.90 vs {0.82, 0.78} -> AEWI ~ 0.39-0.47 (sign reversal, and exactly the zero-margin point where AEWI uniquely predicts maximal between-subject heterogeneity driven by gamma). Pairs 7-8: 0.90 vs {0.82, 0.78, 0.74} -> AEWI ~ 0.08 (strong reversal). Across these four conflict types TTB must predict ONE constant number; AEWI predicts a steep monotone collapse through a sign change (0.99, 0.58, 0.43, 0.08).
(2) TTB-DESCENT CONFLICT (pairs 9-10). The 0.90 cue ties (both options rated 1), so TTB descends to 0.82 and picks that option, while 0.78 and 0.74 jointly oppose. AEWI margin is negative (~-0.6 to -0.7), p(TTB option) ~ 0.21: an opposite-sign prediction where the deciding cue is no longer the salient best expert.
(3) ABSOLUTE-MARGIN CONFIDENCE LADDER (pairs 11-16), direction-free (both models pick the same option). Pairs 11-12: ONLY the 0.55 expert discriminates -> AEWI ~ 0.55 (near chance), TTB must be as decisive as anywhere. Pairs 13-14: ONLY the 0.70 expert discriminates -> AEWI ~ 0.83. Pairs 15-16: unanimous 7-0 -> AEWI ~ 1.00. Because the unanimous and 0.99-margin pairs pin TTB's beta at a high value, TTB is then committed to that same high value on the 0.55-only pairs and on every conflict pair; any observed monotone dependence of choice consistency on the absolute weighted margin (0.55 < 0.83 < 1.00) falsifies TTB and is precisely AEWI's core claim.

DIAGNOSTIC SUMMARY: on the odd (TTB-picks-A) pairs 1,3,5,7,9,11,13,15 TTB predicts the same constant p(A); AEWI predicts the ordered sequence approx (0.99, 0.58, 0.43, 0.08, 0.21, 0.55, 0.83, 1.00), including two sign reversals and a near-zero crossing. No parameter setting of either model can reproduce the other's pattern on this set, and the pairs 3-6 crossing region additionally tests AEWI's unique prediction of inflated between-subject variance where the absolute margin is near zero.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Take-The-Best is rank-based and all-or-none: whatever its beta and epsilon, its choice probability for the top-discriminating-cue option is one constant number on EVERY pair where some cue discriminates. Hence any within-subject difference in TTB-consistency between two trial families must be zero for TTB, no matter how noisy or lapsy an individual subject is - this makes the contrast immune to TTB's very wide beta/epsilon ranges, which would wreck any raw accuracy/consistency measure through between-subject variance.

The two families are selected mechanically from the stimuli (no trial-index hardcoding): (a) AGREEMENT pairs where the AEWI absolute log-odds margin is large (|E| >= 1.5) and points the same way as the top discriminating cue - these are the 0.90-vs-{0.62,0.55} pairs (|E|~2.5) and the unanimous 7-0 pairs (|E|~8.5); (b) CONFLICT pairs where the margin is substantial (|E| >= 0.5) but points AGAINST the top discriminating cue - these are 0.90 vs {0.82,0.78,0.74} (|E|~1.24) and the TTB-descent pairs 0.82 vs {0.78,0.74} (|E|~0.66). Each family contains 24 trials per subject (2 mirrored pair types x 6 reps x 2 A/B orders), so binomial noise per subject is small.

Predictions: AEWI ~0.99 - ~0.15 = ~0.84 with tight per-subject variance (gamma range is narrow, both families sit far from the zero-margin crossing, which I deliberately excluded by the |E| thresholds - pairs 3-6 and 11-14 are dropped precisely because they are the high-variance ones). TTB ~0.00 with only binomial spread (<=0.02 variance). The mirrored A/B design also cancels any side bias.

Source:
def metric(data: pd.DataFrame) -> float:
    """Within-subject contrast: TTB-consistency on large-margin AGREEMENT pairs
    minus TTB-consistency on large-margin CONFLICT pairs (where the weighted
    log-odds evidence points opposite to the single best discriminating cue).
    TTB must give the same choice probability in both sets (=> ~0);
    AEWI predicts ~0.99 vs ~0.15 (=> ~0.85)."""
    v = np.array([0.74, 0.90, 0.55, 0.70, 0.82, 0.62, 0.78], dtype=float)
    d = np.log(v / (1.0 - v))
    w = np.power(d, 1.4)
    order = list(np.argsort(-v, kind='stable'))

    agree = []
    conflict = []

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        if a.size != 7 or b.size != 7:
            continue
        diff = a - b
        E = float(np.dot(w, diff))

        winner = None
        for j in order:
            if diff[j] > 0:
                winner = 0
                break
            if diff[j] < 0:
                winner = 1
                break
        if winner is None:
            continue

        try:
            resp = int(r)
        except Exception:
            continue
        cons = 1.0 if resp == winner else 0.0

        sgn = 1.0 if winner == 0 else -1.0
        signed = E * sgn
        if signed > 0 and abs(E) >= 1.5:
            agree.append(cons)
        elif signed < 0 and abs(E) >= 0.5:
            conflict.append(cons)

    if len(agree) == 0 or len(conflict) == 0:
        return 0.0
    return float(np.mean(agree) - np.mean(conflict))


### RESULTS
- Predicted under pi_1 (simulated): 0.0167 (var=0.0117)
- Predicted under pi_3 (simulated): 0.8408 (var=0.0061)
- Observed on real data: -0.4100 (var=0.0833)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    a_all = data['option_a_ratings'].tolist()
    b_all = data['option_b_ratings'].tolist()
    resp = data['response'].tolist()

    hits = 0
    n = 0
    for a, b, r in zip(a_all, b_all, resp):
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # TTB winner: features are ordered by descending validity in this design,
        # so scan left to right for the first discriminating cue.
        ttb = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb is None:
            continue
        # Tally winner: count strict feature-wise wins.
        aw = float(np.sum(a > b))
        bw = float(np.sum(b > a))
        if aw == bw:
            continue
        tal = 0 if aw > bw else 1
        # Keep only trials where the two heuristics point to OPPOSITE options.
        if tal == ttb:
            continue
        n += 1
        try:
            rr = int(r)
        except Exception:
            continue
        if rr == ttb:
            hits += 1

    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.7367 (var=0.0369)
**Predicted under pi_1:** 0.8425 (var=0.0110)
**Predicted under pi_3:** 0.7625 (var=0.0042)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.60, 0.93, 0.55, 0.80, 0.68, 0.57], dtype=float)

    def ttb_winner(a, b, val):
        order = np.argsort(-val, kind="stable")
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        n = a.shape[0]
        val = validities if validities.shape[0] == n else np.linspace(0.9, 0.55, n)

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_win = 0
        elif margin < 0:
            tally_win = 1
        else:
            continue  # tally undecided: excluded

        tw = ttb_winner(a, b, val)
        if tw < 0 or tw == tally_win:
            continue  # only trials where the two rules point opposite ways
        if abs(margin) < 3:
            continue  # keep only large-margin conflicts (strong tally signal)

        hits.append(1.0 if int(row["response"]) == tally_win else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.1889 (var=0.0124)
**Predicted under pi_1:** 0.1217 (var=0.0145)
**Predicted under pi_3:** 0.1833 (var=0.0042)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Read across all four experiments, the two theories are not equally damaged, but neither is currently right \u2014 and the pattern of the misfit is highly informative.\n\nExperiments 3 and 4 are TTB-vs-tally conflict metrics. Observed: 0.737 top-cue agreement (exp3) and 0.189 tally-agreement, i.e. ~0.81 top-cue agreement (exp4). Both theories reproduce these (pi_1: 0.84 / 0.12; pi_3: 0.76 / 0.18), with pi_3 slightly closer. Crucially, these two metrics are *symmetric under a relabelling of the response*: a value of 0.74 is equally consistent with 'TTB-consistent 74% of the time' and with 'integration-consistent 74% of the time under a flipped option label'. So exps 3\u20134 discriminate almost nothing.\n\nExperiments 1 and 2 are the decisive ones, and they falsify pi_1 in a way that is *immune to any labelling convention*. Both metrics are within-subject differences of TTB-agreement between a large-positive-margin family and a margin-reversed family. Under Take-The-Best the choice probability for the top-discriminating-cue option is one constant number on every discriminating pair, so the difference is 0 whether or not responses are coded the way the metric assumes (flipping the label maps p -> 1-p in *both* families, leaving the difference at 0). The observed values are -0.673 (var 0.021) and -0.410 (var 0.083): large in magnitude, far outside TTB's binomial band (simulated 0.006 and 0.017). Whatever the subjects are doing, their choice consistency depends strongly on the *absolute weighted margin* of the pair, which is exactly the claim TTB forbids and exactly the claim pi_3 was built on. pi_3's predicted magnitudes (0.75, 0.84) are of the right order; only the sign is wrong.\n\nAnd the sign is not something a sane forward model can produce. Reconstructing the trial families: in exp1 the 'agreement' set contains the pairs where one option strictly dominates (only the .85 cue discriminates, trials 13\u201314) and the unanimous 6\u20130 pairs (15\u201316); in exp2 it contains the .90-vs-{.62,.55} pairs and the unanimous 7\u20130 pairs. A difference of -0.67 arithmetically requires near-zero agreement on dominance and unanimity trials while being ~90\u2013100% top-cue consistent on the 3\u20135-cue conflict trials. No evidence-monotone rule \u2014 TTB, tallying, WADD, AEWI, averaging/Anderson-style integration \u2014 can violate dominance while honouring the top cue under conflict. What *does* fit all four numbers with a single mechanism is a sign-reversed (or reverse-labelled) weighted integrator: a rule that ends up selecting the option with the smaller validity-weighted count of positive ratings. Checked quantitatively: pure reversed-polarity integration predicts about -0.83 (exp1), -0.50 (exp2), 1.0 (exp3), 0.0 (exp4); adding a lapse of ~0.2\u20130.35 lands on (-0.67, -0.40, 0.83, 0.15) against the observed (-0.67, -0.41, 0.74, 0.19). That is the only single account that hits all four experiments including the signs.\n\nBottom line: pi_3 (absolute-evidence, margin-graded integration) has the right structural content \u2014 margin-dependent, graded, non-rank-based consistency with large between-subject variance where margins are small (observed vars 0.021 and 0.083 exceed both simulations) \u2014 and it also survives exps 3\u20134. pi_1 (Take-The-Best) predicts a rank-only, label-invariant zero contrast and is decisively contradicted twice, while adding nothing over pi_3 on exps 3\u20134. pi_1 is the theory to replace; and the replacement should be one that can carry the *sign* of the observed contrasts, which pi_3 currently cannot, so that the next round is a genuine test of polarity/evidence-direction rather than another rank-vs-magnitude rerun.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 (Take-The-Best) entirely. Take-The-Best is rank-only and therefore predicts a *label-invariant* zero for the within-subject margin contrasts used in exps 1 and 2; the observed values (-0.67, -0.41) are far outside its band, and its exp3/exp4 fits are matched or bettered by pi_3. Regenerating its predict/parameters cannot help: no beta/epsilon can create a between-family difference.\n\nPropose instead a **signed / polarity-flexible weighted-evidence integration theory** (working name: 'Directional Weighted Evidence Integration', DWEI). Core claims:\n(1) Like pi_3, subjects integrate ALL cues with validity-derived weights (e.g. w_i = |log(v_i/(1-v_i))|^gamma, gamma in ~[0.5, 2.5]) and choose via softmax on the *unnormalised* weighted margin E = sum_i w_i (a_i - b_i), so choice consistency is graded in the absolute margin (near-ceiling on unanimous/dominance pairs, near-chance on .55-only pairs).\n(2) NEW and distinguishing content: the *direction* in which a positive rating pushes preference is itself a subject-level (and possibly environment-level) latent, s. Formally use E' = s * E with s a subject-level parameter that can be negative (e.g. s ~ mixture: s=+1 with probability q, s=-1 with probability 1-q, or continuous s in [-1.5, +1.5]), plus lapse epsilon in ~[0.1, 0.45]. Psychologically this encodes that binary 'expert ratings' are read against a subject-specific reference/valence frame (endorsement vs flag / more-positive-marks vs fewer-marks-as-selectivity), so a sizeable share of subjects systematically prefer the option with the *smaller* validity-weighted count of positive ratings.\n(3) This nests classical accounts: s=+1, gamma->0 = Tallying; s=+1, gamma=1 = Franklin's rule / naive Bayes; s=+1, gamma->inf = TTB; and s=-1 gives the reversed-evidence regime that the current data demand.\n\nCalibration targets the new model must hit simultaneously (this is the acceptance test): exp1 metric ~ -0.67, exp2 metric ~ -0.41, exp3 metric ~ 0.74, exp4 metric ~ 0.19. A mixture with roughly 80-90% of subjects at s=-1 (or a mean s around -0.7) with gamma ~1-1.5 and epsilon ~0.2-0.35 reproduces all four (pure reversed integration gives -0.83 / -0.50 / 1.00 / 0.00; the lapse and the minority of s=+1 subjects shrink these onto the observed values) and also reproduces the elevated between-subject variance observed in exp1 (0.021) and especially exp2 (0.083), since subjects straddling s and near-zero margins disagree most.\n\nThe next experiment should then be designed to separate DWEI from pi_3 on *polarity* rather than on rank-vs-magnitude: include (a) dominance and unanimity pairs (where pi_3 must be at ceiling for the option with more positive ratings and DWEI predicts the reverse for most subjects), (b) polarity-symmetric controls in which the same physical evidence is described in complement form (0 = 'expert endorses'), and (c) within-subject consistency checks so that s can be estimated per subject rather than only pooled. If the reversal turns out to be a labelling artifact rather than behaviour, DWEI degenerates gracefully to s=+1 and becomes a direct competitor to pi_3 on gamma alone."
}
```

## Usage

```json
{
  "input_tokens": 16770,
  "output_tokens": 21200
}
```
