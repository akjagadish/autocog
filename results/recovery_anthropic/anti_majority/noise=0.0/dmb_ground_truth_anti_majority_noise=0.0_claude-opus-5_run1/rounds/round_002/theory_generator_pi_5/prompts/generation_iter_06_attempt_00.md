# generation_iter_06_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Decision Making (Binary Features) domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
The goal of the theory generation process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across the majority of experiments. 
You will see a list of theories that have been proposed in the past but you should only use them as inspiration and not to choose from them. Propose a new theory that is different. 
If they fail to do so, you will receive feedback on their performance on the same experiment(s) and you will have to propose another new theory and model that meet the requirements, iterating until you succeed.

If you think the failure to capture human behavior is due to arbiter feedback that is inaccurate or unhelpful, you can propose a new theory and model that ignore the feedback, but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

## ACCEPT GATE & LOSS TRAJECTORY — HOW THE LOOP HANDLES YOUR EDITS
This propose-loop has a programmatic accept gate: after every iteration the candidate's `aggregate_loss` is compared against the running-best loss; strict improvement -> ACCEPTED (the candidate becomes the new running-best base); otherwise -> REJECTED (the candidate is discarded and the base is unchanged). You do NOT need to manually "revert" a regressed edit — the gate already does that for you.

The block rendered below as `## PREVIOUS CANDIDATE (this loop)` is ALWAYS the running-best (last ACCEPTED) candidate, NEVER your most recent attempt if it was rejected. So:
  * Treat `## PREVIOUS CANDIDATE` as a known-good base. Build on it.
  * The `## LOSS TRAJECTORY` block tags every iteration ACCEPTED or REJECTED. Use this as ground truth on which past critic advice actually moved the loop forward and which didn't.
  * The `## PRIOR FEEDBACK ITERATIONS` block annotates each prior critique with the same ACCEPTED/REJECTED tag of the candidate it elicited. Down-weight critic advice whose previous candidates were REJECTED, and reinforce / extend advice whose candidates were ACCEPTED.
  * Treat the best ACCEPTED iteration's loss as a soft floor — the next edit should plausibly land at-or-below it, otherwise the gate will reject your attempt and the base stays put.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

## PARAMETER NOTATION
`parameters` is a JSON object mapping each parameter name (snake_case string) to a *string* value that specifies its domain. Every value MUST be a string — never a bare list, number, tuple, or expression. Use exactly one of these notations per parameter:

1. Continuous interval — square brackets, two numeric bounds:
   "[min, max]"
   Examples: "[0, 1]", "[1.0, 10.0]", "[10, 1000]"

2. Discrete set — curly braces, comma-separated values:
   "{v1, v2, ...}"
   Example: "{1, 2}"

3. Vector of intervals whose length is set by the experiment — a bracketed tuple repeated by a symbolic length variable:
   "[(min, max)] * length_var"
   Example: "[(0, 1)] * n_features"

4. Symbolic reference — a bare variable name (no brackets, no angle brackets), used when the parameter takes its value from an experiment-defined constant rather than a range:
   "variable_name"
   Example: "n_features"

Rules:
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` are reserved for the vector-of-intervals notation in (3).
- Do not mix notations within a single value (e.g., no "[0, 1] or {2, 3}").
- Do not quote numbers inside the notation (write "[0, 1]", not "['0', '1']").
- Every parameter referenced by `predict` or `policy` must appear as a key in `parameters`, and vice versa.
- Notations 3 and 4 may ONLY reference the experiment-defined symbolic identifiers listed under "ALLOWED SYMBOLIC IDENTIFIERS" below. Do not invent new identifier names. If a parameter's shape doesn't fit any of those variables, fall back to a literal interval (notation 1) or discrete set (notation 2). Use these names so the model adapts to any experiment in this domain instead of hardcoding shapes.

## ALLOWED SYMBOLIC IDENTIFIERS (for notations 3 and 4 above)
- n_features: Number of expert ratings per option (LLM-proposed via `validities` length).
- validities: Per-expert validities (LLM-proposed; each in [0.5, 1.0]); fixed across all trials.

## AVAILABLE IMPORTS inside `predict` and `policy`
- numpy as np
- pandas as pd
- scipy and its submodules
- torch and torch.nn.functional as F
- sklearn and its submodules
- math, random, and other standard Python libraries

## RUNTIME CONTRACT (function signatures and argument shapes)
`predict(parameters, state, history) -> np.ndarray`:
- `parameters`: dict[str, value]. One sample drawn from your declared `parameters` ranges, applied for the entire subject run.
- `state`: the per-trial input delivered by the experiment (shape is domain-specific — see the experiment description above and the `history` key list below, which mirrors the per-trial variables carried in `state`). Convert to an array with `np.asarray(state)` if you need array ops.
- `history`: dict-of-lists for past trials in this subject's run, NOT a list-of-dicts. The per-trial keys are:
  Each value below is a Python list in trial order; entry `i` is the value for trial `i`. On the first trial all lists are empty.
  - `"option_a_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
  - `"option_b_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace Take-The-Best entirely with a compensatory, display-bound weighted-additive theory WITHOUT any coherence/unanimity term. Sketch:

**Name:** Position-Bound Weighted Additive integration with faint-praise endorsement discounting (PWA-FP).

**Core:** D = \u03a3_j w_eff_j (a_j - b_j) - \u03b8\u00b7(\u03a3a - \u03a3b); p(A) = (1-\u03b5)\u00b7logistic(\u03b2 D) + \u03b5/2. All cues are always integrated (no stopping rule) \u2014 this is what buys Exp 4's chance-level conflict performance and Exp 3's ~0.75.

**Weights:** w_j is a convex mixture of (i) a validity-derived weight (e.g. \u221d logit(v_j)^\u03b3) bound to the correct expert, and (ii) a purely positional weight that decays left-to-right over displayed columns, w_pos \u221d (n - j)^\u03bb. The mixture coefficient is a *free* readability parameter that should be allowed to depend on how easy the validity-to-column mapping is (near-pure validity weighting when the display is monotone in validity, near-pure positional weighting when it is scrambled, best cue in a middle column). This is deliberately close to pi_4's kappa mechanism (which the data support: Exp 5 obs -0.208, Exp 6 obs 0.659) but formulated as an explicit position-vs-validity mixture rather than a flattening-toward-equal plus a small gradient, so it can generate the strong positional dominance seen in Exp 6 (0.66) while still letting the .93 cue win in Exp 5's monotone display.

**Crucially, no consensus/coherence premium.** The rho\u00b7|2\u00b7mean-1|^q term must be dropped outright. In its place put a small, strictly MONOTONE endorsement-count discount \u03b8 (0 to ~0.05 of total weight): each extra '1' slightly lowers an option's appeal (faint-praise skepticism). This reproduces (a) Exp 1's ~0 contrast (both the 5/5-vs-4/5 cell and the single-weak-endorsement-vs-empty cell are shifted in the *same* direction, so the difference stays near zero, and can be very slightly negative), (b) Exp 2's small negative -0.15 (U4-U1 and V2-V1 near zero or slightly negative because \u03b8 penalises the higher-endorsement option equally in both cells, while L1-L4 is small once flattened weights let the 4-cue coalition partly offset the .93 cue), and (c) Exp 5's negative agree-minus-conflict value via near-chance/sub-chance responding on the weak-single-endorsement-vs-empty pair, without ever predicting the huge unanimity steps that Exps 1 and 2 refuted. Constrain \u03b8 so that the dominance anchor ([1..1] vs [0..0]) stays at ceiling.

**Parameter ranges:** \u03b2 wide (2\u201312), \u03b3 ~0.8\u20132.0, positional-mixture weight wide (0\u20130.9) and tied to display monotonicity, \u03b8 small (0\u20130.06), \u03b5 0\u20130.15. No stopping/frugality parameter, no q, no rho.

This new theory competes with pi_4 on exactly the mechanism the data can still adjudicate \u2014 smooth endorsement-count effects versus all-or-none unanimity, and flattening-toward-equal versus position-sorted re-weighting \u2014 while both remain compensatory, so future experiments will be genuinely diagnostic rather than re-testing frugality, which is now settled.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.261)

**Description**
**WADD-γ with Positional Validity Misattribution (WADD-γ + position-default blending), with heterogeneous decision determinacy.**

People integrate cues compensatorily: each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, L_j = log(v_j/(1−v_j)), w_j ∝ L_j^γ; the decision rests on the normalised evidence margin d = Σ_j w_j (x_Aj − x_Bj), with p(A) logistic in β·d and an attentional lapse ε. γ ≈ 1.15–1.85 places the decision maker in the near-lexicographic but still compensatory regime: the top expert usually survives a coalition of dissenters, but a near-peer second cue plus a couple of moderate cues can cancel or overturn it.

The substantive claim about *where the weights get attached* is retained and is the theory's main falsifiable commitment. Validities are stated verbally in the instructions, but the decision maker must bind each stated validity to a *column* of the displayed rating vector. This binding is costly and error-prone, and people fall back on a powerful display heuristic: **importance decreases from left to right** (the first-listed expert is the best one, the last-listed the worst). Subjective weights are therefore a blend, with mixing weight λ, of the correctly-bound validity weights and the weights implied by the positional default (the same weight multiset re-sorted into descending order across columns):

  w_eff ∝ (1−λ)·L_j^γ + λ·sort_desc(L^γ)_j.

This is an *environmental* prediction, not a per-experiment fudge: conflict-trial accuracy is a property of the alignment between display order and validity order, while the integration rule itself stays a single stationary compensatory WADD-γ. Concretely, (i) scramble the column order of an environment whose cues are currently presented in descending validity and conflict accuracy must fall toward (and, on some families, below) chance; (ii) re-sort the columns of a scrambled environment into descending validity and the same subjects must become sharply more accurate and more top-cue-consistent; (iii) an explicit column-labelling manipulation that makes the validity→column binding visually trivial should abolish the drop. Because the blend is the exact identity whenever columns are already validity-ordered, λ is behaviourally inert in validity-sorted environments and can only be identified by comparing environments — precisely the invariance signature the theory stakes itself on.

The third claim is about **population structure**: the same stationary rule is held by everyone, but decision determinacy is heterogeneous across people. Some subjects convert modest weighted margins into near-deterministic choices, others remain near chance on all but the most lopsided evidence. That heterogeneity is carried by the logistic gain β and by mild variation in the validity-sensitivity exponent γ, not by discrete strategy switching; the theory therefore predicts that between-subject spread in conflict accuracy should be largest in environments with intermediate margins and should shrink towards zero in environments where every conflict family is near-cancelling.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # ------------------------------------------------------------------
    # WADD-gamma with positional validity misattribution.
    #   w_j      proportional to  logit(v_j) ** gamma
    #   w_eff    = (1-lam) * w + lam * sort_desc(w)     (position default)
    #   d        = sum_j w_eff_j * (x_Aj - x_Bj)        in [-1, 1]
    #   p(A)     = sigmoid(beta * d), mixed with lapse epsilon.
    # gamma -> 0 recovers Tallying, gamma -> large recovers Take-The-Best.
    # lam is the degree of reliance on the display heuristic "importance
    # decreases left-to-right"; it is a no-op when the experiment already
    # presents cues in descending validity order.
    # History is not used: the rule is stationary (no feedback is given).
    # ------------------------------------------------------------------
    import numpy as np

    # ---- unpack the stimulus into two rating vectors -------------------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            # flat concatenation of the two option vectors
            half = stim.shape[0] // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            a, b = stim[0].astype(float), stim[1].astype(float)

    n = int(min(a.shape[0], b.shape[0]))
    a = a[:n]
    b = b[:n]
    if n == 0:
        return np.ones(2) / 2.0

    # ---- subjective importance weights from cue diagnosticity ----------
    val = parameters.get("validities", None)
    if val is None:
        v = np.full(n, 0.75, dtype=float)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]

    # validities live in (0.5, 1); clip for numerical safety
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)
    logit = np.log(v / (1.0 - v))          # diagnosticity on evidence scale
    logit = np.maximum(logit, 1e-9)        # strictly positive

    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 7.0))
    epsilon = float(parameters.get("epsilon", 0.0))
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    lam = float(parameters.get("lam", 0.0))
    lam = float(np.clip(lam, 0.0, 1.0))

    with np.errstate(over="ignore", invalid="ignore"):
        w = np.power(logit, gamma)
    w = np.nan_to_num(w, nan=0.0, posinf=np.finfo(float).max / (10.0 * n), neginf=0.0)

    # ---- positional misattribution: blend with the left-to-right default
    # The default binds the largest importance to the first displayed cue,
    # the next largest to the second, etc.  When the experiment already
    # orders columns by descending validity this blend is the identity.
    if n > 1 and lam > 0.0:
        w_pos = np.sort(w)[::-1]
        w = (1.0 - lam) * w + lam * w_pos

    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)      # degenerate -> equal weights
    else:
        w = w / s

    # ---- weighted additive evidence margin -----------------------------
    d = float(np.dot(w, a - b))            # in [-1, 1]

    # numerically stable two-alternative softmax == logistic on beta*d
    z = np.array([beta * d, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - epsilon) * p_core + epsilon * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if not np.isfinite(tot) or tot <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))

`parameters`:
- gamma: [1.15, 1.85]
- beta: [1.0, 13.0]
- epsilon: [0.0, 0.14]
- lam: [0.64, 1.00]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7533 (var=0.0228) vs this=0.7375 (var=0.0102)
- Experiment 2: real=0.5100 (var=0.0136) vs this=0.5371 (var=0.0083)
- Experiment 3: real=-0.2081 (var=0.0368) vs this=0.6219 (var=0.0370)
- Experiment 4: real=0.6587 (var=0.0379) vs this=0.8380 (var=0.0096)
- Experiment 5: real=-0.0133 (var=0.0143) vs this=-0.0750 (var=0.0313)
- Experiment 6: real=-0.1500 (var=0.0125) vs this=0.0383 (var=0.0077)


---

### `pi_4` (overall score: 0.027)

**Description**
**Readability-limited validity weighting with a strict-unanimity consensus premium (RVW-DE+/U), with broad determinacy heterogeneity.**

A single stationary, fully compensatory integration rule underlies choice in all cue-based binary-feature environments; the apparent switching between 'take-the-best-like' and 'tallying-like' behaviour across environments is produced by modulators that are properties of the *display and the evidence pattern*, not of the strategy, plus wide between-person variation in decision determinacy.

(1) **Graded validity weighting.** Each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, w_j ∝ logit(v_j)^gamma, gamma spread over ~1.05–2.0 across people. At the population mean this is steep enough that a clearly best expert usually survives a coalition of three or four weak experts, yet weak enough that a near-peer plus a couple of moderate cues can overturn the top cue.

(2) **Binding readability, not position.** Stated validities must be bound to displayed columns. When the display order is *monotone* in validity (ascending OR descending) this binding is trivial and the validity weights are used essentially intact. When the order is scrambled, binding fails partially and the weights are flattened toward EQUAL weighting: w_eff = (1-kappa)·w + kappa/n, with kappa = k0 + k1·(1-|Kendall tau(column index, validity)|) + k2·[best expert not in an extreme column]. Direction-free: reversing a sorted display costs nothing, permuting it costs a lot.

(3) **Readability-gated reading-order fallback.** The left-to-right attention gradient is a fallback recruited only when validity-to-column binding fails: phi_eff = phi·(r0 + r1·kappa). It only *scales* weights, never re-sorts them.

(4) **Strict-unanimity consensus premium (dissent aversion).** The premium attaches to *consensus itself*, and consensus is an essentially all-or-none perceptual property: a panel is 'agreed' only when it is unanimous or one voice away from it. Subjects add a direction-blind premium rho·|2·mean(x)-1|^q with a HIGH exponent q (~5–11, person-specific), so a 5/5 or 0/5 panel earns almost the whole premium while a 4/5 or 1/5 panel earns almost none. The magnitude rho of this premium is substantial and heterogeneous (population mean ~0.39): an option unanimously rated 0 can be *preferred* to one carrying a single weak endorsement (sub-chance accuracy on one-endorsement-vs-nothing pairs), while options that are merely lopsided (4-of-5 endorsements) receive essentially no credibility bonus and therefore cannot dilute genuine weighted-evidence conflicts. The premium is a *unanimity* effect, not a mean-rating effect — that is the theory's sharpest falsifiable commitment (a strongly non-linear step at exactly 0 and n dissenters, not a smooth gradient in endorsement proportion).

(5) **Endorsement discount / pull.** Each positive endorsement carries a small credibility adjustment theta (faint-praise skepticism for most, mild endorsement-count pull for some), inert whenever the two options carry equal endorsement counts.

(6) **Population structure.** Everyone holds the same rule; heterogeneity is broad and continuous — in determinacy beta, in gamma, in lapse, in the modulator magnitudes rho, q, theta, phi and the readability coefficients. No discrete strategy switching.

Evidence for A: D = Σ_j w_eff_j (x_Aj - x_Bj) - theta·(Σx_A - Σx_B) + rho·(coh(x_A) - coh(x_B)); p(A) = (1-eps)·logistic(beta·D) + eps/2.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Readability-limited validity weighting with dissent aversion and
    # endorsement discount (RVW-DE+/U).
    #   w_j        proportional to logit(v_j) ** gamma
    #   w_eff      = (1-kappa)*w + kappa/n          (readability flattening)
    #   w_eff     *= (1 + phi_eff * reading-order gradient)
    #                with phi_eff = phi*(r0 + r1*kappa): the reading-order
    #                fallback is recruited only when binding fails.
    #   D          = w_eff.(a-b) - theta*(sum a - sum b) + rho*(coh(a)-coh(b))
    #   p(A)       = (1-eps)*logistic(beta*D) + eps/2
    # kappa is driven by how incongruent the column order is with the
    # stated validity order (direction-free: monotone displays are cheap,
    # scrambled displays are expensive).  coh(x) = |2*mean(x)-1|**q with a
    # HIGH person-specific exponent q: the consensus premium is a strict
    # unanimity effect, not a smooth function of endorsement proportion.
    # History is unused: no feedback is given, the rule is stationary.
    import numpy as np

    # ---------------- unpack the two rating vectors --------------------
    a = None
    b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            half = arr.shape[0] // 2
            a = arr[:half].astype(float)
            b = arr[half:half + half].astype(float)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            a = arr[0].astype(float)
            b = arr[1].astype(float)

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float).ravel()[:n]
    b = np.asarray(b, dtype=float).ravel()[:n]

    # ---------------- validities --------------------------------------
    val = parameters.get("validities", None)
    if val is None:
        v = np.linspace(0.9, 0.55, n)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---------------- parameters --------------------------------------
    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 7.0))
    eps = float(np.clip(float(parameters.get("eps", 0.09)), 0.0, 1.0))
    k0 = float(parameters.get("k0", 0.08))
    k1 = float(parameters.get("k1", 0.78))
    k2 = float(parameters.get("k2", 0.35))
    phi = float(parameters.get("phi", 0.20))
    theta = float(parameters.get("theta", 0.02))
    rho = float(parameters.get("rho", 0.39))
    q = float(parameters.get("q", 8.0))

    # readability-gating of the reading-order fallback (fixed structure)
    r0 = 0.35
    r1 = 1.15

    # ---------------- validity weights --------------------------------
    L = np.log(v / (1.0 - v))
    L = np.maximum(L, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.power(L, gamma)
    w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)
    s = float(np.sum(w))
    if (not np.isfinite(s)) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)
    else:
        w = w / s

    # ---------------- readability flattening --------------------------
    # Kendall tau between column index and stated validity (ties skipped).
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    tot = conc + disc
    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)
    incong = 1.0 - abs(tau)

    jmax = int(np.argmax(v))
    off_extreme = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0

    kappa = k0 + k1 * incong + k2 * off_extreme
    kappa = float(np.clip(kappa, 0.0, 1.0))

    w_eff = (1.0 - kappa) * w + kappa * (1.0 / float(n))

    # ------- reading-order (primacy) gradient, gated by readability ----
    if n > 1 and phi != 0.0:
        phi_eff = phi * (r0 + r1 * kappa)
        phi_eff = float(np.clip(phi_eff, 0.0, 0.45))
        idx = np.arange(n, dtype=float)
        grad = 1.0 + phi_eff * ((float(n - 1) - 2.0 * idx) / float(n - 1))
        grad = np.maximum(grad, 1e-6)
        w_eff = w_eff * grad
    ssum = float(np.sum(w_eff))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        w_eff = np.ones(n, dtype=float) / float(n)
    else:
        w_eff = w_eff / ssum

    # ---------------- evidence ----------------------------------------
    core = float(np.dot(w_eff, a - b))

    # endorsement discount: credibility adjustment per positive rating
    cost = -theta * (float(np.sum(a)) - float(np.sum(b)))

    # dissent aversion: strict-unanimity consensus premium (high exponent q)
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    ua = abs(2.0 * ma - 1.0)
    ub = abs(2.0 * mb - 1.0)
    coh_a = float(np.power(ua, q))
    coh_b = float(np.power(ub, q))
    cons = rho * (coh_a - coh_b)

    D = core + cost + cons

    # ---------------- logistic choice with lapse -----------------------
    z = np.array([beta * D, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if (not np.isfinite(tot)) or tot <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))

`parameters`:
- gamma: [1.05, 1.95]
- beta: [1.2, 13.0]
- eps: [0.0, 0.20]
- k0: [0.02, 0.16]
- k1: [0.55, 1.00]
- k2: [0.10, 0.60]
- phi: [0.02, 0.38]
- theta: [-0.03, 0.07]
- rho: [0.19, 0.59]
- q: [5.0, 11.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7533 (var=0.0228) vs this=0.7196 (var=0.0166)
- Experiment 2: real=0.5100 (var=0.0136) vs this=0.5204 (var=0.0106)
- Experiment 3: real=-0.2081 (var=0.0368) vs this=-0.2069 (var=0.0176)
- Experiment 4: real=0.6587 (var=0.0379) vs this=0.6633 (var=0.0149)
- Experiment 5: real=-0.0133 (var=0.0143) vs this=0.7083 (var=0.0508)
- Experiment 6: real=-0.1500 (var=0.0125) vs this=0.4600 (var=0.0383)


---

### `pi_1` (overall score: 0.011)

**Description**
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


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7533 (var=0.0228) vs this=0.8375 (var=0.0108)
- Experiment 2: real=0.5100 (var=0.0136) vs this=0.8204 (var=0.0123)
- Experiment 3: real=-0.2081 (var=0.0368) vs this=-0.0092 (var=0.0059)
- Experiment 4: real=0.6587 (var=0.0379) vs this=0.1800 (var=0.0087)
- Experiment 5: real=-0.0133 (var=0.0143) vs this=0.0150 (var=0.0203)
- Experiment 6: real=-0.1500 (var=0.0125) vs this=-0.0092 (var=0.0076)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0557 -> ACCEPTED
- iter 2: loss=0.0484 -> ACCEPTED
- iter 3: loss=0.0691 -> REJECTED
- iter 4: loss=0.0475 -> ACCEPTED
- iter 5: loss=0.0384 -> ACCEPTED
- iter 6: loss=0.0428 -> REJECTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.0384 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # Take-The-Best winner: first discriminating cue in validity order.
        # Feature columns are already ordered by descending validity in this design.
        ttb_winner = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        # Tallying winner: majority of feature-wise wins (ties excluded).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        if tally_winner == ttb_winner:
            continue  # not a conflict trial
        resp = row['response']
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        hits += 1.0 if int(resp) == ttb_winner else 0.0
        n += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.7533 (var=0.0228)
**Previous candidate values (this loop):**
  - iter 1: 0.7875 (var=0.0059) (Δ vs real +0.0342)
  - iter 2: 0.7979 (var=0.0078) (Δ vs real +0.0446)
  - iter 3: 0.8383 (var=0.0046) (Δ vs real +0.0850)
  - iter 4: 0.7762 (var=0.0093) (Δ vs real +0.0229)
  - iter 5: 0.7979 (var=0.0053) (Δ vs real +0.0446)
  - iter 6 (most recent): 0.7483 (var=0.0064) (Δ vs real -0.0050)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8375 (var=0.0108)
- pi_2: 0.1425 (var=0.0084)
- pi_3: 0.7375 (var=0.0102)
- pi_4: 0.7196 (var=0.0166)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Fixed design validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65
    val = np.array([0.93, 0.58, 0.86, 0.51, 0.74, 0.65], dtype=float)

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape[0] != b.shape[0]:
            continue
        if a.shape[0] == val.shape[0]:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(a.shape[0])

        tally = int(np.sum(a > b)) - int(np.sum(b > a))

        winner = None
        for j in order:
            j = int(j)
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no cue discriminates: uninformative for both models

        ttb_sign = 1 if winner == 0 else -1
        # Keep only trials where the tally does NOT favour the TTB winner:
        # either the tally is exactly tied (Family D) or it points the
        # other way (Family E, directional conflict).
        if tally * ttb_sign > 0:
            continue

        n += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == winner:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5100 (var=0.0136)
**Previous candidate values (this loop):**
  - iter 1: 0.5004 (var=0.0064) (Δ vs real -0.0096)
  - iter 2: 0.5579 (var=0.0085) (Δ vs real +0.0479)
  - iter 3: 0.5225 (var=0.0069) (Δ vs real +0.0125)
  - iter 4: 0.5342 (var=0.0095) (Δ vs real +0.0242)
  - iter 5: 0.5121 (var=0.0087) (Δ vs real +0.0021)
  - iter 6 (most recent): 0.5267 (var=0.0070) (Δ vs real +0.0167)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3058 (var=0.0050)
- pi_1: 0.8204 (var=0.0123)
- pi_3: 0.5371 (var=0.0083)
- pi_4: 0.5204 (var=0.0106)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = [0.55, 0.62, 0.70, 0.85, 0.93]

    def key(v):
        return ''.join([str(int(round(float(x)))) for x in v])

    # conflict families: TTB (top-validity cue) and a left-to-right positional
    # weighting scheme point to OPPOSITE options
    conflict_keys = {
        '10000|00001', '00001|10000',      # family A
        '01000|00010', '00010|01000',      # family B
        '11000|00011', '00011|11000',      # family E
        '00100|11000', '11000|00100',      # family F
    }
    # agreement families: both accounts favour the same option
    agree_keys = {
        '11111|00000', '00000|11111',      # family D (dominance)
        '00001|00000', '00000|00001',      # family G
        '10000|00000', '00000|10000',      # family H
    }

    def ttb_winner(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    n_conf = 0
    hit_conf = 0
    n_agree = 0
    hit_agree = 0

    for _, row in data.iterrows():
        try:
            a = [int(round(float(x))) for x in row['option_a_ratings']]
            b = [int(round(float(x))) for x in row['option_b_ratings']]
            resp = int(row['response'])
        except Exception:
            continue
        n = len(a)
        if len(b) != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fallback: assume columns are displayed in ascending validity
            v = np.linspace(0.55, 0.93, n)
        order = list(np.argsort(-v, kind='stable'))
        w = ttb_winner(a, b, order)
        if w < 0:
            continue
        k = key(a) + '|' + key(b)
        if k in conflict_keys:
            n_conf += 1
            hit_conf += 1 if resp == w else 0
        elif k in agree_keys:
            n_agree += 1
            hit_agree += 1 if resp == w else 0

    if n_conf == 0 or n_agree == 0:
        return 0.0

    return float(hit_agree) / n_agree - float(hit_conf) / n_conf

```

**Observed (real) value:** -0.2081 (var=0.0368)
**Previous candidate values (this loop):**
  - iter 1: -0.1257 (var=0.0082) (Δ vs real +0.0824)
  - iter 2: -0.1715 (var=0.0082) (Δ vs real +0.0365)
  - iter 3: -0.2093 (var=0.0088) (Δ vs real -0.0013)
  - iter 4: -0.1771 (var=0.0130) (Δ vs real +0.0310)
  - iter 5: -0.1822 (var=0.0149) (Δ vs real +0.0258)
  - iter 6 (most recent): -0.1581 (var=0.0101) (Δ vs real +0.0500)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0092 (var=0.0059)
- pi_3: 0.6219 (var=0.0370)
- pi_2: 0.4544 (var=0.0220)
- pi_4: -0.2069 (var=0.0176)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (display order): col0..col5
    v_default = np.array([0.60, 0.93, 0.55, 0.86, 0.68, 0.78], dtype=float)

    def build(n):
        v = v_default.copy()
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
        v = np.clip(v, 0.5 + 1e-6, 1 - 1e-6)
        L = np.log(v / (1.0 - v))
        w = np.power(L, 1.5)
        wpos = np.sort(w)[::-1]
        weff = 0.2 * w + 0.8 * wpos          # lam = 0.8, mid of advocated box
        s = weff.sum()
        weff = weff / s if s > 0 else np.ones(n) / n
        cascade = list(np.argsort(-v, kind="stable"))
        return weff, cascade

    cache = {}
    num = 0.0
    den = 0.0

    for a_raw, b_raw, resp in zip(data["option_a_ratings"],
                                  data["option_b_ratings"],
                                  data["response"]):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        if n not in cache:
            cache[n] = build(n)
        weff, cascade = cache[n]

        d = float(np.dot(weff, a - b))

        winner = None
        for j in cascade:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue
        if abs(d) < 0.20:                    # ambiguous / near-cancelling for WADD-pos
            continue
        pos_choice = 0 if d > 0 else 1
        if pos_choice == winner:             # models agree -> uninformative
            continue

        den += 1.0
        try:
            r = int(resp)
        except Exception:
            continue
        num += 1.0 if r == pos_choice else 0.0

    if den == 0:
        return 0.5
    return float(num / den)
```

**Observed (real) value:** 0.6587 (var=0.0379)
**Previous candidate values (this loop):**
  - iter 1: 0.7480 (var=0.0085) (Δ vs real +0.0893)
  - iter 2: 0.6840 (var=0.0090) (Δ vs real +0.0253)
  - iter 3: 0.6967 (var=0.0069) (Δ vs real +0.0380)
  - iter 4: 0.6580 (var=0.0097) (Δ vs real -0.0007)
  - iter 5: 0.6460 (var=0.0085) (Δ vs real -0.0127)
  - iter 6 (most recent): 0.6360 (var=0.0101) (Δ vs real -0.0227)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8380 (var=0.0096)
- pi_1: 0.1800 (var=0.0087)
- pi_2: 0.6260 (var=0.0084)
- pi_4: 0.6633 (var=0.0149)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def key(x):
        return tuple(int(v) for v in list(x))

    A = [key(x) for x in data['option_a_ratings']]
    B = [key(x) for x in data['option_b_ratings']]
    R = [int(r) for r in data['response']]

    ones5 = (1, 1, 1, 1, 1)
    four = (1, 1, 1, 1, 0)
    zeros = (0, 0, 0, 0, 0)
    weak_single = [(0, 0, 1, 0, 0), (0, 0, 0, 0, 1)]

    hi_n = 0
    hi_d = 0
    lo_n = 0
    lo_d = 0

    for a, b, r in zip(A, B, R):
        if len(a) != 5 or len(b) != 5:
            continue
        # HIGH-coherence cell: 5/5 unanimous panel vs 4/5 panel, differ on cue5 only.
        if a == ones5 and b == four:
            hi_d += 1
            if r == 0:
                hi_n += 1
        elif b == ones5 and a == four:
            hi_d += 1
            if r == 1:
                hi_n += 1
        # LOW-coherence cell: single weak endorsement vs unanimous all-zero panel.
        elif (a in weak_single) and b == zeros:
            lo_d += 1
            if r == 0:
                lo_n += 1
        elif (b in weak_single) and a == zeros:
            lo_d += 1
            if r == 1:
                lo_n += 1

    if hi_d == 0 or lo_d == 0:
        return float('nan')

    p_hi = hi_n / float(hi_d)
    p_lo = lo_n / float(lo_d)
    return float(p_hi - p_lo)

```

**Observed (real) value:** -0.0133 (var=0.0143)
**Previous candidate values (this loop):**
  - iter 1: -0.0283 (var=0.0377) (Δ vs real -0.0150)
  - iter 2: 0.0167 (var=0.0396) (Δ vs real +0.0300)
  - iter 3: 0.1033 (var=0.0272) (Δ vs real +0.1167)
  - iter 4: 0.0700 (var=0.0374) (Δ vs real +0.0833)
  - iter 5: 0.0475 (var=0.0192) (Δ vs real +0.0608)
  - iter 6 (most recent): 0.0483 (var=0.0412) (Δ vs real +0.0617)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0150 (var=0.0203)
- pi_4: 0.7083 (var=0.0508)
- pi_2: -0.0175 (var=0.0181)
- pi_3: -0.0750 (var=0.0313)

### Experiment 6
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    df = data
    if len(df) == 0:
        return 0.0

    def as_tup(x):
        return tuple(int(v) for v in x)

    A = df['option_a_ratings'].apply(as_tup).values
    B = df['option_b_ratings'].apply(as_tup).values
    R = df['response'].values

    def frac_choose(target, other):
        """Fraction of trials with pair {target, other} on which the
        subject picked `target`. Collapses across A/B display order."""
        num = 0
        den = 0
        for a, b, r in zip(A, B, R):
            if a == target and b == other:
                den += 1
                if int(r) == 0:
                    num += 1
            elif a == other and b == target:
                den += 1
                if int(r) == 1:
                    num += 1
        if den == 0:
            return None
        return float(num) / float(den)

    # ---- cells -------------------------------------------------------
    # Family L (coalition ladder): does the lone top-validity (col2)
    # endorsement keep winning as the opposing coalition grows?
    L1 = frac_choose((0, 0, 1, 0, 0), (1, 0, 0, 0, 0))
    L4 = frac_choose((0, 0, 1, 0, 0), (1, 1, 0, 1, 1))

    # Family U (unanimity step, weighted evidence pinned to col3 only):
    # winner-unanimous cell vs loser-unanimous cell.
    U4 = frac_choose((1, 1, 1, 1, 1), (1, 1, 1, 0, 1))
    U1 = frac_choose((0, 0, 0, 1, 0), (0, 0, 0, 0, 0))

    # Family V (same step at a heavier column, col0).
    V2 = frac_choose((1, 1, 1, 1, 1), (0, 1, 1, 1, 1))
    V1 = frac_choose((1, 0, 0, 0, 0), (0, 0, 0, 0, 0))

    contrasts = []
    if L1 is not None and L4 is not None:
        contrasts.append(L1 - L4)
    if U4 is not None and U1 is not None:
        contrasts.append(U4 - U1)
    if V2 is not None and V1 is not None:
        contrasts.append(V2 - V1)

    if len(contrasts) == 0:
        return 0.0
    return float(np.mean(contrasts))

```

**Observed (real) value:** -0.1500 (var=0.0125)
**Previous candidate values (this loop):**
  - iter 1: -0.1425 (var=0.0166) (Δ vs real +0.0075)
  - iter 2: -0.1425 (var=0.0195) (Δ vs real +0.0075)
  - iter 3: -0.1133 (var=0.0156) (Δ vs real +0.0367)
  - iter 4: -0.1133 (var=0.0181) (Δ vs real +0.0367)
  - iter 5: -0.1292 (var=0.0192) (Δ vs real +0.0208)
  - iter 6 (most recent): -0.1233 (var=0.0174) (Δ vs real +0.0267)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4600 (var=0.0383)
- pi_1: -0.0092 (var=0.0076)
- pi_2: 0.1042 (var=0.0121)
- pi_3: 0.0383 (var=0.0077)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**PWA-DD: Position-Bound Weighted Additive integration with Faint-Praise discounting and Endorsement-Density Dilution.**

Choice in binary-cue environments is always *compensatory*: on every trial the decision maker reads all displayed columns, forms a single weighted difference, and never stops early. There is no lexicographic cascade and no coherence/unanimity premium. Three claims fix the rest of the mechanism.

(1) **Weights are a mixture of validity and display position, and the mixture is governed by how readable the validity-to-column mapping is.** Stated validities must be bound to *columns*. When the display is perfectly monotone in validity (ascending or descending) that binding is trivial and people use validity-derived weights w_val ∝ logit(v_j)^γ essentially intact (the residual positional leak μ0 is small and person-specific, and in some people is exactly zero). When the column order is not monotone in validity — and especially when the best expert sits in an interior column — binding fails and people fall back almost entirely on a purely positional gradient w_pos ∝ (n−j)^λ that decays left-to-right over displayed columns. The mixture weight μ = μ0 + (1−μ0)·min(1, μ1(1−|τ|) + μ2·[best cue interior]) is therefore an *environmental* quantity, not a per-experiment fudge: it is inert (μ = μ0) whenever columns are validity-sorted, and it predicts that scrambling a sorted display converts validity-driven behaviour into position-driven behaviour and that re-sorting a scrambled display restores it. Crucially the gate SATURATES: any display that is appreciably non-monotone is read essentially positionally, with no graded validity leak surviving.

(2) **Faint-praise endorsement discounting.** Each positive endorsement carries a small fixed credibility cost θ: an option that collects more '1's is slightly *less* attractive, all else equal. This is strictly monotone in endorsement count — there is no step at unanimity. It makes a lone weak endorsement no better (usually worse) than an empty panel, and it makes a 5/5 panel no better than a 4/5 panel by exactly the same amount, which is why 'high-coherence' and 'low-coherence' single-cue steps move together rather than apart. Because θ exceeds the subjective weight of the weakest experts but not that of the strongest, endorsement by a weak expert is actively counter-diagnostic while endorsement by a strong expert still wins.

(3) **Endorsement-density dilution of decision sensitivity.** The same absolute weighted difference is less decisive when both panels are densely endorsed. Evidence is divisively normalised by the mean endorsement density T = (Σa + Σb)/2n: D = [Σ_j w_j(a_j−b_j) − θ(Σa−Σb)] / (c0 + κ·T). The dilution has TWO separable constants: a floor c0 that sets how crisply *sparse* panels are compared, and a gain κ that sets how much extra mush dense panels accrue. The floor is *not* negligible: even near-empty panels are compared with bounded, distinctly sub-ceiling determinacy, which is what keeps lone-endorsement-versus-empty cells close to (rather than far below) chance and preserves the graded ladder structure of sparse comparisons. Heavily endorsed panels are compared mushily. Unlike a consensus premium, dilution is *direction-blind* — it never confers an advantage on the more coherent option, it only compresses whatever advantage exists — so it produces the observed shrinkage of a fixed one-cue advantage at high endorsement backgrounds without ever predicting a unanimity bonus.

p(A) = (1−ε)·logistic(βD) + ε/2. Everyone holds the same stationary rule (no feedback is given, so nothing is learned); heterogeneity is broad and continuous — in determinacy β, validity sensitivity γ, discount θ, positional decay λ, dilution (c0, κ), lapse ε and the readability coefficients — never discrete strategy switching. The theory therefore predicts substantial between-subject spread in every conflict metric, largest where margins are intermediate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # PWA-DD: position-bound weighted additive integration with
    # faint-praise endorsement discounting and endorsement-density dilution.
    #
    #   w_val_j  proportional to logit(v_j) ** gamma      (validity weights)
    #   w_pos_j  proportional to (n - j) ** lam           (left-to-right gradient)
    #   mu       = mu0 + (1-mu0)*min(1, mu1*(1-|tau|) + mu2*[best cue interior])
    #   w        = (1-mu)*w_val + mu*w_pos                (normalised)
    #   num      = w.(a-b) - theta*(sum a - sum b)        (faint-praise discount)
    #   T        = (sum a + sum b) / (2n)                 (endorsement density)
    #   D        = num / (c0 + kap*T)                     (density dilution)
    #   p(A)     = (1-eps)*logistic(beta*D) + eps/2
    #
    # No stopping rule, no coherence/unanimity premium.  History unused:
    # the rule is stationary because no feedback is provided.
    import numpy as np

    # ---------------- unpack the two rating vectors --------------------
    a = None
    b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            half = int(arr.shape[0] // 2)
            a = arr[:half].astype(float)
            b = arr[half:2 * half].astype(float)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            a = arr[0].astype(float)
            b = arr[1].astype(float)

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float).ravel()[:n]
    b = np.asarray(b, dtype=float).ravel()[:n]

    # ---------------- validities ---------------------------------------
    val = parameters.get("validities", None)
    if val is None:
        v = np.linspace(0.90, 0.55, n)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---------------- parameters ---------------------------------------
    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 3.8))
    theta = float(parameters.get("theta", 0.12))
    c0 = float(parameters.get("c0", 0.09))
    kap = float(parameters.get("kap", 1.7))
    mu0 = float(np.clip(float(parameters.get("mu0", 0.07)), 0.0, 1.0))
    mu1 = float(parameters.get("mu1", 1.8))
    mu2 = float(parameters.get("mu2", 0.4))
    lam = float(parameters.get("lam", 0.95))
    eps = float(np.clip(float(parameters.get("eps", 0.125)), 0.0, 1.0))
    c0 = max(c0, 1e-3)

    # ---------------- validity-derived weights --------------------------
    L = np.log(v / (1.0 - v))
    L = np.maximum(L, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        wv = np.power(L, gamma)
    wv = np.nan_to_num(wv, nan=0.0, posinf=1e12, neginf=0.0)
    s = float(np.sum(wv))
    if (not np.isfinite(s)) or s <= 0.0:
        wv = np.ones(n, dtype=float) / float(n)
    else:
        wv = wv / s

    # ---------------- positional (reading-order) weights ----------------
    idx = np.arange(n, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        wp = np.power(np.maximum(float(n) - idx, 1e-9), lam)
    wp = np.nan_to_num(wp, nan=0.0, posinf=1e12, neginf=0.0)
    s = float(np.sum(wp))
    if (not np.isfinite(s)) or s <= 0.0:
        wp = np.ones(n, dtype=float) / float(n)
    else:
        wp = wp / s

    # ---------------- display readability -------------------------------
    # Kendall tau between column index and stated validity (ties skipped).
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    tot = conc + disc
    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)
    incong = 1.0 - abs(tau)                     # 0 when display is monotone
    jmax = int(np.argmax(v))
    interior = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0

    mu = mu0 + (1.0 - mu0) * min(1.0, max(0.0, mu1 * incong + mu2 * interior))
    mu = float(np.clip(mu, 0.0, 1.0))

    w = (1.0 - mu) * wv + mu * wp
    s = float(np.sum(w))
    if (not np.isfinite(s)) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)
    else:
        w = w / s

    # ---------------- evidence ------------------------------------------
    core = float(np.dot(w, a - b))
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    num = core - theta * (cnt_a - cnt_b)          # faint-praise discount

    dens = (cnt_a + cnt_b) / (2.0 * float(n))     # endorsement density in [0,1]
    denom = c0 + kap * dens
    if (not np.isfinite(denom)) or denom <= 1e-6:
        denom = 1e-6
    D = num / denom

    # ---------------- logistic choice with lapse -------------------------
    z = np.array([beta * D, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if (not np.isfinite(tot)) or tot <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))

`parameters`:
- gamma: [1.1, 1.9]
- beta: [2.0, 5.6]
- theta: [0.08, 0.16]
- c0: [0.05, 0.13]
- kap: [1.1, 2.3]
- mu0: [0.0, 0.14]
- mu1: [1.4, 2.2]
- mu2: [0.20, 0.60]
- lam: [0.70, 1.20]
- eps: [0.03, 0.22]
- validities: validities

`rationale`: Strict single-knob edit exactly as the critic prescribed. `predict`/`policy` are mechanically byte-identical to the accepted iter-4 base except for one inert `.get()` default (c0 0.07 -> 0.09, only used if the sampler ever omitted the key); ONE parameter range changes and every other box (beta, eps, theta, mu0, mu1, mu2, gamma, lam, kap) is left untouched.

**The edit: c0 [0.03, 0.11] -> [0.05, 0.13] (mid 0.07 -> 0.09), reverting the iter-2 value.** The critic's decomposition of iter 4 is convincing: of the three edits in that pass, mu1 is arithmetically inert on Exps 5 and 6 (Exp 5's display is monotone so the gate is at mu0; Exp 6 saturates through the interior-best-cue term), and raising kap should have pushed Exp 5's hi-minus-lo DOWN, not up. So the c0 reduction is the only remaining candidate for the two regressions (Exp 5 +0.030 -> +0.083, Exp 6 +0.0075 -> +0.037), and it is mechanistically the right suspect: c0 is the floor of the denominator c0 + kap*T, so it governs determinacy precisely in the SPARSE cells out of which both experiments are built.

Hand-check on the mid-box (theta=0.12, kap=1.7, beta=3.8, eps=0.125). Exp 5 lo-cell (weak single endorsement vs empty, T=0.1): denom goes 0.07+0.17=0.24 -> 0.09+0.17=0.26, so the theta-driven sub-chance drive |num|/denom shrinks by ~8% and p_lo moves back toward 0.5; the hi-cell (5/5 vs 4/5, T=0.9) has denom 1.60 -> 1.62, essentially unchanged. Hence hi-minus-lo falls from +0.070 toward ~+0.02, and kap=1.7 (kept) continues to compress the hi-cell. Exp 6's U1/V1 anchors and the L-ladder are all sparse (T=0.1-0.4), so the same de-sharpening restores the cell-proportion differences that had been compressed toward zero, taking the contrast back to ~-0.14 (residual +0.0075, its best-ever level at iters 1-2).

**Explicitly NOT touched:** kap stays at [1.1, 2.3] (the iter-3 correction, directionally right for Exp 5); mu1 stays at [1.4, 2.2] (the one cleanly attributable win, Exp 2 back to ~0.52); mu0, theta and eps have all now been probed in both directions and are closed; gamma is held in reserve per the critic's step-2 instruction, to be used only if Exp 3 slips past ~-0.165 after this pass. Isolating c0 also cleanly disambiguates the confounded kap/c0 change of iter 4.

Guardrail re-check: dominance anchor [1,1,1,1,1] vs [0,0,0,0,0] has num = 1 - 5*0.12 = 0.40, T=0.5, denom = 0.09 + 0.85 = 0.94, D = 0.426, beta*D = 1.62 -> p_core = 0.835, p = 0.855*... = ~0.79 after the mid lapse; comfortably at ceiling on the correct side. Expected residuals: Exp1 ~0.018, Exp2 ~0.024, Exp3 ~0.037, Exp4 ~0.005, Exp5 ~0.03, Exp6 ~0.010 - two large residuals cut by ~0.05 and ~0.03 respectively against a single ~0.006 cost on Exp 3, so the aggregate should land clearly below the 0.0475 floor.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: This candidate is by far the best-fitting model seen in this family: it is within ~0.01 of the data on Exp 2 (0.500 vs 0.510), Exp 5 (-0.028 vs -0.013) and Exp 6 (-0.143 vs -0.150), and within ~0.035 on Exp 1 (0.788 vs 0.753). Critically, it simultaneously reproduces the three patterns that split the earlier theories: the near-zero/slightly-negative unanimity contrasts (Exps 5 and 6) that killed pi_4's coherence premium, the moderate positional dominance of Exp 4 that killed pi_1's validity-locked TTB, and the negative agree-minus-conflict of Exp 3 that killed pi_2/pi_3. It is faithful to the arbiter's prescription: compensatory, all-cue integration, position-vs-validity mixture gated by display readability, monotone faint-praise discount \u03b8, no unanimity/coherence term. The one addition (density dilution) is direction-blind and does not smuggle a coherence bonus back in, so it stays inside the family.

Two residuals remain. (i) Exp 3: simulated -0.126 vs observed -0.208 \u2014 the model does not push the sparse 'single weak endorsement vs empty panel' agreement cells far enough sub-chance (or its conflict cells are not high enough), so the contrast is ~40% too shallow. (ii) Exp 4: simulated 0.748 vs observed 0.659 \u2014 the model is *too* positional/too deterministic on scrambled displays with an interior best cue; \u03bc saturates at ~1 and \u03b2 is high enough that positional WADD wins nearly always, whereas humans are noticeably noisier there.

A third, systematic issue: between-subject variance is far too small. Real var is 0.023 (Exp1), 0.014 (Exp2), 0.037 (Exp3), 0.038 (Exp4); the candidate gives 0.006, 0.006, 0.008, 0.009 \u2014 3\u20135x too homogeneous. The parameter boxes are too tight to generate the observed individual spread, and the theory itself claims heterogeneity is continuous in \u03b2, \u03b8, \u03bc, so the ranges should reflect that.
Rationale: The candidate was ACCEPTED and should be the base \u2014 do NOT redesign it. Keep the exact PWA-DD mechanism (weighted-additive, position/validity mixture gated by display readability, monotone \u03b8 discount, density dilution, no coherence term) and make three targeted, in-family knob adjustments:

1. Deepen Exp 3 (currently -0.126 vs -0.208): raise the faint-praise discount slightly, e.g. \u03b8 \u2208 [0.10, 0.18] instead of [0.06, 0.14]. This drives the sparse 'lone weak endorsement vs empty panel' agreement cells further sub-chance while leaving equal-count conflict trials inert, and (because \u0394count = 1 in both U and V cells) it largely cancels in the Exp-6 contrast and moves the two Exp-5 cells together, so the good fits there should survive. Verify the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor stays at ceiling under the upper end of the new \u03b8 box.

2. Soften Exp 4 (currently 0.748 vs 0.659): the readability gate saturates \u03bc \u2192 1 on the scrambled/interior-best display and \u03b2 is too high there. Reduce the saturation slightly \u2014 e.g. lower \u03bc1 to ~[1.0, 1.6] and \u03bc2 to ~[0.15, 0.45], and/or shift the \u03b2 box down to ~[2.2, 5.0] \u2014 so a residual fraction of validity weighting plus more choice noise pulls the positional-vs-TTB hit rate down toward 0.66. Exp 2 (also scrambled) is currently at 0.500 vs 0.510 and a small validity leak nudges it in the right direction, so this is a two-birds change; check Exp 6 does not drift positive.

3. Fix the heterogeneity deficit: widen the boxes so per-subject metric variance approaches the observed 0.014\u20130.038 rather than the current 0.006\u20130.009. Concretely widen \u03b2 (e.g. [2.0, 7.0]), \u03b5 (e.g. [0.03, 0.22]), \u03bc0 (e.g. [0.0, 0.35]) and \u03b8. Since the point estimates are already close, widening with the mid-points held near current values should preserve pooled fit while restoring realistic between-subject spread.

No mechanism-family change is warranted or permitted here \u2014 the prescribed compensatory position-bound WADD with faint-praise discounting is clearly the right family, and this is a calibration pass on top of it.

**Outcome of this advice:** iter 1 candidate loss=0.0557 -> iter 2 candidate loss=0.0484 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted iter-2 candidate is a genuine improvement (0.0557 -> 0.0484) and it remains faithful to the arbiter's prescribed family: all-cue compensatory integration, validity-vs-position mixture gated by display readability, monotone faint-praise discount \u03b8, direction-blind density dilution, no coherence/unanimity premium. It is still the only theory in the comparison set that simultaneously gets Exp 3 negative (-0.17 vs -0.208), Exp 4 moderate (0.684 vs 0.659) and Exp 6 negative (-0.143 vs -0.150) \u2014 the three contrasts that individually killed pi_1, pi_2/pi_3 and pi_4. Between-subject variance also improved (0.008-0.040 vs the 0.006-0.009 of iter 1), now roughly in the observed range on Exps 4-6 though still low on Exps 1-3.

However, the iter-2 calibration traded residuals rather than uniformly shrinking them. Wins: Exp 3 (+0.082 -> +0.037) and Exp 4 (+0.089 -> +0.025), driven by the \u03bc0 reduction and the \u03b2/\u03bb flattening. Losses: Exp 2 (-0.010 -> +0.048), Exp 5 (-0.015 -> +0.030, sign now flipped positive) and Exp 1 (+0.034 -> +0.045). The Exp 2 regression is directly attributable to lowering \u03bc1: the readability gate no longer saturates on that scrambled display, so a validity leak pushed the metric from a near-perfect 0.500 up to 0.558 (my own iter-1 'two-birds' suggestion overshot here). The Exp 5 sign flip is a \u03b8 x dilution interaction: raising \u03b8 while the sparse lo-cell (weak single endorsement vs empty) is strongly amplified by the 1/(c0+\u03ba T) denominator drives that cell far sub-chance, while the dense hi-cell (5/5 vs 4/5) is compressed \u2014 so hi-lo went positive when the data want it slightly negative. Exp 1 remains systematically ~0.045 too high, i.e. the model is globally a bit too deterministic on validity-sorted displays.
Rationale: Keep PWA-DD exactly as is mechanically \u2014 no family change, no new terms. This is a third calibration pass with three targeted, mostly-orthogonal knobs, chosen to undo the two costs my last critique incurred while preserving its two wins.

1. UNDO the \u03bc1 reduction (my iter-1 advice that backfired). Exp 2 was near-perfect at 0.500 when the readability gate saturated (\u03bc\u21921) and regressed to 0.558 once \u03bc1 was lowered to 1.15. Restore \u03bc1 to roughly [1.4, 2.2] (mid ~1.8) so Exp 2's scrambled display re-saturates and returns to ~0.51. Do NOT try to fix Exp 4 through the gate \u2014 the \u03b2/\u03bb route already worked (0.748 -> 0.684) and should be kept.

2. Reduce global determinacy slightly to fix Exp 1 (and mop up Exp 4/Exp 5). Raise the lapse box to about \u03b5 \u2208 [0.06, 0.26] (mid ~0.16) and/or shave the top of \u03b2 to ~[2.0, 5.0]. This is a single lever that moves four residuals in the right direction at once: Exp 1 0.798 -> ~0.78, Exp 2 (post-fix) stays ~0.51 (it is at chance, so \u03b5 is inert there), Exp 4 0.684 -> ~0.66, Exp 5 0.017 -> ~0.010. Its only cost is compressing Exp 3's contrast.

3. Compensate Exp 3 with the lever that already worked, taken one small step further \u2014 not with more \u03b8. Lower \u03bc0 to [0.0, 0.08] (mid ~0.04). \u03bc0 is arithmetically inert wherever the gate saturates (Exps 2, 4, 6) and, in Exp 3's ASCENDING display, it is the residual leak that props up the lowest-validity cue sitting in column 0 \u2014 exactly what blunts the agree-minus-conflict contrast. Hold \u03b8 where it is (~[0.08, 0.16]); pushing \u03b8 higher again is what flipped Exp 5 positive and inflated Exp 1, so that direction is now closed.

4. One small extra knob for Exp 5's sign: shade the dilution gain down, \u03ba \u2208 [0.7, 1.6] (mid ~1.15). Less dilution differential between the sparse lo-cell and the dense hi-cell shrinks the artificial hi-lo gap and lets the observed slightly-negative value (-0.013) come back, without touching the direction-blindness of the mechanism or reintroducing any coherence bonus.

Guardrails to check before submitting: (a) the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor must stay at ceiling under the upper end of the \u03b8 box combined with the higher \u03b5; (b) Exp 6 must not drift positive \u2014 it is currently the best-fit experiment (-0.143 vs -0.150) and depends on strong column-0 positional dominance, so verify \u03bb and the gate still saturate there; (c) keep the widened \u03b2/\u03b5/\u03b8 boxes for between-subject spread, especially on Exps 1-3 where simulated variance (0.008) is still ~2-4x below observed (0.014-0.037).

**Outcome of this advice:** iter 2 candidate loss=0.0484 -> iter 3 candidate loss=0.0691 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate was REJECTED by the gate (0.0691 vs the running-best 0.0484), so the base remains iter 2. It is still faithful to the arbiter's prescribed family — all-cue compensatory weighted-additive integration, validity-vs-position mixture gated by display readability, monotone faint-praise theta, direction-blind density dilution, no coherence/unanimity premium — and no mechanism drift occurred; this was purely a four-knob calibration pass that traded residuals badly.

Decomposing the four edits against the iter-2 base:
(1) mu1 restored to [1.4,2.2]: WORKED as intended. Exp 2 moved 0.558 -> 0.5225 (Δ +0.048 -> +0.013). The readability gate re-saturating on the scrambled display is the right call and should be kept.
(2) mu0 lowered 0.07 -> 0.04: helped Exp 3 a lot (-0.1715 -> -0.2093, essentially exact) but simultaneously blew up Exp 1 (0.798 -> 0.838, Δ +0.045 -> +0.085). This confirms mu0 is a single shared lever pulling Exps 1 and 3 in opposite directions: less positional leak on ascending/sorted displays = purer validity weighting = more TTB-consistent responding, which Exp 1 does not want.
(3) eps raised / beta ceiling shaved: did NOT deliver the promised Exp 1 relief (it was swamped by the mu0 effect) and cost Exp 6, which drifted from a near-perfect -0.1425 to -0.1133 (Δ +0.0075 -> +0.037). Global lapse is a blunt instrument here: Exp 6's contrast is a difference of cell proportions, so extra lapse compresses it toward zero.
(4) kap lowered 1.5 -> 1.15: my own iter-2 advice, and it had the WRONG SIGN. Exp 5 went 0.017 -> 0.103 (Δ +0.030 -> +0.117), the single largest regression. Less dilution means the dense hi-cell (5/5 vs 4/5, T=0.9) is decompressed and p_hi rises, widening hi-lo, when the data want it slightly negative (-0.013). The correct direction is MORE dilution gain, not less.

So the net picture: one validated win (mu1), one genuine but expensive win (mu0 on Exp 3 at Exp 1's expense), and two clear losses (eps on Exp 6, kap on Exp 5).
Rationale: Regenerate on the UNCHANGED iter-2 accepted base, keeping PWA-DD mechanically identical (no family change, no new terms, no re-introduction of a coherence/unanimity premium). Make a MINIMAL, two-knob edit instead of the four-knob sweep that just got rejected — that sweep failed precisely because coupled levers cancelled.

1. KEEP the mu1 restoration to [1.4, 2.2] (mid ~1.8). This is the one edit in iter 3 that is clearly attributable and clearly positive: Exp 2's scrambled display re-saturates the readability gate and the metric returns from 0.558 to ~0.52. It is arithmetically inert on the monotone displays (Exps 1, 3, 5) and on Exps 4/6, which saturate through the interior-best-cue term anyway.

2. REVERSE my iter-2 kap advice — it had the wrong sign and the gate punished it. Do NOT lower the dilution gain; RAISE it modestly relative to the iter-2 box: kap in roughly [1.1, 2.3] (mid ~1.7) rather than iter-2's [1.0, 2.0] (mid 1.5) or iter-3's [0.7, 1.6]. More dilution compresses the dense hi-cell (5/5 vs 4/5) further, pulling Exp 5's hi-minus-lo down from +0.017 toward the observed -0.013. Take a small step: kap is a global denominator and overshooting it will flatten Exp 1 and Exp 4 as well.

3. REVERT the other two iter-3 edits to their iter-2 values, with one small exception. Put eps and the beta ceiling back to the iter-2 boxes (eps ~[0.03, 0.22], beta ~[2.0, 5.6]) — the extra lapse did not fix Exp 1 and did visibly cost Exp 6, which is currently the best-fitting experiment and must not be sacrificed. For mu0, do not go all the way back to 0.07 nor stay at 0.04: I have now pushed mu0 in both directions across iterations, so split the flanking values and use mu0 in [0.0, 0.11] (mid ~0.055). That should land Exp 3 around -0.19 and Exp 1 around 0.81-0.82, sharing the residual between them rather than concentrating it in Exp 1.

4. Leave theta alone at [0.08, 0.16]. Both the up-direction (flips Exp 5 positive, inflates Exp 1) and further tinkering have now been tested; it is not the productive lever.

If, after this, Exp 1 is still ~0.04-0.05 too high, the next in-family knob to try is NOT global lapse but the validity-sensitivity exponent gamma: shading gamma down (e.g. [0.9, 1.6]) flattens the validity weight profile specifically on sorted displays, which is exactly where Exp 1's excess TTB-consistency comes from, while being largely inert wherever the positional gradient dominates (Exps 2, 4, 6). Flag that as the reserve lever rather than changing it in the same pass.

Guardrails: verify the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor stays at ceiling under the higher kap (the denominator at T=0.5 grows, so check D is still comfortably positive), and confirm Exp 6 returns to ~-0.14 once eps is restored.

**Outcome of this advice:** iter 3 candidate loss=0.0691 -> iter 4 candidate loss=0.0475 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is the new running-best (0.0475) and is fully faithful to the arbiter's prescribed family: all-cue compensatory weighted-additive integration, validity-vs-position mixture gated by display readability, monotone faint-praise \u03b8, direction-blind density dilution, and no coherence/unanimity premium anywhere. It remains the only theory in the comparison set that simultaneously gets Exp 3 clearly negative (-0.177 vs -0.208), Exp 4 moderate (0.658 vs 0.659, essentially exact), and Exp 6 negative (-0.113 vs -0.150) \u2014 the three contrasts that individually falsified pi_1, pi_2/pi_3 and pi_4.

Residuals by experiment: Exp1 +0.023 (best yet), Exp2 +0.024, Exp3 +0.031, Exp4 -0.001 (exact), Exp5 +0.083, Exp6 +0.037. Two of the six now dominate the aggregate loss, and both are REGRESSIONS relative to the iter-2 base rather than untouched weak spots: Exp 5 was +0.030 at iter 2 and is now +0.083; Exp 6 was -0.1425 (residual +0.0075) at iters 1 AND 2 and is now -0.1133 (+0.037). This pass changed three things (mu1 up, kap up, c0 down) but the mu1 change is inert on Exps 5/6 (monotone display / gate saturated through the interior term), and kap up should have pushed Exp 5's hi-minus-lo DOWN. So the c0 reduction (0.09 -> 0.07) is the near-certain culprit for both regressions: shrinking the low-density floor of the denominator sharpens exactly the sparse cells that Exps 5 and 6 are built out of. In Exp 5 the lo-cell (weak single endorsement vs empty panel, T=0.1) is driven further sub-chance by \u03b8 once the tiny denominator amplifies it, so p_hi - p_lo widens positive when the data want ~0. In Exp 6 the U1/V1 anchor cells (single endorsement vs empty, T=0.1) and the ladder cells are all sparse, and over-sharpening them compresses the L1-L4 / U4-U1 / V2-V1 differences toward zero, exactly the observed drift from -0.14 to -0.11.

Between-subject variance is now roughly right on Exps 4-6 (sim 0.010-0.037 vs obs 0.013-0.038) but still ~2x low on Exps 1-3 (sim 0.009-0.013 vs obs 0.014-0.037), and Exp 5's simulated spread (0.037) is now above observed (0.014). This is second-order; the point-estimate residuals on Exps 5 and 6 are the priority.
Rationale: Keep PWA-DD mechanically identical \u2014 no family change, no new terms, no re-introduction of a coherence/unanimity premium, no stopping rule. Make a SINGLE-KNOB edit this time; the four-knob sweep at iter 3 was rejected precisely because coupled levers cancelled, and the current pass shows the same problem in miniature (kap and c0 pushed Exp 5 in opposite directions and c0 won).

1. REVERT c0 to roughly [0.05, 0.13] (mid ~0.09), i.e. back to the iter-2 value, and CHANGE NOTHING ELSE. Keep kap at [1.1, 2.3] (mid 1.7) \u2014 raising it was my iter-3 correction and it is directionally right for Exp 5. Keep mu1 at [1.4, 2.2] \u2014 that is the one edit with a clean, attributable win (Exp 2 back near 0.52). Keep beta, eps, theta, mu0, gamma, lam, mu2 exactly as they are; eps and theta have both been probed in both directions and are closed.

Expected effect: raising the low-density floor pulls the sparse lo-cell in Exp 5 back toward 0.5 (it is currently over-driven sub-chance by \u03b8), so hi-minus-lo should fall from +0.070 toward ~+0.02, and it restores the sparse-cell determinacy gradient in Exp 6, which should return to ~-0.14 (residual +0.0075). The cost is small on Exp 3 (~-0.177 -> ~-0.171, residual +0.031 -> +0.037), and c0 up should also shave Exp 1 slightly, which is currently on the high side (+0.023). Net expected: Exp5 -0.05, Exp6 -0.03, Exp3 +0.006 \u2014 a clear aggregate gain if kap=1.7 is genuinely doing its job, and it also cleanly disambiguates the confounded kap/c0 pass you just ran.

2. If and only if step 1 lands and Exp 3 slips further than ~-0.165, use the reserve lever I flagged last round rather than another global one: shade the validity-sensitivity exponent gamma down slightly (e.g. [0.9, 1.7]). Gamma flattens the validity profile specifically on the monotone displays where Exps 1/3/5 live and is largely inert wherever the positional gradient dominates (Exps 2, 4, 6). Do NOT touch mu0 again \u2014 it is a shared lever that trades Exp 1 against Exp 3 one-for-one and I have now pushed it in both directions.

Guardrails: re-check the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor stays comfortably at ceiling with c0 back at 0.09 combined with kap 1.7 (the T=0.5 denominator is now ~0.94, so verify beta*D still gives p >= 0.78); confirm Exp 4 stays near 0.66 (it is currently exact and must not be sacrificed); and confirm Exp 2 stays near 0.52.

**Outcome of this advice:** iter 4 candidate loss=0.0475 -> iter 5 candidate loss=0.0384 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate is the new running-best (0.0384) and is fully faithful to the arbiter's prescribed family: all-cue compensatory weighted-additive integration, a validity-vs-position weight mixture gated by display readability, a strictly monotone faint-praise discount \u03b8, direction-blind density dilution, and no coherence/unanimity premium or stopping rule anywhere. My iter-4 single-knob advice (revert c0 to ~[0.05,0.13], change nothing else) was validated by the gate: it delivered exactly the predicted repairs on the two sparse-cell experiments, Exp 5 (+0.083 -> +0.061) and Exp 6 (+0.037 -> +0.021), and it also fixed Exp 2 (+0.024 -> +0.002, essentially exact) at only a small cost elsewhere.

Current residuals: Exp1 +0.045, Exp2 +0.002, Exp3 +0.026, Exp4 -0.013, Exp5 +0.061, Exp6 +0.021. Two experiments now carry most of the loss and, importantly, they want the SAME correction. Exp 5's hi-minus-lo is +0.048 vs an observed -0.013: the dense hi-cell (5/5 vs 4/5, T=0.9) is still not compressed enough relative to the sparse lo-cell (weak single endorsement vs empty, T=0.1). Exp 1 is +0.045 too TTB-consistent on the medium-density conflict pairs (T\u22480.4-0.5, e.g. [1,0,0,0,0] vs [0,1,1,1,1]); the model is simply too determinate there. Both cells sit at moderate-to-high endorsement density, so a modest increase in the dilution gain \u03ba pushes both in the right direction simultaneously, whereas the sparse cells that anchor Exps 3 and 6 (T=0.1-0.4) are only weakly touched because the c0 floor now dominates their denominator.

The theory continues to be the only one in the comparison set that jointly reproduces the three diagnostic contrasts that individually falsified the alternatives: Exp 3 clearly negative (-0.182 vs -0.208; pi_2/pi_3 are strongly positive), Exp 4 moderate (0.646 vs 0.659; pi_1 collapses to 0.18), and Exps 5/6 near-zero-to-negative (pi_4 gives +0.71 and +0.46). Between-subject variance is now about right on Exps 4-6 but still ~2-3x low on Exps 1-3 (sim 0.005-0.015 vs obs 0.014-0.037) \u2014 second-order relative to the point-estimate residuals.
Rationale: Keep PWA-DD mechanically identical \u2014 no family change, no new terms, no coherence/unanimity premium, no stopping rule. The single-knob discipline has now been validated twice by the gate (iter 4 c0-only, iter 5 confirmed), while the multi-knob sweep at iter 3 was rejected; stay with one knob.

1. THE EDIT: raise the dilution gain \u03ba from [1.1, 2.3] (mid 1.7) to roughly [1.4, 2.6] (mid ~2.0), and change NOTHING else. This is a small step in a direction the gate has already rewarded (my iter-3 correction of raising \u03ba was part of the accepted iter-4 pass), not an oscillation \u2014 I have never advised lowering \u03ba since that one wrong-signed suggestion at iter 2, which is closed. Mechanistically \u03ba only bites where endorsement density is high, so it is targeted: (a) Exp 5's dense hi-cell (T=0.9, denom ~1.62 -> ~1.89) is compressed further toward 0.5 while the sparse lo-cell (T=0.1, denom ~0.26 -> ~0.29) barely moves, so hi-minus-lo should fall from +0.048 toward ~+0.02 or below; (b) Exp 1's conflict pairs sit at T\u22480.4-0.5, so their hit rate should shave from 0.798 toward ~0.77-0.78. Expected net: Exp5 -0.03, Exp1 -0.02, against a small cost on Exp 3 (whose conflict cells at T=0.2-0.4 will lose a little sharpness, contrast ~-0.182 -> ~-0.175, residual +0.026 -> +0.033) and near-inertness on Exps 2/6, which are chance-level and sparse-cell dominated respectively.

2. EXPLICITLY DO NOT TOUCH: c0 (just validated at [0.05,0.13] \u2014 lowering it demonstrably over-sharpens the sparse cells that Exps 5 and 6 are built from), \u03bc1 (the clean Exp-2 win, now essentially exact at 0.512), \u03bc0 (a shared lever that trades Exp 1 against Exp 3 one-for-one; probed in both directions and closed), \u03b8 and \u03b5 (both probed in both directions and closed; extra global lapse cost Exp 6 last time), \u03bb, \u03bc2, \u03b2.

3. RESERVE LEVER, only if step 1 lands and Exp 3 slips past ~-0.165: shade the validity exponent \u03b3 UP slightly (e.g. [1.3, 2.1]) rather than down \u2014 note that my earlier 'shade \u03b3 down' note would deepen the Exp-1 fix but make Exp 3 shallower, which is the wrong direction now that Exp 3 is the more fragile of the two. Do not change \u03b3 in the same pass as \u03ba.

Guardrails to verify before submitting: (a) the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor must stay comfortably at ceiling with \u03ba=2.0 (T=0.5 gives denom ~1.09, so check \u03b2\u00b7D still yields p \u2273 0.75 on the correct side across the box); (b) Exp 4 must stay near 0.66 \u2014 it is currently at -0.013 and its trials are moderately dense, so if \u03ba=2.0 drags it below ~0.63 take a smaller step (\u03ba mid ~1.85); (c) Exp 2 must stay near 0.51 and Exp 6 must not drift above ~-0.12. Secondarily, if it is free, widen \u03b2 modestly at the top only (e.g. [2.0, 6.0]) to lift the still-too-low between-subject spread on Exps 1-3 \u2014 but only if the point estimates above are unharmed.

**Outcome of this advice:** iter 5 candidate loss=0.0384 -> iter 6 candidate loss=0.0428 -> the gate marked it REJECTED.

### Iteration 6 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate I am grading (iter 6, the single-knob kap raise from mid 1.7 -> 2.0) was REJECTED by the gate (0.0428 vs the running-best 0.0384), so the base stays at iter 5. Mechanically it is still fully faithful to the arbiter's prescribed family 
inc all-cue compensatory weighted-additive integration, validity-vs-position mixture gated by display readability, strictly monotone faint-praise theta, direction-blind density dilution, no coherence/unanimity premium and no stopping rule 
inc so this is purely a calibration failure, not family drift.

Decomposition of the kap edit against the iter-5 base: Exp1 0.798 -> 0.748 (residual +0.045 -> -0.005, a large and genuine win); Exp2 0.512 -> 0.527 (+0.002 -> +0.017); Exp3 -0.182 -> -0.158 (+0.026 -> +0.050); Exp4 0.646 -> 0.636 (-0.013 -> -0.023); Exp5 +0.0475 -> +0.0483 (essentially UNCHANGED, +0.061 -> +0.062); Exp6 -0.129 -> -0.123 (+0.021 -> +0.027). Net: one ~0.04 gain against ~0.055 of spread-out losses.

The crucial diagnostic is that kap did NOTHING for its intended target, Exp 5, and my iter-5 prediction there was sign-wrong. In the hi-cell (5/5 vs 4/5) the numerator is w_weakest - theta, which is NEGATIVE, so p_hi sits just below 0.5. Increasing the denominator on that dense cell pulls p_hi UP toward 0.5, i.e. it WIDENS hi-minus-lo rather than shrinking it (the observed +0.048 barely moved because the two effects nearly cancel). Any lever that compresses a sub-chance dense cell toward 0.5 is the wrong sign for Exp 5. The correct in-family lever for Exp 5 is the floor c0, which acts selectively on the SPARSE lo-cell (weak single endorsement vs empty, T=0.1) that theta drives strongly sub-chance: raising c0 pulls p_lo up toward 0.5 and shrinks hi-minus-lo. This is exactly what the gate already rewarded at iter 4 -> 5, where c0 [0.03,0.11] -> [0.05,0.13] moved Exp5 +0.083 -> +0.061 AND Exp6 +0.037 -> +0.021 AND Exp2 +0.024 -> +0.002, with Exp3 actually getting slightly MORE negative (-0.177 -> -0.182) rather than paying a cost.

The theory itself remains the only one in the comparison set that jointly reproduces the three contrasts that individually falsified the alternatives (Exp3 clearly negative vs pi_2/pi_3's +0.45/+0.62; Exp4 moderate vs pi_1's 0.18; Exps 5/6 near-zero-to-negative vs pi_4's +0.71/+0.46). Between-subject variance is about right on Exps 4-6 and still ~2x low on Exps 1-3, which stays second-order.
Rationale: Rebuild on the UNCHANGED iter-5 accepted base (loss 0.0384), keeping PWA-DD mechanically identical: no family change, no new terms, no coherence/unanimity premium, no stopping rule. Stay with the single-knob discipline that the gate has twice rewarded (iters 4 and 5) and that the four-knob sweep (iter 3) and this last kap step both failed to beat.

1. REVERT kap to the iter-5 box [1.1, 2.3] (mid 1.7). My iter-5 push on kap is now closed: it is arithmetically the wrong sign for Exp 5 (the dense hi-cell numerator w_weakest - theta is negative, so extra dilution moves p_hi UP toward 0.5 and widens hi-minus-lo) and it charged Exp3/Exp2/Exp4/Exp6 for a win it happened to deliver on Exp1. Do not raise kap again, and do not lower it either (that direction was refuted at iter 3).

2. THE EDIT (one knob only): raise the dilution FLOOR c0 from [0.05, 0.13] to roughly [0.08, 0.18] (mid ~0.13), changing nothing else. This is a second, modest step in the one direction the gate has already validated end-to-end. It bites almost exclusively on the sparse cells (T=0.1-0.2), where c0 dominates the denominator, and is nearly inert on dense cells: Exp5's lo-cell denominator goes ~0.26 -> ~0.31 (+19%) so p_lo, currently over-driven sub-chance by theta, rises toward 0.5 and hi-minus-lo should fall from +0.048 toward ~+0.02 or below (observed -0.013), while the hi-cell denominator moves only ~1.62 -> ~1.67. The same mechanism should push Exp6's sparse anchor cells (U1, V1) back toward 0.5, making U4-U1 and V2-V1 more negative and returning the contrast from -0.123/-0.129 toward ~-0.14 (observed -0.150). Exp2 is at chance and stayed unaffected last time; Exp3 empirically did NOT pay a cost when c0 was last raised, so the anticipated 'agree-cell flattening' worry has already been tested and did not materialise at this step size.

3. EXPLICITLY DO NOT TOUCH: mu1 (the clean Exp-2 win, essentially exact at 0.512), mu0 (a shared lever that trades Exp1 against Exp3 one-for-one; probed both ways and closed), theta and eps (both probed both ways; extra global lapse cost Exp6), lam, mu2, beta, gamma.

4. RESERVE, for a LATER pass only if step 2 lands and Exp1 is still ~+0.04 too TTB-consistent: the iter-6 result shows Exp1's excess determinacy is genuinely reachable via kap, so a HALF-step (kap mid ~1.85, i.e. [1.25, 2.45]) may recover part of that 0.04 win at roughly half the collateral cost 
inc but only after c0 has been isolated and scored, never in the same pass.

Guardrails: (a) re-verify the [1,1,1,1,1] vs [0,0,0,0,0] dominance anchor stays comfortably at ceiling with c0 = 0.13 and kap = 1.7 (T=0.5 gives denom ~0.98; check beta*D still yields p >~ 0.75 across the box); (b) Exp4 must not slip below ~0.63 
inc its trials are moderately dense so c0 should be near-inert there; (c) Exp3 must not rise above ~-0.165; (d) if it is free, widen the top of beta only (e.g. [2.0, 6.0]) to lift the still-too-low between-subject spread on Exps 1-3, but only if none of the point estimates above degrade.

**Outcome of this advice:** iter 6 candidate loss=0.0428 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)
The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim above under `## PREVIOUS CANDIDATE (this loop)`. This is the base the loop's accept gate is currently keeping; the most recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) should be applied on top of it. Apply the SMALLEST edit that addresses the critic's diagnosis while staying inside the arbiter's prescribed mechanism family:
  - Re-emit the previous source verbatim, then change ONLY the lines needed to address the critic (a parameter range, a normalization, a softmax temperature, an attention scheme, a gating term, a buggy indexing line, etc.).
  - Keep all unaffected functions, equations, parameter names, and the overall mechanism intact.
  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would do, and do NOT switch mechanism families — that is the arbiter's decision, not yours in this loop.
  - Briefly explain the minimal edit in `rationale`.
If you genuinely believe a larger rewrite is required, you may do one, but justify in `rationale` why the minimal-diff path was insufficient.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- theory: The theoretical claim this theory makes about cognition.
- predict: Python source defining def predict(parameters, state, history) -> np.ndarray.
- policy: Python source defining def policy(probs) -> int.
- parameters: Parameter ranges (e.g. "[1.0, 10.0]") or choices (e.g. "{1, 2}"), sampled per subject.
- rationale: Reasoning behind this theory design; used to explain why this theory is better then the previous ones.


## Response

```json
{
  "theory": "**PWA-DD: Position-Bound Weighted Additive integration with Faint-Praise discounting and Endorsement-Density Dilution.**\n\nChoice in binary-cue environments is always *compensatory*: on every trial the decision maker reads all displayed columns, forms a single weighted difference, and never stops early. There is no lexicographic cascade and no coherence/unanimity premium. Three claims fix the rest of the mechanism.\n\n(1) **Weights are a mixture of validity and display position, and the mixture is governed by how readable the validity-to-column mapping is.** Stated validities must be bound to *columns*. When the display is perfectly monotone in validity (ascending or descending) that binding is trivial and people use validity-derived weights w_val \u221d logit(v_j)^\u03b3 essentially intact (the residual positional leak \u03bc0 is small and person-specific, and in some people is exactly zero). When the column order is not monotone in validity \u2014 and especially when the best expert sits in an interior column \u2014 binding fails and people fall back almost entirely on a purely positional gradient w_pos \u221d (n\u2212j)^\u03bb that decays left-to-right over displayed columns. The mixture weight \u03bc = \u03bc0 + (1\u2212\u03bc0)\u00b7min(1, \u03bc1(1\u2212|\u03c4|) + \u03bc2\u00b7[best cue interior]) is therefore an *environmental* quantity, not a per-experiment fudge: it is inert (\u03bc = \u03bc0) whenever columns are validity-sorted, and it predicts that scrambling a sorted display converts validity-driven behaviour into position-driven behaviour and that re-sorting a scrambled display restores it. Crucially the gate SATURATES: any display that is appreciably non-monotone is read essentially positionally, with no graded validity leak surviving.\n\n(2) **Faint-praise endorsement discounting.** Each positive endorsement carries a small fixed credibility cost \u03b8: an option that collects more '1's is slightly *less* attractive, all else equal. This is strictly monotone in endorsement count \u2014 there is no step at unanimity. It makes a lone weak endorsement no better (usually worse) than an empty panel, and it makes a 5/5 panel no better than a 4/5 panel by exactly the same amount, which is why 'high-coherence' and 'low-coherence' single-cue steps move together rather than apart. Because \u03b8 exceeds the subjective weight of the weakest experts but not that of the strongest, endorsement by a weak expert is actively counter-diagnostic while endorsement by a strong expert still wins.\n\n(3) **Endorsement-density dilution of decision sensitivity.** The same absolute weighted difference is less decisive when both panels are densely endorsed. Evidence is divisively normalised by the mean endorsement density T = (\u03a3a + \u03a3b)/2n: D = [\u03a3_j w_j(a_j\u2212b_j) \u2212 \u03b8(\u03a3a\u2212\u03a3b)] / (c0 + \u03ba\u00b7T). The dilution has TWO separable constants: a floor c0 that sets how crisply *sparse* panels are compared, and a gain \u03ba that sets how much extra mush dense panels accrue. The floor is *not* negligible \u2014 it is in fact appreciable: even near-empty panels are compared with bounded, distinctly sub-ceiling determinacy, so that a lone weak endorsement versus an empty panel lands only slightly below chance rather than far below it, and the graded ladder structure of sparse comparisons is preserved. Heavily endorsed panels are compared mushily. Unlike a consensus premium, dilution is *direction-blind* \u2014 it never confers an advantage on the more coherent option, it only compresses whatever advantage exists \u2014 so it produces the observed shrinkage of a fixed one-cue advantage at high endorsement backgrounds without ever predicting a unanimity bonus.\n\np(A) = (1\u2212\u03b5)\u00b7logistic(\u03b2D) + \u03b5/2. Everyone holds the same stationary rule (no feedback is given, so nothing is learned); heterogeneity is broad and continuous \u2014 in determinacy \u03b2, validity sensitivity \u03b3, discount \u03b8, positional decay \u03bb, dilution (c0, \u03ba), lapse \u03b5 and the readability coefficients \u2014 never discrete strategy switching. The theory therefore predicts substantial between-subject spread in every conflict metric, largest where margins are intermediate.",
  "predict": "def predict(parameters, state, history):\n    # PWA-DD: position-bound weighted additive integration with\n    # faint-praise endorsement discounting and endorsement-density dilution.\n    #\n    #   w_val_j  proportional to logit(v_j) ** gamma      (validity weights)\n    #   w_pos_j  proportional to (n - j) ** lam           (left-to-right gradient)\n    #   mu       = mu0 + (1-mu0)*min(1, mu1*(1-|tau|) + mu2*[best cue interior])\n    #   w        = (1-mu)*w_val + mu*w_pos                (normalised)\n    #   num      = w.(a-b) - theta*(sum a - sum b)        (faint-praise discount)\n    #   T        = (sum a + sum b) / (2n)                 (endorsement density)\n    #   D        = num / (c0 + kap*T)                     (density dilution)\n    #   p(A)     = (1-eps)*logistic(beta*D) + eps/2\n    #\n    # No stopping rule, no coherence/unanimity premium.  History unused:\n    # the rule is stationary because no feedback is provided.\n    import numpy as np\n\n    # ---------------- unpack the two rating vectors --------------------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        if \"option_a_ratings\" in state and \"option_b_ratings\" in state:\n            a = np.asarray(state[\"option_a_ratings\"], dtype=float).ravel()\n            b = np.asarray(state[\"option_b_ratings\"], dtype=float).ravel()\n    if a is None:\n        arr = np.asarray(state, dtype=float)\n        if arr.ndim == 1:\n            half = int(arr.shape[0] // 2)\n            a = arr[:half].astype(float)\n            b = arr[half:2 * half].astype(float)\n        else:\n            arr = arr.reshape(arr.shape[0], -1)\n            a = arr[0].astype(float)\n            b = arr[1].astype(float)\n\n    n = int(min(np.size(a), np.size(b)))\n    if n == 0:\n        return np.array([0.5, 0.5])\n    a = np.asarray(a, dtype=float).ravel()[:n]\n    b = np.asarray(b, dtype=float).ravel()[:n]\n\n    # ---------------- validities ---------------------------------------\n    val = parameters.get(\"validities\", None)\n    if val is None:\n        v = np.linspace(0.90, 0.55, n)\n    else:\n        v = np.asarray(val, dtype=float).ravel().astype(float)\n        if v.shape[0] < n:\n            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])\n        v = v[:n]\n    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)\n\n    # ---------------- parameters ---------------------------------------\n    gamma = float(parameters.get(\"gamma\", 1.5))\n    beta = float(parameters.get(\"beta\", 3.8))\n    theta = float(parameters.get(\"theta\", 0.12))\n    c0 = float(parameters.get(\"c0\", 0.13))\n    kap = float(parameters.get(\"kap\", 1.7))\n    mu0 = float(np.clip(float(parameters.get(\"mu0\", 0.07)), 0.0, 1.0))\n    mu1 = float(parameters.get(\"mu1\", 1.8))\n    mu2 = float(parameters.get(\"mu2\", 0.4))\n    lam = float(parameters.get(\"lam\", 0.95))\n    eps = float(np.clip(float(parameters.get(\"eps\", 0.125)), 0.0, 1.0))\n    c0 = max(c0, 1e-3)\n\n    # ---------------- validity-derived weights --------------------------\n    L = np.log(v / (1.0 - v))\n    L = np.maximum(L, 1e-9)\n    with np.errstate(over=\"ignore\", invalid=\"ignore\"):\n        wv = np.power(L, gamma)\n    wv = np.nan_to_num(wv, nan=0.0, posinf=1e12, neginf=0.0)\n    s = float(np.sum(wv))\n    if (not np.isfinite(s)) or s <= 0.0:\n        wv = np.ones(n, dtype=float) / float(n)\n    else:\n        wv = wv / s\n\n    # ---------------- positional (reading-order) weights ----------------\n    idx = np.arange(n, dtype=float)\n    with np.errstate(over=\"ignore\", invalid=\"ignore\"):\n        wp = np.power(np.maximum(float(n) - idx, 1e-9), lam)\n    wp = np.nan_to_num(wp, nan=0.0, posinf=1e12, neginf=0.0)\n    s = float(np.sum(wp))\n    if (not np.isfinite(s)) or s <= 0.0:\n        wp = np.ones(n, dtype=float) / float(n)\n    else:\n        wp = wp / s\n\n    # ---------------- display readability -------------------------------\n    # Kendall tau between column index and stated validity (ties skipped).\n    conc = 0\n    disc = 0\n    for i in range(n - 1):\n        for j in range(i + 1, n):\n            d = v[j] - v[i]\n            if d > 0:\n                conc += 1\n            elif d < 0:\n                disc += 1\n    tot = conc + disc\n    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)\n    incong = 1.0 - abs(tau)                     # 0 when display is monotone\n    jmax = int(np.argmax(v))\n    interior = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0\n\n    mu = mu0 + (1.0 - mu0) * min(1.0, max(0.0, mu1 * incong + mu2 * interior))\n    mu = float(np.clip(mu, 0.0, 1.0))\n\n    w = (1.0 - mu) * wv + mu * wp\n    s = float(np.sum(w))\n    if (not np.isfinite(s)) or s <= 0.0:\n        w = np.ones(n, dtype=float) / float(n)\n    else:\n        w = w / s\n\n    # ---------------- evidence ------------------------------------------\n    core = float(np.dot(w, a - b))\n    cnt_a = float(np.sum(a))\n    cnt_b = float(np.sum(b))\n    num = core - theta * (cnt_a - cnt_b)          # faint-praise discount\n\n    dens = (cnt_a + cnt_b) / (2.0 * float(n))     # endorsement density in [0,1]\n    denom = c0 + kap * dens\n    if (not np.isfinite(denom)) or denom <= 1e-6:\n        denom = 1e-6\n    D = num / denom\n\n    # ---------------- logistic choice with lapse -------------------------\n    z = np.array([beta * D, 0.0], dtype=float)\n    z = z - np.max(z)\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n\n    p = (1.0 - eps) * p_core + eps * 0.5\n    p = np.clip(p, 1e-12, 1.0)\n    p = p / p.sum()\n    return p",
  "policy": "def policy(probs):\n    import numpy as np\n    p = np.asarray(probs, dtype=np.float64).ravel()\n    p = np.clip(p, 0.0, None)\n    tot = p.sum()\n    if (not np.isfinite(tot)) or tot <= 0.0:\n        p = np.ones_like(p) / float(len(p))\n    else:\n        p = p / tot\n    return int(np.random.choice(len(p), p=p))",
  "parameters": {
    "gamma": "[1.1, 1.9]",
    "beta": "[2.0, 5.6]",
    "theta": "[0.08, 0.16]",
    "c0": "[0.08, 0.18]",
    "kap": "[1.1, 2.3]",
    "mu0": "[0.0, 0.14]",
    "mu1": "[1.4, 2.2]",
    "mu2": "[0.20, 0.60]",
    "lam": "[0.70, 1.20]",
    "eps": "[0.03, 0.22]",
    "validities": "validities"
  },
  "rationale": "Strict single-knob edit exactly as the critic prescribed on top of the running-best iter-5 base. `predict`/`policy` are mechanically byte-identical to iter 5 except for one inert `.get()` default (c0 0.09 -> 0.13, used only if the sampler ever omitted the key); ONE parameter box changes and every other box (beta, eps, theta, mu0, mu1, mu2, gamma, lam, kap) is left untouched.\n\n**The edit: c0 [0.05, 0.13] -> [0.08, 0.18] (mid 0.09 -> 0.13).** kap stays at [1.1, 2.3] (mid 1.7) — the iter-6 kap raise was rejected and the critic's arithmetic shows why it was sign-wrong for its intended target: in Exp 5's dense hi-cell (5/5 vs 4/5) the numerator w_weakest − θ is NEGATIVE, so extra dilution pulls p_hi UP toward 0.5 and WIDENS hi-minus-lo. That direction is now closed in both senses.\n\nWhy c0 is the right lever: c0 is the floor of the denominator c0 + kap·T, so it dominates precisely in the SPARSE cells (T = 0.1–0.2) out of which Exps 5 and 6 are built, and is nearly inert on dense cells. This is the one direction the gate has already validated end-to-end (iter 4 -> 5: c0 [0.03,0.11] -> [0.05,0.13] moved Exp5 +0.083 -> +0.061, Exp6 +0.037 -> +0.021, Exp2 +0.024 -> +0.002, and Exp3 got slightly MORE negative rather than paying a cost).\n\nHand-check on the mid-box (theta=0.12, kap=1.7, beta=3.8, eps=0.125). Exp 5 lo-cell (weak single endorsement vs empty, T=0.1): denom 0.09+0.17=0.26 -> 0.13+0.17=0.30, so the θ-driven sub-chance drive |num|/denom shrinks by ~13% and p_lo rises toward 0.5; the hi-cell (T=0.9) has denom 1.62 -> 1.66, essentially unchanged. Hence hi-minus-lo should fall from +0.048 toward ~+0.02 or below (observed −0.013). Exp 6's U1/V1 anchors (single endorsement vs empty, T=0.1) get the same de-sharpening, making U4−U1 and V2−V1 more negative and returning the contrast from −0.129 toward ~−0.14 (observed −0.150). Exp 2 sits at chance so c0 is near-inert; Exp 4's trials are moderately dense so it should hold near 0.65.\n\nGuardrail re-check: dominance anchor [1,1,1,1,1] vs [0,0,0,0,0] has num = 1 − 5·0.12 = 0.40, T=0.5, denom = 0.13 + 0.85 = 0.98, D = 0.408, beta·D = 1.55 -> p_core = 0.825, ≈0.78 after the mid lapse — comfortably at ceiling on the correct side, and still ≥0.75 at the top of the c0 box (denom 1.03, p ≈ 0.77).\n\n**Explicitly NOT touched:** mu1 (the clean Exp-2 win, 0.512), mu0 (shared Exp1/Exp3 lever, probed both ways and closed), theta and eps (both probed both ways; extra lapse cost Exp 6), lam, mu2, beta, gamma. Isolating c0 also keeps the pass cleanly attributable, which is what the two ACCEPTED single-knob iterations did and what the rejected multi-knob sweeps did not."
}
```

## Usage

```json
{
  "input_tokens": 44418,
  "output_tokens": 6456
}
```
