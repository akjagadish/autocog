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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Replace THEORY 1 with an 'anchored compensatory integration' theory that fuses the surviving parts of both incumbents and, critically, is calibrated to produce intermediate rates rather than ceiling/floor:

1. Cue weights are the PRODUCT of a validity term and an attention term: w_j ∝ (v_j - 0.5)^rho * a_j, with rho near 1 (mild compression/expansion allowed, ~0.8-1.4) so that a crowd of 3 mid cues can out-weigh a single .95 cue (needed for E4's .775 and E3's small positive contrast).
2. Attention term a_j = 1 + alpha * exp(-lambda * position_j) (or simply a boost on the first one or two screen positions), with alpha of moderate size (~0.5-1.5) and lambda ~0.5-1.5. This first-cue anchor must be strong enough to flip validity ties (E1's .82 toward the pos-0 side, E2's .84 toward the pos-0/pos-5 side) but NOT strong enough to override a large validity difference (E5/E6 must stay near .5). A small optional recency term is permissible but must be unimodal jitter, not a 35% discrete recency subpopulation — the low between-subject variance in E5/E6 (.0075-.023) rules out bimodality.
3. Evidence is summed over ALL cues and is NOT normalised by the discriminating weight: E = sum_j w_j (a_j - b_j). Choice = softmax(beta * E) with a lapse epsilon. Because E is unnormalised, cells where the top cue is nearly balanced against a two-cue crowd land near chance (E5 .52, E6 .51) while cells with a real margin land at .75-.90 (E2, E4) — this is the key property both incumbents got wrong in opposite directions.
4. No early stopping / no threshold rule, and no conclusion-graded commitment: the refuted prediction (unresolved trials tilt toward the OPPOSITE option) must be dropped. Confidence is purely margin-graded.
5. Parameter ranges should be chosen so that simulated metric values fall in the .55-.90 band for E1/E2, .45-.60 for E5/E6, .70-.85 for E4 and .03-.15 for E3 — i.e. the model should never predict values above ~.95 or below ~.10 on these designs.

This theory is a direct competitor to pi_3: it keeps a position effect but makes it a graded weight modifier on top of validity rather than the sole driver, and it removes the normalisation that forces pi_3 to ceiling everywhere.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.288)

**Description**
**Threshold-terminated compensatory integration with CONCLUSION-GRADED commitment.**

People treat expert ratings as evidence to be added up, not as a lexicographic hierarchy. Five claims:

1. **Cue weights depend ONLY on the communicated validity, never on where the rating sits on the screen.** The weight of expert j is a compressed, monotone function of its stated validity, w_j ∝ (v_j − 0.5)^rho, normalised to sum to 1. Compression (rho ≈ 1.25–1.45) lets a crowd of medium-validity experts out-vote or tie a single .95 expert.

2. **Integration is compensatory and additive, carried out incrementally in the order the ratings are displayed, and it can terminate early.** After each rating that actually discriminates, the decision maker performs a *best-case reversal check*: "could any single expert I have not yet read still overturn what I now believe?" The sufficiency bar is theta_k = phi · max_{j after the current screen position} w_j, and the probability of stopping is sigmoid(s·(|E_k|/theta_k − 1)). With phi ≈ 1 this is the rational rule "stop when no single remaining expert can reverse my lead". Nothing is discounted by position; position matters *only* through this truncation.

3. **NEW — confidence is conclusion-graded, not merely margin-graded.** Two psychologically distinct states can end a trial. (a) The reversal check *passes*: the person has reached a categorical conclusion ("nothing still unread can overturn this"), and choice is committed sharply — softmax on the raw margin with an amplified sharpness beta·kappa (kappa > 1). Crucially, *which* cue triggered the stop is then almost irrelevant: a mid-validity cue that has already cleared the sufficiency bar is followed about as decisively as the top cue is. (b) The list is *exhausted* without the check ever passing: the person never reached a conclusion and is left holding a raw, ungrounded residual margin; choice is then a shallow softmax on that margin with the ordinary beta, so unresolved trials sit close to chance and are only weakly tilted by the residual. Two trials with identical final margins therefore differ in decisiveness depending on whether the stopping criterion was ever met — the theory's sharpest new prediction.

4. **No renormalisation, no side bias, plus a lapse.** Evidence is never divided by the total discriminating weight (that would make one-cue and many-cue decisions equally confident); a lapse epsilon mixes in a uniform guess.

5. **Individual differences are unimodal jitter** in compression (rho), conservatism (phi), stopping sharpness (s), commitment amplification (kappa), residual sensitivity (beta) and lapse — no discrete strategy types, no primacy/recency "reader types".

Signature predictions: in a validity-sorted display the leading cue clears the bar immediately, search stops, and behaviour looks like Take-The-Best in BOTH conflict and agreement cells — and because stopping commitment is amplified, cells whose first cues tie and which stop on a *middle* cue are followed just as decisively, so the TTB-vs-tally contrast collapses to ≈0 instead of going negative. When the best expert is listed last, the bar stays high until a mid-validity crowd piles past it, search stops before the best expert is read, and behaviour looks like tallying. When a strong cue sits mid-list with another strong cue still unread, the check fails, integration runs to the end, and the shallow residual softmax leaves the choice near chance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b

    # ------------------------------------------------------------------
    # 2. validity-only cue weights, compressed toward equality
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fall back: assume display is validity-sorted, descending
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0)

    rho = float(parameters.get('rho', 1.35))
    rho = float(np.clip(rho, 0.1, 5.0))
    raw = np.clip(val - 0.5, 1e-9, None) ** rho
    tot = float(np.sum(raw))
    if not np.isfinite(tot) or tot <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = raw / tot                      # weights sum to 1, position-free

    phi = float(np.clip(parameters.get('phi', 1.05), 0.05, 3.0))
    s = float(np.clip(parameters.get('s', 20.0), 0.1, 200.0))
    beta = float(np.clip(parameters.get('beta', 7.5), 0.0, 200.0))
    kappa = float(np.clip(parameters.get('kappa', 4.0), 1.0, 20.0))
    eps = float(np.clip(parameters.get('epsilon', 0.0), 0.0, 1.0))

    # ------------------------------------------------------------------
    # 3. incremental integration in screen order with graded stopping
    #    sufficiency bar = phi * (largest weight among cues NOT YET READ)
    #    i.e. "can any single remaining expert still overturn my lead?"
    #    Commitment is CONCLUSION-GRADED: a stop that passes the check is
    #    read out with sharpness beta*kappa; an exhausted list (no
    #    conclusion ever reached) is read out with the shallow beta.
    # ------------------------------------------------------------------
    disc = np.nonzero(np.abs(d) > 1e-12)[0]     # discriminating cues, in order
    if disc.size == 0:
        return np.array([0.5, 0.5])

    steps = w[disc] * np.sign(d[disc])
    cum = np.cumsum(steps)                      # margin after each new reason
    m = cum.shape[0]

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    p_a = 0.0
    remain = 1.0
    for k in range(m):
        E = float(cum[k])
        if k < m - 1:
            idx = int(disc[k])
            if idx + 1 < n:
                suffix_max = float(np.max(w[idx + 1:]))
            else:
                suffix_max = 0.0
            theta_k = max(phi * suffix_max, 1e-12)
            q = _sig(s * (abs(E) / theta_k - 1.0))   # graded termination
            p_choice = _sig(beta * kappa * E)        # committed conclusion
        else:
            q = 1.0                                  # nothing left to read
            p_choice = _sig(beta * E)                # unresolved residual
        p_a += remain * q * p_choice
        remain *= (1.0 - q)
        if remain <= 1e-12:
            break
    if remain > 1e-12:                           # numerical remainder
        p_a += remain * _sig(beta * float(cum[-1]))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5              # lapse, no side bias
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- rho: [1.25, 1.45]
- phi: [1.00, 1.12]
- s: [12.0, 30.0]
- beta: [5.0, 10.0]
- kappa: [3.0, 5.0]
- epsilon: [0.0, 0.06]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0933 (var=0.0064) vs this=-0.0050 (var=0.0008)
- Experiment 2: real=0.7750 (var=0.0072) vs this=0.7860 (var=0.0029)
- Experiment 3: real=0.5175 (var=0.0233) vs this=0.5200 (var=0.0187)
- Experiment 4: real=0.5075 (var=0.0075) vs this=0.4906 (var=0.0065)
- Experiment 5: real=0.8200 (var=0.0126) vs this=0.3500 (var=0.0175)
- Experiment 6: real=0.8413 (var=0.0134) vs this=0.3312 (var=0.0055)


---

### `pi_3` (overall score: 0.164)

**Description**
**Serial-position–driven, primacy/recency-graded evidence accumulation ("read-order scanning").**

People do not re-sort the expert ratings by the stated validities before deciding. They inspect the ratings in the order in which they appear on the screen (reading order) and their attention decays sharply across that inspection sequence, so the evidence carried by a cue is discounted geometrically by its *serial position*, not by its validity. Stated validities exert only a weak, partial pull on the inspection order (parameter gamma, small).

Two further claims:

1. **The gradient is steep but graded.** Weights fall off as w_j = exp(-lambda * rank_j) with |lambda| well above zero, so decisions are usually driven by the first cue(s) that discriminate — near-lexicographic but not strictly all-or-none: when the leading cues tie, the next-inspected cues take over, and when several later cues line up they can occasionally overturn a leading cue if the gradient is shallow.

2. **The gradient has a sign: most readers are primacy-oriented, a substantial minority are recency-oriented.** Reading a short list produces either an anchoring-on-the-first-item strategy (primacy, lambda > 0) or a last-impression-counts strategy (recency, lambda < 0). The population is bimodal: about one third of subjects anchor on the *last-read* cue. Near-zero gradients (i.e., genuine equal-weight tallying) are essentially absent — people do not integrate all six/five cues equally.

Evidence is evaluated *relatively*: the signed weighted difference is normalised by the total weight of the cues that actually discriminate, so a decision based on one early cue is as confident as a decision based on many, and only genuinely near-balanced weighted evidence produces hesitation. Choice is a softmax on this relative evidence with inverse temperature beta, plus a lapse epsilon.

Because validity order and screen order coincide in a validity-sorted display but dissociate in a scrambled display, this theory predicts TTB-looking behaviour in the former and tally-looking behaviour in the latter — without either heuristic being the actual mechanism. The causal driver is *where the cue sits on the screen*.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---- unpack the trial stimulus -> (2, n_features) ------------------
    if isinstance(state, dict):
        a = np.asarray(list(state.get('option_a_ratings', [])), dtype=float)
        b = np.asarray(list(state.get('option_b_ratings', [])), dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            stim = stim.reshape(2, -1)
        elif stim.ndim > 2:
            stim = stim.reshape(2, -1)
    if stim.shape[0] != 2:
        stim = stim.reshape(2, -1)

    a = stim[0].astype(float)
    b = stim[1].astype(float)
    n = a.shape[0]
    d = a - b

    # ---- inspection ranks: screen position, weakly pulled by validity --
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fallback: assume the display is already validity-sorted
        val = np.linspace(0.95, 0.55, n)

    order = np.argsort(-val, kind='stable')       # 0 = highest validity
    vrank = np.empty(n, dtype=float)
    vrank[order] = np.arange(n, dtype=float)
    pos = np.arange(n, dtype=float)               # reading-order index

    gamma = float(parameters.get('gamma', 0.0))
    gamma = min(max(gamma, 0.0), 1.0)
    rank = gamma * vrank + (1.0 - gamma) * pos

    # ---- signed attention gradient (primacy vs recency reader) --------
    lam_mag = float(parameters.get('lam', 2.0))
    orient = float(parameters.get('scan_orientation', 1.0))
    RECENCY_SHARE = 0.35                          # population constant
    lam = -lam_mag if orient < RECENCY_SHARE else lam_mag

    z = -lam * (rank - float(np.mean(rank)))      # centered for stability
    z = z - float(np.max(z))
    w = np.exp(z)

    # ---- relative (normalised) weighted evidence ----------------------
    num = float(np.sum(w * d))
    den = float(np.sum(w * np.abs(d)))
    if den <= 1e-12:
        S = 0.0                                   # nothing discriminates
    else:
        S = num / den                             # in [-1, 1]

    beta = float(parameters.get('beta', 6.0))
    eps = float(parameters.get('epsilon', 0.0))
    eps = min(max(eps, 0.0), 1.0)

    logits = np.array([beta * S, 0.0], dtype=float)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    p = e / np.sum(e)

    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- lam: [1.2, 3.5]
- gamma: [0.0, 0.25]
- beta: [4.0, 12.0]
- epsilon: [0.0, 0.10]
- scan_orientation: [0, 1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0933 (var=0.0064) vs this=0.1027 (var=0.0215)
- Experiment 2: real=0.7750 (var=0.0072) vs this=0.7780 (var=0.0646)
- Experiment 3: real=0.5175 (var=0.0233) vs this=0.0375 (var=0.0022)
- Experiment 4: real=0.5075 (var=0.0075) vs this=0.0269 (var=0.0009)
- Experiment 5: real=0.8200 (var=0.0126) vs this=0.9513 (var=0.0036)
- Experiment 6: real=0.8413 (var=0.0134) vs this=0.9694 (var=0.0014)


---

### `pi_2` (overall score: 0.051)

**Description**
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


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0933 (var=0.0064) vs this=-0.6931 (var=0.0366)
- Experiment 2: real=0.7750 (var=0.0072) vs this=0.8660 (var=0.0083)
- Experiment 3: real=0.5175 (var=0.0233) vs this=0.1400 (var=0.0151)
- Experiment 4: real=0.5075 (var=0.0075) vs this=0.1450 (var=0.0133)
- Experiment 5: real=0.8200 (var=0.0126) vs this=0.4850 (var=0.0117)
- Experiment 6: real=0.8413 (var=0.0134) vs this=0.8825 (var=0.0110)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2158 -> ACCEPTED
- iter 2: loss=0.2242 -> REJECTED
- iter 3: loss=0.2310 -> REJECTED
- iter 4: loss=0.1834 -> ACCEPTED
- iter 5: loss=0.1278 -> ACCEPTED
- iter 6: loss=0.1066 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 6 at loss=0.1066 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    dis_hits = 0.0
    dis_n = 0.0
    agr_hits = 0.0
    agr_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.size == 0:
            continue

        # Take-The-Best winner: features are listed in descending validity
        # order, so scan left to right for the first discriminating cue.
        ttb_winner = None
        for j in range(a.size):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally-tie pairs excluded (uninformative for the contrast)
        tally_winner = 0 if a_wins > b_wins else 1

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hit = 1.0 if r == ttb_winner else 0.0

        if tally_winner != ttb_winner:
            dis_hits += hit
            dis_n += 1.0
        else:
            agr_hits += hit
            agr_n += 1.0

    if dis_n == 0.0 or agr_n == 0.0:
        return float('nan')

    return float(dis_hits / dis_n - agr_hits / agr_n)
```

**Observed (real) value:** 0.0933 (var=0.0064)
**Previous candidate values (this loop):**
  - iter 1: -0.1600 (var=0.0029) (Δ vs real -0.2533)
  - iter 2: -0.1606 (var=0.0029) (Δ vs real -0.2540)
  - iter 3: -0.0910 (var=0.0023) (Δ vs real -0.1844)
  - iter 4: -0.2167 (var=0.0026) (Δ vs real -0.3100)
  - iter 5: -0.2102 (var=0.0044) (Δ vs real -0.3035)
  - iter 6 (most recent): -0.1679 (var=0.0045) (Δ vs real -0.2613)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0038 (var=0.0080)
- pi_2: -0.6931 (var=0.0366)
- pi_3: 0.1027 (var=0.0215)
- pi_4: -0.0050 (var=0.0008)

### Experiment 2
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.9, 0.6, 0.8, 0.55, 0.7, 0.95], dtype=float)

    n_match = 0.0
    n_total = 0.0

    for _, row in data.iterrows():
        a = np.asarray(list(row['option_a_ratings']), dtype=float)
        b = np.asarray(list(row['option_b_ratings']), dtype=float)
        n = a.shape[0]
        if validities.shape[0] == n:
            val = validities
        else:
            # fallback: assume given order is already descending validity
            val = np.arange(n, 0, -1, dtype=float)
        order = np.argsort(-val, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        margin = abs(a_wins - b_wins)
        if margin < 2:
            continue

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
            continue  # only conflict trials

        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        n_total += 1.0
        if resp == tally_winner:
            n_match += 1.0

    if n_total == 0:
        return float('nan')
    return float(n_match / n_total)
```

**Observed (real) value:** 0.7750 (var=0.0072)
**Previous candidate values (this loop):**
  - iter 1: 0.7220 (var=0.0034) (Δ vs real -0.0530)
  - iter 2: 0.6960 (var=0.0040) (Δ vs real -0.0790)
  - iter 3: 0.6655 (var=0.0050) (Δ vs real -0.1095)
  - iter 4: 0.7635 (var=0.0026) (Δ vs real -0.0115)
  - iter 5: 0.7845 (var=0.0028) (Δ vs real +0.0095)
  - iter 6 (most recent): 0.7690 (var=0.0043) (Δ vs real -0.0060)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8660 (var=0.0083)
- pi_1: 0.1530 (var=0.0129)
- pi_3: 0.7780 (var=0.0646)
- pi_4: 0.7860 (var=0.0029)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Rate of choosing the option favoured by the HIGHEST-VALIDITY cue (screen
    # position 3, validity .95) on the 'double-dissociation' pairs, i.e. trials
    # where the two screen-extreme cues (positions 0 and 5) jointly favour one
    # option while the middle top-validity cue favours the other.
    num = 0.0
    den = 0.0
    num_all = 0.0
    den_all = 0.0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        if a.shape[0] < 6 or b.shape[0] < 6:
            continue
        d3 = a[3] - b[3]
        if d3 == 0:
            continue
        top_side = 0 if d3 > 0 else 1
        try:
            resp = int(r)
        except Exception:
            continue
        chose_top = 1.0 if resp == top_side else 0.0
        num_all += chose_top
        den_all += 1.0
        d0 = a[0] - b[0]
        d5 = a[5] - b[5]
        if d0 != 0 and d5 != 0:
            s0 = 1.0 if d0 > 0 else -1.0
            s5 = 1.0 if d5 > 0 else -1.0
            s3 = 1.0 if d3 > 0 else -1.0
            if s0 == s5 and s0 == -s3:
                num += chose_top
                den += 1.0
    if den > 0:
        return float(num / den)
    if den_all > 0:
        return float(num_all / den_all)
    return float('nan')
```

**Observed (real) value:** 0.5175 (var=0.0233)
**Previous candidate values (this loop):**
  - iter 1: 0.6475 (var=0.0197) (Δ vs real +0.1300)
  - iter 2: 0.4075 (var=0.0249) (Δ vs real -0.1100)
  - iter 3: 0.4938 (var=0.0221) (Δ vs real -0.0237)
  - iter 4: 0.4788 (var=0.0198) (Δ vs real -0.0387)
  - iter 5: 0.4437 (var=0.0163) (Δ vs real -0.0737)
  - iter 6 (most recent): 0.6450 (var=0.0132) (Δ vs real +0.1275)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8562 (var=0.0143)
- pi_3: 0.0375 (var=0.0022)
- pi_2: 0.1400 (var=0.0151)
- pi_4: 0.5200 (var=0.0187)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    VAL = np.array([0.78, 0.60, 0.95, 0.88, 0.72], dtype=float)

    def ttb_winner(a, b, val):
        order = np.argsort(-val, kind='stable')
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    def scan_sign(d, lam):
        n = d.shape[0]
        pos = np.arange(n, dtype=float)
        z = -lam * (pos - pos.mean())
        z = z - z.max()
        w = np.exp(z)
        num = float(np.sum(w * d))
        den = float(np.sum(w * np.abs(d)))
        if den <= 1e-12:
            return 0.0
        return num / den

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
        except Exception:
            continue
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        val = VAL if n == VAL.shape[0] else np.linspace(0.95, 0.55, n)
        w_ttb = ttb_winner(a, b, val)
        if w_ttb is None:
            continue
        d = a - b
        s_prim = scan_sign(d, 2.0)    # primacy reader (lambda > 0)
        s_rec = scan_sign(d, -2.0)    # recency reader (lambda < 0)
        if abs(s_prim) < 0.15 or abs(s_rec) < 0.15:
            continue
        # scanner choice: S > 0 -> A (0), S < 0 -> B (1)
        c_prim = 0 if s_prim > 0 else 1
        c_rec = 0 if s_rec > 0 else 1
        # keep only cells where BOTH scanning orientations oppose TTB
        if c_prim == c_rec and c_prim != w_ttb:
            try:
                r = int(row['response'])
            except Exception:
                continue
            hits.append(1.0 if r == w_ttb else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.5075 (var=0.0075)
**Previous candidate values (this loop):**
  - iter 1: 0.4619 (var=0.0075) (Δ vs real -0.0456)
  - iter 2: 0.4225 (var=0.0108) (Δ vs real -0.0850)
  - iter 3: 0.5938 (var=0.0114) (Δ vs real +0.0863)
  - iter 4: 0.3506 (var=0.0088) (Δ vs real -0.1569)
  - iter 5: 0.3287 (var=0.0082) (Δ vs real -0.1787)
  - iter 6 (most recent): 0.4656 (var=0.0070) (Δ vs real -0.0419)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0269 (var=0.0009)
- pi_1: 0.8475 (var=0.0132)
- pi_2: 0.1450 (var=0.0133)
- pi_4: 0.4906 (var=0.0065)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != 5 or b.shape[0] != 5:
            continue
        d = a - b
        # target cell: d = +-[1,-1,-1,1,0]  (pairs 1 and 2)
        pattern_ok = (
            abs(d[4]) < 1e-9
            and abs(abs(d[0]) - 1.0) < 1e-9
            and abs(abs(d[1]) - 1.0) < 1e-9
            and abs(abs(d[2]) - 1.0) < 1e-9
            and abs(abs(d[3]) - 1.0) < 1e-9
            and abs(d[0] - d[3]) < 1e-9
            and abs(d[1] - d[2]) < 1e-9
            and abs(d[0] + d[1]) < 1e-9
        )
        if not pattern_ok:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        # side supported by expert positions 0 and 3 (the first- and last-
        # discriminating cues); opposed by positions 1 and 2
        favored_is_a = d[0] > 0
        chose_favored = (resp == 0) if favored_is_a else (resp == 1)
        hits += 1 if chose_favored else 0
        tot += 1

    if tot == 0:
        return float('nan')
    return float(hits) / float(tot)

```

**Observed (real) value:** 0.8200 (var=0.0126)
**Previous candidate values (this loop):**
  - iter 1: 0.7738 (var=0.0123) (Δ vs real -0.0462)
  - iter 2: 0.7388 (var=0.0109) (Δ vs real -0.0812)
  - iter 3: 0.8125 (var=0.0095) (Δ vs real -0.0075)
  - iter 4: 0.7850 (var=0.0088) (Δ vs real -0.0350)
  - iter 5: 0.7863 (var=0.0149) (Δ vs real -0.0337)
  - iter 6 (most recent): 0.7675 (var=0.0141) (Δ vs real -0.0525)
**Other theories' values on this metric (for reference):**
- pi_4: 0.3500 (var=0.0175)
- pi_3: 0.9513 (var=0.0036)
- pi_1: 0.8250 (var=0.0169)
- pi_2: 0.4850 (var=0.0117)

### Experiment 6
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != b.shape[0] or a.shape[0] < 6:
            continue
        d = a - b
        disc = [int(i) for i in np.nonzero(np.abs(d) > 1e-9)[0]]
        target = None
        # Class (2): d nonzero at {0,2,5}, first and last agree, mid-list top cue opposes
        if disc == [0, 2, 5]:
            if d[0] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        # Class (3): d nonzero at {0,1,2,5}, three cues agree, single top cue opposes
        elif disc == [0, 1, 2, 5]:
            if d[0] == d[1] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        if target is None:
            continue
        n += 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp == target:
            hits += 1
    if n == 0:
        return 0.5
    return float(hits) / float(n)

```

**Observed (real) value:** 0.8413 (var=0.0134)
**Previous candidate values (this loop):**
  - iter 1: 0.3337 (var=0.0099) (Δ vs real -0.5075)
  - iter 2: 0.3906 (var=0.0074) (Δ vs real -0.4506)
  - iter 3: 0.0912 (var=0.0050) (Δ vs real -0.7500)
  - iter 4: 0.5106 (var=0.0126) (Δ vs real -0.3306)
  - iter 5: 0.8344 (var=0.0061) (Δ vs real -0.0069)
  - iter 6 (most recent): 0.8488 (var=0.0066) (Δ vs real +0.0075)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9694 (var=0.0014)
- pi_4: 0.3312 (var=0.0055)
- pi_1: 0.1388 (var=0.0136)
- pi_2: 0.8825 (var=0.0110)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Anchored compensatory integration with load-gated degradation of weighted-additive processing.**

(1) *Diagnosticity is read convexly.* The subjective pull of expert j is d_j = (v_j - 0.5)^rho with rho ~2.3-2.7, so a .95 expert is treated as a near-oracle and .55-.65 experts as near-noise.

(2) *Screen position is a graded attention multiplier, not a search order.* a_j = 1 + alpha*exp(-lam*j) with alpha ~1.5-2.1, lam ~0.9-1.3; the cue weight is the PRODUCT w_j ∝ d_j * a_j, rescaled once to sum to 1 (never renormalised per trial). The first-read cue is rehearsed longest and anchors the running impression, breaking validity ties toward the pos-0 side, but it cannot overturn a genuine diagnosticity gap.

(3) **Processing load is the single switch that degrades weighted-additive integration toward unit-weight counting.** Features on which BOTH products received a positive expert rating carry no discriminating information, yet they are still read, held and compared; they inflate the size of the comparison and make the two options feel similar. Let L be the number of such shared endorsements. Under this load TWO things happen together, both expressions of the same effort–accuracy tradeoff:
  (a) the diagnosticity exponent flattens, rho_eff = rho / (1 + c*L), so validity differentiation degrades toward equal weighting;
  (b) the decision maker falls back on the crude 'how many experts back each side' cue, i.e. supra-additive corroboration appears — each side's summed support is multiplied by k^delta_eff with delta_eff = delta * L/(1+L).
Crucially, corroboration is **silent on sparse displays (L = 0)**: when no expert endorsed both products the comparison is small, the decision maker stays fully weighted-additive, and a mere numerical majority of low-validity, post-anchor cues buys nothing. Counting is a *load-induced shortcut*, not a permanent feature of integration. This is the theory's sharpest claim: the crowd-following bonus and the flattening of validity weights are one and the same phenomenon, gated by display density, and therefore appear and disappear together within subject.

(4) Evidence is the ABSOLUTE (unnormalised) margin E = Score_A - Score_B; choice = logistic(beta*E) with a lapse epsilon. No early stopping, no threshold rule, no conclusion-graded commitment, no per-trial normalisation, and individual differences are unimodal jitter in rho, alpha, lam, delta, c, beta and lapse — no strategy subpopulations, no primacy/recency reader types.

Signature prediction: two trials identical in validity profile, screen positions and cue counts (crowd = first cue + last cue vs a lone mid-list top-validity cue) are decided at chance when the display is sparse and strongly pro-crowd when several experts endorsed both products — and the *size* of the pro-crowd shift grows with L through both channels simultaneously.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. unpack the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b

    if not np.any(np.abs(d) > 1e-12):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> convex diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fallback: assume the display is validity-sorted, descending
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0)

    rho = float(np.clip(parameters.get('rho', 2.5), 0.1, 8.0))
    alpha = float(np.clip(parameters.get('alpha', 1.8), 0.0, 6.0))
    lam = float(np.clip(parameters.get('lam', 1.1), 0.01, 6.0))
    beta = float(np.clip(parameters.get('beta', 7.0), 0.0, 200.0))
    eps = float(np.clip(parameters.get('epsilon', 0.05), 0.0, 1.0))
    delta = float(np.clip(parameters.get('delta', 0.40), 0.0, 2.0))
    c_load = float(np.clip(parameters.get('load_flatten', 0.85), 0.0, 3.0))

    # --- overlapping positive endorsements = processing load.
    # L = number of features on which BOTH options were endorsed.  These
    # cues carry no discriminating information but they enlarge the
    # comparison problem, and under that load validity differentiation
    # degrades toward unit-weight counting.
    L = float(np.count_nonzero((a > 0.5) & (b > 0.5)))
    rho_eff = rho / (1.0 + c_load * L)
    rho_eff = float(np.clip(rho_eff, 0.6, 8.0))

    diag = np.power(np.clip(val - 0.5, 1e-12, None), rho_eff)   # diagnosticity

    # ------------------------------------------------------------------
    # 3. graded primacy attention multiplier (screen-position anchor)
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    att = 1.0 + alpha * np.exp(-lam * pos)

    w = diag * att
    tot = float(np.sum(w))
    if (not np.isfinite(tot)) or tot <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = w / tot            # fixed rescaling only (NOT per-trial normalisation)

    # ------------------------------------------------------------------
    # 4. LOAD-GATED supra-additive corroboration: the crude 'how many
    #    experts back each side' bonus is a shortcut people fall back on
    #    only when the display is dense (L > 0).  On sparse displays
    #    (L = 0) delta_eff = 0 and integration stays purely
    #    weighted-additive.  Evidence remains an ABSOLUTE margin.
    # ------------------------------------------------------------------
    delta_eff = delta * (L / (1.0 + L))

    pos_mask = d > 1e-12
    neg_mask = d < -1e-12
    s_a = float(np.sum(w[pos_mask] * np.abs(d[pos_mask])))
    s_b = float(np.sum(w[neg_mask] * np.abs(d[neg_mask])))
    k_a = float(np.count_nonzero(pos_mask))
    k_b = float(np.count_nonzero(neg_mask))
    f_a = (k_a ** delta_eff) if k_a > 0.0 else 0.0
    f_b = (k_b ** delta_eff) if k_b > 0.0 else 0.0
    E = s_a * f_a - s_b * f_b

    z = np.clip(beta * E, -60.0, 60.0)
    p_a = 1.0 / (1.0 + np.exp(-z))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- rho: [2.3, 2.7]
- alpha: [1.5, 2.1]
- lam: [0.9, 1.3]
- delta: [0.30, 0.55]
- load_flatten: [0.65, 1.10]
- beta: [6.0, 8.5]
- epsilon: [0.02, 0.08]
- validities: validities

`rationale`: MINIMAL DIFF on the accepted iter-5 base, implementing the critic's PRIMARY request (option b, the one-line load-gated corroboration) plus the requested compensation. Exactly three changes:

1. `delta_eff = delta * (L / (1.0 + L))` replaces the raw `delta` in the corroboration exponent. On sparse displays (L = 0) the count bonus is now EXACTLY silent; at L = 1 it is half strength, at L = 2 two-thirds. Nothing else in the evidence equation changes.
2. `load_flatten` widened [0.35,0.75] -> [0.65,1.10] to buy back on E6 through the channel that already worked (rho_eff = rho/(1+cL) drops from ~1.61 to ~1.35 at L = 1, ~1.10 to ~0.95 at L = 2), so the crowd keeps its weight advantage there without the count bonus doing the work.
3. `delta` range nudged [0.20,0.40] -> [0.30,0.55], which is now cheap: because delta is gated it can only act on L > 0 rows, so raising it stabilises E6/E2-rows-9-10 while leaving every L = 0 cell byte-identical to the base.

WHY THIS IS THE RIGHT SIGN, CELL BY CELL. The residual .48 of summed error sits entirely on L = 0 displays and entirely in the crowd-favouring direction, which is precisely where the ungated delta was charged in full.
- E1 (model -.210, real +.093): every scored row has the tally crowd at positions 1-4 (post-anchor, low validity) and the TTB winner at pos 0. Stripping the 4-vs-1 / 3-vs-2 count bonus removes the only term that was pushing those conflict trials toward the crowd; the contrast should rise from -.21 toward -.05/+.05. The two L = 2 conflict rows (7 and 9) keep a partial bonus, so the move is graded rather than abrupt.
- E4 (model .329, real .508): all scored rows are L = 0, crowd = {pos0, pos4} or {pos1,pos2,pos3} against a lone .95/.88 cue. With delta_eff = 0 the weighted margin is close to balanced, so the target rate returns to roughly .45-.55.
- E3 (model .444, real .518): scored rows are L = 0, {pos0, pos5} vs the mid-list .95. Losing the 2-vs-1 bonus lifts the top-cue rate back toward chance (.50-.55).
- E5 (model .786, real .820): counts are 2-vs-2 on the scored pattern, so delta was already silent and L = 0; prediction is unchanged by construction.
- E2 (model .784, real .775): only rows 9/10 have L = 1 and they keep ~half the (now larger) delta; the decisive 3-vs-3 / 5-vs-1 rows are driven by the weighted margin, so E2 should stay in .74-.82.
- E6 (model .834, real .841): all four scored rows have L = 1 or 2, so delta_eff is .15-.37 (vs .30 before) but rho_eff is markedly flatter under the widened load_flatten; the two effects trade off and E6 should hold in the .78-.88 guard band.

FAITHFULNESS. The edit does not leave the arbiter's anchored-compensatory family: product weights d_j*a_j, unnormalised absolute margin, logistic + lapse, unimodal jitter, no early stopping, no threshold rule, no per-trial normalisation, no subpopulations. It also answers the critic's faithfulness caution about L becoming an experiment-specific label: L is no longer a second, independent mechanism bolted onto the count bonus — it is now the SINGLE gating variable controlling both the flattening of validity weights and the appearance of count-based corroboration, i.e. one 'processing load degrades weighted-additive integration' story with one design-general index (shared positive endorsements), and it makes a falsifiable within-subject prediction: add shared endorsements to a sparse design and the same subject should switch from validity-weighting to crowd-following.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the arbiter's prescribed family (validity-term \u00d7 attention-term product weights, summed unnormalised evidence, softmax + lapse, unimodal jitter, no early stopping, no conclusion-graded commitment) and it clearly beats both incumbents on breadth: it avoids the ceiling/floor catastrophes of pi_3 (E3 .04, E4 .03) and the wrong-side catastrophe of pi_4 on E5 (.35 vs .82). E2 (.722 vs .775), E4 (.462 vs .508) and E5 (.774 vs .820) are all within ~.05 of the human values, which is exactly the intermediate-rate calibration the arbiter asked for. Two cells still fail. (i) E6 is a .51 miss in the pi_4 direction: the candidate chooses the mid-list top-validity cue's side (target rate .334) where humans overwhelmingly go with the screen-extreme crowd (.841). This is a direct consequence of the departure from the arbiter's rho range \u2014 rho\u22482.5 makes the near-oracle so dominant that neither the {pos0,pos5} pair nor even the {pos0,pos1,pos5} three-cue crowd can out-weigh it, despite the primacy multiplier. (ii) E1's contrast is -.16 vs +.093, i.e. the model is more tally-like on conflict trials than humans (pi_4 got -.005, pi_3 got +.10). (iii) E3 is .648 vs .518 \u2014 the model over-weights the mid-list .95 cue there too, the same rho symptom. Notably, the rho and edge-attention knobs move E3, E6 and E5 in the *same* helpful direction, so this is a tractable in-family recalibration rather than a structural failure.
Rationale: The candidate was ACCEPTED and should remain the base \u2014 do not restructure it, and stay in the prescribed anchored-compensatory family (product weights, unnormalised sum, margin-graded logistic + lapse). But one cell (E6, error .51) dominates the aggregate loss and it is fixable with the exact knob the candidate deliberately over-shot: convex diagnosticity. Concrete requests: (1) Pull rho back toward the arbiter's band \u2014 try rho \u2208 [1.2, 1.9] rather than [2.2, 2.8]. At rho\u22482.5 the .95 cue is ~4\u00d7 a .78 cue, which is why a three-cue crowd (E6 class {0,1,5} vs {2}) still loses and why E3 sits at .65 instead of .52. Lowering rho raises E6 toward .84 and lowers E3 toward .52 simultaneously; the cost on E4 (currently .462, would drift lower) is bounded and much smaller than the E6 gain, so protect E4 by not going below rho\u22481.2. (2) Make the attention term an EDGE profile rather than pure primacy: att_j = 1 + alpha*exp(-lam*j) + gamma*exp(-mu*(n-1-j)) with a small last-position bump gamma \u2208 [0.2, 0.7], mu \u2248 1. This is still a graded attention multiplier (in-family; the arbiter explicitly permitted a small unimodal recency component) and it is the cheapest way to get E6 class-2 ({pos0,pos5} vs mid-list top cue) and E5 (pos0+pos3 vs pos1+pos2) up without touching the validity term. Keep gamma small so E1 and E4 do not degrade. (3) Keep alpha at the upper end (~1.8-2.4) with lam ~1.0 so that lowering rho does not make E1's contrast even more negative \u2014 the target for E1 is roughly 0 to +.10, and the model is currently at -.16 because late low-validity cues carry too much aggregate weight on conflict trials; a stronger, more sharply decaying primacy multiplier is the in-family lever for that. (4) Re-check that beta stays in a range where E2 lands .75-.85 and E4/E3 stay in .45-.60 \u2014 no cell above ~.92 or below ~.10. Do not add strategy subpopulations, early stopping, or per-trial normalisation.

**Outcome of this advice:** iter 1 candidate loss=0.2158 -> iter 2 candidate loss=0.2242 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: Iter 2 is a strict regression (0.2242 vs 0.2158) and the gate rejected it, so the base is still iter 1. The diagnosis is clean: the added recency term (gamma·exp(-mu·(n-1-j))) bought only +0.057 on E6 (.334 → .391, still a .45 error) while degrading every other cell — E2 .722→.696, E4 .462→.422, E5 .774→.739, and it overshot E3 from +.130 error to -.110 error. Net negative. So the last-position bump is not the right lever, and my iter-1 push in that direction was wrong; I will not repeat it. The structural picture from the accepted base is unchanged and informative. The candidate is faithful to the arbiter's family (product of a convex validity term and a positional attention term, unnormalised summed evidence, margin-graded logistic + lapse, unimodal jitter, no early stopping, no conclusion-graded commitment), and it is the only theory on the board that avoids catastrophic misses in BOTH directions (pi_3 is at ceiling on E5/E6 but floors E3/E4; pi_4 inverts E5). Its one dominating failure is E6: humans go with the screen-extreme crowd against the mid-list top-validity cue at .841, the model goes the opposite way at .33-.39. E3 is the same qualitative cell (edge cues vs mid-list .95) but humans there sit at chance (.518), and in the accepted base the model is .648 — i.e. E3 has ~.13 of slack in the SAME direction that E6 needs a huge move. E1 (-.16 vs +.093) is also a same-direction miss: the model is too tally-like on conflict trials, which a stronger/sharper first-position anchor would correct. The knob that moves E6 up, E3 down, E1 up, E2 up and E5 up simultaneously is the strength and sharpness of the primacy multiplier applied to a still-convex diagnosticity — E4 (currently .462 vs .508) is the only cell that pays.
Rationale: Stay inside the arbiter's anchored-compensatory family (product weights d_j·a_j, unnormalised sum over all cues, logistic on the absolute margin + lapse, unimodal jitter). Build on the ACCEPTED iter-1 base, not on iter 2. Concrete, minimal-diff requests: (1) DROP the recency term (set gamma = 0 / delete it). My iter-1 request for an edge profile was tested and rejected by the gate: it moved E6 by only .06 while costing .03-.06 each on E2, E4, E5 and flipping E3's sign of error. Do not retry it. (2) Instead, strengthen and sharpen the PRIMACY anchor, which I did not push hard last time: alpha from ~2.0 up to ~3.0-4.0 with lam ~1.2-1.7 (currently 1.7-2.3 / 0.95-1.30). Hand-check the direction before committing: a sharper decay leaves the mid-list top cue (E6 pos2, E3 pos3, E4 pos2/pos3) with essentially no attentional boost while pos0 is multiplied 4-5x, which (a) raises E6 toward .5-.7 because pos0 is on the crowd's side in BOTH scored classes, (b) pulls E3 down from .648 toward the observed .518, (c) pushes E1's conflict-minus-agreement contrast up from -.16 toward the observed +.09 (conflict trials' TTB winner is the pos-0 side), (d) raises E2's tally-winner rate on the rows where the crowd occupies pos0-pos2 and the .95 oracle sits at pos5, and (e) raises E5, whose target side contains pos0. (3) Nudge rho down only slightly, to ~[1.9, 2.4], as an intermediate step between the accepted [2.2,2.8] and my earlier (untested) [1.2,1.9] — enough extra crowd weight to keep E6 moving without collapsing E4. Do not go below ~1.8 in one step; the proposer's own arithmetic shows E4 and E1 break at rho≈1.5. (4) Treat E4 as the guard rail: it currently sits .046 below the human value and both knobs push it lower. Accept at most ~.08 total degradation on E4 (i.e. keep it >= .43) and back alpha off if it falls further; also keep E2 >= .70 and E5 >= .77. (5) Sanity-check that the true per-experiment validity vector is actually reaching predict() for E6 and E3 (the `validities` parameter), rather than the descending-linspace fallback — a silent fallback on a scrambled display would by itself explain a .45 miss and would make all of the above tuning moot. (6) No strategy subpopulations, no early stopping, no per-trial normalisation by the discriminating weight, and keep every simulated cell inside .10-.93.

**Outcome of this advice:** iter 2 candidate loss=0.2242 -> iter 3 candidate loss=0.2310 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: Iter 3 is another strict regression (0.2310 vs the accepted 0.2158) and was rejected by the gate, so the base is still iter 1. The per-cell breakdown is informative: the broad recency lobe plus the much higher beta (14-24 vs 6-8.5) did improve E1 (-.160 \u2192 -.091 vs real +.093), E3 (.648 \u2192 .494 vs .518) and E5 (.774 \u2192 .813 vs .820), but it destroyed the two cells that were already carrying the loss: E6 collapsed from .334 to .091 against a real value of .841 (a .75 error, by far the largest single miss on the board), and E2 slid from .722 to .666 (real .775). The E6 collapse falsifies the proposer's own arithmetic: they argued a broad recency lobe would lift the pos0/pos5 crowd in E6, but with rho\u22482.4 the mid-list .95 cue has diagnosticity .45^2.4 \u2248 .25 while a .6 cue has .1^2.4 \u2248 .004 \u2014 a ~60x ratio that no attention multiplier of size 2-3 can overcome. Raising beta then simply sharpened the already wrong-signed evidence toward 0. So the positional-attention lobes are NOT the binding constraint on E6/E3; the convex diagnosticity exponent is. Note also that the candidate has drifted from the arbiter's explicit prescription: the recommendation asked for rho \u2248 0.8-1.4 precisely so that 'a crowd of 3 mid cues can out-weigh a single .95 cue', and every iteration of this loop has instead run rho \u2248 2.2-2.8, i.e. the opposite sign of curvature. That single unfaithful choice mechanically guarantees the E6 (.09-.39 vs .84) and E1 (negative contrast) failures, and it is the one lever that has never actually been tested.
Rationale: The candidate was REJECTED, so build again on the accepted iter-1 base, staying strictly inside the arbiter's anchored-compensatory family (product weights w_j \u221d (v_j-0.5)^rho \u00b7 a_j, unnormalised sum over all cues, logistic on the absolute margin + lapse, unimodal jitter, no early stopping, no per-trial normalisation, no conclusion-graded commitment). Concrete, minimal-diff requests, chosen to avoid repeating advice the gate has already killed:

(1) FIX THE FAITHFULNESS DRIFT ON rho \u2014 this is the priority and it is the only knob never actually tried. The arbiter prescribed rho \u2248 0.8-1.4; every iteration has used rho \u2248 2.2-2.8. Set rho \u2208 [0.8, 1.3]. At rho\u22481 the ratio between a .95 cue and a .60 cue is 4.5:1 rather than ~60:1, so a two- or three-cue edge crowd with a modest primacy boost can genuinely balance or beat the mid-list near-oracle. This is exactly what E6 (real .841, model .09-.39) and E3 (real .518 = chance, model .49-.65) require, and it moves both in the right direction simultaneously. Do NOT compensate by re-inflating alpha.

(2) UNDO the beta escalation. Return beta to roughly the accepted-base band (~6-10). The high beta in iter 3 is what turned a modest wrong-signed margin in E6 into a floor value of .09; with lower rho the E6 margin flips sign, but until it does, keep beta low enough that no simulated cell leaves the .12-.90 band. Combined with lapse \u03b5 \u2248 .05-.10, this preserves the intermediate rates the arbiter asked for.

(3) DROP the recency lobe (gamma = 0) or cap it at gamma \u2264 0.25 with mu \u2265 1. I asked for a recency bump in iter 1 (rejected) and the proposer widened it in iter 3 (rejected again). Two independent tests say this lever costs more on E2/E4 than it buys anywhere. Do not retry it in any broad form.

(4) Keep the primacy anchor MODERATE: alpha \u2208 [1.4, 2.0], lam \u2208 [1.0, 1.4]. I oscillated on this (iter 1: raise to 1.8-2.4; iter 2: raise to 3.0-4.0, rejected; iter 3 lowered to 1.1-1.7, rejected), so settle in the middle. With rho near 1 the anchor no longer needs to be huge to flip validity ties in E1/E5, and a huge anchor was already shown to break E4.

(5) Guard rails to hand-check before submitting (use the true per-experiment `validities`, and verify the descending-linspace fallback in predict() is never triggered \u2014 a silent fallback on E6/E3's scrambled displays would by itself explain the E6 miss): E6 \u2265 .60 (target .84), E2 \u2208 [.72, .85], E5 \u2208 [.75, .88], E3 \u2208 [.42, .60], E4 \u2208 [.43, .58], E1 contrast \u2208 [-.05, +.15]. E6 alone is worth ~.7 of error \u2014 accept a few hundredths of degradation on E4/E5 to buy it, but reject any configuration that pushes E2 below .70.

No strategy subpopulations, no early stopping, no threshold rule, no per-trial normalisation \u2014 stay in the prescribed family; the fix here is a parameter-range correction back toward the arbiter's own specification, not a redesign.

**Outcome of this advice:** iter 3 candidate loss=0.2310 -> iter 4 candidate loss=0.1834 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: Iter 4 is the first strict improvement since iter 1 (0.1834 vs 0.2158) and the corroboration exponent is clearly doing real work: E6 moved .334 -> .511 (error .51 -> .33), E2 moved .722 -> .764 (error .053 -> .012, now essentially exact), and E3 stayed at chance-ish .479 (error .039) and E5 at .785 (error .035). So the accepted base now fits E2, E3, E5 well and is the only theory on the board without a catastrophic sign error anywhere (pi_3 floors E3/E4, pi_4 inverts E5, pi_1 inverts E6, pi_2 inverts E1). Two cells still dominate the residual loss, and they now point in the SAME direction. (i) E6: .511 vs .841 (-.33). (ii) E1 contrast: -.217 vs +.093 (-.31) 
crucially this got WORSE than the base (-.160), i.e. the corroboration term paid for E6/E2 partly out of E1. (iii) E4 also degraded, .462 -> .351 vs .508 (-.157), again a corroboration cost. The structural reading is informative: delta boosts whichever side has more cues, and in E1's conflict trials the crowd (positions 1-4, all low-attention, low-validity) is exactly the side humans do NOT follow, whereas in E6 the crowd contains position 0 (and position 5) and IS the side humans follow. So a raw count exponent cannot separate the two; what distinguishes them is that E6's crowd owns the anchor cue while E1's crowd is entirely post-anchor. That is a graded-attention distinction, still fully inside the prescribed anchored-compensatory family, and it has never been tried.
Rationale: The candidate was ACCEPTED and must remain the base 
do not restructure it, do not leave the arbiter's family (product weights d_j*a_j, unnormalised absolute margin, logistic + lapse, unimodal jitter, no early stopping, no per-trial normalisation, no subpopulations). But .64 of the remaining loss sits in two cells (E6 -.33, E1 -.31) that a single minor in-family edit can move together.

(1) PRIMARY REQUEST 
make the corroboration count ATTENTION-WEIGHTED rather than raw. Replace k_side = count of supporting cues with an effective count k_side = sum_j in side (a_j / a_max), i.e. each supporting cue contributes its own attention share to the 'they agree' bonus. This is a one-line change, keeps delta silent on equal-count/equal-attention cells, and has the right differential sign by construction: in E1 the tally crowd occupies positions 1-4 (attention ~1 each vs ~1+alpha at pos 0), so its corroboration bonus shrinks and the contrast rises from -.22 back toward 0/+.09; in E6 the crowd contains position 0 (and pos 5), so its effective count barely drops and the .511 is preserved or improved; E4's crowd is pos0+pos4 so it loses only a little, which also nudges E4 back up toward .508. Hand-check that E2 stays >= .72 and E5 stays >= .75 before submitting.

(2) SECONDARY, SMALL: with (1) in place you can afford a modest strengthening of the primacy anchor 
alpha in [1.9, 2.5], lam ~1.0-1.2 (currently 1.5-2.1 / 0.9-1.3). Both E1 and E6 want more pos-0 weight (in E6 the anchor is on the crowd's side in both scored classes; in E1 the anchor is on the TTB side). Do NOT go to alpha 3-4 
that step was tested in iter 2 and rejected, and it breaks E4. Treat E4 as the guard rail: keep it >= .43.

(3) delta: keep the range roughly where it is (~[0.20, 0.40]) or shift it very slightly down (~[0.15, 0.35]) once the effective-count version is in, since attention weighting already tempers its E1/E4 cost. Do not push delta above 0.45.

(4) I will not repeat two levers the gate has already killed: no recency/edge lobe (rejected twice), no beta escalation (rejected). Also, my earlier push to flatten rho toward 0.8-1.3 was not followed and the loss improved anyway, so I withdraw it as a hard requirement 
at most allow a mild widening to rho in [2.1, 2.7], and only if E6 does not regress.

(5) Verify (as before) that the true per-experiment `validities` vector reaches predict() for E3/E6 and that the descending-linspace fallback never fires on the scrambled displays.

Targets to hand-check: E6 >= .60 (real .841), E1 contrast in [-.10, +.15] (real +.093), E4 in [.43, .58], E2 in [.72, .85], E5 in [.75, .88], E3 in [.42, .60], and no simulated cell outside .10-.93.

**Outcome of this advice:** iter 4 candidate loss=0.1834 -> iter 5 candidate loss=0.1278 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: Iter 5 is a large, genuine improvement (0.1834 -> 0.1278) and it fixed the cell that had dominated the loss for four iterations: E6 goes .511 -> .834 against a real .841 (error .33 -> .007), while the cells I asked to be protected were protected essentially exactly (E5 .786 vs .820; E3 .444 vs .518; E2 .784 vs .775, now the best on the board). The model remains faithful to the arbiter's anchored-compensatory family in every structural respect: product weights (validity term x positional attention term), unnormalised absolute margin, logistic + lapse, unimodal jitter, no early stopping, no threshold rule, no per-trial normalisation, no strategy subpopulations. The new load term is a stimulus-driven modifier of the weight profile, i.e. still a weight computation inside the prescribed family rather than a family switch, so I grade it as faithful \u2014 with the caveat below.

What is left is now concentrated in exactly two cells, and they point the same way. (i) E1: model contrast -.210 vs real +.093 (error .30, and a SIGN error \u2014 humans are MORE top-cue-consistent on tally-conflict trials, the model is markedly less). (ii) E4: .329 vs .508 (error .18, and it has drifted further down every iteration since delta was introduced: .462 -> .351 -> .329). Both are cells with ZERO shared endorsements (L = 0), so the new load term is inert there by construction and cannot be blamed; the culprit is the raw corroboration exponent delta, which unconditionally rewards whichever side has more discriminating cues. In E1's conflict rows the crowd is post-anchor and low-validity and humans do NOT follow it; in E4 the 2-cue crowd vs the lone .95/.88 cue leaves humans at chance while the model follows the crowd at .33. Importantly, delta's original job \u2014 letting a crowd beat a near-oracle in E6 \u2014 is now done by the load-driven flattening of rho, so delta is largely redundant and is being paid for entirely by E1/E4/E3, all three of which need LESS crowd bonus.

The loop-level trajectory also matters: E6 is now solved, E2 is essentially exact, and the residual .48 of summed error is a single, well-identified over-weighting of cue counts on sparse displays.
Rationale: The candidate was ACCEPTED and must remain the base \u2014 do not restructure it, do not leave the arbiter's family (product weights d_j*a_j, unnormalised absolute margin, logistic + lapse, unimodal jitter, no early stopping, no per-trial normalisation, no subpopulations), and do NOT remove the load-flattening term that just bought E6. One minimal, well-targeted parameter/gating edit should be tried on top of it.

(1) PRIMARY REQUEST \u2014 shrink the raw corroboration exponent on sparse (low-load) displays. delta is now double-counting: with rho_eff flattening under load, the crowd already wins in E6 without a count bonus, but delta is still charged in full on every L = 0 display, which is precisely where E1 (-.210 vs +.093), E4 (.329 vs .508) and E3 (.444 vs .518) all err in the crowd-favouring direction. Either (a) drop delta's range to ~[0.00, 0.15], or (b) make delta itself load-gated in exactly the same way rho is, delta_eff = delta * (c2*L)/(1+c2*L) or delta_eff = delta * L/(1+L), so the count bonus is silent when no expert endorsed both products and full when several did. Option (b) is the cleaner one-line edit and keeps the mechanism unified under the single 'processing load degrades weighted-additive integration' story rather than adding a second free knob. Hand-check the sign: E1's conflict crowd sits at positions 1-4 (no anchor, low validity) so its bonus disappears and the contrast should rise from -.21 toward 0/+.09; E4's {pos0,pos4} crowd loses its bonus against the lone .95/.88 cue so E4 should rise from .33 toward .45-.55; E3's {pos0,pos5} pair loses its bonus against the mid-list .95 so E3 should rise from .444 toward .52. E5 is unaffected (equal counts, delta silent).

(2) COMPENSATE ON E6, NOT VIA delta. If (1) costs E6 anything (its scored rows are 2-vs-1 and 3-vs-1, so they do lose the count bonus), buy it back through the load channel that just worked: widen load_flatten to ~[0.6, 1.1] so rho_eff drops further at L = 1-2. That keeps E6 in the .78-.88 band while leaving every L = 0 cell byte-identical.

(3) GUARD RAILS to hand-check before submitting: E6 >= .75 (real .841), E2 in [.72, .85] (real .775 \u2014 lowering delta will pull rows 9/10 down, so verify), E5 in [.75, .88], E3 in [.45, .60], E4 >= .43, E1 contrast in [-.10, +.15], and no simulated cell outside .10-.93. If E2 falls below .72, recover it with a small beta increase within ~[7, 10] rather than by restoring delta.

(4) Levers I will NOT re-request, because the gate has already killed them: no recency/edge attention lobe (rejected twice), no large beta escalation (rejected), no alpha 3-4 anchor (rejected). My iter-4 attention-weighted corroboration count was declined by the proposer and the alternative worked better, so I withdraw it as a requirement \u2014 but note that request (1) achieves the same differential (strip the crowd bonus where the crowd is post-anchor and sparse) with a simpler gate.

(5) FAITHFULNESS CAUTION, not a redesign order: the load index L is currently defined only as 'features endorsed by BOTH options', and it happens to be non-zero only in E6. That is a real risk of an experiment-specific label. Keep the mechanism, but state (and, if cheap, implement) a graded, design-general version \u2014 e.g. L as the count of non-discriminating POSITIVE endorsements normalised by the number of features \u2014 and verify it leaves E3/E4/E5 unchanged. Also re-verify that the true per-experiment `validities` vector reaches predict() for every design and that the descending-linspace fallback never fires.

**Outcome of this advice:** iter 5 candidate loss=0.1278 -> iter 6 candidate loss=0.1066 -> the gate marked it ACCEPTED.

### Iteration 6 (most recent — address this)
Verdict: regenerate
Interpretation: Iter 6 is another genuine improvement (0.1278 -> 0.1066) and the gate accepted it. My iter-5 primary request (option b: load-gate the corroboration exponent, delta_eff = delta*L/(1+L), and buy E6 back through load_flatten rather than through delta) worked as predicted on three of the four cells I named: E4 recovered .329 -> .466 (error .18 -> .04), E1 improved -.210 -> -.168, E2 held at .769 (error .006) and E6 held at .849 (error .008, essentially exact). The model is still faithful to the arbiter's anchored-compensatory family in every structural respect \97 product weights (convex validity term x positional attention term), unnormalised absolute margin, logistic + lapse, unimodal jitter, no early stopping, no threshold rule, no per-trial normalisation, no subpopulations \97 and it remains the only theory on the board with no catastrophic sign error anywhere (pi_1 inverts E6, pi_2 inverts E1, pi_3 floors E3/E4, pi_4 inverts E5).

The residual (~.50 summed absolute error) is now concentrated in exactly two cells, and they are movable by a SINGLE shared lever. (i) E1: contrast -.168 vs real +.093 (error .26, and still a sign error). (ii) E3: .645 vs real .518 (error +.128) \97 note this one REGRESSED from .444 in iter 5, i.e. removing the sparse-display count bonus overshot in E3, which is the price paid for the E4/E1 gains. E5 (.768 vs .820) and E4 (.466 vs .508) are small same-direction misses.

The key structural observation: in E1 the top-validity cue and the anchor are the SAME cue (position 0), and humans follow it on conflict trials; in E3 the top-validity cue is mid-list (position 3) and the anchor cues (0 and 5) oppose it, and humans sit at chance. So raising the primacy attention multiplier alpha moves E1's contrast UP (conflict rows' TTB winner is the pos-0 side) and E3 DOWN (the pos-0 member of the edge crowd gains weight against the mid-list .95) at the same time. It also nudges E5 up (target side owns pos 0), E2 up (crowd occupies pos 0-2 against the .95 at pos 5) and leaves E6 roughly where it is (crowd contains pos 0 and pos 5). E4 is the only cell that can pay, and it currently has ~.08 of headroom. This is a pure parameter-range recalibration inside the prescribed family, and the mid-range version of it has never actually been tested (iter 2's alpha=3-4 was rejected; iter 4's alpha=1.9-2.5 was declined by the proposer in favour of the load term).
Rationale: The candidate was ACCEPTED and must remain the base \97 do not restructure it, do not remove the load-gated delta or the load-flattening of rho (they just bought E4/E6/E2), and stay strictly inside the arbiter's anchored-compensatory family (product weights d_j*a_j, unnormalised absolute margin, logistic + lapse, unimodal jitter, no early stopping, no per-trial normalisation, no subpopulations). One minimal, well-targeted edit should be tried on top of it.

(1) PRIMARY REQUEST \97 strengthen the primacy anchor moderately: alpha from [1.5, 2.1] to about [2.0, 2.6], with lam kept sharp-ish, ~[1.0, 1.3]. Rationale and predicted signs, cell by cell (hand-check these before submitting): E1's tally-conflict rows have the TTB winner at position 0 and the crowd at positions 1-4 (post-anchor, low validity), so a larger pos-0 multiplier with a sharp decay raises the conflict-trial hit rate while barely touching the already-near-ceiling agreement trials -> contrast should rise from -.168 toward -.05/0. E3's scored rows are {pos0,pos5} against the mid-list .95 at pos3; a bigger pos-0 boost pulls the top-cue rate down from .645 toward the observed .518. E5's target side owns pos 0 -> up from .768 toward .82. E2's decisive rows have the crowd at pos 0-2 -> stays at or slightly above .775. E6's crowd contains pos 0 -> unchanged or marginally up (watch that it does not exceed ~.90). Note I am deliberately choosing a value BETWEEN my rejected iter-2 push (alpha 3-4) and the current range \97 do not go to 3+, that step was tested and killed by the gate and it breaks E4.

(2) GUARD RAIL \97 E4 is the cell that pays for (1), because two of its scored classes have the crowd owning pos 0 (or pos 0/pos 4). Keep E4 >= .43 (real .508, currently .466). If it drops below that, back alpha off to ~2.2 rather than compensating with any other knob.

(3) DO NOT re-introduce a sparse-display count bonus to fix E3. E3's overshoot is a side effect of the (correct) delta gating; fixing it with delta would immediately re-break E4 and E1, which is exactly the trade the last two iterations climbed out of. Use the attention lever instead. Equally, do not touch rho: lowering it would help E3 but would strip the pos-0 validity advantage that E1 needs, and raising it does the reverse.

(4) Levers I will NOT re-request because the gate has already killed them: no recency/edge attention lobe (rejected twice), no large beta escalation (rejected in iter 3). If E2 slips below .72 after (1), recover it with a small beta bump inside ~[7, 10] only, not by restoring delta or adding a recency term.

(5) Targets to hand-check before submitting: E1 contrast in [-.08, +.15] (real +.093), E3 in [.48, .60] (real .518), E4 >= .43 (real .508), E5 in [.78, .88] (real .820), E2 in [.72, .85] (real .775), E6 in [.78, .90] (real .841), and no simulated cell outside .10-.93.

(6) FAITHFULNESS \97 the candidate is in-family and I grade it as faithful; the load term is a stimulus-driven modifier of the weight profile, not a family switch. Two standing hygiene checks: (a) verify the true per-experiment `validities` vector actually reaches predict() on every design so the descending-linspace fallback never fires on the scrambled displays (E3/E4/E6); (b) L (shared positive endorsements) is still near-degenerate across designs \97 non-zero mainly in E6 and a couple of E1/E2 rows \97 so state or implement the graded, design-general normalisation (L as the fraction of features carrying non-discriminating positive endorsements) and confirm it leaves E3/E4/E5 numerically unchanged.

**Outcome of this advice:** iter 6 candidate loss=0.1066 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**Anchored compensatory integration with salience-weighted clutter and partial attentional amplification of the discriminating subset.**\n\n(1) *Diagnosticity is read convexly.* The subjective pull of expert j is d_j = (v_j - 0.5)^rho with rho ~2.3-2.7, so a .95 expert is treated as a near-oracle and .55-.65 experts as near-noise.\n\n(2) *Screen position is a graded attention multiplier, not a search order.* a_j = 1 + alpha*exp(-lam*j); the cue weight is the PRODUCT w_j \u221d d_j * a_j, rescaled once to sum to 1. The first-read cue is rehearsed longest and anchors the running impression, breaking validity ties toward the pos-0 side, but it cannot overturn a genuine diagnosticity gap.\n\n(3) **Clutter \u2014 not mere overlap \u2014 is what degrades weighted-additive processing.** Features on which BOTH products were endorsed carry no discriminating information. But such shared endorsements are not psychologically equivalent: an endorsement of both products by a *weak* expert is pure clutter (noise that has to be read, held and compared, making the options feel similar), whereas an endorsement of both products by the *best* expert is a meaningful, quickly assimilated fact (\"the oracle is neutral here\") that simply hands the decision to the remaining experts. Load is therefore the salience-weighted count L = sum over shared-endorsement features of u_j, with u_j = (v_max - v_j)/(v_max - 0.5) \u2014 zero for the top expert, ~1 for the weakest. Under clutter TWO things happen together, both expressions of one effort\u2013accuracy tradeoff: (a) the diagnosticity exponent flattens, rho_eff = rho/(1 + c*L); (b) the crude 'how many experts back each side' cue appears, each side's summed support multiplied by k^delta_eff with delta_eff = delta*L/(1+L). Both are silent when L = 0 \u2014 including the case where the shared endorsements come from the top experts, which is exactly when people fall back on careful validity weighting of the remaining cues rather than on counting.\n\n(4) **Attention is partially reallocated to the cues that actually discriminate.** Evidence is still the ABSOLUTE (unnormalised in the pi_3 sense) margin, but it is amplified by a sublinear function of how much total weight the trial's discriminating cues carry: E = (S_A - S_B)/D^psi with D = S_A + S_B and psi ~0.4-0.65. psi = 0 is a pure absolute margin (all trials on which only weak experts speak are near chance); psi = 1 is full relative evidence (every trial equally decisive, the ceiling failure of pi_3). Humans sit in between: when the leading experts are silent, the surviving weak-cue difference is magnified but not made fully decisive, so trials whose discriminating cues are all weak are followed clearly but with less confidence than trials in which the anchor cue itself discriminates.\n\n(5) Choice = logistic(beta*E) with a lapse epsilon. No early stopping, no threshold rule, no conclusion-graded commitment, no strategy subpopulations \u2014 individual differences are unimodal jitter in rho, alpha, lam, delta, c, psi, beta and lapse.\n\nSignature prediction: hold the display fixed and let the two best experts endorse both products. The theory says the decision does NOT collapse toward counting (unlike when weak experts endorse both); instead the remaining, lower-validity cues are amplified and the single best surviving expert is followed \u2014 a within-subject dissociation between high-validity and low-validity redundancy that no pure-load or pure-tally account predicts.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ------------------------------------------------------------------\n    # 1. unpack the trial stimulus into two rating vectors (screen order)\n    # ------------------------------------------------------------------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        if 'option_a_ratings' in state and 'option_b_ratings' in state:\n            a = np.asarray(list(state['option_a_ratings']), dtype=float)\n            b = np.asarray(list(state['option_b_ratings']), dtype=float)\n    if a is None:\n        arr = np.asarray(state, dtype=float)\n        if arr.ndim == 1:\n            arr = arr.reshape(2, -1)\n        elif arr.ndim > 2:\n            arr = arr.reshape(2, -1)\n        if arr.shape[0] != 2:\n            arr = arr.reshape(2, -1)\n        a = arr[0].astype(float)\n        b = arr[1].astype(float)\n\n    a = np.ravel(a).astype(float)\n    b = np.ravel(b).astype(float)\n    n = int(min(a.shape[0], b.shape[0]))\n    if n == 0:\n        return np.array([0.5, 0.5])\n    a = a[:n]\n    b = b[:n]\n    d = a - b\n\n    if not np.any(np.abs(d) > 1e-12):\n        return np.array([0.5, 0.5])\n\n    # ------------------------------------------------------------------\n    # 2. communicated validities -> convex diagnosticity\n    # ------------------------------------------------------------------\n    val = parameters.get('validities', None)\n    try:\n        val = np.asarray(val, dtype=float).ravel()\n    except Exception:\n        val = None\n    if val is None or val.shape[0] != n:\n        # fallback: assume the display is validity-sorted, descending\n        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])\n    val = np.clip(val, 0.5 + 1e-6, 1.0)\n\n    rho = float(np.clip(parameters.get('rho', 2.5), 0.1, 8.0))\n    alpha = float(np.clip(parameters.get('alpha', 1.9), 0.0, 6.0))\n    lam = float(np.clip(parameters.get('lam', 1.1), 0.01, 6.0))\n    beta = float(np.clip(parameters.get('beta', 7.0), 0.0, 200.0))\n    eps = float(np.clip(parameters.get('epsilon', 0.05), 0.0, 1.0))\n    delta = float(np.clip(parameters.get('delta', 0.40), 0.0, 2.0))\n    c_load = float(np.clip(parameters.get('load_flatten', 0.85), 0.0, 3.0))\n    psi = float(np.clip(parameters.get('psi', 0.5), 0.0, 1.0))\n\n    # --- SALIENCE-WEIGHTED clutter.  Features endorsed by BOTH options\n    # carry no discriminating information, but only endorsements coming\n    # from comparatively WEAK experts are clutter; a shared endorsement\n    # by the best expert is assimilated at once and simply hands the\n    # decision to the remaining experts.  u_j = 0 for the top expert,\n    # ~1 for the weakest.\n    vmax = float(np.max(val))\n    span = max(vmax - 0.5, 1e-9)\n    u = np.clip((vmax - val) / span, 0.0, 1.0)\n    shared = (a > 0.5) & (b > 0.5)\n    L = float(np.sum(u[shared]))\n\n    rho_eff = rho / (1.0 + c_load * L)\n    rho_eff = float(np.clip(rho_eff, 0.6, 8.0))\n\n    diag = np.power(np.clip(val - 0.5, 1e-12, None), rho_eff)   # diagnosticity\n\n    # ------------------------------------------------------------------\n    # 3. graded primacy attention multiplier (screen-position anchor)\n    # ------------------------------------------------------------------\n    pos = np.arange(n, dtype=float)\n    att = 1.0 + alpha * np.exp(-lam * pos)\n\n    w = diag * att\n    tot = float(np.sum(w))\n    if (not np.isfinite(tot)) or tot <= 0:\n        w = np.ones(n, dtype=float) / n\n    else:\n        w = w / tot            # fixed rescaling only (NOT per-trial normalisation)\n\n    # ------------------------------------------------------------------\n    # 4. clutter-gated supra-additive corroboration (silent at L = 0)\n    # ------------------------------------------------------------------\n    delta_eff = delta * (L / (1.0 + L))\n\n    pos_mask = d > 1e-12\n    neg_mask = d < -1e-12\n    s_a = float(np.sum(w[pos_mask] * np.abs(d[pos_mask])))\n    s_b = float(np.sum(w[neg_mask] * np.abs(d[neg_mask])))\n    k_a = float(np.count_nonzero(pos_mask))\n    k_b = float(np.count_nonzero(neg_mask))\n    f_a = (k_a ** delta_eff) if k_a > 0.0 else 0.0\n    f_b = (k_b ** delta_eff) if k_b > 0.0 else 0.0\n\n    S_a = s_a * f_a\n    S_b = s_b * f_b\n\n    # PARTIAL attentional amplification of the discriminating subset:\n    # E = (S_a - S_b) / D^psi with D the total discriminating support.\n    # psi = 0 -> pure absolute margin; psi = 1 -> fully relative (ceiling).\n    D = S_a + S_b\n    if D <= 1e-12:\n        return np.array([0.5, 0.5])\n    E = (S_a - S_b) / (D ** psi)\n\n    z = np.clip(beta * E, -60.0, 60.0)\n    p_a = 1.0 / (1.0 + np.exp(-z))\n\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    p = (1.0 - eps) * p + eps * 0.5\n    p = np.clip(p, 1e-12, None)\n    p = p / p.sum()\n    return p",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if not np.isfinite(s) or s <= 0:\n        probs = np.ones_like(probs) / len(probs)\n    else:\n        probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "rho": "[2.3, 2.7]",
    "alpha": "[1.7, 2.2]",
    "lam": "[0.9, 1.3]",
    "delta": "[0.30, 0.55]",
    "load_flatten": "[0.65, 1.10]",
    "psi": "[0.40, 0.65]",
    "beta": "[6.0, 8.5]",
    "epsilon": "[0.02, 0.08]",
    "validities": "validities"
  },
  "rationale": "MINIMAL DIFF on the accepted iter-6 base: three edited blocks, everything else byte-identical (product weights d_j*a_j, clutter-gated delta, logistic + lapse, unimodal jitter, no early stopping, no subpopulations, no threshold rule).\n\nWHY I PARTLY DEPART FROM THE CRITIC'S PRIMARY REQUEST. The critic asked for an alpha bump to fix E1 and E3 together. I hand-checked E1's metric row by row with the model's own arithmetic and the alpha lever cannot work there, so I did the arithmetic-driven fix instead and kept only a mild alpha bump. E1's scored rows are: four conflict rows (1,2,8,10) where pos-0 decides and the model is already lapse-capped at ~.94, four agreement rows also lapse-capped at ~.97, and TWO conflict rows (7 and 9) where positions 0 and 1 are endorsed by BOTH options, so the decision rests on pos2 (higher validity) against pos3+pos4. The model scores those two rows at ~.53. Contrast = (4*.94 + 2*h)/6 - .97, so the entire -.17 error is the gap between h=.53 and the ceiling; raising alpha changes h by <.01 (it only moves weight onto the two tied cues) and mathematically cannot move E1. Two things hold h down: (i) rows 7/9 have raw L=2, so the load term flattens rho and hands a count bonus to the pos3+pos4 pair; (ii) even with no load, w2 - (w3+w4) ~= .045 in globally-normalised weight units, and beta*.045 leaves the row near chance. Both are addressed below.\n\nEDIT 1 - SALIENCE-WEIGHTED CLUTTER (replaces the raw shared-endorsement count). L = sum over shared-endorsement features of u_j = (v_max - v_j)/(v_max - 0.5). This is design-general (it also implements the critic's standing request for a graded, non-degenerate load index) and it makes a substantive claim: redundancy from the BEST expert is not clutter, it just passes the decision to the rest, whereas redundancy from weak experts is. Signs: E1 rows 7/9 share pos0 (.90 = top, u=0) and pos1 (u=.25) -> L=.25 instead of 2, so rho stays convex and the count bonus nearly vanishes, and pos2 beats the pos3+pos4 pair. E6's shared cues are all non-top (pos1/pos3/pos4) so L stays ~1.6-2 of its old value and the crowd-following that bought E6 is preserved. E2 rows 9/10 share pos5 (.95 = top) so they lose their load, which costs a few points there - the smallest cost centre on the board.\n\nEDIT 2 - PARTIAL ATTENTIONAL AMPLIFICATION, E = (S_a - S_b)/D^psi, psi in [.40,.65]. This is deliberately NOT the per-trial normalisation the arbiter forbade: psi=1 is pi_3's relative rule that ceilings everywhere (.95/.97 on E5/E6, floors E3/E4), psi=0 is the current absolute margin that leaves every 'only weak cues speak' trial at chance. At psi~.5 the amplification is D^-psi, and D is the fraction of total weight carried by the discriminating cues, so it is ~1.0-1.15 wherever the anchor/top cue discriminates and only bites where the leading experts are silent. Hand-checked signs: E1 rows 7/9 D~.09 -> amplification ~3.3 -> h ~= .75-.90 and the contrast rises from -.168 toward -.10/-.04 (this alone is worth more than every other cell combined, since E1 carries ~three quarters of the squared loss); E1's ceiling and agreement rows have D~1 and are untouched; E5 (4 of 5 cues discriminate, D~.95) rises slightly toward the observed .82; E4 D~.59-.75 -> amplification 1.15-1.31, a bounded cost of ~.03; E3 D~.87 -> amplification only ~1.07, so E3 stays in the .60-.67 band; E6 D~.55 -> some sharpening, which is offset by the (slightly reduced) clutter under Edit 1. |E| <= D^(1-psi) <= 1 by construction, so no cell can be driven outside the .10-.93 band by the new term.\n\nEDIT 3 - alpha [1.5,2.1] -> [1.7,2.2]. A mild version of the critic's request, kept small because my E4 arithmetic (rows 1/2 and 7/8 are the only scored cells and row 7's target cue at pos3 is opposed by pos0 and pos4) shows E4 loses ~.03 per .35 of alpha; this range keeps E4 near the .43 guard rail while pulling E3 down from .645 toward .58-.60 and E5 up toward .80.\n\nNo recency/edge lobe, no beta escalation, no alpha 3-4 anchor - the three levers the gate has already killed."
}
```

## Usage

```json
{
  "input_tokens": 38313,
  "output_tokens": 46122
}
```
