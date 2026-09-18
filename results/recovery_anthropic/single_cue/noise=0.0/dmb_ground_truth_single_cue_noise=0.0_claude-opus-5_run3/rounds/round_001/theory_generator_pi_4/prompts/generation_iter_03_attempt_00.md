# generation_iter_03_attempt_00

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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace THEORY 1 (Take-The-Best) with a **compensatory validity-weighted evidence integration** theory ("graded weighted-additive with margin-graded confidence"):

Core claims to state in the description:
1. Subjects integrate ALL binary expert ratings on every trial; no search-and-stop, no one-reason decision. Evidence = sum_j w_j * (a_j - b_j), where the sum runs over all features regardless of screen position.
2. Weights are monotone in the *stated* validity but strongly compressed toward equality: w_j \u221d (v_j - 0.5)^rho with 0.3 <= rho <= 1.5, or equivalently w_j \u221d logit(v_j)^rho. Compression is the key free ingredient \u2014 it is what makes a single .95 cue lose to three .6\u2013.9 cues (Exp 4) while making one .95 cue roughly TIE a .70+.85 or .78+.72 pair (Exps 1\u20132, observed ~0.51). Parameter ranges must be chosen so that (i) at rho near 1 the model reproduces ~0.75\u20130.85 tally-following on large-margin conflicts and (ii) near-balanced conflicts land at 0.45\u20130.60.
3. The decision is a softmax on the RAW (unnormalised) weighted difference, beta in roughly [1, 8], so confidence is graded by the evidence margin: big-margin conflicts are decided near-deterministically, balanced conflicts are near chance. Do NOT normalise by total discriminating weight (that is pi_3's move and it destroys the chance-level predictions on balanced cells). Add a lapse epsilon in [0, 0.15] and no side bias (designs are mirrored).
4. Explicitly no serial-position/reading-order term: the weight of a cue depends only on its communicated validity, not where it appears on screen. This is the sharp contrast with pi_3 and it is what buys the Exp 1/Exp 2 fit.
5. Optional (small) individual differences: subject-level jitter on rho and beta only, generating unimodal (not bimodal) between-subject distributions \u2014 consistent with the small observed per-subject variances (0.007\u20130.023).

The predict() function should read the numeric `validities` provided with the stimulus (falling back to a descending default only if lengths mismatch), map them to weights via the compression rule, compute the signed weighted difference, and return the softmax-plus-lapse choice probabilities. Sanity-check the implementation against all four observed metrics before accepting it: target ~0.50 on Exp 1, ~0.51 on Exp 2, ~0.05\u20130.15 on Exp 3, and ~0.75\u20130.80 on Exp 4.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.289)

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
- Experiment 1: real=0.0933 (var=0.0064) vs this=0.0038 (var=0.0080)
- Experiment 2: real=0.7750 (var=0.0072) vs this=0.1530 (var=0.0129)
- Experiment 3: real=0.5175 (var=0.0233) vs this=0.8562 (var=0.0143)
- Experiment 4: real=0.5075 (var=0.0075) vs this=0.8475 (var=0.0132)


---

### `pi_2` (overall score: 0.251)

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


---

### `pi_3` (overall score: 0.000)

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


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1030 -> ACCEPTED
- iter 2: loss=0.1206 -> REJECTED
- iter 3: loss=0.0573 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.0573 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.1377 (var=0.0056) (Δ vs real -0.2310)
  - iter 2: -0.0427 (var=0.0023) (Δ vs real -0.1360)
  - iter 3 (most recent): -0.0356 (var=0.0029) (Δ vs real -0.1290)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0038 (var=0.0080)
- pi_2: -0.6931 (var=0.0366)
- pi_3: 0.1027 (var=0.0215)

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
  - iter 1: 0.7175 (var=0.0065) (Δ vs real -0.0575)
  - iter 2: 0.7580 (var=0.0035) (Δ vs real -0.0170)
  - iter 3 (most recent): 0.7695 (var=0.0020) (Δ vs real -0.0055)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8660 (var=0.0083)
- pi_1: 0.1530 (var=0.0129)
- pi_3: 0.7780 (var=0.0646)

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
  - iter 1: 0.5337 (var=0.0197) (Δ vs real +0.0162)
  - iter 2: 0.7312 (var=0.0438) (Δ vs real +0.2137)
  - iter 3 (most recent): 0.4850 (var=0.0190) (Δ vs real -0.0325)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8562 (var=0.0143)
- pi_3: 0.0375 (var=0.0022)
- pi_2: 0.1400 (var=0.0151)

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
  - iter 1: 0.5250 (var=0.0102) (Δ vs real +0.0175)
  - iter 2: 0.5294 (var=0.0138) (Δ vs real +0.0219)
  - iter 3 (most recent): 0.4913 (var=0.0098) (Δ vs real -0.0162)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0269 (var=0.0009)
- pi_1: 0.8475 (var=0.0132)
- pi_2: 0.1450 (var=0.0133)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Threshold-terminated compensatory integration with a "can-anything-still-overturn-this?" stopping rule.**

People treat the expert ratings as *evidence to be added up*, not as a lexicographic hierarchy. Four claims:

1. **Cue weights depend ONLY on the communicated validity, never on where the rating sits on the screen.** The weight of expert j is a compressed, monotone function of its stated validity, w_j ∝ (v_j − 0.5)^rho, normalised to sum to 1. Compression (rho ≈ 1.25–1.45) lets a crowd of medium-validity experts out-vote or tie a single .95 expert.

2. **Integration is compensatory and additive, carried out incrementally in the order the ratings are displayed, and it can terminate early.** After each rating that actually discriminates, the decision maker performs a *best-case reversal check*: "could any single expert I have not yet read still overturn what I now believe?" Formally, the sufficiency bar at that moment is theta_k = phi · max_{j after the current screen position} w_j, and the probability of stopping is sigmoid(s·(|E_k|/theta_k − 1)). With phi ≈ 1 this is the natural rational rule "stop when no remaining single expert can reverse my lead"; phi slightly above 1 encodes mild conservatism. Nothing is discounted by position; position matters *only* through this truncation.

3. **Choice is a softmax on the RAW accumulated margin at the moment integration stops, plus a lapse.** Confidence is margin-graded and never renormalised by the amount of discriminating evidence, so decisive margins are near-deterministic while balanced margins are near chance. No side bias (designs are mirrored).

4. **Individual differences are unimodal jitter** in compression (rho), conservatism (phi), decision sharpness (s), decisiveness (beta) and lapse — no discrete strategy types, no primacy/recency "reader types".

Signature predictions:
* In a validity-sorted display, the leading .95 rating instantly beats the best remaining single expert (w_1 > phi·w_2), search stops, and behaviour *looks* like Take-The-Best — in both conflict and agreement cells, so the TTB-vs-tally contrast collapses toward zero rather than being strongly negative.
* When the .95 expert is listed **last**, the running bar stays high (it is the .95 cue's own weight) until two or three mid-validity ratings pile up past it; search then stops before the best expert is ever read and behaviour *looks* like tallying — the crowd wins.
* When a strong expert sits in the **middle with a strong trailing expert still unread**, the reversal check fails (that trailing expert *could* overturn the lead), so integration continues, the trailing cue cancels the middle one, the final margin is tiny and choice is near chance.
* When the trailing cues are individually **weak**, the same check succeeds and the mid-list cue that just discriminated decides the trial — which is why weak trailing crowds do not get to compensate even though strong trailing single cues do.

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
    beta = float(np.clip(parameters.get('beta', 11.0), 0.0, 200.0))
    eps = float(np.clip(parameters.get('epsilon', 0.0), 0.0, 1.0))

    # ------------------------------------------------------------------
    # 3. incremental integration in screen order with graded stopping
    #    sufficiency bar = phi * (largest weight among cues NOT YET READ)
    #    i.e. "can any single remaining expert still overturn my lead?"
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
        p_choice = _sig(beta * E)               # softmax on RAW margin
        if k < m - 1:
            idx = int(disc[k])
            if idx + 1 < n:
                suffix_max = float(np.max(w[idx + 1:]))
            else:
                suffix_max = 0.0
            theta_k = max(phi * suffix_max, 1e-12)
            q = _sig(s * (abs(E) / theta_k - 1.0))  # graded termination
        else:
            q = 1.0                              # nothing left to read
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
- beta: [8.0, 14.0]
- epsilon: [0.0, 0.06]
- validities: validities

`rationale`: **Minimal diff.** Everything is re-emitted verbatim from the accepted iter-1 base except ONE block: the sufficiency criterion. The list-global bar `theta = phi*max_j w_j` (computed once before the loop) is replaced by a per-step bar computed inside the loop, `theta_k = phi * max_{j > screen_position(k)} w_j` — the weight of the single strongest expert *not yet read*. Parameter ranges are retuned accordingly (phi moved from [0.80,0.95] to [1.00,1.12], beta from [4,8] to [8,14] as the critic advised, s widened to [12,30], rho kept tight at [1.25,1.45] per the critic's guardrail). No KAPPA floor is introduced. Mechanism family is unchanged: compressed validity-only weights, softmax on the RAW unnormalised accumulated margin, lapse, no side bias, position entering only via read-out truncation.

**Why I drop the critic's KAPPA-floor prescription (while keeping its diagnosis).** The critic correctly identified that iter-2's fixed floor 0.35 was hand-tuned for Exp1 and destroyed Exp3, and proposed fitting KAPPA in [0.45,0.65]. I worked the arithmetic through with rho=1.35, phi=0.85 and the two constraints are *mutually exclusive*: Exp1's tie-tie cell needs phi·KAPPA·w(.95) < w(.78), i.e. KAPPA < (0.28/0.45)^1.35/0.85 = 0.62, while Exp3 needs phi·KAPPA·w(.95) > w(.95)−w(pos0), i.e. KAPPA > 0.64 for the pos-0 weight implied by the observed near-perfect cancellation. The KAPPA window is empty, so that edit would trade one experiment for the other again.

**What actually separates the two cells is the suffix MAX, not a floor.** In Exp3 the cue that still lies ahead of the .95 expert (screen position 5) is *individually strong* — strong enough that the observed data's cancellation (w0+w5 ≈ w3) forces |E| after the .95 cue ≈ w5 exactly. In Exp1's broken cell the cues still ahead (.72, .60) are individually weak (each < the just-read .78), even though their SUM is comparable. So a bar equal to the largest *remaining single* weight, with phi just above 1, stops in Exp1's cell (ratio ≈ w(.78)/(phi·w(.72)) ≈ 1.3 → stop) and keeps reading in Exp3 (ratio ≈ 1/phi < 1 → continue, the trailing cue cancels, chance-level output). This ratio-≈-1/phi identity in Exp3 holds for *any* trailing-cue weight given the observed cancellation, so the fix is robust to the Exp3 validities I cannot see — unlike a hard-coded floor.

**Hand-simulated targets (rho=1.35, phi=1.05, s=20, beta=11), calibrated against the reported iter-1 and iter-2 values so the arithmetic is validated on known outputs:** Exp1 conflict cells 0.96/0.96/0.885 vs agreement 0.981 → metric ≈ −0.045 (iter2 achieved −0.043 by the same route); Exp2 = (3×0.98 + 2×0.443)/5 ≈ 0.76 (real 0.775) — crucially the .95 cue at position 5 stays in the suffix on the weak-crowd cells 9/10, so their near-chance value is preserved; Exp3 ≈ 0.49–0.56 (real 0.518) because the reversal check fails at the .95 cue and the trailing extreme cue is read; Exp4 = (0.59+0.39)/2 ≈ 0.49 (real 0.508) because the strong middle cues merely cancel. Expected squared-gap sum drops from ~0.057 (iter-1 base) to ~0.024, i.e. comfortably below the 0.1030 accept floor.

**Experiment invariance.** The stopping rule is now fully scale-free and free of hand-set constants: it is the rational "no single unread expert can reverse my lead" check, expressed only in terms of the stimulus's own validity-derived weights, so it transfers unchanged across feature counts, validity vectors, and screen layouts.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is currently the running best (loss 0.1030, ACCEPTED), and three of the four experiments are essentially nailed: Exp2 0.718 vs 0.775 (small under-shoot), Exp3 0.534 vs 0.518, Exp4 0.525 vs 0.508 \u2014 all within noise-ish range and, crucially, without the catastrophic extremes that pi_1 (0.86/0.85) and pi_3 (0.04/0.03) produced on Exps 3\u20134. Nearly all remaining loss lives in Exp1: -0.138 vs observed +0.093, a 0.23 gap, and it is diagnosable cell-by-cell. In Exp1's validity-sorted display the .95 cue arrives first, |E| = w1 > theta = phi\u00b7w_max, so the model terminates and produces near-ceiling top-cue following in the .95-vs-crowd conflict cells AND in the agreement cells \u2014 those cells cancel out and contribute ~0 to the metric, exactly as the theory intends. The damage comes from the one conflict cell where the first two cues tie (A=[1,1,0,1,1] vs B=[1,1,1,0,0]): here integration starts at the .78 cue, |E| = w(.78) never reaches phi\u00b7w(.95), so the model keeps reading, the .72 and .60 cues almost exactly cancel the .78, and the final margin is ~0.02 \u2192 p(TTB side) \u2248 0.53 while the agreement cells sit at ~0.90. Averaged over the three conflict cells that gives (0.90+0.90+0.53)/3 - 0.90 \u2248 -0.13, which is precisely the simulated value. So the failure is not the mechanism family but the *scaling of the sufficiency criterion*: theta anchored to phi\u00b7max_j w_j is far too high whenever the top cue does not discriminate, forcing exhaustive integration and chance-level output in cells where humans are decisive. On faithfulness: the candidate keeps the arbiter's core (validity-only compressed weights, softmax on the RAW unnormalised margin, lapse, no side bias, unimodal jitter) and injects position only via truncation of the read-out, not via position-discounted weights \u2014 that is a minor, in-family extension rather than a drift into pi_3's territory, and it is what buys the good Exp3/Exp4 fits.
Rationale: Keep the prescribed family exactly as-is (compressed validity weights w_j \u221d (v_j-0.5)^rho, softmax on the raw unnormalised accumulated margin, lapse, no side bias, no position-dependent WEIGHTS) and change one knob: the sufficiency criterion. Replace theta = phi\u00b7max_j w_j (a fixed, list-global bar) with a *remaining-evidence* bar: after reading cue k, stop with probability sigmoid(s\u00b7(|E_k| / (phi\u00b7\u03a3_{j>k} w_j) - 1)) \u2014 i.e. stop when what has been read is already large relative to what could still overturn it (optionally blend: theta_k = phi\u00b7max(\u03a3_{j>k} w_j, w_min_floor) so it never divides by ~0 at the last cue). This is a scale-free, experiment-invariant reformulation of the same self-terminating rule, and it repairs the specific broken cell: in Exp1's tie-tie-.78-vs-.72+.60 conflict, the remaining weight after the .78 cue is only w(.72)+w(.60), so the model terminates and follows the .78 cue at ceiling instead of drifting to 0.53; the agreement cells also terminate at ceiling, so Exp1 lands near 0.00\u2013+0.05 instead of -0.14 (a ~0.15\u20130.20 loss reduction, and comparable to pi_1's 0.004 / pi_3's 0.103 without their Exp3/Exp4 blow-ups). Verify while tuning that (a) Exp2 does NOT regress: with the .95 cue last, the remaining-weight bar is large early on, so the .9 cue at position 0 must not terminate search on its own \u2014 pick phi in roughly [0.4, 0.8] and re-check that tally-following stays in 0.74\u20130.80 (currently 0.718, so a small upward nudge is desirable, e.g. slightly larger s so the 2\u20133-cue crowd terminates reliably); (b) Exp3 stays in 0.48\u20130.56 and Exp4 in 0.48\u20130.55 \u2014 these are the cells where the strong middle cue must merely CANCEL, so make sure the new criterion does not let the leading extreme cue terminate before position 3/2 is read (if it does, raise phi rather than reintroducing any position weighting). Also widen rho slightly upward (e.g. [1.2, 2.5]) since less compression helps the .78-vs-.72+.60 cell without hurting Exp2 (whose tally-following is carried by truncation, not by weight compression). Do not normalise evidence by total discriminating weight, do not add position-discounted weights, and do not add discrete strategy types.

**Outcome of this advice:** iter 1 candidate loss=0.1030 -> iter 2 candidate loss=0.1206 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: This iteration is a clean, informative trade: the suffix-max sufficiency bar did exactly what I asked it to do on the two experiments I flagged \u2014 Exp1 moved from -0.138 to -0.043 (gap to observed +0.093 shrank from 0.231 to 0.136) and Exp2 moved from 0.718 to 0.758 (gap 0.058 \u2192 0.017), and Exp4 stayed put (0.529 vs 0.508). But it blew a hole in Exp3: 0.534 \u2192 0.731 against an observed 0.518, a +0.214 error that is larger than the Exp1 gain, hence the gate rejection (0.1206 vs 0.1030). The mechanism of the regression is diagnosable. In Exp3's double-dissociation cells the two screen-extreme cues (positions 0 and 5) jointly oppose the .95 cue at position 3, and human behaviour is exactly at chance, which the model can only reproduce if it reads ALL the way to position 5 so that w0 + w5 roughly cancels w(.95). Under the new rule, once the .95 cue at position 3 is read the only remaining cue is the weak position-5 extreme, so suffix_max collapses and the bar reduces to the KAPPA floor: theta_3 = phi\u00b70.35\u00b7w_max \u2248 the accumulated |E| = w(.95) - w0. With s\u224820 that is a coin-flip termination probability, so on roughly half the trials search stops the moment the .95 arrives and the model follows the top cue \u2014 output drifts to 0.73, i.e. towards pi_1's 0.86 failure mode. So the KAPPA floor is not merely a numerical guard: it is the single parameter that arbitrates between the Exp1 gain (needs a LOW floor so a mid-validity cue can terminate when only weaker cues remain) and the Exp3 requirement (needs a HIGH floor so a strong middle cue cannot terminate before the trailing extreme cue is read). It was hard-coded at 0.35, i.e. tuned by hand for Exp1 and never checked against Exp3. On faithfulness: the candidate remains inside the arbiter's family (compressed validity-only weights, softmax on the RAW unnormalised margin, lapse, no side bias, position enters only through read-out truncation, unimodal jitter) \u2014 no drift into pi_3's normalisation or into discrete strategy types.
Rationale: The gate rejected this attempt, so the next proposal must be built on the iter-1 accepted base again \u2014 but do NOT throw away this iteration's insight, since the suffix-max bar demonstrably bought ~0.10 on Exp1 and ~0.04 on Exp2. The single change I want is to stop hard-coding the sufficiency floor: promote KAPPA to a fitted parameter, KAPPA \u2208 [0.45, 0.65] (up from the fixed 0.35), keeping theta_k = phi\u00b7max(max_{j>k} w_j, KAPPA\u00b7max_j w_j) otherwise verbatim. Arithmetic sanity check with rho\u22481.35, phi\u22480.85: Exp1's broken cell needs phi\u00b7max(w(.72), KAPPA\u00b7w(.95)) < w(.78), i.e. KAPPA \u2272 0.6, so the Exp1 repair survives; Exp3 needs phi\u00b7KAPPA\u00b7w(.95) > w(.95) - w(pos0), which fails at KAPPA=0.35 (borderline, ~50% premature stop) but holds comfortably at KAPPA\u22480.5\u20130.55, restoring the read-to-the-end cancellation that produced the good 0.534. Target: Exp1 in [-0.06, +0.05], Exp2 in [0.74, 0.80], Exp3 back to [0.48, 0.58], Exp4 in [0.48, 0.55]; explicitly print all four before shipping \u2014 the failure this round was that Exp3 was never re-simulated after the floor was chosen. Two further guardrails. (1) Do NOT follow my earlier suggestion to widen rho upward: I now see that less compression is exactly what pushes Exp3 and Exp4 toward the pi_1 blow-up (a single .95 must roughly TIE the {.78,.72} pair and roughly tie w0+w5), so keep rho tightly around 1.25\u20131.45. That earlier direction was part of a rejected step; drop it. (2) Do NOT go back to the sum-form suffix bar (the proposer showed it collapses Exp1 to ~0.50) and do NOT reintroduce a list-global constant bar (that was iter 1's -0.138 problem). Optionally, if KAPPA alone leaves Exp3 slightly high, lower s to ~8\u201315 so borderline cells (|E|/theta \u2248 1) mix rather than committing, which softens the top-cue-following in Exp3 while barely touching Exp1/Exp2 where |E|/theta is far from 1. Keep beta \u2208 [8,14] (it helped Exp2) and everything else unchanged.

**Outcome of this advice:** iter 2 candidate loss=0.1206 -> iter 3 candidate loss=0.0573 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: This iteration is the best of the loop (0.0573, ACCEPTED) and the improvement is real and diagnosable: swapping the list-global sufficiency bar for the per-step suffix-max bar (theta_k = phi * max weight among unread cues) fixed the two things I flagged without re-breaking Exp3. Exp2 is now essentially exact (0.7695 vs 0.775), Exp3 is back at chance-level (0.485 vs 0.518) because the reversal check fails at the .95 cue and the trailing extreme cue is read and cancels it, and Exp4 is close (0.491 vs 0.508). Faithfulness to the arbiter's family is intact: compressed validity-only weights w_j \u221d (v_j-0.5)^rho, softmax on the RAW unnormalised accumulated margin, lapse, no side bias, unimodal jitter; screen position enters only through read-out truncation, never through position-discounted weights or pi_3-style normalisation. No drift.

All of the residual loss now sits in Exp1: -0.0356 vs observed +0.0933 (gap 0.129), while Exp2/3/4 residuals are 0.006/0.033/0.016. The Exp1 sign is structurally hard for this family: agreement cells always accumulate evidence in a single direction, so continued reading can only help there, whereas the one conflict cell whose first two cues tie (A=[1,1,0,1,1] vs B=[1,1,1,0,0]) can be overturned when the .72/.60 cues outweigh the just-read .78 cue. Any self-terminating additive integrator therefore predicts conflict <= agreement, i.e. metric <= ~0. Note that pure TTB (pi_1) itself only reaches 0.004 on this metric and the per-subject variance is 0.0064 (sd ~0.08), so +0.093 is probably partly sampling noise; the realistic in-family target is ~0.00 to +0.03, not +0.09. Getting from -0.036 to ~0 is worth roughly a third of the remaining loss.

Usefully, the three small residuals all point in the SAME direction as the Exp1 residual: Exp3 is 0.033 too low, Exp4 0.016 too low, and Exp1's broken cell is too permissive \u2014 all four want slightly MORE early termination (a marginally lower effective bar / sharper commitment), not less. That is a single-knob, minimal-diff tuning target rather than a re-design.
Rationale: Keep the mechanism exactly as shipped in iter 3 (this is the accepted base and it must not be re-architected): compressed validity-only weights, incremental read in screen order, suffix-max reversal check theta_k = phi * max_{j>pos(k)} w_j, softmax on the RAW margin, lapse, no side bias, no KAPPA floor, no sum-form bar, no position-dependent weights. Change only the phi/s calibration, jointly, in a small step.

Diagnosis and direction: all four residuals ask for slightly more early stopping. Exp3 (0.485 vs 0.518) and Exp4 (0.491 vs 0.508) are marginally too chance-like, and Exp1's tie-tie conflict cell (~0.885 by the proposer's own arithmetic, versus ~0.98 in the agreement cells) is what drags Exp1 to -0.036. In Exp3 the stop probability at the .95 cue is sigmoid(s*(1/phi - 1)); with phi=1.05, s=20 that is ~0.27, and it is EXTREMELY sensitive to phi \u2014 dropping phi below 1 at s=20 flips it above 0.7 and reproduces the iter-2 blow-up (0.731). So do NOT lower phi at fixed high s. Instead soften the cliff first: lower s to roughly [6, 16] so borderline cells (|E|/theta ~ 1) mix rather than commit, and only then allow phi to come down slightly, e.g. phi in [0.98, 1.10] (overlapping the current [1.00,1.12], a strictly small step, and NOT the sub-0.95 territory that killed Exp3 last time). Keep rho tight at [1.25,1.45] (widening it upward was part of my iter-1 advice that the gate rejected \u2014 do not revisit) and keep beta in [8,14], epsilon <= 0.06.

Acceptance targets to print BEFORE shipping (the iter-2 failure was not re-simulating everything after a knob change): Exp1 in [-0.01, +0.05], Exp2 in [0.75, 0.79], Exp3 in [0.50, 0.57], Exp4 in [0.50, 0.54]. If the phi/s pair cannot lift Exp1 without pushing Exp3 above 0.60, stop and keep the iter-3 setting \u2014 Exp1's remaining gap is largely a structural/noise ceiling (pure TTB itself scores only 0.004 there), so trading Exp3 for Exp1 is a losing move and the gate has already punished that trade once. A cheap secondary lever if phi/s alone stalls: add modest between-subject jitter on phi (unimodal, e.g. sd ~0.03) so the population averages over the sharp ratio-1 boundary; this smooths Exp3/Exp4 toward their observed values without any new mechanism.

**Outcome of this advice:** iter 3 candidate loss=0.0573 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**Threshold-terminated compensatory integration with a \"can-anything-still-overturn-this?\" stopping rule.**\n\nPeople treat the expert ratings as *evidence to be added up*, not as a lexicographic hierarchy. Four claims:\n\n1. **Cue weights depend ONLY on the communicated validity, never on where the rating sits on the screen.** The weight of expert j is a compressed, monotone function of its stated validity, w_j \u221d (v_j \u2212 0.5)^rho, normalised to sum to 1. Compression (rho \u2248 1.25\u20131.45) lets a crowd of medium-validity experts out-vote or tie a single .95 expert.\n\n2. **Integration is compensatory and additive, carried out incrementally in the order the ratings are displayed, and it can terminate early.** After each rating that actually discriminates, the decision maker performs a *best-case reversal check*: \"could any single expert I have not yet read still overturn what I now believe?\" Formally, the sufficiency bar at that moment is theta_k = phi \u00b7 max_{j after the current screen position} w_j, and the probability of stopping is sigmoid(s\u00b7(|E_k|/theta_k \u2212 1)). With phi \u2248 1 this is the rational rule \"stop when no remaining single expert can reverse my lead\"; phi is a mild conservatism/impatience knob that sits *just* around 1, and the population spans both sides of it (some readers are slightly impatient, phi < 1, some slightly conservative, phi > 1). Nothing is discounted by position; position matters *only* through this truncation.\n\n3. **Choice is a softmax on the RAW accumulated margin at the moment integration stops, plus a lapse.** Confidence is margin-graded and never renormalised by the amount of discriminating evidence, so decisive margins are near-deterministic while balanced margins are near chance. No side bias (designs are mirrored).\n\n4. **Individual differences are unimodal jitter** in compression (rho), conservatism/impatience (phi), decision sharpness (s), decisiveness (beta) and lapse \u2014 no discrete strategy types, no primacy/recency \"reader types\".\n\nSignature predictions:\n* In a validity-sorted display, the leading .95 rating instantly beats the best remaining single expert (w_1 > phi\u00b7w_2), search stops, and behaviour *looks* like Take-The-Best \u2014 in both conflict and agreement cells, so the TTB-vs-tally contrast collapses toward zero rather than being strongly negative.\n* When the .95 expert is listed **last**, the running bar stays high (it is the .95 cue's own weight) until two or three mid-validity ratings pile up past it; search then stops before the best expert is ever read and behaviour *looks* like tallying \u2014 the crowd wins.\n* When a strong expert sits in the **middle with a comparably strong trailing expert still unread**, the reversal check sits right at its indifference point (|E|/theta \u2248 1): the population *splits*, roughly a third of trials terminate on the strong middle cue and the rest read on and let the trailing cue cancel it, so the aggregate is slightly above chance rather than at either extreme.\n* When the trailing cues are individually **weak**, the same check succeeds decisively and the mid-list cue that just discriminated decides the trial \u2014 which is why weak trailing crowds do not get to compensate even though strong trailing single cues do.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ------------------------------------------------------------------\n    # 1. parse the trial stimulus into two rating vectors (screen order)\n    # ------------------------------------------------------------------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        if 'option_a_ratings' in state and 'option_b_ratings' in state:\n            a = np.asarray(list(state['option_a_ratings']), dtype=float)\n            b = np.asarray(list(state['option_b_ratings']), dtype=float)\n    if a is None:\n        arr = np.asarray(state, dtype=float)\n        if arr.ndim == 1:\n            arr = arr.reshape(2, -1)\n        elif arr.ndim > 2:\n            arr = arr.reshape(2, -1)\n        if arr.shape[0] != 2:\n            arr = arr.reshape(2, -1)\n        a = arr[0].astype(float)\n        b = arr[1].astype(float)\n\n    a = np.ravel(a).astype(float)\n    b = np.ravel(b).astype(float)\n    n = int(min(a.shape[0], b.shape[0]))\n    if n == 0:\n        return np.array([0.5, 0.5])\n    a = a[:n]\n    b = b[:n]\n    d = a - b\n\n    # ------------------------------------------------------------------\n    # 2. validity-only cue weights, compressed toward equality\n    # ------------------------------------------------------------------\n    val = parameters.get('validities', None)\n    try:\n        val = np.asarray(val, dtype=float).ravel()\n    except Exception:\n        val = None\n    if val is None or val.shape[0] != n:\n        # fall back: assume display is validity-sorted, descending\n        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])\n    val = np.clip(val, 0.5 + 1e-6, 1.0)\n\n    rho = float(parameters.get('rho', 1.35))\n    rho = float(np.clip(rho, 0.1, 5.0))\n    raw = np.clip(val - 0.5, 1e-9, None) ** rho\n    tot = float(np.sum(raw))\n    if not np.isfinite(tot) or tot <= 0:\n        w = np.ones(n, dtype=float) / n\n    else:\n        w = raw / tot                      # weights sum to 1, position-free\n\n    phi = float(np.clip(parameters.get('phi', 1.035), 0.05, 3.0))\n    s = float(np.clip(parameters.get('s', 21.0), 0.1, 200.0))\n    beta = float(np.clip(parameters.get('beta', 11.0), 0.0, 200.0))\n    eps = float(np.clip(parameters.get('epsilon', 0.0), 0.0, 1.0))\n\n    # ------------------------------------------------------------------\n    # 3. incremental integration in screen order with graded stopping\n    #    sufficiency bar = phi * (largest weight among cues NOT YET READ)\n    #    i.e. \"can any single remaining expert still overturn my lead?\"\n    # ------------------------------------------------------------------\n    disc = np.nonzero(np.abs(d) > 1e-12)[0]     # discriminating cues, in order\n    if disc.size == 0:\n        return np.array([0.5, 0.5])\n\n    steps = w[disc] * np.sign(d[disc])\n    cum = np.cumsum(steps)                      # margin after each new reason\n    m = cum.shape[0]\n\n    def _sig(x):\n        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))\n\n    p_a = 0.0\n    remain = 1.0\n    for k in range(m):\n        E = float(cum[k])\n        p_choice = _sig(beta * E)               # softmax on RAW margin\n        if k < m - 1:\n            idx = int(disc[k])\n            if idx + 1 < n:\n                suffix_max = float(np.max(w[idx + 1:]))\n            else:\n                suffix_max = 0.0\n            theta_k = max(phi * suffix_max, 1e-12)\n            q = _sig(s * (abs(E) / theta_k - 1.0))  # graded termination\n        else:\n            q = 1.0                              # nothing left to read\n        p_a += remain * q * p_choice\n        remain *= (1.0 - q)\n        if remain <= 1e-12:\n            break\n    if remain > 1e-12:                           # numerical remainder\n        p_a += remain * _sig(beta * float(cum[-1]))\n\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    p = (1.0 - eps) * p + eps * 0.5              # lapse, no side bias\n    p = np.clip(p, 1e-12, None)\n    p = p / p.sum()\n    return p",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if not np.isfinite(s) or s <= 0:\n        probs = np.ones_like(probs) / len(probs)\n    else:\n        probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "rho": "[1.25, 1.45]",
    "phi": "[0.99, 1.08]",
    "s": "[16.0, 26.0]",
    "beta": "[8.0, 14.0]",
    "epsilon": "[0.0, 0.06]",
    "validities": "validities"
  },
  "rationale": "**Minimal diff: two parameter ranges changed, zero lines of mechanism touched.** `predict` and `policy` are re-emitted verbatim from the accepted iter-3 base (compressed validity-only weights, incremental read in screen order, suffix-max reversal check theta_k = phi\u00b7max_{j>pos(k)} w_j, softmax on the RAW accumulated margin, lapse, no side bias, no KAPPA floor, no position-dependent weights). Only `phi` moves from [1.00, 1.12] to [0.99, 1.08] and `s` from [12, 30] to [16, 26]. `rho`, `beta`, `epsilon` are untouched, exactly as the critic instructed.\n\n**Why this specific pair, and why NOT the critic's literal s\u2208[6,16] suggestion.** The critic is right that all four residuals ask for slightly more early stopping, and right that the Exp3 stop probability sigmoid(s\u00b7(1/phi\u22121)) is the single controlling quantity. But I re-did the arithmetic on the *other* cells before shipping (the iter-2 failure mode was exactly not doing this), and lowering s hurts two cells the critic did not check:\n\n* Exp1 conflict cells 1/2/8/10 (single .95 cue vs a 3\u20134-cue crowd) currently terminate at position 0 with probability sigmoid(s\u00b7(w(.95)/(phi\u00b7w(.88)) \u2212 1)) \u2248 sigmoid(21\u00b70.20) = 0.98; the continuation branch flips the choice (p(top)\u22480.04). Dropping s to ~11 drops that to ~0.90 and costs ~0.03 on each of four dis cells, i.e. it pushes Exp1 *down* from \u22120.036, the opposite of the goal.\n* Exp4's counted cell 7 (|E|/theta \u2248 0.69) is far from the ratio-1 boundary, so lowering s buys almost nothing there while doing the Exp1 damage above.\n\nSo instead of moving s down I keep s at essentially its current mean (21) and move **phi** down by ~0.025 in mean (1.06 \u2192 1.035), spanning both sides of 1 as unimodal between-subject jitter. This is exactly the \"soften the ratio-1 boundary by averaging over phi\" lever the critic listed as the cheap secondary option, implemented with no new mechanism.\n\n**Hand-simulated targets before shipping (rho\u22481.35, phi\u2208[0.99,1.08], s\u2208[16,26], beta\u2208[8,14], eps\u22640.06):**\n* Exp1: cells 7/9 still stop at ratio 1.39/phi \u2248 1.34 (q\u22481.0) and cells 1/2/8/10 still stop at ratio 1.26/phi \u2248 1.21 with q\u22480.97\u20130.99 \u2014 metric stays \u2248 \u22120.035, i.e. no regression on the one experiment carrying most of the loss.\n* Exp2: the .95 cue sits last, so it stays in the suffix throughout the weak-crowd cells 9/10 (\u22480.44) while the 3-cue crowd cells terminate at ceiling \u2192 \u2248 0.77 (real 0.775), unchanged/slightly better.\n* Exp3: at the .95 cue the ratio is \u22481/phi, so E[q] over the new phi\u00d7s box is \u22480.35 (was \u22480.23), giving 0.32 + 0.58\u00b70.35 \u2248 **0.52** (real 0.5175) \u2014 this closes the loop's second-largest residual (0.033 \u2192 \u22480.00) and, crucially, stays far from the iter-2 blow-up (0.73) because phi never goes below 0.99 while s stays high.\n* Exp4: \u2248 (0.591 + 0.395)/2 \u2248 **0.493** (real 0.5075), essentially unchanged.\n\nExpected squared-gap sum \u2248 0.0169 vs the base's 0.0180, driven almost entirely by Exp3 \u2014 a genuine, low-risk improvement rather than an Exp1-for-Exp3 trade, which the gate has already punished once.\n\n**Experiment invariance.** Nothing here is stimulus-specific: the stopping bar remains the scale-free \"no single unread expert can reverse my lead\" check expressed purely in the stimulus's own validity-derived weights, and the only change is that the population's conservatism parameter is now centred on the rational value phi = 1 rather than biased above it."
}
```

## Usage

```json
{
  "input_tokens": 26326,
  "output_tokens": 26218
}
```
