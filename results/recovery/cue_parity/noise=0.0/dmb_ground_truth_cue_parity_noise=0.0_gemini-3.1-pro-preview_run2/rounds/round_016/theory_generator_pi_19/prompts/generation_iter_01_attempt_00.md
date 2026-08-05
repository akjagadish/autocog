# generation_iter_01_attempt_00

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
- THEORY 1 = `pi_16`
- THEORY 2 = `pi_18`
- The recommendation below acts on THEORY 2 (= `pi_18`).

Propose a new theory that moves away from the rigid dual-process mixture of unweighted tally and validity-weighted sums. The new theory should consider 'Cue-Depletion or Diminishing Returns Integration': instead of cues adding up linearly, each additional cue favoring an option provides logarithmically diminishing evidence, and cues might interact competitively. Alternatively, consider a 'Rank-Weighted Sequential Sampling' model where cues are evaluated in order of validity, but the stopping probability depends on the absolute difference in cue values. This would allow the model to sometimes stop early (ignoring tallies) or integrate heavily, capturing the low tally reliance seen in Exp 2, 16, and 21.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_16` (overall score: 0.444)

**Description**
Leaky Competing Accumulator with Non-linear Configural Weighting and Evidence Transduction: Decision-makers evaluate cues simultaneously, with evidence for each option dynamically inhibiting the other in a leaky competing accumulator (LCA). Cues have a configural impact, scaled non-linearly by the total number of supporting cues. Additionally, the integrated evidence for each option is passed through a non-linear transducer (alpha) before entering the accumulation process, allowing the model to amplify the differences driven by high-validity cues and capture strong non-compensatory reversals.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    # Configural weighting: effective validity of a cue is non-linearly modulated by the total number of cues
    # Using max(1e-6, sum) to avoid 0^negative_gamma undefined errors
    sum_a_safe = max(1e-6, sum_a)
    sum_b_safe = max(1e-6, sum_b)
    
    w_a = val * (sum_a_safe ** gamma)
    w_b = val * (sum_b_safe ** gamma)
    
    # Make sure inputs are non-negative and apply non-linear transducer alpha
    I_A = max(0.0, np.sum(w_a * a)) ** alpha
    I_B = max(0.0, np.sum(w_b * b)) ** alpha
    
    # Leaky Competing Accumulator (LCA) simulation
    x_a, x_b = 0.0, 0.0
    dt = 0.1
    steps = 100
    
    for _ in range(steps):
        dx_a = (I_A - leak * x_a - inhibition * x_b) * dt
        dx_b = (I_B - leak * x_b - inhibition * x_a) * dt
        
        x_a = max(0.0, x_a + dx_a)
        x_b = max(0.0, x_b + dx_b)
        
    # Softmax choice based on final activations
    z = theta * np.array([x_a, x_b])
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [-10.0, 10.0]
- leak: [0.1, 2.0]
- inhibition: [0.0, 5.0]
- theta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- alpha: [0.1, 10.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7417 (var=0.0032) vs this=0.4573 (var=0.0351)
- Experiment 2: real=0.2881 (var=0.0100) vs this=0.5750 (var=0.1191)
- Experiment 3: real=0.1389 (var=0.0317) vs this=0.3928 (var=0.2999)
- Experiment 4: real=0.0000 (var=0.0233) vs this=0.0600 (var=0.0228)
- Experiment 5: real=-0.4333 (var=0.0244) vs this=-0.1993 (var=0.0126)
- Experiment 6: real=0.4946 (var=0.0018) vs this=0.6073 (var=0.0105)
- Experiment 7: real=0.3633 (var=0.0090) vs this=0.1622 (var=0.0062)
- Experiment 8: real=0.3389 (var=0.0140) vs this=0.1267 (var=0.0151)
- Experiment 9: real=0.8892 (var=0.0052) vs this=0.6842 (var=0.0493)
- Experiment 10: real=0.4900 (var=0.0032) vs this=0.5425 (var=0.0144)
- Experiment 11: real=0.3556 (var=0.0405) vs this=0.3667 (var=0.0426)
- Experiment 12: real=0.0067 (var=0.0065) vs this=0.1050 (var=0.0357)
- Experiment 13: real=0.1240 (var=0.0162) vs this=0.3700 (var=0.0849)
- Experiment 14: real=0.2000 (var=0.0131) vs this=0.5883 (var=0.1241)
- Experiment 15: real=0.8633 (var=0.0110) vs this=0.6267 (var=0.0842)
- Experiment 16: real=0.6222 (var=0.0019) vs this=0.6447 (var=0.0651)
- Experiment 17: real=0.5092 (var=0.0035) vs this=0.3846 (var=0.0860)
- Experiment 18: real=0.6858 (var=0.0032) vs this=0.5667 (var=0.0670)
- Experiment 19: real=0.3050 (var=0.0041) vs this=0.4083 (var=0.1237)
- Experiment 20: real=0.8700 (var=0.0323) vs this=0.6450 (var=1.2247)
- Experiment 21: real=-0.6578 (var=0.0661) vs this=-0.4867 (var=0.1140)
- Experiment 22: real=0.6108 (var=0.0044) vs this=0.5713 (var=0.0096)
- Experiment 23: real=0.8600 (var=0.0198) vs this=0.4250 (var=0.1701)
- Experiment 24: real=0.5813 (var=0.0017) vs this=0.4517 (var=0.1186)
- Experiment 25: real=0.6217 (var=0.0020) vs this=0.5864 (var=0.0792)
- Experiment 26: real=-0.0111 (var=0.0096) vs this=-0.0178 (var=0.0459)
- Experiment 27: real=0.7800 (var=0.0444) vs this=0.4083 (var=0.1940)
- Experiment 28: real=-0.3346 (var=0.0085) vs this=0.3668 (var=0.2179)
- Experiment 29: real=0.0578 (var=0.0307) vs this=0.2356 (var=0.0941)
- Experiment 30: real=-0.0033 (var=0.0073) vs this=0.1850 (var=0.0634)
- Experiment 31: real=0.7617 (var=0.0238) vs this=0.2429 (var=0.0438)
- Experiment 32: real=0.3800 (var=0.0041) vs this=0.3042 (var=0.0229)
- Experiment 33: real=0.0000 (var=0.0126) vs this=-0.0900 (var=0.0201)
- Experiment 34: real=0.3504 (var=0.0038) vs this=0.6140 (var=0.0735)


---

### `pi_4` (overall score: 0.365)

**Description**
Decision-makers do not exclusively rely on a single decision rule; rather, choices reflect a probabilistic mixture of strategies on a trial-by-trial basis (or heterogeneity across individuals). The Strategy Mixture theory posits that a decision is made by selecting either a non-compensatory strategy (Take The Best) with probability alpha, or a compensatory strategy (Tallying) with probability 1 - alpha. Take The Best relies solely on the most valid discriminating cue, making it highly frugal and non-compensatory, whereas Tallying integrates information by counting the total number of cues favoring each option. By averaging across these two distinct decision rules, the model seamlessly bridges the non-compensatory invariance observed in Experiment 2 and the compensatory sensitivity observed in Experiment 1.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture model expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) component
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    # Tallying component
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for TTB
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # Softmax for Tallying
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_tally
    
    # Lapse rate
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7417 (var=0.0032) vs this=0.5490 (var=0.0315)
- Experiment 2: real=0.2881 (var=0.0100) vs this=0.5057 (var=0.0439)
- Experiment 3: real=0.1389 (var=0.0317) vs this=0.3678 (var=0.0551)
- Experiment 4: real=0.0000 (var=0.0233) vs this=-0.0017 (var=0.0332)
- Experiment 5: real=-0.4333 (var=0.0244) vs this=0.0020 (var=0.0116)
- Experiment 6: real=0.4946 (var=0.0018) vs this=0.6727 (var=0.0153)
- Experiment 7: real=0.3633 (var=0.0090) vs this=0.1686 (var=0.0089)
- Experiment 8: real=0.3389 (var=0.0140) vs this=0.0772 (var=0.0106)
- Experiment 9: real=0.8892 (var=0.0052) vs this=0.4721 (var=0.0460)
- Experiment 10: real=0.4900 (var=0.0032) vs this=0.4167 (var=0.0134)
- Experiment 11: real=0.3556 (var=0.0405) vs this=0.9622 (var=0.2112)
- Experiment 12: real=0.0067 (var=0.0065) vs this=0.3650 (var=0.0600)
- Experiment 13: real=0.1240 (var=0.0162) vs this=0.5140 (var=0.0616)
- Experiment 14: real=0.2000 (var=0.0131) vs this=0.5050 (var=0.0446)
- Experiment 15: real=0.8633 (var=0.0110) vs this=0.4800 (var=0.0604)
- Experiment 16: real=0.6222 (var=0.0019) vs this=0.4867 (var=0.0403)
- Experiment 17: real=0.5092 (var=0.0035) vs this=0.4763 (var=0.0454)
- Experiment 18: real=0.6858 (var=0.0032) vs this=0.4975 (var=0.0537)
- Experiment 19: real=0.3050 (var=0.0041) vs this=0.5154 (var=0.0646)
- Experiment 20: real=0.8700 (var=0.0323) vs this=0.6500 (var=0.4164)
- Experiment 21: real=-0.6578 (var=0.0661) vs this=-0.6022 (var=0.0657)
- Experiment 22: real=0.6108 (var=0.0044) vs this=0.7471 (var=0.0070)
- Experiment 23: real=0.8600 (var=0.0198) vs this=0.4250 (var=0.0656)
- Experiment 24: real=0.5813 (var=0.0017) vs this=0.5117 (var=0.0543)
- Experiment 25: real=0.6217 (var=0.0020) vs this=0.7761 (var=0.0049)
- Experiment 26: real=-0.0111 (var=0.0096) vs this=0.0067 (var=0.0137)
- Experiment 27: real=0.7800 (var=0.0444) vs this=0.3100 (var=0.1225)
- Experiment 28: real=-0.3346 (var=0.0085) vs this=0.2302 (var=0.0323)
- Experiment 29: real=0.0578 (var=0.0307) vs this=-0.0333 (var=0.1038)
- Experiment 30: real=-0.0033 (var=0.0073) vs this=-0.0333 (var=0.0169)
- Experiment 31: real=0.7617 (var=0.0238) vs this=0.0167 (var=0.0216)
- Experiment 32: real=0.3800 (var=0.0041) vs this=0.6467 (var=0.0176)
- Experiment 33: real=0.0000 (var=0.0126) vs this=-0.0106 (var=0.0228)
- Experiment 34: real=0.3504 (var=0.0038) vs this=0.4731 (var=0.0411)


---

### `pi_15` (overall score: 0.361)

**Description**
Environment-Contingent Strategy Selection with Mean-Relative Dominant Cue Sensitivity: Decision-makers select between non-compensatory (Take-The-Best) and compensatory (Tallying) heuristics based on the structural properties of the environment. Specifically, the probability of deploying Take-The-Best increases as a logistic function of the difference between the top cue's validity and the average validity of all cues. In environments where the top cue strongly stands out from the overall cue distribution, individuals rely on TTB; when validities are relatively flat, they fall back to Tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    threshold = float(parameters["threshold"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Structural property: top cue vs mean validity
    sorted_val = np.sort(val)[::-1]
    dispersion = sorted_val[0] - np.mean(val)
    
    # Probability of using TTB based on environment structure
    # Using logistic function to map dispersion to probability
    p_ttb_strategy = 1.0 / (1.0 + np.exp(-gamma * (dispersion - threshold)))
    
    # Take-The-Best (TTB) Strategy
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    z_ttb = beta_ttb * scores_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    p_ttb_choice = e_ttb / np.sum(e_ttb)
    
    # Tallying Strategy
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    scores_tally = np.array([tally_a, tally_b])
    
    z_tally = beta_tally * scores_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    p_tally_choice = e_tally / np.sum(e_tally)
    
    # Mixture of strategies
    p_mix = p_ttb_strategy * p_ttb_choice + (1.0 - p_ttb_strategy) * p_tally_choice
    
    # Lapse rate
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [0.1, 50.0]
- threshold: [0.0, 0.5]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7417 (var=0.0032) vs this=0.4331 (var=0.0393)
- Experiment 2: real=0.2881 (var=0.0100) vs this=0.6150 (var=0.0636)
- Experiment 3: real=0.1389 (var=0.0317) vs this=0.3928 (var=0.0708)
- Experiment 4: real=0.0000 (var=0.0233) vs this=-0.0033 (var=0.0375)
- Experiment 5: real=-0.4333 (var=0.0244) vs this=-0.0127 (var=0.0101)
- Experiment 6: real=0.4946 (var=0.0018) vs this=0.6454 (var=0.0206)
- Experiment 7: real=0.3633 (var=0.0090) vs this=0.1314 (var=0.0081)
- Experiment 8: real=0.3389 (var=0.0140) vs this=0.1061 (var=0.0140)
- Experiment 9: real=0.8892 (var=0.0052) vs this=0.5204 (var=0.0764)
- Experiment 10: real=0.4900 (var=0.0032) vs this=0.3733 (var=0.0123)
- Experiment 11: real=0.3556 (var=0.0405) vs this=0.7733 (var=0.2894)
- Experiment 12: real=0.0067 (var=0.0065) vs this=0.3183 (var=0.0863)
- Experiment 13: real=0.1240 (var=0.0162) vs this=0.4900 (var=0.0929)
- Experiment 14: real=0.2000 (var=0.0131) vs this=0.5733 (var=0.0853)
- Experiment 15: real=0.8633 (var=0.0110) vs this=0.5417 (var=0.0962)
- Experiment 16: real=0.6222 (var=0.0019) vs this=0.5486 (var=0.0806)
- Experiment 17: real=0.5092 (var=0.0035) vs this=0.4462 (var=0.0868)
- Experiment 18: real=0.6858 (var=0.0032) vs this=0.5758 (var=0.0776)
- Experiment 19: real=0.3050 (var=0.0041) vs this=0.5100 (var=0.0832)
- Experiment 20: real=0.8700 (var=0.0323) vs this=0.3917 (var=0.6934)
- Experiment 21: real=-0.6578 (var=0.0661) vs this=-0.4600 (var=0.0815)
- Experiment 22: real=0.6108 (var=0.0044) vs this=0.7504 (var=0.0153)
- Experiment 23: real=0.8600 (var=0.0198) vs this=0.6167 (var=0.0947)
- Experiment 24: real=0.5813 (var=0.0017) vs this=0.5470 (var=0.0708)
- Experiment 25: real=0.6217 (var=0.0020) vs this=0.7608 (var=0.0094)
- Experiment 26: real=-0.0111 (var=0.0096) vs this=0.0244 (var=0.0173)
- Experiment 27: real=0.7800 (var=0.0444) vs this=0.4533 (var=0.1148)
- Experiment 28: real=-0.3346 (var=0.0085) vs this=0.2854 (var=0.0425)
- Experiment 29: real=0.0578 (var=0.0307) vs this=-0.0956 (var=0.0761)
- Experiment 30: real=-0.0033 (var=0.0073) vs this=-0.0100 (var=0.0165)
- Experiment 31: real=0.7617 (var=0.0238) vs this=0.0042 (var=0.0187)
- Experiment 32: real=0.3800 (var=0.0041) vs this=0.6683 (var=0.0270)
- Experiment 33: real=0.0000 (var=0.0126) vs this=0.0275 (var=0.0137)
- Experiment 34: real=0.3504 (var=0.0038) vs this=0.4954 (var=0.0827)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4079 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.4079 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ttb_matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        if ttb_winner is not None:
            if resp == ttb_winner:
                ttb_matches += 1
            total += 1
            
    return float(ttb_matches / total) if total > 0 else 0.5

```

**Observed (real) value:** 0.7417 (var=0.0032)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7948 (var=0.0163) (Δ vs real +0.0531)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8604 (var=0.0104)
- pi_2: 0.2646 (var=0.0053)
- pi_3: 0.6323 (var=0.0207)
- pi_4: 0.5490 (var=0.0315)
- pi_5: 0.6167 (var=0.0387)
- pi_6: 0.3781 (var=0.0027)
- pi_7: 0.5410 (var=0.0715)
- pi_8: 0.3569 (var=0.0044)
- pi_9: 0.7185 (var=0.0447)
- pi_10: 0.4400 (var=0.0282)
- pi_11: 0.6181 (var=0.0572)
- pi_12: 0.3979 (var=0.0313)
- pi_13: 0.4612 (var=0.0573)
- pi_14: 0.5127 (var=0.0109)
- pi_15: 0.4331 (var=0.0393)
- pi_16: 0.4573 (var=0.0351)
- pi_17: 0.6119 (var=0.0115)
- pi_18: 0.2865 (var=0.0041)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            pred = 0
        elif b_wins > a_wins:
            pred = 1
        else:
            continue
            
        matches.append(row['response'] == pred)
        
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.2881 (var=0.0100)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2181 (var=0.0144) (Δ vs real -0.0700)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8286 (var=0.0121)
- pi_1: 0.1267 (var=0.0040)
- pi_3: 0.3557 (var=0.0136)
- pi_4: 0.5057 (var=0.0439)
- pi_5: 0.3776 (var=0.0325)
- pi_6: 0.8555 (var=0.0079)
- pi_7: 0.6126 (var=0.0846)
- pi_8: 0.8533 (var=0.0067)
- pi_9: 0.1950 (var=0.0402)
- pi_10: 0.7310 (var=0.0297)
- pi_11: 0.5714 (var=0.1186)
- pi_12: 0.5488 (var=0.0378)
- pi_13: 0.6488 (var=0.0615)
- pi_14: 0.4814 (var=0.0153)
- pi_15: 0.6150 (var=0.0636)
- pi_16: 0.5750 (var=0.1191)
- pi_17: 0.3464 (var=0.0085)
- pi_18: 0.8669 (var=0.0085)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_winner(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1

    extreme_comp = []
    extreme_noncomp = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_win = get_ttb_winner(a, b)
        if ttb_win == -1:
            continue
            
        ttb_chosen = 1 if resp == ttb_win else 0
        
        # Tallying margin for the TTB winner
        margin = sum(a) - sum(b) if ttb_win == 0 else sum(b) - sum(a)
            
        # Extreme compensatory trials (TTB winner is opposed by many cues)
        if margin <= -2:
            extreme_comp.append(ttb_chosen)
        # Non-compensatory trials (TTB winner is supported by overall cue sum)
        elif margin >= 1:
            extreme_noncomp.append(ttb_chosen)
            
    if not extreme_comp or not extreme_noncomp:
        return 0.0
        
    return float(np.mean(extreme_noncomp) - np.mean(extreme_comp))
```

**Observed (real) value:** 0.1389 (var=0.0317)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0467 (var=0.0266) (Δ vs real -0.0922)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0011 (var=0.0123)
- pi_3: 0.1228 (var=0.0284)
- pi_2: 0.7100 (var=0.0421)
- pi_4: 0.3678 (var=0.0551)
- pi_5: 0.0556 (var=0.0297)
- pi_6: 0.6733 (var=0.0393)
- pi_7: 0.5283 (var=0.1250)
- pi_8: 0.7850 (var=0.0249)
- pi_9: 0.1511 (var=0.1069)
- pi_10: 0.7094 (var=0.0403)
- pi_11: 0.4756 (var=0.1435)
- pi_12: 0.1983 (var=0.0307)
- pi_13: 0.5522 (var=0.0792)
- pi_14: 0.1328 (var=0.0973)
- pi_15: 0.3928 (var=0.0708)
- pi_16: 0.3928 (var=0.2999)
- pi_17: 0.5433 (var=0.0808)
- pi_18: 0.6939 (var=0.0411)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_p_a(a_list, b_list):
        a_tup = tuple(a_list)
        b_tup = tuple(b_list)
        mask = (data['option_a_ratings'].apply(tuple) == a_tup) & (data['option_b_ratings'].apply(tuple) == b_tup)
        if mask.sum() == 0:
            return 0.5
        return 1.0 - data.loc[mask, 'response'].mean()

    # Trial 1: highest validity cue favors A, lowest validity cue favors B
    p_a_t1 = get_p_a([1, 0, 0, 0, 0], [0, 0, 0, 0, 1])
    
    # Trial 4: 4th validity cue favors A, lowest validity cue favors B
    p_a_t4 = get_p_a([0, 0, 0, 1, 0], [0, 0, 0, 0, 1])
    
    return float(p_a_t1 - p_a_t4)
```

**Observed (real) value:** 0.0000 (var=0.0233)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0983 (var=0.0319) (Δ vs real +0.0983)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2517 (var=0.0479)
- pi_1: 0.0167 (var=0.0161)
- pi_2: 0.0467 (var=0.0362)
- pi_4: -0.0017 (var=0.0332)
- pi_5: 0.0383 (var=0.0584)
- pi_6: -0.0117 (var=0.0161)
- pi_7: 0.0733 (var=0.0246)
- pi_8: 0.0867 (var=0.0339)
- pi_9: 0.0300 (var=0.0172)
- pi_10: 0.0533 (var=0.0272)
- pi_11: 0.0183 (var=0.0212)
- pi_12: -0.0350 (var=0.0289)
- pi_13: 0.0533 (var=0.0394)
- pi_14: 0.1583 (var=0.0462)
- pi_15: -0.0033 (var=0.0375)
- pi_16: 0.0600 (var=0.0228)
- pi_17: 0.0667 (var=0.0256)
- pi_18: 0.1500 (var=0.0433)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_first_diff(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i, (val_a, val_b) in enumerate(zip(a, b)):
            if val_a != val_b:
                return i
        return -1

    # Calculate which cue is the highest valid discriminating cue
    first_diffs = data.apply(get_first_diff, axis=1)
    
    # Probability of choosing Option A
    chose_a = 1 - data['response']
    
    # Mean P(Choose A) when Cue 1 (index 0) is the highest discriminating cue
    mask_cue1 = (first_diffs == 0)
    p_a_cue1 = chose_a[mask_cue1].mean() if mask_cue1.sum() > 0 else 0.0
    
    # Mean P(Choose A) when Cue 3 (index 2) is the highest discriminating cue
    mask_cue3 = (first_diffs == 2)
    p_a_cue3 = chose_a[mask_cue3].mean() if mask_cue3.sum() > 0 else 0.0
    
    # The metric is the difference in P(Choose A) between Cue 1 and Cue 3 trials.
    return float(p_a_cue1 - p_a_cue3)
```

**Observed (real) value:** -0.4333 (var=0.0244)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0293 (var=0.0091) (Δ vs real +0.4627)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0020 (var=0.0116)
- pi_3: 0.2487 (var=0.0254)
- pi_1: 0.0033 (var=0.0102)
- pi_2: -0.0073 (var=0.0093)
- pi_5: 0.1173 (var=0.0308)
- pi_6: 0.0027 (var=0.0082)
- pi_7: -0.0087 (var=0.0129)
- pi_8: -0.0047 (var=0.0071)
- pi_9: -0.0153 (var=0.0159)
- pi_10: -0.0013 (var=0.0073)
- pi_11: -0.0120 (var=0.0067)
- pi_12: 0.1887 (var=0.0299)
- pi_13: -0.0133 (var=0.0102)
- pi_14: -0.0207 (var=0.0253)
- pi_15: -0.0127 (var=0.0101)
- pi_16: -0.1993 (var=0.0126)
- pi_17: -0.1180 (var=0.0243)
- pi_18: -0.0500 (var=0.0150)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify if option A has the highest validity cue (feature 0)
    # Feature 0 has validity 0.95, so Take The Best (TTB) always favors the option with a 1 here.
    a_has_f0 = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    
    # Identify if the subject chose A or B
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    # Check if the chosen option is the one with feature 0 (the TTB winner)
    ttb_chosen = (a_has_f0 & chose_a) | (~a_has_f0 & chose_b)
    
    # Return the overall proportion of choosing the TTB winner
    return float(ttb_chosen.mean())
```

**Observed (real) value:** 0.4946 (var=0.0018)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8500 (var=0.0080) (Δ vs real +0.3554)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7946 (var=0.0205)
- pi_4: 0.6727 (var=0.0153)
- pi_1: 0.8510 (var=0.0104)
- pi_2: 0.5025 (var=0.0029)
- pi_5: 0.6531 (var=0.0320)
- pi_6: 0.8752 (var=0.0073)
- pi_7: 0.7392 (var=0.0211)
- pi_8: 0.7325 (var=0.0134)
- pi_9: 0.8577 (var=0.0088)
- pi_10: 0.7358 (var=0.0359)
- pi_11: 0.8565 (var=0.0088)
- pi_12: 0.5600 (var=0.0187)
- pi_13: 0.6594 (var=0.0236)
- pi_14: 0.6085 (var=0.0142)
- pi_15: 0.6454 (var=0.0206)
- pi_16: 0.6073 (var=0.0105)
- pi_17: 0.8521 (var=0.0071)
- pi_18: 0.6419 (var=0.0058)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['a_tup'] = data['option_a_ratings'].apply(tuple)
    data['b_tup'] = data['option_b_ratings'].apply(tuple)
    
    target_trials = [
        ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
        ((1, 0, 0, 1, 0), (0, 1, 1, 0, 0)),
        ((1, 0, 1, 0, 0), (0, 1, 0, 1, 0)),
        ((1, 0, 0, 0, 1), (0, 1, 0, 1, 0)),
        ((0, 1, 1, 0, 0), (1, 0, 0, 0, 1)),
        ((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))
    ]
    
    mad_sum = 0.0
    count = 0
    for a_tup, b_tup in target_trials:
        mask = (data['a_tup'] == a_tup) & (data['b_tup'] == b_tup)
        if mask.sum() > 0:
            p_a = 1.0 - data[mask]['response'].mean()
            mad_sum += abs(p_a - 0.5)
            count += 1
            
    return float(mad_sum / count) if count > 0 else 0.0
```

**Observed (real) value:** 0.3633 (var=0.0090)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.3336 (var=0.0101) (Δ vs real -0.0297)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1125 (var=0.0105)
- pi_3: 0.2997 (var=0.0134)
- pi_1: 0.3322 (var=0.0101)
- pi_2: 0.0781 (var=0.0009)
- pi_4: 0.1686 (var=0.0089)
- pi_6: 0.3372 (var=0.0077)
- pi_7: 0.1961 (var=0.0144)
- pi_8: 0.2197 (var=0.0093)
- pi_9: 0.3461 (var=0.0072)
- pi_10: 0.2503 (var=0.0157)
- pi_11: 0.3053 (var=0.0081)
- pi_12: 0.1183 (var=0.0114)
- pi_13: 0.1453 (var=0.0159)
- pi_14: 0.0961 (var=0.0088)
- pi_15: 0.1314 (var=0.0081)
- pi_16: 0.1622 (var=0.0062)
- pi_17: 0.3428 (var=0.0063)
- pi_18: 0.1700 (var=0.0028)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Convert ratings to hashable tuples
    t_keys = data['option_a_ratings'].apply(tuple)
    
    # Focus on the highly conflicting trials where SCA is likely to fail to reach its evidence threshold
    t6 = (1, 0, 1, 1, 0, 0)
    t7 = (1, 0, 1, 0, 0, 0)
    t8 = (1, 0, 0, 0, 0, 0)
    
    mask = t_keys.isin([t6, t7, t8])
    if not mask.any():
        return 0.0
        
    # Calculate the mean response for each of these trial types
    subset = data[mask]
    t_keys_subset = t_keys[mask]
    means = subset.groupby(t_keys_subset)['response'].mean()
    
    # Calculate 'determinism': how far the mean response is from 0.5 (random guessing)
    determinism = (means - 0.5).abs().mean()
    
    return float(determinism)
```

**Observed (real) value:** 0.3389 (var=0.0140)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.3456 (var=0.0104) (Δ vs real +0.0067)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2617 (var=0.0149)
- pi_5: 0.0994 (var=0.0104)
- pi_1: 0.3183 (var=0.0157)
- pi_2: 0.2611 (var=0.0045)
- pi_4: 0.0772 (var=0.0106)
- pi_6: 0.3189 (var=0.0133)
- pi_7: 0.2783 (var=0.0115)
- pi_8: 0.3550 (var=0.0092)
- pi_9: 0.1633 (var=0.0082)
- pi_10: 0.3056 (var=0.0120)
- pi_11: 0.2450 (var=0.0075)
- pi_12: 0.1472 (var=0.0143)
- pi_13: 0.2239 (var=0.0133)
- pi_14: 0.0578 (var=0.0108)
- pi_15: 0.1061 (var=0.0140)
- pi_16: 0.1267 (var=0.0151)
- pi_17: 0.2900 (var=0.0124)
- pi_18: 0.3422 (var=0.0094)

### Experiment 9
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Calculate tallies (sum of positive cues) for each option
    a_tallies = data['option_a_ratings'].apply(lambda x: sum(x))
    b_tallies = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Filter for trials where the tallies are unequal (trials 1, 2, 6, 7)
    unequal_mask = a_tallies != b_tallies
    if not unequal_mask.any():
        return 0.5
        
    filtered_data = data[unequal_mask]
    a_tally_filtered = a_tallies[unequal_mask]
    b_tally_filtered = b_tallies[unequal_mask]
    
    # Determine which option has the higher tally (0 for A, 1 for B)
    higher_tally_option = (b_tally_filtered > a_tally_filtered).astype(int)
    
    # Calculate the proportion of choices that match the higher tally option
    match = (filtered_data['response'] == higher_tally_option).mean()
    
    return float(match)
```

**Observed (real) value:** 0.8892 (var=0.0052)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1812 (var=0.0135) (Δ vs real -0.7079)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8267 (var=0.0160)
- pi_3: 0.2567 (var=0.0343)
- pi_1: 0.1279 (var=0.0123)
- pi_2: 0.8275 (var=0.0091)
- pi_4: 0.4721 (var=0.0460)
- pi_5: 0.3767 (var=0.0199)
- pi_7: 0.5192 (var=0.1256)
- pi_8: 0.8367 (var=0.0127)
- pi_9: 0.2375 (var=0.0537)
- pi_10: 0.5567 (var=0.1285)
- pi_11: 0.3029 (var=0.0502)
- pi_12: 0.4483 (var=0.0277)
- pi_13: 0.4796 (var=0.0878)
- pi_14: 0.4408 (var=0.0114)
- pi_15: 0.5204 (var=0.0764)
- pi_16: 0.6842 (var=0.0493)
- pi_17: 0.3754 (var=0.0246)
- pi_18: 0.8604 (var=0.0077)

### Experiment 10
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    # Trial 4: A=[1, 0, 0, 0, 1], B=[0, 1, 1, 0, 0]
    mask = (
        ((a_str == '00111') & (b_str == '11000')) |
        ((a_str == '10001') & (b_str == '01100'))
    )
    
    if mask.sum() == 0:
        return 0.0
    
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.4900 (var=0.0032)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5058 (var=0.0056) (Δ vs real +0.0158)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5283 (var=0.0126)
- pi_6: 0.1433 (var=0.0117)
- pi_1: 0.5000 (var=0.0049)
- pi_2: 0.3292 (var=0.0136)
- pi_4: 0.4167 (var=0.0134)
- pi_5: 0.5317 (var=0.0164)
- pi_7: 0.4008 (var=0.0222)
- pi_8: 0.3092 (var=0.0260)
- pi_9: 0.5208 (var=0.0063)
- pi_10: 0.3233 (var=0.0339)
- pi_11: 0.6250 (var=0.0383)
- pi_12: 0.3600 (var=0.0301)
- pi_13: 0.4350 (var=0.0206)
- pi_14: 0.5500 (var=0.0172)
- pi_15: 0.3733 (var=0.0123)
- pi_16: 0.5425 (var=0.0144)
- pi_17: 0.5342 (var=0.0061)
- pi_18: 0.4083 (var=0.0119)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_key(row):
        a = tuple(int(x) for x in row['option_a_ratings'])
        b = tuple(int(x) for x in row['option_b_ratings'])
        return str(a) + str(b)
    
    data['trial_str'] = data.apply(get_key, axis=1)
    
    t5 = str((1, 0, 1, 0, 1, 0)) + str((0, 1, 0, 0, 0, 0))
    t6 = str((0, 0, 1, 1, 1, 0)) + str((0, 0, 0, 0, 0, 1))
    
    t7 = str((1, 0, 1, 1, 1, 0)) + str((0, 1, 0, 0, 0, 0))
    t8 = str((0, 0, 1, 1, 1, 1)) + str((1, 0, 0, 0, 0, 0))
    
    t9 = str((1, 0, 1, 1, 1, 1)) + str((0, 1, 0, 0, 0, 0))
    t10 = str((0, 1, 1, 1, 1, 1)) + str((1, 0, 0, 0, 0, 0))
    
    data['chose_A'] = (data['response'] == 0).astype(float)
    subj_trial = data.groupby(['subject_id', 'trial_str'])['chose_A'].mean().unstack()
    
    diff2 = (subj_trial[t5] - subj_trial[t6]).abs().mean() if t5 in subj_trial and t6 in subj_trial else 0
    diff3 = (subj_trial[t7] - subj_trial[t8]).abs().mean() if t7 in subj_trial and t8 in subj_trial else 0
    diff4 = (subj_trial[t9] - subj_trial[t10]).abs().mean() if t9 in subj_trial and t10 in subj_trial else 0
    
    return float(diff2 + diff3 + diff4)
```

**Observed (real) value:** 0.3556 (var=0.0405)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 1.3933 (var=0.2292) (Δ vs real +1.0378)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6044 (var=0.2751)
- pi_3: 1.3267 (var=0.4358)
- pi_1: 1.5378 (var=0.1681)
- pi_2: 0.3333 (var=0.0385)
- pi_4: 0.9622 (var=0.2112)
- pi_5: 0.7778 (var=0.2430)
- pi_6: 0.3244 (var=0.0488)
- pi_8: 0.2867 (var=0.0302)
- pi_9: 1.1333 (var=0.4168)
- pi_10: 0.3222 (var=0.0288)
- pi_11: 0.5822 (var=0.1682)
- pi_12: 0.7622 (var=0.1240)
- pi_13: 0.4111 (var=0.0594)
- pi_14: 0.5556 (var=0.0879)
- pi_15: 0.7733 (var=0.2894)
- pi_16: 0.3667 (var=0.0426)
- pi_17: 0.6822 (var=0.3590)
- pi_18: 0.3133 (var=0.0418)

### Experiment 12
**Design**
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract whether Option B has the most valid feature (index 0)
    b_has_best_cue = data['option_b_ratings'].apply(lambda x: x[0])
    
    # response == 0 means Option A was chosen
    chose_a = 1.0 - data['response']
    
    # Calculate the choice probability for A when B does NOT have the best cue
    p_a_when_b_lacks_best = chose_a[b_has_best_cue == 0].mean()
    
    # Calculate the choice probability for A when B HAS the best cue
    p_a_when_b_has_best = chose_a[b_has_best_cue == 1].mean()
    
    # The difference reflects sensitivity to cue validities
    return float(p_a_when_b_lacks_best - p_a_when_b_has_best)
```

**Observed (real) value:** 0.0067 (var=0.0065)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6637 (var=0.0447) (Δ vs real +0.6571)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5708 (var=0.0712)
- pi_7: 0.1808 (var=0.0610)
- pi_1: 0.7050 (var=0.0455)
- pi_2: -0.0017 (var=0.0045)
- pi_4: 0.3650 (var=0.0600)
- pi_5: 0.3063 (var=0.0569)
- pi_6: 0.0000 (var=0.0044)
- pi_8: -0.0108 (var=0.0041)
- pi_9: 0.5904 (var=0.0799)
- pi_10: 0.0654 (var=0.0118)
- pi_11: 0.3796 (var=0.0861)
- pi_12: 0.0821 (var=0.1400)
- pi_13: 0.1458 (var=0.0429)
- pi_14: 0.1500 (var=0.0465)
- pi_15: 0.3183 (var=0.0863)
- pi_16: 0.1050 (var=0.0357)
- pi_17: 0.4737 (var=0.1033)
- pi_18: 0.0154 (var=0.0058)

### Experiment 13
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify Trial 1 where Option A has lower tally (2 vs 3) but higher validity under both TTB and WADD
    is_t1 = (data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1))
    if not is_t1.any():
        return 0.0
    # response == 0 means Option A was chosen
    return float((data.loc[is_t1, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1240 (var=0.0162)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8660 (var=0.0138) (Δ vs real +0.7420)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7060 (var=0.0702)
- pi_8: 0.1700 (var=0.0249)
- pi_1: 0.8420 (var=0.0188)
- pi_2: 0.1380 (var=0.0136)
- pi_3: 0.8500 (var=0.0257)
- pi_4: 0.5140 (var=0.0616)
- pi_5: 0.7500 (var=0.0557)
- pi_6: 0.1760 (var=0.0282)
- pi_9: 0.8420 (var=0.0304)
- pi_10: 0.2760 (var=0.1138)
- pi_11: 0.8700 (var=0.0185)
- pi_12: 0.5320 (var=0.0726)
- pi_13: 0.5560 (var=0.0841)
- pi_14: 0.6540 (var=0.0501)
- pi_15: 0.4900 (var=0.0929)
- pi_16: 0.3700 (var=0.0849)
- pi_17: 0.8680 (var=0.0194)
- pi_18: 0.1440 (var=0.0273)

### Experiment 14
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    # Focus purely on trials where the tally difference is exactly 1
    mask = abs(tally_a - tally_b) == 1
    if not mask.any():
        return 0.5
        
    df = data[mask]
    tally_a_filtered = tally_a[mask]
    tally_b_filtered = tally_b[mask]
    
    # Check if the subject chose the option with the higher tally
    chose_a = df['response'] == 0
    higher_tally_a = tally_a_filtered > tally_b_filtered
    
    chose_higher = (chose_a == higher_tally_a)
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.2000 (var=0.0131)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2158 (var=0.0285) (Δ vs real +0.0158)
**Other theories' values on this metric (for reference):**
- pi_8: 0.8333 (var=0.0128)
- pi_7: 0.5958 (var=0.1220)
- pi_1: 0.1433 (var=0.0124)
- pi_2: 0.8575 (var=0.0105)
- pi_3: 0.2775 (var=0.0455)
- pi_4: 0.5050 (var=0.0446)
- pi_5: 0.4208 (var=0.0421)
- pi_6: 0.8417 (var=0.0112)
- pi_9: 0.1792 (var=0.0385)
- pi_10: 0.6217 (var=0.1209)
- pi_11: 0.3892 (var=0.1052)
- pi_12: 0.4800 (var=0.0274)
- pi_13: 0.5875 (var=0.0815)
- pi_14: 0.4808 (var=0.0306)
- pi_15: 0.5733 (var=0.0853)
- pi_16: 0.5883 (var=0.1241)
- pi_17: 0.2350 (var=0.0214)
- pi_18: 0.8525 (var=0.0131)

### Experiment 15
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    mask = (tally_a == 2) & (tally_b == 3)
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8633 (var=0.0110)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1567 (var=0.0118) (Δ vs real -0.7067)
**Other theories' values on this metric (for reference):**
- pi_9: 0.2083 (var=0.0565)
- pi_8: 0.8167 (var=0.0292)
- pi_1: 0.1983 (var=0.0175)
- pi_2: 0.8267 (var=0.0275)
- pi_3: 0.1533 (var=0.0232)
- pi_4: 0.4800 (var=0.0604)
- pi_5: 0.2633 (var=0.0584)
- pi_6: 0.8650 (var=0.0150)
- pi_7: 0.5283 (var=0.1091)
- pi_10: 0.7283 (var=0.1066)
- pi_11: 0.1717 (var=0.0318)
- pi_12: 0.6100 (var=0.0654)
- pi_13: 0.4183 (var=0.0840)
- pi_14: 0.3900 (var=0.0568)
- pi_15: 0.5417 (var=0.0962)
- pi_16: 0.6267 (var=0.0842)
- pi_17: 0.1533 (var=0.0154)
- pi_18: 0.8450 (var=0.0261)

### Experiment 16
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tallies = data['option_a_ratings'].apply(sum)
    b_tallies = data['option_b_ratings'].apply(sum)
    
    unequal_mask = a_tallies != b_tallies
    if not unequal_mask.any():
        return 0.5
        
    unequal_data = data[unequal_mask]
    a_higher = a_tallies[unequal_mask] > b_tallies[unequal_mask]
    b_higher = b_tallies[unequal_mask] > a_tallies[unequal_mask]
    
    chose_a = unequal_data['response'] == 0
    chose_b = unequal_data['response'] == 1
    
    matched = (a_higher & chose_a) | (b_higher & chose_b)
    
    return float(matched.mean())
```

**Observed (real) value:** 0.6222 (var=0.0019)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1900 (var=0.0127) (Δ vs real -0.4322)
**Other theories' values on this metric (for reference):**
- pi_8: 0.8508 (var=0.0074)
- pi_9: 0.2075 (var=0.0600)
- pi_1: 0.1622 (var=0.0127)
- pi_2: 0.8594 (var=0.0108)
- pi_3: 0.2272 (var=0.0425)
- pi_4: 0.4867 (var=0.0403)
- pi_5: 0.3225 (var=0.0343)
- pi_6: 0.8692 (var=0.0095)
- pi_7: 0.5228 (var=0.0763)
- pi_10: 0.7411 (var=0.0529)
- pi_11: 0.3108 (var=0.0618)
- pi_12: 0.5228 (var=0.0660)
- pi_13: 0.5103 (var=0.0662)
- pi_14: 0.4383 (var=0.0344)
- pi_15: 0.5486 (var=0.0806)
- pi_16: 0.6447 (var=0.0651)
- pi_17: 0.1300 (var=0.0098)
- pi_18: 0.8539 (var=0.0082)

### Experiment 17
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    mask = abs(tally_a - tally_b) == 1
    if not mask.any():
        return 0.0
        
    lower_tally_is_b = (tally_b[mask] < tally_a[mask]).astype(int)
    responses = data.loc[mask, 'response']
    
    chose_lower_tally = (responses == lower_tally_is_b).mean()
    return float(chose_lower_tally)
```

**Observed (real) value:** 0.5092 (var=0.0035)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7529 (var=0.0347) (Δ vs real +0.2438)
**Other theories' values on this metric (for reference):**
- pi_10: 0.3525 (var=0.1043)
- pi_8: 0.1492 (var=0.0113)
- pi_1: 0.8658 (var=0.0089)
- pi_2: 0.1554 (var=0.0070)
- pi_3: 0.7300 (var=0.0374)
- pi_4: 0.4763 (var=0.0454)
- pi_5: 0.5896 (var=0.0231)
- pi_6: 0.1313 (var=0.0121)
- pi_7: 0.3721 (var=0.1315)
- pi_9: 0.6696 (var=0.1093)
- pi_11: 0.4921 (var=0.1422)
- pi_12: 0.5154 (var=0.0100)
- pi_13: 0.5408 (var=0.1015)
- pi_14: 0.4750 (var=0.0259)
- pi_15: 0.4462 (var=0.0868)
- pi_16: 0.3846 (var=0.0860)
- pi_17: 0.6629 (var=0.0231)
- pi_18: 0.1475 (var=0.0123)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    diff_1_mask = abs(tally_a - tally_b) == 1
    subset = data[diff_1_mask]
    
    if len(subset) == 0:
        return 0.5
        
    chose_higher_tally = ((tally_a[diff_1_mask] > tally_b[diff_1_mask]) & (subset['response'] == 0)) | ((tally_b[diff_1_mask] > tally_a[diff_1_mask]) & (subset['response'] == 1))
    
    return float(chose_higher_tally.mean())
```

**Observed (real) value:** 0.6858 (var=0.0032)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1429 (var=0.0065) (Δ vs real -0.5429)
**Other theories' values on this metric (for reference):**
- pi_8: 0.8429 (var=0.0125)
- pi_10: 0.6408 (var=0.1068)
- pi_1: 0.1338 (var=0.0102)
- pi_2: 0.8313 (var=0.0146)
- pi_3: 0.2233 (var=0.0242)
- pi_4: 0.4975 (var=0.0537)
- pi_5: 0.3588 (var=0.0329)
- pi_6: 0.8379 (var=0.0134)
- pi_7: 0.5533 (var=0.1170)
- pi_9: 0.2842 (var=0.0560)
- pi_11: 0.2875 (var=0.0376)
- pi_12: 0.4629 (var=0.0325)
- pi_13: 0.4988 (var=0.0855)
- pi_14: 0.4379 (var=0.0077)
- pi_15: 0.5758 (var=0.0776)
- pi_16: 0.5667 (var=0.0670)
- pi_17: 0.2888 (var=0.0150)
- pi_18: 0.8433 (var=0.0126)

### Experiment 19
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    diff = (tally_a - tally_b).abs()
    
    mask = diff >= 2
    if not mask.any():
        return 0.5
    
    subset = data[mask]
    ta = tally_a[mask]
    tb = tally_b[mask]
    resp = subset['response']
    
    chose_higher_tally = ((ta > tb) & (resp == 0)) | ((tb > ta) & (resp == 1))
    
    return float(chose_higher_tally.mean())
```

**Observed (real) value:** 0.3050 (var=0.0041)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2354 (var=0.0300) (Δ vs real -0.0696)
**Other theories' values on this metric (for reference):**
- pi_10: 0.8896 (var=0.0076)
- pi_11: 0.5950 (var=0.1248)
- pi_1: 0.1250 (var=0.0079)
- pi_2: 0.8571 (var=0.0125)
- pi_3: 0.2042 (var=0.0297)
- pi_4: 0.5154 (var=0.0646)
- pi_5: 0.4062 (var=0.0486)
- pi_6: 0.8504 (var=0.0107)
- pi_7: 0.6958 (var=0.0695)
- pi_8: 0.8679 (var=0.0090)
- pi_9: 0.3392 (var=0.1216)
- pi_12: 0.5217 (var=0.0466)
- pi_13: 0.7317 (var=0.0586)
- pi_14: 0.4775 (var=0.0208)
- pi_15: 0.5100 (var=0.0832)
- pi_16: 0.4083 (var=0.1237)
- pi_17: 0.6229 (var=0.1182)
- pi_18: 0.8912 (var=0.0068)

### Experiment 20
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t2_mask = (a_str == '10000') & (b_str == '01110')
    t3_mask = (a_str == '10000') & (b_str == '01111')
    t7_mask = (a_str == '01110') & (b_str == '10000')
    
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    p_a_t7 = (data.loc[t7_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t2): p_a_t2 = 0.5
    if pd.isna(p_a_t3): p_a_t3 = 0.5
    if pd.isna(p_a_t7): p_a_t7 = 0.5
    
    return float(p_a_t2 + p_a_t3 - p_a_t7)
```

**Observed (real) value:** 0.8700 (var=0.0323)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 1.3483 (var=0.2305) (Δ vs real +0.4783)
**Other theories' values on this metric (for reference):**
- pi_11: 0.2150 (var=1.3200)
- pi_10: -0.7017 (var=0.0611)
- pi_1: 1.5433 (var=0.1301)
- pi_2: -0.6633 (var=0.0755)
- pi_3: 1.1367 (var=0.5388)
- pi_4: 0.6500 (var=0.4164)
- pi_5: 0.9700 (var=0.4585)
- pi_6: -0.5567 (var=0.1310)
- pi_7: -0.0500 (var=0.7383)
- pi_8: -0.6267 (var=0.0881)
- pi_9: 0.8767 (var=1.1206)
- pi_12: 0.3933 (var=0.2056)
- pi_13: 0.0383 (var=0.6006)
- pi_14: 0.3583 (var=0.2623)
- pi_15: 0.3917 (var=0.6934)
- pi_16: 0.6450 (var=1.2247)
- pi_17: -0.0533 (var=0.7608)
- pi_18: -0.5350 (var=0.0742)

### Experiment 21
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    t10_mask = (a_keys == (0, 1, 1, 0, 0)) & (b_keys == (1, 0, 0, 1, 1))
    t5_mask = (a_keys == (1, 0, 0, 0, 0)) & (b_keys == (0, 1, 0, 0, 0))
    
    p_a_t10 = (data.loc[t10_mask, 'response'] == 0).mean()
    p_a_t5 = (data.loc[t5_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t10): p_a_t10 = 0.5
    if pd.isna(p_a_t5): p_a_t5 = 0.5
    
    return float(p_a_t10 - p_a_t5)
```

**Observed (real) value:** -0.6578 (var=0.0661)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.6289 (var=0.0670) (Δ vs real +0.0289)
**Other theories' values on this metric (for reference):**
- pi_12: -0.1178 (var=0.0866)
- pi_11: -0.6667 (var=0.0444)
- pi_1: -0.6889 (var=0.0642)
- pi_2: -0.3733 (var=0.0478)
- pi_3: -0.5667 (var=0.1098)
- pi_4: -0.6022 (var=0.0657)
- pi_5: -0.2600 (var=0.1731)
- pi_6: -0.7089 (var=0.0582)
- pi_7: -0.5467 (var=0.0616)
- pi_8: -0.6156 (var=0.0594)
- pi_9: -0.7356 (var=0.0434)
- pi_10: -0.6044 (var=0.0890)
- pi_13: -0.3511 (var=0.1286)
- pi_14: -0.1444 (var=0.0633)
- pi_15: -0.4600 (var=0.0815)
- pi_16: -0.4867 (var=0.1140)
- pi_17: -0.7156 (var=0.0702)
- pi_18: -0.4489 (var=0.0484)

### Experiment 22
**Design**
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where cue 1 and cue 2 conflict
    conflict = (a_ratings[:, 0] != b_ratings[:, 0]) & \
               (a_ratings[:, 1] != b_ratings[:, 1]) & \
               (a_ratings[:, 0] != a_ratings[:, 1])
               
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    
    a_has_cue1 = a_ratings[:, 0] > b_ratings[:, 0]
    b_has_cue1 = b_ratings[:, 0] > a_ratings[:, 0]
    
    # Check if the chosen option is the one favored by cue 1
    chose_cue1 = (a_has_cue1 & (responses == 0)) | (b_has_cue1 & (responses == 1))
    
    return float(np.mean(chose_cue1[conflict]))
```

**Observed (real) value:** 0.6108 (var=0.0044)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8358 (var=0.0129) (Δ vs real +0.2250)
**Other theories' values on this metric (for reference):**
- pi_11: 0.8588 (var=0.0158)
- pi_12: 0.4271 (var=0.0272)
- pi_1: 0.8417 (var=0.0116)
- pi_2: 0.6229 (var=0.0038)
- pi_3: 0.7496 (var=0.0175)
- pi_4: 0.7471 (var=0.0070)
- pi_5: 0.6238 (var=0.0388)
- pi_6: 0.8438 (var=0.0090)
- pi_7: 0.7667 (var=0.0224)
- pi_8: 0.7000 (var=0.0151)
- pi_9: 0.8588 (var=0.0138)
- pi_10: 0.7937 (var=0.0187)
- pi_13: 0.7354 (var=0.0255)
- pi_14: 0.5554 (var=0.0078)
- pi_15: 0.7504 (var=0.0153)
- pi_16: 0.5713 (var=0.0096)
- pi_17: 0.8742 (var=0.0091)
- pi_18: 0.6367 (var=0.0048)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 5 where Option A has the single best cue and Option B has all other 5 cues.
    t5_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 1, 1, 1))
    
    if t5_mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen in this specific trial
    return float((data.loc[t5_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.8600 (var=0.0198)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8433 (var=0.0274) (Δ vs real -0.0167)
**Other theories' values on this metric (for reference):**
- pi_13: 0.1483 (var=0.0209)
- pi_11: 0.6233 (var=0.1042)
- pi_1: 0.8517 (var=0.0176)
- pi_2: 0.1117 (var=0.0124)
- pi_3: 0.8017 (var=0.0475)
- pi_4: 0.4250 (var=0.0656)
- pi_5: 0.6100 (var=0.0660)
- pi_6: 0.1717 (var=0.0198)
- pi_7: 0.1917 (var=0.0528)
- pi_8: 0.1450 (var=0.0105)
- pi_9: 0.8600 (var=0.0365)
- pi_10: 0.1400 (var=0.0182)
- pi_12: 0.5550 (var=0.0466)
- pi_14: 0.6533 (var=0.0332)
- pi_15: 0.6167 (var=0.0947)
- pi_16: 0.4250 (var=0.1701)
- pi_17: 0.8867 (var=0.0127)
- pi_18: 0.1250 (var=0.0198)

### Experiment 24
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.array(data['option_a_ratings'].tolist())
    b = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    sum_a = a.sum(axis=1)
    sum_b = b.sum(axis=1)
    
    # HVO (High Validity Option) is the one with the best cue (index 0)
    # We filter for trials where the HVO is outnumbered in simple tally
    cond_a = (a[:, 0] == 1) & (b[:, 0] == 0) & (sum_a < sum_b)
    cond_b = (b[:, 0] == 1) & (a[:, 0] == 0) & (sum_b < sum_a)
    
    hvo_chosen = np.sum((cond_a & (resp == 0)) | (cond_b & (resp == 1)))
    total = np.sum(cond_a | cond_b)
    
    return float(hvo_chosen / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5813 (var=0.0017)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8350 (var=0.0184) (Δ vs real +0.2537)
**Other theories' values on this metric (for reference):**
- pi_11: 0.7033 (var=0.0842)
- pi_13: 0.3613 (var=0.0536)
- pi_1: 0.8603 (var=0.0076)
- pi_2: 0.1263 (var=0.0083)
- pi_3: 0.8190 (var=0.0231)
- pi_4: 0.5117 (var=0.0543)
- pi_5: 0.6480 (var=0.0349)
- pi_6: 0.1597 (var=0.0108)
- pi_7: 0.4390 (var=0.1182)
- pi_8: 0.1317 (var=0.0096)
- pi_9: 0.8120 (var=0.0415)
- pi_10: 0.2270 (var=0.0276)
- pi_12: 0.5650 (var=0.0325)
- pi_14: 0.6217 (var=0.0282)
- pi_15: 0.5470 (var=0.0708)
- pi_16: 0.4517 (var=0.1186)
- pi_17: 0.8613 (var=0.0088)
- pi_18: 0.1357 (var=0.0071)

### Experiment 25
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Check where high-validity cues (first two) are tied
    tied_high = (a_ratings[:, 0] == b_ratings[:, 0]) & (a_ratings[:, 1] == b_ratings[:, 1])
    
    # Sum the 0.5-validity cues (last three)
    sum_a_low = a_ratings[:, 2:].sum(axis=1)
    sum_b_low = b_ratings[:, 2:].sum(axis=1)
    
    a_more = sum_a_low > sum_b_low
    b_more = sum_b_low > sum_a_low
    
    # Filter to trials where high-validity cues are tied AND low-validity cues differ
    valid_trials = tied_high & (a_more | b_more)
    if not np.any(valid_trials):
        return 0.5
        
    responses = data['response'].values
    
    # Check if subject chose the option with more 0.5-validity cues
    chose_a_more = a_more & (responses == 0)
    chose_b_more = b_more & (responses == 1)
    
    consistent = chose_a_more | chose_b_more
    
    return float(np.sum(consistent[valid_trials]) / np.sum(valid_trials))
```

**Observed (real) value:** 0.6217 (var=0.0020)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7025 (var=0.0118) (Δ vs real +0.0808)
**Other theories' values on this metric (for reference):**
- pi_14: 0.5197 (var=0.0088)
- pi_11: 0.8853 (var=0.0055)
- pi_1: 0.7317 (var=0.0042)
- pi_2: 0.8486 (var=0.0078)
- pi_3: 0.5469 (var=0.0112)
- pi_4: 0.7761 (var=0.0049)
- pi_5: 0.5433 (var=0.0135)
- pi_6: 0.8619 (var=0.0092)
- pi_7: 0.8056 (var=0.0120)
- pi_8: 0.8542 (var=0.0071)
- pi_9: 0.5761 (var=0.0250)
- pi_10: 0.8006 (var=0.0106)
- pi_12: 0.6578 (var=0.0110)
- pi_13: 0.7828 (var=0.0102)
- pi_15: 0.7608 (var=0.0094)
- pi_16: 0.5864 (var=0.0792)
- pi_17: 0.4939 (var=0.0034)
- pi_18: 0.8486 (var=0.0066)

### Experiment 26
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1 vs 7
    is_t1 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (b_tuples == (0, 0, 0, 0, 0, 0, 0, 0))
    is_t1_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (a_tuples == (0, 0, 0, 0, 0, 0, 0, 0))
    is_t7 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (b_tuples == (1, 1, 1, 1, 1, 1, 0, 0))
    is_t7_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (a_tuples == (1, 1, 1, 1, 1, 1, 0, 0))
    
    # Trial 8 vs 12
    is_t8 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 0)) & (b_tuples == (0, 0, 0, 0, 0, 0, 0, 1))
    is_t8_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 0)) & (a_tuples == (0, 0, 0, 0, 0, 0, 0, 1))
    is_t12 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 0)) & (b_tuples == (1, 1, 1, 1, 0, 0, 0, 1))
    is_t12_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 0)) & (a_tuples == (1, 1, 1, 1, 0, 0, 0, 1))
    
    # Trial 13 vs 15
    is_t13 = (a_tuples == (1, 0, 1, 1, 1, 1, 1, 1)) & (b_tuples == (0, 1, 0, 0, 0, 0, 0, 0))
    is_t13_rev = (b_tuples == (1, 0, 1, 1, 1, 1, 1, 1)) & (a_tuples == (0, 1, 0, 0, 0, 0, 0, 0))
    is_t15 = (a_tuples == (1, 1, 1, 0, 1, 1, 1, 1)) & (b_tuples == (1, 1, 0, 1, 0, 0, 0, 0))
    is_t15_rev = (b_tuples == (1, 1, 1, 0, 1, 1, 1, 1)) & (a_tuples == (1, 1, 0, 1, 0, 0, 0, 0))
    
    untied_chosen = []
    tied_chosen = []
    
    for idx, row in data.iterrows():
        # Untied trials
        if is_t1[idx] or is_t8[idx] or is_t13[idx]:
            untied_chosen.append(1.0 if row['response'] == 0 else 0.0)
        elif is_t1_rev[idx] or is_t8_rev[idx] or is_t13_rev[idx]:
            untied_chosen.append(1.0 if row['response'] == 1 else 0.0)
            
        # Tied trials
        if is_t7[idx] or is_t12[idx] or is_t15[idx]:
            tied_chosen.append(1.0 if row['response'] == 0 else 0.0)
        elif is_t7_rev[idx] or is_t12_rev[idx] or is_t15_rev[idx]:
            tied_chosen.append(1.0 if row['response'] == 1 else 0.0)
            
    p_untied = sum(untied_chosen) / len(untied_chosen) if untied_chosen else 0.5
    p_tied = sum(tied_chosen) / len(tied_chosen) if tied_chosen else 0.5
    
    return float(p_untied - p_tied)
```

**Observed (real) value:** -0.0111 (var=0.0096)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0389 (var=0.0174) (Δ vs real +0.0500)
**Other theories' values on this metric (for reference):**
- pi_11: 0.0056 (var=0.0066)
- pi_14: 0.1278 (var=0.0313)
- pi_1: 0.0211 (var=0.0129)
- pi_2: 0.0122 (var=0.0132)
- pi_3: 0.2778 (var=0.0258)
- pi_4: 0.0067 (var=0.0137)
- pi_5: 0.2489 (var=0.0272)
- pi_6: -0.0011 (var=0.0098)
- pi_7: 0.0144 (var=0.0137)
- pi_8: -0.0167 (var=0.0139)
- pi_9: 0.0344 (var=0.0149)
- pi_10: 0.0133 (var=0.0150)
- pi_12: 0.0356 (var=0.0101)
- pi_13: 0.0378 (var=0.0147)
- pi_15: 0.0244 (var=0.0173)
- pi_16: -0.0178 (var=0.0459)
- pi_17: 0.0189 (var=0.0107)
- pi_18: 0.0100 (var=0.0096)

### Experiment 27
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t4_mask = (a_str == '11000') & (b_str == '00111')
    t5_mask = (a_str == '11100') & (b_str == '00011')
    t6_mask = (a_str == '01000') & (b_str == '00111')
    t7_mask = (a_str == '01000') & (b_str == '00011')
    
    p_a = {}
    for i, mask in enumerate([t4_mask, t5_mask, t6_mask, t7_mask]):
        if mask.sum() > 0:
            p_a[i] = (data.loc[mask, 'response'] == 0).mean()
        else:
            p_a[i] = 0.5
            
    return float((p_a[1] - p_a[0]) + (p_a[3] - p_a[2]))
```

**Observed (real) value:** 0.7800 (var=0.0444)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0100 (var=0.0393) (Δ vs real -0.7700)
**Other theories' values on this metric (for reference):**
- pi_14: 0.0083 (var=0.1217)
- pi_15: 0.4533 (var=0.1148)
- pi_1: -0.0050 (var=0.0309)
- pi_2: 0.7483 (var=0.0971)
- pi_3: 0.1383 (var=0.0877)
- pi_4: 0.3100 (var=0.1225)
- pi_5: 0.0983 (var=0.1235)
- pi_6: 0.7850 (var=0.0520)
- pi_7: 0.5433 (var=0.1395)
- pi_8: 0.7200 (var=0.0783)
- pi_9: 0.1467 (var=0.1435)
- pi_10: 0.7750 (var=0.0470)
- pi_11: 0.2567 (var=0.1858)
- pi_12: -0.0233 (var=0.0686)
- pi_13: 0.4500 (var=0.1400)
- pi_16: 0.4083 (var=0.1940)
- pi_17: 0.5317 (var=0.1236)
- pi_18: 0.7717 (var=0.0649)

### Experiment 28
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    # High Tally pool: Tally differences of 0 and +1
    high_mask = (
        ((data['A_str'] == '10000') & (data['B_str'] == '01000')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00100')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00010')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00001')) | 
        ((data['A_str'] == '11000') & (data['B_str'] == '00001')) | 
        ((data['A_str'] == '10100') & (data['B_str'] == '00001')) | 
        ((data['A_str'] == '10010') & (data['B_str'] == '00001'))   
    )
    
    # Low Tally pool: Tally differences of -1 and -2
    low_mask = (
        ((data['A_str'] == '10000') & (data['B_str'] == '01100')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00110')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00011')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '01110')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00111'))   
    )
    
    high_mean = data[high_mask]['chose_A'].mean()
    low_mean = data[low_mask]['chose_A'].mean()
    
    if pd.isna(high_mean): high_mean = 0.5
    if pd.isna(low_mean): low_mean = 0.5
        
    return float(high_mean - low_mean)
```

**Observed (real) value:** -0.3346 (var=0.0085)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0596 (var=0.0113) (Δ vs real +0.3941)
**Other theories' values on this metric (for reference):**
- pi_15: 0.2854 (var=0.0425)
- pi_14: 0.0623 (var=0.0468)
- pi_1: 0.0240 (var=0.0037)
- pi_2: 0.5313 (var=0.0180)
- pi_3: 0.0785 (var=0.0250)
- pi_4: 0.2302 (var=0.0323)
- pi_5: 0.0428 (var=0.0188)
- pi_6: 0.6804 (var=0.0368)
- pi_7: 0.3967 (var=0.1133)
- pi_8: 0.7000 (var=0.0261)
- pi_9: 0.1286 (var=0.0630)
- pi_10: 0.5337 (var=0.0387)
- pi_11: 0.2900 (var=0.0781)
- pi_12: -0.0066 (var=0.0103)
- pi_13: 0.2933 (var=0.0657)
- pi_16: 0.3668 (var=0.2179)
- pi_17: 0.1042 (var=0.0176)
- pi_18: 0.7261 (var=0.0210)

### Experiment 29
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    t3_mask = (data['A_tuple'] == (1, 0, 0, 1, 0)) & (data['B_tuple'] == (0, 1, 1, 0, 0))
    t4_mask = (data['A_tuple'] == (1, 0, 0, 0, 1)) & (data['B_tuple'] == (0, 1, 1, 0, 0))
    t5_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 0, 0))
    t7_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 1))
    
    p_a_t3 = data.loc[t3_mask, 'chose_A'].mean() if t3_mask.sum() > 0 else 0.0
    p_a_t4 = data.loc[t4_mask, 'chose_A'].mean() if t4_mask.sum() > 0 else 0.0
    p_a_t5 = data.loc[t5_mask, 'chose_A'].mean() if t5_mask.sum() > 0 else 0.0
    p_a_t7 = data.loc[t7_mask, 'chose_A'].mean() if t7_mask.sum() > 0 else 0.0
    
    return (p_a_t3 - p_a_t4) + (p_a_t7 - p_a_t5)
```

**Observed (real) value:** 0.0578 (var=0.0307)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.0178 (var=0.0580) (Δ vs real -0.0756)
**Other theories' values on this metric (for reference):**
- pi_16: 0.2356 (var=0.0941)
- pi_15: -0.0956 (var=0.0761)
- pi_1: 0.0156 (var=0.0454)
- pi_2: -0.0311 (var=0.0529)
- pi_3: 0.1000 (var=0.1196)
- pi_4: -0.0333 (var=0.1038)
- pi_5: 0.0044 (var=0.0612)
- pi_6: -0.0111 (var=0.0219)
- pi_7: 0.0378 (var=0.0892)
- pi_8: 0.1333 (var=0.0598)
- pi_9: 0.1978 (var=0.3058)
- pi_10: 0.0444 (var=0.0662)
- pi_11: 0.5422 (var=0.2660)
- pi_12: -0.3156 (var=0.1651)
- pi_13: 0.0756 (var=0.0980)
- pi_14: 0.0844 (var=0.1682)
- pi_17: 0.1111 (var=0.0741)
- pi_18: 0.1911 (var=0.0815)

### Experiment 30
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_has_cue_1 = data['option_a_ratings'].apply(lambda x: x[1] == 1)
    a_has_cue_4 = data['option_a_ratings'].apply(lambda x: x[4] == 1)
    
    p_a_given_cue_1 = (data[a_has_cue_1]['response'] == 0).mean()
    p_a_given_cue_4 = (data[a_has_cue_4]['response'] == 0).mean()
    
    return float(p_a_given_cue_1 - p_a_given_cue_4)
```

**Observed (real) value:** -0.0033 (var=0.0073)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0225 (var=0.0206) (Δ vs real +0.0258)
**Other theories' values on this metric (for reference):**
- pi_15: -0.0100 (var=0.0165)
- pi_16: 0.1850 (var=0.0634)
- pi_1: 0.0025 (var=0.0089)
- pi_2: -0.0217 (var=0.0140)
- pi_3: 0.0108 (var=0.0301)
- pi_4: -0.0333 (var=0.0169)
- pi_5: 0.0633 (var=0.0416)
- pi_6: 0.0042 (var=0.0109)
- pi_7: 0.1067 (var=0.0396)
- pi_8: 0.1175 (var=0.0223)
- pi_9: 0.1983 (var=0.1102)
- pi_10: 0.0050 (var=0.0107)
- pi_11: 0.3333 (var=0.0875)
- pi_12: -0.2208 (var=0.0461)
- pi_13: 0.0450 (var=0.0177)
- pi_14: 0.1017 (var=0.0533)
- pi_17: 0.2767 (var=0.0417)
- pi_18: 0.2367 (var=0.0227)

### Experiment 31
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    scores = []
    
    for subj, subj_df in data.groupby('subject_id'):
        high_choices = []
        mid_choices = []
        
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            
            sum_a = np.sum(a)
            sum_b = np.sum(b)
            
            # Only look at trials where both options have the same total number of cues.
            # This perfectly neutralizes the variance caused by LCA's configural gamma parameter.
            if sum_a == sum_b:
                val_a = np.sum(a * val)
                val_b = np.sum(b * val)
                
                chose_a = 1.0 if row['response'] == 0 else 0.0
                
                # Trials 1-4 and 9-12: The linear sum of validities strongly favors A.
                if val_a > val_b + 1e-4:
                    high_choices.append(chose_a)
                # Trials 13-14: The linear sum of validities is EXACTLY equal for A and B.
                elif abs(val_a - val_b) <= 1e-4:
                    mid_choices.append(chose_a)
                    
        if len(high_choices) > 0 and len(mid_choices) > 0:
            scores.append(np.mean(high_choices) - np.mean(mid_choices))
            
    if not scores:
        return 0.0
        
    return float(np.mean(scores))
```

**Observed (real) value:** 0.7617 (var=0.0238)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0183 (var=0.0145) (Δ vs real -0.7433)
**Other theories' values on this metric (for reference):**
- pi_16: 0.2429 (var=0.0438)
- pi_17: -0.0042 (var=0.0123)
- pi_1: -0.0329 (var=0.0101)
- pi_2: -0.0083 (var=0.0214)
- pi_3: -0.0283 (var=0.0201)
- pi_4: 0.0167 (var=0.0216)
- pi_5: -0.0312 (var=0.0183)
- pi_6: -0.0300 (var=0.0112)
- pi_7: 0.0787 (var=0.0241)
- pi_8: 0.0679 (var=0.0316)
- pi_9: 0.0221 (var=0.0200)
- pi_10: 0.0079 (var=0.0159)
- pi_11: 0.0217 (var=0.0121)
- pi_12: -0.0396 (var=0.0183)
- pi_13: 0.0092 (var=0.0124)
- pi_14: 0.0300 (var=0.0363)
- pi_15: 0.0042 (var=0.0187)
- pi_18: 0.1904 (var=0.0389)

### Experiment 32
**Design**
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_target_trial = data['option_a_ratings'].apply(lambda x: x[0] == 1) & data['option_b_ratings'].apply(lambda x: x[1] == 1 and x[2] == 1)
    target_data = data[is_target_trial]
    if len(target_data) == 0:
        return 0.5
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.3800 (var=0.0041)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8383 (var=0.0133) (Δ vs real +0.4583)
**Other theories' values on this metric (for reference):**
- pi_17: 0.8321 (var=0.0076)
- pi_16: 0.3042 (var=0.0229)
- pi_1: 0.8579 (var=0.0143)
- pi_2: 0.4804 (var=0.0040)
- pi_3: 0.7887 (var=0.0256)
- pi_4: 0.6467 (var=0.0176)
- pi_5: 0.5517 (var=0.0217)
- pi_6: 0.8575 (var=0.0092)
- pi_7: 0.6308 (var=0.0604)
- pi_8: 0.5900 (var=0.0684)
- pi_9: 0.7913 (var=0.0504)
- pi_10: 0.7562 (var=0.0399)
- pi_11: 0.7904 (var=0.0332)
- pi_12: 0.6579 (var=0.0148)
- pi_13: 0.6617 (var=0.0417)
- pi_14: 0.5442 (var=0.0160)
- pi_15: 0.6683 (var=0.0270)
- pi_18: 0.4104 (var=0.0082)

### Experiment 33
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Focus on trials where the tallies are perfectly matched
    mask = sum_a == sum_b
    
    # Compare choice rates between trials with high background tally vs low background tally
    high_mask = mask & (sum_a >= 4)
    low_mask = mask & (sum_a <= 2)
    
    # P(Choose A) = 1.0 - mean(response)
    p_a_high = 1.0 - data.loc[high_mask, 'response'].mean()
    p_a_low = 1.0 - data.loc[low_mask, 'response'].mean()
    
    if pd.isna(p_a_high) or pd.isna(p_a_low):
        return 0.0
        
    return float(p_a_high - p_a_low)
```

**Observed (real) value:** 0.0000 (var=0.0126)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0150 (var=0.0110) (Δ vs real +0.0150)
**Other theories' values on this metric (for reference):**
- pi_16: -0.0900 (var=0.0201)
- pi_18: 0.0300 (var=0.0181)
- pi_1: -0.0088 (var=0.0100)
- pi_2: 0.0056 (var=0.0166)
- pi_3: 0.1262 (var=0.0215)
- pi_4: -0.0106 (var=0.0228)
- pi_5: 0.0537 (var=0.0149)
- pi_6: 0.0206 (var=0.0099)
- pi_7: -0.0088 (var=0.0148)
- pi_8: 0.0081 (var=0.0140)
- pi_9: 0.0081 (var=0.0160)
- pi_10: 0.0106 (var=0.0178)
- pi_11: -0.0050 (var=0.0171)
- pi_12: -0.0194 (var=0.0185)
- pi_13: -0.0144 (var=0.0193)
- pi_14: 0.0575 (var=0.0320)
- pi_15: 0.0275 (var=0.0137)
- pi_17: 0.0162 (var=0.0142)

### Experiment 34
**Design**
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the tally (number of positive cues) for options A and B
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    # Determine if the subject's response matched the option with the higher tally
    # response == 0 means A was chosen, response == 1 means B was chosen
    chose_higher_tally = ((tally_a > tally_b) & (data['response'] == 0)) | \
                         ((tally_b > tally_a) & (data['response'] == 1))
                         
    # Return the overall proportion of trials where the higher tally option was chosen
    return float(chose_higher_tally.mean())
```

**Observed (real) value:** 0.3504 (var=0.0038)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2042 (var=0.0136) (Δ vs real -0.1462)
**Other theories' values on this metric (for reference):**
- pi_18: 0.8515 (var=0.0072)
- pi_16: 0.6140 (var=0.0735)
- pi_1: 0.1650 (var=0.0139)
- pi_2: 0.8523 (var=0.0137)
- pi_3: 0.2946 (var=0.0302)
- pi_4: 0.4731 (var=0.0411)
- pi_5: 0.4348 (var=0.0177)
- pi_6: 0.8535 (var=0.0072)
- pi_7: 0.5721 (var=0.1113)
- pi_8: 0.8325 (var=0.0144)
- pi_9: 0.1683 (var=0.0130)
- pi_10: 0.7027 (var=0.0879)
- pi_11: 0.3065 (var=0.0674)
- pi_12: 0.4973 (var=0.0091)
- pi_13: 0.5140 (var=0.0664)
- pi_14: 0.4494 (var=0.0174)
- pi_15: 0.4954 (var=0.0827)
- pi_17: 0.1319 (var=0.0093)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Rank-Weighted Sequential Sampling: Decision-makers evaluate cues sequentially in descending order of validity, accumulating evidence at each step. After evaluating each cue, there is a probability of halting the evaluation process that increases with the absolute magnitude of accumulated evidence. If evaluation stops, a choice is made based on the accumulated evidence. This allows the model to dynamically transition between non-compensatory (early stopping) and compensatory (full integration) behavior depending on the sequence of cue differences, capturing both Take-The-Best and Tallying-like patterns seamlessly.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    lambda_stop = float(parameters["lambda_stop"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending (stable sort to preserve original order on ties)
    order = np.argsort(-val, kind='stable')
    
    E = 0.0
    p_continue = 1.0
    p_choose_A = 0.0
    
    for i, idx in enumerate(order):
        # Accumulate evidence weighted by validity
        w = val[idx] ** gamma
        E += w * (a[idx] - b[idx])
        
        # Stopping probability depends on absolute accumulated evidence
        if i < len(order) - 1:
            S = 1.0 - np.exp(-lambda_stop * abs(E))
        else:
            S = 1.0  # Must stop at the last cue
            
        p_stop_here = p_continue * S
        p_continue *= (1.0 - S)
        
        # Choice probability if stopped at this step
        z = beta * E
        if z > 20.0:
            p_A_given_stop = 1.0
        elif z < -20.0:
            p_A_given_stop = 0.0
        else:
            p_A_given_stop = 1.0 / (1.0 + np.exp(-z))
            
        p_choose_A += p_stop_here * p_A_given_stop
        
    # Apply lapse rate
    p_A_final = (1.0 - epsilon) * p_choose_A + epsilon * 0.5
    
    return np.array([p_A_final, 1.0 - p_A_final])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- lambda_stop: [0.0, 10.0]
- beta: [0.1, 25.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's suggestion, this theory replaces the rigid strategy mixture with a Rank-Weighted Sequential Sampling model. Cues are processed in order of validity, and evidence is accumulated. Crucially, the probability of stopping the decision process early depends on the absolute accumulated evidence. If early cues strongly favor one option, the model is likely to stop and choose it, mimicking Take-The-Best (which explains the low tally reliance in Experiments 2, 16, and 21). If early cues are tied or provide weak evidence, the model continues integrating later cues, mimicking WADD or Tallying. This single dynamic mechanism elegantly scales between non-compensatory and compensatory behavior based on the specific cue values presented on each trial.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The Rank-Weighted Sequential Sampling model successfully captures the low reliance on tallies in experiments like 2, 14, and 19. However, it severely over-predicts Take-The-Best (early stopping) behavior in experiments where subjects strongly rely on overall tallies, such as Experiments 9, 15, 18, and 34. In Exp 9, for example, the observed match to the higher tally is 0.8892, but the candidate only predicts 0.1812, indicating it is stopping too early on the first few highly valid cues and failing to integrate the remaining cues.
Rationale: The candidate was ACCEPTED by the gate, meaning the Rank-Weighted Sequential Sampling direction is promising, but it currently leans too heavily into non-compensatory early stopping. To fix the severe under-prediction of tallying in Exps 9, 15, and 18 while staying within the prescribed mechanism family, adjust the stopping probability mechanism. Specifically, reduce the upper bound of `lambda_stop` (e.g., from 10.0 to 2.0 or 3.0) so the model is less eager to halt after just one or two cues. You might also consider letting the stopping probability scale with the ratio of accumulated evidence to the maximum possible remaining evidence, or adding a baseline continuation parameter, to ensure compensatory integration can still occur when early cue evidence is moderate.

**Outcome of this advice:** iter 1 candidate loss=0.4079 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
```json
{
  "theory": "Rank-Weighted Sequential Sampling with Forward-Looking Stopping: Decision-makers evaluate cues sequentially in descending order of validity, accumulating evidence at each step. To decide whether to stop or continue, they compare the currently accumulated evidence against the maximum possible evidence that could be provided by the remaining cues. The probability of halting increases as the current evidence outweighs the potential remaining evidence. This forward-looking dynamic prevents premature stopping when remaining cues could overturn the decision, seamlessly bridging Take-The-Best and Tallying-like compensatory behavior.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    lambda_stop = float(parameters[\"lambda_stop\"])\n    beta = float(parameters[\"beta\"])\n    gamma = float(parameters[\"gamma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Sort cues by validity descending (stable sort to preserve original order on ties)\n    order = np.argsort(-val, kind='stable')\n    \n    E = 0.0\n    p_continue = 1.0\n    p_choose_A = 0.0\n    \n    for i, idx in enumerate(order):\n        # Accumulate evidence weighted by validity\n        w = val[idx] ** gamma\n        E += w * (a[idx] - b[idx])\n        \n        # Stopping probability depends on absolute accumulated evidence relative to remaining possible evidence\n        if i < len(order) - 1:\n            max_remaining = sum((val[j] ** gamma) for j in order[i+1:])\n            S = 1.0 - np.exp(-lambda_stop * (abs(E) / (max_remaining + 1e-3)))\n        else:\n            S = 1.0  # Must stop at the last cue\n            \n        p_stop_here = p_continue * S\n        p_continue *= (1.0 - S)\n        \n        # Choice probability if stopped at this step\n        z = beta * E\n        if z > 20.0:\n            p_A_given_stop = 1.0\n        elif z < -20.0:\n            p_A_given_stop = 0.0\n        else:\n            p_A_given_stop = 1.0 / (1.0 + np.exp(-z))\n            \n        p_choose_A += p_stop_here * p_A_given_stop\n        \n    # Apply lapse rate\n    p_A_final = (1.0 - epsilon) * p_choose_A + epsilon * 0.5\n    \n    return np.array([p_A_final, 1.0 - p_A_final])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "lambda_stop": "[0.0, 3.0]",
    "beta": "[0.1, 25.0]",
    "gamma": "[0.0, 5.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "To address the severe under-prediction of tallying behavior in Exps 9, 15, 18, and 34, I modified the stopping probability mechanism. The previous version stopped too eagerly based solely on the absolute magnitude of accumulated evidence. I scaled the accumulated evidence by the maximum possible remaining evidence (`max_remaining`) and reduced the upper bound of `lambda_stop` from 10.0 to 3.0. This 'forward-looking' modification ensures that the model is much less likely to halt after just one or two cues if the remaining cues hold enough weight to potentially overturn the current lead, allowing it to better capture the compensatory integration seen in those specific experiments."
}
```
```

## Usage

```json
{
  "prompt_token_count": 51704,
  "candidates_token_count": 1006,
  "total_token_count": 54472
}
```
