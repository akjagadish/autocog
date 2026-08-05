# generation_iter_00_attempt_00

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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Propose a 'Two-Stage Contingent Processing Theory'. In the first stage, decision-makers perform a rapid, parallel evaluation using a simple heuristic like Tallying (counting winning cues for each option). If the difference between the options in this first stage exceeds a certain threshold, a decision is made immediately based on this simple difference. If the options are too similar (below threshold), the decision-maker transitions to a second, more effortful stage, such as evaluating the options using weighted cues (WADD) or a strict lexicographic rule (Take-The-Best). This naturally limits the exaggerated alignment effects seen in Strategy Mixture Theory (since aligned trials might just be decided in Stage 1 with some noise, rather than summing up multiple confident heuristics) while capturing the context-dependent reliance on Tallying vs. weighted evidence across the other experiments.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.764)

**Description**
Strategy Mixture Theory with Independent Scaling: Individuals use a probabilistic mixture of distinct heuristics (WADD, Tallying, and Take-The-Best), but because the internal evidence scales of these heuristics vary dramatically (log-odds sums vs. integer counts vs. binary indicators), each heuristic applies its own independent temperature parameter to properly calibrate its choice probabilities before mixing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.dot(stim, w)
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
    if winner is None:
        scores_ttb = np.array([0.0, 0.0])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    beta_wadd = float(parameters["beta_wadd"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    
    def get_probs(scores, beta):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        return e / np.sum(e)
        
    p_wadd = get_probs(scores_wadd, beta_wadd)
    p_tally = get_probs(scores_tally, beta_tally)
    p_ttb = get_probs(scores_ttb, beta_ttb)
    
    w1 = float(parameters["w_wadd"])
    w2 = float(parameters["w_tally"])
    w3 = float(parameters["w_ttb"])
    w_sum = w1 + w2 + w3 + 1e-9
    
    p_mix = (w1 * p_wadd + w2 * p_tally + w3 * p_ttb) / w_sum
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

`parameters`:
- beta_wadd: [0.1, 10.0]
- beta_tally: [0.1, 10.0]
- beta_ttb: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- w_wadd: [0.0, 1.0]
- w_tally: [0.0, 1.0]
- w_ttb: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4619 (var=0.0040) vs this=0.5231 (var=0.0144)
- Experiment 2: real=0.5637 (var=0.0028) vs this=0.6109 (var=0.0083)
- Experiment 3: real=0.4629 (var=0.0047) vs this=0.5973 (var=0.0174)
- Experiment 4: real=0.5211 (var=0.0132) vs this=0.3950 (var=0.0238)
- Experiment 5: real=0.4568 (var=0.0095) vs this=0.3784 (var=0.0173)
- Experiment 6: real=0.3875 (var=0.0057) vs this=0.3856 (var=0.0216)
- Experiment 7: real=0.4550 (var=0.0032) vs this=0.4321 (var=0.0139)
- Experiment 8: real=0.4913 (var=0.0096) vs this=0.5325 (var=0.0177)
- Experiment 9: real=0.1133 (var=0.0251) vs this=0.4283 (var=0.0437)
- Experiment 10: real=0.4700 (var=0.0174) vs this=0.4017 (var=0.0219)


---

### `pi_5` (overall score: 0.563)

**Description**
Rank-based Weighting Theory posits that decision-makers do not use complex mathematical transformations like log-odds to weigh evidence. Instead, they rely on the simple ordinal ranking of cue validities. Cues are weighted according to an inverse function of their rank (proportional to 1/rank^rho). By restricting the decay parameter rho to a moderate range, the theory maintains a highly compensatory mechanism that prevents over-reliance on the most valid cue, capturing the softer probability matching observed in human multi-attribute decision making.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute ranks (1 is the highest validity)
    # Using stable sort for consistent tie-breaking if validities are equal
    order = np.argsort(-val, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Apply rank-based weighting
    rho = float(parameters["rho"])
    w = 1.0 / (ranks ** rho)
    
    # Compute weighted sum of features for each option
    scores = np.dot(stim, w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.01, 15.0]
- epsilon: [0.0, 0.5]
- rho: [0.0, 2.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4619 (var=0.0040) vs this=0.3494 (var=0.0478)
- Experiment 2: real=0.5637 (var=0.0028) vs this=0.5731 (var=0.0472)
- Experiment 3: real=0.4629 (var=0.0047) vs this=0.4960 (var=0.0554)
- Experiment 4: real=0.5211 (var=0.0132) vs this=0.5006 (var=0.0790)
- Experiment 5: real=0.4568 (var=0.0095) vs this=0.1663 (var=0.0144)
- Experiment 6: real=0.3875 (var=0.0057) vs this=0.4775 (var=0.0936)
- Experiment 7: real=0.4550 (var=0.0032) vs this=0.6829 (var=0.0424)
- Experiment 8: real=0.4913 (var=0.0096) vs this=0.7531 (var=0.0227)
- Experiment 9: real=0.1133 (var=0.0251) vs this=0.3800 (var=0.1059)
- Experiment 10: real=0.4700 (var=0.0174) vs this=0.5275 (var=0.0890)


---

### `pi_6` (overall score: 0.562)

**Description**
Probabilistic Sequential Search Theory posits that decision-makers evaluate cues sequentially in order of their validity. Instead of adopting a strict stopping rule like Take-The-Best or exhaustively evaluating all cues like WADD, individuals accumulate evidence dynamically. When a cue discriminates between options, there is a constant probability (stop_rate) of stopping the search and deciding based on the accumulated evidence. This creates a flexible, cue-by-cue evidence accumulation process that naturally blends lexicographic and compensatory behaviors, explaining the softer choice probabilities and context-dependent trade-offs seen in human decision-making without forcing an overly aggressive stopping rule.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    
    # Sort cues by validity in descending order
    cue_order = np.argsort(-val_clipped, kind="stable").tolist()
    
    stop_rate = float(parameters["stop_rate"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    P_A = 0.0
    P_B = 0.0
    P_continue = 1.0
    E = 0.0
    
    n_features = len(val)
    
    for i, idx in enumerate(cue_order):
        diff = a[idx] - b[idx]
        w = np.log(val_clipped[idx] / (1.0 - val_clipped[idx]))
        E += w * diff
        
        # Determine stopping probability at this step
        if i == n_features - 1:
            S_t = 1.0
        else:
            if diff != 0:
                # Cue discriminates: stop probability is constant stop_rate
                S_t = stop_rate
            else:
                # Cue does not discriminate: always continue
                S_t = 0.0
                
        # Probability of choosing A if search stops here (using stable sigmoid)
        z = beta * E
        if z > 20:
            p_A_if_stop = 1.0
        elif z < -20:
            p_A_if_stop = 0.0
        else:
            p_A_if_stop = 1.0 / (1.0 + np.exp(-z))
            
        p_B_if_stop = 1.0 - p_A_if_stop
        
        # Accumulate overall choice probabilities
        P_A += P_continue * S_t * p_A_if_stop
        P_B += P_continue * S_t * p_B_if_stop
        
        # Update probability of continuing to the next cue
        P_continue *= (1.0 - S_t)
        
    # Apply lapse rate (epsilon)
    P_A = (1.0 - epsilon) * P_A + epsilon * 0.5
    P_B = (1.0 - epsilon) * P_B + epsilon * 0.5
    
    return np.array([P_A, P_B])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- stop_rate: [0.0, 1.0]
- beta: [0.0, 10.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4619 (var=0.0040) vs this=0.6275 (var=0.0179)
- Experiment 2: real=0.5637 (var=0.0028) vs this=0.4300 (var=0.0089)
- Experiment 3: real=0.4629 (var=0.0047) vs this=0.7327 (var=0.0207)
- Experiment 4: real=0.5211 (var=0.0132) vs this=0.2400 (var=0.0220)
- Experiment 5: real=0.4568 (var=0.0095) vs this=0.4253 (var=0.0250)
- Experiment 6: real=0.3875 (var=0.0057) vs this=0.5913 (var=0.0237)
- Experiment 7: real=0.4550 (var=0.0032) vs this=0.5435 (var=0.0338)
- Experiment 8: real=0.4913 (var=0.0096) vs this=0.5819 (var=0.0327)
- Experiment 9: real=0.1133 (var=0.0251) vs this=0.0875 (var=0.0386)
- Experiment 10: real=0.4700 (var=0.0174) vs this=0.6058 (var=0.0314)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    
    # TTB: winner based on the first discriminating cue
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    
    # Tallying: winner based on total number of winning cues
    a_wins = np.sum(diff > 0, axis=1)
    b_wins = np.sum(diff < 0, axis=1)
    
    # Identify conflict trials where Tallying has a strict winner that disagrees with TTB
    tally_winner = np.where(a_wins > b_wins, 0, 1)
    conflict_mask = (a_wins != b_wins) & (ttb_winner != tally_winner)
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    ttb_consistent = (responses[conflict_mask] == ttb_winner[conflict_mask])
    
    return float(np.mean(ttb_consistent))
```

**Observed (real) value:** 0.4619 (var=0.0040)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8731 (var=0.0080)
- pi_2: 0.1437 (var=0.0116)
- pi_3: 0.5666 (var=0.0056)
- pi_4: 0.5231 (var=0.0144)
- pi_5: 0.3494 (var=0.0478)
- pi_6: 0.6275 (var=0.0179)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    tally_pred = (b_wins[mask] > a_wins[mask]).astype(int)
    return float(np.mean(tally_pred == data['response'].values[mask]))
```

**Observed (real) value:** 0.5637 (var=0.0028)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8631 (var=0.0067)
- pi_1: 0.3444 (var=0.0038)
- pi_3: 0.7028 (var=0.0062)
- pi_4: 0.6109 (var=0.0083)
- pi_5: 0.5731 (var=0.0472)
- pi_6: 0.4300 (var=0.0089)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Log-odds weights for validities: [0.9, 0.8, 0.6, 0.55, 0.51]
    w = np.array([2.19722458, 1.38629436, 0.40546511, 0.2006707 , 0.04000533])
    
    def wadd_predicts_A(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * w) > np.sum(b * w)
    
    wadd_pred_A = data.apply(wadd_predicts_A, axis=1)
    
    # Subject chose A if response == 0, B if response == 1
    match = (wadd_pred_A & (data['response'] == 0)) | (~wadd_pred_A & (data['response'] == 1))
    
    return float(match.mean())
```

**Observed (real) value:** 0.4629 (var=0.0047)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8102 (var=0.0128)
- pi_2: 0.1475 (var=0.0082)
- pi_1: 0.8612 (var=0.0092)
- pi_4: 0.5973 (var=0.0174)
- pi_5: 0.4960 (var=0.0554)
- pi_6: 0.7327 (var=0.0207)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert lists to tuples to allow for equality comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify the critical trials where Tallying and WADD make opposing predictions
    t1_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 0, 0, 0, 0))
    t5_mask = (a_tuples == (0, 1, 1, 1, 0)) & (b_tuples == (1, 0, 0, 0, 1))
    t3_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    
    mask_all = t1_mask | t5_mask | t3_mask
    if not mask_all.any():
        return 0.5
        
    # Tallying predictions: 
    # T1: A wins on 3 features, B on 1 -> prefers A (0)
    # T5: A wins on 3 features, B on 2 -> prefers A (0)
    # T3: A wins on 2 features, B on 3 -> prefers B (1)
    preds = pd.Series(index=data.index, data=np.nan)
    preds.loc[t1_mask] = 0
    preds.loc[t5_mask] = 0
    preds.loc[t3_mask] = 1
    
    # Calculate the proportion of choices matching Tallying's predictions
    match = (data.loc[mask_all, 'response'] == preds.loc[mask_all])
    return float(match.mean())
```

**Observed (real) value:** 0.5211 (var=0.0132)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8511 (var=0.0120)
- pi_3: 0.1539 (var=0.0112)
- pi_1: 0.1417 (var=0.0155)
- pi_4: 0.3950 (var=0.0238)
- pi_5: 0.5006 (var=0.0790)
- pi_6: 0.2400 (var=0.0220)

### Experiment 5
**Design**
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: WADD prefers A, Tallying and TTB prefer B
    is_t1 = a_tuples == (0, 1, 1, 0, 0, 0)
    
    # Trial 2: WADD prefers B, Tallying and TTB prefer A
    is_t2 = a_tuples == (1, 0, 0, 1, 1, 1)
    
    wadd_choice_t1 = (data.loc[is_t1, 'response'] == 0).astype(float)
    wadd_choice_t2 = (data.loc[is_t2, 'response'] == 1).astype(float)
    
    combined = np.concatenate([wadd_choice_t1.values, wadd_choice_t2.values])
    if len(combined) == 0:
        return 0.5
    return float(np.mean(combined))
```

**Observed (real) value:** 0.4568 (var=0.0095)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8463 (var=0.0087)
- pi_4: 0.3784 (var=0.0173)
- pi_1: 0.1432 (var=0.0111)
- pi_2: 0.1232 (var=0.0084)
- pi_5: 0.1663 (var=0.0144)
- pi_6: 0.4253 (var=0.0250)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.8, 0.75, 0.7, 0.6])
    w = np.log(val / (1.0 - val))
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    score_a = a_ratings.dot(w)
    score_b = b_ratings.dot(w)
    
    # TTB prefers A, but WADD prefers B
    cond1 = (a_ratings[:, 0] > b_ratings[:, 0]) & (score_a < score_b)
    # TTB prefers B, but WADD prefers A
    cond2 = (b_ratings[:, 0] > a_ratings[:, 0]) & (score_b < score_a)
    
    responses = data['response'].values
    
    ttb_chose_a = cond1 & (responses == 0)
    ttb_chose_b = cond2 & (responses == 1)
    
    ttb_choices = np.sum(ttb_chose_a) + np.sum(ttb_chose_b)
    total_disagreements = np.sum(cond1) + np.sum(cond2)
    
    if total_disagreements == 0:
        return 0.0
        
    return float(ttb_choices / total_disagreements)

```

**Observed (real) value:** 0.3875 (var=0.0057)
**Other theories' values on this metric (for reference):**
- pi_4: 0.3856 (var=0.0216)
- pi_3: 0.1913 (var=0.0139)
- pi_1: 0.8400 (var=0.0165)
- pi_2: 0.1381 (var=0.0152)
- pi_5: 0.4775 (var=0.0936)
- pi_6: 0.5913 (var=0.0237)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings as 2D numpy arrays
    a_cues = np.vstack(data['option_a_ratings'].values)
    b_cues = np.vstack(data['option_b_ratings'].values)
    
    # Identify which option possesses the highest-validity cue (Cue 1, index 0)
    a_has_cue1 = a_cues[:, 0] == 1
    b_has_cue1 = b_cues[:, 0] == 1
    
    # Determine the subject's choice
    chose_a = data['response'].values == 0
    chose_b = data['response'].values == 1
    
    # Boolean array indicating if the chosen option contained Cue 1
    chose_cue1 = (chose_a & a_has_cue1) | (chose_b & b_has_cue1)
    
    # Return the overall proportion of trials where the Cue 1 option was chosen
    return float(chose_cue1.mean())
```

**Observed (real) value:** 0.4550 (var=0.0032)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6829 (var=0.0424)
- pi_4: 0.4321 (var=0.0139)
- pi_1: 0.8431 (var=0.0103)
- pi_2: 0.3229 (var=0.0034)
- pi_3: 0.1344 (var=0.0077)
- pi_6: 0.5435 (var=0.0338)

### Experiment 8
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    cue1_chosen = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Only consider trials where both options have an EQUAL number of cues
        if sum(a) == sum(b):
            # Identify which option possesses the 1st (most valid) cue
            # We assume the first element (index 0) is the most valid cue based on the design
            if a[0] == 1 and b[0] == 0:
                cue1_chosen.append(1 if resp == 0 else 0)
            elif b[0] == 1 and a[0] == 0:
                cue1_chosen.append(1 if resp == 1 else 0)
                
    if len(cue1_chosen) == 0:
        return 0.5
        
    return float(np.mean(cue1_chosen))
```

**Observed (real) value:** 0.4913 (var=0.0096)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5325 (var=0.0177)
- pi_5: 0.7531 (var=0.0227)
- pi_1: 0.8631 (var=0.0086)
- pi_2: 0.4944 (var=0.0068)
- pi_3: 0.1531 (var=0.0137)
- pi_6: 0.5819 (var=0.0327)

### Experiment 9
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = data['A_str'] == '01111'
    t2_mask = data['A_str'] == '10000'
    t3_mask = data['A_str'] == '01100'
    t4_mask = data['A_str'] == '10011'
    
    p_cue1_t1 = (data.loc[t1_mask, 'response'] == 1).mean()
    p_cue1_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_cue1_t3 = (data.loc[t3_mask, 'response'] == 1).mean()
    p_cue1_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    p_12 = np.nanmean([p_cue1_t1, p_cue1_t2])
    p_34 = np.nanmean([p_cue1_t3, p_cue1_t4])
    
    if np.isnan(p_12): p_12 = 0.5
    if np.isnan(p_34): p_34 = 0.5
    
    return float(p_34 - p_12)
```

**Observed (real) value:** 0.1133 (var=0.0251)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0875 (var=0.0386)
- pi_4: 0.4283 (var=0.0437)
- pi_1: 0.0175 (var=0.0108)
- pi_2: 0.7217 (var=0.0191)
- pi_3: 0.7183 (var=0.0450)
- pi_5: 0.3800 (var=0.1059)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_chosen = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        # Trial 3: Cue 0 winner is B
        if a == (0, 1, 1, 1, 0) and b == (1, 0, 0, 0, 1):
            if row['response'] == 1:
                ttb_chosen += 1
            total += 1
        # Trial 4: Cue 0 winner is A
        elif a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 1, 0):
            if row['response'] == 0:
                ttb_chosen += 1
            total += 1
            
    if total == 0:
        return 0.0
    return ttb_chosen / total
```

**Observed (real) value:** 0.4700 (var=0.0174)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4017 (var=0.0219)
- pi_6: 0.6058 (var=0.0314)
- pi_1: 0.8675 (var=0.0109)
- pi_2: 0.1742 (var=0.0171)
- pi_3: 0.2033 (var=0.0201)
- pi_5: 0.5275 (var=0.0890)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a novel theory from scratch based on all the information available, faithfully implementing the mechanism family the arbiter prescribed above. Do NOT simply reuse anything generated in past rounds.

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
  "theory": "Two-Stage Contingent Processing Theory: Decision-makers first perform a rapid, parallel evaluation using a simple tallying heuristic (counting winning cues). If the evidence difference in this first stage is sufficiently large (exceeding a subjective threshold), a choice is made immediately. If the options are too similar, the decision-maker transitions to a second, more effortful stage, evaluating the options using weighted cues (WADD) or a strict lexicographic rule (Take-The-Best).",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)\n    \n    # Stage 1: Tallying\n    a_wins = float(np.sum(a > b))\n    b_wins = float(np.sum(b > a))\n    abs_diff = abs(a_wins - b_wins)\n    \n    threshold = float(parameters[\"threshold\"])\n    beta_thresh = float(parameters[\"beta_thresh\"])\n    \n    # Probability of stopping at Stage 1 (using stable sigmoid)\n    z_thresh = beta_thresh * (abs_diff - threshold)\n    if z_thresh > 20:\n        p_stage1 = 1.0\n    elif z_thresh < -20:\n        p_stage1 = 0.0\n    else:\n        p_stage1 = 1.0 / (1.0 + np.exp(-z_thresh))\n        \n    def get_probs(scores, beta):\n        z = beta * (scores - np.max(scores))\n        e = np.exp(z)\n        return e / np.sum(e)\n        \n    p_tally = get_probs(np.array([a_wins, b_wins]), float(parameters[\"beta_tally\"]))\n    \n    # Stage 2: Effortful processing (Mixture of WADD and TTB)\n    w = np.log(val_clipped / (1.0 - val_clipped))\n    scores_wadd = np.array([np.dot(a, w), np.dot(b, w)])\n    p_wadd = get_probs(scores_wadd, float(parameters[\"beta_wadd\"]))\n    \n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    winner = None\n    for j in cue_order:\n        if a[j] > b[j]:\n            winner = 0\n            break\n        elif b[j] > a[j]:\n            winner = 1\n            break\n            \n    if winner is None:\n        scores_ttb = np.array([0.0, 0.0])\n    else:\n        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])\n    p_ttb = get_probs(scores_ttb, float(parameters[\"beta_ttb\"]))\n    \n    w_wadd = float(parameters[\"w_wadd\"])\n    p_stage2 = w_wadd * p_wadd + (1.0 - w_wadd) * p_ttb\n    \n    # Combine stages\n    p_mix = p_stage1 * p_tally + (1.0 - p_stage1) * p_stage2\n    \n    # Apply lapse rate\n    epsilon = float(parameters[\"epsilon\"])\n    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])\n    \n    return p_final\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "threshold": "[0.0, 5.0]",
    "beta_thresh": "[0.1, 20.0]",
    "beta_tally": "[0.1, 10.0]",
    "beta_wadd": "[0.1, 10.0]",
    "beta_ttb": "[0.1, 10.0]",
    "w_wadd": "[0.0, 1.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, this theory implements a two-stage contingent processing model. Stage 1 evaluates options using a simple tallying heuristic. If the absolute difference in tally scores exceeds a parameterized threshold, the decision is made immediately using Tallying. If the difference is below the threshold, the decision-maker transitions to Stage 2, which uses a more effortful evaluation (a mixture of WADD and TTB). This naturally limits the exaggerated alignment effects seen in pure strategy mixture models, as aligned trials will typically be resolved quickly in Stage 1, while maintaining sensitivity to context-dependent trade-offs on more difficult trials."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14130,
  "candidates_token_count": 1340,
  "total_token_count": 18280
}
```
