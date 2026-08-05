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
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace Theory 2 with a 'Thresholded Parity' or 'Top-K Parity' theory. Instead of falling back to a completely different strategy like tallying, subjects might apply the strict Cue-Parity rule but only to a subset of cues that they consider 'important' (e.g., the top K most valid cues, or cues with validities above a certain threshold). Low-validity cues are ignored and do not contribute to the parity count. This would introduce a validity-dependent effect (as low-validity cues failing to flip the parity would explain the small deviations from pure parity) without requiring an additive tallying mixture that grossly overestimates the validity effect.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.955)

**Description**
People decide by the PARITY of the evidence rather than its weight or direction. They count the number of cues on which option A strictly beats option B, and prefer A when that count is ODD and B when it is EVEN (zero counts as even). This is a deliberately non-monotone, XOR-like rule: adding one more cue in A's favor flips the preference rather than strengthening it, so option dominance does NOT imply choice — an option that wins on every cue (an even count, when the cue number is even) is rejected. The rule uses no validities and no magnitudes, only the parity of feature-wise wins, which makes it an adversarially hard recovery target: it is uncorrelated with any single cue and with the validity-weighted sum, yet perfectly deterministic, much like the anti-majority ensemble. Response noise enters through a softmax over the binary score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Cue-Parity: prefer A iff the number of cues where A strictly beats B is
    # ODD, else prefer B (zero A-wins is even -> B). Non-monotone in the
    # evidence; uses neither validities nor magnitudes. Stimulus is the pair
    # of option feature vectors: (2, n_features), row 0 = A, row 1 = B.
    # History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Cue-Parity expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    a_wins = int(np.sum(a > b))
    # Odd number of A-favoring cues -> A (0); even (including zero) -> B (1).
    winner = 0 if (a_wins % 2 == 1) else 1

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # score this collapses to sigmoid(beta) for the winner (beta=0 ↔ 50/50;
    # beta ≫ 1 ↔ deterministic).
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
- Experiment 1: real=0.7539 (var=0.0403) vs this=0.7083 (var=0.0555)
- Experiment 2: real=0.8178 (var=0.0117) vs this=0.8596 (var=0.0092)
- Experiment 3: real=0.8358 (var=0.0105) vs this=0.8479 (var=0.0101)
- Experiment 4: real=0.1367 (var=0.0064) vs this=0.1700 (var=0.0169)
- Experiment 5: real=0.0525 (var=0.0125) vs this=0.0006 (var=0.0252)
- Experiment 6: real=-0.1572 (var=0.8155) vs this=0.1336 (var=1.0828)
- Experiment 7: real=0.0250 (var=0.0422) vs this=-0.0625 (var=0.0234)
- Experiment 8: real=0.1067 (var=0.0581) vs this=-0.0017 (var=0.0649)


---

### `pi_5` (overall score: 0.744)

**Description**
Tallying-Biased Parity (Strict Parity Dominance): Decision makers primarily rely on a strict Cue-Parity rule (preferring the option with an odd number of winning cues). However, instead of lapsing into purely uniform random noise, their secondary considerations consist of occasionally falling back to a standard validity-weighted tallying strategy. To ensure that tallying acts only as a weak bias and does not override parity in cases of option dominance, the fallback probability is strictly bounded to a low range.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying-Biased Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Core Strategy: Strict Cue-Parity
    a_wins = int(np.sum(a > b))
    parity_winner = 0 if (a_wins % 2 == 1) else 1
    parity_scores = np.array([1.0, 0.0]) if parity_winner == 0 else np.array([0.0, 1.0])
    
    # Fallback Strategy: Validity-weighted Tallying
    val_a = np.sum(validities * (a > b))
    val_b = np.sum(validities * (b > a))
    tally_scores = np.array([val_a, val_b])
    
    beta_parity = float(parameters["beta_parity"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Parity probabilities via softmax
    z_p = beta_parity * (parity_scores - parity_scores.max())
    e_p = np.exp(z_p)
    p_parity = e_p / e_p.sum()
    
    # Tallying probabilities via softmax
    z_t = beta_tally * (tally_scores - tally_scores.max())
    e_t = np.exp(z_t)
    p_tally = e_t / e_t.sum()
    
    # Mixture of the two strategies
    return (1.0 - epsilon) * p_parity + epsilon * p_tally


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- validities: validities
- beta_parity: [0.1, 20.0]
- beta_tally: [0.1, 10.0]
- epsilon: [0.0, 0.3]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7539 (var=0.0403) vs this=0.6650 (var=0.0488)
- Experiment 2: real=0.8178 (var=0.0117) vs this=0.9009 (var=0.0082)
- Experiment 3: real=0.8358 (var=0.0105) vs this=0.9079 (var=0.0059)
- Experiment 4: real=0.1367 (var=0.0064) vs this=0.0760 (var=0.0021)
- Experiment 5: real=0.0525 (var=0.0125) vs this=0.0037 (var=0.0131)
- Experiment 6: real=-0.1572 (var=0.8155) vs this=-2.1229 (var=1.2146)
- Experiment 7: real=0.0250 (var=0.0422) vs this=0.2612 (var=0.0429)
- Experiment 8: real=0.1067 (var=0.0581) vs this=0.2767 (var=0.0857)


---

### `pi_4` (overall score: 0.688)

**Description**
Continuous Weighted Parity (Restricted Parameter): Decision makers compute a parity-like signal, but the contribution of each cue is weighted by its validity. The evidence is integrated into a continuous sum of validity-weighted wins, which is then passed through a periodic parity function (negative cosine). To avoid chaotic phase shifts that destroy the parity signal, the influence of validity weights is kept small, allowing them to gently perturb the parity score without pushing the sum into the 'dead zones' (half-integers) of the cosine function.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Continuous Weighted Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    w_validity = float(parameters["w_validity"])
    
    # Calculate weights for each cue based on validity
    mean_val = np.mean(validities)
    weights = 1.0 + w_validity * (validities - mean_val)
    
    # Compute weighted sum of strict A wins
    a_wins_mask = (a > b).astype(float)
    weighted_wins = np.sum(weights * a_wins_mask)
    
    # Continuous parity score: maps even integers to -1 (favoring B) and odd integers to +1 (favoring A)
    parity_score = -np.cos(np.pi * weighted_wins)
    
    # Scores for A and B
    score_a = parity_score
    score_b = -parity_score
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax conversion with numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- validities: validities
- w_validity: [0.0, 0.5]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7539 (var=0.0403) vs this=0.7444 (var=0.0359)
- Experiment 2: real=0.8178 (var=0.0117) vs this=0.8662 (var=0.0091)
- Experiment 3: real=0.8358 (var=0.0105) vs this=0.8425 (var=0.0081)
- Experiment 4: real=0.1367 (var=0.0064) vs this=0.1397 (var=0.0083)
- Experiment 5: real=0.0525 (var=0.0125) vs this=0.4313 (var=0.3434)
- Experiment 6: real=-0.1572 (var=0.8155) vs this=0.5248 (var=2.5556)
- Experiment 7: real=0.0250 (var=0.0422) vs this=0.0325 (var=0.0216)
- Experiment 8: real=0.1067 (var=0.0581) vs this=0.0183 (var=0.0395)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.5035 -> ACCEPTED
- iter 2: loss=0.3762 -> ACCEPTED
- iter 3: loss=0.2901 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.2901 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_wins = data.apply(lambda row: sum(a > b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    b_wins = data.apply(lambda row: sum(b > a for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    
    mask_123 = (a_wins.isin([2, 4])) & (b_wins.isin([0, 1]))
    mask_45 = (a_wins == 1) & (b_wins.isin([2, 3]))
    
    val_123 = data.loc[mask_123, 'response'].mean() if mask_123.any() else 0.5
    val_45 = data.loc[mask_45, 'response'].mean() if mask_45.any() else 0.5
    
    return float(val_123 - val_45)
```

**Observed (real) value:** 0.7539 (var=0.0403)
**Previous candidate values (this loop):**
  - iter 1: 0.5156 (var=0.1193) (Δ vs real -0.2383)
  - iter 2: 0.5658 (var=0.0381) (Δ vs real -0.1881)
  - iter 3 (most recent): 0.6092 (var=0.0504) (Δ vs real -0.1447)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7083 (var=0.0555)
- pi_2: -0.7050 (var=0.0284)
- pi_3: 0.5431 (var=0.0498)
- pi_4: 0.7444 (var=0.0359)
- pi_5: 0.6650 (var=0.0488)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_wins = data.apply(lambda row: sum(1 for x, y in zip(row['option_a_ratings'], row['option_b_ratings']) if x > y), axis=1)
    b_wins = data.apply(lambda row: sum(1 for x, y in zip(row['option_a_ratings'], row['option_b_ratings']) if y > x), axis=1)
    
    mask = a_wins == b_wins
    if not mask.any():
        return 0.5
        
    tied_data = data[mask]
    tied_a_wins = a_wins[mask]
    
    # Cue-Parity predicts A (0) if a_wins is odd, B (1) if a_wins is even
    cp_preds = (tied_a_wins % 2 == 0).astype(int)
    
    matches = (tied_data['response'] == cp_preds).astype(float)
    return float(matches.mean())
```

**Observed (real) value:** 0.8178 (var=0.0117)
**Previous candidate values (this loop):**
  - iter 1: 0.6973 (var=0.0466) (Δ vs real -0.1204)
  - iter 2: 0.8356 (var=0.0093) (Δ vs real +0.0178)
  - iter 3 (most recent): 0.7969 (var=0.0185) (Δ vs real -0.0209)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5018 (var=0.0052)
- pi_1: 0.8596 (var=0.0092)
- pi_3: 0.8516 (var=0.0121)
- pi_4: 0.8662 (var=0.0091)
- pi_5: 0.9009 (var=0.0082)

### Experiment 3
**Design**
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract options as 2D arrays
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Compute strict A-wins across all 5 cues
    a_wins = np.sum(a_mat > b_mat, axis=1)
    
    # Cue-Parity rule: Odd A-wins -> A (0), Even A-wins -> B (1)
    cp_preds = (a_wins % 2 == 0).astype(int)
    
    # Compute agreement between subjects' responses and Cue-Parity predictions
    responses = data['response'].values
    return float(np.mean(responses == cp_preds))
```

**Observed (real) value:** 0.8358 (var=0.0105)
**Previous candidate values (this loop):**
  - iter 1: 0.4423 (var=0.0263) (Δ vs real -0.3935)
  - iter 2: 0.3983 (var=0.0358) (Δ vs real -0.4375)
  - iter 3 (most recent): 0.4250 (var=0.0361) (Δ vs real -0.4108)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8479 (var=0.0101)
- pi_3: 0.4615 (var=0.0550)
- pi_2: 0.5721 (var=0.0037)
- pi_4: 0.8425 (var=0.0081)
- pi_5: 0.9079 (var=0.0059)

### Experiment 4
**Design**
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        top3_awins = np.sum(a[:3] > b[:3])
        all_awins = np.sum(a > b)
        
        if (top3_awins % 2) != (all_awins % 2):
            pred = 0 if top3_awins % 2 == 1 else 1
            matches.append(1.0 if row['response'] == pred else 0.0)
            
    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1367 (var=0.0064)
**Previous candidate values (this loop):**
  - iter 1: 0.5620 (var=0.0529) (Δ vs real +0.4253)
  - iter 2: 0.3430 (var=0.0575) (Δ vs real +0.2063)
  - iter 3 (most recent): 0.4720 (var=0.0866) (Δ vs real +0.3353)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3667 (var=0.0840)
- pi_1: 0.1700 (var=0.0169)
- pi_2: 0.5987 (var=0.0025)
- pi_4: 0.1397 (var=0.0083)
- pi_5: 0.0760 (var=0.0021)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the unweighted sum of cues favoring Option A (since B is all 0s)
    a_sums = data['option_a_ratings'].apply(sum)
    
    # Responses: 0 means A was chosen, 1 means B was chosen
    is_a = (data['response'] == 0).astype(float)
    is_b = (data['response'] == 1).astype(float)
    
    # For odd sums (5 and 7), Cue-Parity predicts A with equal strength.
    # For CW-Parity, sum=5 is pushed towards the half-integer dead zone by validities,
    # reducing P(A), while sum=7 is barely affected.
    p_a_7 = is_a[a_sums == 7].mean()
    p_a_5 = is_a[a_sums == 5].mean()
    
    # For even sums (4 and 6), Cue-Parity predicts B with equal strength.
    # For CW-Parity, sum=6 is pushed towards the half-integer dead zone,
    # reducing P(B), while sum=4 is unaffected.
    p_b_4 = is_b[a_sums == 4].mean()
    p_b_6 = is_b[a_sums == 6].mean()
    
    diff1 = (p_a_7 - p_a_5) if pd.notna(p_a_7) and pd.notna(p_a_5) else 0.0
    diff2 = (p_b_4 - p_b_6) if pd.notna(p_b_4) and pd.notna(p_b_6) else 0.0
    
    return float(diff1 + diff2)
```

**Observed (real) value:** 0.0525 (var=0.0125)
**Previous candidate values (this loop):**
  - iter 1: -0.4031 (var=0.0238) (Δ vs real -0.4556)
  - iter 2: -0.3875 (var=0.0227) (Δ vs real -0.4400)
  - iter 3 (most recent): -0.3000 (var=0.1422) (Δ vs real -0.3525)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0006 (var=0.0252)
- pi_4: 0.4313 (var=0.3434)
- pi_2: 0.0050 (var=0.0240)
- pi_3: -0.4175 (var=0.1314)
- pi_5: 0.0037 (var=0.0131)

### Experiment 6
**Design**
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins_mask = (a_ratings > b_ratings).astype(int)
    a_wins = a_wins_mask.sum(axis=1)
    high_wins = a_wins_mask[:, :5].sum(axis=1)
    
    # T1, T2: 5 wins, strictly uniform validities
    mask_12 = (a_wins == 5) & ((high_wins == 0) | (high_wins == 5))
    
    # T3, T4: 4 wins, strictly uniform validities
    mask_34 = (a_wins == 4) & ((high_wins == 0) | (high_wins == 4))
    
    if mask_12.sum() == 0 or mask_34.sum() == 0:
        return 0.0
        
    # h is the number of times Option A was chosen
    h_12 = (data['response'].values[mask_12] == 0).sum()
    m_12 = mask_12.sum() - h_12
    
    h_34 = (data['response'].values[mask_34] == 0).sum()
    m_34 = mask_34.sum() - h_34
    
    # Compute log-odds of choosing A with Laplace smoothing
    lo_12 = np.log((h_12 + 0.5) / (m_12 + 0.5))
    lo_34 = np.log((h_34 + 0.5) / (m_34 + 0.5))
    
    # We return the negative sum of these log-odds
    return float(-(lo_12 + lo_34))
```

**Observed (real) value:** -0.1572 (var=0.8155)
**Previous candidate values (this loop):**
  - iter 1: 1.7323 (var=1.1747) (Δ vs real +1.8894)
  - iter 2: 1.6957 (var=0.9451) (Δ vs real +1.8529)
  - iter 3 (most recent): 0.1893 (var=1.2510) (Δ vs real +0.3464)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5248 (var=2.5556)
- pi_1: 0.1336 (var=1.0828)
- pi_2: -3.7483 (var=3.7447)
- pi_3: 0.2203 (var=0.9312)
- pi_5: -2.1229 (var=1.2146)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = (data['a_str'] == '10000') & (data['b_str'] == '00000')
    t2_mask = (data['a_str'] == '00010') & (data['b_str'] == '11100')
    t3_mask = (data['a_str'] == '11000') & (data['b_str'] == '00000')
    t4_mask = (data['a_str'] == '00011') & (data['b_str'] == '11100')
    
    p_a_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_a_t2 = 1.0 - data.loc[t2_mask, 'response'].mean()
    p_a_t3 = 1.0 - data.loc[t3_mask, 'response'].mean()
    p_a_t4 = 1.0 - data.loc[t4_mask, 'response'].mean()
    
    diff1 = (p_a_t1 - p_a_t2) if pd.notna(p_a_t1) and pd.notna(p_a_t2) else 0.0
    diff2 = (p_a_t3 - p_a_t4) if pd.notna(p_a_t3) and pd.notna(p_a_t4) else 0.0
    
    return float(diff1 + diff2)
```

**Observed (real) value:** 0.0250 (var=0.0422)
**Previous candidate values (this loop):**
  - iter 1: 0.6425 (var=0.4094) (Δ vs real +0.6175)
  - iter 2: 0.2812 (var=0.3566) (Δ vs real +0.2562)
  - iter 3 (most recent): 0.0737 (var=0.4375) (Δ vs real +0.0487)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0625 (var=0.0234)
- pi_5: 0.2612 (var=0.0429)
- pi_2: 1.3538 (var=0.1662)
- pi_3: 0.0150 (var=0.4649)
- pi_4: 0.0325 (var=0.0216)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Identify the trial types based on option A's features.
    # T1: sum=1, A[0]=1 (Parity favors A, Tally favors A)
    # T2: sum=1, A[0]=0 (Parity favors A, Tally favors B)
    # T3: sum=2, A[0]=1 (Parity favors B, Tally favors A)
    # T4: sum=2, A[0]=0 (Parity favors B, Tally favors B)
    # T5: sum=3, A[0]=1 (Parity favors A, Tally favors A)
    # T6: sum=3, A[0]=0 (Parity favors A, Tally favors B)
    
    def get_trial_type(row):
        a = row['option_a_ratings']
        s = sum(a)
        if s == 1:
            return 'T1' if a[0] == 1 else 'T2'
        elif s == 2:
            return 'T3' if a[0] == 1 else 'T4'
        elif s == 3:
            return 'T5' if a[0] == 1 else 'T6'
        return 'Other'
        
    data = data.copy()
    data['trial_type'] = data.apply(get_trial_type, axis=1)
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    # Calculate mean P(choose A) for each trial type
    means = data.groupby('trial_type')['chose_A'].mean()
    
    def get_mean(t):
        return means.get(t, 0.0)
        
    # Difference in P(choose A) when Tally favors A vs when Tally favors B,
    # holding Parity constant.
    diff_1 = get_mean('T1') - get_mean('T2')
    diff_2 = get_mean('T3') - get_mean('T4')
    diff_3 = get_mean('T5') - get_mean('T6')
    
    return float(diff_1 + diff_2 + diff_3)

```

**Observed (real) value:** 0.1067 (var=0.0581)
**Previous candidate values (this loop):**
  - iter 1: 0.7233 (var=0.1315) (Δ vs real +0.6167)
  - iter 2: 0.6233 (var=0.1376) (Δ vs real +0.5167)
  - iter 3 (most recent): 0.0183 (var=0.3212) (Δ vs real -0.0883)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2767 (var=0.0857)
- pi_1: -0.0017 (var=0.0649)
- pi_2: -0.3283 (var=0.1282)
- pi_3: -0.1433 (var=0.3553)
- pi_4: 0.0183 (var=0.0395)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Top-K Parity: Decision makers rely on a strict Cue-Parity rule but apply it selectively to a stable subset of the most valid cues. Instead of using a floating validity threshold, they sort cues by validity and consider only the top K cues (determined as a fraction of the total available cues). This ensures a stable number of cues is evaluated, preserving multi-cue parity effects without collapsing to a single-cue strategy or randomly flipping the parity sum due to minor validity differences.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Top-K Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    k_frac = float(parameters["k_frac"])
    
    n_cues = len(validities)
    k = max(1, int(np.round(k_frac * n_cues)))
    
    # Get indices of the top K validities
    # np.argsort sorts ascending, so we take the last k elements
    top_k_indices = np.argsort(validities)[-k:]
    
    a_filtered = a[top_k_indices]
    b_filtered = b[top_k_indices]
    
    # Strict Cue-Parity on the filtered subset of cues
    a_wins = int(np.sum(a_filtered > b_filtered))
    winner = 0 if (a_wins % 2 == 1) else 1
    
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary score
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- validities: validities
- k_frac: [0.4, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

`rationale`: Following the critic's advice, I replaced the floating `validity_threshold` with a `k_frac` parameter that determines the proportion of top validities to keep. This implements the 'Top-K Parity' strategy, ensuring a stable number of cues is evaluated on every trial. This prevents the parity sum from randomly flipping due to validities falling just below a floating threshold, which was destroying the model's correlation with the 5-cue parity behavior observed in Experiment 3.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The Thresholded Parity model was accepted and provides a good conceptual step forward, but its quantitative fit is struggling, particularly on Experiments 3, 4, 5, and 6. The primary issue is likely the parameter range for `validity_threshold` ([0.5, 1.0]). Because validities are often distributed between 0.5 and 1.0, a high threshold (e.g., > 0.9) frequently filters out almost all cues, triggering the fallback to only the maximum-validity cue. When only one cue is considered, Cue-Parity degenerates into Take-The-Best (1 win = odd = A; 0 wins = even = B). This over-reliance on a single cue destroys the multi-cue parity effects observed in Experiment 3 (where human parity adherence is high but the model drops to ~0.44) and causes wild deviations in Experiments 5 and 6.
Rationale: Tighten the `validity_threshold` parameter range to something lower, such as [0.5, 0.8] or [0.5, 0.75]. This ensures that the model retains a meaningful subset of cues for the parity calculation rather than frequently collapsing to a single-cue (Take-The-Best) strategy when the threshold samples too high. Alternatively, implement a strict 'Top-K Parity' where K is an integer parameter (e.g., K in [2, 5]), ensuring a fixed number of cues are always evaluated for parity.

**Outcome of this advice:** iter 1 candidate loss=0.5035 -> iter 2 candidate loss=0.3762 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The reduction in the validity threshold successfully improved the model's performance on Experiments 2, 4, 7, and 8, leading to an accepted candidate with a lower aggregate loss (0.3762). However, the model still fails dramatically on Experiment 3 (0.3983 vs real 0.8358) and Experiment 5 (-0.3875 vs real 0.0525). The core issue is that a floating validity threshold drops a variable number of cues depending on the specific validity samples. Dropping even a single cue often flips the parity sum (from odd to even or vice versa), which completely destroys the correlation with the 5-cue parity behavior observed in Experiment 3.
Rationale: To fix the instability of the parity sum, switch to the alternative mechanism provided in the arbiter's recommendation: 'Top-K Parity'. Instead of using a validity threshold, sort the cues by validity and keep exactly the top K cues. Make K an integer parameter with a range like [3, 5] or [1, max_cues]. This guarantees a stable number of cues are evaluated and prevents the parity sum from randomly flipping due to validities falling just below a floating threshold.

**Outcome of this advice:** iter 2 candidate loss=0.3762 -> iter 3 candidate loss=0.2901 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The switch to Top-K Parity using a fractional parameter `k_frac` successfully stabilized the number of evaluated cues and improved the aggregate loss, yielding an accepted candidate. The model's performance on Experiments 6, 7, and 8 improved significantly. However, it still fails on Experiment 3 (0.4250 vs real 0.8358) and Experiment 5 (-0.3000 vs real 0.0525). The issue lies in the `k_frac` parameter range [0.4, 1.0]. For a 5-cue experiment like Experiment 3, a `k_frac` of 0.4 means only the top 2 cues are evaluated. Evaluating only 2 or 3 cues frequently flips the parity sum relative to the full 5-cue parity, which destroys the model's ability to match the high empirical adherence to 5-cue parity.
Rationale: Tighten the `k_frac` parameter range from [0.4, 1.0] to [0.7, 1.0] or [0.8, 1.0]. Subjects applying Top-K Parity likely ignore only the least valid 1 or 2 cues rather than discarding more than half of the available information. By raising the lower bound of `k_frac`, the model will consistently evaluate the vast majority of the cues (e.g., 4 or 5 cues out of 5), which is mathematically required to recover the strong full-cue parity effects observed in Experiment 3 and Experiment 5, while still allowing enough flexibility to explain deviations from strict parity.

**Outcome of this advice:** iter 3 candidate loss=0.2901 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Top-K Parity: Decision makers rely on a strict Cue-Parity rule but apply it selectively to a stable subset of the most valid cues. Instead of using a floating validity threshold, they sort cues by validity and consider only the top K cues (determined as a fraction of the total available cues). This ensures a stable number of cues is evaluated, preserving multi-cue parity effects without collapsing to a single-cue strategy or randomly flipping the parity sum due to minor validity differences.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Top-K Parity expects a (2, n_features) stimulus.\")\n    \n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    k_frac = float(parameters[\"k_frac\"])\n    \n    n_cues = len(validities)\n    k = max(1, int(np.round(k_frac * n_cues)))\n    \n    # Get indices of the top K validities\n    # np.argsort sorts ascending, so we take the last k elements\n    top_k_indices = np.argsort(validities)[-k:]\n    \n    a_filtered = a[top_k_indices]\n    b_filtered = b[top_k_indices]\n    \n    # Strict Cue-Parity on the filtered subset of cues\n    a_wins = int(np.sum(a_filtered > b_filtered))\n    winner = 0 if (a_wins % 2 == 1) else 1\n    \n    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax over binary score\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "validities": "validities",
    "k_frac": "[0.8, 1.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]"
  },
  "rationale": "Following the critic's advice, I tightened the `k_frac` parameter range from [0.4, 1.0] to [0.8, 1.0]. A wider range allowed the model to discard too many cues (e.g., evaluating only 2 out of 5 cues), which frequently flipped the parity sum relative to the full-cue parity and destroyed the model's correlation with the 5-cue parity behavior observed in Experiment 3. By raising the lower bound, the model will consistently evaluate the vast majority of cues, recovering strong full-cue parity effects while still allowing minor deviations from strict parity."
}
```
```

## Usage

```json
{
  "prompt_token_count": 16324,
  "candidates_token_count": 866,
  "total_token_count": 17936
}
```
