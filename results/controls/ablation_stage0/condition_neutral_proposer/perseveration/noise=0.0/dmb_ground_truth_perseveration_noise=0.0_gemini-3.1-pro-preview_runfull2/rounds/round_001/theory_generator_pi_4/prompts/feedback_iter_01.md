# feedback_iter_01

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real (human / ground-truth) data, and the value it takes on the candidate's simulated data.

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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a 'Strategy Mixture' or 'Probabilistic Strategy Selection' theory. Instead of assuming all subjects use a single strategy (either pure TTB or pure WADD), this theory should posit that the population consists of a mixture of decision-makers: some rely on a non-compensatory heuristic (TTB) while others use a compensatory strategy (like Tallying or WADD), or individuals probabilistically switch strategies trial-by-trial based on choice difficulty. A mixture model can capture the ~50% alignment rates in conflict trials (by averaging across the population) while potentially accommodating the flat choice probability curves in specific boundary designs like Experiment 2 if the compensatory subset's behavior cancels out or if strategy selection is context-dependent.


## CANDIDATE THEORY
Probabilistic Strategy Selection (TTB and WADD Mixture). The population consists of a mixture of decision-makers: some rely on a non-compensatory heuristic (Take The Best) while others use a compensatory strategy (Weighted Additive, WADD) with choices smoothed by a softmax temperature. The mixture proportion is a fixed parameter alpha.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Determine TTB prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = -1
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        p_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Determine WADD prediction
    tau = float(parameters["tau"])
    scores = np.array([np.sum(a * val), np.sum(b * val)])
    z = (scores - np.max(scores)) / tau
    e = np.exp(z)
    p_wadd = e / np.sum(e)
        
    # Fixed proportion mixture
    alpha = float(parameters["alpha"])
    p_core = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Independent lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- alpha: [0.0, 1.0]
- tau: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the critic's advice, this edit replaces the context-dependent TTB/Tallying mixture with a fixed-proportion mixture of TTB and WADD. The mixture parameter `alpha` controls the population-level proportion of TTB vs WADD. WADD uses the provided validities as weights and generates probabilistic choices via a softmax with temperature `tau`. This allows WADD to capture compensatory behavior in Experiments 1-3, lowering the overpredicted TTB alignment rates from the previous iteration, while still frequently agreeing with TTB in Experiment 4 due to the high validity of the first cue, naturally producing the observed flat choice curves without needing to force 100% TTB on depth=0 trials.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2048 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.2492 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.2048.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # TTB prediction: first discriminating feature in descending validity order
    ttb_preds = np.full(len(data), -1)
    for i in range(a_ratings.shape[1]):
        mask_a = (a_ratings[:, i] > b_ratings[:, i]) & (ttb_preds == -1)
        mask_b = (b_ratings[:, i] > a_ratings[:, i]) & (ttb_preds == -1)
        ttb_preds[mask_a] = 0
        ttb_preds[mask_b] = 1
        
    # Tally prediction: sum of feature-wise wins
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_preds = np.full(len(data), -1)
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    # Identify trials where TTB and Tallying make opposite predictions
    disagree_mask = (ttb_preds != -1) & (tally_preds != -1) & (ttb_preds != tally_preds)
    
    if not np.any(disagree_mask):
        return 0.5
        
    # Calculate proportion of choices aligning with TTB on these crucial trials
    responses = data['response'].values
    matches = (responses[disagree_mask] == ttb_preds[disagree_mask])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5360 (var=0.0087)
**Candidate trajectory (this loop):**
  - iter 1: 0.7430 (var=0.0140) (Δ vs real +0.2070)
  - iter 2 (current): 0.6637 (var=0.0213) (Δ vs real +0.1277)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8393 (var=0.0102)
- pi_2: 0.1500 (var=0.0114)
- pi_3: 0.5487 (var=0.0539)

### Experiment 2
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pref = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    diff = a_mat - b_mat
    nz = diff != 0
    has_nz = np.any(nz, axis=1)
    first_nz_idx = np.argmax(nz, axis=1)
    first_diff = diff[np.arange(len(diff)), first_nz_idx]
    
    ttb_pref = np.where(has_nz, np.where(first_diff > 0, 0, 1), -1)
    
    mask = (tally_pref != -1) & (ttb_pref != -1) & (tally_pref != ttb_pref)
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    tally_prefs = tally_pref[mask]
    
    return float(np.mean(responses == tally_prefs))
```

**Observed (real) value:** 0.5200 (var=0.0096)
**Candidate trajectory (this loop):**
  - iter 1: 0.4200 (var=0.0202) (Δ vs real -0.1000)
  - iter 2 (current): 0.3956 (var=0.0269) (Δ vs real -0.1244)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8751 (var=0.0092)
- pi_1: 0.1409 (var=0.0112)
- pi_3: 0.4533 (var=0.0339)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.vstack(data['option_a_ratings'].values)
    b_mat = np.vstack(data['option_b_ratings'].values)
    
    # Difference in features
    diff = a_mat - b_mat
    
    # Find the first discriminating cue (TTB logic)
    is_diff = diff != 0
    has_diff = is_diff.any(axis=1)
    first_diff_idx = is_diff.argmax(axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # WADD logic (with gamma=1)
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    wadd_diff = diff @ val
    wadd_pred = np.where(wadd_diff > 0, 0, 1)
    
    # Identify conflict trials where TTB and WADD (gamma=1) disagree
    conflict = (ttb_pred != wadd_pred) & has_diff
    
    if not conflict.any():
        return 0.5
        
    # Calculate proportion of choices on conflict trials that align with TTB
    responses = data['response'].values
    matches = (responses[conflict] == ttb_pred[conflict]).sum()
    
    return float(matches / conflict.sum())
```

**Observed (real) value:** 0.4222 (var=0.1296)
**Candidate trajectory (this loop):**
  - iter 1: 0.6447 (var=0.0153) (Δ vs real +0.2225)
  - iter 2 (current): 0.6272 (var=0.0251) (Δ vs real +0.2050)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8228 (var=0.0157)
- pi_3: 0.5472 (var=0.0395)
- pi_2: 0.2175 (var=0.0070)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    diff = sum_a - sum_b
    choose_a = 1.0 - data['response']
    cov = np.cov(diff, choose_a)[0, 1]
    return float(cov)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.0133 (var=0.0055) (Δ vs real +0.0133)
  - iter 2 (current): 0.2654 (var=0.0580) (Δ vs real +0.2654)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5299 (var=0.1410)
- pi_1: 0.0107 (var=0.0072)
- pi_2: 0.8751 (var=0.0369)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate model implements a context-dependent probabilistic strategy selection between TTB and Tallying, using the depth of the first discriminating cue to determine the mixture. This was ACCEPTED by the loop because it successfully captures the 0 covariance in Experiment 4 (by forcing 100% TTB when depth=0). However, it overpredicts TTB alignment in Experiments 1 and 3 (0.74 and 0.64 vs. observed ~0.53 and 0.42). The assumption that depth=0 implies 100% TTB use is too strong, as many conflict trials in Exp 1 and 3 occur at depth=0, yet humans still show significant compensatory behavior.
Rationale: To improve the fit on Experiments 1, 2, and 3 without losing Experiment 4, revise the strategy mixture. Instead of Tallying, use WADD (weighted additive using the provided validities as weights) as the compensatory strategy, and make the mixture proportion a simple fixed population parameter `alpha` (range [0.0, 1.0]) rather than depending on cue depth. In Exp 4, if WADD weights the first cue heavily enough, WADD and TTB will often agree, naturally producing the flat choice curve (0 covariance) without needing to force 100% TTB. Add a temperature parameter `tau` (range [0.1, 5.0]) to the WADD predictions (using softmax) to allow for smooth probabilistic choices, while keeping TTB deterministic. The final prediction should be `(1-epsilon) * (alpha * P_ttb + (1-alpha) * P_wadd) + epsilon * [0.5, 0.5]`.

**Outcome of your advice:** iter 1 candidate loss=0.2048 -> CURRENT candidate loss=0.2492 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The previous attempt (Iter 2) to replace the depth-dependent TTB/Tallying mixture with a fixed TTB/WADD mixture was REJECTED by the gate. The loss increased from 0.2048 to 0.2492. The primary reason for the failure is Experiment 4: the fixed mixture with WADD caused the covariance metric to spike to 0.2654 (real data is 0.0000). In Exp 4, the compensatory strategy's predictions vary across trials as the sum of features changes, which correlates with the difference in sums, breaking the zero covariance. Iter 1 succeeded on Exp 4 by forcing 100% TTB at depth=0, but overpredicted TTB alignment in Exps 1 and 3.",
  "verdict": "regenerate",
  "rationale": "Since the fixed mixture failed, we must build on the ACCEPTED Iter 1 base (depth-dependent mixture of TTB and Tallying). To reduce the overpredicted TTB alignment in Exps 1 and 3 without completely ruining Exp 4, parameterize the depth-dependent rule. Instead of forcing P(TTB) = 1.0 at depth=0, make P(TTB) at depth=0 a free parameter `alpha` (range [0.5, 1.0]), and let P(TTB) decay for deeper cues: e.g., P(TTB) = alpha * (gamma ^ depth), where `gamma` is in [0.0, 1.0]. Also, introduce a softmax temperature `tau` for the Tallying strategy to soften its predictions, rather than making it deterministic. This provides enough flexibility to lower TTB alignment in Exps 1/3 while minimizing the covariance penalty in Exp 4."
}
```

## Usage

```json
{
  "prompt_token_count": 6693,
  "candidates_token_count": 402,
  "total_token_count": 8919
}
```
