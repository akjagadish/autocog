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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a new theory based on Sequential Cue Accumulation or a 'Take-The-Best with Verification' heuristic. Instead of a probabilistic mixture of two distinct rules, decision-makers might search through cues sequentially and accumulate evidence, stopping only when a threshold of evidence is reached. If the most valid cue discriminates but is contradicted by several lower-validity cues, the evidence might fall below the threshold, leading to compensatory-like behavior. This would explain why the TTB winner is often abandoned when secondary cues strongly oppose it, and why the rank of the discriminating cue interacts non-linearly with the tally of remaining cues, capturing the observed negative metric in Experiment 1 and the chance behavior in Experiment 2.


## CANDIDATE THEORY
Sequential Cue Accumulation (Take-The-Best with Verification)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale and normalize weights
    weights = val ** gamma
    weights = weights / np.sum(weights)
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    
    z = beta * (scores - np.max(scores))
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
- theta: [0.0, 1.0]
- gamma: [0.0, 3.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Applied the minimal diff recommended by the critic: restricted the `gamma` range to `[0.0, 3.0]` to prevent extreme weight polarization. Because weights are normalized to sum to 1, `theta` in `[0.0, 1.0]` is already scaled relative to the maximum possible evidence. Lowering the maximum `gamma` ensures that the most valid cue does not trivially exceed `theta` on its own in all cases, allowing the accumulator to process subsequent contradictory cues and correctly produce the compensatory reversals observed in Experiment 5 and the chance-like behavior in Experiment 6.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.5169 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.5697 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.5169.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.5667 (var=0.0480) (Δ vs real -0.1750)
  - iter 2 (current): 0.4933 (var=0.0454) (Δ vs real -0.2483)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8604 (var=0.0104)
- pi_2: 0.2646 (var=0.0053)
- pi_3: 0.6323 (var=0.0207)
- pi_4: 0.5490 (var=0.0315)

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
**Candidate trajectory (this loop):**
  - iter 1: 0.4990 (var=0.0713) (Δ vs real +0.2110)
  - iter 2 (current): 0.5829 (var=0.0674) (Δ vs real +0.2948)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8286 (var=0.0121)
- pi_1: 0.1267 (var=0.0040)
- pi_3: 0.3557 (var=0.0136)
- pi_4: 0.5057 (var=0.0439)

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
**Candidate trajectory (this loop):**
  - iter 1: 0.4694 (var=0.1042) (Δ vs real +0.3306)
  - iter 2 (current): 0.5272 (var=0.1296) (Δ vs real +0.3883)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0011 (var=0.0123)
- pi_3: 0.1228 (var=0.0284)
- pi_2: 0.7100 (var=0.0421)
- pi_4: 0.3678 (var=0.0551)

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
**Candidate trajectory (this loop):**
  - iter 1: 0.1633 (var=0.0347) (Δ vs real +0.1633)
  - iter 2 (current): 0.1867 (var=0.0293) (Δ vs real +0.1867)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2517 (var=0.0479)
- pi_1: 0.0167 (var=0.0161)
- pi_2: 0.0467 (var=0.0362)
- pi_4: -0.0017 (var=0.0332)

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
**Candidate trajectory (this loop):**
  - iter 1: 0.0047 (var=0.0265) (Δ vs real +0.4380)
  - iter 2 (current): 0.0047 (var=0.0144) (Δ vs real +0.4380)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0020 (var=0.0116)
- pi_3: 0.2487 (var=0.0254)
- pi_1: 0.0033 (var=0.0102)
- pi_2: -0.0073 (var=0.0093)

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
**Candidate trajectory (this loop):**
  - iter 1: 0.7633 (var=0.0170) (Δ vs real +0.2687)
  - iter 2 (current): 0.7279 (var=0.0165) (Δ vs real +0.2333)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7946 (var=0.0205)
- pi_4: 0.6727 (var=0.0153)
- pi_1: 0.8510 (var=0.0104)
- pi_2: 0.5025 (var=0.0029)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The Sequential Cue Accumulation model (Take-The-Best with Verification) was accepted and improved the overall loss, but it still struggles to capture the specific empirical patterns in several experiments. In Experiment 1, it underestimates the TTB match rate (0.56 vs 0.74). In Experiment 5, it completely misses the strongly negative observed effect (-0.43 vs 0.00), and in Experiment 6, it overestimates the TTB choice proportion (0.76 vs 0.49). The current parameterization normalizes the weights and uses a threshold `theta` up to 1.0, which theoretically allows both compensatory and non-compensatory behavior, but the model is not hitting the right balance to capture the strong context-dependence seen in the data.
Rationale: Keep the Sequential Cue Accumulation mechanism, but make minor adjustments to improve its fit. First, ensure the threshold `theta` can meaningfully interact with the accumulated evidence: since weights are normalized to sum to 1, a `theta` near 1.0 means exhaustive search. Consider allowing `theta` to be scaled relative to the maximum possible remaining evidence, or adjusting the `gamma` range (e.g., [0.0, 3.0]) to prevent extreme weight polarization that bypasses the threshold logic. Additionally, verify that the evidence accumulation correctly handles negative evidence (contradictory cues) and that the softmax temperature `beta` is calibrated to allow chance-like behavior when evidence falls below the threshold.

**Outcome of your advice:** iter 1 candidate loss=0.5169 -> CURRENT candidate loss=0.5697 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The previous suggestion to restrict the `gamma` range worsened the overall loss (0.5697 vs base 0.5169) and was rejected by the gate. The model continues to struggle with Experiment 2 (overestimating the metric) and Experiment 5 (failing to capture the negative compensatory effect). The current implementation uses Softmax on the accumulated evidence `E` regardless of whether the threshold `theta` was actually reached. This means that if the threshold is never met, the model still behaves like a deterministic weighted-additive model (especially for high `beta`), preventing it from capturing the chance-like behavior observed in Experiment 2.",
  "verdict": "regenerate",
  "rationale": "Since the candidate was rejected, the loop will revert to the Iter 1 base. To improve upon it while staying within the Sequential Cue Accumulation family, do not artificially restrict `gamma` (allow it to go higher, e.g., [0.0, 5.0]). Instead, modify the decision logic at the end of the accumulation loop: if the search exhausts all cues and the final accumulated evidence `abs(E)` is STILL below `theta`, the model should default to guessing (i.e., return `[0.5, 0.5]`) rather than applying the standard Softmax. This directly addresses the arbiter's note about 'chance behavior in Experiment 2' when evidence falls below the threshold, while preserving Take-The-Best behavior when early cues exceed the threshold."
}
```

## Usage

```json
{
  "prompt_token_count": 7611,
  "candidates_token_count": 328,
  "total_token_count": 9471
}
```
