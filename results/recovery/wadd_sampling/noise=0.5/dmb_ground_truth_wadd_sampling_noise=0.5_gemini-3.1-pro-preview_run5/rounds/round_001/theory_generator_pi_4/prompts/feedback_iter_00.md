# feedback_iter_00

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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Replace Tallying with a Strategy Mixture Theory. The new theory should propose that decision-makers probabilistically choose between two distinct heuristics: Tallying (counting winning features) and Take-The-Best (relying solely on the highest-validity discriminating feature). The model should include a mixture parameter (e.g., P_tally) that governs the probability of applying Tallying versus TTB on a given trial, alongside response noise. This will allow the model to capture the intermediate valid-cue preferences in Exps 1-2 and the partial Tallying matches in Exp 4, effectively bridging the gap between the two extremes.


## CANDIDATE THEORY
Strategy Mixture Theory: Decision-makers do not rely on a single heuristic across all trials. Instead, they probabilistically choose between two distinct strategies on any given trial: Tallying (counting the number of winning features for each option) and Take-The-Best (relying solely on the single highest-validity feature that discriminates between the options). The probability of using Tallying versus TTB is governed by a mixture parameter. This allows the model to capture intermediate valid-cue preferences and partial matches to Tallying, bridging the gap between the two extremes of compensatory and non-compensatory decision-making.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_tally_weight = float(parameters["p_tally_weight"])
    
    # --- Tallying Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # --- Take-The-Best (TTB) Prediction ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Mixture & Noise ---
    p_mix = p_tally_weight * p_tally + (1.0 - p_tally_weight) * p_ttb
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_tally_weight: [0.0, 1.0]
- validities: validities

`rationale`:
Based on the arbiter's feedback, this theory implements a mixture of Tallying and Take-The-Best heuristics. By probabilistically selecting between these two strategies on a per-trial basis (governed by p_tally_weight), the model can produce behavior that is intermediate between a purely compensatory Tallying approach and a strictly non-compensatory TTB approach. This flexibility allows it to better match the observed intermediate valid-cue preferences in Exps 1-2 and partial matches in Exp 4, overcoming the failures of the pure Tallying or pure TTB models.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.2409 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.2409.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def ttb_predict(a, b):
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            if b[i] > a[i]:
                return 1
        return -1
    
    matches = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if ttb_predict(a, b) == r:
            matches += 1
            
    return float(matches) / len(data) if len(data) > 0 else 0.0
```

**Observed (real) value:** 0.3747 (var=0.0051)
**Candidate (simulated) value:** 0.4947 (var=0.0452)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8407 (var=0.0059)
- pi_2: 0.1318 (var=0.0071)
- pi_3: 0.3573 (var=0.0363)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate sum of features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Calculate first feature for each option (most valid cue)
    first_a = data['option_a_ratings'].apply(lambda x: x[0])
    first_b = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Identify trials where Tallying and TTB make strictly opposite predictions
    # Tallying decides by sum(A) vs sum(B)
    # TTB decides by A[0] vs B[0]
    opposing_mask = ((sum_a > sum_b) & (first_a < first_b)) | ((sum_a < sum_b) & (first_a > first_b))
    opposing_data = data[opposing_mask]
    
    if len(opposing_data) == 0:
        return 0.5
        
    # Determine what Tallying would choose: 0 (A) if sum_a > sum_b else 1 (B)
    tallying_choice = (opposing_data['option_a_ratings'].apply(sum) < opposing_data['option_b_ratings'].apply(sum)).astype(int)
    
    # Calculate the proportion of responses that match the Tallying prediction
    match_rate = (opposing_data['response'] == tallying_choice).mean()
    
    return float(match_rate)
```

**Observed (real) value:** 0.6400 (var=0.0095)
**Candidate (simulated) value:** 0.5442 (var=0.0534)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8767 (var=0.0063)
- pi_1: 0.1408 (var=0.0115)
- pi_3: 0.7929 (var=0.0253)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    wadd_match = 0
    n_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            wadd_a = np.sum(a * val)
            wadd_b = np.sum(b * val)
            
            if wadd_a > wadd_b:
                expected = 0
            elif wadd_b > wadd_a:
                expected = 1
            else:
                continue
                
            if row['response'] == expected:
                wadd_match += 1
            n_ties += 1
            
    if n_ties == 0:
        return 0.5
    return wadd_match / n_ties
```

**Observed (real) value:** 0.5863 (var=0.0190)
**Candidate (simulated) value:** 0.6681 (var=0.0143)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7831 (var=0.0163)
- pi_2: 0.4850 (var=0.0067)
- pi_1: 0.8438 (var=0.0137)

### Experiment 4
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target_pairs = [
        ( (1,0,0,0), (0,1,0,0) ),
        ( (1,0,1,0), (0,1,0,1) )
    ]
    
    chosen_has_f1 = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        is_target = False
        for p in target_pairs:
            if (a == p[0] and b == p[1]) or (a == p[1] and b == p[0]):
                is_target = True
                break
                
        if is_target:
            chose_a = (row['response'] == 0)
            chosen_opt = a if chose_a else b
            
            if chosen_opt[0] == 1:
                chosen_has_f1.append(1)
            else:
                chosen_has_f1.append(0)
                
    if len(chosen_has_f1) == 0:
        return 0.5
        
    return float(np.mean(chosen_has_f1))
```

**Observed (real) value:** 0.5317 (var=0.0228)
**Candidate (simulated) value:** 0.6850 (var=0.0158)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4892 (var=0.0105)
- pi_3: 0.7533 (var=0.0292)
- pi_1: 0.8408 (var=0.0152)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The Strategy Mixture Theory successfully integrates Tallying and Take-The-Best, yielding an accepted loss of 0.2409. It captures the intermediate nature of the behavior across experiments better than pure models. However, there are still discrepancies: it underpredicts the Tallying match in Exp 2 (0.54 vs 0.64) and overpredicts the TTB match in Exp 1 (0.49 vs 0.37) and Exp 4 (0.68 vs 0.53). A likely reason for this is that the shared `beta` parameter acts asymmetrically on the two strategies: TTB scores are [1.0, 0.0] (max difference of 1), while Tallying scores are raw counts (max difference up to 5). This makes the Tallying policy artificially much more deterministic than TTB for any given `beta`.",
  "verdict": "regenerate",
  "rationale": "The candidate correctly implements the prescribed mechanism family (Tallying + TTB mixture) and improved the loss. To further improve the fit within this family, normalize the Tallying scores (e.g., by dividing by the total number of features or by the total number of discriminating features) so that the maximum score difference is 1.0, matching the scale of the TTB scores. This will allow the shared `beta` parameter to control the determinism of both heuristics symmetrically, enabling `p_tally_weight` to more accurately balance the two strategies without scale distortion."
}
```

## Usage

```json
{
  "prompt_token_count": 5803,
  "candidates_token_count": 341,
  "total_token_count": 7236
}
```
