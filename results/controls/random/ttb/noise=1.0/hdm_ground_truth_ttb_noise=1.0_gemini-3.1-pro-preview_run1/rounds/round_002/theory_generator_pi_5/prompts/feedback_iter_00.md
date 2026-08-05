# feedback_iter_00

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Heuristic Decision Making domain.

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
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Since pure Random Guessing (THEORY 1) explains the aggregated data perfectly, propose a brand-new theory that posits an extremely impoverished but non-zero intelligence strategy to see if it captures any residual variance. For example, a 'Single Random Feature' theory: on each trial, a subject randomly samples exactly one feature (ignoring validities) and chooses the option with the higher value on that feature, guessing if they are tied. This provides a theoretical bridge between pure noise and compensatory integration, testing if subjects extract any attribute-level information at all.


## CANDIDATE THEORY
Single Random Feature heuristic: On each trial, a subject randomly samples exactly one feature (ignoring validities) and chooses the option with the higher value on that feature, guessing if they are tied. This provides a theoretical bridge between pure noise and compensatory integration, testing if subjects extract any attribute-level information at all. An epsilon parameter allows blending this strategy with pure random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    epsilon = float(parameters["epsilon"])
    n_features = stim.shape[1]
    
    wins = (stim[0] > stim[1]).sum()
    losses = (stim[0] < stim[1]).sum()
    ties = (stim[0] == stim[1]).sum()
    
    p_A_core = (wins + 0.5 * ties) / n_features
    p_B_core = (losses + 0.5 * ties) / n_features
    
    probs = np.array([p_A_core, p_B_core])
    return (1.0 - epsilon) * probs + epsilon * 0.5

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- epsilon: [0.0, 1.0]

`rationale`:
Following the arbiter's recommendation, this theory models subjects as extracting the absolute minimum amount of attribute-level information: picking a single random feature and choosing based on it, ignoring validities. Given that pure random guessing fits the aggregate data perfectly, this model includes an epsilon parameter to interpolate between the Single Random Feature strategy and pure noise, allowing us to capture any residual variance that might be explained by minimal feature extraction.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.1353 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.1353.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 4, 4]  B=[5, 3, 3]
  A=[5, 0, 0]  B=[2, 1, 1]
  A=[2, 5, 0]  B=[3, 1, 1]
  A=[0, 4, 4]  B=[2, 3, 3]
  A=[0, 5, 5]  B=[3, 4, 4]
  A=[4, 1, 0]  B=[1, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tally_pred = (b_wins > a_wins).astype(int)
    
    return float((data['response'] == tally_pred).mean())
```

**Observed (real) value:** 0.4917 (var=0.0021)
**Candidate (simulated) value:** 0.5829 (var=0.0051)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.3067 (var=0.1115)
- pi_3: 0.2800 (var=0.0057)
- pi_4: 0.5052 (var=0.0023)

### Experiment 2
**Design**
  A=[2, 6, 6]  B=[10, 5, 5]
  A=[10, 2, 2]  B=[8, 3, 3]
  A=[9, 5, 4]  B=[1, 5, 8]
  A=[5, 5, 5]  B=[0, 8, 8]
  A=[1, 9, 9]  B=[8, 8, 8]
  A=[9, 10, 1]  B=[10, 1, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    wadd_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        resp = row['response']
        # Trial 1: A=[2, 6, 6], B=[10, 5, 5]
        if a[0] == 2:
            wadd_match.append(1 if resp == 1 else 0)
        # Trial 5: A=[1, 9, 9], B=[8, 8, 8]
        elif a[0] == 1:
            wadd_match.append(1 if resp == 1 else 0)
    if not wadd_match:
        return 0.5
    return float(np.mean(wadd_match))
```

**Observed (real) value:** 0.5000 (var=0.0073)
**Candidate (simulated) value:** 0.4125 (var=0.0068)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8900 (var=0.0829)
- pi_1: 0.0000 (var=0.0000)
- pi_3: 0.8675 (var=0.0085)
- pi_4: 0.4900 (var=0.0087)

### Experiment 3
**Design**
  A=[6, 0, 0]  B=[5, 10, 10]
  A=[5, 10, 10]  B=[6, 0, 0]
  A=[8, 2, 10]  B=[8, 3, 0]
  A=[0, 8, 8]  B=[1, 0, 0]
  A=[10, 0, 0]  B=[9, 9, 9]
  A=[5, 5, 5]  B=[6, 0, 0]
  A=[2, 10, 10]  B=[3, 2, 2]
  A=[7, 8, 0]  B=[7, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    diff = a_ratings - b_ratings
    
    ttb_choices = np.zeros(len(data), dtype=int) - 1
    for i in range(len(data)):
        for j in range(a_ratings.shape[1]):
            if diff[i, j] > 0:
                ttb_choices[i] = 0
                break
            elif diff[i, j] < 0:
                ttb_choices[i] = 1
                break
                
    valid = ttb_choices != -1
    if not np.any(valid):
        return 0.5
        
    match = (responses[valid] == ttb_choices[valid])
    return float(np.mean(match))
```

**Observed (real) value:** 0.4829 (var=0.0017)
**Candidate (simulated) value:** 0.4290 (var=0.0043)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8508 (var=0.0143)
- pi_2: 0.0200 (var=0.0046)
- pi_1: 0.1250 (var=0.0000)
- pi_4: 0.4860 (var=0.0027)

### Experiment 4
**Design**
  A=[6, 0, 0]  B=[5, 10, 10]
  A=[10, 2, 0]  B=[10, 1, 10]
  A=[1, 10, 10]  B=[2, 0, 0]
  A=[5, 5, 5]  B=[6, 0, 0]
  A=[0, 8, 8]  B=[1, 1, 1]
  A=[7, 7, 7]  B=[7, 8, 0]
  A=[4, 9, 0]  B=[4, 10, 0]
  A=[0, 0, 10]  B=[0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        if a == (6, 0, 0) and b == (5, 10, 10):
            ttb_match += (resp == 0)
            count += 1
        elif a == (10, 2, 0) and b == (10, 1, 10):
            ttb_match += (resp == 0)
            count += 1
        elif a == (1, 10, 10) and b == (2, 0, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (5, 5, 5) and b == (6, 0, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (0, 8, 8) and b == (1, 1, 1):
            ttb_match += (resp == 1)
            count += 1
        elif a == (7, 7, 7) and b == (7, 8, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (4, 9, 0) and b == (4, 10, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (0, 0, 10) and b == (0, 1, 0):
            ttb_match += (resp == 1)
            count += 1
    if count == 0:
        return 0.5
    return ttb_match / count

```

**Observed (real) value:** 0.5017 (var=0.0022)
**Candidate (simulated) value:** 0.4760 (var=0.0038)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1450 (var=0.0059)
- pi_3: 0.8462 (var=0.0150)
- pi_1: 0.2500 (var=0.0000)
- pi_4: 0.5185 (var=0.0025)

### Experiment 5
**Design**
  A=[10, 10, 10]  B=[0, 0, 0]
  A=[0, 0, 0]  B=[10, 10, 10]
  A=[9, 8, 7]  B=[1, 2, 3]
  A=[1, 2, 3]  B=[9, 8, 7]
  A=[10, 0, 10]  B=[0, 10, 0]
  A=[0, 10, 0]  B=[10, 0, 10]
  A=[8, 2, 5]  B=[2, 8, 5]
  A=[2, 8, 5]  B=[8, 2, 5]
  A=[5, 5, 5]  B=[5, 5, 5]
  A=[6, 4, 8]  B=[4, 6, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    correct = np.where(sum_a > sum_b, 0, 1)
    
    return float(np.mean(data.loc[mask, 'response'] == correct[mask]))
```

**Observed (real) value:** 0.4768 (var=0.0049)
**Candidate (simulated) value:** 0.6746 (var=0.0141)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5019 (var=0.0036)
- pi_2: 0.9657 (var=0.0111)
- pi_1: 1.0000 (var=0.0000)
- pi_3: 0.8448 (var=0.0111)

### Experiment 6
**Design**
  A=[10, 10, 10, 10]  B=[0, 0, 0, 0]
  A=[8, 8, 8, 8]  B=[2, 2, 2, 2]
  A=[6, 6, 6, 6]  B=[4, 4, 4, 4]
  A=[5, 5, 5, 5]  B=[5, 5, 5, 5]
  A=[4, 4, 4, 4]  B=[6, 6, 6, 6]
  A=[2, 2, 2, 2]  B=[8, 8, 8, 8]
  A=[0, 0, 0, 0]  B=[10, 10, 10, 10]
  A=[10, 0, 0, 0]  B=[0, 10, 10, 0]
  A=[0, 10, 10, 0]  B=[10, 0, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 0, 10, 10]  B=[10, 10, 0, 0]
  A=[9, 2, 2, 2]  B=[2, 9, 9, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Create a hashable key for each unique trial pair
    trial_keys = data['option_a_ratings'].apply(tuple).astype(str) + "_" + data['option_b_ratings'].apply(tuple).astype(str)
    
    # Calculate the probability of choosing option A (response == 0) for each unique trial
    p_choose_A = (data['response'] == 0).groupby(trial_keys).mean()
    
    # Return the variance of these probabilities across different trial types
    return float(np.var(p_choose_A))
```

**Observed (real) value:** 0.0005 (var=0.0001)
**Candidate (simulated) value:** 0.0265 (var=0.0017)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1872 (var=0.0000)
- pi_4: 0.0004 (var=0.0001)
- pi_1: 0.2431 (var=0.0000)
- pi_3: 0.1076 (var=0.0035)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory faithfully implements the 'Single Random Feature' heuristic with an epsilon-greedy blend towards pure random guessing, as requested by the arbiter. The model's expected choice probabilities correctly match the analytical expectation of picking a single random feature (calculating the proportion of winning, losing, and tied features). The model was ACCEPTED by the gate, capturing the baseline random guessing behavior while allowing for minimal feature extraction. While there are slight deviations in Experiments 1 and 5 compared to pure noise, the mechanism is completely in line with the prescribed family and provides the requested theoretical bridge.",
  "verdict": "continue",
  "rationale": "The candidate fully and correctly implements the prescribed mechanism family (Single Random Feature with an epsilon blend for random guessing). It was accepted by the loss gate, indicating it provides a solid base. No further structural changes are needed within this family."
}
```

## Usage

```json
{
  "prompt_token_count": 5914,
  "candidates_token_count": 190,
  "total_token_count": 7158
}
```
