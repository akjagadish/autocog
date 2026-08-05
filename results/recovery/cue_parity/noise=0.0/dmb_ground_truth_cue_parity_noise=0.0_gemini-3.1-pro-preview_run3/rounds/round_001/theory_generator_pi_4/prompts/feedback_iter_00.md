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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Propose a 'Strategy Selection' or 'Mixture' theory. This theory should posit that decision-makers maintain a repertoire of strategies (e.g., Take The Best and Tallying/WADD). On any given trial or across subjects, there is a probability of selecting the fast-and-frugal TTB strategy versus a more compensatory strategy. Alternatively, propose a 'Threshold' model where evidence is accumulated in order of validity, but lower-validity cues can override the top cue if their combined weight exceeds a certain threshold. This would allow the model to capture the strong TTB-like behavior in Exps 1 and 2 while accommodating the compensatory behavior observed in Exps 3 and 4.


## CANDIDATE THEORY
Decision makers predominantly use a fast-and-frugal Take The Best (TTB) strategy, sequentially inspecting cues in order of validity. However, they also monitor the total amount of contradictory evidence. If the number of lower-validity cues favoring the TTB-losing option exceeds the number of cues favoring the TTB-winning option by a certain individual-specific threshold, the decision maker abandons TTB and chooses the option with more overall supporting cues (Tallying). This Threshold model explains both the strong TTB adherence in standard environments and the compensatory behavior observed when overwhelming contradictory evidence is present.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the TTB winner
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        if b[j] > a[j]:
            ttb_winner = 1
            break
            
    # Count total wins for each option (Tallying)
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    winner = ttb_winner
    threshold = float(parameters["threshold"])
    
    # Override TTB if contradictory evidence meets the threshold
    if ttb_winner == 0:
        if b_wins - a_wins >= threshold:
            winner = 1
    elif ttb_winner == 1:
        if a_wins - b_wins >= threshold:
            winner = 0
            
    scores = np.array([0.0, 0.0])
    if winner == 0:
        scores = np.array([1.0, 0.0])
    elif winner == 1:
        scores = np.array([0.0, 1.0])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- threshold: [1.0, 4.0]
- validities: validities

`rationale`:
Following the arbiter's suggestion, this Threshold model acts as a mechanistic bridge between non-compensatory (TTB) and compensatory (Tallying) strategies. By default, the model relies on the single best discriminating cue. However, if the opposing option has a significantly higher number of supporting cues, and that difference exceeds an internal threshold, the decision maker switches to the compensatory choice. This perfectly explains the divergence between Experiment 3 (where the contradictory evidence difference is small, keeping behavior TTB-like) and Experiment 4 (where the difference is larger, triggering the compensatory override).

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.2714 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.2714.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Features are pre-sorted by validity due to the design
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred != -1:
            matches.append(1.0 if row['response'] == ttb_pred else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.6369 (var=0.0028)
**Candidate (simulated) value:** 0.7269 (var=0.0213)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8289 (var=0.0104)
- pi_2: 0.2189 (var=0.0062)
- pi_3: 0.5778 (var=0.0301)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def tally_predict(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_w = np.sum(a > b)
        b_w = np.sum(b > a)
        if a_w > b_w:
            return 0
        elif b_w > a_w:
            return 1
        else:
            return -1
            
    preds = data.apply(tally_predict, axis=1)
    valid_trials = preds != -1
    
    if valid_trials.sum() == 0:
        return 0.5
        
    match = (preds[valid_trials] == data.loc[valid_trials, 'response']).mean()
    return float(match)

```

**Observed (real) value:** 0.5855 (var=0.0012)
**Candidate (simulated) value:** 0.5360 (var=0.0250)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8520 (var=0.0107)
- pi_1: 0.3718 (var=0.0030)
- pi_3: 0.6198 (var=0.0427)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_target_trial(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        # Trial 1: A has the best cue, B has all the rest
        if a == [1, 0, 0, 0] and b == [0, 1, 1, 1]:
            return True
        # Trial 2: A has the second best cue, B has the rest
        if a == [0, 1, 0, 0] and b == [0, 0, 1, 1]:
            return True
        return False

    mask = data.apply(is_target_trial, axis=1)
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
    
    # Return the proportion of times Option A was chosen (response == 0)
    return float((subset['response'] == 0).mean())
```

**Observed (real) value:** 0.8287 (var=0.0147)
**Candidate (simulated) value:** 0.7100 (var=0.0472)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8350 (var=0.0172)
- pi_3: 0.4969 (var=0.0682)
- pi_2: 0.1481 (var=0.0127)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where A = [1, 0, 0, 0, 0] and B = [0, 1, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    if not is_trial_1.any():
        return 0.0
    return float(data.loc[is_trial_1, 'response'].mean())
```

**Observed (real) value:** 0.1811 (var=0.0113)
**Candidate (simulated) value:** 0.6800 (var=0.0855)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8126 (var=0.0166)
- pi_1: 0.1126 (var=0.0110)
- pi_2: 0.8663 (var=0.0120)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate's Threshold model fails significantly on Experiment 4. The metric for Exp 4 measures the proportion of times Option B (the compensatory choice with more cues) is chosen. The observed data is 0.1811, meaning participants overwhelmingly stuck with Option A (the TTB choice) ~82% of the time. However, the candidate predicts 0.6800, because the threshold for contradictory evidence (1 to 4) is easily exceeded when B has 4 cues and A has 1, causing the model to incorrectly override TTB.",
  "verdict": "regenerate",
  "rationale": "The deterministic Threshold approach misfires on Experiment 4 because the contradictory evidence difference triggers the override too often, contradicting the strong empirical TTB adherence. Since the arbiter also explicitly suggested a 'Strategy Selection' or 'Mixture' theory, switch to that approach instead. Implement a model where the decision-maker uses TTB with probability `w_ttb` and Tallying with probability `1 - w_ttb` (with `w_ttb` as a parameter in [0.0, 1.0]). This mixture will naturally capture the intermediate TTB/Tallying adherences in Exps 1 and 2, while preventing the catastrophic deterministic failures seen in Exp 4."
}
```

## Usage

```json
{
  "prompt_token_count": 5157,
  "candidates_token_count": 290,
  "total_token_count": 7276
}
```
