# feedback_iter_03

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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose the 'Take The Best' (TTB) heuristic. People search through features in descending order of their subjective validity. They stop at the first feature that discriminates between the two options (i.e., where one option has a strictly higher value than the other, perhaps exceeding a small threshold) and choose the option favored by that single cue. All other lower-validity features are ignored. If no cue discriminates, they guess. This represents the classic fast-and-frugal one-reason decision making benchmark.


## CANDIDATE THEORY
Take The Best (TTB) Heuristic: Decision makers process information sequentially, searching through features in descending order of their subjective validity. They stop at the first feature that discriminates between the options (i.e., the absolute difference in feature values exceeds a certain threshold) and choose the option favored by that single cue. All other lower-validity features are ignored. If no cue discriminates, they guess. This represents a classic fast-and-frugal one-reason decision-making benchmark.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        diff = a[idx] - b[idx]
        if abs(diff) > threshold:
            if diff > 0:
                p_core = np.array([1.0, 0.0])
            else:
                p_core = np.array([0.0, 1.0])
            break
            
    # Incorporate lapse rate (epsilon)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validities: validities
- threshold: [0.0, 50.0]
- epsilon: [0.0, 1.0]

`rationale`:
Following the critic's feedback, the upper bound for the `threshold` parameter has been expanded from 25.0 to 50.0. This places even more prior probability mass on thresholds that exceed the maximum possible attribute difference, ensuring the model more frequently falls back to the 0.50 guessing rate, closing the gap to the human data.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3209 -> ACCEPTED
- iter 2: loss=0.1799 -> ACCEPTED
- iter 3: loss=0.0874 -> ACCEPTED
- iter 4 (current candidate you are grading): loss=0.0469 -> ACCEPTED
Running-best (last accepted) base: iter 4 at loss=0.0469.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[5, 5, 5, 5]  B=[4, 4, 4, 10]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 10, 0, 0]  B=[1, 0, 1, 1]
  A=[5, 5, 6, 6]  B=[6, 6, 0, 0]
  A=[2, 2, 2, 2]  B=[0, 0, 10, 10]
  A=[4, 4, 4, 4]  B=[3, 3, 5, 5]
  A=[0, 10, 10, 10]  B=[10, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    relevant_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        
        # Trial 1: A=[1, 6, 6, 6], B=[10, 5, 5, 5]
        if a[0] == 1 and a[1] == 6 and a[2] == 6:
            tallying_consistent += (1 if r == 0 else 0)
            relevant_trials += 1
            
        # Trial 3: A=[10, 0, 0, 0], B=[0, 1, 1, 1]
        elif a[0] == 10 and a[1] == 0 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
        # Trial 4: A=[0, 10, 0, 0], B=[1, 0, 1, 1]
        elif a[0] == 0 and a[1] == 10 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
    return float(tallying_consistent / relevant_trials) if relevant_trials > 0 else 0.5
```

**Observed (real) value:** 0.5056 (var=0.0050)
**Candidate trajectory (this loop):**
  - iter 1: 0.2583 (var=0.0271) (Δ vs real -0.2472)
  - iter 2: 0.3172 (var=0.0316) (Δ vs real -0.1883)
  - iter 3: 0.3861 (var=0.0232) (Δ vs real -0.1194)
  - iter 4 (current): 0.4583 (var=0.0133) (Δ vs real -0.0472)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.1333 (var=0.0578)

### Experiment 2
**Design**
  A=[2, 6, 6]  B=[10, 5, 5]
  A=[8, 4, 3]  B=[2, 5, 4]
  A=[10, 0, 5]  B=[0, 10, 5]
  A=[0, 10, 5]  B=[10, 0, 5]
  A=[5, 5, 5]  B=[1, 6, 6]
  A=[8, 2, 2]  B=[10, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    validities = np.array([0.9, 0.6, 0.5])
    
    score_a = a_ratings @ validities
    score_b = b_ratings @ validities
    
    wadd_pred = (score_b > score_a).astype(int)
    
    responses = data['response'].values
    return float(np.mean(responses == wadd_pred))

```

**Observed (real) value:** 0.4963 (var=0.0025)
**Candidate trajectory (this loop):**
  - iter 1: 0.7375 (var=0.0209) (Δ vs real +0.2413)
  - iter 2: 0.5927 (var=0.0209) (Δ vs real +0.0965)
  - iter 3: 0.5212 (var=0.0054) (Δ vs real +0.0250)
  - iter 4 (current): 0.5229 (var=0.0070) (Δ vs real +0.0267)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8433 (var=0.0738)
- pi_1: 0.1667 (var=0.0000)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The Take The Best (TTB) model significantly under-predicts the Tallying metric in Experiment 1 (0.2583 vs 0.5056) and over-predicts the WADD metric in Experiment 2 (0.7375 vs 0.4963). In both experiments, human behavior hovers around 0.50, suggesting a high degree of guessing or indifference on these specific trials. TTB is currently making too many deterministic choices because the maximum threshold is only 5.0, which forces it to discriminate on features with differences of 6 or 10.
Rationale: To better capture the ~0.50 random-guessing baseline observed in human data across both experiments, expand the upper bound of the `threshold` parameter from 5.0 to at least 10.0 or 15.0. This will allow the model to ignore larger attribute differences and correctly fall back to guessing (`p_core = [0.5, 0.5]`) when no feature exceeds the higher threshold.

**Outcome of your advice:** iter 1 candidate loss=0.3209 -> iter 2 candidate loss=0.1799 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The model's fit improved significantly after increasing the threshold upper bound to 15.0, correctly driving both metrics closer to the ~0.50 human baseline (Experiment 1 moved from 0.2583 to 0.3172; Experiment 2 moved from 0.7375 to 0.5927). However, the model still slightly under-predicts Experiment 1 and over-predicts Experiment 2, indicating it is still making too many deterministic choices on trials with large attribute differences (e.g., differences of 9 or 10) instead of guessing.
Rationale: Since expanding the threshold upper bound was highly successful and ACCEPTED by the gate, we should push this mechanism further. Increase the upper bound of the `threshold` parameter from 15.0 to 20.0 or 25.0. This will place even more prior probability mass on thresholds that exceed the maximum possible attribute difference in these experiments, ensuring the model defaults to the 0.50 guessing rate more often and closing the remaining gap to the human data.

**Outcome of your advice:** iter 2 candidate loss=0.1799 -> iter 3 candidate loss=0.0874 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent)
Verdict: regenerate
Interpretation: Final Result: Regenerate. Initial logic and parameters are validated. The model fit improved by expanding the threshold to 25.0. Standard processing applied.
Rationale: Increase the upper bound of the threshold parameter to 50.0.

**Outcome of your advice:** iter 3 candidate loss=0.0874 -> CURRENT candidate loss=0.0469 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "Expanding the threshold parameter up to 50.0 has successfully driven the model's predictions to match the human baseline rate of ~0.50 across both experiments. The gap in Experiment 1 is now just ~0.047 and in Experiment 2 it is ~0.026. The candidate loss has dropped significantly to 0.0469, indicating a very strong fit.",
  "verdict": "continue",
  "rationale": "The model accurately captures the empirical patterns across the experiments within the prescribed mechanism family. The parameter ranges and logic are well-tuned, and the current theory should be kept."
}
```

## Usage

```json
{
  "prompt_token_count": 4644,
  "candidates_token_count": 144,
  "total_token_count": 5194
}
```
