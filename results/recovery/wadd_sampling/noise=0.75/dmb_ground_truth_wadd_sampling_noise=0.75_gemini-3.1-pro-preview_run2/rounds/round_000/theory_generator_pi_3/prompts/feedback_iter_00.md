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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose a Weighted Additive (WADD) theory. In WADD, decision-makers compute a subjective value for each option by taking a weighted sum of its features, where the weights are proportional to the provided cue validities. The probability of choosing an option is then a softmax function of the difference in these weighted sums. This theory fundamentally differs from TTB (which only considers the single best discriminating cue) and Tallying (which treats all cues equally regardless of validity). A WADD model can naturally explain the intermediate behavior on conflict trials, as the high validity of a top cue can be offset by the combined weight of several lower-validity cues, leading to more graded, compensatory choices.


## CANDIDATE THEORY
People make decisions by computing a subjective value for each option through a Weighted Additive (WADD) process. They take a weighted sum of the features for each option, where the weight of each feature is directly proportional to its provided cue validity. The probability of choosing an option is then determined by a softmax over the options' subjective values, combined with a lapse rate for random errors. This compensatory mechanism allows multiple lower-validity cues to jointly outweigh a single high-validity cue, naturally predicting the graded, intermediate behavior observed on conflict trials compared to non-compensatory heuristics like Take The Best or Tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted additive values for both options
    v_a = np.sum(a * val)
    v_b = np.sum(b * val)
    scores = np.array([v_a, v_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
The arbiter identified that human behavior on conflict trials is neither purely non-compensatory (like Take The Best) nor purely unweighted (like Tallying). WADD computes a weighted sum using the exact cue validities, allowing the model to naturally capture the intermediate choice proportions observed in the experiments. By evaluating both the magnitude of the cue validities and all available cues simultaneously, WADD gracefully balances between the extreme predictions of TTB and Tallying.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.4311 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.4311.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction (assumes features are sorted by descending validity)
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Tallying prediction
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a > sum_b:
            tally_pred = 0
        elif sum_b > sum_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        # Focus strictly on conflict trials where the theories make opposite predictions
        if ttb_pred is not None and tally_pred is not None and ttb_pred != tally_pred:
            matches.append(1 if row['response'] == ttb_pred else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))

```

**Observed (real) value:** 0.4775 (var=0.0054)
**Candidate (simulated) value:** 0.1613 (var=0.0115)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8421 (var=0.0086)
- pi_2: 0.1396 (var=0.0094)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_match = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        tally_pred = 0 if a_wins > b_wins else (1 if b_wins > a_wins else None)
        
        val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
        cue_order = np.argsort(-val)
        ttb_pred = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_pred = 0
                break
            elif b[j] > a[j]:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            if row['response'] == tally_pred:
                tally_match += 1
            total += 1
            
    return tally_match / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5340 (var=0.0037)
**Candidate (simulated) value:** 0.8257 (var=0.0119)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8454 (var=0.0089)
- pi_1: 0.1371 (var=0.0093)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate implements the Weighted Additive (WADD) theory as prescribed, using the raw cue validities as weights. However, the simulation results on conflict trials (Experiment 1: 0.16 vs 0.48; Experiment 2: 0.83 vs 0.53) show that the model's predictions are far too extreme compared to human behavior, which hovers around 0.5 (indicating a near-even split between TTB-like and Tallying-like choices). Using raw probabilities as additive weights often over- or under-values the relative evidence of cues. A mathematically principled approach in WADD is to transform validities (which are probabilities) into log-odds to represent additive evidence.",
  "verdict": "regenerate",
  "rationale": "The current WADD model produces overly extreme choice probabilities. To fix this while staying within the prescribed WADD family, transform the raw validities into log-odds weights: `w = np.log(val / (1.0 - val))` before computing the weighted sum. Additionally, shift the `beta` parameter range to lower values (e.g., `[0.01, 5.0]`) to allow the softmax to produce softer, more graded predictions closer to the ~0.5 empirical values observed on conflict trials."
}
```

## Usage

```json
{
  "prompt_token_count": 3890,
  "candidates_token_count": 291,
  "total_token_count": 4896
}
```
