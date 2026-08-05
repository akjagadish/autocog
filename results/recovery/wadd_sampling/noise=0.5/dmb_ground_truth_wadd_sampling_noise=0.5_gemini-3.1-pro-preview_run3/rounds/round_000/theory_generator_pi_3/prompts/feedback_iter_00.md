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
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a Weighted Additive (WADD) theory. Unlike TTB, which ignores all lower-validity cues, and Tallying, which ignores differences in cue validities, WADD integrates all available information by multiplying each feature's value by its subjective validity (or weight) and summing these products. This allows multiple lower-validity cues to jointly outweigh a single high-validity cue, naturally predicting the intermediate choice proportions observed in the data.


## CANDIDATE THEORY
Weighted Additive (WADD) Theory: People compare two options by considering all available features, weighting each feature by its subjective validity (informativeness), and summing these weighted values to form an overall valuation for each option. Unlike Take The Best (which ignores lower-validity cues) and Tallying (which ignores differences in cue validities), WADD fully integrates all evidence. This compensatory mechanism allows multiple lower-validity cues to jointly outweigh a single high-validity cue, predicting the intermediate choice proportions observed in the data. Response noise is modeled via a softmax over the WADD scores and an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Convert validities to subjective weights.
    # Subtracting 0.5 ensures that a cue with 0.5 validity (random chance) has 0 weight.
    weights = val - 0.5
    
    a, b = stim[0], stim[1]
    
    # Calculate Weighted Additive scores for both options
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
The arbiter diagnosed that neither Take The Best (which relies solely on the best discriminating cue) nor Tallying (which treats all cues equally) fully captures the decision-making process, as evidenced by the intermediate match proportions in the data. The Weighted Additive (WADD) model directly addresses this by integrating all cues while scaling their contribution by their subjective validity. By using `validities - 0.5` as weights, the model correctly treats chance-level cues as uninformative, while allowing multiple weaker cues to compensate for a single stronger cue. This compensatory mechanism is expected to predict the intermediate choice patterns across both experiments.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.1102 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.1102.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    non_zero = diff != 0
    
    first_diff_idx = np.argmax(non_zero, axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    ttb_choice = np.where(first_diff_val > 0, 0, 1)
    has_diff = np.any(non_zero, axis=1)
    
    if not np.any(has_diff):
        return 0.5
        
    match = (data['response'].values[has_diff] == ttb_choice[has_diff])
    return float(np.mean(match))
```

**Observed (real) value:** 0.4083 (var=0.0075)
**Candidate (simulated) value:** 0.3081 (var=0.0077)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8248 (var=0.0119)
- pi_2: 0.1250 (var=0.0061)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = (a_mat > b_mat).sum(axis=1)
    b_wins = (b_mat > a_mat).sum(axis=1)
    
    preds = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    valid = preds != -1
    if not valid.any():
        return 0.5
        
    matches = (data['response'].values[valid] == preds[valid])
    return float(matches.mean())
```

**Observed (real) value:** 0.6125 (var=0.0092)
**Candidate (simulated) value:** 0.6690 (var=0.0051)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8490 (var=0.0088)
- pi_1: 0.1173 (var=0.0077)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate implements a Weighted Additive (WADD) model using `validities - 0.5` as the cue weights. This yields an aggregate loss of 0.1102, which is a strong improvement over both pure TTB and pure Tallying. However, the model slightly underpredicts the TTB match rate in Experiment 1 (0.3081 simulated vs 0.4083 real) and slightly overpredicts the Tallying match rate in Experiment 2 (0.6690 simulated vs 0.6125 real). This suggests that the highest-validity cues are not being weighted heavily enough relative to the lower-validity cues.",
  "verdict": "regenerate",
  "rationale": "To increase the relative influence of the best cues and better match the intermediate observed proportions, transform the validities into log-odds weights: `weights = np.log(val / (1.0 - val))` (ensuring numerical stability by clipping validities away from 0 and 1 if necessary). This is the normative Bayesian way to integrate independent probabilistic cues and will spread out the weights non-linearly, giving higher-validity cues more power to overcome multiple weaker cues, which should push the Experiment 1 TTB match rate up towards 0.40."
}
```

## Usage

```json
{
  "prompt_token_count": 3702,
  "candidates_token_count": 292,
  "total_token_count": 4571
}
```
