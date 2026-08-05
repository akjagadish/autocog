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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a Weighted Additive (WADD) theory. According to WADD, people evaluate options by multiplying each feature's cardinal rating by its cue validity, summing these products to form an overall expected value for each option, and choosing the option with the higher total value. This differs from TTB (which ignores all cues but the best discriminating one and ignores magnitudes) and Tallying (which ignores cue validities and cardinal magnitudes). Response noise should enter through a softmax over the options' total values.


## CANDIDATE THEORY
Weighted Additive (WADD) theory posits that individuals evaluate options by considering all available features, weighting each feature's cardinal value by its subjective validity or importance. The overall value of an option is the sum of these validity-weighted feature values. Choice is then made by comparing these overall values, with response noise modeled via a softmax function and a base lapse rate. This integrates both cue validity and cardinal magnitudes, distinguishing it from non-compensatory heuristics like Take The Best or unweighted tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    a, b = stim[0], stim[1]
    
    # Calculate weighted additive scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    return int(np.argmax(probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's instructions, this theory implements the Weighted Additive (WADD) model. Unlike Take The Best (which ignores lower-validity cues and cardinal magnitudes) or Tallying (which ignores cue validities and cardinal magnitudes), WADD computes an expected value for each option by multiplying the feature values by their corresponding cue validities and summing them. This compensatory mechanism allows a strong advantage on multiple lower-validity cues to outweigh a small disadvantage on a higher-validity cue, which is likely to capture the low TTB match in Experiment 1 and the relatively high Tallying match in Experiment 2.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.3214 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.3214.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[4, 1, 1, 1]  B=[3, 2, 2, 2]
  A=[2, 3, 3, 3]  B=[3, 1, 1, 1]
  A=[5, 5, 1, 1]  B=[5, 4, 2, 2]
  A=[3, 2, 4, 4]  B=[3, 3, 1, 1]
  A=[4, 2, 1, 1]  B=[3, 3, 2, 2]
  A=[1, 4, 4, 4]  B=[2, 2, 2, 2]
  A=[5, 4, 1, 1]  B=[5, 3, 2, 2]
  A=[2, 2, 4, 4]  B=[2, 3, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_choice = None
        # Features are implicitly ordered by validity (highest to lowest)
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        if ttb_choice is not None:
            if row['response'] == ttb_choice:
                ttb_match += 1
            total += 1
    return ttb_match / total if total > 0 else 0.5
```

**Observed (real) value:** 0.3308 (var=0.0607)
**Candidate (simulated) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0000)

### Experiment 2
**Design**
  A=[5, 2, 2, 2, 2]  B=[2, 5, 5, 5, 2]
  A=[2, 5, 5, 5, 2]  B=[5, 2, 2, 2, 2]
  A=[5, 5, 2, 2, 2]  B=[5, 2, 5, 5, 5]
  A=[4, 4, 4, 2, 2]  B=[4, 4, 2, 5, 5]
  A=[3, 2, 5, 5, 5]  B=[3, 5, 2, 2, 2]
  A=[4, 3, 3, 3, 3]  B=[3, 4, 4, 4, 4]
  A=[1, 5, 1, 1, 1]  B=[1, 1, 5, 5, 5]
  A=[5, 1, 5, 5, 5]  B=[1, 5, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    def tally_predicts(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(x > y for x, y in zip(a, b))
        b_wins = sum(y > x for x, y in zip(a, b))
        if a_wins > b_wins:
            return 0
        elif b_wins > a_wins:
            return 1
        return -1
    predictions = data.apply(tally_predicts, axis=1)
    return float(np.mean(data['response'] == predictions))
```

**Observed (real) value:** 0.7270 (var=0.0429)
**Candidate (simulated) value:** 1.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.0000)
- pi_1: 0.1250 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory correctly implements the WADD mechanism in the `predict` function, incorporating both cue validities and cardinal magnitudes. However, the simulation results show 0.0000 and 1.0000 matches with zero variance, indicating deterministic behavior. This occurs because the `policy` function simply returns `np.argmax(probabilities)`, completely discarding the softmax temperature (`beta`) and lapse rate (`epsilon`) computed in `predict`.",
  "verdict": "regenerate",
  "rationale": "The candidate has the right mechanism in `predict` but fails to simulate realistic human variance because the `policy` function is deterministic. Modify `policy` to sample from the probability distribution (e.g., `return int(np.random.choice(len(probabilities), p=probabilities))`) instead of using `argmax`. This will allow the softmax temperature and lapse rate to properly manifest as intermediate proportions in the simulated data, matching the ~0.33 and ~0.72 human match rates."
}
```

## Usage

```json
{
  "prompt_token_count": 3530,
  "candidates_token_count": 221,
  "total_token_count": 4670
}
```
