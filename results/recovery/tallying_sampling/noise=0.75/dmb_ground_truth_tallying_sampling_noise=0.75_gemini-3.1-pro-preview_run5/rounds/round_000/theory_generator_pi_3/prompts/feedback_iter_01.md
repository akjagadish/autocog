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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace Theory 1 (Take The Best) with a 'Tallying' (Equal-Weight) heuristic. Tallying posits that people compare options by simply counting the number of positive cues (or cues favoring each option) and choosing the option with the higher tally, ignoring the differential validities of the cues. This provides a boundedly rational, compensatory alternative to WADD that can explain why subjects often favor options with a higher quantity of lower-validity cues, without assuming complex weighted integration.


## CANDIDATE THEORY
Tallying (Equal-Weight) Heuristic: People evaluate options by simply counting the number of positive features (or cues favoring each option) and choosing the option with the higher total count. This compensatory strategy ignores the differential validities or subjective importance of different cues, treating all pieces of evidence equally. The choice probability is determined by a softmax over the total feature tallies for each option, combined with a uniform lapse rate. Crucially, the softmax temperature is constrained to produce softer choice probabilities, reflecting that humans do not apply the tallying rule completely deterministically.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tally the number of positive cues for each option
    score_a = np.sum(stim[0])
    score_b = np.sum(stim[1])
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

`parameters`:
- beta: [0.0, 3.0]
- epsilon: [0.0, 0.5]

`rationale`:
Following the arbiter's advice, I reduced the upper bound of the `beta` parameter from 20.0 to 3.0. This makes the softmax over feature tallies less deterministic, leading to softer choice probabilities. This minimal change directly addresses the issue where the previous model over-predicted adherence to the tallying rule (producing metrics around 0.15 instead of the observed ~0.41), bringing the simulated behavior closer to the human data while preserving the core mechanism.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3767 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.2878 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.2878.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f1 = np.array([x[0] for x in data['option_a_ratings']])
    b_f1 = np.array([x[0] for x in data['option_b_ratings']])
    responses = data['response'].values
    
    favors_A = a_f1 > b_f1
    favors_B = b_f1 > a_f1
    
    aligned = (favors_A & (responses == 0)) | (favors_B & (responses == 1))
    return float(np.mean(aligned))
```

**Observed (real) value:** 0.4171 (var=0.0031)
**Candidate trajectory (this loop):**
  - iter 1: 0.1542 (var=0.0099) (Δ vs real -0.2629)
  - iter 2 (current): 0.2675 (var=0.0141) (Δ vs real -0.1496)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8419 (var=0.0118)
- pi_2: 0.3740 (var=0.0225)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_prediction(a, b):
        # Cues are already in descending order of validity
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            elif val_b > val_a:
                return 1
        return -1
        
    ttb_preds = np.array([get_ttb_prediction(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])])
    valid_mask = (ttb_preds != -1)
    
    if not np.any(valid_mask):
        return 0.5
        
    responses = data['response'].values
    matches = (responses[valid_mask] == ttb_preds[valid_mask])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4083 (var=0.0021)
**Candidate trajectory (this loop):**
  - iter 1: 0.1471 (var=0.0090) (Δ vs real -0.2612)
  - iter 2 (current): 0.2046 (var=0.0149) (Δ vs real -0.2037)
**Other theories' values on this metric (for reference):**
- pi_2: 0.2510 (var=0.0145)
- pi_1: 0.8510 (var=0.0082)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The Tallying heuristic correctly captures that subjects often choose the option with more positive cues even if it lacks the highest-validity cue (which is why the metric, representing alignment with the most valid cue / TTB, is below 0.5). However, the simulated values (0.1542 and 0.1471) are much lower than the observed human values (0.4171 and 0.4083). This indicates that the current model is far too deterministic in its tallying choices, choosing the option with the higher tally almost 85% of the time, whereas humans only do so about 59% of the time.
Rationale: The candidate was accepted because it moves in the right direction compared to the previous TTB model, but it overshoots by making the tallying choice too deterministic. To fix this while staying strictly within the prescribed Tallying family, constrain the `beta` parameter range to smaller values (e.g., [0.0, 2.0] or [0.0, 3.0]) instead of [0.1, 20.0]. This will force the softmax to produce softer choice probabilities, bringing the simulated metrics up from ~0.15 closer to the observed ~0.41.

**Outcome of your advice:** iter 1 candidate loss=0.3767 -> CURRENT candidate loss=0.2878 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The Tallying heuristic model correctly captures that subjects often choose the option with the higher count of positive cues, which is why the metric (representing alignment with the most valid cue, or TTB) is below 0.5. The recent adjustment to the `beta` parameter range successfully brought the metrics closer to the human data (loss improved from 0.3767 to 0.2878). However, the simulated values (0.2675 and 0.2046) are still notably lower than the observed human values (~0.41). This indicates that the model is still slightly too deterministic in its tallying choices (choosing the tallying option ~75-80% of the time, whereas humans choose it ~59% of the time).",
  "verdict": "regenerate",
  "rationale": "The previous advice to lower the `beta` upper bound was accepted and improved the fit, but the model still overshoots the tallying choice rate. To further soften the choice probabilities and bring the TTB alignment metrics up to the observed ~0.41, further constrain the `beta` parameter range. Try reducing the upper bound of `beta` to 1.0 or 1.5 (e.g., [0.0, 1.0]). This will force the softmax to produce even softer probabilities, matching the ~59% human adherence to the tallying rule."
}
```

## Usage

```json
{
  "prompt_token_count": 4100,
  "candidates_token_count": 316,
  "total_token_count": 5114
}
```
