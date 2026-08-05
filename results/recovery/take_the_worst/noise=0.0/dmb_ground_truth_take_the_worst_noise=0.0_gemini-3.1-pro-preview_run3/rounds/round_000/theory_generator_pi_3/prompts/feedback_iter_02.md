# feedback_iter_02

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

Replace the degenerate Take The Best theory with a Weighted Additive (WADD) theory. In WADD, participants evaluate options by computing a weighted sum of the features, where each feature is weighted by its subjective validity (or the validity provided in the instructions). The option with the highest total score is chosen, with response noise applied via a softmax function. This theory differs from Tallying because it does not discard the validities (allowing a highly valid cue to potentially outweigh multiple weakly valid cues), and it differs from TTB because it is fully compensatory (integrating all available information rather than stopping at the first discriminating cue).


## CANDIDATE THEORY
Weighted Additive (WADD) Decision Making

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate the weighted sum of features for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores with max-subtraction for numerical stability
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
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.01, 2.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
We maintain the Weighted Additive (WADD) mechanism but further tighten the upper bound on the softmax inverse temperature (beta) from 5.0 to 2.0, as suggested by the critic. This introduces even greater choice stochasticity (softer maximization) when option scores are close, preventing the model from making overly deterministic choices and bringing the Tallying match in Experiment 2 closer to the empirical value of ~0.61.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2576 -> ACCEPTED
- iter 2: loss=0.1875 -> ACCEPTED
- iter 3 (current candidate you are grading): loss=0.1525 -> ACCEPTED
Running-best (last accepted) base: iter 3 at loss=0.1525.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Find the highest validity feature where A and B differ.
    # Since features are ordered by descending validity, this is the first non-zero difference.
    diff = a_mat - b_mat
    first_diff_idx = np.argmax(np.abs(diff), axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    # TTB predicts A (0) if A > B on the discriminating cue, else B (1)
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # Return the proportion of choices that align with TTB's prediction
    return float(np.mean(ttb_pred == resp))
```

**Observed (real) value:** 0.2512 (var=0.0077)
**Candidate trajectory (this loop):**
  - iter 1: 0.2994 (var=0.0037) (Δ vs real +0.0481)
  - iter 2: 0.3052 (var=0.0062) (Δ vs real +0.0540)
  - iter 3 (current): 0.3856 (var=0.0073) (Δ vs real +0.1344)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8650 (var=0.0060)
- pi_2: 0.2373 (var=0.0059)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid_trials = a_wins != b_wins
    if not np.any(valid_trials):
        return 0.5
        
    tally_pred = (b_wins > a_wins).astype(int)
    responses = data['response'].values
    
    match = (tally_pred[valid_trials] == responses[valid_trials])
    return float(np.mean(match))
```

**Observed (real) value:** 0.6089 (var=0.0019)
**Candidate trajectory (this loop):**
  - iter 1: 0.8578 (var=0.0098) (Δ vs real +0.2489)
  - iter 2: 0.7686 (var=0.0085) (Δ vs real +0.1597)
  - iter 3 (current): 0.6592 (var=0.0098) (Δ vs real +0.0503)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8428 (var=0.0102)
- pi_1: 0.2903 (var=0.0064)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The Weighted Additive (WADD) candidate successfully captures the low TTB match in Experiment 1 (observed 0.251, candidate 0.299). However, it overestimates the Tallying match in Experiment 2 (observed 0.609, candidate 0.858). This suggests the model's choices are too deterministic when the weighted sum favors the option with more features, likely because the upper bound of the softmax inverse temperature (`beta`) is too high.
Rationale: The candidate is in the correct mechanism family (WADD) and significantly improves upon the baseline, but the overestimation in Experiment 2 indicates insufficient choice stochasticity. Please regenerate with a tightened range for `beta` (e.g., [0.01, 5.0] instead of [0.1, 20.0]) to encourage softer maximization and bring the Tallying match closer to the empirical ~0.61.

**Outcome of your advice:** iter 1 candidate loss=0.2576 -> iter 2 candidate loss=0.1875 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent)
Verdict: regenerate
Interpretation: The Weighted Additive (WADD) candidate continues to perform well in capturing the overall mechanism, and the previous reduction in the `beta` upper bound successfully reduced the Tallying match in Experiment 2 from 0.858 to 0.769. However, this is still significantly higher than the empirical value of 0.609. Experiment 1's TTB match remains reasonably close to the observed data (0.305 vs 0.251). The model still exhibits slightly too much determinism when evaluating options with similar weighted scores.
Rationale: The previous advice to tighten the `beta` range was accepted and improved the overall fit by introducing more choice stochasticity, but the model still over-predicts the Tallying match in Experiment 2. Please continue in this direction by further tightening the `beta` range (e.g., [0.01, 2.0]) or slightly increasing the `epsilon` range to ensure even softer maximization and bring the Experiment 2 metric closer to ~0.61.

**Outcome of your advice:** iter 2 candidate loss=0.1875 -> CURRENT candidate loss=0.1525 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The Weighted Additive (WADD) candidate successfully reduced the Tallying match in Experiment 2 to 0.659 (very close to the empirical 0.609) by tightening the beta range. However, this increased choice stochasticity pushed the TTB match in Experiment 1 further towards 0.5 (from 0.305 up to 0.386, compared to the empirical 0.251). Because the empirical TTB match in Experiment 1 is below 0.5, the model must make deterministic choices *against* the TTB prediction to fit it well. Too much noise pushes the prediction toward 0.5, worsening the fit for Experiment 1.",
  "verdict": "regenerate",
  "rationale": "The previous reduction in the `beta` upper bound successfully improved Experiment 2 but traded off against Experiment 1's fit by making choices too noisy. To allow the parameter estimation to find the optimal balance between these two opposing forces, please slightly widen the `beta` range back up to [0.01, 3.0] or [0.1, 3.0]. This will give the model enough flexibility to be deterministic enough for Experiment 1 while maintaining enough stochasticity for Experiment 2."
}
```

## Usage

```json
{
  "prompt_token_count": 4422,
  "candidates_token_count": 282,
  "total_token_count": 5435
}
```
