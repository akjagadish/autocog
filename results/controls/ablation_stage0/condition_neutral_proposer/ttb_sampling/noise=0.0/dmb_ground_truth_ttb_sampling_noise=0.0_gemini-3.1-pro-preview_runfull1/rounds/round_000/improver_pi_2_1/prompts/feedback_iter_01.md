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
- The recommendation below acts on THEORY 2 (= `pi_2`).

Revise the WADD model's predict function and parameter ranges. Instead of treating 'validities' and 'weights' as completely independent or random free parameters that multiply together, the model should ideally extract the objective validities directly from the experimental state (if available) or parameterize the subjective weights to be strictly monotonic transformations of the objective validities. Removing redundant weight parameters and ensuring the model systematically assigns higher weights to the high-validity cues will allow WADD to correctly predict the high observed alignment (~0.86) rather than predicting random guessing (~0.50).


## CANDIDATE THEORY
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Use the log-odds of objective validities as the subjective weights
    validities = np.asarray(parameters["validities"], dtype=float)
    validities = np.clip(validities, 1e-5, 1.0 - 1e-5)  # Prevent division by zero or log(0)
    weights = np.log(validities / (1.0 - validities))
    
    # Weighted sum per option (dot product with subjective weights)
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.2]
- validities: validities

`rationale`:
Following the critic's advice, we apply a log-odds transformation to the objective validities to compute the subjective weights. This strictly monotonic transformation stretches out the weights for high-validity cues, increasing the difference in weighted scores between options. We also reduce the upper bound of the lapse rate (epsilon) to 0.2 to prevent artificial capping of the model's accuracy. These changes allow the model to confidently select the WADD-aligned option and match the high alignment (~0.86) observed in the data.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.4194 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0759 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.0759.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    a_wadd = a_mat @ validities
    b_wadd = b_mat @ validities
    
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    wadd_prefers_a = a_wadd > b_wadd
    wadd_prefers_b = b_wadd > a_wadd
    
    disagree = (tally_prefers_a & wadd_prefers_b) | (tally_prefers_b & wadd_prefers_a)
    
    if not np.any(disagree):
        return 0.5
        
    tally_choice = np.where(tally_prefers_a, 0, 1)
    
    match = (data['response'].values[disagree] == tally_choice[disagree])
    return float(np.mean(match))
```

**Observed (real) value:** 0.1067 (var=0.0120)
**Candidate trajectory (this loop):**
  - iter 1: 0.2642 (var=0.0228) (Δ vs real +0.1575)
  - iter 2 (current): 0.0625 (var=0.0048) (Δ vs real -0.0442)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8575 (var=0.0138)
- pi_2: 0.4208 (var=0.0840)

### Experiment 2
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_vals = np.stack(data['option_a_ratings'].values)
    b_vals = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where Tallying and WADD make strictly opposing predictions.
    # Trial 1: A=[0,0,1,1,1], B=[1,1,0,0,0] -> Tallying prefers A, WADD prefers B
    is_trial_1 = (a_vals[:, 0] == 0) & (a_vals[:, 2] == 1) & (b_vals[:, 0] == 1) & (b_vals[:, 2] == 0)
    
    # Trial 2: A=[1,1,0,0,0], B=[0,0,1,1,1] -> Tallying prefers B, WADD prefers A
    is_trial_2 = (a_vals[:, 0] == 1) & (a_vals[:, 2] == 0) & (b_vals[:, 0] == 0) & (b_vals[:, 2] == 1)
    
    mask = is_trial_1 | is_trial_2
    if not mask.any():
        return 0.5
        
    responses = data['response'].values
    wadd_aligned = (is_trial_1 & (responses == 1)) | (is_trial_2 & (responses == 0))
    
    return float(wadd_aligned[mask].mean())
```

**Observed (real) value:** 0.8649 (var=0.0063)
**Candidate trajectory (this loop):**
  - iter 1: 0.4164 (var=0.0032) (Δ vs real -0.4484)
  - iter 2 (current): 0.9409 (var=0.0039) (Δ vs real +0.0760)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5004 (var=0.0514)
- pi_1: 0.1520 (var=0.0057)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate model uses the raw objective validities as weights. Because the raw validities (e.g., 0.95, 0.9 vs 0.6, 0.55, 0.5) sum to relatively similar values for competing options (e.g., 1.85 vs 1.65), the difference in weighted scores is very small (0.20). Even with the maximum beta of 20, the softmax does not strongly enough favor the WADD choice, leading to simulated WADD-alignment values (~0.41 in Exp 2) that are far too low compared to human data (~0.86).
Rationale: To fix the under-prediction of WADD alignment, apply a strictly monotonic transformation to the objective validities to compute the subjective weights. Specifically, use the log-odds transformation: `weights = np.log(validities / (1.0 - validities))`. This standard transformation stretches out probabilities near 1.0, giving much higher relative weight to the most valid cues and allowing the model to confidently choose the WADD-aligned option. Also, consider reducing the upper bound of epsilon (e.g., [0.0, 0.2]) so that lapse noise doesn't artificially cap the model's accuracy.

**Outcome of your advice:** iter 1 candidate loss=0.4194 -> CURRENT candidate loss=0.0759 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate model successfully implemented the log-odds transformation of objective validities, which correctly stretched the subjective weights for high-validity cues. This resolved the under-prediction of WADD alignment from the previous iteration. The model now predicts a WADD alignment of 0.94 in Experiment 2 (up from 0.41, much closer to the observed 0.86) and a tally-alignment of 0.06 in Experiment 1 (down from 0.26, very close to the observed 0.10). The aggregate loss has improved dramatically to 0.0759, indicating a very strong fit to the human data.",
  "verdict": "continue",
  "rationale": "The current model accurately captures the empirical pattern across both experiments by using the log-odds transformation. The fit is excellent and the mechanism works exactly as intended. No further revisions are necessary."
}
```

## Usage

```json
{
  "prompt_token_count": 4766,
  "candidates_token_count": 203,
  "total_token_count": 5492
}
```
