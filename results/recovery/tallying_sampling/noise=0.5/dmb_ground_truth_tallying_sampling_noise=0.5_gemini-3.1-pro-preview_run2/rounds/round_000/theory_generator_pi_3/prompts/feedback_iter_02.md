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

Propose a new theory to replace TTB that is compensatory but still heuristically simplified, such as the Tallying (Equal Weight) heuristic. In Tallying, decision-makers simply count the number of positive features for each option and choose the one with the higher count, completely ignoring the cue validities. This provides a strong, parameter-free compensatory alternative to WADD, testing whether subjects actually weight the features by their validities or simply add up the raw evidence.


## CANDIDATE THEORY
People make decisions between options using a Tallying (Equal Weight) heuristic. Instead of weighting features by their validities or relying on a single discriminating cue, decision-makers simply count the number of positive features for each option. They choose the option with the higher total count, treating all cues as equally important. When counts are tied, they guess. Response noise is modeled via a softmax over the tally scores and a uniform lapse rate. The choice is relatively noisy, preventing the strategy from becoming perfectly deterministic even when one option has a clear tally advantage.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: compute the sum of features for each option (equal weighting)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 1.5]
- epsilon: [0.0, 0.5]

`rationale`:
Following the critic's feedback, the Tallying mechanism remains exactly the same, but the range of the `beta` parameter has been further restricted from [0.1, 5.0] to [0.1, 1.5]. This will continue to soften the softmax choices, pulling the Experiment 2 prediction down closer to 0.67 and the Experiment 1 prediction up closer to 0.35, resulting in an even better fit across both datasets.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2298 -> ACCEPTED
- iter 2: loss=0.1779 -> ACCEPTED
- iter 3 (current candidate you are grading): loss=0.0363 -> ACCEPTED
Running-best (last accepted) base: iter 3 at loss=0.0363.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    weights = np.array([1000, 100, 10, 1])
    score_diff = diff @ weights
    ttb_preds = np.where(score_diff > 0, 0, 1)
    matches = (data['response'].values == ttb_preds)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3504 (var=0.0026)
**Candidate trajectory (this loop):**
  - iter 1: 0.2406 (var=0.0071) (Δ vs real -0.1098)
  - iter 2: 0.2812 (var=0.0068) (Δ vs real -0.0692)
  - iter 3 (current): 0.3654 (var=0.0070) (Δ vs real +0.0150)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8598 (var=0.0115)
- pi_2: 0.4281 (var=0.0279)

### Experiment 2
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    mask = sum_b > sum_a
    if mask.sum() == 0:
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6741 (var=0.0053)
**Candidate trajectory (this loop):**
  - iter 1: 0.8841 (var=0.0087) (Δ vs real +0.2100)
  - iter 2: 0.8393 (var=0.0116) (Δ vs real +0.1652)
  - iter 3 (current): 0.6981 (var=0.0114) (Δ vs real +0.0241)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6756 (var=0.0280)
- pi_1: 0.1400 (var=0.0055)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory implements the Tallying (Equal Weight) heuristic. In Experiment 2, it predicts an 88.4% preference for the option with more positive features, overshooting the observed 67.4%. In Experiment 1, it predicts a 24.1% match to TTB, undershooting the observed 35.0%. Interestingly, in Experiment 1, the TTB-winner is often the Tally-loser. If the model were less deterministic (i.e., choosing the Tally-winner ~67% of the time instead of ~88%), it would perfectly hit the Experiment 2 target (67%) and simultaneously hit the Experiment 1 target (100% - 67% = 33%, very close to the observed 35%).
Rationale: The Tallying mechanism is correct and captures the qualitative direction perfectly, but the current parameter ranges allow it to become too deterministic, causing it to overshoot the effect size. Narrow the `beta` parameter range to something like [0.1, 5.0] (down from 20.0). This will force a softer softmax, naturally pulling the Experiment 2 prediction down to ~0.67 and the Experiment 1 prediction up to ~0.33, resulting in an excellent fit across both datasets while staying strictly within the prescribed Tallying family.

**Outcome of your advice:** iter 1 candidate loss=0.2298 -> iter 2 candidate loss=0.1779 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent)
Verdict: regenerate
Interpretation: The candidate theory successfully implemented the Tallying heuristic and the previous suggestion to narrow the `beta` range to [0.1, 5.0] was accepted, improving the overall fit. However, the model is still too deterministic. In Experiment 2, the model predicts an 83.9% preference for the option with more positive features, which still overshoots the observed 67.4%. In Experiment 1, it predicts a 28.1% match to TTB, which still undershoots the observed 35.0%. Since the previous adjustment in this direction improved the loss, we should continue softening the model's choices.
Rationale: The Tallying mechanism is correct, but the parameter ranges still allow for too much determinism. Further restrict the `beta` parameter range to [0.1, 1.5] or [0.1, 2.0]. This will continue the successful trajectory of softening the softmax, pulling the Experiment 2 prediction down closer to 0.67 and the Experiment 1 prediction up closer to 0.35.

**Outcome of your advice:** iter 2 candidate loss=0.1779 -> CURRENT candidate loss=0.0363 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory successfully implements the Tallying heuristic as prescribed by the arbiter. Following the previous advice to further restrict the `beta` range to [0.1, 1.5], the model's determinism has been perfectly calibrated. In Experiment 1, the model predicts a 36.5% match to TTB, which is extremely close to the observed 35.0%. In Experiment 2, it predicts a 69.8% preference for the option with more positive features, closely matching the observed 67.4%. The aggregate loss has dropped to an excellent 0.0363.",
  "verdict": "continue",
  "rationale": "The Tallying mechanism with the current parameter ranges captures the empirical phenomena across both experiments exceptionally well. The choice to soften the softmax via a restricted beta range has successfully brought the model's predictions into alignment with human data. The candidate is good to go."
}
```

## Usage

```json
{
  "prompt_token_count": 4360,
  "candidates_token_count": 210,
  "total_token_count": 4768
}
```
