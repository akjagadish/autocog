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

Propose a Weighted Additive (WADD) theory. Unlike TTB (which only uses the single best discriminating cue) and Tallying (which weights all cues equally regardless of validity), WADD computes a total value for each option by summing its features weighted by their respective validities. The choice is then made via a softmax over these weighted sums. This compensatory mechanism naturally explains why options with one highly valid cue and options with many lower-validity cues are perceived as roughly equal in value, producing the observed ~0.5 choice proportions in these adversarial designs.


## CANDIDATE THEORY
Weighted Additive (WADD) Theory with Subjective Validity Scaling: People compare options by computing a comprehensive value for each option, integrating all available cues by weighting each feature. However, instead of using raw objective validities, individuals apply a non-linear subjective scaling to these validities. This scaling allows a single highly valid cue to be disproportionately weighted to balance out multiple less valid cues, gracefully explaining why individuals exhibit near-indifference (choice proportions around 0.5) when faced with adversarial choices where one option excels on the best cue but the other excels on numerous lesser cues. Response noise is captured via a softmax over the options' weighted sums, along with an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
    
    # Apply non-linear subjective scaling to validities
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Compute weighted additive values for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 10.0]
- validities: validities

`rationale`:
The previous WADD model used raw objective validities, which resulted in options with many lower-validity cues easily dominating those with fewer, higher-validity cues, failing to capture the ~0.50 choice proportions observed in adversarial experiments. Following the critic's feedback, we introduce a subjective validity scaling parameter, `gamma`. By exponentiating the validities by `gamma`, the model can disproportionately weight the highest-validity cues (when gamma > 1), allowing a single highly valid cue to mathematically balance out multiple lower-validity cues. This brings the weighted sums of the options closer together, yielding the observed ~50/50 choice proportions via the softmax decision rule.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3785 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0983 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.0983.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(np.mean(data['response'] == 0))
```

**Observed (real) value:** 0.4554 (var=0.0025)
**Candidate trajectory (this loop):**
  - iter 1: 0.2029 (var=0.0149) (Δ vs real -0.2525)
  - iter 2 (current): 0.5123 (var=0.0398) (Δ vs real +0.0569)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8592 (var=0.0090)
- pi_2: 0.1310 (var=0.0058)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    tally_pred = (sum_b > sum_a).astype(int)
    return float((data['response'] == tally_pred).mean())
```

**Observed (real) value:** 0.5387 (var=0.0030)
**Candidate trajectory (this loop):**
  - iter 1: 0.8331 (var=0.0113) (Δ vs real +0.2944)
  - iter 2 (current): 0.4537 (var=0.0530) (Δ vs real -0.0850)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8683 (var=0.0052)
- pi_1: 0.1506 (var=0.0094)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate WADD model fails to capture the near-indifference (~0.50 choice proportions) observed in the adversarial designs. In Experiment 1, humans choose Option A (the option with fewer, but more valid cues) at a rate of 0.45, but the model predicts 0.20. In Experiment 2, humans choose the tally-predicted option at a rate of 0.54, but the model predicts 0.83. Because the raw objective validities are used directly as weights, the sum of multiple lower-validity cues easily dwarfs the single highest-validity cue, causing the model to strongly favor the option with more cues rather than treating them as roughly equivalent.
Rationale: To fix this while strictly remaining within the prescribed WADD family, you need to account for 'subjective validities'. Introduce a non-linear scaling parameter `gamma` (e.g., range [0.1, 10.0]) and transform the validities before computing the weighted sum: `val = np.asarray(parameters['validities']) ** float(parameters['gamma'])`. If gamma > 1, the model will disproportionately heavily weight the highest-validity cues, allowing a single highly valid cue to mathematically balance out multiple lower-validity cues. This will let the weighted sums for both options be nearly equal in these adversarial designs, producing the observed ~50/50 choice proportions via the softmax without completely flattening the temperature.

**Outcome of your advice:** iter 1 candidate loss=0.3785 -> CURRENT candidate loss=0.0983 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The introduction of the non-linear subjective validity scaling parameter (`gamma`) successfully corrected the model's behavior. By allowing the model to disproportionately weight the highest-validity cues, the weighted sums of the adversarial options are now much closer together. This leads to choice proportions around 0.45-0.51, closely matching the empirical data (0.4554 in Experiment 1, 0.5387 in Experiment 2). The aggregate loss has dropped significantly to 0.0983, indicating an excellent fit across both experiments.",
  "verdict": "continue",
  "rationale": "The current WADD theory with subjective validity scaling captures the empirical phenomena very well. It stays within the prescribed mechanism family while correctly explaining why subjects are nearly indifferent in these adversarial designs. The parameterization is sound, and the performance is strong enough to proceed to the next round."
}
```

## Usage

```json
{
  "prompt_token_count": 4053,
  "candidates_token_count": 199,
  "total_token_count": 4494
}
```
