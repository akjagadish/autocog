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

Propose a Weighted Additive (WADD) theory. Instead of relying on a single discriminating cue (like TTB) or simply counting features while ignoring validities (like Tallying), the WADD theory should posit that participants compute a weighted sum of the features for each option, using the cue validities as weights. The choice is then made probabilistically (e.g., via a softmax function) based on the difference between these weighted sums. This allows for compensatory decision-making that can naturally capture the intermediate agreement rates observed across both experiments.


## CANDIDATE THEORY
People make choices by computing a weighted sum of all available features for each option, but the weights they use are a non-linear transformation of the objective cue validities. By exponentiating the validities with a free parameter gamma, the strategy can smoothly interpolate between equal weighting (Tallying, gamma=0) and a strong reliance on the most valid cues (approximating Take The Best, gamma > 1). The choice is then made probabilistically based on the difference between the options' weighted sums, with a bounded inverse temperature beta to allow for more stochastic choices.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities
    subjective_weights = val ** gamma
    
    # Calculate weighted sums for both options
    a, b = stim[0], stim[1]
    score_a = np.sum(a * subjective_weights)
    score_b = np.sum(b * subjective_weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate response noise (lapse rate)
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 5.0]
- validities: validities

`rationale`:
Following the critic's advice, we retain the highly successful WADD mechanism with non-linear validity scaling (gamma). To address the remaining over-prediction of Tallying agreement in Experiment 2, we reduce the upper bound of the beta parameter from 20.0 to 5.0. This enforces more stochasticity in choices when weighted sums are close, which should naturally lower the Experiment 2 agreement rates while preserving the balance achieved in Experiment 1.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2809 -> ACCEPTED
- iter 2: loss=0.1450 -> ACCEPTED
- iter 3 (current candidate you are grading): loss=0.0467 -> ACCEPTED
Running-best (last accepted) base: iter 3 at loss=0.0467.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    
    ttb_choices = []
    for a, b in zip(a_mat, b_mat):
        choice = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                choice = 0
                break
            elif b[i] > a[i]:
                choice = 1
                break
        ttb_choices.append(choice)
        
    ttb_choices = np.array(ttb_choices)
    responses = data['response'].values
    
    valid = ttb_choices != 0.5
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))
```

**Observed (real) value:** 0.4408 (var=0.0085)
**Candidate trajectory (this loop):**
  - iter 1: 0.3283 (var=0.0031) (Δ vs real -0.1125)
  - iter 2: 0.4567 (var=0.0195) (Δ vs real +0.0158)
  - iter 3 (current): 0.4292 (var=0.0115) (Δ vs real -0.0117)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8546 (var=0.0116)
- pi_2: 0.2252 (var=0.0040)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying counts the number of features where one option strictly dominates the other
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pred = np.zeros(len(data))
    tally_pred[b_wins > a_wins] = 1
    tally_pred[a_wins == b_wins] = -1 # Ignore ties for matching purposes
    
    matches = (data['response'].values == tally_pred)
    valid = (tally_pred != -1)
    
    return float(np.mean(matches[valid])) if np.any(valid) else 0.5
```

**Observed (real) value:** 0.6317 (var=0.0047)
**Candidate trajectory (this loop):**
  - iter 1: 0.8573 (var=0.0099) (Δ vs real +0.2256)
  - iter 2: 0.7871 (var=0.0148) (Δ vs real +0.1554)
  - iter 3 (current): 0.6756 (var=0.0176) (Δ vs real +0.0440)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8548 (var=0.0074)
- pi_1: 0.2681 (var=0.0045)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The proposed Weighted Additive (WADD) model captures the compensatory nature of decision-making, but it struggles to match the exact empirical agreement rates. It under-predicts TTB agreement in Experiment 1 (0.3283 vs 0.4408) and over-predicts Tallying agreement in Experiment 2 (0.8573 vs 0.6317). The fixed validities do not seem to perfectly reflect the subjective weights participants assign to the features.
Rationale: To better fit the intermediate agreement rates, keep the WADD mechanism but introduce a `gamma` parameter (e.g., range [0.0, 5.0]) that exponentiates the validities (`val ** gamma`) before computing the weighted sum. This non-linear scaling allows the model to flexibly interpolate between pure Tallying (gamma = 0) and a more TTB-like strong reliance on high-validity cues (gamma > 1), which should help balance the metrics across both experiments.

**Outcome of your advice:** iter 1 candidate loss=0.2809 -> iter 2 candidate loss=0.1450 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent)
Verdict: regenerate
Interpretation: The introduction of the `gamma` parameter significantly improved the model's fit, effectively reducing the aggregate loss from 0.2809 to 0.1450. The model now matches Experiment 1's TTB agreement very closely (0.4567 vs real 0.4408). However, it still over-predicts the Tallying agreement in Experiment 2 (0.7871 vs real 0.6317). The non-linear scaling of validities is working as intended, but the choice function might still be too deterministic, leading to the over-prediction in Experiment 2.
Rationale: Keep the exact same WADD mechanism with the `gamma` parameter, as it was highly successful. To address the remaining over-prediction of Tallying agreement in Experiment 2 without disrupting Experiment 1, reduce the upper bound of the `beta` (inverse temperature) parameter range (e.g., from [0.1, 20.0] to [0.01, 5.0]). A lower maximum beta will encourage slightly softer, more stochastic choices when weighted sums are close, which should naturally pull the Experiment 2 Tallying agreement down closer to the empirical 0.6317.

**Outcome of your advice:** iter 2 candidate loss=0.1450 -> CURRENT candidate loss=0.0467 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate model with the non-linear validity scaling (gamma) and the restricted inverse temperature (beta) parameter range has successfully captured the empirical patterns across both experiments. The aggregate loss has plummeted to 0.0467. In Experiment 1, the predicted TTB agreement (0.4292) is extremely close to the observed value (0.4408). In Experiment 2, the previously high Tallying agreement has been successfully reigned in, dropping from 0.7871 to 0.6756, which is now very close to the true empirical value of 0.6317. The minor adjustments to the softmax temperature bounds provided exactly the right amount of stochasticity without breaking the core compensatory mechanism.",
  "verdict": "continue",
  "rationale": "The model fits both experiments well within an acceptable margin of error. The combination of WADD, non-linear validity scaling via gamma, and a well-calibrated softmax temperature effectively interpolates between the strategies and matches the human data. No further architectural or parameter bound changes are necessary."
}
```

## Usage

```json
{
  "prompt_token_count": 4560,
  "candidates_token_count": 240,
  "total_token_count": 5047
}
```
