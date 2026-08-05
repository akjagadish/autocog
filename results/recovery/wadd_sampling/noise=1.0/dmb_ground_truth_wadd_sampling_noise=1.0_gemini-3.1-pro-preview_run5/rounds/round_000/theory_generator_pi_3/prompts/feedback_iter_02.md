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
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose a Weighted Additive (WADD) theory. Unlike TTB (which uses only the top discriminating cue) and Tallying (which gives equal weight to all cues), WADD computes a total score for each option by summing the validities of the features it possesses (or multiplying features by validities). It then chooses the option with the higher total score, subject to softmax noise. This compensatory model uses all available information and scales it by cue validity, potentially explaining the intermediate behavior observed when a single high-validity cue is pitted against multiple lower-validity cues.


## CANDIDATE THEORY
People compare options by computing a compensatory overall value for each option. This is done by summing the features of each option weighted by their respective validities (Weighted Additive rule). Because empirical behavior in these experiments is highly stochastic (near random guessing), choice is subject to significant softmax noise and lapse rates. Narrowing the inverse temperature bounds forces the model to capture this high level of noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match number of features.")

    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    a_score = np.sum(a * val)
    b_score = np.sum(b * val)
    scores = np.array([a_score, b_score])

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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

`rationale`:
Following the critic's advice, the upper bound of beta is narrowed from 20.0 to 5.0. This restricts the optimizer to the higher-noise regime, allowing the WADD model to more easily match the ~0.50 empirical targets observed in the experiments without being drawn to highly deterministic parameter points.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3650 -> ACCEPTED
- iter 2: loss=0.2690 -> ACCEPTED
- iter 3 (current candidate you are grading): loss=0.1879 -> ACCEPTED
Running-best (last accepted) base: iter 3 at loss=0.1879.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    def ttb_predict(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif a[i] < b[i]:
                return 1
        return 0.5
        
    ttb_choices = data.apply(ttb_predict, axis=1)
    matches = (data['response'] == ttb_choices)
    return float(matches.mean())
```

**Observed (real) value:** 0.5050 (var=0.0019)
**Candidate trajectory (this loop):**
  - iter 1: 0.3429 (var=0.0024) (Δ vs real -0.1621)
  - iter 2: 0.3812 (var=0.0048) (Δ vs real -0.1238)
  - iter 3 (current): 0.4148 (var=0.0068) (Δ vs real -0.0902)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8398 (var=0.0110)
- pi_2: 0.3215 (var=0.0026)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Only consider trials where Tallying has a strict preference
        if a_wins > b_wins:
            tally_choice = 0
            matches.append(int(row['response'] == tally_choice))
        elif b_wins > a_wins:
            tally_choice = 1
            matches.append(int(row['response'] == tally_choice))
            
    if not matches:
        return 0.5
    return float(np.mean(matches))

```

**Observed (real) value:** 0.5107 (var=0.0040)
**Candidate trajectory (this loop):**
  - iter 1: 0.8203 (var=0.0132) (Δ vs real +0.3097)
  - iter 2: 0.7327 (var=0.0219) (Δ vs real +0.2220)
  - iter 3 (current): 0.6603 (var=0.0148) (Δ vs real +0.1497)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8623 (var=0.0098)
- pi_1: 0.1203 (var=0.0068)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate implements the prescribed Weighted Additive (WADD) strategy. However, the simulation results show that the model strongly over-predicts matches with Tallying (0.82 vs 0.51) and under-predicts matches with TTB (0.34 vs 0.50). The empirical data, sitting right around 0.50 for both metrics, suggests that human behavior in these specific experiments is highly stochastic or near random. The current parameter ranges restrict the model from capturing this level of noise.
Rationale: The candidate successfully implements the WADD family, but fails to capture the empirical data because it cannot generate enough stochasticity to hit the ~0.50 observed metrics. Widen the parameter ranges to allow for near-random guessing: expand the lower bound of `beta` to 0.0 (e.g., `[0.0, 20.0]`) and the upper bound of `epsilon` to 1.0 (e.g., `[0.0, 1.0]`). This will allow the optimization loop to push the model closer to the observed ~0.5 behavior while keeping the WADD mechanism intact.

**Outcome of your advice:** iter 1 candidate loss=0.3650 -> iter 2 candidate loss=0.2690 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent)
Verdict: regenerate
Interpretation: The candidate incorporated the expanded parameter ranges (beta [0.0, 20.0], epsilon [0.0, 1.0]), which successfully improved the aggregate loss from 0.3650 to 0.2690. The model is moving in the right direction: the Experiment 1 metric increased from 0.34 to 0.38 (target 0.50), and the Experiment 2 metric decreased from 0.82 to 0.73 (target 0.51). However, the model still has not reached the highly stochastic ~0.50 baseline observed in the human data. The optimizer might be struggling to sample enough low-beta/high-epsilon combinations in the wide parameter space.
Rationale: The previous advice to expand the bounds was accepted and improved the fit, but the model still falls short of the ~0.50 empirical targets, likely because the parameter space for `beta` is too wide for the optimizer to reliably converge on the highly noisy regime. To force the model closer to the empirical randomness, narrow the upper bound of `beta` significantly (e.g., `[0.0, 5.0]`) while keeping `epsilon` at `[0.0, 1.0]`. This will concentrate the search space on lower determinism, allowing the WADD mechanism to hit the 0.50 mark.

**Outcome of your advice:** iter 2 candidate loss=0.2690 -> CURRENT candidate loss=0.1879 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate incorporated the narrowed beta range [0.0, 5.0], which successfully improved the aggregate loss from 0.2690 to 0.1879. The model continues to move in the right direction: Experiment 1 increased from 0.38 to 0.41 (target 0.50), and Experiment 2 decreased from 0.73 to 0.66 (target 0.51). The empirical data strongly indicates near-random guessing (~0.50 for both metrics), and while the current restrictions helped, the model is still slightly too deterministic.",
  "verdict": "regenerate",
  "rationale": "The previous advice to narrow the beta upper bound to 5.0 was accepted and improved the fit, but the model is still over-predicting determinism (e.g., 0.66 vs 0.51 in Exp 2). To push the model even closer to the highly stochastic empirical targets, narrow the upper bound of `beta` even further (e.g., `[0.0, 2.0]`) while keeping `epsilon` at `[0.0, 1.0]`. This will force the WADD mechanism into an even noisier regime, allowing it to fully capture the ~0.50 guessing behavior observed in the human data."
}
```

## Usage

```json
{
  "prompt_token_count": 4474,
  "candidates_token_count": 302,
  "total_token_count": 5216
}
```
