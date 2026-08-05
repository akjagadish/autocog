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

Propose a 'Mixture of Heuristics' theory. The new theory should posit that human decision-making in this domain is not governed by a single universal heuristic. Instead, individuals differ in their strategies, or switch between them, such that the population's choices reflect a mixture of Tallying (which ignores magnitudes and validities) and WADD (which integrates both). The model should introduce a mixture weight parameter 'alpha' that dictates the probability of using Tallying versus WADD on any given trial, along with the standard noise parameters.


## CANDIDATE THEORY
Human decision-making in multi-attribute choice is not governed by a single universal heuristic. Instead, individuals differ in their strategies or switch between them, such that the population's choices reflect a mixture of Tallying (which counts strict feature-wise wins and ignores magnitudes and validities) and Weighted Additive (WADD, which integrates both magnitudes and validities). A mixture weight parameter 'alpha' dictates the probability of using Tallying versus WADD on any given trial. Response noise enters through a softmax over the scores of the chosen heuristic, plus an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    
    # Tallying heuristic
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # WADD heuristic
    w = np.asarray(parameters["validities"], dtype=float)
    wadd_scores = stim @ w
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of heuristics
    alpha = float(parameters["alpha"])
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

`parameters`:
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: [(0.0, 1.0)] * n_features

`rationale`:
Following the arbiter's feedback, this theory implements a Mixture of Heuristics model. The previous models (Tallying and WADD) each systematically failed on certain experiments where the other succeeded, suggesting that human behavior relies on a combination of both strategies across the population. By introducing the 'alpha' parameter, the model can probabilistically mix Tallying (which ignores magnitudes and validities) and WADD (which accounts for both), allowing it to capture the heterogeneous choice patterns observed in the data.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0978 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0978.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[5, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[0, 8, 8, 8]  B=[10, 7, 7, 7]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[10, 2, 2, 2]  B=[0, 3, 3, 3]
  A=[2, 8, 8, 8]  B=[10, 5, 5, 5]
  A=[10, 1, 1, 1]  B=[0, 2, 2, 2]
  A=[0, 5, 5, 5]  B=[10, 4, 4, 4]
  A=[9, 3, 3, 3]  B=[0, 4, 4, 4]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_choice = (b_wins > a_wins).astype(int)
    return float((data['response'] == tally_choice).mean())

```

**Observed (real) value:** 0.6400 (var=0.2304)
**Candidate (simulated) value:** 0.5460 (var=0.0567)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.1950 (var=0.0838)

### Experiment 2
**Design**
  A=[10, 0, 0, 0]  B=[0, 3, 3, 3]
  A=[2, 2, 2, 0]  B=[0, 0, 0, 10]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 2, 2, 2]  B=[10, 0, 0, 0]
  A=[5, 5, 5, 5]  B=[6, 4, 4, 4]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_match = 0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (10, 0, 0, 0) and b == (0, 3, 3, 3):
            if resp == 0: wadd_match += 1
            count += 1
        elif a == (2, 2, 2, 0) and b == (0, 0, 0, 10):
            if resp == 1: wadd_match += 1
            count += 1
        elif a == (0, 2, 2, 2) and b == (10, 0, 0, 0):
            if resp == 1: wadd_match += 1
            count += 1
            
    return float(wadd_match / count) if count > 0 else 0.5
```

**Observed (real) value:** 0.3667 (var=0.0856)
**Candidate (simulated) value:** 0.4204 (var=0.0430)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6800 (var=0.0576)
- pi_1: 0.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The proposed Mixture of Heuristics model successfully captures the intermediate behavior observed in the human data. In Experiment 1, pure Tallying predicted 1.0 and pure WADD predicted 0.195, while human data was 0.6400. The candidate's mixture achieved 0.5460, a substantial improvement. In Experiment 2, pure Tallying predicted 0.0 and pure WADD predicted 0.6800, while human data was 0.3667. The candidate achieved 0.4204, again bridging the gap and reflecting a much closer fit to empirical patterns. The model appropriately balances these strategies via the alpha parameter.",
  "verdict": "continue",
  "rationale": "The candidate theory has successfully implemented the requested mixture mechanism and substantially improved the fit to the data, balancing the extremes of pure Tallying and pure WADD. The predictions are close enough to the empirical means to suggest the mechanism is conceptually correct and well-calibrated. No further structural changes are needed."
}
```

## Usage

```json
{
  "prompt_token_count": 3605,
  "candidates_token_count": 236,
  "total_token_count": 4099
}
```
