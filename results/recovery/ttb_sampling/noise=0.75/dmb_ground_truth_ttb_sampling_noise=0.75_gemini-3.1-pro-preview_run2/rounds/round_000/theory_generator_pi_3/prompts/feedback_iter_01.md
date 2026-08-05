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

Replace Tallying with 'Take-The-Best' (TTB). TTB is a lexicographic, non-compensatory heuristic where subjects rank features by validity and decide based solely on the first feature that discriminates between the two options. This provides a strong non-compensatory alternative to the compensatory WADD model, differing from Tallying by relying entirely on the validity hierarchy rather than an unweighted count.


## CANDIDATE THEORY
Take-The-Best (TTB) heuristic: People make decisions by ranking features according to their validities and choosing the option that is favored by the single most valid discriminating feature. If no feature discriminates, they guess. This is a lexicographic, non-compensatory strategy. However, human execution of this strategy is highly noisy, so choice probabilities are heavily tempered by response noise (low beta) and random guessing lapses (high epsilon).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to preserve the original feature order in case of ties.
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    
    scores = np.array([0.0, 0.0])
    # Find the first feature that discriminates between the two options
    for idx in ranked_features:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    # If no feature discriminates, the core preference is uniform
    if scores[0] == 0.0 and scores[1] == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        # Softmax over the scores to introduce response noise
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

`rationale`:
Following the critic's feedback, the core Take-The-Best logic is preserved, but the parameter ranges for response noise have been adjusted. The upper bound of `beta` is reduced to 5.0 and the range of `epsilon` is expanded to 1.0. This allows the model to capture the heavily noisy nature of human choices in these experiments, bringing its predictions closer to the observed 50% baseline without abandoning the underlying non-compensatory mechanism.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3629 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.1740 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.1740.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_choices = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: A has 3 wins on weak features, B has 2 wins on strong features
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            # Tallying prefers A
            tally_choices.append(1.0 if row['response'] == 0 else 0.0)
        # Trial 2: A has 2 wins on strong features, B has 3 wins on weak features
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            # Tallying prefers B
            tally_choices.append(1.0 if row['response'] == 1 else 0.0)
            
    if not tally_choices:
        return 0.5
    return float(np.mean(tally_choices))
```

**Observed (real) value:** 0.3975 (var=0.0118)
**Candidate trajectory (this loop):**
  - iter 1: 0.1350 (var=0.0105) (Δ vs real -0.2625)
  - iter 2 (current): 0.3212 (var=0.0320) (Δ vs real -0.0763)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8788 (var=0.0105)
- pi_2: 0.4338 (var=0.0954)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the options are [1, 1, 0, 0, 0] and [0, 0, 1, 1, 1]
    is_11000_A = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_11000_B = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    mask = is_11000_A | is_11000_B
    if mask.sum() == 0:
        return 0.5
    
    subset = data[mask]
    is_11000_A_sub = is_11000_A[mask]
    
    # Check if the subject chose the option with [1, 1, 0, 0, 0]
    chose_11000 = (is_11000_A_sub & (subset['response'] == 0)) | (~is_11000_A_sub & (subset['response'] == 1))
    return float(chose_11000.mean())
```

**Observed (real) value:** 0.5833 (var=0.0110)
**Candidate trajectory (this loop):**
  - iter 1: 0.8533 (var=0.0134) (Δ vs real +0.2700)
  - iter 2 (current): 0.7050 (var=0.0238) (Δ vs real +0.1217)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5050 (var=0.0797)
- pi_1: 0.1292 (var=0.0114)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate Take-The-Best (TTB) model successfully captures the qualitative direction of the human data, which strongly rejects the Tallying model (as seen in Experiment 1, where Tallying choice is below 0.5). However, the candidate model is far too extreme. In Experiment 1, it predicts a tallying choice rate of 0.1350 compared to the observed 0.3975. In Experiment 2, it predicts a TTB choice rate of 0.8533 compared to the observed 0.5833. Human behavior is much noisier and closer to 50% than pure TTB predicts.
Rationale: The TTB mechanism is correctly implemented and is moving in the right direction, but it is currently too deterministic. To capture the fact that human choices are much closer to uniform (0.39 and 0.58), reduce the upper bound of the `beta` parameter (e.g., [0.0, 5.0]) to soften the softmax, and widen the `epsilon` lapse rate range (e.g., [0.0, 1.0]) to allow the model to rely more heavily on random guessing when needed. Keep the core TTB logic intact.

**Outcome of your advice:** iter 1 candidate loss=0.3629 -> CURRENT candidate loss=0.1740 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate model, following the previous advice to increase noise, has significantly improved its fit to the human data (loss decreased from 0.3629 to 0.1740). However, the model is still slightly too deterministic in its Take-The-Best predictions. In Experiment 1, it predicts a tallying choice rate of 0.3212 (implying a TTB choice rate of ~0.68), compared to the observed 0.3975 (TTB rate ~0.60). In Experiment 2, it predicts a TTB choice rate of 0.7050 compared to the observed 0.5833. The human data is even closer to 50% than the current parameter ranges naturally encourage during fitting.",
  "verdict": "regenerate",
  "rationale": "The previous suggestion to lower the upper bound of `beta` and widen `epsilon` was successful and accepted by the gate. To close the remaining gap, push further in the same direction: tighten the upper bound of `beta` even more (e.g., [0.0, 2.0]) to force the softmax to be even softer, ensuring the model captures the heavily noisy, near-guessing behavior observed in the human data. Keep the core TTB mechanism and `epsilon` range intact."
}
```

## Usage

```json
{
  "prompt_token_count": 4639,
  "candidates_token_count": 298,
  "total_token_count": 5537
}
```
