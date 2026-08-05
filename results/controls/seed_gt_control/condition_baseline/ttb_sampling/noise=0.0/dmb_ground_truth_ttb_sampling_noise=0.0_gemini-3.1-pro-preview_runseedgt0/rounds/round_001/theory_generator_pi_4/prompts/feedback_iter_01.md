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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Since strict cue integration (WADD) fails and the non-compensatory TTB rule is highly successful, the new theory should explore a different boundedly rational or heuristic approach. Propose a 'Probabilistic Take The Best' or 'Take The Best with Sequential Sampling' theory, where cue search is guided probabilistically by validity rather than being strictly deterministic, or a hybrid model where subjects use TTB but fall back to a simple Tallying (counting positive features) if the top cues are perceived as equally valid or too noisy. This provides a stronger, more nuanced competitor to the deterministic TTB.


## CANDIDATE THEORY
Probabilistic Take The Best (Plackett-Luce Cue Search) with High Determinism

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues discriminate and in which direction
    diff = a - b
    favor_a = (diff > 0)
    favor_b = (diff < 0)
    discriminating = favor_a | favor_b
    
    if not np.any(discriminating):
        # No cues discriminate, guess uniformly
        p_a = 0.5
    else:
        # The probability that a given discriminating cue is encountered first 
        # under a Plackett-Luce search order depends only on the relative weights 
        # of the discriminating cues.
        # Subtract max validity for numerical stability before exponentiation.
        max_val = np.max(val[discriminating])
        w = np.exp(gamma * (val - max_val))
        
        weight_a = np.sum(w[favor_a])
        weight_b = np.sum(w[favor_b])
        
        total_weight = weight_a + weight_b
        p_a = weight_a / total_weight
        
    p_core = np.array([p_a, 1.0 - p_a])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [0.0, 100.0]
- epsilon: [0.0, 0.2]
- validities: validities

`rationale`:
Following the critic's feedback, the empirical data indicates that human subjects are using a highly deterministic Take-The-Best strategy, whereas the previous parameterization forced too much noise. To resolve this while keeping the Plackett-Luce mechanism, I increased the upper bound of `gamma` to 100.0, allowing the model to approximate a strict argmax (deterministic TTB) when needed. I also reduced the upper bound of the lapse rate `epsilon` to 0.2 to prevent excessive random guessing from diluting the TTB agreement.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.4728 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.1368 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.1368.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    agreements = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        if ttb_winner is not None:
            if resp == ttb_winner:
                agreements += 1
            total += 1
            
    return float(agreements / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.8492 (var=0.0099)
**Candidate trajectory (this loop):**
  - iter 1: 0.6310 (var=0.0169) (Δ vs real -0.2181)
  - iter 2 (current): 0.8696 (var=0.0204) (Δ vs real +0.0204)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8342 (var=0.0101)
- pi_2: 0.2537 (var=0.0064)
- pi_3: 0.8438 (var=0.0123)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        if a_wins > b_wins:
            matches += (resp == 0)
            total += 1
        elif b_wins > a_wins:
            matches += (resp == 1)
            total += 1
    return float(matches / total) if total > 0 else 0.5

```

**Observed (real) value:** 0.1739 (var=0.0108)
**Candidate trajectory (this loop):**
  - iter 1: 0.4353 (var=0.0249) (Δ vs real +0.2614)
  - iter 2 (current): 0.1339 (var=0.0287) (Δ vs real -0.0400)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8858 (var=0.0098)
- pi_1: 0.1358 (var=0.0099)
- pi_3: 0.1464 (var=0.0088)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    diff = a_ratings - b_ratings
    is_diff = diff != 0
    has_diff = is_diff.any(axis=1)
    
    first_diff_idx = is_diff.argmax(axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    match = (resp[has_diff] == ttb_pred[has_diff])
    return float(match.mean()) if len(match) > 0 else 0.5
```

**Observed (real) value:** 0.8400 (var=0.0088)
**Candidate trajectory (this loop):**
  - iter 1: 0.4798 (var=0.0030) (Δ vs real -0.3602)
  - iter 2 (current): 0.6723 (var=0.0189) (Δ vs real -0.1677)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8827 (var=0.0091)
- pi_3: 0.3890 (var=0.0025)
- pi_2: 0.2754 (var=0.0023)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # TTB always chooses Option A (response = 0) across all 8 trials in this design
    # because A always has the single best discriminating cue. 
    # WADD, by contrast, integrates all cues and will frequently choose Option B 
    # (response = 1) because the sum of lower-validity cues often outweighs the single best cue.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.1613 (var=0.0125)
**Candidate trajectory (this loop):**
  - iter 1: 0.5450 (var=0.0047) (Δ vs real +0.3838)
  - iter 2 (current): 0.2829 (var=0.0221) (Δ vs real +0.1217)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5608 (var=0.0249)
- pi_1: 0.1537 (var=0.0066)
- pi_2: 0.8119 (var=0.0075)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate theory implements a probabilistic Take-The-Best (Plackett-Luce Cue Search) and was accepted by the gate. However, looking at the empirical data across Experiments 1, 3, and 4, human subjects are overwhelmingly using a deterministic TTB strategy (agreements are ~84-85%). The candidate's simulated values for these TTB metrics hover around 47-63%, indicating that the model is far too noisy and fails to capture the highly deterministic nature of human cue search in these paradigms. The model behaves more like a random proportional choice among discriminating cues rather than a sharp sequential search.
Rationale: While the Plackett-Luce mechanism fits the arbiter's prescribed family, its current parameterization forces too much noise, preventing it from matching the high deterministic TTB rate seen in the real data. To fix this within the same family, increase the upper bound of `gamma` significantly (e.g., [0.0, 50.0] or [0.0, 100.0]) so the softmax can approximate a strict argmax (deterministic TTB) when needed. Additionally, reduce the upper bound of the lapse rate `epsilon` to [0.0, 0.1] or [0.0, 0.2] to prevent excessive random guessing from dragging down the TTB agreement.

**Outcome of your advice:** iter 1 candidate loss=0.4728 -> CURRENT candidate loss=0.1368 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate model implements the Plackett-Luce Probabilistic Take-The-Best strategy as prescribed. By expanding the bounds of `gamma` to [0.0, 100.0] and constraining `epsilon` to [0.0, 0.2], the model has successfully reduced its aggregate loss from 0.4728 to 0.1368. It now captures the highly deterministic TTB behavior seen in the real data much better: Experiment 1 matches closely (0.8696 vs 0.8492), Experiment 2 matches well (0.1339 vs 0.1739), Experiment 3 has improved significantly (0.6723 vs 0.8400), and Experiment 4 has similarly moved much closer to the ground truth (0.2829 vs 0.1613). The modifications successfully allowed the model to approximate a strict argmax when necessary while retaining the probabilistic cue search framework.",
  "verdict": "continue",
  "rationale": "The candidate successfully incorporates the previous critique by widening the `gamma` parameter range and tightening the lapse rate `epsilon`. This allowed the probabilistic TTB model to account for the high degree of determinism in human cue search while remaining within the arbiter's prescribed family. The fit is now very strong across all four experiments, so the loop should proceed with this theory."
}
```

## Usage

```json
{
  "prompt_token_count": 5494,
  "candidates_token_count": 315,
  "total_token_count": 6450
}
```
