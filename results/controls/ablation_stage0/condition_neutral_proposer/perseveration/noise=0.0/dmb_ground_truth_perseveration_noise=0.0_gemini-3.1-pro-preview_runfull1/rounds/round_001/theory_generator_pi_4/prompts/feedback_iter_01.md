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
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a 'Strategy Selection' or 'Mixed Heuristics' theory. Instead of assuming all subjects use TTB or WADD exclusively, this theory should propose that decision-makers probabilistically sample between a non-compensatory heuristic (like TTB) and a simple compensatory heuristic (like equal-weights Tallying) on a trial-by-trial basis, depending on the difficulty of the choice. This would naturally pull the population-level predictions closer to 0.5 in highly conflicting trials while retaining some sensitivity to the dominant cue, better matching the observed data across all four experiments.


## CANDIDATE THEORY
Strategy Selection (Mixed Heuristics): Decision-makers do not rely on a single heuristic for all choices. Instead, they probabilistically sample between a non-compensatory heuristic (Take The Best) and a simple compensatory heuristic (Tallying) on a trial-by-trial basis. The probability of using Tallying increases with the ease of the choice, defined by the absolute difference in the number of features favoring each option. When the options are highly similar in tally (a difficult choice), decision-makers fall back on the most valid cue (TTB). When Tallying clearly discriminates, they are more likely to use it. In highly conflicting trials where the heuristics point in opposite directions, this mixture naturally pulls choice probabilities toward 0.5, while retaining sensitivity to dominant cues when the heuristics agree.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) score
    ttb_score = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            ttb_score = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            ttb_score = np.array([0.0, 1.0])
            break
            
    # Tallying (Equal-Weights) score
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    if a_wins > b_wins:
        tally_score = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        tally_score = np.array([0.0, 1.0])
    else:
        tally_score = np.array([0.5, 0.5])
        
    # Difficulty defined by tally difference
    diff = abs(a_wins - b_wins)
    
    # Probability of using Tallying over TTB
    gamma = float(parameters["gamma"])
    w_tally = 1.0 - np.exp(-gamma * diff)
    w_ttb = 1.0 - w_tally
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for TTB
    z_ttb = beta * ttb_score
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * tally_score
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of probabilities
    p_core = w_ttb * p_ttb + w_tally * p_tally
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 5.0]
- validities: validities

`rationale`:
Following the critic's feedback, the model now applies the softmax function to the TTB and Tallying scores separately to compute their respective choice probabilities. The final choice probability is then computed as a weighted average of these two probability distributions (`w_ttb * p_ttb + w_tally * p_tally`). This correctly implements a probabilistic sampling interpretation, ensuring that when the heuristics disagree, the aggregate probability averages out (pulling toward 0.5) instead of collapsing deterministically due to a high beta parameter.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.4583 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.4066 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.4066.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    is_1 = a_tuples == (1, 0, 0, 0)
    is_3 = a_tuples == (0, 1, 1, 1)
    is_5 = a_tuples == (1, 1, 0, 0)
    is_6 = a_tuples == (0, 0, 1, 1)
    
    correct = 0
    total = 0
    
    if is_1.any():
        correct += (data.loc[is_1, 'response'] == 0).sum()
        total += is_1.sum()
    if is_3.any():
        correct += (data.loc[is_3, 'response'] == 1).sum()
        total += is_3.sum()
    if is_5.any():
        correct += (data.loc[is_5, 'response'] == 0).sum()
        total += is_5.sum()
    if is_6.any():
        correct += (data.loc[is_6, 'response'] == 1).sum()
        total += is_6.sum()
        
    if total == 0:
        return 0.5
    return float(correct / total)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.2037 (var=0.0341) (Δ vs real -0.2963)
  - iter 2 (current): 0.2525 (var=0.0333) (Δ vs real -0.2475)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8250 (var=0.0123)
- pi_2: 0.1369 (var=0.0055)
- pi_3: 0.4153 (var=0.0068)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            continue
            
        tally_match.append(row['response'] == tally_choice)
        
    if not tally_match:
        return 0.5
    return float(np.mean(tally_match))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.7967 (var=0.0361) (Δ vs real +0.2967)
  - iter 2 (current): 0.7275 (var=0.0376) (Δ vs real +0.2275)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8575 (var=0.0067)
- pi_1: 0.1575 (var=0.0102)
- pi_3: 0.5614 (var=0.0029)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None and row['response'] == ttb_pred:
            ttb_matches += 1
            
    return float(ttb_matches / total)
```

**Observed (real) value:** 0.6950 (var=0.1026)
**Candidate trajectory (this loop):**
  - iter 1: 0.2296 (var=0.0360) (Δ vs real -0.4654)
  - iter 2 (current): 0.2594 (var=0.0462) (Δ vs real -0.4356)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8502 (var=0.0120)
- pi_3: 0.3042 (var=0.0231)
- pi_2: 0.1544 (var=0.0074)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Candidate trajectory (this loop):**
  - iter 1: 0.7352 (var=0.0440) (Δ vs real +0.2152)
  - iter 2 (current): 0.7392 (var=0.0460) (Δ vs real +0.2192)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6940 (var=0.0168)
- pi_1: 0.1644 (var=0.0130)
- pi_2: 0.8583 (var=0.0075)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate theory attempts to implement the Strategy Selection (Mixed Heuristics) model by taking a weighted average of the TTB and Tallying scores based on choice difficulty, and then applying a softmax function. However, the model fails to match the observed data, predicting extreme probabilities (e.g., 0.20 in Exp 1, 0.79 in Exp 2, and 0.22 in Exp 3) where the human data is closer to 0.5 or 0.69. The core issue is that mixing the heuristic *scores* before applying a softmax with a potentially large `beta` (up to 20.0) causes the model to deterministically choose whichever heuristic has a slightly higher weight, completely destroying the probabilistic sampling aspect of the mixture.
Rationale: To faithfully implement 'probabilistically sampling' between heuristics, the mixture must occur *after* the choice probabilities for each heuristic are computed, or the mixture weights themselves should act as the probabilities of using each strategy. Revise the `predict` function to compute the choice probabilities for TTB and Tallying separately (e.g., using softmax or just returning the one-hot scores if deterministic), and then compute `p_mixed = w_ttb * p_ttb + w_tally * p_tally`. This will ensure that when the heuristics disagree, the aggregate choice probability naturally averages out (closer to 0.5), rather than collapsing to 0 or 1 due to the high beta softmax.

**Outcome of your advice:** iter 1 candidate loss=0.4583 -> CURRENT candidate loss=0.4066 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The model correctly implemented the probabilistic mixture of TTB and Tallying probabilities, which improved the aggregate loss and was accepted by the gate. However, the predictions are still far from the observed data: in Exp 3, the model predicts 0.259 TTB matches compared to the human 0.695, indicating it relies far too heavily on Tallying. In Exps 1 and 2, predictions are around 0.25 and 0.72 respectively, while human data is exactly 0.50. The issue lies in the parameterization of the mixture weight `w_tally = 1.0 - np.exp(-gamma * diff)`. With `gamma` ranging up to 5.0 and `diff` up to 5, `w_tally` rapidly saturates to near 1.0 on any trial where Tallying clearly discriminates, completely suppressing TTB and contradicting the empirical reliance on TTB in Exp 3.",
  "verdict": "regenerate",
  "rationale": "The current mechanism is correct in family but miscalibrated in its parameter ranges, causing Tallying to overwhelm TTB. To fix this, restrict the `gamma` parameter range to something much smaller (e.g., `[0.0, 0.5]` or `[0.0, 1.0]`) so that `w_tally` does not immediately saturate to 1.0 on high-difference trials, preserving a baseline probability of using TTB. Additionally, consider widening the `epsilon` range (e.g., up to 1.0) or lowering the maximum `beta` to help pull the highly conflicting trials in Exps 1 and 2 closer to the observed 0.50."
}
```

## Usage

```json
{
  "prompt_token_count": 5929,
  "candidates_token_count": 389,
  "total_token_count": 7515
}
```
