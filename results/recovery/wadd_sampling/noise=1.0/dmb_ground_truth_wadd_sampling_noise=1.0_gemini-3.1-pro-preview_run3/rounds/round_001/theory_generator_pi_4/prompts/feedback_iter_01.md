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

Propose a 'Conflict-Induced Guessing' or 'Random Choice' theory. The new theory should posit that when participants face adversarial trials where the most valid cue conflicts with the sheer number of supporting cues (or the compensatory weight of lower cues), they experience decision conflict and resort to uniform guessing. This would perfectly explain the consistent ~0.50 choice rates and low between-subject variance observed across all experiments, distinguishing it from both the deterministic one-reason TTB and the deterministic compensatory WADD.


## CANDIDATE THEORY
Conflict-Induced Guessing Theory posits that decision makers concurrently evaluate options using both a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). When these two strategies agree, the decision maker confidently chooses the favored option (subject to standard softmax noise). However, when the strategies conflict—or when one strategy fails to corroborate the other (e.g., Tallying is tied while TTB prefers one option)—the decision maker experiences cognitive conflict or ambiguity. Unable to easily resolve this conflict, they resort to uniform guessing. This explains the consistent ~50% choice rates observed on adversarial trials and trials with weak corroborating evidence without requiring extreme parameter values.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Take The Best (TTB) prediction
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_winner = 0
    elif b_wins > a_wins:
        tally_winner = 1
    else:
        tally_winner = None
        
    # Determine choice probabilities
    if ttb_winner != tally_winner or ttb_winner is None:
        # Conflict or lack of clear corroboration leads to guessing
        p_core = np.array([0.5, 0.5])
    else:
        # No conflict: both strategies agree
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
            
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Modified the conflict condition so that if Tallying results in a tie (tally_winner is None), it no longer defaults to following TTB. Instead, a tie in the compensatory strategy fails to corroborate the non-compensatory strategy, which triggers the same guessing behavior as a direct contradiction. This resolves the failure in Experiment 4 where participants guessed on trials with a TTB winner but tied Tallying scores.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0474 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0352 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.0352.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                winner = 0
                break
            elif b[i] > a[i]:
                winner = 1
                break
        if winner is not None:
            matches.append(1 if resp == winner else 0)
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.4900 (var=0.0012)
**Candidate trajectory (this loop):**
  - iter 1: 0.5360 (var=0.0026) (Δ vs real +0.0460)
  - iter 2 (current): 0.5471 (var=0.0026) (Δ vs real +0.0571)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8477 (var=0.0132)
- pi_2: 0.2398 (var=0.0060)
- pi_3: 0.6058 (var=0.0018)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4981 (var=0.0023)
**Candidate trajectory (this loop):**
  - iter 1: 0.5055 (var=0.0019) (Δ vs real +0.0074)
  - iter 2 (current): 0.5086 (var=0.0021) (Δ vs real +0.0105)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8589 (var=0.0080)
- pi_1: 0.1488 (var=0.0095)
- pi_3: 0.4067 (var=0.0021)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_aligned = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None:
            if resp == ttb_pred:
                ttb_aligned += 1
            total += 1
            
    return ttb_aligned / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5033 (var=0.0016)
**Candidate trajectory (this loop):**
  - iter 1: 0.5052 (var=0.0023) (Δ vs real +0.0019)
  - iter 2 (current): 0.5077 (var=0.0019) (Δ vs real +0.0044)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8465 (var=0.0074)
- pi_3: 0.2437 (var=0.0109)
- pi_2: 0.1525 (var=0.0120)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5054 (var=0.0018)
**Candidate trajectory (this loop):**
  - iter 1: 0.4365 (var=0.0031) (Δ vs real -0.0690)
  - iter 2 (current): 0.4877 (var=0.0028) (Δ vs real -0.0177)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7238 (var=0.0085)
- pi_1: 0.1356 (var=0.0082)
- pi_2: 0.8200 (var=0.0068)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The Conflict-Induced Guessing theory performs excellently, matching the ~0.50 choice rates across most experiments (Loss=0.0474, ACCEPTED). However, it slightly misses the mark on Experiment 4 (0.4365 vs 0.5054) and Experiment 1 (0.5360 vs 0.4900). In Experiment 4, the final trial is A=[0,0,0,1,0] vs B=[0,0,0,0,1]. Here, TTB favors A, and Tallying is tied (1 to 1). The current model assumes that if Tallying is tied, there is 'no conflict' and it defaults to TTB, choosing A deterministically. However, the empirical data shows participants are still guessing (~0.50) on this trial. A tie in the compensatory strategy fails to corroborate the non-compensatory strategy, which still leaves the decision maker without a clear, confident choice.
Rationale: The candidate is very close and was accepted by the gate. To perfect the fit within the prescribed family, make a minor adjustment to the conflict logic: when Tallying is a tie, it should NOT default to TTB. Instead, a tie in the compensatory strategy means it fails to support TTB, which should also be treated as a state of conflict/ambiguity leading to uniform guessing (p=0.5). Update the `predict` function so that if `tally_winner` is None (a tie), it triggers the guessing behavior just like a direct contradiction does.

**Outcome of your advice:** iter 1 candidate loss=0.0474 -> CURRENT candidate loss=0.0352 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The model successfully implements the Conflict-Induced Guessing theory and the recent adjustment to treat tally ties as conflict has significantly improved the fit, particularly on Experiment 4 (moving from 0.4365 to 0.4877, closer to the empirical 0.5054). Across all four experiments, the model consistently predicts choice rates around ~0.50, closely matching the human ground-truth data. The loss has improved to 0.0352, demonstrating that the theoretical mechanism is robust and experiment-invariant.",
  "verdict": "continue",
  "rationale": "The candidate captures the human behavior well across all experiments, with minimal deviations from the empirical ~0.50 choice rates. The conflict logic correctly identifies adversarial and weak-corroboration trials, leading to the observed uniform guessing. The theory is good enough to proceed."
}
```

## Usage

```json
{
  "prompt_token_count": 5265,
  "candidates_token_count": 196,
  "total_token_count": 5761
}
```
