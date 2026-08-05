# feedback_iter_00

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

Propose 'Take The Best' (TTB) as the new theory. Unlike Tallying (which counts unweighted wins) and WADD (which integrates all weighted features), TTB is a non-compensatory heuristic that searches through features in descending order of their validity, stopping at the first feature that discriminates between the two options and choosing the option with the higher value on that feature. This will provide a much stronger and more realistic heuristic baseline to contrast against the fully compensatory WADD model.


## CANDIDATE THEORY
People use a non-compensatory heuristic called 'Take The Best' (TTB) to choose between options. They search through features in descending order of subjective validity, stopping at the first feature that discriminates between the two options, and choose the option with the higher value on that feature. If no features discriminate, they guess randomly.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a_wins = 0.0
    b_wins = 0.0
    
    # Search for the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            a_wins = 1.0
            break
        elif b[idx] > a[idx]:
            b_wins = 1.0
            break
            
    # If no feature discriminates, baseline preference is uniform
    if a_wins == 0.0 and b_wins == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([a_wins, b_wins])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- epsilon: [0.0, 1.0]
- validities: validities

`rationale`:
Following the arbiter's guidance, I have implemented the Take The Best (TTB) heuristic. TTB is a one-reason, non-compensatory decision rule that orders features by their validity and bases the choice solely on the first feature that discriminates between the options. It completely ignores lower-validity cues once a discriminating cue is found, contrasting heavily with compensatory models like WADD and unweighted models like Tallying. Response noise is handled via an independent lapse rate (epsilon) which blends the deterministic TTB choice with uniform guessing.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0785 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0785.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    
    agreements = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        a_wadd = np.sum(a * validities)
        b_wadd = np.sum(b * validities)
        
        # Check if both models have strict preferences
        if a_wins != b_wins and a_wadd != b_wadd:
            tally_pref = 0 if a_wins > b_wins else 1
            wadd_pref = 0 if a_wadd > b_wadd else 1
            
            # Focus only on trials where Tallying and WADD disagree
            if tally_pref != wadd_pref:
                agreements.append(1.0 if row['response'] == tally_pref else 0.0)
                
    if len(agreements) == 0:
        return 0.5
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.3133 (var=0.0068)
**Candidate (simulated) value:** 0.2552 (var=0.0273)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8633 (var=0.0089)
- pi_2: 0.4407 (var=0.0546)

### Experiment 2
**Design**
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option A as a 2D array
    A = np.vstack(data['option_a_ratings'].values)
    
    # Identify Trial 1: A = [0, 0, 1, 1, 1, 0]
    # WADD prefers B (response 1), Tallying prefers A (response 0)
    is_t1 = (A[:, 0] == 0) & (A[:, 1] == 0) & (A[:, 2] == 1) & (A[:, 3] == 1)
    
    # Identify Trial 3: A = [1, 1, 0, 0, 0, 0]
    # WADD prefers A (response 0), Tallying prefers B (response 1)
    is_t3 = (A[:, 0] == 1) & (A[:, 1] == 1) & (A[:, 2] == 0) & (A[:, 3] == 0)
    
    responses = data['response'].values
    
    wadd_match = 0
    total = 0
    
    wadd_match += np.sum(responses[is_t1] == 1)
    total += np.sum(is_t1)
    
    wadd_match += np.sum(responses[is_t3] == 0)
    total += np.sum(is_t3)
    
    return float(wadd_match / total) if total > 0 else 0.5

```

**Observed (real) value:** 0.7188 (var=0.0044)
**Candidate (simulated) value:** 0.7562 (var=0.0282)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5719 (var=0.0907)
- pi_1: 0.1456 (var=0.0073)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The Take The Best (TTB) model successfully captures the empirical trends across both experiments. In Experiment 1, the model predicts the rate of Tallying agreement when it conflicts with WADD at 0.2552, which is reasonably close to the observed 0.3133 (and vastly outperforms the previous pi_1 model). In Experiment 2, the model predicts WADD agreement on diagnostic trials at 0.7562, which is an excellent match to the observed 0.7188. The implementation faithfully follows the arbiter's recommendation for a non-compensatory heuristic based on sorting by validity and stopping at the first discriminating feature.",
  "verdict": "continue",
  "rationale": "The TTB implementation is faithful to the arbiter's prescribed mechanism family and yields a strong fit to the human data across both experiments. The use of a lapse rate (epsilon) effectively captures the stochasticity in subjects' choices. The candidate is good enough to proceed."
}
```

## Usage

```json
{
  "prompt_token_count": 4026,
  "candidates_token_count": 222,
  "total_token_count": 4486
}
```
