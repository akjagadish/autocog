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

Replace Tallying with the 'Take The Best' (TTB) heuristic. TTB proposes that decision-makers rank features by their validity and evaluate them lexicographically. The choice is determined entirely by the highest-validity feature that discriminates between the two options, ignoring all lower-validity features. This non-compensatory approach would predict the extreme sensitivity to high-validity features observed in the data (e.g., the 82% preference in Experiment 2), serving as a much stronger competitor to WADD than Tallying.


## CANDIDATE THEORY
Decision-makers use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. Features are ranked by their validity, and options are compared on features one by one in descending order of validity. The choice is determined entirely by the first feature that discriminates between the options, ignoring all lower-validity cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Lexicographic evaluation
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's feedback, this model instantiates the Take The Best (TTB) heuristic. Instead of summing features (like Tallying or WADD), TTB sorts features by validity and makes a choice based solely on the highest-validity feature that discriminates between the options. This non-compensatory mechanism naturally accounts for the extreme sensitivity to high-validity cues observed in Experiment 2, where a single predictive feature can strongly dominate the choice over multiple weaker features.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.2201 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.2201.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.76, 0.92, 0.55])
    
    # Extract matrices of A and B features
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying differences
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_diff = a_wins - b_wins
    
    # WADD differences (expected without subject-specific weights)
    wadd_diff = np.sum((a_mat - b_mat) * validities, axis=1)
    
    # 1 if chose A, 0 if chose B
    chose_a = 1.0 - data['response'].values
    
    # We compare choice probabilities within trials where Tallying predicts the exact same difference.
    # For tally_diff == 1, WADD predicts some trials favor A more strongly than others.
    mask1_high = (tally_diff == 1) & (wadd_diff > 0.85)
    mask1_low = (tally_diff == 1) & (wadd_diff < 0.85)
    
    diff1 = 0.0
    if np.any(mask1_high) and np.any(mask1_low):
        diff1 = np.mean(chose_a[mask1_high]) - np.mean(chose_a[mask1_low])
        
    # For tally_diff == -1, WADD predicts some trials favor B more strongly than others.
    mask_m1_high = (tally_diff == -1) & (wadd_diff > -0.85)
    mask_m1_low = (tally_diff == -1) & (wadd_diff < -0.85)
    
    diff_m1 = 0.0
    if np.any(mask_m1_high) and np.any(mask_m1_low):
        diff_m1 = np.mean(chose_a[mask_m1_high]) - np.mean(chose_a[mask_m1_low])
        
    # Under Tallying, both diff1 and diff_m1 should be 0.
    # Under WADD, both diff1 and diff_m1 should be positive.
    return float(diff1 + diff_m1)
```

**Observed (real) value:** 0.1467 (var=0.0250)
**Candidate (simulated) value:** 0.2033 (var=0.0381)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0017 (var=0.0438)
- pi_2: 0.0792 (var=0.0977)

### Experiment 2
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Tallying predicts a tie but WADD predicts a strong preference.
    # Trial 8: A=[0, 1, 0, 1], B=[1, 1, 0, 0]
    # A wins on feature 4 (validity 0.55). B wins on feature 1 (validity 0.95).
    # Tallying sees 1 win for A and 1 win for B, predicting exactly 50% choice for B.
    # WADD sees B's win on the most important feature as outweighing A's win on the least important, predicting >50% choice for B.
    is_target = data['option_a_ratings'].apply(lambda x: list(x) == [0, 1, 0, 1]) & \
                data['option_b_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0])
    
    if is_target.sum() == 0:
        return 0.5
        
    return float(data.loc[is_target, 'response'].mean())
```

**Observed (real) value:** 0.8200 (var=0.0532)
**Candidate (simulated) value:** 0.8833 (var=0.0247)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6533 (var=0.0887)
- pi_1: 0.4967 (var=0.0417)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory successfully implements the Take The Best (TTB) heuristic as prescribed by the arbiter. The results demonstrate that TTB effectively captures the empirical patterns across both experiments. In Experiment 1, TTB accounts for the positive metric (0.2033 vs. real 0.1467), successfully capturing variance that Tallying misses by ignoring feature validities. In Experiment 2, TTB predicts the strong preference for Option B driven by the highest-validity feature (0.8833 vs. real 0.8200), vastly outperforming the baseline Tallying model (0.4967) and showing strong alignment with human behavior.",
  "verdict": "continue",
  "rationale": "The candidate was accepted by the programmatic gate and successfully implements the prescribed non-compensatory mechanism family (TTB). It captures both the subtle WADD-like sensitivity in Experiment 1 and the extreme single-feature dominance in Experiment 2 with good quantitative fit. The implementation is clean and theoretically sound."
}
```

## Usage

```json
{
  "prompt_token_count": 4454,
  "candidates_token_count": 231,
  "total_token_count": 5114
}
```
