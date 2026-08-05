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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a 'High-Noise / Random Guessing' theory. The new theory should posit that due to the abstract nature of the task and lack of trial-by-trial feedback, subjects do not consistently apply structured multi-attribute heuristics. Instead, their choices are dominated by high response noise or a massive lapse rate, effectively resulting in random guessing on most trials. The model should predict uniform (0.5) probabilities across most, if not all, stimulus pairs, differentiating itself from both Tallying and TTB-Mixture models which predict decisive choices in at least some conditions.


## CANDIDATE THEORY
Due to the abstract nature of the task and lack of trial-by-trial feedback, subjects do not consistently apply structured multi-attribute heuristics. Instead, their choices are dominated by high response noise or a massive lapse rate, effectively resulting in random guessing on most trials. While they may occasionally attempt to tally features or look at the most valid cue, the lack of grounding leads to a near-uniform choice probability across all stimulus pairs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    # Calculate a weak tallying signal
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Massive lapse rate dominates the choice
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]

`rationale`:
The real experimental data shows choice alignments hovering extremely close to 0.5 across all metrics (e.g., 0.5025, 0.5121, 0.5117, 0.4842). Previous models like pure TTB or Tallying vastly overpredicted decisive choices (reaching ~0.85). Even the mixture model predicted ~0.67 in some experiments. By explicitly modeling a massive lapse rate (epsilon between 0.8 and 1.0) and very low inverse temperature (beta near 0), this theory posits that subjects are essentially guessing due to the abstract task and lack of feedback, directly capturing the near-0.5 alignment observed in the empirical data.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0182 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0182.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract matrices of option features
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Calculate the difference
    diff = a_mat - b_mat
    
    # Using exponentially decreasing weights, the sign of the weighted sum
    # is completely determined by the first non-zero difference.
    # This exactly mimics the one-reason stopping rule of Take The Best.
    weights = np.array([10000, 1000, 100, 10, 1])
    weighted_diff = np.dot(diff, weights)
    
    # If weighted_diff > 0, A is favored by the first discriminating cue (TTB predicts 0).
    # If weighted_diff < 0, B is favored by the first discriminating cue (TTB predicts 1).
    ttb_pred = (weighted_diff < 0).astype(int)
    
    # Return the proportion of choices that align with the Take The Best prediction
    return float(np.mean(ttb_pred == data['response']))
```

**Observed (real) value:** 0.5025 (var=0.0034)
**Candidate (simulated) value:** 0.4985 (var=0.0027)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8379 (var=0.0113)
- pi_2: 0.1288 (var=0.0054)
- pi_3: 0.5142 (var=0.0491)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

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
        tally_pred = 0 if a_wins > b_wins else 1
        matches.append(row['response'] == tally_pred)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5121 (var=0.0032)
**Candidate (simulated) value:** 0.5112 (var=0.0026)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8623 (var=0.0092)
- pi_1: 0.1646 (var=0.0113)
- pi_3: 0.4844 (var=0.0419)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        # Focus only on trials where Tallying predicts a tie
        if tally_a == tally_b:
            ttb_winner = None
            # Cue order matches index order since validities are strictly decreasing
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_winner = 0
                    break
                elif b[i] > a[i]:
                    ttb_winner = 1
                    break
            
            if ttb_winner is not None:
                matches.append(row['response'] == ttb_winner)
                
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5117 (var=0.0085)
**Candidate (simulated) value:** 0.5142 (var=0.0091)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6675 (var=0.0266)
- pi_2: 0.5067 (var=0.0049)
- pi_1: 0.8725 (var=0.0113)

### Experiment 4
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_aligned = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_choice = 0
                    break
                elif b[i] > a[i]:
                    ttb_choice = 1
                    break
            else:
                continue
                
            ttb_aligned.append(int(row['response'] == ttb_choice))
            
    if len(ttb_aligned) == 0:
        return 0.5
    return float(np.mean(ttb_aligned))
```

**Observed (real) value:** 0.4842 (var=0.0049)
**Candidate (simulated) value:** 0.5058 (var=0.0064)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5271 (var=0.0046)
- pi_3: 0.6754 (var=0.0184)
- pi_1: 0.8512 (var=0.0146)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory effectively captures the empirical data across all four experiments. The real human data shows alignments hovering remarkably close to 0.50 for both Take-The-Best and Tallying metrics, regardless of whether the heuristics predict decisive outcomes or ties. The proposed 'High-Noise / Random Guessing' model, which implements a massive lapse rate (epsilon between 0.8 and 1.0) and a low inverse temperature (beta near 0), accurately produces near-uniform choice probabilities. The simulated metric values (0.4985, 0.5112, 0.5142, and 0.5058) are excellent matches to the observed values (0.5025, 0.5121, 0.5117, and 0.4842).",
  "verdict": "continue",
  "rationale": "The proposer faithfully implemented the arbiter's recommendation for a 'High-Noise / Random Guessing' theory. The resulting model fits the empirical data exceptionally well, successfully capturing the near-random behavior of subjects in this abstract task. No further adjustments are needed; the current model should be kept."
}
```

## Usage

```json
{
  "prompt_token_count": 5036,
  "candidates_token_count": 266,
  "total_token_count": 5619
}
```
