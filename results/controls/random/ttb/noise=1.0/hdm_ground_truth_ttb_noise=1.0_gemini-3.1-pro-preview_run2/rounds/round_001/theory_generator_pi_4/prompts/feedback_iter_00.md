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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

The current WADD theory (THEORY 2) is degenerate for this dataset as participants do not exhibit strong compensatory behavior. Given that aggregate choices are consistently near 0.5, replace THEORY 2 with a 'Random Guessing / High-Lapse Baseline' theory, or a 'Tallying' heuristic with a very high baseline noise/lapse rate. A pure Random Guessing model would explicitly instantiate the hypothesis that subjects are overwhelmed by the cardinal feature magnitudes and simply guess, perfectly capturing the ~0.5 metric values across all experiments without relying on a mismatched mechanism (like TTB with an artificially inflated threshold).


## CANDIDATE THEORY
Random Guessing / High-Lapse Baseline: When confronted with complex multi-attribute choices involving cardinal feature magnitudes that do not easily yield to simple heuristics, decision makers become overwhelmed and resort to random guessing. Their choices reflect a near-uniform probability distribution over the available options, occasionally influenced by a negligible spatial or option-order bias.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    bias = float(parameters["bias"])
    return np.array([bias, 1.0 - bias])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- bias: [0.48, 0.52]

`rationale`:
The arbiter noted that the previous WADD theory was degenerate because participants do not exhibit strong compensatory behavior, and aggregate choices are consistently near 0.5. This implies that subjects are likely overwhelmed by the cardinal feature magnitudes and simply guess. A pure Random Guessing model explicitly instantiates this hypothesis, capturing the ~0.5 metric values across all experiments without relying on a mismatched mechanism like TTB with inflated thresholds.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0179 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0179.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[5, 5, 5, 5]  B=[4, 4, 4, 10]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 10, 0, 0]  B=[1, 0, 1, 1]
  A=[5, 5, 6, 6]  B=[6, 6, 0, 0]
  A=[2, 2, 2, 2]  B=[0, 0, 10, 10]
  A=[4, 4, 4, 4]  B=[3, 3, 5, 5]
  A=[0, 10, 10, 10]  B=[10, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    relevant_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        
        # Trial 1: A=[1, 6, 6, 6], B=[10, 5, 5, 5]
        if a[0] == 1 and a[1] == 6 and a[2] == 6:
            tallying_consistent += (1 if r == 0 else 0)
            relevant_trials += 1
            
        # Trial 3: A=[10, 0, 0, 0], B=[0, 1, 1, 1]
        elif a[0] == 10 and a[1] == 0 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
        # Trial 4: A=[0, 10, 0, 0], B=[1, 0, 1, 1]
        elif a[0] == 0 and a[1] == 10 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
    return float(tallying_consistent / relevant_trials) if relevant_trials > 0 else 0.5
```

**Observed (real) value:** 0.5056 (var=0.0050)
**Candidate (simulated) value:** 0.5028 (var=0.0089)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.1333 (var=0.0578)
- pi_3: 0.4572 (var=0.0238)

### Experiment 2
**Design**
  A=[2, 6, 6]  B=[10, 5, 5]
  A=[8, 4, 3]  B=[2, 5, 4]
  A=[10, 0, 5]  B=[0, 10, 5]
  A=[0, 10, 5]  B=[10, 0, 5]
  A=[5, 5, 5]  B=[1, 6, 6]
  A=[8, 2, 2]  B=[10, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    validities = np.array([0.9, 0.6, 0.5])
    
    score_a = a_ratings @ validities
    score_b = b_ratings @ validities
    
    wadd_pred = (score_b > score_a).astype(int)
    
    responses = data['response'].values
    return float(np.mean(responses == wadd_pred))

```

**Observed (real) value:** 0.4963 (var=0.0025)
**Candidate (simulated) value:** 0.4996 (var=0.0026)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8433 (var=0.0738)
- pi_1: 0.1667 (var=0.0000)
- pi_3: 0.5246 (var=0.0059)

### Experiment 3
**Design**
  A=[7, 2, 2]  B=[5, 9, 9]
  A=[4, 8, 8]  B=[6, 1, 1]
  A=[8, 3, 3]  B=[7, 9, 8]
  A=[5, 10, 10]  B=[8, 0, 0]
  A=[9, 1, 1]  B=[7, 8, 8]
  A=[3, 9, 9]  B=[6, 2, 2]
  A=[6, 5, 5]  B=[5, 10, 10]
  A=[2, 7, 7]  B=[4, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    cue1_a = np.array([x[0] for x in data['option_a_ratings']])
    cue1_b = np.array([x[0] for x in data['option_b_ratings']])
    
    a_better = (cue1_a > cue1_b)
    b_better = (cue1_b > cue1_a)
    
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    aligned = (a_better & chose_a) | (b_better & chose_b)
    
    return float(aligned.mean())
```

**Observed (real) value:** 0.5138 (var=0.0022)
**Candidate (simulated) value:** 0.4963 (var=0.0021)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4906 (var=0.0131)
- pi_2: 0.0250 (var=0.0206)
- pi_1: 0.0000 (var=0.0000)

### Experiment 4
**Design**
  A=[6, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[7, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[8, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[9, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[10, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[10, 2, 2, 2]  B=[0, 10, 10, 10]
  A=[3, 10, 10, 10]  B=[8, 1, 1, 1]
  A=[10, 5, 5, 5]  B=[0, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Trial 1: A=[6, 1, 1, 1], B=[5, 10, 10, 10]
    t1_mask = data['option_a_ratings'].apply(lambda x: list(x) == [6, 1, 1, 1])
    # Trial 7: A=[3, 10, 10, 10], B=[8, 1, 1, 1]
    t7_mask = data['option_a_ratings'].apply(lambda x: list(x) == [3, 10, 10, 10])
    
    t1_resp = data[t1_mask]['response'].mean()
    t7_resp = data[t7_mask]['response'].mean()
    
    if pd.isna(t1_resp):
        t1_resp = 0.5
    if pd.isna(t7_resp):
        t7_resp = 0.5
        
    return float(t1_resp - t7_resp)
```

**Observed (real) value:** 0.0033 (var=0.0308)
**Candidate (simulated) value:** -0.0267 (var=0.0362)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.0000)
- pi_3: 0.0450 (var=0.0420)
- pi_1: 1.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate implements the arbiter's recommendation to use a Random Guessing baseline. The empirical data across all four experiments demonstrates metric values hovering around 0.5 (or ~0.0 for the difference metric in Experiment 4). The candidate model's simulated values match the observed ground-truth values exceptionally well, confirming the hypothesis that subjects are essentially guessing at random when faced with these complex cardinal feature choices without trial-by-trial feedback.",
  "verdict": "continue",
  "rationale": "The candidate perfectly follows the arbiter's prescription to implement a pure Random Guessing model with a slight bias parameter. This simple model accurately captures the empirical metrics across all experiments, which cluster tightly around chance levels. The loss is very low, and no further adjustments are needed for this family."
}
```

## Usage

```json
{
  "prompt_token_count": 4497,
  "candidates_token_count": 175,
  "total_token_count": 4937
}
```
