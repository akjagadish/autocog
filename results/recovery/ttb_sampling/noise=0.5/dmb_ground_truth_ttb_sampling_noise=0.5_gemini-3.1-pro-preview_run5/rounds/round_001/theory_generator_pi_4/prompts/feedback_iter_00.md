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
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose a 'Strategy Selection' or 'Mixture' theory. The current data strongly supports a non-compensatory heuristic like Take The Best, but the slight deviations suggest some subjects or some trials use a compensatory rule. The new theory should posit that decision-makers probabilistically mix between a lexicographic strategy (TTB) and a compensatory strategy (such as Tallying or WADD) based on a strategy-selection parameter, or that the population consists of a mixture of TTB users and compensatory users.


## CANDIDATE THEORY
Strategy Selection Theory: Individuals probabilistically mix between a non-compensatory lexicographic heuristic (Take The Best) and a compensatory heuristic (Tallying). Depending on individual differences or trial-by-trial strategy selection, a decision-maker relies on the single best discriminating cue a certain fraction of the time, and otherwise considers the unweighted sum of all feature differences. This accounts for both the strong baseline reliance on TTB and the subtle deviations towards compensatory behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Take The Best (TTB) prediction
    validities = np.asarray(parameters["validities"], dtype=float)
    order = np.argsort(validities)[::-1]
    
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Tallying prediction
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_tally = e / e.sum()
    
    # Mix strategies
    w_ttb = float(parameters["w_ttb"])
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Add lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- w_ttb: [0.0, 1.0]
- validities: validities

`rationale`:
The arbiter noted that while behavior is largely driven by a non-compensatory heuristic like Take The Best (TTB), deviations suggest some use of a compensatory rule. By introducing a mixture parameter (w_ttb) that interpolates between TTB and Tallying, this model can capture the primary lexicographic reliance while allowing for compensatory integration on trials where TTB alone fails to explain subject responses. Tallying is used as the compensatory rule for simplicity and robustness, avoiding the explosion of free parameters that WADD would entail.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.2317 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.2317.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_consistent = 0
    total_relevant = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: Tallying prefers A (count 3 vs 2), WADD prefers B (weight 1.8 vs 1.7)
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if row['response'] == 0:
                tally_consistent += 1
            total_relevant += 1
        # Trial 2: Tallying prefers B (count 3 vs 2), WADD prefers A (weight 1.8 vs 1.7)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if row['response'] == 1:
                tally_consistent += 1
            total_relevant += 1
            
    if total_relevant == 0:
        return 0.5
    return float(tally_consistent / total_relevant)
```

**Observed (real) value:** 0.3683 (var=0.0161)
**Candidate (simulated) value:** 0.4850 (var=0.0623)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8408 (var=0.0182)
- pi_2: 0.5383 (var=0.0797)
- pi_3: 0.2442 (var=0.0244)

### Experiment 2
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1))
    is_t4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0))
    
    m1 = data.loc[is_t1, 'response'].mean()
    m4 = data.loc[is_t4, 'response'].mean()
    
    if pd.isna(m1):
        m1 = 0.5
    if pd.isna(m4):
        m4 = 0.5
        
    return float(m1 - m4)
```

**Observed (real) value:** 0.3600 (var=0.0410)
**Candidate (simulated) value:** 0.0367 (var=0.2203)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0733 (var=0.4268)
- pi_1: -0.6600 (var=0.0611)
- pi_3: 0.5000 (var=0.1061)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    a_sum = data['option_a_ratings'].apply(sum)
    mask = b_sum > a_sum
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.3228 (var=0.0034)
**Candidate (simulated) value:** 0.4672 (var=0.0586)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2447 (var=0.0248)
- pi_2: 0.6783 (var=0.0163)
- pi_1: 0.8519 (var=0.0086)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    preds = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        pred = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        preds.append(pred)
    matches = (data['response'] == preds).mean()
    return float(matches)
```

**Observed (real) value:** 0.6746 (var=0.0031)
**Candidate (simulated) value:** 0.6027 (var=0.0219)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4290 (var=0.0142)
- pi_3: 0.7031 (var=0.0172)
- pi_1: 0.3683 (var=0.0021)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate implements the prescribed Strategy Selection mixture model using Tallying as the compensatory rule. However, it severely misses the empirical pattern in Experiment 2 (candidate: 0.0367 vs real: 0.3600) and pulls predictions too close to 0.5 across the board. In Experiment 2, the metric measures the difference in choice probability when TTB and the compensatory rule conflict. The observed value of 0.36 indicates a strong but imperfect reliance on TTB. The candidate's near-zero value implies that the Tallying predictions are overly aggressive and washing out the TTB signal when mixed, or that Tallying is simply the wrong compensatory rule to mix with.",
  "verdict": "regenerate",
  "rationale": "The candidate correctly implements the mixture framework but chooses Tallying (unweighted sum) as the compensatory component, which ignores cue validities entirely. This leads to a poor fit, especially in Experiment 2 where validity weights likely matter even when subjects deviate from TTB. Switch the compensatory strategy from Tallying to WADD (Weighted Additive), using the provided `validities` as weights for the feature differences. Additionally, constrain `epsilon` to a smaller range (e.g., [0.0, 0.2]) to prevent the model from using the lapse rate to flatten predictions excessively."
}
```

## Usage

```json
{
  "prompt_token_count": 4912,
  "candidates_token_count": 298,
  "total_token_count": 6201
}
```
