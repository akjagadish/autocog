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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a 'Position Bias / Task Disengagement' theory. The new theory should state that in the absence of trial-by-trial feedback and facing complex multi-cue fictitious products, subjects do not integrate or search through features. Instead, they fall back on stimulus-independent response strategies, such as a strong position bias (e.g., always selecting Option A) or random guessing. The model should predict choices based solely on spatial or temporal position (ignoring the 'validities' and feature vectors entirely) with parameters controlling the strength and direction of the spatial bias.


## CANDIDATE THEORY
Position Bias / Task Disengagement: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects often disengage from effortful feature integration or search. Instead, they fall back on stimulus-independent response strategies, such as a strong position bias (e.g., a tendency to select the first option, Option A) or random guessing. Decisions are driven entirely by these spatial/temporal biases, and the actual feature validities and values are ignored.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the stimulus and history entirely.
    # It predicts based solely on a spatial position bias for Option A.
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


`parameters`:
- bias_a: [0.0, 1.0]

`rationale`:
The arbiter feedback indicates that the previous mechanistic models (which assume subjects actively process features via TTB or Tallying) fail because the empirical alignment scores hover around chance (0.50). This suggests that subjects are not systematically using the features. By implementing a 'Position Bias / Task Disengagement' theory, the model predicts choices based purely on a spatial bias (or random guessing if bias_a ~ 0.5). This naturally captures the experiment-invariant finding that subjects' choices are largely uncorrelated with the feature-based heuristic predictions.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0243 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0243.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    preds = []
    for i in range(len(data)):
        a_vec = data['option_a_ratings'].iloc[i]
        b_vec = data['option_b_ratings'].iloc[i]
        pred = -1
        for j in range(len(a_vec)):
            if a_vec[j] > b_vec[j]:
                pred = 0
                break
            elif b_vec[j] > a_vec[j]:
                pred = 1
                break
        preds.append(pred)
        
    return float(np.mean(data['response'].values == np.array(preds)))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate (simulated) value:** 0.5046 (var=0.0014)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8444 (var=0.0101)
- pi_2: 0.1360 (var=0.0099)
- pi_3: 0.3513 (var=0.0200)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            pred = 0
        elif b_wins > a_wins:
            pred = 1
        else:
            continue
            
        if row['response'] == pred:
            match_count += 1
        total += 1
        
    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate (simulated) value:** 0.5079 (var=0.0017)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8760 (var=0.0081)
- pi_1: 0.1310 (var=0.0075)
- pi_3: 0.6421 (var=0.0265)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_ttb_aligned(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_pred = 0 if a[i] > b[i] else 1
                return row['response'] == ttb_pred
        return False
        
    return float(data.apply(is_ttb_aligned, axis=1).mean())
```

**Observed (real) value:** 0.4250 (var=0.1350)
**Candidate (simulated) value:** 0.4788 (var=0.0521)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8413 (var=0.0121)
- pi_3: 0.2840 (var=0.0195)
- pi_2: 0.1325 (var=0.0080)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                break
    return float(matches / len(data))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate (simulated) value:** 0.4971 (var=0.0016)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2888 (var=0.0249)
- pi_1: 0.8498 (var=0.0105)
- pi_2: 0.1650 (var=0.0102)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate model successfully implements the 'Position Bias / Task Disengagement' theory prescribed by the arbiter. Across all four experiments, the observed empirical values for the various feature-integration heuristics (TTB, Tallying) are near 0.50 (chance) or slightly below, indicating that subjects are not systematically using these features and are likely guessing or exhibiting a simple position bias. The candidate model simulates this perfectly, producing alignment metrics around 0.50 (e.g., 0.5046 vs 0.5000 in Exp 1, 0.5079 vs 0.5000 in Exp 2, 0.4971 vs 0.5000 in Exp 4). It also captures the trend in Exp 3 reasonably well (0.4788 vs 0.4250).",
  "verdict": "continue",
  "rationale": "The candidate theory fits the empirical data extremely well and faithfully executes the arbiter's recommendation. The remarkably low loss and close correspondence to the observed chance-level heuristic alignment confirm that subjects are likely disengaged or relying on spatial biases rather than complex multi-cue integration. No further tuning is necessary."
}
```

## Usage

```json
{
  "prompt_token_count": 4561,
  "candidates_token_count": 273,
  "total_token_count": 5281
}
```
