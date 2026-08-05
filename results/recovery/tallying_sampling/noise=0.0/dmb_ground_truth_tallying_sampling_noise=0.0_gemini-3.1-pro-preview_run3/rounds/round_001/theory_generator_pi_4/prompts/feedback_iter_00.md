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

Theory 1 (Tallying) is currently dominating and perfectly explains the data. Theory 2 (WADD) is degenerate for this domain since subjects do not appear to use the provided validities in a fully compensatory, magnitude-sensitive way. Propose a brand-new theory to replace Theory 2 that offers a different non-compensatory or boundedly rational mechanism. For example, consider 'Majority of Confirming Dimensions' (which focuses on pairwise feature differences rather than independent sums) or a 'Rank-Weighted Heuristic' (where cues are weighted solely by their ordinal rank rather than explicit validity magnitudes) to see if a slightly more sophisticated heuristic can challenge the pure Equal Weight model.


## CANDIDATE THEORY
Majority of Confirming Dimensions (MCD): People make decisions by comparing options dimension by dimension rather than computing holistic values. On each feature, the option with the superior value earns a point (a 'confirming dimension'). The option with the most points across all features is chosen. For binary features, this dimension-wise comparison yields the same relative preference as Tallying, but through a fundamentally different cognitive process (pairwise comparisons rather than independent counting).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("MCD expects a (2, n_features) state.")
        
    a, b = stim[0], stim[1]
    
    # MCD: Count how many features strictly favor each option
    score_a = np.sum(a > b)
    score_b = np.sum(b > a)
    scores = np.array([score_a, score_b], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over MCD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

`rationale`:
Following the arbiter's suggestion, we introduce the Majority of Confirming Dimensions (MCD) heuristic. Rather than evaluating options holistically (like WADD or Tallying), MCD compares options dimension-by-dimension. It counts how many features strictly favor option A versus option B. In environments with binary features, this boundedly rational, non-compensatory pairwise comparison mathematically mirrors the relative evidence of Tallying, thus perfectly capturing the human data while offering a distinct, cognitively plausible mechanistic explanation based on dimensional comparisons.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0317 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0317.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        if ttb_winner is not None:
            matches.append(1.0 if resp == ttb_winner else 0.0)
            
    return float(np.mean(matches)) if matches else 0.5

```

**Observed (real) value:** 0.2787 (var=0.0031)
**Candidate (simulated) value:** 0.2873 (var=0.0037)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8596 (var=0.0092)
- pi_2: 0.3973 (var=0.0146)
- pi_3: 0.2789 (var=0.0049)

### Experiment 2
**Design**
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.65, 0.95, 0.55, 0.8])
    cue_order = [1, 3, 0, 2]
    
    wadd_aligned = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        wadd_a = np.dot(a, val)
        wadd_b = np.dot(b, val)
        
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
                
        if ttb_winner is None:
            continue
            
        wadd_winner = 0 if wadd_a > wadd_b else (1 if wadd_b > wadd_a else None)
        
        if wadd_winner is not None and ttb_winner != wadd_winner:
            resp = row['response']
            if resp == wadd_winner:
                wadd_aligned += 1
            total += 1
            
    if total == 0:
        return 0.5
    return wadd_aligned / total
```

**Observed (real) value:** 0.8967 (var=0.0081)
**Candidate (simulated) value:** 0.8467 (var=0.0119)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6538 (var=0.0294)
- pi_1: 0.1525 (var=0.0116)
- pi_3: 0.8508 (var=0.0095)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_consistent = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: Tally prefers A, WADD prefers B
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            tally_consistent.append(1 if row['response'] == 0 else 0)
        # Trial 2: Tally prefers B, WADD prefers A
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            tally_consistent.append(1 if row['response'] == 1 else 0)
            
    if not tally_consistent:
        return 0.5
    return float(np.mean(tally_consistent))
```

**Observed (real) value:** 0.8500 (var=0.0104)
**Candidate (simulated) value:** 0.8642 (var=0.0141)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8383 (var=0.0147)
- pi_2: 0.4933 (var=0.1001)
- pi_1: 0.1425 (var=0.0124)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    mask = (
        data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) &
        data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    )
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8489 (var=0.0265)
**Candidate (simulated) value:** 0.8667 (var=0.0217)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4067 (var=0.1005)
- pi_3: 0.8667 (var=0.0198)
- pi_1: 0.1600 (var=0.0179)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory (Majority of Confirming Dimensions, MCD) provides an excellent fit to the empirical data across all four experiments. By counting the number of features that strictly favor each option, it mathematically mirrors the predictions of Tallying (pi_3) but offers a distinct, cognitively plausible mechanism based on dimension-by-dimension pairwise comparisons. The simulated point estimates are extremely close to the observed human data, confirming that subjects' behavior is well-described by this non-compensatory heuristic.",
  "verdict": "continue",
  "rationale": "The MCD candidate successfully captures the human behavior across all experiments, achieving a very low loss and closely tracking the observed metric values. It faithfully implements the arbiter's recommendation for a boundedly rational heuristic, so no further tuning is necessary. The theory should be accepted as the current best model."
}
```

## Usage

```json
{
  "prompt_token_count": 5110,
  "candidates_token_count": 184,
  "total_token_count": 5609
}
```
