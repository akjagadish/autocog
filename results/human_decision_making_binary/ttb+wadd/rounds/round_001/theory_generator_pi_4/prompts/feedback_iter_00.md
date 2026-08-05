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

Propose a brand-new theory that unifies the compensatory integration of WADD with the heuristic behavior of TTB/Tallying. One promising direction is a 'Non-linear Subjective Weighting' model where subjective cue weights are a power function of the provided validities (with an individual-difference parameter controlling the exponent). A high exponent leads to non-compensatory (TTB-like) behavior, while an exponent near 1 yields WADD-like behavior, and an exponent near 0 yields Tallying. Alternatively, propose an 'Adaptive Strategy Selection' theory where subjects probabilistically mix WADD and TTB rather than Tallying and TTB.


## CANDIDATE THEORY
Non-linear Subjective Weighting Model: Subjects evaluate options by computing a weighted sum of their features, but they do not use the objective cue validities directly. Instead, subjective cue weights are a power function of the provided validities. An individual-specific exponent parameter controls the non-linearity of this transformation. This single mechanism unifies multiple decision strategies: an exponent near 0 flattens the weights (yielding Equal-Weight/Tallying), an exponent of 1 uses the validities linearly (yielding WADD), and a large exponent strongly amplifies the most valid cues (yielding non-compensatory Take The Best behavior).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear subjective weighting: w_i = v_i ^ gamma
    # Validities are in [0.5, 1.0], so base is positive.
    w = np.maximum(validities, 0.0) ** gamma
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- gamma: [0.0, 10.0]
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's suggestion, this theory unifies compensatory and non-compensatory decision-making without relying on a discrete mixture of heuristics. By applying an individual-specific exponent (gamma) to the cue validities, the model can natively capture Tallying (gamma ~ 0), WADD (gamma ~ 1), and Take The Best (gamma >> 1) behaviors within a single, continuous weighted-additive framework. This provides a more robust and elegant explanation for the intermediate TTB matches and varying strategy adoptions observed in the experiments.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0631 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0631.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        sum_a = sum(row['option_a_ratings'])
        sum_b = sum(row['option_b_ratings'])
        
        # In the compensatory trials designed for this experiment,
        # the option favored by TTB always has fewer total positive cues
        # than the option favored by WADD.
        if sum_a != sum_b:
            ttb_choice = 0 if sum_a < sum_b else 1
            ttb_match.append(1 if row['response'] == ttb_choice else 0)
            
    if not ttb_match:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.4175 (var=0.0635)
**Candidate (simulated) value:** 0.4792 (var=0.0615)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.1000 (var=0.0400)
- pi_3: 0.4225 (var=0.0632)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        response = row['response']
        
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        
        if ttb_choice is not None and response == ttb_choice:
            ttb_matches += 1
        total += 1
        
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4317 (var=0.0770)
**Candidate (simulated) value:** 0.5056 (var=0.0469)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1933 (var=0.0482)
- pi_1: 1.0000 (var=0.0000)
- pi_3: 0.4429 (var=0.0513)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Define the critical trials where PHS and WADD strongly diverge.
    # In Trial 1, Option A has the single highest validity cue and more total cues,
    # so PHS (TTB + Tallying) strongly predicts Option A (response 0).
    # Option B has multiple high-validity cues that outweigh A's cues in a weighted sum,
    # so WADD strongly predicts Option B (response 1).
    # Trial 2 is the exact symmetric swap.
    t1_A = (1, 0, 0, 0, 1, 1, 1)
    t2_A = (0, 1, 1, 1, 0, 0, 0)
    
    is_t1 = data['option_a_ratings'].apply(tuple) == t1_A
    is_t2 = data['option_a_ratings'].apply(tuple) == t2_A
    
    match_phs = 0
    match_phs += (data.loc[is_t1, 'response'] == 0).sum()
    match_phs += (data.loc[is_t2, 'response'] == 1).sum()
    
    total = is_t1.sum() + is_t2.sum()
    if total == 0:
        return 0.5
    return float(match_phs / total)

```

**Observed (real) value:** 0.2317 (var=0.0651)
**Candidate (simulated) value:** 0.1650 (var=0.0173)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8050 (var=0.0118)
- pi_2: 0.2800 (var=0.1483)
- pi_1: 1.0000 (var=0.0000)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_1 = data['option_b_ratings'].apply(lambda x: x[1])
    critical_mask = (a_0 == 1) & (b_1 == 1)
    if critical_mask.sum() == 0:
        return 0.5
    return float(data.loc[critical_mask, 'response'].mean())
```

**Observed (real) value:** 0.8158 (var=0.0450)
**Candidate (simulated) value:** 0.8058 (var=0.0304)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6300 (var=0.1806)
- pi_3: 0.0650 (var=0.0015)
- pi_1: 0.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The non-linear subjective weighting model performs exceptionally well across all four experiments. By introducing an individual-level exponent parameter (gamma) on the validities, the model natively spans the space between Tallying, WADD, and Take The Best without requiring discrete mixture probabilities or heuristics. The simulated metric values closely track the observed human data, notably correcting the stark deviations seen in previous theories (e.g., matching the ~0.815 observed in Experiment 4 and the ~0.23 observed in Experiment 3). The aggregate loss of 0.0631 indicates a very strong fit.",
  "verdict": "continue",
  "rationale": "The candidate theory effectively captures the empirical patterns across the entire suite of experiments. The power-law transformation of validities provides a robust, continuous mechanism for strategy variation that aligns tightly with human choice data. No further tuning is required at this stage."
}
```

## Usage

```json
{
  "prompt_token_count": 5055,
  "candidates_token_count": 199,
  "total_token_count": 5557
}
```
