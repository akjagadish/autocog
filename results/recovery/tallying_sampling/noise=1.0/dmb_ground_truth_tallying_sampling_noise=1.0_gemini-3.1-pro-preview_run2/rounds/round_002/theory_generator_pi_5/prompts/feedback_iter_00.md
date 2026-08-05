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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a 'Random Choice (Pure Guessing) Theory' or a 'Weighted Additive (WADD) with Extreme Noise Theory'. If subjects find the fictitious validities too difficult to integrate, they might resort to pure guessing, making choices completely randomly. Alternatively, a WADD model that linearly weights cues by their provided validities combined with a very high lapse rate might perfectly balance the conflicting cues in these specific experimental designs, resulting in exactly 0.50 predictions without relying on a non-compensatory heuristic like TTB.


## CANDIDATE THEORY
Weighted Additive (WADD) with Extreme Noise Theory: Subjects attempt to integrate all available cues by weighting them according to their provided validities. However, the cognitive demand of integrating multiple conflicting fictitious validities is overwhelming. This results in extreme decision conflict and distraction, leading to a near-total reliance on guessing (a very high lapse rate). Consequently, choice probabilities are pulled almost entirely toward chance (0.50), masking the underlying compensatory process in the aggregate behavioral data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Weighted sum per option (dot product with per-feature weights)
    scores = stim @ (validities * w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores with max-subtraction for stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with extreme uniform lapse (guessing)
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.0, 2.0]
- epsilon: [0.95, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

`rationale`:
Following the arbiter's suggestion, this theory implements a Weighted Additive (WADD) model combined with an extreme noise component (lapse rate between 0.95 and 1.0). The previous TTB with Extreme Noise model (pi_4) performed perfectly by recognizing that choices in these experiments are heavily dominated by guessing, clustering around 0.50. This new model tests the hypothesis that subjects might be attempting a fully compensatory integration of cues rather than a non-compensatory one, but the extreme difficulty of the task still drives them to guess almost all the time. This perfectly balances conflicting cues and reproduces the ~0.50 choice proportions across all metrics without relying on a one-reason heuristic.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0163 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0163.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the total number of positive cues for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify critical compensatory trials where Option B has more positive cues overall,
    # but Option A possesses the single most valid discriminating cue.
    mask = sum_b > sum_a
    
    if not mask.any():
        return 0.0
        
    # Return the proportion of times the subject chose Option B (response == 1)
    # TTB will consistently choose A (yielding ~0.0) because it only looks at the top valid cue.
    # WADD will frequently choose B (yielding a higher value) because the numerous lower-validity cues compensate.
    return float(data.loc[mask, 'response'].mean())

```

**Observed (real) value:** 0.4961 (var=0.0021)
**Candidate (simulated) value:** 0.4986 (var=0.0052)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1519 (var=0.0070)
- pi_2: 0.7075 (var=0.0196)
- pi_3: 0.5386 (var=0.0041)
- pi_4: 0.4650 (var=0.0031)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # The experimental design is set up such that Take The Best (TTB) almost
    # always chooses option A (response = 0) because option A always has the
    # advantage on the single most valid discriminating cue.
    # Conversely, Weighted Additive (WADD) will frequently choose option B 
    # (response = 1) because option B has a large number of lower-validity 
    # cues that cumulatively outweigh option A's single best cue.
    # Thus, the simple overall proportion of choosing option B perfectly 
    # discriminates the two theories.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4996 (var=0.0028)
**Candidate (simulated) value:** 0.4952 (var=0.0015)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5729 (var=0.0092)
- pi_1: 0.1487 (var=0.0133)
- pi_3: 0.5337 (var=0.0030)
- pi_4: 0.4794 (var=0.0033)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Create a string representation for grouping by trial type
    data = data.copy()
    data['trial_key'] = data.apply(lambda row: str(tuple(row['option_a_ratings'])) + '_' + str(tuple(row['option_b_ratings'])), axis=1)
    
    # Calculate the mean response (probability of choosing B) for each subject and trial type
    subject_trial_means = data.groupby(['subject_id', 'trial_key'])['response'].mean().reset_index()
    
    # Calculate the absolute deviation from 0.5 (guessing)
    subject_trial_means['dev'] = (subject_trial_means['response'] - 0.5).abs()
    
    # Average across trial types for each subject, then average over subjects
    return float(subject_trial_means.groupby('subject_id')['dev'].mean().mean())
```

**Observed (real) value:** 0.1071 (var=0.0007)
**Candidate (simulated) value:** 0.1071 (var=0.0008)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1165 (var=0.0011)
- pi_2: 0.2760 (var=0.0095)
- pi_1: 0.3713 (var=0.0089)
- pi_4: 0.1092 (var=0.0010)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    wadd_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        score_a = np.dot(a, validities)
        score_b = np.dot(b, validities)
        if score_a > score_b:
            pred = 0
        elif score_b > score_a:
            pred = 1
        else:
            continue
        if row['response'] == pred:
            wadd_matches += 1
        total += 1
    return wadd_matches / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5065 (var=0.0027)
**Candidate (simulated) value:** 0.4983 (var=0.0023)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6055 (var=0.0238)
- pi_3: 0.4861 (var=0.0026)
- pi_1: 0.5535 (var=0.0021)
- pi_4: 0.5038 (var=0.0027)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    tally_prediction = (sum_b > sum_a).astype(int)
    return float((data['response'] == tally_prediction).mean())
```

**Observed (real) value:** 0.4992 (var=0.0026)
**Candidate (simulated) value:** 0.4996 (var=0.0033)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5390 (var=0.0042)
- pi_4: 0.4765 (var=0.0029)
- pi_1: 0.1846 (var=0.0131)
- pi_2: 0.7373 (var=0.0303)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    
    ttb_prefs = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(a_mat.shape[1]):
            if diff[i, j] > 0:
                ttb_prefs[i] = 0
                break
            elif diff[i, j] < 0:
                ttb_prefs[i] = 1
                break
                
    responses = data['response'].values
    return float(np.mean(responses == ttb_prefs))
```

**Observed (real) value:** 0.5071 (var=0.0025)
**Candidate (simulated) value:** 0.4996 (var=0.0025)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5150 (var=0.0028)
- pi_3: 0.4329 (var=0.0059)
- pi_1: 0.8646 (var=0.0113)
- pi_2: 0.3000 (var=0.0176)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory implements a Weighted Additive (WADD) model with an extreme lapse rate (epsilon between 0.95 and 1.0), capturing the hypothesis that subjects are overwhelmed by the cognitive demand of integrating conflicting fictitious validities and resort almost entirely to guessing. The simulated values across all six experiments match the observed human data exceptionally well, reproducing the ~0.50 choice proportions and the specific deviation metric in Experiment 3 perfectly. The aggregate loss is very low (0.0163), indicating an excellent fit to the empirical pattern.",
  "verdict": "continue",
  "rationale": "The current WADD with Extreme Noise model successfully captures the empirical data, which overwhelmingly reflects pure guessing behavior (choice proportions around 0.50) in these specific experimental designs. The extreme lapse parameter correctly pulls the predictions to chance, masking any underlying compensatory process and aligning perfectly with the aggregate behavioral metrics. The model is good enough and should be retained."
}
```

## Usage

```json
{
  "prompt_token_count": 6029,
  "candidates_token_count": 210,
  "total_token_count": 6540
}
```
