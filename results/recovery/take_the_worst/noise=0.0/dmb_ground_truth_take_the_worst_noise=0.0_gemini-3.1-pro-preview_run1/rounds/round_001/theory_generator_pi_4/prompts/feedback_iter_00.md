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

Since Tallying (pi_2) is already highly successful, the new theory should explore a nuanced boundary condition of Tallying rather than reverting to a fully compensatory WADD model. Propose a 'Take-The-Best with Tallying Fallback' (or Lexicographic Semi-Tallying) theory: people first check if the most valid cue strongly discriminates the options (perhaps only if its validity is exceptionally high or above a threshold). If it doesn't, they fall back to a simple Tallying strategy. Alternatively, propose a 'Top-K Tallying' model where individuals tally feature wins but only among the top K most valid features, ignoring the least valid ones to save cognitive effort.


## CANDIDATE THEORY
People use a Thresholded Tallying strategy to save cognitive effort: they do not integrate all available information, nor do they rely on just a single cue. Instead, they ignore cues with low validities and only tally wins on features that meet a certain subjective validity threshold. This combines the compensatory, robust nature of Tallying with the frugality of heuristic decision-making, allowing individuals to filter out weak evidence before counting feature-wise wins.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["validity_threshold"])
    
    # Filter cues based on subjective validity threshold
    valid_cues = val >= threshold
    # Fallback to all cues if the threshold is too strict to leave any cues
    if not np.any(valid_cues):
        valid_cues = np.ones_like(val, dtype=bool)
        
    # Tally strict feature-wise wins only among the valid cues
    a_wins = float(np.sum((a > b) & valid_cues))
    b_wins = float(np.sum((b > a) & valid_cues))
    
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validity_threshold: [0.5, 1.0]
- validities: validities

`rationale`:
Given the high success of the Tallying model (pi_2) and the poor performance of pure Take-The-Best (pi_1) and WADD (pi_3), it's clear that human behavior in these tasks is highly compensatory but insensitive to exact validity weights. The arbiter rightly suggests exploring a boundary condition of Tallying. This Thresholded Tallying model introduces a 'validity_threshold' parameter: individuals ignore low-validity cues to save cognitive effort and only tally feature wins among the subset of cues they deem sufficiently reliable. If the threshold is low, this gracefully collapses back to pure Tallying, preserving its excellent fit while allowing for nuanced exclusion of weak cues.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.6738 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.6738.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    ttb_choices = np.zeros(len(data))
    mask = np.zeros(len(data), dtype=bool)
    
    for i in range(len(data)):
        a = a_mat[i]
        b = b_mat[i]
        
        # TTB choice (first discriminating cue)
        ttb_c = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_c = 0
                break
            elif b[j] > a[j]:
                ttb_c = 1
                break
                
        # Tallying choice (most feature wins)
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        tally_c = -1
        if a_wins > b_wins:
            tally_c = 0
        elif b_wins > a_wins:
            tally_c = 1
            
        # Only consider trials where TTB and Tallying make strictly opposing predictions
        if ttb_c != -1 and tally_c != -1 and ttb_c != tally_c:
            mask[i] = True
            ttb_choices[i] = ttb_c
            
    if not np.any(mask):
        return 0.5
        
    return float(np.mean(resp[mask] == ttb_choices[mask]))
```

**Observed (real) value:** 0.1383 (var=0.0087)
**Candidate (simulated) value:** 0.4192 (var=0.0681)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8742 (var=0.0109)
- pi_2: 0.1297 (var=0.0093)
- pi_3: 0.1503 (var=0.0082)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

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
        
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue
            
        matches.append(row['response'] == tally_pred)
        
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8286 (var=0.0105)
**Candidate (simulated) value:** 0.5643 (var=0.0631)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8479 (var=0.0103)
- pi_1: 0.1536 (var=0.0070)
- pi_3: 0.8264 (var=0.0085)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    df = data.copy()
    df['A_str'] = df['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    df['B_str'] = df['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Trial 1: A='11000', B='00111' -> WADD prefers A (1.9 vs 1.6), Tally prefers B (2 vs 3)
    # Trial 2: A='00111', B='11000' -> WADD prefers B (1.6 vs 1.9), Tally prefers A (3 vs 2)
    
    t1 = df[(df['A_str'] == '11000') & (df['B_str'] == '00111')]
    t2 = df[(df['A_str'] == '00111') & (df['B_str'] == '11000')]
    
    score = 0.0
    n = 0
    
    if len(t1) > 0:
        score += (t1['response'] == 0).sum()
        n += len(t1)
    if len(t2) > 0:
        score += (t2['response'] == 1).sum()
        n += len(t2)
        
    if n == 0:
        return 0.5
    return float(score / n)

```

**Observed (real) value:** 0.1333 (var=0.0128)
**Candidate (simulated) value:** 0.8525 (var=0.0088)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7842 (var=0.0284)
- pi_2: 0.1667 (var=0.0219)
- pi_1: 0.8317 (var=0.0133)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    mask1 = (a_str == '11000') & (b_str == '00111')
    mask2 = (a_str == '00111') & (b_str == '11000')
    
    wadd_chosen = 0
    total = 0
    
    if mask1.sum() > 0:
        wadd_chosen += (data.loc[mask1, 'response'] == 0).sum()
        total += mask1.sum()
        
    if mask2.sum() > 0:
        wadd_chosen += (data.loc[mask2, 'response'] == 1).sum()
        total += mask2.sum()
        
    if total == 0:
        return 0.5
        
    return float(wadd_chosen / total)
```

**Observed (real) value:** 0.1956 (var=0.0240)
**Candidate (simulated) value:** 0.8622 (var=0.0235)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1511 (var=0.0177)
- pi_3: 0.7733 (var=0.0291)
- pi_1: 0.8333 (var=0.0283)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate's Thresholded Tallying model is performing poorly compared to pure Tallying (pi_2). On Experiments 3 and 4, human subjects score around 0.15 (strongly favoring Tallying), but the candidate scores ~0.85 (mimicking TTB or WADD). This happens because the validity threshold (0.5 to 1.0) typically filters out all but the top 1 or 2 cues. By tallying only the top cues, the model effectively degenerates into Take-The-Best, completely losing the robust compensatory nature of Tallying that fits the human data.",
  "verdict": "regenerate",
  "rationale": "The current thresholding mechanism is too aggressive, discarding too many cues and causing the model to behave like TTB. To stay within the arbiter's prescribed family while fixing this, implement the 'Take-The-Best with Tallying Fallback' more explicitly. Create a parameter `ttb_threshold` with a high range (e.g., [0.8, 1.0]). The decision process should be: check the single most valid cue. IF its validity is above `ttb_threshold` AND it discriminates between A and B, use it (TTB). OTHERWISE, fall back to tallying ALL available cues. This ensures the model defaults to full Tallying in most scenarios (preserving the excellent fit of pi_2) and only acts lexicographically when a cue is overwhelmingly reliable."
}
```

## Usage

```json
{
  "prompt_token_count": 5589,
  "candidates_token_count": 326,
  "total_token_count": 6905
}
```
