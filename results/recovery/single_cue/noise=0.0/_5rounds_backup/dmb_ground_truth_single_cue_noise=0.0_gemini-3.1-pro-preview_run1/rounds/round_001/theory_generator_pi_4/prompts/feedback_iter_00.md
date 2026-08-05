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

Theory 1 (WADD) is fundamentally degenerate for this dataset because subjects clearly do not integrate all validities in a compensatory manner. Propose a brand-new theory that acts as a stronger competitor to Tallying. For example, you could propose 'Validity-Biased Tallying' (where subjects primarily count wins/losses but use the validity of the most predictive cue to break ties or softly weight the tally), or a 'Take-The-Majority' rule with an attention mechanism that drops low-validity cues. This new theory should remain largely based on counting, but introduce a bounded or simplified use of the validities to see if it can capture the slight deviations from pure Tallying observed in Experiment 4.


## CANDIDATE THEORY
Validity-Biased Tallying: People primarily compare options by counting the number of features where one option strictly dominates the other (Tallying), ignoring the magnitudes of the differences and the exact cue validities. However, they are not completely blind to cue validities. Instead of integrating all validities in a compensatory manner (WADD) or strictly following a non-compensatory rule (Take The Best), they give a soft bonus weight to the single most predictive cue. This validity bias acts primarily as a tie-breaker when the tallies are close or equal, slightly skewing choice probabilities toward the option endorsed by the best cue, while still preserving the overall dominance of the Tallying heuristic. This explains why human choices overwhelmingly follow Tallying but show slight deviations toward the most valid cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Validity-Biased Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Bonus for winning the most valid cue
    val = np.asarray(parameters["validities"], dtype=float)
    best_cue = np.argmax(val)
    
    w = float(parameters["w"])
    a_best_win = float(a[best_cue] > b[best_cue])
    b_best_win = float(b[best_cue] > a[best_cue])
    
    score_a = a_wins + w * a_best_win
    score_b = b_wins + w * b_best_win
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w: [0.0, 2.0]
- validities: validities

`rationale`:
Following the arbiter's diagnosis, the previous WADD theory failed because it assumed full compensatory integration of all validities, which drastically mispredicted human data that overwhelmingly aligns with Tallying. To provide a stronger competitor to pure Tallying, this 'Validity-Biased Tallying' theory assumes subjects primarily count wins and losses (Tallying) but apply a bounded use of validities by adding a soft bonus weight ('w') to the option that wins on the single most predictive cue. This allows the model to capture the slight deviations from pure Tallying observed in the experiments without falling into the degenerate predictions of WADD or the overly frugal predictions of Take The Best.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.2939 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.2939.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        # Only consider trials where Tallying does not predict a tie
        if a_wins != b_wins:
            ttb_pred = None
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_pred = 0
                    break
                elif b[i] > a[i]:
                    ttb_pred = 1
                    break
            
            if ttb_pred is not None:
                if resp == ttb_pred:
                    ttb_matches += 1
                total += 1
                
    return float(ttb_matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.1194 (var=0.0110)
**Candidate (simulated) value:** 0.1391 (var=0.0093)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8616 (var=0.0098)
- pi_2: 0.1291 (var=0.0066)
- pi_3: 0.1731 (var=0.0077)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    total = 0
    
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
            
        if row['response'] == tally_pred:
            matches += 1
        total += 1
        
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.7608 (var=0.0062)
**Candidate (simulated) value:** 0.8427 (var=0.0078)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8690 (var=0.0080)
- pi_1: 0.3217 (var=0.0032)
- pi_3: 0.8319 (var=0.0117)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_match = 0
    count = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        resp = row['response']
        
        # Trial 1: A has top 2 validities (0.9+0.8=1.7), B has bottom 3 (0.6+0.5+0.5=1.6)
        # WADD prefers A (0), Tallying prefers B (1) since B wins 3-2
        if a == (1, 1, 0, 0, 0):
            if resp == 0:
                wadd_match += 1
            count += 1
        # Trial 2: A has bottom 3, B has top 2
        # WADD prefers B (1), Tallying prefers A (0)
        elif a == (0, 0, 1, 1, 1):
            if resp == 1:
                wadd_match += 1
            count += 1
            
    if count == 0:
        return 0.5
    return float(wadd_match / count)
```

**Observed (real) value:** 0.1562 (var=0.0200)
**Candidate (simulated) value:** 0.5138 (var=0.1077)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6056 (var=0.0172)
- pi_2: 0.1475 (var=0.0146)
- pi_1: 0.8644 (var=0.0082)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Tallying predicts B (3 wins vs 2) but WADD predicts A (1.8 vs 1.65)
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if not is_trial_1.any():
        return 0.0
    # Return the proportion of times option A was chosen (response == 0)
    return float((data.loc[is_trial_1, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1533 (var=0.0127)
**Candidate (simulated) value:** 0.5058 (var=0.1141)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1467 (var=0.0111)
- pi_3: 0.7233 (var=0.0238)
- pi_1: 0.8508 (var=0.0143)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory 'Validity-Biased Tallying' correctly captures the overall dominance of Tallying in Experiments 1 and 2. However, in Experiments 3 and 4, the model predicts a much higher match with WADD/TTB (around 0.50) than observed in human data (around 0.15). In these trials, Tallying prefers one option by a 1-point margin (e.g., 3 wins vs 2 wins), but the other option wins the highest validity cue. Because the parameter 'w' can go up to 2.0, the model allows the validity bonus to completely override a 1-point tally difference (e.g., 2 + 2.0 = 4.0 > 3.0). This violates the intention that the validity bias should act primarily as a tie-breaker or soft bias, causing the model to over-predict choices aligned with the most valid cue.",
  "verdict": "regenerate",
  "rationale": "The candidate is conceptually on the right track and was ACCEPTED by the loop, but its parameter bounds allow it to violate the core premise of the theory. To ensure the validity bonus 'w' acts strictly as a tie-breaker or a very soft bias that cannot override a strict tally advantage, restrict the upper bound of 'w' to be less than 1.0 (e.g., `[0.0, 0.9]`). This will force the model to preserve Tallying dominance when there is a clear winner, while still breaking ties using the best cue."
}
```

## Usage

```json
{
  "prompt_token_count": 4964,
  "candidates_token_count": 347,
  "total_token_count": 6043
}
```
