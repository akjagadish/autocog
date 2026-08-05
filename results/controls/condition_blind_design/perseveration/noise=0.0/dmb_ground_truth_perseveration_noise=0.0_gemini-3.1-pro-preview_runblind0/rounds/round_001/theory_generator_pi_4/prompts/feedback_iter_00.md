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

Propose a 'Noisy Take-The-Best' (Lexicographic) theory. In this theory, decision-makers compare options sequentially starting from the most valid cue. However, to account for the ~0.50 observed data, assume that the execution of this heuristic is highly stochastic (e.g., a high lapse rate or a probability of stopping/guessing early). This provides a classic non-compensatory heuristic competitor to the compensatory WADD model, while still acknowledging the massive response noise evident in the human data.


## CANDIDATE THEORY
Noisy Take-The-Best: Decision-makers evaluate options lexicographically, starting with the most valid cue and stopping at the first cue that discriminates between the options. However, the execution of this heuristic is highly stochastic, characterized by a massive lapse rate. Because the observed aggregate behavior is very close to random guessing (~0.50), the theory posits that subjects frequently suffer from attentional lapses or guess uniformly, effectively washing out the deterministic TTB predictions with heavy noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_core = np.ones(2) / 2.0
    else:
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 5.0]
- epsilon: [0.7, 1.0]
- validities: validities

`rationale`:
Following the arbiter's suggestion, this model instantiates a 'Noisy Take-The-Best' (Lexicographic) theory. It uses the standard TTB stopping rule (checking cues in descending validity order until one discriminates) but incorporates massive response noise to capture the ~0.50 empirical metrics across experiments. By constraining the lapse rate `epsilon` to a very high range [0.7, 1.0], the model acknowledges the heavy stochasticity in human data, providing a non-compensatory heuristic alternative to WADD while correcting the deterministic failure of the previous TTB model.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.1316 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.1316.

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
    # Identify trials where TTB and Tallying make opposite deterministic predictions.
    # Specifically, Trial 3: A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
    # TTB: Feature 0 (highest validity) favors A, so TTB chooses A.
    # Tallying: A wins on Feature 0, but B wins on Features 1 and 2. Tallying chooses B.
    is_reversal_trial = data.apply(
        lambda row: tuple(row['option_a_ratings']) == (1, 0, 0, 1) and tuple(row['option_b_ratings']) == (0, 1, 1, 1),
        axis=1
    )
    if is_reversal_trial.sum() == 0:
        return 0.5
    
    # Return the proportion of times option A was chosen on this trial.
    # TTB will have a value close to 1, while Tallying will have a value close to 0.
    return float((data.loc[is_reversal_trial, 'response'] == 0).mean())
```

**Observed (real) value:** 0.4000 (var=0.2400)
**Candidate (simulated) value:** 0.5433 (var=0.0387)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9000 (var=0.0167)
- pi_2: 0.1567 (var=0.0260)
- pi_3: 0.3467 (var=0.0398)

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
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    is_trial_8 = (a_str == '0101') & (b_str == '1100')
    is_trial_11 = (a_str == '1101') & (b_str == '1011')
    
    ttb_consistent_8 = data[is_trial_8]['response'] == 1
    ttb_consistent_11 = data[is_trial_11]['response'] == 0
    
    total_trials = len(ttb_consistent_8) + len(ttb_consistent_11)
    if total_trials == 0:
        return 0.5
        
    ttb_consistent_count = ttb_consistent_8.sum() + ttb_consistent_11.sum()
    return float(ttb_consistent_count / total_trials)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate (simulated) value:** 0.5600 (var=0.0272)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5150 (var=0.0221)
- pi_1: 0.8617 (var=0.0152)
- pi_3: 0.5383 (var=0.0262)

### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['pair'] = data['A_str'] + data['B_str']
    
    tally_0 = {'11010001', '01101000', '11110000', '11111101', '00110001', '11100101', '10111001', '01100100'}
    tally_1 = {'10101011', '00010101', '10100111', '11001101', '11001110'}
    
    is_0 = data['pair'].isin(tally_0)
    is_1 = data['pair'].isin(tally_1)
    
    matches = (is_0 & (data['response'] == 0)) | (is_1 & (data['response'] == 1))
    valid = is_0 | is_1
    
    return float(matches.sum() / valid.sum()) if valid.sum() > 0 else 0.5
```

**Observed (real) value:** 0.5046 (var=0.0133)
**Candidate (simulated) value:** 0.5374 (var=0.0032)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6451 (var=0.0098)
- pi_2: 0.8569 (var=0.0102)
- pi_1: 0.7438 (var=0.0075)

### Experiment 4
**Design**
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    favored_chosen = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 10: A=[0, 0, 1, 1], B=[0, 1, 0, 0]
        # Tallying difference = 1 (A wins 2 features, B wins 1)
        # WADD difference = 0.27 (A=1.21, B=0.94)
        if a == (0, 0, 1, 1) and b == (0, 1, 0, 0):
            favored_chosen.append(1 if resp == 0 else 0)
            
        # Trial 14: A=[1, 1, 0, 0], B=[1, 0, 1, 1]
        # Tallying difference = 1 (B wins 2 features, A wins 1)
        # WADD difference = 0.27 (B=2.16, A=1.89)
        elif a == (1, 1, 0, 0) and b == (1, 0, 1, 1):
            favored_chosen.append(1 if resp == 1 else 0)
            
    if not favored_chosen:
        return 0.5
    return float(np.mean(favored_chosen))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate (simulated) value:** 0.4483 (var=0.0214)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8583 (var=0.0192)
- pi_3: 0.5483 (var=0.0225)
- pi_1: 0.1667 (var=0.0258)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The proposed Noisy Take-The-Best model successfully implements the arbiter's recommendation by applying a massive lapse rate (epsilon in [0.7, 1.0]) to the deterministic TTB heuristic. This accurately pushes the predictions toward the ~0.50 mark observed in the human data across Experiments 2, 3, and 4. While Experiment 1's empirical metric is slightly below 0.50 (0.40), which a pure TTB-based model cannot perfectly reach (since TTB fundamentally favors the opposite direction), the heavy noise effectively prevents the model from making the extreme, incorrect predictions seen in prior deterministic TTB iterations. The model provides a well-regularized non-compensatory baseline.",
  "verdict": "continue",
  "rationale": "The candidate was accepted and perfectly executes the arbiter's prescribed mechanism family. The extremely high lapse rate correctly captures the near-randomness of the aggregate human data in this paradigm, neutralizing the severe overconfidence of deterministic TTB. The model is good to keep as the running baseline."
}
```

## Usage

```json
{
  "prompt_token_count": 6197,
  "candidates_token_count": 236,
  "total_token_count": 7213
}
```
