# feedback_iter_01

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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

THEORY 1 (Zero-Intelligence) is clearly the superior model. To provide a stronger, more plausible competitor than WADD, propose a new theory based on an extremely simplified heuristic, such as 'Tallying with High Lapse' (participants simply count the number of positive features but are subject to near-total decision noise) or a 'Single-Cue / First-Feature' heuristic where only the most valid cue is ever considered. This will test whether there is any residual, highly simplified cognitive strategy at play rather than the computationally expensive weighted integration assumed by WADD.


## CANDIDATE THEORY
Single-Cue Heuristic with Very High Lapse: Participants avoid the cognitive overload of integrating multiple cues by relying exclusively on the single most valid cue. They compare the two options on this single best feature, choosing the one that is superior. If the options tie on this cue, they guess randomly. However, due to task disengagement, lack of feedback, or cognitive fatigue, participants exhibit a near-total lapse rate, meaning they almost entirely guess randomly, with only a tiny residual trace of the single-cue strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Single-Cue expects a (2, n_features) state.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the single most valid cue
    best_cue_idx = np.argmax(val)
    
    a_val = stim[0, best_cue_idx]
    b_val = stim[1, best_cue_idx]
    
    epsilon = float(parameters["epsilon"])
    
    # Deterministic choice based on the single best cue
    if a_val > b_val:
        p_core = np.array([1.0, 0.0])
    elif b_val > a_val:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    # Blend with random guessing based on the lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- epsilon: [0.9, 1.0]
- validities: validities

`rationale`:
Following the critic's advice, I restricted the range of the lapse parameter 'epsilon' from [0.0, 1.0] to [0.9, 1.0]. The empirical data consistently hovers around 0.5000, indicating near-random guessing. By constraining epsilon to be very high, the model correctly acts primarily as a zero-intelligence guesser, while still allowing for a minuscule residual effect of the single-cue heuristic. This prevents the large deviations from 0.5000 that were observed in the previous iteration.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3520 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0462 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.0462.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB choice: first cue that discriminates determines choice
        ttb_choice = None
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_choice = 0
                break
            elif b[j] > a[j]:
                ttb_choice = 1
                break
                
        if ttb_choice is None:
            ttb_choice = 0.5
            
        if ttb_choice == row['response']:
            matches.append(1.0)
        elif ttb_choice == 0.5:
            matches.append(0.5)
        else:
            matches.append(0.0)
            
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.6250 (var=0.0081) (Δ vs real +0.1250)
  - iter 2 (current): 0.5210 (var=0.0023) (Δ vs real +0.0210)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8527 (var=0.0108)
- pi_2: 0.1663 (var=0.0108)
- pi_3: 0.4742 (var=0.0035)
- pi_4: 0.5046 (var=0.0028)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = 0
    valid_trials = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_arr = np.array(a)
        b_arr = np.array(b)
        a_wins = np.sum(a_arr > b_arr)
        b_wins = np.sum(b_arr > a_arr)
        if a_wins > b_wins:
            if r == 0:
                matches += 1
            valid_trials += 1
        elif b_wins > a_wins:
            if r == 1:
                matches += 1
            valid_trials += 1
    return float(matches / valid_trials) if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.5042 (var=0.0108)
**Candidate trajectory (this loop):**
  - iter 1: 0.3371 (var=0.0191) (Δ vs real -0.1671)
  - iter 2 (current): 0.4760 (var=0.0032) (Δ vs real -0.0281)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8631 (var=0.0101)
- pi_1: 0.1667 (var=0.0111)
- pi_3: 0.5233 (var=0.0024)
- pi_4: 0.4954 (var=0.0020)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.3546 (var=0.0132) (Δ vs real -0.1454)
  - iter 2 (current): 0.4792 (var=0.0017) (Δ vs real -0.0208)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1421 (var=0.0096)
- pi_3: 0.5192 (var=0.0039)
- pi_2: 0.8592 (var=0.0081)
- pi_4: 0.5029 (var=0.0019)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4985 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.2886 (var=0.0226) (Δ vs real -0.2099)
  - iter 2 (current): 0.4705 (var=0.0024) (Δ vs real -0.0280)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5147 (var=0.0036)
- pi_1: 0.1318 (var=0.0101)
- pi_2: 0.6425 (var=0.0025)
- pi_4: 0.5069 (var=0.0030)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    def get_wadd_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * val) - np.sum(b * val)
        
    diffs = data.apply(get_wadd_diff, axis=1)
    
    # WADD predicts choice A (0) when diffs > 0, and choice B (1) when diffs < 0
    is_correct = ((diffs > 0) & (data['response'] == 0)) | ((diffs < 0) & (data['response'] == 1))
    
    return float(is_correct.mean())
```

**Observed (real) value:** 0.5008 (var=0.0004)
**Candidate trajectory (this loop):**
  - iter 1: 0.6027 (var=0.0068) (Δ vs real +0.1019)
  - iter 2 (current): 0.5125 (var=0.0033) (Δ vs real +0.0117)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4856 (var=0.0028)
- pi_3: 0.5494 (var=0.0039)
- pi_1: 0.6802 (var=0.0030)
- pi_2: 0.7510 (var=0.0058)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    val_a = A.dot(val)
    val_b = B.dot(val)
    
    # The expected choice under deterministic WADD
    expected = (val_a < val_b).astype(int)
    correct = (data['response'].values == expected).astype(float)
    
    # Weight the accuracy by the absolute difference in weighted sums
    diff = np.abs(val_a - val_b)
    
    # Return the weighted accuracy
    return float(np.sum(correct * diff) / np.sum(diff))
```

**Observed (real) value:** 0.4990 (var=0.0006)
**Candidate trajectory (this loop):**
  - iter 1: 0.7445 (var=0.0268) (Δ vs real +0.2455)
  - iter 2 (current): 0.5269 (var=0.0060) (Δ vs real +0.0278)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5956 (var=0.0133)
- pi_4: 0.4894 (var=0.0064)
- pi_1: 0.8439 (var=0.0127)
- pi_2: 0.7734 (var=0.0079)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The empirical data across all experiments hovers extremely close to 0.5000, indicating near-random guessing or zero intelligence behavior by the participants. The current candidate, a Single-Cue heuristic with a lapse rate, yields predictions that deviate significantly from 0.5000 (e.g., 0.2886 on Exp 4, 0.7445 on Exp 6). This suggests that the residual single-cue strategy is exerting too much influence, likely because the lapse parameter 'epsilon' is allowed to take values that are too low.
Rationale: The candidate successfully implements the Arbiter's prescribed Single-Cue with High Lapse family, but the parameter ranges allow it to deviate too far from the near-random empirical data. Regenerate the model by restricting the range of the lapse parameter 'epsilon' to a much higher interval (e.g., [0.8, 1.0] or [0.9, 1.0]). This will force the model to behave primarily as a zero-intelligence guesser while still testing for the tiny residual effect of the single-cue heuristic, thereby reducing the large deviations observed in experiments 2, 4, and 6.

**Outcome of your advice:** iter 1 candidate loss=0.3520 -> CURRENT candidate loss=0.0462 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The empirical data across all six experiments consistently shows a value extremely close to 0.5000, indicating that participants are essentially guessing randomly (zero intelligence). The current candidate theory, which implements a Single-Cue heuristic with a constrained high lapse rate (epsilon between 0.9 and 1.0), captures this near-random behavior remarkably well. The deviations from the real data have been dramatically reduced (all within ~0.02 to ~0.03), and the aggregate loss has dropped from 0.3520 to 0.0462. The candidate faithfully follows the Arbiter's prescribed mechanism family while correctly calibrating the parameter range to match the human data.",
  "verdict": "continue",
  "rationale": "The candidate successfully implements the 'Single-Cue with High Lapse' family as prescribed by the Arbiter. By restricting the lapse parameter 'epsilon' to [0.9, 1.0], the model accurately reflects the near-total random guessing observed in the empirical data while still allowing for the minuscule residual trace of the heuristic. The fit is excellent, and no further tuning is necessary."
}
```

## Usage

```json
{
  "prompt_token_count": 6794,
  "candidates_token_count": 250,
  "total_token_count": 7455
}
```
