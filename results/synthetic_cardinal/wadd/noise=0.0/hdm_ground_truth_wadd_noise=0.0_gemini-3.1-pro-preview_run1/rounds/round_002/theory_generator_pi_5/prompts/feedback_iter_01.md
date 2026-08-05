# feedback_iter_01

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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace Equal Weight with 'Tallying' (also known as Majority of Confirming Dimensions). Under Tallying, decision makers simply count the number of features on which each option is strictly superior to the other, ignoring both the validities of the features and the cardinal magnitudes of the differences. This provides a distinct, non-compensatory heuristic baseline that accounts for feature-by-feature comparisons rather than summing values.


## CANDIDATE THEORY
Tallying (Majority of Confirming Dimensions) heuristic: People evaluate multi-attribute options by counting the number of features on which each option is strictly superior to the other. They ignore both the validities of the features and the cardinal magnitudes of the differences, choosing the option that wins on the most features. Choice probabilities are softened by decision noise and lapses to capture human variability.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Count strict superiority
    tally_a = np.sum(a > b)
    tally_b = np.sum(b > a)
    scores = np.array([tally_a, tally_b], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    return int(np.argmax(probabilities))


`parameters`:
- beta: [0.0, 5.0]
- epsilon: [0.05, 0.5]

`rationale`:
Following the critic's feedback, the Tallying heuristic logic is maintained, but the parameter ranges are adjusted to prevent the model from becoming overly deterministic. We lowered the maximum `beta` to 5.0 and introduced a minimum `epsilon` of 0.05 to ensure a baseline level of behavioral noise, capturing the probabilistic nature of the human data.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.6471 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.6471 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.6471.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[4, 2, 2, 2, 2]  B=[3, 4, 4, 4, 4]
  A=[2, 4, 4, 4, 4]  B=[4, 2, 2, 2, 2]
  A=[3, 4, 2, 2, 2]  B=[3, 3, 4, 4, 4]
  A=[5, 2, 4, 4, 4]  B=[5, 4, 2, 2, 2]
  A=[1, 1, 4, 2, 2]  B=[1, 1, 3, 4, 4]
  A=[2, 2, 2, 4, 4]  B=[2, 2, 4, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = None
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                ttb_pred = 0
                break
            elif val_b > val_a:
                ttb_pred = 1
                break
        if ttb_pred is not None:
            matches.append(row['response'] == ttb_pred)
    if not matches:
        return 0.0
    return float(np.mean(matches))
```

**Observed (real) value:** 0.0733 (var=0.0124)
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.0000) (Δ vs real -0.0733)
  - iter 2 (current): 0.0000 (var=0.0000) (Δ vs real -0.0733)
**Other theories' values on this metric (for reference):**
- pi_1: 0.2933 (var=0.0695)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.0467 (var=0.0112)
- pi_4: 0.0000 (var=0.0000)

### Experiment 2
**Design**
  A=[5, 2, 2, 2]  B=[4, 4, 4, 4]
  A=[2, 5, 5, 5]  B=[4, 3, 3, 3]
  A=[5, 2, 3, 3]  B=[4, 4, 3, 3]
  A=[2, 5, 3, 3]  B=[4, 4, 3, 3]
  A=[3, 5, 2, 2]  B=[3, 4, 4, 4]
  A=[3, 2, 5, 5]  B=[3, 4, 4, 4]
  A=[3, 3, 5, 2]  B=[3, 3, 4, 4]
  A=[3, 3, 2, 5]  B=[3, 3, 4, 4]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins != b_wins:
            tally_pred = 0 if a_wins > b_wins else 1
            tally_match.append(1.0 if row['response'] == tally_pred else 0.0)
            
    if not tally_match:
        return 0.5
    return float(np.mean(tally_match))
```

**Observed (real) value:** 0.8350 (var=0.0340)
**Candidate trajectory (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.1650)
  - iter 2 (current): 1.0000 (var=0.0000) (Δ vs real +0.1650)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.0000)
- pi_1: 0.7100 (var=0.0809)
- pi_3: 0.8750 (var=0.0231)
- pi_4: 1.0000 (var=0.0000)

### Experiment 3
**Design**
  A=[5, 5, 6, 6]  B=[10, 10, 0, 0]
  A=[0, 0, 8, 8]  B=[2, 2, 0, 0]
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 10]
  A=[10, 0, 0, 0]  B=[7, 1, 1, 1]
  A=[0, 10, 0, 0]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert response to probability of choosing A (where response == 0)
    is_a = 1.0 - data['response']
    
    def get_trial_type(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (5, 5, 6, 6) and b == (10, 10, 0, 0): return 1
        if a == (0, 0, 8, 8) and b == (2, 2, 0, 0): return 2
        if a == (10, 0, 0, 0) and b == (0, 2, 2, 2): return 3
        if a == (1, 1, 1, 1) and b == (0, 0, 0, 10): return 4
        if a == (10, 0, 0, 0) and b == (7, 1, 1, 1): return 5
        if a == (0, 10, 0, 0) and b == (1, 0, 1, 1): return 6
        return 0
        
    tt = data.apply(get_trial_type, axis=1)
    
    # WADD predicts A on trials 2, 3, 5, 6 and B on trials 1, 4.
    # Tallying predicts B on 3, 5, 6, A on 4, and tie on 1, 2.
    wadd_pro_trials = is_a[tt.isin([2, 3, 5, 6])]
    wadd_anti_trials = is_a[tt.isin([1, 4])]
    
    wadd_pro = wadd_pro_trials.mean() if len(wadd_pro_trials) > 0 else 0.0
    wadd_anti = wadd_anti_trials.mean() if len(wadd_anti_trials) > 0 else 0.0
    
    return float(wadd_pro - wadd_anti)
```

**Observed (real) value:** 0.3600 (var=0.1554)
**Candidate trajectory (this loop):**
  - iter 1: -0.7500 (var=0.0000) (Δ vs real -1.1100)
  - iter 2 (current): -0.7500 (var=0.0000) (Δ vs real -1.1100)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3400 (var=0.1219)
- pi_2: -0.7500 (var=0.0000)
- pi_1: -0.3550 (var=0.0902)
- pi_4: 0.5000 (var=0.0000)

### Experiment 4
**Design**
  A=[0, 3, 3, 3, 3]  B=[10, 0, 0, 0, 0]
  A=[10, 0, 0, 0, 0]  B=[0, 3, 3, 3, 3]
  A=[0, 0, 8, 8, 8]  B=[10, 10, 0, 0, 0]
  A=[10, 10, 0, 0, 0]  B=[0, 0, 8, 8, 8]
  A=[2, 2, 2, 2, 2]  B=[10, 10, 0, 0, 0]
  A=[10, 10, 0, 0, 0]  B=[2, 2, 2, 2, 2]
  A=[0, 0, 10, 10, 10]  B=[10, 10, 0, 0, 0]
  A=[6, 6, 6, 6, 6]  B=[10, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_a = np.sum(a_mat > b_mat, axis=1)
    tally_b = np.sum(b_mat > a_mat, axis=1)
    
    valid_trials = tally_a != tally_b
    if not np.any(valid_trials):
        return 0.5
        
    tally_pred = (tally_b[valid_trials] > tally_a[valid_trials]).astype(int)
    actual_responses = data['response'].values[valid_trials]
    
    return float(np.mean(tally_pred == actual_responses))
```

**Observed (real) value:** 0.5250 (var=0.0619)
**Candidate trajectory (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.4750)
  - iter 2 (current): 1.0000 (var=0.0000) (Δ vs real +0.4750)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.0000)
- pi_3: 0.5725 (var=0.0682)
- pi_1: 0.7325 (var=0.1631)
- pi_4: 0.7500 (var=0.0000)

### Experiment 5
**Design**
  A=[10, 0, 0]  B=[0, 6, 6]
  A=[0, 10, 10]  B=[10, 5, 5]
  A=[9, 2, 2]  B=[2, 7, 7]
  A=[5, 8, 2]  B=[8, 0, 7]
  A=[5, 10, 0]  B=[9, 4, 0]
  A=[7, 5, 0]  B=[5, 8, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (10, 0, 0))
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (9, 2, 2))
    mask = is_t1 | is_t3
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.7000 (var=0.2000)
**Candidate trajectory (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.3000)
  - iter 2 (current): 1.0000 (var=0.0000) (Δ vs real +0.3000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6800 (var=0.1776)
- pi_4: 1.0000 (var=0.0000)
- pi_1: 0.6400 (var=0.2304)
- pi_2: 1.0000 (var=0.0000)

### Experiment 6
**Design**
  A=[0, 0, 10, 10]  B=[8, 8, 0, 0]
  A=[9, 9, 0, 0]  B=[0, 0, 10, 10]
  A=[10, 0, 0, 0]  B=[0, 0, 5, 5]
  A=[0, 0, 10, 8]  B=[10, 0, 0, 2]
  A=[2, 2, 8, 8]  B=[9, 5, 2, 2]
  A=[8, 8, 2, 2]  B=[2, 2, 10, 10]
  A=[0, 0, 8, 8]  B=[8, 8, 0, 0]
  A=[8, 0, 0, 4]  B=[0, 0, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    eqw_choices = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        resp = row['response']
        
        if a == (0, 0, 10, 10):
            eqw_choices += (resp == 0)
            total += 1
        elif a == (9, 9, 0, 0):
            eqw_choices += (resp == 1)
            total += 1
        elif a == (2, 2, 8, 8):
            eqw_choices += (resp == 0)
            total += 1
        elif a == (8, 8, 2, 2):
            eqw_choices += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.5
    return float(eqw_choices / total)
```

**Observed (real) value:** 0.6850 (var=0.1870)
**Candidate trajectory (this loop):**
  - iter 1: 0.5000 (var=0.0000) (Δ vs real -0.1850)
  - iter 2 (current): 0.5000 (var=0.0000) (Δ vs real -0.1850)
**Other theories' values on this metric (for reference):**
- pi_4: 1.0000 (var=0.0000)
- pi_3: 0.5200 (var=0.2296)
- pi_1: 0.4600 (var=0.2484)
- pi_2: 0.5000 (var=0.0000)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate successfully implements the Tallying (Majority of Confirming Dimensions) heuristic as prescribed by the arbiter. However, the simulated values across the experiments are extremely deterministic (e.g., 1.0000, -0.7500, 0.5000) with zero variance across subjects. This indicates that the model is fitting with a very high beta and zero epsilon, failing to capture the graded, probabilistic nature of human responses (as seen in the real data, which typically lie between 0.3 and 0.8). For instance, in Experiment 3, Tallying predicts -0.7500 while humans are at 0.3600; in Experiment 4, Tallying is at 1.0000 while humans are at 0.5250.
Rationale: While the core Tallying mechanism is faithful to the arbiter's recommendation, the current parameter ranges allow the model to become overly deterministic, hurting its cross-experiment fit. Please keep the exact Tallying logic but adjust the parameter ranges: lower the maximum `beta` (e.g., [0.0, 5.0]) to encourage softer softmax distributions, and consider setting a non-zero minimum for `epsilon` (e.g., [0.05, 0.5]) to ensure a baseline level of behavioral noise. This will help the model capture the probabilistic nuances of the human data rather than snapping to extreme values.

**Outcome of your advice:** iter 1 candidate loss=0.6471 -> CURRENT candidate loss=0.6471 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate was rejected because the model's output remains entirely deterministic (all variances are 0.0000), producing exactly the same metric values as the previous iteration. The root cause is a bug in the `policy` function: it uses `np.argmax(probabilities)`. By taking the argmax, the policy completely overrides the softmax temperature (`beta`) and the lapse rate (`epsilon`) implemented in the `predict` function, forcing the model to always pick the option with the highest probability deterministically.",
  "verdict": "regenerate",
  "rationale": "The Tallying mechanism in `predict` is correct, but the `policy` function must be fixed. Change `policy` to probabilistically sample from the provided probabilities rather than choosing deterministically (e.g., use `np.random.choice(len(probabilities), p=probabilities)` instead of `np.argmax`). Keep the current Tallying logic and the adjusted parameter ranges, but fix this sampling bug so the model can actually produce the graded probabilistic responses needed to fit the human data."
}
```

## Usage

```json
{
  "prompt_token_count": 6970,
  "candidates_token_count": 233,
  "total_token_count": 7955
}
```
