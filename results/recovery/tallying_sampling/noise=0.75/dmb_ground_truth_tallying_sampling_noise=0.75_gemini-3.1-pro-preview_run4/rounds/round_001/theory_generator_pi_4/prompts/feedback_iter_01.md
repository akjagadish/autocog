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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a brand-new theory to replace the Strategy Mixture Model. A promising direction is a 'Non-linear Weighting' or 'Lexicographic Semi-Order' model. Instead of probabilistically mixing two separate strategies, subjects might use a single integration process where cue validities are transformed non-linearly (e.g., via a rank-dependent or exponential decay function). This would naturally produce behavior that looks partially like TTB (due to steep drops in subjective weight for lower-ranked cues) but remains fundamentally compensatory, allowing it to better match the ~0.40-0.45 TTB selection rate on conflict trials while naturally predicting guessing on effectively tied trials.


## CANDIDATE THEORY
Non-linear Weighting Model: Decision-makers rely on a single compensatory integration process where cue validities are transformed non-linearly. By exponentiating the objective validities by a subjective parameter gamma, the model scales between equal-weighting (gamma near 0), objective WADD (gamma = 1), and mild lexicographic behavior (gamma > 1). Constraining gamma prevents pure TTB-like behavior, capturing the intermediate TTB-match rates on conflict trials while still producing compensatory ties and guessing when the non-linearly weighted sums are close.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear transformation of validities
    weights = val ** gamma
    
    # Compensatory integration using subjective weights
    scores = stim @ weights
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- gamma: [0.1, 3.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the critic's advice, I applied a minimal edit to tighten the upper bound of the `gamma` parameter from 10.0 to 3.0. This prevents the model from frequently sampling highly convex transformations that over-weight the top cue and collapse into pure TTB behavior. By restricting gamma, the model will reduce its overprediction of TTB choices on conflict trials and remain closer to guessing on objective WADD-tied trials, better aligning with the empirical ~0.40-0.48 rates.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2493 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.2561 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.2493.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_ttb_response(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for j in range(len(a)):
            if a[j] > b[j]:
                return 0
            elif b[j] > a[j]:
                return 1
        return 0.5
    
    ttb_preds = data.apply(get_ttb_response, axis=1)
    return float(np.mean(data['response'] == ttb_preds))
```

**Observed (real) value:** 0.4392 (var=0.0021)
**Candidate trajectory (this loop):**
  - iter 1: 0.5640 (var=0.0501) (Δ vs real +0.1248)
  - iter 2 (current): 0.2440 (var=0.0055) (Δ vs real -0.1952)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8633 (var=0.0057)
- pi_2: 0.3152 (var=0.0188)
- pi_3: 0.5523 (var=0.0387)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    wadd_a = a_ratings @ validities
    wadd_b = b_ratings @ validities
    wadd_prefers_a = wadd_a > wadd_b
    wadd_prefers_b = wadd_b > wadd_a
    
    diff = a_ratings - b_ratings
    ttb_prefers_a = np.zeros(len(data), dtype=bool)
    ttb_prefers_b = np.zeros(len(data), dtype=bool)
    
    for i in range(len(data)):
        for j in range(5):
            if diff[i, j] > 0:
                ttb_prefers_a[i] = True
                break
            elif diff[i, j] < 0:
                ttb_prefers_b[i] = True
                break
                
    conflict = (ttb_prefers_a & wadd_prefers_b) | (ttb_prefers_b & wadd_prefers_a)
    
    if not np.any(conflict):
        return 0.5
        
    response = data['response'].values
    ttb_match = (ttb_prefers_a & (response == 0)) | (ttb_prefers_b & (response == 1))
    
    return float(np.mean(ttb_match[conflict]))
```

**Observed (real) value:** 0.4100 (var=0.0050)
**Candidate trajectory (this loop):**
  - iter 1: 0.4919 (var=0.0452) (Δ vs real +0.0819)
  - iter 2 (current): 0.2297 (var=0.0172) (Δ vs real -0.1803)
**Other theories' values on this metric (for reference):**
- pi_2: 0.2984 (var=0.0285)
- pi_1: 0.8697 (var=0.0064)
- pi_3: 0.4878 (var=0.0578)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    abs_diff = np.abs(diff)
    
    # TTB winner: first cue where options differ
    first_diff_idx = np.argmax(abs_diff, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    
    # WADD winner
    wadd_a = a_mat @ val
    wadd_b = b_mat @ val
    
    wadd_winner = np.where(wadd_a > wadd_b, 0, 1)
    valid = (wadd_a != wadd_b) & (np.max(abs_diff, axis=1) > 0)
    
    conflict = valid & (ttb_winner != wadd_winner)
    
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    ttb_chosen = (responses[conflict] == ttb_winner[conflict]).sum()
    
    return float(ttb_chosen / np.sum(conflict))
```

**Observed (real) value:** 0.4044 (var=0.0030)
**Candidate trajectory (this loop):**
  - iter 1: 0.4756 (var=0.0561) (Δ vs real +0.0713)
  - iter 2 (current): 0.1806 (var=0.0090) (Δ vs real -0.2237)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4631 (var=0.0390)
- pi_2: 0.2669 (var=0.0255)
- pi_1: 0.8631 (var=0.0130)

### Experiment 4
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    wadd_a = a_ratings @ val
    wadd_b = b_ratings @ val
    
    # Identify trials where the objective WADD model predicts a tie
    tied_trials = np.abs(wadd_a - wadd_b) < 1e-5
    if not np.any(tied_trials):
        return 0.5
    
    a_tied = a_ratings[tied_trials]
    b_tied = b_ratings[tied_trials]
    responses = data['response'].values[tied_trials]
    
    # In all tied trials, cue 1 (the highest validity cue) discriminates the options.
    # TTB predicts A (0) if A has cue 1, else B (1).
    ttb_preds = (b_tied[:, 0] > a_tied[:, 0]).astype(int)
    
    return float(np.mean(responses == ttb_preds))
```

**Observed (real) value:** 0.4867 (var=0.0072)
**Candidate trajectory (this loop):**
  - iter 1: 0.6756 (var=0.0212) (Δ vs real +0.1889)
  - iter 2 (current): 0.5239 (var=0.0133) (Δ vs real +0.0372)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5050 (var=0.0527)
- pi_3: 0.6789 (var=0.0228)
- pi_1: 0.8594 (var=0.0142)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The Non-linear Weighting Model successfully implements the arbiter's recommendation and provides a solid base, which is why it was accepted by the loop. However, across all experiments, it still consistently overpredicts the proportion of TTB-consistent choices. In Experiments 1-3 (conflict trials), the observed TTB match rate is ~0.40-0.44, while the model predicts ~0.47-0.56. More importantly, in Experiment 4 (objective WADD tie trials), subjects guess (~0.48), but the model strongly leans TTB (~0.67). This happens because the upper bound of the gamma parameter (up to 10.0) allows the model to frequently sample highly convex transformations that heavily over-weight the top cue, breaking WADD ties in favor of TTB and driving up the TTB match rate on conflict trials.
Rationale: Keep the exact same Non-linear Weighting mechanism, but tighten the parameter range for `gamma`. By lowering the upper bound of `gamma` from 10.0 to something much lower (e.g., `[0.1, 3.0]`), the model will be prevented from collapsing into pure TTB behavior. This will reduce the overprediction of TTB choices on conflict trials and keep the model closer to guessing on WADD-tied trials, better matching the empirical ~0.40-0.48 rates.

**Outcome of your advice:** iter 1 candidate loss=0.2493 -> CURRENT candidate loss=0.2561 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The current candidate was REJECTED by the accept gate because it over-corrected. By restricting the `gamma` parameter to [0.1, 3.0], the model's TTB match rate on conflict trials (Experiments 1-3) plummeted to ~0.18-0.24, falling far below the observed ~0.40-0.44. While this successfully reduced the overprediction on Experiment 4's WADD tie trials, the penalty on the conflict trials was too severe, resulting in an overall worse aggregate loss.",
  "verdict": "regenerate",
  "rationale": "Since the previous restriction of `gamma` to [0.1, 3.0] was too aggressive and resulted in underpredicting TTB choices, and the base iteration's [0.1, 10.0] overpredicted them, we need an intermediate range. Keep the Non-linear Weighting mechanism exactly as is, but adjust the `gamma` range to something like [0.1, 5.0] or [0.5, 6.0]. This should provide a balanced mix of convex transformations that allows the model to hit the ~0.40-0.45 TTB match rate on conflict trials without excessively breaking WADD ties in Experiment 4."
}
```

## Usage

```json
{
  "prompt_token_count": 6268,
  "candidates_token_count": 290,
  "total_token_count": 7428
}
```
