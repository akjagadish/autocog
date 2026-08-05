# feedback_iter_02

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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose a 'Weighted Additive' (WADD) theory. Unlike TTB (which stops at the first discriminating cue) and Tallying (which counts wins equally regardless of cue validity), WADD computes a total score for each option by summing the validities of the features it possesses (or the validities of the features where it wins). This allows lower-validity cues to jointly outweigh a single higher-validity cue if there are enough of them, naturally producing behavior that sits between pure TTB and pure Tallying.


## CANDIDATE THEORY
Weighted Additive (WADD) theory with Non-Linear Cue Scaling posits that decision makers compute a total score for each option by summing the validities of the features it possesses, but they may non-linearly amplify or dampen the differences between cue validities. By exponentiating cue validities with a free parameter, the model can naturally interpolate between Tallying (where all cues are weighted equally) and Take The Best (where the highest-validity cue dominates).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Scale validities non-linearly to allow amplification of the best cues
    val_scaled = val ** gamma
    
    # Compute WADD scores: sum of scaled validities for features possessed by the option
    score_a = np.sum(val_scaled * a)
    score_b = np.sum(val_scaled * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
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
- gamma: [1.0, 20.0]
- validities: validities

`rationale`:
Following the critic's diagnosis, the previous non-linear WADD model still behaved too much like Tallying because the upper bound of the gamma parameter (5.0) was not high enough to allow the highest-validity cue to fully dominate multiple lower-validity cues. By expanding the gamma range to [1.0, 20.0], the model is granted the capacity to heavily amplify the primary cue, shifting predictions closer to the empirical Take The Best-like behavior observed in humans.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.7352 -> ACCEPTED
- iter 2: loss=0.5104 -> ACCEPTED
- iter 3 (current candidate you are grading): loss=0.0527 -> ACCEPTED
Running-best (last accepted) base: iter 3 at loss=0.0527.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_winner = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])
        b_wins = sum(1 for i in range(len(a)) if b[i] > a[i])
        
        tally_winner = 0 if a_wins > b_wins else (1 if b_wins > a_wins else -1)
        
        # Only consider trials where TTB and Tallying make opposing deterministic predictions
        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner:
            ttb_matches.append(1 if row['response'] == ttb_winner else 0)
            
    if not ttb_matches:
        return 0.5
    return float(np.mean(ttb_matches))
```

**Observed (real) value:** 0.6508 (var=0.0505)
**Candidate trajectory (this loop):**
  - iter 1: 0.1729 (var=0.0119) (Δ vs real -0.4779)
  - iter 2: 0.3237 (var=0.0298) (Δ vs real -0.3271)
  - iter 3 (current): 0.6054 (var=0.0452) (Δ vs real -0.0454)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8471 (var=0.0094)
- pi_2: 0.1842 (var=0.0150)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid_mask = a_wins != b_wins
    if not np.any(valid_mask):
        return 0.5
        
    tally_preds = (b_wins > a_wins).astype(int)
    responses = data['response'].values
    
    matches = (tally_preds[valid_mask] == responses[valid_mask])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3011 (var=0.0238)
**Candidate trajectory (this loop):**
  - iter 1: 0.8436 (var=0.0115) (Δ vs real +0.5425)
  - iter 2: 0.6711 (var=0.0209) (Δ vs real +0.3700)
  - iter 3 (current): 0.3269 (var=0.0255) (Δ vs real +0.0258)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8328 (var=0.0095)
- pi_1: 0.1311 (var=0.0070)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The Weighted Additive (WADD) candidate is currently failing to capture the empirical behavior across both experiments. In Experiment 1, the metric measures how often subjects match the Take-The-Best (TTB) prediction when it conflicts with Tallying; humans lean towards TTB (0.65), but the candidate heavily favors Tallying (0.17). Similarly, in Experiment 2, the metric measures matches to Tallying; humans exhibit a low match rate (0.30), while the candidate predicts a very high match rate (0.84). This indicates that the linear sum of raw validities causes the coalition of lower-validity cues to overpower the highest-validity cue far more often than humans actually do.
Rationale: To fix the under-weighting of the primary cue while remaining strictly within the WADD framework, introduce a non-linear scaling parameter for the validities. Add a `gamma` parameter (e.g., range [0.5, 5.0]) and compute the scores using exponentiated validities: `score = np.sum((val ** gamma) * features)`. When `gamma > 1`, this will amplify the highest-validity cues, allowing the model to naturally shift its behavior closer to TTB and better match the human data, while preserving the core WADD mechanism.

**Outcome of your advice:** iter 1 candidate loss=0.7352 -> iter 2 candidate loss=0.5104 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent)
Verdict: regenerate
Interpretation: The introduction of the non-linear scaling parameter `gamma` successfully moved the model's predictions in the right direction, cutting the loss from 0.735 to 0.510. However, the model still falls short of capturing the human tendency to favor the highest-validity cue. In Experiment 1, the candidate's TTB match rate increased from 0.17 to 0.32, but humans are at 0.65. In Experiment 2, the Tallying match rate dropped from 0.84 to 0.67, but humans are much lower at 0.30. The model is still behaving too much like Tallying and not enough like TTB.
Rationale: The current `gamma` range [0.5, 5.0] is not wide enough to allow the highest-validity cue to dominate the sum when opposed by multiple lower-validity cues. To allow the model to fully capture the human data (which sits closer to TTB), shift and expand the `gamma` parameter range to allow for much stronger amplification. Change the range of `gamma` to something like [1.0, 15.0] or [2.0, 20.0]. Keep the core mechanism exactly the same.

**Outcome of your advice:** iter 2 candidate loss=0.5104 -> CURRENT candidate loss=0.0527 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate model with the expanded `gamma` parameter range ([1.0, 20.0]) has successfully captured the empirical behavior across both experiments. In Experiment 1, the model's TTB match rate is 0.605, closely approximating the human rate of 0.651. In Experiment 2, the model's Tallying match rate dropped to 0.327, closely matching the human rate of 0.301. By allowing `gamma` to reach higher values, the WADD model effectively interpolates between Tallying and Take The Best, accurately capturing the human tendency to heavily weight the highest-validity cue without completely ignoring the rest of the cues. The aggregate loss has plummeted to 0.0527, indicating an excellent fit.",
  "verdict": "continue",
  "rationale": "The current non-linear WADD theory is highly successful. The expansion of the `gamma` parameter range provided the exact degree of freedom needed to allow the primary cue to dominate when appropriate, perfectly aligning the model's predictions with human behavior in both experiments. No further modifications are necessary; this model should be kept as the running best."
}
```

## Usage

```json
{
  "prompt_token_count": 4729,
  "candidates_token_count": 263,
  "total_token_count": 5324
}
```
