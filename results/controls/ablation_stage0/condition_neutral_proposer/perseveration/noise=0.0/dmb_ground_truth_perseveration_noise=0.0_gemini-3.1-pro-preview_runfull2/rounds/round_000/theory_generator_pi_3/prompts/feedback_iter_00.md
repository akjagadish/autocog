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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose a Weighted Additive (WADD) theory. Unlike TTB (which ignores all but the top discriminating cue) and Tallying (which ignores cue validities), WADD posits that people evaluate options by summing the features weighted by their respective validities (or perceived importance). In conflict trials where a high-validity cue opposes several lower-validity cues, WADD naturally predicts a closer, more compensatory balance of evidence, leading to choice probabilities much closer to 0.5, perfectly aligning with the observed data.


## CANDIDATE THEORY
People evaluate options using a Weighted Additive (WADD) strategy, where each option's value is the sum of its features weighted by their respective cue validities. This compensatory mechanism allows multiple lower-validity cues to balance out a single high-validity cue, producing choice probabilities near 0.5 on conflict trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's suggestion, this model implements the Weighted Additive (WADD) strategy. It computes a score for each option by summing its features multiplied by their validities. In conflict trials where one high-validity cue opposes several lower-validity cues, WADD naturally balances these out, resulting in scores that are closer together. This leads to choice probabilities near 0.5, which aligns perfectly with the observed data where humans side with TTB roughly 53.6% of the time in Exp 1 and Tallying 52% of the time in Exp 2.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.4818 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.4818.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # TTB prediction: first discriminating feature in descending validity order
    ttb_preds = np.full(len(data), -1)
    for i in range(a_ratings.shape[1]):
        mask_a = (a_ratings[:, i] > b_ratings[:, i]) & (ttb_preds == -1)
        mask_b = (b_ratings[:, i] > a_ratings[:, i]) & (ttb_preds == -1)
        ttb_preds[mask_a] = 0
        ttb_preds[mask_b] = 1
        
    # Tally prediction: sum of feature-wise wins
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_preds = np.full(len(data), -1)
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    # Identify trials where TTB and Tallying make opposite predictions
    disagree_mask = (ttb_preds != -1) & (tally_preds != -1) & (ttb_preds != tally_preds)
    
    if not np.any(disagree_mask):
        return 0.5
        
    # Calculate proportion of choices aligning with TTB on these crucial trials
    responses = data['response'].values
    matches = (responses[disagree_mask] == ttb_preds[disagree_mask])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5360 (var=0.0087)
**Candidate (simulated) value:** 0.1760 (var=0.0130)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8393 (var=0.0102)
- pi_2: 0.1500 (var=0.0114)

### Experiment 2
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pref = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    diff = a_mat - b_mat
    nz = diff != 0
    has_nz = np.any(nz, axis=1)
    first_nz_idx = np.argmax(nz, axis=1)
    first_diff = diff[np.arange(len(diff)), first_nz_idx]
    
    ttb_pref = np.where(has_nz, np.where(first_diff > 0, 0, 1), -1)
    
    mask = (tally_pref != -1) & (ttb_pref != -1) & (tally_pref != ttb_pref)
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    tally_prefs = tally_pref[mask]
    
    return float(np.mean(responses == tally_prefs))
```

**Observed (real) value:** 0.5200 (var=0.0096)
**Candidate (simulated) value:** 0.8440 (var=0.0129)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8751 (var=0.0092)
- pi_1: 0.1409 (var=0.0112)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory implements a Weighted Additive (WADD) strategy, which the arbiter prescribed. However, the simulation results show that the model heavily favors Tallying (the option with more positive cues) over Take-The-Best (the option with the single highest-validity cue) on conflict trials. In Exp 1, TTB alignment is 0.1760 (humans: 0.5360); in Exp 2, Tally alignment is 0.8440 (humans: 0.5200). Because non-conflict trials likely force the inverse-temperature `beta` to be high, the linear sum of validities deterministically favors the Tally option on conflict trials, missing the ~0.50 human indifference.",
  "verdict": "regenerate",
  "rationale": "To capture the ~0.50 choice probability on conflict trials while remaining strictly within the WADD family, the model needs to be able to balance the weight of the single highest-validity cue against the sum of the lower-validity cues. Currently, using raw linear validities makes the sum of lower cues much larger. Introduce a non-linear scaling parameter for the validities, such as `weights = validities ** gamma` (with `gamma` as a new parameter in `[1.0, 10.0]`). This will allow the model to learn a weight distribution where the top cue perfectly balances the rest, yielding equal scores and thus 50/50 choices on conflict trials, even when `beta` is high."
}
```

## Usage

```json
{
  "prompt_token_count": 3997,
  "candidates_token_count": 344,
  "total_token_count": 5405
}
```
