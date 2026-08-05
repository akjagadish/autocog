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

Propose a Weighted Additive (WADD) theory. Unlike Tallying (which gives equal weight to all cues) or TTB (which only uses the single best discriminating cue), WADD assumes subjects compute a weighted sum of the features for each option, where the weights are proportional to the cues' validities or their log-odds. This provides a formal compensatory model that scales with both cue validity and the number of cues, offering a more realistic alternative to the non-compensatory TTB.


## CANDIDATE THEORY
Weighted Additive (WADD) model: Decision makers evaluate options by computing a weighted sum of their feature values. Unlike non-compensatory heuristics (e.g., Take-The-Best) that rely on a single cue, or equal-weight heuristics (e.g., Tallying) that ignore cue validities, WADD uses all available information and scales the contribution of each feature by its validity or log-odds. The option with the highest weighted sum is selected, subject to softmax noise and a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # The arbiter suggested weights proportional to validities or their log-odds.
    # We allow the model to select which weighting scheme best fits the population.
    if int(parameters["use_log_odds"]) == 1:
        # Clip to avoid log(0) or division by zero if validity is exactly 0.5 or 1.0
        val_clipped = np.clip(val, 0.5001, 0.9999)
        w = np.log(val_clipped / (1.0 - val_clipped))
    else:
        w = val
        
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- use_log_odds: {0, 1}
- validities: validities

`rationale`:
Following the arbiter's instructions, we implement the Weighted Additive (WADD) theory. TTB and Tallying failed to capture the choice patterns because they are extreme approximations (using only one cue or weighting all cues equally). WADD computes a compensatory weighted sum of features, using either raw validities or their log-odds (Naive Bayes optimal weights) as coefficients. This provides a formal compensatory mechanism that trades off cues of different validities appropriately.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.5618 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.5618.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None:
            matches.append(1 if resp == ttb_pred else 0)
            
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate (simulated) value:** 0.3515 (var=0.0216)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8444 (var=0.0093)
- pi_2: 0.2267 (var=0.0038)

### Experiment 2
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Determine which option is favored by the highest validity cue (cue 0)
    cue_0_a = a_ratings[:, 0]
    cue_0_b = b_ratings[:, 0]
    cue_0_winner = (cue_0_b > cue_0_a).astype(int)
    
    # Did the subject choose the option favored by cue 0?
    chose_cue_0 = (data['response'] == cue_0_winner).astype(float)
    
    # Calculate the tally difference in favor of the cue 0 winner
    tally_a = (a_ratings > b_ratings).sum(axis=1)
    tally_b = (b_ratings > a_ratings).sum(axis=1)
    tally_diff = np.where(cue_0_winner == 0, tally_a - tally_b, tally_b - tally_a)
    
    # Compare choice probabilities when tally strongly agrees vs. disagrees/ties with cue 0
    high_diff_mask = tally_diff >= 3
    low_diff_mask = tally_diff <= 0
    
    if not np.any(high_diff_mask) or not np.any(low_diff_mask):
        return 0.0
        
    high_diff_rate = chose_cue_0[high_diff_mask].mean()
    low_diff_rate = chose_cue_0[low_diff_mask].mean()
    
    return float(high_diff_rate - low_diff_rate)

```

**Observed (real) value:** -0.0237 (var=0.0386)
**Candidate (simulated) value:** 0.5571 (var=0.0210)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6340 (var=0.0248)
- pi_1: -0.0025 (var=0.0080)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate WADD model was accepted and provides a baseline for a compensatory strategy. However, it currently misses the empirical targets significantly. In Experiment 1, humans match the TTB prediction exactly 50% of the time (metric = 0.5000, var = 0.0000), whereas the candidate yields 0.3515. In Experiment 2, humans show almost no difference in choosing the top-cue winner whether the tally strongly agrees or disagrees (metric = -0.0237), whereas the candidate yields a large positive difference (0.5571), indicating it is far too sensitive to the tally of remaining cues.",
  "verdict": "regenerate",
  "rationale": "To better capture the human data within the prescribed WADD family, the model needs more flexibility in how cue validities are translated into weights. Currently, it strictly uses either raw validities or log-odds. Introduce a non-linear scaling parameter `gamma` (e.g., `w = val ** gamma` or `w = log_odds ** gamma`) with a range like [0.0, 5.0]. This will allow the model to smoothly interpolate between Tallying (gamma = 0), proportional WADD (gamma = 1), and TTB-like non-compensatory behavior (gamma > 1), enabling it to find the exact weighting scheme that resolves the tension between Experiment 1 (50% TTB match) and Experiment 2 (insensitivity to tally differences)."
}
```

## Usage

```json
{
  "prompt_token_count": 3997,
  "candidates_token_count": 334,
  "total_token_count": 5618
}
```
