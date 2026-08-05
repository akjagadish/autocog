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
- The recommendation below acts on THEORY 1 (= `pi_1`).

The 'Take The Best' theory is completely rejected by the data, as subjects show clear compensatory behavior (integrating multiple lower-validity cues to override a single higher-validity cue). Replace THEORY 1 with a 'Weighted Additive Model' (WADD). WADD is a compensatory strategy like Tallying, but instead of weighting all features equally, it weights each feature by its given validity (or a subjective transformation of it) before summing them. This will provide a strong, validity-sensitive compensatory baseline to test whether subjects are truly ignoring validities (as Tallying claims) or just integrating them.


## CANDIDATE THEORY
Weighted Additive Model (WADD): People evaluate options by integrating all available features, weighting each feature by its given validity. Unlike Take The Best, WADD is fully compensatory, meaning multiple lower-validity cues can override a single higher-validity cue. A subjective transformation parameter (gamma) allows the model to scale the validities, seamlessly interpolating between equal-weighting (Tallying, gamma=0) and heavy reliance on the most valid cues (gamma > 1).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Subjective transformation of validities
    gamma = float(parameters["gamma"])
    w = val ** gamma
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 10.0]
- validities: validities

`rationale`:
Following the arbiter's suggestion, this theory replaces Take The Best with a Weighted Additive Model (WADD). While Tallying assumes subjects ignore validities completely, and TTB assumes they ignore all but the highest validity cue, WADD provides a compensatory mechanism that integrates all cues weighted by their validities. The inclusion of the `gamma` parameter enables the model to capture subjective scaling of these validities, allowing it to span the continuum from Tallying (gamma=0) to validity-sensitive compensatory evaluation.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.4819 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.4819.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.array(data['option_a_ratings'].tolist())
    b = np.array(data['option_b_ratings'].tolist())
    
    diff = a - b
    mask = diff != 0
    first_diff_idx = np.argmax(mask, axis=1)
    has_diff = np.any(mask, axis=1)
    
    row_indices = np.arange(len(data))
    first_diff_val = diff[row_indices, first_diff_idx]
    
    ttb_winner = np.where(first_diff_val == 1, 0, 1)
    
    a_wins = np.sum(diff == 1, axis=1)
    b_wins = np.sum(diff == -1, axis=1)
    tally_winner = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    disagree = has_diff & (tally_winner != -1) & (ttb_winner != tally_winner)
    
    if not np.any(disagree):
        return 0.5
        
    responses = data['response'].values
    ttb_matches = (responses[disagree] == ttb_winner[disagree])
    
    return float(np.mean(ttb_matches))
```

**Observed (real) value:** 0.1425 (var=0.0068)
**Candidate (simulated) value:** 0.4988 (var=0.0916)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8433 (var=0.0153)
- pi_2: 0.1579 (var=0.0134)

### Experiment 2
**Design**
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    mask = sum_a != sum_b
    if not np.any(mask):
        return 0.5
    tally_choices = (sum_b > sum_a).astype(int)
    responses = np.array(data['response'].values)
    agreements = (responses[mask] == tally_choices[mask]).astype(float)
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.8556 (var=0.0092)
**Candidate (simulated) value:** 0.5303 (var=0.0504)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8500 (var=0.0101)
- pi_1: 0.1414 (var=0.0091)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The proposed Weighted Additive Model (WADD) was accepted by the gate because it improved upon the previous base, but it still falls short of capturing the strong compensatory (Tallying-like) behavior observed in the human data. In Experiment 1, human subjects only match TTB predictions 14.2% of the time, heavily favoring the compensatory option. In Experiment 2, subjects match the Tallying prediction 85.6% of the time. The candidate model yields ~50% for both metrics. This suggests the model is stuck in a regime that is too non-compensatory.",
  "verdict": "regenerate",
  "rationale": "The `gamma` parameter range `[0.0, 10.0]` is heavily skewed toward large values, which mathematically forces the model to behave like Take The Best (heavily overweighting the most valid cue). Since the empirical data strongly supports compensatory/Tallying-like behavior (which corresponds to gamma near 0), the optimizer struggles to find the correct regime. Please keep the WADD mechanism exactly as is, but restrict the `gamma` parameter range to `[0.0, 2.0]` or `[-1.0, 2.0]` so the model can more easily discover the compensatory weighting that matches human behavior."
}
```

## Usage

```json
{
  "prompt_token_count": 3727,
  "candidates_token_count": 288,
  "total_token_count": 4826
}
```
