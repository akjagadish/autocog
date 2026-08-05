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

Propose the 'Take The Best' (TTB) heuristic as a brand-new theory. Unlike Tallying (which ignores validities) or WADD (which compensates for low validities by summing them up), TTB posits a non-compensatory lexicographic process: people search through features in descending order of their validities and stop at the first feature that discriminates between the two options, choosing the option with the higher value on that cue. The model should implement this by sorting features by validity, finding the first index where the options differ, and choosing deterministically based on that feature (subject to softmax/lapse noise).


## CANDIDATE THEORY
People use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. They evaluate features sequentially in descending order of their validities and stop at the first feature that discriminates between the options. The option with the higher value on this decisive feature is chosen. If no feature discriminates, they guess.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    
    score_a = 0.0
    score_b = 0.0
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for response noise
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
The Take The Best (TTB) heuristic is a non-compensatory, lexicographic decision strategy. Unlike compensatory models (like WADD) or unweighted models (like Tallying), TTB assumes that decision-makers order features by their subjective validities and base their choice entirely on the first feature that discriminates between the two options. The search stops immediately once a difference is found, disregarding all other features. This captures the bounded rationality of human decision-making, where individuals often rely on a single, highly valid reason rather than integrating all available information. The model incorporates both a softmax temperature and a lapse rate to account for response noise and occasional deviations from the strict heuristic.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.2474 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.2474.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_matches.append(1 if row['response'] == 0 else 0)
        elif b_wins > a_wins:
            tally_matches.append(1 if row['response'] == 1 else 0)
    if len(tally_matches) == 0:
        return 0.5
    return float(np.mean(tally_matches))
```

**Observed (real) value:** 0.2967 (var=0.0025)
**Candidate (simulated) value:** 0.1520 (var=0.0078)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8713 (var=0.0084)
- pi_2: 0.5197 (var=0.0620)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Tallying and WADD make opposite predictions
    # A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \
                 data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    is_trial_1_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                     data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
                     
    tally_choices = 0
    total = 0
    
    if is_trial_1.sum() > 0:
        # Tallying favors B (response 1) because it has 3 features vs 2.
        tally_choices += data[is_trial_1]['response'].sum()
        total += is_trial_1.sum()
        
    if is_trial_1_rev.sum() > 0:
        # Tallying favors A (response 0).
        tally_choices += (1 - data[is_trial_1_rev]['response']).sum()
        total += is_trial_1_rev.sum()
        
    if total == 0:
        return 0.5
        
    return float(tally_choices / total)
```

**Observed (real) value:** 0.3600 (var=0.0119)
**Candidate (simulated) value:** 0.1463 (var=0.0145)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4853 (var=0.0941)
- pi_1: 0.8737 (var=0.0110)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate successfully implements the 'Take The Best' (TTB) heuristic prescribed by the arbiter, searching features in descending order of validity and stopping at the first discriminator. However, the candidate's simulated metrics (0.1520 and 0.1463) are noticeably lower than the human observed values (0.2967 and 0.3600). The metrics track the proportion of choices that align with Tallying. Pure TTB strongly opposes Tallying in these designs, yielding very low metric values. To bring the candidate's predictions closer to human data, the model needs to allow for more noise (either through a higher lapse rate or a lower softmax beta), which would pull the extreme TTB predictions back towards 0.5.",
  "verdict": "regenerate",
  "rationale": "The implementation of the TTB mechanism is faithful to the arbiter's recommendation. However, to improve the quantitative fit, adjust the parameter ranges to allow for more stochasticity. Specifically, shift the `beta` range to lower values (e.g., [0.01, 5.0]) and/or increase the upper bound of the lapse rate `epsilon` to [0.0, 1.0]. This will allow the optimization to inject enough noise to bridge the gap between pure TTB (which predicts ~0.15 on these metrics) and human behavior (~0.30-0.36)."
}
```

## Usage

```json
{
  "prompt_token_count": 3777,
  "candidates_token_count": 321,
  "total_token_count": 4858
}
```
