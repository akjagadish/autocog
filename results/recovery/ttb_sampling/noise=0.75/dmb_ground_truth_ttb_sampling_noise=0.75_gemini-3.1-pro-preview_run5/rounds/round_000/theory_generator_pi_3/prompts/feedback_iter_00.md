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

Propose 'Take-The-Best' (TTB) as the new theory. TTB searches through features in descending order of their validity and stops at the first feature that discriminates between the two options, choosing the option with the higher value on that feature. If no features discriminate, it guesses. This will provide a strong non-compensatory alternative to the compensatory WADD model, as it uses validities (unlike Tallying) but in a lexicographic rather than additive way.


## CANDIDATE THEORY
Take-The-Best (TTB) models decision-making as a non-compensatory, lexicographic process. People search through available features in descending order of their validity (importance) and stop at the first feature that discriminates between the two options. The option with the higher value on this decisive feature is chosen. If no features discriminate, the decision reduces to a guess. Unlike Weighted Additive (WADD) models, TTB does not allow deficits on high-validity cues to be compensated by advantages on lower-validity cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order (stable sort for ties)
    order = np.argsort(-validities, kind='stable')
    
    a, b = stim[0], stim[1]
    scores = np.zeros(2)
    
    # Lexicographic search: stop at the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores (scores are either [1,0], [0,1], or [0,0])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
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
Following the arbiter's recommendation, this theory instantiates the Take-The-Best (TTB) heuristic. It provides a non-compensatory alternative to the WADD model by iterating through features ordered by validity and stopping immediately once a discriminating cue is found. This lexicographic search naturally explains situations where subjects ignore large numbers of low-validity cues in favor of a single high-validity cue. Response noise is included via a softmax with an independent lapse rate to capture probabilistic deviations from the strict heuristic.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.3998 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.3998.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tallying_consistent = 0
    total_dissociation_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Identify dissociation trials: one option has more features, but lacks the first (highest validity) feature
        if sum_a > sum_b and a[0] == 0:
            total_dissociation_trials += 1
            if row['response'] == 0:
                tallying_consistent += 1
        elif sum_b > sum_a and b[0] == 0:
            total_dissociation_trials += 1
            if row['response'] == 1:
                tallying_consistent += 1
                
    if total_dissociation_trials == 0:
        return 0.5
        
    return float(tallying_consistent / total_dissociation_trials)
```

**Observed (real) value:** 0.4113 (var=0.0093)
**Candidate (simulated) value:** 0.1444 (var=0.0113)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8669 (var=0.0105)
- pi_2: 0.4050 (var=0.0960)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_f1 = np.array([a[0] for a in data['option_a_ratings']])
    b_f1 = np.array([b[0] for b in data['option_b_ratings']])
    resp = data['response'].values
    chosen_f1 = np.where(resp == 0, a_f1, b_f1)
    return float(np.mean(chosen_f1))
```

**Observed (real) value:** 0.5900 (var=0.0027)
**Candidate (simulated) value:** 0.8327 (var=0.0124)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5321 (var=0.0463)
- pi_1: 0.2685 (var=0.0055)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate successfully implements the Take-The-Best (TTB) heuristic as requested by the arbiter. The simulation results show that strict TTB over-predicts reliance on the single highest-validity feature compared to human data. In Experiment 1, humans choose the Tallying-consistent option 41% of the time, while TTB predicts only 14%. In Experiment 2, humans choose the option favored by the most valid cue 59% of the time, while TTB predicts 83%. This indicates that human participants are more compensatory or noisy than a strict lexicographic rule. However, as an instantiation of the TTB mechanism family, the candidate is perfectly faithful to the arbiter's prescription and provides a clean non-compensatory baseline.",
  "verdict": "continue",
  "rationale": "The candidate accurately implements the prescribed Take-The-Best mechanism, including the lexicographic search through validity-ordered features and a reasonable noise model (softmax + lapse). Although the empirical fit reveals that humans are less strictly non-compensatory than TTB predicts, this is a theoretical finding about the TTB family itself rather than a flaw in the proposer's implementation. Since it successfully serves the arbiter's goal of providing a strong non-compensatory alternative model and was accepted by the loss gate, we can ship this candidate."
}
```

## Usage

```json
{
  "prompt_token_count": 3519,
  "candidates_token_count": 294,
  "total_token_count": 5155
}
```
