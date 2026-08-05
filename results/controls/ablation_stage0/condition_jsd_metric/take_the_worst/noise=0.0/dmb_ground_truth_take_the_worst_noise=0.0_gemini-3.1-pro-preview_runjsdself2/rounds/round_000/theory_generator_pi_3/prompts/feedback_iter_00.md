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

Replace Take The Best with a Weighted Additive (WADD) theory. Unlike TTB (which relies only on the single best discriminating cue) and Tallying (which gives equal weight to all cues), WADD assumes that decision-makers integrate all available features but weight them proportionally to their validities. The overall value for each option is computed as the validity-weighted sum of its features, and choices are made via a softmax over these aggregated values.


## CANDIDATE THEORY
Weighted Additive (WADD) theory posits that decision-makers integrate all available cues by weighting them proportionally to their validities. The overall value for each option is the validity-weighted sum of its features. Choices are made probabilistically via a softmax over these aggregated values, incorporating response noise through an inverse temperature parameter and an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
    
    # Calculate validity-weighted sum of features for each option
    scores = np.sum(stim * val, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
- validities: validities

`rationale`:
Following the arbiter's recommendation, this theory implements the Weighted Additive (WADD) heuristic. Unlike Take The Best (which relies only on the single best discriminating cue) and Tallying (which weights all cues equally), WADD considers all features but weights them proportionally to their validities. This captures compensatory decision-making where a combination of weaker cues can override a single strong cue, which aligns with the observed human behavior in these experiments.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.2574 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.2574.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 0), (0, 0, 1, 1))|0': 0.1259320629660315, '((1, 1, 0, 0), (0, 0, 1, 1))|1': 0.14502529510961215, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.1400454201362604, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.12531581606872158, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.8697047496790757, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.8687561214495593, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.1400296882731321, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.14186193793540217, '((1, 0, 0, 0), (0, 1, 1, 1))|0': 0.1333997013439522, '((1, 0, 0, 0), (0, 1, 1, 1))|1': 0.12696417347580138, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.8359240069084629, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.8751023751023751, '((0, 1, 0, 1), (1, 0, 0, 0))|0': 0.8776622090143635, '((0, 1, 0, 1), (1, 0, 0, 0))|1': 0.8621125869702719, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.85957213384531, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.8773213280810355}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2429 (var=0.0060)
**Candidate (simulated) value:** 0.1542 (var=0.0020)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0004 (var=0.0001)
- pi_2: 0.1888 (var=0.0025)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.15036743923120408, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.1583833970507919, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.12419070041200707, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.13992635455023672, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.8402915838303512, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8589191774270684, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8630310716550412, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8645575877409788, '((0, 0, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.16549069916613213, '((0, 0, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.14061734443900048, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.4925219185146983, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.473208910295003, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.8657289002557544, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.861984282907662, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.855036855036855, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8638768638768639}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0405 (var=0.0004)
**Candidate (simulated) value:** 0.0030 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0010 (var=0.0002)
- pi_1: 0.2525 (var=0.0091)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate model successfully implemented the Weighted Additive (WADD) heuristic as recommended by the arbiter, computing the validity-weighted sums of features and applying a softmax choice rule with lapse. The model was ACCEPTED by the gate, significantly improving the aggregate loss compared to the previous baseline. It strikes a balance between the experiments, capturing the compensatory behavior observed without overfitting to the non-compensatory patterns of a single experiment.",
  "verdict": "continue",
  "rationale": "The candidate faithfully incorporates the WADD mechanism specified by the arbiter. The parameterization (beta for softmax noise and epsilon for lapses) is standard and cleanly implemented, resulting in an accepted loss improvement. The model is good to go as the new baseline."
}
```

## Usage

```json
{
  "prompt_token_count": 5475,
  "candidates_token_count": 161,
  "total_token_count": 6147
}
```
