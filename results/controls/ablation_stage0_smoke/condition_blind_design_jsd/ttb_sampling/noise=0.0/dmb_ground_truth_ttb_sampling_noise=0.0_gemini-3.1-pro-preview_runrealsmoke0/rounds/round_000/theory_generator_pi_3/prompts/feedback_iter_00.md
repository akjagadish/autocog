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

Propose the 'Take The Best' (TTB) heuristic. Unlike Tallying (which equally weighs all features) and WADD (which computes a validity-weighted sum across all features), TTB is a non-compensatory lexicographic strategy. It assumes people rank features by their validity, compare the options on the most valid feature first, and if one option is strictly better, they choose it immediately without considering any other features. If there is a tie, they move to the next most valid feature, and so on. This introduces a fundamentally different mechanism for combining binary features that may better explain the observed choice patterns.


## CANDIDATE THEORY
Take The Best (TTB) is a non-compensatory lexicographic heuristic for multi-attribute choice. It posits that people rank features by their subjective validity and evaluate options sequentially. They compare the options on the most valid feature first; if one option is strictly better on that feature, they choose it immediately and ignore all remaining features. If the options tie, they move to the next most valid feature, continuing until a discriminating feature is found. If all features tie, they guess randomly. This mechanism explains how people can make fast, accurate decisions without integrating all available information.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    # Default to a random guess if all features tie
    p_core = np.array([0.5, 0.5])
    
    # Sequential search for the first discriminating feature
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif stim[1, idx] > stim[0, idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Apply lapse rate (epsilon) for response noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's guidance, this model implements the 'Take The Best' (TTB) heuristic. TTB offers a fundamentally different, non-compensatory mechanism compared to Tallying or WADD. Instead of aggregating information across all features (either equally or weighted), TTB assumes a lexicographic stopping rule: the decision-maker searches through features ordered by validity and stops at the first one that discriminates between the options. This captures the cognitive efficiency of human decision-making, where highly valid cues can decisively drive choices without the need for exhaustive computation.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0893 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0893.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.8431952662721893, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.8665480427046264, '((1, 0, 0, 1), (1, 1, 1, 0))|0': 0.8254649499284692, '((1, 0, 0, 1), (1, 1, 1, 0))|1': 0.8592188919164396, '((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.8591772151898734, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.8553082191780822, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.1513859275053305, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.1585274229902329, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.10526315789473684, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.10247747747747747, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.5180327868852459, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.519774011299435, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8691335740072202, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8265895953757225, '((1, 1, 1, 0), (0, 1, 0, 0))|0': 0.16736401673640167, '((1, 1, 1, 0), (0, 1, 0, 0))|1': 0.12934947049924356, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8623853211009175, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8492957746478873, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8579710144927536, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8621621621621621, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8696604600219058, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8680947012401353, '((1, 0, 1, 0), (0, 0, 0, 1))|0': 0.17297297297297298, '((1, 0, 1, 0), (0, 0, 0, 1))|1': 0.1614457831325301, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.1541095890410959, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.13240131578947367, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8710407239819005, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.8565400843881856, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8608534322820037, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8751300728407908, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.157439446366782, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.12520458265139117}
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

**Observed (real) value:** 0.0197 (var=0.0002)
**Candidate (simulated) value:** 0.0228 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0015 (var=0.0002)
- pi_2: 0.0091 (var=0.0004)

### Experiment 2
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 0, 0, 0), (1, 0, 0, 0))|0': 0.707843137254902, '((0, 0, 0, 0), (1, 0, 0, 0))|1': 0.7930232558139535, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.20941558441558442, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.16385135135135134, '((0, 1, 0, 1), (1, 0, 1, 1))|0': 0.7039848197343453, '((0, 1, 0, 1), (1, 0, 1, 1))|1': 0.7148468185388845, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.691764705882353, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.7185454545454546, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8456973293768546, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8374777975133215, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.5599078341013825, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.6317715959004392, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.829686013320647, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8277703604806409, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.6508728179551122, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.6769790718835305, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.23106796116504855, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.20311284046692607, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.512743628185907, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.47572815533980584, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.835990888382688, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.8394793926247288, '((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.7595419847328244, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.7830374753451677, '((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.7261698440207972, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.7375306623058054, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.193390452876377, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.23194303153611392, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8232611174458381, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8418201516793066, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.7265372168284789, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.7986463620981388}
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

**Observed (real) value:** 0.0203 (var=0.0002)
**Candidate (simulated) value:** 0.0210 (var=0.0002)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0010 (var=0.0002)
- pi_1: 0.0081 (var=0.0001)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate successfully implements the 'Take The Best' (TTB) heuristic as prescribed by the arbiter. The metric evaluates the Jensen-Shannon divergence between the simulated sequence-dependent choices and the ground truth reference probabilities. The candidate yields a simulation metric value (Exp 1: 0.0228, Exp 2: 0.0210) that is comparable to the empirical variance in the real data (Exp 1: 0.0197, Exp 2: 0.0203), and the loop accepted this initial formulation.",
  "verdict": "continue",
  "rationale": "The candidate faithfully implements the non-compensatory lexicographic mechanism requested by the arbiter, correctly sorting features by validity and stopping at the first discriminating cue. The parameterization (epsilon for lapse rate) is appropriate for this family. Since it was accepted by the gate and correctly reflects the specified mechanism, it is good to continue and use this as the baseline."
}
```

## Usage

```json
{
  "prompt_token_count": 7436,
  "candidates_token_count": 217,
  "total_token_count": 8308
}
```
