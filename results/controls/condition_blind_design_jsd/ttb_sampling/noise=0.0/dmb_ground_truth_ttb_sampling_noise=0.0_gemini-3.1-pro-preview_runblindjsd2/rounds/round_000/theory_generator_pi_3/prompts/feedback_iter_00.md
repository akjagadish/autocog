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

Propose the 'Take The Best' (TTB) heuristic. Under TTB, individuals search through features in descending order of their validity. The first feature that discriminates between the two options (i.e., one option has a 1 and the other has a 0) strictly determines the choice. If no features discriminate, the subject guesses uniformly. This differs from Tallying (which counts all discriminating features equally regardless of validity) and WADD (which compensates for deficits on highly valid cues with advantages on multiple less valid cues).


## CANDIDATE THEORY
People use the 'Take The Best' (TTB) heuristic to make binary choices. They search through features in descending order of their subjective or stated validities. The first feature that discriminates between the two options (i.e., one option has a positive feature and the other does not) strictly determines the choice, ignoring all other features. If no features discriminate, they guess uniformly. Response noise is modeled via a simple lapse rate where the individual occasionally guesses randomly instead of following the deterministic rule.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by descending validity
    order = np.argsort(validities)[::-1]
    
    epsilon = float(parameters["epsilon"])
    
    # Default to uniform guess if no features discriminate
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's recommendation, this model instantiates the Take The Best (TTB) heuristic. It orders features by their validities and makes a strict choice based on the first discriminating feature, acting as a non-compensatory lexicographic rule. This prevents lower-validity cues from overriding the most valid available cue. A lapse rate 'epsilon' is added to account for response noise and unmodeled errors.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0327 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0327.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (1, 0, 1, 1))|0': 0.43703703703703706, '((0, 1, 1, 1), (1, 0, 1, 1))|1': 0.5026143790849673, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.8568872987477638, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.8424336973478939, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.15728155339805824, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.17142857142857143, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.1486784140969163, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.1547085201793722, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.125, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.1565217391304348, '((0, 1, 0, 0), (1, 1, 1, 0))|0': 0.8710166919575114, '((0, 1, 0, 0), (1, 1, 1, 0))|1': 0.8692033293697978, '((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8631578947368421, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8813008130081301, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8595360824742269, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8349609375, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.11976047904191617, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.1157243816254417, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.5056980056980057, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.5151515151515151, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.527147087857848, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4714104193138501, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8561643835616438, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8524229074889867, '((1, 0, 0, 1), (1, 1, 0, 1))|0': 0.8854961832061069, '((1, 0, 0, 1), (1, 1, 0, 1))|1': 0.8607021517553793, '((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.17066666666666666, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.1368421052631579, '((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.8222222222222222, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.8921810699588477}
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

**Observed (real) value:** 0.0748 (var=0.0005)
**Candidate (simulated) value:** 0.0707 (var=0.0007)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0010 (var=0.0001)
- pi_2: 0.0115 (var=0.0007)

### Experiment 2
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8401037165082109, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.80248833592535, '((0, 0, 1, 1), (0, 1, 1, 0))|0': 0.6113207547169811, '((0, 0, 1, 1), (0, 1, 1, 0))|1': 0.6108108108108108, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.34727061556329847, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.3184238551650692, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8583850931677018, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8170854271356784, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.22069317023445464, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.23318385650224216, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.38506224066390043, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.3277310924369748, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.15520833333333334, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.17023809523809524, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.760061919504644, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.7381889763779528, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.1650114591291062, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.1955193482688391, '((1, 1, 1, 1), (1, 0, 0, 0))|0': 0.14410134600158353, '((1, 1, 1, 1), (1, 0, 0, 0))|1': 0.19366852886405958, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.7028301886792453, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.6401515151515151, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.20399666944213155, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.28046744574290483, '((1, 0, 1, 0), (0, 0, 1, 1))|0': 0.3624242424242424, '((1, 0, 1, 0), (0, 0, 1, 1))|1': 0.37333333333333335, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.148636763412489, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.16289592760180996}
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

**Observed (real) value:** 0.0238 (var=0.0003)
**Candidate (simulated) value:** 0.0236 (var=0.0002)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0008 (var=0.0002)
- pi_1: 0.0090 (var=0.0002)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate model implements the Take The Best (TTB) heuristic as requested. The simulated metric values (0.0707 on Exp 1 and 0.0236 on Exp 2) closely match the observed real data values (0.0748 and 0.0238, respectively). The implementation is faithful to the arbiter's instructions, successfully capturing the non-compensatory, lexicographic decision process along with a lapse rate for noise.",
  "verdict": "continue",
  "rationale": "The candidate has been accepted by the gate, accurately implements the prescribed Take The Best (TTB) mechanism, and produces metric values that are highly consistent with the ground-truth human data. No further modifications are necessary."
}
```

## Usage

```json
{
  "prompt_token_count": 7051,
  "candidates_token_count": 170,
  "total_token_count": 7985
}
```
