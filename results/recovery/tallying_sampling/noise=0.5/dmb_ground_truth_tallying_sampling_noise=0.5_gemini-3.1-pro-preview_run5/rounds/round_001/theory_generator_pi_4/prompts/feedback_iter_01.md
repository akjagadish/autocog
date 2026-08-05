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
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose the 'Take-The-Best' (TTB) heuristic as a brand-new theory. TTB is a non-compensatory, lexicographic strategy where the decision-maker searches through cues in descending order of their validity. The search stops at the first cue that discriminates between the two options, and the option with the positive value on that cue is chosen. If no cues discriminate, the model guesses. This provides a stark contrast to both the unweighted compensatory nature of Tallying and the weighted compensatory nature of WADD.


## CANDIDATE THEORY
Probabilistic Take-The-Best (PTTB) Model: Decision makers search through cues in descending order of validity, but instead of strictly stopping at the first discriminating cue, they exhibit a probabilistic stopping rule. At each discriminating cue, the search terminates with probability p_stop (and a choice is made based on that cue), or continues with probability 1 - p_stop. If the search exhausts all cues without stopping, the decision maker guesses. This allows the model to span a spectrum from pure Take-The-Best (p_stop = 1) to a softer, Tallying-like integration where multiple discriminating cues contribute to the decision (p_stop near 0).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    p_stop = float(parameters["p_stop"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Order cues by descending validity
    cue_order = np.argsort(-validities, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    
    p_a = 0.0
    p_b = 0.0
    p_reach = 1.0  # Probability that the search reaches the current cue
    
    for j in cue_order:
        if a[j] != b[j]:
            # Cue discriminates
            p_stop_here = p_stop * p_reach
            
            # Softmax choice on the current cue
            scores = np.array([a[j], b[j]])
            z = beta * (scores - np.max(scores))
            e = np.exp(z)
            p_choice = e / np.sum(e)
            
            p_a += p_stop_here * p_choice[0]
            p_b += p_stop_here * p_choice[1]
            
            # Update probability of continuing the search
            p_reach *= (1.0 - p_stop)
            
    # If the search exhausts all cues without stopping, guess uniformly
    p_a += p_reach * 0.5
    p_b += p_reach * 0.5
    
    p_core = np.array([p_a, p_b])
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- p_stop: [0.0, 1.0]
- beta: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the critic's guidance, I have implemented a Probabilistic Take-The-Best model to strictly adhere to the arbiter's prescribed sequential/lexicographic search mechanism family. To overcome the empirical failures of pure TTB, I introduced a probabilistic stopping rule (`p_stop`) and a softmax temperature (`beta`) on the choice at the stopping cue. When `p_stop` is high, the model mimics pure TTB. When `p_stop` is low, the search is likely to continue past the first discriminating cue, effectively distributing decision weight across multiple discriminating cues. This allows the model to approximate the high-performing Tallying behavior (equal weighting of cues) while mechanistically remaining a sequential cue search.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.4566 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.5266 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.4566.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_cues = data['option_b_ratings'].apply(sum)
    a_cues = data['option_a_ratings'].apply(sum)
    mask = b_cues > a_cues
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6919 (var=0.0034)
**Candidate trajectory (this loop):**
  - iter 1: 0.6012 (var=0.0228) (Δ vs real -0.0907)
  - iter 2 (current): 0.3895 (var=0.0205) (Δ vs real -0.3024)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1636 (var=0.0114)
- pi_2: 0.6681 (var=0.0277)
- pi_3: 0.7638 (var=0.0082)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                total += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                total += 1
                break
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.3892 (var=0.0034)
**Candidate trajectory (this loop):**
  - iter 1: 0.5254 (var=0.0134) (Δ vs real +0.1362)
  - iter 2 (current): 0.5894 (var=0.0136) (Δ vs real +0.2002)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4569 (var=0.0152)
- pi_1: 0.8435 (var=0.0127)
- pi_3: 0.3602 (var=0.0033)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.85, 0.55, 0.5, 0.5])
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    
    wadd_a = np.dot(a_ratings, validities)
    wadd_b = np.dot(b_ratings, validities)
    
    tally_pred = (sum_b > sum_a).astype(int)
    wadd_pred = (wadd_b > wadd_a).astype(int)
    
    critical_mask = (sum_a != sum_b) & (wadd_a != wadd_b) & (tally_pred != wadd_pred)
    
    if not np.any(critical_mask):
        return 0.5
        
    responses = data['response'].values[critical_mask]
    tally_preds = tally_pred[critical_mask]
    
    return float(np.mean(responses == tally_preds))

```

**Observed (real) value:** 0.6783 (var=0.0060)
**Candidate trajectory (this loop):**
  - iter 1: 0.2917 (var=0.0475) (Δ vs real -0.3867)
  - iter 2 (current): 0.3225 (var=0.0223) (Δ vs real -0.3558)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7333 (var=0.0207)
- pi_2: 0.4296 (var=0.0575)
- pi_1: 0.1713 (var=0.0169)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            wadd_consistent += (1 if resp == 0 else 0)
            total += 1
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            wadd_consistent += (1 if resp == 1 else 0)
            total += 1
            
    if total == 0:
        return 0.5
    return wadd_consistent / total
```

**Observed (real) value:** 0.3467 (var=0.0161)
**Candidate trajectory (this loop):**
  - iter 1: 0.7617 (var=0.0533) (Δ vs real +0.4150)
  - iter 2 (current): 0.6733 (var=0.0265) (Δ vs real +0.3267)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5575 (var=0.0876)
- pi_3: 0.2508 (var=0.0216)
- pi_1: 0.8392 (var=0.0186)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate model achieves better empirical fit than pure Take-The-Best by using a Power-Weighted Additive model. However, it explicitly ignores the arbiter's recommendation to implement the TTB (lexicographic/sequential search) mechanism family, instead reverting to a compensatory additive model.
Rationale: You explicitly ignored the arbiter's recommendation to implement the Take-The-Best (TTB) family, drifting instead into a Power-Weighted Additive model. You must strictly adhere to the prescribed mechanism family (sequential/lexicographic cue search). To address pure TTB's poor empirical fit without abandoning the mechanism, introduce probabilistic elements within the TTB framework. For example, add a probabilistic stopping rule (e.g., at each discriminating cue, the search stops with probability `p_stop` and otherwise continues), a temperature parameter on the choice at the stopping cue, or a high lapse rate (`epsilon`). Return to the TTB family with these adjustments.

**Outcome of your advice:** iter 1 candidate loss=0.4566 -> CURRENT candidate loss=0.5266 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate implemented Probabilistic Take-The-Best (PTTB) as requested, successfully returning to the prescribed sequential/lexicographic mechanism family. However, this candidate was REJECTED by the accept gate because its aggregate loss (0.5266) was worse than the running-best base (0.4566, which was a compensatory model). PTTB struggled particularly on Experiment 1 (0.3895 vs real 0.6919) and Experiment 4 (0.6733 vs real 0.3467). We need to find a different way to introduce probabilistic variation into TTB that fits the data better than the probabilistic stopping rule.",
  "verdict": "regenerate",
  "rationale": "The previous attempt to make TTB probabilistic via a stopping rule (`p_stop`) was rejected by the gate. To improve fit while strictly remaining within the prescribed TTB family, try a different source of noise: probabilistic cue ordering. Maintain the strict TTB stopping rule (always stop at the first discriminating cue and choose the option it favors, no `p_stop`), but assume subjects have noisy representations of cue validities. On each trial, add Gaussian noise to the validities before sorting them to determine the search order (e.g., `perceived_validity = validity + Normal(0, sigma)`). This allows the model to capture aggregate deviations from pure TTB while preserving the strict lexicographic mechanism at the single-trial level. Include a lapse rate (`epsilon`) as well."
}
```

## Usage

```json
{
  "prompt_token_count": 5937,
  "candidates_token_count": 336,
  "total_token_count": 7083
}
```
