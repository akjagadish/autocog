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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a 'Probabilistic Cue Sampling' theory to replace Tallying. In this theory, rather than integrating all cues (like WADD) or counting wins (like Tallying), the decision-maker samples a single feature on each trial with probability proportional to its subjective validity. They then choose the option that has a higher value on that sampled feature. If the options tie on the sampled feature, they guess uniformly. This will naturally predict the ~0.5 choice proportions with low variance on trials where the validity sums of the favoring cues are roughly equal.


## CANDIDATE THEORY
Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- gamma: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Applying the minimal diff requested by the critic: replacing the power function transformation for subjective validities with a softmax transformation (`np.exp(gamma * validities)`) and expanding the `gamma` range to `[0.0, 10.0]`. This allows the model to flexibly capture choice proportions closer to 0.5 on incongruent trials by better calibrating the sampling probabilities of the cues.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.4334 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.3618 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.3618.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_consistent = 0
    total_incongruent = 0
    
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        resp = row['response']
        
        # Incongruent trial 1: Option A has fewer but higher-validity features
        if a == [1, 1, 0, 0, 0] and b == [0, 0, 1, 1, 1]:
            total_incongruent += 1
            if resp == 1:  # Tallying prefers B (3 features > 2 features)
                tally_consistent += 1
        
        # Incongruent trial 4: Option B has fewer but higher-validity features
        elif a == [0, 0, 1, 1, 1] and b == [1, 1, 0, 0, 0]:
            total_incongruent += 1
            if resp == 0:  # Tallying prefers A (3 features > 2 features)
                tally_consistent += 1
                
    if total_incongruent == 0:
        return 0.5
    return tally_consistent / total_incongruent
```

**Observed (real) value:** 0.5067 (var=0.0125)
**Candidate trajectory (this loop):**
  - iter 1: 0.2633 (var=0.0145) (Δ vs real -0.2433)
  - iter 2 (current): 0.3083 (var=0.0235) (Δ vs real -0.1983)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8458 (var=0.0094)
- pi_2: 0.4117 (var=0.1164)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where one option has the two highest validity features [1, 1, 0, 0, 0]
    # and the other has the three lowest validity features [0, 0, 1, 1, 1].
    is_t1 = (data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)))
    is_t2 = (data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)))
    
    wadd_choices = 0
    total = 0
    
    if is_t1.any():
        wadd_choices += (data.loc[is_t1, 'response'] == 0).sum()
        total += is_t1.sum()
        
    if is_t2.any():
        wadd_choices += (data.loc[is_t2, 'response'] == 1).sum()
        total += is_t2.sum()
        
    return float(wadd_choices / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.4800 (var=0.0099)
**Candidate trajectory (this loop):**
  - iter 1: 0.7558 (var=0.0270) (Δ vs real +0.2758)
  - iter 2 (current): 0.6658 (var=0.0321) (Δ vs real +0.1858)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5533 (var=0.0817)
- pi_1: 0.1417 (var=0.0117)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate successfully implements the Probabilistic Cue Sampling mechanism prescribed by the arbiter. However, it fails to capture the empirical ~0.50 choice proportion on incongruent trials (where one option has fewer but higher-validity cues). Instead, the model heavily favors the WADD-consistent option, yielding ~0.75 for WADD choices in Experiment 2 and ~0.26 for Tallying choices in Experiment 1. This suggests that the subjective validity transformation (`validities ** gamma`) is poorly calibrated across the full set of trials, forcing the model to overweight the highest-validity cues.
Rationale: To fix the calibration and hit the ~0.5 empirical mark on incongruent trials without leaving the prescribed mechanism family, change the subjective validity transformation from a power function to a softmax function: `p_feat = np.exp(gamma * validities) / np.sum(np.exp(gamma * validities))`. This is a more standard and flexible way to map validities to sampling probabilities. You can also adjust the `gamma` range to `[0.0, 10.0]` to allow for both uniform sampling (gamma=0) and highly selective sampling. Keep the rest of the Probabilistic Cue Sampling mechanism exactly the same.

**Outcome of your advice:** iter 1 candidate loss=0.4334 -> CURRENT candidate loss=0.3618 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate successfully implemented the softmax transformation for subjective validities, which was ACCEPTED by the gate and improved the aggregate loss from 0.4334 to 0.3618. The predictions for incongruent trials moved in the right direction (e.g., Experiment 1 tally-consistent choices increased from 0.26 to 0.31; Experiment 2 WADD-consistent choices decreased from 0.75 to 0.66). However, the model still overestimates WADD-consistent choices and underestimates Tally-consistent choices compared to the empirical ~0.50 mark. This indicates that the model is still sampling the highest-validity cues too frequently.",
  "verdict": "regenerate",
  "rationale": "The softmax modification was successful, but the model still over-weights the most valid cues. To bring the sampling probabilities closer to uniform (which yields the ~0.50 choice proportions seen in the human data), narrow the range of the `gamma` parameter from `[0.0, 10.0]` to `[0.0, 2.0]`. This will prevent the optimizer from selecting overly high inverse temperatures and force the model to distribute its sampling more evenly across cues, while strictly remaining within the Probabilistic Cue Sampling family."
}
```

## Usage

```json
{
  "prompt_token_count": 4552,
  "candidates_token_count": 286,
  "total_token_count": 5957
}
```
