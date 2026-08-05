# feedback_iter_03

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

Propose 'Take The Best' (TTB) as a new theory. TTB states that people compare options by inspecting features in descending order of their subjective validity. They stop at the first feature that discriminates between the two options and choose the option with the positive value on that feature. If no features discriminate, they guess. This introduces a non-compensatory lexicographic mechanism that differs from both the equal-weight exhaustive counting of Tallying and the compensatory exhaustive weighting of WADD.


## CANDIDATE THEORY
Take The Best (TTB) with Probabilistic Stopping: Decision-makers use a lexicographic heuristic, ranking features by subjective validity and stopping at the first discriminating feature. However, rather than making a strictly deterministic choice based on this feature, the decision is probabilistic. The probability of choosing the winning option scales with the validity of that discriminating feature via a softmax function with a highly regularized inverse temperature (beta). This allows confidence to vary depending on how valid the deciding feature is, capturing empirical noise without relying entirely on a global random lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    beta = float(parameters["beta"])
    
    a, b = stim[0], stim[1]
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            scores = np.array([0.0, validities[f]])
            break
            
    # If no feature discriminates, default to uniform guessing
    if scores[0] == scores[1]:
        p_core = np.array([0.5, 0.5])
    else:
        # Probabilistic choice scaling with the validity of the discriminating feature
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    # Apply lapse rate
    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.0, 2.5]
- epsilon: [0.0, 1.0]
- validities: validities

`rationale`:
Following the critic's feedback, the upper bound of the `beta` parameter range is further reduced from 5.0 to 2.5. This tighter range forces the model to produce softer choice probabilities on the discriminating features, bringing the overpredicted determinism in Experiment 1 and the overpredicted preference for the single most valid feature in Experiment 2 closer to the empirical levels.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.6292 -> ACCEPTED
- iter 2: loss=0.6189 -> ACCEPTED
- iter 3: loss=0.4607 -> ACCEPTED
- iter 4 (current candidate you are grading): loss=0.1612 -> ACCEPTED
Running-best (last accepted) base: iter 4 at loss=0.1612.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_dev(df):
        m3 = df['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x)) == (1, 0, 0, 0, 0)
        m4 = df['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x)) == (0, 1, 1, 0, 0)
        
        dev = 0.0
        count = 0
        if m3.sum() > 0:
            dev += abs((df.loc[m3, 'response'] == 0).mean() - 0.5)
            count += 1
        if m4.sum() > 0:
            dev += abs((df.loc[m4, 'response'] == 0).mean() - 0.5)
            count += 1
            
        return dev / count if count > 0 else 0.0

    return float(data.groupby('subject_id').apply(get_dev).mean())
```

**Observed (real) value:** 0.1217 (var=0.0056)
**Candidate trajectory (this loop):**
  - iter 1: 0.2433 (var=0.0191) (Δ vs real +0.1217)
  - iter 2: 0.2346 (var=0.0201) (Δ vs real +0.1129)
  - iter 3: 0.2104 (var=0.0181) (Δ vs real +0.0888)
  - iter 4 (current): 0.1567 (var=0.0110) (Δ vs real +0.0350)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1508 (var=0.0038)
- pi_2: 0.2325 (var=0.0158)

### Experiment 2
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    count = 0
    match_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        sum_a = sum(a)
        sum_b = sum(b)
        
        # Isolate conflict trials where one option has fewer features (sum=2 vs sum=3)
        # but the fewer features include the most predictive one (index 0 is 1).
        if sum_a == 2 and sum_b == 3 and a[0] == 1:
            match_trials += 1
            if resp == 0:  # Chose the option with fewer but more valid features
                count += 1
        elif sum_b == 2 and sum_a == 3 and b[0] == 1:
            match_trials += 1
            if resp == 1:  # Chose the option with fewer but more valid features
                count += 1
                
    if match_trials == 0:
        return 0.0
    return count / match_trials
```

**Observed (real) value:** 0.6062 (var=0.0088)
**Candidate trajectory (this loop):**
  - iter 1: 0.7731 (var=0.0286) (Δ vs real +0.1669)
  - iter 2: 0.7556 (var=0.0303) (Δ vs real +0.1494)
  - iter 3: 0.6719 (var=0.0192) (Δ vs real +0.0656)
  - iter 4 (current): 0.6031 (var=0.0196) (Δ vs real -0.0031)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5181 (var=0.0768)
- pi_1: 0.1275 (var=0.0098)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The Take The Best (TTB) model successfully captures the qualitative effect in Experiment 2 (preferring the option with the single most valid feature despite having fewer total features), yielding an accepted improvement in loss. However, it overpredicts the strength of this effect (0.77 vs 0.61) and also overpredicts the choice determinism in Experiment 1 (deviation of 0.24 vs 0.12). Currently, a single lapse rate (`epsilon`) has to do all the work of softening the strict deterministic choices of TTB. Because a very high lapse rate would penalize the likelihood on easier trials, the fitted epsilon compromises, leaving the model too deterministic on the focal trials.
Rationale: Keep the TTB mechanism family (lexicographic search, stopping at the first discriminating feature), but replace the strict deterministic choice (1.0 vs 0.0) with a probabilistic choice that scales with the validity of the discriminating feature. Specifically, introduce a `beta` (inverse temperature) parameter. When the search stops at feature `f`, the probability of choosing the option that wins on `f` should be computed using a softmax or logistic function based on `beta * validity[f]`. This allows the model to be less confident when the discriminating feature has low validity, fitting the empirical deviations better without requiring a massive global lapse rate.

**Outcome of your advice:** iter 1 candidate loss=0.6292 -> iter 2 candidate loss=0.6189 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The introduction of a probabilistic stopping rule (softmax over the discriminating feature's validity) was successful, leading to an accepted improvement in aggregate loss (0.6292 -> 0.6189). However, the model still overpredicts choice determinism in Experiment 1 (0.23 vs 0.12) and overestimates the preference for the single most valid feature in Experiment 2 (0.75 vs 0.60). This suggests that the model is still acting too deterministically. The parameter range for the inverse temperature `beta` currently extends up to 20.0, which allows the softmax to approximate a strict step function, partially defeating the purpose of the probabilistic choice mechanism.
Rationale: Keep the exact same TTB mechanism and probabilistic choice rule, as this direction was accepted and reduced the loss. To further pull the predictions down toward the empirical values, tighten the upper bound of the `beta` parameter range. Change the range of `beta` from [0.1, 20.0] to [0.0, 5.0]. This will prevent the model from falling back into near-deterministic behavior and force it to better utilize the probabilistic scaling, which should soften the predictions in both Experiment 1 and Experiment 2.

**Outcome of your advice:** iter 2 candidate loss=0.6189 -> iter 3 candidate loss=0.4607 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent)
Verdict: regenerate
Interpretation: The model's performance continues to improve significantly. By tightening the upper bound of the `beta` parameter to 5.0, the aggregate loss dropped from 0.6189 to 0.4607, an accepted and substantial improvement. The predictions are now much closer to the empirical data: in Experiment 1, the deviation dropped from 0.23 to 0.21 (real is 0.12), and in Experiment 2, the choice proportion dropped from 0.75 to 0.67 (real is 0.60). However, the model still slightly overpredicts the determinism of choices in both experiments, indicating that the fitter is still finding compromises that leave focal trials a bit too deterministic.
Rationale: The previous advice to restrict the `beta` range was highly successful and ACCEPTED by the gate. To fully close the gap and bring the predictions down to the empirical levels (0.12 for Exp 1 and 0.60 for Exp 2), we should push this exact same mechanism slightly further. Tighten the `beta` parameter range once more, from [0.0, 5.0] to [0.0, 2.5]. This will further regularize the softmax, forcing the model to produce softer choice probabilities on the discriminating features without altering the core TTB mechanism.

**Outcome of your advice:** iter 3 candidate loss=0.4607 -> CURRENT candidate loss=0.1612 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The model's performance has reached an excellent level. Tightening the upper bound of the `beta` parameter to 2.5 led to a massive reduction in aggregate loss from 0.4607 to 0.1612. The candidate now perfectly matches the empirical data in Experiment 2 (0.6031 vs 0.6062) and is very close in Experiment 1 (0.1567 vs 0.1217). The probabilistic Take The Best mechanism, constrained to produce softer choices, accurately captures the human behavior across both experiments.",
  "verdict": "continue",
  "rationale": "The current model fits the data exceptionally well. The mechanism is working exactly as intended, and the parameter ranges are properly calibrated to prevent over-determinism. No further modifications are needed; the theory should be kept as the running-best baseline."
}
```

## Usage

```json
{
  "prompt_token_count": 5196,
  "candidates_token_count": 200,
  "total_token_count": 5657
}
```
