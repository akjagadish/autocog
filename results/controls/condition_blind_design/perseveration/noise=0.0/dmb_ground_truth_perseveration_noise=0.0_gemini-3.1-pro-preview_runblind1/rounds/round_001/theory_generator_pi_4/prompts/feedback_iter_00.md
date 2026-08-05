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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose 'Take The Best' (TTB) as the new theory. In TTB, decision-makers rank features by their validities and compare options lexicographically. They check the cue with the highest validity first; if one option is better, they choose it. If they tie, they move to the next most valid cue, and so on. If all cues tie, they guess randomly. This explains why subjects appear to guess on certain trials (if the top cues tie and they don't integrate the rest efficiently) while still showing sensitivity to the most predictive cues on other trials, capturing the deviations seen in Exps 3 and 4.


## CANDIDATE THEORY
Noisy Take The Best: Decision-makers use a lexicographic heuristic (Take The Best), checking features in descending order of validity and stopping at the first feature that discriminates between options. However, human application of this rule is highly noisy. A heavily restricted inverse temperature parameter accounts for the empirical observation that subjects' choices on these trials hover near chance level (0.50-0.55), rather than the highly deterministic choices (0.85+) predicted by standard TTB.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity descending; stable sort preserves original order on ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.0, 0.5]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
The arbiter recommended implementing Take The Best (TTB). However, as seen in the leaderboard (pi_1), a standard TTB model with a wide beta range severely overpredicts the metrics, scoring ~0.85-0.87 whereas human subjects score ~0.50-0.55. The data clearly shows that on these specific trials, subjects are operating very close to chance. To comply with the arbiter's suggestion while fixing the mechanistic failure of pi_1, I propose a Noisy Take The Best model. By restricting the softmax inverse temperature (beta) to a very low range [0.0, 0.5], the model captures the fact that subjects might be using lexicographic rules but with extreme inconsistency or high decision noise, bringing the predictions down to the ~0.50-0.55 range actually observed in humans.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0649 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0649.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # The validities are [0.95, 0.72, 0.73, 0.55]
    # Thus, the descending order of cue validities for Take The Best is:
    # Feature 0 (0.95), Feature 2 (0.73), Feature 1 (0.72), Feature 3 (0.55)
    cue_order = [0, 2, 1, 3]
    matches = []
    
    for a_vals, b_vals, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a = np.array(a_vals)
        b = np.array(b_vals)
        
        # Isolate trials where Tallying predicts a tie (equal number of feature wins)
        if np.sum(a > b) == np.sum(b > a):
            # Take The Best will still make a deterministic prediction based on the first discriminating cue
            for j in cue_order:
                if a[j] > b[j]:
                    matches.append(1.0 if resp == 0 else 0.0)
                    break
                elif b[j] > a[j]:
                    matches.append(1.0 if resp == 1 else 0.0)
                    break
                    
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.5500 (var=0.0600)
**Candidate (simulated) value:** 0.5375 (var=0.0147)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8792 (var=0.0084)
- pi_2: 0.4942 (var=0.0107)
- pi_3: 0.5583 (var=0.0103)

### Experiment 2
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    score = 0.0
    count = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_list = list(a)
        b_list = list(b)
        # Trial 1: Tallying ties, TTB predicts A (0)
        if a_list == [1, 1, 0, 1] and b_list == [0, 1, 1, 1]:
            score += (1 if r == 0 else 0)
            count += 1
        # Trial 2: Tallying ties, TTB predicts B (1)
        elif a_list == [0, 1, 1, 0] and b_list == [1, 0, 1, 0]:
            score += (1 if r == 1 else 0)
            count += 1
        # Trial 9: Tallying predicts A, TTB predicts B (1)
        elif a_list == [0, 1, 0, 1] and b_list == [0, 0, 1, 0]:
            score += (1 if r == 1 else 0)
            count += 1
            
    if count == 0:
        return 0.5
    return float(score / count)
```

**Observed (real) value:** 0.5333 (var=0.0267)
**Candidate (simulated) value:** 0.5300 (var=0.0171)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3489 (var=0.0083)
- pi_1: 0.8667 (var=0.0110)
- pi_3: 0.4856 (var=0.0137)

### Experiment 3
**Design**
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.6, 0.87, 0.55])
    
    wadd_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        if tally_a == tally_b:
            wadd_a = np.sum(a * val)
            wadd_b = np.sum(b * val)
            if wadd_a > wadd_b:
                wadd_match.append(1 if row['response'] == 0 else 0)
            elif wadd_b > wadd_a:
                wadd_match.append(1 if row['response'] == 1 else 0)
                
    return float(np.mean(wadd_match)) if len(wadd_match) > 0 else 0.5
```

**Observed (real) value:** 0.4900 (var=0.0624)
**Candidate (simulated) value:** 0.5442 (var=0.0171)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5425 (var=0.0123)
- pi_2: 0.5017 (var=0.0103)
- pi_1: 0.8500 (var=0.0156)

### Experiment 4
**Design**
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    is_t2 = (a_str == '0110') & (b_str == '1001')
    is_t16 = (a_str == '1011') & (b_str == '0111')
    
    t2_wadd_choices = (data.loc[is_t2, 'response'] == 1).sum()
    t16_wadd_choices = (data.loc[is_t16, 'response'] == 0).sum()
    
    total_trials = is_t2.sum() + is_t16.sum()
    
    if total_trials == 0:
        return 0.5
        
    return float((t2_wadd_choices + t16_wadd_choices) / total_trials)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Candidate (simulated) value:** 0.5283 (var=0.0216)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4650 (var=0.0192)
- pi_3: 0.5550 (var=0.0238)
- pi_1: 0.8683 (var=0.0161)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The Noisy Take The Best candidate performs exceptionally well across all four experiments. By restricting the inverse temperature (beta) and adding an epsilon-greedy component, the model accurately captures the human tendency to perform near chance level (0.50-0.55) on these specific trial types, avoiding the severe overprediction (0.85+) seen in standard deterministic TTB models. The simulated values closely track the observed data: Exp 1 (0.55 vs 0.5375), Exp 2 (0.533 vs 0.53), Exp 3 (0.49 vs 0.544), and Exp 4 (0.50 vs 0.528).",
  "verdict": "continue",
  "rationale": "The candidate was accepted and substantially reduced the aggregate loss. It faithfully implemented the arbiter's recommendation within the prescribed mechanism family (Take The Best) and correctly diagnosed that human choices on these specific tie-breaking configurations are heavily noise-driven. The current theory and its parameter ranges are well-calibrated and explain the experimental data effectively."
}
```

## Usage

```json
{
  "prompt_token_count": 6163,
  "candidates_token_count": 248,
  "total_token_count": 6666
}
```
