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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Propose a new theory such as 'Stochastic Feature Comparison' or 'Tallying with Attention Allocation'. Instead of a linear additive validity bias, suggest that subjects primarily tally, but their attention to features is probabilistically driven by validities. If a tie occurs, they do not default to validities to break it, but guess randomly because the tallying process itself has concluded. Alternatively, propose a 'Lexicographic Tallying' model where validities only matter if the tally difference is large enough to trigger confidence, or a 'Feature Cancellation' model where shared features are ignored and validities only weight the unique features.


## CANDIDATE THEORY
Tally-Gated Validity Bias: Decision-makers primarily rely on a Tallying heuristic, simply counting the number of positive features for each option. If the tally results in a tie, the decision process abruptly concludes and they guess randomly, without falling back on cue validities. However, if there is a difference in tally scores, the strength of their preference is modulated by the explicit cue validities. This means validities act as a secondary confidence-adjuster rather than a tie-breaker, explaining why validity bias appears in overall choices but is absent when options have an equal number of positive features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    # Primary strategy: Tallying
    tally_scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_val = float(parameters["w_val"])
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # If tallying results in a tie, guess randomly (no validity tie-breaking)
    if tally_scores[0] == tally_scores[1]:
        p_core = np.ones(2) / 2.0
    else:
        # If there is a tally difference, validities modulate the response strength
        val_scores = stim @ validities
        scores = (1.0 - w_val) * tally_scores + w_val * val_scores
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 3.0]
- epsilon: [0.0, 0.5]
- w_val: [0.0, 0.5]
- validities: validities

`rationale`:
The previous model (pi_4) applied a continuous validity bias to all choices, which failed to capture human behavior on tally ties. As seen in Experiment 6, humans exhibit a tie-breaking probability of ~0.5, meaning they do not use validities to break ties. However, Experiment 5 shows a positive validity bias overall. To reconcile this, we propose the 'Tally-Gated Validity Bias' theory: subjects primarily use a tallying heuristic. If the tally results in a tie, the decision process concludes and they guess randomly. If the tally yields a clear winner, their choice probability is modulated by the cue validities. This preserves the validity bias on non-tied trials (improving Exp 5) while perfectly capturing the random guessing behavior on tied trials (improving Exp 6).

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0571 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0571.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    weights = np.array([1000, 100, 10, 1])
    score_diff = diff @ weights
    ttb_preds = np.where(score_diff > 0, 0, 1)
    matches = (data['response'].values == ttb_preds)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3504 (var=0.0026)
**Candidate (simulated) value:** 0.3329 (var=0.0033)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8598 (var=0.0115)
- pi_2: 0.4281 (var=0.0279)
- pi_3: 0.3846 (var=0.0045)
- pi_4: 0.3565 (var=0.0084)

### Experiment 2
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    mask = sum_b > sum_a
    if mask.sum() == 0:
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6741 (var=0.0053)
**Candidate (simulated) value:** 0.7570 (var=0.0152)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6756 (var=0.0280)
- pi_1: 0.1400 (var=0.0055)
- pi_3: 0.6930 (var=0.0104)
- pi_4: 0.7648 (var=0.0137)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify Trial 3 where both options have exactly 1 positive feature.
    # Tallying predicts a perfect tie (score 1 vs 1) for all subjects.
    # WADD predicts a preference based on subjective weights for the specific features.
    def is_t3(row):
        return sum(row['option_a_ratings']) == 1 and sum(row['option_b_ratings']) == 1
    
    mask = data.apply(is_t3, axis=1)
    t3_data = data[mask]
    
    if len(t3_data) == 0:
        return 0.0
        
    # Calculate each subject's absolute deviation from 0.5 probability of choosing A
    subject_devs = []
    for subj, subj_df in t3_data.groupby('subject_id'):
        pA = (subj_df['response'] == 0).mean()
        subject_devs.append(abs(pA - 0.5))
        
    if not subject_devs:
        return 0.0
        
    return float(np.mean(subject_devs))
```

**Observed (real) value:** 0.1000 (var=0.0041)
**Candidate (simulated) value:** 0.0825 (var=0.0037)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0988 (var=0.0038)
- pi_2: 0.2900 (var=0.0195)
- pi_1: 0.3362 (var=0.0144)
- pi_4: 0.0875 (var=0.0048)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def is_11000(x):
        return tuple(x) == (1, 1, 0, 0, 0)
    
    def is_00111(x):
        return tuple(x) == (0, 0, 1, 1, 1)

    a_11000 = data['option_a_ratings'].apply(is_11000)
    b_00111 = data['option_b_ratings'].apply(is_00111)
    
    a_00111 = data['option_a_ratings'].apply(is_00111)
    b_11000 = data['option_b_ratings'].apply(is_11000)
    
    trial_1 = a_11000 & b_00111
    trial_2 = a_00111 & b_11000
    
    mask = trial_1 | trial_2
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
        
    chose_00111 = (trial_1 & (data['response'] == 1)) | (trial_2 & (data['response'] == 0))
    chose_00111_subset = chose_00111[mask]
    
    return float(chose_00111_subset.mean())
```

**Observed (real) value:** 0.7017 (var=0.0062)
**Candidate (simulated) value:** 0.6783 (var=0.0156)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4475 (var=0.0872)
- pi_3: 0.6583 (var=0.0175)
- pi_1: 0.1433 (var=0.0129)
- pi_4: 0.6900 (var=0.0185)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Use the exact validities specified in the experimental design
    validities = np.array([1.0, 0.9, 0.6, 0.5, 0.5])
    
    # Safely convert lists to numpy arrays for vectorized operations
    a_mat = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b_mat = np.array(data['option_b_ratings'].tolist(), dtype=float)
    
    # Compute the Tally Difference and Validity Difference (Option B - Option A)
    tally_diff = b_mat.sum(axis=1) - a_mat.sum(axis=1)
    val_diff = b_mat.dot(validities) - a_mat.dot(validities)
    
    df = pd.DataFrame({
        'td': tally_diff,
        'vd': val_diff,
        'resp': data['response'].values
    })
    
    # Compute the empirical mean response per subject per tally difference
    if 'subject_id' in data.columns:
        df['subject_id'] = data['subject_id'].values
        td_means = df.groupby(['subject_id', 'td'])['resp'].transform('mean')
    else:
        td_means = df.groupby('td')['resp'].transform('mean')
        
    # The residual choice perfectly partials out the main effect of the Tallying heuristic
    res = df['resp'] - td_means
    
    # The covariance between the residual choice and the validity difference 
    # isolates the unique contribution of the Validity Bias.
    return float(np.mean(res * df['vd']))
```

**Observed (real) value:** 0.0049 (var=0.0003)
**Candidate (simulated) value:** 0.0002 (var=0.0004)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0017 (var=0.0004)
- pi_4: 0.0031 (var=0.0003)
- pi_1: 0.0838 (var=0.0006)
- pi_2: 0.0219 (var=0.0026)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    
    def get_tally(x):
        return sum(x)
        
    def get_val(x):
        return sum(v * r for v, r in zip(validities, x))
        
    tally_a = data['option_a_ratings'].apply(get_tally)
    tally_b = data['option_b_ratings'].apply(get_tally)
    
    # Only consider trials where the tally scores are equal (ties)
    tie_mask = tally_a == tally_b
    if tie_mask.sum() == 0:
        return 0.5
        
    tie_data = data[tie_mask]
    
    val_a = tie_data['option_a_ratings'].apply(get_val)
    val_b = tie_data['option_b_ratings'].apply(get_val)
    
    # Determine which option has the higher validity sum
    higher_val_is_b = (val_b > val_a).astype(int)
    
    # Calculate the proportion of choices that align with the higher validity option
    match = (tie_data['response'] == higher_val_is_b).mean()
    
    return float(match)
```

**Observed (real) value:** 0.4964 (var=0.0074)
**Candidate (simulated) value:** 0.4923 (var=0.0098)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5800 (var=0.0079)
- pi_3: 0.4974 (var=0.0052)
- pi_1: 0.8615 (var=0.0095)
- pi_2: 0.5892 (var=0.0508)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory 'Tally-Gated Validity Bias' was ACCEPTED by the gate and achieves an excellent fit on Experiment 6 by correctly predicting random guessing (0.5) on tally ties. However, its performance on Experiment 5 dropped significantly (0.0002 vs observed 0.0049) compared to the previous iteration. Experiment 5 measures the covariance between residual choice and validity differences, partialling out the tally difference. Because the model now completely zeros out validity bias on tied trials, all the validity bias in Experiment 5 must come from the non-tied trials. The current parameterization (e.g., beta up to 3.0, w_val up to 0.5) likely causes the tally difference to dominate and saturate the softmax on non-tied trials, leaving too little variance for the validities to modulate the choice probabilities.",
  "verdict": "regenerate",
  "rationale": "Keep the exact 'Tally-Gated Validity Bias' mechanism (random guessing on ties, validity modulation on non-ties) as it perfectly captures Experiment 6. To restore the validity bias observed in Experiment 5, adjust the parameter ranges to prevent the softmax from saturating on non-tied trials. Specifically, lower the upper bound of `beta` (e.g., [0.1, 1.5]) or increase the upper bound of `w_val` (e.g., [0.0, 1.0]) so that validities can exert a stronger, observable modulatory effect when a tally difference exists."
}
```

## Usage

```json
{
  "prompt_token_count": 6870,
  "candidates_token_count": 340,
  "total_token_count": 8811
}
```
