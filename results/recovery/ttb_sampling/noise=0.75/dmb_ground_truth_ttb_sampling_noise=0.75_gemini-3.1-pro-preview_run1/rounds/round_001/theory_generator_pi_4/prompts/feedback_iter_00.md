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

Propose a brand-new theory to replace the Weighted Additive (WADD) model. Since subjects exhibit predominantly non-compensatory behavior but with some noise (as captured well by Theory 1), a strong competitor could be a 'Strategy Selection' or 'Mixture' theory. This new theory would posit that a population consists of a majority of TTB users and a minority of WADD (or Tallying) users, capturing the fact that aggregate behavior is slightly softer than pure TTB. Alternatively, propose a 'Two-Stage Heuristic' where subjects first attempt a fast tally of features and only fall back to rigorous validity-based lexicographic search if the tally is too close.


## CANDIDATE THEORY
Strategy Mixture Model (TTB and Tallying): Decision-makers exhibit heterogeneous strategy use, with the population consisting of a majority of Take-The-Best (TTB) users and a minority of Tallying users. Rather than a pure one-reason heuristic or a fully compensatory process, aggregate behavior reflects a probabilistic mixture. On any given trial, a subject's choice is a weighted blend of a lexicographic TTB process (which chooses based on the single most valid discriminating cue) and a Tallying process (which counts the number of feature-wise wins for each option). This captures the empirical finding that aggregate behavior is predominantly non-compensatory (driven by high-validity cues) but slightly softer than pure TTB due to the minority influence of compensatory tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    w_tally = float(parameters["w_tally"])
    epsilon = float(parameters["epsilon"])
    
    # --- TTB Prediction ---
    order = np.argsort(validities)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for f in order:
        if a[f] > b[f]:
            ttb_scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            ttb_scores = np.array([0.0, validities[f]])
            break
            
    if ttb_scores[0] == ttb_scores[1]:
        p_ttb = np.array([0.5, 0.5])
    else:
        z = beta_ttb * (ttb_scores - ttb_scores.max())
        e = np.exp(z)
        p_ttb = e / e.sum()
        
    # --- Tallying Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    
    if tally_scores[0] == tally_scores[1]:
        p_tally = np.array([0.5, 0.5])
    else:
        z = beta_tally * (tally_scores - tally_scores.max())
        e = np.exp(z)
        p_tally = e / e.sum()
        
    # --- Mixture ---
    p_core = (1.0 - w_tally) * p_ttb + w_tally * p_tally
    
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta_ttb: [0.0, 10.0]
- beta_tally: [0.0, 5.0]
- w_tally: [0.0, 0.5]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's suggestion, this model replaces the strictly compensatory WADD framework with a Strategy Mixture theory. It explicitly instantiates the hypothesis that the population is composed of a majority of TTB users and a minority of Tallying users. By restricting the mixture weight `w_tally` to [0.0, 0.5], the model structurally enforces that TTB remains the dominant strategy while allowing Tallying to soften the aggregate predictions. This addresses the mechanistic failure of pure TTB (which is too rigid) and pure WADD (which overly penalizes high-validity cues), providing a balanced fit to the 'compensatory' and 'conflict' trials highlighted in the experiments.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.3210 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.3210.

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
**Candidate (simulated) value:** 0.2338 (var=0.0121)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1508 (var=0.0038)
- pi_2: 0.2325 (var=0.0158)
- pi_3: 0.1783 (var=0.0133)

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
**Candidate (simulated) value:** 0.6238 (var=0.0253)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5181 (var=0.0768)
- pi_1: 0.1275 (var=0.0098)
- pi_3: 0.6200 (var=0.0141)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    trial_count = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        sa = sum(a)
        sb = sum(b)
        
        if sa == sb:
            continue
            
        ttb = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
                
        # Target "compensatory" trials where TTB chooses the option with fewer positive features
        if (ttb == 0 and sa < sb) or (ttb == 1 and sb < sa):
            trial_count += 1
            if row['response'] == ttb:
                match_count += 1
                
    if trial_count == 0:
        return 0.5
        
    return float(match_count / trial_count)
```

**Observed (real) value:** 0.5920 (var=0.0046)
**Candidate (simulated) value:** 0.6547 (var=0.0170)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6173 (var=0.0123)
- pi_2: 0.3450 (var=0.0137)
- pi_1: 0.1400 (var=0.0109)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    v = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    wadd_matches = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        resp = row['response']
        
        wadd_a = np.dot(a, v)
        wadd_b = np.dot(b, v)
        wadd_pred = 0 if wadd_a > wadd_b else 1
        
        diff = a - b
        ttb_pred = None
        for i in range(len(v)):
            if diff[i] == 1:
                ttb_pred = 0
                break
            elif diff[i] == -1:
                ttb_pred = 1
                break
                
        if ttb_pred is not None and wadd_pred != ttb_pred:
            wadd_matches.append(1 if resp == wadd_pred else 0)
            
    if not wadd_matches:
        return 0.5
        
    return float(np.mean(wadd_matches))
```

**Observed (real) value:** 0.4011 (var=0.0034)
**Candidate (simulated) value:** 0.3436 (var=0.0145)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6883 (var=0.0243)
- pi_3: 0.3628 (var=0.0139)
- pi_1: 0.7967 (var=0.0065)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The Strategy Mixture Model (TTB and Tallying) performs quite well, capturing the predominantly non-compensatory aggregate behavior while allowing for softer, compensatory responses. It aligns closely with the real data in Experiments 2 and 3. However, it overestimates the deviation in Experiment 1 (0.2338 vs real 0.1217) and slightly underestimates the WADD-match rate in Experiment 4 (0.3436 vs real 0.4011).",
  "verdict": "regenerate",
  "rationale": "The current model restricts the mixture weight `w_tally` to [0.0, 0.5]. Because parameters are fitted per subject, this upper bound artificially prevents the model from capturing subjects who predominantly or exclusively use the Tallying strategy. To properly represent a population with a *mixture of strategies* across individuals (where some subjects might be pure Tallying users), expand the parameter range of `w_tally` to [0.0, 1.0]. Keep the rest of the mechanism exactly the same."
}
```

## Usage

```json
{
  "prompt_token_count": 5661,
  "candidates_token_count": 243,
  "total_token_count": 6858
}
```
