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

Propose a 'Strategy Selection / Mixture Model' theory. Instead of a single deterministic rule with exponential weights, posit that decision-makers maintain a repertoire of heuristics (e.g., Take The Best and Equal-Weight Tallying). On any given trial, a subject samples a strategy probabilistically, perhaps biased by the dispersion of validities or simply as a fixed individual difference (a mixture parameter). This would allow the model to capture the exact lexicographic tie-breaking of TTB on the majority of trials (explaining Exp 1) while correctly predicting a baseline rate of compensatory choices (explaining Exp 2) without relying on arbitrary exponential rank weights.


## CANDIDATE THEORY
Strategy Selection / Mixture Model: Decision-makers maintain a repertoire of distinct heuristics rather than relying on a single deterministic rule with arbitrary weights. On any given trial, a subject probabilistic samples a strategy from this repertoire—specifically, mixing a non-compensatory lexicographic rule (Take The Best) and a simple compensatory rule (Equal-Weight Tallying). This allows the model to capture the exact tie-breaking behavior of TTB on the majority of trials while correctly predicting a baseline rate of compensatory choices across the population, without relying on unstable exponential rank weights.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Strategy 1: Take The Best (TTB)
    order = np.argsort(-validities, kind='stable')
    score_a_ttb = 0.5
    score_b_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            score_a_ttb = 1.0
            score_b_ttb = 0.0
            break
        elif b[idx] > a[idx]:
            score_a_ttb = 0.0
            score_b_ttb = 1.0
            break
    p_ttb = np.array([score_a_ttb, score_b_ttb])
    
    # Strategy 2: Equal-Weight Tallying (EQW)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_eqw = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_eqw = np.array([0.0, 1.0])
    else:
        p_eqw = np.array([0.5, 0.5])
        
    # Mixture
    p_mix = float(parameters["p_ttb"])
    p_core = p_mix * p_ttb + (1.0 - p_mix) * p_eqw
    
    # Response noise
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- p_ttb: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's suggestion, we propose a Strategy Selection Mixture Model that interpolates between Take The Best (TTB) and Equal-Weight Tallying (EQW). Instead of relying on a single compensatory mechanism with extreme exponential weights (which failed mechanically in pi_4), this theory posits that individuals sample from a repertoire of heuristics. By mixing TTB's strict lexicographic search with EQW's simple compensatory counting, the model can natively capture both the strong non-compensatory signals in Exp 3-5 and the baseline compensatory choices in Exp 2 and 6. The mixture parameter `p_ttb` naturally accommodates individual differences in strategy preference.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.4231 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.4231.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    X = A - B
    y = np.where(data['response'] == 0, 1.0, -1.0)
    
    # Fit OLS to estimate the subjective weight of each feature
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    # We apply a carefully constructed contrast vector k to the estimated weights.
    # The design matrix X has rank 4, with a null space spanned by [1, 0, 0, 1, 0].
    # For Tallying, all features are treated equally (true weights = [1, 1, 1, 1, 1]).
    # The projection of Tallying's weights onto the row space of X is [0, 1, 1, 0, 1].
    # Our contrast vector k is orthogonal to the null space AND satisfies k @ [0, 1, 1, 0, 1] = 0.
    # This mathematically guarantees that Tallying evaluates to exactly 0 in expectation.
    # For WADD, features are weighted by validities. The expected projected weights are 
    # [0.1, 0.45, 0.3, -0.1, 0.25]. 
    # k @ expected_WADD = 0.88(0.1) + 1(0.45) - 0.88(-0.1) - 1(0.25) = 0.376 > 0.
    k = np.array([0.88, 1.0, 0.0, -0.88, -1.0])
    
    return float(np.dot(k, beta_hat))

```

**Observed (real) value:** 0.7537 (var=0.0444)
**Candidate (simulated) value:** -0.0061 (var=0.2999)
**Other theories' values on this metric (for reference):**
- pi_1: -0.6677 (var=0.1254)
- pi_2: 0.3889 (var=0.4719)
- pi_3: 0.7379 (var=0.0518)
- pi_4: 0.6553 (var=0.0675)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1 = (1, 1, 0, 0, 0)
    t2 = (0, 0, 1, 1, 1)
    
    is_t1 = (a_tuples == t1) & (b_tuples == t2)
    is_t2 = (a_tuples == t2) & (b_tuples == t1)
    
    wadd_consistent = 0
    wadd_consistent += (data.loc[is_t1, 'response'] == 0).sum()
    wadd_consistent += (data.loc[is_t2, 'response'] == 1).sum()
    
    total = is_t1.sum() + is_t2.sum()
    
    if total == 0:
        return 0.5
    return float(wadd_consistent / total)
```

**Observed (real) value:** 0.9117 (var=0.0062)
**Candidate (simulated) value:** 0.5850 (var=0.0564)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4558 (var=0.0898)
- pi_1: 0.1342 (var=0.0129)
- pi_3: 0.8692 (var=0.0105)
- pi_4: 0.8500 (var=0.0140)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    ttb_match = 0
    total = 0
    
    for a, b, resp in zip(a_tuples, b_tuples, data['response']):
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (1, 1, 0, 0) and b == (1, 0, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (0, 1, 0, 1) and b == (1, 0, 0, 0):
            ttb_match += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_match / total)
```

**Observed (real) value:** 0.8492 (var=0.0120)
**Candidate (simulated) value:** 0.5483 (var=0.0407)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8883 (var=0.0044)
- pi_2: 0.3517 (var=0.0227)
- pi_1: 0.1600 (var=0.0119)
- pi_4: 0.8708 (var=0.0079)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Features are sorted by validity in the design (index 0 is highest)
        for i in range(len(a)):
            if a[i] > b[i]:
                if resp == 0:
                    matches += 1
                break
            elif b[i] > a[i]:
                if resp == 1:
                    matches += 1
                break
                
    return float(matches) / len(data)
```

**Observed (real) value:** 0.8279 (var=0.0105)
**Candidate (simulated) value:** 0.5650 (var=0.0414)
**Other theories' values on this metric (for reference):**
- pi_2: 0.2387 (var=0.0149)
- pi_3: 0.8777 (var=0.0063)
- pi_1: 0.1702 (var=0.0126)
- pi_4: 0.8413 (var=0.0317)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_0 = data['option_b_ratings'].apply(lambda x: x[0])
    mask = a_0 != b_0
    if not mask.any():
        return 0.5
    
    chose_feat0 = ((data['response'] == 0) & (a_0 == 1)) | ((data['response'] == 1) & (b_0 == 1))
    return float(chose_feat0[mask].mean())
```

**Observed (real) value:** 0.8489 (var=0.0084)
**Candidate (simulated) value:** 0.5397 (var=0.0391)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8806 (var=0.0065)
- pi_4: 0.1378 (var=0.0072)
- pi_1: 0.2475 (var=0.0054)
- pi_2: 0.3042 (var=0.0295)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    non_ttb_choices = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_pred = 0 if a[i] > b[i] else 1
                break
        if row['response'] != ttb_pred:
            non_ttb_choices += 1
            
    rate = non_ttb_choices / len(data) if len(data) > 0 else 0.0
    
    # Variance stabilization for single-subject slices
    n_subj = data['subject_id'].nunique()
    if n_subj == 1:
        return rate / 10000.0
        
    return rate
```

**Observed (real) value:** 0.1693 (var=0.0000)
**Candidate (simulated) value:** 0.4587 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1216 (var=0.0000)
- pi_3: 0.1113 (var=0.0000)
- pi_1: 0.8669 (var=0.0000)
- pi_2: 0.7671 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The proposed Strategy Selection / Mixture Model successfully implemented the arbiter's recommendation of probabilistically mixing TTB with a compensatory rule. However, it fails dramatically across all experiments, particularly Experiments 1 and 2. Experiment 1's metric specifically isolates whether compensatory choices use validities (WADD) or ignore them (Equal-Weight Tallying), with Tallying mathematically yielding an expected value of 0. The candidate's score of -0.0061 definitively shows that EQW is the wrong compensatory rule for this dataset, whereas human subjects score ~0.75 (strongly indicating WADD). Consequently, the model also fails to match human WADD-consistency in Experiment 2 and under-predicts TTB usage in Experiments 3-6 because the likelihood optimization struggles to balance the poorly-fitting EQW rule against the data.",
  "verdict": "regenerate",
  "rationale": "The mixture model framework is the correct approach, but the choice of Equal-Weight Tallying (EQW) as the compensatory heuristic directly conflicts with the empirical data. To fix this while staying within the prescribed family, replace EQW with a Weighted Additive (WADD) rule as the second strategy in the repertoire. For WADD, compute the dot product of the features and the validities for each option, and assign probability 1.0 to the option with the higher WADD score (or use a softmax over the WADD scores). Keep the TTB implementation and the probabilistic mixture (`p_mix`) exactly as they are."
}
```

## Usage

```json
{
  "prompt_token_count": 6978,
  "candidates_token_count": 332,
  "total_token_count": 8870
}
```
