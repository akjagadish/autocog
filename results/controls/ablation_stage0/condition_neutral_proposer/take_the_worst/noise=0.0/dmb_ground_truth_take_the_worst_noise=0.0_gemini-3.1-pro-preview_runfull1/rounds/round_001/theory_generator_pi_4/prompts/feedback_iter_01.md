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
- The recommendation below acts on THEORY 1 (= `pi_3`).

Replace the Weighted Additive (WADD) theory with a 'Take The Best' (TTB) or 'Lexicographic' theory. TTB represents the other extreme of the bounded rationality spectrum: it searches through cues in order of validity, stops at the first cue that discriminates between the options, and entirely ignores all remaining cues. Comparing Tallying (which integrates all cues but ignores validities) against TTB (which strictly adheres to validity order but ignores cue quantity) will provide a stringent test of whether subjects employ non-compensatory stopping rules or non-compensatory weighting rules.


## CANDIDATE THEORY
Take The Best (Lexicographic Rule): People evaluate options by considering cues one at a time, strictly in order of their validity. The process stops at the first cue that discriminates between the options. To account for response noise and subjective confidence, the choice probability is derived via a softmax function over the validity of the discriminating cue, plus a uniform lapse rate. This makes it a non-compensatory stopping rule while allowing choice determinism to scale with the importance of the deciding feature.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Lexicographic stopping rule
    score_a, score_b = 0.0, 0.0
    for j in cue_order:
        if a[j] > b[j]:
            score_a = val[j]
            break
        elif b[j] > a[j]:
            score_b = val[j]
            break
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the critic's explicit instruction, I have replaced the Equal-Weight mechanism with a strict Take The Best (TTB) lexicographic rule. The model iterates through cues in order of validity and stops at the first discriminating cue, completely ignoring all subsequent cues. To improve fit and introduce graded determinism as suggested, the model assigns the validity of the discriminating cue as the score for the winning option and passes it through a softmax function with an epsilon lapse rate.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0258 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.9797 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.0258.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    
    winner = np.zeros(len(data))
    for i in range(len(data)):
        w = -1
        for j in range(a_mat.shape[1]):
            if diff[i, j] > 0:
                w = 0
                break
            elif diff[i, j] < 0:
                w = 1
                break
        winner[i] = w
        
    match = (data['response'].values == winner)
    return float(np.mean(match))
```

**Observed (real) value:** 0.2350 (var=0.0036)
**Candidate trajectory (this loop):**
  - iter 1: 0.2381 (var=0.0068) (Δ vs real +0.0031)
  - iter 2 (current): 0.8708 (var=0.0060) (Δ vs real +0.6358)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8379 (var=0.0095)
- pi_2: 0.2358 (var=0.0067)
- pi_3: 0.2625 (var=0.0084)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option ratings into 2D arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Calculate the number of features each option strictly wins
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    # Filter for trials where Tallying makes a deterministic prediction (no ties)
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    # Tallying predicts the option with more winning features
    tallying_choice = (b_wins > a_wins).astype(int)
    
    responses = data['response'].values
    
    # Calculate the proportion of choices consistent with Tallying
    consistent = (responses[mask] == tallying_choice[mask]).astype(float)
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.8275 (var=0.0100)
**Candidate trajectory (this loop):**
  - iter 1: 0.8397 (var=0.0135) (Δ vs real +0.0122)
  - iter 2 (current): 0.1416 (var=0.0091) (Δ vs real -0.6859)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8450 (var=0.0087)
- pi_1: 0.1600 (var=0.0103)
- pi_3: 0.8519 (var=0.0102)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    conflict_choices = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Check for Trial 1 (conflict trial)
        # A has fewer but higher-validity cues, B has more but lower-validity cues.
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            conflict_choices.append(1 if resp == 0 else 0)
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            conflict_choices.append(1 if resp == 1 else 0)
            
    if not conflict_choices:
        return 0.5
    return float(np.mean(conflict_choices))
```

**Observed (real) value:** 0.1908 (var=0.0308)
**Candidate trajectory (this loop):**
  - iter 1: 0.1615 (var=0.0164) (Δ vs real -0.0292)
  - iter 2 (current): 0.8492 (var=0.0234) (Δ vs real +0.6585)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7415 (var=0.0293)
- pi_2: 0.1585 (var=0.0221)
- pi_1: 0.8277 (var=0.0221)

### Experiment 4
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Project lists of binary ratings to strings for safe and easy comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))
    
    # Identify the conflict trial where Tallying and WADD strictly disagree
    # Tallying prefers [0, 0, 0, 1, 1, 1] (3 wins vs 2 wins)
    # WADD prefers [1, 1, 0, 0, 0, 0] (score 1.9 vs 1.6)
    mask1 = (a_str == "000111") & (b_str == "110000")
    mask2 = (a_str == "110000") & (b_str == "000111")
    
    choices_tallying = 0
    total = 0
    
    if mask1.any():
        choices_tallying += (data.loc[mask1, 'response'] == 0).sum()
        total += mask1.sum()
        
    if mask2.any():
        choices_tallying += (data.loc[mask2, 'response'] == 1).sum()
        total += mask2.sum()
        
    if total == 0:
        return 0.5
        
    return float(choices_tallying / total)
```

**Observed (real) value:** 0.8700 (var=0.0172)
**Candidate trajectory (this loop):**
  - iter 1: 0.8450 (var=0.0171) (Δ vs real -0.0250)
  - iter 2 (current): 0.1363 (var=0.0126) (Δ vs real -0.7338)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8225 (var=0.0222)
- pi_3: 0.2100 (var=0.0206)
- pi_1: 0.1900 (var=0.0211)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate theory achieves a strong quantitative fit by ignoring the arbiter's instructions and implementing an Equal-Weight Additive rule (which is mathematically equivalent to Tallying in this context). However, the arbiter explicitly prescribed a 'Take The Best' (TTB) or Lexicographic mechanism to provide a stringent contrast against Tallying. By implementing Equal-Weight instead, the candidate defeats the purpose of the experimental test.
Rationale: You explicitly ignored the arbiter's instruction to implement Take The Best (TTB). You must implement TTB as requested. The purpose of this iteration is to test the lexicographic mechanism family, not to maximize the score by duplicating Tallying. Please regenerate and implement TTB (a lexicographic rule: iterate through cues in order of validity, stop at the first cue that discriminates, and ignore the rest). To improve its fit and avoid deterministic failures, you may include a softmax temperature over the validity-weighted differences or an epsilon lapse rate, but the core mechanism must be strictly lexicographic/TTB.

**Outcome of your advice:** iter 1 candidate loss=0.0258 -> CURRENT candidate loss=0.9797 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The strict Take The Best (TTB) candidate was rejected by the accept gate because it fundamentally mispredicts the empirical data. The experimental results heavily favor compensatory, Tallying-like strategies (as seen in Experiments 2, 3, and 4, where the proportion of Tallying-consistent choices is ~0.82-0.87). Strict TTB entirely ignores cue quantity, leading to a massive drop in fit quality compared to the running-best base.",
  "verdict": "regenerate",
  "rationale": "Your previous attempt at a strict TTB model was rejected because it cannot capture the highly compensatory behavior shown by the subjects. Since you must adhere to the arbiter's prescription of the Lexicographic/TTB family while improving upon the accepted base, you should implement a 'Soft Lexicographic' or 'Probabilistic Stopping' model. For instance, instead of strictly stopping at the first discriminating cue, the model could have a probability `gamma` of stopping and making a decision based on that cue, and `1-gamma` of continuing the search to integrate further cues. Alternatively, you could implement a weighted additive model where the weights decay exponentially according to the validity rank (a continuous relaxation of lexicographic search). This allows the model to remain within the prescribed family while flexibly capturing the compensatory (tallying) behavior if the stopping probability/decay rate is tuned appropriately."
}
```

## Usage

```json
{
  "prompt_token_count": 5740,
  "candidates_token_count": 303,
  "total_token_count": 7180
}
```
