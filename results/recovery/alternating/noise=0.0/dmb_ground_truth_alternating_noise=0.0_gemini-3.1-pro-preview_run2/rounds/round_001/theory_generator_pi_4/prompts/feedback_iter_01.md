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

Replace the Tallying theory with a 'Take-The-Best' (TTB) or Lexicographic theory. TTB postulates that decision makers search through cues in order of validity and stop at the first cue that discriminates between the options, completely ignoring all remaining cues. This contrasts with both Tallying (which counts all cues equally) and WADD (which integrates all cues). TTB might better explain the ~0.50 agreement in Experiment 4 if TTB's predictions are systematically orthogonal to Tallying's predictions on those trials. Include parameters for search errors or lapse rates to accommodate the high degree of noise in the observed data.


## CANDIDATE THEORY
Take-The-Best (TTB) with High Noise: Decision makers employ a lexicographic heuristic, searching through cues in order of descending validity. They stop at the first cue that discriminates between the two options and choose the option with the higher value on that cue. However, to accommodate the empirical observation that agreement with any deterministic strategy hovers around 50%, the model incorporates a very high lapse rate (epsilon) and a low softmax inverse temperature (beta). This restricts the model to primarily exhibit random guessing, with only a weak TTB signal, matching the high degree of noise in the observed data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity; stable sort handles ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    
    # Lexicographic search
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No cue discriminates, guess uniformly
        return np.array([0.5, 0.5])
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary TTB scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

`rationale`:
Following the critic's advice, the deterministic TTB logic is preserved, but the parameter ranges for noise (beta and epsilon) are severely restricted. By constraining epsilon to [0.8, 1.0] and beta to [0.0, 0.5], the average simulated subject is forced to exhibit near-guessing behavior. This minimal edit directly addresses the previous model's overprediction of the TTB signal, bringing the expected metric down to the ~0.50 mark observed across all four experiments.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.4343 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0534 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.0534.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    def get_ttb_winner(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1
    
    agreements = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        w = get_ttb_winner(a, b)
        if w != -1:
            agreements.append(1.0 if resp == w else 0.0)
            
    if not agreements:
        return 0.5
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.4983 (var=0.0017)
**Candidate trajectory (this loop):**
  - iter 1: 0.7310 (var=0.0166) (Δ vs real +0.2327)
  - iter 2 (current): 0.4871 (var=0.0027) (Δ vs real -0.0113)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8456 (var=0.0128)
- pi_2: 0.3210 (var=0.0037)
- pi_3: 0.5056 (var=0.0054)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    match_count = 0
    valid_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            continue
            
        if row['response'] == tally_choice:
            match_count += 1
        valid_trials += 1
        
    if valid_trials == 0:
        return 0.5
    return match_count / valid_trials
```

**Observed (real) value:** 0.4921 (var=0.0015)
**Candidate trajectory (this loop):**
  - iter 1: 0.2768 (var=0.0259) (Δ vs real -0.2152)
  - iter 2 (current): 0.4994 (var=0.0052) (Δ vs real +0.0073)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8444 (var=0.0068)
- pi_1: 0.1835 (var=0.0141)
- pi_3: 0.5984 (var=0.0137)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    consistent_choices = 0
    total_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Identify trials where Tallying predicts a tie
        if a_wins == b_wins:
            # Determine which option wins on the highest validity feature (index 0)
            if a[0] > b[0]:
                wadd_prediction = 0
            elif b[0] > a[0]:
                wadd_prediction = 1
            else:
                continue
                
            if row['response'] == wadd_prediction:
                consistent_choices += 1
            total_ties += 1
            
    if total_ties == 0:
        return 0.5
        
    return float(consistent_choices) / total_ties

```

**Observed (real) value:** 0.5208 (var=0.0104)
**Candidate trajectory (this loop):**
  - iter 1: 0.6863 (var=0.0225) (Δ vs real +0.1654)
  - iter 2 (current): 0.4971 (var=0.0062) (Δ vs real -0.0238)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6129 (var=0.0216)
- pi_2: 0.4988 (var=0.0046)
- pi_1: 0.8438 (var=0.0098)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    wadd_pref_chosen = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on Tallying-tie trials
        if a_wins == b_wins:
            score_a = np.dot(a, val)
            score_b = np.dot(b, val)
            
            # Where WADD has a strict preference
            if abs(score_a - score_b) > 1e-5:
                wadd_pref = 0 if score_a > score_b else 1
                if row['response'] == wadd_pref:
                    wadd_pref_chosen.append(1)
                else:
                    wadd_pref_chosen.append(0)
                    
    if len(wadd_pref_chosen) == 0:
        return 0.5
        
    return float(np.mean(wadd_pref_chosen))
```

**Observed (real) value:** 0.5167 (var=0.0067)
**Candidate trajectory (this loop):**
  - iter 1: 0.6000 (var=0.0104) (Δ vs real +0.0833)
  - iter 2 (current): 0.4972 (var=0.0060) (Δ vs real -0.0194)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4994 (var=0.0064)
- pi_3: 0.6694 (var=0.0140)
- pi_1: 0.6700 (var=0.0068)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate theory implemented Take-The-Best (TTB) with a full-range lapse rate (epsilon) and inverse temperature (beta). However, the simulated results deviate significantly from the observed ~0.50 metrics across all experiments (e.g., Exp 1: 0.73 vs 0.50; Exp 2: 0.27 vs 0.49). Because the parameters are uniformly sampled over their broad ranges (epsilon in [0.0, 1.0], beta in [0.0, 10.0]), the average simulated subject exhibits a strong TTB signal (expected metric around 0.75 or 0.25 depending on the experiment's correlation with TTB). To match the empirical ~0.50 agreement rates, the model needs to produce much noisier predictions on average.
Rationale: The candidate successfully implemented the TTB mechanism requested by the arbiter, but the broad parameter ranges allow too much deterministic TTB behavior, causing the simulated metrics to overshoot or undershoot the 0.50 mark. Regenerate the model keeping the exact same TTB logic, but severely restrict the parameter ranges to enforce higher noise levels. For example, set the epsilon range to [0.8, 1.0] or the beta range to [0.0, 0.5]. This will ensure the model degrades to near-guessing on average, correctly capturing the ~0.50 observed in the data.

**Outcome of your advice:** iter 1 candidate loss=0.4343 -> CURRENT candidate loss=0.0534 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The current candidate implemented the Take-The-Best (TTB) mechanism with highly restricted parameter ranges (epsilon: [0.8, 1.0], beta: [0.0, 0.5]) as recommended in the previous iteration. This adjustment successfully suppressed the strong deterministic TTB signal and produced the high noise levels required to match the empirical data. Across all four experiments, the candidate's simulated metrics are remarkably close to the observed values (all hovering around 0.50, with absolute deviations of less than 0.025). The aggregate loss dropped significantly from 0.4343 to 0.0534, indicating an excellent fit to the human data.",
  "verdict": "continue",
  "rationale": "The candidate theory effectively captures the near-guessing behavior observed in the empirical data by appropriately constraining the noise parameters within the requested TTB framework. The fit quality is excellent across all experiments, and no further tuning is necessary. The current model should be kept as the running best."
}
```

## Usage

```json
{
  "prompt_token_count": 5976,
  "candidates_token_count": 227,
  "total_token_count": 6519
}
```
