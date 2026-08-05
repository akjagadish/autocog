# feedback_iter_01

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Heuristic Decision Making domain.

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
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Propose a new theory to replace Equal Weight. Since subjects are clearly sensitive to the number of winning features but both current theories ignore the explicitly provided expert validities, a logical next step is a theory that incorporates validities. Propose 'Take-The-Best' (TTB) or a 'Weighted Tallying' heuristic. TTB would search through features in descending order of their validity and stop at the first feature that discriminates between the options, ignoring all other features and cardinal magnitudes. This would provide a strong contrast to simple Tallying by predicting that subjects heavily overweight the most valid cue rather than treating all cues equally.


## CANDIDATE THEORY
Weighted Tallying Heuristic: People make decisions by integrating across multiple features, but rather than treating all cues equally (as in simple Tallying), they weight each feature-wise win by the explicitly provided expert validity. The score for each option is the sum of the validities for the features where that option has a strictly higher rating than the other. The option with the highest validity-weighted tally is chosen. This captures both the tendency to integrate across features and the sensitivity to explicit validities.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Weighted Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sum validities for features where an option strictly wins
    score_a = np.sum(validities[a > b])
    score_b = np.sum(validities[b > a])
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the critic's feedback, we replace Take-The-Best with Weighted Tallying. TTB under-predicted the strong tallying behavior seen in Experiments 1 and 3 because it stopped at the first discriminating feature. Weighted Tallying resolves this by integrating across all features (explaining the tallying-like behavior) while still weighting the feature-wise wins by the explicitly provided expert validities, honoring the arbiter's original instruction to incorporate validities into the heuristic without falling back to a fully compensatory magnitude-based model like WADD.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3140 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.1220 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.1220.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[5, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[2, 8, 8, 8]  B=[9, 7, 7, 7]
  A=[10, 4, 4, 4]  B=[5, 5, 5, 5]
  A=[9, 3, 3, 3]  B=[2, 4, 4, 4]
  A=[6, 6, 6, 2]  B=[5, 5, 5, 9]
  A=[3, 3, 3, 10]  B=[4, 4, 4, 1]
  A=[8, 8, 2, 2]  B=[7, 7, 4, 4]
  A=[2, 2, 8, 8]  B=[4, 4, 7, 7]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_a = np.sum(a_mat > b_mat, axis=1)
    tally_b = np.sum(b_mat > a_mat, axis=1)
    
    mask = tally_a != tally_b
    if not np.any(mask):
        return 0.5
        
    preds = np.where(tally_a > tally_b, 0, 1)
    responses = data['response'].values
    
    accuracy = np.mean(preds[mask] == responses[mask])
    return float(accuracy)

```

**Observed (real) value:** 0.8000 (var=0.0667)
**Candidate trajectory (this loop):**
  - iter 1: 0.3922 (var=0.0028) (Δ vs real -0.4078)
  - iter 2 (current): 0.8506 (var=0.0081) (Δ vs real +0.0506)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.2233 (var=0.0673)
- pi_3: 0.1189 (var=0.0057)

### Experiment 2
**Design**
  A=[0, 4, 4, 4]  B=[5, 3, 3, 3]
  A=[1, 5, 5, 5]  B=[5, 4, 4, 4]
  A=[5, 4, 1, 1]  B=[1, 1, 5, 5]
  A=[5, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[4, 1, 1, 1]  B=[0, 2, 2, 2]
  A=[1, 2, 3, 5]  B=[2, 3, 4, 0]
  A=[5, 1, 1, 1]  B=[1, 2, 2, 2]
  A=[2, 5, 1, 1]  B=[3, 1, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trial 4 (A=[5,0,0,0], sum=5) and trial 1 (A=[0,4,4,4], sum=12)
    a_sum = data['option_a_ratings'].apply(sum)
    chose_a = (data['response'] == 0).astype(float)
    
    mask_t4 = (a_sum == 5)
    mask_t1 = (a_sum == 12)
    
    p_a_t4 = chose_a[mask_t4].mean() if mask_t4.any() else 0.5
    p_a_t1 = chose_a[mask_t1].mean() if mask_t1.any() else 0.5
    
    return float(p_a_t4 - p_a_t1)
```

**Observed (real) value:** -0.4000 (var=0.8400)
**Candidate trajectory (this loop):**
  - iter 1: 0.6617 (var=0.0646) (Δ vs real +1.0617)
  - iter 2 (current): -0.7033 (var=0.0509) (Δ vs real -0.3033)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5600 (var=0.6864)
- pi_1: -1.0000 (var=0.0000)
- pi_3: 0.7383 (var=0.0522)

### Experiment 3
**Design**
  A=[2, 2, 2, 0]  B=[0, 0, 0, 10]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 8]
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[9, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[5, 5, 0, 0]  B=[0, 0, 5, 6]
  A=[6, 6, 0, 0]  B=[0, 0, 5, 5]
  A=[0, 0, 0, 10]  B=[2, 2, 2, 2]
  A=[2, 2, 2, 0]  B=[0, 0, 0, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(A > B, axis=1)
    b_wins = np.sum(B > A, axis=1)
    
    a_sum = np.sum(A, axis=1)
    b_sum = np.sum(B, axis=1)
    
    tally_pref = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    ew_pref = np.where(a_sum > b_sum, 0, np.where(b_sum > a_sum, 1, -1))
    
    discordant = (tally_pref != -1) & (ew_pref != -1) & (tally_pref != ew_pref)
    
    if not np.any(discordant):
        return 0.5
        
    responses = data['response'].values
    matches = (responses[discordant] == tally_pref[discordant])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7467 (var=0.0780)
**Candidate trajectory (this loop):**
  - iter 1: 0.6150 (var=0.0027) (Δ vs real -0.1317)
  - iter 2 (current): 0.8481 (var=0.0078) (Δ vs real +0.1014)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_3: 0.1489 (var=0.0087)
- pi_2: 0.2600 (var=0.1057)

### Experiment 4
**Design**
  A=[10, 2, 2, 2]  B=[0, 3, 3, 3]
  A=[4, 4, 4, 0]  B=[3, 3, 3, 10]
  A=[10, 8, 2, 1]  B=[0, 1, 3, 4]
  A=[1, 1, 0, 0]  B=[0, 0, 10, 10]
  A=[6, 6, 6, 0]  B=[5, 5, 5, 3]
  A=[10, 2, 2, 2]  B=[1, 5, 5, 5]
  A=[0, 0, 0, 10]  B=[1, 1, 1, 0]
  A=[8, 8, 8, 0]  B=[7, 7, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_data = data[mask]
    sum_a_filtered = sum_a[mask]
    sum_b_filtered = sum_b[mask]
    
    eqw_pred = (sum_a_filtered < sum_b_filtered).astype(int)
    matches = (filtered_data['response'] == eqw_pred).mean()
    
    return float(matches)
```

**Observed (real) value:** 0.3500 (var=0.0469)
**Candidate trajectory (this loop):**
  - iter 1: 0.3719 (var=0.0018) (Δ vs real +0.0219)
  - iter 2 (current): 0.2417 (var=0.0028) (Δ vs real -0.1083)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8767 (var=0.0074)
- pi_1: 0.1667 (var=0.0000)
- pi_2: 0.9567 (var=0.0109)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate implemented Take-The-Best (TTB) as suggested by the arbiter, and it was accepted as the new running best. However, looking at the fit across experiments, TTB struggles significantly on Experiment 1 (observed = 0.80, candidate = 0.39) and Experiment 2 (observed = -0.40, candidate = 0.66). Experiment 1 explicitly measures how often subjects follow a Tallying strategy (counting winning features), and the low candidate value indicates TTB is under-predicting the tallying behavior. Experiment 3 also shows strong tallying preference (observed = 0.74). Since TTB stops at the first discriminating feature, it completely ignores the count of winning features, which contradicts the strong evidence for tallying-like behavior in these experiments.
Rationale: While TTB was an improvement over Equal Weights, it fails to capture the strong tallying behavior seen in Experiments 1 and 3. The arbiter's recommendation explicitly allowed for a 'Weighted Tallying' heuristic. I recommend regenerating the model to implement Weighted Tallying instead of TTB. In Weighted Tallying, the score for each option should be the sum of the validities for the features where that option has a strictly higher rating than the other (e.g., `score_a = sum(validities[i] for i in range(n) if a[i] > b[i])`). This remains perfectly faithful to the arbiter's instruction to incorporate explicit validities, but will better capture the empirical fact that subjects integrate across multiple features rather than stopping at the first one.

**Outcome of your advice:** iter 1 candidate loss=0.3140 -> CURRENT candidate loss=0.1220 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The switch to the Weighted Tallying heuristic has been highly successful, significantly reducing the aggregate loss from 0.3140 to 0.1220. By summing the explicit validities of the features where an option strictly wins, the model now accurately captures the strong tallying behavior seen in Experiments 1 and 3 (achieving 0.85 vs observed 0.80, and 0.85 vs observed 0.75, respectively). Furthermore, it correctly predicts the direction of the effect in Experiment 2 (-0.70 vs observed -0.40) and performs well on Experiment 4 (0.24 vs observed 0.35). The model effectively integrates across multiple features while remaining sensitive to explicit validities.",
  "verdict": "continue",
  "rationale": "The current Weighted Tallying theory provides an excellent fit across all experiments, capturing both the qualitative directions and quantitative magnitudes of the empirical data. It faithfully implements the arbiter's prescribed mechanism family and resolves the under-prediction of tallying behavior seen in the previous Take-The-Best iteration. No further regeneration is needed; this candidate should be shipped as the final model for this round."
}
```

## Usage

```json
{
  "prompt_token_count": 5653,
  "candidates_token_count": 267,
  "total_token_count": 6308
}
```
