# feedback_iter_02

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

Propose a 'Mixture of Heuristics' or 'Strategy Selection' theory. This new theory should posit that the population consists of a mixture of decision-makers: a proportion of the trials/subjects rely on a simple Tallying (equal-weight) heuristic, while the rest use a validity-sensitive strategy like WADD or Take-The-Best. Alternatively, propose a 'Rank-Weighted Additive' theory where features are weighted by their ordinal rank rather than their cardinal validities, naturally interpolating between Tallying and WADD.


## CANDIDATE THEORY
Mixture of Heuristics (Tallying and WADD)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- WADD ---
    scores_wadd = stim @ val
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.1]
- w_tally: [0.0, 1.0]
- validities: validities

`rationale`:
Following the critic's advice, we revert to the successful probability-mixture architecture from Iteration 1 but replace TTB with WADD (using objective validities as weights) to capture validity-sensitive compensatory behavior. To address the issue of predictions hovering near 0.5, we restrict the uniform lapse parameter `epsilon` to [0.0, 0.1] and `beta` to [0.1, 10.0]. We also explicitly mix Tallying and WADD probabilities via a `w_tally` parameter. This allows the model to capture the strong (~67%) preference for Tallying over validity-based strategies observed empirically without decaying into random noise.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2201 -> ACCEPTED
- iter 2: loss=0.5739 -> REJECTED
- iter 3 (current candidate you are grading): loss=0.1813 -> ACCEPTED
Running-best (last accepted) base: iter 3 at loss=0.1813.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = a_sum != b_sum
    if not mask.any():
        return 0.5
    filtered_data = data[mask]
    a_sum_f = a_sum[mask]
    b_sum_f = b_sum[mask]
    chose_fewer = ((filtered_data['response'] == 0) & (a_sum_f < b_sum_f)) | ((filtered_data['response'] == 1) & (b_sum_f < a_sum_f))
    return float(chose_fewer.mean())
```

**Observed (real) value:** 0.3172 (var=0.0051)
**Candidate trajectory (this loop):**
  - iter 1: 0.4325 (var=0.0302) (Δ vs real +0.1153)
  - iter 2: 0.6639 (var=0.0562) (Δ vs real +0.3467)
  - iter 3 (current): 0.1017 (var=0.0173) (Δ vs real -0.2156)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8558 (var=0.0110)
- pi_2: 0.3389 (var=0.0293)
- pi_3: 0.1508 (var=0.0087)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    diff = a - b
    # Find the index of the first discriminating cue (highest validity first)
    idx = (diff != 0).argmax(axis=1)
    # TTB chooses option A (0) if A > B on this cue, else option B (1)
    ttb_winner = np.where(diff[np.arange(len(diff)), idx] > 0, 0, 1)
    return float((data['response'].values == ttb_winner).mean())
```

**Observed (real) value:** 0.3383 (var=0.0035)
**Candidate trajectory (this loop):**
  - iter 1: 0.4910 (var=0.0427) (Δ vs real +0.1527)
  - iter 2: 0.6325 (var=0.0257) (Δ vs real +0.2942)
  - iter 3 (current): 0.1833 (var=0.0144) (Δ vs real -0.1550)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3296 (var=0.0225)
- pi_1: 0.8242 (var=0.0146)
- pi_3: 0.1598 (var=0.0064)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    validities = np.array([1.0, 0.9, 0.6, 0.5, 0.5])
    wadd_a = a_ratings @ validities
    wadd_b = b_ratings @ validities
    
    conflict_mask = ((sum_a > sum_b) & (wadd_a < wadd_b)) | ((sum_a < sum_b) & (wadd_a > wadd_b))
    
    if not np.any(conflict_mask):
        return 0.5
        
    sum_a_conf = sum_a[conflict_mask]
    sum_b_conf = sum_b[conflict_mask]
    responses = data['response'].values[conflict_mask]
    
    tallying_predictions = (sum_a_conf < sum_b_conf).astype(int)
    
    return float(np.mean(responses == tallying_predictions))
```

**Observed (real) value:** 0.6700 (var=0.0082)
**Candidate trajectory (this loop):**
  - iter 1: 0.4875 (var=0.0533) (Δ vs real -0.1825)
  - iter 2: 0.1833 (var=0.0310) (Δ vs real -0.4867)
  - iter 3 (current): 0.5617 (var=0.0514) (Δ vs real -0.1083)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8433 (var=0.0144)
- pi_2: 0.3683 (var=0.0688)
- pi_1: 0.1608 (var=0.0140)

### Experiment 4
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to strings for easy comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Identify trials where one option has fewer but more valid features (11000)
    # and the other has more but less valid features (00111)
    mask1 = (a_str == '11000') & (b_str == '00111')
    mask2 = (a_str == '00111') & (b_str == '11000')
    
    total_critical_trials = mask1.sum() + mask2.sum()
    if total_critical_trials == 0:
        return 0.5
        
    # Count how often the subject chose the WADD-favored option (11000)
    wadd_choices = (data.loc[mask1, 'response'] == 0).sum() + (data.loc[mask2, 'response'] == 1).sum()
    
    return float(wadd_choices / total_critical_trials)
```

**Observed (real) value:** 0.3233 (var=0.0099)
**Candidate trajectory (this loop):**
  - iter 1: 0.4783 (var=0.0562) (Δ vs real +0.1550)
  - iter 2: 0.7808 (var=0.0625) (Δ vs real +0.4575)
  - iter 3 (current): 0.3567 (var=0.0496) (Δ vs real +0.0333)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5450 (var=0.0989)
- pi_3: 0.1500 (var=0.0126)
- pi_1: 0.8558 (var=0.0136)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate model implements a probability-level mixture of Tallying and Take-The-Best. While this was accepted and improves over the base, the simulated point estimates (0.43, 0.49, 0.48, 0.47) are hovering near 0.5 and fail to hit the observed empirical values (0.31, 0.33, 0.67, 0.32). The model is essentially predicting random choices on average, likely because a probability-level mixture with a broad `w_ttb` range [0.0, 1.0] and independent `beta` scaling for two very different heuristics creates a flat or poorly-behaved loss landscape during subject-level fitting.
Rationale: To achieve tighter fits within this mechanism family, switch from a probability-level mixture to a unified 'Rank-Weighted Additive' rule. Instead of explicitly calculating TTB and Tallying probabilities and mixing them, assign a weight to each cue based on its validity rank. For example, `weight_i = 1 / (rank_i ** gamma)`, where `gamma` is a free parameter (e.g., [0.0, 5.0]). When `gamma=0`, this perfectly matches Tallying (equal weights); as `gamma` increases, it naturally approximates TTB. Compute a single score for each option using these rank-based weights, and pass the difference into a standard softmax. This provides a much smoother interpolation between the two heuristics and eliminates the need to balance two separate probability distributions.

**Outcome of your advice:** iter 1 candidate loss=0.2201 -> iter 2 candidate loss=0.5739 -> the gate marked it REJECTED.

### Iteration 2 (most recent)
Verdict: regenerate
Interpretation: The Rank-Weighted Additive model (iter 2) was rejected by the gate because it pushed predictions in the wrong direction. The empirical data across all four experiments shows a strong preference (~67-68%) for the Tallying strategy (choosing the option with more positive cues) over validity-based strategies like WADD or TTB (which are chosen ~32-33% of the time). The iter 2 model overshot in favor of the validity-based options. The accepted base from iter 1 (a probability mixture) hovered near 0.5, likely due to a high random-guessing parameter washing out the signal.
Rationale: Since the Rank-Weighted Additive approach was rejected, we will build upon the accepted iter 1 base (the probability-level mixture). To fix the issue of predictions hovering near 0.5 in that base, the proposer should implement a probability-level mixture of Tallying and WADD (or TTB), but strictly limit the random-guessing parameter `epsilon` to [0.0, 0.1]. Use a single shared `beta` parameter [0.1, 10.0] to compute the softmax probabilities for both the Tallying score and the WADD score, and then mix these two probability distributions using a `w_tally` parameter [0.0, 1.0]. This will allow the model to confidently capture the ~67% Tallying behavior observed in the data without decaying into random noise.

**Outcome of your advice:** iter 2 candidate loss=0.5739 -> CURRENT candidate loss=0.1813 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The probability-level mixture of Tallying and WADD with a shared beta and restricted epsilon was highly successful, bringing the aggregate loss down to 0.1813 and being accepted by the gate. The model successfully captures the qualitative trends across all experiments: it correctly predicts that subjects favor the Tallying option in Experiment 3 (>0.5) while favoring the non-WADD/TTB options in Experiment 1, 2, and 4 (<0.5). Although there is still some numerical under-prediction in Experiments 1 and 2, the current mechanism is structurally sound and effectively balances the two strategies without decaying into random noise.",
  "verdict": "continue",
  "rationale": "The current theory and parameterizations are well-calibrated. Using a shared beta and restricting the lapse rate allowed the model to confidently express the mixture of Tallying and WADD, successfully matching the empirical direction of the metrics across all four experiments. The model is good enough to serve as the baseline for future rounds."
}
```

## Usage

```json
{
  "prompt_token_count": 6338,
  "candidates_token_count": 225,
  "total_token_count": 7608
}
```
