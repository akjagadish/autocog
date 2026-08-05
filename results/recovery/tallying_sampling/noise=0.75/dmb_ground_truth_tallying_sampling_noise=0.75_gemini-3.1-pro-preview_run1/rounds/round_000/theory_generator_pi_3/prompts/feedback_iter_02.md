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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a 'Tallying' (Equal Weights) theory or a 'Rank-Weighted Additive' theory. Tallying posits that subjects simply count the number of positive features for each option and choose the one with the higher count, ignoring the precise cue validities. A Rank-Weighted model posits that subjects assign weights based on the ordinal rank of the cues rather than their cardinal validities. Both serve as strong, classic compensatory alternatives to the WADD model that might better capture the exact mixture of choices observed.


## CANDIDATE THEORY
Rank-Weighted Additive Theory: People evaluate options by computing a weighted sum of features, but instead of using cardinal validities, they assign weights based on the ordinal rank of the cues' validities. Specifically, weights decay linearly with rank (e.g., n_features - rank), providing a compensatory mechanism where the top cue has higher importance but not so much that it completely dominates lower-ranked cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    # Rank-Weighted Additive: weight cues by linear rank decay (n_features - rank)
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    cue_order = np.argsort(-val, kind="stable")
    weights = np.zeros(n_features)
    for rank, idx in enumerate(cue_order):
        weights[idx] = float(n_features - rank)
        
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
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
Following the critic's advice, we adjust the rank-weighting scheme from 1/rank to a linear decay (n_features - rank). The 1/rank weighting gave too much dominance to the top cue, behaving too similarly to Take-The-Best and overestimating the TTB match rate in Exp 1 while underestimating the Tallying match rate in Exp 2. A linear decay provides a gentler slope, reducing the top cue's dominance and allowing lower-ranked cues to compensate more effectively, striking a better balance between TTB and Tallying.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3689 -> ACCEPTED
- iter 2: loss=0.1993 -> ACCEPTED
- iter 3 (current candidate you are grading): loss=0.0804 -> ACCEPTED
Running-best (last accepted) base: iter 3 at loss=0.0804.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice is not None and row['response'] == ttb_choice:
            ttb_matches += 1
            
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4071 (var=0.0029)
**Candidate trajectory (this loop):**
  - iter 1: 0.1327 (var=0.0071) (Δ vs real -0.2744)
  - iter 2: 0.5669 (var=0.0024) (Δ vs real +0.1598)
  - iter 3 (current): 0.4179 (var=0.0020) (Δ vs real +0.0108)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8638 (var=0.0118)
- pi_2: 0.3210 (var=0.0381)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_resp = data.loc[mask, 'response']
    a_higher = sum_a[mask] > sum_b[mask]
    
    chose_higher = (a_higher & (filtered_resp == 0)) | (~a_higher & (filtered_resp == 1))
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.6008 (var=0.0065)
**Candidate trajectory (this loop):**
  - iter 1: 0.8596 (var=0.0110) (Δ vs real +0.2588)
  - iter 2: 0.5467 (var=0.0046) (Δ vs real -0.0542)
  - iter 3 (current): 0.6754 (var=0.0046) (Δ vs real +0.0746)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6663 (var=0.0281)
- pi_1: 0.1458 (var=0.0146)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The Tallying candidate theory correctly captures the qualitative direction in Experiment 2 (preferring the option with the higher feature sum), but it severely overshoots the effect size (predicting 0.85 vs observed 0.60). In Experiment 1, it significantly underestimates the TTB match rate (predicting 0.13 vs observed 0.40). Because pure Tallying strongly prefers the option with more features, it makes overly extreme predictions that do not reflect the more nuanced, noisy, or heterogeneous behavior of the human subjects, who hover closer to 0.5 in both experiments.
Rationale: The current Tallying model is too deterministic and extreme in its predictions. Since the arbiter explicitly provided the option of a 'Rank-Weighted Additive' theory, you should switch to implementing that mechanism (e.g., weighting cues by 1/rank or n-rank+1). A Rank-Weighted approach provides a middle ground between Tallying and TTB, which should naturally elevate the TTB match rate in Exp 1 while softening the higher-sum preference in Exp 2. Alternatively, if you keep Tallying, you must drastically shift the parameter ranges to allow for much more noise (e.g., lower beta range like [0.01, 2.0] and higher epsilon range up to 0.8) so the model can flatten its predictions closer to the observed 0.40 and 0.60.

**Outcome of your advice:** iter 1 candidate loss=0.3689 -> iter 2 candidate loss=0.1993 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent)
Verdict: regenerate
Interpretation: The switch to the Rank-Weighted Additive model successfully balanced the extremes of the previous Tallying model, leading to a significantly improved aggregate loss (ACCEPTED). However, the specific weighting scheme used (1/rank) places too much emphasis on the top cue. This causes the model to overshoot the Take-The-Best match rate in Experiment 1 (predicting 0.56 vs observed 0.40) and slightly underestimate the higher-sum tallying preference in Experiment 2 (predicting 0.54 vs observed 0.60).
Rationale: The current 1/rank weighting scheme drops off too steeply (e.g., 1, 0.5, 0.33, 0.25), which makes it behave a bit too much like Take-The-Best and not enough like Tallying. To perfectly split the difference, keep the Rank-Weighted Additive family but change the weighting function to a linear rank decay, such as `weights = n_features - rank` (e.g., 4, 3, 2, 1). This will slightly reduce the dominance of the top cue, bringing the Exp 1 TTB match rate down towards 0.40 and pushing the Exp 2 higher-sum match rate up towards 0.60.

**Outcome of your advice:** iter 2 candidate loss=0.1993 -> CURRENT candidate loss=0.0804 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The linear rank decay weighting scheme has proven to be an excellent mechanism for capturing the empirical data. By assigning weights as `n_features - rank`, the model strikes a near-perfect balance. In Experiment 1, the model's TTB match rate is 0.4179, which is incredibly close to the observed 0.4071. In Experiment 2, the model predicts a higher-sum choice rate of 0.6754, which is a reasonable approximation of the observed 0.6008 and a huge improvement from the pure Tallying model's 0.8596. The aggregate loss has plummeted to 0.0804, confirming that this intermediate compensatory weighting is highly descriptive of human behavior across these tasks.",
  "verdict": "continue",
  "rationale": "The Rank-Weighted Additive model with linear rank decay accurately captures the core behavioral patterns in both experiments, cleanly balancing the influence of the top cue with the compensatory power of lower-ranked cues. The fit is exceptionally tight, particularly for the Take-The-Best match rate. No further regeneration is needed; the current theory and parameter ranges are well-calibrated and ready to be maintained."
}
```

## Usage

```json
{
  "prompt_token_count": 4473,
  "candidates_token_count": 269,
  "total_token_count": 5015
}
```
