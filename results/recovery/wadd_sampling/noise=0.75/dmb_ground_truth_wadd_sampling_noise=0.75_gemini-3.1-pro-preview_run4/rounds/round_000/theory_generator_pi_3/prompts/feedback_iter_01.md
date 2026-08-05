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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a Weighted Additive (WADD) theory. Unlike Take The Best (which ignores all lower-validity cues once one discriminates) and Tallying (which ignores cue validities entirely), WADD assumes decision makers compute a score for each option by summing its feature values weighted by their respective cue validities. The choice is then made by comparing these weighted sums, with response noise modeled via a softmax function. This compensatory theory can naturally capture the intermediate choice patterns observed on conflict trials, as multiple lower-validity cues can collectively outweigh a single higher-validity cue depending on the exact validities.


## CANDIDATE THEORY
People evaluate options by computing a comprehensive weighted sum of all available features. Each feature's contribution is proportional to its objective validity transformed into log-odds (log-likelihood ratios). This allows the compensatory mechanism to properly weigh probabilistic evidence, preventing multiple weak cues from inappropriately dominating a single strong cue. Choices are generated probabilistically by applying a softmax function over the options' weighted scores, subject to an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    # Transform validities into log-odds to correctly accumulate probabilistic evidence
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute weighted sum of features for each option
    scores = np.dot(stim, w)
    
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
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's feedback, we transform the raw validities into log-odds (log-likelihood ratios) before computing the weighted sum. This is the mathematically principled way to accumulate probabilistic evidence and prevents the WADD model from behaving like a simple tallying model where multiple weak cues easily outweigh a strong cue. This should better capture the intermediate choice proportions observed in the human data.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.4947 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.2128 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.2128.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    
    # TTB: winner based on the first discriminating cue
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    
    # Tallying: winner based on total number of winning cues
    a_wins = np.sum(diff > 0, axis=1)
    b_wins = np.sum(diff < 0, axis=1)
    
    # Identify conflict trials where Tallying has a strict winner that disagrees with TTB
    tally_winner = np.where(a_wins > b_wins, 0, 1)
    conflict_mask = (a_wins != b_wins) & (ttb_winner != tally_winner)
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    ttb_consistent = (responses[conflict_mask] == ttb_winner[conflict_mask])
    
    return float(np.mean(ttb_consistent))
```

**Observed (real) value:** 0.4619 (var=0.0040)
**Candidate trajectory (this loop):**
  - iter 1: 0.1625 (var=0.0099) (Δ vs real -0.2994)
  - iter 2 (current): 0.5566 (var=0.0044) (Δ vs real +0.0947)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8731 (var=0.0080)
- pi_2: 0.1437 (var=0.0116)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    tally_pred = (b_wins[mask] > a_wins[mask]).astype(int)
    return float(np.mean(tally_pred == data['response'].values[mask]))
```

**Observed (real) value:** 0.5637 (var=0.0028)
**Candidate trajectory (this loop):**
  - iter 1: 0.8653 (var=0.0070) (Δ vs real +0.3016)
  - iter 2 (current): 0.7172 (var=0.0054) (Δ vs real +0.1534)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8631 (var=0.0067)
- pi_1: 0.3444 (var=0.0038)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The Weighted Additive (WADD) candidate integrates cues by taking a dot product of the feature values and the raw validities. However, the simulation results show that the candidate is behaving almost identically to a Tallying model (pi_2), severely underpredicting the TTB-consistent choices in Experiment 1 (0.16 vs 0.46) and overpredicting the Tallying-consistent choices in Experiment 2 (0.86 vs 0.56). This happens because summing raw probabilities (e.g., 0.8 + 0.7 + 0.6 = 2.1) vastly outweighs a single highly valid cue (e.g., 0.9), stripping the model of its ability to correctly weigh strong evidence against multiple weak pieces of evidence.
Rationale: To fix the extreme Tallying-like behavior while strictly remaining within the prescribed WADD family, transform the raw validities into log-odds (or log-likelihood ratios) before computing the weighted sum. Specifically, compute weights as `w = np.log(val / (1.0 - val))` (with appropriate clipping to avoid log(0) or division by zero). This is the mathematically principled way to accumulate probabilistic evidence, and it will allow the WADD model to achieve the intermediate choice proportions observed in the human data.

**Outcome of your advice:** iter 1 candidate loss=0.4947 -> CURRENT candidate loss=0.2128 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The introduction of the log-odds transformation for the validities has significantly improved the model's fit. The candidate's aggregate loss dropped from 0.4947 to 0.2128. In Experiment 1, the proportion of TTB-consistent choices on conflict trials shifted from an extreme Tallying-like 0.16 up to 0.55, which is much closer to the observed human value of 0.46. In Experiment 2, the Tallying-consistent choices dropped from 0.86 to 0.71, moving towards the observed 0.56. By properly accumulating probabilistic evidence using log-likelihood ratios, the WADD model effectively captures the intermediate, compensatory nature of human decision-making on these tasks, balancing between the extremes of pure Tallying and pure Take-The-Best.",
  "verdict": "continue",
  "rationale": "The current WADD model with log-odds transformed weights successfully captures the intermediate choice patterns across both experiments. The mechanism is mathematically principled for accumulating independent probabilistic evidence and its quantitative fit is strong enough to serve as a reliable baseline. No further architectural changes are necessary."
}
```

## Usage

```json
{
  "prompt_token_count": 4148,
  "candidates_token_count": 260,
  "total_token_count": 4965
}
```
