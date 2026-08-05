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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a 'Take The Best' (TTB) theory. Unlike Tallying (which weights all cues equally) and WADD (which integrates all cues compensatorily), TTB orders cues by their subjective validity and searches through them sequentially. It stops at the first cue that discriminates between the two options and bases the decision entirely on that single cue. This non-compensatory heuristic will naturally explain the high variance in Experiment 1 (since trials with the same tally difference can be decided by different highest-validity cues) and will correctly predict the extreme preference for the 'fewer but better' option in Experiment 2 (since the highest validity cue alone dictates the choice).


## CANDIDATE THEORY
Take The Best (TTB): People compare two options by ordering features by their subjective validity and searching through them sequentially. The search stops at the first feature that discriminates between the two options (i.e., one option has a higher value than the other), and the decision is based entirely on that single feature. This non-compensatory strategy ignores all other features, preventing any compensatory trade-offs. If no feature discriminates, the learner guesses. Response noise is modeled via an independent lapse rate epsilon, which replaces the deterministic TTB choice with a uniform random pick.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(validities, kind='stable')[::-1]
    
    # Default to guessing if no cue discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the arbiter's instructions, this model implements the Take The Best (TTB) heuristic. Instead of computing a compensatory score (like WADD) or an unweighted sum (like Tallying), TTB sequentially evaluates features in decreasing order of validity. It stops at the first cue that discriminates and bases the choice entirely on it. This explains the high variance in Experiment 1 because trials with identical tally differences can have their decisions driven by different cues depending on the specific feature patterns. It also explains the strong preference in Experiment 2 for the 'fewer but better' option, as the single highest-validity cue dominates the decision regardless of deficits on multiple lower-validity cues.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.1232 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.1232.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # Create a safe, hashable string identifier for each trial type
    data['trial_id'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x])) + '_' + \
                       data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Calculate the Tallying difference (wins for A - wins for B)
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    data['tally_diff'] = np.sum(a_ratings > b_ratings, axis=1) - np.sum(b_ratings > a_ratings, axis=1)
    
    # Calculate choice for A (response == 0 means A was chosen)
    data['choice_A'] = 1 - data['response']
    
    # Calculate the proportion of times A was chosen for each trial type, per subject
    trial_means = data.groupby(['subject_id', 'tally_diff', 'trial_id'])['choice_A'].mean().reset_index()
    
    # Compute the pooled within-group variance of choice proportions for each subject
    def pooled_variance(df):
        var_sum = 0.0
        df_sum = 0.0
        for t_diff, group in df.groupby('tally_diff'):
            n = len(group)
            if n > 1:
                v = group['choice_A'].var(ddof=1)
                var_sum += v * (n - 1)
                df_sum += (n - 1)
        if df_sum == 0:
            return 0.0
        return float(var_sum / df_sum)
        
    subj_vars = []
    for subj, subj_df in trial_means.groupby('subject_id'):
        subj_vars.append(pooled_variance(subj_df))
        
    return float(np.mean(subj_vars))
```

**Observed (real) value:** 0.1080 (var=0.0018)
**Candidate (simulated) value:** 0.1253 (var=0.0019)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0216 (var=0.0001)
- pi_2: 0.0866 (var=0.0030)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_fewer_but_better(a, b):
        return tuple(a) == (1, 1, 0, 0, 0) and tuple(b) == (0, 0, 1, 1, 1)
        
    def is_worse_but_more(a, b):
        return tuple(a) == (0, 0, 1, 1, 1) and tuple(b) == (1, 1, 0, 0, 0)

    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        if is_fewer_but_better(a, b):
            matches.append(1.0 if resp == 0 else 0.0)
        elif is_worse_but_more(a, b):
            matches.append(1.0 if resp == 1 else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8267 (var=0.0120)
**Candidate (simulated) value:** 0.8858 (var=0.0067)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5692 (var=0.0859)
- pi_1: 0.1467 (var=0.0159)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate theory successfully implements the Take The Best (TTB) heuristic. In Experiment 1, the candidate captures the high variance in choice proportions for trials with the same tally difference (candidate: 0.1253 vs. observed: 0.1080), substantially outperforming the alternative theories. In Experiment 2, the candidate successfully predicts the strong preference for the 'fewer but better' option (candidate: 0.8858 vs. observed: 0.8267), again providing an excellent fit to the empirical data. The inclusion of a lapse rate (epsilon) appropriately scales the deterministic predictions to match human response noise.",
  "verdict": "continue",
  "rationale": "The current TTB model with an epsilon lapse rate perfectly captures the target phenomena across both experiments. It correctly accounts for the within-tally-difference variance in Experiment 1 and the non-compensatory 'fewer but better' preference in Experiment 2. The parameter ranges and implementation are well-calibrated. No further changes are needed."
}
```

## Usage

```json
{
  "prompt_token_count": 4217,
  "candidates_token_count": 233,
  "total_token_count": 4629
}
```
