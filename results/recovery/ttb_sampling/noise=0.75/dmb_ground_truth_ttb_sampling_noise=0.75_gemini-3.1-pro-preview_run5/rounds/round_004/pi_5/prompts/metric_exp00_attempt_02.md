# metric_exp00_attempt_02

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To dissociate the Sequential Evidence Accumulation model from the Parallel Constraint Satisfaction (PCS) network, we use a 5-feature design with linearly decreasing validities. The PCS model uses a softmax transformation of validities, bounding its predictions between a purely unweighted compensatory strategy (Tallying, when tau is low) and a purely lexicographic strategy based only on the highest validity cue (TTB, when tau is high). The Sequential model, however, can exhibit intermediate stopping behaviors—such as stopping after accumulating evidence from exactly two or three cues—which cannot be captured by the global softmax weighting of PCS. By including trials where the first cue is tied, or where the first cue slightly favors one option but the subsequent cues strongly favor the other, we can pinpoint sequential stopping that bypasses the strict TTB vs. Tallying interpolation of the PCS model.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation. Decision-makers inspect features sequentially in decreasing order of validity. At each step, the validity-weighted difference between the options' features is added to an accumulated evidence tally. If the absolute value of this accumulated evidence exceeds a critical threshold, search stops immediately and a decision is made based on the current tally (mimicking Take-The-Best when the threshold is low). If all features are exhausted without crossing the threshold, a choice is made probabilistically based on the final accumulated tally (mimicking compensatory strategies like Weighted Additive when the threshold is high).

**Parameters:**
- threshold: [0.01, 1.5]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order (stable sort for ties)
    order = np.argsort(-validities, kind='stable')
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    
    # Sequential search and evidence accumulation
    for idx in order:
        diff = a[idx] - b[idx]
        evidence += validities[idx] * diff
        
        # Stop search if accumulated evidence exceeds the threshold
        if abs(evidence) >= threshold:
            break
            
    # Convert the final evidence into choice probabilities.
    # Evidence represents the accumulated advantage of Option A over Option B.
    scores = np.array([evidence, 0.0])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Parallel Constraint Satisfaction (PCS) Decision Network with Softmax Cue Weighting. Choices emerge from a recurrent neural network where options and features bidirectionally interact. Options accumulate evidence from features and mutually inhibit each other. Options also send feedback to features, amplifying features that support the leading option (coherence shift). To allow the model to flexibly transition between compensatory and non-compensatory (lexicographic) behavior without destabilizing the network, the initial cue validities are transformed via a softmax function controlled by a temperature parameter (tau). This ensures the highest-validity cue can decisively dominate the parallel accumulation when necessary, while bounded inhibition and feedback terms prevent runaway dynamics.

**Parameters:**
- tau: [0.0, 10.0]
- lam: [0.0, 2.0]
- gamma: [0.0, 2.0]
- kappa: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    tau = float(parameters["tau"])
    lam = float(parameters["lam"])
    gamma = float(parameters["gamma"])
    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax transformation of validities to allow lexicographic dominance
    z_v = tau * validities
    z_v = z_v - np.max(z_v)
    weights = np.exp(z_v) / np.sum(np.exp(z_v))
    
    # Initialize activations: Options at 0, Cues at their resting weights
    O = np.array([0.0, 0.0])
    C = weights.copy()
    
    w_A = a - b
    w_B = b - a
    
    alpha = 0.1
    n_iter = 50
    
    for _ in range(n_iter):
        # Net input to options: evidence from cues minus lateral inhibition
        net_O_A = np.sum(w_A * C) - lam * O[1]
        net_O_B = np.sum(w_B * C) - lam * O[0]
        
        # Net input to cues: feedback from options
        net_C = gamma * w_A * O[0] + gamma * w_B * O[1]
        
        # Update options (McClelland & Rumelhart rule)
        net_O = np.array([net_O_A, net_O_B])
        delta_O = np.where(net_O > 0,
                           alpha * net_O * (1.0 - O) - kappa * O,
                           alpha * net_O * O - kappa * O)
        O = np.clip(O + delta_O, 0.0, 1.0)
        
        # Update cues (decay towards resting weights)
        delta_C = np.where(net_C > 0,
                           alpha * net_C * (1.0 - C) - kappa * (C - weights),
                           alpha * net_C * C - kappa * (C - weights))
        C = np.clip(C + delta_C, 0.0, 1.0)
        
    scores = O
    
    # Softmax choice rule
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
[0] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 1 (where Option A has only the highest-validity cue and B has all others) and Trial 3 (where the highest-validity cue is tied, A has the second-highest, and B has the rest). Under Sequential Evidence Accumulation, a decision-maker with an intermediate stopping threshold (e.g., between 0.8 and 0.9) will stop immediately on Trial 1 and reliably choose A, but will fail to stop on Trial 3, evaluate all remaining cues, and ultimately choose B. This yields a large positive difference. Conversely, the PCS model uses a softmax transformation over validities. Because of the exponential spacing, if the network strongly favors A in Trial 1, it must also strongly favor A in Trial 3; if it is more compensatory, it favors B in both, or favors B in Trial 1 and A in Trial 3. PCS mathematically cannot produce a strong preference for A in Trial 1 and B in Trial 3, meaning its metric value will be near zero or negative, strongly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    t3_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    if t1_mask.sum() == 0 or t3_mask.sum() == 0:
        return 0.0
        
    p_a_t1 = 1.0 - data[t1_mask]['response'].mean()
    p_a_t3 = 1.0 - data[t3_mask]['response'].mean()
    
    return float(p_a_t1 - p_a_t3)
outcome: self_sim=0.0200 (var=0.0250) adversary_sim=-0.0337 (var=0.0198) welch_t=+1.268 p=0.2111 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 5 and Trial 2. Under the Sequential Evidence Accumulation model, the decision-maker inspects cues in order of validity. In both Trial 2 and Trial 5, the first two cues strongly favor Option A (validities 0.9 and 0.8). The accumulated evidence after two cues is 1.7. Because the maximum possible threshold in the parameter space is 1.5, the sequential model will ALWAYS stop by the second cue at the latest, making its choices for Trial 2 and Trial 5 completely identical in expectation. Thus, the metric will be near zero. In contrast, the Parallel Constraint Satisfaction (PCS) model integrates all cues simultaneously. Trial 5 provides an additional cue favoring Option A compared to Trial 2, which strictly increases the net evidence for A and decreases it for B. Consequently, PCS predicts a robustly positive difference in choice probabilities between these two trials, allowing for clear discrimination between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    t5_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1))
    
    if t2_mask.sum() == 0 or t5_mask.sum() == 0:
        return 0.0
        
    p_a_t2 = 1.0 - data[t2_mask]['response'].mean()
    p_a_t5 = 1.0 - data[t5_mask]['response'].mean()
    
    return float(p_a_t5 - p_a_t2)
outcome: self_sim=0.0063 (var=0.0118) adversary_sim=0.0926 (var=0.0326) welch_t=-2.047 p=0.04738 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # T1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]\n    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & \\\n              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))\n              \n    # T4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]\n    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0)) & \\\n              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 1))\n              \n    if t1_mask.sum() == 0 or t4_mask.sum() == 0:\n        return 0.0\n        \n    p_a_t1 = 1.0 - data[t1_mask]['response'].mean()\n    p_a_t4 = 1.0 - data[t4_mask]['response'].mean()\n    \n    return float(p_a_t4 - p_a_t1)",
  "rationale": "This metric calculates the difference in the probability of choosing Option A between Trial 4 and Trial 1. Under Sequential Evidence Accumulation, both trials share the same first cue favoring A (+0.9). For about 60% of subjects (stopping threshold <= 0.9), search stops immediately at cue 1, yielding perfectly identical choices for both trials. For the remaining subjects, search continues and the evidence tally ultimately favors Option B in both trials (ending at -0.3 and -1.7 respectively), resulting in a difference near zero with extremely low between-subject variance. In contrast, the PCS model integrates all cues simultaneously. Trial 4 has an additional cue favoring A compared to Trial 1, causing Trial 4 to switch to choosing A at a much more compensatory temperature (tau ~ 2.6) than Trial 1 (tau ~ 6.5). This creates a broad regime (nearly 40% of the parameter space) where PCS strongly favors A in Trial 4 but B in Trial 1, yielding a large positive mean difference that reliably discriminates the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 4810,
  "candidates_token_count": 601,
  "total_token_count": 33202
}
```
