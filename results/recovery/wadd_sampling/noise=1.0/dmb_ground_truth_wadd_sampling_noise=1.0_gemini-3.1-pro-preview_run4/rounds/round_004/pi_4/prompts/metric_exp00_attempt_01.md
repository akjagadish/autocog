# metric_exp00_attempt_01

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
**Validities (n_features=5):** [1.0, 0.9, 0.8, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Rationale:** This design systematically dissociates the Advocated Theory (Tallying-based with a massive lapse rate) from the Competing Theory (Weighted Additive with extremely low inverse temperature). By including trials where the WADD scores are perfectly tied but the Tallying scores differ (e.g., one cue of validity 1.0 vs two cues of validity 0.5), we can isolate the Tallying signal. Conversely, by including trials where Tallying scores are tied but WADD scores differ, we isolate the WADD signal. The high noise in both models will compress choice probabilities near 0.5, but the direction of the small deviations will cleanly identify which underlying compensatory process (Tallying vs. WADD) is driving the behavior.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Due to the abstract nature of the task and lack of trial-by-trial feedback, subjects do not consistently apply structured multi-attribute heuristics. Instead, their choices are dominated by high response noise or a massive lapse rate, effectively resulting in random guessing on most trials. While they may occasionally attempt to tally features or look at the most valid cue, the lack of grounding leads to a near-uniform choice probability across all stimulus pairs.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    # Calculate a weak tallying signal
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Massive lapse rate dominates the choice
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Subjects attempt to integrate all available information by computing the subjective value of each option based on the provided feature validities (Weighted Additive strategy). However, due to the abstract nature of the task and the absence of trial-by-trial feedback, they suffer from extremely high decision noise. This translates to a very low inverse temperature in their choice rule, producing behavior that appears near-random but is actually generated by a structured, bounded-rational compensatory process.

**Parameters:**
- beta: [0.0, 0.05]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate weighted additive values for each option
    val_a = np.sum(validities * a)
    val_b = np.sum(validities * b)
    scores = np.array([val_a, val_b])
    
    # Apply softmax with extremely high decision noise (low beta)
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
[0] rationale: This metric isolates the specific trials designed to dissociate the two theories. In trials 1 and 2, the WADD scores are perfectly tied but Tallying prefers the option with two validities of 0.5 over one validity of 1.0. In trials 3 and 4, the Tallying scores are perfectly tied but WADD prefers the option with validity 1.0 over validity 0.9. By taking the difference in the proportion of choices aligning with the Tallying preference in trials 1/2 and the WADD preference in trials 3/4, we obtain a metric that is positive under the Advocated Theory (Tallying-based) and negative under the Competing Theory (WADD-based).
metric_source:
def metric(data: pd.DataFrame) -> float:
    tally_choices = 0
    tally_trials = 0
    wadd_choices = 0
    wadd_trials = 0
    
    for _, row in data.iterrows():
        a_str = ''.join(map(str, row['option_a_ratings']))
        b_str = ''.join(map(str, row['option_b_ratings']))
        
        # Trials 1 & 2: WADD is indifferent (values 1.0 vs 1.0), Tally prefers [0,0,0,1,1] (2 wins vs 1 win)
        if a_str == '10000' and b_str == '00011':
            tally_trials += 1
            if row['response'] == 1:
                tally_choices += 1
        elif a_str == '00011' and b_str == '10000':
            tally_trials += 1
            if row['response'] == 0:
                tally_choices += 1
                
        # Trials 3 & 4: Tally is indifferent (1 win vs 1 win), WADD prefers [1,0,0,0,0] (value 1.0 vs 0.9)
        elif a_str == '10000' and b_str == '01000':
            wadd_trials += 1
            if row['response'] == 0:
                wadd_choices += 1
        elif a_str == '01000' and b_str == '10000':
            wadd_trials += 1
            if row['response'] == 1:
                wadd_choices += 1
                
    tally_rate = tally_choices / tally_trials if tally_trials > 0 else 0.5
    wadd_rate = wadd_choices / wadd_trials if wadd_trials > 0 else 0.5
    
    return float(tally_rate - wadd_rate)
outcome: self_sim=0.0217 (var=0.0128) adversary_sim=-0.0050 (var=0.0144) welch_t=+0.808 p=0.423 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    tally_score = 0.0\n    for _, row in data.iterrows():\n        a = np.array(row['option_a_ratings'])\n        b = np.array(row['option_b_ratings'])\n        a_wins = np.sum(a > b)\n        b_wins = np.sum(b > a)\n        if row['response'] == 0:\n            tally_score += (a_wins - b_wins)\n        else:\n            tally_score += (b_wins - a_wins)\n    return float(tally_score / len(data))",
  "rationale": "To maximize the signal of the Tallying mechanism despite the massive lapse rate, this metric computes the average Tallying score difference in favor of the chosen option across all trials. By aggregating the continuous Tallying advantage of the chosen options rather than just binary alignment on a subset of trials, we pool the weak signal across the entire experiment (96 trials) to reduce binomial noise. The Advocated Theory (Tallying) will show a reliably higher mean on this metric than the Competing Theory (WADD), which only aligns with Tallying on a subset of trials and with an even weaker decision signal."
}
```

## Usage

```json
{
  "prompt_token_count": 3722,
  "candidates_token_count": 307,
  "total_token_count": 9595
}
```
