# metric_exp00_attempt_03

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
**Validities (n_features=5):** [0.95, 0.9, 0.55, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Equal Weights (Tallying) from Probabilistic Cue Selection (Random Cue), we exploit Tallying's complete insensitivity to cue validities. Tallying simply counts the number of positive features for each option and strictly prefers the one with the higher count. Random Cue, however, samples cues proportionally to their subjective validities and chooses the option favored by the sampled cue. By presenting trials where one option has fewer positive cues but those cues possess significantly higher validities, Tallying will consistently predict a preference for the option with more positive cues, whereas Random Cue will predict a preference for the option supported by fewer, more valid cues. We also include trials where the total number of positive cues is tied; Tallying predicts random guessing (0.5 probability), while Random Cue predicts strong deterministic-like preferences based on the validities of the active cues.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Equal Weights (Tallying)

**Parameters:**
- beta: [0.0, 3.0]
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    
    # Count the number of positive features for each option
    tally_a = np.sum(stim[0])
    tally_b = np.sum(stim[1])
    
    if tally_a > tally_b:
        scores = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        scores = np.array([0.0, 1.0])
    else:
        # Guess randomly if there is a tie
        return np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary tally score
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Independent lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Probabilistic Cue Selection (Random Cue) posits that decision-makers do not deterministically use the most valid cue (like Take-The-Best) nor do they integrate all cues simultaneously (like WADD). Instead, on each trial, they sample a single cue with a probability proportional to its subjective validity. They then choose the option favored by that sampled cue, guessing uniformly if the sampled cue ties. This single-cue sampling process naturally generates probabilistic choices across trials, producing choice shares near 0.50 for conflict trials where different cues favor different options, without relying on extreme softmax noise.

**Parameters:**
- epsilon: [0.0, 0.5]
- weights: [(0.0, 1.0)] * n_features
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Subjective validities used as sampling weights
    subj_weights = val * w
    sum_w = np.sum(subj_weights)
    
    if sum_w <= 1e-9:
        p_core = np.array([0.5, 0.5])
    else:
        p_cue = subj_weights / sum_w
        a, b = stim[0], stim[1]
        
        p_a = 0.0
        for j in range(n_features):
            if a[j] > b[j]:
                p_a += p_cue[j]
            elif a[j] == b[j]:
                p_a += p_cue[j] * 0.5
                
        p_core = np.array([p_a, 1.0 - p_a])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: This metric isolates the difference in preference for Option A between Trial 1 and Trial 2. In Trial 1, Option A has more positive cues but lower validity, so Tallying strongly prefers A while Random Cue slightly prefers B. In Trial 2, the options are tied in tally (both have 2 positive cues), so Tallying guesses randomly (50% for A), whereas Random Cue strongly prefers A due to its higher validity cues. By subtracting the proportion of choosing A in Trial 2 from Trial 1, Tallying yields a positive difference (~0.5), while Random Cue yields a negative difference (~ -0.2), creating a large and reliable contrast between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert ratings to string for easy trial identification
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Trial 1: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    mask1 = a_str == '00111'
    # Trial 2: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 0]
    mask2 = a_str == '11000'
    
    # Proportion of choosing A (response == 0)
    p1 = (data.loc[mask1, 'response'] == 0).mean()
    p2 = (data.loc[mask2, 'response'] == 0).mean()
    
    if pd.isna(p1): p1 = 0.5
    if pd.isna(p2): p2 = 0.5
    
    return float(p1 - p2)
outcome: self_sim=0.0900 (var=0.0591) adversary_sim=-0.0887 (var=0.0788) welch_t=+2.407 p=0.02005 (N=25, alpha=0.01) -> reject

[1] rationale: To create a highly discriminative and robust metric, we compute a specific linear combination of the empirical probabilities of choosing Option A across the 6 unique trial types. By assigning a +1 weight to trials where Tallying deterministically prefers A (T1, T4, T6) and a -1 weight to trials where Tallying deterministically prefers B or guesses while Random Cue strongly prefers A (T2, T3, T5), we maximize the separation between the two models. In the absence of noise, Tallying yields a score of ~2.5, whereas Random Cue yields a score near ~0.08, creating an overwhelming gap that easily survives parameter noise and between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert option A ratings to string for exact trial identification
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Masks for the 6 unique trials based on Option A's features
    m1 = a_str == '00111'
    m2 = a_str == '11000'
    m3 = a_str == '10000'
    m4 = a_str == '01110'
    m5 = a_str == '10100'
    m6 = a_str == '00110'
    
    def get_p_a(mask):
        subset = data.loc[mask, 'response']
        if len(subset) == 0:
            return 0.5
        return (subset == 0).mean()
        
    # Calculate the proportion of times the subject chose Option A in each trial type
    p1 = get_p_a(m1)
    p2 = get_p_a(m2)
    p3 = get_p_a(m3)
    p4 = get_p_a(m4)
    p5 = get_p_a(m5)
    p6 = get_p_a(m6)
    
    # We construct a linear combination of these probabilities that maximizes the theoretical
    # difference between Tallying and Random Cue.
    # Tallying theoretical P(A): T1=1, T2=0.5, T3=0, T4=1, T5=0, T6=1. Sum with signs = 2.5
    # Random Cue theoretical P(A): T1=0.45, T2=0.63, T3=0.39, T4=0.57, T5=0.44, T6=0.52. Sum with signs = 0.08
    return float(p1 - p2 - p3 + p4 - p5 + p6)

outcome: self_sim=0.5800 (var=0.3744) adversary_sim=0.1562 (var=0.2632) welch_t=+2.654 p=0.01086 (N=25, alpha=0.01) -> reject

[2] rationale: To achieve strict discrimination, we must minimize the massive between-subject variance caused by Tallying's noise parameters (beta and epsilon). We do this by constructing a contrast where the theoretical expected value for Tallying is identical across all subjects, reducing its variance to mere binomial sampling noise.

In Trial 4 and Trial 1, Option A has exactly 3 positive features and Option B has 2. For any given subject, Tallying predicts the exact same probability of choosing A in both trials. Thus, the difference `P(A|T4) - P(A|T1)` is theoretically exactly 0 for Tallying, with zero parameter variance. In Trial 2, both options have 2 positive features, so Tallying predicts a tie; `P(A|T2)` is exactly 0.5 for all subjects.

Therefore, the metric `P(A|T4) - P(A|T1) + P(A|T2)` has a mathematically guaranteed expected value of 0.5 for Tallying with exceptionally low variance.

For Random Cue, choices depend on cue validities. In T4, the cues favoring A have higher validities than in T1, making `P(A|T4) > P(A|T1)` (a difference of ~0.12). In T2, Option A's cues have vastly higher validities (0.95, 0.9) than B's (0.55, 0.5), making `P(A|T2)` strongly favor A (~0.64). Random Cue's expected value for this metric is ~0.76. This creates a highly reliable mean difference of ~0.26, which cleanly overcomes the tightly controlled variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert option A ratings to string for exact trial identification
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Trial 1: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    m1 = a_str == '00111'
    # Trial 2: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 0]
    m2 = a_str == '11000'
    # Trial 4: A=[0, 1, 1, 1, 0], B=[1, 0, 0, 0, 1]
    m4 = a_str == '01110'
    
    def get_p_a(mask):
        subset = data.loc[mask, 'response']
        if len(subset) == 0:
            return 0.5
        return float((subset == 0).mean())
        
    p1 = get_p_a(m1)
    p2 = get_p_a(m2)
    p4 = get_p_a(m4)
    
    # We compute a contrast that has zero parameter variance for Tallying.
    return float(p4 - p1 + p2)
outcome: self_sim=0.5525 (var=0.0457) adversary_sim=0.6850 (var=0.0756) welch_t=-1.902 p=0.06352 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # Convert option A ratings to string for exact trial identification\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    \n    # Masks for the 6 unique trials based on Option A's features\n    m1 = a_str == '00111'\n    m2 = a_str == '11000'\n    m3 = a_str == '10000'\n    m4 = a_str == '01110'\n    m5 = a_str == '10100'\n    m6 = a_str == '00110'\n    \n    def get_p_a(mask):\n        subset = data.loc[mask, 'response']\n        if len(subset) == 0:\n            return 0.5\n        return float((subset == 0).mean())\n        \n    # Calculate the proportion of times the subject chose Option A in each trial type\n    p1 = get_p_a(m1)\n    p2 = get_p_a(m2)\n    p3 = get_p_a(m3)\n    p4 = get_p_a(m4)\n    p5 = get_p_a(m5)\n    p6 = get_p_a(m6)\n    \n    # Compute the zero-variance contrast for Tallying\n    return float(-p1 + p2 - 0.25 * p3 + 0.25 * p4 - 0.25 * p5 + 0.25 * p6)",
  "rationale": "By constructing a specific linear combination of P(A) across the 6 trial types (-p1 + p2 - 0.25*p3 + 0.25*p4 - 0.25*p5 + 0.25*p6), we create a metric whose theoretical expected value for Tallying is exactly 0 for any combination of noise parameters (beta and epsilon). This mathematically eliminates parameter-driven between-subject variance for Tallying, reducing its variance to mere binomial sampling noise. For Random Cue, however, this same linear combination yields an expected value of approximately 0.25 due to its sensitivity to cue validities. The resulting large mean difference combined with tightly controlled variance easily discriminates the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 5109,
  "candidates_token_count": 600,
  "total_token_count": 11629
}
```
