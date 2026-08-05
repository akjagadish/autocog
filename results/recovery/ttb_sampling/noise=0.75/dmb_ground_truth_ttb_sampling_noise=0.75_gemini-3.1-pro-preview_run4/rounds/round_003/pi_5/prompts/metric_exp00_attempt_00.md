# metric_exp00_attempt_00

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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=20):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 9: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 11: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 18: A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 19: A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  trial 20: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate Subjective (Noisy-Validity) TTB from Pure TTB, we restrict the discriminating cues to exactly two per trial: one favoring Option A and one favoring Option B. Across trials, we fix the winning cue for the favored option but parametrically decrease the validity of the opposing cue. Because Pure TTB evaluates cues strictly in descending order of objective validity, it predicts a completely flat choice probability across these trials: the favored option is always chosen with the same probability as long as its winning cue has a higher validity than the opposing cue. In contrast, Subjective TTB samples the primary cue via a softmax distribution over the validities of all discriminating cues. Consequently, Subjective TTB predicts that the probability of choosing the favored option will monotonically increase as the validity of the opposing cue decreases, because the lower-validity opposing cue becomes less likely to win the softmax competition.

**Computed schedule:** 20 unique pairs × 4 reps = 80 trials per subject.



## ADVOCATED THEORY
**Description:** Subjective Take The Best (Noisy-Validity TTB). Decision makers strictly follow the non-compensatory Take The Best (TTB) heuristic on any given trial, meaning they base their choice solely on the first discriminating cue they consider and ignore all others. However, their internal ranking of cue validities is noisy. This is modeled by sampling the primary discriminating cue via a softmax distribution over the objective validities of the cues that differ between the options. The inverse temperature parameter beta controls the noise in the validity ranking: as beta increases, the model converges to deterministic TTB, whereas lower beta values allow sub-optimal cues to occasionally be evaluated first. Because decisions rely on a single cue, the model captures the flat sensitivity curves to supporting cue quantities (Exps 1, 2, 6). Meanwhile, the stochastic cue selection suppresses the overall agreement with the objective TTB predictions, matching the lower empirical performance in Exps 3, 4, and 5. Response noise is included via a simple lapse rate (epsilon).

**Parameters:**
- beta: [0.0, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify discriminating cues
    diff = stim[0] - stim[1]
    discrim_idx = np.where(diff != 0)[0]
    
    if len(discrim_idx) == 0:
        p_core = np.array([0.5, 0.5])
    else:
        # Softmax over validities of discriminating cues
        v_discrim = validities[discrim_idx]
        # Numerically stable softmax
        z = beta * v_discrim
        z -= np.max(z)
        w = np.exp(z)
        p_discrim = w / np.sum(w)
        
        p_a = 0.0
        p_b = 0.0
        for i, idx in enumerate(discrim_idx):
            if diff[idx] > 0:
                p_a += p_discrim[i]
            else:
                p_b += p_discrim[i]
                
        p_core = np.array([p_a, p_b])
        
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
**Description:** Take The Best (TTB) is a non-compensatory lexicographic heuristic. Decision makers rank features according to their validity. To choose between two options, they compare them on the most valid feature. If one option has a higher value on this feature, it is chosen immediately, and all remaining features are ignored. If the options are tied on this feature, the decision maker moves to the next most valid feature, and so on. If the options tie on all features, the decision maker guesses randomly. Response noise is modeled via a simple lapse rate (epsilon) where the subject makes a random choice instead of following the TTB rule. The lapse rate can be high, reflecting significant guessing in the empirical data.

**Parameters:**
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Order features by validity, descending
    order = np.argsort(validities)[::-1]
    
    # Find the first discriminating feature
    chosen = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen = 1
            break
            
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # Tie on all features
        p_core = np.array([0.5, 0.5])
        
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    diffs = []\n    favored = []\n    for _, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        \n        idx_a = -1\n        idx_b = -1\n        for i, (va, vb) in enumerate(zip(a, b)):\n            if va > vb:\n                idx_a = i\n            elif vb > va:\n                idx_b = i\n                \n        if idx_a != -1 and idx_b != -1:\n            if idx_a < idx_b:\n                diff = idx_b - idx_a\n                fav = 0\n            else:\n                diff = idx_a - idx_b\n                fav = 1\n            diffs.append(diff)\n            favored.append(1 if row['response'] == fav else 0)\n            \n    df = pd.DataFrame({'diff': diffs, 'fav': favored})\n    high = df[df['diff'] >= 3]['fav'].mean()\n    low = df[df['diff'] == 1]['fav'].mean()\n    return float(high - low)",
  "rationale": "By design, each trial pits exactly two discriminating cues against each other. Pure Take The Best (TTB) evaluates cues strictly in order of validity, so the probability of choosing the option supported by the higher-validity cue is constant regardless of the opposing cue's validity. Subjective TTB, however, samples the primary cue via a softmax distribution over the validities of the discriminating cues. Therefore, as the validity gap between the two cues increases, the probability of sampling the higher-validity cue increases. This metric computes the difference in the proportion of choices favoring the higher-validity cue between trials with a large validity gap (index difference >= 3) and trials with a small validity gap (index difference == 1). Pure TTB predicts a difference near zero, while Subjective TTB predicts a significantly positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 3904,
  "candidates_token_count": 503,
  "total_token_count": 6543
}
```
