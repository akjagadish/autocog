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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 8: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** Both the Position Bias and Zero-Intelligence theories assume that participants completely ignore the feature values and validities. To dissociate them, we must look at the baseline choice probability for Option A across trials that vary drastically in objective quality. We include trials where Option B overwhelmingly dominates Option A (e.g., B has all positive ratings, A has none), trials where Option A dominates, and mixed trials. The Competing Theory (Zero-Intelligence) predicts exactly 50% choice of Option A across all trial types. The Advocated Theory (Position Bias) predicts a flat but elevated choice probability for Option A (e.g., 55-60%) regardless of whether Option A is objectively terrible or perfect. By showing that p(A) remains stably above 0.5 even when B is clearly superior, we can quantitatively support Position Bias over pure Random Choice while simultaneously ruling out feature-integrating heuristics.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Position Bias / Option A Default Theory: Participants do not systematically integrate feature validities or use complex heuristics. Instead, due to reading order (e.g., left-to-right or top-to-bottom) or motor ease, they exhibit a slight systematic bias toward choosing the first option presented (Option A). This structural bias is mixed with random guessing, leading to a state-independent choice probability where Option A is chosen slightly more often than Option B, regardless of the features of the options.

**Parameters:**
- p_A: [0.5, 0.6]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the state and history, relying solely on a structural bias toward Option A.
    p_A = float(parameters["p_A"])
    return np.array([p_A, 1.0 - p_A])
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
**Description:** Zero-Intelligence / Random Choice Theory: In this specific experimental paradigm, participants do not systematically integrate feature validities or use non-compensatory heuristics. Due to task disengagement, cognitive overload, or the lack of trial-by-trial feedback, participants do not learn or apply any complex decision rules. Instead, they simply guess randomly between the two options on every trial, resulting in a 50/50 probability for any pair of options regardless of their features.

**Parameters:**
(none)

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The Zero-Intelligence model completely ignores the state and history,
    # assuming participants guess uniformly at random on every trial.
    return np.array([0.5, 0.5])
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
[0] rationale: The Advocated Theory (Position Bias) posits that subjects will choose Option A with a probability slightly greater than 0.5 (between 0.5 and 0.6), uniformly across all trials. The Competing Theory (Zero-Intelligence) predicts exactly 0.5 probability of choosing Option A. Since both theories assume features are ignored, the simplest and most statistically powerful metric to discriminate them is the overall base rate of choosing Option A.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # The metric simply calculates the overall proportion of trials where Option A was chosen.
    # Since 'response' is 0 for Option A and 1 for Option B, 1.0 - mean(response) gives p(A).
    return 1.0 - float(data['response'].mean())
outcome: self_sim=0.5413 (var=0.0032) adversary_sim=0.5142 (var=0.0026) welch_t=+1.766 p=0.08382 (N=25, alpha=0.01) -> reject

[1] rationale: Both theories ignore feature validities, so the sufficient statistic for discriminating them is the overall rate of choosing Option A. The raw proportion previously suffered from high binomial noise relative to the effect size. Applying the log-odds (logit) transformation stretches out probabilities as they move away from 0.5, mathematically amplifying the signal of the Position Bias theory (which predicts p(A) > 0.5) while Zero-Intelligence remains anchored at a logit of approximately 0. The small smoothing constant ensures stability.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    # Calculate the empirical probability of choosing Option A
    p_a = 1.0 - data['response'].mean()
    # Apply an empirical logit transformation.
    # This non-linear transformation amplifies differences away from 0.5,
    # providing better discriminability between exactly 0.5 (Zero-Intelligence)
    # and elevated probabilities (Position Bias) than the raw proportion.
    return float(np.log((p_a + 0.01) / (1.0 - p_a + 0.01)))
outcome: self_sim=0.1621 (var=0.0517) adversary_sim=0.0556 (var=0.0415) welch_t=+1.745 p=0.08751 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Count choices for Option A\n    k = np.sum(data['response'] == 0)\n    n = len(data)\n    \n    # The Advocated Theory (Position Bias) posits p_A is uniformly distributed between 0.5 and 0.6.\n    # The Competing Theory (Zero-Intelligence) posits p_A is exactly 0.5.\n    # We construct the log-likelihood ratio (LLR) of the observed choices under PB vs ZI.\n    # This is the theoretically optimal test statistic (Neyman-Pearson lemma) for discriminating them.\n    \n    ps = np.linspace(0.5, 0.6, 51)\n    \n    # For each candidate p in the PB theory, calculate the log likelihood ratio compared to p=0.5\n    # LLR_p = k * log(p / 0.5) + (n - k) * log((1 - p) / 0.5)\n    log_terms = k * np.log(2 * ps) + (n - k) * np.log(2 * (1 - ps))\n    \n    # Integrate out the uniform prior over p using the log-sum-exp trick for numerical stability\n    max_log = np.max(log_terms)\n    llr = max_log + np.log(np.mean(np.exp(log_terms - max_log)))\n    \n    return float(llr)",
  "rationale": "Since both theories assume features are ignored, the sequence of choices is fully summarized by the total number of times Option A was chosen. However, simple linear transformations like the raw proportion or empirical logit failed to reach statistical significance because they do not optimally separate the specific distributions of the two theories (a point mass at p=0.5 vs a uniform distribution of p in [0.5, 0.6]). By computing the exact Log-Likelihood Ratio (LLR) of the data under the two theories' structural assumptions, we apply the Neyman-Pearson lemma to construct the most statistically powerful metric possible for this exact contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 3248,
  "candidates_token_count": 503,
  "total_token_count": 9675
}
```
