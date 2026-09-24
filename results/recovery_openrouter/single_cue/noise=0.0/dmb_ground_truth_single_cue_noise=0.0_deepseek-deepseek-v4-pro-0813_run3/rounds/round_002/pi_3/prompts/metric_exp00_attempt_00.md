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
**Validities (n_features=16):** [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 2: A=[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 12: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 15: A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 16: A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** Sixteen experts: eight with validity 0.50 and eight with validity 1.00, so centered validities are -0.25 and +0.25. This binary spread maximizes the validity-composition score S = sum_j (v_j - 0.75)*(A_j - B_j) for a fixed tally difference and makes the competing fallback weights (v_j - 0.5)^gamma exactly zero for low-validity experts. The 16 unique pairs are organized into three blocks. (1) Six d=+1 pairs (A wins one more feature than B) with S = -3.75, -2.75, -1.75, 0.25, 1.25, 3.75. (2) Six d=-1 pairs, the A/B swaps of the first block, with S = 3.75, 2.75, 1.75, -0.25, -1.25, -3.75. (3) Four d=0 tally-tie pairs in which the high-validity tally H = #high A wins - #high B wins is +8, +4, -4, -8. On the non-tie pairs the competing validity-gated tallying rule never opens its validity gate because |d| > tau, so it predicts a single p(A) for all six d=+1 pairs and a single p(A) for all six d=-1 pairs, regardless of S. The advocated validity-graded integration predicts advantage D = d + kappa*S, so within each d block p(A) must vary with S for any subject with kappa != 0 (e.g., D shifts by +/-0.375 at kappa=0.1), giving a within-block S psychometric curve and overdispersion. On the d=0 pairs the competing rule opens its validity gate and, with high beta_v, predicts p(A) strongly follows the high-validity tally (high for H=+8/+4, low for H=-4/-8, scaled by fallback_p), whereas the advocated rule predicts D = kappa*S and, because kappa has a symmetric zero-centered prior, aggregate p(A)=0.5 for all four tie pairs. Thus the design produces a double dissociation: non-tie trials test continuous validity reweighting against a closed gate, and tie trials test the gated validity fallback against a zero-average compensatory rule. Swapped pairs control for A/B response bias; 16 unique pairs x 6 repeats = 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People use a validity-graded compensatory integration rule. On every choice they inspect all features, compute a signed option advantage by summing each discriminating feature's contribution, and then choose via a softmax over that advantage plus an independent lapse. Unlike Take The Best there is no early stopping. Unlike pure Tallying, each feature's contribution is weighted by a validity-sensitive multiplier. The multiplier is anchored around equal weighting, w_j = 1 + kappa * (v_j - mean(v)), normalized so the average weight is exactly 1. Because kappa has a symmetric prior centered at zero, the population-level expected weight vector is the equal-weight vector, but individual subjects may slightly overweight or underweight higher-validity cues. Tied features contribute nothing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- kappa: [-0.10, 0.10]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    # Validity-graded compensatory integration.
    # state is the current pair of option feature vectors, shape (2, n_features).
    # Row 0 = option A, row 1 = option B. History is irrelevant because there is
    # no trial-by-trial feedback and the rule is applied independently each trial.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-graded integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Validity-sensitive weights around a strict equal-weight anchor.
    # The centered validities sum to zero, so the raw weights sum to n_features.
    # Normalizing by their mean keeps the scale of the advantage score directly
    # comparable to an unweighted signed tally when kappa is small.
    kappa = float(parameters["kappa"])
    centered_validities = validities - np.mean(validities)
    weights = 1.0 + kappa * centered_validities
    weights = weights / np.mean(weights)

    # Discriminating features contribute according to their weights; ties
    # have difference zero and thus contribute nothing.
    advantage = float(np.dot(weights, stim[0] - stim[1]))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Stable softmax over the signed advantage versus a zero-advantage boundary.
    # This reduces to logistic choice in the weighted advantage.
    scores = np.array([advantage, 0.0])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** People use a validity-gated tallying rule. On every trial they first compute the unweighted tally difference d = #A wins - #B wins. If the tallies clearly differ, choice is driven by a softmax over the two unweighted tallies, exactly as in Tallying, so majority-tally behavior is preserved. Only when the tallies are tied, or within a small threshold tau of tied, is communicated validity used: the decision maker computes validity-weighted evidence V = sum_j w_j * (A_j - B_j), with weights w_j = (v_j - 0.5)^gamma increasing steeply and convexly in validity. Choice on these tie trials is then a separate softmax over V and -V, with its own inverse temperature beta_v and lapse epsilon_v, and with probability 1 - fallback_p the subject simply guesses instead of using this validity fallback. Thus validity does not continuously reweight every trial, as in a compensatory rule, nor are ties resolved by blind guessing, as in Tallying.

**Parameters:**
- beta_tally: [0.1, 20.0]
- epsilon_tally: [0.0, 0.5]
- tau: [0.0, 0.5]
- gamma: [6.0, 10.0]
- beta_v: [100.0, 300.0]
- epsilon_v: [0.0, 0.2]
- fallback_p: [0.7, 1.0]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    # Validity-gated tallying.
    # state is the current pair of option feature vectors, shape (2, n_features).
    # Row 0 = option A, row 1 = option B. History is irrelevant because choices
    # are made independently on each trial with no feedback.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-gated tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    a, b = stim[0], stim[1]
    diff = a - b

    # Unweighted tally of strict feature-wise wins; tied features contribute nothing.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_d = a_wins - b_wins

    tau = float(parameters["tau"])

    def stable_softmax(scores, inv_temp):
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        p = e / np.sum(e)
        return p

    if np.abs(tally_d) > tau:
        # Unequal tallies: exactly the Tallying softmax, preserving majority-tally
        # choice behavior on clearly discriminating trials.
        scores = np.array([a_wins, b_wins])
        p_core = stable_softmax(scores, float(parameters["beta_tally"]))
        eps_tally = float(parameters["epsilon_tally"])
        probs = (1.0 - eps_tally) * p_core + eps_tally * np.array([0.5, 0.5])
    else:
        # Tally tie or near-tie: validity-gated integration instead of guessing.
        gamma = float(parameters["gamma"])
        weights = np.maximum(validities - 0.5, 0.0) ** gamma
        v = float(np.dot(weights, diff))

        scores_v = np.array([v, -v])
        p_core_v = stable_softmax(scores_v, float(parameters["beta_v"]))
        eps_v = float(parameters["epsilon_v"])
        p_valid = (1.0 - eps_v) * p_core_v + eps_v * np.array([0.5, 0.5])

        fallback_p = float(parameters["fallback_p"])
        probs = fallback_p * p_valid + (1.0 - fallback_p) * np.array([0.5, 0.5])

    probs = np.clip(probs, 0.0, 1.0)
    total = probs.sum()
    if total <= 0.0:
        probs = np.array([0.5, 0.5])
    else:
        probs = probs / total
    return probs
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0.0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\ndef metric(data: pd.DataFrame) -> float:\n    a = np.asarray(data['option_a_ratings'].tolist(), dtype=float)\n    b = np.asarray(data['option_b_ratings'].tolist(), dtype=float)\n    diff = a - b\n    total_tally = diff.sum(axis=1)\n    high_tally = diff[:, 8:].sum(axis=1)\n    chose_a = (data['response'].to_numpy() == 0).astype(float)\n    subject = data['subject_id'].to_numpy()\n\n    per_subject = []\n    for sid in np.unique(subject):\n        idx = subject == sid\n        tie = total_tally[idx] == 0.0\n        high_pos = high_tally[idx] > 0\n        high_neg = high_tally[idx] < 0\n        pos_trials = tie & high_pos\n        neg_trials = tie & high_neg\n        if pos_trials.sum() > 0 and neg_trials.sum() > 0:\n            diff_rate = float(chose_a[idx][pos_trials].mean() - chose_a[idx][neg_trials].mean())\n            per_subject.append(diff_rate)\n\n    if not per_subject:\n        return float('nan')\n\n    return float(np.mean(per_subject))",
  "rationale": "This metric isolates the four tied-tally trial types where the total feature tally difference is 0. For each subject it computes the proportion of A choices on ties where the eight high-validity experts favor A (high-validity tally H > 0) minus the proportion of A choices on ties where they favor B (H < 0), then averages across subjects. The gated-tallying theory predicts a large positive value because on ties it opens its validity gate and follows the high-validity experts. The advocated validity-graded theory has symmetric zero-centered kappa, so positive-H and negative-H tie advantages cancel in expectation across subjects, giving a near-zero mean while retaining per-subject variation."
}
```

## Usage

```json
{
  "input_tokens": 6200,
  "output_tokens": 24085,
  "reasoning_tokens": 23606,
  "provider": "Baidu"
}
```
