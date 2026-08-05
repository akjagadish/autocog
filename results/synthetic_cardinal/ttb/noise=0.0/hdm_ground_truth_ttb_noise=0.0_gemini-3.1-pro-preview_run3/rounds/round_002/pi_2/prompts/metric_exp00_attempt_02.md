# metric_exp00_attempt_02

## System Prompt

You are a psychology researcher proposing a metric in the Heuristic Decision Making domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 50). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` integer expert ratings in [0, rating_max]. The per-feature validities, n_features, and rating_max are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. All three — `n_features`, `validities`, and `rating_max` — are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=4):** [1.0, 0.5, 0.5, 0.5]

**rating_max:** 10

**Trial pairs (n=10):**
  trial 1: A=[0, 10, 10, 10]  B=[10, 3, 3, 4]
  trial 2: A=[10, 3, 3, 4]  B=[0, 10, 10, 10]
  trial 3: A=[0, 10, 10, 10]  B=[8, 3, 3, 4]
  trial 4: A=[8, 3, 3, 4]  B=[0, 10, 10, 10]
  trial 5: A=[10, 0, 0, 0]  B=[0, 6, 7, 7]
  trial 6: A=[0, 6, 7, 7]  B=[10, 0, 0, 0]
  trial 7: A=[10, 5, 0, 0]  B=[0, 0, 10, 10]
  trial 8: A=[0, 0, 10, 10]  B=[10, 5, 0, 0]
  trial 9: A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  trial 10: A=[0, 0, 10, 10]  B=[10, 10, 0, 0]

**Rationale:** To quantitatively dissociate the advocated pure WADD theory from the competing Mixture (Tallying + WADD) theory, we exploit trials where the two models make qualitatively different predictions about indifference. Specifically, we construct 'WADD-tie' trials where the weighted sums of the two options are exactly identical, meaning pure WADD predicts a 50/50 guess. However, on these same trials, one option strictly wins on a majority of features, meaning Tallying strongly prefers it. The competing mixture model predicts a significant deviation from 50/50 towards the Tallying winner, whereas the advocated pure WADD model predicts strict indifference. We also include 'Tallying-tie' trials where options win on an equal number of features (Tallying predicts 50/50) but the weighted sums differ, allowing us to observe if choice probabilities are extreme (as pure WADD predicts) or dampened towards 0.5 (as the mixture model predicts).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    w = np.asarray(parameters["validities"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"validities length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ w

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    return int(np.argmax(probabilities))
```


## COMPETING THEORY
**Description:** Human decision-making in multi-attribute choice is not governed by a single universal heuristic. Instead, individuals differ in their strategies or switch between them, such that the population's choices reflect a mixture of Tallying (which counts strict feature-wise wins and ignores magnitudes and validities) and Weighted Additive (WADD, which integrates both magnitudes and validities). A mixture weight parameter 'alpha' dictates the probability of using Tallying versus WADD on any given trial. Response noise enters through a softmax over the scores of the chosen heuristic, plus an independent lapse rate.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- validities: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    
    # Tallying heuristic
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # WADD heuristic
    w = np.asarray(parameters["validities"], dtype=float)
    wadd_scores = stim @ w
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of heuristics
    alpha = float(parameters["alpha"])
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    return p_mixed
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    return int(np.argmax(probs))
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features integer expert ratings (each in [0, rating_max]) for option A on this trial.
- option_b_ratings: List of n_features integer expert ratings (each in [0, rating_max]) for option B on this trial.
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
[0] rationale: This metric isolates 'WADD-tie' trials where the two options have exactly equal Weighted Additive scores but one option dominates on a strict count of features (Tallying). According to the pure WADD model, subjects should be indifferent on these trials (predicting a 0.5 choice proportion for the Tallying-preferred option). In contrast, the Mixture model (Tallying + WADD) predicts a systematic deviation from 0.5 towards the option favored by Tallying. By measuring the proportion of choices aligning with Tallying on WADD-tie trials, we cleanly discriminate the two theories.
metric_source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    w = np.array([1.0, 0.5, 0.5, 0.5])
    
    tally_aligned = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        wadd_a = np.dot(a, w)
        wadd_b = np.dot(b, w)
        
        # Check for WADD tie
        if abs(wadd_a - wadd_b) < 1e-5:
            tally_a = np.sum(a > b)
            tally_b = np.sum(b > a)
            
            if tally_a != tally_b:
                tally_choice = 0 if tally_a > tally_b else 1
                tally_aligned.append(1 if row['response'] == tally_choice else 0)
                
    if not tally_aligned:
        return 0.5
        
    return float(np.mean(tally_aligned))
outcome: self_sim=0.8500 (var=0.1225) adversary_sim=0.9100 (var=0.0769) welch_t=-0.950 p=0.3445 (N=50, alpha=0.01) -> reject

[1] rationale: By isolating the trial where Tallying and WADD's predictions diverge most frequently (Trial 5/6), we maximize the observable difference between the two deterministic policies. Pure WADD chooses the '10'-option whenever the subjective weight for feature 1 is dominant (~15% of subjects). The Mixture model, however, will dampen this rate because subjects with a dominant feature 1 weight but a high 'alpha' parameter will still follow Tallying and choose the other option. This metric directly captures the maximal probability gap between the two theories.
metric_source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # We focus on Trial 5 where Tallying and WADD have the highest chance of disagreement.
    # Trial 5: option A has ratings [10, 0, 0, 0] and B has [0, 6, 7, 7].
    # Tallying prefers B (3 wins to 1).
    # WADD prefers A if the subjective validity for feature 1 is overwhelmingly high.
    # Pure WADD subjects with a dominant feature 1 weight will consistently choose A.
    # Mixture subjects with a dominant feature 1 weight will still often choose B if their alpha (Tallying weight) is high.
    # We calculate the proportion of times a subject chooses A on this specific trial.
    
    subject_means = []
    for subj, subj_df in data.groupby('subject_id'):
        trial_5_responses = []
        for _, row in subj_df.iterrows():
            a = tuple(row['option_a_ratings'])
            b = tuple(row['option_b_ratings'])
            
            # Identify Trial 5 (or its symmetric equivalent Trial 6)
            if a == (10, 0, 0, 0) and b == (0, 6, 7, 7):
                trial_5_responses.append(1 if row['response'] == 0 else 0)
            elif a == (0, 6, 7, 7) and b == (10, 0, 0, 0):
                trial_5_responses.append(1 if row['response'] == 1 else 0)
                
        if trial_5_responses:
            subject_means.append(np.mean(trial_5_responses))
            
    if not subject_means:
        return 0.0
        
    return float(np.mean(subject_means))
outcome: self_sim=0.1400 (var=0.1204) adversary_sim=0.1000 (var=0.0900) welch_t=+0.617 p=0.5389 (N=50, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\nimport pandas as pd\n\ndef metric(data: pd.DataFrame) -> float:\n    subject_devs = []\n    \n    for subj, subj_df in data.groupby('subject_id'):\n        p1_choices = []\n        p2_choices = []\n        \n        for _, row in subj_df.iterrows():\n            a = tuple(row['option_a_ratings'])\n            b = tuple(row['option_b_ratings'])\n            resp = row['response']\n            \n            # Identify Tallying-tie Pair 1\n            if a == (10, 5, 0, 0) and b == (0, 0, 10, 10):\n                p1_choices.append(1 if resp == 0 else 0)\n            elif a == (0, 0, 10, 10) and b == (10, 5, 0, 0):\n                p1_choices.append(1 if resp == 1 else 0)\n                \n            # Identify Tallying-tie Pair 2\n            elif a == (10, 10, 0, 0) and b == (0, 0, 10, 10):\n                p2_choices.append(1 if resp == 0 else 0)\n            elif a == (0, 0, 10, 10) and b == (10, 10, 0, 0):\n                p2_choices.append(1 if resp == 1 else 0)\n                \n        devs = []\n        if len(p1_choices) > 0:\n            devs.append(abs(np.mean(p1_choices) - 0.5))\n        if len(p2_choices) > 0:\n            devs.append(abs(np.mean(p2_choices) - 0.5))\n            \n        if devs:\n            subject_devs.append(np.mean(devs))\n            \n    if not subject_devs:\n        return 0.0\n        \n    return float(np.mean(subject_devs))",
  "rationale": "This metric focuses on 'Tallying-tie' trials where both options win on exactly two features (e.g., [10, 5, 0, 0] vs [0, 0, 10, 10]). On these trials, the Tallying heuristic is completely indifferent, outputting a 50/50 choice probability. The pure WADD theory, which integrates magnitudes, will generally strongly favor one option depending on the subject's validities, leading to extreme choice proportions (close to 0 or 1). The Mixture theory, however, blends the extreme WADD preference with the 50/50 Tallying preference, dampening the choice probabilities toward 0.5. By calculating the absolute deviation of each subject's choice proportion from 0.5 on these trials, we can cleanly separate the theories: pure WADD predicts a high deviation, while the Mixture model predicts a significantly lower deviation."
}
```

## Usage

```json
{
  "prompt_token_count": 4752,
  "candidates_token_count": 741,
  "total_token_count": 8449
}
```
