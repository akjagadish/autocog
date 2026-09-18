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
**Validities (n_features=7):** [0.52, 0.58, 0.65, 0.72, 0.8, 0.88, 0.95]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1]
  trial 3: A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 11: A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1]
  trial 12: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1]

**Rationale:** This 7-cue design holds the tally margin constant while systematically relocating the same 3-versus-2 cue coalition from early, low-validity positions to late, high-validity positions. In each of the first six pairs, A wins exactly three cues, B wins exactly two, and two cues tie. Tallying therefore predicts exactly the same A-choice probability on all six: (1-epsilon)*sigmoid(beta)+epsilon/2. The final six pairs are A/B-swapped mirrors, for which Tallying predicts the complementary constant probability. Accessibility-Balanced Evidence Integration instead assigns progressively greater weight to later cues; the ascending validities reinforce this positional gradient throughout the advocated parameter ranges. Consequently, its predicted A-choice probability should increase monotonically across the first six configurations. At the low end, A's three early wins should be overcome by B's two late wins, reversing the tally preference; at the high end, A's late coalition should produce a strong A preference. Coalition saturation further penalizes A's three-cue majority relative to B's two-cue coalition but is constant in form across the progression. The mirrored pairs should exhibit the complementary graded pattern, controlling response-side bias. Thus the decisive signature is a large configuration-dependent slope, including a preference reversal despite an invariant tally margin; Tallying permits neither. There are 12 unique pairs, each repeated eight times, yielding exactly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Accessibility-Balanced Evidence Integration proposes that people encode each communicated cue validity as a subjective diagnosticity signal and combine it with a separate positional-accessibility signal. Diagnosticity favors objectively valid cues, with nonlinear compression or sharpening of validity differences, whereas accessibility can favor cues appearing later or lower in the display. A subject-specific balance determines how strongly each code influences cue weight. On each choice, all discriminating cues contribute their signed evidence to a common accumulator; therefore, several opposing cues can gradually overcome an initially favored cue. Because the accumulator retains evidence magnitude and configuration, the theory predicts graded coalition effects rather than selecting either the first or last discriminating cue. Evidence supporting each option exhibits mild, imbalance-dependent within-coalition diminishing returns before the two directional totals are compared, followed by ordinary response noise and occasional lapses.

**Parameters:**
- validities: validities
- validity_position_balance: [0.25, 0.35]
- recency_gradient: [1.8, 2.4]
- validity_curvature: [0.65, 1.0]
- coalition_saturation: [0.65, 0.8]
- beta: [0.65, 1.15]
- epsilon: [0.02, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Accessibility-Balanced Integration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    balance = float(parameters["validity_position_balance"])
    recency = float(parameters["recency_gradient"])
    curvature = float(parameters["validity_curvature"])
    coalition_saturation = float(parameters["coalition_saturation"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Communicated validities are represented as log-odds diagnosticities.
    # Curvature captures sharpening or compression of perceived differences.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    max_diagnosticity = float(np.max(diagnosticity))
    if max_diagnosticity > 0.0:
        diagnosticity = diagnosticity / max_diagnosticity
    else:
        diagnosticity = np.ones(n_features, dtype=np.float64)
    diagnosticity = np.power(np.maximum(diagnosticity, 0.0), curvature)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)

    # Later/lower display positions have greater accessibility when recency > 0.
    if n_features == 1:
        position = np.zeros(1, dtype=np.float64)
    else:
        position = np.arange(n_features, dtype=np.float64) / float(n_features - 1)
    accessibility = np.exp(recency * position)
    accessibility /= max(float(np.mean(accessibility)), 1e-12)

    # A convex mixture makes the validity-versus-position tradeoff explicit.
    weights = balance * diagnosticity + (1.0 - balance) * accessibility
    weights /= max(float(np.mean(weights)), 1e-12)

    differences = a - b
    discriminating = differences != 0.0
    n_discriminating = int(np.sum(discriminating))
    if n_discriminating == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Evidence is accumulated separately for the two options. Diminishing
    # returns depend on directional coalition imbalance rather than absolute
    # coalition size, preserving full configuration evidence for equal-sized
    # coalitions while tempering a numerical majority against a minority.
    positive = differences > 0.0
    negative = differences < 0.0
    n_positive = max(int(np.sum(positive)), 1)
    n_negative = max(int(np.sum(negative)), 1)
    min_coalition = min(n_positive, n_negative)
    positive_evidence = float(np.dot(weights, np.maximum(differences, 0.0)))
    negative_evidence = float(np.dot(weights, np.maximum(-differences, 0.0)))
    evidence_a = (
        positive_evidence
        / (max(1.0, float(n_positive) / float(min_coalition)) ** coalition_saturation)
        - negative_evidence
        / (max(1.0, float(n_negative) / float(min_coalition)) ** coalition_saturation)
    )

    logits = np.array(
        [0.5 * beta * evidence_a, -0.5 * beta * evidence_a],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    if len(data) == 0:\n        return 0.0\n\n    # Fixed, preregistered midpoint instantiation of the advocated model.\n    validities = np.array([0.52, 0.58, 0.65, 0.72, 0.80, 0.88, 0.95], dtype=float)\n    balance = 0.30\n    recency = 2.10\n    curvature = 0.825\n    saturation = 0.725\n    beta = 0.90\n    epsilon = 0.05\n\n    v = np.clip(validities, 0.500001, 0.999999)\n    diagnosticity = np.log(v / (1.0 - v))\n    diagnosticity /= np.max(diagnosticity)\n    diagnosticity = np.power(np.maximum(diagnosticity, 0.0), curvature)\n    diagnosticity /= np.mean(diagnosticity)\n\n    position = np.arange(len(validities), dtype=float) / float(len(validities) - 1)\n    accessibility = np.exp(recency * position)\n    accessibility /= np.mean(accessibility)\n\n    cue_weights = balance * diagnosticity + (1.0 - balance) * accessibility\n    cue_weights /= np.mean(cue_weights)\n\n    predicted_majority_probs = []\n    chose_majority = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        d = a - b\n\n        # Orient every pair toward the option winning the 3-versus-2 tally.\n        tally_margin = float(np.sum(d > 0) - np.sum(d < 0))\n        if tally_margin == 0.0 or len(d) != len(cue_weights):\n            continue\n        majority_is_a = tally_margin > 0.0\n        oriented = d if majority_is_a else -d\n\n        positive = oriented > 0.0\n        negative = oriented < 0.0\n        n_pos = max(int(np.sum(positive)), 1)\n        n_neg = max(int(np.sum(negative)), 1)\n        min_n = min(n_pos, n_neg)\n\n        pos_evidence = float(np.dot(cue_weights, np.maximum(oriented, 0.0)))\n        neg_evidence = float(np.dot(cue_weights, np.maximum(-oriented, 0.0)))\n        evidence = (\n            pos_evidence / (max(1.0, float(n_pos) / float(min_n)) ** saturation)\n            - neg_evidence / (max(1.0, float(n_neg) / float(min_n)) ** saturation)\n        )\n\n        core_p = 1.0 / (1.0 + np.exp(-np.clip(beta * evidence, -50.0, 50.0)))\n        predicted_p = (1.0 - epsilon) * core_p + epsilon * 0.5\n        predicted_majority_probs.append(predicted_p)\n\n        response = int(row[\"response\"])\n        chose_majority.append(float((response == 0) if majority_is_a else (response == 1)))\n\n    if len(predicted_majority_probs) == 0:\n        return 0.0\n\n    p = np.asarray(predicted_majority_probs, dtype=float)\n    y = np.asarray(chose_majority, dtype=float)\n    contrast = p - np.mean(p)\n    return float(np.mean(contrast * (y - 0.5)))\n",
  "rationale": "This is a preregistered configuration-sensitivity covariance after orienting every stimulus toward its 3-cue tally winner. The centered weight is the majority-choice probability predicted by a fixed midpoint instantiation of Accessibility-Balanced Evidence Integration. Thus configurations where the majority coalition occupies strong late cues receive positive weights, while configurations where its three cues are outweighed by two late cues receive negative weights. Under Tallying, every oriented trial has the identical 3-versus-2 margin and hence the same expected majority-choice rate; centering the weights makes its expected metric exactly zero for every beta and epsilon. The advocated theory predicts that observed majority choice covaries strongly with these weights. All 96 responses contribute to the scalar, mirror trials are automatically placed in the same canonical orientation, and the balanced repeated schedule keeps the per-subject estimate comparatively stable. The weights are fixed independently of responses rather than fitted to the observed data."
}
```

## Usage

```json
{
  "input_tokens": 4511,
  "output_tokens": 3030
}
```
