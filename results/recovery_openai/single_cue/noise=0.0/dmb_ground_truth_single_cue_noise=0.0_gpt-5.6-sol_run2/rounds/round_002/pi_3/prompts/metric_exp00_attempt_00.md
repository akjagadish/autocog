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
**Validities (n_features=10):** [0.7, 0.95, 0.55, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 3: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 5: A=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  trial 12: A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]

**Rationale:** This 10-cue design pits universal positional accessibility against switch-closure in balanced two-versus-two conflicts. Experts 1 and 4-10 all have validity 0.70; only these equal-validity experts discriminate. Experts 2 and 3 provide the required validity spread (0.95 and 0.55) but are tied, so validity differences cannot explain choices. In each of the first six pairs, A is supported by two cues including the final discriminating cue, whereas B is supported by two cues whose combined positions are later on average. Switch-Closure Coalition Integration enters its balanced-coalition branch: because A supplies the final discriminating cue and all four discriminating cues have identical validity, it predicts the same above-chance A-choice probability for all six configurations (approximately 0.58-0.66 across its parameter ranges), irrespective of spacing. Accessibility-Balanced Evidence Integration instead sums the cues' exponentially increasing positional weights. In every pair, B's two adjacent late cues have greater total accessibility than A's early-plus-terminal pair, so it robustly predicts P(A)<0.5. It additionally predicts graded changes as A's early cue moves later and the accessibility deficit shrinks, whereas the competing theory remains flat because coalition size and closure direction are unchanged. The final six pairs are exact A/B mirrors, producing complementary predictions and controlling response-side bias. Thus the decisive signature is a sign reversal—advocated-theory preference for B versus competing-theory preference for A—together with a graded spacing effect predicted only by the advocated theory. There are 12 unique pairs, each repeated eight times, for 96 trials.

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
**Description:** Switch-Closure Coalition Integration proposes that people encode communicated cue validities in a strongly compressed form and then organize simultaneously supporting cues into directional coalitions. Evidence contributed by a coalition grows sublinearly with its size, so each additional cue matters but has diminishing impact. Attention is configuration-dependent rather than governed by a universal positional gradient. A conflict between two singleton cues receives no positional weighting. When equally sized multi-cue coalitions compete, completing the later coalition produces a small switch-closure advantage because it is the most recently completed coherent interpretation. When coalition sizes differ, attention favors coalitions that begin early and remain locally coherent, but this structural gate is strongest for nearly balanced multi-cue conflicts such as three-versus-two and attenuated when a singleton competes with a growing coalition. Thus positional effects can reverse across configurations: terminal closure can favor the later side in balanced coalitions, early coherent organization can dominate nearly balanced multi-cue conflicts, and singleton-versus-coalition decisions remain governed primarily by gradual accumulation. Subject-specific compression, accumulation, attention, response sensitivity, and lapse parameters produce heterogeneity without trial-by-trial learning, which is appropriate because the task provides no outcome feedback.

**Parameters:**
- validities: validities
- validity_compression: [0.20, 0.45]
- validity_reliance: [0.03, 0.08]
- accumulation_saturation: [0.72, 0.84]
- balanced_terminal_attention: [0.42, 0.56]
- coalition_primacy: [0.12, 0.25]
- coherence_gain: [0.03, 0.10]
- near_balance_gain: [4.75, 6.25]
- singleton_gate_scale: [0.20, 0.45]
- beta: [0.75, 1.05]
- epsilon: [0.03, 0.09]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Switch-Closure Coalition Integration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    accumulation_saturation = float(parameters["accumulation_saturation"])
    balanced_terminal_attention = float(parameters["balanced_terminal_attention"])
    coalition_primacy = float(parameters["coalition_primacy"])
    coherence_gain = float(parameters["coherence_gain"])
    near_balance_gain = float(parameters["near_balance_gain"])
    singleton_gate_scale = float(parameters["singleton_gate_scale"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Compress communicated diagnosticities and mix them with a common
    # baseline. This preserves validity information without allowing one
    # instructed number to become lexicographically decisive.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)
    cue_weights = (1.0 - validity_reliance) + validity_reliance * diagnosticity
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    pos_idx = np.flatnonzero(differences > 0.0)
    neg_idx = np.flatnonzero(differences < 0.0)
    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)

    if n_pos == 0 and n_neg == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_evidence(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # Division by n**saturation gives sublinear accumulation:
        # total evidence grows as approximately n**(1-saturation).
        return float(np.sum(cue_weights[indices])) / (float(n) ** accumulation_saturation)

    def coalition_structure(indices):
        n = int(indices.size)
        if n < 2:
            return 0.0, 0.0
        scale = float(max(n_features - 1, 1))
        onset_primacy = 1.0 - float(indices[0]) / scale
        adjacent_links = float(np.sum(np.diff(indices) == 1))
        coherence = adjacent_links / float(n - 1)
        return onset_primacy, coherence

    evidence_a = coalition_evidence(pos_idx)
    evidence_b = coalition_evidence(neg_idx)
    log_gate_a = 0.0
    log_gate_b = 0.0

    if n_pos == 1 and n_neg == 1:
        # A pure singleton-versus-singleton conflict has no positional gate.
        pass
    elif n_pos == n_neg and n_pos >= 2:
        # Balanced multi-cue interpretations receive a bounded closure effect.
        # It depends on which coalition supplies the final piece of
        # discriminating evidence, not on a fixed weight for every position.
        last_pos = int(pos_idx[-1])
        last_neg = int(neg_idx[-1])
        half = 0.5 * balanced_terminal_attention
        if last_pos > last_neg:
            log_gate_a += half
            log_gate_b -= half
        elif last_neg > last_pos:
            log_gate_b += half
            log_gate_a -= half
    else:
        # In unequal conflicts, early and coherent coalitions are chunked and
        # maintained more effectively. The gate is amplified when both sides
        # form multi-cue, nearly balanced coalitions, but attenuated when one
        # side is a singleton so that growing opposition accumulates gradually.
        onset_a, coherence_a = coalition_structure(pos_idx)
        onset_b, coherence_b = coalition_structure(neg_idx)
        raw_a = coalition_primacy * onset_a + coherence_gain * coherence_a
        raw_b = coalition_primacy * onset_b + coherence_gain * coherence_b
        center = 0.5 * (raw_a + raw_b)
        log_gate_a = raw_a - center
        log_gate_b = raw_b - center

        if min(n_pos, n_neg) >= 2 and abs(n_pos - n_neg) == 1:
            log_gate_a *= near_balance_gain
            log_gate_b *= near_balance_gain
        elif min(n_pos, n_neg) == 1:
            log_gate_a *= singleton_gate_scale
            log_gate_b *= singleton_gate_scale

    evidence_a *= np.exp(np.clip(log_gate_a, -20.0, 20.0))
    evidence_b *= np.exp(np.clip(log_gate_b, -20.0, 20.0))
    net_a = evidence_a - evidence_b

    logits = np.array([0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    if len(data) == 0:\n        return 0.5\n\n    a = np.vstack(data[\"option_a_ratings\"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())\n    b = np.vstack(data[\"option_b_ratings\"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())\n    responses = pd.to_numeric(data[\"response\"], errors=\"coerce\").to_numpy(dtype=float)\n\n    n_features = a.shape[1]\n    if n_features == 1:\n        positional_weights = np.ones(1, dtype=float)\n    else:\n        positions = np.arange(n_features, dtype=float) / float(n_features - 1)\n        positional_weights = np.exp(2.1 * positions)\n\n    # Positive evidence means positional accessibility favors A; negative\n    # evidence means it favors B.\n    accessibility_difference = np.dot(a - b, positional_weights)\n    valid = np.isfinite(responses) & (accessibility_difference != 0.0)\n    if not np.any(valid):\n        return 0.5\n\n    chose_a = responses[valid] == 0.0\n    accessibility_favors_a = accessibility_difference[valid] > 0.0\n    accessibility_aligned = chose_a == accessibility_favors_a\n    return float(np.mean(accessibility_aligned))",
  "rationale": "This is the proportion of choices favoring the option with greater exponentially recency-weighted cue support. In every designed conflict, that option is the adjacent late-cue coalition rather than the coalition supplying the final discriminating cue. Accessibility-Balanced Evidence Integration therefore predicts values above 0.5, whereas Switch-Closure Coalition Integration predicts values below 0.5 because its balanced-coalition closure gate favors the opposite option. The six exact A/B mirrors make the measure insensitive to a stable A-versus-B response bias. Averaging all 96 binary observations gives a stable per-subject estimate, while the fixed midpoint recency gradient avoids estimating noisy subject-level parameters."
}
```

## Usage

```json
{
  "input_tokens": 5954,
  "output_tokens": 2003
}
```
