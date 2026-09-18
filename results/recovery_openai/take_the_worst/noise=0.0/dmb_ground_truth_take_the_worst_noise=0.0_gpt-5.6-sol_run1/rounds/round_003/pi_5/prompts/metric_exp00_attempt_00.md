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
**Validities (n_features=8):** [0.91, 0.91, 0.82, 0.75, 0.69, 0.63, 0.57, 0.52]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 7: A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  trial 17: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  trial 20: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  trial 23: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0, 0]

**Rationale:** This design creates matched low-load confidence and recency dissociations while making Prospective Capacity-Gated Comparison exactly invariant. Experts 1 and 2 form a distinctive partially tied leading-validity tier. Each unreversed trial has either three or four discriminating cues, the highest-validity active cue favors A, and the equal-weight majority also favors A. Consequently, both branches of the competing theory select A on every unreversed trial, predicting the same P(A) throughout (approximately 0.80-0.89 over its parameter ranges), regardless of load, unanimity, or the losing cue's display position. Within each fixed active-cue set, Load-Indexed Format Arbitration instead predicts a graded profile. The unanimous condition has evidence +1 in every format and should produce the strongest A preference. Moving to a narrow A majority reduces normalized diffuse evidence to 1/3 at load 3 or 1/2 at load 4; moreover, the partially tied leading tier selectively attenuates low-format confidence at these loads. Among narrow-majority trials, placing the sole B-favoring cue late rather than at an intermediate position should further reduce A choice because sequential retrieval has a strong recency gradient. Thus the critical predicted ordering is unanimous > intermediate-position loser > latest-position loser, especially at load 4 where retrieval recruitment is greater. The competing theory predicts no such within-set ordering because tally direction and lexicographic direction are unchanged. Exact A/B reversals provide response-side balance. There are 24 unique pairs, yielding four repetitions each for 96 trials.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Load-Indexed Format Arbitration theory: people first register the observable number of cues that discriminate between the options and use it to recruit among three comparison formats. At low discriminating-cue load, they diffusely pool all diagnostic cues with equal weight, but the confidence attached to this normalized majority evidence varies stably across individuals. When the display contains a partially tied leading-validity tier, confidence in diffuse pooling is selectively attenuated at loads 3–4 because the tied priority structure interferes with forming a unitary low-load comparison. At intermediate load, people retrieve diagnostic cues sequentially from the display; retrieval accessibility combines communicated-validity tier with a recency gradient that is attenuated at very low loads and reaches full strength by overload loads. At high load, they compress the display to the highest-validity active tier, with a format-specific and person-specific evidence gain. The high-load transition distinguishes loads 6–7 from load 8, and its location varies coherently with recency strength. Observable validity-tier structure changes compressibility: a distinctive partially tied leading tier facilitates compression, whereas an entirely undifferentiated validity display delays it. Equal-validity tiers invoke a stable person-specific convention: some individuals use the first displayed diagnostic member of the tier, whereas others pool graded signed evidence across all diagnostic members of that tier. Under high load, whole-tier poolers obtain a modest coherence benefit when compressing a partially tied leading tier, strengthening—but not discretizing—the tier's graded mean evidence. These representations and their format-specific confidence generate signed evidence before ordinary response noise is applied.

**Parameters:**
- validities: validities
- low_center: [4.3, 4.8]
- high_center: [7.0, 7.35]
- transition_width: [0.9, 1.3]
- high_transition_width: [0.3, 0.5]
- partial_tie_acceleration: [0.9, 1.4]
- all_equal_delay: [0.9, 1.3]
- recency_compression_coupling: [0.25, 0.45]
- recency_strength: [4.8, 6.8]
- validity_strength: [0.2, 0.65]
- tie_rule: {0, 1}
- partial_tie_low_attenuation: [0.35, 0.6]
- low_evidence_gain: [0.6, 0.85]
- high_evidence_gain: [0.4, 0.65]
- partial_tier_pooling_gain: [1.45, 1.8]
- beta: [4.8, 6.2]
- epsilon: [0.0, 0.06]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Load-Indexed Format Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    signs = np.sign(a - b)
    active = np.flatnonzero(signs != 0.0)
    load = float(active.size)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    def stable_sigmoid(x):
        x = float(x)
        if x >= 0.0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        z = np.exp(x)
        return z / (1.0 + z)

    # Observable load gradually opens sequential retrieval and, later,
    # validity-tier compression. The low transition begins slightly earlier
    # so retrieval can contribute on three-cue conflicts.
    low_width = float(parameters["transition_width"])
    entered_retrieval = stable_sigmoid(
        (load - float(parameters["low_center"])) / low_width
    )

    global_best = float(np.max(validities))
    global_top_tier = np.flatnonzero(np.isclose(
        validities, global_best, rtol=1e-9, atol=1e-12
    ))
    partial_top_tie = 1.0 if 1 < global_top_tier.size < n_features else 0.0
    all_equal = 1.0 if global_top_tier.size == n_features else 0.0

    # Recency strength and compression timing are stably correlated within a
    # person. Strong-recency subjects retain retrieval longer; validity-led
    # subjects compress earlier. Distinct partial tiers facilitate chunking,
    # whereas an all-equal display supplies no validity boundary to compress.
    recency_strength = float(parameters["recency_strength"])
    recency_profile = (recency_strength - 5.8) / 1.0
    effective_high_center = (
        float(parameters["high_center"])
        + float(parameters["recency_compression_coupling"]) * recency_profile
        - partial_top_tie * float(parameters["partial_tie_acceleration"])
        + all_equal * float(parameters["all_equal_delay"])
    )
    compressed = stable_sigmoid(
        (load - effective_high_center)
        / float(parameters["high_transition_width"])
    )

    w_low = 1.0 - entered_retrieval
    w_high = entered_retrieval * compressed
    w_mid = entered_retrieval * (1.0 - compressed)

    # Low-load format: diffuse equal-weight comparison. Normalization makes
    # the evidence the diagnostic-cue majority margin rather than display size.
    low_evidence = float(np.mean(signs[active]))

    # A partially tied leading tier selectively lowers confidence in diffuse
    # pooling at loads 3-4, without changing strictly ordered low-load displays.
    low_gain_multiplier = 1.0
    if partial_top_tie > 0.0 and 3.0 <= load <= 4.0:
        low_gain_multiplier = float(parameters["partial_tie_low_attenuation"])

    # Construct communicated-validity tiers. Cues tied in communicated
    # validity have the same rank; display position remains separately visible.
    tier_rank = np.asarray(
        [np.sum(validities > validities[j] + 1e-12) for j in range(n_features)],
        dtype=float,
    )
    rank_scale = max(1.0, float(n_features - 1))
    normalized_rank = tier_rank / rank_scale
    if n_features > 1:
        display_position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        display_position = np.zeros(1, dtype=float)

    # Intermediate format: expected evidence from sequential retrieval.
    # Positional accessibility is mildly attenuated at three-cue load and
    # reaches its original strength by load six, preserving overload gradients.
    recency_load_scale = 0.85 + 0.15 * np.clip((load - 3.0) / 3.0, 0.0, 1.0)
    retrieval_log_access = (
        recency_load_scale * recency_strength * display_position[active]
        - float(parameters["validity_strength"]) * normalized_rank[active]
    )
    retrieval_log_access -= np.max(retrieval_log_access)
    retrieval_weights = np.exp(retrieval_log_access)
    retrieval_weights /= retrieval_weights.sum()
    mid_evidence = float(np.dot(retrieval_weights, signs[active]))

    # High-load format: compress to the best communicated-validity tier that
    # is diagnostic on this trial. Stable tie_rule=0 uses its first displayed
    # member; tie_rule=1 pools graded evidence over the complete active tier.
    best_validity = float(np.max(validities[active]))
    top_tier = active[np.isclose(
        validities[active], best_validity, rtol=1e-9, atol=1e-12
    )]

    tie_rule = int(parameters["tie_rule"])
    if tie_rule == 0:
        first_member = int(np.min(top_tier))
        high_evidence = float(signs[first_member])
    else:
        high_evidence = float(np.mean(signs[top_tier]))

    # At overload, coherent retrieval of an entire partially tied leading tier
    # gives whole-tier poolers a bounded confidence benefit. The underlying
    # evidence remains its graded mean rather than being collapsed to its sign.
    high_gain = float(parameters["high_evidence_gain"])
    if tie_rule == 1 and partial_top_tie > 0.0 and load >= 6.0:
        high_gain *= float(parameters["partial_tier_pooling_gain"])

    beta = float(parameters["beta"])

    def evidence_probabilities(evidence, gain=1.0):
        logits = np.array([
            0.5 * beta * float(gain) * float(evidence),
            -0.5 * beta * float(gain) * float(evidence),
        ], dtype=float)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    # Format recruitment is part of representation, not response noise.
    # Stable format-specific gains calibrate confidence without weakening the
    # intermediate positional profile that is needed at overload loads.
    p_core = (
        w_low * evidence_probabilities(
            low_evidence,
            low_gain_multiplier * float(parameters["low_evidence_gain"])
        )
        + w_mid * evidence_probabilities(mid_evidence)
        + w_high * evidence_probabilities(high_evidence, high_gain)
    )

    # Ordinary lapses occur only after each format has generated evidence.
    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Prospective Capacity-Gated Comparison theory: before choosing, people perceptually register how many cues discriminate between the options and compare that comparison load with a stable cognitive capacity. If the discriminating set is manageable, they integrate all discriminating cues with an equal-weight majority rule. If it exceeds capacity, they compress the comparison into a validity-ranked lexicographic search and rely on the first discriminating cue. Strategy recruitment changes smoothly near the capacity boundary, reflecting variability in momentary load, but is determined by an observable property of the option pair rather than by experiment identity. Ordinary response noise is applied only after a strategy has produced its intended choice.

**Parameters:**
- validities: validities
- capacity: [4.5, 5.5]
- gate_slope: [2.0, 5.0]
- beta: [1.55, 2.05]
- epsilon: [0.0, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Gated Comparison expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    discriminating = np.flatnonzero(differences != 0)
    load = float(discriminating.size)
    if load == 0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["capacity"])
    gate_slope = float(parameters["gate_slope"])

    # Probability that exhaustive equal-weight integration is recruited.
    # This stable logistic gate approaches one below capacity and zero above it.
    gate_logit = gate_slope * (capacity - load)
    if gate_logit >= 0.0:
        exp_neg = np.exp(-gate_logit)
        p_integrate = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(gate_logit)
        p_integrate = exp_pos / (1.0 + exp_pos)

    # Manageable-load strategy: all discriminating cues receive equal weight.
    tally_margin = float(np.sum(np.sign(differences)))
    if tally_margin > 0.0:
        tally_scores = np.array([1.0, 0.0])
    elif tally_margin < 0.0:
        tally_scores = np.array([0.0, 1.0])
    else:
        tally_scores = np.array([0.0, 0.0])

    # Overload strategy: validity-ranked compression to the first diagnostic cue.
    cue_order = np.argsort(-validities, kind="stable")
    lex_scores = np.array([0.0, 0.0])
    for cue in cue_order:
        if differences[cue] > 0.0:
            lex_scores = np.array([1.0, 0.0])
            break
        if differences[cue] < 0.0:
            lex_scores = np.array([0.0, 1.0])
            break

    beta = float(parameters["beta"])

    def noisy_strategy_choice(scores):
        logits = beta * np.asarray(scores, dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    # Response noise follows, rather than determines, strategy recruitment.
    p_tally = noisy_strategy_choice(tally_scores)
    p_lex = noisy_strategy_choice(lex_scores)
    p_core = p_integrate * p_tally + (1.0 - p_integrate) * p_lex

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    unanimous = []\n    late_loser = []\n\n    for row in data.itertuples(index=False):\n        a = np.asarray(row.option_a_ratings, dtype=float)\n        b = np.asarray(row.option_b_ratings, dtype=float)\n        signs = np.sign(a - b)\n        active = np.flatnonzero(signs != 0.0)\n        if active.size == 0:\n            continue\n\n        margin = float(np.sum(signs[active]))\n        if margin == 0.0:\n            continue\n\n        # Reorient every trial toward its equal-weight-majority winner, thereby\n        # combining A/B-reversed pairs and eliminating response-side direction.\n        winner_sign = 1.0 if margin > 0.0 else -1.0\n        relative_signs = signs[active] * winner_sign\n        winner_response = 0 if winner_sign > 0.0 else 1\n        followed_winner = float(int(row.response) == winner_response)\n\n        if np.all(relative_signs > 0.0):\n            unanimous.append(followed_winner)\n        elif np.sum(relative_signs < 0.0) == 1:\n            losing_active_position = int(active[np.flatnonzero(relative_signs < 0.0)[0]])\n            if losing_active_position == int(np.max(active)):\n                late_loser.append(followed_winner)\n\n    if len(unanimous) == 0 or len(late_loser) == 0:\n        return 0.0\n\n    return float(np.mean(unanimous) - np.mean(late_loser))",
  "rationale": "The metric contrasts majority-winner choice rates on unanimous trials with otherwise majority-consistent trials whose sole dissenting cue is the latest displayed active cue. Exact A/B reversals are reoriented toward the majority winner, balancing response side and reducing sensitivity to stable A/B response biases. The advocated theory predicts a large positive contrast: unanimous evidence is strong in every format, whereas a late dissenting cue can dominate recency-weighted sequential retrieval, particularly at load four. The competing theory predicts approximately zero because both its equal-weight tally and validity-ranked lexicographic strategy select the same majority winner in both conditions. Pooling all load-3 and load-4 instances supplies 32 observations per condition per subject, keeping the subject-level contrast comparatively stable."
}
```

## Usage

```json
{
  "input_tokens": 6739,
  "output_tokens": 2397
}
```
