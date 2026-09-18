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
**Validities (n_features=10):** [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.95, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0]

**Rationale:** This 10-cue design isolates the advocated theory's translation-invariant run-asymmetry mechanism in nearly balanced three-versus-two conflicts. In each of the first six trials, A is supported by three equal-validity cues and B by the fixed adjacent pair at Experts 7-8. A's coalition always begins at Expert 1 and always contains exactly one adjacent link, but its internal ordering alternates between an entry-compact form [1,1,g] and an exit-compact form [g,1], for gaps g=2, 3, and 4. Thus each matched pair has identical coalition sizes, onset positions, total validity, and conventional adjacency coherence. Switch-Closure Coalition Integration consequently predicts exactly the same A-choice probability across all six configurations: its unequal-coalition gate only represents onset primacy and the proportion of adjacent links, both of which are held constant. Reliability-Compressed Parsing Mixtures instead assigns positive centered run coherence to entry-compact coalitions and negative coherence to exit-compact coalitions. It therefore predicts higher P(A) for every entry-compact configuration than for its matched exit-compact configuration. Moreover, the entry-minus-exit contrast should increase from gap 2 to gap 4 because the compact-versus-fragmented distinction becomes more pronounced. This contrast has the same sign for every integration style, although the stable size-calibration split can shift subjects' overall preference for the three-cue coalition. The final six trials are exact A/B mirrors, so the corresponding prediction is higher endorsement of the three-cue coalition when it is entry-compact, while controlling response-side bias. Experts 9-10 provide the required validity spread but are tied; every discriminating expert has validity 0.70. Twelve unique pairs repeated eight times yield exactly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Reliability-Compressed Parsing Mixtures proposes that communicated cue validities are strongly compressed and all discriminating cues enter a diminishing-returns accumulator. Stable subject-level styles allocate attention among reliability, coalition size, and locally parsed cue groups. Display position has no universal primacy or recency gradient: configuration effects arise from chunk binding and weak interpretation completion. In nearly balanced three-versus-two conflicts, parsing is sensitive to the internal ordering of compact and fragmented portions of a coalition; this translation-invariant run-asymmetry can reverse associations with recency-weighted predictions without assigning greater weight to later cues themselves. Balanced two-versus-two conflicts retain sign-varying parsing effects that cancel at the population level. Completion in conflicts between two coalitions of at least three cues is population-common but weak, graded, and heterogeneous rather than deterministic. Signed size calibration creates stable divisions between subjects who treat an additional cue as corroboration and those who treat it as redundant.

**Parameters:**
- validities: validities
- integration_style: {0, 1, 2, 3, 4, 5, 6, 7, 8}
- validity_compression: [0.20, 0.42]
- coalition_saturation: [0.74, 0.86]
- size_calibration: [1.25, 1.65]
- parsing_strength: [0.12, 0.30]
- imbalanced_group_gain: [4.50, 6.50]
- completion_strength: [0.38, 0.50]
- beta: [0.95, 1.30]
- epsilon: [0.03, 0.09]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Compressed Parsing Mixtures expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    style = int(parameters["integration_style"])
    validity_compression = float(parameters["validity_compression"])
    coalition_saturation = float(parameters["coalition_saturation"])
    size_calibration = float(parameters["size_calibration"])
    parsing_strength = float(parameters["parsing_strength"])
    imbalanced_group_gain = float(parameters["imbalanced_group_gain"])
    completion_strength = float(parameters["completion_strength"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Reliability differences are represented but strongly compressed.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)

    # Stable styles differ in reliability attention, but even nominally
    # reliability-led subjects strongly compress instructed differences.
    if style <= 3:
        reliability_attention = 0.02 + 0.01 * style
    elif style <= 5:
        reliability_attention = 0.08 + 0.02 * (style - 4)
    else:
        reliability_attention = 0.04 + 0.015 * (style - 6)

    cue_weights = (
        (1.0 - reliability_attention) * np.ones(n_features, dtype=np.float64)
        + reliability_attention * diagnosticity
    )
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    pos_idx = np.flatnonzero(differences > 0.0)
    neg_idx = np.flatnonzero(differences < 0.0)
    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)

    if n_pos == 0 and n_neg == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def accumulated_evidence(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # All cues contribute, while total evidence grows sublinearly.
        return float(np.sum(cue_weights[indices])) / (float(n) ** coalition_saturation)

    evidence_a = accumulated_evidence(pos_idx)
    evidence_b = accumulated_evidence(neg_idx)

    # There is deliberately no fixed position gradient. A subject instead
    # parses the display into short local chunks whose width and phase are
    # stable properties of that subject's style.
    chunk_width = 2 + (style % 3)
    chunk_phase = style % chunk_width

    def parsing_quality(indices):
        n = int(indices.size)
        if n < 2:
            return 0.0
        idx = np.asarray(indices, dtype=int)
        adjacent = float(np.sum(np.diff(idx) == 1)) / float(n - 1)
        chunk_ids = np.floor_divide(idx + chunk_phase, chunk_width)
        same_chunk_pairs = 0.0
        total_pairs = float(n * (n - 1) // 2)
        for i in range(n):
            for j in range(i + 1, n):
                if chunk_ids[i] == chunk_ids[j]:
                    same_chunk_pairs += 1.0
        local_binding = same_chunk_pairs / max(total_pairs, 1.0)
        return 0.2 * adjacent + 0.8 * local_binding

    def centered_run_coherence(indices):
        # This descriptor depends only on within-coalition gap structure, not
        # absolute position. Positive values mean compact binding occurs near
        # the coalition's parsed entry; negative values mean it occurs near
        # its parsed exit. It is zero for singleton and two-cue coalitions.
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 3:
            return 0.0
        gaps = np.diff(idx).astype(np.float64)
        binding = np.exp(-(gaps - 1.0))
        centered_order = np.linspace(1.0, -1.0, binding.size)
        norm = max(float(np.sum(np.abs(centered_order))), 1.0)
        return float(np.dot(centered_order, binding) / norm)

    # Multi-cue conflicts receive a weak, signed structural calibration.
    # Styles 0--3 regard larger coalitions as corroborative; styles 4--8
    # regard excess members as increasingly redundant. This creates a stable
    # population split in 3-versus-2 conflicts without changing singleton
    # comparisons or discarding any cue from the base accumulator.
    if n_pos >= 2 and n_neg >= 2:
        size_orientation = 1.0 if style <= 3 else -1.0
        log_size_ratio = np.log(float(n_pos) / float(n_neg))
        size_shift = size_orientation * size_calibration * log_size_ratio

        parsing_valences = np.array(
            [-1.0, 0.75, -0.50, 1.0, -0.75, 0.50, -1.0, 0.75, 0.25],
            dtype=np.float64,
        )
        parse_difference = parsing_quality(pos_idx) - parsing_quality(neg_idx)

        # Nearly balanced 3-versus-2 configurations use centered run
        # coherence. This distinguishes where compact binding occurs within a
        # coalition without imposing an absolute early/late accessibility
        # gradient. Balanced conflicts retain the centered style valences.
        if min(n_pos, n_neg) == 2 and abs(n_pos - n_neg) == 1:
            group_multipliers = np.array(
                [0.80, 1.10, 0.90, 1.20, 0.75, 1.05, 0.85, 1.15, 1.00],
                dtype=np.float64,
            )
            run_difference = (
                centered_run_coherence(pos_idx)
                - centered_run_coherence(neg_idx)
            )
            parsing_shift = (
                parsing_strength
                * imbalanced_group_gain
                * float(group_multipliers[style])
                * run_difference
            )
        else:
            parsing_shift = (
                parsing_strength
                * float(parsing_valences[style])
                * parse_difference
            )

        log_gate_a = 0.5 * (size_shift + parsing_shift)
        log_gate_b = -0.5 * (size_shift + parsing_shift)
        evidence_a *= np.exp(np.clip(log_gate_a, -10.0, 10.0))
        evidence_b *= np.exp(np.clip(log_gate_b, -10.0, 10.0))

    net_a = evidence_a - evidence_b

    # Completion is weak and graded, is absent from singleton and 2-versus-2
    # conflicts, and never closes the accumulator. Stable differences in its
    # strength are smaller than the former opposing-sign split.
    if n_pos == n_neg and n_pos >= 3:
        completion_multipliers = np.array(
            [0.85, 0.925, 1.00, 1.075, 1.15, 0.95, 1.05, 0.90, 1.10],
            dtype=np.float64,
        )
        completion = completion_strength * float(completion_multipliers[style])
        if int(pos_idx[-1]) > int(neg_idx[-1]):
            net_a += completion
        elif int(neg_idx[-1]) > int(pos_idx[-1]):
            net_a -= completion

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    entry_sum = 0.0\n    entry_weight = 0.0\n    exit_sum = 0.0\n    exit_weight = 0.0\n\n    for row in data.itertuples(index=False):\n        a = np.asarray(row.option_a_ratings, dtype=float)\n        b = np.asarray(row.option_b_ratings, dtype=float)\n        diff = a - b\n        pos = np.flatnonzero(diff > 0)\n        neg = np.flatnonzero(diff < 0)\n\n        # Identify the three-cue coalition and whether it supports A or B.\n        if pos.size == 3 and neg.size == 2:\n            coalition = pos\n            endorsed_three = 1.0 if int(row.response) == 0 else 0.0\n        elif neg.size == 3 and pos.size == 2:\n            coalition = neg\n            endorsed_three = 1.0 if int(row.response) == 1 else 0.0\n        else:\n            continue\n\n        gaps = np.diff(coalition).astype(float)\n        if gaps.size != 2:\n            continue\n\n        first_compact = bool(gaps[0] == 1 and gaps[1] > 1)\n        last_compact = bool(gaps[1] == 1 and gaps[0] > 1)\n        if not (first_compact or last_compact):\n            continue\n\n        # Larger separation makes entry- versus exit-compact organization\n        # more distinct. This bounded weight is the difference between the\n        # two adjacent-link binding strengths.\n        weight = abs(np.exp(-(gaps[0] - 1.0)) - np.exp(-(gaps[1] - 1.0)))\n\n        if first_compact:\n            entry_sum += weight * endorsed_three\n            entry_weight += weight\n        else:\n            exit_sum += weight * endorsed_three\n            exit_weight += weight\n\n    if entry_weight <= 0.0 or exit_weight <= 0.0:\n        return 0.0\n\n    return float(entry_sum / entry_weight - exit_sum / exit_weight)\n",
  "rationale": "This metric is the weighted difference in endorsement of the three-cue coalition between entry-compact configurations ([adjacent cue, then gap]) and their exit-compact counterparts ([gap, then adjacent cue]). It recodes choices relative to the three-cue coalition, so the exact A/B mirrors cancel response-side bias. It also contrasts configurations within the same 3-versus-2 coalition sizes, cue validities, onset, and number of adjacent links, removing stable subject differences in overall preference for larger coalitions. Reliability-Compressed Parsing Mixtures predicts a positive contrast for every integration style, with a larger effect at wider gaps; the bounded gap weight emphasizes those more diagnostic trials while retaining all 96 observations. Switch-Closure Coalition Integration assigns matched entry- and exit-compact configurations the same structural gate because their onset and adjacency coherence are identical, predicting a contrast near zero."
}
```

## Usage

```json
{
  "input_tokens": 7074,
  "output_tokens": 2180
}
```
