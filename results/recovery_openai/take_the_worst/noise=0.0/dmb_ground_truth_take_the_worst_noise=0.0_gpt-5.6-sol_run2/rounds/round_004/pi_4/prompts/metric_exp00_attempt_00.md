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
**Validities (n_features=10):** [0.9, 0.84, 0.79, 0.74, 0.69, 0.64, 0.6, 0.57, 0.54, 0.52]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 8: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 11: A=[0, 1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
  trial 13: A=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
  trial 15: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Rationale:** Use 10 experts and hold all structural quantities used by Ecological Encoding Hysteresis constant across conditions. Every trial has exactly seven discriminating cues (density 0.70), a four-to-three majority, normalized majority margin 1/7, and Expert 1 as the highest-validity discriminator. In each base orientation Expert 1 favors A while four lower experts favor B, so ordered search favors A and equal-weight pooling favors B. The manipulation only permutes which lower-validity experts constitute the B-majority coalition. The four coalition conditions range from the four strongest lower experts supporting B to the four weakest experts supporting B; two common tie-value variants make distinct displays without changing any theoretical evidence. Ecological Encoding Hysteresis ignores lower-cue validity magnitudes after identifying the first discriminator and sign-compresses the tally. Because density, lexical direction, and pooled direction are identical on every trial, it predicts a common choice-probability trajectory across all coalition permutations, conditional on serial position, even though density 0.70 lies near its contextual transition. Contextual Margin Arbitration instead retains aggregate advertised-validity support. It therefore predicts a graded decline in B-majority choices as the B coalition moves from high- to low-validity experts: high-validity majority support increases compression accessibility and reinforces compressed B evidence, whereas low-validity majority support leaves strong residual opposition and strengthens the A-favoring rank route. The fixed, subthreshold margin of 1/7 prevents this gradient from being attributed to varying tally margins. Exact A/B reversals balance response side. Six repetitions of each of the 16 unique pairs yield 96 trials. The decisive analysis is the within-position coalition-validity slope: approximately zero under the advocated theory but substantial and monotonic under the competitor.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Ecological Encoding Hysteresis theory claims that decision makers learn how to encode an entire block of comparisons. They begin with a sparse-ecology prior favoring maintenance of the advertised validity hierarchy. After each comparison, they update a slowly changing estimate of the proportion of cues that discriminate. When discrimination remains sparse and ties are common, cue ranks remain cognitively useful and an ordered-search task set persists: the highest-validity discriminating cue controls choice, even on an occasional dense trial. When comparisons repeatedly contain many simultaneous discriminations, maintaining separate cue ranks becomes inefficient. People then compress cue directions into a relational gist and approximately pool them with equal weights. A steep contextual transition and a ceiling on compression implement hysteresis-like persistence rather than trial-by-trial load arbitration. Individual differences in the contextual threshold, prior strength, compression ceiling, sensitivity, and lapses produce heterogeneous behavior near the transition. Thus, identical diagnostic comparisons can elicit lexical choices in sparse filler blocks but majority choices in saturated filler blocks.

**Parameters:**
- n_features: n_features
- validities: validities
- context_prior: [0.50, 0.60]
- prior_strength: [1.0, 3.0]
- context_threshold: [0.68, 0.71]
- context_steepness: [35.0, 55.0]
- pooling_ceiling: [0.68, 0.76]
- evidence_sensitivity: [1.6, 3.0]
- lapse: [0.0, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Ecological Encoding Hysteresis expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Reconstruct the slowly accumulated block ecology. Each observation is
    # the fraction of cues that discriminate in a comparison; its complement
    # is tie prevalence. A prior prevents one unusual early trial from
    # immediately replacing the initial ordered-search task set.
    density_sum = 0.0
    n_observed = 0
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        density_sum += float(np.mean(a_i != b_i))
        n_observed += 1

    # Viewing the current display supplies another ecological observation,
    # but its effect shrinks as evidence about the block accumulates.
    current_density = float(np.mean(stim[0] != stim[1]))
    density_sum += current_density
    n_observed += 1

    prior_density = float(parameters["context_prior"])
    prior_strength = float(parameters["prior_strength"])
    context_density = (
        prior_strength * prior_density + density_sum
    ) / (prior_strength + float(n_observed))

    # Saturated ecologies trigger a compressed relational encoding. The
    # ceiling allows some ordered processing to survive even in dense blocks.
    threshold = float(parameters["context_threshold"])
    steepness = float(parameters["context_steepness"])
    gate_logit = steepness * (context_density - threshold)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    compression_activation = 1.0 / (1.0 + np.exp(-gate_logit))
    pooling_probability = (
        float(parameters["pooling_ceiling"]) * compression_activation
    )

    directions = np.sign(stim[0] - stim[1])

    # Ordered encoding: consult cues by advertised validity and stop at the
    # first discrimination.
    lexical_delta = 0.0
    cue_order = np.argsort(-validities, kind="stable")
    for cue in cue_order:
        if directions[cue] != 0.0:
            lexical_delta = float(directions[cue])
            break

    # Compressed encoding: retain all cue directions but discard fine-grained
    # rank and magnitude information. Sign normalization puts its response
    # scale on the same footing as the lexical representation.
    pooled_delta = float(np.sign(np.sum(directions)))

    sensitivity = float(parameters["evidence_sensitivity"])

    def choice_probs(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_ordered = choice_probs(lexical_delta)
    p_compressed = choice_probs(pooled_delta)
    p_core = (
        (1.0 - pooling_probability) * p_ordered
        + pooling_probability * p_compressed
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Contextual Margin Arbitration theory proposes that people maintain two competing representations of binary-feature comparisons. The advertised-rank route retrieves the most valid discriminating expert and favors its option. The conflict-compression route represents all discriminating experts as a directional tally while retaining the tally's graded normalized magnitude. Accessibility of compression is jointly controlled by a slowly persistent estimate of block density, the current normalized majority margin, a recency-weighted trace of recent margins, and the aggregate advertised-validity support for the current majority. Dense blocks make compression chronically accessible, whereas sparse blocks require a clearly suprathreshold majority to overcome rank processing. Density and margin provide compensatory routes to compression, so their overlap does not produce a superadditive accessibility boost. Aggregate validity shifts route accessibility and evidence continuously. A smoothly saturating diagnosticity representation preserves unusually extreme advertised-validity dominance without turning it into a categorical veto: dense context strongly reduces validity opposition, whereas margin-driven release is weaker and leaves a residual that grows continuously with the extremity of anti-majority support. Compression uses a mildly convex transform of normalized tally magnitude, making conspicuously large majorities more persuasive than small anti-leader coalitions while retaining continuous evidence. Repetition of particular cue identities has no independent learning effect because choices receive no correctness feedback. Stable individual differences in contextual thresholds, margin thresholds, gating sensitivities, evidence precision, and lapses generate heterogeneous intermediate behavior.

**Parameters:**
- n_features: n_features
- validities: validities
- validity_saturation: [5.0, 7.0]
- context_memory_span: [14.0, 26.0]
- density_prior: [0.48, 0.58]
- margin_prior: [0.10, 0.22]
- density_threshold: [0.68, 0.71]
- margin_threshold: [0.24, 0.28]
- margin_smoothness: [45.0, 70.0]
- base_compression_access: [-2.8, -2.0]
- density_sensitivity: [30.0, 40.0]
- margin_sensitivity: [30.0, 42.0]
- recent_margin_sensitivity: [0.5, 1.6]
- density_margin_interaction: [0.75, 1.0]
- validity_access_bias: [2.0, 4.0]
- density_validity_release: [10.0, 16.0]
- margin_validity_release: [1.0, 3.0]
- validity_release_floor: [0.20, 0.40]
- validity_evidence_gain: [0.15, 0.65]
- tally_margin_exponent: [1.05, 1.15]
- tally_evidence_scale: [4.8, 5.4]
- rank_sensitivity: [1.8, 3.2]
- tally_sensitivity: [0.70, 0.90]
- lapse: [0.0, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Contextual Margin Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float).reshape(-1)
    if validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Diagnosticity weights express advertised reliability. Smooth saturation
    # preserves the special extremity of a near-perfect expert while ensuring
    # that no finite advertised validity becomes a logical veto.
    v = np.clip(validities, 0.500001, 0.999999)
    raw_diagnosticity = np.log(v / (1.0 - v))
    validity_saturation = float(parameters["validity_saturation"])
    diagnosticity = validity_saturation * np.tanh(
        raw_diagnosticity / validity_saturation
    )

    memory_span = float(parameters["context_memory_span"])
    alpha = 1.0 / max(memory_span, 1.0)
    density_trace = float(parameters["density_prior"])
    margin_trace = float(parameters["margin_prior"])

    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    # Density and margin traces are learned from displays alone. Cue identities
    # and previous responses do not enter because there is no outcome feedback.
    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        d_i = np.sign(a_i - b_i)
        k_i = int(np.count_nonzero(d_i))
        density_i = float(k_i) / float(max(n_features, 1))
        margin_i = (
            abs(float(np.sum(d_i))) / float(k_i) if k_i > 0 else 0.0
        )
        density_trace += alpha * (density_i - density_trace)
        margin_trace += alpha * (margin_i - margin_trace)

    directions = np.sign(stim[0] - stim[1])
    discriminating = np.flatnonzero(directions != 0.0)
    n_discriminating = int(discriminating.size)
    current_density = float(n_discriminating) / float(max(n_features, 1))
    raw_tally = float(np.sum(directions))
    current_margin = (
        abs(raw_tally) / float(n_discriminating)
        if n_discriminating > 0 else 0.0
    )

    # The current display is incorporated weakly into the persistent context.
    context_density = density_trace + alpha * (current_density - density_trace)
    recent_margin = margin_trace + alpha * (current_margin - margin_trace)

    # Advertised-rank evidence comes from the most valid discriminating cue.
    lexical_delta = 0.0
    if n_discriminating > 0:
        cue_order = np.argsort(-validities, kind="stable")
        for cue in cue_order:
            if directions[cue] != 0.0:
                lexical_delta = float(directions[cue])
                break

    # Signed aggregate validity support lies in [-1, 1]. Positive values favor
    # A and negative values favor B. Smoothly saturated weights retain graded
    # information about exceptionally dominant advertised experts.
    if n_discriminating > 0:
        active_weights = diagnosticity[discriminating]
        denom = float(np.sum(active_weights))
        if denom > 1e-12:
            validity_direction = float(
                np.dot(active_weights, directions[discriminating]) / denom
            )
        else:
            validity_direction = 0.0
    else:
        validity_direction = 0.0

    tally_sign = float(np.sign(raw_tally))
    validity_for_majority = tally_sign * validity_direction

    # Density remains an asymmetric contextual bonus. A sharper, individually
    # varying soft hinge makes margins below threshold contribute negligibly.
    density_excess = max(
        context_density - float(parameters["density_threshold"]), 0.0
    )
    margin_offset = current_margin - float(parameters["margin_threshold"])
    recent_offset = recent_margin - float(parameters["margin_threshold"])
    smoothness = float(parameters["margin_smoothness"])
    margin_excess = float(np.logaddexp(0.0, smoothness * margin_offset) / smoothness)
    recent_excess = float(np.logaddexp(0.0, smoothness * recent_offset) / smoothness)

    # Opposing advertised-validity support is released strongly by a genuinely
    # dense context but only weakly by momentary margin salience. A nonzero
    # attenuation floor preserves graded residual validity opposition without
    # allowing it to become a categorical veto.
    positive_validity = max(validity_for_majority, 0.0)
    negative_validity = min(validity_for_majority, 0.0)
    release_floor = float(parameters["validity_release_floor"])
    release_exponent = (
        float(parameters["density_validity_release"]) * density_excess
        + float(parameters["margin_validity_release"]) * margin_excess
    )
    negative_attenuation = release_floor + (1.0 - release_floor) * np.exp(
        -release_exponent
    )
    effective_validity_access = (
        positive_validity + negative_validity * negative_attenuation
    )

    # Density and current margin are compensatory routes to compression. The
    # overlap correction makes their combination approximately max/noisy-OR
    # like instead of granting a superadditive bonus when both are salient.
    density_drive = (
        float(parameters["density_sensitivity"]) * density_excess
    )
    margin_drive = (
        float(parameters["margin_sensitivity"]) * margin_excess
    )
    overlap_correction = (
        float(parameters["density_margin_interaction"])
        * min(density_drive, margin_drive)
    )

    gate_logit = (
        float(parameters["base_compression_access"])
        + density_drive
        + margin_drive
        - overlap_correction
        + float(parameters["recent_margin_sensitivity"]) * recent_excess
        + float(parameters["validity_access_bias"]) * effective_validity_access
    )
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    compression_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    validity_evidence_gain = float(parameters["validity_evidence_gain"])
    rank_delta = lexical_delta + validity_evidence_gain * validity_direction

    # Carry the same salience-dependent release into compressed evidence. Once
    # a majority representation is active, opposing validity remains a graded
    # residual bias instead of re-entering as a full anti-majority safeguard.
    if tally_sign != 0.0:
        compressed_validity_direction = (
            tally_sign * effective_validity_access
        )
    else:
        compressed_validity_direction = validity_direction

    # Preserve continuous normalized tally magnitude. Mild convexity separates
    # conspicuously large majorities from small anti-leader margins without
    # introducing dependence on the experiment's number of active experts.
    transformed_margin = current_margin ** float(
        parameters["tally_margin_exponent"]
    )
    compressed_delta = (
        tally_sign
        * transformed_margin
        * float(parameters["tally_evidence_scale"])
        + validity_evidence_gain * compressed_validity_direction
    )

    rank_sensitivity = float(parameters["rank_sensitivity"])
    tally_sensitivity = float(parameters["tally_sensitivity"])

    def binary_probs(delta, sensitivity):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_rank = binary_probs(rank_delta, rank_sensitivity)
    p_compressed = binary_probs(compressed_delta, tally_sensitivity)
    p_core = (
        (1.0 - compression_probability) * p_rank
        + compression_probability * p_compressed
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    validities = np.array([0.90, 0.84, 0.79, 0.74, 0.69, 0.64, 0.60, 0.57, 0.54, 0.52], dtype=float)\n    diagnosticity = np.log(validities / (1.0 - validities))\n\n    def subject_slope(df):\n        df = df.reset_index(drop=True)\n        n = len(df)\n        if n < 4:\n            return np.nan\n\n        coalition_scores = np.empty(n, dtype=float)\n        majority_b = np.empty(n, dtype=float)\n        majority_choice = np.empty(n, dtype=float)\n\n        for i in range(n):\n            a = np.asarray(df.loc[i, \"option_a_ratings\"], dtype=float)\n            b = np.asarray(df.loc[i, \"option_b_ratings\"], dtype=float)\n            directions = np.sign(a - b)\n            active = directions != 0.0\n            tally = float(np.sum(directions))\n            majority_sign = float(np.sign(tally))\n\n            if majority_sign == 0.0 or not np.any(active):\n                coalition_scores[i] = 0.0\n                majority_b[i] = 0.5\n                majority_choice[i] = 0.0\n                continue\n\n            denom = float(np.sum(diagnosticity[active]))\n            coalition_scores[i] = majority_sign * float(\n                np.dot(diagnosticity[active], directions[active])\n            ) / denom\n\n            maj_b = 1.0 if majority_sign < 0.0 else 0.0\n            majority_b[i] = maj_b\n            majority_choice[i] = 1.0 if float(df.loc[i, \"response\"]) == maj_b else 0.0\n\n        rounded = np.round(coalition_scores, 10)\n        levels = np.sort(np.unique(rounded))\n        if levels.size < 2:\n            return np.nan\n\n        rank_map = {float(v): float(j) for j, v in enumerate(levels)}\n        x = np.array([rank_map[float(v)] for v in rounded], dtype=float)\n        x -= (levels.size - 1.0) / 2.0\n\n        # Remove assignment to short serial-position strata, as well as any\n        # incidental association with which screen side contains the majority.\n        bins = np.arange(n, dtype=int) // 6\n        unique_bins = np.unique(bins)\n        design = np.zeros((n, unique_bins.size + 1), dtype=float)\n        for j, value in enumerate(unique_bins):\n            design[:, j] = (bins == value).astype(float)\n        design[:, -1] = majority_b\n\n        beta = np.linalg.lstsq(design, x, rcond=None)[0]\n        x_resid = x - design.dot(beta)\n        denom = float(np.dot(x_resid, x_resid))\n        if denom <= 1e-12:\n            return np.nan\n\n        return float(np.dot(x_resid, majority_choice) / denom)\n\n    if \"subject_id\" in data.columns:\n        values = []\n        for _, subject_df in data.groupby(\"subject_id\", sort=False):\n            value = subject_slope(subject_df)\n            if np.isfinite(value):\n                values.append(value)\n        return float(np.mean(values)) if values else np.nan\n\n    return float(subject_slope(data))\n",
  "rationale": "This is the within-position slope of choosing the four-cue majority on the advertised-validity strength of that majority coalition. Coalition strength is computed from log-odds diagnosticity and converted to its four ordinal levels, avoiding excessive leverage from the strongest expert. Ecological Encoding Hysteresis discards lower-cue validity magnitudes once lexical and tally directions are fixed, so it predicts a slope near zero. Contextual Margin Arbitration retains aggregate validity support in both route accessibility and evidence, predicting a positive slope. Residualizing the coalition rank within six-trial serial strata removes the advocated theory's potentially steep context trajectory and substantially reduces randomization-induced between-subject variance. Residualization against majority side also removes incidental response-side imbalance. The pooled result is the mean subject slope, making the same estimand valid for pooled and single-subject inputs."
}
```

## Usage

```json
{
  "input_tokens": 7343,
  "output_tokens": 4646
}
```
