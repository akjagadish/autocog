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
**Validities (n_features=35):** [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This 35-feature, 2-by-4 factorial design isolates the advocated theory's margin-selective diagnostic-overload compression. The validity profile contains seven mean-validity experts (0.75) and fourteen high/low pairs symmetric around 0.75. Base conditions cross A-minus-B count margins of 1, 3, 5, and 7 with sparse versus dense diagnostic fields. In sparse trials, only the required number of mean-validity experts favors A. In dense trials, 28 additional cues discriminate: seven complete high/low pairs favor A and seven complete high/low pairs favor B. These additions contribute zero net count and zero centered-validity residual. Consequently, within each count margin, Channel-Specific Coherence Integration predicts exactly identical sparse and dense choice probabilities: its count channel depends only on the signed margin, and its reliability signal is exactly zero. Resource-Adaptive Diagnostic Field Integration instead predicts a diagnostic-density penalty whose magnitude increases sharply with count margin. Dense trials contain 29, 31, 33, or 35 discriminating cues and therefore exceed its diagnostic span, but its powered margin activation makes overload compression relatively weak at margin 1 and progressively stronger at margins 3, 5, and 7. The critical signature is thus a density-by-margin interaction: little sparse-to-dense attenuation at margin 1 but a pronounced movement toward indifference at larger margins. The competing theory predicts a zero density effect at every margin, although it may show ordinary count saturation across margins. All eight base trials are also A/B reversed, balancing response side. Sixteen unique pairs repeated six times yield exactly 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Resource-Adaptive Diagnostic Field Integration theory: people establish a stable interpretation of whether marked ratings are adverse or beneficial, then allocate attention to currently discriminating evidence rather than total displayed feature count. Count and reliability information form a joint diagnostic field. Tally evidence accumulates smoothly with count margin, while reliability attention increases when discriminating cues expose a salient and internally organized reliability contrast. Accessibility is additionally calibrated by the clustering structure of the full communicated validity profile: coherent high-versus-low reliability blocks preserve attention, whereas uniformly graded profiles receive less reliability amplification. Only an unusually large set of simultaneously discriminating cues creates overload; under such overload, count evidence is divisively compressed, especially at larger margins. Opposing count and reliability signals usually inhibit one another through smooth conflict normalization. However, this normalization is locally weaker when the first count advantage appears, because a single additional mark is treated as diagnostic evidence rather than as a reason to globally discount the pre-existing reliability field. Stable subject-level integration style follows a softly polarized continuum between predominantly tally-oriented and predominantly reliability-sensitive policies. Subjects also differ stably in how deterministically they express their integrated evidence. This precision heterogeneity is mean-preserving and becomes less behaviorally expressed as diagnostic demand and overload increase, producing persistent individual differences in ordinary tasks without destabilizing dense displays or changing semantic polarity.

**Parameters:**
- validities: validities
- reliability_style: [0.0, 1.0]
- count_gain: [0.80, 1.24]
- count_scale: [3.2, 5.8]
- diagnostic_span: [20.0, 23.0]
- overload_softness: [0.6, 1.2]
- compression_margin_scale: [2.2, 3.4]
- margin_activation_power: [2.6, 3.4]
- overload_compression: [0.45, 0.80]
- baseline_reliability_attention: [0.25, 0.61]
- contrast_attention_gain: [0.90, 1.50]
- diffuse_reliability_compression: [0.01, 0.08]
- conflict_normalization: [0.05, 0.35]
- response_precision: [0.78, 1.30]
- precision_deviation: [-0.28, 0.28]
- adverse_polarity_confidence: [0.82, 1.0]
- lapse_rate: [0.0, 0.12]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Resource-Adaptive Diagnostic Field Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    difference = a - b
    discriminating = np.abs(difference) > 0.5
    n_discriminating = float(np.sum(discriminating))
    count_difference = float(np.sum(difference))
    margin = abs(count_difference)

    # A stable integration style creates a softly polarized continuum between
    # tally-oriented and reliability-sensitive subjects. It changes relative
    # allocation but does not create trial-specific strategy switches.
    raw_style = float(parameters["reliability_style"])
    style = 0.5 + 0.5 * np.tanh(2.5 * (2.0 * raw_style - 1.0)) / np.tanh(2.5)
    count_allocation = 1.25 - 0.50 * style
    reliability_allocation = 0.45 + 1.10 * style

    # Ordinary count margins accumulate smoothly and approximately linearly
    # near zero, with gradual bounded compression rather than margin-one
    # saturation.
    count_scale = float(parameters["count_scale"])
    base_count = count_scale * np.tanh(count_difference / count_scale)

    # Resource demand is determined by the number of cues that actually need
    # comparison. Non-discriminating display elements impose no direct load.
    # Softplus supplies a differentiable transition with no feature-count knee.
    span = float(parameters["diagnostic_span"])
    softness = float(parameters["overload_softness"])
    z = (n_discriminating - span) / softness
    overload = softness * (np.log1p(np.exp(-abs(z))) + max(z, 0.0))

    # Dense diagnostic fields selectively compress larger count margins. The
    # powered activation is negligible at margin one but rises smoothly for
    # margins around five and above.
    margin_scale = float(parameters["compression_margin_scale"])
    margin_power = float(parameters["margin_activation_power"])
    if margin > 0.0:
        margin_activation = margin ** margin_power / (
            margin ** margin_power + margin_scale ** margin_power
        )
    else:
        margin_activation = 0.0
    count_divisor = 1.0 + float(parameters["overload_compression"]) * overload * margin_activation
    count_signal = (
        float(parameters["count_gain"])
        * count_allocation
        * base_count
        / count_divisor
    )

    # Reliability evidence is residualized against the common tally
    # component. Its accessibility depends on diagnostic contrast and on how
    # coherently the currently discriminating cues organize that contrast.
    centered_validities = validities - float(np.mean(validities))
    reliability_residual = float(np.dot(centered_validities, difference))

    # Clustering is computed from the full communicated validity profile, not
    # from the active subset. A large gap relative to ordinary profile spacing
    # identifies a genuinely organized high-versus-low reliability structure.
    sorted_validities = np.sort(validities)
    profile_gaps = np.diff(sorted_validities)
    positive_gaps = profile_gaps[profile_gaps > 1e-9]
    if positive_gaps.size > 0:
        largest_gap = float(np.max(positive_gaps))
        typical_gap = float(np.median(positive_gaps))
        profile_clustering = np.clip(
            (largest_gap - typical_gap) / (largest_gap + typical_gap + 1e-9),
            0.0,
            1.0,
        )
    else:
        profile_clustering = 0.0

    if n_discriminating > 0.0:
        active_centered = centered_validities[discriminating]
        contrast = float(np.std(active_centered))
        profile_range = float(np.max(validities) - np.min(validities))
        contrast_salience = contrast / (contrast + 0.25 * profile_range + 1e-9)

        absolute_reliability = float(
            np.sum(np.abs(centered_validities[discriminating] * difference[discriminating]))
        )
        organization = abs(reliability_residual) / (absolute_reliability + 1e-9)
    else:
        contrast_salience = 0.0
        organization = 0.0

    # Smooth validity gradients receive less amplification, while clustered
    # profiles retain nearly all of the accepted model's diagnostic attention.
    profile_attention = 0.70 + 0.30 * profile_clustering
    diagnostic_attention = (
        float(parameters["baseline_reliability_attention"])
        + float(parameters["contrast_attention_gain"])
        * contrast_salience
        * organization
        * profile_attention
    )

    # Unorganized reliability detail can suffer under heavy diagnostic load,
    # while a coherent high-versus-low contrast remains available even in a
    # large display.
    reliability_divisor = 1.0 + (
        float(parameters["diffuse_reliability_compression"])
        * overload
        * (1.0 - organization)
    )
    reliability_signal = (
        reliability_allocation
        * diagnostic_attention
        * reliability_residual
        / reliability_divisor
    )

    # Count and reliability are jointly represented. When they recommend
    # opposite markedness directions, divisive cross-inhibition usually
    # reduces total confidence smoothly. A narrow notch around margin one
    # prevents the first count cue from producing excessive normalization.
    scale_sum = abs(count_signal) + abs(reliability_signal)
    if scale_sum > 1e-12 and count_signal * reliability_signal < 0.0:
        conflict = (
            4.0 * abs(count_signal * reliability_signal)
            / (scale_sum * scale_sum + 1e-12)
        )
    else:
        conflict = 0.0

    margin_one_notch = np.exp(-((margin - 1.0) / 0.55) ** 2)
    conflict *= 1.0 - 0.40 * margin_one_notch

    joint_divisor = 1.0 + float(parameters["conflict_normalization"]) * conflict

    # Stable, symmetric precision deviations increase ordinary-task
    # heterogeneity without shifting mean precision. Their expression is
    # attenuated when many cues must be compared and further attenuated by the
    # same overload signal that regulates dense diagnostic fields.
    precision_shrinkage = (
        1.0 + overload + n_discriminating / max(span, 1e-9)
    )
    effective_response_precision = (
        float(parameters["response_precision"])
        + float(parameters["precision_deviation"]) / precision_shrinkage
    )

    markedness_evidence = effective_response_precision * (
        count_signal + reliability_signal
    ) / joint_divisor

    # Positive markedness means A carries more adverse evidence and therefore
    # favors choosing B under the dominant adverse-mark interpretation.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    beneficial_probs = adverse_probs[::-1]
    polarity = float(parameters["adverse_polarity_confidence"])
    core_probs = polarity * adverse_probs + (1.0 - polarity) * beneficial_probs

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Channel-Specific Coherence Integration theory: binary counts and cue reliabilities are encoded by distinct cognitive channels before being combined. The count channel rapidly computes the difference between option-level numbers of marked ratings. It depends only on that signed margin, so adding mutually canceling cues does not dilute count evidence. Its response is smooth and bounded, reflecting saturation of an approximate magnitude representation rather than unlimited linear accumulation. The reliability channel instead binds each discriminating cue's identity to its communicated validity. It first removes the validity component redundant with the count margin and then measures the signed coherence of the remaining cue-validity contributions. A reliability residual supported by mutually consistent contributions receives high gain, whereas the same residual embedded in opposing contributions receives lower gain because maintaining the required cue-identity bindings is difficult. This loss is determined by evidence coherence and communicated validity-profile contrast, not by feature count, diagnostic-cue count, or a generic load threshold. Stable subject differences govern relative allocation to the two channels and response precision. A stable semantic-polarity belief determines whether marks are interpreted as adverse or beneficial, and a small lapse process captures stimulus-independent choices.

**Parameters:**
- validities: validities
- reliability_style: [0.0, 1.0]
- count_gain: [1.9, 3.0]
- count_scale: [2.8, 4.8]
- reliability_floor: [0.08, 0.30]
- coherent_binding_gain: [0.25, 0.78]
- coherence_power: [0.65, 1.65]
- profile_contrast_gain: [0.0, 0.45]
- response_precision: [0.68, 1.48]
- adverse_polarity_confidence: [0.80, 1.0]
- lapse_rate: [0.0, 0.12]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Channel-Specific Coherence Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    difference = a - b
    count_difference = float(np.sum(difference))

    # Stable integration style allocates expression strength between two
    # separately computed channels. The smooth transformation allows broad
    # subject heterogeneity without imposing discrete strategy classes.
    raw_style = float(parameters["reliability_style"])
    style = 0.5 + 0.5 * np.tanh(2.2 * (2.0 * raw_style - 1.0)) / np.tanh(2.2)
    count_allocation = 1.30 - 0.60 * style
    reliability_allocation = 0.35 + 1.35 * style

    # The count channel reads only the option-level mark difference. Hence
    # mutually canceling added cues have exactly no effect. Saturation is
    # gradual and smooth rather than a hard margin rule or load-dependent
    # divisive compression.
    count_scale = float(parameters["count_scale"])
    count_signal = (
        float(parameters["count_gain"])
        * count_allocation
        * np.tanh(count_difference / count_scale)
    )

    # Reliability is represented as evidence orthogonal to ordinary tallying.
    # Centering ensures that a common shift of all communicated validities
    # cannot masquerade as additional count evidence.
    centered_validities = validities - float(np.mean(validities))
    signed_components = centered_validities * difference
    reliability_residual = float(np.sum(signed_components))
    absolute_support = float(np.sum(np.abs(signed_components)))

    # Coherence compares the net residual with all reliability evidence that
    # had to be identity-bound. It is one for uniformly directed support and
    # approaches zero when a fixed residual is embedded in opposing groups.
    # It contains no explicit count-margin, display-size, or cue-count term.
    if absolute_support > 1e-12:
        coherence = abs(reliability_residual) / absolute_support
    else:
        coherence = 0.0
    coherence = float(np.clip(coherence, 0.0, 1.0))

    # Communicated profiles with meaningful reliability contrast make identity
    # bindings more distinguishable. This is a global property of the stated
    # profile, not a trial-level overload proxy.
    profile_mean = float(np.mean(validities))
    profile_spread = float(np.sqrt(np.mean((validities - profile_mean) ** 2)))
    profile_range = float(np.max(validities) - np.min(validities))
    if profile_range > 1e-12:
        profile_contrast = np.clip(2.0 * profile_spread / profile_range, 0.0, 1.0)
    else:
        profile_contrast = 0.0

    coherence_power = float(parameters["coherence_power"])
    coherent_access = coherence ** coherence_power
    profile_factor = (
        1.0
        + float(parameters["profile_contrast_gain"])
        * profile_contrast
    )

    # A nonzero floor permits reliability sensitivity even for diffuse fields,
    # while the binding term produces selective attenuation when opposing
    # reliability contributions proliferate.
    reliability_gain = (
        float(parameters["reliability_floor"])
        + float(parameters["coherent_binding_gain"])
        * coherent_access
        * profile_factor
    )
    reliability_signal = (
        reliability_allocation
        * reliability_gain
        * reliability_residual
    )

    # The independently summarized channels meet only after their respective
    # transformations. Precision is stable within subject and introduces
    # persistent heterogeneity in the expression of both summaries.
    markedness_evidence = float(parameters["response_precision"]) * (
        count_signal + reliability_signal
    )

    # Positive markedness indicates that A carries more adverse evidence and
    # therefore favors B under the dominant adverse-mark interpretation.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # Semantic polarity is stable across trials. The alternative beneficial
    # interpretation reverses the complete markedness representation and thus
    # preserves exact A/B reversal symmetry.
    beneficial_probs = adverse_probs[::-1]
    polarity = float(parameters["adverse_polarity_confidence"])
    core_probs = polarity * adverse_probs + (1.0 - polarity) * beneficial_probs

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs):
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    sparse = []\n    dense = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        difference = a - b\n        signed_margin = float(np.sum(difference))\n        margin = abs(signed_margin)\n        n_discriminating = int(np.sum(np.abs(difference) > 0.5))\n\n        # Focus on margins where overload compression should be pronounced.\n        if margin < 2.5 or margin > 7.5 or signed_margin == 0.0:\n            continue\n\n        response = int(row[\"response\"])\n        # One means choosing the option with fewer marked ratings, aligning\n        # A/B-reversed trials onto the dominant adverse-mark interpretation.\n        adverse_consistent = float(response if signed_margin > 0.0 else 1 - response)\n\n        if n_discriminating <= 7:\n            sparse.append(adverse_consistent)\n        elif n_discriminating >= 29:\n            dense.append(adverse_consistent)\n\n    if len(sparse) == 0 or len(dense) == 0:\n        return float(\"nan\")\n\n    return float(np.mean(sparse) - np.mean(dense))",
  "rationale": "This metric is the high-margin diagnostic-density attenuation effect. It aligns A/B reversals and compares adverse-consistent choice rates between sparse and dense trials at margins 3, 5, and 7. Resource-Adaptive Diagnostic Field Integration predicts substantially stronger adverse-consistent responding in sparse fields because dense fields overload and divisively compress large count margins. Channel-Specific Coherence Integration predicts the same choice probability in each matched sparse/dense pair, since the added cues have zero net count and zero reliability residual. Pooling three margins, both orientations, and all repetitions provides many observations per subject, reducing binomial and between-subject variance without adding the weakly diagnostic margin-1 condition."
}
```

## Usage

```json
{
  "input_tokens": 9845,
  "output_tokens": 1756
}
```
