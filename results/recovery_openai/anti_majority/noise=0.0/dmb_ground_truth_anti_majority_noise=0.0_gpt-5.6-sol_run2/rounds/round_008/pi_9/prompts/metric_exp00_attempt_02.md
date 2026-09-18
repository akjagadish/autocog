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
**Validities (n_features=24):** [0.8, 0.7, 0.8, 0.7, 0.85, 0.65, 0.85, 0.65, 0.9, 0.6, 0.9, 0.6, 0.95, 0.55, 0.95, 0.55, 0.99, 0.51, 0.99, 0.51, 0.99, 0.51, 0.99, 0.51]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
  trial 11: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
  trial 12: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 13: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 23: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  trial 24: A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Rationale:** This 24-feature design isolates magnitude-dependent reliability access and the competitor's intermediate-magnitude calibration notch while eliminating count evidence and reliability-coherence changes. The validity profile consists of 12 high/low pairs symmetric around 0.75. In each base condition, an increasing prefix of pairs discriminates: the higher-validity member marks A and the lower-validity member marks B. Consequently, A and B always have equal mark counts, every nonzero centered-validity contribution has the same positive sign, and reliability coherence is exactly 1. The centered-validity residual increases through 0.10, 0.20, 0.40, 0.60, 0.90, 1.20, 1.60, 2.00, 2.48, 2.96, 3.44, and 3.92. Channel-Specific Coherence Integration therefore predicts an exactly proportional latent reliability signal: for each subject, reliability_signal/R is constant across all 12 conditions, with only the common response mapping producing bounded observable probabilities. Relative-Evidence Conflict Arbitration necessarily violates that proportionality. Its magnitude-access term strongly increases reliability_signal/R between small and moderate residuals, its notch selectively discounts residuals around 1.6–2.2, and access recovers beyond that band. It thus predicts a curved profile with accelerating per-unit sensitivity at low magnitudes, localized attenuation around the intermediate conditions, and recovery at high magnitudes. The critical analysis fits each theory's common polarity, lapse, precision, and bounded response mapping and tests the residual-magnitude profile: a single linear latent trajectory supports the advocated theory, whereas the competitor predicts magnitude-specific curvature even though count margin and coherence never change. Every base pair is also A/B reversed. The 24 unique pairs are each repeated four times, yielding exactly 96 trials.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Relative-Evidence Conflict Arbitration theory: people construct two density-invariant summaries of a binary-feature display. A signed count accumulator represents only the difference in the numbers of marks carried by the options. It grows approximately linearly across ordinary margins and saturates only at relatively large margins, so adding equal numbers of opposing marks cannot dilute count evidence. In parallel, communicated validities are centered around their profile mean and bound to cue directions. Access to this reliability residual is governed by its absolute magnitude and signed coherence: consistently directed reliability contributions become accessible, whereas additions containing genuinely opposing contributions reduce access by increasing gross support without increasing the residual. Accessibility never depends directly on display size, the number of discriminating cues, or active-validity dispersion. A shallow, band-limited calibration discounts intermediate reliability magnitudes but recovers for genuinely strong residuals, representing temporary ambiguity in mapping moderately accumulated reliability evidence onto a categorical preference rather than generic overload. The two summaries are combined by relative-evidence arbitration. Congruent summaries receive modest mutual reinforcement. During conflict, balanced channels inhibit confidence, while sufficiently asymmetric conflicts recruit a winner-enhancement process favoring whichever signed summary is stronger. Thus pure count problems show robust margin growth, but structured count-reliability conflicts can be compressed or reversed without invoking density overload. After arbitration, a mild bounded-confidence transformation imposes diminishing decisional returns on repeatedly accumulated coherent evidence without altering the raw channel ratio used to resolve conflicts. Stable but moderate differences in channel balance and response precision coexist with stable semantic polarity and occasional lapses.

**Parameters:**
- validities: validities
- channel_balance: [0.0, 1.0]
- count_gain: [0.92, 1.18]
- count_scale: [7.0, 11.5]
- reliability_floor: [0.14, 0.32]
- coherence_gain: [0.66, 1.00]
- coherence_power: [1.00, 1.70]
- magnitude_half_saturation: [0.08, 0.28]
- reliability_gain: [0.72, 1.16]
- reliability_notch_center: [1.6, 2.2]
- reliability_notch_width: [0.45, 0.80]
- reliability_notch_depth: [0.12, 0.26]
- congruent_synergy: [0.0, 0.14]
- conflict_compression: [0.35, 0.95]
- winner_enhancement: [0.18, 0.62]
- arbitration_curvature: [0.65, 1.35]
- decision_bound: [2.8, 4.8]
- response_precision: [0.82, 1.22]
- adverse_polarity_confidence: [0.84, 0.98]
- lapse_rate: [0.0, 0.10]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Relative-Evidence Conflict Arbitration expects shape "
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

    # Stable channel balance produces moderate individual differences without
    # creating discrete strategy classes or trial-dependent attention shifts.
    raw_balance = float(parameters["channel_balance"])
    centered_balance = np.tanh(1.6 * (2.0 * raw_balance - 1.0)) / np.tanh(1.6)
    count_allocation = 1.0 - 0.14 * centered_balance
    reliability_allocation = 1.0 + 0.22 * centered_balance

    # The count accumulator reads only the signed mark margin. Its large scale
    # gives approximately linear growth through ordinary margins and delays
    # saturation. Equal and opposite added marks leave it exactly unchanged.
    count_scale = float(parameters["count_scale"])
    count_signal = (
        count_allocation
        * float(parameters["count_gain"])
        * count_scale
        * np.tanh(count_difference / count_scale)
    )

    # Reliability information is the component of validity-weighted evidence
    # not already represented by an undifferentiated count. Centering also
    # prevents the common validity level from acting as a second tally.
    centered_validities = validities - float(np.mean(validities))
    signed_components = centered_validities * difference
    reliability_residual = float(np.sum(signed_components))
    gross_reliability_support = float(np.sum(np.abs(signed_components)))

    # Coherence is determined solely by the geometry of signed reliability
    # contributions. Consistent contributions have coherence one; opposing
    # additions increase gross support and reduce coherence even if the net
    # residual is unchanged. No cue-count or display-density term appears.
    if gross_reliability_support > 1e-12:
        coherence = abs(reliability_residual) / gross_reliability_support
    else:
        coherence = 0.0
    coherence = float(np.clip(coherence, 0.0, 1.0))

    residual_magnitude = abs(reliability_residual)
    half_saturation = float(parameters["magnitude_half_saturation"])
    magnitude_access = residual_magnitude / (
        residual_magnitude + half_saturation + 1e-12
    )

    accessibility = (
        float(parameters["reliability_floor"])
        + float(parameters["coherence_gain"])
        * (coherence ** float(parameters["coherence_power"]))
        * magnitude_access
    )

    # Intermediate reliability magnitudes receive a shallow ambiguity cost,
    # but accessibility recovers for genuinely strong residuals. This smooth
    # notch depends only on signed reliability magnitude, not display density.
    notch_center = float(parameters["reliability_notch_center"])
    notch_width = float(parameters["reliability_notch_width"])
    notch = np.exp(
        -0.5 * ((residual_magnitude - notch_center) / notch_width) ** 2
    )
    magnitude_calibration = (
        1.0 - float(parameters["reliability_notch_depth"]) * notch
    )

    reliability_signal = (
        reliability_allocation
        * float(parameters["reliability_gain"])
        * accessibility
        * magnitude_calibration
        * reliability_residual
    )

    c_abs = abs(count_signal)
    r_abs = abs(reliability_signal)
    total_abs = c_abs + r_abs

    if total_abs <= 1e-12:
        integrated_signal = 0.0
    elif count_signal * reliability_signal > 0.0:
        # Agreement is mildly superadditive but remains a function of relative
        # signed support, not the number of displayed or active features.
        agreement_balance = 4.0 * c_abs * r_abs / (total_abs * total_abs + 1e-12)
        integrated_signal = (count_signal + reliability_signal) * (
            1.0
            + float(parameters["congruent_synergy"]) * agreement_balance
        )
    elif count_signal * reliability_signal < 0.0:
        # Balanced conflict compresses confidence. Asymmetric conflict recruits
        # winner enhancement, permitting either robust count choices or
        # reliability-driven reversals according to relative evidence alone.
        conflict_balance = 4.0 * c_abs * r_abs / (
            total_abs * total_abs + 1e-12
        )
        relative_dominance = abs(c_abs - r_abs) / (total_abs + 1e-12)
        dominance_gate = relative_dominance ** float(
            parameters["arbitration_curvature"]
        )

        compressed_sum = (count_signal + reliability_signal) / (
            1.0
            + float(parameters["conflict_compression"])
            * conflict_balance
        )
        winner_signal = count_signal if c_abs >= r_abs else reliability_signal
        winner_bonus = (
            float(parameters["winner_enhancement"])
            * conflict_balance
            * dominance_gate
            * winner_signal
        )
        integrated_signal = compressed_sum + winner_bonus
    else:
        integrated_signal = count_signal + reliability_signal

    # Arbitration uses the unbounded channel summaries above, preserving the
    # raw count-to-reliability ratio in structured conflicts. Only the final
    # decisional representation has diminishing returns, so repeated coherent
    # contributions cannot increase confidence without limit.
    decision_bound = float(parameters["decision_bound"])
    integrated_signal = decision_bound * np.tanh(
        integrated_signal / decision_bound
    )

    markedness_evidence = (
        float(parameters["response_precision"]) * integrated_signal
    )

    # Positive markedness means A carries the greater adverse burden and thus
    # favors B. Semantic polarity is fixed across the entire subject run.
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
[0] rationale: This metric measures the prevalence of weak responding when the coherent reliability residual is already large. Responses are first aligned across each A/B-reversed pair, so positive values always represent selection of the option favored by the dominant adverse-mark interpretation. It then averages aligned choices over the six largest-residual conditions and applies a bounded soft threshold centered at an aligned preference of 0.45. Channel-Specific Coherence Integration permits a substantial lower tail of subjects for whom low reliability-channel allocation, binding gain, and precision jointly leave even strong coherent residuals weakly expressed. Relative-Evidence Conflict Arbitration has narrower channel allocation and increasingly complete magnitude access at these residuals, so it predicts fewer such weak responders despite its intermediate notch. Using 48 trials per subject stabilizes the tail estimate, while the smooth threshold avoids the variance of a hard subject classification. The pooled implementation averages the same subject-level score used by the between-subject evaluation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        aligned = []
        supports = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            difference = a - b

            # Every designed pair contains the first, highest-validity cue.
            # Its direction identifies whether this is the base or A/B-reversed
            # version without relying on trial order.
            direction = np.sign(difference[0])
            if direction == 0:
                nz = np.flatnonzero(difference)
                if nz.size == 0:
                    continue
                direction = np.sign(difference[nz[0]])

            x = direction * (2.0 * float(row["response"]) - 1.0)
            aligned.append(x)
            supports.append(float(np.sum(np.abs(difference))))

        if len(aligned) == 0:
            return 0.5

        aligned = np.asarray(aligned, dtype=float)
        supports = np.asarray(supports, dtype=float)

        # The six largest-residual conditions have 14--24 discriminating cues.
        # They provide 48 observations per subject and sharply distinguish the
        # advocated theory's low-reliability-allocation tail from the competing
        # theory's more uniformly accessible strong residuals.
        use = supports >= 14.0
        if not np.any(use):
            use = np.ones(len(aligned), dtype=bool)

        high_residual_preference = float(np.mean(aligned[use]))

        # Smooth, bounded lower-tail indicator. Values near one indicate that
        # even strong coherent reliability residuals produce weak preferences.
        z = np.clip((high_residual_preference - 0.45) / 0.08, -40.0, 40.0)
        return float(1.0 / (1.0 + np.exp(z)))

    if "subject_id" in data.columns:
        scores = [subject_score(g) for _, g in data.groupby("subject_id", sort=False)]
        if len(scores) == 0:
            return float("nan")
        return float(np.mean(scores))

    return subject_score(data)

outcome: self_sim=0.3515 (var=0.1092) adversary_sim=0.2657 (var=0.0551) welch_t=+1.058 p=0.2958 (N=25, alpha=0.01) -> reject

[1] rationale: This is a preregistered generalized least-squares contrast of the complete 12-point reliability-magnitude profile. Choices are aligned across A/B reversals, producing one signed preference estimate from all eight observations at each residual magnitude. The contrast weights are derived deterministically from the stated parameter ranges: they emphasize the competitor's low-magnitude access curvature, intermediate calibration notch, and high-magnitude recovery, while downweighting profile directions that are noisy because of stable subject-level gain, polarity, precision, saturation, or finite trial sampling. Thus it is qualitatively different from the failed high-residual tail score, which discarded half the trials and was dominated by between-subject strength differences. The resulting scalar is not an identity or a parameter-recovery criterion: it is an empirical weighted shape contrast that can favor either theory on human data. Centering at the two predicted mean profiles makes positive values support Channel-Specific Coherence Integration and negative values support Relative-Evidence Conflict Arbitration.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Reliability residuals associated with the 12 prefix conditions.
    residuals = np.array(
        [0.10, 0.20, 0.40, 0.60, 0.90, 1.20,
         1.60, 2.00, 2.48, 2.96, 3.44, 3.92],
        dtype=float
    )
    n_cond = residuals.size

    # Construct a deterministic generalized least-squares contrast. The
    # contrast emphasizes profile locations where the theories differ while
    # downweighting directions dominated by stable parameter heterogeneity or
    # by binomial response noise.
    rng = np.random.RandomState(73129)
    n_sim = 16000

    def uniform(lo, hi):
        return lo + (hi - lo) * rng.random_sample(n_sim)

    # Predicted subject-level aligned response profiles under
    # Channel-Specific Coherence Integration.
    raw_style = uniform(0.0, 1.0)
    style = (
        0.5
        + 0.5
        * np.tanh(2.2 * (2.0 * raw_style - 1.0))
        / np.tanh(2.2)
    )
    rel_allocation_s = 0.35 + 1.35 * style
    rel_floor_s = uniform(0.08, 0.30)
    binding_gain_s = uniform(0.25, 0.78)
    profile_gain_s = uniform(0.0, 0.45)
    precision_s = uniform(0.68, 1.48)
    polarity_s = uniform(0.80, 1.0)
    lapse_s = uniform(0.0, 0.12)

    validities = np.array(
        [0.8, 0.7, 0.8, 0.7, 0.85, 0.65, 0.85, 0.65,
         0.9, 0.6, 0.9, 0.6, 0.95, 0.55, 0.95, 0.55,
         0.99, 0.51, 0.99, 0.51, 0.99, 0.51, 0.99, 0.51],
        dtype=float
    )
    spread = float(np.sqrt(np.mean((validities - np.mean(validities)) ** 2)))
    value_range = float(np.max(validities) - np.min(validities))
    profile_contrast = 2.0 * spread / value_range

    gain_s = (
        rel_floor_s
        + binding_gain_s * (1.0 + profile_gain_s * profile_contrast)
    )
    evidence_s = (
        precision_s[:, None]
        * rel_allocation_s[:, None]
        * gain_s[:, None]
        * residuals[None, :]
    )
    mean_y_s = (
        (1.0 - lapse_s)[:, None]
        * (2.0 * polarity_s - 1.0)[:, None]
        * np.tanh(0.5 * evidence_s)
    )

    # Predicted profiles under Relative-Evidence Conflict Arbitration. Count
    # evidence and conflict arbitration vanish in this equal-count design, but
    # magnitude access, the calibration notch, and the final decision bound
    # remain operative.
    raw_balance = uniform(0.0, 1.0)
    centered_balance = (
        np.tanh(1.6 * (2.0 * raw_balance - 1.0)) / np.tanh(1.6)
    )
    rel_allocation_a = 1.0 + 0.22 * centered_balance
    rel_floor_a = uniform(0.14, 0.32)
    coherence_gain_a = uniform(0.66, 1.00)
    half_sat_a = uniform(0.08, 0.28)
    rel_gain_a = uniform(0.72, 1.16)
    notch_center_a = uniform(1.6, 2.2)
    notch_width_a = uniform(0.45, 0.80)
    notch_depth_a = uniform(0.12, 0.26)
    decision_bound_a = uniform(2.8, 4.8)
    precision_a = uniform(0.82, 1.22)
    polarity_a = uniform(0.84, 0.98)
    lapse_a = uniform(0.0, 0.10)

    r = residuals[None, :]
    access_a = (
        rel_floor_a[:, None]
        + coherence_gain_a[:, None]
        * r / (r + half_sat_a[:, None] + 1e-12)
    )
    notch_a = np.exp(
        -0.5
        * ((r - notch_center_a[:, None]) / notch_width_a[:, None]) ** 2
    )
    calibration_a = 1.0 - notch_depth_a[:, None] * notch_a
    rel_signal_a = (
        rel_allocation_a[:, None]
        * rel_gain_a[:, None]
        * access_a
        * calibration_a
        * r
    )
    bounded_a = (
        decision_bound_a[:, None]
        * np.tanh(rel_signal_a / decision_bound_a[:, None])
    )
    evidence_a = precision_a[:, None] * bounded_a
    mean_y_a = (
        (1.0 - lapse_a)[:, None]
        * (2.0 * polarity_a - 1.0)[:, None]
        * np.tanh(0.5 * evidence_a)
    )

    mu_s = np.mean(mean_y_s, axis=0)
    mu_a = np.mean(mean_y_a, axis=0)
    delta = mu_s - mu_a

    cov_s = np.cov(mean_y_s, rowvar=False, bias=True)
    cov_a = np.cov(mean_y_a, rowvar=False, bias=True)

    # Each residual condition has four presentations in each orientation,
    # hence eight aligned Bernoulli observations per subject. Add their
    # expected conditional sampling variance to the parameter covariance.
    noise_s = np.mean(1.0 - mean_y_s ** 2, axis=0) / 8.0
    noise_a = np.mean(1.0 - mean_y_a ** 2, axis=0) / 8.0
    cov_s = cov_s + np.diag(noise_s)
    cov_a = cov_a + np.diag(noise_a)

    pooled_cov = cov_s + cov_a + 1e-6 * np.eye(n_cond)
    weights = np.linalg.solve(pooled_cov, delta)
    scale = float(np.sqrt(weights @ pooled_cov @ weights))
    if np.isfinite(scale) and scale > 1e-12:
        weights = weights / scale

    midpoint = 0.5 * (mu_s + mu_a)

    def subject_score(df):
        sums = np.zeros(n_cond, dtype=float)
        counts = np.zeros(n_cond, dtype=float)

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            difference = a - b
            support = int(np.rint(np.sum(np.abs(difference))))
            condition = support // 2 - 1
            if condition < 0 or condition >= n_cond:
                continue

            direction = float(np.sign(difference[0]))
            if direction == 0.0:
                nonzero = np.flatnonzero(difference)
                if nonzero.size == 0:
                    continue
                direction = float(np.sign(difference[nonzero[0]]))

            aligned_response = direction * (
                2.0 * float(row["response"]) - 1.0
            )
            sums[condition] += aligned_response
            counts[condition] += 1.0

        if np.any(counts == 0.0):
            return float("nan")

        observed_profile = sums / counts
        return float(weights @ (observed_profile - midpoint))

    if "subject_id" in data.columns:
        scores = [
            subject_score(group)
            for _, group in data.groupby("subject_id", sort=False)
        ]
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            return float("nan")
        return float(np.mean(scores))

    return subject_score(data)

outcome: self_sim=-0.0366 (var=0.6529) adversary_sim=-0.0720 (var=0.2833) welch_t=+0.183 p=0.8558 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def subject_score(df):\n        # Estimate response certainty from independent repetitions of exactly\n        # the same display.  For responses coded z in {-1,+1}, the mean of\n        # z_i*z_j over distinct repetition pairs is an unbiased estimate of\n        # the squared latent choice preference for that display.\n        groups = {}\n\n        for _, row in df.iterrows():\n            a = tuple(int(v) for v in row[\"option_a_ratings\"])\n            b = tuple(int(v) for v in row[\"option_b_ratings\"])\n            support = int(sum(abs(x - y) for x, y in zip(a, b)))\n\n            # Use the eight strongest residual conditions. They contain most\n            # of the reliability-allocation separation while retaining 64\n            # observations per subject.\n            if support < 10:\n                continue\n\n            key = (a, b)\n            z = 2.0 * float(row[\"response\"]) - 1.0\n            if key not in groups:\n                groups[key] = []\n            groups[key].append(z)\n\n        estimates = []\n        weights = []\n        for values in groups.values():\n            z = np.asarray(values, dtype=float)\n            n = int(z.size)\n            if n < 2:\n                continue\n\n            # Sum over ordered distinct pairs, equivalent to the average over\n            # unordered repetition pairs.\n            pair_product = (float(np.sum(z)) ** 2 - float(n)) / (\n                float(n) * float(n - 1)\n            )\n            estimates.append(pair_product)\n            weights.append(n * (n - 1) / 2.0)\n\n        if len(estimates) == 0:\n            return float(\"nan\")\n        return float(np.average(np.asarray(estimates), weights=np.asarray(weights)))\n\n    if \"subject_id\" in data.columns:\n        scores = [\n            subject_score(group)\n            for _, group in data.groupby(\"subject_id\", sort=False)\n        ]\n        scores = np.asarray(scores, dtype=float)\n        scores = scores[np.isfinite(scores)]\n        if scores.size == 0:\n            return float(\"nan\")\n        return float(np.mean(scores))\n\n    return subject_score(data)\n",
  "rationale": "This metric measures polarity-invariant repeated-choice certainty in the eight strongest reliability-residual conditions. For each exact A/B display, it averages products of the \u00b11 responses across distinct repetitions, yielding an unbiased estimate of squared latent preference rather than merely counting majority-aligned choices. It therefore uses information discarded by the failed mean-profile contrasts: two theories can have similar average aligned choice rates while differing in how consistently individual subjects repeat those choices. Channel-Specific Coherence Integration has a broad reliability-allocation range and permits many low-allocation subjects to remain weakly committed even at substantial residuals. Relative-Evidence Conflict Arbitration has substantially narrower allocation and near-complete magnitude access in these conditions, predicting more repeatable responses despite its localized notch. Averaging pair-product estimates over 16 independently repeated displays uses 96 within-display response pairs per subject, reduces trial noise, and remains insensitive to stable beneficial-versus-adverse polarity."
}
```

## Usage

```json
{
  "input_tokens": 12676,
  "output_tokens": 3835
}
```
