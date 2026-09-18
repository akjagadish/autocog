# metric_exp01_attempt_00

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
**Validities (n_features=48):** [0.99, 0.51, 0.87, 0.63, 0.87, 0.63, 0.83, 0.67, 0.83, 0.67, 0.83, 0.67, 0.81, 0.69, 0.81, 0.69, 0.81, 0.69, 0.81, 0.69, 0.79, 0.71, 0.79, 0.71, 0.79, 0.71, 0.79, 0.71, 0.79, 0.71, 0.79, 0.71, 0.78, 0.72, 0.78, 0.72, 0.78, 0.72, 0.78, 0.72, 0.78, 0.72, 0.78, 0.72, 0.78, 0.72, 0.78, 0.72]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Rationale:** This 48-feature design isolates the competitor's active-field validity-contrast mechanism while holding every input to Relative-Evidence Conflict Arbitration exactly constant. The validity profile comprises six disjoint blocks of symmetric high/low pairs around the common mean 0.75. The blocks contain m = 1, 2, 3, 4, 6, or 8 pairs, with each pair's deviation from the mean equal to 0.24/m. In each base trial, only one block discriminates: every high-validity member marks A and every low-validity partner marks B. Consequently, both products receive exactly m marks, the count difference is zero, the centered-validity residual is always +0.48, gross reliability support is always 0.48, and reliability coherence is exactly 1. Relative-Evidence Conflict Arbitration therefore predicts exactly the same choice probability in all six conditions, including identical magnitude access and notch calibration. Contrast-Calibrated Reliability-Precision Competition instead predicts a monotonic decline in choices of B, the option favored by the positive residual. Its active-field RMS validity contrast falls from 0.24 to 0.12, 0.08, 0.06, 0.04, and 0.03 as the fixed residual is distributed across progressively more moderate experts. This lowers selective dilution confidence and reliability precision; because count signal is zero but count precision retains its floor, the reliability estimate is increasingly pulled toward indifference. Unlike a design that merely adds mean-validity null cues, this manipulation redistributes the same coherent reliability evidence across fields of different contrast. The final six trials exchange A and B, balancing response side. Twelve unique pairs repeated eight times yield exactly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Contrast-Calibrated Reliability-Precision Competition theory: people encode a binary-feature display through two stable-polarity evidence channels. The count channel accumulates only the signed difference in marks and is therefore invariant to additions that cancel in the tally. It saturates smoothly after relatively small margins, while retaining a substantial precision floor, so ordinary and large margins differ less sharply in confidence. The reliability channel computes centered validity evidence but treats that residual as an uncertain estimate. Its precision is inferred jointly from residual magnitude, signed coherence, and active-field validity contrast relative to the characteristic dispersion of the communicated validity profile. Weak residuals retain partial availability through a confidence-weighted precision floor: coherent residuals in a distinct validity field remain available, whereas diffuse weak residuals receive little precision. Reliability is selectively diluted when active validity contrast falls relative to the full validity field, rather than merely because more cues are active. Count and reliability estimates enter a smooth precision-weighted competition: count controls choices when reliability is uncertain, whereas sufficiently precise coherent reliability can dominate count or resolve count ties. Broad continuous subject variation in count weighting, relative channel precision, and response precision produces persistent heterogeneity without discrete strategy classes.

**Parameters:**
- validities: validities
- count_weight: [1.25, 1.90]
- count_scale: [2.8, 4.2]
- count_precision: [0.35, 3.10]
- count_precision_floor: [0.52, 0.82]
- count_precision_half: [0.80, 2.00]
- reliability_gain: [1.8, 4.4]
- reliability_precision: [0.15, 3.10]
- reliability_floor: [0.10, 0.28]
- magnitude_half: [0.08, 0.30]
- magnitude_power: [1.00, 1.70]
- coherence_floor: [0.15, 0.35]
- coherence_power: [0.85, 1.65]
- contrast_half: [0.55, 1.25]
- dilution_floor: [0.12, 0.42]
- competition_scale: [1.45, 2.45]
- decision_bound: [3.4, 6.8]
- response_precision: [0.40, 2.10]
- adverse_polarity_confidence: [0.82, 0.99]
- lapse_rate: [0.0, 0.12]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Contrast-Calibrated Reliability-Precision Competition expects "
            f"shape (2, n_features); got {stim.shape}."
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
    count_magnitude = abs(count_difference)

    # This accumulator depends only on signed tally margin. Adding equally
    # many marks to opposing sides leaves both its signal and precision fixed.
    # Its relatively fast saturation flattens small-to-large margin growth.
    count_scale = float(parameters["count_scale"])
    count_signal = (
        float(parameters["count_weight"])
        * count_scale
        * np.tanh(count_difference / count_scale)
    )

    # A substantial baseline precision makes tallying available at ordinary
    # margins without making precision grow excessively with margin size.
    count_access = count_magnitude / (
        count_magnitude + float(parameters["count_precision_half"]) + 1e-12
    )
    count_precision = float(parameters["count_precision"]) * (
        float(parameters["count_precision_floor"])
        + (1.0 - float(parameters["count_precision_floor"])) * count_access
    )

    # Centering removes the validity component redundant with ordinary count.
    centered_validities = validities - float(np.mean(validities))
    signed_components = centered_validities * difference
    reliability_residual = float(np.sum(signed_components))
    residual_magnitude = abs(reliability_residual)
    gross_support = float(np.sum(np.abs(signed_components)))

    if gross_support > 1e-12:
        coherence = residual_magnitude / gross_support
        coherence = float(np.clip(coherence, 0.0, 1.0))
    else:
        coherence = 0.0

    # Active contrast is interpreted relative to the characteristic dispersion
    # of the communicated profile. This dimensionless reference distinguishes
    # a genuinely weakening active field from a mere change in measurement
    # units, while introducing no direct cue-count or display-size penalty.
    active = np.abs(difference) > 0.5
    if np.any(active):
        active_deviations = centered_validities[active]
        active_contrast = float(np.sqrt(np.mean(active_deviations ** 2)))
    else:
        active_contrast = 0.0

    profile_contrast = float(
        np.sqrt(np.mean(centered_validities ** 2))
    )
    if profile_contrast > 1e-12:
        relative_contrast = active_contrast / profile_contrast
    else:
        relative_contrast = 0.0

    contrast_confidence = relative_contrast / (
        relative_contrast + float(parameters["contrast_half"]) + 1e-12
    )
    contrast_confidence = float(np.clip(contrast_confidence, 0.0, 1.0))

    # Selective dilution follows relative active-field signal quality, not raw
    # feature count. The smooth cubic mapping makes deliberately declining
    # contrast diagnostic while retaining the nonzero accessibility floor.
    dilution_floor = float(parameters["dilution_floor"])
    selective_dilution = (
        dilution_floor
        + (1.0 - dilution_floor) * contrast_confidence ** 3
    )

    # Reliability precision has smooth magnitude-by-coherence curvature. Weak
    # residuals remain partly available, while coherent large residuals become
    # disproportionately trustworthy. The softened coherence transformation
    # prevents an unrealistically extreme coherent/diffuse endpoint.
    magnitude_confidence = residual_magnitude / (
        residual_magnitude + float(parameters["magnitude_half"]) + 1e-12
    )
    magnitude_confidence = magnitude_confidence ** float(
        parameters["magnitude_power"]
    )
    coherence_confidence = coherence ** float(parameters["coherence_power"])
    coherence_term = (
        float(parameters["coherence_floor"])
        + (1.0 - float(parameters["coherence_floor"]))
        * coherence_confidence
    )

    diagnostic_confidence = (
        magnitude_confidence * coherence_term * selective_dilution
    )

    # The weak-residual floor is itself confidence weighted. Small coherent
    # residuals in a distinctive active field remain accessible, whereas weak
    # diffuse residuals no longer receive the same unconditional precision.
    reliability_floor = float(parameters["reliability_floor"])
    floor_confidence = coherence_term * selective_dilution
    reliability_quality = (
        reliability_floor * floor_confidence
        + (1.0 - reliability_floor) * diagnostic_confidence
    )

    if residual_magnitude <= 1e-12:
        reliability_precision = 0.0
        reliability_signal = 0.0
    else:
        reliability_precision = (
            float(parameters["reliability_precision"])
            * reliability_quality
        )
        reliability_signal = (
            float(parameters["reliability_gain"])
            * reliability_residual
        )

    # Precision-weighted averaging implements smooth competition without a
    # notch, hard strategy switch, or explicit winner-enhancement operation.
    # It applies identically to congruent and conflicting channel directions.
    total_precision = count_precision + reliability_precision
    if total_precision <= 1e-12:
        integrated_signal = 0.0
    else:
        integrated_signal = float(parameters["competition_scale"]) * (
            count_precision * count_signal
            + reliability_precision * reliability_signal
        ) / total_precision

    # A final bounded decision representation supplies diminishing returns but
    # does not alter the precision estimates used to arbitrate the channels.
    decision_bound = float(parameters["decision_bound"])
    integrated_signal = decision_bound * np.tanh(
        integrated_signal / decision_bound
    )
    markedness_evidence = (
        float(parameters["response_precision"]) * integrated_signal
    )

    # Positive evidence means A carries greater adverse markedness and favors B.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # Semantic polarity remains fixed over the subject run.
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
[0] rationale: The metric is the proportion of side-balanced choices following the signed centered-validity evidence over residual magnitudes 0.8–2.2 (n=4–11). This range includes weak evidence and the advocated theory's subject-varying calibration-notch region. Relative-Evidence Conflict Arbitration predicts restrained reliability-consistent choice in this band because its lower-gain reliability channel is additionally discounted near intermediate magnitudes. The competing precision-competition theory has no localized discount and its larger reliability gain should produce substantially stronger reliability-aligned choices even at these magnitudes. Pooling eight n-levels, both display orientations, and all four repetitions uses 64 trials per subject, limiting Bernoulli noise and cancelling stable A/B response bias. This is an empirical confidence/alignment measure rather than a direct check that a notch parameter exists.
metric_source:
def metric(data: pd.DataFrame) -> float:
    aligned = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Each diagnostic pair contributes one mark to each option, so the
        # number of marks on A identifies n and hence residual magnitude 0.2*n.
        n = int(round(float(np.sum(a))))
        if 4 <= n <= 11:
            # Feature 0 is a high-validity expert. If it marks A, positive
            # reliability markedness favors choosing B; after the display swap
            # it marks B and the corresponding choice is A.
            high_validity_option_is_a = bool(a[0] > b[0])
            response = int(row["response"])
            adverse_consistent = (
                response if high_validity_option_is_a else 1 - response
            )
            aligned.append(float(adverse_consistent))

    if len(aligned) == 0:
        return float("nan")
    return float(np.mean(aligned))
outcome: self_sim=0.7159 (var=0.0052) adversary_sim=0.7622 (var=0.0128) welch_t=-1.722 p=0.09257 (N=25, alpha=0.01) -> reject

[1] rationale: This is a within-subject reliability-evidence dynamic-range contrast: reliability-aligned choice at strong residuals (n=12–15; magnitudes 2.4–3.0) minus aligned choice at weak residuals (n=4–7; magnitudes 0.8–1.4). It is qualitatively different from averaging alignment through the notch region. The advocated theory should show appreciable growth from the weak range to the recovered post-notch range. In the competing theory, reliability precision and its high reliability gain make even weak coherent residuals relatively decisive, after which precision weighting, the decision bound, and response probabilities compress further growth. Subtracting each subject's weak-range performance removes much of the stable variation in polarity, lapse tendency, and overall response confidence that inflated variance in the failed level metric. Each endpoint estimate uses 32 trials per subject, and recoding by which option receives the high-validity marks cancels side bias.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df: pd.DataFrame) -> float:
        low = []
        high = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            n = int(round(float(np.sum(a))))

            # Recode choices relative to the reliability-evidence direction,
            # thereby cancelling the A/B swap used in the design.
            high_validity_option_is_a = bool(a[0] > b[0])
            response = int(row["response"])
            aligned = float(response if high_validity_option_is_a else 1 - response)

            if 4 <= n <= 7:
                low.append(aligned)
            elif 12 <= n <= 15:
                high.append(aligned)

        if len(low) == 0 or len(high) == 0:
            return float("nan")
        return float(np.mean(high) - np.mean(low))

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        values = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            value = subject_contrast(subj_df)
            if np.isfinite(value):
                values.append(value)
        if len(values) == 0:
            return float("nan")
        return float(np.mean(values))

    return subject_contrast(data)

outcome: self_sim=0.1294 (var=0.0128) adversary_sim=0.0906 (var=0.0120) welch_t=+1.231 p=0.2244 (N=25, alpha=0.01) -> reject

[2] rationale: This metric measures the relative dynamic range of reliability-guided choice across all 12 residual magnitudes. It estimates each subject's linear increase in side-balanced reliability alignment and divides that increase by the subject's overall alignment amplitude. This differs qualitatively from the failed raw endpoint contrast: stable semantic polarity, lapse rate, and general response determinism largely multiply both the level and the evidence-dependent increase, so normalization suppresses precisely the subject heterogeneity that inflated the previous contrasts. The advocated theory predicts a comparatively low intermediate-strength response followed by recovered growth, yielding a larger slope relative to its overall response amplitude. The competing theory's high-gain precision channel makes weak coherent residuals already influential and then compresses additional growth through smooth precision weighting and bounded responding, yielding a smaller normalized slope. Using all 96 trials improves precision, while a ridge and winsorization stabilize the statistic for noisy near-random subjects without directly testing for the presence of a notch parameter.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subject_score(df: pd.DataFrame) -> float:
        xs = []
        ys = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)

            # The design has n marks on each option. Map n=4,...,15 onto
            # an approximately [-1, 1] evidence-strength coordinate.
            n = int(round(float(np.sum(a))))
            if n < 4 or n > 15:
                continue
            x = (float(n) - 9.5) / 5.5

            # Recode the response so 1 always denotes following the adverse
            # centered-validity direction, independently of display side.
            high_validity_option_is_a = bool(a[0] > b[0])
            response = int(row["response"])
            aligned = float(response if high_validity_option_is_a else 1 - response)

            xs.append(x)
            ys.append(aligned)

        if len(ys) < 2:
            return float("nan")

        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        x = x - float(np.mean(x))
        denom_x = float(np.sum(x * x))
        if denom_x <= 1e-12:
            return float("nan")

        # OLS growth in alignment over the complete 12-level schedule.
        slope = float(np.sum(x * (y - float(np.mean(y)))) / denom_x)

        # Normalize growth by the subject's overall reliability-directed
        # response amplitude. The ridge limits noise for near-random subjects.
        amplitude = max(float(np.mean(y)) - 0.5, 0.0)
        score = slope / (0.10 + amplitude)

        # Winsorization prevents a few low-amplitude empirical curves from
        # dominating between-subject variance.
        return float(np.clip(score, -1.0, 1.0))

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        scores = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            value = subject_score(subj_df)
            if np.isfinite(value):
                scores.append(value)
        if len(scores) == 0:
            return float("nan")
        return float(np.mean(scores))

    return subject_score(data)

outcome: self_sim=0.2633 (var=0.0519) adversary_sim=0.1875 (var=0.0537) welch_t=+1.166 p=0.2495 (N=25, alpha=0.01) -> reject

[3] rationale: This is a nuisance-orthogonalized matched-filter measure of localized reliability calibration. It first constructs each subject's 12-point, side-balanced reliability-alignment curve. The diagnostic template averages the expected intermediate-strength attenuation across the advocated theory's full range of notch centers and widths, making the metric robust to subject-specific notch location. Projecting that template off intercept, linear, quadratic, and cubic components removes overall reliability preference, ordinary evidence growth, ceiling effects, and broad smooth saturation—the main sources of heterogeneity and the patterns the competing theory can generate. The remaining scalar therefore measures a localized midrange deformation followed by recovery rather than merely lower alignment or greater endpoint growth, both of which previously had insufficient separation. All 96 responses contribute, and the contrast weights sum to zero, reducing variance from semantic polarity, lapse rate, and general response determinism.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subject_score(df: pd.DataFrame) -> float:
        sums = {n: [] for n in range(4, 16)}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            n = int(round(float(np.sum(a))))
            if n < 4 or n > 15:
                continue

            # Recode so that one always means choosing in the direction of the
            # high-validity experts, independent of the A/B display swap.
            high_marks_a = bool(a[0] > b[0])
            response = int(row["response"])
            aligned = float(response if high_marks_a else 1 - response)
            sums[n].append(aligned)

        if any(len(sums[n]) == 0 for n in range(4, 16)):
            return float("nan")

        nvals = np.arange(4.0, 16.0)
        y = np.asarray([np.mean(sums[n]) for n in range(4, 16)], dtype=float)

        # Average the predicted calibration shape across the stated range of
        # notch centers and widths. Multiplication by n reflects the fact that
        # the discounted reliability signal itself grows with residual size.
        centers = np.linspace(8.0, 11.0, 7)
        widths = np.linspace(2.25, 4.0, 5)
        template = np.zeros_like(nvals)
        for center in centers:
            for width in widths:
                template += nvals * np.exp(-0.5 * ((nvals - center) / width) ** 2)
        template /= float(len(centers) * len(widths))

        # Remove every cubic trend from the template. The resulting matched
        # contrast is insensitive to level, linear growth, ordinary saturation,
        # and broad curvature, retaining only a localized midrange deformation
        # followed by recovery.
        x = (nvals - np.mean(nvals)) / np.std(nvals)
        smooth = np.column_stack((np.ones_like(x), x, x ** 2, x ** 3))
        beta = np.linalg.lstsq(smooth, template, rcond=None)[0]
        weights = template - smooth @ beta
        norm = float(np.sum(np.abs(weights)))
        if norm <= 1e-12:
            return float("nan")
        weights /= norm

        return float(np.dot(weights, y))

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        scores = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            value = subject_score(subj_df)
            if np.isfinite(value):
                scores.append(value)
        if len(scores) == 0:
            return float("nan")
        return float(np.mean(scores))

    return subject_score(data)

outcome: self_sim=0.0046 (var=0.0010) adversary_sim=0.0009 (var=0.0015) welch_t=+0.363 p=0.7184 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Matched-filter weights for the monotonic attenuation predicted when the\n    # fixed reliability residual is distributed over increasingly many cues.\n    # They sum to zero, so any condition-invariant choice probability cancels.\n    weights = {\n        1: 51.0,\n        2: 19.0,\n        3: -1.0,\n        4: -12.0,\n        6: -26.0,\n        8: -31.0,\n    }\n\n    weighted_sum = 0.0\n    weight_mass = 0.0\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        difference = a - b\n\n        # Each option has m marks in this design. Using the number of active\n        # differences also remains valid if ratings are represented as floats.\n        m = int(round(float(np.sum(np.abs(difference))) / 2.0))\n        if m not in weights:\n            continue\n\n        # High-validity members occupy even zero-based indices in every block.\n        # Recode so one always denotes the reliability-consistent adverse\n        # choice, regardless of which display side receives those members.\n        even_support_for_a = float(np.sum(difference[::2]))\n        odd_support_for_a = float(np.sum(difference[1::2]))\n        high_validity_option_is_a = even_support_for_a > odd_support_for_a\n\n        response = int(row[\"response\"])\n        aligned = float(response if high_validity_option_is_a else 1 - response)\n\n        w = weights[m]\n        weighted_sum += w * aligned\n        weight_mass += abs(w)\n\n    if weight_mass <= 0.0:\n        return float(\"nan\")\n    return float(weighted_sum / weight_mass)",
  "rationale": "This is a zero-sum matched-filter contrast over the actual six active-block sizes m = 1, 2, 3, 4, 6, and 8. Responses are first recoded as choices aligned with the high-validity experts, independently of A/B orientation. Relative-Evidence Conflict Arbitration gives every condition the same count difference, reliability residual, gross support, coherence, and magnitude calibration, so its expected score is near zero for each subject. The competing theory predicts progressively weaker reliability-aligned choice as active-field contrast falls with m; the weights approximate that predicted nonlinear decline while using all 96 trials. Their zero sum removes each subject's invariant reliability preference, polarity, lapse tendency, and general response confidence, and the side-balanced recoding suppresses response-side bias. Unlike the failed metrics, this uses the experiment's true block sizes rather than treating mark counts as a 4\u201315 residual-magnitude manipulation."
}
```

## Usage

```json
{
  "input_tokens": 13038,
  "output_tokens": 2785
}
```
