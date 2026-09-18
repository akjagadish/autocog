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
**Validities (n_features=14):** [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 2: A=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** This eight-pair design targets the theories' opposite use of communicated validity while holding A's categorical tally advantage fixed at 7 endorsements versus 6. The first seven experts have validity .95 and the last seven validity .50. Across the four base pairs, the high-validity endorsement imbalance favoring A is +7, +3, -3, and -6. Under Uncertain-Credibility Normalized Voting, the corresponding signed residual-deviation sums are approximately +4.07, +1.60, -2.10, and -3.96. Its total evidence for A therefore decreases from approximately 2.58–3.53 to 1.62–1.99, then to approximately -0.30–0.19, and finally to -1.50–-0.55 over the reliability-shrinkage range. It predicts a steep conventional-validity gradient: strong A choice in the first conditions, near-chance choice in the third, and a B preference in the fourth despite A's one-vote tally advantage. The competing theory's dominant latent construal reverses validity direction. Its bounded reliability signal is nearly saturated whenever the high-validity imbalance is nonzero; hence it predicts the opposite gradient—B choices for the +7 and +3 conditions because reversed reliability overturns A's weak tally, but confident A choices for the -3 and -6 conditions because reversed reliability aligns with A's tally. The conflicts involve 13 discriminating experts, so the competing theory's compact-coalition attenuation is essentially inactive. Thus the theories predict oppositely signed condition gradients and opposite majority choices at both ends, not merely different overall accuracy. The final four pairs are exact A/B reversals, balancing response side and requiring mirrored effects. Eight unique pairs are each repeated 12 times for 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Uncertain-Credibility Normalized Voting theory proposes that people do not treat instructed expert validities as precisely known reliabilities. In the absence of feedback, each communicated validity is shrunk toward a shared, moderately credible prototype. Consequently, every discriminating expert supplies one dominant categorical vote, while the incompletely trusted deviation of that expert's validity from the prototype supplies only a residual adjustment. Extreme communicated deviations are compressed smoothly because participants regard unusually high or low precision claims as especially uncertain, while still retaining limited distinctions among the most credible experts. These residual adjustments are accumulated cue by cue, preserving expert identity, but their total is divisively normalized and increasingly bounded once a clear tally coalition exists. Residual reliability is also contextually gated: it is strongly discounted in exact tally ties, where conflicting reliability claims provide no categorical direction, while otherwise retaining its ordinary residual weight without a special margin-one amplification. Before saturation, total evidence undergoes a moderately convex magnitude transformation, making small vote margins comparatively uncertain while preserving confidence for large coalitions. The resulting evidence is passed through a saturating response transformation, so additional agreement has diminishing effects once evidence is decisive. Between-person differences arise in reliability shrinkage, response sensitivity, and lapse rate. Because choices reveal no correctness information, the representation is not updated from choice history.

**Parameters:**
- reliability_shrinkage: [0.58, 0.84]
- choice_sensitivity: [2.20, 2.70]
- lapse_rate: [0.0, 0.12]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Uncertain-Credibility Normalized Voting. History is intentionally ignored:
    # without outcome feedback, previous choices cannot identify expert accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # A difference of +1 is an endorsement of A and -1 an endorsement of B.
    # Nondiscriminating experts contribute neither a vote nor reliability evidence.
    signed_endorsements = np.sign(stimulus[0] - stimulus[1])
    discriminating = signed_endorsements != 0.0
    tally_evidence = float(np.sum(signed_endorsements))

    # Map validity onto above-chance diagnostic strength. The reference value
    # zero corresponds to a common validity of .75. Lower-tail deviations are
    # winsorized, while upper-tail deviations are smoothly compressed so that
    # very credible experts remain distinguishable without receiving full weight.
    v = np.clip(validities, 0.5, 1.0)
    communicated_strength = 2.0 * (v - 0.5)
    raw_deviation = communicated_strength - 0.5
    lower_bounded = np.maximum(raw_deviation, -0.25)
    upper_excess = np.maximum(lower_bounded - 0.25, 0.0)
    reliability_deviation = (
        np.minimum(lower_bounded, 0.25)
        + 0.13 * np.tanh(upper_excess / 0.10)
    )

    # Strong shrinkage leaves only a small fraction of communicated deviations.
    # Deviations are nevertheless summed expert by expert, unlike a summary of
    # the two endorsing groups. Divisive normalization bounds their joint impact.
    shrinkage = float(parameters["reliability_shrinkage"])
    retained_precision = 1.0 - shrinkage
    active_deviations = reliability_deviation[discriminating]
    active_signs = signed_endorsements[discriminating]

    if active_deviations.size == 0:
        reliability_adjustment = 0.0
    else:
        signed_sum = float(np.dot(active_signs, active_deviations))
        deviation_mass = float(np.sum(np.abs(active_deviations)))
        coalition_protection = 0.12 * max(abs(tally_evidence) - 1.0, 0.0) ** 2
        reliability_adjustment = (
            4.0 * retained_precision * signed_sum
            / (1.0 + retained_precision * deviation_mass + coalition_protection)
        )

        # Reliability is least trusted when the categorical tally is exactly
        # tied. Outside exact ties it receives its ordinary residual weight;
        # there is no separate amplification at a one-vote margin.
        absolute_tally = abs(tally_evidence)
        tie_gate = 1.0 - 0.94 * np.exp(-((absolute_tally / 0.35) ** 2))
        reliability_adjustment *= tie_gate

    total_evidence_for_a = tally_evidence + reliability_adjustment

    # Slightly stronger convexity compresses low evidence further. The revised
    # scale is calibrated to residual-shifted large-margin evidence rather than
    # to a raw tally of exactly five.
    response_curvature = 1.28
    saturation_scale = 4.50
    curved_evidence = np.sign(total_evidence_for_a) * (
        abs(total_evidence_for_a) / saturation_scale
    ) ** response_curvature
    normalized_evidence = np.tanh(curved_evidence)
    sensitivity = float(parameters["choice_sensitivity"])
    decision_variable = sensitivity * normalized_evidence

    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * p_core + lapse * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size, 1.0 / probabilities.size, dtype=np.float64
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))
```


## COMPETING THEORY
**Description:** Latent Validity-Construal Tally theory proposes that participants first encode each discriminating expert as one categorical vote and form an unweighted vote margin. Communicated validities are represented separately as monotonic diagnostic-strength codes whose signed sum is compressed into bounded reliability evidence. Crucially, participants differ in their interpretation of the communicated validity scale. The dominant latent construal treats lower stated values as safer or more diagnostic, reversing the direction of reliability evidence. Small subgroups either disregard the validity information or interpret it conventionally. Because reliability evidence is bounded, the dominant reversed construal can overturn a one-vote tally or resolve an exact tally, but it cannot overturn a decisive many-versus-few coalition. When reliability opposes a one-vote tally, its influence is selectively attenuated if that tally advantage arises from a compact, proportionally coherent coalition; reliability remains nearly unchanged in diffuse many-versus-many conflicts. Reliability-based commitment applies only when validity overturns a nonzero tally; exact-tally decisions remain graded rather than receiving the same categorical amplification. In exact tallies, reliability is further gated by its relative salience: the absolute signed validity imbalance is compared with the total encoded validity strength on both sides. Diffuse conflicts between two substantial coalitions receive little reliability influence, whereas endpoint contrasts involving a near-chance expert remain salient. The raw tally is also smoothly saturated, reducing excessive confidence differences between moderate and large margins while retaining its dominance in decisive coalitions. The model is reversal-symmetric and history-independent because no outcome feedback is available.

**Parameters:**
- construal_quantile: [0, 1]
- orientation_strength: [1.9, 2.4]
- reliability_accumulation: [1.8, 3.2]
- exact_tie_salience_threshold: [0.45, 0.60]
- exact_tie_salience_slope: [12, 18]
- conflict_consensus_floor: [0.35, 0.60]
- evidence_scale: [2.3, 2.9]
- choice_sensitivity: [1.55, 2.15]
- construal_commitment: [2.1, 2.5]
- lapse_rate: [0.0, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Latent Validity-Construal Tally model. History is intentionally ignored:
    # choices without correctness feedback cannot reveal expert accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # +1 denotes an expert uniquely endorsing A, -1 an expert uniquely
    # endorsing B, and zero a nondiscriminating expert.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    tally_margin = float(np.sum(signed_votes))

    # The communicated scale is encoded monotonically before its psychological
    # direction is selected. Near-chance experts retain a categorical vote but
    # supply little reliability evidence. Extreme values are compressed.
    v = np.clip(validities, 0.5, 1.0)
    validity_code = np.tanh(3.0 * (v - 0.5)) / np.tanh(1.5)
    signed_validity_sum = float(np.dot(signed_votes, validity_code))

    # Accumulation is cue-by-cue but saturates, so validity can settle a weak
    # conflict without growing enough to defeat a large tally coalition.
    accumulation = float(parameters["reliability_accumulation"])
    bounded_reliability = float(np.tanh(accumulation * signed_validity_sum))

    # In an exact tally, distinguish a concentrated diagnostic imbalance from
    # a diffuse conflict between two substantial endorsing coalitions. The
    # normalization is used only at exact ties, leaving every nonzero-tally
    # prediction and its successful margin profile unchanged.
    if abs(tally_margin) < 1e-12:
        total_validity_strength = float(
            np.dot(np.abs(signed_votes), validity_code)
        )
        if total_validity_strength > 1e-12:
            relative_salience = np.clip(
                abs(signed_validity_sum) / total_validity_strength, 0.0, 1.0
            )
        else:
            relative_salience = 0.0

        threshold = float(parameters["exact_tie_salience_threshold"])
        slope = float(parameters["exact_tie_salience_slope"])
        raw_gate = 1.0 / (
            1.0 + np.exp(-slope * (relative_salience - threshold))
        )
        endpoint_gate = 1.0 / (
            1.0 + np.exp(-slope * (1.0 - threshold))
        )
        bounded_reliability *= float(raw_gate / endpoint_gate)

    # The continuous latent index implements a task-invariant population
    # mixture. Nearly all participants reverse the instructed scale; very small
    # groups ignore it or use its conventional direction.
    construal_quantile = float(parameters["construal_quantile"])
    if construal_quantile < 0.99:
        validity_orientation = -1.0
    elif construal_quantile < 0.998:
        validity_orientation = 0.0
    else:
        validity_orientation = 1.0

    orientation_strength = float(parameters["orientation_strength"])
    reliability_evidence = (
        validity_orientation * orientation_strength * bounded_reliability
    )

    # In a nonzero one-vote conflict, coalition structure regulates only the
    # opposing reliability signal. A compact proportional majority protects
    # its tally, whereas a diffuse many-versus-many margin leaves validity
    # almost unchanged. Larger tally margins and aligned reliability are not
    # modified, preserving the established margin-confidence profile.
    discriminating_count = int(np.count_nonzero(signed_votes))
    if (
        discriminating_count > 0
        and 0.5 < abs(tally_margin) < 1.5
        and reliability_evidence * tally_margin < 0.0
    ):
        relative_consensus = abs(tally_margin) / float(discriminating_count)
        coherent_conflict = 1.0 / (
            1.0 + np.exp(-24.0 * (relative_consensus - 0.25))
        )
        conflict_floor = float(parameters["conflict_consensus_floor"])
        conflict_gate = 1.0 - (1.0 - conflict_floor) * coherent_conflict
        reliability_evidence *= float(conflict_gate)

    # Smoothly saturating the raw tally reduces excessive margin-dependent
    # confidence while preserving its sign and its dominance for large margins.
    tally_evidence = 3.8 * np.tanh(tally_margin / 3.8)
    total_evidence = tally_evidence + reliability_evidence

    # A reliability-based construal that reverses a weak nonzero tally is
    # treated as a categorical resolution. Exact-tally reliability choices are
    # left graded, avoiding excessive amplification in balanced coalitions.
    reliability_controls = False
    if abs(total_evidence) > 1e-12 and abs(reliability_evidence) > 1e-12:
        if abs(tally_margin) >= 1e-12:
            if np.sign(total_evidence) != np.sign(tally_margin):
                reliability_controls = True

    evidence_scale = float(parameters["evidence_scale"])
    normalized_evidence = float(np.tanh(total_evidence / evidence_scale))
    decision_variable = (
        float(parameters["choice_sensitivity"]) * normalized_evidence
    )
    if reliability_controls:
        decision_variable *= float(parameters["construal_commitment"])

    # Positive evidence favors A. Construct symmetric logits to preserve exact
    # invariance under swapping the two displayed options.
    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size, 1.0 / probabilities.size, dtype=np.float64
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    if data is None or len(data) == 0:\n        return float('nan')\n\n    weighted_success = 0.0\n    total_weight = 0.0\n\n    # Conditions are represented in a canonical orientation using the sign of\n    # the overall tally. This makes exact A/B-reversed pairs contribute to the\n    # same planned condition.\n    condition_weights = {\n        7: 2.00,\n        3: 1.45,\n        -3: 0.80,\n        -6: 1.10,\n    }\n\n    for _, row in data.iterrows():\n        a = np.asarray(row['option_a_ratings'], dtype=np.float64)\n        b = np.asarray(row['option_b_ratings'], dtype=np.float64)\n        if a.size < 7 or b.size != a.size:\n            continue\n\n        signed_votes = np.sign(a - b)\n        high_validity_margin = float(np.sum(signed_votes[:7]))\n        tally_margin = float(np.sum(signed_votes))\n        if high_validity_margin == 0.0 or tally_margin == 0.0:\n            continue\n\n        canonical_margin = int(round(high_validity_margin * np.sign(tally_margin)))\n        weight = condition_weights.get(canonical_margin, 1.0)\n\n        response = int(row['response'])\n        chose_high_validity_side = (\n            (high_validity_margin > 0.0 and response == 0) or\n            (high_validity_margin < 0.0 and response == 1)\n        )\n        weighted_success += weight * float(chose_high_validity_side)\n        total_weight += weight\n\n    if total_weight == 0.0:\n        return float('nan')\n    return float(weighted_success / total_weight)",
  "rationale": "This is a condition-weighted rate of choosing the option endorsed by more of the seven .95-validity experts. It collapses each A/B reversal into the same canonical condition, eliminating response-side effects. The advocated theory predicts predominantly validity-consistent choices, especially for canonical high-validity margins +7 and +3, whereas the competing theory's dominant reversed construal predicts validity-inconsistent choices across all four conditions. Weights emphasize the conditions with the largest predicted probability gaps while retaining the full 96-trial sample to keep subject-level binomial variance low. The statistic remains an interpretable behavioral measure\u2014adherence to the communicated validity ordering\u2014rather than a direct restatement of either model's internal computation."
}
```

## Usage

```json
{
  "input_tokens": 6317,
  "output_tokens": 3117
}
```
