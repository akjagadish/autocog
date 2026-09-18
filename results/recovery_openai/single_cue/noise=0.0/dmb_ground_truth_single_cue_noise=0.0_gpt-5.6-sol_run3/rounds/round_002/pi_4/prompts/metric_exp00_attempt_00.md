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
**Validities (n_features=21):** [0.5, 0.5, 0.5, 0.5, 0.5, 0.689974, 0.802184, 0.750522, 0.750522, 0.622459, 0.845535, 0.751327, 0.751327, 0.549834, 0.880797, 0.752669, 0.752669, 0.512497, 0.895669, 0.753525, 0.753525]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

**Rationale:** This design targets both a robust confidence dissociation and a secondary composition-sensitive gradient. Every base pair gives A seven discriminating endorsements and B two, producing the same tally margin of five. Five of A's endorsements always come from 50%-valid experts. Compressed-Reliability Consensus treats these as full categorical votes and applies its saturated margin-five consensus multiplier; it therefore predicts substantially above-chance A choice. Even at its least confident parameter settings, its predicted A probability is approximately 0.67 or greater. Communicated-Reliability Accumulation gives each 50%-valid expert only the reliability floor, so these five endorsements contribute total evidence of only 5c = 0.50–0.70. The remaining two-versus-two coalitions were selected to have approximately equal summed transformed log-odds weights throughout the accumulation model's narrow gamma range. Consequently, accumulation predicts a nearly invariant and much weaker A preference, approximately 0.53–0.60 across base pairs. The four conditions also increase the dispersion of A's two non-chance reliabilities while preserving their accumulated weight: from 69%/80% to 51%/90%, opposed by two experts near 75%. Softmax compression increasingly emphasizes A's high-reliability member, moving the advocated model's reliability contrast from slightly negative to positive and producing an increasing choice-confidence gradient. Accumulation predicts essentially no such condition gradient because the transformed-weight sums are matched. Exact reversals balance response side and require mirrored predictions. There are eight unique pairs, each repeated 12 times for 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Compressed-Reliability Consensus theory holds that people represent every discriminating expert primarily as one categorical vote and encode communicated reliability through a separate, bounded comparison of the two endorsing groups. For each option, attention is biased moderately toward its more reliable endorsers, but their reliabilities are summarized as a softmax-weighted mean rather than accumulated expert by expert. Net evidence is therefore the full signed tally plus an identity-sensitive contrast between the two bounded group summaries. Agreement additionally amplifies confidence as the absolute tally margin grows, with amplification saturating after a decisive coalition has formed. This representation permits expert identity to determine equal-tally choices, preserves substantial influence for near-chance experts, and allows sufficiently numerous weak experts to override the strongest expert. Reliability sensitivity, consensus amplification, choice sensitivity, and lapse rate vary across subjects. Because the task provides no correctness feedback, choices are reversal-symmetric and independent of history.

**Parameters:**
- reliability_sensitivity: [0.65, 1.05]
- consensus_amplification: [0.03, 0.07]
- choice_sensitivity: [0.13, 0.30]
- lapse_rate: [0.0, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Compressed-Reliability Consensus model. History is intentionally ignored
    # because the task supplies no outcome feedback from which to recalibrate.
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

    # Every discriminating expert contributes a full baseline vote through the
    # signed tally. Reliability is encoded separately as a bounded summary of
    # each endorsing side, preventing identity increments from accumulating
    # without bound as a coalition grows.
    v = np.clip(validities, 0.5, 1.0)
    reliability_code = np.tanh(3.0 * (v - 0.5)) / np.tanh(1.5)

    difference = stimulus[0] - stimulus[1]
    a_endorsers = difference > 0.0
    b_endorsers = difference < 0.0
    tally_margin = float(np.sum(a_endorsers) - np.sum(b_endorsers))

    def side_reliability(mask):
        values = reliability_code[mask]
        if values.size == 0:
            return 0.0
        attention_logits = 4.0 * values
        attention_logits = attention_logits - np.max(attention_logits)
        attention = np.exp(attention_logits)
        attention /= attention.sum()
        return float(np.dot(attention, values))

    reliability_contrast = (
        side_reliability(a_endorsers) - side_reliability(b_endorsers)
    )
    alpha = float(parameters["reliability_sensitivity"])
    weighted_balance = tally_margin + alpha * reliability_contrast

    # Agreement is represented separately by the unweighted vote margin.
    # It changes confidence but does not erase which experts supplied the
    # votes. Leaving margin-one and tied tallies unamplified preserves
    # uncertainty in sparse conflicts and identity effects in equal tallies.
    # Amplification saturates at the level reached by a margin-five coalition.
    excess_consensus = max(abs(tally_margin) - 1.0, 0.0)
    consensus_gain = float(parameters["consensus_amplification"])
    confidence_multiplier = 1.0 + consensus_gain * min(excess_consensus ** 1.5, 8.0)
    evidence_for_a = weighted_balance * confidence_multiplier

    sensitivity = float(parameters["choice_sensitivity"])
    logits = np.array(
        [0.5 * sensitivity * evidence_for_a,
         -0.5 * sensitivity * evidence_for_a],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + lapse * 0.5
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
        probabilities = np.full(probabilities.size, 1.0 / probabilities.size)
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))
```


## COMPETING THEORY
**Description:** Communicated-Reliability Accumulation theory proposes that people integrate every discriminating expert rating after translating communicated validity into subjective diagnostic weight. The weight is a mildly superlinear log-odds transformation plus a modest common reliability floor: w_j = logit(v_j)^gamma + c. The log-odds component privileges more reliable experts, while the positive floor represents a minimal contribution assigned to any expert judged better than chance. Net evidence favoring A is E = sum_j w_j(A_j-B_j). Consequently, sufficiently numerous weaker experts can overturn a stronger cue, and concordant cue-number margins generate graded confidence. Choice sensitivity varies moderately across people, while only very rare lapses produce validity-independent guessing. Choice history is ignored because the task provides no outcome feedback.

**Parameters:**
- beta: [0.24, 0.58]
- epsilon: [0.0, 0.005]
- gamma: [1.02, 1.05]
- reliability_floor: [0.10, 0.14]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Communicated-Reliability Accumulation.
    # state has shape (2, n_features), with rows for options A and B.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Reliability accumulation expects shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Clipping only protects the log-odds transform at its mathematical
    # endpoints. A modest common floor strengthens accumulation by cue number,
    # while mild superlinearity preserves the leverage of highly valid cues.
    v = np.clip(validities, 0.5, 1.0 - 1e-12)
    log_odds = np.log(v) - np.log1p(-v)
    gamma = float(parameters["gamma"])
    reliability_floor = float(parameters["reliability_floor"])
    reliability_weights = np.power(log_odds, gamma) + reliability_floor

    a = stimulus[0]
    b = stimulus[1]
    evidence_for_a = float(np.dot(reliability_weights, a - b))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Symmetric logits make their difference beta * evidence_for_a, so the
    # core probability of A is sigmoid(beta * evidence_for_a).
    logits = np.array(
        [0.5 * beta * evidence_for_a, -0.5 * beta * evidence_for_a],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    probabilities = (1.0 - epsilon) * p_core + epsilon * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.ones(len(probabilities), dtype=np.float64) / len(probabilities)
    else:
        probabilities /= total
    return int(np.random.choice(len(probabilities), p=probabilities))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Proportion of choices favoring the option with the larger raw vote tally.\"\"\"\n    scores = []\n    for a_cell, b_cell, response in zip(\n        data[\"option_a_ratings\"],\n        data[\"option_b_ratings\"],\n        data[\"response\"],\n    ):\n        a = np.asarray(a_cell, dtype=np.float64)\n        b = np.asarray(b_cell, dtype=np.float64)\n        margin = float(np.sum(a - b))\n        r = int(response)\n\n        if margin > 0.0:\n            scores.append(1.0 if r == 0 else 0.0)\n        elif margin < 0.0:\n            scores.append(1.0 if r == 1 else 0.0)\n        else:\n            scores.append(0.5)\n\n    if len(scores) == 0:\n        return float(\"nan\")\n    return float(np.mean(scores))",
  "rationale": "This is the raw-coalition choice rate: the probability that a subject selects the option endorsed by more discriminating experts, irrespective of response side or expert identity. All eight stimuli have a 7-versus-2 endorsement split, and the four reversed pairs ensure that choosing the tally winner is not confounded with choosing A. Compressed-Reliability Consensus assigns full baseline votes to the five 50%-valid experts and amplifies the margin-five coalition, so it predicts a substantially higher rate on this measure. Communicated-Reliability Accumulation gives those experts only the small reliability floor and predicts a much weaker tally-winner preference. Pooling all 96 balanced trials yields a low-noise estimate, while the same definition remains meaningful and stable for each individual subject."
}
```

## Usage

```json
{
  "input_tokens": 5132,
  "output_tokens": 1959
}
```
