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
**Validities (n_features=5):** [0.6, 0.95, 0.9, 0.9, 0.9]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0]

**Rationale:** Short-list n=5 design that holds the competing sampler's capacity at (5/5)^0.5 = 1, so every discriminating cue is always sampled and its first-cue bonus is unattenuated because there is no n-5 penalty. The advocated accumulator has length_log = 0 at n=5, so its primacy gradient is steep and its first-discriminator shift is positive rather than negative. Validities are [0.60, 0.95, 0.55, 0.55, 0.55]: a moderately weak first cue, one high-validity second cue, and three matched low-validity tail cues. Every unique trial has exactly three discriminators. The first cue at position 0 with validity 0.60 favors one product; the high cue at position 1 with validity 0.95 always favors the other product; one of the matched low cues with validity 0.55 at position 2, 3, or 4 also favors that other product; the remaining cues are tied. Mirrors swap A and B. Under the advocated position-weighted tally, the later high-plus-low block outweighs the first cue plus its moderate first-cue bonus in all three positive trials, so P(A) is below chance; moving the low tail cue from position 2 to 3 and 4 shrinks its position weight, producing a graded increase in P(A) that stays below 0.5. Under the competing bounded-sampling accumulator, the first cue receives a large fixed first-cue bonus and all three discriminators are sampled, but evidence is not discounted by display position, so the matched low tail cue contributes equally at positions 2, 3, and 4. The first-cue direction wins, so P(A) is above chance and nearly flat across the three tail positions. Mirrors reverse both predictions. This gives both an opposite directional dissociation and a flat-versus-position-gradient quantitative dissociation. Six unique pairs repeated 16 times give 96 trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make binary choices by accumulating signed, count-sensitive evidence across every discriminating feature comparison, without normalizing by total attended evidence. Each cue contributes additive evidence equal to a positive validity-contrast weight times a length-damped serial-position gradient times a subject-specific cue multiplier. The serial-position gradient flattens with list length both by shrinking its primacy range and by lifting its tail, so long lists attend broadly rather than dropping most cues. A first-discriminator adjustment is positive on short lists and crosses smoothly to negative by mid-length lists, but its negative excursion is smoothly bounded by a subject-level cap so long displays show a modest anti-first-discriminator tendency rather than a strong reversal. Accumulated evidence passes through a fairly sharp noisy softmax with lapse and trial-level noise, so majority low-validity blocks can move choice steeply while balanced high-validity blocks cancel and per-pattern choice variance remains large.

**Parameters:**
- validity_intercept: [0.65, 0.85]
- validity_slope: [1.10, 1.60]
- primacy_gain: [2.00, 2.40]
- primacy_tail: [0.50, 0.60]
- tail_length_gain: [0.03, 0.08]
- primacy_center: [1.30, 1.70]
- primacy_temp: [0.30, 0.50]
- length_flat: [0.40, 0.60]
- center_length_shift: [0.10, 0.25]
- first_disc_intercept: [0.42, 0.55]
- first_disc_length_slope: [0.45, 0.60]
- first_disc_neg_cap: [0.15, 0.22]
- subject_scale: [0.40, 1.10]
- evidence_scale: [1.05, 1.30]
- beta: [2.20, 2.80]
- bias: [-0.10, 0.10]
- lapse: [0.09, 0.14]
- trial_noise_sd: [0.05, 0.12]
- cue_noise: [(-0.70, 0.70)] * n_features
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    import math

    if isinstance(state, dict):
        a = np.asarray(state.get('option_a_ratings'), dtype=float)
        b = np.asarray(state.get('option_b_ratings'), dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        return np.full(2, 0.5, dtype=float)

    a = stim[0]
    b = stim[1]
    n = int(stim.shape[1])

    validities = parameters.get('validities')
    if validities is None:
        validities = np.full(n, 0.75, dtype=float)
    else:
        validities = np.asarray(validities, dtype=float)
        if validities.size != n:
            validities = np.full(n, 0.75, dtype=float)

    validity_intercept = float(parameters['validity_intercept'])
    validity_slope = float(parameters['validity_slope'])
    primacy_gain = float(parameters['primacy_gain'])
    primacy_tail = float(parameters['primacy_tail'])
    tail_length_gain = float(parameters['tail_length_gain'])
    primacy_center = float(parameters['primacy_center'])
    primacy_temp = float(parameters['primacy_temp'])
    length_flat = float(parameters['length_flat'])
    center_length_shift = float(parameters['center_length_shift'])
    first_disc_intercept = float(parameters['first_disc_intercept'])
    first_disc_length_slope = float(parameters['first_disc_length_slope'])
    first_disc_neg_cap = float(parameters['first_disc_neg_cap'])
    subject_scale = float(parameters['subject_scale'])
    evidence_scale = float(parameters['evidence_scale'])
    beta = float(parameters['beta'])
    bias = float(parameters['bias'])
    lapse = float(parameters['lapse'])
    trial_noise_sd = float(parameters['trial_noise_sd'])

    cue_noise = parameters.get('cue_noise')
    if cue_noise is None:
        cue_noise = np.zeros(n, dtype=float)
    else:
        cue_noise = np.asarray(cue_noise, dtype=float)
        if cue_noise.ndim != 1 or cue_noise.size != n:
            cue_noise = np.zeros(n, dtype=float)
        cue_noise = np.clip(cue_noise, -0.7, 0.7)

    v = np.clip(validities, 0.5, 1.0)
    validity_dev = v - 0.5
    validity_weight = np.clip(
        validity_intercept + validity_slope * validity_dev, 0.15, 1.8
    )

    length_log = math.log(max(1.0, float(n) - 4.0))
    length_factor = 1.0 + length_flat * length_log
    center_eff = primacy_center + center_length_shift * length_log
    primacy_tail_eff = primacy_tail + tail_length_gain * length_log
    primacy_range = (primacy_gain - primacy_tail_eff) / length_factor

    positions = np.arange(n, dtype=float)
    z_pos = (center_eff - positions) / primacy_temp
    sig = 1.0 / (1.0 + np.exp(-np.clip(z_pos, -30.0, 30.0)))
    position_weight = primacy_tail_eff + primacy_range * sig

    cue_factor = 1.0 + cue_noise
    evidence_weights = (
        evidence_scale
        * subject_scale
        * validity_weight
        * position_weight
        * cue_factor
    )

    diff = a - b
    disc = diff != 0
    if not np.any(disc):
        return np.full(2, 0.5, dtype=float)

    signs = np.where(disc, np.sign(diff), 0.0)
    score = float(np.sum(evidence_weights * signs))

    first_disc = int(np.argmax(disc))
    raw_first_shift = first_disc_intercept - first_disc_length_slope * length_log
    if raw_first_shift < 0.0:
        first_shift = -first_disc_neg_cap * math.tanh(-raw_first_shift / first_disc_neg_cap)
    else:
        first_shift = raw_first_shift

    if diff[first_disc] > 0:
        score += first_shift
    elif diff[first_disc] < 0:
        score -= first_shift

    score += bias

    trial_noise = np.random.normal(0.0, trial_noise_sd)
    logits = np.array([beta * score + trial_noise, 0.0], dtype=float)
    logits = logits - np.max(logits)
    exponentials = np.exp(logits)
    core_probs = exponentials / np.sum(exponentials)

    probs = (1.0 - lapse) * core_probs + lapse * np.full(2, 0.5, dtype=float)
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People make binary choices by stochastically sampling a bounded, capacity-scaled subset of discriminating cues and accumulating signed evidence. Cue sampling is governed jointly by a moderate serial-position primacy gradient that flattens on longer lists and by stated cue validity, whose influence on evidence is compressive rather than all-or-none. The first discriminating cue receives a serial-position-sensitive bonus: early positions carry a calibrated primacy boost, while later first-cues receive only a modest residual bonus, so behavior is neither deterministic first-cue commitment nor a pure equal-weight tally. On longer lists, some subjects carry partially anti-cue weights, producing subject-level reversals of first-cue effects. Accumulated evidence passes through a noisy softmax with lapse, preserving between-subject heterogeneity.

**Parameters:**
- capacity_gamma: [0.45, 0.55]
- position_decay: [0.20, 0.40]
- att_validity_intercept: [0.48, 0.52]
- att_validity_slope: [0.75, 0.90]
- evidence_intercept: [0.50, 0.54]
- evidence_slope: [0.52, 0.66]
- first_cue_bonus: [1.65, 1.95]
- first_cue_validity_power: [0.45, 0.75]
- first_cue_capacity_power: [1.50, 2.50]
- first_position_decay: [0.12, 0.22]
- subject_scale: [0.40, 1.60]
- subject_bias: [-0.30, 0.30]
- beta: [1.10, 2.30]
- epsilon: [0.12, 0.18]
- long_jitter_shift: [0.78, 0.98]
- validities: validities
- evidence_jitter: [(-1.5, 3.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Bounded-sampling accumulator expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    validities = parameters.get('validities')
    if validities is None:
        validities = np.full(n_features, 0.75, dtype=float)
    else:
        validities = np.asarray(validities, dtype=float)
        if validities.shape[0] != n_features:
            validities = np.full(n_features, 0.75, dtype=float)

    jitter = parameters.get('evidence_jitter')
    if jitter is None:
        jitter = np.ones(n_features, dtype=float)
    else:
        jitter = np.asarray(jitter, dtype=float)
        if jitter.ndim != 1 or jitter.shape[0] != n_features:
            jitter = np.ones(n_features, dtype=float)
        else:
            if n_features > 5:
                jitter = np.clip(jitter, -2.5, 3.0)
                jitter = jitter - float(parameters['long_jitter_shift'])
            else:
                jitter = np.clip(jitter, 0.1, 3.0)

    capacity_gamma = float(parameters['capacity_gamma'])
    position_decay = float(parameters['position_decay'])
    att_validity_intercept = float(parameters['att_validity_intercept'])
    att_validity_slope = float(parameters['att_validity_slope'])
    evidence_intercept = float(parameters['evidence_intercept'])
    evidence_slope = float(parameters['evidence_slope'])
    first_cue_bonus = float(parameters['first_cue_bonus'])
    first_cue_validity_power = float(parameters['first_cue_validity_power'])
    first_cue_capacity_power = float(parameters['first_cue_capacity_power'])
    first_position_decay = float(parameters['first_position_decay'])
    subject_scale = float(parameters['subject_scale'])
    subject_bias = float(parameters['subject_bias'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    capacity = (5.0 / n_features) ** capacity_gamma

    positions = np.arange(n_features, dtype=float)
    att_validity = np.clip(
        att_validity_intercept + att_validity_slope * (validities - 0.5), 0.05, 1.0
    )

    sampling_weights = (
        capacity
        * np.exp(-position_decay * positions * (capacity ** 2))
        * att_validity
    )

    diff = a - b
    discrim = diff != 0
    signs = np.where(discrim, np.sign(diff), 0.0)
    discrim_idx = np.flatnonzero(discrim)

    if discrim_idx.size == 0:
        return np.full(2, 0.5, dtype=float)

    sample_size = max(2, int(np.ceil(capacity * discrim_idx.size)))

    if discrim_idx.size <= sample_size:
        sampled = discrim_idx.copy()
    else:
        w = sampling_weights[discrim_idx].astype(float)
        if w.sum() <= 0.0:
            w = np.ones(discrim_idx.size, dtype=float)
        w = w / w.sum()
        sampled = np.random.choice(
            discrim_idx, size=sample_size, replace=False, p=w
        )

    first_disc = int(np.argmax(discrim))

    max_validity = float(np.max(validities))
    first_bonus = (
        first_cue_bonus
        * (validities[first_disc] / max_validity) ** first_cue_validity_power
        * np.exp(-first_cue_capacity_power * max(0.0, n_features - 5.0))
        * np.exp(-first_position_decay * max(0.0, first_disc - 1.0))
    )

    evidence_base = subject_scale * np.clip(
        evidence_intercept + evidence_slope * (validities - 0.5), 0.1, 1.2
    )
    evidence = evidence_base * jitter

    score = 0.0
    for j in sampled:
        j = int(j)
        score += signs[j] * evidence[j]
        if j == first_disc:
            score += signs[j] * first_bonus

    logits = np.array([beta * score + subject_bias, 0.0], dtype=float)
    logits = logits - np.max(logits)
    exponentials = np.exp(logits)
    core_probs = exponentials / np.sum(exponentials)

    probs = (1.0 - epsilon) * core_probs + epsilon * np.full(2, 0.5, dtype=float)
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    first_b = []\n    mirror_b = []\n    for _, row in data.iterrows():\n        a = np.asarray(row['option_a_ratings'], dtype=float)\n        b = np.asarray(row['option_b_ratings'], dtype=float)\n        diff = a - b\n        disc = np.flatnonzero(diff != 0.0)\n        if disc.size < 2:\n            continue\n        signs = np.sign(diff[disc])\n        if signs[0] == 1.0 and np.all(signs[1:] == -1.0):\n            first_b.append(float(row['response']))\n        elif signs[0] == -1.0 and np.all(signs[1:] == 1.0):\n            mirror_b.append(float(row['response']))\n    if not first_b or not mirror_b:\n        return 0.0\n    return float(np.mean(mirror_b) - np.mean(first_b))",
  "rationale": "This metric isolates the core forced-choice conflict in the design. For A-first conflict trials, the first discriminating cue favors A and all later discriminating cues favor B; for mirror trials, the first discriminating cue favors B and all later discriminating cues favor A. Let m_Afirst be the average B-choice rate on A-first conflict trials and m_Bfirst be the average B-choice rate on mirror trials. The metric is m_Bfirst - m_Afirst. The advocated position-weighted accumulator predicts that the later high-validity block outweighs the moderate first cue, so m_Afirst is high, m_Bfirst is low, and the metric is strongly negative. The competing bounded-sampling accumulator predicts first-cue dominance and a large first-cue bonus with no position discounting of the later cues, so m_Afirst is low, m_Bfirst is high, and the metric is positive. Each subject has 48 trials per group because the three conflict pairs and their mirrors are repeated 16 times, which keeps the within-subject aggregate stable."
}
```

## Usage

```json
{
  "input_tokens": 6129,
  "output_tokens": 12222,
  "reasoning_tokens": 11758,
  "provider": "Baidu"
}
```
