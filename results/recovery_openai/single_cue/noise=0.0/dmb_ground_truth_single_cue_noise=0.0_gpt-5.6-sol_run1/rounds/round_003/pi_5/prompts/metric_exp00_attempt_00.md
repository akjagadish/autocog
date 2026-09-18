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
**Validities (n_features=6):** [0.55, 0.6, 0.99, 0.95, 0.85, 0.75]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 5: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Rationale:** This fixed-length scan-order conflict test exploits the theories' different definitions of cue order. The experts are displayed in index order, but their validities imply the descending-validity order 3, 4, 5, 6, 2, 1. In every base trial, the displayed sequence consists of one A run followed by a terminal B run: AAAAAB, AAAABB, AAABBB, or AABBBB. Directional-Run Chunking therefore predicts choices favoring B. It additionally predicts a nonlinear suffix-length profile: the singleton B suffix should be weakest, adding a second B cue should produce a marked increase through run completion and suffix replacement, and terminal runs of two through four cues should be relatively similar because run confidence saturates rapidly. Length, discriminating-cue count, and switch count are fixed at six, six, and one, respectively. In contrast, the competing theory scans in descending validity, where Experts 2 and 1 are always the final two attended discriminating cues and both favor A on every base trial. Two near-certain terminal overwrites should thus yield a strong, approximately condition-invariant A preference; common six-cue fatigue only attenuates that preference. The decisive dissociation is therefore opposite modal choices, supplemented by a singleton-versus-repeated-suffix increase and a two-to-four-cue plateau under chunking versus a nearly flat profile under overwrite. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times produce 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Saturated Directional-Run Chunking with Boundary Interference. People scan discriminating cues in their instructed order and compress consecutive cues favoring the same option into a single directional chunk. Chunk confidence rises rapidly with run length but saturates, so two agreeing terminal cues can be nearly as effective as a much longer terminal run. At each directional boundary, the newly completed chunk suppresses the accumulated representation of earlier chunks. A coherent terminal run receives an additional replacement advantage, causing stimuli with the same strong suffix to converge despite differences in prefix segmentation. Short four-reason sequences preserve run structure at high temporal resolution and express terminal chunks with a heterogeneous gain, producing strong but individually variable terminal control. In longer sequences, switch-dense patterns suffer source interference, while long low-switch conflicts undergo a separate modest loss of response precision without recovering early high-validity cues. Instructed validity only modestly scales chunk encoding and cannot become the retrieval asymptote. Choices arise from recency-weighted competition among run summaries plus a small response lapse.

**Parameters:**
- validities: validities
- run_saturation: [1.8, 3.2]
- boundary_suppression: [0.35, 0.55]
- chunk_recency_decay: [0.78, 0.94]
- short_resolution_boost: [0.12, 0.68]
- suffix_replacement: [0.08, 0.25]
- four_reason_expression_gain: [0.35, 2.60]
- switch_interference: [1.8, 2.8]
- length_interference: [0.06, 0.12]
- long_low_switch_loss: [0.08, 0.34]
- validity_encoding_scale: [0.02, 0.20]
- choice_gain: [2.6, 3.8]
- lapse_rate: [0.04, 0.14]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Directional-run chunking expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    saturation = float(parameters["run_saturation"])
    suppression = float(parameters["boundary_suppression"])
    recency_decay = float(parameters["chunk_recency_decay"])
    short_boost = float(parameters["short_resolution_boost"])
    suffix_replacement = float(parameters["suffix_replacement"])
    four_gain = float(parameters["four_reason_expression_gain"])
    switch_interference = float(parameters["switch_interference"])
    length_interference = float(parameters["length_interference"])
    long_low_switch_loss = float(parameters["long_low_switch_loss"])
    validity_scale = float(parameters["validity_encoding_scale"])
    choice_gain = float(parameters["choice_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_indices = np.flatnonzero(diff != 0)
    m = int(cue_indices.size)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.sign(diff[cue_indices]).astype(float)
    cue_validities = np.clip(validities[cue_indices], 0.5, 1.0)

    # Validity changes encoding strength only modestly. Every discriminating
    # cue has a substantial baseline representation, preventing overload from
    # turning retrieval into a validity-biased early-cue process.
    encoded_strengths = 1.0 + validity_scale * (2.0 * cue_validities - 1.0)

    # Compress contiguous cues with a common direction into run summaries.
    runs = []
    start = 0
    for i in range(1, m + 1):
        if i == m or directions[i] != directions[start]:
            direction = float(directions[start])
            encoding_mass = float(np.sum(encoded_strengths[start:i]))
            # Repetition increases chunk confidence, but confidence rapidly
            # saturates rather than growing as an unbounded cue tally.
            confidence = 1.0 - np.exp(-saturation * encoding_mass)
            runs.append((direction, confidence, i - start))
            start = i

    n_switches = max(0, len(runs) - 1)

    # Compete run summaries online. A new coherent chunk both enters memory
    # and suppresses prior chunks. Short sequences retain sharper boundary
    # resolution, yielding especially effective terminal replacement.
    directional_memory = 0.0
    represented_mass = 0.0
    for run_number, (direction, confidence, run_length) in enumerate(runs):
        if run_number == 0:
            directional_memory = direction * confidence
            represented_mass = confidence
            continue

        effective_suppression = suppression
        if m <= 4:
            effective_suppression += short_boost
        # A repeated-direction suffix is encoded as a completed terminal
        # decision chunk and therefore more fully replaces prefix summaries.
        if run_number == len(runs) - 1 and run_length >= 2:
            effective_suppression += suffix_replacement
        effective_suppression = float(np.clip(effective_suppression, 0.0, 0.98))

        retention = recency_decay * (1.0 - effective_suppression * confidence)
        retention = float(np.clip(retention, 0.0, 1.0))
        directional_memory = retention * directional_memory + direction * confidence
        represented_mass = retention * represented_mass + confidence

    if represented_mass <= 0.0 or not np.isfinite(represented_mass):
        normalized_direction = 0.0
    else:
        normalized_direction = float(
            np.clip(directional_memory / represented_mass, -1.0, 1.0)
        )

    # Switch-dense sequences lose directional source resolution. Generic
    # length uncertainty remains mild, while long low-switch conflicts have
    # a separate expression loss that attenuates extremity without restoring
    # privileged access to early cues.
    excess_switches = max(0, n_switches - 2)
    low_switch_excess = max(0, m - 5) if n_switches <= 2 else 0
    precision_denominator = (
        1.0
        + switch_interference * float(excess_switches ** 2)
        + length_interference * float(max(0, m - 4))
        + long_low_switch_loss * float(low_switch_excess)
    )
    effective_gain = choice_gain / precision_denominator

    # Four-reason displays preserve unusually high temporal resolution. The
    # broad subject-level gain range captures heterogeneity in expressing the
    # terminal chunk without changing longer conflicts.
    if m == 4:
        effective_gain *= four_gain

    logit_a = float(np.clip(effective_gain * normalized_direction, -60.0, 60.0))
    p_a_core = 1.0 / (1.0 + np.exp(-logit_a))
    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)

    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
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
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

**Parameters:**
- validities: validities
- cue_attention: [0.95, 1.0]
- overwrite_rate: [0.95, 1.0]
- update_fatigue: [0.06, 0.17]
- lapse_rate: [0.15, 0.41]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recency-Gated Reason Overwrite expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["cue_attention"])
    overwrite = float(parameters["overwrite_rate"])
    fatigue = float(parameters["update_fatigue"])
    lapse = float(parameters["lapse_rate"])

    # Stable sorting represents the displayed high-to-low validity scan;
    # ties retain their display order.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]

    # Exact marginal distribution over the provisional state:
    # A selected, B selected, or no discriminating cue encoded yet.
    p_a, p_b, p_uncommitted = 0.0, 0.0, 1.0
    discriminating_count = 0

    for j in cue_order:
        direction = diff[j]
        if direction == 0:
            continue

        discriminating_count += 1
        old_a, old_b, old_u = p_a, p_b, p_uncommitted

        if direction > 0:
            # An attended A cue initializes an uncommitted decision and
            # overwrites a conflicting B decision with the overwrite rate.
            p_a = old_a + attention * overwrite * old_b + attention * old_u
            p_b = old_b * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)
        else:
            p_b = old_b + attention * overwrite * old_a + attention * old_u
            p_a = old_a * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)

    # If no discriminating cue was encoded, either response is an equal guess.
    p_core = np.array([
        p_a + 0.5 * p_uncommitted,
        p_b + 0.5 * p_uncommitted
    ], dtype=np.float64)

    # Processing the first reason and its immediate successor is reliable.
    # Additional discriminating updates progressively weaken expression of
    # the retained choice without disrupting the crucial second-cue reversal.
    excess_updates = max(0, discriminating_count - 2)
    retention = (1.0 - fatigue) ** excess_updates
    p_core = retention * p_core + (1.0 - retention) * np.array([0.5, 0.5])

    # Independent execution/inattention lapse.
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    if len(data) == 0:\n        return float('nan')\n\n    agreements = []\n    for _, row in data.iterrows():\n        a = np.asarray(row['option_a_ratings'], dtype=float)\n        b = np.asarray(row['option_b_ratings'], dtype=float)\n        diff = a - b\n        discriminating = np.flatnonzero(diff != 0)\n        if discriminating.size == 0:\n            continue\n\n        # Direction of the terminal discriminating cue in displayed order.\n        terminal_direction = diff[discriminating[-1]]\n        terminal_response = 0 if terminal_direction > 0 else 1\n        agreements.append(float(int(row['response']) == terminal_response))\n\n    if len(agreements) == 0:\n        return float('nan')\n    return float(np.mean(agreements))",
  "rationale": "Terminal displayed-cue agreement pools all 96 trials into a low-variance, side-balanced diagnostic. Directional-run chunking predicts that the terminal displayed run controls choice, so this proportion should be well above 0.5. Overwrite scans by descending validity instead; its final attended cues are Experts 2 and 1, which favor the option opposite the terminal displayed cue on every designed pair, so it predicts a value well below 0.5. Exact A/B reversals prevent a generic response-side preference from creating the contrast."
}
```

## Usage

```json
{
  "input_tokens": 5175,
  "output_tokens": 1362
}
```
