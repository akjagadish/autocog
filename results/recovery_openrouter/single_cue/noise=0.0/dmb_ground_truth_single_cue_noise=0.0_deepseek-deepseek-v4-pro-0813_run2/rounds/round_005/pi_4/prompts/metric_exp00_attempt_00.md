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
**Validities (n_features=6):** [0.9, 0.8, 0.72, 0.61, 0.55, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 9: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 11: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 12: A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 13: A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 14: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 15: A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 16: A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 17: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 18: A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 19: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 20: A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 21: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 22: A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 24: A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 25: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 26: A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 27: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 28: A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 29: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 30: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 31: A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 32: A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0]

**Rationale:** This design crosses two factors that the conflict-gated model treats as interacting but the stable-mixture model treats as irrelevant: (i) a gate toggle achieved by tying the highest-validity cue's rating while the second-highest cue carries the same direction, and (ii) the feature-win tally difference d. Validities are [0.90, 0.80, 0.72, 0.61, 0.55, 0.50]; feature 0 is the unique highest cue and feature 5 is the unique lowest cue. On every trial feature 5 discriminates opposite the top-direction winner. In gate-active trials feature 0 itself discriminates in the top direction; in matched gate-inactive trials feature 0 is tied and feature 1 discriminates in the same top direction. This keeps the forward lexicographic winner, backward lexicographic winner, and tally difference d identical across matched pairs, so the stable-mixture model must predict identical P(A). The conflict-gated model, however, is live only when the top cue discriminates and tally does not oppose it: top-A d>=0 and top-B d<=0. There it suspends tallying and boosts backward scanning, producing a large gate gap; when d opposes the top cue (top-A d=-1, top-B d=+1), the gate is off and the gap disappears. With representative parameters (advocated: w_forward=.27, w_backward=.48, beta_forward=.14, beta_backward=1.5, beta_tally=2.2; competing: w_forward=.25, w_backward=.48, beta_forward=.30, beta_backward=1.5, beta_tally=2.2, w_conflict_back=.85, beta_conflict_back=2.5), top-A d=+1 gives advocated P(A)=.457 in both gate conditions, whereas competing gives .474 gate-inactive but .151 gate-active; top-B d=-1 gives advocated .543 in both, competing .526 gate-inactive but .849 gate-active. Top-A d=-1 and top-B d=+1 are controls where both models predict no gate gap. The 32 unique pairs are balanced over top direction, gate condition, d, and tie value, repeated K=3 for 96 trials.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People choose between two options through a stable, subject-level mixture of three comparison routes: (1) forward lexicographic scanning, inspecting cues in descending validity order and stopping at the first discriminating cue; (2) backward lexicographic scanning, inspecting cues in ascending validity order and stopping at the first discriminating cue in that reversed order; and (3) feature-win tallying, where each option receives one point per feature on which it strictly beats the other and ties contribute nothing. The forward and backward lexicographic weights jointly act as a subject-level polarity parameter for first-validity-cue effects: forward weight supports a positive first-discriminator effect, while backward weight supports a negative one. Each route's preferred option is passed through its own softmax choice rule, so responding is noisy but not controlled by a uniform lapse.

**Parameters:**
- w_forward: [0.22, 0.32]
- w_backward: [0.42, 0.54]
- beta_forward: [0.08, 0.20]
- beta_backward: [1.30, 1.70]
- beta_tally: [1.80, 2.60]
- epsilon: {0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Preferred validity vector comes from the experiment.  If the
    # parameter resolver ever leaves it unresolved, fall back to a
    # descending validity vector so the model is still scorable.
    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.ones(2, dtype=float) / 2.0

        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People compare options through a conflict-gated dual-lexicographic/tallying system. A baseline mixture combines a weak forward Take-The-Best route, a moderate backward lexicographic route, and feature-win tallying. When the highest-validity and lowest-validity cues discriminate in opposite directions and the tally does not oppose the highest cue, the decision maker enters a conflict regime: tallying is suspended and the choice is made with high probability by a strong backward lexicographic scan over ascending validities. Subjects differ substantially in how strongly and deterministically they reverse on such conflict trials.

**Parameters:**
- w_forward: [0.17, 0.30]
- w_backward: [0.36, 0.60]
- beta_forward: [0.0, 0.6]
- beta_backward: [1.0, 2.5]
- beta_tally: [1.5, 3.0]
- w_conflict_back: [0.75, 0.95]
- beta_conflict_back: [1.8, 4.0]
- epsilon: {0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_conflict_back = float(parameters['beta_conflict_back'])

    p_forward = lex_probabilities(descending, beta_forward)
    p_backward_base = lex_probabilities(ascending, beta_backward)
    p_backward_conflict = lex_probabilities(ascending, beta_conflict_back)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (scores - np.max(scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally_base = 1.0 - w_forward - w_backward
    if w_tally_base < 0.0:
        w_tally_base = 0.0
        denom = w_forward + w_backward
        w_forward /= denom
        w_backward /= denom

    v_max = float(np.max(validities))
    v_min = float(np.min(validities))
    top_positions = np.where(validities == v_max)[0]
    bottom_positions = np.where(validities == v_min)[0]

    if top_positions.size == 1 and bottom_positions.size == 1:
        j_hi = int(top_positions[0])
        j_lo = int(bottom_positions[0])
        hi_dir = 0
        if a[j_hi] > b[j_hi]:
            hi_dir = 1
        elif b[j_hi] > a[j_hi]:
            hi_dir = -1
        lo_dir = 0
        if a[j_lo] > b[j_lo]:
            lo_dir = 1
        elif b[j_lo] > a[j_lo]:
            lo_dir = -1
        tally_d = int(a_wins - b_wins)
        extreme_cue_conflict = (hi_dir != 0 and lo_dir != 0 and hi_dir == -lo_dir)
        live_conflict = bool(extreme_cue_conflict and (tally_d * hi_dir >= 0))
    else:
        live_conflict = False

    if live_conflict:
        w_conflict_back = float(parameters['w_conflict_back'])
        w_back_eff = w_conflict_back
        w_forward_eff = 1.0 - w_back_eff
        if w_forward_eff < 0.0:
            w_forward_eff = 0.0
            w_back_eff = 1.0
        w_tally_eff = 0.0
        p_backward = p_backward_conflict
    else:
        w_forward_eff = w_forward
        w_back_eff = w_backward
        w_tally_eff = w_tally_base
        p_backward = p_backward_base

    weights = np.array([w_forward_eff, w_back_eff, w_tally_eff], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.full(2, 0.5, dtype=float)

    p = np.clip(p, 0.0, None)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
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
  "metric_source": "import numpy as np\n\ndef metric(data):\n    if data is None or len(data) == 0:\n        return 0.0\n    validities = np.array([0.9, 0.8, 0.72, 0.61, 0.55, 0.5])\n    order = np.argsort(-validities, kind='stable')\n    hi_idx = int(order[0])\n    second_idx = int(order[1])\n    lo_idx = int(order[-1])\n    active_low = []\n    inactive_low = []\n    for row in data.itertuples(index=False):\n        a = np.asarray(row.option_a_ratings, dtype=float)\n        b = np.asarray(row.option_b_ratings, dtype=float)\n        resp = row.response\n        if a.shape != validities.shape or b.shape != validities.shape:\n            continue\n        if a[lo_idx] == b[lo_idx]:\n            continue\n        low_dir = 1 if a[lo_idx] > b[lo_idx] else -1\n        low_chosen = 1.0 if ((low_dir == 1 and resp == 0) or (low_dir == -1 and resp == 1)) else 0.0\n        tally_d = int(np.sum(a > b) - np.sum(b > a))\n        if a[hi_idx] != b[hi_idx]:\n            hi_dir = 1 if a[hi_idx] > b[hi_idx] else -1\n            if hi_dir == -low_dir and tally_d * hi_dir >= 0:\n                active_low.append(low_chosen)\n        else:\n            if a[second_idx] != b[second_idx]:\n                hi_dir = 1 if a[second_idx] > b[second_idx] else -1\n                if hi_dir == -low_dir and tally_d * hi_dir >= 0:\n                    inactive_low.append(low_chosen)\n    if not active_low or not inactive_low:\n        return 0.0\n    return float(np.mean(active_low) - np.mean(inactive_low))",
  "rationale": "On every designed trial the lowest-validity cue discriminates opposite the highest discriminating cue, so choosing the low-validity-cue option is the backward-lexicographic choice. This metric computes the rate of choosing that low-cue option in two matched sets: gate-active live trials, where the highest cue discriminates and the tally does not oppose it, and gate-inactive live trials, where the highest cue is tied and the second cue discriminates in the same direction. It returns the active rate minus the inactive rate. The stable-mixture theory has identical forward winner, backward winner, and tally difference across matched active-inactive pairs, so the expected value is exactly zero. The conflict-gated theory instead enters a strong backward-lexicographic conflict regime only on gate-active trials, producing a large positive difference, estimated in the range of roughly 0.20 to 0.35."
}
```

## Usage

```json
{
  "input_tokens": 6526,
  "output_tokens": 26096,
  "reasoning_tokens": 25408,
  "provider": "Baidu"
}
```
