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
**Validities (n_features=12):** [1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 10: A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 11: A=[1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 12: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Rationale:** All 12 unique pairs are unweighted tally ties, d = sum(A - B) = 0, so both fallback gates open. The validities define a three-cue elite: the first three experts have validity 1.0 and the remaining nine have 0.8, giving a gap of 0.20 and top_size = 3, which meets both theories' small-elite conditions. The first three features define T = #top A wins - #top B wins among positions 0-2; the last three define L = #late A wins - #late B wins among positions 9-11. Middle features 3-8 are tally-balancing fillers. For the advocated model, combined relevance = validity * exp(rho * (position - max_position)) makes the final three positions dominate every earlier 0.8 cue for rho in [0.60, 1.00], so its restricted block is positions 9-11 and its weighted sub-tally is exactly +3 on L=+3 trials and -3 on L=-3 trials. The competing model ignores position and uses the three 1.0 experts, so its sub-tally equals T. The design is a 4 x 2 factorial over T in {+3,+1,-1,-3} and L in {+3,-3}; trials 1-8 are sign-reversal cells. Critical predictions: in T=+3,L=-3 cells, competing p(A) is roughly 0.85-0.91 while advocated p(A) is roughly 0.03-0.07; in T=-3,L=+3 cells these reverse. Same-sign cells provide magnitude controls. A validity-only fallback cannot produce an L effect within a fixed T, and a recency-weighted fallback cannot produce a T effect within a fixed L, giving a quantitative double dissociation. The 12 unique pairs are repeated 8 times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People choose by a confidence-gated unweighted signed tally. On clear tallies, choice follows the tally through a moderate softmax plus a small lapse. When the tally is tied or nearly tied, people may switch to a restricted sub-tally only if the advertised cue validities themselves flag a small coherent elite (two to three cues separated from the rest, or a large validity gap with at most six top cues). Critically, that restricted block is not defined by validity alone: cues are re-ranked by a mixture of advertised validity and serial display position, where later cues receive exponentially greater recency weight. This allows late relevant cues to dominate early equally-valid cue blocks, producing reversals like Experiment 10 while preserving validity use in Experiments 3 and 4. Large-feature environments with small tallies are treated as low confidence and heavily regularized with near-chance lapse, preventing deterministic regression blow-ups.

**Parameters:**
- beta_tally: [0.6, 1.0]
- epsilon_tally: [0.02, 0.08]
- beta_near: [0.3, 0.6]
- epsilon_near: [0.80, 0.97]
- near_tally_bound: {1.5}
- n_low_conf: {15}
- tau_tie: {0.5}
- top_tol: {0.001}
- small_size_min: {2}
- small_size_max: {3}
- gap_lo: [0.05, 0.10]
- gap_hi: [0.30, 0.45]
- block_max: {6}
- rho: [0.60, 1.00]
- beta_sub: [0.60, 1.00]
- epsilon_sub: [0.05, 0.10]
- validities: validities

**`predict source code`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length mismatch.')

    a = stim[0]
    b = stim[1]
    diff = a - b
    tally = float(np.sum(diff))
    positions = np.arange(n_features, dtype=float)

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        return e / np.sum(e)

    def lapse_mix(core, eps):
        probs = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
        probs = np.clip(probs, 0.005, 0.995)
        total = float(np.sum(probs))
        if total <= 1e-12:
            return np.array([0.5, 0.5])
        return probs / total

    vmax = float(np.max(validities))
    top_tol = float(parameters['top_tol'])
    top_mask = validities >= (vmax - top_tol)
    top_size = int(np.sum(top_mask))

    if top_size < n_features:
        second_max = float(np.max(validities[~top_mask]))
    else:
        second_max = vmax
    validity_gap = vmax - second_max

    gap_hi = float(parameters['gap_hi'])
    gap_lo = float(parameters['gap_lo'])
    size_min = int(parameters['small_size_min'])
    size_max = int(parameters['small_size_max'])
    block_max = int(parameters['block_max'])

    small_elite = (size_min <= top_size <= size_max) and (validity_gap >= gap_lo)
    block_clear = (validity_gap >= gap_hi) and (top_size <= block_max)
    salience = 1.0 if (small_elite or block_clear) else 0.0

    tau_tie = float(parameters['tau_tie'])

    if salience > 0.5 and abs(tally) <= tau_tie:
        rho = float(parameters['rho'])

        # Stable recency-weighted cue relevance. Subtracting the max position
        # prevents overflow while preserving the ranking and normalized weights.
        combined = np.asarray(validities, dtype=float) * np.exp(
            rho * (positions - positions.max())
        )

        order = np.argsort(-combined)
        selected = order[:top_size]
        if selected.size == 0:
            return np.array([0.5, 0.5])

        sel_weights = combined[selected]
        denom = float(np.sum(sel_weights))
        if denom <= 1e-12:
            weights = np.ones_like(sel_weights)
        else:
            weights = sel_weights / denom * float(selected.size)

        sub_tally = float(np.dot(weights, diff[selected]))

        beta_sub = float(parameters['beta_sub'])
        eps_sub = float(parameters['epsilon_sub'])

        if abs(sub_tally) > 1e-12:
            core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_sub)
            return lapse_mix(core, eps_sub)

        return np.array([0.5, 0.5])

    near_bound = float(parameters['near_tally_bound'])
    n_low_conf = int(parameters['n_low_conf'])

    if abs(tally) <= near_bound and n_features >= n_low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = float(parameters['beta_tally'])
        eps = float(parameters['epsilon_tally'])

    core = stable_softmax(np.array([tally, -tally]), beta)
    return lapse_mix(core, eps)
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** Tally-first choice with a salience-gated and size-regularized validity tie-break. On every trial the unweighted signed tally of the feature differences is the default choice signal. When the tally is clear, choice follows it through a moderate softmax plus an explicit lapse. When the tally is tied or near-tied, validity is allowed to matter only if the task's validity distribution itself advertises a clear elite: either a small elite block of two to three maximally valid cues separated from the remaining cues, or a large validity gap combined with a bounded elite-block size. In those gated cases people compute a restricted sub-tally over the elite cues and choose its favored option stochastically. Larger elite blocks receive an additional lapse, preventing overconfident validity-following for wide top tiers. In large stimulus arrays, near-zero tallies are treated as low-confidence and are heavily regularized with extra choice noise, so deterministic regression blows-up do not occur.

**Parameters:**
- beta_tally: [2.0, 4.0]
- epsilon_tally: [0.02, 0.08]
- beta_near: [0.3, 0.8]
- epsilon_near: [0.78, 0.97]
- near_tally_bound: {1.5}
- n_low_conf: {15}
- tau_tie: [0.0, 0.5]
- salience_gap_hi: [0.30, 0.45]
- salience_gap_lo: [0.04, 0.06]
- top_tol: {0.01}
- top_size_small_min: {2}
- top_size_small_max: {3}
- top_size_block_max: {6}
- beta_sub: [0.35, 0.50]
- epsilon_sub: [0.08, 0.14]
- epsilon_sub_size: [0.16, 0.20]
- validities: validities

**`predict source code`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length mismatch.')

    a = stim[0]
    b = stim[1]
    diff = a - b
    tally = float(np.sum(diff))

    vmax = float(np.max(validities))
    top_tol = float(parameters['top_tol'])
    top_mask = validities >= (vmax - top_tol)
    top_size = int(np.sum(top_mask))

    if top_size < n_features:
        second_max = float(np.max(validities[~top_mask]))
    else:
        second_max = vmax
    validity_gap = vmax - second_max

    gap_hi = float(parameters['salience_gap_hi'])
    gap_lo = float(parameters['salience_gap_lo'])
    size_min = int(parameters['top_size_small_min'])
    size_max = int(parameters['top_size_small_max'])
    block_max = int(parameters['top_size_block_max'])

    block_clear = (validity_gap >= gap_hi) and (top_size <= block_max)
    small_elite = (size_min <= top_size <= size_max) and (validity_gap >= gap_lo)
    salience = 1.0 if (block_clear or small_elite) else 0.0

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        return e / np.sum(e)

    tau_tie = float(parameters['tau_tie'])

    if salience > 0.5 and abs(tally) <= tau_tie:
        top_diff = diff[top_mask]
        sub_a = float(np.sum(top_diff > 0.0))
        sub_b = float(np.sum(top_diff < 0.0))
        sub_tally = sub_a - sub_b

        beta_sub = float(parameters['beta_sub'])
        eps_sub = float(parameters['epsilon_sub'])
        eps_sub_size = float(parameters['epsilon_sub_size'])
        eps_sub_eff = eps_sub + eps_sub_size * max(0.0, float(top_size) - 3.0)
        eps_sub_eff = float(np.clip(eps_sub_eff, 0.0, 0.95))

        if abs(sub_tally) > 1e-12:
            core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_sub)
            probs = (1.0 - eps_sub_eff) * core + eps_sub_eff * np.array([0.5, 0.5])
        else:
            probs = np.array([0.5, 0.5])
        return probs

    near_bound = float(parameters['near_tally_bound'])
    n_low_conf = int(parameters['n_low_conf'])

    if abs(tally) <= near_bound and n_features >= n_low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = float(parameters['beta_tally'])
        eps = float(parameters['epsilon_tally'])

    core = stable_softmax(np.array([tally, -tally]), beta)
    probs = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        return np.array([0.5, 0.5])
    return probs / total
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    chose_a_c1 = []\n    chose_a_c2 = []\n\n    for _, row in data.iterrows():\n        a = [int(v) for v in row['option_a_ratings']]\n        b = [int(v) for v in row['option_b_ratings']]\n        diff = [a[j] - b[j] for j in range(len(a))]\n\n        t_top = int(sum(diff[:3]))\n        l_late = int(sum(diff[-3:]))\n        chose_a = 1.0 if int(row['response']) == 0 else 0.0\n\n        if t_top == 3 and l_late == -3:\n            chose_a_c1.append(chose_a)\n        elif t_top == -3 and l_late == 3:\n            chose_a_c2.append(chose_a)\n\n    if len(chose_a_c1) == 0 or len(chose_a_c2) == 0:\n        return 0.0\n\n    return float(np.mean(chose_a_c2) - np.mean(chose_a_c1))",
  "rationale": "This metric measures the double-dissociation reversal contrast using the two critical sign-reversal cells: top tally T=+3 with late tally L=-3, and top tally T=-3 with late tally L=+3. All trials are unweighted tally ties, so both theories enter their gated sub-tally path rather than using the full tally. The validity-only theory follows the three high-validity early cues, producing a high A-choice rate when T=+3 and a low A-choice rate when T=-3. The advocated recency-weighted theory follows the late three cues, producing the opposite pattern. Subtracting the A-choice rate in the T=+3,L=-3 cell from the A-choice rate in the T=-3,L=+3 cell therefore yields approximately +0.9 under the advocated theory and approximately -0.8 under the competing theory. Each subject provides 16 trials per critical cell, so the binomial variability of this contrast is modest, giving a large and stable between-theory separation."
}
```

## Usage

```json
{
  "input_tokens": 6330,
  "output_tokens": 12582,
  "reasoning_tokens": 12083,
  "provider": "Baidu"
}
```
