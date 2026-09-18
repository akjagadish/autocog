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
**Validities (n_features=6):** [0.72, 0.85, 0.55, 0.78, 0.6, 0.66]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 4: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 10: A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 11: A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 12: A=[1, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  trial 13: A=[0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  trial 14: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 15: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate Take-The-Best (one-reason, validity-ordered, all-or-none confidence) from Absolute-Evidence Weighted Integration (AEWI: unnormalised log-odds^gamma weights, softmax in absolute evidence units). Both are *weight-order* consistent, so a Tallying-style design will not separate them; what separates them is (a) the SIGN flip AEWI produces when several mid/low cues jointly outweigh the top cue in absolute log-odds units, and (b) the GRADED magnitude of choice consistency that AEWI predicts across pairs where TTB necessarily predicts one constant probability sigmoid(beta).

ENVIRONMENT: six experts with a deliberately COMPRESSED top (0.85) and a long tail (0.78, 0.72, 0.66, 0.60, 0.55), displayed in scrambled order [0.72, 0.85, 0.55, 0.78, 0.60, 0.66] so that no 'use the first-listed expert' strategy can mimic TTB and genuine validity re-sorting is required. Log-odds diagnosticities are d = (0.944, 1.735, 0.201, 1.266, 0.406, 0.663); at the theory's own gamma range (1.355-1.475) the AEWI weights are approx w(.85)=2.11-2.25, w(.78)=1.38-1.42, w(.72)=0.92-0.93, w(.66)=0.55-0.57, w(.60)=0.26-0.29, w(.55)=0.09-0.11. Crucially the top cue's weight is LESS than the sum of any two-or-more of the next cues, so AEWI reverses TTB's choice on most conflict trials, while TTB (which only reads validity RANK) is unaffected.

TRIAL FAMILIES (16 unique pairs, each mirrored across A/B so response bias cannot mimic either model; K = 6 repetitions, 96 trials):
(1) GRADED TOP-CUE CONFLICTS (pairs 1-8). The 0.85 expert favors one option while 1, 2, 3 or all 5 lower experts favor the other. TTB predicts the top-cue option with a CONSTANT probability sigmoid(beta) in every case. AEWI predicts a steep monotone collapse and a sign reversal: p(top-cue option) approx 0.83 (1 opposing), 0.43 (2 opposing), 0.20 (3 opposing), 0.10 (5 opposing). This single curve is the primary discriminator: it contains both an ordinal reversal and a within-family gradient that TTB forbids. The 2-opposing case sits essentially at the AEWI zero-margin point (0.41-0.46 across the gamma range), which is also where AEWI uniquely predicts maximal between-subject heterogeneity.
(2) TTB-DESCENT CONFLICT (pairs 9-10). The 0.85 cue ties, so TTB descends to 0.78 and picks that option; the remaining four cues (0.72, 0.66, 0.60, 0.55) all oppose. AEWI margin is -0.41 to -0.53, i.e. p(TTB option) approx 0.26-0.31. Opposite-sign prediction with the deciding cue no longer being the visually salient best expert.
(3) ABSOLUTE-MARGIN / CONFIDENCE SCALING (pairs 11-16), direction-free. Pairs 11-12: ONLY the 0.55 expert discriminates (all other cues tie) - TTB must be as decisive here as anywhere (sigmoid(beta)), AEWI predicts near chance (approx 0.55). Pairs 13-14: ONLY the 0.85 expert discriminates - AEWI approx 0.99. Pairs 15-16: unanimous 6-0 - AEWI approx 1.00. Because pairs 13-16 pin TTB's beta at a high value, TTB is then forced to predict that same high value on the 0.55-only pairs and on all conflict pairs; any observed monotone dependence of choice consistency on the absolute weighted margin is fatal for TTB and is exactly AEWI's core claim.

DIAGNOSTIC SUMMARY: TTB predicts one number (roughly constant agreement with the top discriminating cue) across pairs 1,3,5,7,9,11,13,15; AEWI predicts the ordered sequence approx (0.83, 0.43, 0.20, 0.10, 0.28, 0.55, 0.99, 1.00). No parameter setting of either model can imitate the other's pattern on this set.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** **Absolute-Evidence Weighted Integration (log-odds^gamma diagnosticity weights, no divisive normalization).**

People integrate ALL stated expert ratings, but with diagnosticity weights derived from the *magnitude* of each expert's stated validity: d_i = log(v_i/(1-v_i)) (the normative evidence a binary cue carries), passed through a non-compensatoriness exponent gamma, w_i = d_i^gamma. Evidence for option A is E = sum_i w_i (a_i - b_i), converted to choice by a softmax with inverse temperature beta plus a lapse epsilon. The family nests both classical heuristics as limits: gamma -> 0 gives equal weights (Tallying); gamma = 1 gives naive-Bayes / Franklin's rule; gamma -> infinity collapses weight onto the single most valid discriminating expert (Take-The-Best).

The crucial claim is that evidence is accumulated in ABSOLUTE log-odds-derived units: the weight vector is NOT renormalised to sum to one, and decision noise (beta) is constant in those absolute units. Consequently the *total diagnosticity* of an environment, D = sum_i d_i^gamma, governs how decisively choices are made. Three consequences follow that no rank-based rule and no scale-invariant (normalised) weighted-additive rule can produce:

(1) Environment-dependent non-compensatoriness. A structurally identical conflict - top cue versus 4-5 opposing lower cues - is resolved MORE top-cue-consistently in an environment containing a dominant, highly valid expert (e.g. .93 alongside .80) than in a smoothly graded environment topping out at .90, because the former yields larger absolute margins. Humans show exactly this asymmetry (~81% top-cue agreement in the 6-cue .93 environment vs ~74% in the graded 5-cue one).

(2) Graded, sub-ceiling conflict agreement. In the graded environment the absolute margin for 'top cue vs the four lower cues' is small, so agreement is only ~0.62-0.69, while 'top two cues vs the bottom three' is near-deterministic (~1.0) - a within-family gradient in the weighted margin that TTB forbids and that produces sub-ceiling pooled agreement without a large lapse rate.

(3) Environment-dependent heterogeneity. Because the graded environment's pivotal margins sit near zero, individual differences in gamma translate into large between-subject differences in conflict agreement there, whereas the dominant-cue environment keeps every subject on the top-cue side - predicting larger between-subject variance in the graded environment specifically.

It further predicts near-deterministic choice on unanimous (all-cue) pairs, and near-chance responding on pairs decided only by a .55-validity expert, since those margins are small in absolute units.

**Parameters:**
- gamma: [1.355, 1.475]
- beta: [1.7, 2.3]
- epsilon: [0.0, 0.03]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Absolute-Evidence Weighted Integration.
    # Weights come from the *magnitudes* of the stated validities:
    #     d_i = log(v_i / (1 - v_i))          (log-odds diagnosticity)
    #     w_i  =  d_i ** gamma                 (gamma = non-compensatoriness)
    # gamma -> 0  : equal weights            == Tallying
    # gamma  = 1  : log-odds weights         == naive Bayes / Franklin's rule
    # gamma -> inf: top cue dominates        == Take-The-Best
    # Evidence  E = sum_i w_i * (a_i - b_i)  -> softmax(beta * E) -> lapse.
    # The weight vector is NOT renormalised to sum to 1.  Evidence lives in
    # absolute log-odds units, so environments with greater total
    # diagnosticity (sum_i d_i**gamma) yield larger margins and hence more
    # decisive, more top-cue-consistent choices.
    # History is ignored (no feedback is given in this domain).
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        # Flat vector: assume [A features..., B features...]
        if stim.shape[0] % 2 != 0:
            return np.ones(2) / 2.0
        half = stim.shape[0] // 2
        stim = np.vstack([stim[:half], stim[half:]])
    if stim.ndim != 2 or stim.shape[0] != 2:
        return np.ones(2) / 2.0

    n_features = int(stim.shape[1])
    if n_features == 0:
        return np.ones(2) / 2.0

    # --- validities -------------------------------------------------
    val = parameters.get("validities", None)
    try:
        v = np.asarray(val, dtype=float).ravel()
    except Exception:
        v = np.array([])
    if v.size != n_features or not np.all(np.isfinite(v)):
        v = np.linspace(0.90, 0.55, n_features)
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # --- diagnosticity weights (ABSOLUTE, un-normalised) ------------
    gamma = float(parameters["gamma"])
    d = np.log(v / (1.0 - v))              # >= 0, larger = more diagnostic
    d = np.maximum(d, 1e-12)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        w = np.power(d, gamma)
    w = np.where(np.isfinite(w), w, 0.0)
    tot = float(np.sum(w))
    if tot <= 0.0 or not np.isfinite(tot):
        w = np.ones(n_features)
    # NOTE: deliberately no `w = w / tot` here.  The absolute scale of
    # the weights is the theoretical content of this model.

    # --- weighted-additive evidence ---------------------------------
    a = stim[0]
    b = stim[1]
    E = float(np.dot(w, a - b))
    if not np.isfinite(E):
        E = 0.0

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over [beta*E, 0]  (== sigmoid(beta*E) for A)
    z = np.array([beta * E, 0.0], dtype=float)
    z = np.clip(z, -700.0, 700.0)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        p = np.ones(p.shape[0]) / p.shape[0]
    else:
        p = p / s
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    if len(data) == 0:\n        return 0.0\n\n    A = np.array([[int(x) for x in r] for r in data['option_a_ratings']], dtype=float)\n    B = np.array([[int(x) for x in r] for r in data['option_b_ratings']], dtype=float)\n    resp = np.asarray(data['response'], dtype=int)\n    n_feat = A.shape[1]\n\n    val = np.array([0.72, 0.85, 0.55, 0.78, 0.60, 0.66], dtype=float)\n    if val.shape[0] != n_feat:\n        val = np.linspace(0.90, 0.55, n_feat)\n    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-6)\n\n    order = np.argsort(-val, kind='stable')\n\n    # log-odds diagnosticity weights with a mid-range gamma (reference AEWI weights)\n    d = np.log(val / (1.0 - val))\n    w = np.power(np.maximum(d, 1e-12), 1.4)\n    beta_ref = 2.0\n\n    diff = A - B  # +1 favours A on that cue, -1 favours B\n\n    # Take-The-Best winner per trial: first discriminating cue in validity order\n    n = A.shape[0]\n    ttb = np.full(n, -1, dtype=int)  # 0 = A, 1 = B, -1 = no discrimination\n    undecided = np.ones(n, dtype=bool)\n    for j in order:\n        dj = diff[:, j]\n        pickA = undecided & (dj > 0)\n        pickB = undecided & (dj < 0)\n        ttb[pickA] = 0\n        ttb[pickB] = 1\n        undecided = undecided & (dj == 0)\n\n    valid = ttb >= 0\n    if not np.any(valid):\n        return 0.0\n\n    # Signed absolute-evidence margin IN FAVOUR OF the TTB-chosen option\n    E = diff.dot(w)\n    sign_ttb = np.where(ttb == 0, 1.0, -1.0)\n    margin = E * sign_ttb\n    p_aewi = 1.0 / (1.0 + np.exp(-beta_ref * margin))\n\n    agree = (resp == ttb).astype(float)\n\n    high = valid & (p_aewi > 0.65)   # integration and one-reason agree strongly\n    low = valid & (p_aewi < 0.35)    # integration reverses the one-reason choice\n\n    if high.sum() == 0 or low.sum() == 0:\n        return 0.0\n\n    return float(agree[high].mean() - agree[low].mean())",
  "rationale": "Both models are compared on the SAME quantity: agreement with the Take-The-Best choice (the option favoured by the highest-validity discriminating expert). Trials are split, using the design's own structure, into (i) pairs where an absolute-evidence integrator's margin also strongly favours the TTB option (single-cue-discriminating .85 pairs, unanimous pairs, and the 1-opposing-cue conflicts) and (ii) pairs where an integrator's summed log-odds margin REVERSES the TTB choice (3-opposing, 5-opposing, and the TTB-descent pairs). Ambiguous near-zero-margin pairs are dropped. The metric is the within-subject difference in TTB-agreement between these two trial sets. Under Take-The-Best only the RANK of the first discriminating cue matters, so agreement is the same constant sigmoid(beta)-based value in both sets for every subject regardless of beta and epsilon: the difference is 0 up to binomial sampling noise (36 trials per set, sd ~0.10). Under AEWI the difference is large and positive (~0.94 vs ~0.19, i.e. ~0.75) and is insensitive to its gamma/beta ranges. Because the contrast is taken within subject, individual differences in noise/lapse cancel for TTB, giving a huge mean gap with tight per-subject variance in both theories."
}
```

## Usage

```json
{
  "input_tokens": 9044,
  "output_tokens": 5995
}
```
