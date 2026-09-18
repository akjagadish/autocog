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
**Validities (n_features=7):** [0.74, 0.9, 0.55, 0.7, 0.82, 0.62, 0.78]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0]
  trial 2: A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  trial 9: A=[0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  trial 10: A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  trial 14: A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate Take-The-Best (one-reason, validity-RANK only, all-or-none confidence) from Absolute-Evidence Weighted Integration (AEWI: unnormalised log-odds^gamma weights, softmax in absolute evidence units). Both models are weight-order consistent, so the separation must come from (a) SIGN reversals when two or three mid-validity cues jointly outweigh the top cue in absolute log-odds units, and (b) a GRADED confidence ladder across pairs where TTB is forced to predict a single constant probability sigmoid(beta).

ENVIRONMENT: seven experts, one clearly dominant cue (0.90) plus a dense mid band (0.82, 0.78, 0.74, 0.70) and two weak cues (0.62, 0.55), displayed in SCRAMBLED order [0.74, 0.90, 0.55, 0.70, 0.82, 0.62, 0.78] so no 'use the first-listed expert' rule can imitate TTB and genuine validity re-sorting is required. Log-odds diagnosticities d = (1.046, 2.197, 0.201, 0.847, 1.516, 0.489, 1.266). At the theory's gamma range (1.355-1.475) the AEWI weights are approx w(.90)=2.91-3.19, w(.82)=1.76-1.85, w(.78)=1.38-1.42, w(.74)=1.06-1.07, w(.70)=0.78-0.80, w(.62)=0.36-0.37, w(.55)=0.10-0.11. The band is tuned so that the top cue BEATS any single mid cue and both weak cues combined, is roughly TIED with {0.82+0.74}, is slightly BEATEN by {0.82+0.78}, and is decisively beaten by three mid cues. TTB, reading only rank, is blind to all of these distinctions.

TRIAL FAMILIES (16 unique pairs, every pair mirrored across A/B so response bias cannot mimic either model; K = 6, 96 trials). In every odd-numbered pair TTB chooses A, in every even-numbered pair TTB chooses B.
(1) GRADED TOP-CUE CONFLICTS, spanning the AEWI zero-margin crossing. Pairs 1-2: 0.90 vs {0.62, 0.55} -> AEWI p(top option) ~ 0.99 (agrees with TTB). Pairs 3-4: 0.90 vs {0.82, 0.74} -> AEWI ~ 0.54-0.63 (top side, but barely). Pairs 5-6: 0.90 vs {0.82, 0.78} -> AEWI ~ 0.39-0.47 (sign reversal, and exactly the zero-margin point where AEWI uniquely predicts maximal between-subject heterogeneity driven by gamma). Pairs 7-8: 0.90 vs {0.82, 0.78, 0.74} -> AEWI ~ 0.08 (strong reversal). Across these four conflict types TTB must predict ONE constant number; AEWI predicts a steep monotone collapse through a sign change (0.99, 0.58, 0.43, 0.08).
(2) TTB-DESCENT CONFLICT (pairs 9-10). The 0.90 cue ties (both options rated 1), so TTB descends to 0.82 and picks that option, while 0.78 and 0.74 jointly oppose. AEWI margin is negative (~-0.6 to -0.7), p(TTB option) ~ 0.21: an opposite-sign prediction where the deciding cue is no longer the salient best expert.
(3) ABSOLUTE-MARGIN CONFIDENCE LADDER (pairs 11-16), direction-free (both models pick the same option). Pairs 11-12: ONLY the 0.55 expert discriminates -> AEWI ~ 0.55 (near chance), TTB must be as decisive as anywhere. Pairs 13-14: ONLY the 0.70 expert discriminates -> AEWI ~ 0.83. Pairs 15-16: unanimous 7-0 -> AEWI ~ 1.00. Because the unanimous and 0.99-margin pairs pin TTB's beta at a high value, TTB is then committed to that same high value on the 0.55-only pairs and on every conflict pair; any observed monotone dependence of choice consistency on the absolute weighted margin (0.55 < 0.83 < 1.00) falsifies TTB and is precisely AEWI's core claim.

DIAGNOSTIC SUMMARY: on the odd (TTB-picks-A) pairs 1,3,5,7,9,11,13,15 TTB predicts the same constant p(A); AEWI predicts the ordered sequence approx (0.99, 0.58, 0.43, 0.08, 0.21, 0.55, 0.83, 1.00), including two sign reversals and a near-zero crossing. No parameter setting of either model can reproduce the other's pattern on this set, and the pairs 3-6 crossing region additionally tests AEWI's unique prediction of inflated between-subject variance where the absolute margin is near zero.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Within-subject contrast: TTB-consistency on large-margin AGREEMENT pairs\n    minus TTB-consistency on large-margin CONFLICT pairs (where the weighted\n    log-odds evidence points opposite to the single best discriminating cue).\n    TTB must give the same choice probability in both sets (=> ~0);\n    AEWI predicts ~0.99 vs ~0.15 (=> ~0.85).\"\"\"\n    v = np.array([0.74, 0.90, 0.55, 0.70, 0.82, 0.62, 0.78], dtype=float)\n    d = np.log(v / (1.0 - v))\n    w = np.power(d, 1.4)\n    order = list(np.argsort(-v, kind='stable'))\n\n    agree = []\n    conflict = []\n\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_raw, b_raw, r in zip(a_col, b_col, r_col):\n        try:\n            a = np.asarray(a_raw, dtype=float).ravel()\n            b = np.asarray(b_raw, dtype=float).ravel()\n        except Exception:\n            continue\n        if a.size != 7 or b.size != 7:\n            continue\n        diff = a - b\n        E = float(np.dot(w, diff))\n\n        winner = None\n        for j in order:\n            if diff[j] > 0:\n                winner = 0\n                break\n            if diff[j] < 0:\n                winner = 1\n                break\n        if winner is None:\n            continue\n\n        try:\n            resp = int(r)\n        except Exception:\n            continue\n        cons = 1.0 if resp == winner else 0.0\n\n        sgn = 1.0 if winner == 0 else -1.0\n        signed = E * sgn\n        if signed > 0 and abs(E) >= 1.5:\n            agree.append(cons)\n        elif signed < 0 and abs(E) >= 0.5:\n            conflict.append(cons)\n\n    if len(agree) == 0 or len(conflict) == 0:\n        return 0.0\n    return float(np.mean(agree) - np.mean(conflict))\n",
  "rationale": "Take-The-Best is rank-based and all-or-none: whatever its beta and epsilon, its choice probability for the top-discriminating-cue option is one constant number on EVERY pair where some cue discriminates. Hence any within-subject difference in TTB-consistency between two trial families must be zero for TTB, no matter how noisy or lapsy an individual subject is - this makes the contrast immune to TTB's very wide beta/epsilon ranges, which would wreck any raw accuracy/consistency measure through between-subject variance.\n\nThe two families are selected mechanically from the stimuli (no trial-index hardcoding): (a) AGREEMENT pairs where the AEWI absolute log-odds margin is large (|E| >= 1.5) and points the same way as the top discriminating cue - these are the 0.90-vs-{0.62,0.55} pairs (|E|~2.5) and the unanimous 7-0 pairs (|E|~8.5); (b) CONFLICT pairs where the margin is substantial (|E| >= 0.5) but points AGAINST the top discriminating cue - these are 0.90 vs {0.82,0.78,0.74} (|E|~1.24) and the TTB-descent pairs 0.82 vs {0.78,0.74} (|E|~0.66). Each family contains 24 trials per subject (2 mirrored pair types x 6 reps x 2 A/B orders), so binomial noise per subject is small.\n\nPredictions: AEWI ~0.99 - ~0.15 = ~0.84 with tight per-subject variance (gamma range is narrow, both families sit far from the zero-margin crossing, which I deliberately excluded by the |E| thresholds - pairs 3-6 and 11-14 are dropped precisely because they are the high-variance ones). TTB ~0.00 with only binomial spread (<=0.02 variance). The mirrored A/B design also cancels any side bias."
}
```

## Usage

```json
{
  "input_tokens": 9345,
  "output_tokens": 5591
}
```
