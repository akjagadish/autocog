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
**Validities (n_features=5):** [0.93, 0.84, 0.72, 0.62, 0.54]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 13: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 15: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 16: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** I advocate TAKE-THE-BEST (TTB) against Polarised Weighted Evidence Accumulation (PWEA). PWEA makes two commitments that TTB flatly denies: (i) a rating of '1' is read as a NEGATIVE flag (polarity s<0), so the option carrying the greater validity-weighted mass of 1s is REJECTED; and (ii) choice probability scales smoothly with that weighted mass margin. TTB instead says the option holding the '1' on the highest discriminating expert is chosen, with a single, margin-independent accuracy (1-eps)*sigmoid(beta) on every discriminating trial.

Crucially, the two earlier experiments in this series pitted TTB against Tallying, using only 'conflict' pairs of the form A=[1,0,0,0,0] vs B=[0,1,1,1,1]. On exactly those pairs PWEA MIMICS TTB (the top-cue option also carries the least mass of 1s), so nothing so far can separate them. This design therefore concentrates on the trial class where PWEA and TTB predict OPPOSITE choices: pairs where the option favoured by the top expert also carries MORE 1s.

Five experts with well-spread, strictly ordered validities (0.93, 0.84, 0.72, 0.62, 0.54) make TTB's cue cascade unambiguous. gamma=1 weights (mean-1 normalised) are w=[1.87, 1.48, 0.96, 0.52, 0.17]. All 8 base pairs are shown in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials) so any side bias cancels.

CLASS I - POLARITY REVERSAL, LARGE MARGIN (pairs 1-2 and mirrors 9-10):
[1,1,1,1,1] vs [0,0,0,0,0] (mass 5.00 vs 0) and [1,1,1,0,0] vs [0,0,0,1,1] (4.31 vs 0.70). TTB picks the all-/mostly-1 option essentially deterministically; PWEA picks the OPPOSITE option essentially deterministically for every admissible parameter setting (any gamma, any s<0, beta>=2). These are the hard falsification trials: they cannot be reconciled with PWEA by any parameter tuning, and equally, a reliable preference for the sparse-1 option would kill TTB outright.

CLASS II - GRADED REVERSAL (pairs 3, 4, 6 and mirrors): [1,0,0,0,0] vs [0,0,0,0,0] (PWEA margin 1.87), [0,0,0,0,1] vs [0,0,0,0,0] (margin 0.17 at gamma=1), and [1,0,0,0,0] vs [0,0,0,1,0] (margin 1.35). TTB predicts the SAME accuracy toward the option with the single 1 (or with the higher-validity 1) in all three, because only the sign on the first discriminating cue matters. PWEA predicts the opposite direction with strength graded ~10:1 between the top-cue and bottom-cue trials. So the design delivers a within-subject direction test AND a magnitude-gradient test on the same trial family: flat, above-chance, top-cue-consistent accuracy across margins is uniquely TTB; a margin-graded reversal is uniquely PWEA.

CLASS III - CASCADE / NEAR-TIE TRIAL (pair 7 and mirror): [1,1,0,0,0] vs [1,0,1,1,1]. Expert 1 does not discriminate, so TTB descends to expert 2 and decisively picks A. PWEA's weighted masses are 3.35 vs 3.52 at gamma=1 (and reverse to favour A at gamma<=0.5): across essentially the whole admissible gamma range PWEA is pinned near 50/50 here, whereas TTB predicts full, undiminished accuracy. Systematic choice of A on this pair is diagnostic of one-reason stopping and cannot be produced by PWEA at any evidence sensitivity.

CLASS IV - AGREEMENT / CALIBRATION (pairs 5 and 8, plus mirrors): [1,0,0,0,0] vs [0,1,1,1,1] and [1,0,0,0,0] vs [0,1,1,0,0]. Here both theories favour the same option (the classic TTB-vs-Tallying conflict items on which PWEA mimics TTB). They pin down beta and epsilon for BOTH models independently of the diagnostic classes, so the reversal trials are interpreted against well-identified noise parameters, and they guarantee the design is not simply stacked toward one family.

PREDICTIONS. If TTB is right: on every pair the option bearing the '1' on the highest discriminating expert is chosen at a single, roughly constant above-chance rate, including the mostly-1 dominance pairs (Class I), the bottom-cue pairs (Class II), and the cascade pair (Class III). If PWEA is right: choices REVERSE on Classes I-II (sparse-1 option chosen), near-deterministically on Class I and only weakly on the small-margin Class II items, sit at chance on Class III, and agree with TTB only on Class IV.

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
**Description:** **Polarised Weighted Evidence Accumulation (PWEA).**

People do NOT stop at the first discriminating cue. On every trial they integrate *all* expert ratings for each option into a single scalar impression, and then choose between the two impressions with a graded (margin-sensitive) rule.

Three claims:

1. **Compensatory integration with validity-graded weights.** Each cue j contributes w_j = ((v_j - 0.5)/0.5)^gamma to the option's score, where the weights are rescaled to have mean 1 so that the total evidence scale grows with the number of cues. The steepness exponent gamma is a free, subject-level parameter: gamma -> 0 gives equal weighting (pure tallying), gamma ~ 1 gives classic validity-proportional weighting, and large gamma approaches a lexicographic, Take-The-Best-like regime in which the top cue can outvote all the others. The family therefore *nests* both incumbent heuristics as special cases rather than opposing them.

2. **Polarity of endorsement is a free interpretive parameter s.** A binary rating of 1 is not intrinsically 'good'. It is a symbol whose desirability sign the subject must infer from the instructions/response mapping. In this task population the sign is systematically *negative*: an expert '1' is read as a flag/warning (or the A/B response mapping is inverted), so the option carrying the greater weighted mass of 1s is *rejected*. score_i = s * sum_j w_j x_ij with s < 0.

3. **Graded choice on the evidence margin, with genuinely heterogeneous evidence sensitivity.** Choice is a softmax over the two scores with inverse temperature beta, so the probability of choosing an option increases smoothly with the *magnitude* of the weighted difference (a strict prediction that one-reason stopping rules cannot make), plus an independent attention-lapse epsilon that replaces the choice with a coin flip. Exact ties in weighted score produce exactly 50/50. Critically, the effective evidence sensitivity k = beta*|s| is only *moderate* in this population and varies widely across subjects: many subjects are near-threshold on small-margin pairs, so their choices on narrow-margin conflict items are only weakly above chance, whereas the same subjects are near-deterministic on wide-margin (dominance-like) items. This within-family heterogeneity in k and gamma — not lapse alone — is what produces the between-subject dispersion of the data and keeps the pooled conflict-hit rate well below ceiling.

Because s < 0 and gamma stays in the compensatory range, the model reproduces both signature results with one mechanism: in conflict pairs the option carrying the single highest-validity 1 also carries the *smallest* weighted mass of 1s, so it is chosen (looking exactly like TTB), whereas in agreement/dominance pairs the option that both TTB and tallying favour carries the *largest* mass of 1s and is therefore rejected (looking anti-TTB). Because Exp-1 conflict margins are small and Exp-2 conflict/dominance margins are large, moderate k depresses the Exp-1 hit rate while leaving the Exp-2 contrast saturated — exactly the asymmetry the human data show.

**Parameters:**
- polarity: [-1.0, -0.35]
- gamma: [0.1, 1.15]
- beta: [2.0, 9.0]
- epsilon: [0.10, 0.55]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            # flat concatenation of the two options
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    # ---------- 2. validity-derived weights ----------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n_features:
        # graceful fallback: linearly descending validities
        if n_features == 1:
            val = np.array([0.9])
        else:
            val = np.linspace(0.9, 0.55, n_features)

    gamma = float(parameters["gamma"])
    r = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)          # normalised validity strength
    w = np.power(r, max(gamma, 0.0))
    m = float(np.mean(w))
    if not np.isfinite(m) or m <= 0.0:
        w = np.ones(n_features)
    else:
        w = w / m                                       # mean weight == 1 (scale grows with n cues)

    # ---------- 3. signed compensatory integration ----------
    s = float(parameters["polarity"])                   # negative: a '1' counts AGAINST the option
    score_a = s * float(np.dot(w, a))
    score_b = s * float(np.dot(w, b))
    scores = np.array([score_a, score_b], dtype=float)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # ---------- 4. margin-graded softmax + lapse ----------
    z = beta * scores
    z = z - np.max(z)                                    # numerically stable
    e = np.exp(z)
    p_core = e / e.sum()
    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def key(x):\n        return ''.join([str(int(v)) for v in list(x)])\n\n    # Class I / II 'polarity-reversal' pairs: the option favoured by the\n    # top-validity expert ALSO carries the larger weighted mass of 1s.\n    rev = {\n        ('11111', '00000'), ('00000', '11111'),\n        ('11100', '00011'), ('00011', '11100'),\n        ('10000', '00000'), ('00000', '10000'),\n        ('10000', '00010'), ('00010', '10000'),\n    }\n    # Class IV 'agreement' pairs: top-cue option carries the SMALLER mass of 1s.\n    agr = {\n        ('10000', '01111'), ('01111', '10000'),\n        ('10000', '01100'), ('01100', '10000'),\n    }\n\n    hits_rev = []\n    hits_agr = []\n    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):\n        try:\n            ka = key(a)\n            kb = key(b)\n        except Exception:\n            continue\n        pair = (ka, kb)\n        in_rev = pair in rev\n        in_agr = pair in agr\n        if not (in_rev or in_agr):\n            continue\n        aa = int(list(a)[0])\n        bb = int(list(b)[0])\n        if aa == bb:\n            continue\n        winner = 0 if aa == 1 else 1\n        hit = 1.0 if int(r) == winner else 0.0\n        if in_rev:\n            hits_rev.append(hit)\n        else:\n            hits_agr.append(hit)\n\n    if len(hits_rev) == 0 or len(hits_agr) == 0:\n        return 0.0\n    return float(np.mean(hits_rev) - np.mean(hits_agr))",
  "rationale": "Contrast score: (rate of choosing the option favoured by the highest-validity discriminating expert on POLARITY-REVERSAL pairs) minus (the same rate on AGREEMENT pairs). Reversal pairs are those where the top-cue winner also carries the larger validity-weighted mass of 1s ([11111]v[00000], [11100]v[00011], [10000]v[00000], [10000]v[00010] and mirrors); agreement pairs are the classic conflict items ([10000]v[01111], [10000]v[01100] and mirrors) where the top-cue winner carries the smaller mass of 1s. Under TTB only the sign on the first discriminating cue matters, so the hit rate is the SAME quantity ((1-eps)*sigmoid(beta)+eps/2) on both sets for every subject regardless of beta and epsilon: the difference is identically 0 up to binomial sampling noise (48 vs 24 trials -> per-subject SD ~0.09). Under PWEA a '1' is a negative flag, so on the reversal set the top-cue option is REJECTED (margins of 5.0, ~3.6, ~1.9, ~1.35 mean-weight units times k=beta*|s|>=0.7 make this near-deterministic), while on the agreement set PWEA mimics TTB and the same option is chosen. The difference therefore lands near -(1-eps)*0.9, i.e. roughly -0.4 to -0.8, with epsilon-driven dispersion only. Because the metric is a within-subject difference of two rates that share the same noise parameters, it cancels the lapse/temperature heterogeneity that otherwise inflates between-subject variance, giving a large mean gap (~0.55-0.6) with tight per-subject variance."
}
```

## Usage

```json
{
  "input_tokens": 9075,
  "output_tokens": 8168
}
```
