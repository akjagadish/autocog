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
**Validities (n_features=5):** [0.95, 0.8, 0.7, 0.62, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 8: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 9: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 14: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** I advocate Polarised Weighted Evidence Accumulation (PWEA) against Take-The-Best (TTB). The two prior TTB-vs-PWEA designs contrasted a few isolated reversal pairs; this design instead builds a *continuous margin ladder* on which TTB is mathematically forced to a single flat accuracy while PWEA must produce a monotone psychometric function with a parameter-robust crossing point, plus a gamma-graded pair and a robust weighted-tie pair.

Validities (0.95, 0.80, 0.70, 0.62, 0.55) are strictly ordered so TTB's cue cascade is unambiguous. At gamma=1 the mean-1 normalised weights are w=[2.01, 1.34, 0.89, 0.54, 0.22]; at gamma=0.1 they are nearly flat, w=[1.10, 1.05, 1.01, 0.96, 0.88]. Every base pair appears in both A/B orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly.

LADDER (base pairs 1-4, all of the form A=[1,0,0,0,0] versus a B with progressively more low-validity 1s): B = [0,0,0,0,0] / [0,1,0,0,0] / [0,1,1,0,0] / [0,1,1,1,1]. On ALL FOUR, expert 1 discriminates in favour of A, so TTB predicts exactly the same accuracy (1-eps)*sigmoid(beta) toward A on each; nothing in TTB can make these four differ. PWEA's signed weighted-mass margin (A minus B) is +2.01, +0.67, -0.22, -0.98 at gamma=1 and +1.10, +0.04, -0.92, -2.80 at gamma=0.1: monotone in the same order for the whole admissible gamma range, and with s<0 this yields a monotonically INCREASING probability of choosing A across pairs 1->4, crossing 50% between pair 2 and pair 3 for every gamma in [0.1, 1.15]. A four-point graded curve with a sign flip located between pairs 2 and 3 is uniquely PWEA; flat above-chance A-choice is uniquely TTB. Pair 4 is the classic 'conflict' item on which the two families agree in direction, so it also pins down beta/epsilon for both models.

WEIGHT-vs-COUNT GRADIENT (pairs 6 and 7): [1,0,0,0,0] vs [0,0,0,0,1] and [0,0,1,0,0] vs [0,0,0,1,0]. Both have one 1 per option, so TTB again predicts identical accuracy toward the higher-validity-1 option (A) in both. PWEA predicts the opposite direction (B) with margins in an approximately 5:1 ratio (1.79 vs 0.36 at gamma=1), collapsing toward chance for both as gamma->0. These items therefore separate direction (PWEA vs TTB) and simultaneously identify gamma, which no TTB parameter can mimic.

CASCADE REVERSAL (pair 5): [1,1,1,0,0] vs [1,0,0,1,1]. Expert 1 does not discriminate, so TTB descends to expert 2 and picks A decisively. PWEA's margin (w2+w3-w4-w5) is positive for every gamma (1.47 at gamma=1, 0.22 at gamma=0.1), so PWEA chooses B, strongly if gamma is near 1 and weakly if gamma is near 0. This tests one-reason stopping against signed integration on a trial where the top cue is uninformative.

ROBUST WEIGHTED TIE (pair 8): [0,1,0,0,1] vs [0,0,1,1,0]. Counts are equal and the weighted masses nearly cancel across the whole gamma range (|margin| <= 0.19 for gamma in [0.1, 1.15], and it even changes sign), so PWEA is pinned essentially at chance for any beta and |s|. TTB, however, stops at expert 2 and picks A with its full, undiminished accuracy. Systematic A-choice here would falsify PWEA; near-chance responding here combined with decisive responding on pairs 1 and 5 would falsify TTB, because TTB's accuracy must be constant across all discriminating trials.

PREDICTIONS. PWEA: choice of A rises monotonically across ladder pairs 1->4 with the crossover between 2 and 3; strong B-preference on 6, weak B-preference on 7 (ratio ~5:1); B-preference on 5; chance on 8; overall dispersion driven by subject-level k=beta*|s| and gamma. TTB: a single, roughly constant above-chance preference for A on every one of pairs 1-8 (and for the mirrored option in the mirrors), with no dependence on weighted mass, cue count, or margin.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    def as_str(x):\n        return ''.join([str(int(v)) for v in list(x)])\n\n    dom_hits = []\n    conf_hits = []\n\n    for _, row in data.iterrows():\n        a = [int(v) for v in list(row['option_a_ratings'])]\n        b = [int(v) for v in list(row['option_b_ratings'])]\n        if len(a) != len(b) or len(a) == 0:\n            continue\n        # Cue order = feature index order (validities are strictly descending by design)\n        winner = None\n        for j in range(len(a)):\n            if a[j] > b[j]:\n                winner = 0\n                break\n            if b[j] > a[j]:\n                winner = 1\n                break\n        if winner is None:\n            continue\n        try:\n            resp = int(row['response'])\n        except Exception:\n            continue\n        hit = 1.0 if resp == winner else 0.0\n\n        sa = as_str(a)\n        sb = as_str(b)\n        key = tuple(sorted([sa, sb]))\n\n        # DOMINANCE base pair: [1,0,0,0,0] vs [0,0,0,0,0]  (trials 1 and 9)\n        if key == ('00000', '10000'):\n            dom_hits.append(hit)\n        # CONFLICT base pair: [1,0,0,0,0] vs [0,1,1,1,1]  (trials 4 and 12)\n        elif key == ('01111', '10000'):\n            conf_hits.append(hit)\n\n    if len(dom_hits) == 0 or len(conf_hits) == 0:\n        return 0.0\n\n    return float(np.mean(conf_hits) - np.mean(dom_hits))",
  "rationale": "Within-subject contrast between two base pairs that TTB must treat identically but PWEA must treat oppositely.\n\nDefine 'hit' = choosing the option favoured by the Take-The-Best cascade (the option with the 1 on the highest-validity discriminating cue). The metric is hit-rate on the CONFLICT pair {[1,0,0,0,0] vs [0,1,1,1,1]} minus hit-rate on the DOMINANCE pair {[1,0,0,0,0] vs [0,0,0,0,0]}. Both base pairs appear in both A/B orders, 12 trials each per subject, so side bias cancels.\n\nTTB: cue 1 discriminates on BOTH pairs, so the predicted choice probability for the TTB winner is exactly (1-eps)*sigmoid(beta)+eps/2 on both. The difference is identically 0 for every beta and epsilon, i.e. the metric is pinned at 0 with only binomial noise (var ~0.02-0.03), independent of subject parameters.\n\nPWEA (polarity s<0, compensatory weights): on the dominance pair the weighted mass of 1s is 2.0 (gamma=1) or 1.1 (gamma=0.1) for A and 0 for B, so PWEA REJECTS A -> hit rate well below 0.5. On the conflict pair the mass is 2.0 for A vs 3.0 for B (gamma=1) or 1.1 vs 3.9 (gamma=0.1), so PWEA ACCEPTS A -> hit rate well above 0.5, agreeing with TTB. Hence the difference is strongly positive (~0.5-0.9) for essentially the whole admissible parameter box, and crucially beta, |s| and epsilon largely cancel within subject, keeping between-subject variance modest.\n\nSo the expected values are ~0 (TTB) vs ~0.6-0.7 (PWEA) with small per-subject spread: a large, tight contrast that is not trivially true of PWEA (it is a signed, direction-reversing prediction about two specific stimulus structures that any one-reason rule cannot produce)."
}
```

## Usage

```json
{
  "input_tokens": 8958,
  "output_tokens": 6965
}
```
