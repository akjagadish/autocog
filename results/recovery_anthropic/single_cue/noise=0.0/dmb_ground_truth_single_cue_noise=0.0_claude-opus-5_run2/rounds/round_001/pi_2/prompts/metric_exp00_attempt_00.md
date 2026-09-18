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
**Validities (n_features=8):** [0.93, 0.55, 0.78, 0.86, 0.6, 0.71, 0.66, 0.82]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  trial 4: A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 1]
  trial 5: A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 1]
  trial 9: A=[1, 1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 1, 0]
  trial 11: A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0]
  trial 12: A=[0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 1, 0, 1]

**Rationale:** GOAL: dissociate softmax Tallying (advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: P(A) = (1-eps)*sigmoid(beta*M) + eps/2. Therefore any two profiles with the same M MUST yield exactly the same P(A) for every beta and epsilon -- a parameter-free invariance. LSNC instead depends on the full composition (nA, nB, nT): because the inspected subset is drawn uniformly from ALL features, tied cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so adherence must fall monotonically as a fixed margin is realized with fewer ties and more loser-wins.

DESIGN: 8 experts, scrambled validity spread [0.93, 0.55, 0.78, 0.86, 0.60, 0.71, 0.66, 0.82] (descending rank f0 > f3 > f7 > f2 > f5 > f6 > f4 > f1, so validity order is decoupled from screen position). 2 (margin: M = 2, 4) x 3 (composition: SPARSE / MID / DENSE) factorial, each cell fully mirrored for side => 12 unique pairs x 8 repetitions ~ 96 trials.

Cells (A-favouring version; mirror swaps A and B):
- S2 sparse M=2: (nA,nB,nT) = (2,0,6)
- M2 mid    M=2: (3,1,4)
- D2 dense  M=2: (5,3,0)
- S4 sparse M=4: (4,0,4)
- M4 mid    M=4: (5,1,2)
- D4 dense  M=4: (6,2,0)

QUANTITATIVE PREDICTIONS. Tallying: a pure margin main effect and EXACTLY ZERO composition effect -- the three M=2 cells must be statistically identical, the three M=4 cells must be statistically identical, and the M=4 level must exceed M=2 by the sigmoid gap sigmoid(4b) - sigmoid(2b). LSNC (exact enumeration, k=2.5, nu=0.20, eps=0): M=2 -> 0.661 (sparse) vs ~0.62 (mid) vs 0.598 (dense); M=4 -> 0.750 (sparse) vs ~0.72 (mid) vs 0.689 (dense). I.e. LSNC predicts a monotone ~6-point decline across the composition ladder at BOTH margins, plus adherence capped near 0.75-0.80 even at M=4. Note that at sample size K=1 all same-margin cells coincide (P = 0.5 + 0.3*M/n), so any composition effect in the data is diagnostic of multi-cue but sub-total sampling; a flat ladder is diagnostic of full integration (Tallying).

CONFOUND CONTROLS. (i) In every A-favouring cell option A wins the highest-validity cue f0 (0.93), which discriminates in all six cells, so Take-The-Best / any 'best cue' rule predicts a CONSTANT P(A) across all cells and cannot generate either the margin effect or a composition ladder. (ii) Validity-weighted additive evidence (log-odds weights) was deliberately matched WITHIN each margin level: M=2 cells give net +2.99 (sparse), +3.04 (mid), +3.12 (dense); M=4 cells give +4.55, +4.78, +4.39. Hence WADD-type integration also predicts no composition effect, so a sparse>mid>dense ordering can only come from sample-limited counting with tie dilution, while a flat ladder plus a clear margin effect uniquely supports Tallying. (iii) Tied cues are encoded sometimes 1-1 and sometimes 0-0 so that display density of 1s is not itself a cue to composition. (iv) Every configuration appears in both an A-favouring and a B-favouring mirror, so side bias cancels exactly.

INTERPRETATION: margin-graded but composition-invariant adherence (with the M=4 cells approaching the ceiling implied by the beta estimated from the M=2 cells) supports Tallying; a systematic sparse > mid > dense decline at each margin, with adherence capped around 0.75 at M=4, supports LSNC.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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
**Description:** **Limited-Sample Noisy Cue Counting (LSNC).** People neither integrate all cues (Tallying) nor consult a single best cue (Take-The-Best). Instead, on each choice they inspect only a *limited, randomly selected subset* of the presented expert ratings, tally feature-wise wins **within that sample only**, and pick the option that is ahead in the sample; a within-sample tie (including a sample containing no discriminating cue) forces a guess.

Three commitments define the theory:
1. **Equal-probability sampling.** The inspected subset is drawn uniformly at random from the whole profile — it is *not* validity-ordered and *not* validity-weighted. Stated validities are too weakly represented to steer the search order, so no cue enjoys priority. Consequently, on globally tally-tied, side-counterbalanced profiles the process is exactly symmetric and choice is 50/50, with no tendency to follow the most valid discriminating cue.
2. **Sample-size limitation with tie dilution.** The number of cues actually inspected, K, is a random variable with mean k < n (implemented as K = 1 + Binomial(n−1, q), so at least one cue is always read). Because the subset is drawn from *all* features, non-discriminating (tied) features occupy sampling slots and dilute the evidence: two profiles with the same raw tally margin M produce different adherence depending on how many uninformative cues pad the profile, and on the proportion (not the raw count) of informative cues won. This is the signature that dissociates LSNC from softmax Tallying, whose predictions depend only on M.
3. **Encoding noise + lapse.** Each inspected feature-wise comparison is registered with the wrong sign with probability nu (attention/encoding noise), and with probability epsilon the whole decision is replaced by a coin flip.

Because a small, noisy sample can easily reverse the sign of a large global margin, adherence to the full-tally winner is *capped well below 1* even on high-margin conflict trials — the systematic attenuation that full Tallying cannot produce without degenerate temperature — while symmetry guarantees exactly chance behaviour on tally-tied pairs.

**Parameters:**
- k: [1.5, 3.5]
- nu: [0.12, 0.32]
- epsilon: [0.0, 0.12]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from math import comb

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("LSNC expects a (2, n_features) stimulus; got shape %s." % (stim.shape,))
    a, b = stim[0], stim[1]
    n = int(a.shape[0])

    nA = int(np.sum(a > b))          # cues on which A wins
    nB = int(np.sum(b > a))          # cues on which B wins
    nT = n - nA - nB                 # uninformative (tied) cues

    k = float(parameters["k"])            # mean number of cues inspected
    nu = float(parameters["nu"])          # per-comparison encoding error
    eps = float(parameters["epsilon"])    # lapse rate
    nu = float(min(max(nu, 0.0), 0.5))
    eps = float(min(max(eps, 0.0), 1.0))

    if n == 0:
        return np.ones(2) / 2.0

    # ---- distribution of the inspected sample size K in {1..n} -------------
    if n == 1:
        pK = np.array([1.0])
    else:
        q = (k - 1.0) / (n - 1.0)
        q = float(min(max(q, 0.0), 1.0))
        pK = np.array([comb(n - 1, s - 1) * (q ** (s - 1)) * ((1.0 - q) ** (n - s))
                       for s in range(1, n + 1)], dtype=float)
        tot = pK.sum()
        pK = pK / tot if tot > 0 else np.ones(n) / float(n)

    # ---- exact enumeration over sample compositions ------------------------
    pA = 0.0
    for si, s in enumerate(range(1, n + 1)):
        w = float(pK[si])
        if w <= 0.0:
            continue
        denom = float(comb(n, s))
        for ai in range(0, min(nA, s) + 1):
            for bi in range(0, min(nB, s - ai) + 1):
                ti = s - ai - bi
                if ti < 0 or ti > nT:
                    continue
                cnt = comb(nA, ai) * comb(nB, bi) * comb(nT, ti)
                if cnt == 0:
                    continue
                ph = float(cnt) / denom          # multivariate hypergeometric
                m = ai + bi                      # informative cues in sample
                if m == 0:
                    pA += w * ph * 0.5           # nothing discriminates -> guess
                    continue
                # X = number of sampled comparisons REGISTERED as A-wins
                d1 = np.array([comb(ai, x) * ((1.0 - nu) ** x) * (nu ** (ai - x))
                               for x in range(ai + 1)], dtype=float)
                d2 = np.array([comb(bi, x) * (nu ** x) * ((1.0 - nu) ** (bi - x))
                               for x in range(bi + 1)], dtype=float)
                dist = np.convolve(d1, d2)
                xs = np.arange(m + 1)
                p_win = float(dist[2 * xs > m].sum())
                p_tie = float(dist[2 * xs == m].sum())
                pA += w * ph * (p_win + 0.5 * p_tie)

    pA = float(min(max(pA, 0.0), 1.0))
    p = np.array([pA, 1.0 - pA], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Adherence to the full-profile tally winner.\n\n    For every trial we recompute the global feature-wise win counts\n    nA = #(A > B) and nB = #(B > A) over ALL features, identify the\n    option favoured by the complete tally, and score whether the subject\n    chose it. The returned scalar is the proportion of such trials on\n    which the subject followed the complete-tally winner.\n\n    Works identically on the pooled DataFrame and on a single subject's\n    slice.\n    \"\"\"\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    hits = []\n    for a_raw, b_raw, r in zip(a_col, b_col, r_col):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        if a.shape[0] != b.shape[0] or a.shape[0] == 0:\n            continue\n        nA = float(np.sum(a > b))\n        nB = float(np.sum(b > a))\n        if nA == nB:\n            continue  # undiagnostic (no complete-tally winner)\n        winner = 0 if nA > nB else 1\n        try:\n            resp = int(r)\n        except (TypeError, ValueError):\n            continue\n        if resp not in (0, 1):\n            continue\n        hits.append(1.0 if resp == winner else 0.0)\n\n    if len(hits) == 0:\n        return float('nan')\n    return float(np.mean(hits))",
  "rationale": "The single sharpest quantitative divergence between the two accounts is the ATTAINABLE LEVEL of adherence to the complete-profile tally winner on clearly discriminating pairs. Every trial in this design has a strict global tally winner with margin M = 2 or M = 4, and (by construction) the option favoured by the tally also wins the most valid cue, so the metric is not confounded by cue-order strategies.\n\nUnder softmax Tallying the decision variable is the FULL margin: P(winner) = (1-eps)*sigmoid(beta*M) + eps/2. Since M >= 2 on every trial, sigmoid(beta*M) is already essentially at ceiling for most of the beta range, so adherence is pinned near 1 - eps/2, i.e. roughly 0.85-0.90 on average and, crucially, only weakly dispersed (its across-subject spread comes from the lapse parameter alone, since 48 + 48 trials wash out binomial noise).\n\nUnder LSNC the inspected subset averages ~2.5 of 8 cues and each comparison is mis-registered with prob nu ~ 0.2, so a large global margin is routinely reversed or diluted by tied cues occupying sampling slots. Exact enumeration for this design gives adherence ~0.60-0.66 at M=2 and ~0.69-0.75 at M=4, i.e. a pooled value around 0.66-0.70 that is CAPPED well below what full integration produces, and again tightly distributed across subjects because the parameter ranges (k in [1.5,3.5], nu in [0.12,0.32]) move it only a few points.\n\nSo the expected contrast is roughly 0.87 vs 0.68 with per-subject variances of order 0.005-0.008 in both cases, which at N=25 is a very large Welch t. This is not trivially true of Tallying: had subjects been strongly noise-limited, Tallying's own epsilon/beta would drag adherence down toward the LSNC band; the prediction that adherence sits high above the sample-limited ceiling is a substantive commitment of full-cue integration."
}
```

## Usage

```json
{
  "input_tokens": 8218,
  "output_tokens": 7147
}
```
