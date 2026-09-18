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
**Validities (n_features=6):** [0.92, 0.6, 0.83, 0.71, 0.55, 0.77]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 5: A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  trial 7: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 8: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 9: A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  trial 10: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  trial 11: A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Rationale:** GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from softmax Tallying (competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: any two pairs with the same M must produce exactly the same P(choose A) for every possible beta and epsilon. LSNC instead depends on the full composition (nA, nB, nT): tied/uninformative cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so two profiles with identical M but different numbers of tied cues yield different adherence. The design is therefore a 3 (margin: M = 1, 2, 3) x 2 (informative density: SPARSE, all discriminating cues won by one option but many tied cues; DENSE, no/few tied cues but the loser also wins several cues) factorial, fully mirrored for side. Tallying predicts a pure main effect of margin and EXACTLY ZERO density effect (parameter-free); LSNC predicts sparse > dense at every margin.

CELLS (A-favouring version; the mirror swaps A and B):
- S1 sparse M=+1: (nA,nB,nT) = (1,0,5)
- D1 dense  M=+1: (3,2,1)
- S2 sparse M=+2: (2,0,4)
- D2 dense  M=+2: (4,2,0)
- S3 sparse M=+3: (3,0,3)
- D3 dense  M=+3: (4,1,1)

QUANTITATIVE LSNC PREDICTIONS (k=2.5, nu=0.20, epsilon=0, exact enumeration of the hypergeometric sample compositions): M=1: 0.625 (sparse) vs 0.572 (dense); M=2: 0.702 vs 0.630; M=3: 0.754 vs 0.707. So LSNC predicts a ~5-7 point sparse-minus-dense advantage at EVERY margin (and the gap grows with k), while adherence stays capped well below 1 even at M=3. Tallying predicts 0 at every margin. Aggregated over the 3 margins each subject contributes 48 sparse and 48 dense trials, so the parameter-free density contrast is powerfully estimated within subject.

CONFOUND CONTROL (critical): (i) In every A-favouring cell option A wins the highest-validity discriminating cue f0 (0.92), which also discriminates in every cell, so Take-The-Best (or any 'best cue' strategy) predicts a CONSTANT P(A) across all six cells and cannot generate a density effect. (ii) Validity-weighted additive evidence (log-odds weights w = [2.44, 0.41, 1.59, 0.90, 0.20, 1.21]) was deliberately matched between the sparse and dense member of each margin pair: M=1: +2.44 vs +2.13; M=2: +2.64 vs +2.53; M=3: +3.05 vs +3.16. Hence a WADD-type subject also cannot produce the sparse>dense pattern; only sample-limited counting with tie dilution can. (iii) Validity order is scrambled relative to screen position (descending order is f0, f2, f5, f3, f1, f4) so positional heuristics do not mimic validity-ordered search. (iv) Tied cues are encoded sometimes as 1-1 and sometimes as 0-0 so that the number of displayed 1s is not a constant cue to density. (v) Every configuration appears in both an A-favouring and a B-favouring mirror (12 unique pairs x 8 repetitions = 96 trials), so side bias cancels exactly.

INTERPRETATION: a graded, margin-monotone function that is IDENTICAL for sparse and dense cells supports Tallying; a margin-monotone function that is systematically shifted upward in the sparse (tie-padded, proportion-1.0) cells relative to the dense cells with the same margin, together with adherence ceilings around 0.70-0.80 at M=3, uniquely supports LSNC and cannot be produced by Tallying under any beta/epsilon.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    hits = []\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n    for a, b, r in zip(a_col, b_col, r_col):\n        av = np.asarray(a, dtype=float).ravel()\n        bv = np.asarray(b, dtype=float).ravel()\n        if av.size == 0 or av.size != bv.size:\n            continue\n        nA = float(np.sum(av > bv))\n        nB = float(np.sum(bv > av))\n        if nA == nB:\n            continue\n        winner = 0 if nA > nB else 1\n        try:\n            resp = int(r)\n        except Exception:\n            continue\n        hits.append(1.0 if resp == winner else 0.0)\n    if len(hits) == 0:\n        return float('nan')\n    return float(np.mean(hits))\n",
  "rationale": "This is the overall adherence rate to the full-tally winner (the option that wins on strictly more features), pooled over the 96 trials of the design (every trial in this design has a non-zero tally margin, so no trial is dropped).\n\nWhy it separates the theories: LSNC's core commitment is that only a small, randomly drawn subset of cues (mean k between 1.5 and 3.5 of the 6 features) is ever inspected, that tied cues occupy sampling slots (tie dilution), and that each inspected comparison is mis-registered with probability nu = .12-.32. A small noisy sample frequently reverses or fails to recover the sign of the global margin, so LSNC's adherence is structurally CAPPED: exact enumeration for this design gives ~0.57-0.75 across the six cells, i.e. an aggregate near 0.65-0.68, and it cannot approach 1 for any admissible parameter setting. Tallying, in contrast, reads every feature and passes the exact margin (M = 1, 2 or 3 here, never 0) through a softmax; over the bulk of its inverse-temperature range the softmax is essentially deterministic, so its adherence sits near 1 - epsilon/2, i.e. roughly 0.85-0.90 on average and never structurally attenuated by the composition of the profile.\n\nThis is not trivially true of my theory: Tallying is perfectly free to produce low adherence (small beta, large epsilon), and LSNC is free to produce moderately high adherence (large k, small nu) - the metric measures a behavioural ceiling, not a theory label. The prediction that humans in a 6-cue, feedback-free, high-conflict task fall far short of margin-consistent responding (adherence well below 0.8 even at margin 3) is the substantive LSNC claim; Tallying, which by construction integrates all cues, has to attribute any such shortfall to a degenerate temperature that would then also flatten the margin gradient. The metric is a simple proportion over ~96 trials per subject, so its per-subject sampling noise is small (sd ~0.04-0.05), keeping the between-subject variance tight relative to the ~0.2 mean gap."
}
```

## Usage

```json
{
  "input_tokens": 7963,
  "output_tokens": 7325
}
```
