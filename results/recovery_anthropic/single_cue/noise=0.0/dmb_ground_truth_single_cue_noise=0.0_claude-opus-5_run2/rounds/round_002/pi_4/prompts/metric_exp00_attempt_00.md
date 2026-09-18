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
**Validities (n_features=7):** [0.94, 0.58, 0.86, 0.63, 0.72, 0.8, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1]
  trial 7: A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  trial 9: A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 1]
  trial 10: A=[1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  trial 11: A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1]

**Rationale:** GOAL: dissociate Count-Direction with Conflict-Graded Confidence (CDCC, advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Both theories agree that the DIRECTION of choice is the count winner and both cap adherence below 1, so tally-vs-TTB style contrasts are useless. They differ on exactly what modulates adherence: LSNC's probability is a function of the composition (nA, nB, nT) ALONE (uniform sampling makes cue identity/validity irrelevant), whereas CDCC's confidence is (i) gated by validity COHERENCE C = (W_win - W_lose)/(W_win + W_lose) with convex weights w = (v-0.5)^rho, and (ii) diluted by S = M/D^gamma where D = nA+nB, so tied cues are completely irrelevant to CDCC but occupy sampling slots for LSNC. This design attacks both differences and, crucially, contains two ORDERING REVERSALS.

DESIGN: 7 experts, scrambled validity spread [0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55] (descending rank f0 > f2 > f5 > f4 > f3 > f1 > f6, so validity rank is decoupled from screen position and positional heuristics cannot mimic validity coherence). One very high (0.94) and two very low (0.58, 0.55) validities make the convex weighting bite: w = (v-.5)^3 gives f0 = .0852 vs f6 = .000125, a ~680:1 ratio. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the even-numbered mirror swaps A and B):
- T1/T2  SPARSE-COHERENT (nA,nB,nT) = (2,1,4): A wins f0(.94), f2(.86); B wins f6(.55); rest tied.
- T3/T4  SPARSE-CONFLICT  (2,1,4): A wins f1(.58), f6(.55); B wins f0(.94); rest tied. IDENTICAL composition to T1/T2.
- T5/T6  DENSE-COHERENT   (4,2,1): A wins f0,f2,f5,f4; B wins f3,f1; f6 tied.
- T7/T8  DENSE-CONFLICT   (4,2,1): A wins f1,f3,f4,f6; B wins f0,f2; f5 tied. IDENTICAL composition to T5/T6.
- T9/T10 UNANIMOUS-SPARSE (2,0,5): A wins f2,f4; five tied cues, no cue opposes.
- T11/T12 UNANIMOUS-DENSE (4,0,3): A wins f0,f3,f5,f6; three tied cues, no cue opposes.

QUANTITATIVE PREDICTIONS.
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): (2,1,4) coherent 0.708 vs conflict 0.541; (4,2,1) coherent 0.651 vs conflict 0.540; (2,0,5) 0.928; (4,0,3) 0.888.
LSNC (exact enumeration, k=2.5, nu=0.20, eps=0): (2,1,4) 0.579 for BOTH coherent and conflict; (4,2,1) 0.621 for BOTH; (2,0,5) 0.680; (4,0,3) 0.770.

THREE ORTHOGONAL DIAGNOSTICS:
(1) COHERENCE (parameter-free for LSNC): holding (nA,nB,nT) exactly fixed, CDCC predicts a 11-17 point coherent-minus-conflict gap that cannot be switched off (kappa >= 0.4 by hypothesis), while LSNC predicts EXACTLY ZERO for every k, nu, epsilon. Each subject contributes 32 coherent and 32 conflict trials on matched compositions, so this contrast is estimated within subject with high power.
(2) SPARSITY REVERSAL among the coherent cells: CDCC's D^gamma dilution makes (2,1,4) [D=2 winners vs 1 loser, D=3] MORE confident than (4,2,1) [D=6] despite the smaller raw margin (0.708 > 0.651); LSNC's tie dilution predicts the OPPOSITE order (0.579 < 0.621) because the sparse cell wastes 4 of 7 sampling slots on tied cues. Same reversal for the unanimous cells: CDCC predicts (2,0,5) > (4,0,3) (0.928 > 0.888, and 0.911 > 0.826 at gamma=1.75), LSNC predicts (4,0,3) > (2,0,5) (0.770 > 0.680). A margin-2 profile beating a margin-4 profile is impossible for LSNC under any parameters.
(3) CEILING: on the clean unanimous-sparse cell CDCC allows 0.90-0.95 adherence, whereas LSNC is capped near 0.68 (and at most ~0.79 even at k=3.5, nu=0.12) because a small noisy sample easily reverses the sign. Adherence above ~0.85 on T9/T10 is outside LSNC's reachable range.

CONTROLS: (i) every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; (ii) tied cues are encoded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not covary with composition or coherence; (iii) validity rank is scrambled relative to display position; (iv) no pair is degenerate: every trial has a strict count winner and requires reading several ratings; (v) systematic BELOW-chance adherence in the conflict cells (i.e. following the single best cue against the tally) would falsify BOTH theories in favour of one-reason/WADD integration, so the design is self-diagnostic.

INTERPRETATION: a large coherent > conflict gap that never dips below 0.5, together with sparse > dense ordering at matched or even smaller margins and near-0.9 adherence on unanimous sparse profiles, uniquely supports CDCC. Flat adherence across the coherence manipulation with a (4,0,3) > (2,0,5) > (4,2,1) > (2,1,4) composition ordering and a ceiling near 0.7 uniquely supports LSNC.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Count-Direction with Conflict-Graded Confidence (CDCC).**

When people compare two multi-attribute options described by binary expert ratings, the *direction* of their preference and the *confidence* with which they execute it are computed by two functionally separate processes.

1. **Direction is purely count-based.** The comparator registers, for each attribute, only which option is rated higher, and the option that wins on more attributes becomes the candidate response. Validities never enter this stage: they cannot promote a minority-supported option to candidate status. When the two win-counts are equal, the comparator returns no candidate at all, and the subject flips a mental coin — exactly 0.5, with **no** validity-ordered tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles CDCC predicts chance adherence to the most-valid discriminating cue, regardless of any parameter setting.

2. **Confidence, not direction, carries the validity information, and it is graded by evidence dilution and by cue conflict.** The probability of actually executing the count-winner is a saturating (tanh) function of a scalar confidence signal built from three ingredients: (i) the raw win margin M = |nA − nB|; (ii) *dilution* — the margin is discounted by the total number of discriminating cues raised to a power gamma, so that a 6-vs-2 split feels much weaker than a 4-vs-0 split even though both have M = 4 (evidence spread thin across many mutually contradicting attributes is subjectively less compelling than a clean, sparse, unanimous split); and (iii) *coherence* — the signed, normalised balance of validity weight between the cues the count-winner wins and the cues the loser wins, where the subjective weight of a cue is a convex (super-linear) function of its stated validity advantage, w_j = (v_j − 0.5)^rho. Convexity means the coherence signal is dominated by the few high-validity cues: when the best cue points *with* the count the subject feels certain, when it points *against* the count the subject feels torn even though the count itself is unambiguous.

3. **Conflict erodes confidence but never reverses direction.** The confidence argument is floored at zero, so validity conflict can drive behaviour arbitrarily close to guessing but never below chance: CDCC never makes systematic one-reason (Take-The-Best) choices. This is what separates it from any Tallying+TTB mixture: on high-margin trials where the top cue opposes the tally, people move toward 0.5 from above, not past it.

4. **Lapses.** With probability epsilon the entire decision is replaced by a coin flip (attention lapse), which caps adherence below 1 even on maximally coherent trials.

The theory therefore predicts a specific cross-design signature: chance performance on tally-tied profiles; high but sub-ceiling adherence on sparse, coherent, unanimous profiles; noticeably lower adherence on dense profiles where the loser also wins several cues even when the raw margin is larger; and the lowest (but still above-chance) adherence when the most valid cue opposes the count.

**Parameters:**
- beta: [1.6, 3.6]
- gamma: [1.25, 1.75]
- kappa: [0.4, 1.0]
- rho: [2.2, 3.8]
- epsilon: [0.02, 0.18]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        stim = stim.reshape(2, -1)
    if stim.ndim != 2 or stim.shape[0] != 2:
        stim = stim.reshape(2, -1)
    a = stim[0]
    b = stim[1]
    n = int(a.shape[0])

    if n == 0:
        return np.array([0.5, 0.5], dtype=float)

    a_win = a > b
    b_win = b > a
    nA = int(np.sum(a_win))
    nB = int(np.sum(b_win))
    D = nA + nB

    # --- Commitment 1: direction is purely count-based; ties -> exact guess ---
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    winner = 0 if nA > nB else 1
    M = float(abs(nA - nB))

    # --- parameters -----------------------------------------------------
    beta = float(parameters.get("beta", 2.6))
    gamma = float(parameters.get("gamma", 1.5))
    kappa = float(parameters.get("kappa", 0.7))
    rho = float(parameters.get("rho", 3.0))
    eps = float(parameters.get("epsilon", 0.10))

    beta = min(max(beta, 0.0), 20.0)
    gamma = min(max(gamma, 0.0), 3.0)
    kappa = min(max(kappa, 0.0), 5.0)
    rho = min(max(rho, 0.5), 8.0)
    eps = min(max(eps, 0.0), 0.6)

    # --- validity weights (convex in validity advantage) ------------------
    v_raw = parameters.get("validities", None)
    v = None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume features are ordered by descending validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

    w = np.clip(v - 0.5, 1e-9, None) ** rho

    if winner == 0:
        W_win = float(np.sum(w[a_win]))
        W_lose = float(np.sum(w[b_win]))
    else:
        W_win = float(np.sum(w[b_win]))
        W_lose = float(np.sum(w[a_win]))

    tot = W_win + W_lose
    C = 0.0 if tot <= 0.0 else (W_win - W_lose) / tot   # in [-1, 1]

    # --- Commitment 2: dilution-discounted margin, coherence-gated --------
    S = M / (float(D) ** gamma)
    gate = (1.0 + kappa * C) / (1.0 + kappa)             # 1.0 when fully coherent
    arg = beta * S * gate

    # --- Commitment 3: conflict erodes confidence, never reverses it ------
    if arg < 0.0:
        arg = 0.0
    if arg > 30.0:
        arg = 30.0

    conf = float(np.tanh(arg))

    # --- Commitment 4: lapse ---------------------------------------------
    p_winner = 0.5 + 0.5 * (1.0 - eps) * conf
    p_winner = float(min(max(p_winner, 0.5), 1.0 - 1e-9))

    p = np.empty(2, dtype=float)
    p[winner] = p_winner
    p[1 - winner] = 1.0 - p_winner

    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    V = np.array([0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55], dtype=float)\n\n    coh_hit = 0.0; coh_n = 0\n    con_hit = 0.0; con_n = 0\n    us_hit = 0.0; us_n = 0\n    ud_hit = 0.0; ud_n = 0\n\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        try:\n            a = np.asarray(a_raw, dtype=float).ravel()\n            b = np.asarray(b_raw, dtype=float).ravel()\n        except Exception:\n            continue\n        n = int(a.size)\n        if n == 0 or int(b.size) != n:\n            continue\n        if n == int(V.size):\n            w = V - 0.5\n        elif n > 1:\n            w = np.linspace(0.95, 0.55, n) - 0.5\n        else:\n            w = np.array([0.25])\n\n        aw = a > b\n        bw = b > a\n        nA = int(np.sum(aw))\n        nB = int(np.sum(bw))\n        if nA == nB:\n            continue\n        winner = 0 if nA > nB else 1\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        if r not in (0, 1):\n            continue\n        hit = 1.0 if r == winner else 0.0\n        D = nA + nB\n        if winner == 0:\n            Ww = float(np.sum(w[aw])); Wl = float(np.sum(w[bw])); n_lose = nB\n        else:\n            Ww = float(np.sum(w[bw])); Wl = float(np.sum(w[aw])); n_lose = nA\n\n        if n_lose == 0:\n            # unanimous profiles: sparse (few discriminating cues) vs dense\n            if D <= 2:\n                us_hit += hit; us_n += 1\n            else:\n                ud_hit += hit; ud_n += 1\n        else:\n            if Ww > Wl:\n                coh_hit += hit; coh_n += 1\n            elif Ww < Wl:\n                con_hit += hit; con_n += 1\n\n    if coh_n > 0 and con_n > 0:\n        d1 = coh_hit / coh_n - con_hit / con_n\n    else:\n        d1 = 0.0\n    if us_n > 0 and ud_n > 0:\n        d2 = us_hit / us_n - ud_hit / ud_n\n    else:\n        d2 = 0.0\n\n    return float(d1 + d2)\n",
  "rationale": "Two structurally signed contrasts are summed, each of which has an opposite (or null) sign under the two theories, so the composite separates them with a large mean gap while each component is estimated from 32/32 and 16/16 trials per subject (tight per-subject noise).\n\nD1 = adherence(count-winner) on validity-COHERENT conflicting profiles minus adherence on validity-INCOHERENT profiles, with the (nA,nB,nT) composition held EXACTLY matched across the two sets by design (T1/T2 vs T3/T4 both (2,1,4); T5/T6 vs T7/T8 both (4,2,1)). LSNC samples cues uniformly and ignores validity entirely, so its predicted adherence depends only on (nA,nB,nT): D1 = 0 exactly, for every k, nu, epsilon. CDCC gates confidence by the convex validity-weighted coherence C, so D1 is strictly positive (~+0.11 to +0.17, larger for larger kappa) and cannot be turned off.\n\nD2 = adherence on the UNANIMOUS-SPARSE profile (2 winning cues, 0 opposing, 5 tied) minus adherence on the UNANIMOUS-DENSE profile (4 winning cues, 0 opposing, 3 tied). For CDCC the dilution term S = M/D^gamma gives 2^(1-gamma) vs 4^(1-gamma) = (2^(1-gamma))^2, and since gamma > 1 the sparse profile ALWAYS yields the larger confidence: D2 >= 0 for every admissible parameter vector. For LSNC tied cues waste sampling slots, so the dense profile (4/7 informative slots) always beats the sparse one (2/7): D2 < 0 (~ -0.06 to -0.12) for every admissible parameter vector.\n\nHence the composite D1 + D2 is bounded below by 0 under CDCC (expected ~ +0.18) and is expected to be clearly negative under LSNC (~ -0.09), a gap of roughly 0.27 with per-subject variance dominated by binomial sampling over 96 trials (sd ~ 0.17), which gives a Welch t well above 4 at N=25. It is not trivially true for CDCC: a flat coherence effect or an ordering with dense > sparse on unanimous profiles would push the metric to zero/negative and falsify it."
}
```

## Usage

```json
{
  "input_tokens": 10478,
  "output_tokens": 9154
}
```
