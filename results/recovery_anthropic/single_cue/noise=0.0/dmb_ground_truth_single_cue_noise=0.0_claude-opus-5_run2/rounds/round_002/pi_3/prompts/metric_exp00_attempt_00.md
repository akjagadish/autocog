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
**Validities (n_features=6):** [0.94, 0.57, 0.86, 0.62, 0.7, 0.78]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 3: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 11: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 12: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Rationale:** GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from Count-Direction with Conflict-Graded Confidence (CDCC, competing). Both theories agree on the count-based DIRECTION of choice and both cap adherence below 1, so the classic tally-vs-TTB contrasts are useless here. They differ on exactly two things, and this design isolates both.

KEY DISSOCIATION 1 (parameter-free for LSNC): COHERENCE. LSNC's predicted probability is a function of the composition (nA, nB, nT) ALONE 
cue identity/validity is irrelevant because the inspected subset is drawn uniformly at random. CDCC's confidence is gated by the coherence term C = (W_win - W_lose)/(W_win + W_lose) with convex weights w = (v-0.5)^rho, so it must show a large drop when the highest-validity cue is won by the count LOSER, holding (nA, nB, nT) exactly fixed. Every composition therefore appears in a COHERENT version (count winner also wins the top-validity cues, loser wins only the weakest ones) and a CONFLICT version (count winner wins only the weakest cues, loser wins the top cue(s)), with the identical (nA, nB, nT).

KEY DISSOCIATION 2 (composition ordering reversal): with n = 6 fixed, CDCC's dilution term is S = M/D^gamma (D = nA+nB, ties irrelevant), whereas LSNC's dilution comes from tied cues occupying sampling slots and from the PROPORTION of sampled informative cues won. These give opposite orderings for two of the three cells: LSNC ranks (4,2,0) ABOVE (2,1,3) (0.630 vs 0.589), while CDCC ranks (2,1,3) ABOVE (4,2,0) (0.633 vs 0.602 averaged over coherence; 0.707 vs 0.651 in the coherent versions), because CDCC penalises D = 6 heavily via D^1.5 while LSNC penalises the 3 tied slots.

DESIGN: 6 experts, scrambled validity spread [0.94, 0.57, 0.86, 0.62, 0.70, 0.78] (descending rank f0 > f2 > f5 > f4 > f3 > f1, so validity rank is decoupled from screen position and positional heuristics cannot mimic validity-based coherence). One very high (0.94) and two low (0.57, 0.62) validities make the convex weighting in CDCC bite hard (w = (v-.5)^3 gives f0 = .085 vs f1 = .0003, a 250:1 ratio). 3 compositions x 2 coherence levels x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the mirror swaps A and B):
- T1/T2  C1-COHERENT (nA,nB,nT) = (2,1,3): A wins f0(.94), f2(.86); B wins f1(.57); f3,f4,f5 tied.
- T3/T4  C1-CONFLICT (2,1,3): A wins f1(.57), f3(.62); B wins f0(.94); f2,f4,f5 tied.
- T5/T6  C2-COHERENT (3,1,2): A wins f0,f2,f5; B wins f1; f3,f4 tied.
- T7/T8  C2-CONFLICT (3,1,2): A wins f1,f3,f4; B wins f0; f2,f5 tied.
- T9/T10 C3-COHERENT (4,2,0): A wins f0,f2,f4,f5; B wins f1,f3.
- T11/T12 C3-CONFLICT (4,2,0): A wins f1,f3,f4,f5; B wins f0,f2.

QUANTITATIVE PREDICTIONS. LSNC (exact enumeration, k = 2.5, nu = 0.20, eps = 0): P(count winner) = 0.589 for (2,1,3), 0.656 for (3,1,2), 0.630 for (4,2,0) 
 IDENTICAL for the coherent and conflict member of each pair (a strictly zero coherence effect, for every k, nu, eps). CDCC (beta = 2.6, gamma = 1.5, kappa = 0.7, rho = 3.0, eps = 0.10): coherent 0.707 / 0.757 / 0.651 versus conflict 0.544 / 0.576 / 0.553 
 a 10-18 point coherence effect that cannot be switched off (kappa > 0.4 by hypothesis) and that is largest for the sparse (2,1,3) and (3,1,2) cells where the loser's single win is the top expert.

Each subject contributes 48 coherent and 48 conflict trials, so the coherence contrast is estimated within subject with high power. INTERPRETATION: adherence that depends only on the (nA,nB,nT) composition, is flat across the coherence manipulation, and shows the (3,1,2) > (4,2,0) > (2,1,3) ordering with a ceiling near 0.65, uniquely supports LSNC. A systematic coherent > conflict gap (never dropping below 0.5) together with the (3,1,2) > (2,1,3) > (4,2,0) ordering uniquely supports CDCC. Below-chance adherence in the conflict cells would falsify BOTH theories in favour of validity-weighted / one-reason integration, so the design is also self-diagnostic.

CONTROLS: (i) every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; (ii) tied cues are encoded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not covary with composition; (iii) validity order is scrambled relative to display position; (iv) no pair is degenerate 
 every trial has a strict count winner and every trial requires reading more than one rating.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    validities = np.array([0.94, 0.57, 0.86, 0.62, 0.70, 0.78], dtype=float)\n\n    coh_hit = 0.0\n    coh_n = 0.0\n    con_hit = 0.0\n    con_n = 0.0\n\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'],\n                                  data['option_b_ratings'],\n                                  data['response']):\n        a = np.asarray(list(a_raw), dtype=float)\n        b = np.asarray(list(b_raw), dtype=float)\n        n = a.shape[0]\n        if n == 0:\n            continue\n        if validities.shape[0] == n:\n            v = validities\n        else:\n            v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])\n\n        a_win = a > b\n        b_win = b > a\n        nA = int(a_win.sum())\n        nB = int(b_win.sum())\n        if nA == nB:\n            continue  # no count winner; excluded\n        winner = 0 if nA > nB else 1\n\n        disc = a_win | b_win\n        if not disc.any():\n            continue\n        # highest-validity discriminating cue\n        idx = int(np.argmax(np.where(disc, v, -np.inf)))\n        top_owner = 0 if a_win[idx] else 1\n        coherent = (top_owner == winner)\n\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        hit = 1.0 if r == winner else 0.0\n\n        if coherent:\n            coh_hit += hit\n            coh_n += 1.0\n        else:\n            con_hit += hit\n            con_n += 1.0\n\n    if coh_n == 0.0 or con_n == 0.0:\n        return 0.0\n    return float(coh_hit / coh_n - con_hit / con_n)\n",
  "rationale": "The design holds the win-count composition (nA,nB,nT) exactly fixed while flipping whether the highest-validity expert (0.94) sides with or against the count winner. LSNC (advocated) samples cues uniformly at random, so its predicted adherence depends only on (nA,nB,nT) and is mathematically invariant to which cues the winner owns: the coherent-minus-conflict adherence difference is exactly 0 for every parameter setting (k, nu, epsilon). CDCC gates confidence through the convex validity-weighted coherence term C, with kappa >= 0.4 by hypothesis, so it must show a large positive gap (~0.15, i.e. ~0.71 coherent vs ~0.56 conflict). Each subject supplies 48 coherent and 48 conflict trials, so the within-subject difference has SE ~0.10, giving a very tight per-subject estimate and a between-theory separation of many standard errors at N=25. Side mirrors within each cell cancel any side bias, and the coherence label is derived generically (winner owns the most valid discriminating cue), so no trial-index bookkeeping is required."
}
```

## Usage

```json
{
  "input_tokens": 10204,
  "output_tokens": 2327
}
```
