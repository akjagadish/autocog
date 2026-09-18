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
**Validities (n_features=7):** [0.66, 0.95, 0.55, 0.8, 0.6, 0.88, 0.72]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  trial 5: A=[1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1]
  trial 7: A=[0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 1]
  trial 10: A=[1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1, 0]
  trial 11: A=[0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 12: A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 1]
  trial 13: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  trial 15: A=[0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1]
  trial 16: A=[0, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0]

**Rationale:** GOAL: dissociate AEWI (absolute, un-normalised log-odds^gamma weights, gamma~1.4, forward polarity) from RD-WEI (compressed near-tally weights gamma~0.33, reversed subject-level polarity s=-1, and divisive normalization of the margin by the weight of the *discriminating* cues, kappa~0.5). Separation is engineered on three orthogonal axes: (a) the SHAPE of the single-cue confidence ladder, (b) SIGN reversals on multi-cue splits, and (c) WHERE responding is noisy vs deterministic.

ENVIRONMENT: seven experts in SCRAMBLED display order [0.66, 0.95, 0.55, 0.80, 0.60, 0.88, 0.72], so no 'use the first-listed expert' rule can imitate any model and genuine validity re-sorting is required. Log-odds diagnosticities d = (0.663, 2.944, 0.201, 1.386, 0.406, 1.992, 0.944). At AEWI's gamma (~1.415) the weights are w = (0.56, 4.61, 0.10, 1.59, 0.28, 2.65, 0.92): a 45:1 best-to-worst ratio, strongly non-compensatory in ABSOLUTE units. At RD-WEI's gamma (~0.33) the same environment gives w = (0.87, 1.43, 0.59, 1.11, 0.74, 1.26, 0.98): only a ~2.4:1 spread, i.e. a validity-tinted tally. Every family below exploits that 45:1 vs 2.4:1 gap plus the normalization/polarity terms.

EIGHT TRIAL TYPES, each mirrored across the A/B sides (16 unique pairs, K=6, 96 trials) so no side bias can mimic either model. Predicted p(A) at representative parameters (AEWI: gamma=1.415, beta=2.0; RD-WEI: gamma=0.33, kappa=0.52, beta=2.1, s=-1):

(T1, pairs 1-2) ONLY the 0.55 expert discriminates. AEWI ~0.55 (near chance: absolute margin 0.10). RD-WEI ~0.16 (confident REVERSED choice, since after normalization a 1-vs-0 discrimination gives E' = w^(1-kappa) ~ 0.78 even for the worst cue).
(T2, pairs 3-4) ONLY the 0.95 expert discriminates. AEWI ~0.9999 (margin 4.61). RD-WEI ~0.08.
(T3, pairs 5-6) ONLY the 0.72 expert discriminates. AEWI ~0.86. RD-WEI ~0.11.
T1/T2/T3 together are the cleanest quantitative test: AEWI is forced into a STEEP monotone single-cue ladder (0.55 -> 0.86 -> 0.9999) scaled by absolute cue diagnosticity, while RD-WEI's compression + divisive normalization make the ladder nearly FLAT and on the opposite side of chance (0.16 -> 0.11 -> 0.08). No reparameterisation of either model reproduces the other's shape here.

(T4, pairs 7-8) 3-vs-4 SPLIT, top cue on the small side: A holds 0.95+0.55+0.60, B holds 0.88+0.80+0.72+0.66. AEWI margin is slightly NEGATIVE in absolute units (4.99 vs 5.72) -> p(A) ~ 0.19 (chooses B). RD-WEI, reading fewer weighted marks as better, chooses A: ~0.75. Full sign reversal.
(T5, pairs 9-10) ALL SEVEN cues discriminate, 4-vs-3 count but grossly weight-lopsided: A holds 0.95+0.88+0.60+0.55, B holds 0.80+0.72+0.66. AEWI ~0.9999 (absolute margin 4.58); RD-WEI ~0.31 - a near-even compressed tally divided by the full discriminating mass, hence a small relative share, mild reversed preference and (uniquely) large predicted between-subject spread exactly here.
(T6, pairs 11-12) 1-vs-3 with the top cue alone: A holds 0.95, B holds 0.66+0.60+0.55. Both models pick A but for opposite reasons (AEWI: 4.61 > 0.94 in absolute units, ~0.999; RD-WEI: A carries fewer marks, ~0.70). A direction-free MAGNITUDE test where RD-WEI masquerades as one-reason decision making.
(T7, pairs 13-14) UNANIMOUS 7-0 dominance. AEWI ~1.00 for the all-1 option; RD-WEI ~0.005 for it (reversed frame), and capped at S^(1-kappa) by normalization. This is the single most decisive polarity test.
(T8, pairs 15-16) 1v1 MID CONFLICT: 0.80 favours A, 0.72 favours B, all else ties. AEWI ~0.79 (1.59 vs 0.92 in absolute units); RD-WEI ~0.45 (compressed weights 1.11 vs 0.98 make it a coin flip). Isolates the compression exponent with only two cues in play.

DIAGNOSTIC SUMMARY of p(A) across T1..T8: AEWI = (0.55, 0.9999, 0.86, 0.19, 0.9999, 0.999, 1.00, 0.79); RD-WEI = (0.16, 0.08, 0.11, 0.75, 0.31, 0.70, 0.005, 0.45). The vectors disagree in SIGN on six of eight types and, crucially, differ in SHAPE: AEWI cannot produce a flat single-cue profile (T1-T3) at any gamma/beta, nor near-chance responding on the weight-lopsided seven-cue pair T5 while staying decisive on T2; RD-WEI cannot produce a validity-scaled single-cue gradient, cannot be deterministic on the unanimous pair in the all-1 direction, and cannot be decisive on T5. Even a sign-flipped AEWI fails, since it would predict 0.45/0.0001/0.14/0.81/0.0001 on T1-T5 versus RD-WEI's 0.16/0.08/0.11/0.75/0.31.

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
**Description:** **RD-WEI: Relative (Normalized) Directional Weighted Evidence Integration.**

This keeps the two claims that carried the previous candidate and adds a third, which is the theoretical content of this revision.

(1) COMPRESSED, COMPENSATORY INTEGRATION. Subjects integrate ALL expert ratings. Each stated validity v_i is turned into normative evidence d_i = log(v_i/(1-v_i)), which enters the comparison after a strongly *compressive* transform w_i = d_i^gamma with gamma well below 1 (~0.2-0.5). The weight profile is therefore only mildly validity-ordered - much closer to a tally than to log-odds weighting - so no single expert can lexicographically out-weigh three or four others, yet validity still decides 1-vs-1 comparisons.

(2) SUBJECT-LEVEL POLARITY FRAME. The direction in which a positive binary rating pushes preference is a latent subject-level frame s in {-1,+1}: the decision variable is s * E. When the frame is inverted (s = -1, the overwhelming majority in these environments) a '1' is read as a mark/flag/non-selective endorsement and the subject prefers the option carrying the SMALLER validity-weighted count of positive ratings. Because in these designs the option with fewer marks is usually also the option favoured by the single most valid discriminating cue, reversed integration *masquerades* as one-reason (Take-The-Best) decision making on cue-conflict trials while producing strongly NEGATIVE weighted-margin contrasts on dominance/unanimity pairs - a signature no forward integrator and no lexicographic rule can produce.

(3) NEW: EVIDENCE IS READ IN RELATIVE, NOT ABSOLUTE, UNITS (divisive normalization). The margin is not evaluated in absolute log-odds units; it is evaluated as a *share of the evidence actually in play* on that trial. Only cues that discriminate enter the comparison, and the margin is divided by the total weight of those discriminating cues raised to a power kappa (~0.4-0.6):

    E' = s * [ sum_i w_i (a_i - b_i) ] / ( sum_{i: a_i != b_i} w_i )^kappa ,   P(A) = sigmoid(beta * E'), then lapse.

Psychologically this is a Weber-like normalization of the comparison: what governs confidence is how *lopsided* the discriminating evidence is, not how much of it there is. Two consequences distinguish RD-WEI from the un-normalized version: (a) a one-cue-versus-one-cue disagreement (few cues in play) is resolved almost as decisively as a 1-versus-5 rout, so single-cue and small-set discriminations stay sharp; (b) trials in which many cues discriminate but the split is nearly even (e.g. 4 marks vs 6 marks, or 2 vs 3 across a 7-cue array) become markedly noisier than their absolute margin would imply, and even unanimous/dominance pairs stop being perfectly deterministic once every cue is in play. This is exactly the graded-ness pattern the data show: near-ceiling reversed responding on lopsided splits, sub-ceiling responding on many-cue near-even splits, and sizeable between-subject disagreement precisely on the near-zero-share pairs.

The family still nests the classics: s=+1, gamma->0, kappa=0 = Tallying; s=+1, gamma=1, kappa=0 = Franklin's rule / naive Bayes; s=+1, gamma->inf = TTB; kappa=1 = pure average-diagnosticity-per-discriminating-cue; s=-1 gives the reversed-frame regime these environments demand.

**Parameters:**
- gamma: [0.18, 0.48]
- kappa: [0.42, 0.62]
- beta: [1.8, 2.5]
- epsilon: [0.0, 0.12]
- p_reverse: [0.94, 1.0]
- polarity_u: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Relative Directional Weighted Evidence Integration (RD-WEI).
    #   d_i = log(v_i/(1-v_i))                    normative cue evidence
    #   w_i = d_i ** gamma,  gamma < 1            COMPRESSED (near-tally) weights
    #   E   = sum_i w_i (a_i - b_i)               un-normalised weighted margin
    #   S   = sum of w_i over DISCRIMINATING cues (a_i != b_i)
    #   E'  = E / S**kappa                        relative (share-of-evidence) margin
    #   s   = -1 with prob p_reverse, else +1     subject-level polarity frame
    #   P(A) = sigmoid(s * beta * E'), then lapse epsilon toward uniform.
    # History is ignored (no feedback in this domain).
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        if stim.shape[0] % 2 != 0:
            return np.ones(2) / 2.0
        half = stim.shape[0] // 2
        stim = np.vstack([stim[:half], stim[half:]])
    if stim.ndim != 2 or stim.shape[0] != 2:
        return np.ones(2) / 2.0

    n_features = int(stim.shape[1])
    if n_features == 0:
        return np.ones(2) / 2.0

    # ---- validities ------------------------------------------------
    try:
        v = np.asarray(parameters.get("validities", None), dtype=float).ravel()
    except Exception:
        v = np.array([])
    if v.size != n_features or not np.all(np.isfinite(v)):
        v = np.linspace(0.90, 0.55, n_features)
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---- compressed diagnosticity weights (un-normalised) ----------
    gamma = float(parameters["gamma"])
    d = np.maximum(np.log(v / (1.0 - v)), 1e-12)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        w = np.power(d, gamma)
    w = np.where(np.isfinite(w), w, 0.0)
    if not np.isfinite(np.sum(w)) or np.sum(w) <= 0.0:
        w = np.ones(n_features, dtype=float)

    # ---- weighted margin (forward frame) ---------------------------
    diff = stim[0] - stim[1]
    E = float(np.dot(w, diff))
    if not np.isfinite(E):
        E = 0.0

    # ---- divisive normalization by the evidence actually in play ----
    kappa = float(parameters["kappa"])
    disc = np.abs(diff) > 0.0
    S = float(np.sum(w[disc])) if np.any(disc) else 0.0
    if S > 1e-12 and kappa != 0.0:
        denom = S ** kappa
        if np.isfinite(denom) and denom > 1e-12:
            E = E / denom
    if not np.isfinite(E):
        E = 0.0

    # ---- subject-level polarity frame ------------------------------
    u = float(parameters["polarity_u"])
    q = float(parameters["p_reverse"])
    s = -1.0 if u < q else 1.0

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    z = np.array([s * beta * E, 0.0], dtype=float)
    z = np.clip(z, -500.0, 500.0)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    V = np.array([0.66, 0.95, 0.55, 0.80, 0.60, 0.88, 0.72], dtype=float)\n    hits = 0\n    n = 0\n    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):\n        try:\n            av = np.asarray(list(a), dtype=float)\n            bv = np.asarray(list(b), dtype=float)\n        except Exception:\n            continue\n        if av.shape[0] != bv.shape[0] or av.shape[0] == 0:\n            continue\n        d = av - bv\n        nz = np.nonzero(d)[0]\n        if nz.size == 0:\n            continue\n        include = False\n        # (1) unanimous / total-dominance pairs: every cue discriminates, same direction\n        if nz.size == d.shape[0] and (np.all(d > 0) or np.all(d < 0)):\n            include = True\n        # (2) single-cue pairs where the lone discriminating cue is highly valid\n        elif nz.size == 1 and d.shape[0] == V.shape[0] and V[nz[0]] >= 0.85:\n            include = True\n        if not include:\n            continue\n        # target = the option carrying the positive rating(s) on the discriminating cue(s)\n        target = 0 if d[nz[0]] > 0 else 1\n        try:\n            chosen = int(r)\n        except Exception:\n            continue\n        if chosen == target:\n            hits += 1\n        n += 1\n    if n == 0:\n        return 0.5\n    return float(hits) / float(n)",
  "rationale": "This isolates the two trial families where the two theories make maximally opposed, low-variance predictions: (a) the unanimous 7-0 dominance pairs (trials 13/14) and (b) the pairs decided by the single highly diagnostic 0.95 expert (trials 3/4). In both families I score the fraction of choices for the option carrying the POSITIVE expert ratings on the discriminating cue(s). AEWI is a forward integrator in absolute log-odds units: the margin is 4.61 (single 0.95 cue) or ~10.7 (unanimous), so choices are essentially deterministic toward the positively-rated option, giving ~0.98-1.00 with only the small lapse (eps<=0.03) pulling it down. RD-WEI's subject-level reversed polarity frame (s=-1 for ~97% of subjects) means the positively-marked option is systematically AVOIDED, and divisive normalization keeps the reversed choice confident even on the single-cue pairs, giving ~0.02-0.08. The metric therefore has a mean gap near 0.95 while being extremely tight per subject (24 trials each, near-ceiling/near-floor binomials), so it survives Welch's test at N=25. It is not trivially true for AEWI: any near-tally forward integrator would also score high, and a lexicographic TTB rule would too \\u2014 the metric is diagnostic of the polarity/absolute-margin claim, not of the specific weighting exponent."
}
```

## Usage

```json
{
  "input_tokens": 11362,
  "output_tokens": 3459
}
```
