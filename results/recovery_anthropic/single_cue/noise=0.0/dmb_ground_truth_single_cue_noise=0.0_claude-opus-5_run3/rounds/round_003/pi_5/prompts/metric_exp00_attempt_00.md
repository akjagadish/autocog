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
**Validities (n_features=6):** [0.68, 0.55, 0.95, 0.6, 0.86, 0.72]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 4: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 7: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 11: A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 12: A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 13: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 14: A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 15: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** GOAL: dissociate anchored compensatory integration with LOAD-GATED weight flattening + supra-additive corroboration + PARTIAL (psi~0.3) margin normalisation from read-order scanning (screen-position geometric weights, FULL per-trial normalisation, ~65% primacy / ~35% recency readers).

VALIDITIES BY SCREEN POSITION: [.68, .55, .95, .60, .86, .72]. The single best expert (95%) sits in the MIDDLE (pos2), a mid expert leads (pos0 = .68) and a mid expert closes (pos5 = .72). Under the competitor (rank = gamma*vrank + (1-gamma)*pos, gamma<=.25) the ranks are [0.75, 2.0, 1.5, 3.25, 3.25, 4.25], so pos0 is the primacy anchor and pos5 the recency anchor for EVERY (lam, gamma) in its box, while the advocated model's diagnosticity term keeps pos2 dominant at low load. Three distinct anchors, robustly.

THE CENTRAL ADVERSARIAL MOVE: several cells come in PAIRS with an IDENTICAL difference vector d = a - b, differing only in how many features carry a shared positive endorsement (L = both options rated 1). The competitor's evidence S = sum(w*d)/sum(w*|d|) depends only on d and screen position, so it predicts EXACTLY THE SAME choice probability within each such pair u2014 no parameter setting can produce any load effect. The advocated model predicts large, signed load effects (rho_eff = rho/(1+cL) flattens the validity gradient; delta_eff = delta*L/(1+L) turns on counting).

CELLS (8 base designs, each mirrored across sides -> 16 unique pairs, K = 6 -> 96 trials; advocated favours A on 8 and B on 8 overall, so side bias cannot mimic anything).

(1) SINGLE-CUE LADDER, trials 1-2 (+mirrors 9-10). Exactly one expert discriminates and L = 0: pos2 (95%) in trial 1, pos3 (60%) in trial 2. The competitor is pinned at |S| = 1 in BOTH cells for every lam, gamma and orientation -> a FLAT ceiling (~.95-1.00). The advocated model reads a single cue against a partially attenuated yardstick, E = w_j^(1-psi), giving a steep ladder: P ~ .99 for the 95% cue but only ~.55-.60 for the 60% cue. Flat ceiling vs. a 40-point ladder is a purely quantitative signature the normalising model cannot produce.

(2) LOAD x CUE-STRENGTH CROSSOVER, trials 3-4 (+11-12). Trial 3 repeats trial 2's d (only pos3 discriminates) but adds three shared positive endorsements (L = 3); trial 4 repeats trial 1's d (only pos2 discriminates) with L = 4. Competitor: identical to trials 2 and 1 respectively (zero effect). Advocated: flattening RAISES the weak cue's relative weight (P .55-.60 -> .72-.80, about +18 pts) and LOWERS the strong cue's (P ~.99 -> ~.92-.93, about -6 pts). A crossover interaction of ~25 points produced by ratings that discriminate nothing at all is impossible for any model that only reads d.

(3) LOAD-INDUCED DIRECTION REVERSAL, trials 5-6 (+13-14) u2014 the sharpest cells. Both have d = [+1, 0, -1, 0, 0, +1]: the first-read 68% cue and the last-read 72% cue favour A, the mid-list 95% cue favours B. Trial 5 has L = 0, trial 6 pads three shared positives (L = 3). The competitor predicts A in BOTH and in both orientations (primacy S ~ +.43, recency S ~ +.93; P(A) ~ .85-.99) with no difference between them u2014 no orientation escape hatch, no load sensitivity. The advocated model predicts B at L = 0 (P(A) ~ .07-.10, checked at the extreme primacy corner alpha=2.1/lam=0.9/rho=2.3) and A at L = 3 (P(A) ~ .84-.92, checked across c in [.65,1.10] and rho in [2.3,2.7]) because flattening plus the 2-vs-1 corroboration bonus flips the sign of E. Same d, opposite direction: within-subject reversal across load is diagnostic no matter what the individual's parameters are.

(4) UNIMODAL vs BIMODAL, trial 7 (+15). d = [0,0,+1,0,0,-1]: the middle 95% expert against the last-read 72% expert, L = 0. Advocated: unimodal P(A) ~ .95-.98 for every subject. Competitor: primacy readers S ~ +.93 (P(A) ~ .98) but the ~35% recency readers S ~ -.93 (P(A) ~ .02) -> a strongly BIMODAL subject-level distribution with aggregate ~ .65. Both the aggregate level and the shape of the between-subject distribution differ.

(5) MAJORITY-DIRECTION DISAGREEMENT, trial 8 (+16). d = [+1,0,-1,0,0,0]: leading 68% cue vs. middle 95% cue, L = 0. Advocated: B, confidently (P(A) ~ .04-.09) because diagnosticity beats the primacy multiplier when rho is convex and load is zero. Competitor: the 65% primacy majority chooses A (P(A) ~ .85-.99), the recency minority chooses B; aggregate ~ .60-.65, bimodal. This cell also anchors the interpretation of trials 5/6: if subjects show the primacy-anchored A response here AND no load effect in 5-vs-6, the competitor wins outright.

Every pair has at least one discriminating cue, no cell is degenerate, and each of the three load-matched contrasts (2 vs 3, 1 vs 4, 5 vs 6) is a within-subject test on which the competing theory is mathematically forced to predict a null.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Anchored compensatory integration with load-gated degradation of weighted-additive processing and partial evidence amplification.**

(1) *Diagnosticity is read convexly.* The subjective pull of expert j is d_j = (v_j - 0.5)^rho with rho ~2.3-2.7, so a .95 expert is treated as a near-oracle and .55-.65 experts as near-noise.

(2) *Screen position is a graded attention multiplier, not a search order.* a_j = 1 + alpha*exp(-lam*j) with alpha ~1.5-2.1, lam ~0.9-1.3; the cue weight is the PRODUCT w_j ∝ d_j * a_j, rescaled once to sum to 1 (never renormalised per trial). The first-read cue is rehearsed longest and anchors the running impression, breaking validity ties toward the pos-0 side, but it cannot overturn a genuine diagnosticity gap.

(3) **Processing load is the single switch that degrades weighted-additive integration toward unit-weight counting.** Let L be the number of features on which BOTH products received a positive expert rating (design-general form: the count of non-discriminating positive endorsements; equivalently L/n_features as a density). Under this load two things happen together, both expressions of the same effort–accuracy tradeoff: (a) the diagnosticity exponent flattens, rho_eff = rho / (1 + c*L), so validity differentiation degrades toward equal weighting; (b) the decision maker falls back on the crude 'how many experts back each side' cue, i.e. supra-additive corroboration appears with delta_eff = delta * L/(1+L). Corroboration is silent on sparse displays (L = 0): counting is a load-induced shortcut, not a permanent feature of integration.

(4) **NEW — the margin is read against a partially-attenuated yardstick, not against the raw amount of evidence considered.** Two trials with the same signed weighted margin are not equally decisive: a margin obtained while very little total diagnostic weight was in play feels proportionally larger than the same margin obtained amid a large mass of conflicting weight, but the correction is only PARTIAL. Evidence is E = (Score_A - Score_B) / D^psi, where D = total discriminating weight and psi ~0.2-0.4 is well below 1. At psi = 0 this is the pure absolute margin (all thin-evidence trials near chance); at psi = 1 it would be the fully relative, per-trial-normalised evidence the arbiter rejected (every decision equally confident). Real subjects sit near the low end: thin, near-balanced displays are magnified enough to escape chance, while dense displays keep an essentially absolute margin. This is the theory's sharpest quantitative claim — decisiveness scales sub-linearly with the reciprocal of the evidence mass.

(5) Choice = logistic(beta*E) with a lapse epsilon. No early stopping, no threshold rule, no conclusion-graded commitment, no full per-trial normalisation, and individual differences are unimodal jitter in rho, alpha, lam, delta, c, psi, beta and lapse — no strategy subpopulations, no primacy/recency reader types.

**Parameters:**
- rho: [2.3, 2.7]
- alpha: [1.5, 2.1]
- lam: [0.9, 1.3]
- delta: [0.30, 0.55]
- load_flatten: [0.65, 1.10]
- psi: [0.20, 0.40]
- beta: [6.0, 8.5]
- epsilon: [0.02, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. unpack the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b

    if not np.any(np.abs(d) > 1e-12):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> convex diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fallback: assume the display is validity-sorted, descending
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0)

    rho = float(np.clip(parameters.get('rho', 2.5), 0.1, 8.0))
    alpha = float(np.clip(parameters.get('alpha', 1.8), 0.0, 6.0))
    lam = float(np.clip(parameters.get('lam', 1.1), 0.01, 6.0))
    beta = float(np.clip(parameters.get('beta', 7.0), 0.0, 200.0))
    eps = float(np.clip(parameters.get('epsilon', 0.05), 0.0, 1.0))
    delta = float(np.clip(parameters.get('delta', 0.40), 0.0, 2.0))
    c_load = float(np.clip(parameters.get('load_flatten', 0.85), 0.0, 3.0))
    psi = float(np.clip(parameters.get('psi', 0.30), 0.0, 0.65))

    # --- overlapping positive endorsements = processing load.
    # L = number of features on which BOTH options were endorsed.  These
    # cues carry no discriminating information but they enlarge the
    # comparison problem, and under that load validity differentiation
    # degrades toward unit-weight counting.
    L = float(np.count_nonzero((a > 0.5) & (b > 0.5)))
    rho_eff = rho / (1.0 + c_load * L)
    rho_eff = float(np.clip(rho_eff, 0.6, 8.0))

    diag = np.power(np.clip(val - 0.5, 1e-12, None), rho_eff)   # diagnosticity

    # ------------------------------------------------------------------
    # 3. graded primacy attention multiplier (screen-position anchor)
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    att = 1.0 + alpha * np.exp(-lam * pos)

    w = diag * att
    tot = float(np.sum(w))
    if (not np.isfinite(tot)) or tot <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = w / tot            # fixed rescaling only (NOT per-trial normalisation)

    # ------------------------------------------------------------------
    # 4. LOAD-GATED supra-additive corroboration: the crude 'how many
    #    experts back each side' bonus is a shortcut people fall back on
    #    only when the display is dense (L > 0).  On sparse displays
    #    (L = 0) delta_eff = 0 and integration stays purely
    #    weighted-additive.
    # ------------------------------------------------------------------
    delta_eff = delta * (L / (1.0 + L))

    pos_mask = d > 1e-12
    neg_mask = d < -1e-12
    s_a = float(np.sum(w[pos_mask] * np.abs(d[pos_mask])))
    s_b = float(np.sum(w[neg_mask] * np.abs(d[neg_mask])))
    k_a = float(np.count_nonzero(pos_mask))
    k_b = float(np.count_nonzero(neg_mask))
    f_a = (k_a ** delta_eff) if k_a > 0.0 else 0.0
    f_b = (k_b ** delta_eff) if k_b > 0.0 else 0.0

    # ------------------------------------------------------------------
    # 5. PARTIAL amplification: the raw margin is read against a
    #    sub-linearly attenuated yardstick D^psi (psi well below 1), so
    #    thin displays are magnified somewhat while dense displays keep an
    #    essentially absolute margin.  psi = 0 -> pure absolute margin;
    #    psi = 1 would be the forbidden full per-trial normalisation.
    # ------------------------------------------------------------------
    D = s_a + s_b
    if not np.isfinite(D) or D <= 1e-12:
        return np.array([0.5, 0.5])
    E = (s_a * f_a - s_b * f_b) / (D ** psi)

    z = np.clip(beta * E, -60.0, 60.0)
    p_a = 1.0 / (1.0 + np.exp(-z))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p
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
**Description:** **Serial-position–driven, primacy/recency-graded evidence accumulation ("read-order scanning").**

People do not re-sort the expert ratings by the stated validities before deciding. They inspect the ratings in the order in which they appear on the screen (reading order) and their attention decays sharply across that inspection sequence, so the evidence carried by a cue is discounted geometrically by its *serial position*, not by its validity. Stated validities exert only a weak, partial pull on the inspection order (parameter gamma, small).

Two further claims:

1. **The gradient is steep but graded.** Weights fall off as w_j = exp(-lambda * rank_j) with |lambda| well above zero, so decisions are usually driven by the first cue(s) that discriminate — near-lexicographic but not strictly all-or-none: when the leading cues tie, the next-inspected cues take over, and when several later cues line up they can occasionally overturn a leading cue if the gradient is shallow.

2. **The gradient has a sign: most readers are primacy-oriented, a substantial minority are recency-oriented.** Reading a short list produces either an anchoring-on-the-first-item strategy (primacy, lambda > 0) or a last-impression-counts strategy (recency, lambda < 0). The population is bimodal: about one third of subjects anchor on the *last-read* cue. Near-zero gradients (i.e., genuine equal-weight tallying) are essentially absent — people do not integrate all six/five cues equally.

Evidence is evaluated *relatively*: the signed weighted difference is normalised by the total weight of the cues that actually discriminate, so a decision based on one early cue is as confident as a decision based on many, and only genuinely near-balanced weighted evidence produces hesitation. Choice is a softmax on this relative evidence with inverse temperature beta, plus a lapse epsilon.

Because validity order and screen order coincide in a validity-sorted display but dissociate in a scrambled display, this theory predicts TTB-looking behaviour in the former and tally-looking behaviour in the latter — without either heuristic being the actual mechanism. The causal driver is *where the cue sits on the screen*.

**Parameters:**
- lam: [1.2, 3.5]
- gamma: [0.0, 0.25]
- beta: [4.0, 12.0]
- epsilon: [0.0, 0.10]
- scan_orientation: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---- unpack the trial stimulus -> (2, n_features) ------------------
    if isinstance(state, dict):
        a = np.asarray(list(state.get('option_a_ratings', [])), dtype=float)
        b = np.asarray(list(state.get('option_b_ratings', [])), dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            stim = stim.reshape(2, -1)
        elif stim.ndim > 2:
            stim = stim.reshape(2, -1)
    if stim.shape[0] != 2:
        stim = stim.reshape(2, -1)

    a = stim[0].astype(float)
    b = stim[1].astype(float)
    n = a.shape[0]
    d = a - b

    # ---- inspection ranks: screen position, weakly pulled by validity --
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fallback: assume the display is already validity-sorted
        val = np.linspace(0.95, 0.55, n)

    order = np.argsort(-val, kind='stable')       # 0 = highest validity
    vrank = np.empty(n, dtype=float)
    vrank[order] = np.arange(n, dtype=float)
    pos = np.arange(n, dtype=float)               # reading-order index

    gamma = float(parameters.get('gamma', 0.0))
    gamma = min(max(gamma, 0.0), 1.0)
    rank = gamma * vrank + (1.0 - gamma) * pos

    # ---- signed attention gradient (primacy vs recency reader) --------
    lam_mag = float(parameters.get('lam', 2.0))
    orient = float(parameters.get('scan_orientation', 1.0))
    RECENCY_SHARE = 0.35                          # population constant
    lam = -lam_mag if orient < RECENCY_SHARE else lam_mag

    z = -lam * (rank - float(np.mean(rank)))      # centered for stability
    z = z - float(np.max(z))
    w = np.exp(z)

    # ---- relative (normalised) weighted evidence ----------------------
    num = float(np.sum(w * d))
    den = float(np.sum(w * np.abs(d)))
    if den <= 1e-12:
        S = 0.0                                   # nothing discriminates
    else:
        S = num / den                             # in [-1, 1]

    beta = float(parameters.get('beta', 6.0))
    eps = float(parameters.get('epsilon', 0.0))
    eps = min(max(eps, 0.0), 1.0)

    logits = np.array([beta * S, 0.0], dtype=float)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    p = e / np.sum(e)

    p = (1.0 - eps) * p + eps * 0.5
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    hi_load_hits, hi_load_n = 0, 0\n    lo_load_hits, lo_load_n = 0, 0\n\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'],\n                                  data['option_b_ratings'],\n                                  data['response']):\n        try:\n            a = np.asarray(list(a_raw), dtype=float)\n            b = np.asarray(list(b_raw), dtype=float)\n        except Exception:\n            continue\n        n = int(min(a.shape[0], b.shape[0]))\n        if n == 0:\n            continue\n        a = a[:n]\n        b = b[:n]\n        d = a - b\n\n        pos_idx = np.where(d > 0.5)[0]   # features favouring A\n        neg_idx = np.where(d < -0.5)[0]  # features favouring B\n        k_a = pos_idx.size\n        k_b = neg_idx.size\n\n        # keep only the 3-cue conflict cells: two cues on one side, one on the other\n        if not ((k_a == 2 and k_b == 1) or (k_a == 1 and k_b == 2)):\n            continue\n\n        # did the subject pick the side backed by the LARGER NUMBER of cues?\n        if k_a == 2:\n            chose_many = 1 if int(resp) == 0 else 0\n        else:\n            chose_many = 1 if int(resp) == 1 else 0\n\n        # processing load = shared positive endorsements (non-discriminating 1s)\n        L = int(np.count_nonzero((a > 0.5) & (b > 0.5)))\n\n        if L > 0:\n            hi_load_hits += chose_many\n            hi_load_n += 1\n        else:\n            lo_load_hits += chose_many\n            lo_load_n += 1\n\n    if hi_load_n == 0 or lo_load_n == 0:\n        return 0.0\n\n    return float(hi_load_hits / hi_load_n - lo_load_hits / lo_load_n)\n",
  "rationale": "This isolates the design's sharpest within-subject dissociation: trials 5/13 (L=0) and 6/14 (L=3) share the EXACT same difference vector d = (+1,0,-1,0,0,+1) up to side-mirroring \\u2014 two mid-validity cues (screen positions 0 and 5) against the single 95% cue at position 2. They differ only in the number of shared positive endorsements, which discriminate nothing.\n\nThe metric is P(choose the 2-cue side | dense display) - P(choose the 2-cue side | sparse display), pooled over both mirror orientations so any side bias cancels.\n\nCompeting theory (read-order scanning): its evidence S = sum(w*d)/sum(w*|d|) is a function of d and screen position ONLY. Shared 1s are invisible to it, so it is mathematically forced to give the identical choice probability in both cells, for every lam, gamma, beta, epsilon and for both primacy and recency orientations. Predicted value: 0 (only binomial noise). Its per-subject variance is small because both proportions sit near the same ceiling and their sampling errors are of the same modest size (~12 trials each).\n\nAdvocated theory: load flattens the diagnosticity exponent (rho_eff = rho/(1+cL)) and switches on supra-additive corroboration (delta_eff = delta*L/(1+L)), flipping the decision from the single strong cue to the 2-cue majority. Hand-computing at mid parameters gives P(2-cue side) ~ .10 at L=0 and ~ .88 at L=3, i.e. a metric value near +0.75 to +0.80, stable across the whole parameter box (the sign flip is robust to rho, alpha, lam, c).\n\nSo the expected contrast is roughly 0.78 vs 0.00 with per-subject SDs on the order of 0.1-0.15 \\u2014 a very large, tight separation at N=25. It is not trivially true for my theory: it is a genuine empirical claim that adding informationally irrelevant shared endorsements reverses which option people pick."
}
```

## Usage

```json
{
  "input_tokens": 11430,
  "output_tokens": 5692
}
```
