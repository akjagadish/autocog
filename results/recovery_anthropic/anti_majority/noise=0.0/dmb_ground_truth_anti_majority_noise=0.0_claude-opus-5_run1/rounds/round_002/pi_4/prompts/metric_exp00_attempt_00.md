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
**Validities (n_features=5):** [0.62, 0.83, 0.93, 0.54, 0.72]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 8: A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 9: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 11: A=[1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 12: A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0]
  trial 13: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 14: A=[1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0]
  trial 15: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  trial 16: A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1]
  trial 17: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 18: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 19: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 21: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 24: A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate RVW-DE+/U (advocated) from Take-The-Best (competing) on three orthogonal signatures: (i) READABILITY FLATTENING, (ii) the STRICT-UNANIMITY consensus premium, (iii) flat-vs-graded choice probability. TTB's core commitment is that on every discriminating trial it sides with the highest-validity discriminating expert at ONE flat level sigmoid(beta) (plus lapse), regardless of the other cues, of endorsement counts, of coherence, and of where the expert is displayed.

DISPLAY (deliberately scrambled): col0=.62, col1=.83, col2=.93, col3=.54, col4=.72. Kendall tau between column index and validity is exactly 0 (5 concordant / 5 discordant pairs) and the best expert (.93) sits in a MIDDLE column, so under RVW kappa = k0 + k1*1 + k2*1 lies in [0.67, 1.0] across the whole parameter box: validity weights are strongly flattened toward equal weighting and the reading-order fallback (phi_eff = phi*(0.35+1.15*kappa)) is maximally recruited. Effective weights therefore run roughly [.18-.26, .22-.24, .20-.34, .12-.17, .13-.16] by column (mid-parameter ~[.223,.237,.253,.148,.138]) - the .93 expert never commands more than ~1/3 of the weight, and two or three other columns always outweigh it. TTB's cascade, by contrast, is col2 > col1 > col4 > col0 > col3, bound to experts, not columns.

12 base pairs, each mirrored (A/B swapped) so neither key is systematically correct: 24 unique pairs, K = 4, 96 trials. TTB predicts A on 11 of the 12 base pairs (and B on P1) at IDENTICAL probability throughout.

FAMILY L - COALITION LADDER (pairs 1-4 + mirrors). A carries only the .93 endorsement (col2); B carries a growing coalition of other columns: {col0}, {col0,col3}, {col0,col3,col4}, {col0,col1,col3,col4}. TTB stops at col2 and picks A at ceiling in all four, with zero modulation. RVW predicts a steep monotone ladder that CROSSES CHANCE: mid-parameter cores +.03 / -.12 / -.26 / -.49, i.e. p(A) ~ .55 / .30 / .14 / .03 at beta=7. Coherence is exactly cancelling in L1 and L4 (1/5 vs 1/5 and 1/5 vs 4/5 give identical |2m-1|) and negligible elsewhere, so this family is a pure weighted-integration/flattening test. L1 additionally is the near-cancelling cell where the RVW sign flips inside its own parameter box, so it should show the LARGEST between-subject spread of any family.

FAMILY U - UNANIMITY STEP WITH WEIGHTED EVIDENCE HELD EXACTLY CONSTANT (pairs 5-8 + mirrors). All four cells differ on col3 and ONLY col3, so the weighted-evidence term is identical and TTB must give the same ceiling probability in all four. Only panel coherence changes: (U1) 1/5 vs 0/5 - the LOSER is unanimous, so it collects nearly the whole premium; (U2) 3/5 vs 2/5 - neither panel coherent; (U3) 4/5 vs 3/5 - still no premium under high q even though mean rating rose; (U4) 5/5 vs 4/5 - the WINNER is unanimous. RVW predicts p(A) ~ .15 / .74 / .75 / .97: a sub-chance reversal when a lone weak endorsement faces an all-zero panel, a flat plateau across U2-U3 (which a smooth, low-q coherence rule could not produce - it would make U3 clearly exceed U2), and a jump at exactly 0 dissenters. This non-monotone step at the unanimity boundary, with weighted evidence pinned, is the sharpest falsifiable signature and is impossible under any re-weighting of cues.

FAMILY V - PREMIUM ADDITIVITY AT A HEAVIER COLUMN (pairs 11-12 + mirrors). Same manipulation on col0 (weight ~.22 rather than ~.15): [1,0,0,0,0] vs all-zeros, and [1,1,1,1,1] vs [0,1,1,1,1]. RVW predicts p(A) ~ .22 and ~.98 - the same rho-sized step, shifted upward by the larger cue weight (V1 > U1, V2 > U4). TTB again predicts one flat ceiling.

FAMILY P - POSITION VERSUS VALIDITY (pair 9 + mirror). Only col0 (.62, leftmost) and col4 (.72, rightmost) discriminate, endorsement counts and coherence identical, so theta and rho are inert. TTB decisively picks the col4 (.72) option. RVW, with flattened weights plus the recruited reading-order gradient, predicts the OPPOSITE (col0) option at p ~ .64 mid-parameter, ranging from chance to clearly reversed. Any failure to prefer the higher-validity expert here is fatal for TTB.

FAMILY D - DOMINANCE ANCHOR (pair 10 + mirror). [1,1,1,1,1] vs [0,0,0,0,0]: both panels unanimous so the premium cancels exactly, all weights point the same way. Both models are at ceiling; this pins beta and epsilon for BOTH models, making every other family an out-of-sample prediction rather than a free fit.

DECISION RULE. If TTB is right: one flat probability level, calibrated by Family D, accounts for L1-L4, U1-U4, V1-V2 alike (all choosing the option favoured by the first discriminating cue), no ladder gradation, no coherence effect, and Family P shows decisive choice of the .72 option. If RVW-DE+/U is right: the L ladder falls monotonically and crosses chance, the U ladder shows the non-monotone unanimity step (.15 / .74 / .75 / .97) with U2 = U3, V mirrors U at a higher level, Family P reverses or sits at chance, and between-subject spread peaks in L1 and P.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Readability-limited validity weighting with a strict-unanimity consensus premium (RVW-DE+/U), with broad determinacy heterogeneity.**

A single stationary, fully compensatory integration rule underlies choice in all cue-based binary-feature environments; the apparent switching between 'take-the-best-like' and 'tallying-like' behaviour across environments is produced by modulators that are properties of the *display and the evidence pattern*, not of the strategy, plus wide between-person variation in decision determinacy.

(1) **Graded validity weighting.** Each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, w_j ∝ logit(v_j)^gamma, gamma spread over ~1.05–2.0 across people. At the population mean this is steep enough that a clearly best expert usually survives a coalition of three or four weak experts, yet weak enough that a near-peer plus a couple of moderate cues can overturn the top cue.

(2) **Binding readability, not position.** Stated validities must be bound to displayed columns. When the display order is *monotone* in validity (ascending OR descending) this binding is trivial and the validity weights are used essentially intact. When the order is scrambled, binding fails partially and the weights are flattened toward EQUAL weighting: w_eff = (1-kappa)·w + kappa/n, with kappa = k0 + k1·(1-|Kendall tau(column index, validity)|) + k2·[best expert not in an extreme column]. Direction-free: reversing a sorted display costs nothing, permuting it costs a lot.

(3) **Readability-gated reading-order fallback.** The left-to-right attention gradient is a fallback recruited only when validity-to-column binding fails: phi_eff = phi·(r0 + r1·kappa). It only *scales* weights, never re-sorts them.

(4) **Strict-unanimity consensus premium (dissent aversion).** The premium attaches to *consensus itself*, and consensus is an essentially all-or-none perceptual property: a panel is 'agreed' only when it is unanimous or one voice away from it. Subjects add a direction-blind premium rho·|2·mean(x)-1|^q with a HIGH exponent q (~5–11, person-specific), so a 5/5 or 0/5 panel earns almost the whole premium while a 4/5 or 1/5 panel earns almost none. The magnitude rho of this premium is substantial and heterogeneous (population mean ~0.39): an option unanimously rated 0 can be *preferred* to one carrying a single weak endorsement (sub-chance accuracy on one-endorsement-vs-nothing pairs), while options that are merely lopsided (4-of-5 endorsements) receive essentially no credibility bonus and therefore cannot dilute genuine weighted-evidence conflicts. The premium is a *unanimity* effect, not a mean-rating effect — that is the theory's sharpest falsifiable commitment (a strongly non-linear step at exactly 0 and n dissenters, not a smooth gradient in endorsement proportion).

(5) **Endorsement discount / pull.** Each positive endorsement carries a small credibility adjustment theta (faint-praise skepticism for most, mild endorsement-count pull for some), inert whenever the two options carry equal endorsement counts.

(6) **Population structure.** Everyone holds the same rule; heterogeneity is broad and continuous — in determinacy beta, in gamma, in lapse, in the modulator magnitudes rho, q, theta, phi and the readability coefficients. No discrete strategy switching.

Evidence for A: D = Σ_j w_eff_j (x_Aj - x_Bj) - theta·(Σx_A - Σx_B) + rho·(coh(x_A) - coh(x_B)); p(A) = (1-eps)·logistic(beta·D) + eps/2.

**Parameters:**
- gamma: [1.05, 1.95]
- beta: [1.2, 13.0]
- eps: [0.0, 0.20]
- k0: [0.02, 0.16]
- k1: [0.55, 1.00]
- k2: [0.10, 0.60]
- phi: [0.02, 0.38]
- theta: [-0.03, 0.07]
- rho: [0.19, 0.59]
- q: [5.0, 11.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Readability-limited validity weighting with dissent aversion and
    # endorsement discount (RVW-DE+/U).
    #   w_j        proportional to logit(v_j) ** gamma
    #   w_eff      = (1-kappa)*w + kappa/n          (readability flattening)
    #   w_eff     *= (1 + phi_eff * reading-order gradient)
    #                with phi_eff = phi*(r0 + r1*kappa): the reading-order
    #                fallback is recruited only when binding fails.
    #   D          = w_eff.(a-b) - theta*(sum a - sum b) + rho*(coh(a)-coh(b))
    #   p(A)       = (1-eps)*logistic(beta*D) + eps/2
    # kappa is driven by how incongruent the column order is with the
    # stated validity order (direction-free: monotone displays are cheap,
    # scrambled displays are expensive).  coh(x) = |2*mean(x)-1|**q with a
    # HIGH person-specific exponent q: the consensus premium is a strict
    # unanimity effect, not a smooth function of endorsement proportion.
    # History is unused: no feedback is given, the rule is stationary.
    import numpy as np

    # ---------------- unpack the two rating vectors --------------------
    a = None
    b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            half = arr.shape[0] // 2
            a = arr[:half].astype(float)
            b = arr[half:half + half].astype(float)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            a = arr[0].astype(float)
            b = arr[1].astype(float)

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float).ravel()[:n]
    b = np.asarray(b, dtype=float).ravel()[:n]

    # ---------------- validities --------------------------------------
    val = parameters.get("validities", None)
    if val is None:
        v = np.linspace(0.9, 0.55, n)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---------------- parameters --------------------------------------
    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 7.0))
    eps = float(np.clip(float(parameters.get("eps", 0.09)), 0.0, 1.0))
    k0 = float(parameters.get("k0", 0.08))
    k1 = float(parameters.get("k1", 0.78))
    k2 = float(parameters.get("k2", 0.35))
    phi = float(parameters.get("phi", 0.20))
    theta = float(parameters.get("theta", 0.02))
    rho = float(parameters.get("rho", 0.39))
    q = float(parameters.get("q", 8.0))

    # readability-gating of the reading-order fallback (fixed structure)
    r0 = 0.35
    r1 = 1.15

    # ---------------- validity weights --------------------------------
    L = np.log(v / (1.0 - v))
    L = np.maximum(L, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.power(L, gamma)
    w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)
    s = float(np.sum(w))
    if (not np.isfinite(s)) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)
    else:
        w = w / s

    # ---------------- readability flattening --------------------------
    # Kendall tau between column index and stated validity (ties skipped).
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    tot = conc + disc
    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)
    incong = 1.0 - abs(tau)

    jmax = int(np.argmax(v))
    off_extreme = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0

    kappa = k0 + k1 * incong + k2 * off_extreme
    kappa = float(np.clip(kappa, 0.0, 1.0))

    w_eff = (1.0 - kappa) * w + kappa * (1.0 / float(n))

    # ------- reading-order (primacy) gradient, gated by readability ----
    if n > 1 and phi != 0.0:
        phi_eff = phi * (r0 + r1 * kappa)
        phi_eff = float(np.clip(phi_eff, 0.0, 0.45))
        idx = np.arange(n, dtype=float)
        grad = 1.0 + phi_eff * ((float(n - 1) - 2.0 * idx) / float(n - 1))
        grad = np.maximum(grad, 1e-6)
        w_eff = w_eff * grad
    ssum = float(np.sum(w_eff))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        w_eff = np.ones(n, dtype=float) / float(n)
    else:
        w_eff = w_eff / ssum

    # ---------------- evidence ----------------------------------------
    core = float(np.dot(w_eff, a - b))

    # endorsement discount: credibility adjustment per positive rating
    cost = -theta * (float(np.sum(a)) - float(np.sum(b)))

    # dissent aversion: strict-unanimity consensus premium (high exponent q)
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    ua = abs(2.0 * ma - 1.0)
    ub = abs(2.0 * mb - 1.0)
    coh_a = float(np.power(ua, q))
    coh_b = float(np.power(ub, q))
    cons = rho * (coh_a - coh_b)

    D = core + cost + cons

    # ---------------- logistic choice with lapse -----------------------
    z = np.array([beta * D, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * 0.5
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
    tot = p.sum()
    if (not np.isfinite(tot)) or tot <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / tot
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    df = data\n    if len(df) == 0:\n        return 0.0\n\n    def as_tup(x):\n        return tuple(int(v) for v in x)\n\n    A = df['option_a_ratings'].apply(as_tup).values\n    B = df['option_b_ratings'].apply(as_tup).values\n    R = df['response'].values\n\n    def frac_choose(target, other):\n        \"\"\"Fraction of trials with pair {target, other} on which the\n        subject picked `target`. Collapses across A/B display order.\"\"\"\n        num = 0\n        den = 0\n        for a, b, r in zip(A, B, R):\n            if a == target and b == other:\n                den += 1\n                if int(r) == 0:\n                    num += 1\n            elif a == other and b == target:\n                den += 1\n                if int(r) == 1:\n                    num += 1\n        if den == 0:\n            return None\n        return float(num) / float(den)\n\n    # ---- cells -------------------------------------------------------\n    # Family L (coalition ladder): does the lone top-validity (col2)\n    # endorsement keep winning as the opposing coalition grows?\n    L1 = frac_choose((0, 0, 1, 0, 0), (1, 0, 0, 0, 0))\n    L4 = frac_choose((0, 0, 1, 0, 0), (1, 1, 0, 1, 1))\n\n    # Family U (unanimity step, weighted evidence pinned to col3 only):\n    # winner-unanimous cell vs loser-unanimous cell.\n    U4 = frac_choose((1, 1, 1, 1, 1), (1, 1, 1, 0, 1))\n    U1 = frac_choose((0, 0, 0, 1, 0), (0, 0, 0, 0, 0))\n\n    # Family V (same step at a heavier column, col0).\n    V2 = frac_choose((1, 1, 1, 1, 1), (0, 1, 1, 1, 1))\n    V1 = frac_choose((1, 0, 0, 0, 0), (0, 0, 0, 0, 0))\n\n    contrasts = []\n    if L1 is not None and L4 is not None:\n        contrasts.append(L1 - L4)\n    if U4 is not None and U1 is not None:\n        contrasts.append(U4 - U1)\n    if V2 is not None and V1 is not None:\n        contrasts.append(V2 - V1)\n\n    if len(contrasts) == 0:\n        return 0.0\n    return float(np.mean(contrasts))\n",
  "rationale": "Every term is a WITHIN-subject contrast between two cells in which Take-The-Best makes the *same* one-reason decision at the *same* flat probability sigmoid(beta)(1-eps)+eps/2, so each contrast has expectation exactly 0 under TTB regardless of that subject's beta and lapse rate (only binomial noise remains, ~8 trials/cell). (i) L1-L4: TTB stops at the .93 cue in both, picking the col2 option identically whether the rival carries one weak endorsement or four; RVW's flattened weights let the growing coalition overturn the top cue, so the col2 option falls from ~.55 to ~.03. (ii) U4-U1: both cells differ on col3 ALONE, so TTB's discriminating cue and its winner are identical; RVW's strict-unanimity premium makes the all-zero panel beat a lone weak endorsement (~.15) while a 5/5 panel beats a 4/5 panel (~.97). (iii) V2-V1: same manipulation at col0, again a single flat TTB level but a ~.22 vs ~.98 split under RVW. Averaging the three contrasts cuts per-subject binomial noise while preserving a mean gap of roughly 0.7 (RVW) versus 0.0 (TTB); because the TTB side is pinned at zero with only sampling variance, and the RVW side is driven by shared, strongly-signed effects, the between-subject spread stays small relative to the gap."
}
```

## Usage

```json
{
  "input_tokens": 11536,
  "output_tokens": 5096
}
```
