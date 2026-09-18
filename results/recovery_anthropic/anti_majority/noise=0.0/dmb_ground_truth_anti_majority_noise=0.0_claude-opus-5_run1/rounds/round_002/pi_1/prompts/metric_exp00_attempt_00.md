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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 7: A=[1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1]
  trial 9: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 10: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  trial 11: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 14: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate Take-The-Best (advocated) from RVW-DE+/U (competing). The competitor's sharpest falsifiable commitment is the STRICT-UNANIMITY consensus premium: rho*|2*mean(x)-1|^q with high q (5-11), so a panel that is unanimous (0/5 or 5/5) collects almost the whole premium (rho ~ .19-.59, mean .39) while a merely lopsided 4/5 or 1/5 panel collects essentially none. TTB has no such term: on every discriminating trial it sides with the highest-validity discriminating expert at one flat level sigmoid(beta) (plus lapse), regardless of endorsement counts, coherence, or lower cues.

DISPLAY: five experts shown in strictly DESCENDING validity (.95,.85,.75,.65,.55). This is deliberately the competitor's *best case*: Kendall tau = -1 so incongruence = 0, the best expert sits in an extreme column, hence kappa = k0 (.02-.16) and the reading-order fallback phi_eff = phi*(0.35+1.15*k0) is nearly inert. Readability nuisance is therefore removed and the competitor's predictions are driven almost entirely by its graded validity weights and by the unanimity premium. Normalised weights across its gamma box (1.05-1.95) are approximately [.46-.64, .23-.26, .09-.16, .03-.09, .003-.03], i.e. cue1 >> cue2 > cue3 > cue4 > cue5 for every subject.

16 unique pairs (K=6, 96 trials), every pair mirrored so neither response key is systematically correct. TTB predicts A on pairs 1,3,5,7,9,11,13,15 and B on their mirrors, always at the SAME probability.

FAMILY S - SINGLE ENDORSEMENT vs AN EMPTY (UNANIMOUS-ZERO) PANEL (pairs 1-6). One option has exactly one '1' (on cue1, cue3 or cue5) and the other is all zeros. TTB: the endorsed option wins in all three cases at an identical level (~.99 at moderate beta). Competitor: D(endorsed) = w_j - rho*(1 - 0.6^q) ~ w_j - .38, so with beta=7, gamma=1.5, rho=.39, q=8 it predicts p(endorsed) ~ .70 (cue1), .13 (cue3), .06 (cue5) - a steep monotone gradient that CROSSES CHANCE between cue1 and cue3, i.e. explicit sub-chance accuracy when only a weak expert speaks against an all-zero panel. Flat-at-ceiling versus sign-reversing-gradient is the core directional dissociation.

FAMILY P - UNANIMITY STEP WITH WEIGHTED EVIDENCE HELD EXACTLY CONSTANT (pair 5/6 reused as the baseline, plus pairs 7-10). In all three cells the two options differ on cue5 and only cue5, so the weighted-evidence term is identical (w5 ~ .01-.03) and TTB must give the SAME probability in all three. Only the coherence of the panels changes: (i) [0,0,0,0,1] vs [0,0,0,0,0] - loser panel unanimous, full premium; (ii) [1,1,0,0,1] vs [1,1,0,0,0] - both panels middling (2/5 vs 3/5), premium ~ 0; (iii) [1,1,1,1,1] vs [1,1,1,1,0] - winner panel unanimous, full premium in the same direction as the cue. Competitor predicts p(cue5-winner) ~ .06 / .51 / .94 respectively (a non-monotone step at exactly 0 and n dissenters), TTB predicts ~.99 / .99 / .99. This triple is the sharpest quantitative signature in the design and cannot be mimicked by any re-weighting of cues.

FAMILY D - COMPENSATION / GAMMA IDENTIFICATION (pairs 11-12): [1,0,0,0,0] vs [0,1,1,1,1]. Coherence cancels exactly (both panels 1/5 vs 4/5, |2m-1| identical), so this is a pure weighted-integration test. TTB stops at cue1 and is at ceiling for the top-cue option; the competitor is near-cancelling and its sign flips inside its own gamma box (core = +.05 at gamma=1.5, -.08 at gamma=1.05, +.15 at gamma=1.95), so it must predict the LARGEST between-subject spread of any family here and an overall level close to chance.

FAMILY E - DEEP CASCADE vs COALITION (pairs 13-14): [1,1,0,0,0] vs [1,0,1,1,1]. Cue1 ties, cue2 (.85) favours one option, cues 3-5 favour the other. TTB stops at cue2 and is at ceiling; the competitor computes core ~ .02 and lands near chance (p ~ .59). Confirms the cascade (not merely the single best cue) drives TTB, while remaining maximally diagnostic.

FAMILY F - DOMINANCE ANCHOR (pairs 15-16): [1,1,1,1,1] vs [0,0,0,0,0]. Both panels unanimous so the premium cancels exactly; both models are at ceiling. This pins beta and epsilon for BOTH models, so every other family is an out-of-sample prediction rather than a free fit.

DECISION RULE. If TTB is right: a single flat probability level (calibrated by Family F) accounts for Families S, P, D and E alike; accuracy on the single-endorsement pairs is high and IDENTICAL for cue1, cue3 and cue5; the three cue5-only cells (S5, P2, P3) are indistinguishable; and no coherence/unanimity effect appears anywhere. If RVW-DE+/U is right: accuracy on Family S falls steeply with cue validity and goes SUB-CHANCE for cue3 and cue5 (the all-zero panel is preferred to a weakly endorsed one), the cue5-only ladder shows a large non-monotone step (~.06 / ~.5 / ~.94) driven purely by unanimity, Family D shows near-chance means with maximal individual spread, and Family E hovers near chance.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    def key(x):\n        return tuple(int(v) for v in list(x))\n\n    A = [key(x) for x in data['option_a_ratings']]\n    B = [key(x) for x in data['option_b_ratings']]\n    R = [int(r) for r in data['response']]\n\n    ones5 = (1, 1, 1, 1, 1)\n    four = (1, 1, 1, 1, 0)\n    zeros = (0, 0, 0, 0, 0)\n    weak_single = [(0, 0, 1, 0, 0), (0, 0, 0, 0, 1)]\n\n    hi_n = 0\n    hi_d = 0\n    lo_n = 0\n    lo_d = 0\n\n    for a, b, r in zip(A, B, R):\n        if len(a) != 5 or len(b) != 5:\n            continue\n        # HIGH-coherence cell: 5/5 unanimous panel vs 4/5 panel, differ on cue5 only.\n        if a == ones5 and b == four:\n            hi_d += 1\n            if r == 0:\n                hi_n += 1\n        elif b == ones5 and a == four:\n            hi_d += 1\n            if r == 1:\n                hi_n += 1\n        # LOW-coherence cell: single weak endorsement vs unanimous all-zero panel.\n        elif (a in weak_single) and b == zeros:\n            lo_d += 1\n            if r == 0:\n                lo_n += 1\n        elif (b in weak_single) and a == zeros:\n            lo_d += 1\n            if r == 1:\n                lo_n += 1\n\n    if hi_d == 0 or lo_d == 0:\n        return float('nan')\n\n    p_hi = hi_n / float(hi_d)\n    p_lo = lo_n / float(lo_d)\n    return float(p_hi - p_lo)\n",
  "rationale": "Within-subject contrast designed to cancel every free noise parameter of Take-The-Best while maximally loading the competitor's strict-unanimity consensus premium.\n\nCell HIGH = pairs [1,1,1,1,1] vs [1,1,1,1,0] (and mirror): the two options differ on the LOWEST-validity cue only, and the winner's panel is unanimous. Cell LOW = pairs where one option carries a single weak endorsement (cue3 or cue5) and the other is the all-zero (also unanimous) panel; here the weakly endorsed option is the TTB winner but its panel is maximally incoherent while its rival is unanimous.\n\nTTB: in both cells the first discriminating cue favours the target option, so the choice probability is exactly (1-eps)*sigmoid(beta)+eps/2 in BOTH cells, whatever beta and eps are. The contrast is therefore identically 0 in expectation for every subject, regardless of that subject's determinacy or lapse rate \u2014 the metric is immune to TTB's (very wide) parameter heterogeneity, leaving only small binomial sampling noise (12 + 24 trials per subject).\n\nRVW-DE+/U: the weighted-evidence term is essentially the same tiny w5 (or w3) in both cells, but the direction-blind premium rho*|2m-1|^q with high q flips sign between cells: in HIGH the premium (~+rho) adds to the cue, in LOW it (~-rho) opposes the single endorsement, driving predicted accuracy from ~.94 down to ~.06-.13. Its predicted contrast is therefore large and positive (~.7-.9), with only moderate spread from rho/q/beta heterogeneity.\n\nSo: TTB predicts ~0, the competitor predicts a large positive step, and neither prediction is trivially guaranteed by the metric's construction (both cells are ordinary accuracy proportions on distinct trial types)."
}
```

## Usage

```json
{
  "input_tokens": 11185,
  "output_tokens": 3541
}
```
