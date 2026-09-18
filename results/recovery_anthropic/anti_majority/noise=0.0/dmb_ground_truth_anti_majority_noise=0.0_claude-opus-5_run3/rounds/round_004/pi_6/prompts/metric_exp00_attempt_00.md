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
**Validities (n_features=7):** [0.72, 0.95, 0.55, 0.85, 0.6, 0.72, 0.6]

**Trial pairs (n=18):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1]
  trial 2: A=[0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0]
  trial 10: A=[0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1]
  trial 14: A=[1, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 1]
  trial 15: A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1]
  trial 17: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate R-SED (advocated: reversed polarity, mildly compressed weights gamma~0.34, kappa~0.70, beta~3.6, lapse~0.20, plus the SPARSITY-GATED SEVERITY BOOST that multiplies the weight of the single most valid DISCRIMINATING cue by 1+phi*exp(-lam*(n_disc-2)) when n_disc>=2) from RD-WEI (reversed polarity, gamma~0.33, kappa~0.52, beta~2.1, small lapse, NO severity gate). Both are reversed-frame accounts, so overall direction cannot separate them; the design attacks three orthogonal SHAPE signatures: (i) the LEVEL of 1-flag-vs-1-flag duel resolution and its (in)dependence of the validity GAP, (ii) the n_disc ladder at an EXACTLY matched signed weighted margin, and (iii) sparse/dominance magnitude anchors that pin beta and epsilon so no noise-only reparameterisation can mimic the pattern.

ENVIRONMENT: seven experts in scrambled display order [0.72, 0.95, 0.55, 0.85, 0.60, 0.72, 0.60] (idx0..idx6), so no 'use the first-listed expert' rule can imitate either model. Two exact DUPLICATE pairs are built in (idx0/idx5 = 0.72; idx4/idx6 = 0.60) so that cancelling flag pairs add EXACTLY zero to the signed weighted margin while inflating the discriminating mass S. Log-odds d = (0.944, 2.944, 0.201, 1.735, 0.406, 0.944, 0.406). At the shared compression exponent the base weights are w ~ (0.98, 1.43, 0.59, 1.20, 0.74, 0.98, 0.74) - only a ~2.4:1 spread - so ALL divergences below come from the severity gate and the kappa/beta/epsilon regime, never from weight ordering.

NINE TYPES, each mirrored across A/B so no side bias can imitate either model (18 unique pairs, K=5, 90 trials). Predicted p(A) at representative parameters (R-SED: gamma=.34, kappa=.70, phi=.5, lam=1.0, beta=3.6, eps=.20, s=-1 | RD-WEI: gamma=.33, kappa=.52, beta=2.1, eps=.06, s=-1):
(T1, pairs 1-2) DUEL HIGH, small gap: 0.95 flags A, 0.85 flags B (n_disc=2). R-SED .25 (the severity gate multiplies the 0.95 flag by 1.5 and condemns A) | RD-WEI .44 (compressed weights 1.43 vs 1.20 differ by almost nothing, so after normalisation this is nearly a coin flip). PRIMARY DISCRIMINATOR.
(T2, pairs 3-4) DUEL MID, small gap: 0.85 flags A, 0.72 flags B. R-SED .25 | RD-WEI .43.
(T3, pairs 5-6) DUEL LOW, small gap: 0.60 flags A, 0.55 flags B. R-SED .27 | RD-WEI .44. T1-T3 test R-SED's headline claim that duel resolution is roughly CONSTANT and independent of validity LEVEL, while RD-WEI must hover near chance throughout.
(T4, pairs 7-8) DUEL MAX GAP: 0.95 flags A, 0.55 flags B. Both models ~.245 - they AGREE here. The T1/T4 contrast is the gap test: RD-WEI's deviation from chance must grow ~4-fold from small-gap to max-gap duels, while R-SED's stays flat (.25 -> .245). Neither model can reproduce the other's duel profile at any admissible parameters.
(T5, pairs 9-10) MATCHED-MARGIN LADDER, n_disc=4: T1's duel plus an exactly cancelling 0.72/0.72 flag pair (idx0 on A, idx5 on B). Signed margin identical to T1; only S grows and R-SED's boost decays to 1.068. R-SED .42 | RD-WEI .45.
(T6, pairs 11-12) MATCHED-MARGIN LADDER, n_disc=6: T1's duel plus cancelling 0.72/0.72 and 0.60/0.60 pairs. R-SED .45 | RD-WEI .46. THE LADDER T1->T5->T6 is the decisive shape test: at a CONSTANT weighted margin R-SED predicts a steep collapse toward chance (.25, .42, .45; deviation shrinks ~5-fold as the severity gate closes) whereas RD-WEI predicts an essentially FLAT profile (.44, .45, .46).
(T7, pairs 13-14) SINGLE CUE 0.95, n_disc=1 (a tied 1 at idx0 and idx6 keeps the display non-degenerate). R-SED .11 | RD-WEI .10 - the models nearly coincide. Because R-SED's boost is OFF at n_disc=1 and ON at n_disc=2, the T7->T1 STEP is diagnostic: RD-WEI predicts a large jump toward chance when a 0.85 counter-flag is added (+.34), R-SED only +.14.
(T8, pairs 15-16) LOPSIDED 5-vs-1: five weak/mid flags (0.72,0.72,0.60,0.60,0.55) on A against a single 0.95 flag on B (n_disc=6, boost ~1.01, i.e. inert). R-SED .245 | RD-WEI .13 - a pure magnitude test isolating the kappa/beta/epsilon regime once the severity gate is switched off, and a check that one flag from the best expert loses to four/five weak flags.
(T9, pairs 17-18) UNANIMOUS 7-0 DOMINANCE. R-SED .10 | RD-WEI .033. LAPSE ANCHOR: it pins epsilon for both models, so R-SED's near-chance ladder tail and RD-WEI's near-chance duels cannot be traded off against global noise.

DIAGNOSTIC SUMMARY of p(A) across T1..T9: R-SED = (.25, .25, .27, .245, .42, .45, .11, .245, .10); RD-WEI = (.44, .43, .44, .245, .45, .46, .10, .13, .033). The vectors agree in sign everywhere (both reversed-polarity) but differ in SHAPE in three independent, non-reparameterisable ways: (i) small-gap duels strongly resolved vs near chance, with the two accounts converging exactly at the max-gap duel; (ii) a steeply decaying vs flat n_disc ladder at an exactly matched signed margin; (iii) opposite orderings of the sparse (T7), lopsided (T8) and dominance (T9) anchors, which fix beta and epsilon and block any noise-only mimicry.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** **R-SED: Reversed Evidence with Sparsity-Gated Severity Dominance.**

Four claims.

(1) POLARITY (retained; the one robustly confirmed claim). In these instruction-only, feedback-free product-rating environments a binary '1' is read as a WARNING MARK / flag against an option rather than as an endorsement. The decision variable is therefore the *negated* weighted mark load: the option carrying the SMALLER validity-weighted number of 1s is preferred. A small minority of subjects keep the forward frame (subject-level latent s in {-1,+1}, P(s=-1)=p_reverse ~0.94-1.0), which is what generates the observed between-subject spread on dominance/unanimity pairs.

(2) NO CONSULTATION BOTTLENECK. Every displayed rating is registered; there is no per-expert reading probability and no capacity that shrinks with panel size. A lone flag on a 6-expert panel is read exactly as reliably as on a 4-expert panel. This kills the falsified panel-size dilution of RFS-C: sparse non-conflicting displays land at ~0.86 fewer-marks choices regardless of n_features.

(3) EVIDENCE IS RELATIVE, WITH MILDLY COMPRESSED BASE WEIGHTS. Stated validities become log-odds d_i = log(v_i/(1-v_i)), compressed to w_i = d_i^gamma with gamma ~0.3 so that many-cue conflicts stay essentially count-driven (no single expert can lexicographically beat four others). The comparison is evaluated in *relative* units: the signed weighted margin is divided by the total weight of the cues actually in play raised to kappa ~0.7 (Weber-like divisive normalisation). Because kappa is close to (but below) 1, decisiveness saturates: 1-cue, 2-cue-same-side and 7-cue unanimous displays all sit at roughly the same ~0.86 reversed-choice level rather than becoming deterministic as the panel grows, while lopsided many-cue conflicts remain only slightly less decisive (~0.82).

(4) NEW THEORETICAL CONTENT — SPARSITY-GATED SEVERITY DOMINANCE. When only a couple of experts disagree, the comparison stops being a tally and becomes a *duel between named experts*: the reader asks which flag is the more damning, and the single most valid DISCRIMINATING cue acquires an amplified effective weight,

    w_eff(top) = w(top) * (1 + phi * exp(-lam * (n_disc - 2))),  applied only when n_disc >= 2,

so the more credible flag nearly dominates the comparison. Crucially the amplification enters BOTH the margin and the normaliser, so duels resolve at a roughly constant ~0.72-0.78 rejection of the option flagged by the more credible expert, essentially INDEPENDENT of the validity gap and of the validity level (0.95-vs-0.90 behaves like 0.60-vs-0.55). As the difference set grows the identities of the flagging experts are lost, the boost decays geometrically (already ~1.07 at n_disc = 4, ~1.01 at n_disc = 6) and behaviour reverts to a near-tally weighted count of marks, which is why one flag from the best expert loses to four flags from weak experts and why top-heavy near-even splits (2-vs-3, 2-vs-2) sit near chance.

The family nests the classics: s=+1, gamma->0, kappa=0, phi=0 = Tallying; s=+1, gamma=1, kappa=0, phi=0 = Franklin's rule; phi->inf with lam->0 = (reversed) Take-The-Best; kappa=1 = pure share-of-evidence. Its sharp, falsifiable signature relative to RD-WEI is the n_disc ladder at matched weighted margin: R-SED predicts a steep drop in duel resolution as n_disc goes 2 -> 3 -> 4 (severity gate closing), whereas RD-WEI predicts smooth share-graded behaviour and near-chance duels throughout.

**Parameters:**
- gamma: [0.28, 0.40]
- kappa: [0.64, 0.76]
- beta: [3.2, 4.0]
- phi: [0.38, 0.60]
- lam: [0.8, 1.3]
- epsilon: [0.15, 0.25]
- p_reverse: [0.94, 1.0]
- polarity_u: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # R-SED: Reversed Evidence with Sparsity-Gated Severity Dominance.
    #   d_i = log(v_i/(1-v_i));  w_i = d_i ** gamma          (mild compression)
    #   if n_disc >= 2: w_eff[top discriminating cue] *= 1 + phi*exp(-lam*(n_disc-2))
    #   E  = sum_i w_eff_i (a_i - b_i)      (positive = A carries more marks)
    #   S  = sum_{i in disc} w_eff_i
    #   E' = E / S**kappa                    (relative / divisive normalisation)
    #   s  = -1 with prob p_reverse (mark-reading frame), else +1
    #   P(A) = sigmoid(s * beta * E'), then lapse epsilon toward uniform.
    # No consultation bottleneck: every rating is registered. History unused
    # (there is no feedback in this domain).
    import numpy as np

    # ---------------- unpack stimulus ----------------
    stim = None
    if isinstance(state, dict):
        a_raw = state.get("option_a_ratings", None)
        b_raw = state.get("option_b_ratings", None)
        if a_raw is not None and b_raw is not None:
            try:
                a_v = np.asarray(list(a_raw), dtype=float).ravel()
                b_v = np.asarray(list(b_raw), dtype=float).ravel()
                if a_v.size == b_v.size and a_v.size > 0:
                    stim = np.vstack([a_v, b_v])
            except Exception:
                stim = None
    if stim is None:
        try:
            stim = np.asarray(state, dtype=float)
        except Exception:
            return np.ones(2) / 2.0
        if stim.ndim == 1:
            if stim.shape[0] % 2 != 0:
                return np.ones(2) / 2.0
            h = stim.shape[0] // 2
            stim = np.vstack([stim[:h], stim[h:]])
    if stim.ndim != 2 or stim.shape[0] != 2 or stim.shape[1] == 0:
        return np.ones(2) / 2.0

    n_features = int(stim.shape[1])

    # ---------------- validities ----------------
    try:
        v = np.asarray(parameters.get("validities", None), dtype=float).ravel()
    except Exception:
        v = np.array([])
    if v.size != n_features or not np.all(np.isfinite(v)):
        v = np.linspace(0.90, 0.55, n_features)
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    gamma = float(parameters["gamma"])
    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    phi = float(parameters["phi"])
    lam = float(parameters["lam"])
    eps = float(parameters["epsilon"])
    p_rev = float(parameters["p_reverse"])
    u = float(parameters["polarity_u"])

    # ---------------- compressed diagnosticity weights ----------------
    d = np.maximum(np.log(v / (1.0 - v)), 1e-12)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        w = np.power(d, gamma)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)

    diff = stim[0] - stim[1]
    disc = np.nonzero(np.abs(diff) > 0.0)[0]
    n_disc = int(disc.size)
    if n_disc == 0:
        return np.ones(2) / 2.0

    # ---------------- sparsity-gated severity amplification ----------------
    w_eff = w.astype(float).copy()
    if n_disc >= 2:
        # most valid DISCRIMINATING cue
        top = int(disc[int(np.argmax(v[disc]))])
        expo = -lam * float(n_disc - 2)
        expo = float(np.clip(expo, -50.0, 50.0))
        amp = 1.0 + phi * float(np.exp(expo))
        if not np.isfinite(amp) or amp < 1.0:
            amp = 1.0
        w_eff[top] = w_eff[top] * amp

    # ---------------- relative (normalised) margin ----------------
    E = float(np.dot(w_eff, diff))          # >0 : A carries more weighted marks
    S = float(np.sum(w_eff[disc]))
    if not np.isfinite(E):
        E = 0.0
    if np.isfinite(S) and S > 1e-12:
        denom = S ** kappa
        if np.isfinite(denom) and denom > 1e-12:
            E = E / denom
    if not np.isfinite(E):
        E = 0.0

    # ---------------- subject-level polarity frame ----------------
    s = -1.0 if u < p_rev else 1.0

    z = np.array([s * beta * E, 0.0], dtype=float)
    z = np.clip(z, -500.0, 500.0)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    V = np.array([0.72, 0.95, 0.55, 0.85, 0.60, 0.72, 0.60], dtype=float)\n\n    def _subj(df):\n        dom = []   # 1 if subject picked the ALL-ZERO option on 7-0 dominance trials\n        duel = []  # 1 if subject picked the option NOT flagged by the more valid expert\n        for a_raw, b_raw, r in zip(df['option_a_ratings'], df['option_b_ratings'], df['response']):\n            try:\n                a = np.asarray(list(a_raw), dtype=float).ravel()\n                b = np.asarray(list(b_raw), dtype=float).ravel()\n            except Exception:\n                continue\n            if a.size != b.size or a.size != V.size:\n                continue\n            try:\n                resp = int(r)\n            except Exception:\n                continue\n            n = a.size\n            diff = a - b\n            disc = np.nonzero(diff != 0.0)[0]\n\n            # --- dominance / unanimity trials: frame anchor ---\n            if (a.sum() == n and b.sum() == 0.0) or (b.sum() == n and a.sum() == 0.0):\n                zero_side = 0 if a.sum() == 0.0 else 1\n                dom.append(1.0 if resp == zero_side else 0.0)\n                continue\n\n            # --- small-gap 1-vs-1 duels (exactly two discriminating cues, one per side) ---\n            if disc.size == 2:\n                i, j = int(disc[0]), int(disc[1])\n                if diff[i] * diff[j] < 0.0:\n                    vi, vj = V[i], V[j]\n                    if vi != vj and abs(vi - vj) <= 0.20:\n                        hi = i if vi > vj else j\n                        hi_side = 0 if diff[hi] > 0.0 else 1  # option flagged by more valid expert\n                        duel.append(1.0 if resp == (1 - hi_side) else 0.0)\n\n        if len(duel) == 0:\n            return np.nan\n        # subject-level polarity frame, read off the unanimous dominance pairs\n        frame = 1.0\n        if len(dom) > 0 and float(np.mean(dom)) < 0.5:\n            frame = -1.0\n        rate = float(np.mean(duel))\n        return rate if frame > 0.0 else 1.0 - rate\n\n    vals = []\n    for _, g in data.groupby('subject_id', sort=False):\n        v = _subj(g)\n        if v == v:\n            vals.append(v)\n    if len(vals) == 0:\n        return 0.5\n    return float(np.mean(vals))",
  "rationale": "The two accounts are both reversed-polarity, so any overall direction measure is useless. Their sharpest non-reparameterisable divergence is the LEVEL at which a 1-flag-vs-1-flag duel with a SMALL validity gap is resolved. R-SED's sparsity-gated severity boost amplifies the single most valid discriminating cue when n_disc=2, so the option flagged by the more credible expert is rejected at roughly 0.72-0.78 regardless of how small the validity gap is (trials 1-6: .95 vs .85, .85 vs .72, .60 vs .55 all resolve alike, ~0.75). RD-WEI has no such gate: after its compressive gamma transform the two competing weights are nearly equal (e.g. 1.43 vs 1.20), and divisive normalisation cannot rescue a near-zero share, so it must hover near chance (~0.56). Predicted gap ~0.19 in a bounded rate estimated from 30 trials per subject (binomial sd ~0.08), which is tight enough for N=25.\n\nTwo design choices keep per-subject variance small. (1) Max-gap duels (trials 7-8, .95 vs .55), where the two theories AGREE (~0.755 both), are excluded by the |v_hi - v_lo| <= 0.20 filter, so only the discriminating small-gap duels contribute. (2) The dominant nuisance source of between-subject spread in both models is the latent polarity frame s (a few percent of subjects run the forward frame and mirror every prediction). I read each subject's frame off the unanimous 7-0 dominance trials \\u2014 which BOTH theories predict at >0.90/0.96 accuracy, so frame recovery from 10 trials is essentially error-free \\u2014 and align the duel rate to that frame. This removes the bimodality without abs()-style bias, leaving mostly binomial noise. The metric is identically defined on the pooled frame (mean over subject-level rates) and on any single-subject slice."
}
```

## Usage

```json
{
  "input_tokens": 12742,
  "output_tokens": 6106
}
```
