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
**Validities (n_features=8):** [0.75, 0.93, 0.55, 0.68, 0.88, 0.6, 0.75, 0.68]

**Trial pairs (n=18):**
  trial 1: A=[0, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  trial 2: A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  trial 11: A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  trial 12: A=[0, 0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  trial 15: A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  trial 17: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate RD-WEI (compressed near-tally weights gamma~0.33, reversed polarity s=-1, divisive normalisation with kappa~0.52, beta~2.1, small lapse) from R-SED (same reversed-polarity family but kappa~0.70, beta~3.6, large lapse ~0.20, and crucially a SPARSITY-GATED SEVERITY BOOST that multiplies the weight of the single most valid DISCRIMINATING cue by 1+phi*exp(-lam*(n_disc-2)) whenever n_disc>=2). Both are reversed-frame accounts, so direction cannot separate them; the design targets (i) the level of 1-flag-vs-1-flag DUEL resolution, (ii) the n_disc LADDER at matched weighted margin, and (iii) magnitude anchors that pin beta and epsilon so no global-noise reparameterisation can mimic the shape.

ENVIRONMENT: eight experts in scrambled display order [0.75, 0.93, 0.55, 0.68, 0.88, 0.60, 0.75, 0.68] (idx0..idx7), so no 'first-listed expert' rule can imitate either model. Two exact DUPLICATE pairs are built in (idx0/idx6 both 0.75; idx3/idx7 both 0.68) so that cancelling flag pairs add EXACTLY zero to the signed weighted margin while inflating the discriminating mass S. Log-odds d = (1.099, 2.587, 0.201, 0.754, 1.992, 0.406, 1.099, 0.754). At RD-WEI's gamma=0.33 the weights are w = (1.032, 1.368, 0.589, 0.911, 1.255, 0.742, 1.032, 0.911) (only ~2.3:1 spread); R-SED's gamma=0.34 gives virtually the same base profile, so ALL differences below come from the severity gate, kappa and the beta/epsilon regime, not from weight ordering.

NINE TYPES, each mirrored across A/B so no side bias can imitate either model (18 unique pairs, K=5, 90 trials). Predicted p(A) at representative parameters (RD-WEI: gamma=.33, kappa=.52, beta=2.1, eps=.06, s=-1 | R-SED: gamma=.34, kappa=.70, phi=.5, lam=1.0, beta=3.6, eps=.20, s=-1):
(T1, pairs 1-2) DUEL HIGH, small gap: 0.93 flags A, 0.88 flags B (n_disc=2). RD-WEI .466 (compressed weights 1.368 vs 1.255 differ by almost nothing, so after normalisation this is essentially a coin flip) | R-SED .278 (the severity gate amplifies the 0.93 flag by 1.5 and condemns A). PRIMARY DISCRIMINATOR.
(T2, pairs 3-4) DUEL MID, small gap: 0.75 flags A, 0.68 flags B. RD-WEI .457 | R-SED .281.
(T3, pairs 5-6) DUEL LOW, small gap: 0.60 flags A, 0.55 flags B. RD-WEI .435 | R-SED .270. T1-T3 test R-SED's headline claim that duel resolution is roughly constant (~.27-.28) and independent of validity LEVEL, while RD-WEI must sit near chance (.43-.47) throughout. Parameter sweeps over the stated ranges keep the two sets disjoint (RD-WEI .39-.48 vs R-SED .20-.36).
(T4, pairs 7-8) DUEL MAX GAP: 0.93 flags A, 0.55 flags B. RD-WEI .265 | R-SED .150. Contrasted with T1-T3 this gives the GAP-gradedness test: RD-WEI's deviation from chance grows ~7-fold from small-gap to max-gap duels, R-SED's only ~1.6-fold.
(T5, pairs 9-10) MATCHED-MARGIN LADDER, n_disc=4: T1's duel plus an exactly cancelling 0.75/0.75 flag pair (idx0 on A, idx6 on B). Signed margin identical to T1; only S grows and R-SED's boost decays to 1.068. RD-WEI .476 | R-SED .450.
(T6, pairs 11-12) MATCHED-MARGIN LADDER, n_disc=6: T1's duel plus cancelling 0.75/0.75 and 0.68/0.68 pairs. RD-WEI .479 | R-SED .481. THE LADDER T1->T5->T6 is the decisive shape test: at a CONSTANT weighted margin RD-WEI predicts an essentially FLAT profile (.466, .476, .479; deviation ratio ~1.4 across the ladder even at extreme parameters) whereas R-SED predicts a STEEP collapse (.278, .450, .481; deviation ratio ~12) as the severity gate closes. No reparameterisation of either model reproduces the other's ladder.
(T7, pairs 13-14) SINGLE CUE 0.93 (n_disc=1, two tied 1s present so the display is non-degenerate). RD-WEI .095 | R-SED .115. Because R-SED's boost is switched OFF at n_disc=1 and ON at n_disc=2, the T7->T1 step is diagnostic: RD-WEI predicts a huge jump toward chance (+.37) when a 0.88 counter-flag is added, R-SED only +.16.
(T8, pairs 15-16) LOPSIDED 5-vs-1: five weak/mid flags (0.75,0.68,0.60,0.75,0.68) on A against a single 0.93 flag on B (n_disc=6, boost ~1.009, i.e. off). RD-WEI .089 | R-SED .228 - a pure magnitude test isolating the kappa/beta/epsilon regime once the severity gate is inert.
(T9, pairs 17-18) UNANIMOUS 8-0 DOMINANCE. RD-WEI .033 | R-SED .101. LAPSE ANCHOR: it pins epsilon for both models, so R-SED's near-chance ladder tail and RD-WEI's near-chance duels cannot be traded off against global noise.

DIAGNOSTIC SUMMARY of p(A) across T1..T9: RD-WEI = (.466, .457, .435, .265, .476, .479, .095, .089, .033); R-SED = (.278, .281, .270, .150, .450, .481, .115, .228, .101). The vectors agree in sign everywhere (both reversed-polarity), but differ in SHAPE in three independent, non-reparameterisable ways: (i) duels near chance vs strongly resolved; (ii) a flat vs steeply decaying n_disc ladder at matched margin; (iii) different sparse/dominance magnitude anchors that block a noise-only mimicry.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    V = np.array([0.75, 0.93, 0.55, 0.68, 0.88, 0.60, 0.75, 0.68], dtype=float)\n    duel = []\n    dom = []\n    for _, row in data.iterrows():\n        try:\n            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()\n            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()\n            resp = int(row['response'])\n        except Exception:\n            continue\n        if a.size == 0 or a.size != b.size:\n            continue\n        if a.size == V.size:\n            v = V\n        else:\n            v = np.linspace(0.90, 0.55, a.size)\n        d = a - b\n        disc = np.nonzero(np.abs(d) > 0.0)[0]\n        sa = float(a.sum())\n        sb = float(b.sum())\n        if disc.size == 2 and sa == sb:\n            vs = v[disc]\n            if abs(float(vs[0]) - float(vs[1])) <= 0.20:\n                top = int(disc[int(np.argmax(vs))])\n                # reference = the option NOT flagged by the more valid duelling cue\n                ref = 1 if d[top] > 0 else 0\n                duel.append(1.0 if resp == ref else 0.0)\n        elif abs(sa - sb) >= 4.0:\n            # reference = option carrying fewer marks\n            ref = 1 if sa > sb else 0\n            dom.append(1.0 if resp == ref else 0.0)\n    if len(duel) == 0 or len(dom) == 0:\n        return float('nan')\n    p_dom = float(np.mean(dom))\n    p_duel = float(np.mean(duel))\n    s = 1.0 if p_dom >= 0.5 else -1.0\n    return float(s * (p_dom - 0.5) - s * (p_duel - 0.5))",
  "rationale": "The two theories agree in polarity everywhere, so the discriminating signal must be a SHAPE contrast between trial classes. RD-WEI (divisive normalisation with small kappa~0.52) says a 1-flag-vs-1-flag duel between two cues of similar validity is essentially a coin flip (p(A)~.44-.47, deviation from chance ~.03-.07), because the compressed near-tally weights of the two duelling cues barely differ and normalisation removes any absolute-magnitude advantage. R-SED's sparsity-gated severity boost instead turns such duels into decisive 'which flag is more damning' judgements (p(A)~.27-.28, deviation ~.22) essentially independent of the validity gap. Conversely, on lopsided/dominance displays (5-vs-1 and 8-vs-0 marks) RD-WEI is near-ceiling (~.91-.97 for the fewer-marks option) while R-SED's larger kappa (~.70) plus large lapse (~.20) caps it around .77-.90.\n\nThe metric therefore computes: (deviation-from-chance on high-mark-difference dominance trials) minus (signed deviation on SMALL-VALIDITY-GAP duel trials, sign-aligned to the subject's own dominance-trial polarity). Predicted values: RD-WEI ~0.40, R-SED ~0.11.\n\nSign-aligning the duel term to the polarity revealed by the dominance trials makes the metric invariant to the rare forward-frame subject (s=+1), preventing outlier subjects from inflating between-subject variance in either simulation. Each component is estimated from 20-30 trials per subject, so per-subject SD is ~0.10 while the between-theory gap is ~0.29 -- comfortably separable at N=25. Restricting the duel class to matched-mark, n_disc=2 pairs with validity gap <= 0.20 excludes the max-gap duel (0.93 vs 0.55), where RD-WEI also deviates strongly from chance, thereby maximising the contrast."
}
```

## Usage

```json
{
  "input_tokens": 12827,
  "output_tokens": 6888
}
```
