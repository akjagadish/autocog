# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
**RD-WEI: Relative (Normalized) Directional Weighted Evidence Integration.**

This keeps the two claims that carried the previous candidate and adds a third, which is the theoretical content of this revision.

(1) COMPRESSED, COMPENSATORY INTEGRATION. Subjects integrate ALL expert ratings. Each stated validity v_i is turned into normative evidence d_i = log(v_i/(1-v_i)), which enters the comparison after a strongly *compressive* transform w_i = d_i^gamma with gamma well below 1 (~0.2-0.5). The weight profile is therefore only mildly validity-ordered - much closer to a tally than to log-odds weighting - so no single expert can lexicographically out-weigh three or four others, yet validity still decides 1-vs-1 comparisons.

(2) SUBJECT-LEVEL POLARITY FRAME. The direction in which a positive binary rating pushes preference is a latent subject-level frame s in {-1,+1}: the decision variable is s * E. When the frame is inverted (s = -1, the overwhelming majority in these environments) a '1' is read as a mark/flag/non-selective endorsement and the subject prefers the option carrying the SMALLER validity-weighted count of positive ratings. Because in these designs the option with fewer marks is usually also the option favoured by the single most valid discriminating cue, reversed integration *masquerades* as one-reason (Take-The-Best) decision making on cue-conflict trials while producing strongly NEGATIVE weighted-margin contrasts on dominance/unanimity pairs - a signature no forward integrator and no lexicographic rule can produce.

(3) NEW: EVIDENCE IS READ IN RELATIVE, NOT ABSOLUTE, UNITS (divisive normalization). The margin is not evaluated in absolute log-odds units; it is evaluated as a *share of the evidence actually in play* on that trial. Only cues that discriminate enter the comparison, and the margin is divided by the total weight of those discriminating cues raised to a power kappa (~0.4-0.6):

    E' = s * [ sum_i w_i (a_i - b_i) ] / ( sum_{i: a_i != b_i} w_i )^kappa ,   P(A) = sigmoid(beta * E'), then lapse.

Psychologically this is a Weber-like normalization of the comparison: what governs confidence is how *lopsided* the discriminating evidence is, not how much of it there is. Two consequences distinguish RD-WEI from the un-normalized version: (a) a one-cue-versus-one-cue disagreement (few cues in play) is resolved almost as decisively as a 1-versus-5 rout, so single-cue and small-set discriminations stay sharp; (b) trials in which many cues discriminate but the split is nearly even (e.g. 4 marks vs 6 marks, or 2 vs 3 across a 7-cue array) become markedly noisier than their absolute margin would imply, and even unanimous/dominance pairs stop being perfectly deterministic once every cue is in play. This is exactly the graded-ness pattern the data show: near-ceiling reversed responding on lopsided splits, sub-ceiling responding on many-cue near-even splits, and sizeable between-subject disagreement precisely on the near-zero-share pairs.

The family still nests the classics: s=+1, gamma->0, kappa=0 = Tallying; s=+1, gamma=1, kappa=0 = Franklin's rule / naive Bayes; s=+1, gamma->inf = TTB; kappa=1 = pure average-diagnosticity-per-discriminating-cue; s=-1 gives the reversed-frame regime these environments demand.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## THEORY 2 — pi_3
**Absolute-Evidence Weighted Integration (log-odds^gamma diagnosticity weights, no divisive normalization).**

People integrate ALL stated expert ratings, but with diagnosticity weights derived from the *magnitude* of each expert's stated validity: d_i = log(v_i/(1-v_i)) (the normative evidence a binary cue carries), passed through a non-compensatoriness exponent gamma, w_i = d_i^gamma. Evidence for option A is E = sum_i w_i (a_i - b_i), converted to choice by a softmax with inverse temperature beta plus a lapse epsilon. The family nests both classical heuristics as limits: gamma -> 0 gives equal weights (Tallying); gamma = 1 gives naive-Bayes / Franklin's rule; gamma -> infinity collapses weight onto the single most valid discriminating expert (Take-The-Best).

The crucial claim is that evidence is accumulated in ABSOLUTE log-odds-derived units: the weight vector is NOT renormalised to sum to one, and decision noise (beta) is constant in those absolute units. Consequently the *total diagnosticity* of an environment, D = sum_i d_i^gamma, governs how decisively choices are made. Three consequences follow that no rank-based rule and no scale-invariant (normalised) weighted-additive rule can produce:

(1) Environment-dependent non-compensatoriness. A structurally identical conflict - top cue versus 4-5 opposing lower cues - is resolved MORE top-cue-consistently in an environment containing a dominant, highly valid expert (e.g. .93 alongside .80) than in a smoothly graded environment topping out at .90, because the former yields larger absolute margins. Humans show exactly this asymmetry (~81% top-cue agreement in the 6-cue .93 environment vs ~74% in the graded 5-cue one).

(2) Graded, sub-ceiling conflict agreement. In the graded environment the absolute margin for 'top cue vs the four lower cues' is small, so agreement is only ~0.62-0.69, while 'top two cues vs the bottom three' is near-deterministic (~1.0) - a within-family gradient in the weighted margin that TTB forbids and that produces sub-ceiling pooled agreement without a large lapse rate.

(3) Environment-dependent heterogeneity. Because the graded environment's pivotal margins sit near zero, individual differences in gamma translate into large between-subject differences in conflict agreement there, whereas the dominant-cue environment keeps every subject on the top-cue side - predicting larger between-subject variance in the graded environment specifically.

It further predicts near-deterministic choice on unanimous (all-cue) pairs, and near-chance responding on pairs decided only by a .55-validity expert, since those margins are small in absolute units.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.7, 0.92, 0.55, 0.84, 0.62, 0.76]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 6: A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  trial 7: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 9: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 10: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 11: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 12: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 16: A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]

**Rationale:** GOAL: dissociate RD-WEI (compressed near-tally weights, subject-level REVERSED polarity frame s=-1, and divisive normalization of the margin by the weight of the discriminating cues) from AEWI (absolute, un-normalised log-odds^gamma weights with gamma~1.4, forward polarity). Three orthogonal handles are built into the design, so the two models are separated by SIGN, by the SHAPE of the confidence gradient, and by WHERE choices are noisy \u2014 not by overall accuracy alone.

ENVIRONMENT: six experts, displayed in scrambled (non-descending) order [0.70, 0.92, 0.55, 0.84, 0.62, 0.76], so no 'first-listed expert' rule can imitate any model. Log-odds diagnosticities d = (0.847, 2.442, 0.201, 1.658, 0.489, 1.153). At AEWI's own gamma (~1.41) the weights are w = (0.79, 3.49, 0.11, 2.03, 0.37, 1.22): a 33:1 ratio between the best and worst expert, i.e. strongly non-compensatory. At RD-WEI's gamma (0.18-0.48) the same environment yields w = (0.92-0.97, 1.17-1.54, 0.46-0.75, 1.10-1.28, 0.71-0.88, 1.03-1.07): only a ~2:1 spread, i.e. essentially a validity-tinted tally. Every trial family below is tuned to exploit that 33:1 vs 2:1 gap.

TRIAL FAMILIES (8 types, each mirrored across A/B so no side bias can mimic either model; 16 unique pairs, K=6, 96 trials). Predicted p(A) at representative parameters (RD-WEI: gamma=0.33, kappa=0.52, beta=2.1, s=-1; AEWI: gamma=1.41, beta=2.0):

(T1, pairs 1-2) ONLY the 0.55 expert discriminates. AEWI ~0.55 (near chance, tiny absolute margin). RD-WEI ~0.16 (confident REVERSED choice, because after divisive normalization a 1-vs-0 discrimination is E' = w^(1-kappa), which is large even for the worst cue).
(T2, pairs 3-4) ONLY the 0.92 expert discriminates. AEWI ~0.999. RD-WEI ~0.08. The T1-vs-T2 contrast is the cleanest quantitative test: AEWI must show a huge single-cue confidence gradient (0.55 -> 0.999) driven by absolute cue diagnosticity, whereas RD-WEI's normalization plus weight compression predicts an almost FLAT (and reversed) profile (0.16 -> 0.08).
(T3, pairs 5-6) The two weakest experts (0.55, 0.62) jointly favour one option, everything else ties. AEWI ~0.72; RD-WEI ~0.08 (reversed and near-ceiling). Combined with T1/T2 this pins how confidence scales with the amount vs the lopsidedness of the discriminating evidence.
(T4, pairs 7-8) TOP-CUE CONFLICT: 0.92 favours A while 0.70, 0.62 and 0.55 all favour B (0.84 and 0.76 tie). Both models pick A, but for opposite reasons: AEWI because 3.49 > 1.27 in absolute units (~0.99); RD-WEI because A carries FEWER weighted marks (~0.65-0.75). A direction-free magnitude test.
(T5, pairs 9-10) EQUAL-TALLY, TOP-HEAVY 3v3: A holds 0.92+0.84+0.55, B holds 0.76+0.70+0.62. AEWI ~0.999 (weight mass is grossly lopsided in absolute log-odds units); RD-WEI ~0.38-0.44 (compressed weights make this a near-tie, and with all six cues in play the normalized share is tiny, so RD-WEI uniquely predicts NOISY, near-chance responding with large between-subject spread exactly here).
(T6, pairs 11-12) 2v4 with the top cue on the SMALL side: A holds 0.92+0.55, B holds 0.84+0.76+0.70+0.62. AEWI ~0.16 (B); RD-WEI ~0.85 (A: fewer marks). Full sign reversal in the direction opposite to T5's disagreement, so no single response bias reproduces both.
(T7, pairs 13-14) UNANIMOUS 6-0 dominance. AEWI ~1.00 for the all-1 option; RD-WEI ~0.01 for it (reversed frame), and \u2014 crucially \u2014 sub-ceiling relative to its own single-cue trials in logit terms because normalization by the full discriminating set caps the margin at S^(1-kappa).
(T8, pairs 15-16) 1v1 MID CONFLICT: 0.84 favours A, 0.76 favours B, all else ties. AEWI ~0.84 (2.03 vs 1.22 in absolute units); RD-WEI ~0.45 (compressed weights 1.18 vs 1.05 make this a coin flip). Isolates the compression exponent with only two cues in play.

DIAGNOSTIC SUMMARY of p(A) across T1..T8: AEWI = (0.55, 0.999, 0.72, 0.99, 0.999, 0.16, 1.00, 0.84); RD-WEI = (0.16, 0.08, 0.08, 0.70, 0.41, 0.85, 0.01, 0.45). The two vectors are anti-correlated on six of eight types and, more importantly, differ in shape in ways no reparameterisation can fix: AEWI cannot produce a flat single-cue profile (T1 vs T2) at any gamma/beta, and it cannot produce near-chance responding on T5 while being decisive on T1-T3; RD-WEI cannot produce a steep validity-scaled single-cue gradient nor decisive choice on the equal-tally top-heavy pair. Even a hypothetical sign-flipped AEWI fails: it would predict 0.45/0.001/0.28/0.01/0.001 on T1-T5, versus RD-WEI's 0.16/0.08/0.08/0.70/0.41.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The two accounts are dissociated most sharply on a subset of the designed pairs: (a) single-strong-cue trials (only the .92 expert discriminates), (b) two-weak-cue trials (.55 + .62 discriminate), (c) 2-vs-4 splits with the top cue on the small side, and (d) unanimous 6-0 dominance pairs. I select trials programmatically by simulating both models at representative parameter values and keeping only pairs where (i) the two models' predicted p(A) fall on opposite sides of 0.5 and (ii) both models are decisive there (|p-0.5| not tiny). This automatically isolates exactly those four mirrored trial families (48 of the 96 trials) and excludes the pairs where the models agree (top-cue-conflict, mid 1v1, equal-tally) or where one model sits at chance and would only inject noise.

The score is the proportion of those trials on which the subject chose the option favoured by RD-WEI (i.e. the option carrying the SMALLER compressed-weight count of positive marks). Under RD-WEI this should be ~0.85-0.92 (reversed frame, near-ceiling on lopsided normalized splits, attenuated only by lapses and the small forward-frame minority). Under AEWI it should be ~0.10-0.15, since absolute log-odds weighting makes the .92 cue and the unanimous pairs near-deterministic in the opposite direction. Because the selected trials are the ones where BOTH models are confident, per-subject variability is small on each side, so the mean gap (~0.75) should dwarf between-subject spread at N=25. Crucially the metric is not trivially true for my theory: it is a directional accuracy score defined on trial families where any forward integrator (tallying, Franklin, TTB, naive Bayes) would land near 0, so it tests the reversed-plus-normalized signature rather than restating it.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Fixed design validities (communicated in instructions)
    v = np.array([0.70, 0.92, 0.55, 0.84, 0.62, 0.76], dtype=float)
    d = np.log(v / (1.0 - v))

    # Representative parameter settings for the two competing accounts
    w_rd = d ** 0.33          # compressed (near-tally) weights
    w_ae = d ** 1.41          # absolute log-odds^gamma weights
    beta_rd, kappa = 2.1, 0.52
    beta_ae = 2.0

    def _p_rd_A(a, b):
        diff = a - b
        E = float(np.dot(w_rd, diff))
        disc = np.abs(diff) > 0.0
        S = float(w_rd[disc].sum()) if np.any(disc) else 0.0
        if S > 1e-12:
            E = E / (S ** kappa)
        z = -beta_rd * E      # reversed polarity frame s = -1
        z = float(np.clip(z, -500.0, 500.0))
        return 1.0 / (1.0 + np.exp(-z))

    def _p_ae_A(a, b):
        E = float(np.dot(w_ae, a - b))
        z = float(np.clip(beta_ae * E, -500.0, 500.0))
        return 1.0 / (1.0 + np.exp(-z))

    memo = {}

    def _target(a_list, b_list):
        key = (tuple(int(x) for x in a_list), tuple(int(x) for x in b_list))
        if key in memo:
            return memo[key]
        a = np.asarray(key[0], dtype=float)
        b = np.asarray(key[1], dtype=float)
        out = None
        if a.shape[0] == v.shape[0] and b.shape[0] == v.shape[0]:
            pr = _p_rd_A(a, b)
            pa = _p_ae_A(a, b)
            # keep only pairs where the two accounts point in OPPOSITE
            # directions and BOTH are reasonably decisive there
            if (pr - 0.5) * (pa - 0.5) < 0 and abs(pa - 0.5) >= 0.15 and abs(pr - 0.5) >= 0.25:
                out = 0 if pr > 0.5 else 1   # option favoured by RD-WEI
        memo[key] = out
        return out

    hits = 0
    n = 0
    for a_list, b_list, resp in zip(data['option_a_ratings'],
                                    data['option_b_ratings'],
                                    data['response']):
        try:
            t = _target(a_list, b_list)
        except Exception:
            t = None
        if t is None:
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        n += 1
        if r == t:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)


### RESULTS
- Predicted under pi_4 (simulated): 0.8321 (var=0.0476)
- Predicted under pi_3 (simulated): 0.1250 (var=0.0019)
- Observed on real data: 0.8458 (var=0.0103)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
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



### METRIC
Rationale:
This isolates the two trial families where the two theories make maximally opposed, low-variance predictions: (a) the unanimous 7-0 dominance pairs (trials 13/14) and (b) the pairs decided by the single highly diagnostic 0.95 expert (trials 3/4). In both families I score the fraction of choices for the option carrying the POSITIVE expert ratings on the discriminating cue(s). AEWI is a forward integrator in absolute log-odds units: the margin is 4.61 (single 0.95 cue) or ~10.7 (unanimous), so choices are essentially deterministic toward the positively-rated option, giving ~0.98-1.00 with only the small lapse (eps<=0.03) pulling it down. RD-WEI's subject-level reversed polarity frame (s=-1 for ~97% of subjects) means the positively-marked option is systematically AVOIDED, and divisive normalization keeps the reversed choice confident even on the single-cue pairs, giving ~0.02-0.08. The metric therefore has a mean gap near 0.95 while being extremely tight per subject (24 trials each, near-ceiling/near-floor binomials), so it survives Welch's test at N=25. It is not trivially true for AEWI: any near-tally forward integrator would also score high, and a lexicographic TTB rule would too \u2014 the metric is diagnostic of the polarity/absolute-margin claim, not of the specific weighting exponent.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    V = np.array([0.66, 0.95, 0.55, 0.80, 0.60, 0.88, 0.72], dtype=float)
    hits = 0
    n = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            av = np.asarray(list(a), dtype=float)
            bv = np.asarray(list(b), dtype=float)
        except Exception:
            continue
        if av.shape[0] != bv.shape[0] or av.shape[0] == 0:
            continue
        d = av - bv
        nz = np.nonzero(d)[0]
        if nz.size == 0:
            continue
        include = False
        # (1) unanimous / total-dominance pairs: every cue discriminates, same direction
        if nz.size == d.shape[0] and (np.all(d > 0) or np.all(d < 0)):
            include = True
        # (2) single-cue pairs where the lone discriminating cue is highly valid
        elif nz.size == 1 and d.shape[0] == V.shape[0] and V[nz[0]] >= 0.85:
            include = True
        if not include:
            continue
        # target = the option carrying the positive rating(s) on the discriminating cue(s)
        target = 0 if d[nz[0]] > 0 else 1
        try:
            chosen = int(r)
        except Exception:
            continue
        if chosen == target:
            hits += 1
        n += 1
    if n == 0:
        return 0.5
    return float(hits) / float(n)

### RESULTS
- Predicted under pi_4 (simulated): 0.0958 (var=0.0153)
- Predicted under pi_3 (simulated): 0.9917 (var=0.0004)
- Observed on real data: 0.1383 (var=0.0102)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    a_all = data['option_a_ratings'].tolist()
    b_all = data['option_b_ratings'].tolist()
    resp = data['response'].tolist()

    hits = 0
    n = 0
    for a, b, r in zip(a_all, b_all, resp):
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # TTB winner: features are ordered by descending validity in this design,
        # so scan left to right for the first discriminating cue.
        ttb = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb is None:
            continue
        # Tally winner: count strict feature-wise wins.
        aw = float(np.sum(a > b))
        bw = float(np.sum(b > a))
        if aw == bw:
            continue
        tal = 0 if aw > bw else 1
        # Keep only trials where the two heuristics point to OPPOSITE options.
        if tal == ttb:
            continue
        n += 1
        try:
            rr = int(r)
        except Exception:
            continue
        if rr == ttb:
            hits += 1

    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.7367 (var=0.0369)
**Predicted under pi_4:** 0.7429 (var=0.0044)
**Predicted under pi_3:** 0.7625 (var=0.0042)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.60, 0.93, 0.55, 0.80, 0.68, 0.57], dtype=float)

    def ttb_winner(a, b, val):
        order = np.argsort(-val, kind="stable")
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        n = a.shape[0]
        val = validities if validities.shape[0] == n else np.linspace(0.9, 0.55, n)

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_win = 0
        elif margin < 0:
            tally_win = 1
        else:
            continue  # tally undecided: excluded

        tw = ttb_winner(a, b, val)
        if tw < 0 or tw == tally_win:
            continue  # only trials where the two rules point opposite ways
        if abs(margin) < 3:
            continue  # keep only large-margin conflicts (strong tally signal)

        hits.append(1.0 if int(row["response"]) == tally_win else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.1889 (var=0.0124)
**Predicted under pi_4:** 0.1622 (var=0.0193)
**Predicted under pi_3:** 0.1833 (var=0.0042)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    if len(data) == 0:
        return 0.0

    A = np.array([[int(x) for x in r] for r in data['option_a_ratings']], dtype=float)
    B = np.array([[int(x) for x in r] for r in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int)
    n_feat = A.shape[1]

    val = np.array([0.72, 0.85, 0.55, 0.78, 0.60, 0.66], dtype=float)
    if val.shape[0] != n_feat:
        val = np.linspace(0.90, 0.55, n_feat)
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-6)

    order = np.argsort(-val, kind='stable')

    # log-odds diagnosticity weights with a mid-range gamma (reference AEWI weights)
    d = np.log(val / (1.0 - val))
    w = np.power(np.maximum(d, 1e-12), 1.4)
    beta_ref = 2.0

    diff = A - B  # +1 favours A on that cue, -1 favours B

    # Take-The-Best winner per trial: first discriminating cue in validity order
    n = A.shape[0]
    ttb = np.full(n, -1, dtype=int)  # 0 = A, 1 = B, -1 = no discrimination
    undecided = np.ones(n, dtype=bool)
    for j in order:
        dj = diff[:, j]
        pickA = undecided & (dj > 0)
        pickB = undecided & (dj < 0)
        ttb[pickA] = 0
        ttb[pickB] = 1
        undecided = undecided & (dj == 0)

    valid = ttb >= 0
    if not np.any(valid):
        return 0.0

    # Signed absolute-evidence margin IN FAVOUR OF the TTB-chosen option
    E = diff.dot(w)
    sign_ttb = np.where(ttb == 0, 1.0, -1.0)
    margin = E * sign_ttb
    p_aewi = 1.0 / (1.0 + np.exp(-beta_ref * margin))

    agree = (resp == ttb).astype(float)

    high = valid & (p_aewi > 0.65)   # integration and one-reason agree strongly
    low = valid & (p_aewi < 0.35)    # integration reverses the one-reason choice

    if high.sum() == 0 or low.sum() == 0:
        return 0.0

    return float(agree[high].mean() - agree[low].mean())
```

**Observed (real) value:** -0.6733 (var=0.0205)
**Predicted under pi_4:** -0.6700 (var=0.0086)
**Predicted under pi_3:** 0.7472 (var=0.0055)

### Experiment 6
**Design**
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Within-subject contrast: TTB-consistency on large-margin AGREEMENT pairs
    minus TTB-consistency on large-margin CONFLICT pairs (where the weighted
    log-odds evidence points opposite to the single best discriminating cue).
    TTB must give the same choice probability in both sets (=> ~0);
    AEWI predicts ~0.99 vs ~0.15 (=> ~0.85)."""
    v = np.array([0.74, 0.90, 0.55, 0.70, 0.82, 0.62, 0.78], dtype=float)
    d = np.log(v / (1.0 - v))
    w = np.power(d, 1.4)
    order = list(np.argsort(-v, kind='stable'))

    agree = []
    conflict = []

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        if a.size != 7 or b.size != 7:
            continue
        diff = a - b
        E = float(np.dot(w, diff))

        winner = None
        for j in order:
            if diff[j] > 0:
                winner = 0
                break
            if diff[j] < 0:
                winner = 1
                break
        if winner is None:
            continue

        try:
            resp = int(r)
        except Exception:
            continue
        cons = 1.0 if resp == winner else 0.0

        sgn = 1.0 if winner == 0 else -1.0
        signed = E * sgn
        if signed > 0 and abs(E) >= 1.5:
            agree.append(cons)
        elif signed < 0 and abs(E) >= 0.5:
            conflict.append(cons)

    if len(agree) == 0 or len(conflict) == 0:
        return 0.0
    return float(np.mean(agree) - np.mean(conflict))

```

**Observed (real) value:** -0.4100 (var=0.0833)
**Predicted under pi_4:** -0.4683 (var=0.0405)
**Predicted under pi_3:** 0.8408 (var=0.0061)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments the picture is lopsided. Experiment 1 (pi_4's design): observed 0.846 vs pi_4's 0.832 and pi_3's 0.125 \n\ndecisive win for RD-WEI. Experiment 2 (pi_3's own design, engineered to favor it): observed 0.138 vs pi_4's 0.096 and pi_3's 0.992 \n\nagain a decisive win for RD-WEI, on pi_3's home turf. Experiment 5: observed -0.673, pi_4 -0.670, pi_3 +0.747. Experiment 6: observed -0.410, pi_4 -0.468, pi_3 +0.841. Experiments 3 and 4 are non-diagnostic (both theories predict essentially the observed value: 0.74 vs 0.74/0.76, and 0.19 vs 0.16/0.18) because those metrics were built on TTB-vs-tally contrasts that a reversed integrator mimics. So on every diagnostic experiment, the direction of choice is REVERSED relative to forward integration, and pi_4 (RD-WEI) nails both the sign and, approximately, the magnitude. pi_3 (AEWI) is not merely mis-parameterized \n\nit is sign-wrong on four independent designs, at every gamma/beta. Its central claim (forward polarity plus absolute log-odds margins) is falsified. Notably, pi_4 is not perfect either: it slightly overshoots reversal (E2 observed 0.138 vs predicted 0.096; E6 observed -0.410 vs predicted -0.468; E1 observed 0.846 vs 0.832), and observed between-subject variance is much lower on E1 (0.010 vs 0.048) but higher on E4/E6, suggesting the mixture over polarity frames and the amount of graded noise are not quite right. But relative to pi_3 it is clearly the task-invariant account. Since one theory is dead and the other is good, the productive move is to replace the loser with a genuinely competitive alternative that shares the empirically established reversal but derives it from a different mechanism, so the next round tests mechanism rather than sign.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_3 (AEWI) entirely. It is sign-wrong on all four diagnostic experiments (1, 2, 5, 6) and only matches on the two non-diagnostic ones; no reparameterization can rescue forward polarity. Propose a NEW theory that accepts the empirically robust reversal but explains it by a different mechanism than RD-WEI's compressed-weight + divisive-normalization scheme, so the two are separable. Suggested sketch: 'Rank-Ordered Reversed Lexicographic Screening with Cue-Count Penalty' \n\n subjects interpret a '1' as a warning flag/mark rather than an endorsement, and rather than integrating all cues compensatorily they apply an ordered search over experts sorted by validity: they scan cues in descending validity, and choose the option lacking the flag on the FIRST discriminating cue (a reversed Take-The-Best), with a probabilistic search-depth / attention parameter that occasionally lets a lower cue or a raw flag-count tally decide. Key contrasts with RD-WEI: (a) it predicts a STEEP validity-graded single-cue ladder in the reversed direction (near-ceiling reversal when only the .95 cue discriminates, near-chance when only the .55 cue discriminates), whereas RD-WEI predicts a nearly FLAT reversed ladder \n\n Experiment 2's T1/T2/T3 families and Experiment 1's T1/T2 give direct leverage; (b) it predicts reversal magnitude on many-cue near-even splits governed by cue RANK, not by a normalized share, so it makes different predictions on the equal-tally top-heavy pairs (E1 T5, E2 T5); (c) it can accommodate the observed reduced between-subject variance on E1 and larger variance on E4/E6 via a search-depth distribution rather than a polarity mixture. Parameterize with: validity-ordered search, a stopping/attention probability per cue, an optional flag-count tie-breaker weight, inverse temperature, and a lapse. This produces a strong, non-degenerate competitor to RD-WEI that agrees on the sign of every existing observation but disagrees on the SHAPE of the single-cue confidence gradient \n\n exactly the axis future experiments should probe."
}
```

## Usage

```json
{
  "input_tokens": 22606,
  "output_tokens": 1531
}
```
