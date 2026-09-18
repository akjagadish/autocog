# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Goal: dissociate Take-The-Best (one-reason, validity-ordered, all-or-none confidence) from Tallying (unweighted vote count, graded confidence in the tally margin). Five experts with well-spread validities (0.90, 0.80, 0.70, 0.60, 0.55) are used so that the top cue is clearly the most diagnostic while the lower four collectively outvote it.

Four trial families, each balanced for A/B side so that no response bias can mimic either model:
(1) DIRECTIONAL CONFLICTS (pairs 1-6, 13-14): the highest-validity cue favors one option while a majority of lower cues favors the other. TTB predicts the top-cue option with constant high probability; Tallying predicts the opposite option, with probability graded by the tally margin (3, 2, 1). These trials give opposite-sign predictions and are the primary discriminator. Pairs 13-14 add a variant where cue 1 ties so TTB must descend to cue 2, again against the majority.
(2) TALLY-TIE / TTB-DECIDES (pairs 7-8): each option wins two cues, so Tallying is exactly at chance (50/50 irrespective of beta), while TTB decides confidently via the 0.90 cue. Any systematic deviation from 50% here is fatal for Tallying and predicted by TTB.
(3) MARGIN-SCALING CONTROLS (pairs 9-12, 15-16): both models pick the same option, but the tally margin varies from 1 (pairs 9-10, decided only by the lowest-validity 0.55 cue; and 15-16, 3-vs-2) to 5 (pairs 11-12, unanimous). TTB predicts an identical accuracy/confidence level across all of these (sigmoid(beta) whichever cue was used), whereas Tallying predicts a strongly graded increase in choice consistency with margin. This is a purely quantitative dissociation that does not depend on choice direction, and it also stresses TTB's claim that a decision resting on a 55%-valid cue is made just as decisively as a unanimous one.
Together these 16 unique pairs (K=6 repetitions each, 96 trials) yield both sign-level and magnitude-level separation of the two heuristics.

[1] Goal: separate Take-The-Best (one-reason, validity-ordered, all-or-none confidence) from Tallying (unweighted vote count, confidence graded by the tally margin) using a design that is deliberately different from the previously explored one in three ways: (i) six cues instead of five, (ii) the validities are presented in a SCRAMBLED (non-descending) display order, so the top cue is Expert 2 (0.93) and the worst is Expert 3 (0.55) \-- this prevents a simple 'use the first-listed expert' strategy from mimicking TTB and forces genuine validity re-sorting, and (iii) the tally margins on conflict trials are pushed to a wider range (1, 3, 5 cue-wins against the top cue).

Validities: [0.60, 0.93, 0.55, 0.80, 0.68, 0.57] (display order). Descending validity order is therefore expert index 1 > 3 > 4 > 0 > 5 > 2.

Four trial families (16 unique pairs, K = 6, 96 trials), each mirrored across the A/B sides so that no response bias can imitate either model:

(1) DIRECTIONAL CONFLICTS with graded margin (pairs 1-6). The 0.93 cue favors one option while 2, 4, or 5 lower-validity cues favor the other (tally margins of 1, 3 and 4 against the top cue). TTB predicts the top-cue option with a CONSTANT probability sigmoid(beta) regardless of how lopsided the vote is; Tallying predicts the OPPOSITE option with probability increasing monotonically in the margin. These trials give opposite-sign predictions plus a within-model gradient, so a single conflict-choice curve identifies the model.

(2) TTB-DESCENT CONFLICT (pairs 7-8). The 0.93 cue ties, so TTB must descend to the 0.80 cue, which favors one option, while the remaining four cues (0.60, 0.68, 0.57, 0.55) all favor the other. Again opposite-sign predictions, but now the TTB-preferred option is supported by a non-top cue, testing whether the one-reason rule really tracks validity rank rather than salience of the best expert.

(3) TALLY-TIE / TTB-DECIDES (pairs 9-12). Each option wins an equal number of cues, so Tallying is exactly at chance (50/50 for every beta), while TTB decides confidently. In pairs 9-10 the deciding cue is the 0.93 expert; in pairs 11-12 the two highest cues tie and the tie-breaking cue is the mid-validity 0.68 expert. Systematic above-chance agreement with TTB here is impossible for Tallying at any parameter setting, and the pair 9-10 vs 11-12 contrast further checks TTB's claim of equal confidence irrespective of which cue happened to discriminate.

(4) MARGIN-SCALING CONTROLS, models agree in direction (pairs 13-16). Pairs 13-14: only the WORST cue (0.55) discriminates, everything else ties (tally margin 1). Pairs 15-16: unanimous 6-0 dominance (tally margin 6). Both models pick the same option, but TTB predicts identical choice consistency in both cases (sigmoid(beta) after a lapse), whereas Tallying predicts near-chance-plus-a-little on the 0.55-only trials and near-deterministic choice on the unanimous ones. This is a purely quantitative, direction-free dissociation, and it stresses the strongest counterintuitive commitment of TTB: a decision resting on a 55%-accurate expert should be made just as decisively as a 6-0 consensus.

Together the families deliver sign-level separation (families 1-2), a parameter-free zero-information prediction for Tallying (family 3), and a magnitude/confidence-gradient test (families 1 and 4).

[2] GOAL: dissociate Take-The-Best (one-reason, validity-ordered, all-or-none confidence) from Absolute-Evidence Weighted Integration (AEWI: unnormalised log-odds^gamma weights, softmax in absolute evidence units). Both are *weight-order* consistent, so a Tallying-style design will not separate them; what separates them is (a) the SIGN flip AEWI produces when several mid/low cues jointly outweigh the top cue in absolute log-odds units, and (b) the GRADED magnitude of choice consistency that AEWI predicts across pairs where TTB necessarily predicts one constant probability sigmoid(beta).

ENVIRONMENT: six experts with a deliberately COMPRESSED top (0.85) and a long tail (0.78, 0.72, 0.66, 0.60, 0.55), displayed in scrambled order [0.72, 0.85, 0.55, 0.78, 0.60, 0.66] so that no 'use the first-listed expert' strategy can mimic TTB and genuine validity re-sorting is required. Log-odds diagnosticities are d = (0.944, 1.735, 0.201, 1.266, 0.406, 0.663); at the theory's own gamma range (1.355-1.475) the AEWI weights are approx w(.85)=2.11-2.25, w(.78)=1.38-1.42, w(.72)=0.92-0.93, w(.66)=0.55-0.57, w(.60)=0.26-0.29, w(.55)=0.09-0.11. Crucially the top cue's weight is LESS than the sum of any two-or-more of the next cues, so AEWI reverses TTB's choice on most conflict trials, while TTB (which only reads validity RANK) is unaffected.

TRIAL FAMILIES (16 unique pairs, each mirrored across A/B so response bias cannot mimic either model; K = 6 repetitions, 96 trials):
(1) GRADED TOP-CUE CONFLICTS (pairs 1-8). The 0.85 expert favors one option while 1, 2, 3 or all 5 lower experts favor the other. TTB predicts the top-cue option with a CONSTANT probability sigmoid(beta) in every case. AEWI predicts a steep monotone collapse and a sign reversal: p(top-cue option) approx 0.83 (1 opposing), 0.43 (2 opposing), 0.20 (3 opposing), 0.10 (5 opposing). This single curve is the primary discriminator: it contains both an ordinal reversal and a within-family gradient that TTB forbids. The 2-opposing case sits essentially at the AEWI zero-margin point (0.41-0.46 across the gamma range), which is also where AEWI uniquely predicts maximal between-subject heterogeneity.
(2) TTB-DESCENT CONFLICT (pairs 9-10). The 0.85 cue ties, so TTB descends to 0.78 and picks that option; the remaining four cues (0.72, 0.66, 0.60, 0.55) all oppose. AEWI margin is -0.41 to -0.53, i.e. p(TTB option) approx 0.26-0.31. Opposite-sign prediction with the deciding cue no longer being the visually salient best expert.
(3) ABSOLUTE-MARGIN / CONFIDENCE SCALING (pairs 11-16), direction-free. Pairs 11-12: ONLY the 0.55 expert discriminates (all other cues tie) - TTB must be as decisive here as anywhere (sigmoid(beta)), AEWI predicts near chance (approx 0.55). Pairs 13-14: ONLY the 0.85 expert discriminates - AEWI approx 0.99. Pairs 15-16: unanimous 6-0 - AEWI approx 1.00. Because pairs 13-16 pin TTB's beta at a high value, TTB is then forced to predict that same high value on the 0.55-only pairs and on all conflict pairs; any observed monotone dependence of choice consistency on the absolute weighted margin is fatal for TTB and is exactly AEWI's core claim.

DIAGNOSTIC SUMMARY: TTB predicts one number (roughly constant agreement with the top discriminating cue) across pairs 1,3,5,7,9,11,13,15; AEWI predicts the ordered sequence approx (0.83, 0.43, 0.20, 0.10, 0.28, 0.55, 0.99, 1.00). No parameter setting of either model can imitate the other's pattern on this set.

[3] GOAL: dissociate Take-The-Best (one-reason, validity-RANK only, all-or-none confidence) from Absolute-Evidence Weighted Integration (AEWI: unnormalised log-odds^gamma weights, softmax in absolute evidence units). Both models are weight-order consistent, so the separation must come from (a) SIGN reversals when two or three mid-validity cues jointly outweigh the top cue in absolute log-odds units, and (b) a GRADED confidence ladder across pairs where TTB is forced to predict a single constant probability sigmoid(beta).

ENVIRONMENT: seven experts, one clearly dominant cue (0.90) plus a dense mid band (0.82, 0.78, 0.74, 0.70) and two weak cues (0.62, 0.55), displayed in SCRAMBLED order [0.74, 0.90, 0.55, 0.70, 0.82, 0.62, 0.78] so no 'use the first-listed expert' rule can imitate TTB and genuine validity re-sorting is required. Log-odds diagnosticities d = (1.046, 2.197, 0.201, 0.847, 1.516, 0.489, 1.266). At the theory's gamma range (1.355-1.475) the AEWI weights are approx w(.90)=2.91-3.19, w(.82)=1.76-1.85, w(.78)=1.38-1.42, w(.74)=1.06-1.07, w(.70)=0.78-0.80, w(.62)=0.36-0.37, w(.55)=0.10-0.11. The band is tuned so that the top cue BEATS any single mid cue and both weak cues combined, is roughly TIED with {0.82+0.74}, is slightly BEATEN by {0.82+0.78}, and is decisively beaten by three mid cues. TTB, reading only rank, is blind to all of these distinctions.

TRIAL FAMILIES (16 unique pairs, every pair mirrored across A/B so response bias cannot mimic either model; K = 6, 96 trials). In every odd-numbered pair TTB chooses A, in every even-numbered pair TTB chooses B.
(1) GRADED TOP-CUE CONFLICTS, spanning the AEWI zero-margin crossing. Pairs 1-2: 0.90 vs {0.62, 0.55} -> AEWI p(top option) ~ 0.99 (agrees with TTB). Pairs 3-4: 0.90 vs {0.82, 0.74} -> AEWI ~ 0.54-0.63 (top side, but barely). Pairs 5-6: 0.90 vs {0.82, 0.78} -> AEWI ~ 0.39-0.47 (sign reversal, and exactly the zero-margin point where AEWI uniquely predicts maximal between-subject heterogeneity driven by gamma). Pairs 7-8: 0.90 vs {0.82, 0.78, 0.74} -> AEWI ~ 0.08 (strong reversal). Across these four conflict types TTB must predict ONE constant number; AEWI predicts a steep monotone collapse through a sign change (0.99, 0.58, 0.43, 0.08).
(2) TTB-DESCENT CONFLICT (pairs 9-10). The 0.90 cue ties (both options rated 1), so TTB descends to 0.82 and picks that option, while 0.78 and 0.74 jointly oppose. AEWI margin is negative (~-0.6 to -0.7), p(TTB option) ~ 0.21: an opposite-sign prediction where the deciding cue is no longer the salient best expert.
(3) ABSOLUTE-MARGIN CONFIDENCE LADDER (pairs 11-16), direction-free (both models pick the same option). Pairs 11-12: ONLY the 0.55 expert discriminates -> AEWI ~ 0.55 (near chance), TTB must be as decisive as anywhere. Pairs 13-14: ONLY the 0.70 expert discriminates -> AEWI ~ 0.83. Pairs 15-16: unanimous 7-0 -> AEWI ~ 1.00. Because the unanimous and 0.99-margin pairs pin TTB's beta at a high value, TTB is then committed to that same high value on the 0.55-only pairs and on every conflict pair; any observed monotone dependence of choice consistency on the absolute weighted margin (0.55 < 0.83 < 1.00) falsifies TTB and is precisely AEWI's core claim.

DIAGNOSTIC SUMMARY: on the odd (TTB-picks-A) pairs 1,3,5,7,9,11,13,15 TTB predicts the same constant p(A); AEWI predicts the ordered sequence approx (0.99, 0.58, 0.43, 0.08, 0.21, 0.55, 0.83, 1.00), including two sign reversals and a near-zero crossing. No parameter setting of either model can reproduce the other's pattern on this set, and the pairs 3-6 crossing region additionally tests AEWI's unique prediction of inflated between-subject variance where the absolute margin is near zero.

[4] GOAL: dissociate RD-WEI (compressed near-tally weights, subject-level REVERSED polarity frame s=-1, and divisive normalization of the margin by the weight of the discriminating cues) from AEWI (absolute, un-normalised log-odds^gamma weights with gamma~1.4, forward polarity). Three orthogonal handles are built into the design, so the two models are separated by SIGN, by the SHAPE of the confidence gradient, and by WHERE choices are noisy \u2014 not by overall accuracy alone.

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

[5] GOAL: dissociate AEWI (absolute, un-normalised log-odds^gamma weights, gamma~1.4, forward polarity) from RD-WEI (compressed near-tally weights gamma~0.33, reversed subject-level polarity s=-1, and divisive normalization of the margin by the weight of the *discriminating* cues, kappa~0.5). Separation is engineered on three orthogonal axes: (a) the SHAPE of the single-cue confidence ladder, (b) SIGN reversals on multi-cue splits, and (c) WHERE responding is noisy vs deterministic.

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

[6] GOAL: dissociate RD-WEI (compressed near-tally weights, reversed polarity s=-1, divisive normalization of the margin by the weight of the DISCRIMINATING cues) from RFS-C+D (reversed flag COUNTING under an absolute attention budget, plus a rank-lexicographic DUEL mode on sparse difference sets). Both theories share the reversed polarity claim (fewer marks = better), so overall direction cannot separate them; the design therefore targets three quantitative signatures where they diverge: (i) how a 1-flag-vs-1-flag duel is resolved (graded in the WEIGHT GAP for RD-WEI vs rank-lexicographic and graded in CONSULTATION LEVEL for RFS-C+D), (ii) the height and slope of the single-cue ladder (normalization makes RD-WEI near-ceiling and almost flat; consultation failure makes RFS-C+D much lower and steeper), and (iii) a count-vs-weight magnitude test on a 2-vs-2 top-heavy conflict.

ENVIRONMENT: six experts in SCRAMBLED display order [0.72, 0.95, 0.55, 0.90, 0.60, 0.80] so no 'first-listed expert' rule can imitate either model. Log-odds d = (0.944, 2.944, 0.201, 2.197, 0.405, 1.386). At RD-WEI's gamma~0.33 the weights are w = (0.98, 1.43, 0.59, 1.30, 0.74, 1.11) (only ~2.4:1 spread). At RFS-C+D's rho~0.35, q_max~0.85, capacity~3.1 the consultation probabilities are q ~ (0.52, 0.665, 0.31, 0.64, 0.39, 0.58). Crucially the panel is built with TWO near-equal high experts (0.95/0.90), TWO near-equal mid experts (0.80/0.72), TWO near-equal weak experts (0.60/0.55) and one maximal gap (0.95 vs 0.55).

NINE TYPES, each mirrored across A/B so no side bias can mimic either model (18 unique pairs, K=5, 90 trials). Predicted p(A) at representative parameters (RD-WEI: gamma=0.33, kappa=0.52, beta=2.1, s=-1; RFS-C+D: rho=0.35, q_max=0.85, C=3.1, beta=4.5, omega=0.92, lam=1.4):

(TA, pairs 1-2) DUEL, small gap / HIGH level: 0.95 flags A, 0.90 flags B. RD-WEI ~0.46 (the compressed weights 1.43 vs 1.30 differ by almost nothing, so after normalization by S the trial is essentially a coin flip). RFS-C+D ~0.29 (duel mode is RANK-lexicographic and magnitude-blind: the 0.95 expert is consulted first with q=0.665 and its flag condemns A).
(TB, pairs 3-4) DUEL, small gap / LOW level: 0.60 flags A, 0.55 flags B. RD-WEI ~0.43 (again near chance, essentially the SAME value as TA: RD-WEI's duel resolution depends on the weight GAP, not on the validity level). RFS-C+D ~0.40 (both experts are rarely consulted, so the duel usually ends in a guess). TA vs TB is the cleanest interaction test: RD-WEI predicts NO level effect (0.46 vs 0.43), RFS-C+D predicts a large one (0.29 vs 0.40).
(TC, pairs 5-6) DUEL, small gap / MID level: 0.80 flags A, 0.72 flags B. RD-WEI ~0.45; RFS-C+D ~0.33.
(TD, pairs 7-8) DUEL, MAXIMAL gap: 0.95 flags A, 0.55 flags B. Both models ~0.23 (they agree here). Combined with TA/TB/TC this yields the decisive ladder: RD-WEI predicts strong GAP-gradedness (0.46, 0.45, 0.43 -> 0.23), RFS-C+D predicts a nearly FLAT duel profile (0.29, 0.33, 0.40 -> 0.23) that is ordered by consultation probability instead. No reparameterisation lets one model produce the other's ladder: RFS-C+D's duel mode cannot return to chance for two high-validity, near-equal experts, and RD-WEI cannot be decisive when the two duelling weights are nearly identical.
(TE, pairs 9-10) SINGLE CUE 0.55 (other cues tie, two of them tied at 1 so the displays are non-degenerate). RD-WEI ~0.84 (after normalization a 1-vs-0 discrimination gives E' = w^(1-kappa) ~ 0.78 even for the worst expert). RFS-C+D ~0.655 (the flag is read only ~31% of the time; otherwise a guess).
(TF, pairs 11-12) SINGLE CUE 0.72. RD-WEI ~0.89; RFS-C+D ~0.76.
(TG, pairs 13-14) SINGLE CUE 0.95. RD-WEI ~0.92; RFS-C+D ~0.83. TE-TG give the second orthogonal handle: RD-WEI predicts a HIGH, nearly FLAT single-cue ladder (0.84, 0.89, 0.92; span ~0.09) because divisive normalization cancels most of the cue-strength difference, whereas RFS-C+D predicts a much LOWER, twice-as-steep ladder (0.655, 0.76, 0.83; span ~0.18) tracking q_i = q_max*(2v-1)^rho under the capacity budget.
(TH, pairs 15-16) UNANIMOUS 6-0 dominance. Both models ~0.99 for the all-0 option. This pair is included as a LAPSE ANCHOR: it pins epsilon for both models so the TE-TG levels cannot be mimicked by simply raising the lapse rate (a large epsilon that pushes RD-WEI down to RFS-C+D's 0.655 on TE would also push dominance well below ceiling).
(TI, pairs 17-18) 2-vs-2 TOP-HEAVY CONFLICT: A flagged by 0.95+0.90, B flagged by 0.60+0.55 (four cues discriminate, so RFS-C+D's duel mode is switched off, p_duel ~ 0.003). RD-WEI ~0.20 (weighted marks 2.73 vs 1.33, decisive against A); RFS-C+D ~0.28 (counts are EQUAL, 2 vs 2; only the asymmetry in reading probability tilts it). A pure magnitude test in a regime where both models agree in direction.

DIAGNOSTIC SUMMARY of p(A) across TA..TI: RD-WEI = (0.46, 0.43, 0.45, 0.23, 0.84, 0.89, 0.92, 0.99, 0.20); RFS-C+D = (0.29, 0.40, 0.33, 0.23, 0.655, 0.76, 0.83, 0.99, 0.28). The vectors agree in sign everywhere (both are reversed-polarity accounts) but differ in SHAPE in ways neither can reparameterise away: RD-WEI's duel resolution is gap-graded and level-independent while RFS-C+D's is level-graded and gap-independent; RD-WEI's single-cue profile is high and flat while RFS-C+D's is low and steep, with the dominance pair fixing the lapse rate that would otherwise trade off against that level difference.

[7] GOAL: dissociate RFS-C+D (reversed flag COUNTING under an absolute attention budget, with a rank-lexicographic DUEL mode on sparse difference sets) from RD-WEI (compressed near-tally weights, reversed polarity, divisive normalization by the discriminating-cue mass). Both are reversed-polarity accounts, so direction cannot separate them; earlier work (experiment [6]) already probed a 6-expert panel. Here I exploit two handles that a 6-expert panel cannot deliver and that no reparameterisation can erase.

HANDLE 1 - PANEL-SIZE / CAPACITY DILUTION. I use EIGHT experts, displayed in SCRAMBLED order [0.70, 0.97, 0.55, 0.88, 0.62, 0.78, 0.93, 0.66] (idx 0..7), so no 'first-listed expert' rule can mimic anything. Raw consultation probabilities q_i = q_max(2v-1)^rho (q_max=.85, rho=.35) sum to ~5.19, far above capacity C~3.1, so every q is rescaled by .60: q = (.369, .497, .227, .461, .308, .415, .482, .341). Consequently RFS-C+D is HARD-CAPPED on sparse trials: a single flag can be read at most ~50% of the time, so the best possible single-cue accuracy is ~0.75 and even two top-expert flags cap near 0.86 - while unanimity is still ~0.99 (some flag is almost surely read). RD-WEI has no such bound: after divisive normalization a 1-vs-0 discrimination gives E' = w^(1-kappa) regardless of how many experts are on the panel, so single-cue trials sit at 0.84-0.93 and 2-flag trials at 0.92-0.97. The dominance pair pins the lapse rate, so RD-WEI cannot buy RFS's low sparse-trial level by raising epsilon or lowering beta without also sinking the dominance anchor.

HANDLE 2 - DUEL LEVEL x GAP CROSSOVER (the headline, parameter-robust interaction). Three 1-flag-vs-1-flag duels are matched in structure but placed at three validity LEVELS whose compressed-weight GAPS run in the OPPOSITE direction: H = .97 vs .93 (q .497/.482, w gap small, large S), M = .88 vs .78, L = .62 vs .55 (q .308/.227, w gap relatively large, tiny S). RFS-C+D's duel is rank-lexicographic and magnitude-blind, so its decisiveness tracks CONSULTATION probability and must FALL as validity level falls; RD-WEI's duel resolution tracks the normalized weight gap (w1-w2)/S^kappa, which here RISES as level falls. Deviations from chance: RFS (.115, .109, .071) versus RD-WEI (.042, .059, .088) - a full crossover that holds across the whole stated parameter ranges (gamma .18-.48 all keep RD-WEI's ordering; any rho/C keep RFS's ordering).

EIGHT TYPES, each mirrored across A/B so no side bias can imitate either model (16 unique pairs, K=6, 96 trials). Predicted p(A) at representative parameters (RFS-C+D: rho=.35, q_max=.85, C=3.1, beta=4.5, omega=.92, lam=1.4, eps=0; RD-WEI: gamma=.33, kappa=.52, beta=2.1, s=-1, eps=0):
(T1, pairs 1-2) SINGLE CUE 0.55 (idx2), all else tied: RFS .611 | RD-WEI .836.
(T2, pairs 3-4) SINGLE CUE 0.97 (idx1): RFS .746 | RD-WEI .928. T1/T2 give the single-cue ladder: RFS LOW and STEEP (.611->.746, span .135, exactly q-graded), RD-WEI HIGH and FLAT (.836->.928, span .09).
(T3, pairs 5-6) DUEL H: 0.97 flags A, 0.93 flags B: RFS .385 | RD-WEI .458.
(T4, pairs 7-8) DUEL M: 0.88 flags A, 0.78 flags B: RFS .391 | RD-WEI .441.
(T5, pairs 9-10) DUEL L: 0.62 flags A, 0.55 flags B: RFS .429 | RD-WEI .412. T3-T5 is the crossover: RFS predicts monotonically WEAKER duel resolution as level drops, RD-WEI predicts monotonically STRONGER.
(T6, pairs 11-12) TWO WEAK FLAGS vs ZERO (0.55+0.62 on one side): RFS .727 | RD-WEI .921.
(T7, pairs 13-14) UNANIMOUS 8-0 dominance: both ~.99 - the LAPSE/BETA ANCHOR that blocks a global-noise mimicry of the sparse-trial level difference.
(T8, pairs 15-16) TWO STRONG FLAGS vs ZERO (0.97+0.93 on one side): RFS .862 | RD-WEI .970. T6 vs T8 isolates the consultation bound: RFS predicts a large weak-vs-strong flag-pair spread (.135) far below ceiling; RD-WEI predicts both near ceiling with a small spread (.049).

DIAGNOSTIC SUMMARY of p(A) across T1..T8: RFS-C+D = (.611, .746, .385, .391, .429, .727, .991, .862); RD-WEI = (.836, .928, .458, .441, .412, .921, .997, .970). The vectors agree in sign everywhere (both reversed-polarity) but differ in SHAPE in two independent, non-reparameterisable ways: (i) sparse-trial decisiveness capped well below ceiling and steeply q-graded versus near-ceiling and flat, with dominance fixed at ceiling for both; (ii) an outright crossover in how a two-expert duel is resolved as a function of validity LEVEL versus weight GAP.

[8] GOAL: dissociate RD-WEI (compressed near-tally weights gamma~0.33, reversed polarity s=-1, divisive normalisation with kappa~0.52, beta~2.1, small lapse) from R-SED (same reversed-polarity family but kappa~0.70, beta~3.6, large lapse ~0.20, and crucially a SPARSITY-GATED SEVERITY BOOST that multiplies the weight of the single most valid DISCRIMINATING cue by 1+phi*exp(-lam*(n_disc-2)) whenever n_disc>=2). Both are reversed-frame accounts, so direction cannot separate them; the design targets (i) the level of 1-flag-vs-1-flag DUEL resolution, (ii) the n_disc LADDER at matched weighted margin, and (iii) magnitude anchors that pin beta and epsilon so no global-noise reparameterisation can mimic the shape.

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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate R-SED (advocated: reversed polarity, mildly compressed weights gamma~0.34, kappa~0.70, beta~3.6, lapse~0.20, plus the SPARSITY-GATED SEVERITY BOOST that multiplies the weight of the single most valid DISCRIMINATING cue by 1+phi*exp(-lam*(n_disc-2)) when n_disc>=2) from RD-WEI (reversed polarity, gamma~0.33, kappa~0.52, beta~2.1, small lapse, NO severity gate). Both are reversed-frame accounts, so overall direction cannot separate them; the design attacks three orthogonal SHAPE signatures: (i) the LEVEL of 1-flag-vs-1-flag duel resolution and its (in)dependence of the validity GAP, (ii) the n_disc ladder at an EXACTLY matched signed weighted margin, and (iii) sparse/dominance magnitude anchors that pin beta and epsilon so no noise-only reparameterisation can mimic the pattern.\n\nENVIRONMENT: seven experts in scrambled display order [0.72, 0.95, 0.55, 0.85, 0.60, 0.72, 0.60] (idx0..idx6), so no 'use the first-listed expert' rule can imitate either model. Two exact DUPLICATE pairs are built in (idx0/idx5 = 0.72; idx4/idx6 = 0.60) so that cancelling flag pairs add EXACTLY zero to the signed weighted margin while inflating the discriminating mass S. Log-odds d = (0.944, 2.944, 0.201, 1.735, 0.406, 0.944, 0.406). At the shared compression exponent the base weights are w ~ (0.98, 1.43, 0.59, 1.20, 0.74, 0.98, 0.74) - only a ~2.4:1 spread - so ALL divergences below come from the severity gate and the kappa/beta/epsilon regime, never from weight ordering.\n\nNINE TYPES, each mirrored across A/B so no side bias can imitate either model (18 unique pairs, K=5, 90 trials). Predicted p(A) at representative parameters (R-SED: gamma=.34, kappa=.70, phi=.5, lam=1.0, beta=3.6, eps=.20, s=-1 | RD-WEI: gamma=.33, kappa=.52, beta=2.1, eps=.06, s=-1):\n(T1, pairs 1-2) DUEL HIGH, small gap: 0.95 flags A, 0.85 flags B (n_disc=2). R-SED .25 (the severity gate multiplies the 0.95 flag by 1.5 and condemns A) | RD-WEI .44 (compressed weights 1.43 vs 1.20 differ by almost nothing, so after normalisation this is nearly a coin flip). PRIMARY DISCRIMINATOR.\n(T2, pairs 3-4) DUEL MID, small gap: 0.85 flags A, 0.72 flags B. R-SED .25 | RD-WEI .43.\n(T3, pairs 5-6) DUEL LOW, small gap: 0.60 flags A, 0.55 flags B. R-SED .27 | RD-WEI .44. T1-T3 test R-SED's headline claim that duel resolution is roughly CONSTANT and independent of validity LEVEL, while RD-WEI must hover near chance throughout.\n(T4, pairs 7-8) DUEL MAX GAP: 0.95 flags A, 0.55 flags B. Both models ~.245 - they AGREE here. The T1/T4 contrast is the gap test: RD-WEI's deviation from chance must grow ~4-fold from small-gap to max-gap duels, while R-SED's stays flat (.25 -> .245). Neither model can reproduce the other's duel profile at any admissible parameters.\n(T5, pairs 9-10) MATCHED-MARGIN LADDER, n_disc=4: T1's duel plus an exactly cancelling 0.72/0.72 flag pair (idx0 on A, idx5 on B). Signed margin identical to T1; only S grows and R-SED's boost decays to 1.068. R-SED .42 | RD-WEI .45.\n(T6, pairs 11-12) MATCHED-MARGIN LADDER, n_disc=6: T1's duel plus cancelling 0.72/0.72 and 0.60/0.60 pairs. R-SED .45 | RD-WEI .46. THE LADDER T1->T5->T6 is the decisive shape test: at a CONSTANT weighted margin R-SED predicts a steep collapse toward chance (.25, .42, .45; deviation shrinks ~5-fold as the severity gate closes) whereas RD-WEI predicts an essentially FLAT profile (.44, .45, .46).\n(T7, pairs 13-14) SINGLE CUE 0.95, n_disc=1 (a tied 1 at idx0 and idx6 keeps the display non-degenerate). R-SED .11 | RD-WEI .10 - the models nearly coincide. Because R-SED's boost is OFF at n_disc=1 and ON at n_disc=2, the T7->T1 STEP is diagnostic: RD-WEI predicts a large jump toward chance when a 0.85 counter-flag is added (+.34), R-SED only +.14.\n(T8, pairs 15-16) LOPSIDED 5-vs-1: five weak/mid flags (0.72,0.72,0.60,0.60,0.55) on A against a single 0.95 flag on B (n_disc=6, boost ~1.01, i.e. inert). R-SED .245 | RD-WEI .13 - a pure magnitude test isolating the kappa/beta/epsilon regime once the severity gate is switched off, and a check that one flag from the best expert loses to four/five weak flags.\n(T9, pairs 17-18) UNANIMOUS 7-0 DOMINANCE. R-SED .10 | RD-WEI .033. LAPSE ANCHOR: it pins epsilon for both models, so R-SED's near-chance ladder tail and RD-WEI's near-chance duels cannot be traded off against global noise.\n\nDIAGNOSTIC SUMMARY of p(A) across T1..T9: R-SED = (.25, .25, .27, .245, .42, .45, .11, .245, .10); RD-WEI = (.44, .43, .44, .245, .45, .46, .10, .13, .033). The vectors agree in sign everywhere (both reversed-polarity) but differ in SHAPE in three independent, non-reparameterisable ways: (i) small-gap duels strongly resolved vs near chance, with the two accounts converging exactly at the max-gap duel; (ii) a steeply decaying vs flat n_disc ladder at an exactly matched signed margin; (iii) opposite orderings of the sparse (T7), lopsided (T8) and dominance (T9) anchors, which fix beta and epsilon and block any noise-only mimicry.",
  "validities": [
    0.72,
    0.95,
    0.55,
    0.85,
    0.6,
    0.72,
    0.6
  ],
  "trial_a_ratings": [
    [
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 25909,
  "output_tokens": 13332
}
```
