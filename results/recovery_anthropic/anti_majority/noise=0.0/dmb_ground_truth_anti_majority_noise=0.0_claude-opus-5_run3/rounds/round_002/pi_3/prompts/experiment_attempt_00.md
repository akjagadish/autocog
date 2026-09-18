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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate AEWI (absolute, un-normalised log-odds^gamma weights, gamma~1.4, forward polarity) from RD-WEI (compressed near-tally weights gamma~0.33, reversed subject-level polarity s=-1, and divisive normalization of the margin by the weight of the *discriminating* cues, kappa~0.5). Separation is engineered on three orthogonal axes: (a) the SHAPE of the single-cue confidence ladder, (b) SIGN reversals on multi-cue splits, and (c) WHERE responding is noisy vs deterministic.\n\nENVIRONMENT: seven experts in SCRAMBLED display order [0.66, 0.95, 0.55, 0.80, 0.60, 0.88, 0.72], so no 'use the first-listed expert' rule can imitate any model and genuine validity re-sorting is required. Log-odds diagnosticities d = (0.663, 2.944, 0.201, 1.386, 0.406, 1.992, 0.944). At AEWI's gamma (~1.415) the weights are w = (0.56, 4.61, 0.10, 1.59, 0.28, 2.65, 0.92): a 45:1 best-to-worst ratio, strongly non-compensatory in ABSOLUTE units. At RD-WEI's gamma (~0.33) the same environment gives w = (0.87, 1.43, 0.59, 1.11, 0.74, 1.26, 0.98): only a ~2.4:1 spread, i.e. a validity-tinted tally. Every family below exploits that 45:1 vs 2.4:1 gap plus the normalization/polarity terms.\n\nEIGHT TRIAL TYPES, each mirrored across the A/B sides (16 unique pairs, K=6, 96 trials) so no side bias can mimic either model. Predicted p(A) at representative parameters (AEWI: gamma=1.415, beta=2.0; RD-WEI: gamma=0.33, kappa=0.52, beta=2.1, s=-1):\n\n(T1, pairs 1-2) ONLY the 0.55 expert discriminates. AEWI ~0.55 (near chance: absolute margin 0.10). RD-WEI ~0.16 (confident REVERSED choice, since after normalization a 1-vs-0 discrimination gives E' = w^(1-kappa) ~ 0.78 even for the worst cue).\n(T2, pairs 3-4) ONLY the 0.95 expert discriminates. AEWI ~0.9999 (margin 4.61). RD-WEI ~0.08.\n(T3, pairs 5-6) ONLY the 0.72 expert discriminates. AEWI ~0.86. RD-WEI ~0.11.\nT1/T2/T3 together are the cleanest quantitative test: AEWI is forced into a STEEP monotone single-cue ladder (0.55 -> 0.86 -> 0.9999) scaled by absolute cue diagnosticity, while RD-WEI's compression + divisive normalization make the ladder nearly FLAT and on the opposite side of chance (0.16 -> 0.11 -> 0.08). No reparameterisation of either model reproduces the other's shape here.\n\n(T4, pairs 7-8) 3-vs-4 SPLIT, top cue on the small side: A holds 0.95+0.55+0.60, B holds 0.88+0.80+0.72+0.66. AEWI margin is slightly NEGATIVE in absolute units (4.99 vs 5.72) -> p(A) ~ 0.19 (chooses B). RD-WEI, reading fewer weighted marks as better, chooses A: ~0.75. Full sign reversal.\n(T5, pairs 9-10) ALL SEVEN cues discriminate, 4-vs-3 count but grossly weight-lopsided: A holds 0.95+0.88+0.60+0.55, B holds 0.80+0.72+0.66. AEWI ~0.9999 (absolute margin 4.58); RD-WEI ~0.31 - a near-even compressed tally divided by the full discriminating mass, hence a small relative share, mild reversed preference and (uniquely) large predicted between-subject spread exactly here.\n(T6, pairs 11-12) 1-vs-3 with the top cue alone: A holds 0.95, B holds 0.66+0.60+0.55. Both models pick A but for opposite reasons (AEWI: 4.61 > 0.94 in absolute units, ~0.999; RD-WEI: A carries fewer marks, ~0.70). A direction-free MAGNITUDE test where RD-WEI masquerades as one-reason decision making.\n(T7, pairs 13-14) UNANIMOUS 7-0 dominance. AEWI ~1.00 for the all-1 option; RD-WEI ~0.005 for it (reversed frame), and capped at S^(1-kappa) by normalization. This is the single most decisive polarity test.\n(T8, pairs 15-16) 1v1 MID CONFLICT: 0.80 favours A, 0.72 favours B, all else ties. AEWI ~0.79 (1.59 vs 0.92 in absolute units); RD-WEI ~0.45 (compressed weights 1.11 vs 0.98 make it a coin flip). Isolates the compression exponent with only two cues in play.\n\nDIAGNOSTIC SUMMARY of p(A) across T1..T8: AEWI = (0.55, 0.9999, 0.86, 0.19, 0.9999, 0.999, 1.00, 0.79); RD-WEI = (0.16, 0.08, 0.11, 0.75, 0.31, 0.70, 0.005, 0.45). The vectors disagree in SIGN on six of eight types and, crucially, differ in SHAPE: AEWI cannot produce a flat single-cue profile (T1-T3) at any gamma/beta, nor near-chance responding on the weight-lopsided seven-cue pair T5 while staying decisive on T2; RD-WEI cannot produce a validity-scaled single-cue gradient, cannot be deterministic on the unanimous pair in the all-1 direction, and cannot be decisive on T5. Even a sign-flipped AEWI fails, since it would predict 0.45/0.0001/0.14/0.81/0.0001 on T1-T5 versus RD-WEI's 0.16/0.08/0.11/0.75/0.31.",
  "validities": [
    0.66,
    0.95,
    0.55,
    0.8,
    0.6,
    0.88,
    0.72
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
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
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      1,
      1,
      1,
      1
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
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
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
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
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
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 15524,
  "output_tokens": 8319
}
```
