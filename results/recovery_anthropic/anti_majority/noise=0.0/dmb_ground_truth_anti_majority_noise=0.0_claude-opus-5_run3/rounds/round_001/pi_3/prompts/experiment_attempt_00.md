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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate Take-The-Best (one-reason, validity-RANK only, all-or-none confidence) from Absolute-Evidence Weighted Integration (AEWI: unnormalised log-odds^gamma weights, softmax in absolute evidence units). Both models are weight-order consistent, so the separation must come from (a) SIGN reversals when two or three mid-validity cues jointly outweigh the top cue in absolute log-odds units, and (b) a GRADED confidence ladder across pairs where TTB is forced to predict a single constant probability sigmoid(beta).\n\nENVIRONMENT: seven experts, one clearly dominant cue (0.90) plus a dense mid band (0.82, 0.78, 0.74, 0.70) and two weak cues (0.62, 0.55), displayed in SCRAMBLED order [0.74, 0.90, 0.55, 0.70, 0.82, 0.62, 0.78] so no 'use the first-listed expert' rule can imitate TTB and genuine validity re-sorting is required. Log-odds diagnosticities d = (1.046, 2.197, 0.201, 0.847, 1.516, 0.489, 1.266). At the theory's gamma range (1.355-1.475) the AEWI weights are approx w(.90)=2.91-3.19, w(.82)=1.76-1.85, w(.78)=1.38-1.42, w(.74)=1.06-1.07, w(.70)=0.78-0.80, w(.62)=0.36-0.37, w(.55)=0.10-0.11. The band is tuned so that the top cue BEATS any single mid cue and both weak cues combined, is roughly TIED with {0.82+0.74}, is slightly BEATEN by {0.82+0.78}, and is decisively beaten by three mid cues. TTB, reading only rank, is blind to all of these distinctions.\n\nTRIAL FAMILIES (16 unique pairs, every pair mirrored across A/B so response bias cannot mimic either model; K = 6, 96 trials). In every odd-numbered pair TTB chooses A, in every even-numbered pair TTB chooses B.\n(1) GRADED TOP-CUE CONFLICTS, spanning the AEWI zero-margin crossing. Pairs 1-2: 0.90 vs {0.62, 0.55} -> AEWI p(top option) ~ 0.99 (agrees with TTB). Pairs 3-4: 0.90 vs {0.82, 0.74} -> AEWI ~ 0.54-0.63 (top side, but barely). Pairs 5-6: 0.90 vs {0.82, 0.78} -> AEWI ~ 0.39-0.47 (sign reversal, and exactly the zero-margin point where AEWI uniquely predicts maximal between-subject heterogeneity driven by gamma). Pairs 7-8: 0.90 vs {0.82, 0.78, 0.74} -> AEWI ~ 0.08 (strong reversal). Across these four conflict types TTB must predict ONE constant number; AEWI predicts a steep monotone collapse through a sign change (0.99, 0.58, 0.43, 0.08).\n(2) TTB-DESCENT CONFLICT (pairs 9-10). The 0.90 cue ties (both options rated 1), so TTB descends to 0.82 and picks that option, while 0.78 and 0.74 jointly oppose. AEWI margin is negative (~-0.6 to -0.7), p(TTB option) ~ 0.21: an opposite-sign prediction where the deciding cue is no longer the salient best expert.\n(3) ABSOLUTE-MARGIN CONFIDENCE LADDER (pairs 11-16), direction-free (both models pick the same option). Pairs 11-12: ONLY the 0.55 expert discriminates -> AEWI ~ 0.55 (near chance), TTB must be as decisive as anywhere. Pairs 13-14: ONLY the 0.70 expert discriminates -> AEWI ~ 0.83. Pairs 15-16: unanimous 7-0 -> AEWI ~ 1.00. Because the unanimous and 0.99-margin pairs pin TTB's beta at a high value, TTB is then committed to that same high value on the 0.55-only pairs and on every conflict pair; any observed monotone dependence of choice consistency on the absolute weighted margin (0.55 < 0.83 < 1.00) falsifies TTB and is precisely AEWI's core claim.\n\nDIAGNOSTIC SUMMARY: on the odd (TTB-picks-A) pairs 1,3,5,7,9,11,13,15 TTB predicts the same constant p(A); AEWI predicts the ordered sequence approx (0.99, 0.58, 0.43, 0.08, 0.21, 0.55, 0.83, 1.00), including two sign reversals and a near-zero crossing. No parameter setting of either model can reproduce the other's pattern on this set, and the pairs 3-6 crossing region additionally tests AEWI's unique prediction of inflated between-subject variance where the absolute margin is near zero.",
  "validities": [
    0.74,
    0.9,
    0.55,
    0.7,
    0.82,
    0.62,
    0.78
  ],
  "trial_a_ratings": [
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
      1,
      0,
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
      1,
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
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
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
      0
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
      1,
      1,
      1,
      0,
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
    ]
  ],
  "trial_b_ratings": [
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
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
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
      0,
      0,
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
      1,
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
      1
    ],
    [
      0,
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
      0,
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
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
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
  "input_tokens": 10020,
  "output_tokens": 10478
}
```
