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
**Description:** **WADD-γ with Positional Validity Misattribution (WADD-γ + position-default blending), with heterogeneous decision determinacy.**

People integrate cues compensatorily: each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, L_j = log(v_j/(1−v_j)), w_j ∝ L_j^γ; the decision rests on the normalised evidence margin d = Σ_j w_j (x_Aj − x_Bj), with p(A) logistic in β·d and an attentional lapse ε. γ ≈ 1.15–1.85 places the decision maker in the near-lexicographic but still compensatory regime: the top expert usually survives a coalition of dissenters, but a near-peer second cue plus a couple of moderate cues can cancel or overturn it.

The substantive claim about *where the weights get attached* is retained and is the theory's main falsifiable commitment. Validities are stated verbally in the instructions, but the decision maker must bind each stated validity to a *column* of the displayed rating vector. This binding is costly and error-prone, and people fall back on a powerful display heuristic: **importance decreases from left to right** (the first-listed expert is the best one, the last-listed the worst). Subjective weights are therefore a blend, with mixing weight λ, of the correctly-bound validity weights and the weights implied by the positional default (the same weight multiset re-sorted into descending order across columns):

  w_eff ∝ (1−λ)·L_j^γ + λ·sort_desc(L^γ)_j.

This is an *environmental* prediction, not a per-experiment fudge: conflict-trial accuracy is a property of the alignment between display order and validity order, while the integration rule itself stays a single stationary compensatory WADD-γ. Concretely, (i) scramble the column order of an environment whose cues are currently presented in descending validity and conflict accuracy must fall toward (and, on some families, below) chance; (ii) re-sort the columns of a scrambled environment into descending validity and the same subjects must become sharply more accurate and more top-cue-consistent; (iii) an explicit column-labelling manipulation that makes the validity→column binding visually trivial should abolish the drop. Because the blend is the exact identity whenever columns are already validity-ordered, λ is behaviourally inert in validity-sorted environments and can only be identified by comparing environments — precisely the invariance signature the theory stakes itself on.

The third claim is about **population structure**: the same stationary rule is held by everyone, but decision determinacy is heterogeneous across people. Some subjects convert modest weighted margins into near-deterministic choices, others remain near chance on all but the most lopsided evidence. That heterogeneity is carried by the logistic gain β and by mild variation in the validity-sensitivity exponent γ, not by discrete strategy switching; the theory therefore predicts that between-subject spread in conflict accuracy should be largest in environments with intermediate margins and should shrink towards zero in environments where every conflict family is near-cancelling.

**Parameters:**
- gamma: [1.15, 1.85]
- beta: [1.0, 13.0]
- epsilon: [0.0, 0.14]
- lam: [0.64, 1.00]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # ------------------------------------------------------------------
    # WADD-gamma with positional validity misattribution.
    #   w_j      proportional to  logit(v_j) ** gamma
    #   w_eff    = (1-lam) * w + lam * sort_desc(w)     (position default)
    #   d        = sum_j w_eff_j * (x_Aj - x_Bj)        in [-1, 1]
    #   p(A)     = sigmoid(beta * d), mixed with lapse epsilon.
    # gamma -> 0 recovers Tallying, gamma -> large recovers Take-The-Best.
    # lam is the degree of reliance on the display heuristic "importance
    # decreases left-to-right"; it is a no-op when the experiment already
    # presents cues in descending validity order.
    # History is not used: the rule is stationary (no feedback is given).
    # ------------------------------------------------------------------
    import numpy as np

    # ---- unpack the stimulus into two rating vectors -------------------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            # flat concatenation of the two option vectors
            half = stim.shape[0] // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            a, b = stim[0].astype(float), stim[1].astype(float)

    n = int(min(a.shape[0], b.shape[0]))
    a = a[:n]
    b = b[:n]
    if n == 0:
        return np.ones(2) / 2.0

    # ---- subjective importance weights from cue diagnosticity ----------
    val = parameters.get("validities", None)
    if val is None:
        v = np.full(n, 0.75, dtype=float)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]

    # validities live in (0.5, 1); clip for numerical safety
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)
    logit = np.log(v / (1.0 - v))          # diagnosticity on evidence scale
    logit = np.maximum(logit, 1e-9)        # strictly positive

    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 7.0))
    epsilon = float(parameters.get("epsilon", 0.0))
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    lam = float(parameters.get("lam", 0.0))
    lam = float(np.clip(lam, 0.0, 1.0))

    with np.errstate(over="ignore", invalid="ignore"):
        w = np.power(logit, gamma)
    w = np.nan_to_num(w, nan=0.0, posinf=np.finfo(float).max / (10.0 * n), neginf=0.0)

    # ---- positional misattribution: blend with the left-to-right default
    # The default binds the largest importance to the first displayed cue,
    # the next largest to the second, etc.  When the experiment already
    # orders columns by descending validity this blend is the identity.
    if n > 1 and lam > 0.0:
        w_pos = np.sort(w)[::-1]
        w = (1.0 - lam) * w + lam * w_pos

    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)      # degenerate -> equal weights
    else:
        w = w / s

    # ---- weighted additive evidence margin -----------------------------
    d = float(np.dot(w, a - b))            # in [-1, 1]

    # numerically stable two-alternative softmax == logistic on beta*d
    z = np.array([beta * d, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - epsilon) * p_core + epsilon * 0.5
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
    if not np.isfinite(tot) or tot <= 0:
        p = np.ones_like(p) / len(p)
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Goal: dissociate Take-The-Best (one-reason, cue-order-driven, magnitude-blind) from Tallying (equal-weight vote count, margin-graded) both in *direction* and in *graded magnitude* of choice probabilities.

Five experts with clearly spread validities (0.92, 0.80, 0.70, 0.60, 0.53) so the cue cascade is unambiguous and subjects are told expert 1 is much better than experts 3-5.

The 16 unique pairs (each repeated 6x, 96 trials) fall into five diagnostic families, each mirrored so that neither response key is systematically correct:

(1) DIRECTIONAL CONFLICT WITH GRADED TALLY MARGIN. Pairs 1-6: the option winning the top (92%) cue loses on all remaining cues, with the tally margin varied at 3 (1 vs 4 wins), 2 (1 vs 3 wins, one tie) and 1 (2 vs 3 wins). TTB predicts the SAME high probability of choosing the top-cue option in all three, independent of margin; Tallying predicts the OPPOSITE option with a probability that increases monotonically with margin (sigmoid(beta*margin)). This is the core quantitative dissociation: a flat-versus-graded profile across margin, plus opposite signs.

(2) TALLY-TIE / TTB-DISCRIMINATES. Pairs 7-8: A=[1,0,1,0,0] vs B=[0,1,0,1,0]. Feature wins are 2-2, so Tallying must guess (exactly 50/50 regardless of beta), while TTB is driven deterministically by the 92% cue. Any systematic departure from 50% here is fatal to Tallying.

(3) TTB-STRONG / TALLY-WEAK ON A SINGLE LOW CUE. Pairs 9-10: the options tie on the four top cues and differ only on the 53% cue. TTB is fully committed (its stopping rule gives the winner probability sigmoid(beta), the same as for a top-cue win), whereas Tallying only has a margin of 1, its weakest signal. Comparing accuracy here against family (5) below separates the two models' beta scalings.

(4) DEEP-CUE CASCADE CONFLICT. Pairs 13-14: options tie on the 92% cue, the 80% cue favors one option, and the three lower cues all favor the other. TTB stops at cue 2 and chooses against a 3-vs-1 tally. This checks that the cascade (not just the single best cue) governs choice.

(5) AGREEMENT ANCHORS. Pairs 11-12 (4 wins vs 1 win, same direction for both models) and 15-16 (only the top cue discriminates, all else tied) provide low-conflict baselines that pin down epsilon and beta for both models so the conflict trials are interpreted against a calibrated noise level.

Predicted pattern if TTB is right: near-ceiling choice of the top-cue option in families 1-4 with NO modulation by tally margin, and above-chance choice in the tally-tie pairs. If Tallying is right: choice reverses in family 1, scales with margin (m=1 < m=2 < m=3), is at chance in family 7-8, and is weak in pairs 9-10.

[1] GOAL: dissociate Tallying (advocated) from Take-The-Best (competing) by exploiting the fact that TTB's choice probability is *constant* (sigmoid(beta)) whenever ANY cue discriminates, whereas Tallying's is a graded function of the vote margin, sigmoid(beta*m). The previously explored design put the gradation manipulation only inside directional-conflict trials; here the core diagnostic is a *no-conflict* margin ladder, which cleanly separates the two models on magnitude alone (no direction confound), and it is cross-validated against conflict and tally-tie families that separate them on direction.

Six experts with widely spread, deliberately non-monotonic validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65. TTB's cue cascade is therefore f0 > f2 > f4 > f5 > f1 > f3, which is NOT the display order, so cue-identity effects cannot be mimicked by a simple left-to-right reading strategy. 16 unique pairs, each repeated 6x (96 trials), fully key-balanced (every pair appears with A and B swapped).

FAMILY C 	rairairs 1-8): MARGIN LADDER WITHOUT CONFLICT. The 93% cue always favors the same option that wins the tally, and the tally margin takes the values m = 1, 2, 3, 4 (each in both key directions). TTB predicts EXACTLY the same accuracy at m=1 as at m=4 (flat line, sigmoid(beta) after lapse). Tallying predicts a steeply rising curve, e.g. at beta=0.6 roughly .65 / .77 / .86 / .92. This is the central quantitative signature: a flat-versus-monotone accuracy profile over margin with the direction of choice held constant, so no directional confound can explain it.

FAMILY D (pairs 9-12): TALLY TIES, TTB DISCRIMINATES. Pairs 9-10 are 2-2 splits where the 93% cue discriminates; pairs 11-12 are 2-2 splits where f0 and f2 TIE and the first discriminating cue is the 74% expert (with the 65%, 58% and 51% experts split against it). Tallying must be exactly 50/50 in all four, irrespective of beta and of which cue is on top. TTB predicts systematic above-chance choice of the top-discriminating-cue option, and — importantly — the SAME magnitude for the f0-driven and f4-driven pairs. Any deviation from chance here falsifies Tallying; equality of the f0- and f4-driven deviations is a further TTB-specific prediction.

FAMILY E (pairs 13-16): GRADED DIRECTIONAL CONFLICT. The 93% cue favors one option while the tally favors the other, with tally margin 4 (pair 13/14: 5 wins vs 1) or 2 (pair 15/16: 3 wins vs 1). TTB predicts the same probability of choosing the f0-winner in both, and this probability must equal its Family-C and Family-D accuracy (all are sigmoid(beta)). Tallying predicts the opposite option, with a much larger effect at m=4 than at m=2, and with beta already pinned down by the Family-C ladder — so the conflict trials are a genuine out-of-sample prediction rather than a free fit.

DECISION RULE: If Tallying is correct, accuracy in Family C rises monotonically with margin, Family D is at chance, and Family E reverses against the 93% expert more strongly at m=4 than m=2, with all effects captured by a single (beta, epsilon). If TTB is correct, one flat probability level sigmoid(beta) accounts for Families C, D (top-cue side) and E (top-cue side) simultaneously, with zero margin modulation anywhere.

[2] GOAL: dissociate Take-The-Best (advocated) from WADD-\u03b3 with Positional Validity Misattribution (competing). The competitor's central falsifiable commitment is that subjective weights are largely bound to *display columns* (importance assumed to fall left-to-right, mixing weight \u03bb \u2265 0.64), whereas TTB binds validities to experts correctly and consults cues in true validity order, stopping at the first discriminator. The blend is the identity only when columns are already sorted by descending validity, so I present the five experts in strictly ASCENDING validity order (col0=.55, col1=.62, col2=.70, col3=.85, col4=.93). This maximises the mismatch: the positional default assigns the LARGEST weight to col0 (the 55% expert, TTB's very last cue) and the SMALLEST to col4 (the 93% expert, TTB's first cue).\n\nEffective competitor weights (normalised) across its parameter range: at \u03bb=.64,\u03b3=1.15 -> [.31,.22,.13,.15,.19]; at \u03bb=.80,\u03b3=1.50 -> [.44,.25,.10,.10,.12]; at \u03bb=1.0,\u03b3=1.85 -> [.60,.29,.08,.03,.01]. In every case col0 > col1 > {col2,col3,col4}. TTB's order is exactly col4 > col3 > col2 > col1 > col0. This gives sign reversals that hold over the entire competitor parameter box.\n\n16 unique pairs (K=6, 96 trials), every family mirrored so neither key is systematically correct:\n\nFAMILY A (pairs 1-2, EXTREME POSITION CONFLICT): only col0 and col4 discriminate, in opposite directions. TTB decisively picks the 93%-cue option; the competitor picks the col0 option with |d| = .13 to .60 (p up to ~1 at high \u03b2). Opposite signs, robust to \u03bb and \u03b3.\n\nFAMILY B (pairs 3-4, MILD POSITION CONFLICT): only col1 (62%) and col3 (85%) discriminate. TTB again picks the higher-validity (col3) option with EXACTLY the same probability as in Family A (sigmoid(\u03b2) is invariant across all discriminating trials); the competitor again reverses but with a smaller margin (|d| = .06 to .26). The TTB-flat vs competitor-graded profile across A > B is the core quantitative signature.\n\nFAMILY C (pairs 5-6, NEAR-CANCELLING): col2 (70%) vs col3 (85%). Their positional weights are nearly identical, so the competitor is pinned near chance (|d| \u2264 .05, and its sign even flips within the parameter box), while TTB is fully committed to the col3 option at its usual sigmoid(\u03b2) level. Any systematic above-chance choice here is fatal for the competitor's positional binding.\n\nFAMILY D (pairs 7-8, DOMINANCE ANCHOR): all five cues favour one option. Both models agree and near ceiling; this pins \u03b2 and \u03b5 for both models so the conflict families are out-of-sample predictions rather than free fits.\n\nFAMILY E (pairs 9-10, TWO-VS-TWO POSITION CONFLICT): A wins the two leftmost/least valid cues (col0,col1), B wins the two rightmost/most valid cues (col3,col4). TTB \u2192 B via col4; competitor \u2192 A with a large margin (|d| = .19 to .57). This is the strongest directional test.\n\nFAMILY F (pairs 11-12, DEEP-CASCADE vs POSITION): the top two validity cues (col4, col3) tie; col2 favours one option and col0+col1 favour the other. TTB stops at col2 and chooses against a 2-vs-1 vote; the competitor chooses the col0+col1 option strongly. Tests that the cascade, not just the single best cue, governs TTB choice, while remaining maximally diagnostic against positional weighting.\n\nFAMILIES G and H (pairs 13-16, MAGNITUDE-ONLY CONTRAST WITH DIRECTION HELD CONSTANT): in G only col4 (93%) discriminates; in H only col0 (55%) discriminates. Both models pick the option with the single 1, so there is no directional confound. TTB predicts IDENTICAL accuracy in G and H (and identical to Families A-F and D, all sigmoid(\u03b2)). The competitor predicts H \u226b G (weights .44 vs .12 at \u03bb=.8,\u03b3=1.5; .60 vs .01 at \u03bb=1), i.e. subjects should be far more decisive when only the worst-but-leftmost expert speaks than when only the best-but-rightmost expert speaks. That inversion is impossible under TTB.\n\nDECISION RULE: If TTB is right, one flat probability level accounts for all eight families, with choices always tracking the higher-validity discriminating expert (col4 > col3 > col2 > col1 > col0) irrespective of column position, and no G-vs-H asymmetry. If the positional WADD is right, choices reverse in Families A, B, E and F, sit at chance in Family C, and show a large H > G accuracy gap with graded magnitudes ordered D > F > E > H > A > B > G > C.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate WADD-\\u03b3 with Positional Validity Misattribution (advocated) from Take-The-Best (competing). TTB binds stated validities to the correct expert and consults cues in true validity order, stopping at the first discriminator, so its choice probability is a single flat level sigmoid(beta) (plus lapse) on EVERY discriminating trial, always siding with the highest-validity discriminating expert. The advocated model claims that the validity->column binding is largely replaced by the display default 'importance falls left to right' (lam >= 0.64), and that the resulting weighted margin d produces GRADED choice probabilities.\n\nDISPLAY ORDER (deliberately scrambled, not merely ascending): col0=.60, col1=.93, col2=.55, col3=.86, col4=.68, col5=.78. TTB's cascade is col1 > col3 > col5 > col4 > col0 > col2. The positional default instead assigns the weight multiset in descending order to columns, so col0 (a 60% expert) receives the LARGEST subjective weight and col5 (a 78% expert) receives nearly none. Effective normalised weights over the whole competitor-free parameter box: lam=.64,gamma=1.15 -> [.271,.312,.119,.157,.065,.076]; lam=.80,gamma=1.50 -> [.374,.309,.128,.112,.037,.039]; lam=1.0,gamma=1.85 -> [.518,.269,.138,.053,.017,.005]. Across the whole box col0 and col1 dominate, col2 > col3 for most of it, and col4/col5 are near zero even though col5's true validity (.78) is third best.\n\n18 unique pairs (9 families, each mirrored so neither key is systematically correct), K=5 -> 90 trials. Shared 1s are added as background so no option is trivially empty.\n\nFAMILY A (pairs 1-2, col0 .60 vs col3 .86 discriminate in opposite directions): TTB -> the col3 option; WADD-pos -> the col0 option, |d| ~ .26-.47. Sign reversal robust over the entire parameter box.\nFAMILY B (pairs 3-4, col0 vs col5 .78): TTB -> col5 option with EXACTLY the same probability as in Family A; WADD-pos -> the col0 option again but with a LARGER margin (|d| ~ .34-.51). A>B ordering with direction held constant is the core flat-vs-graded signature.\nFAMILY C (pairs 5-6, col2 .55 vs col4 .68): TTB -> the col4 option (col4 precedes col2 in the true cascade); WADD-pos -> the col2 option but only weakly (|d| ~ .05-.12). Reversal with a small, gradable margin.\nFAMILY D (pairs 7-8, col0 .60 vs col1 .93): TTB is decisive for the col1 option. WADD-pos is near-cancelling and its sign flips inside the parameter box (col1 wins at lam=.64/gamma=1.15, col0 wins at lam>=.8). This is the lam-identifying family and, per the heterogeneity claim, should show the LARGEST between-subject spread of any family.\nFAMILY E (pairs 9-10, coalition): one option wins col1 (the true 93% expert) alone, the other wins col0 + col5. TTB stops at col1; WADD-pos reverses (B wins for every (lam,gamma) in the box, though with a modest margin .04-.25).\nFAMILY F (pairs 11-12, deep cascade vs position): col1 ties, col3 (.86) favours A while col0 + col2 favour B. TTB stops at col3 and picks A against a 2-vs-1 vote; WADD-pos picks B strongly (|d| ~ .23-.42). Strongest directional test that also verifies TTB uses the cascade, not just the single best cue.\nFAMILIES G and H (pairs 13-16, MAGNITUDE ONLY, NO DIRECTIONAL CONFOUND): in G only col1 (93%) discriminates; in H only col5 (78%) discriminates. Both models favour the same option in both, so TTB must produce IDENTICAL accuracy in G and H (and identical to A-F). WADD-pos predicts G (|d| ~ .27-.31) far above H (|d| ~ .005-.08, essentially chance): subjects should be near-indifferent when only a 78% expert speaks from the rightmost column.\nFAMILY I (pairs 17-18, full dominance): all six cues agree; both models near ceiling. This pins beta and epsilon for both models so every conflict family is an out-of-sample prediction rather than a free fit.\n\nDECISION RULE: If TTB is right, one flat probability level fits families A-I, choices always track the highest-validity discriminating column (col1 > col3 > col5 > col4 > col0 > col2), and G = H exactly. If positional WADD-\\u03b3 is right, choices REVERSE in A, B, C, E, F, sit near chance in D and H, and the graded ordering of |accuracy - .5| should be I > F > B > A > G > E > C > D > H, with the between-subject spread peaking in the intermediate-margin families (D, E, C) and collapsing in I and H.",
  "validities": [
    0.6,
    0.93,
    0.55,
    0.86,
    0.68,
    0.78
  ],
  "trial_a_ratings": [
    [
      1,
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
      1,
      0
    ],
    [
      1,
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
      1,
      1
    ],
    [
      0,
      1,
      1,
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
      1
    ],
    [
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
      0,
      1
    ],
    [
      0,
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
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
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
      0
    ],
    [
      0,
      1,
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
      0
    ],
    [
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
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
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
      0
    ],
    [
      0,
      1,
      0,
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
      0
    ],
    [
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
      1,
      1,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
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
      0
    ],
    [
      0,
      1,
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
      1
    ],
    [
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
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 11021,
  "output_tokens": 10717
}
```
