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
**Description:** **MDSC-NV (sharpened) — Maximum-Doubt Single-Cue selection with Null-Verdict capture.**

Four claims; the mechanism family is unchanged from the accepted base, but one quantitative commitment is sharpened.

**(1) One-reason choice by MAXIMUM DOUBT (no summation).** On each trial the subject settles A-vs-B with the verdict of a SINGLE expert. Which expert is consulted is decided by *doubt*: a disagreement voiced by a near-chance expert (v≈.53) is a strange, newsworthy fact that has to be adjudicated, whereas a disagreement voiced by a near-certain expert (v≈.97) is discounted as a mere echo of the impression the subject already has. Salience is log u_j = delta*log(1-v_j) + lam*j and the consulted row is drawn from softmax(u). **The doubt gain is LARGER than previously assumed (delta ≈ 13-17, not 8-12): the doubt rule is essentially argmax, not a graded share.** This is the sharpened commitment. It matters wherever two moderately-doubtful rows are within ~0.4 log-validity of one another (e.g. v=.55 vs v=.70 rows two columns apart): a share-based rule lets the second-most-doubtful row steal ~18% of the verdict and blunts the choice, whereas the near-argmax rule leaves the display fully decided by the single most doubtful row. Coalitions still buy nothing: five agreeing rows never outvote one more-doubtful row, which is why choice is systematically anti-validity in every design where the printed labels are usable.

**(2) Reading position is a WEAK tie-breaker.** lam = pos_ratio*delta with pos_ratio ≈ 0.09-0.14 — an order of magnitude below the doubt term's dynamic range. Position therefore only arbitrates between rows whose stated validities are (nearly) *equal* — exactly the .955/.950/.945 and .958/.954/.950 designs, where it makes the last low-validity row win. Whenever validities genuinely differ, doubt beats recency: a low-validity EARLY column defeats a late .93/.99 column. Because delta is raised and pos_ratio held fixed, the *ratio* of doubt-range to positional gain is preserved, so all orderings in the corpus are unchanged.

**(3) NULL-VERDICT CAPTURE.** Doubt is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the most doubtful expert **in the whole display**, before the subject checks whether that expert actually discriminates. With probability q (≈ .41, slightly higher than previously estimated) the subject simply goes with whoever captured attention; if that expert gave both products the same rating it returns no verdict, the subject has nothing to adjudicate and guesses. With probability 1-q the subject re-targets deliberately, restricting the doubt competition to the rows that differ. Sharp prediction: *holding the discriminating rows fixed, adding agreement on a MORE doubtful (lower-validity) row pushes choice toward chance*, whereas adding agreement on a high-validity row does nothing. Displays in which only two high/mid-validity rows disagree while the display's most doubtful expert stays silent are therefore near chance — no compensatory or one-reason model predicts that non-discriminating rows matter at all.

**(4) LABEL-DEGENERACY GATE (all-or-none) and LOAD LAPSE.** The doubt rule presupposes that each printed validity picks out one expert. If any label is printed twice (tolerance .002 — it must NOT fire for the .004/.005-gap designs) the experts are interchangeable, the doubt ordering is unavailable, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). Independently, the more rows disagree the more often the subject loses the thread: p(guess) = 1-(1-eps0)*rho^(m-2), which compresses conflict displays toward chance and keeps even unanimous 6-row displays well below ceiling. Individual subjects differ mainly in how leaky (eps0, rho) and how capture-prone (q) they are, not in the doubt rule itself.

No feedback is given, so the rule is stationary across the block.

**Parameters:**
- doubt_gain: [13.0, 17.0]
- pos_ratio: [0.09, 0.14]
- capture: [0.33, 0.49]
- eps0: [0.19, 0.29]
- rho: [0.77, 0.85]
- tally_kappa: [1.00, 1.40]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # MDSC-NV: Maximum-Doubt Single-Cue selection with Null-Verdict capture
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed labels) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9975)

    # ---------- parameters ----------
    delta = float(parameters['doubt_gain'])
    lam = float(parameters['pos_ratio']) * delta        # weak positional gain
    q = float(np.clip(float(parameters['capture']), 0.0, 1.0))
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    kap = float(parameters['tally_kappa'])

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    fav_a = diff[disc] > 0.0
    n_a = float(np.sum(fav_a))
    n_b = float(m) - n_a

    # ---------- label-degeneracy gate (all-or-none, exact repetition) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        # experts cannot be individuated -> unweighted graded row tally
        z = float(np.clip(kap * (n_a - n_b), -60.0, 60.0))
        p_core_a = 1.0 / (1.0 + np.exp(-z))
    else:
        # doubt salience over ALL rows
        logu = delta * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)

        lg = logu - np.max(logu)
        s = np.exp(lg)
        ssum = float(np.sum(s))
        s = s / ssum if (np.isfinite(ssum) and ssum > 0.0) else np.full(n, 1.0 / n)

        ld = logu[disc]
        ld = ld - np.max(ld)
        t = np.exp(ld)
        tsum = float(np.sum(t))
        t = t / tsum if (np.isfinite(tsum) and tsum > 0.0) else np.full(m, 1.0 / m)

        # (i) deliberate re-targeting: doubt competition among disagreeing rows
        p_rest_a = float(np.sum(t[fav_a]))

        # (ii) null-verdict capture: attention grabbed by the most doubtful
        #      expert in the whole display; if it is silent -> guess
        mask = np.zeros(n, dtype=bool)
        mask[disc] = True
        s_nondisc = float(np.sum(s[~mask]))
        s_disc = s[disc]
        p_glob_a = float(np.sum(s_disc[fav_a])) + 0.5 * s_nondisc

        p_core_a = q * p_glob_a + (1.0 - q) * p_rest_a

    # ---------- load-dependent lapse ----------
    expo = max(m - 2, 0)
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
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
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** SSI-DS+ (Strategy Selection by Label-Code Injectivity, with Distinctive-Cue Summation and Competition-Scaled Interference).

(1) STRATEGY SWITCH. Before the block the subject reads the printed validity list as a naming code. If all labels are distinct printed symbols (0.955 vs 0.950 counts as distinct) the experts are individuable and the subject adopts a WEIGHTED mode; if any label is repeated the experts are interchangeable and the subject adopts an unweighted MAJORITY TALLY mode. The switch is all-or-none at the design level (only a small per-trial slip probability omega), not a graded function of numeric gap size.

(2) WEIGHTED MODE = COMPENSATORY SUMMATION over all discriminating rows: E = sum_j sign(a_j-b_j) * (1-v_j)^gamma * exp(lam*j), rescaled so the strongest discriminating row has magnitude 1, then p(A) = sigmoid(beta*E). Validity enters with INVERTED sign (near-certain experts are redundant echoes of the general impression; near-chance disagreements are the news that must be adjudicated) and position enters as a reading-order recency gain. Coalitions ADD, so several early distinctive rows can outvote a single later row.

(3) TALLY MODE = graded majority p(A) = sigmoid(kappa*(n_A - n_B)): 1-1 splits give exactly 0.50, 2-1 splits ~0.77 (harder than probability matching, softer than a hard majority rule).

(4) CONFLICT INTERFERENCE IS COMPETITION-SCALED (the refinement). Guessing is driven by the load of holding several discrepancies in mind while ADJUDICATING BETWEEN THEM. The load is therefore not the raw number of discriminating rows but the number of rows that actually have to be pitted against one another: when the display is unanimous (min(n_A,n_B) = 0, i.e. every discrepancy points the same way) there is nothing to adjudicate and the extra rows impose only a discounted reading load eta*(m-2) rather than the full (m-2). Formally g = 1 - (1-eps0) * rho^L with L = (m-2) on opposed displays and L = eta*(m-2) on unanimous displays, and the core probability is mixed toward 0.5 by g. This predicts that dominance accuracy degrades with the number of agreeing experts only mildly, whereas conflict displays degrade steeply with the number of opposed experts - the two are dissociated, not a single monotone function of display size.

No feedback, hence no learning; the rule is stationary across the block.

**Parameters:**
- gamma: [0.86, 0.94]
- lam: [1.45, 1.55]
- beta: [3.2, 4.0]
- kappa: [1.10, 1.30]
- eps0: [0.22, 0.26]
- rho: [0.755, 0.80]
- omega: [0.0, 0.04]
- eta: [0.62, 0.84]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            flat = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = flat[0].ravel(), flat[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed label code) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    beta = float(parameters['beta'])
    kappa = float(parameters['kappa'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    omega = float(np.clip(float(parameters['omega']), 0.0, 0.5))
    eta = float(np.clip(float(parameters['eta']), 0.0, 1.0))

    # ---------- STRATEGY SWITCH: injectivity of the printed label code ----------
    LABEL_TOL = 2.5e-3   # two printed labels closer than this are the same symbol
    if n <= 1:
        injective = True
    else:
        vs = np.sort(v)
        gaps = np.diff(vs)
        injective = bool(np.all(gaps > LABEL_TOL))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])

    # ---------- weighted mode: distinctive-cue SUMMATION ----------
    # log weight = gamma*log(1 - v_j) + lam*j ; rescaled so strongest row = 1
    logw = gamma * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + lam * disc.astype(float)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    E = float(np.sum(signs * w))
    z = float(np.clip(beta * E, -60.0, 60.0))
    p_weighted_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- tally mode: graded majority ----------
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    zt = float(np.clip(kappa * (n_a - n_b), -60.0, 60.0))
    p_tally_a = 1.0 / (1.0 + np.exp(-zt))

    if injective:
        p_core_a = (1.0 - omega) * p_weighted_a + omega * p_tally_a
    else:
        p_core_a = (1.0 - omega) * p_tally_a + omega * p_weighted_a

    # ---------- conflict interference / lapse (competition-scaled load) ----------
    # load counts only rows that must be ADJUDICATED against one another:
    # unanimous (dominance) displays impose only a discounted reading load.
    conflict = float(min(n_a, n_b))
    expo = float(max(m - 2, 0))
    if conflict <= 0.0:
        expo = eta * expo
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
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
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with clearly spread, strictly ordered validities (0.95, 0.85, 0.75, 0.65, 0.55) so the TTB cue cascade is unambiguous and matches feature index order. The 12 unique pairs (repeated 8x each = 96 trials) are built to dissociate one-reason decision making from unit-weight tallying along three quantitative dimensions:

(1) CONFLICT MAGNITUDE. Pairs 1-4 pit the single highest-validity cue against a coalition of lower cues, varying the tally margin from 4 (A=[1,0,0,0,0] vs B=[0,1,1,1,1]) to 1 (A=[1,0,0,0,0] vs B=[0,1,1,0,0]). TTB predicts identical choice probabilities across these pairs (always the option winning cue 1, with p fixed by beta and epsilon), whereas Tallying predicts an ordered gradient: near-deterministic choice of the coalition option at margin 4 and a shallower, beta-scaled preference at margin 1. This is a quantitative signature, not merely a directional one: TTB is flat over margins, Tallying is monotone increasing.

(2) TALLY-TIE TRIALS. Pairs 5-6 (A=[1,0,1,0,0] vs B=[0,1,0,1,0] and its mirror) give each option exactly two feature wins, so Tallying must guess at exactly 50/50 regardless of its parameters, while TTB is fully decisive in favour of the option winning cue 1. Any systematic deviation from chance here cannot be produced by Tallying at all, making these trials a parameter-free falsification test.

(3) CASCADE POSITION. Pairs 9-11 place the first discriminating cue lower in the cascade (cue 2 after a tie on cue 1; cue 3 after ties on cues 1-2) while the remaining lower cues favour the other option. TTB still predicts the same high, constant probability for the option winning the first discriminating cue; Tallying predicts the opposite choice. If subjects' accuracy on the discriminating-cue option degrades as that cue moves down the cascade, that also constrains the subjective-validity assumption inside TTB.

Pairs 7-8 are agreement controls where both heuristics select the same option, giving a baseline for lapse rate (epsilon) and beta that is uncontaminated by the model contrast. Every pair and its mirror image appear so that the correct response is balanced across the A and B keys for both models, ruling out side bias as an explanation. No pair is degenerate: on every trial at least one model is decisive, and on 10 of 12 the models disagree either in direction or in predicted confidence.

[1] GOAL: dissociate one-reason decision making (TTB) from unit-weight vote counting (Tallying) using (i) a validity order that is deliberately NOT the display order, (ii) a tally-margin gradient over which TTB is flat, (iii) parameter-free tally-tie trials, and (iv) cascade-depth manipulations.

VALIDITIES: six experts shown in display order with validities [0.60, 0.93, 0.55, 0.87, 0.70, 0.78]. The descending cue cascade is therefore feature index 1 (.93) > 3 (.87) > 5 (.78) > 4 (.70) > 0 (.60) > 2 (.55). Because the top cue is Expert 2 rather than Expert 1, any subject who follows a genuine validity cascade must reorder the display, whereas Tallying is completely invariant to the permutation. This also guards against a confound present when validities descend with index: 'use the leftmost cue' can no longer masquerade as TTB.

16 unique pairs, each repeated 6x (96 trials), every pair present together with its A/B mirror so the model-predicted response is exactly balanced across keys (no side-bias explanation).

(1) TALLY-MARGIN GRADIENT (pairs 1-6). One option wins only the top cue (index 1); the other wins a coalition of lower-validity cues, with the tally margin escalating 1 -> 2 -> 4: [0,1,0,0,0,0] vs [1,0,1,0,0,0], vs [1,0,1,0,1,0], vs [1,0,1,1,1,1]. TTB is decisive for the top-cue option with the SAME probability sigmoid(beta)-scaled value on all three, i.e. flat across margins. Tallying reverses direction (choosing the coalition option) and its confidence increases monotonically with margin, from a shallow beta-scaled preference at margin 1 to near-determinism at margin 4. Flat-vs-monotone over margin is a quantitative, not merely directional, signature.

(2) PARAMETER-FREE TALLY TIES (pairs 7-10). Pair 7: [0,1,0,0,0,0] vs [1,0,0,0,0,0] — each option wins exactly one cue, so Tallying MUST be exactly 50/50 for any beta and epsilon, while TTB is decisive for A (cue 1, .93, beats cue 0, .60). Note the naive 'read left to right' strategy predicts the opposite option here, so this pair simultaneously falsifies Tallying and index-order pseudo-TTB. Pair 9: [0,0,1,1,0,0] vs [0,0,0,0,1,1] — a 2-2 tally tie in which TTB's winner is set by a mid-cascade comparison (cue 3, .87, vs cues 5/4), again forcing Tallying to chance while TTB is decisive for A.

(3) CASCADE DEPTH WITH CONSTANT TALLY (pairs 11-14). Pair 11 ([0,0,0,0,0,1] vs [1,0,1,0,1,0]) has the identical tally structure to pair 3 (1 vs 3) but the first discriminating cue sits third in the cascade (index 5) after two ties; pair 13 ([0,0,0,0,1,0] vs [1,0,1,0,0,0]) matches pair 1's margin of 1 but the discriminator is fourth in the cascade (index 4). Tallying predicts exactly the same probabilities as its margin-matched counterparts, while TTB predicts the same high probability for the discriminating-cue option regardless of depth. Any systematic decay in choosing the discriminating-cue option as it moves down the cascade would violate both models as specified and constrain subjective validities inside TTB.

(4) AGREEMENT BASELINE (pairs 15-16). [1,1,1,0,1,0] vs [0,0,0,1,0,1]: the top cue and the tally (4-2) point the same way, so both models predict the same option. These trials identify epsilon and beta free of the model contrast, so the contrast trials can be read as genuine model divergence rather than overall noise.

No pair is degenerate: on 14 of 16 pairs the models disagree in direction, and on the remaining agreement pairs they differ in nothing but supply the noise baseline.

[2] GOAL: dissociate Amplified-Validity Additive Integration (AVAI) from unit-weight Tallying. Both models are permutation-invariant over cue position, so the crucial contrasts are (i) trials where the WIN COUNT is tied but the validity-weighted margin varies parametrically, and (ii) trials where the win count and the weighted sum point in OPPOSITE directions or where a large count margin corresponds to a near-zero weighted margin.

VALIDITIES: six experts in a deliberately non-monotone display order, [0.68, 0.96, 0.52, 0.88, 0.58, 0.78]. Ranked: H1 = idx1 (.96), H2 = idx3 (.88), M = idx5 (.78), L1 = idx0 (.68), L2 = idx4 (.58), L3 = idx2 (.52). Scrambling display order removes any 'read leftmost cue' artifact. With gamma = 2 the AVAI weights are H1=.922, H2=.774, M=.608, L1=.462, L2=.336, L3=.270 \u2014 all the same order of magnitude, exactly the compressed regime AVAI posits.

16 unique pairs (8 base structures, each with its A/B mirror so the predicted response is exactly key-balanced for both models), 6 repetitions each = 96 trials.

(1) TALLY-TIE LADDER (pairs 1-4, plus mirrors). Every one of these is a 2-2 win tie, so Tallying is EXACTLY 50/50 for any beta and epsilon ">AVAI, in contrast, predicts a graded, monotone ladder of choice probabilities because the logistic acts on the continuous weighted difference:
  \u2022 Pair 1 {H1,H2} vs {L2,L3}: d = +1.09 \u2192 p(A) \u2248 .99 (near-deterministic).
  \u2022 Pair 2 {H1,L1} vs {M,L3}: d = +0.51 \u2192 p(A) \u2248 .95.
  \u2022 Pair 3 {H2,L1} vs {M,L2}: d = +0.29 \u2192 p(A) \u2248 .85.
  \u2022 Pair 4 {H1,L3} vs {H2,L1}: d = -0.045 \u2192 p(A) \u2248 .47 (indistinguishable from chance).
A flat 50% profile across all four falsifies AVAI and confirms Tallying; a monotone spread from ~.99 down to ~.50 with the SAME tally structure cannot be produced by Tallying under any parameter setting. Pair 4 additionally pins gamma: the sign of d for {H1,L3} vs {H2,L1} flips from B-favoring to A-favoring near gamma \u2248 2.7, so it is the most gamma-diagnostic trial in the set.

(2) COUNT-VS-WEIGHT CONFLICT LADDER (pairs 5-7, plus mirrors). One option holds only the single best cue (or the two best); the other holds a coalition of weak cues, with the coalition's tally margin escalating 1 \u2192 2 \u2192 3 while the weighted margin sweeps through zero:
  \u2022 Pair 5: A={H1} vs B={L2,L3}. Tallying: B wins by 1 (p(A) < .5, beta-scaled). AVAI: d = +0.31 \u2192 p(A) \u2248 .87. DIRECTIONAL REVERSAL \u2014 the two models predict opposite modal responses.
  \u2022 Pair 6: A={H1,H2} vs B={M,L1,L2,L3}. Tallying: B wins by 2, i.e. MORE confident than pair 5. AVAI: d = +0.02 \u2192 p(A) \u2248 .50, i.e. maximally noisy. A large count margin producing chance-level behaviour is impossible for Tallying and is the signature 'big count margin, small weighted margin' prediction of AVAI.
  \u2022 Pair 7: A={H1} vs B={L1,L2,L3}. Tallying: B wins by 3 (near-deterministic). AVAI: d = -0.15 \u2192 p(A) \u2248 .29 (same direction, much weaker). Across pairs 5-7 Tallying predicts a monotone increase in coalition choice (.5+ \u2192 strong \u2192 near-1) driven purely by margin, whereas AVAI predicts the coalition being chosen at rates .13, .50, .71 \u2014 crossing chance at margin 2 rather than at margin 0. Flat-margin-crossing location is a quantitative, parameter-robust separator.

(3) AGREEMENT BASELINE (pair 8, plus mirror). A={H1,H2,L1,M} vs B={L2,L3}: tally 4-2 and weighted margin both strongly favour A. Both models predict the same near-deterministic response, which identifies epsilon (lapse rate) and the overall beta scale independently of the contrast trials, so deviations on pairs 1-7 can be attributed to model structure rather than global noise.

No pair is degenerate: on 6 of the 8 base structures the models disagree either in direction or by more than 30 percentage points in predicted probability, and the four tie trials constitute a parameter-free test that Tallying can only pass by producing exactly 50% on all of them.

[3] GOAL: dissociate unit-weight Tallying (advocated) from Amplified-Validity Additive Integration (AVAI). Both models are permutation-invariant over display position, so the decisive manipulation is to hold the WIN COUNT constant while sweeping the validity-weighted margin across its whole range, and then to break the rank-order between count margin and weighted margin.

VALIDITIES: six experts in deliberately scrambled display order, [0.58, 0.97, 0.53, 0.80, 0.66, 0.90]. Ranked: H1 = idx1 (.97), H2 = idx5 (.90), M = idx3 (.80), L1 = idx4 (.66), L2 = idx0 (.58), L3 = idx2 (.53). Scrambling removes any 'read the leftmost cue' artifact and prevents position from proxying validity. AVAI weights at gamma = 2: H1=.941, H2=.810, M=.640, L1=.436, L2=.336, L3=.281 (the compressed regime AVAI assumes). Cues that are irrelevant on a trial are set to 1 for BOTH options, so every display is dense and requires actual comparison rather than counting 1s.

8 base structures, each accompanied by its exact A/B mirror (16 unique pairs, 6 repetitions = 96 trials), so the predicted response of BOTH models is exactly key-balanced and side bias cannot mimic either model.

(1) CONSTANT-COUNT-MARGIN LADDER (bases 1-4, plus mirrors). All four are +1 win-count margins for A, so Tallying must produce the SAME p(A) = sigmoid(beta) on every one of them, for any beta and epsilon. AVAI's weighted difference, in contrast, sweeps the entire probability range:
  - Base 1: A wins {H1} only. d = +0.94 -> AVAI p(A) ~ .998.
  - Base 2: A wins {H1,H2}, B wins {L2}. d = +1.42 -> AVAI p(A) ~ 1.00.
  - Base 3: A wins {L2,L3}, B wins {H1}. d = -0.32 -> AVAI p(A) ~ .11 (DIRECTIONAL REVERSAL relative to Tallying).
  - Base 4: A wins {L1,L2,L3}, B wins {H1,H2}. d = -0.70 -> AVAI p(A) ~ .01 (near-deterministic reversal).
A flat profile across these four (all equal, all above .5) is a signature Tallying alone can produce; a monotone sweep from ~1.00 down to ~.01 at constant count margin is impossible for Tallying under any parameters. This is the crown-jewel contrast: flat vs. full-range sweep.

(2) RANK-ORDER VIOLATION (bases 5-6, plus mirrors). Both are +2 count margins for A, so Tallying predicts them equal AND strictly more extreme than every margin-1 pair in block (1).
  - Base 5: A wins {M,L1,L2,L3}, B wins {H1,H2}. d = -0.06 -> AVAI p(A) ~ .41, i.e. essentially chance despite the LARGEST count margin in the design.
  - Base 6: A wins {H1,H2}, B wins nothing. d = +1.75 -> AVAI p(A) ~ 1.00.
AVAI therefore predicts that a margin-2 trial (base 5) is noisier than a margin-1 trial (base 1 or 2) — a rank-order violation Tallying strictly forbids. Base 6 doubles as an agreement baseline that identifies epsilon and the global beta scale free of the model contrast.

(3) TALLY TIES (bases 7-8, plus mirrors). Each option wins exactly the same number of cues, so Tallying is EXACTLY 50/50 for any beta and epsilon — a parameter-free prediction.
  - Base 7: A wins {H1}, B wins {L3}. d = +0.66 -> AVAI p(A) ~ .99.
  - Base 8: A wins {H1,L3}, B wins {H2,L1}. d = -0.02 -> AVAI p(A) ~ .46 (also near chance).
If subjects are at chance on base 7 as well as base 8, AVAI is falsified while Tallying is confirmed with no free parameters; if base 7 is near-deterministic while base 8 is at chance, that graded split is exactly AVAI's magnitude-sensitivity claim and cannot be produced by Tallying. Base 8 is additionally gamma-diagnostic: the sign of d for {H1,L3} vs {H2,L1} is near zero and flips with gamma, so it pins the amplification exponent if AVAI wins.

No pair is degenerate: on 5 of the 8 base structures the models disagree in direction or by more than 40 percentage points, the two tie trials are parameter-free tests of Tallying, and the remaining structures supply the noise/lapse baseline.

[4] GOAL: dissociate Amplified-Validity Additive Integration (AVAI, advocated) from Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS). The two models differ on three orthogonal axes: (i) POSITION-INVARIANCE \u2014 AVAI depends only on the validity-weighted difference, RDS is dominated by exp(lam*j) recency, so column position of the deciding row is everything; (ii) SIGN OF THE VALIDITY EFFECT \u2014 AVAI weights cues by v^gamma (high validity wins), RDS discounts high validity as 'redundant' via (1-v)^delta (low validity wins); (iii) EFFECT OF THE NUMBER OF DISCRIMINATING ROWS \u2014 AVAI's confidence GROWS with more agreeing cues, RDS's conflict-interference term drives choices toward chance as |D| grows.

VALIDITIES: a deliberately PALINDROMIC vector [0.93, 0.57, 0.75, 0.75, 0.57, 0.93]. This is the design's key trick: every validity level appears twice, once early and once late in the display. Validity content can therefore be held EXACTLY constant while column position is manipulated. Under AVAI (weights at gamma=2: .865, .325, .563, .563, .325, .865) any pair in which A wins column j and B wins the mirror column 5-j has weighted difference EXACTLY ZERO, so AVAI must predict exactly 50/50 for ANY gamma, beta and epsilon \u2014 a parameter-free prediction. Under RDS the same trials are strongly lopsided toward the later column (u_j = e^{1.6 j}(1-v_j)^{0.9}: u = [0.09, 2.32, 7.05, 34.9, 281.5, 272.4]).

18 unique pairs (9 base structures, each with its exact A/B mirror so both models' predicted responses are perfectly key-balanced), K = 5 repetitions = 90 trials. Non-discriminating rows are filled with matched 1s (and some matched 0s) so displays are dense and cannot be solved by counting 1s.

BLOCK 1 \u2014 PARAMETER-FREE MIRROR TIES (bases 1, 2, 5; pairs 1-4 and 9-10).
\u2022 Base 1: A wins col0 (.93), B wins col5 (.93). AVAI = .500 exactly. RDS \u2248 .14 (the late .93 row captures the decision, m=2 so interference is minimal).
\u2022 Base 2: A wins col2 (.75), B wins col3 (.75). AVAI = .500 exactly. RDS \u2248 .26 (adjacent columns \u2192 only one factor of e^lam).
\u2022 Base 5: A wins cols 4,5; B wins cols 0,1. AVAI = .500 exactly. RDS \u2248 .72 (both late rows favour A; m=4 pulls it back toward chance).
A flat 50% profile across these three, with the mirrors also at 50%, is uniquely AVAI. RDS is forced to produce an ordered spread (~.14 < ~.26 < ~.72) whose ordering is set purely by column distance and |D|.

BLOCK 2 \u2014 POSITION x VALIDITY INTERACTION (bases 3, 4; pairs 5-8). Both bases contain the identical validity contrast (.93 vs .57), only relocated.
\u2022 Base 3: A wins col0 (.93), B wins col1 (.57). AVAI p(A) \u2248 .96; RDS p(A) \u2248 .17 (DIRECTIONAL REVERSAL, both position and distinctiveness push to B).
\u2022 Base 4: A wins col5 (.93), B wins col4 (.57). AVAI p(A) \u2248 .96 \u2014 EXACTLY the same as base 3; RDS \u2248 .49 (recency of col5 and distinctiveness of col4 nearly cancel).
AVAI predicts base3 = base4; RDS predicts a ~32-point gap between two trials with identical validity content. This is the crown-jewel quantitative dissociation.

BLOCK 3 \u2014 WEIGHTED MARGIN VS LATE-CUE COALITION (base 6; pairs 11-12). A wins cols 0 and 5 (.93,.93), B wins cols 3 and 4 (.75,.57). AVAI d = +0.84 \u2192 p(A) \u2248 .99. RDS \u2248 .48 (the two late B rows roughly balance the late A row, and m=4 interference flattens it). Near-determinism vs chance on the same display.

BLOCK 4 \u2014 CONFLICT-INTERFERENCE GRADIENT (bases 7, 8, 9; pairs 13-18). All discriminating rows favour the same option, with |D| = 2, 4, 6.
\u2022 |D|=2 (cols 0,1): AVAI \u2248 1.00, RDS \u2248 .86.
\u2022 |D|=4 (cols 0,2,3,5): AVAI \u2248 1.00, RDS \u2248 .72.
\u2022 |D|=6 (all rows): AVAI \u2248 1.00, RDS \u2248 .63.
AVAI predicts accuracy that is flat-at-ceiling or weakly INCREASING with the number of agreeing cues (the weighted difference grows monotonically); RDS predicts a monotone DECREASE, because difficulty is driven by the count of differences rather than their diagnosticity. Observing >95% correct on a unanimous 6-0 trial is essentially impossible for RDS (its ceiling there is ~.63 given eps_base and rho ranges), while a drop to ~.63 on unanimous trials is impossible for AVAI. These trials also double as the lapse/beta baseline for AVAI.

No pair is degenerate: on 6 of the 9 bases the models disagree in direction or by more than 30 percentage points, and the three mirror-tie bases constitute a parameter-free test that AVAI can only pass by sitting exactly at chance.

[5] GOAL: dissociate Recency-Anchored Distinctive-Cue Sampling with Conflict Interference (RDS, advocated) from Amplified-Validity Additive Integration (AVAI, competitor). The two models differ on three orthogonal axes: (i) DIRECTION OF THE VALIDITY EFFECT \u2014 AVAI weights each row by v^gamma so the option holding the more valid experts wins, whereas RDS discounts high validity as redundant via (1-v)^delta so the option holding the LESS valid discriminating expert wins; (ii) COLUMN POSITION \u2014 AVAI is completely permutation-invariant, RDS is dominated by exp(lam*j) recency, so the later of two discriminating rows almost always captures the decision; (iii) NUMBER OF DISCRIMINATING ROWS \u2014 AVAI's confidence grows monotonically as more cues agree, RDS's conflict-interference term 1-(1-eps0)rho^(m-2) drags choices toward chance as |D| grows.

VALIDITIES: a REPEATED-BLOCK vector [0.95, 0.70, 0.55, 0.95, 0.70, 0.55] (deliberately not the palindrome used in an earlier design). Every validity level occurs exactly twice \u2014 once in the early block (cols 0-2) and once in the late block (cols 3-5) \u2014 so validity content can be held EXACTLY constant while the position of the high vs. low expert is swapped. Under AVAI (weights at gamma=2: .903, .490, .303, .903, .490, .303) any pair in which A wins column j and B wins column j+3 has weighted difference EXACTLY ZERO, i.e. p(A)=.500 for ANY gamma, beta, epsilon \u2014 a parameter-free prediction. Under RDS the same displays are strongly lopsided toward the LATER column (u_j = e^{1.6j}(1-v_j)^{0.9}: u = [0.067, 1.68, 12.0, 8.19, 204, 1460]).

8 base structures, each accompanied by its exact A/B mirror (16 unique pairs, K=6 \u2192 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows are filled with matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s.

BLOCK 1 \u2014 PARAMETER-FREE CROSS-BLOCK TIES (pairs 1-4).
\u2022 Base 1: A wins col0 (.95), B wins col3 (.95). AVAI = .500 exactly. RDS \u2248 .14 (the late .95 row wins the salience competition, m=2 so interference is minimal).
\u2022 Base 2: A wins col2 (.55), B wins col5 (.55). AVAI = .500 exactly. RDS \u2248 .15.
A flat 50% profile (with mirrors also at 50%) is uniquely AVAI; a strong pull toward the option holding the LATER identical-validity row is uniquely RDS and cannot be produced by any AVAI parameterisation.

BLOCK 2 \u2014 CROWN JEWEL: HIGH-EARLY vs LOW-LATE (pairs 5-8). Both bases contain the identical validity contrast (.95 vs .55), only relocated across the two blocks.
\u2022 Base 3: A wins col0 (.95), B wins col5 (.55). AVAI d=+0.60 \u2192 p(A) \u2248 .98. RDS \u2248 .14 \u2014 a near-total DIRECTIONAL REVERSAL (recency and distinctiveness both hand the decision to the late .55 row).
\u2022 Base 4: A wins col5 (.55), B wins col3 (.95). AVAI d=-0.60 \u2192 p(A) \u2248 .02. RDS \u2248 .86 \u2014 the mirror-image reversal.
AVAI says the .95-holder is chosen ~98% in both bases regardless of where the rows sit; RDS says the option holding the LATE row is chosen ~86% in both bases regardless of its validity. These two bases alone separate the models by ~85 percentage points in opposite directions.

BLOCK 3 \u2014 WEIGHTED MARGIN vs LATE-CUE COALITION (pairs 9-10). Base 5: A wins cols 0 and 3 (both .95), B wins cols 4 (.70) and 5 (.55). AVAI d=+1.01 \u2192 p(A) \u2248 1.00 (its largest margin outside the unanimous trials). RDS \u2248 .28: the two late B rows dominate salience and m=4 interference flattens the rest. Near-determinism vs. sub-chance on the same display.

BLOCK 4 \u2014 CONFLICT-INTERFERENCE GRADIENT (pairs 11-16). All discriminating rows favour the same option, with |D| = 2 (cols 4,5), 4 (cols 0,2,3,5) and 6 (all rows).
\u2022 |D|=2: AVAI \u2248 .99, RDS \u2248 .86.
\u2022 |D|=4: AVAI \u2248 1.00, RDS \u2248 .72.
\u2022 |D|=6: AVAI \u2248 1.00, RDS \u2248 .63.
AVAI predicts accuracy flat at ceiling (weakly increasing with the number of agreeing cues, since d grows monotonically); RDS predicts a monotone DECREASE, because difficulty tracks the count of differences rather than their diagnosticity. Observing >95% correct on the unanimous 6-0 trial is essentially impossible for RDS (ceiling ~.63 given eps_base/rho ranges), while a drop to ~.63 on a unanimous display is impossible for AVAI. These trials also serve as AVAI's lapse/beta baseline.

No pair is degenerate: on 6 of the 8 bases the models disagree in direction or by more than 30 percentage points, and the two cross-block tie bases constitute a parameter-free test that AVAI can only pass by sitting exactly at chance.

[6] GOAL: dissociate OCO-LI (advocated) from RDS (competitor). The two models are algebraically identical EXCEPT for OCO-LI's label-individuation gate phi: with probability 1-phi the odd-cue-out/recency anchoring competition cannot run (columns are not individuable) and the choice collapses to a probability-matched tally over the discriminating rows. Everything else (salience form exp(lam*j)*(1-v)^delta, conflict interference 1-(1-eps0)*rho^(m-2)) is shared and its parameter ranges essentially overlap. The only way to separate them is therefore a design in which phi is FORCED to 0, and this is achieved parameter-free (independent of tau) by giving every stated validity an exact duplicate.

VALIDITIES: [0.95, 0.55, 0.75, 0.95, 0.75, 0.55]. Each level (.95, .75, .55) appears exactly twice, so d_j = min_k|v_j-v_k| = 0 for every column, s_j = 0, and phi = 0 EXACTLY for any tau in [0.02,0.045]. The arrangement is neither the palindrome nor the repeated-block vector used in earlier designs: the .95 rows sit at columns 0 and 3, the .55 rows at 1 and 5, the .75 rows at 2 and 4, so every same-validity pair spans a different column distance and low/high validities alternate in both display halves. With lam=1.6, delta=0.9 the RDS salience weights are u = [0.067, 2.42, 7.05, 8.19, 172.8, 1453] \u2014 a five-order-of-magnitude spread \u2014 whereas OCO-LI (phi=0) treats every discriminating row as interchangeable.

8 base structures, each with its exact A/B mirror (16 unique pairs, K = 6 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows are filled with matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s.

BLOCK 1 \u2014 SHARED-PARAMETER BASELINES (pairs 1-4). Base 1: unanimous m=2 (A wins cols 0 and 5). Base 2: unanimous m=6 (A wins every row). All discriminating rows point the same way, so both models predict identically (p(A) ~ .86 at m=2, ~ .64 at m=6 for the stated parameter ranges). These pin eps_base and rho for BOTH models, so the contrast trials cannot be explained away as global noise, and they also test the shared claim that unanimity gets WORSE with more agreeing rows.

BLOCK 2 \u2014 CROWN JEWEL: PARAMETER-FREE m=2 SPLIT TIES (pairs 5-12). Four bases in which exactly one row favours A and exactly one row favours B. Under OCO-LI with phi=0 the flat tally is 1/1 -> p_core = .500, and mixing with the guess term leaves p(A) = .500 EXACTLY, for every base and every parameter setting (lam, delta, tau, eps0, rho all drop out). Under RDS the same four displays are strongly lopsided and, crucially, ORDERED by column position and validity:
  \u2022 Base 3: A wins col0 (.95) vs B wins col3 (.95) \u2014 identical validity, distant columns. RDS ~ .15.
  \u2022 Base 4: A wins col2 (.75) vs B wins col4 (.75) \u2014 identical validity, medium distance. RDS ~ .17.
  \u2022 Base 5: A wins col0 (.95) vs B wins col1 (.55) \u2014 adjacent, recency and anti-validity ALIGNED. RDS ~ .16.
  \u2022 Base 6: A wins col1 (.55) vs B wins col3 (.95) \u2014 recency and anti-validity OPPOSED. RDS ~ .30.
A flat .50 profile across all four bases (and their mirrors) is producible ONLY by the gated model; RDS with eps_base <= .32 cannot come within 18 points of chance on any of them, and it is additionally forced to produce the ordered spread .15 < .16 < .17 < .30. Flat-vs-ordered-spread at constant m is the decisive quantitative signature.

BLOCK 3 \u2014 TALLY PROPORTION VS SALIENCE MASS (pairs 13-16).
  \u2022 Base 7 (m=3): A wins cols 0 (.95) and 2 (.75), B wins col5 (.55, last column). OCO-LI: proportion 2/3 -> p(A) ~ .60 (ABOVE chance). RDS: the single late .55 row carries 99.5% of the salience mass -> p(A) ~ .22 (well BELOW chance). ~38-point directional disagreement.
  \u2022 Base 8 (m=4): A wins col5 alone (.55, last), B wins cols 0,1,2. OCO-LI: proportion 1/4 -> p(A) ~ .39 (below chance, the majority option wins). RDS: the lone late row dominates salience -> p(A) ~ .72 (above chance, the minority option wins). A full directional reversal on the same display.

INTERPRETATION: if choices on the four m=2 split bases sit at 50/50 while the m=3 and m=4 conflict bases track the PROPORTION of discriminating rows (above chance for 2-vs-1, below chance for 1-vs-3), the label-ambiguity gate is confirmed and RDS is falsified with essentially no free parameters. If instead the m=2 splits show the strong sub-chance, position-ordered anti-validity pattern (~.15-.30) and the minority-but-late option is chosen ~72% of the time in Base 8, RDS is confirmed and OCO-LI's gating claim is dead. Together with the already-run all-distinct-label designs (where both models coincide), this experiment supplies the exact 'duplicate-versus-de-duplicate the stated validity vector' contrast that OCO-LI names as its own falsification test.

[7] GOAL: dissociate RDS (advocated) from OCO-LI (competitor). The two models share their salience form exp(lam*j)*(1-v_j)^delta and their conflict-interference term 1-(1-eps0)*rho^(m-2); the ONLY structural difference is OCO-LI's label-individuation gate phi = mean_j min(1, d_j/tau), where d_j is the distance of expert j's stated validity to its nearest neighbour. With probability 1-phi OCO-LI abandons the anchoring competition and probability-matches the raw PROPORTION of discriminating rows. Any design with all-distinct labels (gaps >= .05) drives phi -> 1 and makes the models identical; a design with EXACT duplicates was already run. This design instead uses NEAR-duplicates: v = [0.955, 0.545, 0.950, 0.540, 0.945, 0.535]. Every expert's nearest neighbour is exactly 0.005 away, so d_j = 0.005 for all j and phi = min(1, 0.005/tau) is pinned to [0.11, 0.25] for tau in [0.02, 0.045] (point estimate phi ~ 0.156). This is not degenerate (all six labels are numerically distinct, and high vs low validities differ by 0.41, so the anti-validity/oddness factor is fully alive) but OCO-LI must be ~85% probability-matching while RDS runs the pure salience competition. Layout alternates high/low validity across columns (H,L,H,L,H,L), so recency and anti-validity can be aligned or opposed. RDS salience with lam=1.6, delta=0.9: u = [0.06, 2.44, 1.65, 60.4, 44.2, 1496]. Note u1 > u2 and u3 > u4: one step of oddness slightly outweighs one column of recency, a within-RDS constraint tested by base 5.

9 base structures, each with its exact A/B mirror (18 unique pairs, K = 5 -> 90 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions below use lam=1.6, delta=0.9, eps0=0.28, rho=0.78 (RDS) and the same plus tau=0.032 -> phi=0.156 (OCO-LI).

BLOCK 1 - SHARED BASELINES (bases 1-2, pairs 1-4). Base 1: unanimous m=2 (A wins cols 2,3). Base 2: unanimous m=6 (A wins every row). All discriminating rows point the same way, so p_flat = p_anchor = 1 and BOTH models predict identically (~.86 at m=2, ~.63 at m=6). These pin eps_base and rho for both models, so the contrast trials cannot be dismissed as global noise, and they test the shared claim that unanimity gets WORSE as more rows speak.

BLOCK 2 - m=2 SPLIT TIES, WHERE OCO-LI IS FORCED TOWARD CHANCE (bases 3-5, pairs 5-10). Exactly one row favours A and one favours B, so the flat tally is 1/1 = .500 and OCO-LI can move away from chance only by the factor phi <= .25.
  - Base 3 (A wins col0 .955, B wins col1 .545; adjacent, recency and oddness ALIGNED against A): RDS ~ .16, OCO-LI ~ .45.
  - Base 4 (A wins col0 .955, B wins col5 .535; maximal column distance): RDS ~ .14 (its floor eps0/2), OCO-LI ~ .45.
  - Base 5 (A wins col1 .545, B wins col2 .950; recency and oddness OPPOSED): RDS ~ .57 (oddness narrowly beats one column of recency), OCO-LI ~ .51. This base has little cross-model leverage but is the sharpest internal test of the lam/delta trade-off inside the shared salience rule, and it rules out pure position accounts.
A ~.14-.16 profile on bases 3-4 with a crossover to ~.57 on base 5 is uniquely RDS; a compressed .45/.45/.51 profile hugging chance on all three is the signature of the gate.

BLOCK 3 - CROWN JEWEL: DIRECTIONAL REVERSALS AT ASYMMETRIC m (bases 6, 7, 9; pairs 11-14, 17-18). Here the proportion of discriminating rows and the salience mass point in OPPOSITE directions.
  - Base 6 (m=3): A wins cols 0 (.955) and 2 (.950), B wins col5 (.535, last column). Proportion favours A (2/3) but 99.7% of the salience mass sits on B's late low-validity row. RDS ~ .22 (well BELOW chance); OCO-LI ~ .54 (ABOVE chance). ~32-point directional disagreement.
  - Base 7 (m=4): A wins col5 alone (.535, last), B wins cols 0,1,2. Proportion favours B (3/4); salience favours A. RDS ~ .72; OCO-LI ~ .44. Full directional reversal on one display.
  - Base 9 (m=5): A wins cols 0,1,2,3, B wins col5 alone. Proportion 4/5 for A; salience 96% for B. RDS ~ .34; OCO-LI ~ .56.
Across bases 7, 6, 9 A's share of discriminating rows rises .25 -> .67 -> .80. OCO-LI must therefore produce a MONOTONE INCREASING profile (~.44, .54, .56) tracking the tally proportion, whereas RDS produces a NON-MONOTONE profile dictated entirely by which option owns the last discriminating column (~.72, .22, .34). Monotone-in-proportion versus governed-by-the-last-column is the decisive quantitative signature and needs no free parameters to read off.

BLOCK 4 - BALANCED HIGH-m SPLITS (base 8, pairs 15-16). A wins cols 0,1 (early), B wins cols 4,5 (late); proportion is exactly 1/2 so OCO-LI's ungated component is pinned to .500, while RDS's salience is 99.8% on B's late rows. RDS ~ .28, OCO-LI ~ .47 at m=4. This isolates the position effect at a tally tie with more conflict than block 2, and jointly with base 4 it maps how far interference erodes the anti-validity/recency pull as m grows.

INTERPRETATION: if the m=2 splits sit at .14-.16 (and their mirrors at .84-.86), the 2-vs-1 base 6 sits below chance while the 1-vs-3 base 7 sits above chance, and choices track the OWNER OF THE LAST DISCRIMINATING COLUMN rather than the row count, RDS is confirmed and the label-individuation gate is falsified (with near-duplicate labels the gate should have suppressed exactly these effects). If instead every split trial collapses toward chance and the three asymmetric bases order themselves monotonically by the proportion of rows favouring A, OCO-LI's gating claim is confirmed. No pair is degenerate: on 6 of the 9 bases the models disagree by >18 points or in direction, and the two unanimous bases supply the shared lapse/interference baseline.

[8] GOAL: dissociate OCO-LI (advocated) from SSI-DS+ (competitor). The two accounts share a salience form ((1-v)^exponent * exp(lam*j)) and a conflict-interference term, so they can only be separated on three axes: (A) HOW the salience is used u2014 OCO-LI samples ONE anchoring row (choice = normalized salience share) whereas SSI-DS+ SUMS the rows and passes the rescaled sum through a sigmoid (extremizing, coalitions add); (B) THE GATE u2014 OCO-LI's individuation gate is a GRADED function of the numeric validity gap, phi = mean_j min(1, d_j/tau) with tau in [.02,.045], while SSI-DS+ switches all-or-none on printed-label injectivity with a tolerance of only .0025; (C) INTERFERENCE u2014 OCO-LI's guess term grows with the raw number of discriminating rows m, SSI-DS+ discounts unanimous displays by eta.

VALIDITIES: [0.958, 0.546, 0.954, 0.542, 0.950, 0.538] u2014 three near-identical high labels and three near-identical low labels, arranged H,L,H,L,H,L. Every expert's nearest neighbour is EXACTLY 0.004 away, so (i) all six printed labels are numerically distinct and every consecutive gap (0.004) exceeds SSI-DS+'s LABEL_TOL = 0.0025, forcing SSI-DS+ into full WEIGHTED (compensatory summation) mode; (ii) OCO-LI's gate is pinned to phi = 0.004/tau in [0.089, 0.20] (point ~0.13), so OCO-LI must be ~87% probability-matching the raw PROPORTION of discriminating rows. High vs low validity still differ by 0.41, so the anti-validity/oddness factor is fully alive for both models; the alternating layout lets recency and oddness be aligned or opposed. Salience mass (lam~1.6, exponent~0.9) is u ~ [0.06, 2.6, 1.7, 70, 50, 1910]: column 5 (late, low-validity) carries almost all of it.

8 base structures, each with its exact A/B mirror (16 unique pairs, K=6 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions use lam=1.65/1.5, delta=gamma=0.9, eps=0.28/0.24, rho=0.79/0.78, beta=3.6, eta=0.73, phi=0.13.

BLOCK 1 u2014 SHARED-ISH BASELINES / INTERFERENCE (bases 1-2, pairs 1-4). Base 1: unanimous m=2 (A wins cols 4,5) u2192 both ~.86, pinning eps0. Base 2: unanimous m=6 u2192 OCO-LI ~.64 (full load) vs SSI-DS+ ~.68-.70 (load discounted by eta because there is nothing to adjudicate). A steep unanimity decay (.86 -> .64) supports OCO-LI's raw-m interference; a shallow one (.86 -> .70) supports the competition-scaled version.

BLOCK 2 u2014 GATE-DIAGNOSTIC m=2 SPLITS (bases 3-4, pairs 5-8). Proportion is exactly 1/1, so OCO-LI can leave chance only by the factor phi<=0.20.
u2022 Base 3: A wins col0 (.958, first) vs B wins col5 (.538, last); recency and oddness both hand the salience to B. OCO-LI ~.43-.47 (barely below chance); SSI-DS+ ~.14 (E is essentially -1, sigmoid saturates). ~30-point gap.
u2022 Base 4: A wins col3 (.542) vs B wins col4 (.950); oddness and recency OPPOSED, salience narrowly favours A. OCO-LI ~.51 (chance); SSI-DS+ ~.72-.75. ~21-point gap. Compressed-toward-chance vs saturated is the signature of the graded gate.

BLOCK 3 u2014 CROWN JEWEL: PROPORTION vs SUMMATION AT ASYMMETRIC m (bases 5-7, pairs 9-14). Here the row COUNT and the salience mass point in opposite directions.
u2022 Base 5 (m=3): A wins cols 0,2 (both high-validity, early), B wins col5 (low, last). Proportion 2/3 for A; >99% of salience mass on B. OCO-LI ~.55 (ABOVE chance); SSI-DS+ ~.22 (WELL BELOW). ~33-point directional reversal.
u2022 Base 6 (m=4): A wins col5 alone, B wins cols 0,1,2. Proportion 1/4 for A; salience 99.8% for A. OCO-LI ~.43 (below chance, majority wins); SSI-DS+ ~.72 (above chance, the lone late low-validity row wins). Full reversal on one display.
u2022 Base 7 (m=5): A wins cols 0,1,2,3; B wins col5. Proportion 4/5 for A; salience 95% for B. OCO-LI ~.57; SSI-DS+ ~.33.
Across bases 6, 5, 7 A's share of discriminating rows rises .25 -> .67 -> .80: OCO-LI must produce a MONOTONE INCREASING profile (~.43, .55, .57) tracking the tally proportion, whereas SSI-DS+ produces a profile dictated by the summed (last-column-dominated) weight (~.72, .22, .33) that is non-monotone in proportion. Monotone-in-proportion versus governed-by-the-summed-salience is the decisive quantitative signature and needs no free parameters to read off.

BLOCK 4 u2014 BALANCED HIGH-m SPLIT (base 8, pairs 15-16). A wins cols 0,2 (early highs), B wins cols 3,5 (late lows); proportion is exactly 1/2 so OCO-LI's ungated component is pinned to .500 (predicted ~.47 after interference) while SSI-DS+'s summed weight is ~-1.05 rescaled -> ~.28. This isolates the gate at a tally tie with more conflict than block 2 and maps how interference erodes the pull as m grows.

INTERPRETATION: if the m=2 splits hug chance (.43-.51), the 2-vs-1 base 5 sits ABOVE chance while the 1-vs-3 base 6 sits BELOW chance, and the asymmetric bases order themselves monotonically by the proportion of rows favouring A, the graded label-individuation gate is confirmed and the all-or-none injectivity switch plus compensatory summation is falsified. If instead every split saturates toward the option owning the late low-validity row (.14-.28) and base 6 is chosen ~72% of the time despite being outvoted 1-vs-3, SSI-DS+ wins. No pair is degenerate: on 6 of 8 bases the models disagree in direction or by >19 points; the two unanimous bases supply the lapse baseline and simultaneously test the eta discount.

[9] GOAL: dissociate SSI-DS+ (advocated) from OCO-LI (competitor) on the two axes that remain once the label gate is neutralised: (A) COMPENSATORY SUMMATION-THROUGH-A-SIGMOID versus ONE-ROW ANCHOR SAMPLING (proportional salience share), and (B) COMPETITION-SCALED versus RAW-m CONFLICT INTERFERENCE (the eta discount on unanimous displays).

VALIDITIES: [0.53, 0.72, 0.86, 0.93, 0.99]. Deliberately five experts (not six) so the exp(lam*j) recency span is small enough that validity can counteract it, and validity RISES with column index so that oddness ((1-v)^exp) and recency pull in opposite directions and the resulting salience weights are all within one order of magnitude. With lam=1.5, exp=0.9 the shared salience weights are w = [0.51, 1.43, 3.42, 8.23, 6.41] \u2014 note the non-monotone peak at column 3, which lets me build coalitions whose SUM exceeds the single largest row without any row dominating. Crucially all pairwise validity gaps are >= 0.06, i.e. above OCO-LI's tau (max .045), so phi = 1 EXACTLY: OCO-LI runs its odd-cue-out anchoring at full strength and cannot be rescued (or blamed) by the individuation gate. All gaps also exceed SSI-DS+'s LABEL_TOL = .0025, so SSI-DS+ is locked into WEIGHTED summation mode. Both models therefore use the same salience form and always agree in DIRECTION; the entire contrast is quantitative, which is what makes it sharp.

Key algebra: with S_A, S_B the salience mass on each side and M the largest single discriminating weight, OCO-LI predicts p_core = S_A/(S_A+S_B) (proportional, bounded, near-linear in the salience share) whereas SSI-DS+ predicts p_core = sigmoid(beta*(S_A-S_B)/M) (saturating, and boosted whenever the winning side is a COALITION of sub-maximal rows, because the sum is normalised by the max row rather than by the total).

24 unique pairs (12 base structures each with its exact A/B mirror), K = 4 -> 96 trials, so both models' predicted responses are perfectly key-balanced. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions below use SSI-DS+ (gamma=.90, lam=1.50, beta=3.6, eps0=.24, rho=.78, eta=.73) and OCO-LI (delta=.90, lam=1.65, eps_base=.28, rho=.79, phi=1).

BLOCK A \u2014 m=2 SALIENCE-RATIO LADDER (pairs 1-10). Exactly one row favours each option; the ratio r = w_small/w_large is swept: A2 (col3 vs col4, r~.78), A5 (col4 vs col2, r~.53), A1 (col2 vs col1, r~.42), A4 (col1 vs col0, r~.36), A3 (col3 vs col1, r~.17). Interference is constant across all five (m=2), so this isolates the response function. OCO-LI must produce a nearly LINEAR climb in the salience share: ~.52, .66, .67, .69, .78. SSI-DS+ must produce a SATURATING profile that is 10-15 points higher at intermediate ratios and then flattens: ~.64, .76, .80, .81, .84. Same ordering, very different curvature and intercept \u2014 regressing p on the salience share yields slope ~.7 for OCO-LI versus a shallow, high-intercept curve for SSI-DS+.

BLOCK B \u2014 CROWN JEWEL: COALITION SUMMATION (pairs 11-18). Here several sub-maximal rows are pitted against one large row.
\u2022 B1 (m=3): A wins cols 2+3 (sum 11.65), B wins col 4 (6.41). SSI-DS+: E = +0.64 of the max row -> p ~ .74. OCO-LI: share .60 -> p ~ .56. ~18-point gap, the largest in the design.
\u2022 B2 (m=3): A wins cols 0+1 (1.93), B wins col 2 (3.42). SSI-DS+ ~ .31, OCO-LI ~ .40 (both below chance, 9-point gap in the mirror-image direction, ruling out a global response-scaling explanation).
\u2022 B3 (m=4): A wins cols 1,2,3, B wins col 4. SSI-DS+ ~ .71 vs OCO-LI ~ .55 (~15 points): three added rows push SSI-DS+ up despite the extra interference, whereas OCO-LI's share barely moves off .6 and its raw-m guessing drags it toward chance.
\u2022 B4 (m=4): A wins cols 3+4, B wins cols 1+2. SSI-DS+ ~ .73, OCO-LI ~ .63.
Across B1/B3 versus their m=2 counterparts, SSI-DS+ predicts that ADDING agreeing rows to the winning coalition keeps or increases decisiveness, while OCO-LI predicts erosion toward chance \u2014 the summation-versus-sampling signature.

BLOCK C \u2014 UNANIMITY DECAY / eta TEST (pairs 19-24). Unanimous displays with m=2 (C1: cols 3,4), m=4 (C3: cols 1-4) and m=5 (C2: all rows). Both models coincide at m=2 (~.86-.88), which pins eps0/eps_base free of the contrast. Then SSI-DS+, whose load is discounted by eta because there is nothing to adjudicate, decays only to ~.77 (m=4) and ~.72 (m=5), while OCO-LI's raw-m interference decays to ~.72 and ~.67. A shallow unanimity decay together with steep conflict decay (Block A/B m=4 trials well below the m=4 unanimous trials) is the competition-scaled interference signature; a single monotone function of display size supports OCO-LI.

INTERPRETATION: if the m=2 ladder is compressed and high (~.64-.84) and the coalition trials B1/B3 sit ~15-18 points above what the raw salience share predicts, compensatory summation with max-normalisation is confirmed and one-row anchor sampling is falsified. If instead choice probabilities track the salience SHARE almost linearly (~.52-.78) and coalitions buy nothing beyond their proportional mass, OCO-LI wins. No pair is degenerate: on 9 of the 12 bases the models differ by 9-18 percentage points, and the unanimous bases double as the shared lapse baseline and the eta test.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: separate MDSC-NV (advocated) from SSI-DS+ (competitor). Both models invert validity (near-chance experts carry the decision) and both add a reading-position gain, so they can only be separated on three axes: (A) DOUBT GAIN MAGNITUDE \u2014 MDSC-NV uses (1-v)^delta with delta=13-17, SSI-DS+ uses (1-v)^gamma with gamma~0.9, i.e. ~17x weaker, so with comparable positional gains (lam_MDSC = pos_ratio*delta ~ 1.2-2.4 vs lam_SSI ~ 1.5) the two models put their weight on DIFFERENT rows: MDSC is governed by the lowest-validity row wherever it sits, SSI is governed by the LAST row unless it is extremely non-distinctive; (B) SINGLE-CUE vs SUMMATION \u2014 MDSC draws one row and cannot let a coalition add up, SSI sums and passes the max-normalised sum through a sigmoid; (C) NULL-VERDICT CAPTURE \u2014 MDSC says non-discriminating rows matter (with prob q~.41 attention is captured by the most doubtful expert in the WHOLE display, and if that expert is silent the subject guesses), whereas SSI (like every compensatory or one-reason model) says silent rows are strictly irrelevant.\n\nVALIDITIES [0.55, 0.90, 0.75, 0.95, 0.68, 0.85]: scrambled so validity does not track column order; the two most doubtful experts sit at column 0 (.55, FIRST) and column 4 (.68, late), the two most certain at columns 1 (.90) and 3 (.95). All pairwise gaps >= .05, far above both models' label tolerances (.002 / .0025), so neither model is in tally/degenerate mode \u2014 the contrast is purely mechanistic. MDSC log-salience (delta=15, lam=1.725): col4 -10.2 > col0 -12.0 > col2 -17.3 > col5 -19.8 > col1 -32.8 > col3 -39.8. SSI log-weight (gamma=.9, lam=1.5): col5 5.79 > col4 4.98 > col3 1.80 ~ col2 1.75 >> col1 -0.57 ~ col0 -0.72. Note the near-perfect crossing of the two orderings at the extremes (col0 is MDSC's #2 and SSI's last; col5 is SSI's #1 and MDSC's #4).\n\n8 base structures, each with its exact A/B mirror (16 unique pairs, K=6 -> 96 trials), so both models' predicted responses are perfectly key-balanced; non-discriminating rows carry matched 1s AND matched 0s so displays are dense.\n\nBLOCK 1 \u2014 CROWN JEWEL: NULL-VERDICT CAPTURE ON UNANIMOUS m=2 DISPLAYS (bases 1-3). All three are 2-0 dominance displays that differ ONLY in which rows are silent. SSI must predict the same saturated value (~.88) on all three, because E is computed over discriminating rows only and a unanimous 2-row display always gives E in (1,2). MDSC predicts a graded decline set by how much global doubt mass sits on SILENT rows: base 1 (disc = cols 0 & 4, the two most doubtful rows -> s_nondisc ~ 0) ~ .88; base 3 (disc = cols 0 & 2, so the globally most doubtful expert col4 is silent -> s_nondisc ~ .86) ~ .75; base 2 (disc = cols 1 & 3, the two highest-validity rows, BOTH doubtful experts silent -> s_nondisc ~ 1) ~ .72. A flat .88/.88/.88 profile falsifies null-verdict capture; a 15-16 point drop from base 1 to base 2 with identical 2-0 dominance structure is producible by no compensatory or standard one-reason model. Base 1 simultaneously pins eps0 for both models.\n\nBLOCK 2 \u2014 DOUBT vs RECENCY DIRECTIONAL REVERSALS (bases 4-5, m=2 splits).\n- Base 4: A wins col0 (.55, first column), B wins col5 (.85, last column). MDSC: the .55 row wins the doubt competition by ~8 log units -> p(A) ~ .75. SSI: col5's weight is ~670x col0's -> E ~ -1 -> p(A) ~ .14. ~60-point reversal.\n- Base 5: A wins col1 (.90), B wins col3 (.95), with both doubtful rows silent. MDSC: doubt still picks col1 (.90 is 2x as doubtful as .95) but capture drags it toward chance -> p(A) ~ .72. SSI: position dominates -> p(A) ~ .15. ~57-point reversal. Bases 4 and 5 jointly estimate the delta/lam ratio: only a doubt gain above ~10 can make an EARLY .55 (or a mid-display .90) row beat a late row.\n\nBLOCK 3 \u2014 COALITIONS BUY NOTHING (bases 6, 8).\n- Base 6 (m=6): A wins col4 alone (.68, the globally most doubtful expert); B wins the other five rows. MDSC: one more-doubtful row outvotes five -> p(A) ~ .62 (above chance) despite being outvoted 1-vs-5. SSI: the summed coalition (dominated by col5) wins -> p(A) ~ .39 (below chance). A clean directional split at a 1-vs-5 row count.\n- Base 8 (m=4, 2-vs-2): A wins cols 0 & 4 (the two most doubtful), B wins cols 3 & 5 (SSI's heavy late rows). MDSC ~ .75; SSI ~ .32 (its summed weight is dominated by col5). ~43-point reversal at an exact row-count tie, which no tally account can produce either way.\n\nBLOCK 4 \u2014 UNANIMITY DECAY / eta TEST (base 7): A wins all six rows. MDSC's raw-m interference gives ~.66; SSI's competition-scaled load (eta discount, nothing to adjudicate) gives ~.69. Together with base 1 (~.88 for both) this pins the lapse parameters free of the contrast and gives a mild test of the eta discount.\n\nINTERPRETATION: if unanimous 2-0 displays vary by ~15 points purely as a function of WHICH experts are silent (base1 > base3 > base2), if the early .55 row and the .90 row beat the late .85/.95 rows (bases 4-5 above chance), and if the lone doubtful row beats a five-row coalition (base 6 above chance) and the doubt pair beats the late pair (base 8 ~ .75), MDSC-NV is confirmed and SSI-DS+ is falsified. If instead all three unanimous m=2 displays sit flat at ~.88, and bases 4, 5, 6, 8 all land in the .14-.39 range (the option holding the LATE rows being chosen), compensatory summation over discriminating rows with an all-or-none injectivity switch wins and the null-verdict-capture claim is dead.",
  "validities": [
    0.55,
    0.9,
    0.75,
    0.95,
    0.68,
    0.85
  ],
  "trial_a_ratings": [
    [
      1,
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
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
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
      1,
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
      1,
      0,
      1,
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
      0
    ],
    [
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
      1
    ],
    [
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
      1,
      1,
      0,
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
      0,
      0,
      1,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
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
      1
    ],
    [
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
      1,
      1,
      0,
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
      0,
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
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
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
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
      1,
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
      1,
      0,
      1,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 29100,
  "output_tokens": 18986
}
```
