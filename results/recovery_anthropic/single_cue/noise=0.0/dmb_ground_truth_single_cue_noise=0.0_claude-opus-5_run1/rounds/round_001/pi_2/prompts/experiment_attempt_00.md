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
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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
**Description:** **Amplified-Validity Additive Integration (AVAI).**

People do not stop at one reason (Take The Best) and do not count cues as equal (Tallying). Instead, on every trial they read all expert ratings, convert each communicated validity v_j into a subjective *pull* w_j = v_j^gamma, and form a single continuous evidence difference d = sum_j w_j (a_j - b_j). Choice is a logistic function of that continuous difference, p(A) = (1-eps)*sigmoid(beta*d) + eps/2.

Three claims give the theory its empirical bite:

1. **No stopping rule.** Every cue enters the sum, so a single high-validity cue can be outvoted by a coalition of weaker cues. This is why, when the top cue points one way and three or four lesser cues point the other, people overwhelmingly follow the coalition (unlike TTB).

2. **Unequal but compressed weights.** Because all validities lie in (0.5, 1], raising them to a modest power gamma (~2) yields weights that are all of the same order of magnitude (roughly 0.2-0.9). Integration therefore *looks* like counting most of the time, but with a systematic tilt toward the option holding the more valid cues. Crucially, the exponent gamma is a *subjective amplification* of stated validity, not the normatively correct log-odds transform: people over-weight validity relative to plain counting yet massively under-weight it relative to Bayes (log(v/(1-v)) would make one 0.95 cue beat two mid cues; humans do not do this). gamma is the single psychological parameter that locates a person on the tallying-to-lexicographic continuum.

3. **Magnitude sensitivity.** Because the logistic acts on the raw weighted difference rather than on a binarised winner code, trials differ *gradedly* in confidence. Unit-tally ties are not forced to chance: they resolve toward whichever side owns the higher-validity cues, but only weakly when the weighted margin is small (e.g. 0.87+0.55 vs 0.78+0.70 stays near chance) and strongly when it is large (0.93 vs 0.60 is near-deterministic). Symmetrically, trials with a big count margin but a small weighted margin become noisy rather than deterministic.

Residual attentional lapses are captured by an epsilon-rate uniform guess. No learning occurs across trials (there is no feedback), so the rule is applied stationarily.

**Parameters:**
- gamma: [1.8, 2.7]
- beta: [4.5, 8.5]
- epsilon: [0.0, 0.04]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Amplified-Validity Additive Integration (AVAI).
    # d = sum_j (v_j ** gamma) * (a_j - b_j); p(A) = sigmoid(beta * d),
    # mixed with an epsilon lapse. All cues are consulted (no stopping
    # rule) and the logistic acts on the CONTINUOUS weighted difference,
    # so both tie-trials and large-margin trials are graded.
    import numpy as np

    # ---- parse the stimulus into two rating vectors -------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            a = np.asarray(vals[0], dtype=float).ravel()
            b = np.asarray(vals[1], dtype=float).ravel()
    else:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            half = arr.shape[0] // 2
            a, b = arr[:half], arr[half:2 * half]
        else:
            flat = arr.reshape(2, -1)
            a, b = flat[0], flat[1]

    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]

    # ---- validities -> subjective weights ----------------------------
    v_raw = parameters.get('validities', None)
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.shape[0] < n:
            pad = np.full(n - v.shape[0], 0.55)
            v = np.concatenate([v, pad])
        v = v[:n]
    # validities are probabilities of being right; keep them in (0.5, 1]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-9)

    gamma = float(parameters['gamma'])
    gamma = max(gamma, 0.0)
    w = np.power(v, gamma)

    # ---- continuous weighted evidence difference ---------------------
    d = float(np.sum(w * (a - b)))

    beta = float(parameters['beta'])
    eps = float(parameters['epsilon'])
    eps = min(max(eps, 0.0), 1.0)

    # numerically stable two-option softmax on [beta*d, 0]
    z = np.array([beta * d, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / e.sum()

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
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
    if not np.isfinite(s) or s <= 0:
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate unit-weight Tallying (advocated) from Amplified-Validity Additive Integration (AVAI). Both models are permutation-invariant over display position, so the decisive manipulation is to hold the WIN COUNT constant while sweeping the validity-weighted margin across its whole range, and then to break the rank-order between count margin and weighted margin.\n\nVALIDITIES: six experts in deliberately scrambled display order, [0.58, 0.97, 0.53, 0.80, 0.66, 0.90]. Ranked: H1 = idx1 (.97), H2 = idx5 (.90), M = idx3 (.80), L1 = idx4 (.66), L2 = idx0 (.58), L3 = idx2 (.53). Scrambling removes any 'read the leftmost cue' artifact and prevents position from proxying validity. AVAI weights at gamma = 2: H1=.941, H2=.810, M=.640, L1=.436, L2=.336, L3=.281 (the compressed regime AVAI assumes). Cues that are irrelevant on a trial are set to 1 for BOTH options, so every display is dense and requires actual comparison rather than counting 1s.\n\n8 base structures, each accompanied by its exact A/B mirror (16 unique pairs, 6 repetitions = 96 trials), so the predicted response of BOTH models is exactly key-balanced and side bias cannot mimic either model.\n\n(1) CONSTANT-COUNT-MARGIN LADDER (bases 1-4, plus mirrors). All four are +1 win-count margins for A, so Tallying must produce the SAME p(A) = sigmoid(beta) on every one of them, for any beta and epsilon. AVAI's weighted difference, in contrast, sweeps the entire probability range:\n  - Base 1: A wins {H1} only. d = +0.94 -> AVAI p(A) ~ .998.\n  - Base 2: A wins {H1,H2}, B wins {L2}. d = +1.42 -> AVAI p(A) ~ 1.00.\n  - Base 3: A wins {L2,L3}, B wins {H1}. d = -0.32 -> AVAI p(A) ~ .11 (DIRECTIONAL REVERSAL relative to Tallying).\n  - Base 4: A wins {L1,L2,L3}, B wins {H1,H2}. d = -0.70 -> AVAI p(A) ~ .01 (near-deterministic reversal).\nA flat profile across these four (all equal, all above .5) is a signature Tallying alone can produce; a monotone sweep from ~1.00 down to ~.01 at constant count margin is impossible for Tallying under any parameters. This is the crown-jewel contrast: flat vs. full-range sweep.\n\n(2) RANK-ORDER VIOLATION (bases 5-6, plus mirrors). Both are +2 count margins for A, so Tallying predicts them equal AND strictly more extreme than every margin-1 pair in block (1).\n  - Base 5: A wins {M,L1,L2,L3}, B wins {H1,H2}. d = -0.06 -> AVAI p(A) ~ .41, i.e. essentially chance despite the LARGEST count margin in the design.\n  - Base 6: A wins {H1,H2}, B wins nothing. d = +1.75 -> AVAI p(A) ~ 1.00.\nAVAI therefore predicts that a margin-2 trial (base 5) is noisier than a margin-1 trial (base 1 or 2) \u2014 a rank-order violation Tallying strictly forbids. Base 6 doubles as an agreement baseline that identifies epsilon and the global beta scale free of the model contrast.\n\n(3) TALLY TIES (bases 7-8, plus mirrors). Each option wins exactly the same number of cues, so Tallying is EXACTLY 50/50 for any beta and epsilon \u2014 a parameter-free prediction.\n  - Base 7: A wins {H1}, B wins {L3}. d = +0.66 -> AVAI p(A) ~ .99.\n  - Base 8: A wins {H1,L3}, B wins {H2,L1}. d = -0.02 -> AVAI p(A) ~ .46 (also near chance).\nIf subjects are at chance on base 7 as well as base 8, AVAI is falsified while Tallying is confirmed with no free parameters; if base 7 is near-deterministic while base 8 is at chance, that graded split is exactly AVAI's magnitude-sensitivity claim and cannot be produced by Tallying. Base 8 is additionally gamma-diagnostic: the sign of d for {H1,L3} vs {H2,L1} is near zero and flips with gamma, so it pins the amplification exponent if AVAI wins.\n\nNo pair is degenerate: on 5 of the 8 base structures the models disagree in direction or by more than 40 percentage points, the two tie trials are parameter-free tests of Tallying, and the remaining structures supply the noise/lapse baseline.",
  "validities": [
    0.58,
    0.97,
    0.53,
    0.8,
    0.66,
    0.9
  ],
  "trial_a_ratings": [
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
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
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
      0,
      0
    ],
    [
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
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
      1,
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
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
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
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
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
      1,
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
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
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
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 9930,
  "output_tokens": 9985
}
```
