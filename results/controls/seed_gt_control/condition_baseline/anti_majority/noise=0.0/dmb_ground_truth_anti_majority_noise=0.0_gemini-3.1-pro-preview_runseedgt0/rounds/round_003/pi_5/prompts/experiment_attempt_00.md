# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Reverse Weighted Additive (Negative Utility) Theory: Subjects systematically interpret the features or validities as negative signals (such as costs, penalties, or negative reviews) rather than positive attributes. Consequently, they compute a standard Weighted Additive (WADD) score—the sum of feature values weighted by their respective validities—but aim to minimize this score. Choice behavior is modeled as a softmax over the negative WADD scores, offering an elegant, single-process cognitive explanation for the choice inversions observed across experiments. The noise parameters are tuned to reflect a higher degree of stochasticity observed in human data.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Reverse WADD expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Compute standard WADD scores (sum of features weighted by validities)
    wadd_scores = stim @ validities
    
    # Reverse the scores to model negative utility (minimization)
    scores = -wadd_scores
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the negative WADD scores with numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People are modeled as running three classic heuristics in parallel — Take The Best (one-reason, top-validity cue), Tallying (count of feature-wise wins), and Weighted Additive (validity-weighted sum) — reading off the single option each heuristic deterministically prefers, and then leaning toward whichever option the majority of the three did NOT prefer. With three binary voters at least two always agree, so a majority option always exists; the model targets its complement. Response noise enters only at the final, ensemble level: a softmax over the binary "flipped-majority" score with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options. The individual heuristics contribute only their hard preference (their argmax) to the vote — they carry no internal noise — so beta and epsilon describe noise in the contrarian response itself, not in the constituent heuristics.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Anti-majority-vote ensemble. Each of Tallying / TTB / WADD
    # contributes a single hard preference (the argmax of its
    # deterministic score — no per-model softmax or lapse). We take the
    # majority vote across the three binary votes and flip it: when at
    # least two models agree on an option, we target the other one.
    # Choice noise (softmax over beta, lapse over epsilon) is applied
    # ONCE here, to the flipped-majority decision.

    def _pref_tallying(stimulus):
        # Tallying (Dawes 1979; Gigerenzer & Goldstein 1999): count
        # strict feature-wise wins; prefer the option with more wins.
        # Ties (equal win counts) fall to option A via argmax.
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim != 2 or stim.shape[0] != 2:
            raise ValueError(
                f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
            )
        a, b = stim[0], stim[1]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        return int(np.argmax(np.array([a_wins, b_wins])))

    def _pref_ttb(parameters, stimulus):
        # Take The Best (Gigerenzer & Goldstein 1996): consult cues in
        # descending validity; the first discriminating cue decides.
        # If no cue discriminates, the score is uniform and argmax
        # falls to option A.
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
            scores = np.ones(2) / 2.0
        else:
            scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        return int(np.argmax(scores))

    def _pref_wadd(parameters, stimulus):
        # Weighted Additive (Payne, Bettman & Johnson 1993): prefer the
        # option with the higher validity-weighted sum. Ties fall to
        # option A via argmax.
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim != 2 or stim.shape[0] != 2:
            raise ValueError(
                f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
            )
        n_features = stim.shape[1]
        validities = np.asarray(parameters["validities"], dtype=float)
        w = np.asarray(parameters["weights"], dtype=float)
        if w.shape[0] != n_features:
            raise ValueError(
                f"weights length {w.shape[0]} != n_features {n_features}."
            )
        scores = stim @ (validities * w)
        return int(np.argmax(scores))

    # Each model's hard preference (no noise inside the models).
    c_tallying = _pref_tallying(stimulus)
    c_ttb = _pref_ttb(parameters, stimulus)
    c_wadd = _pref_wadd(parameters, stimulus)

    # Majority vote across the three binary votes, then flip it.
    votes = [c_tallying, c_ttb, c_wadd]
    majority = 0 if votes.count(0) >= 2 else 1
    flipped = 1 - majority

    # Binary score for the flipped-majority choice (winner = 1, loser = 0),
    # then ONE softmax + lapse over the two options.
    scores = np.array([1.0, 0.0]) if flipped == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # flipped score this collapses to sigmoid(beta) for the targeted
    # option (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic flip).
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
[0] The advocated theory posits a 'flipped-majority' ensemble in which the decision-maker computes the deterministic preferences of Tallying, Take The Best (TTB), and Weighted Additive (WADD), and then deliberately chooses the option that the majority of these heuristics do NOT prefer. The competing theory posits a standard Tallying heuristic with softmax noise. To maximally dissociate these theories, we design trials where Tallying has a strict preference for one option, and the majority of the three heuristics (often including Tallying itself) also agree on that same option. Consequently, the advocated theory will strongly predict choosing the exact opposite of what the competing Tallying theory predicts.

[1] We design trials where the Advocated Theory (Tallying) either strongly prefers one option or predicts indifference, while the Competing Theory (Anti-majority ensemble) predicts the exact opposite. By pitting a single highly valid cue against multiple lower-validity cues, we create scenarios where Tallying prefers the option with more wins, whereas Take The Best (TTB) prefers the option with the single best cue. When Weighted Additive (WADD) aligns with Tallying, the majority vote aligns with Tallying, causing the Anti-majority model to deterministically flip and choose the TTB-preferred option. This yields a strong quantitative and qualitative dissociation between the two theories.

[2] To maximally dissociate the Advocated Theory (Anti-majority ensemble) from the Competing Theory (Anti-Tallying), we exploit the fact that Anti-Tallying is effectively the exact opposite of Tallying. We design trials where Tallying prefers Option A (because it has more winning features), but both Take The Best (TTB) and Weighted Additive (WADD) prefer Option B (because Option B wins on fewer but much higher-validity features). In this scenario, the majority of the three heuristics (TTB and WADD) prefer Option B. The Advocated Theory flips this majority and thus strongly predicts Option A. The Competing Theory, interpreting 0 as better than 1, counts the 'anti-wins' and strongly prefers Option B, as Option B has more 0s where Option A has 1s. This creates a stark qualitative and quantitative dissociation.

[3] To maximally dissociate the Advocated Theory (Anti-Tallying) from the Competing Theory (Anti-Majority), we construct trials where Tallying prefers one option, while both Take The Best (TTB) and Weighted Additive (WADD) prefer the other. In such a scenario, the majority of the three standard heuristics (TTB and WADD) agree on an option, causing the Anti-Majority model to strongly predict the *opposite* option (which aligns with Tallying's choice). Conversely, the Anti-Tallying model, which treats 0s as wins and thus acts as the exact opposite of Tallying, will prefer the option that Tallying dislikes. This results in the two models predicting completely opposite choices.

[4] To maximally dissociate the Advocated Theory (Anti-majority ensemble) from the Competing Theory (Soft Take-The-Best), we design trials where the majority of the three standard heuristics (Take The Best, Tallying, Weighted Additive) agree on one option, causing the Anti-majority model to deterministically flip its preference to the opposite option. In these trials, Take The Best (TTB) is always part of the majority. Because Soft TTB always predicts the option favored by the highest-validity discriminating cue (the TTB choice), it will align with the majority. Consequently, the Anti-majority model will strongly predict the exact opposite of Soft TTB.

[5] To maximally dissociate the Advocated Theory (Soft Take-The-Best) from the Competing Theory (Anti-majority ensemble), we design trials where the majority of the three standard heuristics (TTB, Tallying, WADD) agree on an option. Because Soft TTB aligns with standard TTB's directional preference, it will consistently predict the majority-preferred option. The Anti-majority model, however, deterministically flips the majority decision, thus predicting the exact opposite. Furthermore, by varying which feature is the first to discriminate (highest validity, second highest, or third highest), we test Soft TTB's unique prediction that the determinism of the choice scales with the validity of the stopping feature, a quantitative nuance entirely absent in the Anti-majority model.

[6] To maximally dissociate the Advocated Theory (Anti-majority ensemble) from the Competing Theory (Reverse WADD), we design trials where Standard WADD strongly disagrees with both Take The Best (TTB) and Tallying. When TTB and Tallying agree on an option (e.g., Option B), they form the majority, causing the Anti-majority model to predict the opposite (Option A). Meanwhile, since Standard WADD prefers Option A, the Reverse WADD model (which minimizes the WADD score) will prefer Option B. This creates a stark, deterministic divergence in the choices predicted by the two models.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To maximally dissociate the Advocated Theory (Reverse WADD) from the Competing Theory (Anti-majority ensemble), we design trials where Standard WADD strongly disagrees with both Take The Best (TTB) and Tallying. The Anti-majority model takes a majority vote of TTB, Tallying, and Standard WADD, and then deterministically flips it. By ensuring TTB and Tallying agree on an option (e.g., Option B), they form the majority, causing the Anti-majority model to predict the opposite (Option A). Meanwhile, because Standard WADD prefers Option A, the Reverse WADD model (which seeks to minimize the WADD score) will prefer Option B. This creates a stark, deterministic divergence in the choices predicted by the two models, completely separating a negative-utility WADD process from a contrarian heuristic ensemble.",
  "validities": [
    0.95,
    0.94,
    0.93,
    0.92,
    0.91,
    0.55,
    0.54,
    0.53,
    0.52,
    0.51
  ],
  "trial_a_ratings": [
    [
      0,
      1,
      1,
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
      1,
      1,
      1,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      0,
      0,
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
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4282,
  "candidates_token_count": 651,
  "total_token_count": 10998
}
```
