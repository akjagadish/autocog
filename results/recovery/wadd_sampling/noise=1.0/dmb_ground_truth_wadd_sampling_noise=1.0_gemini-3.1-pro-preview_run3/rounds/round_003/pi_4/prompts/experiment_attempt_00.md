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
**Description:** Conflict-Induced Guessing Theory posits that decision makers concurrently evaluate options using both a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). When these two strategies agree, the decision maker confidently chooses the favored option (subject to standard softmax noise). However, when the strategies conflict—or when one strategy fails to corroborate the other (e.g., Tallying is tied while TTB prefers one option)—the decision maker experiences cognitive conflict or ambiguity. Unable to easily resolve this conflict, they resort to uniform guessing. This explains the consistent ~50% choice rates observed on adversarial trials and trials with weak corroborating evidence without requiring extreme parameter values.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Take The Best (TTB) prediction
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_winner = 0
    elif b_wins > a_wins:
        tally_winner = 1
    else:
        tally_winner = None
        
    # Determine choice probabilities
    if ttb_winner != tally_winner or ttb_winner is None:
        # Conflict or lack of clear corroboration leads to guessing
        p_core = np.array([0.5, 0.5])
    else:
        # No conflict: both strategies agree
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
            
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Threshold Tallying Theory posits that decision makers evaluate options by counting the number of features where one option strictly dominates the other (unweighted tallying). However, they only make a confident directional choice if the difference in winning features between the two options meets or exceeds a certain cognitive threshold. If the difference in evidence is below this threshold (e.g., a difference of 0 or 1), the decision maker finds the evidence too ambiguous or weak, leading to cognitive overload or uncertainty, and they resort to uniform guessing. This captures the persistent ~50% choice rates observed across many adversarial trials where the feature counts are closely matched.

**Parameters:**
- threshold: [1.5, 3.5]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Count strict feature-wise wins for each option
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    diff = abs(a_wins - b_wins)
    threshold = float(parameters["threshold"])
    
    # If the difference is below the threshold, the evidence is deemed too weak -> guessing
    if diff < threshold:
        p_core = np.array([0.5, 0.5])
    else:
        # Otherwise, make a choice based on the tally scores using softmax
        scores = np.array([float(a_wins), float(b_wins)])
        beta = float(parameters["beta"])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, we use a 5-feature design with spread validities. We design trials where the option with the single highest-validity discriminating cue (the TTB winner) is opposed by the option that has a greater overall number of positive features (the Tallying winner). TTB decides based solely on the first discriminating cue in descending validity order, while Tallying simply counts the number of cues favoring each option, ignoring validities entirely.

[1] To quantitatively dissociate Tallying from Take The Best (TTB), this design uses 5 features with strictly descending validities. In each trial, Option A is constructed to win on the highest-validity discriminating cue (making it the TTB choice), while Option B is constructed to win on a greater number of lower-validity cues (making it the Tallying choice). Because Tallying ignores validities and simply counts feature-wise wins, it will consistently select Option B. TTB, which stops at the first discriminating cue, will consistently select Option A.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) theory, we use a 4-feature design with specific validities that create opposing predictions. TTB is a non-compensatory strategy that stops at the first discriminating cue in order of validity. In contrast, WADD computes a weighted sum of all features using the log-odds of the validities. We chose validities [0.85, 0.75, 0.70, 0.65] such that the log-odds of the highest validity cue (weight ~1.73) is less than the sum of the log-odds of the remaining lower-validity cues (weight ~2.56). This allows us to construct 'compensatory' trials where Option A wins on the single highest-validity cue (chosen by TTB), but Option B has a combination of lower-validity cues whose total weight exceeds that of the top cue (chosen by WADD). We also include trials where the top cue is tied, testing the same non-compensatory vs. compensatory dynamic on the remaining cues.

[3] To quantitatively dissociate WADD from TTB, we use a 5-feature design with validities [0.90, 0.80, 0.75, 0.70, 0.65]. WADD evaluates options using log-odds weights, which here are approximately 2.20, 1.39, 1.10, 0.85, and 0.62. We design trials where the single highest-validity discriminating cue favors Option A (making it the TTB choice), but a combination of lower-validity cues favors Option B such that their summed log-odds weight exceeds that of the top cue (making it the WADD choice). We include variations with top-cue ties to test this compensatory versus non-compensatory dynamic across different subsets of cues.

[4] To quantitatively dissociate Conflict-Induced Guessing Theory from Weighted Additive (WADD) theory, we use a 5-feature design with validities [0.85, 0.80, 0.75, 0.60, 0.55]. Conflict-Induced Guessing Theory posits that decision makers use both Take The Best (TTB) and Tallying concurrently; if they agree, the subject chooses confidently, but if they conflict or one ties, the subject guesses (~50%). WADD posits a continuous integration of log-odds weights. We design specific trials to contrast these. In Trial 1, Option A is favored by both TTB (highest validity cue) and Tallying (3 cues to 2), so Conflict Theory predicts a confident choice for A. However, Option B's two cues have a higher combined log-odds weight than A's three cues, so WADD predicts a confident choice for B. In Trial 2, TTB favors A while Tallying favors B; Conflict Theory predicts ~50% guessing, whereas WADD strongly prefers A. Additional trials systematically vary these alignments to ensure robust dissociation across the full parameter space of both models.

[5] To quantitatively dissociate Weighted Additive (WADD) theory from Conflict-Induced Guessing Theory, we use a 5-feature design with validities set to [0.88, 0.82, 0.75, 0.65, 0.55]. Conflict-Induced Guessing Theory predicts that when Take The Best (TTB) and Tallying conflict or tie, subjects will experience ambiguity and guess (~50%). In contrast, WADD integrates all features using log-odds weights (approximately 1.99, 1.52, 1.10, 0.62, and 0.20) and makes directional choices based on the weighted sum. We designed trials where TTB and Tallying conflict (e.g., TTB prefers one option, Tallying prefers the other) or one heuristic ties, leading Conflict Theory to predict guessing. Meanwhile, WADD's log-odds weights heavily favor one option, leading to confident directional predictions, thereby strongly dissociating the two theories.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Conflict-Induced Guessing Theory from Threshold Tallying Theory, we use a 6-feature design. Conflict-Induced Guessing Theory predicts uniform guessing whenever Take The Best (TTB) and Tallying conflict, and confident choices when they agree. Threshold Tallying Theory predicts uniform guessing when the absolute difference in feature wins (Diff) is below a cognitive threshold (1.5 to 3.5), and confident choices when Diff is above this threshold. We construct two types of critical trials. In Type 1, one option wins on the single highest-validity cue (TTB winner) but loses on all remaining cues (Tallying loser). Here, TTB and Tallying conflict, so Conflict Theory predicts guessing; however, the Diff is 4 (5 wins vs 1 win), which exceeds the maximum threshold of 3.5, so Threshold Tallying predicts a highly confident choice. In Type 2, one option wins on the highest-validity cue and one other cue, while the alternative wins on only a single lower-validity cue. Here, TTB and Tallying agree, so Conflict Theory predicts a confident choice; however, the Diff is 1 (2 wins vs 1 win), which is strictly below the minimum threshold of 1.5, so Threshold Tallying predicts uniform guessing. These opposing predictions cleanly separate the two theories across their entire parameter spaces.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55,
    0.51
  ],
  "trial_a_ratings": [
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
      0
    ],
    [
      0,
      0,
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
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
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
      0
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
      0,
      0,
      1,
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
      0
    ],
    [
      0,
      1,
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3528,
  "candidates_token_count": 622,
  "total_token_count": 6924
}
```
