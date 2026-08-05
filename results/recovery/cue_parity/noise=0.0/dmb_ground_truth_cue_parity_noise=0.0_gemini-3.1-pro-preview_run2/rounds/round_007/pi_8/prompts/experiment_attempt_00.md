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
**Description:** Strict Tallying with Validity Tie-Breaker: Decision-makers primarily rely on the unweighted tally of positive features to evaluate options. They always compute this simple tally first. If one option has a strictly higher tally, they choose it (subject to decision noise), ignoring cue validities entirely. Only when the tallies are exactly equal do they incorporate cue validities as a tie-breaker, utilizing either a lexicographic (Take-The-Best) or Weighted Additive strategy to resolve the decision.

**Parameters:**
- beta_tally: [0.1, 20.0]
- beta_tie: [0.1, 20.0]
- use_wadd: {0, 1}
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_tally = float(parameters["beta_tally"])
    beta_tie = float(parameters["beta_tie"])
    use_wadd = int(parameters["use_wadd"])
    epsilon = float(parameters["epsilon"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    if tally_a != tally_b:
        scores = np.array([tally_a, tally_b])
        active_beta = beta_tally
    else:
        active_beta = beta_tie
        if use_wadd == 1:
            wadd_a = np.sum(val * a)
            wadd_b = np.sum(val * b)
            scores = np.array([wadd_a, wadd_b])
        else:
            cue_order = np.argsort(-val, kind="stable").tolist()
            winner = None
            for j in cue_order:
                if a[j] > b[j]:
                    winner = 0
                    break
                if b[j] > a[j]:
                    winner = 1
                    break
            if winner == 0:
                scores = np.array([1.0, 0.0])
            elif winner == 1:
                scores = np.array([0.0, 1.0])
            else:
                scores = np.array([0.0, 0.0])
                
    z = active_beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Configural Log-Odds Evidence Accumulation with Bounded Non-Linearity

**Parameters:**
- gamma: [0.0, 4.0]
- lambda_pen: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    lambda_pen = float(parameters["lambda_pen"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Clip validities to avoid division by zero or log(1)
    v_clipped = np.clip(val, 0.5001, 0.9999)
    
    # Calculate log-odds (Naive Bayes evidence)
    log_odds = np.log(v_clipped / (1.0 - v_clipped))
    
    # Apply non-linear scaling to capture individual differences in extreme cue weighting
    w = log_odds ** gamma
    
    # Configural penalty: missing cues interact, heavily penalizing options with multiple absent cues
    penalty_a = lambda_pen * (np.sum(w * (1.0 - a))) ** 2
    penalty_b = lambda_pen * (np.sum(w * (1.0 - b))) ** 2
    
    score_a = np.sum(w * a) - penalty_a
    score_b = np.sum(w * b) - penalty_b
    
    scores = np.array([score_a, score_b])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
[0] To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design with descending validities. TTB decides solely based on the highest-validity cue that discriminates between the options. Tallying ignores validities and simply counts the number of features favoring each option. The trials are constructed so that one option always wins on the highest-validity discriminating cue (favored by TTB), while the other option wins on a greater number of lower-validity cues (favored by Tallying), or the total tally results in a tie while TTB makes a deterministic choice.

[1] To strongly dissociate Tallying (which gives equal weight to all features and chooses based on the simple sum of winning features) from Take The Best (which relies exclusively on the single highest-validity discriminating cue), we use a 5-feature design with descending validities. We construct trial pairs where one option wins on the highest-validity discriminating feature (so TTB strongly prefers it) but loses on a larger number of lower-validity features (so Tallying prefers the other option). We also include trials where Tallying results in a tie (no preference) while TTB makes a deterministic choice based on the first discriminating cue.

[2] This design quantitatively dissociates Take The Best (TTB) from Weighted Additive (WADD) theory by exploiting TTB's strict one-reason decision making. Under TTB, the choice probability depends only on the first discriminating cue and is entirely insensitive to the number or weight of any lower-validity cues. Thus, TTB predicts identical choice probabilities across trials where the top discriminating cue favors an option, regardless of whether the other option is favored by one, none, or all remaining lower-validity cues. WADD, by contrast, integrates all features. Because it scales validities but does not discard them, WADD predicts that the choice probability will vary systematically with the margin of integrated scores (e.g., an option will be chosen less frequently if it is opposed by many lower-validity cues compared to when it is unopposed). By holding the TTB choice constant while varying the compensatory evidence, we can cleanly separate the two models.

[3] This design quantitatively dissociates WADD from Take The Best (TTB) by exploiting TTB's invariance to the rank of the highest discriminating cue. Under TTB, the choice probability is determined solely by the first cue that discriminates. Because TTB assigns a fixed binary score (1 for winner, 0 for loser) before applying response noise, it predicts exactly the same choice probability across trials regardless of whether the decision is triggered by the 1st, 2nd, 3rd, or 4th most valid cue. WADD, conversely, computes a compensatory score based on scaled validities. It predicts that the choice probability will systematically decrease as the validity of the highest discriminating cue decreases. By pitting a shifting highest discriminating cue (from most to least valid) against a constant lowest-validity opposing cue, we can cleanly test whether choice confidence is invariant to the cue's rank (TTB) or proportional to its scaled validity (WADD).

[4] This design quantitatively dissociates the Strategy Mixture model from WADD by exploiting a fundamental difference in how they process cue validities. The Strategy Mixture model combines Take The Best (TTB) and Tallying. Crucially, its TTB component is completely invariant to the absolute validity of the highest discriminating cue (it assigns a fixed 1/0 score to the winner), and its Tallying component is invariant to validities entirely. Therefore, the Mixture model predicts that choice probabilities will remain identical across trials as long as the TTB winner is unchanged and the Tally difference is held constant. WADD, by contrast, scales validities and integrates them; it predicts that choice probabilities will systematically decline if the highest discriminating cue shifts to a lower-validity feature, even if the Tally difference is constant. We present a 3x3 grid of trials varying the rank of the highest discriminating cue (1st, 2nd, or 3rd) and the Tally difference (-1, 0, +1). Strategy Mixture predicts three perfectly flat lines across cue ranks, whereas WADD predicts three downward-sloping lines.

[5] This design quantitatively dissociates the Weighted Additive (WADD) theory from the Strategy Mixture (TTB + Tallying) model by exploiting how secondary cues are integrated. Across a set of trials, we hold two things strictly constant: the highest discriminating cue (which fixes the TTB choice) and the total number of cues favoring each option (which fixes the Tallying difference at zero). Because the Strategy Mixture model's TTB component only cares about the top cue and its Tallying component only counts the sheer number of wins, it predicts identical choice probabilities across all these trials. WADD, however, integrates the scaled validities of all cues. By systematically swapping which secondary cues belong to Option A versus Option B, WADD predicts a steep, systematic change in choice probabilities as the secondary support for the TTB winner increases or decreases in validity.

[6] This design quantitatively dissociates Sequential Cue Accumulation (SCA / Take-The-Best with Verification) from Weighted Additive (WADD) theory by probing for a 'dead zone' of guessing. SCA posits a threshold of evidence; if the accumulated evidence across all cues fails to reach this threshold (e.g., due to highly contradictory cues), the decision-maker defaults to exact 50/50 guessing. WADD, conversely, always computes a graded choice probability based on the integrated score difference, no matter how small. By presenting a spectrum of trials with highly contradictory cues that result in very small but non-zero net validity differences, SCA predicts a flat 50% choice rate for all trials falling under the threshold, whereas WADD predicts a smooth, graded curve of probabilities that strictly tracks the small variations in net evidence.

[7] This design quantitatively dissociates Weighted Additive (WADD) theory from Sequential Cue Accumulation (SCA / Take-The-Best with Verification) by probing for 'early stopping' behavior. SCA assumes decision-makers accumulate evidence sequentially by validity and stop searching as soon as a threshold is reached. If the first few high-validity cues strongly favor one option and exceed this threshold, SCA completely ignores all remaining lower-validity cues, predicting identical choice probabilities across trials regardless of how the remaining cues are distributed. WADD, by contrast, integrates all information. It predicts that choice probabilities will systematically shift as the distribution of lower-validity cues changes, even when the top cues provide overwhelming evidence. By fixing the top two highly valid cues to favor Option A and systematically varying whether the remaining four cues favor A or B, we can cleanly test whether lower-validity cues are ignored (SCA) or integrated (WADD).

[8] This design quantitatively dissociates 'Tallying with Validity Tie-Breaking' from Weighted Additive (WADD) theory by contrasting trials where Tallying and WADD conflict, and by probing the nature of tie-breaking. Under the advocated theory, decision-makers strictly choose the option with the most positive cues (Tallying) regardless of validities; validities are only consulted via Take-The-Best (TTB) when the tallies are tied. Therefore, the advocated model predicts that choice probabilities will be identical across all tally-tie trials as long as TTB selects a winner, regardless of the validity of the tie-breaking cue. WADD, by contrast, integrates all scaled validities. It predicts graded choice probabilities that systematically decrease as the highest discriminating cue shifts to lower validities. Furthermore, in trials where one option has more total cues but the other has fewer, higher-validity cues, the advocated model predicts a categorical preference for the high-tally option, whereas WADD predicts a preference for the high-validity option.

[9] This design quantitatively dissociates the advocated Weighted Additive (WADD) theory from the competing 'Tallying with Validity Tie-Breaking' model by contrasting scenarios where the two models fundamentally disagree on the integration of evidence. The competing theory posits a strict lexical process: it first compares the sheer number of positive cues (Tallying), and only if these are equal does it consult validities via Take-The-Best (TTB). Thus, it predicts categorical, invariant choice probabilities across all trials where the tally difference is the same (e.g., all trials where A has more cues than B, or all tally-tie trials where TTB favors A). WADD, in contrast, integrates all exponentially scaled validities. It predicts graded choice probabilities that systematically shift as the validities of the specific cues favoring each option change, regardless of the raw tally. By presenting pairs where Option A wins the tally but loses on high-validity cues, and tally-tie pairs where the TTB-determining cue shifts down the validity hierarchy, we can cleanly test whether choices are categorically driven by tallies/ties or continuously driven by integrated validities.

[10] This design quantitatively dissociates the advocated Dual-Process Strategy Selection theory from the competing Weighted Additive (WADD) theory by probing for a discrete threshold in evidence integration. The advocated theory posits that if the raw tally difference between two options meets or exceeds a certain threshold, the decision-maker entirely ignores validities and relies purely on the tally difference. If the tally difference is below the threshold, they fall back to a validity-sensitive strategy (TTB or WADD). WADD, in contrast, continuously integrates scaled validities across all trials regardless of the tally difference. By presenting pairs of trials where the tally difference is held constant (at 0, 1, 2, 3, or 4) but the validities of the discriminating features vary wildly (e.g., Option B holds the highest validity cue in one trial, but the lowest in the other), we can cleanly test for this threshold. The advocated theory predicts that once the tally difference is large enough, choice probabilities will suddenly become identical across matched-tally pairs, completely insensitive to the validities. WADD predicts continuous sensitivity to validities (and thus different choice probabilities) across all pairs.

[11] To quantitatively dissociate the advocated Weighted Additive (WADD) theory from the competing Dual-Process Strategy Selection theory, we probe for the existence of a discrete tally-difference threshold. The competing theory posits that if the raw tally difference between two options meets or exceeds a certain threshold (from 1 to 5), the decision-maker entirely ignores cue validities and relies purely on the unweighted tally difference. WADD, in contrast, continuously integrates scaled validities across all trials regardless of the tally difference. By presenting pairs of trials where the tally difference is held constant (at 1, 2, 3, and 4) but the validities of the discriminating features vary drastically (e.g., Option B holds the highest validity cue in one trial, but the lowest in the other), we can cleanly test for this threshold. The competing theory predicts that once the tally difference reaches the threshold, choice probabilities will suddenly become identical across matched-tally pairs, completely insensitive to the validities. WADD predicts continuous sensitivity to validities (and thus divergent choice probabilities) across all pairs.

[12] This design quantitatively dissociates the advocated Dual-Process Strategy Selection theory from the competing Strict Tallying theory by exploiting their divergent handling of small tally differences. The competing theory posits that any non-zero tally difference strictly dictates the choice, ignoring validities entirely. Thus, across trials where the tally difference is exactly 1 (e.g., Option A has 2 cues, Option B has 3 cues), the competing theory predicts invariant choice probabilities favoring the higher-tally option. The advocated theory, however, includes a variable threshold. If the tally difference is below this threshold (e.g., a threshold of 2), the decision-maker falls back to a validity-sensitive strategy (TTB or WADD). By presenting multiple trials with a constant tally difference of 1 but drastically shifting which option holds the highest-validity cues, the advocated theory predicts robust, validity-driven shifts in choice probabilities, whereas the competing theory predicts completely flat, tally-driven responses.

[13] To quantitatively dissociate the advocated 'Strict Tallying with Validity Tie-Breaker' theory from the competing 'Dual-Process Strategy Selection' theory, we probe the core difference in their threshold mechanisms. The advocated theory asserts that ANY non-zero tally difference strictly determines the choice, with validities only used to break perfect ties. The competing theory posits a variable threshold (1 to 5), predicting that for small tally differences (e.g., 1, 2, or 3), decision-makers might fall back to a validity-sensitive strategy (TTB or WADD). We present a series of trials where Option A has a lower tally but possesses the single highest-validity cue, while Option B has a higher tally built from lower-validity cues. By systematically varying the tally difference (1, 2, 3) against the validities, the advocated theory rigidly predicts the higher-tally option will always be chosen, regardless of the difference magnitude. The competing theory predicts a reversal in choice (favoring the high-validity, low-tally option) whenever the tally difference falls below the subject's threshold.

[14] To quantitatively dissociate the advocated 'Configural Log-Odds Evidence Accumulation' theory from the competing 'Strict Tallying with Validity Tie-Breaker' theory, we systematically vary the raw tally difference against the log-odds validity of the cues, and manipulate the distribution of missing cues. The competing theory posits that any non-zero tally difference strictly dictates the choice, ignoring validities entirely, and only uses validities (via TTB or WADD) when tallies are exactly tied. The advocated theory, however, evaluates options using non-linearly scaled log-odds validities and applies a squared configural penalty for missing cues. By presenting pairs where one option has a higher tally but the other holds the highest-validity cue, the competing theory predicts a strict preference for the higher-tally option, whereas the advocated theory predicts graded responses that can favor the lower-tally option. Furthermore, in tally-tie trials, the advocated theory's configural penalty heavily punishes options missing the most critical cues, leading to divergent predictions from simple linear WADD or TTB tie-breaking.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the advocated 'Strict Tallying with Validity Tie-Breaker' theory from the competing 'Configural Log-Odds Evidence Accumulation' theory, we construct trials that pit raw tally differences against extreme log-odds validities and configural penalties. The advocated theory strictly chooses the option with the higher tally, completely ignoring validities unless there is a perfect tie. The competing theory uses log-odds scaling and a squared configural penalty for missing cues, meaning an option with a lower tally but possessing the most valid cues will be preferred, as the higher-tally option will suffer a massive penalty for missing the critical high-validity cues. By varying the tally difference (e.g., A has 2 cues but the highest validities, B has 3 or 4 cues with lower validities) and including tally-tie trials where the squared penalty diverges from linear tie-breaking, we can cleanly separate the two models.",
  "validities": [
    0.95,
    0.85,
    0.7,
    0.6,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
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
      1
    ],
    [
      1,
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
      0
    ],
    [
      0,
      1,
      1,
      1,
      1
    ],
    [
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
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
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
      1,
      0
    ],
    [
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
      1,
      1
    ],
    [
      1,
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
      0
    ],
    [
      0,
      1,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 5432,
  "candidates_token_count": 544,
  "total_token_count": 7203
}
```
