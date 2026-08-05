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
**Description:** First-Trial Inertia / Rational Disengagement: Subjects do not possess an innate spatial bias prior to the experiment. Instead, they actively evaluate the very first trial using a simple heuristic (like Tallying or Take-The-Best) to make an initial choice. However, due to the lack of trial-by-trial feedback and the repetitive nature of the multi-attribute stimuli, they rationally disengage after this first trial to minimize cognitive effort. For all subsequent trials, they exhibit 'first-trial inertia' by simply repeating their initial motor action. This produces 100% perseveration at the individual level, but grounds the origin of the bias in an initial stimulus-driven evaluation rather than an arbitrary trait.

**Parameters:**
- heuristic_type: {0, 1}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # If this is the first trial, evaluate using a simple heuristic
    if len(history.get("response", [])) == 0:
        state = np.asarray(state, dtype=float)
        a, b = state[0], state[1]
        
        heuristic_type = int(parameters["heuristic_type"])
        score_a, score_b = 0, 0
        
        if heuristic_type == 0:
            # Tallying (equal weights)
            score_a = np.sum(a)
            score_b = np.sum(b)
        else:
            # Take-The-Best (lexicographic)
            val = np.asarray(parameters["validities"], dtype=float)
            order = np.argsort(val)[::-1]
            for idx in order:
                if a[idx] > b[idx]:
                    score_a = 1
                    break
                elif b[idx] > a[idx]:
                    score_b = 1
                    break
                    
        if score_a > score_b:
            return np.array([1.0, 0.0])
        elif score_b > score_a:
            return np.array([0.0, 1.0])
        else:
            return np.array([0.5, 0.5])
            
    else:
        # Rational disengagement: repeat the initial motor action
        first_resp = history["response"][0]
        if first_resp == 0:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])
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
**Description:** Extreme Spatial Bias / Key Perseveration: Subjects completely disengage from the multi-attribute decision task and instead adopt a deterministic response strategy. They choose the exact same option (either always Option A or always Option B) on every single trial, completely ignoring the stimuli and feature validities. The preference for Option A versus Option B is fixed per subject, creating a population split between 'Always-A' and 'Always-B' responders. At the individual level, the choice policy is entirely deterministic and repetitive.

**Parameters:**
- preferred_option: {0, 1}

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The subject adopts a deterministic response strategy, always choosing the same option.
    pref = int(parameters["preferred_option"])
    if pref == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, the experimental design pits the highest-validity discriminating cue against the sheer number of winning cues. In TTB, the option with the higher value on the most valid discriminating cue is chosen, ignoring all other cues. In Tallying, the option that has a higher value on more cues is chosen, ignoring cue validities completely. The trials are constructed such that one option wins on the single most valid discriminating cue, while the other option wins on a larger number of less valid cues. This leads to completely opposite predictions between the two models.

[1] This design systematically pits the single most valid discriminating cue against the overall number of winning cues. In Take The Best (TTB), only the highest validity feature that discriminates between the two options is considered, completely ignoring the sheer volume of supporting features. In contrast, Tallying ignores feature validities and simply counts the number of features favoring each option. The trial set includes pairs where TTB strongly prefers one option while Tallying prefers the other (due to a larger number of lower-validity wins), as well as pairs where Tallying predicts a tie (equal number of winning features) but TTB makes a deterministic prediction based on the highest-validity cue.

[2] To quantitatively dissociate Take The Best (TTB) from a Weighted Additive (WADD) strategy that is subject to strong spatial biases and lapses, we design trials where the single highest-validity discriminating cue strongly opposes the sum of the remaining lower-validity cues. In TTB, the choice is deterministically driven by the first discriminating cue in descending order of validity, completely ignoring the cumulative weight of all other cues. In WADD, the sum of the lower-validity cues can easily outweigh the single highest-validity cue. By symmetrically alternating whether Option A or Option B possesses the highest-validity cue or the bulk of the lower-validity cues, we can also separate true WADD integration from a simple Option A positional bias.

[3] This design dissociates WADD (with its potential for spatial bias and feature integration) from Take The Best (TTB). We include 'compensatory' trials where the highest-validity cue points to one option while the sum of the remaining cues points to the other, separating TTB's non-compensatory one-reason decision making from WADD's integration. Crucially, we also include 'tie' trials where both options have identical feature profiles; TTB must guess (predicting exactly 50/50), whereas the advocated WADD model will reveal any underlying spatial/positional bias (e.g., a baseline preference for Option A).

[4] To quantitatively dissociate the Advocated Theory (pure random guessing / complete disengagement) from the Competing Theory (WADD with spatial bias and lapse rate), this design includes three types of trials: (1) Extreme dominance trials where Option A or Option B has all positive features, maximizing the WADD difference. (2) Compensatory trials where the highest-validity cue opposes the sum of lower-validity cues, testing sensitivity to weighted feature integration. (3) Identical-profile (tie) trials where both options have the exact same features. The Advocated Theory rigidly predicts exactly 50/50 choice probabilities across all trial types. In contrast, the Competing Theory will predict strong deviations from 50/50 on dominance and compensatory trials (via the beta parameter) and will reveal any baseline positional preference on the identical-profile trials (via the bias_A parameter).

[5] To quantitatively dissociate the Advocated Theory (WADD with spatial bias and lapse rate) from the Competing Theory (pure random guessing), we must expose variations in choice probabilities that random guessing cannot capture. The Competing Theory rigidly predicts a 50/50 choice split on every trial, regardless of the stimuli. The Advocated Theory predicts that choices will track the weighted sums of features (modulated by sensitivity 'beta') and may exhibit a baseline preference for Option A (via 'bias_A'). We include extreme dominance trials (where one option has all positive features and the other has none) to elicit maximum deviation from 50/50. We also include compensatory trials to test sensitivity to feature integration, and identical-profile (tie) trials where both options have the same features. In tie trials, random guessing still predicts exactly 0.5, whereas the Advocated Theory will reveal any underlying spatial bias.

[6] To quantitatively dissociate the Advocated Theory (Extreme Spatial Bias / Key Perseveration) from the Competing Theory (WADD with spatial bias and lapse rate), this design introduces extreme variations in the objective quality (weighted feature sum) between Option A and Option B across trials. The Advocated Theory predicts that an individual subject will rigidly choose the exact same option (either 100% A or 100% B) on every single trial, completely ignoring the stimulus features. In contrast, the Competing Theory predicts that unless the subject is completely insensitive to the stimuli (beta=0), their probability of choosing Option A will systematically vary according to the difference in weighted additive scores between the options. By including trials where Option A strongly dominates Option B, trials where Option B strongly dominates Option A, and mixed compensatory trials, we can cleanly separate individuals who are merely biased but still integrating features (Competing Theory) from those who are entirely perseverative and stimulus-blind (Advocated Theory).

[7] To quantitatively dissociate the Advocated Theory (WADD with spatial bias and lapse rate) from the Competing Theory (Extreme Spatial Bias / Key Perseveration), this design introduces extreme variations in the objective quality (weighted feature sum) between Option A and Option B across trials. The Competing Theory rigidly predicts that an individual subject will deterministically choose the exact same option (either 100% A or 100% B) on every single trial, completely ignoring the stimulus features. In contrast, the Advocated Theory predicts that unless the subject is completely insensitive to the stimuli (beta=0), their probability of choosing Option A will systematically vary according to the difference in weighted additive scores between the options. By including trials where Option A strongly dominates Option B, trials where Option B strongly dominates Option A, mixed compensatory trials, and tie trials, we can cleanly separate individuals who are merely biased but still integrating features (Advocated Theory) from those who are entirely perseverative and stimulus-blind (Competing Theory).

[8] To dissociate the Advocated Theory (innate extreme spatial bias) from the Competing Theory (first-trial inertia based on heuristic evaluation), we exploit the randomized trial order. We include extreme dominance trials alongside compensatory trials that dissociate TTB from Tallying. The Competing Theory predicts that a subject's lifelong perseveration is perfectly predicted by the heuristic evaluation of their randomly assigned *first* trial (e.g., if the first trial is one where Option B dominates, they must become an 'Always-B' responder). The Advocated Theory predicts that the perseverated choice is an innate trait and thus entirely independent of the first trial's stimulus properties, allowing for 'Always-A' responders even when their very first trial objectively and strongly favored Option B.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate the Advocated Theory (First-Trial Inertia based on heuristic evaluation) from the Competing Theory (innate Extreme Spatial Bias), we exploit the randomized trial order. Both theories predict that an individual will make the exact same choice (e.g., always A or always B) across all trials. However, the Advocated Theory predicts that this perseveration is caused by the heuristic evaluation of the randomly assigned *first* trial. Thus, if a subject's first trial objectively and strongly favors Option A, they must become an 'Always-A' responder. The Competing Theory predicts that the perseveration is an innate trait, meaning a subject might be an 'Always-B' responder even if their very first trial strongly favored Option A. By including extreme dominance trials for both A and B, as well as compensatory trials, we can test whether a subject's lifelong perseveration is perfectly predicted by the stimulus properties of their first trial.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
    [
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
      0,
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
      0
    ],
    [
      0,
      0,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
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
      1
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
      1
    ],
    [
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
  "prompt_token_count": 3523,
  "candidates_token_count": 473,
  "total_token_count": 5346
}
```
