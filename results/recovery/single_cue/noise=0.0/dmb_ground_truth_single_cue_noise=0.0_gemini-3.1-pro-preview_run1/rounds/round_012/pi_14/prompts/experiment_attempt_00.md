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
**Description:** Decision-making in binary choice tasks is driven by a probabilistic mixture of two primary heuristics: Tallying and Take-The-Best (TTB). On any given trial, a subject may evaluate the options by counting the number of winning features (Tallying) or by relying solely on the single highest-validity discriminating cue (TTB). When Tallying results in a tie, its choice probabilities become uniform, allowing the TTB preference to naturally act as a tie-breaker without requiring a separate, contrived tie-breaking mechanism. To maintain the strong dominance of Tallying observed in non-tie scenarios, the prior probability of using TTB is constrained to be relatively low.

**Parameters:**
- validities: validities
- p_ttb: [0.0, 0.3]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    p_ttb = float(parameters["p_ttb"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Take-The-Best (TTB): rely on the highest validity discriminating cue
    # Use stable sort to preserve left-to-right order for equal validities
    order = np.argsort(-validities, kind='stable')
    ttb_a, ttb_b = 0.0, 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_b = 1.0
            break
            
    # Tallying: count of strict feature-wise wins
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    
    # Compute probabilities for each heuristic
    z_ttb = beta_ttb * np.array([ttb_a, ttb_b])
    probs_ttb = np.exp(z_ttb - np.max(z_ttb))
    probs_ttb /= np.sum(probs_ttb)
    
    z_tally = beta_tally * np.array([tally_a, tally_b])
    probs_tally = np.exp(z_tally - np.max(z_tally))
    probs_tally /= np.sum(probs_tally)
    
    # Mix the strategies
    mixed_probs = p_ttb * probs_ttb + (1.0 - p_ttb) * probs_tally
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
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
**Description:** Inverse Validity Tie-Breaking with Non-linear Scaling: Decision-makers primarily rely on a Tallying heuristic. When this primary mechanism results in a tie, subjects do not guess randomly, but exhibit a recency bias or systematically misinterpret the cue validities, breaking the tie by heavily weighting the lower-validity (or right-most) features. A non-linear scaling parameter exaggerates this inverse-validity preference to better capture the magnitude of the recency effect.

**Parameters:**
- validities: validities
- w_tie: [0.0, 0.95]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 10.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    w_tie = float(parameters["w_tie"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    # Primary mechanism: Tallying (count of strict wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Inverse Validity Tie-Breaker
    # Weight lower-validity features more heavily, with a non-linear scaling (gamma)
    tie_weights = (1.0 - validities) ** gamma
    if np.sum(tie_weights) > 0:
        tie_weights /= np.sum(tie_weights)
    else:
        tie_weights = np.ones_like(tie_weights) / len(tie_weights)
        
    tie_score_a = np.sum(a * tie_weights)
    tie_score_b = np.sum(b * tie_weights)
    
    # Combine scores. Since w_tie < 1.0 and tie_score difference is <= 1.0,
    # the tie-breaker will never override a strict Tallying win (difference >= 1.0).
    score_a = a_wins + w_tie * tie_score_a
    score_b = b_wins + w_tie * tie_score_b
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
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
[0] To maximally dissociate Take The Best (TTB) from Tallying, we use 5 binary features with monotonically decreasing validities. This allows us to construct 'compensatory' trials where the single highest-validity discriminating cue favors one option, but a larger number of lower-validity cues favor the other option. In these critical trials, TTB will choose the option favored by the top cue, whereas Tallying will choose the opposite option based on the sheer count of winning features. We also include trials where Tallying ties (equal number of winning features) but TTB has a strong preference, as well as baseline trials where both models agree.

[1] To cleanly dissociate Tallying from Take The Best (TTB), we use 4 features with descending validities. We design trials where the highest valid discriminating cue points to one option, but a larger number of lower-validity cues point to the other. In these cases, TTB chooses based on the single most valid cue, while Tallying sums the feature-wise wins and chooses the opposite option. We also include trials where Tallying results in a tie (equal number of feature wins) but TTB has a strict preference, allowing us to capture differences in both deterministic choice and guessing behavior.

[2] To quantitatively dissociate the Weighted Additive (WADD) model from Tallying, we exploit their divergent treatment of cue validities. Tallying treats all features equally, merely counting the number of features on which one option strictly dominates the other. WADD, by contrast, weights each feature's contribution by its explicitly provided validity. We employ 5 binary features with a steep drop-off in validities (e.g., two highly valid cues and three less valid cues). The critical trials pair an option that wins on the two highly valid cues against an option that wins on the three less valid cues. Tallying predicts a preference for the latter (3 wins vs 2 wins), whereas WADD predicts a preference for the former due to the larger summed validity. We also include trials where the options tie in the number of winning features (leading Tallying to guess uniformly) but differ in their weighted sums, allowing WADD to make a deterministic prediction.

[3] To quantitatively dissociate Tallying from WADD, we use 5 binary features where two cues have very high validity (0.95) and three have low validity (0.55). Tallying ignores validities and simply counts the number of winning features, whereas WADD integrates the validities into a weighted sum. The critical dissociation trials pit an option that wins on the three low-validity features against an option that wins on the two high-validity features. Tallying predicts a preference for the former (3 wins vs 2 wins), whereas WADD predicts a strong preference for the latter (weighted sum 1.90 vs 1.65). We also include trials where Tallying results in a tie (equal number of winning features) but WADD has a strong preference, as well as baseline trials where both models agree.

[4] To robustly dissociate Take The Best (TTB) from Tallying, we employ 5 binary features with monotonically decreasing validities. TTB decides based solely on the single most valid discriminating cue, ignoring all others. Tallying, on the other hand, ignores validities and simply counts the number of features favoring each option. We designed a set of trials where the highest-validity discriminating cue points to one option, but a strictly greater number of lower-validity cues point to the other option. In these 'compensatory' trials, TTB and Tallying will choose opposite options. We also include trials where Tallying results in a tie (equal number of winning features) but TTB has a strict preference, yielding differences in deterministic choice versus guessing.

[5] This design explicitly pits Tallying against Take The Best (TTB) by systematically manipulating the number of winning features versus the validity of the highest discriminating feature. We employ 5 features with monotonically decreasing validities. In the critical dissociation trials, one option is favored by the single most valid cue, but the other option is favored by a strictly greater number of lower-validity cues. TTB will choose the former, while Tallying (which ignores validities and simply counts feature-wise wins) will choose the latter. We also include trials where the options tie in the number of winning features (leading Tallying to predict a 50/50 guess) but differ on the highest validity cue (leading TTB to make a deterministic choice). Finally, baseline trials where both models agree are included to ensure basic task engagement.

[6] To dissociate the advocated WADD theory (with its flexible gamma parameter) from Tallying, we use 5 features with linearly decreasing validities. Tallying simply counts the number of winning features, ignoring validities. WADD computes a weighted sum where weights are validities exponentiated by gamma. We include 'compensatory' trials where one option wins on fewer but more valid features (e.g., the top 2) while the other wins on more but less valid features (e.g., the bottom 3). Tallying strictly predicts the latter, whereas WADD predicts the former for typical positive gamma values. We also include trials that result in a tie under Tallying but yield a strict preference under WADD, and trials that differentiate low gamma (close to Tallying) from high gamma (close to Take-The-Best) to fully exercise the advocated model's parameter space.

[7] To quantitatively dissociate Tallying from the Weighted Additive (WADD) model, we exploit their fundamentally different treatment of feature validities. Tallying completely ignores validities, relying only on the count of features where one option dominates another. WADD, conversely, weights features by their validities (exponentiated by a free parameter gamma). We use 5 features with a large spread in validities: two high-validity experts (0.95, 0.90) and three low-validity experts (0.60, 0.55, 0.50). The trial set includes 'compensatory' trials where one option wins on more features (favored by Tallying) but the competing option wins on fewer, much higher-validity features (favored by WADD). We also include 'tie' trials where both options win on the same number of features—causing Tallying to predict a uniform guess—but where the weighted sums differ, allowing WADD to predict a deterministic preference. Variations in these combinations target the varying gamma parameter of the WADD model.

[8] To quantitatively dissociate the advocated mixture model (TTB + Tallying) from the competing pure Tallying model, we use 5 features with linearly decreasing validities. Pure Tallying strictly counts feature-wise wins, predicting a choice for the option with more wins, or exactly 50/50 guessing when the counts tie, entirely ignoring validities. The advocated mixture model, however, includes a Take-The-Best (TTB) component that bases decisions on the single most valid discriminating cue. We designed trials where the option with fewer total wins is favored by the highest-validity cue (e.g., A wins on the single best cue, B wins on three lower-validity cues). Pure Tallying strongly predicts B, while the mixture model predicts a shifted probability toward A depending on the mixture weight (p_ttb). We also include 'tie' trials where both options win on the same number of features, leading pure Tallying to guess, but where the mixture model predicts a deterministic preference due to the TTB component.

[9] To cleanly dissociate the advocated Tallying model from the competing TTB+Tallying mixture model, we exploit their differing predictions on 'Tally Tie' and 'Compensatory' trials. Tallying ignores cue validities and relies purely on counting the number of features each option wins on. When the number of winning features is equal, Tallying predicts a 50/50 guess. The mixture model, however, includes a Take-The-Best (TTB) component that breaks ties deterministically based on the single most valid discriminating cue. We include trials where Tallying ties but TTB has a strict preference, causing the mixture model to predict a directional preference proportional to the mixture weight (p_ttb). We also include compensatory trials where the option with fewer overall wins is favored by the highest-validity cue. Tallying strongly predicts the option with more wins, whereas the mixture model predicts a softer probability or even a flipped preference depending on the TTB weight.

[10] To dissociate the advocated Thresholded Weighted Additive (WADD) model from the competing Tallying model, we use 5 features with a steep drop-off in validities (two high, three low). The advocated model acts as WADD when the weighted difference between options exceeds a threshold, but falls back to Tallying when the options are perceived as similar in overall value. We design trials with varying degrees of weighted value differences (wadd_diff) and opposing Tallying predictions. For example, when Option A wins on the two high-validity features and B wins on the three low-validity features, WADD strongly prefers A while Tallying prefers B. By including trials where the wadd_diff is very small but Tallying has a strict preference, we can observe the thresholding behavior: if the difference is below the subject's threshold, they will fall back to Tallying, aligning with the competing model; if above, they will choose based on WADD. This provides a clear quantitative signature of the threshold parameter.

[11] To effectively dissociate the advocated Tallying model from the competing Thresholded Weighted Additive (WADD) model, we exploit the thresholding mechanism and validity-weighting of the latter. Tallying ignores cue validities and relies purely on the count of winning features. If the counts tie, it guesses uniformly. The competing model uses WADD if the normalized difference between options exceeds a threshold; otherwise, it falls back to Tallying. We use 5 features with a steep validity drop-off. We include 'Compensatory' trials where Option A wins on the two highest-validity features but B wins on the three lowest-validity features (Tallying prefers B, WADD prefers A). Crucially, we also include 'Tally Tie' trials where both options win on an equal number of features but have large WADD differences. In these trials, Tallying predicts a strict 50/50 guess, whereas the competing model will predict a deterministic choice if the WADD difference exceeds the threshold, allowing us to quantitatively identify the threshold and the underlying strategy.

[12] To quantitatively dissociate the Advocated Theory (a mixture of Left-to-Right Take-The-Best, Tallying, and a 'Top-Heavy Penalty') from the Competing Theory (Pure Tallying), we exploit the unique structural sensitivities of the advocated model. Pure Tallying strictly counts feature-wise wins and guesses uniformly on ties, completely ignoring the spatial/index position of the features. The Advocated Theory, however, uses the Left-to-Right (LTR) primacy rule and a Top-Heavy Penalty (THP) that explicitly favors features further to the right (higher indices). Crucially, when Tallying results in a tie, the Advocated Theory boosts the weight of the THP. We include 'Tally Tie' trials where Tallying predicts exactly 50/50, but LTR and THP pull strongly in specific directions (e.g., LTR prefers A, but THP strongly prefers B). We also include 'Compensatory' trials where Tallying prefers one option (e.g., 3 wins vs 2 wins), but LTR and/or THP favor the other option, allowing us to tease apart the specific mixture components and the conditional tie-breaker boost.

[13] To quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (a mixture of Left-to-Right Take-The-Best, Tallying, and a Top-Heavy Penalty), we exploit the Competing Theory's unique sensitivity to feature order and its conditional tie-breaking mechanism. Pure Tallying counts feature-wise wins, strictly ignoring feature validities and positions, and guesses uniformly when the win counts are equal. The Competing Theory uses an LTR primacy rule (favoring early features) and a THP anti-primacy rule (favoring late features), and crucially boosts the THP weight when Tallying results in a tie. We designed trials that pit these mechanisms against each other. For example, in 'Tally Tie' trials, Tallying predicts a 50/50 guess, but LTR and THP pull in opposite directions, allowing us to observe the THP tie-breaker boost. In 'Compensatory' trials, Tallying favors one option based on win count, but LTR and/or THP strongly favor the other option due to feature positioning. Pure Tallying will stubbornly track the win count, while the Competing Theory will exhibit predictable shifts based on the spatial distribution of the features.

[14] To quantitatively dissociate the Advocated Theory (Tallying with Recency Tie-Breaker) from the Competing Theory (Pure Tallying), we focus on trials where the primary Tallying score results in a tie. Both theories rely on Tallying as the primary decision mechanism, meaning they will strongly agree when one option has strictly more feature-wise wins than the other. However, when the feature-wise wins are tied, Pure Tallying predicts exactly a 50/50 guess, while the Advocated Theory uses a secondary recency (anti-primacy) tie-breaker, favoring the option whose winning features appear later in the sequence. By systematically varying the positions of the winning features in 'Tally Tie' trials, we can observe deterministic shifts in choice probabilities under the Advocated Theory that Pure Tallying cannot explain.

[15] To quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Tallying with Recency Tie-Breaker), we focus on trials where the primary Tallying score results in a tie. Both theories rely on Tallying as the primary decision mechanism, meaning they will strongly agree when one option has strictly more feature-wise wins than the other. However, when the feature-wise wins are tied, Pure Tallying predicts exactly a 50/50 guess, while the Competing Theory uses a secondary recency (anti-primacy) tie-breaker, favoring the option whose winning features appear later in the sequence. By systematically varying the positions of the winning features in 'Tally Tie' trials, we can observe deterministic shifts in choice probabilities under the Competing Theory that Pure Tallying cannot explain.

[16] To quantitatively dissociate the Advocated Theory (Take-The-Best with Thresholded Compensatory Tallying and a Recency Tie-Breaker) from the Competing Theory (Pure Tallying), we exploit the thresholding and tie-breaking mechanisms unique to the advocated model. Pure Tallying always decides based on the strict count of winning features, ignoring feature order entirely, and guesses uniformly when counts tie. The Advocated Theory, however, starts with a Left-to-Right TTB approach and only falls back to Tallying if the number of opposing wins exceeds a threshold. Furthermore, if it falls back to Tallying and the counts tie, it employs a recency (anti-primacy) tie-breaker. We designed 'Compensatory' trials where TTB and Tallying conflict and the number of opposing wins varies (e.g., 2, 3, or 4), allowing us to identify the threshold parameter. We also included 'Tally Tie' trials where Pure Tallying predicts a 50/50 guess, but the Advocated Theory predicts a deterministic preference due to either sticking with the TTB winner (if opposing wins are below threshold) or applying the recency tie-breaker (if above threshold).

[17] To quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Left-to-Right TTB with Thresholded Compensatory Tallying and Recency Tie-Breaker), we focus on two critical types of trials. First, we use 'Tally Tie' trials where both options have an equal number of feature wins. Pure Tallying predicts an exact 50/50 guess on these trials. The Competing Theory, however, will either make a deterministic choice based on the first discriminating feature (TTB) if the opposing wins are below threshold, or use a recency-weighted tie-breaker if above threshold. By carefully arranging the winning features, we pit the TTB prediction against the recency prediction, ensuring the Competing Theory always diverges from 50/50. Second, we include 'Compensatory' trials where the option with fewer wins is favored by the first discriminating cue. Pure Tallying strictly favors the option with more wins, while the Competing Theory may favor the TTB winner depending on the subject's threshold parameter.

[18] To quantitatively dissociate the Soft WADD model (Advocated Theory) from pure Tallying (Competing Theory), we use 5 features with a steep drop-off in validities. Tallying relies solely on the count of winning features, ignoring validities and spatial position, and predicts a uniform guess on ties. The Soft WADD model, however, computes a weighted sum that incorporates both the explicitly provided validities and a spatial position bias. We include 'Compensatory' trials where one option wins on fewer but highly valid features (Soft WADD prefers) while the other wins on more but less valid features (Tallying prefers). We also include 'Tally Tie' trials where both options win on an equal number of features. In these ties, pure Tallying strictly predicts a 50/50 guess, while Soft WADD will predict a deterministic choice driven by the validities and the position bias (w_pos). By carefully balancing the validity sums and position indices across these ties, we can isolate the unique position-dependent bias of the Soft WADD model.

[19] To quantitatively dissociate the Advocated Theory (pure Tallying) from the Competing Theory (Soft WADD with Convex Mixture), we exploit the Competing Theory's sensitivity to both explicitly provided validities and feature position bias. Pure Tallying strictly counts feature-wise wins, entirely ignoring validities and feature indices, and predicts a uniform 50/50 guess when the win counts are tied. The Soft WADD model computes a weighted sum combining a baseline tallying weight, validities, and a position-dependent bias. We include 'Compensatory' trials where one option wins on more features (favored by Tallying) while the other option wins on fewer but highly valid features (favored by Soft WADD). We also include 'Tally Tie' trials where both options win on an equal number of features. In these ties, pure Tallying predicts exact guessing, whereas Soft WADD predicts deterministic preferences driven by validity differences and spatial position biases. By systematically varying which features comprise the ties and compensatory pairs, we can isolate the unique validity and position components of the Soft WADD model against the rigid counting of Tallying.

[20] To quantitatively dissociate the Advocated Theory (Tallying with Strict Validity Tie-Breaking) from the Competing Theory (Pure Tallying), we focus on 'Tally Tie' trials. Both models use Tallying as their primary decision mechanism, so they make identical predictions when one option strictly wins on more features. However, when the feature-wise wins are equal, Pure Tallying predicts a strict 50/50 guess, while the Advocated Theory resolves the tie deterministically using the explicitly provided validities of the winning features. By including trials where both options win on an equal number of features but differ in the validities of those features, we isolate the secondary validity tie-breaker. We also include 'Strict Win' trials where the option with more wins has lower overall validity, confirming that the validity component in the Advocated Theory is strictly bounded and does not override the primary Tallying mechanism.

[21] To quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Tallying with Strict Validity Tie-Breaking), we focus on 'Tally Tie' trials. Both models use Tallying as their primary decision mechanism, so they make identical predictions when one option strictly wins on more features. However, when the feature-wise wins are equal, Pure Tallying predicts a strict 50/50 guess, while the Competing Theory resolves the tie deterministically using the explicitly provided validities of the winning features. By including trials where both options win on an equal number of features but differ in the validities of those features, we isolate the secondary validity tie-breaker. We also include 'Strict Win' trials where the option with more wins has lower overall validity, confirming that the validity component in the Competing Theory is strictly bounded and does not override the primary Tallying mechanism.

[22] To quantitatively dissociate the Advocated Theory (Tallying with an Inverse Validity Tie-Breaker) from the Competing Theory (Pure Tallying), we focus on 'Tally Tie' trials. Both models use Tallying as their primary decision mechanism, so they make identical predictions when one option strictly wins on more features. However, when the feature-wise wins are equal, Pure Tallying predicts a strict 50/50 guess, while the Advocated Theory resolves the tie deterministically by heavily weighting the lower-validity features (an inverse validity bias). By including trials where both options win on an equal number of features but differ in the validities of those features, we isolate the secondary inverse validity tie-breaker. We also include 'Strict Win' trials to confirm that the inverse validity component in the Advocated Theory is bounded and does not override the primary Tallying mechanism.

[23] To quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Tallying with Inverse Validity Tie-Breaker), we focus on 'Tally Tie' trials. Both models use Tallying as their primary decision mechanism, so they make identical predictions when one option strictly wins on more features. However, when the feature-wise wins are equal, Pure Tallying predicts a strict 50/50 guess, while the Competing Theory resolves the tie deterministically by heavily weighting the lower-validity features (an inverse validity bias). By including trials where both options win on an equal number of features but differ in the validities of those features, we isolate the secondary inverse validity tie-breaker. We also include 'Strict Win' trials to confirm that the inverse validity component in the Competing Theory is bounded and does not override the primary Tallying mechanism.

[24] To quantitatively dissociate the Advocated Theory (Tallying with an Inverse Validity Tie-Breaker) from the Competing Theory (a mixture of Tallying and Take-The-Best), we focus on 'Tally Tie' trials. Both models use Tallying as a primary mechanism and will generally agree when one option has strictly more feature-wise wins. However, when the feature-wise wins are tied, the Competing Theory relies on its TTB component, breaking the tie in favor of the option with the highest-validity winning cue. In contrast, the Advocated Theory employs an inverse validity tie-breaker, strongly favoring the option whose winning features have the lowest validities. By pairing high-validity wins against low-validity wins in tie scenarios, we create a stark qualitative divergence in predictions. We also include 'Compensatory' trials where the option with fewer wins is favored by the highest-validity cue; here, the Advocated Theory strictly follows Tallying, while the Competing Theory's TTB mixture component causes a measurable shift in choice probabilities.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Advocated Theory (a mixture of Tallying and Take-The-Best) from the Competing Theory (Tallying with an Inverse Validity Tie-Breaker), we focus on two critical trial types: 'Tally Ties' and 'Compensatory' trials. In 'Tally Tie' trials, both options have an equal number of feature-wise wins. The Advocated Theory resolves these ties naturally via its TTB component, favoring the option that wins on the highest-validity discriminating cue. In stark contrast, the Competing Theory uses an inverse validity tie-breaker, strongly favoring the option that wins on the lowest-validity cues. By pitting high-validity wins against low-validity wins in tie scenarios, we create diametrically opposed predictions. Furthermore, we include 'Compensatory' trials where the option with fewer wins is favored by the highest-validity cue. Here, the Competing Theory strictly follows Tallying (as its tie-breaker cannot override a strict win), whereas the Advocated Theory's TTB mixture component induces a measurable shift in choice probabilities away from the pure Tallying prediction.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55
  ],
  "trial_a_ratings": [
    [
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
      0,
      0,
      0
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
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
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
      1
    ],
    [
      0,
      0,
      0,
      1,
      1
    ],
    [
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
      0
    ],
    [
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
  "prompt_token_count": 7796,
  "candidates_token_count": 522,
  "total_token_count": 9561
}
```
