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
**Description:** Tally-then-TTB (Tally with Validity Tie-Breaker): Decision-makers first compare options by tallying the total number of winning features for each option, ignoring cue validities. If one option has strictly more wins, it is chosen. This captures the compensatory nature of decision-making when there is a clear majority of supporting cues. However, if the tally results in a tie, the decision-maker falls back to a non-compensatory 'Take-The-Best' strategy to break the tie, choosing the option that wins on the single most valid cue. This hybrid model preserves robust Tallying performance on unequal-tally trials while capturing deterministic validity-based tie-breaking on ambiguous trials.

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
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying phase: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    score_a = a_wins
    score_b = b_wins
    
    # Tie-breaking phase: if tallies are equal, use TTB
    if a_wins == b_wins:
        cue_order = np.argsort(-val, kind="stable")
        for idx in cue_order:
            if a[idx] > b[idx]:
                score_a += 1.0
                break
            elif b[idx] > a[idx]:
                score_b += 1.0
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Thresholded WADD (Tally-then-WADD) with Normalized Exponentiated Validities: Decision-makers primarily rely on a non-compensatory Tallying heuristic, counting the number of winning features. If the difference in tally scores exceeds a certain threshold (tau), they make a choice based strictly on this tally difference. If the tally difference is small or zero, they switch to a compensatory Weighted Additive (WADD) strategy. To capture strong non-compensatory choices on tied trials without destabilizing the softmax temperature, the cue validities are exponentiated and normalized. This allows the WADD component to become steep enough to mimic Take-The-Best (TTB) behavior when needed, bridging the gap between compensatory evaluation and strict non-compensatory tie-breaking.

**Parameters:**
- tau: {1, 2}
- beta_tally: [0.1, 20.0]
- beta_wadd: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [1.0, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    tau = float(parameters["tau"])
    beta_tally = float(parameters["beta_tally"])
    beta_wadd = float(parameters["beta_wadd"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    # Tallying scores
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_diff = abs(a_wins - b_wins)
    
    if tally_diff >= tau:
        # Use Tallying
        scores = np.array([a_wins, b_wins])
        beta = beta_tally
    else:
        # Use WADD with normalized exponentiated validities
        val_transformed = val ** gamma
        val_transformed = val_transformed / np.sum(val_transformed)
        wadd_a = np.sum(val_transformed * a)
        wadd_b = np.sum(val_transformed * b)
        scores = np.array([wadd_a, wadd_b])
        beta = beta_wadd
        
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] To dissociate Take The Best (TTB) and Tallying, we use a 4-feature design where the validities are strictly ordered. The trials are constructed such that the single highest-validity discriminating cue points to one option, while the simple sum of winning features (the tally) points to the other option or results in a tie. TTB will reliably choose the option favored by the top discriminating cue, ignoring the sheer number of lower-validity cues, whereas Tallying will choose the option with more total feature wins, ignoring the validities entirely.

[1] To strongly dissociate Tallying (advocated) from Take The Best (competing), we use a 5-feature design with strictly ordered validities. The trials are constructed so that the highest-validity discriminating cue points to one option, while the simple sum of winning features (the tally) points to the other option or results in a tie. TTB will reliably choose the option favored by the top discriminating cue, ignoring the sheer number of lower-validity cues, whereas Tallying will choose the option with more total feature wins, ignoring the validities entirely.

[2] To dissociate Weighted Additive (WADD) from Tallying, we exploit the fact that WADD integrates cue validities into a weighted sum, whereas Tallying simply counts the number of features favoring each option, treating all validities equally. We use a 5-feature design with a clear split between two high-validity cues and three low-validity cues. In key trials, one option wins on the two high-validity cues (resulting in a higher WADD score), while the other option wins on the three low-validity cues (resulting in a higher Tallying score). Additionally, we include trials where Tallying predicts a tie (equal number of winning features), but WADD strongly prefers one option due to the validity differences. This quantitative and qualitative divergence ensures the models yield distinct, distinguishable choice patterns.

[3] This design strictly dissociates Tallying from Weighted Additive (WADD) by contrasting the quantity of winning cues with their cumulative validities. We use a 5-feature design where two cues have very high validities and three have low validities. In the key dissociation trials, one option is favored by the two high-validity cues (winning the WADD evaluation), while the other option is favored by the three low-validity cues (winning the Tallying evaluation). We also include trials where Tallying predicts a perfect tie (each option wins on an equal number of cues), but WADD strongly prefers the option supported by higher-validity cues. This ensures a robust quantitative and qualitative divergence in the choice probabilities generated by the two models.

[4] To dissociate the Take-K heuristic from Tallying, we exploit their differential treatment of the lowest-validity cues. Take-K sorts cues by validity and ignores the bottom fraction (depending on its k_prop parameter), whereas Tallying counts wins across all cues regardless of validity. We use a 6-feature design where the 6th cue has a substantially lower validity than the rest. The trials are constructed so that the top 5 cues result in a tie (which Take-K will guess on) but the 6th cue breaks the tie (which Tallying will follow), or conversely, the top 5 cues favor one option but the 6th cue balances the overall tally (leading Tallying to guess, while Take-K makes a deterministic choice).

[5] To dissociate Tallying (the advocated theory) from the Take-K heuristic, we exploit both the structural difference in cue usage and the bounds on their noise parameters. Take-K sorts cues by validity and may discard the lowest-validity cues (depending on its k_prop parameter), whereas Tallying counts wins across all cues equally. Furthermore, Take-K is constrained to have high determinism (beta >= 2.0, epsilon <= 0.1), while Tallying allows for much softer, noisier choices (beta down to 0.1, epsilon up to 0.5). We use a 7-feature design where the 7th cue has the lowest validity. In key trials, the top 6 cues result in a perfect tie (3 wins each), but the 7th cue breaks the tie. Take-K (if k_prop < 1.0) will ignore the 7th cue and predict a 50/50 guess, while Tallying will use the 7th cue to predict a preference. Even if Take-K sets k_prop = 1.0 to mimic Tallying, its high-beta constraint forces it to predict very sharp probabilities (>= 88% for a 1-win difference), whereas Tallying can correctly capture a much softer empirical preference (e.g., 60%) driven by the small 1-cue tally difference.

[6] To cleanly dissociate Tally-then-TTB from pure Tallying, we use a 5-feature design with distinct validities. The core difference between the two models arises when the simple tally of features results in a tie. Pure Tallying has no mechanism to break ties and will strictly predict a 50/50 guess regardless of its noise parameters. In contrast, Tally-then-TTB will fall back to the highest-validity discriminating cue to break the tie, resulting in a deterministic preference. We include trials where the tallies are tied but the highest-validity cue points to either A or B, as well as trials where the highest-validity cue is tied so the second-highest must be used. We also include unequal-tally trials where both models agree, to ensure baseline tallying behavior is captured.

[7] To quantitatively dissociate Tallying (the advocated theory) from Tally-then-TTB (the competing theory), we use a 5-feature design with distinct validities. The core divergence between the models occurs when the simple tally of features results in a tie. Pure Tallying has no mechanism to break ties and strictly predicts a 50/50 guess on these trials, regardless of its noise parameters. In contrast, Tally-then-TTB breaks the tie using the highest-validity discriminating cue, resulting in a deterministic preference. We include trials where the tallies are tied but the highest-validity cue points to either A or B, as well as trials where the highest-validity cue is tied so the second-highest must be used. We also include unequal-tally trials where both models agree on the tally winner, ensuring baseline tallying behavior is captured and that the models only diverge on tie-breaker trials.

[8] To dissociate Tally-then-TTB (advocated) from Validity-Biased Tallying (competing), we focus on trials where the simple tally of features results in a tie. Tally-then-TTB resolves ties by looking exclusively at the single highest-validity discriminating cue (Take-The-Best), ignoring the remaining cues. In contrast, Validity-Biased Tallying resolves ties by comparing the sum of validity ranks of all winning cues for each option. We designed key trials where one option wins on the single highest-validity cue, but the other option wins on multiple moderately high-validity cues such that its sum of validity ranks is greater. On these tied-tally trials, Tally-then-TTB will deterministically choose the option with the top cue, while Validity-Biased Tallying will probabilistically favor the option with the higher sum of validity ranks. We also include unequal-tally baseline trials where both models agree.

[9] To dissociate Validity-Biased Tallying (advocated) from Tally-then-TTB (competing), we focus on trials where the simple tally of features results in a tie. Both models first tally the cues and agree on unequal-tally trials. However, they resolve ties differently. Tally-then-TTB uses a non-compensatory tie-breaker, strictly choosing the option favored by the single highest-validity discriminating cue. In contrast, Validity-Biased Tallying breaks ties by comparing the sum of the validity ranks of all winning cues for each option. We designed key dissociation trials where one option wins on the single highest-validity cue, but the other option wins on multiple moderately high-validity cues such that its total sum of validity ranks is greater. On these trials, Tally-then-TTB will deterministically choose the option with the top cue, while Validity-Biased Tallying will probabilistically favor the option with the higher sum of validity ranks.

[10] To dissociate Tally-then-TTB (advocated) from the Exponentially-Weighted Validity Tie-Breaker (competing), we exploit the structural constraints on how each model scales tie-breaker preferences. In Tally-then-TTB, a tie broken by the highest-validity cue yields a score difference of exactly 1.0, which is identical to the score difference produced by a 1-point tally advantage. Thus, Tally-then-TTB can predict equal determinism across 1-point tally wins and tie-breaker wins. In contrast, the competing model bounds its tie-breaker bonus by a parameter gamma < 1.0, and further dilutes it by normalizing against the sum of exponentiated validities. This forces the competing model to predict that choices on tied-tally trials must be significantly noisier (closer to 50/50) than choices on 1-point tally-advantage trials. By including both 1-point tally-advantage trials and tied-tally trials where the top cue breaks the tie, we can isolate this quantitative constraint.

[11] To dissociate the Exponentially-Weighted Validity Tie-Breaker (advocated) from Tally-then-TTB (competing), we focus on trials where the simple tally of features results in a tie. Tally-then-TTB resolves ties by looking exclusively at the single highest-validity discriminating cue, acting in a strictly non-compensatory manner. In contrast, the Exponentially-Weighted model resolves ties by comparing a weighted sum of exponentiated validities. Because its exponent (theta) is a free parameter, it can behave in a compensatory manner (e.g., if theta is low), allowing multiple moderately-high validity cues to overpower a single higher validity cue during the tie-breaker. We designed key dissociation trials where one option is favored by the single highest-validity cue, but the other option is favored by multiple moderately-high validity cues such that the sum of their validities is greater. On these tied-tally trials, Tally-then-TTB is structurally forced to favor the option with the top cue, whereas the Exponentially-Weighted model can predict a preference for the option supported by multiple cues. We also include baseline trials where tallies are unequal to ensure tallying dominance is captured.

[12] To dissociate Tally-then-TTB (advocated) from Generalized WADD (competing), we exploit the tension between tied-tally trials and unequal-tally trials. On tied-tally trials, Tally-then-TTB deterministically uses the highest validity cue to break the tie, which requires WADD to use a highly non-linear weighting (theta > 1) to match. However, if WADD uses theta > 1, it will act non-compensatorily on unequal-tally trials, favoring a minority of high-validity cues over a majority of low-validity cues. Tally-then-TTB, conversely, strictly follows the tally majority regardless of validities. By including both tied-tally trials and unequal-tally trials where the minority of cues has high validity, we force WADD into a contradiction where it cannot simultaneously fit Tally-then-TTB's deterministic tie-breaking and its strict unweighted tallying.

[13] To dissociate Generalized WADD (advocated) from Tally-then-TTB (competing), we construct a 5-feature design that challenges both the strict tallying assumption and the non-compensatory tie-breaker assumption of Tally-then-TTB. In Trial 1, Option B has more winning features (3 vs 2), but Option A wins on the two highest-validity cues. Tally-then-TTB strictly follows the tally and predicts B, whereas WADD can use its compensatory weighting to favor A. In Trial 2, the tallies are tied (2 vs 2). The single highest-validity cue points to A, so Tally-then-TTB deterministically breaks the tie in favor of A. However, Option B's winning cues sum to a higher total validity, allowing WADD to predict B. By including both types of trials, we show that WADD can capture compensatory trade-offs that violate both stages of the Tally-then-TTB model.

[14] To dissociate Tally-then-TTB (advocated) from Rank-Based Threshold Tallying (competing), we exploit the tension between global tallying and subset tallying. Tally-then-TTB integrates all cues to form a tally, falling back to the single highest-validity cue only if there is a global tie. Rank-Based Threshold Tallying restricts its primary tally to the top-K valid cues, ignoring lower-validity cues unless there is a tie within the top-K. We use a 6-feature design. In baseline trials (e.g., A wins the top 2 cues, B wins the next 3), Tally-then-TTB chooses B based on the global tally. To match this, the Rank-Based model must use a large consideration set (K >= 4). In our key dissociation trials, the global tally is perfectly tied (3 vs 3), with one option winning the single highest-validity cue and the lowest two cues, while the other wins the middle three cues. Tally-then-TTB deterministically breaks the global tie using the top cue. However, the Rank-Based model (forced to K >= 4 by the baseline trials) will see a primary tally advantage for the option winning the middle cues, leading it to predict the opposite choice or a 50/50 guess.

[15] To dissociate Rank-Based Threshold Tallying (advocated) from Tally-then-TTB (competing), we use a 6-feature design that pits global tallying against subset tallying. Tally-then-TTB integrates all cues for its primary decision, only falling back to the single highest-validity cue if the global tally is perfectly tied. Rank-Based Threshold Tallying restricts its primary tally to the top-K valid cues. In key dissociation trials, the global tally is either tied or favors one option, but the tally restricted to the top 2 or 3 cues favors the opposite option. For example, when the global tally is tied (3 vs 3) and Option A wins the single best cue but Option B wins the 2nd and 3rd best cues, Tally-then-TTB deterministically chooses A (via TTB tie-breaker), whereas the Rank-Based model chooses B (via top-K tally). Furthermore, we include unequal global tally trials where the minority of cues are high-validity, forcing Tally-then-TTB to follow the global majority while the Rank-Based model follows the top-K majority.

[16] To dissociate Tally-then-TTB (advocated) from the Strategy Selection Mixture model (competing), we exploit the fact that the mixing parameter in the competing model is fixed across trials by the global maximum validity of the environment. Tally-then-TTB acts as pure Tallying on unequal-tally trials and pure Take-The-Best (TTB) on tied-tally trials. We include unequal-tally trials where Tallying and TTB strongly disagree, forcing the Mixture model to rely heavily on its Tallying component to match the advocated theory's choices. However, we also include tied-tally trials where the advocated theory uses TTB to deterministically break the tie. Because the Mixture model's Tallying component predicts a 50/50 guess on ties, and its TTB weight is constrained to be low by the unequal-tally trials, it will be forced to predict highly ambiguous choices on tied-tally trials, failing to capture the deterministic tie-breaking of Tally-then-TTB.

[17] To dissociate the advocated Strategy Selection model from the competing Tally-then-TTB model, we exploit the differing roles of the highest validity cue. In Tally-then-TTB, the top cue is only consulted if the simple tally of features results in a tie; otherwise, the tally strictly determines the choice. In contrast, the Strategy Selection model probabilistically applies Take-The-Best (TTB) on every trial, with the probability of using TTB driven by the maximum validity in the environment. We include unequal-tally trials where the single best cue strongly favors one option but the majority of cues favor the other. Tally-then-TTB is structurally forced to choose the tally winner, whereas Strategy Selection can correctly capture a preference for the TTB winner if the top cue's validity is sufficiently high.

[18] To dissociate Tally-then-TTB (advocated) from WADD with Rank-Based Exponential Decay (competing), we exploit the tension in the competing model's decay parameter (gamma). Tally-then-TTB is purely compensatory on unequal-tally trials (strictly following the tally) and purely non-compensatory on tied-tally trials (strictly following the highest validity cue). In contrast, WADD uses a single rank-based exponential decay for all trials. We designed an unequal-tally trial where Option B wins the tally (3 vs 2) but Option A has the top two cues. For WADD to predict B (matching the advocated theory), it requires a very slow decay (gamma must be small, so weights are flat). We also designed a tied-tally trial (2 vs 2) where Option A has the top cue and the bottom cue, while Option B has the 2nd and 3rd cues. Tally-then-TTB deterministically chooses A because of the top cue. However, if WADD has a slow decay (to fit the unequal-tally trial), the combined weight of the 2nd and 3rd cues will outweigh the 1st and 5th cues, forcing WADD to predict B. Thus, WADD cannot simultaneously capture both the strict tallying and the deterministic tie-breaking of Tally-then-TTB.

[19] To dissociate WADD with Rank-Based Exponential Decay (advocated) from Tally-then-TTB (competing), we construct trials that challenge both stages of the competing model's rigid decision rules. Tally-then-TTB strictly predicts the tally winner on unequal-tally trials and the Take-The-Best (TTB) winner on tied-tally trials. We include unequal-tally trials where the minority of cues has high validity, and tied-tally trials where the TTB winner is opposed by multiple moderately high-validity cues. Tally-then-TTB is structurally forced into a fixed choice pattern across these trials (e.g., choosing the tally winner in the first case, and the top-cue winner in the second). In contrast, WADD's continuous exponential decay allows it to capture compensatory behavior that overrides the tally, or distributed weighting that overrides the single top cue, allowing it to fit diverse choice patterns (such as consistent preference for high-validity cues across both trial types) that Tally-then-TTB cannot match regardless of its noise parameters.

[20] To dissociate Tally-then-TTB (advocated) from the Probabilistic Strategy Mixture (competing), we exploit the competing model's upper bound on TTB reliance. On tied-tally trials, Tally-then-TTB deterministically uses the highest-validity cue to break the tie, allowing it to predict a very strong preference (approaching 100% depending on beta). In contrast, the Mixture model generates a 50/50 guess from its Tallying component on tied-tally trials, and its TTB component weight is strictly bounded by p_ttb <= 0.3. Consequently, the Mixture model is structurally constrained to predict at most a 65% preference for the TTB winner on these trials. By including tied-tally trials alongside unequal-tally baseline trials, we can robustly test if subjects exhibit deterministic tie-breaking that exceeds the Mixture model's mathematical limits.

[21] To dissociate the Probabilistic Strategy Mixture (advocated) from Tally-then-TTB (competing), we exploit a strict structural constraint in the competing model. Tally-then-TTB assigns a 1-point score difference to a 1-point tally advantage (e.g., 3 vs 2 winning features) and exactly the same 1-point score difference to a tied tally broken by the highest-validity cue. Thus, it is forced to predict identical choice determinism across both trial types. In contrast, the advocated Mixture model predicts strong preferences on 1-point tally advantages (driven by its Tallying component) but is strictly bounded to a weak preference (maximum 65%) on tied-tally trials because its Tallying component guesses 50/50 and its TTB weight is capped at 0.3. By including both 1-point tally advantage trials and tied-tally trials, we can demonstrate that subjects show high determinism on tally advantages but much lower determinism on ties—a quantitative pattern the advocated model naturally fits but the competing model cannot.

[22] To dissociate Tally-then-TTB (advocated) from WADD with Power Transformation (competing), we exploit the competing model's single gamma parameter, which controls its position on the compensatory/non-compensatory spectrum. Tally-then-TTB is strictly compensatory on unequal-tally trials (following the tally) and strictly non-compensatory on tied-tally trials (following the single best cue). In Trial 1 (unequal tally), Option A wins the tally but Option B wins the top cues. For WADD to predict A (matching Tally-then-TTB), it must use a low gamma (compensatory). In Trial 2 (tied tally), Option A has the highest validity cue but Option B has the next two highest. Tally-then-TTB deterministically chooses A via its tie-breaker. However, if WADD uses a low gamma to fit Trial 1, the combined weight of Option B's cues will exceed Option A's top cue, forcing WADD to predict B. Thus, WADD cannot simultaneously capture the strict tallying and deterministic tie-breaking of the advocated theory.

[23] To dissociate WADD with Power Transformation (advocated) from Tally-then-TTB (competing), we construct trials that pit the rigid two-stage logic of Tally-then-TTB against the continuous validity integration of WADD. Tally-then-TTB is structurally forced to choose the tally winner on unequal-tally trials, and the single highest-validity cue winner on tied-tally trials. We designed Trial 1 such that Option A wins the tally (3 vs 2), but Option B wins the two highest-validity cues. We designed Trial 2 such that the tally is tied (2 vs 2) and Option A wins the single highest-validity cue, but Option B wins the second and third highest-validity cues. Tally-then-TTB strictly predicts Option A in both trials. However, WADD can predict Option B in both trials by using a moderate power transformation (e.g., gamma = 2), which scales validities such that the top two cues outweigh the bottom three (Trial 1), and the second plus third cues outweigh the first plus fifth (Trial 2). This demonstrates WADD's ability to capture compensatory validity integration that violates both stages of Tally-then-TTB.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Tally-then-TTB (advocated) from Thresholded WADD (competing), we exploit how each model scales tie-breaker determinism. In Tally-then-TTB, any tie broken by a single cue (whether it is the 1st or 2nd highest validity) yields a fixed score difference of 1.0, predicting identical choice determinism across different tie-breaker trials. In contrast, Thresholded WADD resolves ties using exponentiated validities. Because the exponentiated validity of the top cue is mathematically larger than that of the second cue, Thresholded WADD is structurally forced to predict higher determinism when the 1st cue breaks a tie compared to when the 2nd cue breaks a tie. By including both types of tied-tally trials, we can isolate this signature of equal vs. unequal determinism.",
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
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
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
      1,
      1,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
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
  "prompt_token_count": 7765,
  "candidates_token_count": 399,
  "total_token_count": 11703
}
```
