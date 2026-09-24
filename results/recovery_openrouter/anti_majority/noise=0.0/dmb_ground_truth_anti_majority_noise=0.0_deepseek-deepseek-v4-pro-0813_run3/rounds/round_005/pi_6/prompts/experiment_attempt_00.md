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
**Description:** Subjects carry a latent valence orientation, with roughly 86% treating a binary expert rating of 1 as a defect and 0 as clean, while the remainder use the opposite positive valence. On each trial, each feature is probabilistically gated into an unweighted tally according to a validity logistic whose top-validity cue is included with high probability, while lower-validity cues are included often enough for their majorities to matter. Tally choices follow a softmax with subject-specific gain. Exact zero tallies are resolved by scanning only included cues in descending validity order and choosing the first valence-consistent discriminator. In addition, when the included tally is just one cue away from zero, the single highest-validity included discriminator receives a small attention bonus inside the tally, making the top cue somewhat more decisive without overriding larger lower-cue majorities. Subject-level parameters are drawn from narrowed heavy-tailed or logit-normal distributions to keep between-subject variance modest where the data are consistent while preserving heterogeneity where it is large.

**Parameters:**
- orientation_u: [0, 1]
- alpha_raw: [0.01, 0.99]
- gamma_raw: [0.01, 0.99]
- beta_tally_raw: [0.01, 0.99]
- beta_tiebreak_raw: [0.01, 0.99]
- epsilon_raw: [0.01, 0.99]
- top_bonus_raw: [0.01, 0.99]
- validities: validities

**`predict source code`:**
```python
import numpy as np
from scipy.stats import norm


def _trunc_qnorm(u, mu, sigma, lo, hi):
    u = float(np.clip(u, 0.0, 1.0))
    q_lo = norm.cdf((lo - mu) / sigma)
    q_hi = norm.cdf((hi - mu) / sigma)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * norm.ppf(q)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # Subject-level transforms: raw uniform draws are mapped through
    # truncated normal / log-normal / logit-normal distributions.
    u_alpha = (float(parameters['alpha_raw']) - 0.01) / 0.98
    u_alpha = float(np.clip(u_alpha, 0.0, 1.0))
    alpha = float(np.exp(_trunc_qnorm(
        u_alpha, np.log(6.5), 0.20, np.log(5.5), np.log(8.5))))

    # Live gamma for the validity gate.  It is not overwritten below; only
    # the single highest-validity cue's inclusion probability is pinned high.
    u_gamma = (float(parameters['gamma_raw']) - 0.01) / 0.98
    u_gamma = float(np.clip(u_gamma, 0.0, 1.0))
    gamma = float(_trunc_qnorm(u_gamma, -2.95, 0.35, -3.85, -2.45))

    u_bt = (float(parameters['beta_tally_raw']) - 0.01) / 0.98
    u_bt = float(np.clip(u_bt, 0.0, 1.0))
    beta_tally = float(np.exp(_trunc_qnorm(
        u_bt, np.log(1.85), 0.28, np.log(1.20), np.log(2.90))))

    u_tb = (float(parameters['beta_tiebreak_raw']) - 0.01) / 0.98
    u_tb = float(np.clip(u_tb, 0.0, 1.0))
    beta_tiebreak = float(np.exp(_trunc_qnorm(
        u_tb, np.log(2.00), 0.35, np.log(1.20), np.log(3.20))))

    u_eps = (float(parameters['epsilon_raw']) - 0.01) / 0.98
    u_eps = float(np.clip(u_eps, 0.0, 1.0))
    logit_lo = np.log(0.03 / 0.97)
    logit_hi = np.log(0.16 / 0.84)
    logit_mu = np.log(0.07 / 0.93)
    z_eps = _trunc_qnorm(u_eps, logit_mu, 0.45, logit_lo, logit_hi)
    epsilon = float(1.0 / (1.0 + np.exp(-z_eps)))

    # Small top-cue attention bonus, used only for near-zero tally margins.
    u_top = (float(parameters['top_bonus_raw']) - 0.01) / 0.98
    u_top = float(np.clip(u_top, 0.0, 1.0))
    extra = float(_trunc_qnorm(u_top, 0.40, 0.15, 0.00, 0.80))
    w_top = 1.0 + max(0.0, min(0.80, extra))

    # Latent valence orientation.  Approximately 86% use inverse valence.
    inv = float(parameters['orientation_u']) < 0.86

    # Inclusion probabilities from a validity logistic gate.
    # Only the highest-validity cue's inclusion probability is forced high.
    logit_j = alpha * val + gamma
    p_j = _sigmoid(logit_j)
    if n_features > 0:
        j_top = int(np.argmax(val))
        p_j[j_top] = float(np.clip(p_j[j_top], 0.95, 0.998))
    p_j = np.clip(p_j, 0.02, 0.99)

    included = np.flatnonzero(np.random.uniform(size=n_features) < p_j)
    if included.size == 0:
        return np.ones(2) / 2.0

    # Valence-consistent cue comparison on the included cues only.
    if inv:
        good_a = (a == 0) & (b == 1)
        good_b = (a == 1) & (b == 0)
    else:
        good_a = (a == 1) & (b == 0)
        good_b = (a == 0) & (b == 1)

    d_raw = float(np.sum(good_a[included]) - np.sum(good_b[included]))
    d = d_raw

    # Top-cue attention bonus.  It operates inside the tally path only, and
    # only when the current tally is one cue away from zero.  Larger
    # lower-cue majorities are not overridden, and exact-zero ties still
    # go to the validity-ordered tie-break scan.
    if d_raw != 0.0 and abs(d_raw) <= 1.0:
        disc = [j for j in included if a[j] != b[j]]
        if disc:
            j_att = max(disc, key=lambda j: float(val[j]))
            if inv:
                if a[j_att] == 0 and b[j_att] == 1:
                    d = d_raw + (w_top - 1.0)
                elif a[j_att] == 1 and b[j_att] == 0:
                    d = d_raw - (w_top - 1.0)
            else:
                if a[j_att] == 1 and b[j_att] == 0:
                    d = d_raw + (w_top - 1.0)
                elif a[j_att] == 0 and b[j_att] == 1:
                    d = d_raw - (w_top - 1.0)

    if d != 0.0:
        scores = np.array([d, -d], dtype=float)
        beta = beta_tally
    else:
        discriminating = [j for j in included if a[j] != b[j]]
        if len(discriminating) == 0:
            return np.ones(2) / 2.0

        discriminating = sorted(discriminating, key=lambda j: -val[j])
        winner = None
        for j in discriminating:
            if inv:
                if a[j] == 0 and b[j] == 1:
                    winner = 0
                    break
                elif a[j] == 1 and b[j] == 0:
                    winner = 1
                    break
            else:
                if a[j] == 1 and b[j] == 0:
                    winner = 0
                    break
                elif a[j] == 0 and b[j] == 1:
                    winner = 1
                    break

        if winner is None:
            return np.ones(2) / 2.0

        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = beta_tiebreak

    logits = beta * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People invert the valence of binary expert ratings: a 1 is treated as a defect and a 0 as a clean, favorable signal. Each feature j probabilistically enters an inverse clean-cue tally with inclusion probability p_include_j = logistic(alpha * validity_j + gamma); excluded cues receive exactly zero weight. The included clean-cue tally drives a noisy softmax choice, and exact zero tallies are resolved by consulting the highest-validity included discriminating cue and choosing the option with the clean 0 there. If no included cue discriminates, people guess. Subject-level alpha, gamma, beta_tally, beta_tiebreak, and epsilon create individual differences, with heavy-tailed/logit-transformed subject-level distributions so that some subjects are nearly deterministic while others remain noisy, preserving the pooled inverse-valence pattern and realistically large between-subject variance.

**Parameters:**
- alpha: [5.0, 9.0]
- gamma_logit: [-2.50, 2.50]
- beta_tally_log: [-0.30, 0.50]
- beta_tiebreak_log: [0.40, 1.10]
- epsilon_logit: [-3.00, -0.40]
- validities: validities

**`predict source code`:**
```python
import numpy as np
from scipy.special import expit

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # Heavy-tailed subject-level transforms of the sampled raw parameters.
    beta_tally = float(np.exp(float(parameters['beta_tally_log'])))
    beta_tiebreak = float(np.exp(float(parameters['beta_tiebreak_log'])))
    epsilon = float(expit(float(parameters['epsilon_logit'])))
    alpha = float(parameters['alpha'])
    gamma = -4.5 + 2.0 * float(expit(float(parameters['gamma_logit'])))

    p_include = expit(alpha * val + gamma)
    include = np.random.random(n_features) < p_include

    clean_a = ((b > a) & include).astype(float)
    clean_b = ((a > b) & include).astype(float)
    s = float(np.sum(clean_a) - np.sum(clean_b))

    if s != 0.0:
        p_a = expit(2.0 * beta_tally * s)
        p_core = np.array([p_a, 1.0 - p_a])
    else:
        winner = None
        cue_order = np.argsort(-val, kind='stable')
        for j in cue_order:
            if include[j] and a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        p_a = expit(beta_tiebreak) if winner == 0 else expit(-beta_tiebreak)
        p_core = np.array([p_a, 1.0 - p_a])

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Six features are used. The highest-validity expert (0.90) is always the first cue TTB tests. I hold that cue's discriminating direction constant across blocks and vary the lower five cues so Tallying's net evidence changes while TTB sees the same first discriminating cue. In the top-A block the top cue is 1 vs 0 and lower-cue net wins take values +5, +3, +1, -1, -3, -5; in the top-B block the top cue is reversed. TTB predicts identical p(A) for all six top-A pairs and identical p(A) for all six top-B pairs because the first discriminating cue is unchanged; Tallying predicts a graded logistic change because its tally advantage is 1 + lower_net for top-A pairs and lower_net - 1 for top-B pairs. At lower_net = -1 in the top-A block and lower_net = +1 in the top-B block, Tallying is exactly tied while TTB commits to A or B. At lower_net = -3/-5 in the top-A block and +3/+5 in the top-B block, the two theories make opposite directional predictions. The critical signature is flat TTB choice probabilities across the six lower-net levels versus a monotone logistic gradient for Tallying. Twelve unique pairs repeated K = 8 gives 96 trials.

[1] Twelve unique pairs form a 4 × 3 factorial. Six features have validities [0.92, 0.84, 0.76, 0.68, 0.60, 0.55], so TTB's cue cascade is feature 0 → 1 → 2 → 3 → 4 → 5. The first factor is the direction of the first discriminating cue: feature 0 favors A; feature 0 favors B; feature 0 ties and feature 1 favors A; feature 0 ties and feature 1 favors B. The second factor is the tally difference (+2, 0, -2), implemented by the remaining lower features. Tallying ignores which features discriminate, so it predicts three choice-probability levels that depend only on tally difference and are identical across the four first-cue conditions. TTB ignores all lower features once a discriminating cue is found, so it predicts only two choice-probability levels (high for first-cue A, low for first-cue B) and no effect of tally difference. This yields opposite directional predictions on conflict trials: top A with tally -2 (pair 3), top B with tally +2 (pair 4), top tie/second A with tally -2 (pair 9), and top tie/second B with tally +2 (pair 10). It is the complement of the already-explored design: instead of holding the top cue constant and varying lower net, it holds tally difference constant across cue-order conditions and varies the first discriminating cue. With 12 unique pairs repeated K=8, total trials = 96.

[2] Take The Best stops at the first discriminating cue in validity order and ignores all lower cues. Inverse tallying aggregates all cues with reversed valence (0 = favorable) and chooses the option with more zeros. This design holds the inverse-tallying score at exactly zero for every trial: each option has exactly one 1 and one 0 in the two discriminating positions, and all other positions tie. Thus inverse tallying predicts p(A)=0.5 for every pair, regardless of its beta and epsilon. TTB, however, consults cues in validity order and stops at the first discriminating cue. By shifting the position of that first discriminating cue from feature 0 to feature 5 and varying its direction (A=1/B=0 vs A=0/B=1), TTB predicts p(A) to be high when the first cue favors A and low when it favors B, regardless of the lower balancing cue. This yields a strong quantitative dissociation: a flat 0.5 psychometric function across all 12 pairs for inverse tallying versus a step function of first-cue direction for TTB. Unlike the already-explored designs, which held the top cue constant while varying lower-cue net (experiment 0) or varied first-cue direction and tally difference with standard Tallying (experiment 1), this design uses the inverse-tallying tie prediction on every trial and varies the first-cue position across all six possible positions.

[3] Six features with validities [0.90, 0.80, 0.70, 0.60, 0.55, 0.50] fix TTB's cue order as feature 0 through 5. Every trial has all six features discriminating, so TTB always stops at feature 0; its direction is manipulated across trials. The remaining five features are arranged so that the inverse-tally score s_A = #(A=0,B=1) - #(A=1,B=0) takes values +2, 0, or -2. This creates a 2 (feature-0 direction: A=1/B=0 vs A=0/B=1) x 3 (inverse tally score) factorial with two replications per cell. Inverse tallying ignores which feature discriminates and uses only s_A, so it predicts p(A) ordered by s (+2 > 0 > -2) and no effect of feature-0 direction. TTB stops at feature 0 and ignores all lower features, so it predicts p(A) high when feature 0 favors A and low when feature 0 favors B, with no effect of s. The conflict cells (feature-0 A with s=-2; feature-0 B with s=+2) yield opposite directional predictions, and the s=0 cells yield p(A)=0.5 for inverse tallying while TTB commits to the top cue. This double dissociation quantitatively separates the two theories.

[4] This design dissociates the advocated inverse weighted-tally plus validity-ordered tie-break theory from plain inverse tallying. Six validities are spread as [0.95, 0.85, 0.75, 0.65, 0.55, 0.50], so the advocated weak validity weighting (1.9-2.1) makes high-validity cues roughly 2.5 times as influential as low-validity cues. Six tie trials hold the raw inverse tally at exactly 0 (one A-clean and one B-clean discriminating feature). Competing inverse tallying must predict p(A)=0.5 for all six because the tallies are equal, whereas the advocated tie-break predicts p(A) near 0.71 when the first discriminating cue in descending validity order is A=0/B=1 and near 0.29 when it is A=1/B=0; the first discriminator is varied across features 0, 1, and 2. Six conflict trials hold the raw inverse tally at +1 or -1 but place the high-validity wins on the opposite side, so the advocated validity-weighted score reverses sign. For raw +1 trials the weighted scores are about -0.62, -0.42, and -0.22, predicting p(A) around 0.38-0.46, while competing inverse tallying predicts high p(A) around 0.65-0.85. For raw -1 trials the mirror arrangement gives advocated p(A) around 0.54-0.62, while competing inverse tallying predicts low p(A) around 0.15-0.35. Twelve unique pairs repeated K=8 gives 96 trials.

[5] Eight features with symmetric validities around mean 0.75. At validity_weight=2.0 the competing weights are [1.4,1.3,1.2,1.1,0.9,0.8,0.7,0.6]. The 12 unique pairs form a 2x6 factorial: raw inverse tally d=#(A=0,B=1)-#(A=1,B=0) is +1 or -1, while the validity-weighted score s is set to -0.9,-0.6,-0.3,0,+0.3,+0.6 for d=+1 and the mirror for d=-1. No trial has d=0, so the competing tie-break is never engaged; this isolates the validity-weighting component and avoids repeating design [4]. Plain inverse tallying ignores validities: all d=+1 trials predict p(A) about 0.64-0.85 and all d=-1 trials predict p(A) about 0.15-0.36, flat across s. The competing weighted tally predicts p(A) increasing with s: about 0.33,0.38,0.44,0.50,0.56,0.62,0.67 for s=-0.9,-0.6,-0.3,0,+0.3,+0.6,+0.9. Therefore conflict trials (d=+1 with s<0; d=-1 with s>0) give opposite directional predictions, and s=0 trials oppose p=0.5 to high/low. K=8 repetitions gives 96 trials.

[6] Eight features with validities symmetric around 0.75. For the advocated theory this gives validity weights [1.4,1.3,1.2,1.1,0.9,0.8,0.7,0.6] when the raw tally is nonzero, but every trial here has raw inverse tally exactly zero, so the advocated theory always uses its validity-ordered tie-break. Feature 0 (0.95) discriminates on every trial; top-A trials have feature 0 A=0/B=1 and top-B trials have A=1/B=0, so advocated predicts p(A) approximately 0.715 for all six top-A trials and approximately 0.285 for all six top-B trials, flat across lower-cue composition. The competing inclusion theory does not tie-break on raw zero: cue inclusion is probabilistic, so the included tally is usually nonzero. Lower features are arranged to have net -1 (top A) or +1 (top B) but with increasing numbers of opposite-side lower cues across three levels. With typical inclusion parameters (e.g., alpha=7, gamma=-4.0), competing expected p(A) moves from about 0.81 (level 1) to 0.72 (level 2) to 0.54 (level 3) for top A, and mirror 0.19/0.28/0.46 for top B, crossing the advocated flat predictions. This yields a quantitative interaction: advocated predicts no effect of lower-cue composition, while competing predicts a monotone gradient.

[7] Eight features with three perfect experts (1.0) and five moderate experts (0.70). The three high-validity cues are always unanimous for one product; the five low-validity cues oppose them in 3, 4, or 5 positions (remaining low cues tied). This yields six unique pairs: 2 high-cue direction × 3 low-cue count. Competing theory uses all cues with validity weights 1.375 (high) and 0.775 (low); for high-B/low-A pairs the weighted score is negative (n=4: -1.025, n=5: -0.25), predicting p(A)≈0.31/0.45, and the n=3 raw tie is broken by the high cue, p(A)≈0.29. The mirror high-A/low-B pairs predict p(A)≈0.72/0.69/0.55. The advocated inclusion theory includes high cues with p≈0.97 and low cues with p≈0.80 and tallies included cues equally. It predicts p(A)≈0.18 (n=3), 0.46 (n=4), 0.71 (n=5) for high-B/low-A, and p(A)≈0.83, 0.54, 0.29 for high-A/low-B. Thus the theories make opposite directional predictions in the n=4/n=5 high-B/low-A cells and the n=5 high-A/low-B cell, and show different slopes across low-cue count. K=16 repetitions of the six unique pairs gives 96 trials.

[8] This design holds the raw inverse tally at exactly 0 on every unique pair, so the advocated theory is always in its validity-ordered tie-break mode and chooses using only the first discriminating feature. The top two experts (validities 0.95 and 0.90) are tied in all pairs, so the first discriminator is placed at validity 0.65 or 0.64 after those ties. The six pairs form a first-direction by lower-cue-mass design. Advocated theory predicts a flat step of approximately p(A)=0.71 when the first discriminating cue is A=0/B=1 and approximately p(A)=0.29 when it is A=1/B=0, with no effect of later cues. The competing stochastic-inclusion theory does not have this deterministic raw-tie mode: it includes later cues probabilistically, and in the k=3 pairs the three mid-validity opposite cues (0.64, 0.63, 0.62) plus two 0.50 same-side cues give an expected included tally of about -0.40. Under typical parameters alpha=7, gamma=-3.5, beta_tally around 1.1, beta_tiebreak around 2, p(A) is approximately 0.47 for the first-cue-A k=3 pair and approximately 0.53 for its mirror, opposite in direction to the advocated tie-break. The k=2 pairs provide a weaker opposite lower-cue mass and therefore an intermediate quantitative gradient. This separates a deterministic validity-ordered raw-tie-break from stochastic all-cue inclusion while avoiding the previously explored top-cue tie-break designs.

[9] All six pairs have a raw inverse tally of +2 or -2, so the competing theory is forced to use its mean-centered validity-weighted tally rather than a raw tie-break. The validities are deliberately skewed: features 0-5 are high (1.00 to 0.95), but features 3-5 are tied on every trial and act as high-validity anchors that raise the validity mean to about 0.795. The five low-validity features 6-10 (0.62, 0.60, 0.58, 0.56, 0.54) actually do the discriminating work. At median advocated parameters alpha=7 and gamma=-3.5, inclusion probabilities are about 0.97 for the high cues and 0.70, 0.67, 0.64, 0.60, 0.57 for the low cues. In the competing theory the tied high anchors lower the mean-centered weights of the low cues to about 0.72 to 0.59, while high-cue weights are about 1.33. For the raw +2 pairs, a low-validity majority of 3, 4, or 5 A-clean cues is opposed by 1, 2, or 3 high-validity B-clean cues. The competing weighted score goes from +0.73 through -0.08 to -0.66, so p(A) drops from about 0.64 to 0.49 to 0.37. The advocated expected included tally stays positive at about 1.03, 0.54, and 0.27, so advised p(A) stays at about 0.84, 0.73, and 0.62. The raw -2 mirrors reverse the signs and create a crossover: at the 4- and 5-low-cue levels the two theories make opposite directional predictions. The key manipulation is the tied high anchors, which change the competing mean-centered weights without changing the advocated inclusion tally. Six unique pairs repeated K=16 gives 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Nine unique pairs are built from eight binary cues. Every cue discriminates on every trial, so there are no degenerate display ties. The pairs are indexed by k = 0 through 8: the k highest-validity cues are A=0/B=1 and the remaining 8-k cues are A=1/B=0. This makes the inverse-clean tally (0=clean) take the values -8, -6, -4, -2, 0, +2, +4, +6, +8. The pure inverse-valence competing theory predicts a steep monotone trace, near p(A)=0 at k=0 and near p(A)=1 at k=8. The advocated theory has the same inverse tally path for about 86 percent of subjects, but about 14 percent of subjects carry positive valence and treat 1 as clean; on every discriminating cue those subjects choose the opposite product. Thus the advocated prediction is approximately P_adv(k) = 0.14 + 0.72 * P_inv(k), a linear contraction of the inverse trace toward 0.5. At k=0 the advocated model is pulled up to roughly 0.14 to 0.18 instead of near zero; at k=8 it is capped near 0.82 to 0.86 instead of near one; and the overall deflection across k is only about 72 percent as steep. The validities are spread but kept moderately high so inclusion probabilities stay high and the large raw tallies dominate stochastic exclusion, making the orientation-mixture compression the main quantitative dissociation. The k=4 pair also carries an exact raw tie, but its advocated tie-break prediction still lies closer to 0.5 than the pure inverse tie-break prediction. The nine unique pairs are repeated K = max(1, 96 // 9) = 10 times per subject for about 90 trials.",
  "validities": [
    0.96,
    0.92,
    0.88,
    0.84,
    0.8,
    0.76,
    0.72,
    0.68
  ],
  "trial_a_ratings": [
    [
      1,
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
      0,
      1
    ],
    [
      0,
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
      0,
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
      0,
      0,
      0
    ],
    [
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
      1,
      1,
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
      1,
      1,
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
      1,
      0
    ],
    [
      1,
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
  "input_tokens": 8347,
  "output_tokens": 25202,
  "reasoning_tokens": 24280,
  "provider": "Baidu"
}
```
