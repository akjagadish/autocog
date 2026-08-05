# Concave Value Functions in Multi-Feature Choice: A Two-Experiment Preregistration (Pairwise Model Comparison and Value-Curvature Tests)

## Title

Concave Value Functions in a Multi-Feature Choice Task: Pairwise Model Comparison (Experiment 1) and Value-Curvature Tests (Experiment 2)

## Description

This project tests whether people apply a concave value function to numerical feature values in a multi-feature choice task. On each trial, participants choose between two options, each described by four cardinal feature values on a common 0–5 scale. Each feature also has a displayed validity (shown in strictly descending order) indicating how strongly it should count toward the choice. The central claim is that displayed feature values are passed through a monotone concave value function before being combined by the displayed validities, so the marginal impact of a one-unit improvement shrinks as the absolute feature value grows: a change from 0 to 1 affects choice more than an objectively equal change from 4 to 5.

The project comprises two separate, between-subjects experiments, run as two independent recruitments and analyzed independently. Each participant takes part in only one experiment.

**Experiment 1 — Pairwise model comparison (the first experiment).** This experiment asks whether the concave value-function account out-predicts each of three classic decision rules — weighted-additive (WADD), tallying, and take-the-best (TTB) — in a pairwise fashion. Rather than contrasting the concave account against the three heuristics jointly, we contrast it against each heuristic separately, on stimuli built specifically to separate the concave account from that one heuristic. The battery therefore explicitly includes stimuli on which the three heuristics disagree with one another.

**Experiment 2 — Value curvature (the second experiment).** This experiment characterizes the shape of the value function directly, with two components. The steep-vs-flat component tests whether an advantage in the low range of the scale beats an objectively equal advantage in the high range. The level-shift (offset) component tests whether the same feature-value structure has a weaker effect on choice when it is shifted upward by a constant into a higher value range.

The exact frozen stimulus battery and the power simulation that fixed the sample sizes are provided in the attached code (`build.py`, which writes `battery.json`); the confirmatory analysis is provided in `analyse.py`.

## Contributors

Younes Strittmatter

## Subjects

Psychology; Social and Behavioral Sciences

# Overview

## Research questions or hypotheses

The project registers three confirmatory hypotheses. H1 belongs to Experiment 1; H2 and H3 belong to Experiment 2.

### Experiment 1 — Pairwise model comparison

**H1. Pairwise model-discrimination hypothesis.** On stimuli constructed to separate the concave account from a given heuristic m (WADD, tallying, or TTB), participants' choices will match the concave account more often than they match m. This is tested separately for each of the three heuristics.

### Experiment 2 — Value curvature

**H2. Steep-vs-flat hypothesis.** Participants will choose the option whose advantage falls in the low-value region more often than the option whose equal-sized advantage falls in the high-value region (above 50%).

**H3. Level-shift / offset hypothesis.** The same feature-value structure will have a stronger effect on choice when presented in a low value range than when shifted upward into a higher value range.

## Foreknowledge of data or evidence

No. No data have been collected for either experiment. The study will be run prospectively after registration, and no part of the data to be analyzed exists yet.

# Research Design

## Study type

Randomized behavioral experiments. Option side, trial order, and value/validity assignment are randomized or counterbalanced. The two experiments are separate between-subjects studies; each participant is recruited into one experiment and sees only that experiment's trials.

## Models

All four models map a stimulus (two options × four features) and the displayed validities to a predicted option. They are defined as follows.

**Concave value-function account (focal model).** Each option's value is V = sum over features i of w_i · u(x_i), where x_i is the displayed value of feature i, the weights w_i are the displayed validities normalized to sum to 1, and u(x) = (x + s)^α − s^α is a concave value function with curvature α in (0, 1] and offset s > 0. Choice is a softmax with lapse: P(choose B) = (1 − ε)·logistic(β·(V_B − V_A)) + ε·0.5, with inverse temperature β > 0 and lapse rate ε in [0, 0.5]. At α = 1 the value function is linear (the account reduces to WADD); at α < 1 it is strictly concave. To define each option pair's concave-predicted option we use the deterministic core (the argmax of V, i.e. β → ∞, ε = 0) evaluated over a grid of α in {0.1, 0.2, …, 1.0} crossed with s in {0.1, 0.644, …, 5.0} (10 × 10); the concave-predicted option is the majority choice across the grid points with α < 1.

**WADD (weighted additive).** Choose the option maximizing the validity-weighted sum of raw feature values, sum over i of validity_i · x_i (a linear value function).

**Tallying.** Count the features on which each option is strictly higher; choose the option with more such wins (an equal tally leaves the model undecided).

**Take-the-best (TTB).** Inspect features in descending validity order and choose the option favored by the first feature that discriminates between the options (if no feature discriminates, the model is undecided).

## Stimuli

Each option is described by four cardinal feature values, integers from 0 to 5. Each feature has a fixed displayed validity, shown in strictly descending order. Five validity vectors are used: the five strictly-descending 4-tuples of distinct values from {0.5, 0.6, 0.7, 0.8, 0.9}, namely (0.9, 0.8, 0.7, 0.6), (0.9, 0.8, 0.7, 0.5), (0.9, 0.8, 0.6, 0.5), (0.9, 0.7, 0.6, 0.5), and (0.8, 0.7, 0.6, 0.5). Each participant is assigned one validity vector, counterbalanced in equal numbers across that experiment's participants. Each participant completes 96 choice trials; trial order and the left/right screen position of the two options are randomized independently for each participant. The exact feature values of every option pair are fixed before data collection and are provided in the attached `battery.json`.

**Experiment 1 (pairwise model comparison).** Under each validity vector the battery contains twelve option pairs, organized into three discriminating subsets of four pairs each — one subset per heuristic (concave-vs-WADD, concave-vs-tallying, concave-vs-TTB). Within the subset for heuristic m, each pair is selected so that (a) the concave account's deterministic core chooses the option opposite to m on at least 85% of the α < 1 grid, and (b) m makes a decisive (non-tie) choice. Each of the two base pairs in a subset is presented together with its option-swapped mirror (A↔B), so the concave-predicted option is option A on half the trials and option B on the other half. Each of the twelve pairs is presented 8 times, for 96 trials (32 trials per pairwise comparison). For every pair we record all four models' predicted options; because each subset targets one heuristic, the three heuristics frequently disagree with one another across the battery.

**Experiment 2 (value curvature).** Under each validity vector the battery contains four steep-vs-flat counterbalanced pairs and four level-shift pairs. In each steep-vs-flat pair, one option gains a value step in the low range of the scale and the other gains an equal step in the high range, on a different feature; across the two trials of the pair (X and Y), which feature carries the low-range step and which option holds it are swapped, so every linear (validity-weighted) rule chooses the low-range option exactly 50% of the time over the pair and any constant side preference cancels. The concave-predicted option is the one whose advantage is in the low range. The four pairs use four different gain sizes (steps of 1, 2, 3, and 4 units) on different feature pairs, so the steep/flat trade-off is probed at several points of the value scale; the two features not carrying the trade-off hold an identical (tied) filler value in both options, which therefore cancels in every model and does not affect any prediction (these filler values are varied across pairs rather than fixed at zero). Each steep-vs-flat trial (X or Y) is presented 4 times (8 trials per pair). In level-shift pairs, the same option pair is shown once with values in a low range and once shifted upward by a constant into a higher range; every linear rule predicts identical choices for the two versions, and the target option is the one a concave value function favors in the low range. The four level-shift pairs use different low/high ranges (low values 0–2 shifted up by 3 into 3–5; low 0–3 shifted up by 2 into 2–5; and low 0–1 shifted up by 4 into 4–5), so the level-shift is probed at several absolute value ranges. Each level-shift version (low or shifted-up) is presented 8 times (16 trials per pair). This yields 96 trials (steep-vs-flat 8 trials × 4 pairs = 32; level-shift 16 trials × 4 pairs = 64); the subtler level-shift effect therefore receives more trials per participant.

## Randomization and blinding

Within each experiment, each participant is assigned one of the five validity vectors, counterbalanced in equal numbers (Experiment 1: 10 participants per vector; Experiment 2: 20 participants per vector). Within a participant, trial order is randomized and the left/right screen position of the two options is counterbalanced, so the predicted option appears equally often on each side. Participants are not told the hypotheses, which option any model predicts, or which experiment they are in; they receive only the task instructions. The predicted option for every model and every trial is fixed by the stimulus-generation procedure and model definitions before data collection (frozen in `battery.json`).

# Sampling

## Data collection procedures

Data will be collected online via Prolific (recruitment) and a web-based task. The target population is adults who are at least 18, fluent in English, and eligible for online studies on Prolific; we use Prolific prescreening for age and language. Participants provide informed consent and receive instructions explaining the feature values, displayed validities, and choice procedure. We record each trial's choice, feature values, validities, option side, trial type, the experiment, and trial order. Response time is recorded for exploratory purposes only.

## Sample size and stopping rule

Two separate samples, each balanced across the five validity vectors:

- Experiment 1 (pairwise model comparison): 50 participants who complete the study (10 per validity vector).
- Experiment 2 (value curvature): 100 participants who complete the study (20 per validity vector).

For each experiment we collect data until that experiment reaches its target number of completed participants, balanced across the five vectors; collection stops once the target is reached. A participant counts as completed when all 96 of their choice trials are recorded.

## Sample size rationale

Sample sizes were fixed before data collection by the power simulation in the attached `build.py`. The simulation draws each participant's value-function curvature, offset, and softmax sensitivity from the concave account, applies each experiment's repetitions per stimulus (Experiment 1: 8 per pair, 32 per pairwise comparison; Experiment 2: 4 per steep-vs-flat trial and 8 per level-shift version, i.e. 32 steep-vs-flat and 64 level-shift trials; 96 trials per participant in both), and adds 30% observation noise — on 30% of trials the response is a uniform random choice between the two options. All confirmatory tests are evaluated at a one-tailed alpha of .05.

Under this primary generating model, the smallest balanced sample reaching at least 99% power for every confirmatory test is 30 participants for Experiment 1 (its binding constraint is the concave-vs-WADD comparison) and 75 participants for Experiment 2 (its binding constraint is the level-shift test; the steep-vs-flat test reaches 99% by about 40 participants). We collect more than these minimums — 50 and 100 — to (a) retain high power under a more conservative generating model, (b) buffer against incomplete sessions, and (c) support the balanced per-vector design. Under a conservative model with only weak concavity (curvature drawn from [0.5, 0.9], i.e. close to linear) and a higher 40% lapse rate, the planned samples still yield about 100% power for Experiment 1 and for the Experiment 2 steep-vs-flat test, and about 90% power for the Experiment 2 level-shift test (the subtlest effect). Under a linear (non-concave) generating model with the same noise, every test's rejection rate is at its nominal .05 level (the joint Experiment-1 criterion rejects essentially never, because the concave-vs-WADD comparison cannot be passed by a linear chooser).

# Variables

## Manipulated variables

In Experiment 1 we manipulate, across the three subsets, which heuristic the trial structure pits the concave account against. In Experiment 2 we manipulate whether a feature advantage falls in the low vs high region of the value scale (steep-vs-flat) and whether the same feature-value structure is shown in a low range or shifted upward into a higher range (level-shift). Value/validity assignment, feature identity, option side, and trial order are randomized or counterbalanced as described above.

## Measured variable and derived indices

The measured variable is the participant's binary choice (option A vs option B) on each trial. From these choices we derive participant-level proportions.

**Experiment 1.** For each heuristic m in {WADD, tallying, TTB}, restricted to m's own discriminating subset (where the concave account predicts the option opposite to m):
- cmr_m = the proportion of m-subset trials on which the participant chose the concave-predicted option (concave-match rate).
- hmr_m = the proportion of m-subset trials on which the participant chose m's predicted option (m-match rate).
Because the concave account and m predict opposite options on every trial of the subset, cmr_m + hmr_m = 1; the pairwise comparison statistic is D_m = cmr_m − hmr_m.

**Experiment 2.**
- SteepChoiceRate = the proportion of steep-vs-flat trials on which the participant chose the low-range-advantage option.
- OffsetEffect = (proportion choosing the target option in the low-range version) − (proportion choosing it in the shifted-up version), where the target option is fixed per pair in advance.

# Analysis Plan

All confirmatory tests are one-tailed at alpha = .05 in the pre-registered predicted direction, computed on participant-level proportions pooled across the five validity vectors within each experiment. For every test we report the mean effect, standard error, 95% confidence interval, t statistic, degrees of freedom, one-tailed p-value, and effect size. The analysis is implemented in the attached `analyse.py` (`--experiment exp1` and `--experiment exp2`).

## Experiment 1 — Pairwise model comparison (H1)

For each heuristic m in {WADD, tallying, TTB} we test, with a one-sided paired t-test across participants, whether D_m = cmr_m − hmr_m > 0 (equivalently, whether the concave-match rate exceeds the m-match rate on m's subset). Because H1 comprises three planned comparisons (one per heuristic), the three one-tailed p-values are corrected with the Holm procedure: ordered from smallest to largest and compared in turn to .05/3, .05/2, and .05. We infer support for a given pairwise comparison if it is significant under this Holm criterion, and support for H1 overall if all three comparisons are significant after Holm correction.

## Experiment 2 — Value curvature (H2, H3)

H2 (steep-vs-flat): a one-sided one-sample t-test of whether SteepChoiceRate > 0.5 across participants. We infer support if the one-tailed p-value is below .05 in the predicted direction.

H3 (level-shift / offset): a one-sided one-sample t-test of whether OffsetEffect > 0 across participants. We infer support if the one-tailed p-value is below .05 in the predicted direction.

## Data inclusion, exclusion, and missing data

A participant is included in their experiment's confirmatory analyses if and only if they provide consent and complete the study (all 96 choice trials recorded). No outlier removal, trimming, or winsorizing is applied to the choice variables, and no exclusions are based on response patterns, response times, or accuracy. Participants whose data are incomplete or missing because of a technical failure or non-completion are excluded; no imputation is used.

## Exploratory analyses

Any analysis not described above is exploratory. This includes response-time analyses and any per-validity-vector or per-stimulus breakdowns. We will label such analyses as exploratory when reported.
