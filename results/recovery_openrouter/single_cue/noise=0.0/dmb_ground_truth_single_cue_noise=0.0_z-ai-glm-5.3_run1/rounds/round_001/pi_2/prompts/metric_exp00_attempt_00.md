# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=8):** [0.99, 0.98, 0.97, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  trial 9: A=[1, 0, 0, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1]
  trial 12: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1]
  trial 13: A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  trial 14: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  trial 15: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Rationale:** ADVOCATED THEORY: unweighted Tallying (choice depends ONLY on the raw tally margin d = #A-cue-wins - #B-cue-wins, with unbounded beta up to 20). COMPETITOR: validity-weighted tallying whose own theoretical claim pins it to a bounded box: gamma in [0.05, 0.7], beta in [0.03, 0.55], epsilon in [0, 0.08] ('unsaturated regime'). The design exploits two structural facts about that box. FACT 1 (HARD CEILINGS): the competitor's maximum achievable confidence on a trial is (1-eps)*sigmoid(2*beta*m) with m = sum_j v_j^gamma*sign(a_j-b_j). Because beta <= 0.55 and any single cue's weight is <= 1, on ANY trial where exactly one more cue favors A than B and only one cue discriminates (S1, S2), m <= 0.9995, so the competitor's absolute ceiling is p(A) = 0.750 -- no parameter setting in its box can exceed it. On minimal-m d=+2 (A wins only the two lowest-validity cues, L2) the ceiling is 0.894, and on minimal-m d=+3 (L3) it is 0.961. Tallying, with beta up to 20, can predict ~0.99 on all of these. So if subjects choose the count-winner on >80% of S1/S2 trials or >92% of L2 trials, the competitor is falsified outright with zero free parameters -- the exact mirror-image knife of Experiment [2], but aimed at the bounded regime rather than at gamma. FACT 2 (MINIMAL-m LADDER): on the ladder S2 -> L2 -> L3 every A-win is on a low-validity cue, so the competitor's evidence per tally step is only ~0.62-0.97 of a raw count unit; its steepest possible psychometric slope in d is 2*0.55*0.97 ~ 1.07 logits per step, whereas Tallying's is 2*beta, unbounded. A subject psychometric function in d that is steeper than ~1.07 logits/unit (e.g., 0.75 at d=1 rising to ~0.99 at d=3) exceeds the competitor's box. DISSOCIATION FAMILY d=+/-1 (S1, S2, S3, S4 and mirrors): all four A-tilted trials have d = +1, so Tallying makes the PARAMETER-FREE prediction p(S1)=p(S2)=p(S3)=p(S4) exactly, for every (beta, epsilon). The competitor predicts a large within-d spread because m differs sharply across compositions: at gamma=0.7, beta=0.55 its predictions are S1=0.749, S2=0.665, S4=0.816, and -- the sharpest cell -- S3=0.389, i.e., BELOW 0.50. S3 is the sign-flip trial: A wins the four weakest experts (v=0.54,0.53,0.52,0.51) while B wins the three super-experts (v=0.99,0.98,0.97); the raw count favors A (4 vs 3) but the validity-weighted margin at high gamma is m = -0.41, so the competitor predicts a B preference while Tallying predicts the same strong A preference as on S1. Subject behavior on S3 is a direct gamma-map: p(A) near p(S1) kills the competitor twice (ceiling violation + zero validity weighting); p(A) < 0.50 falsifies Tallying on a choice (not just confidence) measure. The validity vector [0.99, 0.98, 0.97, 0.55, 0.54, 0.53, 0.52, 0.51] is engineered for this: the top-two/top-three weights (0.993+0.986+0.979 = 2.958 at gamma=0.7) exceed any three or four low-cue weights (e.g., 2.548 for the four lowest), which is what makes the sign flip achievable at all with binary cues. HONEST ADVERSARIAL KNIVES AGAINST MY OWN THEORY: (i) T0 (d=0: A wins the three super-experts, B wins the three worst): Tallying MUST predict exactly 0.50 (tied softmax is uniform and the lapse injects uniform), while the competitor predicts p(A) up to 0.76 -- any systematic A-bias on T0 falsifies Tallying with no parameters; (ii) the within-d=1 spread S3<S2<S1<S4 and the L2-vs-M2 comparison (both d=+2; Tallying demands exact equality, competitor predicts L2 < M2 by up to ~10 points since M2's wins are on the two super-experts, m ~ 2.0 vs 1.26-1.93) are the competitor's best cells, included deliberately; (iii) displaying the three ~99% experts first maximizes validity salience, favoring the competitor's processing assumptions. ANCHORS: G+/G- (d = +/-6) are near-agreement trials (both theories ~1) that pin lapse rates; even here the competitor's epsilon <= 0.08 forces p in [0.958, 0.998] on G+, so an anchor error rate above ~5% strains it while Tallying's epsilon up to 0.5 absorbs any sloppiness. PREDICTED PATTERN IF MY THEORY IS RIGHT: flat near-ceiling performance across the entire d=1 family (all four compositions equal, ~0.95+), steep sigmoid in d reaching ~0.99 by d=3, exactly 0.50 on T0, L2 = M2. PREDICTED PATTERN IF THE COMPETITOR IS RIGHT: graded 0.65-0.75 on d=1 with a 15-40 point within-d spread, S3 at or below 0.50, shallow d-slope, A-bias on T0. The two patterns are disjoint almost everywhere: 16 unique pairs x K=6 = 96 trials; sides balanced (8 count-favors-A, 7 count-favors-B, 1 tied); no all-tie trials; every trial requires scanning all 8 experts to locate discriminators.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



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
**Description:** Validity-Weighted Tallying (graded-evidence regime): people integrate evidence from ALL binary expert ratings, casting each feature a sign vote (+1 win / -1 loss / 0 tie) weighted by w_j = validity_j^gamma, summing into a zero-sum evidence score, with choice a softmax (inverse temperature beta) over the two scores plus an independent uniform lapse (epsilon). The theoretical claim added this round is about where humans sit in this family: in the UNSATURATED softmax regime. Because the evidence scores are integer-scale sign tallies, beta >= 3 already drives tanh(beta*m) ~ 1 for every unit margin, making the model behaviorally deterministic and pinning it to the tallying corner; the graded dose-response in tally margin that both experiments demand (Exp 1 metric at 44% of the deterministic-tally value; Exp 2's small positive b_d) lives at beta ~ 0.2-0.3, where p(A) is approximately linear in the tally margin. Gamma stays in a mild, provably sub-lexicographic band so that no subject leaps into the TTB regime.

**Parameters:**
- gamma: [0.05, 0.7]
- beta: [0.03, 0.55]
- epsilon: [0.0, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Validity-Weighted Tallying.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each feature j casts a vote sign(a_j - b_j) for A's score and
    # the opposite for B's score; votes are weighted by w_j =
    # validity_j ** gamma. gamma = 0 -> unweighted Tallying;
    # gamma -> large -> lexicographic TTB (top cue dominates).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    # Validity-based weights. Since validities are in [0.5, 1.0],
    # gamma = 0 gives all-ones weights (pure Tallying); increasing
    # gamma increasingly privileges high-validity cues, with the
    # top cue dominating as gamma grows (TTB limit).
    w = np.power(val, gamma)

    # Weighted evidence scores. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable softmax over the two weighted scores.
    # When all cues tie (s_a == 0) the softmax is exactly uniform,
    # which is the correct behavior for an undiscriminating tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Hard-ceiling test on the |d|=1 family (FACT 1 of the design).\n\n    Metric = proportion of choices that agree with the UNWEIGHTED tally\n    count-winner, restricted to trials whose raw tally margin is exactly\n    +/-1 (the 8 trial types t1-t8; 48 of the 96 trials per subject).\n\n    Tallying (any beta >= 1, any epsilon) predicts ~1 - epsilon/2 here,\n    i.e. >= ~0.85 for any moderate lapse. The validity-weighted\n    competitor is structurally capped: its choice probability on a |d|=1\n    trial is (1-eps)*sigmoid(2*beta*m) with m = sum_j v_j^gamma*sign diff;\n    the mean of |m| over the 8 |d|=1 trial types is <= 1 for every\n    gamma in [0.05, 0.7] (equality only at gamma=0), and beta <= 0.55,\n    so by concavity of tanh its mean agreement on this family is\n    <= 0.5 + tanh(0.55)/2 = 0.750 over its ENTIRE parameter box.\n    \"\"\"\n    def _mat(col):\n        vals = []\n        for v in col:\n            if isinstance(v, np.ndarray):\n                vals.append(v)\n            else:\n                vals.append(list(v))\n        return np.asarray(vals, dtype=float)\n\n    A = _mat(data[\"option_a_ratings\"])\n    B = _mat(data[\"option_b_ratings\"])\n    resp = np.asarray(data[\"response\"], dtype=float)\n\n    if A.ndim != 2 or A.shape[0] == 0 or A.shape != B.shape:\n        return 0.5\n\n    # Raw (unweighted) tally margin per trial.\n    d = (A > B).sum(axis=1) - (B > A).sum(axis=1)\n\n    # The |d| = 1 family: 8 unique pairs x 6 reps = 48 trials/subject.\n    mask = np.abs(d) == 1\n    if mask.sum() < 10:\n        # Defensive fallback: use all non-tied trials.\n        mask = d != 0\n    if mask.sum() == 0:\n        return 0.5\n\n    dm = d[mask]\n    rm = resp[mask]\n    # Count-winner is A when d>0 (response 0), B when d<0 (response 1).\n    consistent = np.where(dm > 0, rm == 0, rm == 1)\n    return float(np.mean(consistent))",
  "rationale": "This metric implements the HARD-CEILING knife (FACT 1) of the design on the trial family where the two theories are maximally separated in a parameter-robust way. On every trial with raw tally margin |d|=1 (trial types t1-t8: single top-cue discriminator, single low-cue discriminator, 4-low-vs-3-top sign-flip pair, 2-top-vs-1-low pair, and their mirrors; 48 of 96 trials per subject), the competitor's predicted agreement with the count-winner is (1-eps)*sigmoid(2*beta*m) with m = sum_j v_j^gamma * sign(a_j - b_j). Because gamma <= 0.7, beta <= 0.55, epsilon <= 0.08, and the mean of |m| across the 8 trial types is at most 1 (it equals 1 only in the gamma->0 limit and shrinks as gamma grows, since the low-validity discriminators lose weight), concavity of tanh bounds the competitor's expected value on this metric by 0.5 + tanh(0.55)/2 = 0.750 over its ENTIRE box \u2014 and at its self-described operating point (beta ~ 0.25-0.3, mild gamma) it sits near 0.55-0.62. Tallying, in contrast, predicts (1-eps)*sigmoid(2*beta) + eps/2 on every one of these trials: for any beta >= 1 (most of its [0.1, 20] range) this is >= 0.88*(1-eps) + eps/2, i.e. roughly 0.85-0.95 for moderate lapses. So the theories make almost disjoint predictions on this scalar, and the gap (~0.3 in typical parameterizations, >= 0.12 even in the competitor's most favorable corner gamma->0, beta=0.55) is large relative to per-subject noise: each subject contributes 48 Bernoulli trials, so the per-subject standard deviation is only ~0.05 (tallying-like data) to ~0.10 (competitor-like data), giving Welch t on the order of 10-20 at N=25. The metric is well-defined both on the pooled data and on each single subject's slice (every subject sees all 8 |d|=1 trial types 6 times), so the between-subject variance is a proper binomial-style variance rather than n/a. It is not trivially fixed under my own theory (its value depends on beta and epsilon, and would drop toward 0.5 if subjects did not tally), and on human data it directly adjudicates the theories' conflicting predictions: near-ceiling count-winner adherence on minimal-margin trials falsifies the bounded unsaturated-regime claim, while values pinned below 0.75 falsify high-beta tallying."
}
```

## Usage

```json
{
  "input_tokens": 5436,
  "output_tokens": 42068,
  "reasoning_tokens": 40823,
  "provider": "Sail Research"
}
```
