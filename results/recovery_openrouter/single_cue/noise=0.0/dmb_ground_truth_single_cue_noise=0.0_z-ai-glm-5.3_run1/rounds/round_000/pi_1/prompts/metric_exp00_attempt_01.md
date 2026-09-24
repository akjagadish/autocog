# metric_exp00_attempt_01

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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** This design creates two orthogonal quantitative dissociations between Take-The-Best (TTB) and Tallying, using 5 cues with well-spread validities (0.95, 0.85, 0.75, 0.65, 0.55). DISSOCIATION 1 (graded vs. flat dose-response curve): Trials C1-C3 hold the TTB-relevant structure constant (the top cue v=0.95 favors A, so TTB always picks A with probability (1-eps)*sigmoid(beta)+eps/2) while systematically increasing the number of lower-validity cues favoring B (tally margins 1-vs-2, 1-vs-3, 1-vs-4). TTB predicts a FLAT p(A) across these three trial types — the number of opposing low-validity cues is irrelevant to a one-reason rule. Tallying predicts a strictly MONOTONE DECREASING p(A) = sigmoid(-k*beta) for k=1,2,3 — a graded dose-response in tally margin. Trials C5-C7 mirror this with B favored by the top cue (TTB flat-high p(B); Tallying increasingly favors A). This is a pure quantitative dissociation: the slope of p(choice) against tally margin is exactly 0 for TTB and strictly negative for Tallying at any beta > 0. DISSOCIATION 2 (parameter-free point prediction): Trials T1-T4 have equal tallies (1 win each) but a clear validity-ordered discriminator (e.g., T1: A wins the v=0.95 cue, B wins only the v=0.55 cue). Tallying MUST predict exactly p(A)=0.50 on these trials for ALL parameter values (the softmax over tied scores is uniform, and the lapse also injects uniform). TTB predicts p(A) = (1-eps)*sigmoid(beta)+eps/2, which is strictly above 0.50 for any beta>0. Thus, if subjects choose systematically (not at chance) on T1/T3 (toward A) or T2/T4 (toward B), Tallying is falsified outright with no free parameters to rescue it — an adversarial knife. Trials G1-G2 are agreement controls (both theories pick the same option) that anchor the noise parameters. 12 unique pairs x K=8 = 96 trials per subject. Note the design also avoids any all-tie (fully degenerate) trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
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
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
[0] rationale: This is a per-trial 'dissociation score' that exploits both orthogonal dissociations built into the design. On every trial I compute (i) the TTB pick (features are in descending-validity order, so the TTB cascade is just the lowest-index discriminating feature — no validity values needed) and (ii) the Tallying pick (strict win counts). Trials where the two theories agree (types 11-12) are skipped as uninformative. On dissociation trials 1-6 (TTB backs the option with FEWER cue wins, Tallying backs the majority-tally option), each choice contributes +1 if it matches TTB and -1 if it matches Tallying. On tied-tally trials 7-10, Tallying is structurally forced to predict exactly p=0.5 for every parameter value (softmax over tied scores is uniform, and the lapse injects uniform), so a choice toward TTB's pick contributes +0.5 and away contributes -0.5. Under TTB-simulated data the expected score is strongly positive: on dissociation trials it is 2*[(1-eps)*sigmoid(beta)+eps/2] - 1 > 0, and on tied trials it is (1-eps)*(sigmoid(beta)-0.5) > 0. Under Tallying-simulated data the expected score is strongly negative: -(2*p_majority - 1) on dissociation trials and exactly 0 on tied trials. The metric pools ~80 informative trials per subject, so per-subject sampling noise is small (sd roughly 0.05-0.09 depending on parameters), while the between-theory mean gap is large (order 0.5-1.4 across plausible beta/epsilon), giving a decisive Welch's t-test at N=25. It works identically on the pooled DataFrame and on single-subject slices, and it hedges against weak parameters: if TTB is simulated with low beta, the tied-trial signal shrinks but the slope-based dissociation trials still separate; if Tallying is simulated with low beta, the dissociation trials shrink but TTB's tied-trial bias persists. Unlike a raw dose-response slope, this combined score uses every bit of diagnostic signal in the design in a single scalar.
metric_source:
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def _to_int_vec(cell):
        # Robustly convert a ratings cell (list/tuple/ndarray/str) to list of ints.
        if isinstance(cell, str):
            cell = cell.strip().lstrip("[").rstrip("]")
            return [int(float(v)) for v in cell.replace(",", " ").split()]
        return [int(float(v)) for v in list(cell)]

    df = data.reset_index(drop=True)
    contribs = []

    for i in range(len(df)):
        # --- response ---
        r_raw = df["response"].iloc[i]
        try:
            r = int(r_raw)
        except (TypeError, ValueError):
            continue

        # --- feature vectors ---
        try:
            a = _to_int_vec(df["option_a_ratings"].iloc[i])
            b = _to_int_vec(df["option_b_ratings"].iloc[i])
        except (TypeError, ValueError):
            continue
        n = min(len(a), len(b))
        if n == 0:
            continue

        # --- Take-The-Best pick: features are presented in descending
        # validity order in this design, so the cascade is simply the
        # lowest-index discriminating feature. ---
        ttb = None
        for j in range(n):
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb is None:
            continue  # fully tied pair: both theories guess; uninformative

        # --- Tallying pick: strict feature-wise win counts ---
        a_wins = sum(1 for j in range(n) if a[j] > b[j])
        b_wins = sum(1 for j in range(n) if b[j] > a[j])

        if a_wins == b_wins:
            # Tied tally: Tallying is exactly indifferent (p = 0.5 for all
            # beta, epsilon), while TTB has a systematic preference.
            contribs.append(0.5 if r == ttb else -0.5)
        else:
            tal = 0 if a_wins > b_wins else 1
            if tal == ttb:
                continue  # theories agree here; uninformative for the contrast
            # Full dissociation trial: TTB backs the minority-tally option,
            # Tallying backs the majority-tally option.
            contribs.append(1.0 if r == ttb else -1.0)

    if len(contribs) == 0:
        return 0.0
    return float(np.mean(contribs))
```
outcome: evaluation failed (SyntaxError: invalid syntax (<string>, line 1))

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "```python\ndef metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    def _vec(cell):\n        # Robustly convert a ratings cell (list/tuple/ndarray/str) to a list of ints.\n        if isinstance(cell, str):\n            s = cell.strip().strip(\"[]\")\n            return [int(float(v)) for v in s.replace(\",\", \" \").split()]\n        return [int(float(v)) for v in list(cell)]\n\n    df = data.reset_index(drop=True)\n\n    a_col = df[\"option_a_ratings\"].tolist()\n    b_col = df[\"option_b_ratings\"].tolist()\n    r_col = df[\"response\"].tolist()\n\n    contribs = []\n\n    for i in range(len(df)):\n        try:\n            r = int(r_col[i])\n            a = _vec(a_col[i])\n            b = _vec(b_col[i])\n        except (TypeError, ValueError):\n            continue\n\n        n = min(len(a), len(b))\n        if n == 0:\n            continue\n\n        # --- Take-The-Best pick ---\n        # In this design the features are listed in descending-validity\n        # order (validities = [0.95, 0.85, 0.75, 0.65, 0.5]), so the TTB\n        # cascade is simply the lowest-index feature on which the two\n        # options differ. No validity values are needed.\n        ttb = -1\n        for j in range(n):\n            if a[j] > b[j]:\n                ttb = 0\n                break\n            if b[j] > a[j]:\n                ttb = 1\n                break\n        if ttb < 0:\n            # Fully tied pair: both theories guess; uninformative.\n            continue\n\n        # --- Tallying pick: strict feature-wise win counts ---\n        a_wins = len([j for j in range(n) if a[j] > b[j]])\n        b_wins = len([j for j in range(n) if b[j] > a[j]])\n\n        if a_wins == b_wins:\n            # Tied tally: Tallying is structurally forced to p = 0.5 for\n            # EVERY parameter value (softmax over tied scores is uniform,\n            # and the lapse also injects uniform). Any systematic choice\n            # here is pure TTB signal.\n            contribs.append(0.5 if r == ttb else -0.5)\n        else:\n            tal = 0 if a_wins > b_wins else 1\n            if tal == ttb:\n                # Theories agree on this trial; uninformative for the contrast.\n                continue\n            # Full dissociation trial: TTB backs the option with FEWER cue\n            # wins, Tallying backs the majority-tally option.\n            contribs.append(1.0 if r == ttb else -1.0)\n\n    if len(contribs) == 0:\n        return 0.0\n    return float(np.mean(contribs))\n```",
  "rationale": "The previous submission of this metric failed purely on a SyntaxError (a generator expression was mangled by the string pipeline), not on statistics \u2014 so this version keeps the same statistically strong construction while using only plain list comprehensions and indexed row access, which are syntactically bulletproof. Construction: on every trial I compute (i) the TTB pick (features are in descending-validity order in this design, so the cascade is just the lowest-index discriminating feature) and (ii) the Tallying pick (strict win counts). Trials 11-12 (types where both theories agree) are skipped as uninformative. On dissociation trials 1-6 (TTB backs the option with FEWER cue wins, Tallying backs the majority-tally option), each choice contributes +1 if it matches TTB and -1 if it matches Tallying. On tied-tally trials 7-10, Tallying is structurally forced to predict exactly p=0.5 for all parameter values, so a choice toward TTB's pick contributes +0.5 and away contributes -0.5. Expected value under TTB-simulated data is strongly positive: 2*[(1-eps)*sigmoid(beta)+eps/2] - 1 > 0 on dissociation trials and half that on tied trials. Under Tallying-simulated data it is strongly negative: -(2*p_majority - 1) on dissociation trials and exactly 0 on tied trials. The metric pools all ~80 informative trials per subject, so per-subject sampling noise is small (sd roughly 0.05-0.10 across plausible parameters) while the between-theory mean gap is large (order 0.5-1.4), giving a decisive Welch's t-test at N=25. It is also self-hedging: if TTB is simulated with low beta, the dissociation-trial signal shrinks but tied-trial bias toward the top cue persists; if Tallying is simulated with low beta, the tied trials contribute pure noise around zero while the dissociation trials still separate. The function works identically on the pooled DataFrame and on single-subject slices (it is a simple mean of per-trial contributions), so the between-subject variance is well defined."
}
```

## Usage

```json
{
  "input_tokens": 5103,
  "output_tokens": 2963,
  "reasoning_tokens": 1750,
  "provider": "Sail Research"
}
```
