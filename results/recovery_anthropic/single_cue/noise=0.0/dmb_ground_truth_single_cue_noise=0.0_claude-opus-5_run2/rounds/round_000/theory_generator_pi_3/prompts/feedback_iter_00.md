# feedback_iter_00

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace pi_1 (Take The Best) entirely. It is parameter-free falsified: on Experiment 1's tally-tie trials it must predict ~0.90 for the top-cue winner and the data sit at 0.474, and it predicts 0.14 where the data show 0.65 in Experiment 2. No re-parameterisation rescues it except degenerate all-lapse.

Propose instead a 'limited-sample / noisy feature counting' theory that is a genuine rival to full Tallying while remaining consistent with BOTH observations:

Core claim: subjects do not integrate all cues, nor a single best cue. On each trial they inspect a limited random subset of the features (subset size drawn from a distribution with mean k < n; features sampled with EQUAL probability, not validity-weighted, since Exp 1's chance performance on tally-ties rules out validity weighting), count feature-wise wins within that sample, and choose the option ahead within the sample; a within-sample tie forces a guess. Optionally each inspected comparison is registered with error probability nu (attention/encoding noise), plus a lapse epsilon.

Why it fits both experiments: (i) On globally tally-tied, side-counterbalanced pairs, uniform subset sampling is symmetric, so choice is exactly 50/50 — reproducing Exp 1's 0.474 as parameter-freely as Tallying does. (ii) On large-margin conflict trials, adherence to the global tally winner is capped well below 1 because a small sample can reverse the sign — naturally yielding the attenuated ~0.65 observed in Exp 2 without needing an implausibly small beta.

Critical dissociation from pi_2 for the next experiment: softmax Tallying's choice probability depends only on the RAW margin M, whereas the sampling account depends on the PROPORTION of informative cues won (and on the number of tied/uninformative cues present). So design pairs holding M constant while varying the informative-cue base rate — e.g. a 2-0 split (M=2, 2 informative cues, 4 tied) versus a 4-2 split (M=2, 6 informative cues). Tallying predicts identical choice probabilities; the sampling model predicts markedly higher adherence for 2-0 and dilution when many tied cues pad the profile. Ensure the new theory's parameters (mean sample size k, encoding noise nu, epsilon) are ranged so that its Exp 1 prediction stays at 0.5 and its Exp 2 prediction lands near 0.60-0.70.


## CANDIDATE THEORY
**Limited-Sample Noisy Cue Counting (LSNC).** People neither integrate all cues (Tallying) nor consult a single best cue (Take-The-Best). Instead, on each choice they inspect only a *limited, randomly selected subset* of the presented expert ratings, tally feature-wise wins **within that sample only**, and pick the option that is ahead in the sample; a within-sample tie (including a sample containing no discriminating cue) forces a guess.

Three commitments define the theory:
1. **Equal-probability sampling.** The inspected subset is drawn uniformly at random from the whole profile — it is *not* validity-ordered and *not* validity-weighted. Stated validities are too weakly represented to steer the search order, so no cue enjoys priority. Consequently, on globally tally-tied, side-counterbalanced profiles the process is exactly symmetric and choice is 50/50, with no tendency to follow the most valid discriminating cue.
2. **Sample-size limitation with tie dilution.** The number of cues actually inspected, K, is a random variable with mean k < n (implemented as K = 1 + Binomial(n−1, q), so at least one cue is always read). Because the subset is drawn from *all* features, non-discriminating (tied) features occupy sampling slots and dilute the evidence: two profiles with the same raw tally margin M produce different adherence depending on how many uninformative cues pad the profile, and on the proportion (not the raw count) of informative cues won. This is the signature that dissociates LSNC from softmax Tallying, whose predictions depend only on M.
3. **Encoding noise + lapse.** Each inspected feature-wise comparison is registered with the wrong sign with probability nu (attention/encoding noise), and with probability epsilon the whole decision is replaced by a coin flip.

Because a small, noisy sample can easily reverse the sign of a large global margin, adherence to the full-tally winner is *capped well below 1* even on high-margin conflict trials — the systematic attenuation that full Tallying cannot produce without degenerate temperature — while symmetry guarantees exactly chance behaviour on tally-tied pairs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    from math import comb

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("LSNC expects a (2, n_features) stimulus; got shape %s." % (stim.shape,))
    a, b = stim[0], stim[1]
    n = int(a.shape[0])

    nA = int(np.sum(a > b))          # cues on which A wins
    nB = int(np.sum(b > a))          # cues on which B wins
    nT = n - nA - nB                 # uninformative (tied) cues

    k = float(parameters["k"])            # mean number of cues inspected
    nu = float(parameters["nu"])          # per-comparison encoding error
    eps = float(parameters["epsilon"])    # lapse rate
    nu = float(min(max(nu, 0.0), 0.5))
    eps = float(min(max(eps, 0.0), 1.0))

    if n == 0:
        return np.ones(2) / 2.0

    # ---- distribution of the inspected sample size K in {1..n} -------------
    if n == 1:
        pK = np.array([1.0])
    else:
        q = (k - 1.0) / (n - 1.0)
        q = float(min(max(q, 0.0), 1.0))
        pK = np.array([comb(n - 1, s - 1) * (q ** (s - 1)) * ((1.0 - q) ** (n - s))
                       for s in range(1, n + 1)], dtype=float)
        tot = pK.sum()
        pK = pK / tot if tot > 0 else np.ones(n) / float(n)

    # ---- exact enumeration over sample compositions ------------------------
    pA = 0.0
    for si, s in enumerate(range(1, n + 1)):
        w = float(pK[si])
        if w <= 0.0:
            continue
        denom = float(comb(n, s))
        for ai in range(0, min(nA, s) + 1):
            for bi in range(0, min(nB, s - ai) + 1):
                ti = s - ai - bi
                if ti < 0 or ti > nT:
                    continue
                cnt = comb(nA, ai) * comb(nB, bi) * comb(nT, ti)
                if cnt == 0:
                    continue
                ph = float(cnt) / denom          # multivariate hypergeometric
                m = ai + bi                      # informative cues in sample
                if m == 0:
                    pA += w * ph * 0.5           # nothing discriminates -> guess
                    continue
                # X = number of sampled comparisons REGISTERED as A-wins
                d1 = np.array([comb(ai, x) * ((1.0 - nu) ** x) * (nu ** (ai - x))
                               for x in range(ai + 1)], dtype=float)
                d2 = np.array([comb(bi, x) * (nu ** x) * ((1.0 - nu) ** (bi - x))
                               for x in range(bi + 1)], dtype=float)
                dist = np.convolve(d1, d2)
                xs = np.arange(m + 1)
                p_win = float(dist[2 * xs > m].sum())
                p_tie = float(dist[2 * xs == m].sum())
                pA += w * ph * (p_win + 0.5 * p_tie)

    pA = float(min(max(pA, 0.0), 1.0))
    p = np.array([pA, 1.0 - pA], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- k: [1.5, 3.5]
- nu: [0.12, 0.32]
- epsilon: [0.0, 0.12]

`rationale`:
Take-The-Best (pi_1) is falsified in both directions (0.90 vs 0.474; 0.14 vs 0.65) and softmax Tallying (pi_2), while parameter-freely correct at chance on Experiment 1's tally ties, over-predicts conflict-trial adherence (0.85 vs 0.65) because a large raw margin drives its softmax to near-determinism unless beta collapses toward guessing everywhere. LSNC dissolves this tension with a single mechanism.

(1) **Experiment 1 is fit parameter-freely.** On tally-tied, side-counterbalanced profiles nA = nB, and uniform (validity-blind) subset sampling plus symmetric encoding noise make the whole process exchangeable in A/B, so p(A) = 0.5 exactly for every parameter setting. Predicted TTB-agreement = 0.50 (real 0.4738), and the only across-subject variance is binomial (0.25/32 ≈ 0.0078 vs. observed 0.0073) — matching not just the mean but the observed between-subject spread.

(2) **Experiment 2's attenuation falls out mechanically.** The two conflict configurations are (nA=1, nB=4, nT=1) and (nA=1, nB=3, nT=2). Exact enumeration gives adherence to the tally winner of ~0.60 (K=1), ~0.63 (K=2), ~0.67 (K=3) at nu ≈ 0.25, rising only slowly toward 1 as K → n. With the declared ranges (mean k ≈ 2.5, nu ≈ 0.22, eps ≈ 0.06) the pooled prediction is ≈ 0.655, essentially the observed 0.6538, and the parameter spread contributes ≈ 0.002 on top of the 0.007 binomial variance, landing near the observed 0.0092. Crucially, adherence is *capped*: no amount of increasing determinism drives it to 1 unless the sample becomes the full profile, so the model is not merely a re-tuned Tallying.

(3) **It is a genuine rival with a testable dissociation.** Because the subset is drawn from all cues — including tied ones — adherence depends on the *proportion* of informative cues won and is *diluted* by uninformative cues, not on the raw margin M. Note above that the 1-vs-4 (margin 3, one tied cue) and 1-vs-3 (margin 2, two tied cues) cases already differ by ~0.03–0.05 at fixed parameters in a direction Tallying cannot represent; a 2-0 versus 4-2 design (both M=2) yields a large predicted gap under LSNC and none under Tallying. The model uses no validity information at all, honouring the Experiment 1 evidence against validity-ordered or validity-weighted search, and it generalises to any n_features and any validity vector without refitting.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0151 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0151.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = [0.95, 0.85, 0.75, 0.62, 0.55]
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fall back: assume features already ordered by descending validity
            v = np.arange(n, 0, -1).astype(float)
        order = np.argsort(-v, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # only tally-tie trials

        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no discriminating cue at all

        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4738 (var=0.0073)
**Candidate (simulated) value:** 0.4844 (var=0.0072)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8988 (var=0.0090)
- pi_2: 0.4988 (var=0.0068)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    val = np.array([0.93, 0.57, 0.79, 0.86, 0.62, 0.71], dtype=float)
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if val.shape[0] == n:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(n)
        # tally margin (A wins minus B wins)
        margin = int(np.sum(a > b) - np.sum(b > a))
        if abs(margin) < 2:
            continue
        tally_winner = 0 if margin > 0 else 1
        # take-the-best winner
        ttb_winner = None
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        if ttb_winner == tally_winner:
            continue  # keep only conflict trials
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == tally_winner else 0.0)
    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6538 (var=0.0092)
**Candidate (simulated) value:** 0.6575 (var=0.0100)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8525 (var=0.0139)
- pi_1: 0.1431 (var=0.0111)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate implements exactly the mechanism family the arbiter prescribed \u2014 uniform (validity-blind) limited-subset sampling, within-sample tallying with tie-guessing, per-comparison encoding noise nu, and a lapse epsilon \u2014 and it recovers both empirical anchors. Experiment 1: the model is exchangeable in A/B on tally-tied, side-counterbalanced profiles, so its TTB-agreement is a parameter-free 0.50; simulation gives 0.4844 against an observed 0.4738 (difference well inside the binomial spread), and the between-subject variance is matched almost exactly (0.0072 vs 0.0073). This is as good as pi_2 on this metric and decisively better than pi_1 (0.8988), which the arbiter had already ruled out. Experiment 2: the sampling cap produces the attenuated conflict-trial adherence that neither rival can produce \u2014 0.6575 simulated vs 0.6538 observed, versus 0.8525 for softmax Tallying and 0.1431 for TTB. The between-subject variance (0.0100 vs 0.0092) is also in the right neighbourhood, indicating the parameter ranges (k in [1.5,3.5], nu in [0.12,0.32], eps in [0,0.12]) generate roughly the right heterogeneity rather than merely hitting the pooled mean with a degenerate setting. Importantly, the fit is not achieved by re-tuning a determinism parameter: adherence stays bounded below 1 by construction, which is the theoretically load-bearing property. The predict function is an exact enumeration (multivariate hypergeometric over sample composition, convolution over sign-flip noise) rather than a sampling approximation, so predictions are stable, and it generalises to arbitrary n and validity vectors without refitting. The only residual is that Exp 1's data sit ~0.026 below the model's structural 0.5; that is a slight side/first-cue avoidance that no symmetric account can express, and it is small enough not to motivate breaking symmetry (doing so would jeopardise the parameter-free chance prediction that is the theory's main asset).",
  "verdict": "continue",
  "rationale": "The candidate is faithful to the arbiter's prescribed 'limited-sample / noisy feature counting' family \u2014 equal-probability subset sampling, within-sample tally, tie-guess, encoding noise, lapse \u2014 with no drift toward validity-weighted search or a softmax-on-raw-margin rule. It was ACCEPTED by the gate at loss=0.0151 and matches both experiments on point estimate and between-subject variance (Exp 1: 0.4844 vs 0.4738, var 0.0072 vs 0.0073; Exp 2: 0.6575 vs 0.6538, var 0.0100 vs 0.0092), which is simultaneously better than either rival on the metric they respectively fail. Crucially the fit is experiment-invariant in the intended sense: Exp 1 is captured parameter-freely by the model's A/B exchangeability, and Exp 2's attenuation falls out of the sample-size cap rather than from tuning a determinism knob, so the same parameter region explains both without trade-off. It also carries the arbiter's requested dissociation from softmax Tallying (dependence on the proportion of informative cues and dilution by tied cues rather than on raw margin M), which makes it directly testable by a 2-0 vs 4-2 design in the next round. No in-family knob is obviously mis-set: k~2.5 and nu~0.22 land Exp 2 in the 0.60-0.70 target band as specified, and shrinking nu or widening k would only push Exp 2 adherence off target while leaving Exp 1 untouched. Ship this candidate."
}
```

## Usage

```json
{
  "input_tokens": 9506,
  "output_tokens": 1417
}
```
