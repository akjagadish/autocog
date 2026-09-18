# feedback_iter_01

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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace pi_4 entirely. Its strict-unanimity consensus premium is falsified in four independent experiments (1, 2, 7, 8) with large, sign-reversed errors; no re-parameterisation can save it, because the premium is direction-carrying and the data show the *opposite* direction at both unanimity boundaries.

Proposed replacement \u2014 'Expectation-Referenced Cue Coding (ERC)': choice is compensatory and stationary, but each displayed rating is scored relative to a subjective endorsement threshold rather than as pure positive evidence.

(1) Signed cue values: s_j = c*(logit v_j)^gamma - theta, with theta calibrated so the crossover validity v* sits around .65-.78 (person-specific). A '1' from a high-validity expert is positive evidence; a '1' from a low-validity expert is *negative* ('faint praise is damning' \u2014 an indiscriminate endorser's approval signals a mediocre product). D = sum_j s_j (a_j - b_j). This is a single mechanism that replaces both pi_4's rho-premium and pi_5's separate theta term.

(2) NO endorsement-density dilution. Instead, if any normalisation is needed, divide by a *contrast count* term (c0 + kap * number of discriminating columns), which is blind to the shared endorsement background. This is the sharp point of departure from pi_5: it predicts that 5/5-vs-4/5 (weak dissent) is strongly sub-chance (~.20, as observed in Exp 2) rather than near-chance, that single-endorsement-vs-empty and 5/5-vs-4/5 cells sit at similar depths below chance (Exp 7 diff ~0, Exp 1 diff ~0), and that pi_5's 'Family T' density ladders (matched counts, matched coherence, growing shared background) should be FLAT, not compressed \u2014 an immediately testable dissociation.

(3) Retain a readability/column-binding module, since Exps 4 and 6 require it: when the display is monotone in validity the validity-derived signed weights are used intact; when it is scrambled (or the best expert is interior) the weights are flattened toward equal weighting and a left-to-right positional gradient is recruited. Keep this direction-free (ascending or descending both cheap).

(4) Broad continuous heterogeneity in beta, gamma, theta (i.e. in the crossover validity v*), the readability coefficients and lapse; no discrete strategy switching, no learning (no feedback given).

Parameter ranges should be chosen so that: Exp 2 metric lands near 0.20-0.30; Exp 1 metric near 0.0-0.15; Exp 7 near 0; Exp 8 near -0.15; Exp 5 near -0.21; Exp 3 near 0.75; Exp 4 near 0.51; Exp 6 near 0.66. The next discriminating experiment should pit ERC against pi_5 on matched-count / matched-coherence density ladders (flat vs compressed) and on cells that vary which expert supplies the extra endorsement (ERC predicts a sign flip as v crosses v*; pi_5 predicts only a magnitude change scaled by the shared background).


## CANDIDATE THEORY
**ERC-K2: Expectation-Referenced Cue Coding with a deep crossover and weak contrast-count normalisation.**

Choice between two binary-cue panels is compensatory, stationary (no feedback is given, so nothing is learned) and rests on a single mechanism: **every displayed rating is scored relative to a subjective endorsement threshold, not as pure positive evidence.**

(1) *Signed cue coding with a DEEP crossover.* Expert j's stated validity v_j is converted to diagnosticity on the log-odds scale, x_j = (logit v_j)^γ, and referenced against a person-specific expectation θ = (logit v*)^γ, giving a **signed** cue value s_j = x_j − θ. The theory's substantive commitment, sharpened here, is that the crossover v* sits HIGH — above all but the top one or two experts in a typical panel (v* ≈ .81–.92). Approval from anyone appreciably less than near-certain is faint praise and is read as *negative* evidence; its absence is reassuring. Consequently a unanimous 5/5 panel loses to a 4/5 panel whose lone dissent sits on a weak expert (~.20), a lone weak endorsement is *worse* than an empty panel, and a lone strong endorsement is only mildly preferred — so the ‘extra endorsement from a strong expert’ and ‘extra endorsement from a weak expert’ cells sit close together (small positive Exp-7 contrast) rather than far apart.

(2) *No endorsement-density dilution; weak contrast-count normalisation.* Sensitivity is normalised only by the **contrast count** K, the number of columns on which the two panels actually disagree: D = Σ_j s_j(a_j−b_j) / (c0 + κ·K), and κ is small — multi-column contrasts are compressed only mildly, never flattened to chance. The normaliser is blind to the shared endorsement background, so density ladders holding contrast and coherence fixed must be FLAT. The weak compression is what lets a *coalition* of low-validity endorsements remain net-negative evidence: a lone top-cue endorsement is chosen MORE often when opposed by a large weak coalition (K=5) than when opposed by a single mid-validity endorsement (K=2) — the L4 > L1 reversal, a signature no consensus-premium or tallying account can produce.

(3) *Readability / column binding.* Stated validities must be bound to displayed columns. When the display is monotone in validity (ascending or descending — direction-free) binding is trivial and the signed validity values are used intact. When the mapping is scrambled — especially when the best expert sits interior — binding fails: validity-derived values **flatten toward their mean** and a left-to-right positional gradient is recruited in their place (with a somewhat stronger gradient than previously assumed). The gate μ is environmental, inert whenever columns are validity-sorted.

(4) *Population structure.* One stationary rule for everyone, with broad continuous heterogeneity in determinacy β, validity sensitivity γ, the crossover v*, the contrast normalisation (c0, κ), the readability coefficients and lapse ε. No discrete strategy switching, no learning, no consensus premium, no density term.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # ERC-K2: Expectation-Referenced Cue Coding with contrast-count normalisation.
    #
    #   x_j     = logit(v_j) ** gamma                (raw diagnosticity)
    #   theta   = logit(v*) ** gamma                 (subjective endorsement threshold)
    #   mu      = readability gate (0 = validity-sorted display, 1 = unreadable)
    #   x~_j    = (1-mu)*x_j + mu*mean(x)            (flatten toward equal weighting)
    #   g_j     = ((n-j)/n)^lam  (centred)           (left-to-right positional gradient)
    #   s_j     = x~_j + rho*mu*g_j - theta          (SIGNED cue value)
    #   K       = number of discriminating columns
    #   D       = sum_j s_j (a_j - b_j) / (c0 + kap*K)
    #   p(A)    = (1-eps)*logistic(beta*D) + eps/2
    #
    # No density dilution, no coherence/unanimity premium, no learning.
    import numpy as np

    # ---------------- unpack the two rating vectors --------------------
    a = None
    b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            half = int(arr.shape[0] // 2)
            a = arr[:half].astype(float)
            b = arr[half:2 * half].astype(float)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            a = arr[0].astype(float)
            b = arr[1].astype(float)

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]

    # ---------------- validities ---------------------------------------
    val = parameters.get("validities", None)
    if val is None:
        v = np.linspace(0.90, 0.55, n)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---------------- parameters ---------------------------------------
    gamma = float(parameters.get("gamma", 1.525))
    vstar = float(np.clip(float(parameters.get("vstar", 0.865)), 0.51, 0.99))
    beta = float(parameters.get("beta", 0.42))
    c0 = max(float(parameters.get("c0", 0.485)), 1e-3)
    kap = max(float(parameters.get("kap", 0.45)), 1e-3)
    mu0 = float(np.clip(float(parameters.get("mu0", 0.05)), 0.0, 1.0))
    mu1 = float(parameters.get("mu1", 1.9))
    mu2 = float(parameters.get("mu2", 0.6))
    rho = float(parameters.get("rho", 2.5))
    lam = float(parameters.get("lam", 1.0))
    eps = float(np.clip(float(parameters.get("eps", 0.085)), 0.0, 1.0))

    # ---------------- diagnosticity and expectation reference -----------
    L = np.log(v / (1.0 - v))
    L = np.maximum(L, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.power(L, gamma)
    x = np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=0.0)

    Ls = np.log(vstar / (1.0 - vstar))
    Ls = max(Ls, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        theta = float(np.power(Ls, gamma))
    if not np.isfinite(theta):
        theta = float(np.mean(x))

    # ---------------- readability of the validity->column mapping -------
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    tot = conc + disc
    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)
    incong = 1.0 - abs(tau)                      # 0 when display is monotone
    jmax = int(np.argmax(v))
    interior = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0

    gate = mu1 * incong + mu2 * interior
    gate = min(1.0, max(0.0, gate))
    mu = mu0 + (1.0 - mu0) * gate
    mu = float(np.clip(mu, 0.0, 1.0))

    # ---------------- flattening + positional gradient ------------------
    xbar = float(np.mean(x))
    xt = (1.0 - mu) * x + mu * xbar

    idx = np.arange(n, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        pos = np.power(np.maximum((float(n) - idx) / float(n), 1e-9), lam)
    pos = np.nan_to_num(pos, nan=0.0, posinf=1.0, neginf=0.0)
    g = pos - float(np.mean(pos))                # zero-mean reading gradient

    s = xt + rho * mu * g - theta                # SIGNED cue values

    # ---------------- evidence: signed sum over discriminating columns --
    diff = a - b
    K = float(np.sum(np.abs(diff) > 1e-9))
    if K < 1.0:
        return np.array([0.5, 0.5])

    num = float(np.dot(s, diff))
    denom = c0 + kap * K
    if (not np.isfinite(denom)) or denom <= 1e-6:
        denom = 1e-6
    D = num / denom
    if not np.isfinite(D):
        D = 0.0

    # ---------------- logistic choice with lapse -------------------------
    z = np.array([beta * D, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if (not np.isfinite(tot)) or tot <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))

`parameters`:
- gamma: [1.30, 1.80]
- vstar: [0.81, 0.92]
- beta: [0.25, 0.60]
- c0: [0.35, 0.62]
- kap: [0.32, 0.58]
- mu0: [0.0, 0.10]
- mu1: [1.5, 2.4]
- mu2: [0.35, 0.90]
- rho: [1.8, 3.2]
- lam: [0.7, 1.3]
- eps: [0.02, 0.15]
- validities: validities

`rationale`:
MINIMAL DIFF: the `predict`/`policy` code is re-emitted verbatim except for updated default values; the only substantive change is to five parameter ranges, exactly as the critic prescribed (deepen the crossover, weaken the contrast compression, protect Exps 1-3 by lowering beta and widening lapse), plus a modest widening of rho which the critic's Exp-6 diagnosis implies.

(1) **Deepen the expectation reference**: vstar [0.73,0.85] -> [0.81,0.92], gamma up to 1.80. Hand-computation on a representative 5-cue set (validities ~{.55,.62,.70,.85,.93}, gamma 1.525 => theta ~2.57) makes mid-validity endorsements genuinely aversive: Exp 8's two admissible cells (unanimous vs weak-column dissent at K=1 and K=2) land at ~.25 and ~.20, average ~.23 (observed .205; previous .30). In Exp 7 the metric reduces algebraically to [p(s_strong) - p(s_weak)]/2; pushing theta above the second-best expert's diagnosticity moves p(s_strong) from ~.64 down to ~.45 while p(s_weak) is already floored at ~.25, shrinking the contrast to ~.09-.10 (observed .0675; previous .1725). Both previous errors were same-signed and this single knob fixes both, as the critic diagnosed.

(2) **Weaken the contrast-count compression**: kap [0.45,0.85] -> [0.32,0.58], c0 [0.25,0.60] -> [0.35,0.62]. The critic's Exp-6 diagnostic is L4 > L1. With mu~1 in Exp 6 (best cue is interior => readability gate saturates) the algebra gives L1 numerator = -rho*(g_0-g_2) (pure positional, always negative for the lone-col2 option) and L4 numerator = 3*(theta - xbar) (positive once the crossover is deep). With the new bands L1 ~ .43 and L4 ~ .66-.70, i.e. L1-L4 ~ -0.25, and with the two structurally-zero U/V contrasts the metric lands near -0.08 (previous -0.003, observed -0.15). Raising rho to [1.8,3.2] deepens L1 further and simultaneously pushes Exp 4 (scrambled, positional) up from .605 toward the observed .659, since Exp 4's metric literally scores agreement with a descending-position weighting.

(3) **Protect the existing fits**: raising theta and lowering kap both inflate Exp 1 (its conflict cells all have the TTB winner carrying FEWER weak endorsements, which signed coding rewards). Lowering beta to [0.25,0.60] and widening eps to [0.02,0.15] restores Exp 1 to ~.75 (observed .753) at the same time as keeping the K=1 sensitivity beta/(c0+kap) ~ .43 that Exps 5/7/8 need; Exp 5 computes to ~-0.03 (observed -0.013; previous -0.065).

(4) **Accepted cost, explicitly**: the deeper crossover makes the 5-vs-0 dominance cell of Exp 3 sub-chance, so Exp 3 is expected to drift from -0.19 to about -0.28/-0.31 (observed -0.208), and Exp 2 from .537 to ~.56. I evaluated the obvious in-family remedy (a compressive count reference, theta*sgn(D)|D|^psi with psi<1) and it trades almost exactly 1:1 against Exp 6 (it is the same linear count penalty that makes the weak coalition net-negative), so it buys nothing and adds a parameter; I therefore left the mechanism untouched. Summing squared errors over the eight metrics, the projected total is ~0.019 versus ~0.049 for the accepted base, driven by simultaneous ~0.07-0.08 gains on Exps 6, 7 and 8 against ~0.09 and ~0.03 losses on Exps 3 and 2.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0917 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0931 -> REJECTED
Running-best (last accepted) base: iter 1 at loss=0.0917.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # Take-The-Best winner: first discriminating cue in validity order.
        # Feature columns are already ordered by descending validity in this design.
        ttb_winner = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        # Tallying winner: majority of feature-wise wins (ties excluded).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        if tally_winner == ttb_winner:
            continue  # not a conflict trial
        resp = row['response']
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        hits += 1.0 if int(resp) == ttb_winner else 0.0
        n += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.7533 (var=0.0228)
**Candidate trajectory (this loop):**
  - iter 1: 0.7592 (var=0.0060) (Δ vs real +0.0058)
  - iter 2 (current): 0.7592 (var=0.0076) (Δ vs real +0.0058)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8375 (var=0.0108)
- pi_2: 0.1425 (var=0.0084)
- pi_3: 0.7375 (var=0.0102)
- pi_4: 0.7196 (var=0.0166)
- pi_5: 0.7683 (var=0.0072)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Fixed design validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65
    val = np.array([0.93, 0.58, 0.86, 0.51, 0.74, 0.65], dtype=float)

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape[0] != b.shape[0]:
            continue
        if a.shape[0] == val.shape[0]:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(a.shape[0])

        tally = int(np.sum(a > b)) - int(np.sum(b > a))

        winner = None
        for j in order:
            j = int(j)
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no cue discriminates: uninformative for both models

        ttb_sign = 1 if winner == 0 else -1
        # Keep only trials where the tally does NOT favour the TTB winner:
        # either the tally is exactly tied (Family D) or it points the
        # other way (Family E, directional conflict).
        if tally * ttb_sign > 0:
            continue

        n += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == winner:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5100 (var=0.0136)
**Candidate trajectory (this loop):**
  - iter 1: 0.5367 (var=0.0063) (Δ vs real +0.0267)
  - iter 2 (current): 0.5867 (var=0.0050) (Δ vs real +0.0767)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3058 (var=0.0050)
- pi_1: 0.8204 (var=0.0123)
- pi_3: 0.5371 (var=0.0083)
- pi_4: 0.5204 (var=0.0106)
- pi_5: 0.4938 (var=0.0059)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = [0.55, 0.62, 0.70, 0.85, 0.93]

    def key(v):
        return ''.join([str(int(round(float(x)))) for x in v])

    # conflict families: TTB (top-validity cue) and a left-to-right positional
    # weighting scheme point to OPPOSITE options
    conflict_keys = {
        '10000|00001', '00001|10000',      # family A
        '01000|00010', '00010|01000',      # family B
        '11000|00011', '00011|11000',      # family E
        '00100|11000', '11000|00100',      # family F
    }
    # agreement families: both accounts favour the same option
    agree_keys = {
        '11111|00000', '00000|11111',      # family D (dominance)
        '00001|00000', '00000|00001',      # family G
        '10000|00000', '00000|10000',      # family H
    }

    def ttb_winner(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    n_conf = 0
    hit_conf = 0
    n_agree = 0
    hit_agree = 0

    for _, row in data.iterrows():
        try:
            a = [int(round(float(x))) for x in row['option_a_ratings']]
            b = [int(round(float(x))) for x in row['option_b_ratings']]
            resp = int(row['response'])
        except Exception:
            continue
        n = len(a)
        if len(b) != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fallback: assume columns are displayed in ascending validity
            v = np.linspace(0.55, 0.93, n)
        order = list(np.argsort(-v, kind='stable'))
        w = ttb_winner(a, b, order)
        if w < 0:
            continue
        k = key(a) + '|' + key(b)
        if k in conflict_keys:
            n_conf += 1
            hit_conf += 1 if resp == w else 0
        elif k in agree_keys:
            n_agree += 1
            hit_agree += 1 if resp == w else 0

    if n_conf == 0 or n_agree == 0:
        return 0.0

    return float(hit_agree) / n_agree - float(hit_conf) / n_conf

```

**Observed (real) value:** -0.2081 (var=0.0368)
**Candidate trajectory (this loop):**
  - iter 1: -0.1885 (var=0.0193) (Δ vs real +0.0196)
  - iter 2 (current): -0.2896 (var=0.0185) (Δ vs real -0.0815)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0092 (var=0.0059)
- pi_3: 0.6219 (var=0.0370)
- pi_2: 0.4544 (var=0.0220)
- pi_4: -0.2069 (var=0.0176)
- pi_5: -0.1193 (var=0.0132)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (display order): col0..col5
    v_default = np.array([0.60, 0.93, 0.55, 0.86, 0.68, 0.78], dtype=float)

    def build(n):
        v = v_default.copy()
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
        v = np.clip(v, 0.5 + 1e-6, 1 - 1e-6)
        L = np.log(v / (1.0 - v))
        w = np.power(L, 1.5)
        wpos = np.sort(w)[::-1]
        weff = 0.2 * w + 0.8 * wpos          # lam = 0.8, mid of advocated box
        s = weff.sum()
        weff = weff / s if s > 0 else np.ones(n) / n
        cascade = list(np.argsort(-v, kind="stable"))
        return weff, cascade

    cache = {}
    num = 0.0
    den = 0.0

    for a_raw, b_raw, resp in zip(data["option_a_ratings"],
                                  data["option_b_ratings"],
                                  data["response"]):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        if n not in cache:
            cache[n] = build(n)
        weff, cascade = cache[n]

        d = float(np.dot(weff, a - b))

        winner = None
        for j in cascade:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue
        if abs(d) < 0.20:                    # ambiguous / near-cancelling for WADD-pos
            continue
        pos_choice = 0 if d > 0 else 1
        if pos_choice == winner:             # models agree -> uninformative
            continue

        den += 1.0
        try:
            r = int(resp)
        except Exception:
            continue
        num += 1.0 if r == pos_choice else 0.0

    if den == 0:
        return 0.5
    return float(num / den)
```

**Observed (real) value:** 0.6587 (var=0.0379)
**Candidate trajectory (this loop):**
  - iter 1: 0.6047 (var=0.0062) (Δ vs real -0.0540)
  - iter 2 (current): 0.5513 (var=0.0089) (Δ vs real -0.1073)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8380 (var=0.0096)
- pi_1: 0.1800 (var=0.0087)
- pi_2: 0.6260 (var=0.0084)
- pi_4: 0.6633 (var=0.0149)
- pi_5: 0.6767 (var=0.0145)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def key(x):
        return tuple(int(v) for v in list(x))

    A = [key(x) for x in data['option_a_ratings']]
    B = [key(x) for x in data['option_b_ratings']]
    R = [int(r) for r in data['response']]

    ones5 = (1, 1, 1, 1, 1)
    four = (1, 1, 1, 1, 0)
    zeros = (0, 0, 0, 0, 0)
    weak_single = [(0, 0, 1, 0, 0), (0, 0, 0, 0, 1)]

    hi_n = 0
    hi_d = 0
    lo_n = 0
    lo_d = 0

    for a, b, r in zip(A, B, R):
        if len(a) != 5 or len(b) != 5:
            continue
        # HIGH-coherence cell: 5/5 unanimous panel vs 4/5 panel, differ on cue5 only.
        if a == ones5 and b == four:
            hi_d += 1
            if r == 0:
                hi_n += 1
        elif b == ones5 and a == four:
            hi_d += 1
            if r == 1:
                hi_n += 1
        # LOW-coherence cell: single weak endorsement vs unanimous all-zero panel.
        elif (a in weak_single) and b == zeros:
            lo_d += 1
            if r == 0:
                lo_n += 1
        elif (b in weak_single) and a == zeros:
            lo_d += 1
            if r == 1:
                lo_n += 1

    if hi_d == 0 or lo_d == 0:
        return float('nan')

    p_hi = hi_n / float(hi_d)
    p_lo = lo_n / float(lo_d)
    return float(p_hi - p_lo)

```

**Observed (real) value:** -0.0133 (var=0.0143)
**Candidate trajectory (this loop):**
  - iter 1: -0.0650 (var=0.0251) (Δ vs real -0.0517)
  - iter 2 (current): -0.0442 (var=0.0225) (Δ vs real -0.0308)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0150 (var=0.0203)
- pi_4: 0.7083 (var=0.0508)
- pi_2: -0.0175 (var=0.0181)
- pi_3: -0.0750 (var=0.0313)
- pi_5: -0.0008 (var=0.0364)

### Experiment 6
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    df = data
    if len(df) == 0:
        return 0.0

    def as_tup(x):
        return tuple(int(v) for v in x)

    A = df['option_a_ratings'].apply(as_tup).values
    B = df['option_b_ratings'].apply(as_tup).values
    R = df['response'].values

    def frac_choose(target, other):
        """Fraction of trials with pair {target, other} on which the
        subject picked `target`. Collapses across A/B display order."""
        num = 0
        den = 0
        for a, b, r in zip(A, B, R):
            if a == target and b == other:
                den += 1
                if int(r) == 0:
                    num += 1
            elif a == other and b == target:
                den += 1
                if int(r) == 1:
                    num += 1
        if den == 0:
            return None
        return float(num) / float(den)

    # ---- cells -------------------------------------------------------
    # Family L (coalition ladder): does the lone top-validity (col2)
    # endorsement keep winning as the opposing coalition grows?
    L1 = frac_choose((0, 0, 1, 0, 0), (1, 0, 0, 0, 0))
    L4 = frac_choose((0, 0, 1, 0, 0), (1, 1, 0, 1, 1))

    # Family U (unanimity step, weighted evidence pinned to col3 only):
    # winner-unanimous cell vs loser-unanimous cell.
    U4 = frac_choose((1, 1, 1, 1, 1), (1, 1, 1, 0, 1))
    U1 = frac_choose((0, 0, 0, 1, 0), (0, 0, 0, 0, 0))

    # Family V (same step at a heavier column, col0).
    V2 = frac_choose((1, 1, 1, 1, 1), (0, 1, 1, 1, 1))
    V1 = frac_choose((1, 0, 0, 0, 0), (0, 0, 0, 0, 0))

    contrasts = []
    if L1 is not None and L4 is not None:
        contrasts.append(L1 - L4)
    if U4 is not None and U1 is not None:
        contrasts.append(U4 - U1)
    if V2 is not None and V1 is not None:
        contrasts.append(V2 - V1)

    if len(contrasts) == 0:
        return 0.0
    return float(np.mean(contrasts))

```

**Observed (real) value:** -0.1500 (var=0.0125)
**Candidate trajectory (this loop):**
  - iter 1: -0.0025 (var=0.0175) (Δ vs real +0.1475)
  - iter 2 (current): -0.0342 (var=0.0240) (Δ vs real +0.1158)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4600 (var=0.0383)
- pi_1: -0.0092 (var=0.0076)
- pi_2: 0.1042 (var=0.0121)
- pi_3: 0.0383 (var=0.0077)
- pi_5: -0.1183 (var=0.0157)

### Experiment 7
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def key(x):
        try:
            return ''.join([str(int(round(float(v)))) for v in list(x)])
        except Exception:
            return ''

    P_key = frozenset(['01000', '00000'])          # lone strong endorsement vs empty panel
    N_keys = [frozenset(['11111', '10111']),        # unanimous panel vs one dissent (strong cue)
              frozenset(['11111', '11110'])]        # unanimous panel vs one dissent (weak cue)

    p_hits = []
    n_hits = []

    for _, row in data.iterrows():
        ka = key(row['option_a_ratings'])
        kb = key(row['option_b_ratings'])
        if len(ka) == 0 or len(kb) == 0 or ka == kb:
            continue
        pairset = frozenset([ka, kb])
        sa = ka.count('1')
        sb = kb.count('1')
        if sa == sb:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        # did the subject pick the option carrying the extra endorsement?
        if sa > sb:
            chose_endorsed = 1.0 if resp == 0 else 0.0
        else:
            chose_endorsed = 1.0 if resp == 1 else 0.0

        if pairset == P_key:
            p_hits.append(chose_endorsed)
        else:
            for nk in N_keys:
                if pairset == nk:
                    n_hits.append(chose_endorsed)
                    break

    mp = float(np.mean(p_hits)) if len(p_hits) > 0 else 0.5
    mn = float(np.mean(n_hits)) if len(n_hits) > 0 else 0.5
    return mp - mn

```

**Observed (real) value:** 0.0675 (var=0.0281)
**Candidate trajectory (this loop):**
  - iter 1: 0.1725 (var=0.0438) (Δ vs real +0.1050)
  - iter 2 (current): 0.0413 (var=0.0512) (Δ vs real -0.0262)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3337 (var=0.0409)
- pi_4: -0.5475 (var=0.0849)
- pi_1: -0.0275 (var=0.0274)
- pi_2: 0.0050 (var=0.0120)
- pi_3: 0.0975 (var=0.0386)

### Experiment 8
**Design**
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = 0
    tot = 0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = [int(x) for x in a_raw]
            b = [int(x) for x in b_raw]
        except Exception:
            continue
        n = min(len(a), len(b))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        a_all = all(x == 1 for x in a)
        b_all = all(x == 1 for x in b)
        # exactly one option must be the unanimous all-ones panel
        if a_all == b_all:
            continue
        other = b if a_all else a
        zeros = [i for i, x in enumerate(other) if x == 0]
        if len(zeros) == 0:
            continue
        # the dissenting (zero) columns must all be WEAK experts:
        # with validities displayed in ascending order, the weak experts
        # occupy the leftmost columns.
        cutoff = max(1, n - 3)
        if any(i >= cutoff for i in zeros):
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        chose_unanimous = (r == 0) if a_all else (r == 1)
        tot += 1
        if chose_unanimous:
            hits += 1
    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.2050 (var=0.0274)
**Candidate trajectory (this loop):**
  - iter 1: 0.3000 (var=0.0181) (Δ vs real +0.0950)
  - iter 2 (current): 0.2562 (var=0.0160) (Δ vs real +0.0512)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8800 (var=0.0125)
- pi_5: 0.4938 (var=0.0166)
- pi_1: 0.8187 (var=0.0246)
- pi_2: 0.8438 (var=0.0202)
- pi_3: 0.8450 (var=0.0113)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: ERC-K is clearly the best single account so far: it is the only theory that gets Exp 3 (-0.19 vs -0.21), Exp 8 (0.30 vs 0.205) and Exp 1 (0.759 vs 0.753) into the right region simultaneously, and it does so with the prescribed mechanism (signed expectation-referenced cue values + contrast-count normalisation + an environment-gated readability module). Cell-by-cell errors: Exp1 +0.006, Exp2 +0.027, Exp3 +0.020, Exp4 -0.054, Exp5 -0.052, Exp7 +0.105, Exp8 +0.095, Exp6 +0.148. So six of eight experiments are essentially fit; the residual loss (0.0917) is dominated by two related failures, both in the *same direction*: the model does not make weak/low-validity endorsements aversive enough, and it over-compresses (or fails to differentiate) many-column contrasts.

Exp 8 (0.30 vs 0.205) and Exp 7 (0.173 vs 0.068) are the same miscalibration: with vstar in [0.73,0.85] and gamma ~1.5, the signed value s_j for mid-validity experts is only mildly negative, so unanimous-vs-weak-dissent panels sit only slightly below chance and the lone-strong-endorsement cell P stays too far above the N cells. Both errors point the same way (candidate too high), so a single knob \u2014 pushing the crossover vstar up (and/or gamma up) so more of the displayed panel falls below the expectation reference \u2014 moves both toward the data.

Exp 6 (-0.003 vs -0.150) is claimed to be structurally unreachable, but that claim is only true for the U and V contrasts (identical difference vectors, different backgrounds, which are indeed 0 for any background-blind model). The L contrast is NOT structurally zero: L1 (lone col2 vs lone col0, K=2) versus L4 (lone col2 vs a four-column weak coalition, K=5). Under ERC with a sufficiently high crossover, the weak coalition is *net negative evidence*, so L4 should be HIGHER than L1 \u2014 giving a negative L1-L4 of roughly -0.45 and hence a metric near -0.15 with U=V=0. The current parameterisation kills this because kappa (0.45-0.85) compresses the K=5 cell hard toward chance, cancelling exactly the effect ERC predicts. So Exp 6 is reachable inside the prescribed family; it is a normalisation-strength problem, not a missing-mechanism problem.

Exp 4 (-0.054) and Exp 5 (-0.052) are close enough that they should be treated as constraints not targets.
Rationale: Stay in the arbiter's ERC family \u2014 the candidate is faithful to it and should be kept as the base; only re-tune two knobs, do NOT add a shared-background dilution term (that is out of family and is what sank pi_5 on Exps 7/8).

(1) Deepen the expectation reference. Raise the crossover range to roughly vstar in [0.78, 0.90] (and allow gamma up to ~1.9). This makes mid/low-validity endorsements genuinely aversive, which pushes Exp 8 down from 0.30 toward the observed 0.205 and lowers the lone-endorsement cell P in Exp 7, shrinking P-N from 0.17 toward the observed 0.07. Both current errors are same-signed and this is the single knob that fixes both.

(2) Weaken the contrast-count compression. Lower kappa to roughly [0.25, 0.60] (and c0 to ~[0.30, 0.70]) so that the K=5 cell of Exp 6 (lone strong cue vs a four-column weak coalition) is not flattened to chance. With a higher vstar the coalition becomes net-negative evidence, so L4 > L1 and L1-L4 goes strongly negative; averaged with the two structurally-zero U/V contrasts this lands the Exp 6 metric near the observed -0.15 without any background term. Verify explicitly that L4 exceeds L1 in the simulation \u2014 that is the diagnostic.

(3) Protect the fits you already have. Reducing kappa and raising vstar will both push Exp 1 up (currently 0.759 vs 0.753) and Exp 2 up (0.537 vs 0.510). Compensate by lowering the beta band (e.g. [0.40, 1.00]) and/or widening lapse eps to ~[0.02, 0.16], and re-check Exp 3 (target -0.21) and Exp 5 (target -0.013) after retuning \u2014 Exp 5 currently sits 0.05 too negative, and a slightly higher lapse in the dense-cell direction should help there too.

No change to the readability/column-binding module (Exps 2/4 are within 0.06 and it is doing its job), no learning, no discrete strategy switching, and no unanimity/coherence premium. If the retune cannot get Exp 6 below about -0.08 without costing more than 0.05 on Exps 7/8 combined, keep the vstar increase (which is the safer, same-signed fix for Exps 7 and 8) and back off the kappa change.

**Outcome of your advice:** iter 1 candidate loss=0.0917 -> CURRENT candidate loss=0.0931 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "Iter-2 was a controlled test of my own iter-1 advice, and the results split cleanly. The parts I predicted worked: Exp 7 error fell from +0.105 to -0.026 and Exp 8 from +0.095 to +0.051 (both driven by the deeper crossover vstar 0.81-0.92), and Exp 5 improved slightly (-0.065 -> -0.044). But the collateral damage I flagged as a risk materialised and was bigger than the gain: Exp 3 went from -0.189 (err +0.020) to -0.290 (err -0.082), Exp 2 from 0.537 (err +0.027) to 0.587 (err +0.077), Exp 4 from 0.605 (err -0.054) to 0.551 (err -0.107). Net loss 0.0917 -> 0.0931, REJECTED. Crucially, the *target* of the kappa change \"\"\" Exp 6 \"\"\" barely moved (-0.003 -> -0.034 vs observed -0.150), so the compression relaxation bought almost nothing while the crossover deepening cost three experiments. Diagnosis of why Exp 6 did not move: with mu saturated (best cue interior), the L4 cell numerator is ~3*(theta - xbar) but the *linear* count normaliser c0 + kappa*K puts K=5 through a denominator ~2.7 while the L1 cell (K=2) sits at ~1.4; the linear-in-K denominator cancels almost exactly the coalition effect that ERC needs to produce L4 > L1. Reducing kappa uniformly cannot fix this because it also inflates the K=1/K=2 sensitivity that Exps 5/7/8 depend on \"\"\" the two demands are in direct conflict under a linear normaliser. Separately, pushing vstar above the top expert's diagnosticity makes the 5-vs-0 dominance cell sub-chance, which is what dragged Exp 3 (agreement families) and Exp 4 down. The candidate remains faithful to the arbiter's ERC family (signed expectation-referenced cue values, contrast-count normalisation, environment-gated readability, no density dilution, no consensus premium, no learning), so this is a calibration problem, not a family problem.",
  "verdict": "regenerate",
  "rationale": "The candidate was rejected by the gate, so the next iteration must build on the iter-1 accepted base (vstar [0.73,0.85], kappa [0.45,0.85], beta/eps as in iter 1) with a DIFFERENT in-family knob than the one I pushed last time. Three concrete recommendations, all inside the prescribed ERC family:\n\n(1) STOP oscillating on vstar; take the midpoint. Iter 1 used [0.73,0.85] (Exps 7/8 too high), iter 2 used [0.81,0.92] (Exps 2/3/4 broke). Set vstar to roughly [0.76, 0.87] and gamma back to ~[1.30,1.65]. That should retain most of the Exp 7/8 gain (target ~0.08 and ~0.23) while keeping Exp 3 near -0.21 and Exp 2 near 0.52. Additionally, guard the dominance cell explicitly: theta must stay strictly below the top expert's diagnosticity x_max so that 5-vs-0 (and Exp 4's 111111-vs-000000) remain clearly above chance. The cleanest in-family way to enforce this is to calibrate theta as a panel-referenced quantity, e.g. theta = xbar + phi*(x_max - xbar) with phi in ~[0.55,0.85], instead of an absolute logit(v*)^gamma that can exceed every cue in the panel. This is a re-calibration of the same expectation-reference mechanism, not a new mechanism.\n\n(2) Replace the LINEAR contrast-count normaliser with a SUBLINEAR one: denom = c0 + kappa*K^psi with psi in ~[0.45,0.70] (or simply kappa*sqrt(K)). This is the key structural fix for Exp 6 and it is explicitly a normalisation-scheme swap, not a family change. It decouples the two conflicting demands: K=1/K=2 sensitivity (Exps 5, 7, 8) is left essentially untouched, while the K=5 coalition cell (L4 = lone col2 vs the four-column weak coalition) is no longer flattened to chance. Hand-check with psi=0.5, kappa~0.45, c0~0.49: L4 denom drops from ~2.7 to ~1.5 and L1 denom from ~1.39 to ~1.12, so L1-L4 goes from roughly -0.10 to roughly -0.30, putting the Exp 6 metric near -0.10/-0.12 (U and V contrasts are structurally 0 for any background-blind model, so -0.10 is close to the in-family ceiling and is worth ~0.015 of loss). Report L1 and L4 separately in the rationale as the diagnostic \"\"\" the signature is L4 > L1, not just a negative metric.\n\n(3) Do NOT repeat the iter-2 rho increase in isolation. Raising rho to 3.2 moved Exp 4 the wrong way (0.605 -> 0.551); keep rho near the iter-1 band (~[1.8,2.6]) and let the sublinear normaliser supply the extra L1 depth instead. After the change, re-check the constraint set: Exp 1 ~0.753 (a less compressive normaliser will push multi-column trials up, so be prepared to shave beta slightly or keep eps ~[0.03,0.13]), Exp 2 ~0.51, Exp 3 ~-0.21, Exp 5 ~-0.01. If the sublinear normaliser costs more than ~0.03 combined on Exps 1-3, back psi up toward 0.7 rather than abandoning it."
}
```

## Usage

```json
{
  "input_tokens": 25161,
  "output_tokens": 5776
}
```
