from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, ClassVar

from typing import Callable

import numpy as np
import pandas as pd
import yaml
from pydantic import Field, PrivateAttr
from tqdm.auto import tqdm

from src.experiment import Experiment
from src.logger import info
from src.online_config import OnlineConfig

def _bayesian_normative_choice(
    a_ratings: list[int],
    b_ratings: list[int],
    validities: list[float],
) -> str:
    """Lee & Cummins (2004) Bayesian normative choice given cue ratings.

    Returns ``"a"`` / ``"b"`` for the option with higher posterior log-odds,
    or ``""`` for ties (so jsPsych marks ``bean_correct`` as null and the
    trial drops out of the end-of-task accuracy summary). Each expert's
    contribution is ``(a_i - b_i) * log(v_i / (1 - v_i))``, which reduces
    to the standard binary log-odds rule when ratings are 0/1 and extends
    naturally to cardinal ratings.
    """
    log_odds = 0.0
    for ai, bi, v in zip(a_ratings, b_ratings, validities):
        v_safe = min(max(float(v), 0.5 + 1e-6), 1.0 - 1e-6)
        log_odds += (int(ai) - int(bi)) * math.log(v_safe / (1.0 - v_safe))
    if log_odds > 0:
        return "a"
    if log_odds < 0:
        return "b"
    return ""


class _RawJsExpr:
    """Wrap a JavaScript expression so SweetBean's ``_var_to_js`` emits it
    verbatim. Used to render the end-of-task feedback HTML from the
    browser-side ``jsPsych.data`` collection — SweetBean has no first-class
    way to inject a raw JS expression into a stimulus parameter, but it
    duck-types on ``to_js()``."""

    def __init__(self, js: str) -> None:
        self._js = js

    def to_js(self) -> str:
        return self._js


class DecisionMakingBinaryExperiment(Experiment):
    """Two-option, k-integer-feature decision-making.

    Subjects choose between two fictitious products A and B, each described
    by k binary expert ratings (each 0 or 1). The feature count
    (k = n_features) and per-feature validities are LLM-proposed
    per experiment and fixed across all
    trials in that experiment. This is the classic
    binary-feature protocol (Hilbig & Moshagen 2014). Validities are
    communicated to subjects up front; there is no trial-by-trial
    correctness feedback.
    """

    name: ClassVar[str] = "Decision Making (Binary Features)"
    description: ClassVar[str] = (
        "Subjects repeatedly choose between two fictitious products, A and "
        "B. Each option is described by a vector of binary expert ratings "
        "(each 0 or 1). Every experiment fixes its own feature count "
        "(via `validities` length) and per-expert validities; both are "
        "LLM-proposed. The validities are "
        "communicated to the subject in the instructions. Subjects pick "
        "whichever product they believe is of higher quality. There is no "
        "trial-by-trial correctness feedback."
    )
    parameter_variables: ClassVar[dict[str, str]] = {
        "n_features": "Number of expert ratings per option (LLM-proposed via `validities` length).",
        "validities": "Per-expert validities (LLM-proposed; each in [0.5, 1.0]); fixed across all trials.",
    }

    # --- fixed protocol parameters (single source of truth) ----------------------

    N_OPTIONS: ClassVar[int] = 2
    MAX_TRIALS: ClassVar[int] = 96  # Source: Hilbig & Moshagen (2014)

    # --- output columns (descriptions use n_features generically since it's instance-set) ----

    output_columns: ClassVar[dict[str, str]] = {
        "subject_id": "Subject identifier (one row per trial per subject).",
        "option_a_ratings": "List of n_features binary expert ratings (each 0 or 1) for option A on this trial.",
        "option_b_ratings": "List of n_features binary expert ratings (each 0 or 1) for option B on this trial.",
        "response": "0 if subject chose A, 1 if subject chose B.",
    }

    # --- abstract class-level framing (subject-facing; instance renders concrete values) ---

    # This text is wired into the experiment-PROPOSER prompt under the
    # heading "Subjects see the following instructions:" (see
    # `src/prompts/experiment_proposal.py::TEMPLATE`). The proposer reads
    # it to understand exactly what subjects see — which determines what
    # kinds of trial pairs actually pay off — so it MUST describe the
    # UX concretely, including:
    #   * the rating type (binary: each rating is 0 or 1),
    #   * how ratings are displayed (numeric bar widget, not pips),
    #   * the response keys (A / B), and
    #   * the per-trial min-RT gate (the keys are locked briefly each
    #     trial — proposers should know that subjects are *forced* to
    #     view the stimulus, so trivial "always pick A" trial sets won't
    #     produce data).
    # Per-experiment values (validities, trial count) stay as
    # PLACEHOLDERS here because the proposer is the one choosing them;
    # the same placeholders are filled in for real participants by
    # `_online_instructions_inner_html()` at build time (wrapped by
    # SweetBean ``RichInstructions``). Keep this in sync with that method
    # whenever the participant-facing
    # copy changes.
    introduction_text: ClassVar[str] = (
        "In this experiment you will repeatedly choose between two "
        "fictitious products, A and B. On every trial you will see "
        "`n_features` expert ratings for each product (the number of "
        "experts is fixed across all trials and is set by the length of "
        "`validities`).\n\n"
        "Each rating is an integer in [0, 1]. The ratings are displayed as a "
        "horizontal filled bar with the numeric value (e.g. \"0/1\") "
        "shown next to it. Higher = more positive.\n\n"
        "The same experts (in the same order) provide ratings for both "
        "products on every trial. Each expert's accuracy (their validity "
        "expressed as a percentage, e.g. \"Expert 1 (80%)\") is shown "
        "next to their rating on every trial AND is also listed up front "
        "in an \"Expert accuracies\" panel.\n\n"
        "On each trial, decide which product is of higher quality and "
        "press A for product A or B for product B. There is no time "
        "limit and no feedback. Note that for the first ~`min_rt_ms` of "
        "each trial the answer prompt is hidden and the keys are locked, "
        "so subjects first see the full ratings and can answer once the "
        "A / B prompt appears — design pairs that actually require "
        "comparing the ratings.\n\n"
        "Total trials per subject is roughly `MAX_TRIALS`: each unique "
        "pair you propose is repeated `K = max(1, MAX_TRIALS // "
        "n_unique_pairs)` times in an independently-randomized order "
        "per subject."
    )

    # --- LLM-set instance fields (part of the response schema) -------------------

    validities: list[Annotated[float, Field(ge=0.5, le=1.0)]] = Field(
        ...,
        description=(
            "Per-expert validities, one per feature. Each must be in "
            "[0.5, 1.0]; order is free (no descending requirement — "
            "heuristics that rely on validity-ordering re-sort internally). "
            "Length determines n_features for the whole experiment and must "
            "match the rating-list length in every trial pair. Pick a "
            "spread (at least one high validity and at least one low one) "
            "— uniform validities (e.g. all 0.7) collapse WADD into a "
            "scaled Equal-Weight rule and make them indistinguishable by "
            "any decision."
        ),
    )
    trial_a_ratings: list[list[int]] = Field(
        ...,
        description=(
            "List of option-A rating vectors, one per trial. Each inner "
            "list has length equal to len(validities); each value is "
            "binary (0 or 1). `trial_a_ratings[i]` pairs with "
            "`trial_b_ratings[i]` on trial i."
        ),
    )
    trial_b_ratings: list[list[int]] = Field(
        ...,
        description=(
            "List of option-B rating vectors, one per trial. Same length "
            "and shape constraints as `trial_a_ratings`. Pick trials that "
            "let the intended heuristics (TTB, EQW, Tallying, WADD) be "
            "dissociated; avoid trials where every heuristic agrees."
        ),
    )

    # --- runtime state -----------------------------------------------------------

    _trials: list[tuple[tuple[int, ...], tuple[int, ...]]] = PrivateAttr(default_factory=list)
    _idx: int = PrivateAttr(default=0)
    _n_repeats: int = PrivateAttr(default=1)

    # --- construction ------------------------------------------------------------

    def model_post_init(self, _context: Any) -> None:
        # Validate (validities, trial_a_ratings/trial_b_ratings) against each other, then
        # replicate the LLM-supplied unique pairs to fill MAX_TRIALS and
        # randomize independently per subject. Floor the repeat count at 1
        # so designs with more than MAX_TRIALS unique pairs slightly exceed
        # the cap rather than dropping pairs.
        if not self.validities:
            raise ValueError("validities must contain at least one entry.")
        for v in self.validities:
            if not (0.5 <= v <= 1.0):
                raise ValueError(
                    f"each validity must be in [0.5, 1.0]; got {v!r}."
                )
        n_features = len(self.validities)

        if len(self.trial_a_ratings) != len(self.trial_b_ratings):
            raise ValueError(
                f"trial_a_ratings and trial_b_ratings must have the same "
                f"length; got {len(self.trial_a_ratings)} vs "
                f"{len(self.trial_b_ratings)}."
            )
        levels = [
            (tuple(a), tuple(b))
            for a, b in zip(self.trial_a_ratings, self.trial_b_ratings)
        ]
        if not levels:
            raise ValueError("trial_a_ratings/trial_b_ratings must contain at least one trial.")
        for a, b in levels:
            if len(a) != n_features or len(b) != n_features:
                raise ValueError(
                    f"each rating list must have length n_features="
                    f"{n_features} (derived from validities); got "
                    f"len(a)={len(a)}, len(b)={len(b)}."
                )
            for v in (*a, *b):
                if not (0 <= v <= 1):
                    raise ValueError(
                        f"each rating value must be binary "
                        f"(0 or 1); got {v!r}."
                    )

        self._n_repeats = max(1, self.MAX_TRIALS // len(levels))
        rng = np.random.default_rng()
        sequence = list(levels) * self._n_repeats
        rng.shuffle(sequence)
        self._trials = sequence

        info(
            f"DecisionMakingBinaryExperiment: prepared "
            f"{len(levels)} unique pairs × {self._n_repeats} reps = "
            f"{len(self._trials)} trials "
            f"(n_features={n_features}, MAX_TRIALS={self.MAX_TRIALS})"
        )
        self._reset_history()

    def _reset_history(self) -> None:
        self._history = {
            "option_a_ratings": [],
            "option_b_ratings": [],
            "response": [],
        }

    # --- prompt-facing summaries -------------------------------------------------

    @classmethod
    def pretty_print_header(cls) -> str:
        """Design summary aimed at the experiment PROPOSER (chooses validities + trial_a_ratings/trial_b_ratings)."""
        return (
            f"A multi-attribute decision-making experiment. On each trial "
            f"the subject sees two options (A, B), each described by "
            f"`n_features` integer expert ratings (`n_features` is set by "
            f"the length of `validities` you propose). Choose `validities` "
            f"— one per feature, each in [0.5, 1.0], order free — to fix "
            f"each expert's advertised accuracy; subjects are told these "
            f"values up front. Then choose "
            f"`trial_a_ratings/trial_b_ratings` (each rating value in "
            f"[0, 1]) so that the intended heuristics (e.g. TTB, "
            f"EQW, Tallying, WADD) make distinguishable predictions: avoid "
            f"degenerate pairs where every heuristic agrees, and prefer "
            f"pairs that dissociate single-feature focus from "
            f"feature-summing strategies. Validities and the "
            f"trial ratings together define the design; they are fixed "
            f"across all trials in this experiment. No trial-by-trial "
            f"correctness feedback. The total number of trials per subject "
            f"is held at roughly {cls.MAX_TRIALS}: each unique pair is "
            f"repeated K = max(1, {cls.MAX_TRIALS} // n_unique_pairs) "
            f"times in an independently-randomized order per subject."
        )

    @classmethod
    def pretty_print_protocol(cls) -> str:
        """Protocol summary aimed at the THEORIST / MODELER."""
        return (
            f"Each subject completes ~{cls.MAX_TRIALS} trials in a single "
            f"block, with order randomized independently per subject. On "
            f"every trial the subject sees two options A and B, each "
            f"described by `n_features` binary expert ratings "
            f"(each 0 or 1). The per-feature validities and n_features "
            f"are fixed per experiment (design-time "
            f"choices). Validities are communicated to the subject in the "
            f"instructions. Both `n_features` and `validities` "
            f"are exposed to your `predict` via the "
            f"`parameters` dict. The subject chooses A or B; no "
            f"correctness feedback is provided after the choice."
        )

    def pretty_print(self) -> str:
        """Compact view of the design: rationale + validities + trial table + schedule."""
        rationale = self.rationale or "(no rationale)"
        n_unique = len(self.trial_a_ratings)
        total_trials = n_unique * self._n_repeats
        table = "\n".join(
            f"  trial {i + 1}: A={list(a)}  B={list(b)}"
            for i, (a, b) in enumerate(
                zip(self.trial_a_ratings, self.trial_b_ratings)
            )
        )
        return (
            f"**Validities (n_features={len(self.validities)}):** "
            f"{list(self.validities)}\n\n"
            f"**Trial pairs (n={n_unique}):**\n{table}\n\n"
            f"**Rationale:** {rationale}\n\n"
            f"**Computed schedule:** {n_unique} unique pairs × "
            f"{self._n_repeats} reps = {total_trials} trials per subject.\n\n"
        )

    def pretty_print_design(self) -> str:
        """Trial-pairs table only (no rationale, no schedule chrome)."""
        return "\n".join(
            f"  A={list(a)}  B={list(b)}"
            for a, b in zip(self.trial_a_ratings, self.trial_b_ratings)
        )

    # --- iteration interface (used by Experiment.simulate) -----------------------

    def __iter__(self) -> "DecisionMakingBinaryExperiment":
        self._idx = 0
        self._reset_history()
        return self

    def __next__(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], dict[str, list[Any]]]:
        if self._idx >= len(self._trials):
            raise StopIteration
        a, b = self._trials[self._idx]
        self._idx += 1
        return (a, b), self._history

    def update(self, response: int) -> None:
        a, b = self._trials[self._idx - 1]
        self._history["option_a_ratings"].append(list(a))
        self._history["option_b_ratings"].append(list(b))
        self._history["response"].append(int(response))

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def parameter_context(self) -> dict[str, Any]:
        return {
            "n_features": len(self.validities),
            "validities": list(self.validities),
        }

    # --- centaur: prompted simulation -------------------------------------------

    def _centaur_prompt_intro(
        self,
        label_a: str,
        label_b: str,
        validities_pct_sorted: list[int],
    ) -> str:
        """Plain-text preamble shown to Centaur once per subject.

        `validities_pct_sorted` is the sorted-descending list of integer
        validity percentages — same permutation that `_centaur_prompt_trial_open`
        applies to per-trial rating vectors, so position `i` of the prompt's
        listed validity matches position `i` of every trial line's rating vector.
        """
        n_features = len(self.validities)
        
        rating_phrase = (
                "with 1 representing a positive and 0 representing a negative rating"
            )
        
        validities_str = ", ".join(f"{v}%" for v in validities_pct_sorted)
        return (
            f"You are repeatedly presented with two options, labeled {label_a} and {label_b}.\n"
            "Each option represents a fictitious product and you have to infer "
            "which product is superior in terms of quality.\n"
            "You select a product by pressing the corresponding key.\n"
            f"For each decision, you are provided with {n_features} expert ratings "
            f"({rating_phrase}).\n"
            f"The {n_features} experts differ in their validity (defined as the "
            "probability that a product is superior given that it has received a "
            "positive rating by a particular expert).\n"
            "The ratings of experts are given in descending order of their "
            f"validity (taking values of {validities_str}).\n\n"
        )

    @staticmethod
    def _format_ratings_numpy(ratings: tuple[int, ...]) -> str:
        """Render `(0, 1, 1, 0)` as `'[0 1 1 0]'` — numpy-style, no commas."""
        return "[" + " ".join(str(int(r)) for r in ratings) + "]"

    def _centaur_prompt_trial_open(
        self,
        a_sorted: tuple[int, ...],
        b_sorted: tuple[int, ...],
        label_a: str,
        label_b: str,
    ) -> str:
        """One trial line up to (but not including) the next-token slot.

        Returns the line ending in `'You chose: <'`. The caller appends the
        generator's chosen letter and the closing `'>.\\n'` after the model
        emits a token.
        """
        return (
            f"Product {label_a} ratings: {self._format_ratings_numpy(a_sorted)}. "
            f"Product {label_b} ratings: {self._format_ratings_numpy(b_sorted)}. "
            "You chose: <"
        )

    def simulate_centaur(
        self,
        n_runs: int,
        *,
        generator: Callable[[str], str],
        rng: "np.random.Generator | None" = None,
    ) -> pd.DataFrame:
        """Simulate `n_runs` Centaur "subjects" on this design.

        `generator(prompt) -> str` returns the next token text after `prompt`
        (whitespace allowed; we strip). For production runs `generator` is
        `UnslothAgent(...)` loaded once at the top of the run; for tests a
        small stub returning one of the two per-subject labels is sufficient.

        The prompt lists validities in **descending** order — canonical
        multi-attribute decision-making framing — and applies the same
        permutation to every trial's rating vectors. The output frame stores
        ratings in **proposed** order (unchanged from `Experiment.simulate` /
        `run_online`) so downstream theories' `predict` keep aligning
        rating-position `i` to `validities[i]`.

        Adds one centaur-only diagnostic column `centaur_random_fallback`
        (bool) recording per-trial whether the response came from the random
        fallback (Centaur emitted a token outside `{label_a, label_b}`).
        """
        rng = rng if rng is not None else np.random.default_rng()
        # Permutation that sorts validities descending; ties stable.
        # `argsort` is ascending so we negate to get descending.
        perm = np.argsort([-v for v in self.validities], kind="stable").tolist()
        validities_pct_sorted = [int(round(100 * self.validities[i])) for i in perm]

        info(
            f"DecisionMakingBinaryExperiment: simulate_centaur — "
            f"n_runs={n_runs}, n_trials={len(self._trials)}, "
            f"n_features={len(self.validities)}"
        )

        frames: list[pd.DataFrame] = []
        letters_pool = list(map(chr, range(65, 91)))   # A..Z
        n_trials = len(self._trials)
        subject_bar = tqdm(
            range(n_runs),
            desc="simulate_centaur subjects",
            unit="subj",
        )
        for run in subject_bar:
            # Per-subject randomization: pick two distinct letters and assign
            # which one names option A vs B. Mirrors centaur/utils.py's
            # `randomized_choice_options`, but driven off `rng` so the test
            # suite gets reproducibility.
            picks = rng.choice(letters_pool, size=2, replace=False)
            L0, L1 = str(picks[0]), str(picks[1])
            if rng.random() < 0.5:
                label_a, label_b = L0, L1
            else:
                label_a, label_b = L1, L0
            valid_letters = (label_a, label_b)

            prompt = self._centaur_prompt_intro(
                label_a, label_b, validities_pct_sorted
            )
            fallback_flags: list[bool] = []
            trial_bar = tqdm(
                self,
                total=n_trials,
                desc=f"  subject {run + 1}/{n_runs} trials",
                unit="trial",
                leave=False,
            )
            for (a, b), _hist in trial_bar:
                a_sorted = tuple(a[i] for i in perm)
                b_sorted = tuple(b[i] for i in perm)
                prompt += self._centaur_prompt_trial_open(
                    a_sorted, b_sorted, label_a, label_b
                )
                tok = str(generator(prompt)).strip()
                if tok == label_a:
                    response, fell_back = 0, False
                elif tok == label_b:
                    response, fell_back = 1, False
                else:
                    fallback = str(rng.choice(valid_letters))
                    info(
                        f"simulate_centaur: invalid token {tok!r}; "
                        f"falling back to random {fallback!r}."
                    )
                    response = 0 if fallback == label_a else 1
                    fell_back = True
                fallback_flags.append(fell_back)
                chosen_letter = label_a if response == 0 else label_b
                prompt += f"{chosen_letter}>.\n"
                self.update(int(response))

            df = self.to_df()
            df["subject_id"] = run
            df["centaur_random_fallback"] = fallback_flags
            frames.append(df)
            if run == 0:
                info(f"simulate_centaur: subject 0 prompt:\n{prompt}")
            n_fb = sum(fallback_flags)
            if n_fb:
                info(
                    f"simulate_centaur: subject {run} fell back to random on "
                    f"{n_fb}/{len(fallback_flags)} trials."
                )

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        canonical = self.to_canonical_data(df)
        # `to_canonical_data` reindexes to `output_columns` and would strip the
        # diagnostic; re-attach so callers can audit fallbacks per trial.
        canonical["centaur_random_fallback"] = df["centaur_random_fallback"].values
        return canonical

    def simulate_centaur_batched(
        self,
        n_runs: int,
        *,
        batch_generator: Callable[[list[str]], list[str]],
        rng: "np.random.Generator | None" = None,
    ) -> pd.DataFrame:
        """Subjects-batched variant of :meth:`simulate_centaur`.

        Same return shape and semantics as the serial version, but at every
        trial step we hand all `n_runs` subject prompts to `batch_generator`
        in one call so the underlying GPU runs them in parallel. Trials are
        shared across subjects (matching the serial version's design), so
        subjects stay in lockstep and a single batched forward pass per trial
        is sufficient.

        `batch_generator(prompts) -> list[str]` must return one next-token
        string per input prompt, in the same order. For production use this
        wraps `transformers.pipeline(prompts, batch_size=...)`; tests use a
        per-prompt stub.
        """
        rng = rng if rng is not None else np.random.default_rng()
        perm = np.argsort([-v for v in self.validities], kind="stable").tolist()
        validities_pct_sorted = [int(round(100 * self.validities[i])) for i in perm]

        info(
            f"DecisionMakingBinaryExperiment: simulate_centaur_batched — "
            f"n_runs={n_runs}, n_trials={len(self._trials)}, "
            f"n_features={len(self.validities)}"
        )

        letters_pool = list(map(chr, range(65, 91)))   # A..Z
        n_trials = len(self._trials)

        prompts: list[str] = []
        label_a_list: list[str] = []
        label_b_list: list[str] = []
        fallback_flags_list: list[list[bool]] = [[] for _ in range(n_runs)]
        histories: list[dict[str, list[Any]]] = [
            {"option_a_ratings": [], "option_b_ratings": [], "response": []}
            for _ in range(n_runs)
        ]

        for _run in range(n_runs):
            picks = rng.choice(letters_pool, size=2, replace=False)
            L0, L1 = str(picks[0]), str(picks[1])
            if rng.random() < 0.5:
                la, lb = L0, L1
            else:
                la, lb = L1, L0
            label_a_list.append(la)
            label_b_list.append(lb)
            prompts.append(
                self._centaur_prompt_intro(la, lb, validities_pct_sorted)
            )

        trial_bar = tqdm(
            range(n_trials),
            desc="simulate_centaur_batched trials",
            unit="trial",
        )
        for t in trial_bar:
            a, b = self._trials[t]
            a_sorted = tuple(a[i] for i in perm)
            b_sorted = tuple(b[i] for i in perm)
            for run in range(n_runs):
                prompts[run] += self._centaur_prompt_trial_open(
                    a_sorted, b_sorted, label_a_list[run], label_b_list[run]
                )
            toks = batch_generator(prompts)
            if len(toks) != n_runs:
                raise ValueError(
                    f"batch_generator returned {len(toks)} tokens for "
                    f"{n_runs} prompts."
                )
            for run in range(n_runs):
                tok = str(toks[run]).strip()
                la, lb = label_a_list[run], label_b_list[run]
                if tok == la:
                    response, fell_back = 0, False
                elif tok == lb:
                    response, fell_back = 1, False
                else:
                    fallback = str(rng.choice([la, lb]))
                    info(
                        f"simulate_centaur_batched: subject {run} invalid token "
                        f"{tok!r}; falling back to random {fallback!r}."
                    )
                    response = 0 if fallback == la else 1
                    fell_back = True
                fallback_flags_list[run].append(fell_back)
                chosen_letter = la if response == 0 else lb
                prompts[run] += f"{chosen_letter}>.\n"
                histories[run]["option_a_ratings"].append(list(a))
                histories[run]["option_b_ratings"].append(list(b))
                histories[run]["response"].append(int(response))

        frames: list[pd.DataFrame] = []
        for run in range(n_runs):
            df = pd.DataFrame(histories[run])
            df["subject_id"] = run
            df["centaur_random_fallback"] = fallback_flags_list[run]
            frames.append(df)
            n_fb = sum(fallback_flags_list[run])
            if n_fb:
                info(
                    f"simulate_centaur_batched: subject {run} fell back to random "
                    f"on {n_fb}/{len(fallback_flags_list[run])} trials."
                )
        if n_runs:
            info(f"simulate_centaur_batched: subject 0 prompt:\n{prompts[0]}")

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        canonical = self.to_canonical_data(df)
        canonical["centaur_random_fallback"] = df["centaur_random_fallback"].values
        return canonical

    # --- online: instructions screen --------------------------------------------

    def _online_instructions_inner_html(self, *, min_rt_ms: int = 0) -> str:
        """Body HTML for :class:`sweetbean.stimulus.RichInstructions` (no ``<style>``).

        Typography and layout live in SweetBean's reusable
        ``RichInstructions`` stimulus; this method only supplies
        experiment-specific copy and the worked example markup.
        """
        n_features = len(self.validities)
        validities_pct = [int(round(v * 100)) for v in self.validities]
        try:
            n_trials = len(self._trials)
        except AttributeError:
            n_trials = self.MAX_TRIALS

        if int(1) == 1:
            rating_scale_phrase = (
                "Each rating is either <strong>0</strong> or "
                "<strong>1</strong> (1 means the expert thinks this product "
                "is high quality)."
            )
        else:
            rating_scale_phrase = (
                "Each rating is an integer between <strong>0</strong> and "
                f"<strong>{int(1)}</strong> &mdash; higher "
                "means the expert is more positive about the product."
            )

        accuracy_items = "".join(
            f"<li><strong>Expert {i + 1}</strong>: "
            f"<span class='sbri-accent'>{pct}%</span></li>"
            for i, pct in enumerate(validities_pct)
        )

        if int(1) == 1:
            example_a = [1] * n_features
            example_b = [0] * n_features
        else:
            high = int(1)
            low = 0
            mid_high = max(1, high - 1)
            mid_low = min(high - 1, 1)
            example_a = [high if i % 2 == 0 else mid_high for i in range(n_features)]
            example_b = [low if i % 2 == 0 else mid_low for i in range(n_features)]

        from sweetbean.util.rating import rating_card_from_values_html

        labels = [
            f"Expert {i + 1} ({validities_pct[i]}%)" for i in range(n_features)
        ]
        example_card_a = rating_card_from_values_html(
            "Product A",
            labels,
            example_a,
            int(1),
            card_kwargs={"background": "rgba(0,0,0,0.15)", "min_width_px": 240},
        )
        example_card_b = rating_card_from_values_html(
            "Product B",
            labels,
            example_b,
            int(1),
            card_kwargs={"background": "rgba(0,0,0,0.15)", "min_width_px": 240},
        )

        example_html = (
            "<div class='sbri-example'>"
            "<div class='sbri-example-prompt'>Which product is higher quality?</div>"
            "<div class='sbri-example-row'>"
            f"{example_card_a}"
            f"{example_card_b}"
            "</div>"
            "<div class='sbri-example-keys'>"
            "Press <kbd>A</kbd> for A, <kbd>B</kbd> for B."
            "</div>"
            "</div>"
        )

        if min_rt_ms and int(min_rt_ms) > 0:
            min_rt_note = (
                "<p class='sbri-muted'>Note: each trial "
                "shows the product ratings first. The A/B answer prompt "
                f"appears after about {int(min_rt_ms)} ms, and responses "
                "are accepted then.</p>"
            )
        else:
            min_rt_note = ""

        return (
            "<p>"
            "<strong>Important:</strong> please stay on this tab and keep "
            "this window in focus until the task is finished; if you switch to "
            "another tab or app for an extended period you will be "
            "<strong>excluded</strong>.</p>"
            "In this experiment you will repeatedly choose between two "
            "fictitious products, <strong>A</strong> and <strong>B</strong>. "
            f"On every trial you will see <strong>{n_features}</strong> "
            "expert ratings for each product."
            "</p>"
            f"<p>{rating_scale_phrase} The same {n_features} experts "
            "(in the same order) provide ratings for both products on every "
            "trial.</p>"
            "<h3 class='sbri-section'>The experts &amp; their accuracies</h3>"
            "<p>Some experts are more accurate at predicting product quality "
            "than others. <strong>Each expert's accuracy is shown next to "
            "their rating on every trial.</strong> Here are the accuracies "
            "you will see throughout this experiment:</p>"
            f"<ul class='sbri-panel-list'>{accuracy_items}</ul>"
            "<h3 class='sbri-section'>How to respond</h3>"
            "<p>On each trial, decide which product you believe is of higher "
            "quality, then press <kbd>A</kbd> for product <strong>A</strong> "
            "or <kbd>B</kbd> for product <strong>B</strong>. There is no "
            "time limit and no trial-by-trial feedback &mdash; please respond "
            "as accurately as you can based on the ratings and the experts' "
            "accuracies. <strong>At the end of the task you will see how "
            "often your choices matched the optimal benchmark.</strong></p>"
            "<p style='border-left:3px solid #c0392b;padding:6px 12px;"
            "background:rgba(192,57,43,0.08);'>"
            f"{min_rt_note}"
            "<h3 class='sbri-section'>What a trial looks like</h3>"
            f"{example_html}"
            "<p>You will complete approximately "
            f"<strong>{n_trials}</strong> trials in total.</p>"
        )

    # --- online: build (no firebase) --------------------------------------------

    def build_online_experiment(
        self,
        *,
        consent_config: dict | None = None,
        min_rt_ms: int = 0,
    ) -> tuple[Any, list]:
        """Build the sweetbean Experiment + per-trial timelines (no network).

        Parameters
        ----------
        consent_config:
            Flat dict of fields injected into the consent screen. See
            ``InformedConsent.from_sections``.
        min_rt_ms:
            Minimum response time (in ms) per choice trial. The trial
            cannot be advanced for the first ``min_rt_ms`` after it
            renders — useful as a low-effort guardrail against
            single-key spamming on Prolific. Set to 0 to disable.
            Implemented by ``HtmlKeyboardResponse(min_rt=...)`` (see
            ``vendor/sweetbean/.../AUTHORING.md`` §"Minimum response
            time"). Only the per-trial choice screen is gated; the
            consent and instruction screens are unaffected.

        Pattern A from `vendor/sweetbean/src/sweetbean/stimulus/AUTHORING.md`:
        every trial's HTML is pre-rendered in Python (using the
        `render_rating_html` helper, which is a Python-time API) and shipped
        as a string TimelineVariable. The raw `option_a` / `option_b` arrays
        are also kept on each timeline row and promoted into trial data via
        `arg.update`, so they appear as `bean_option_a` / `bean_option_b`
        for `_observations_to_df` to read back.

        We do **not** use a `FunctionVariable` here because all rendering
        inputs (the rating arrays, the validities) are known at build time.
        Using one would force the rendering function to be Transcrypt-safe
        (literals only, no `self`, no module-level helpers like
        `render_rating_html`); see AUTHORING.md §Footguns.
        """
        from sweetbean import Block, Experiment as JsPsychExperiment
        from sweetbean.protection import BotDetection
        from sweetbean.stimulus import (
            HtmlKeyboardResponse,
            InformedConsent,
            RichInstructions,
        )
        from sweetbean.variable import FunctionVariable, TimelineVariable

        instructions_trial = RichInstructions(
            title="Instructions",
            inner_html=self._online_instructions_inner_html(min_rt_ms=min_rt_ms),
            footer_html=(
                "<p class='sbri-press'><strong>Press SPACE to start.</strong></p>"
            ),
            choices=[" "],
            correct_key="",
            fit_to_viewport=False,
        )

        validities_pct = [int(v * 100) for v in self.validities]

        # Keep timeline rows as pure trial parameters (`option_a`, `option_b`).
        # Render HTML from those parameters at runtime so uploaded conditions
        # stay compact and only carry fields needed for downstream decoding.
        #
        # Rendering uses a horizontal-bar widget per expert (filled rectangle
        # whose width = rating / rating_max, plus a "<value>/<max>" label).
        # Bars scale naturally for any `rating_max` — the old "row of pips"
        # widget became visually noisy past `rating_max=4` and arbitrary
        # past ~7. The bar markup mirrors `sweetbean.util.rating.rating_*`
        # but is inlined here because this function is transpiled to JS by
        # SweetBean, and Transcrypt can't follow the import.
        # NOTE: FunctionVariable callables are transpiled to JS — keep this
        # function flat (no nested defs, no module-level references).
        def _trial_html_runtime(
            option_a: list[int],
            option_b: list[int],
            validities_pct_: list[int],
            rating_max_: int,
            reveal_after_ms_: int,
            swap_order_: bool,
        ) -> str:
            html = (
                "<div style='text-align:center;'>"
                "<h2 style='margin:0 0 10px 0;font-size:18px;'>"
                "Which product is higher quality?</h2>"
            )
            # Counterbalance left/right column position across trials
            # (Hilbig & Moshagen 2014, p. 1438). Labels and key bindings stay
            # tied to the products — only the visual order swaps.
            if swap_order_:
                columns = (("B", option_b), ("A", option_a))
            else:
                columns = (("A", option_a), ("B", option_b))
            for product_label, ratings in columns:
                html += (
                    "<div style='display:inline-block;vertical-align:top;"
                    "margin:0 14px;padding:10px 14px;border:2px solid #888;"
                    "border-radius:8px;min-width:240px;text-align:left;'>"
                    f"<h3 style='margin:0 0 6px 0;font-size:16px;text-align:center;'>"
                    f"Product {product_label}</h3>"
                )
                rmax = int(rating_max_)
                if rmax < 1:
                    rmax = 1
                for i, rating in enumerate(ratings):
                    rating_i = int(rating)
                    if rating_i < 0:
                        rating_i = 0
                    if rating_i > rmax:
                        rating_i = rmax
                    pct = int(round(100.0 * rating_i / rmax))
                    html += (
                        "<div style='display:flex;align-items:center;"
                        "justify-content:space-between;gap:10px;margin:4px 0;"
                        "font-size:13px;color:#ddd;'>"
                        "<span style='display:inline-block;min-width:110px;"
                        "white-space:nowrap;'>"
                        f"Expert {i + 1} ({int(validities_pct_[i])}%)</span>"
                        "<span style='display:inline-flex;align-items:center;'>"
                        # bar track
                        "<span style='display:inline-block;vertical-align:middle;"
                        "width:110px;height:14px;background:rgba(255,255,255,0.08);"
                        "border:1px solid #888;border-radius:3px;overflow:hidden;'>"
                        # bar fill (width = pct%)
                        f"<span style='display:block;height:100%;width:{pct}%;"
                        "background:#4682B4;border-radius:2px;'></span>"
                        "</span>"
                        # numeric label
                        "<span style='display:inline-block;vertical-align:middle;"
                        "margin-left:8px;font-variant-numeric:tabular-nums;'>"
                        f"{rating_i}/{rmax}</span>"
                        "</span></div>"
                    )
                html += "</div>"
            if reveal_after_ms_ > 0:
                answer_attrs = (
                    " data-sb-min-rt-reveal='answer' "
                    "style='margin:10px 0 0 0;font-size:13px;"
                    "visibility:hidden;'"
                )
            else:
                answer_attrs = " style='margin:10px 0 0 0;font-size:13px;'"
            html += (
                f"<p{answer_attrs}>Press <strong>A</strong> for A, "
                "<strong>B</strong> for B.</p></div>"
            )
            return html

        # Pre-compute the Lee & Cummins (2004) Bayesian normative answer per
        # trial and ship it as a per-trial `correct_choice` ("a" / "b" / "" for
        # ties). SweetBean wires `correct_key=TimelineVariable("correct_choice")`
        # into jsPsych so each trial gets a `bean_correct: true|false|null`
        # field — the end-of-task feedback screen aggregates over those.
        # `swap_order` counterbalances the left/right column position across
        # trials (Hilbig & Moshagen 2014, p. 1438): half the trials render B
        # in the left column. Labels and key bindings stay tied to the
        # products, so option_a / option_b / correct_choice / response decoding
        # are unaffected.
        n_trials = len(self._trials)
        swap_flags = [True] * (n_trials // 2) + [False] * (n_trials - n_trials // 2)
        np.random.default_rng().shuffle(swap_flags)
        timeline = [
            {
                "option_a": list(a),
                "option_b": list(b),
                "correct_choice": _bayesian_normative_choice(
                    list(a), list(b), list(self.validities)
                ),
                "swap_order": bool(swap),
            }
            for (a, b), swap in zip(self._trials, swap_flags)
        ]

        intro_block = Block(
            [
                InformedConsent.from_sections(
                    research_config={},
                    consent_config=consent_config or {},
                    style="princeton",
                ),
                instructions_trial,
            ],
            [{}],
        )

        trial_stim = HtmlKeyboardResponse(
            stimulus=FunctionVariable(
                "trial_html_runtime",
                _trial_html_runtime,
                [
                    TimelineVariable("option_a"),
                    TimelineVariable("option_b"),
                    validities_pct,
                    int(1),
                    int(min_rt_ms) if min_rt_ms else 0,
                    TimelineVariable("swap_order"),
                ],
            ),
            choices=["a", "b"],
            correct_key=TimelineVariable("correct_choice"),
            min_rt=int(min_rt_ms) if min_rt_ms else None,
        )
        # The rendered per-trial HTML is fully reconstructible from
        # `option_a`, `option_b`, `validities`, and `rating_max` (which we
        # keep on the trial). Drop both the SweetBean-saved `bean_stimulus`
        # and the jsPsych-auto-saved `stimulus` so a 96-trial timeline
        # stays well under the Firestore 1 MB observation-document limit.
        # See AUTHORING.md §"Suppressing fields from trial data".
        trial_stim.skip_data("stimulus")
        # Promote the structured rating arrays into per-trial data so they
        # come back as bean_option_a / bean_option_b in the JSON observation
        # for _observations_to_df. See AUTHORING.md §"Promoting fields into
        # trial data".
        trial_stim.arg["option_a"] = TimelineVariable("option_a")
        trial_stim.arg["option_b"] = TimelineVariable("option_b")

        trial_block = Block([trial_stim], timeline)

        # End-of-task feedback (Hilbig & Moshagen 2014, p. 1438): a single
        # summary screen showing how often the participant's choices matched
        # the Bayesian normative benchmark. Built as a raw JS expression that
        # walks `jsPsych.data` in the browser; only choice trials (those with
        # a non-null `bean_correct`) are counted.
        feedback_js = (
            "(()=>{"
            "const all=jsPsych.data.get().values();"
            "const ch=all.filter(t=>t.bean_correct===true||t.bean_correct===false);"
            "const n=ch.length;"
            "const k=ch.filter(t=>t.bean_correct===true).length;"
            "const pct=n>0?Math.round(100*k/n):0;"
            "return \"<div style='text-align:center;max-width:560px;margin:0 auto;'>\""
            "+\"<h2 style='margin:0 0 14px 0;'>You're done &mdash; thank you!</h2>\""
            "+\"<p style='font-size:15px;line-height:1.5;'>Your choices agreed with the optimal benchmark on <strong>\"+pct+\"%</strong> of trials (\"+k+\" of \"+n+\").</p>\""
            "+\"<p style='margin-top:22px;font-size:13px;opacity:0.85;'>Press <kbd>SPACE</kbd> to finish.</p>\""
            "+\"</div>\";"
            "})()"
        )
        feedback_trial = HtmlKeyboardResponse(
            stimulus=_RawJsExpr(feedback_js),
            choices=[" "],
            correct_key="",
            fit_to_viewport=False,
        )
        feedback_trial.skip_data("stimulus")
        feedback_block = Block([feedback_trial], [{}])

        protections = [BotDetection(track_sliders=False)]
        return (
            JsPsychExperiment(
                [intro_block, trial_block, feedback_block],
                protections=protections,
            ),
            [timeline],
        )

    # --- online: deploy + collect -----------------------------------------------

    def run_online(self, config: OnlineConfig) -> pd.DataFrame:
        from src.online_workflow import run_online as _run_online

        repo_root = Path(__file__).resolve().parents[2]

        def _resolve(p: str | None) -> str | None:
            if not p:
                return None
            path = Path(p)
            return str(path if path.is_absolute() else repo_root / path)

        consent_path = _resolve(config.consent_yaml_path)
        consent_config: dict = {}
        if consent_path and Path(consent_path).exists():
            with open(consent_path, encoding="utf-8") as f:
                consent_config = yaml.safe_load(f) or {}

        experiment, timelines = self.build_online_experiment(
            consent_config=consent_config,
            min_rt_ms=int(getattr(config, "min_rt_ms", 0) or 0),
        )

        credentials_path = _resolve(config.firebase_credentials_path)
        if credentials_path is None:
            raise ValueError("firebase_credentials_path is required to run online.")

        observations = _run_online(
            [experiment] * config.n_subjects,
            [timelines] * config.n_subjects,
            credentials_path=credentials_path,
            is_prolific=config.is_prolific,
            study_completion_time=config.duration,
            prolific_settings_path=_resolve(config.prolific_settings_path),
            study_url=config.study_url,
            completion_code=config.completion_code,
        )
        return self._observations_to_df(observations)

    @classmethod
    def _observations_to_df(cls, observations: list) -> pd.DataFrame:
        """Turn the runner's per-subject observations into the canonical
        modeling frame, plus stash audit sidecars on ``df.attrs``.

        Heavy lifting (parsing the autora envelope, flattening
        ``bot_detection.*``, computing the per-subject quality report,
        splitting kept vs. dropped) happens in
        :mod:`sweetbean.util.data_process` so every domain experiment can
        reuse it. This method only adds HDM-specific column shaping.

        Stashed on the returned frame's ``attrs``:

        * ``participant_report`` — one row per subject, including
          ``prolific_pid`` and ``slot_key`` so we can pay every recruited
          participant (kept *and* dropped). Saved as a CSV sidecar by
          :meth:`Observations.save`.
        * ``excluded_trials``   — raw per-trial rows for dropped subjects,
          with the drop reason attached. Saved as a JSONL sidecar.
        """
        from sweetbean.util.data_process import (
            parse_autora_observations,
            participant_quality_report,
            split_by_quality,
        )

        # Step 1: flatten everything to one row per trial — including
        # bot_detection columns and the prolific_pid envelope.
        all_trials = parse_autora_observations(observations)

        # Step 2: keep only the HDM choice trials. The HtmlKeyboardResponse
        # with a/b choices is the only place those response keys appear; we
        # use the bean_response field (set by sweetbean's _process_response)
        # rather than the raw jsPsych response so this also rejects the
        # SPACE-press feedback screen.
        if not all_trials.empty and "bean_response" in all_trials.columns:
            response_lower = (
                all_trials["bean_response"]
                .astype("string")
                .str.lower()
            )
            choice_mask = response_lower.isin(["a", "b"])
            choice = all_trials[choice_mask].copy()
            choice["response"] = (response_lower[choice_mask] == "b").astype(int)
            # Promote the bean_option_* arrays to the canonical column names.
            for canonical, source in (
                ("option_a_ratings", "bean_option_a"),
                ("option_b_ratings", "bean_option_b"),
            ):
                if source in choice.columns:
                    choice[canonical] = choice[source].apply(
                        lambda v: list(v) if v is not None else None
                    )
            # bean_rt is the choice trial's RT; preserve it for the report.
            if "bean_rt" in choice.columns:
                choice["rt_ms"] = pd.to_numeric(choice["bean_rt"], errors="coerce")
        else:
            choice = pd.DataFrame()

        # Step 3: build the per-subject quality report (uses our `response`
        # column for single-key + alternation; rt_ms for fast-RT;
        # bot_detection_* fields for browser / honeypot / bot_flag).
        report = participant_quality_report(
            choice,
            response_col="response",
            rt_col="rt_ms",
            valid_responses={0, 1},
        )

        # Step 4: split kept vs. dropped trials based on the report.
        kept_trials, excluded_trials = split_by_quality(choice, report)

        if not report.empty:
            dropped_ids = sorted(report.loc[report["drop"], "subject_id"].tolist())
            if dropped_ids:
                info(
                    "DecisionMakingBinaryExperiment: participant filter "
                    f"dropped subject_id(s) {dropped_ids} — see participant "
                    "report sidecar for reasons."
                )

        out = cls.to_canonical_data(kept_trials)
        # Stash sidecars on attrs so Observations.save() can persist them
        # alongside the canonical .jsonl without leaking into the modeling
        # schema. attrs survives reindex/copy on modern pandas.
        out.attrs["participant_report"] = report
        out.attrs["excluded_trials"] = excluded_trials
        return out
