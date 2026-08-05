"""Guard against ground-truth leakage in LLM-visible experiment strings.

The recovery pipeline hides which decision-making theory generated the
"real" data and asks an LLM to rediscover it. If the experiment class's
LLM-facing text (schema field descriptions, design summaries, task framing)
names the candidate strategies, the LLM is handed a multiple-choice answer
key instead of having to search theory space — which inflates recovery.

These tests pin the de-biased state: no strategy *name* (TTB, WADD,
Tallying, Equal-Weight/EW/EQW, take-the-best, weighted-additive) may appear
in any string that reaches a prompt. The names legitimately live only in the
theory YAMLs under `theories/`, never in the experiment machinery.
"""

import re
from pathlib import Path

import pytest

from src.decision_making_binary_features.experiment import (
    DecisionMakingBinaryExperiment,
)
from src.heuristic_decision_making.experiment import (
    HeuristicDecisionMakingExperiment,
)
from src.theory import Theory

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Strategy names that must not appear in LLM-visible strings. Short acronyms
# are matched on word boundaries so they don't trip on unrelated substrings
# (e.g. "EW" inside "fewer"); the multi-word names are matched literally.
_BANNED = [
    r"\bTTB\b",
    r"\bWADD\b",
    r"\btally",  # "tally", "Tallying" (anchored so "totally" doesn't match)
    r"\bEQW\b",
    r"\bEW\b",
    r"equal[\s-]?weight",
    r"take[\s-]the[\s-]best",
    r"weighted[\s-]additive",
]
_BANNED_RE = re.compile("|".join(_BANNED), re.IGNORECASE)

_CLASSES = [DecisionMakingBinaryExperiment, HeuristicDecisionMakingExperiment]


def _llm_visible_strings(cls) -> dict[str, str]:
    """Every string on `cls` that can be substituted into a prompt.

    Covers the class docstring (may be surfaced by Promptable), the
    `description` framing, the two `pretty_print_*` summaries, the
    subject-facing `introduction_text`, and every response-schema field
    description (pydantic v2 `model_fields`).
    """
    strings: dict[str, str] = {
        "__doc__": cls.__doc__ or "",
        "description": cls.description,
        "introduction_text": cls.introduction_text,
        "pretty_print_header": cls.pretty_print_header(),
        "pretty_print_protocol": cls.pretty_print_protocol(),
    }
    for name, info in cls.model_fields.items():
        strings[f"field:{name}"] = info.description or ""
    for name, desc in cls.parameter_variables.items():
        strings[f"param:{name}"] = desc
    return strings


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: c.__name__)
def test_no_strategy_name_in_llm_visible_strings(cls):
    offenders: list[str] = []
    for where, text in _llm_visible_strings(cls).items():
        hit = _BANNED_RE.search(text)
        if hit:
            offenders.append(f"{where}: matched {hit.group(0)!r} in {text!r}")
    assert not offenders, "GT strategy name leaked into LLM-visible text:\n" + "\n".join(
        offenders
    )


# The two "gibberish" seed theories that start the adversarial loop in the
# non-canonical recovery runs. Their `predict`/`policy` source is rendered
# verbatim into every proposal/arbitration prompt (comments included), so they
# must not name a strategy OR prime the cue-based family. Beyond the strategy
# names, a gibberish seed should never mention "cue", "validit(y/ies)", or call
# anything "canonical" — a maximally family-neutral seed has no reason to.
_SEED_YAMLS = [
    _REPO_ROOT / "theories/heuristic_decision_making/coin_flip.yaml",
    _REPO_ROOT / "theories/heuristic_decision_making/stimulus_parity.yaml",
]
_SEED_BANNED_RE = re.compile(
    "|".join(_BANNED + [r"\bcue", r"validit", r"canonical"]), re.IGNORECASE
)


@pytest.mark.parametrize("yaml_path", _SEED_YAMLS, ids=lambda p: p.stem)
def test_gibberish_seed_source_has_no_family_priming(yaml_path):
    theory = Theory.from_yaml(str(yaml_path))
    # Exactly the fields that reach a prompt via `Theory.pretty_print()`.
    surfaces = {
        "description": theory.description,
        "predict_source": theory.predict_source,
        "policy_source": theory.policy_source,
    }
    offenders: list[str] = []
    for where, text in surfaces.items():
        hit = _SEED_BANNED_RE.search(text or "")
        if hit:
            offenders.append(f"{where}: matched {hit.group(0)!r} in {text!r}")
    assert not offenders, (
        f"gibberish seed {yaml_path.name} primes the GT family in prompt-"
        f"visible source:\n" + "\n".join(offenders)
    )


# The shared prompt templates (arbitration, feedback, model_improvement, …)
# are domain-agnostic and must never name a decision-making strategy: any
# strategy name in a prompt-builder module would reach the LLM verbatim. Scan
# the raw module text so module-level constants, f-strings, and comments are
# all covered. Only the strategy *names* (`_BANNED`) are forbidden here — the
# phrase "ground truth" is used legitimately in these files for the
# accept/reject signal ("use this as ground truth on which advice helped"),
# so it must NOT be banned.
_PROMPT_TEMPLATE_FILES = sorted((_REPO_ROOT / "src" / "prompts").glob("*.py"))


@pytest.mark.parametrize(
    "template_path", _PROMPT_TEMPLATE_FILES, ids=lambda p: p.stem
)
def test_prompt_template_names_no_strategy(template_path):
    text = template_path.read_text()
    offenders = [
        f"line {i}: matched {m.group(0)!r} in {line.strip()!r}"
        for i, line in enumerate(text.splitlines(), start=1)
        if (m := _BANNED_RE.search(line))
    ]
    assert not offenders, (
        f"prompt template {template_path.name} names a GT strategy:\n"
        + "\n".join(offenders)
    )
