"""autocog is a decision-making codebase; category learning is not part of it.

The category-learning domain (GCM / RULEX / SUSTAIN on the Shepard I-VI
structures) was dropped when this repo was split out of the older `autopi`
tree. Its Python package (`domains.category_learning`, `src.category_learning`)
and its seed YAMLs (`theories/category_learning/`) were never carried over, so
anything still referring to them is dead weight that fails the moment it runs.

These tests fail if a category-learning *dependency* creeps back into shipped
source or config: an import of a category-learning module, or a path into a
category-learning data directory. They deliberately do NOT scan ``results/``
(committed run output — grepping recorded LLM prose for "sustain" would produce
false alarms), and they deliberately do NOT flag bare model names. A run-name
parser that still recognises the legacy token ``gcm`` costs nothing and breaks
nothing; an ``import src.category_learning`` is dead code that crashes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]

# Live dependencies only: importing a category-learning module, or resolving a
# path into one of its data directories. Mentioning "gcm" in a docstring or in a
# tuple of legacy run-name tokens is inert and is not flagged.
FORBIDDEN = re.compile(
    r"""(
          (?:from|import)\s+(?:src\.)?category_learning\b   # import src.category_learning
        | (?:from|import)\s+domains\b                       # import domains(.category_learning)
        | domains\.category_learning\b
        | src\.category_learning\b
        | theories/category_learning\b                      # seed-YAML dir
        | results/category_learning\b                       # results dir
    )""",
    re.IGNORECASE | re.VERBOSE,
)

SCANNED_DIRS = ("src", "scripts", "configs", "theories")
SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".sh"}


def _scanned_files() -> list[Path]:
    out: list[Path] = []
    for d in SCANNED_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and p.suffix in SCANNED_SUFFIXES:
                if "__pycache__" in p.parts:
                    continue
                out.append(p)
    return out


def test_scan_covers_the_tree() -> None:
    """Guard the guard: a broken glob would make the check vacuous."""
    files = _scanned_files()
    assert len(files) > 40, f"only scanned {len(files)} files"
    assert any(p.name == "autocog.py" for p in files)


def test_no_category_learning_references_in_source() -> None:
    offenders: list[str] = []
    for path in _scanned_files():
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                rel = path.relative_to(ROOT)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "category-learning references remain:\n" + "\n".join(offenders)


def test_category_learning_directories_are_absent() -> None:
    for rel in ("domains", "src/category_learning", "theories/category_learning"):
        assert not (ROOT / rel).exists(), f"{rel} should not exist in autocog"


@pytest.mark.parametrize(
    ("config_name", "expected_provider"),
    [("default.yaml", "gemini"), ("mock.yaml", "mock")],
)
def test_configs_load_with_expected_llm_settings(
    config_name: str, expected_provider: str
) -> None:
    """The shipped configs must still load, and carry the exact settings runs rely on.

    Every caller of ``load_config`` reads only ``.llm``, so these are the
    values that actually reach a run.
    """
    cfg = load_config(ROOT / "configs" / config_name)
    assert cfg.llm.provider == expected_provider
    assert cfg.llm.model == "gemini-3.1-pro-preview"
    assert cfg.llm.temperature == 0.7
    assert cfg.llm.max_tokens == 32768
    assert cfg.llm.thinking_budget == 8096
    assert cfg.n_rounds == 5
    assert cfg.seed == 42
