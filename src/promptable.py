import json
import re
from pathlib import Path
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict
from pydantic.fields import FieldInfo


class Promptable(BaseModel):
    """
    Base for artifacts that triple-duty as:
      1. an LLM response schema (passed as response_schema=...),
      2. a validator over the LLM's parsed output (cls(**parsed)),
      3. a source for the prompt's "return JSON with these fields" block (instruction_format()).

    Also supports loading instances from YAML so the same class can be hydrated
    from disk or from an LLM response with no extra plumbing.
    """

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def instruction_format(cls) -> str:
        """
        Render the field list for the LLM, derived from `model_fields`.
        Skips underscore-prefixed fields. Uses the LLM-facing alias when present.
        """
        lines: list[str] = []
        for name, finfo in cls.model_fields.items():
            if name.startswith("_"):
                continue
            label = _resolve_label(finfo) or name
            desc = (finfo.description or "").strip()
            lines.append(f"- {label}: {desc}" if desc else f"- {label}")
        return "\n".join(lines)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    @classmethod
    def from_llm_text(cls, text: str) -> Self:
        """Parse a freeform LLM JSON response into this Promptable.

        Use when `response_schema=` can't be passed to the LLM client.
        Gemini's MLDev API rejects any JSON-schema with `additionalProperties`
        (which Pydantic emits for `dict[str, ...]` fields), so dict-bearing
        Promptables (e.g. `Theory`, `Model`) must be requested in freeform
        text mode and decoded here. Strips an optional ```json … ``` fence.
        """
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        payload = match.group(1) if match else text
        return cls.model_validate(json.loads(payload))


def _resolve_label(finfo: FieldInfo) -> str | None:
    """Pick the LLM-facing key for a field: alias > first AliasChoices entry > None."""
    va = finfo.validation_alias
    if isinstance(va, str):
        return finfo.alias or va
    if isinstance(va, AliasChoices):
        first = va.choices[0]
        return first if isinstance(first, str) else None
    return finfo.alias
