from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.promptable import Promptable


class ArbiterVerdict(Promptable):
    """The arbiter's structured judgment over one round.

    `verdict` is a categorical decision about what the loop should do next:

      * `"new_model"`  — keep the same theory description, but regenerate the
                          model instantiation: the `predict` function, the
                          `policy` function, and the `parameters` ranges.
                          Nothing about the parameter *values* (samples) is
                          implied — those are sampled per simulation. The
                          theory's prose claim is preserved verbatim.
      * `"new_theory"` — the current theory is degenerate; gecco should
                          propose a brand-new Theory (description + model).

    `interpretation` is freeform analysis. `recommendation` is the actionable
    guide the next agent (gecco for `"new_theory"`, the model-improver for
    `"new_model"`) should follow.
    """

    interpretation: str = Field(
        ...,
        description=(
            "Freeform analysis of which theory better captured the observed "
            "data across both experiments, and why."
        ),
    )
    verdict: Literal["new_model", "new_theory"] = Field(
        ...,
        description=(
            "Either 'new_model' (keep the same theory description, regenerate "
            "the predict / policy / parameter-ranges) or 'new_theory' "
            "(the current theory is degenerate; propose a brand-new theory)."
        ),
    )
    target_theory_idx: int = Field(
        ...,
        ge=1,
        le=2,
        description=(
            "Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. "
            "If verdict='new_model', this is the theory whose predict / policy / "
            "parameter-ranges should be regenerated (description preserved); "
            "if verdict='new_theory', this is the theory that will be replaced completely."
        ),
    )
    recommendation: str = Field(
        ...,
        description=(
            "If verdict='new_model': how the current predict / policy / "
            "parameter-ranges should be revised. If verdict='new_theory': a "
            "sketch of the new theory that should be proposed and how it "
            "differs from both."
        ),
    )

class FeedbackVerdict(Promptable):
    """The feedbackverdict's structured judgment over one round.

    `interpretation` is freeform analysis of the experimental results. 
    `verdict` is a categorical decision about what the loop should do next:

      * `"regenerate"`  — regenerate the model/theory instantiation: the 'theory' is prose description, `predict` function, the
                            `policy` function, and the `parameters` ranges.
      * `"continue"` — the current theor/model is good enough; keep the same 
                         theory and model for the next round of experiments.
    `rationale` is the justification for the verdict.
    """

    interpretation: str = Field(
        ...,
        description=(
            "Freeform analysis of the model simulation results on existing experiments and how they do or do not "
            "support the proposed model/theory."
        ),
    )
    verdict: Literal["regenerate", "continue"] = Field(
        ...,
        description=(
            "Either 'regenerate' (regenerate a new model/theory) or 'continue' "
            "(keep the current model/theory)."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "If verdict='regenerate', why the current predict / policy / "
            "parameter-ranges should be revised. If verdict='continue', a "
            "sketch of why the current theory should be kept."
        ),
    )