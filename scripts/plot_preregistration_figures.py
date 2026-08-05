"""Render the preregistration flagship figure bundle for one human-dataset run.

Folder-parameterized driver over the self-contained bundle that lives inside the
run folder (its `style.py` holds the standardized look — sandy headline, gray
WADD, slate Tallying, sage TTB; axis labels 14 / ticks 12; no big suptitles).
This script just imports that bundle and calls its render entry points, writing
PNG/SVG/PDF into the bundle's `output/`.

Usage:
    python scripts/plot_preregistration_figures.py            # default bundle
    python scripts/plot_preregistration_figures.py --bundle-dir PATH
    python scripts/plot_preregistration_figures.py --layout flagship
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    _REPO_ROOT / "results/heuristic_decision_making/humans/"
    "hdm_full_prolific_run_full/ttb+tallying/preregistration_visualization"
)

# layout name -> (bundle module, render function). Each function returns the
# list of written paths and writes into the bundle's own output/ directory.
LAYOUTS: dict[str, tuple[str, str]] = {
    "flagship": ("plot_flagship", "plot_flagship"),
    "columns": ("plot_flagship_columns", "plot_flagship_columns"),
    "rows": ("plot_flagship_columns", "plot_flagship_rows"),
    "single_row": ("plot_flagship_columns", "plot_flagship_single_row"),
    "single_row_h1_first": (
        "plot_flagship_columns", "plot_flagship_single_row_h1_first",
    ),
    "panels": ("plot_flagship_columns", "plot_standalone_panels"),
}


def _ensure_on_path(bundle_dir: Path) -> Path:
    """Put the bundle on sys.path so its absolute peer imports resolve."""
    bundle_dir = Path(bundle_dir).resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"bundle dir not found: {bundle_dir}")
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))
    return bundle_dir


def render_layout(bundle_dir: Path, layout: str) -> list[Path]:
    """Render one layout from the bundle; returns the written paths."""
    if layout not in LAYOUTS:
        raise KeyError(f"unknown layout {layout!r}; choose from {list(LAYOUTS)}")
    _ensure_on_path(bundle_dir)
    module_name, fn_name = LAYOUTS[layout]
    fn = getattr(importlib.import_module(module_name), fn_name)
    return fn()


def render_bundle(bundle_dir: Path = DEFAULT_BUNDLE) -> list[Path]:
    """Render every layout + standalone panel for the bundle."""
    paths: list[Path] = []
    for layout in LAYOUTS:
        paths.extend(render_layout(bundle_dir, layout))
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render the preregistration bundle.")
    p.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument(
        "--layout", choices=list(LAYOUTS), default=None,
        help="Render only this layout; default renders all.",
    )
    args = p.parse_args(argv)

    if args.layout:
        paths = render_layout(args.bundle_dir, args.layout)
    else:
        paths = render_bundle(args.bundle_dir)
    for path in paths:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
