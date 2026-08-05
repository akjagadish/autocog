# Diminishing Returns WADD — flagship figure bundle

Self-contained code + data + rendered figures for the preregistration paper's
flagship figure (the concave theory plus the three preregistered tests, H1–H3).

## Run

```bash
pip install -r requirements.txt
python plot_flagship_columns.py
```

This regenerates everything in `output/` (SVG + PDF). No other setup is needed;
all paths are relative to this folder.

## What gets produced (`output/`)

Four full-figure layouts (same content, different arrangement):

| File | Layout |
|------|--------|
| `single_row.*`          | one wide row: theory curve, then H1, H2, H3 |
| `single_row_h1_first.*` | one wide row: H1, then theory curve, then H2, H3 |
| `columns.*`             | theory curve on top, the three hypotheses as columns |
| `rows.*`                | theory curve on top, one row per hypothesis |

Plus every panel on its own (`panel_hero.*`, `panel_h1_example.*`,
`panel_h1_result.*`, …) so the figure can be reassembled in a layout tool.

Each hypothesis has an **example trial** (the stimulus + model predictions) above
its **result** (the preregistered test).

## Where to adjust style

- **`style.py`** — single source of truth for the look:
  - Colours: `CONCAVE`, `WADD`, `RIVAL`, `STEEP`, `FLAT`, `SHIFT`, `OPTION_A/B`, …
    (one named constant = one meaning across every panel).
  - Axes: `COLORS["axis_spine"]`, `COLORS["tick"]`, and `style_axes()` (spine
    colour, tick marks, despining).
  - Fonts: `FONTS`, and `apply_style()` (matplotlib rcParams).
- **`plot_flagship.py`** — the result bar charts:
  - `SHOW_BAR_VALUES` — print the % label inside each bar (on/off).
  - `BAR_EDGE`, `BAR_EDGE_COLOR`, `BAR_EDGE_WIDTH` — outline around each bar.
  - `ERRBAR` — error-bar style (`elinewidth`, `capsize`, colour).
  - `RESULT_YLIM`, `RESULT_YTICKS` — shared y-axis for H1/H2 (H3 keeps its own zoom).
- **`plot_flagship_columns.py`** — the example trials and figure assembly:
  - Tile sizes / column positions (`H1_TILE`, `H1_STEP`, `H1_CX`, `H1_RX`,
    `EX_XLIM`, `EX_YLIM`).
  - Example contents (`H1_TRIALS`, `H2_TRIALS`; H3 reads `data/h3_example.json`).
  - `SUPTITLE` and the per-layout `build_*_figure()` functions.

## Files

```
plot_flagship_columns.py   layouts + example trials (entry point)
plot_flagship.py           hero curve + result bar charts
plot_h1.py                 H1 data loading
plot_h2_h3.py              H2/H3 data loading
trial_diagram.py           model scoring (concave / WADD / tallying / TTB)
data_io.py                 CSV/JSON helpers
style.py                   shared colours, fonts, axis styling
data/                      h1_results.csv, h2_participants.csv,
                           h3_participants.csv, h3_example.json
output/                    rendered figures (SVG + PDF)
```
