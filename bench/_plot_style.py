"""Shared visual style for the bench/plot_*.py figure scripts.

Each plot_*.py imports STYLE from this module so the five report figures
look like a coherent set. Per-figure layout (axes, what's plotted, sizes)
lives in each script's body. To override a single setting in one figure,
do `STYLE["ring_color"] = "..."` after import.
"""

STYLE = {
    # series colors
    "ring_color":   "#4C72B0",          # muted blue
    "fc_color":     "#C44E52",          # muted red
    "single_color": "#888888",          # gray (pso_cuda baseline)

    # stacked-bar component colors (used by fig_breakdown.png)
    "stack_colors": {
        "eval":   "#4C72B0",
        "reduce": "#55A868",
        "update": "#C44E52",
        "sync":   "#8172B2",
    },

    # markers and line styles per series
    "ring_marker": "o",
    "fc_marker":   "s",
    "ring_ls":     "-",
    "fc_ls":       "--",
    "baseline_ls": ":",

    # type sizes
    "axis_label_size":  12,
    "tick_label_size":  10,
    "legend_size":       9,
    "panel_title_size": 11,

    # render
    "dpi":         150,
    "grid_alpha":  0.3,
}
