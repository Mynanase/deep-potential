"""Shared journal-style matplotlib defaults for experiment figures."""

from __future__ import annotations

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "figure.titlesize": 10.0,
    "axes.prop_cycle": __import__("cycler").cycler(
        color=["#3070b3", "#e1812c", "#3a923a", "#c03d3e", "#8f63b8"]
    ),
    "savefig.bbox": "tight",
}

DATA_COLOR = "black"
MODEL_COLOR = "#3070b3"
TRUTH_COLOR = "#e1812c"
RESIDUAL_CMAP = "coolwarm"


def format_display_unit(unit: str | None) -> str:
    """Format a user-supplied display string without interpreting its value."""
    if not unit:
        return ""
    value = str(unit)
    if "$" not in value and any(token in value for token in ("\\", "^", "_")):
        return f"${value}$"
    return value


def label_with_unit(symbol: str, unit: str | None) -> str:
    """Attach an optional display-only unit to an axis symbol."""
    formatted = format_display_unit(unit)
    return f"{symbol} [{formatted}]" if formatted else symbol
