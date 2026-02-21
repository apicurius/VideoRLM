"""Rich CLI output for KUAVi, using a Tokyo Night color theme (aligned with RLM)."""

from __future__ import annotations

from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ============================================================================
# Tokyo Night Color Theme (shared with rlm/logger/verbose.py)
# ============================================================================
COLORS = {
    "primary": "#7AA2F7",  # Soft blue - headers, titles
    "secondary": "#BB9AF7",  # Soft purple - emphasis
    "success": "#9ECE6A",  # Soft green - success, code
    "warning": "#E0AF68",  # Soft amber - warnings
    "error": "#F7768E",  # Soft red/pink - errors
    "text": "#A9B1D6",  # Soft gray-blue - regular text
    "muted": "#565F89",  # Muted gray - less important
    "accent": "#7DCFFF",  # Bright cyan - accents
    "bg_subtle": "#1A1B26",  # Dark background
    "border": "#3B4261",  # Border color
    "code_bg": "#24283B",  # Code background
}

STYLE_PRIMARY = Style(color=COLORS["primary"], bold=True)
STYLE_SECONDARY = Style(color=COLORS["secondary"])
STYLE_SUCCESS = Style(color=COLORS["success"])
STYLE_WARNING = Style(color=COLORS["warning"])
STYLE_ERROR = Style(color=COLORS["error"])
STYLE_TEXT = Style(color=COLORS["text"])
STYLE_MUTED = Style(color=COLORS["muted"])
STYLE_ACCENT = Style(color=COLORS["accent"], bold=True)


class KUAViPrinter:
    """Rich console printer for KUAVi CLI output."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.console = Console() if enabled else None

    def print_header(self, command: str, config: dict[str, Any]) -> None:
        """Print a configuration panel for the current command."""
        if not self.enabled:
            return

        title = Text()
        title.append("◆ ", style=STYLE_ACCENT)
        title.append("KUAVi", style=Style(color=COLORS["primary"], bold=True))
        title.append(f" ━ {command}", style=STYLE_MUTED)

        config_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
            expand=True,
        )
        config_table.add_column("key", style=STYLE_MUTED, width=20)
        config_table.add_column("value", style=STYLE_TEXT)
        config_table.add_column("key2", style=STYLE_MUTED, width=20)
        config_table.add_column("value2", style=STYLE_TEXT)

        items = list(config.items())
        for i in range(0, len(items), 2):
            k1, v1 = items[i]
            row: list[str | Text] = [k1, Text(str(v1), style=STYLE_ACCENT)]
            if i + 1 < len(items):
                k2, v2 = items[i + 1]
                row.extend([k2, Text(str(v2), style=STYLE_ACCENT)])
            else:
                row.extend(["", ""])
            config_table.add_row(*row)

        panel = Panel(
            config_table,
            title=title,
            title_align="left",
            border_style=COLORS["border"],
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_step(self, name: str, detail: str = "") -> None:
        """Print a pipeline step indicator."""
        if not self.enabled:
            return
        text = Text()
        text.append("▸ ", style=STYLE_SUCCESS)
        text.append(name, style=Style(color=COLORS["success"], bold=True))
        if detail:
            text.append(f"  {detail}", style=STYLE_MUTED)
        self.console.print(text)

    def print_step_done(self, name: str, detail: str = "", elapsed: float | None = None) -> None:
        """Print step completion with optional timing."""
        if not self.enabled:
            return
        text = Text()
        text.append("  ✓ ", style=STYLE_SUCCESS)
        text.append(name, style=STYLE_TEXT)
        if detail:
            text.append(f"  {detail}", style=STYLE_MUTED)
        if elapsed is not None:
            text.append(f"  ({elapsed:.2f}s)", style=STYLE_MUTED)
        self.console.print(text)

    def print_search_results(self, results: list[dict[str, Any]], field: str) -> None:
        """Print semantic search results in a table."""
        if not self.enabled:
            return

        if not results:
            text = Text()
            text.append("  No results found.", style=STYLE_MUTED)
            self.console.print(text)
            return

        title = Text()
        title.append("◇ ", style=STYLE_ACCENT)
        title.append(f"Search Results ", style=STYLE_PRIMARY)
        title.append(f"(field={field})", style=STYLE_MUTED)

        table = Table(
            show_edge=False,
            padding=(0, 1),
            expand=True,
            border_style=COLORS["border"],
        )
        table.add_column("#", style=STYLE_MUTED, width=3, justify="right")
        table.add_column("Time Range", style=STYLE_ACCENT, width=18)
        table.add_column("Score", style=STYLE_WARNING, width=8, justify="right")
        table.add_column("Caption", style=STYLE_TEXT)

        for i, r in enumerate(results):
            time_range = f"{r['start_time']:.1f}s - {r['end_time']:.1f}s"
            score = f"{r['score']:.4f}"
            caption = r.get("caption", "")[:100]
            table.add_row(str(i + 1), time_range, score, caption)

        panel = Panel(
            table,
            title=title,
            title_align="left",
            border_style=COLORS["muted"],
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_transcript_results(self, results: list[dict[str, Any]]) -> None:
        """Print transcript search results in a table."""
        if not self.enabled:
            return

        if not results:
            return

        title = Text()
        title.append("◇ ", style=STYLE_ACCENT)
        title.append("Transcript Matches", style=STYLE_PRIMARY)

        table = Table(
            show_edge=False,
            padding=(0, 1),
            expand=True,
            border_style=COLORS["border"],
        )
        table.add_column("Time Range", style=STYLE_ACCENT, width=18)
        table.add_column("Text", style=STYLE_TEXT)

        for r in results[:5]:
            time_range = f"{r['start_time']:.1f}s - {r['end_time']:.1f}s"
            table.add_row(time_range, r["text"])

        panel = Panel(
            table,
            title=title,
            title_align="left",
            border_style=COLORS["muted"],
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_final_summary(self, stats: dict[str, Any]) -> None:
        """Print a summary panel with key stats."""
        if not self.enabled:
            return

        summary_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
        )
        summary_table.add_column("metric", style=STYLE_MUTED)
        summary_table.add_column("value", style=STYLE_ACCENT)

        for key, value in stats.items():
            summary_table.add_row(key, str(value))

        self.console.print()
        self.console.print(Rule(style=COLORS["border"], characters="═"))
        self.console.print(summary_table, justify="center")
        self.console.print(Rule(style=COLORS["border"], characters="═"))
        self.console.print()

    def print_error(self, message: str) -> None:
        """Print an error panel."""
        if not self.enabled:
            return

        title = Text()
        title.append("✗ ", style=STYLE_ERROR)
        title.append("Error", style=Style(color=COLORS["error"], bold=True))

        panel = Panel(
            Text(message, style=STYLE_ERROR),
            title=title,
            title_align="left",
            border_style=COLORS["error"],
            padding=(0, 2),
        )
        self.console.print(panel)
