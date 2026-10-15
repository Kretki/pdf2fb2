"""Pure text formatting of the tqdm-style progress line."""

PARTIAL_BLOCKS = " ▏▎▍▌▋▊▉"
FULL_BLOCK = "█"
MIN_BAR_WIDTH = 10
MAX_BAR_WIDTH = 60
SECONDS_PER_HOUR = 3600
MIN_FRACTION_FOR_ESTIMATE = 0.005


def format_duration(duration_seconds: float | None) -> str:
    """Format a duration like tqdm does.

    Args:
        duration_seconds: Duration in seconds (>= 0), or None when unknown.

    Returns:
        ``MM:SS``, ``H:MM:SS`` for an hour or more, or ``?`` when unknown.
    """
    if duration_seconds is None:
        return "?"
    whole_seconds = int(duration_seconds)
    hour_count, minute_rest = divmod(whole_seconds, SECONDS_PER_HOUR)
    minute_count, second_count = divmod(minute_rest, 60)
    if hour_count:
        return f"{hour_count}:{minute_count:02d}:{second_count:02d}"
    return f"{minute_count:02d}:{second_count:02d}"


def estimate_remaining(elapsed_seconds: float, overall_fraction: float) -> float | None:
    """Estimate the remaining time by linear extrapolation.

    Args:
        elapsed_seconds: Time since the start, in seconds.
        overall_fraction: Completed share, 0..1.

    Returns:
        Remaining seconds, or None while too little is done to estimate.
    """
    if overall_fraction < MIN_FRACTION_FOR_ESTIMATE:
        return None
    return elapsed_seconds * (1.0 - overall_fraction) / overall_fraction


def render_bar(overall_fraction: float, bar_width: int) -> str:
    """Draw the bar body with eighth-block resolution.

    Args:
        overall_fraction: Completed share, clamped to 0..1.
        bar_width: Bar width in characters (> 0).

    Returns:
        String of exactly ``bar_width`` characters.
    """
    eighth_count = int(min(max(overall_fraction, 0.0), 1.0) * bar_width * 8)
    full_count, partial_index = divmod(eighth_count, 8)
    bar_text = FULL_BLOCK * full_count
    if full_count < bar_width:
        bar_text += PARTIAL_BLOCKS[partial_index]
    return bar_text.ljust(bar_width)


def compose_progress_line(title_text: str, overall_fraction: float, elapsed_seconds: float,
                          terminal_columns: int) -> str:
    """Build ``title:  42%|████▏     | 00:31<00:43``, fitted to the terminal width.

    Usage:
        Called by the terminal display on every redraw.

    Args:
        title_text: Short label in front of the bar.
        overall_fraction: Completed share, 0..1.
        elapsed_seconds: Time since the start, in seconds.
        terminal_columns: Terminal width in characters.

    Returns:
        One line without a newline, at most ``terminal_columns - 1`` characters.
    """
    remaining_text = format_duration(estimate_remaining(elapsed_seconds, overall_fraction))
    line_prefix = f"{title_text}: {int(overall_fraction * 100):3d}%|"
    line_suffix = f"| {format_duration(elapsed_seconds)}<{remaining_text}"
    free_columns = terminal_columns - 1 - len(line_prefix) - len(line_suffix)
    bar_width = min(max(free_columns, MIN_BAR_WIDTH), MAX_BAR_WIDTH)
    return fit_to_width(line_prefix + render_bar(overall_fraction, bar_width) + line_suffix, terminal_columns)


def fit_to_width(line_text: str, terminal_columns: int) -> str:
    """Cut a line so it never wraps (wrapping would break the in-place redraw).

    Args:
        line_text: Single-line text.
        terminal_columns: Terminal width in characters (>= 2).

    Returns:
        The text, shortened with ``…`` to at most ``terminal_columns - 1`` characters.
    """
    usable_columns = max(terminal_columns - 1, 1)
    if len(line_text) <= usable_columns:
        return line_text
    return line_text[:usable_columns - 1] + "…"
