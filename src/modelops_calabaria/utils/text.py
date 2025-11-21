"""Text helpers used across CLI commands."""

import re
from pathlib import Path


def slugify(value: str, max_length: int = 64, default: str = "study") -> str:
    """Convert arbitrary text to a safe filename slug.

    Args:
        value: Source text (e.g., --name)
        max_length: Maximum length of the slug
        default: Fallback slug when nothing remains after cleaning

    Returns:
        Safe, lowercase slug suitable for filenames
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    if not value:
        value = default
    if len(value) > max_length:
        value = value[:max_length].rstrip("-")
    return value or default


def default_output_path(name: str, suffix: str) -> Path:
    """Return a default output path using the name slug."""
    slug = slugify(name)
    return Path(f"{slug}{suffix}")
