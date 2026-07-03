"""argus wordmark + the all-seeing-eye glyph (terminal splash + status mark)."""

GLYPH = "◉"  # the iris — the watching eye, used in the status bar

BANNER = r"""   ▄▀█ █▀█ █▀▀ █░█ █▀    ◉
   █▀█ █▀▄ █▄█ █▄█ ▄█"""

TAGLINE = "survivorship-free · point-in-time · honest by construction"


def splash() -> str:
    """The launch banner (wordmark + eye mark + tagline)."""
    return f"{BANNER}\n   {TAGLINE}"
