from matplotlib import style

from . import default

STYLES = {"default": default.STYLE}


def use(style_name: str):
    """Update matplotlib style settings from a given style."""
    style_name = style_name.lower()
    if style_name not in STYLES:
        raise ValueError(f"style '{style_name}' not defined")
    style.use(STYLES[style_name])
