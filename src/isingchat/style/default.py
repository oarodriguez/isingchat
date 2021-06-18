from matplotlib import cycler

COLOR_PALETTE = [
    "E24A33",  # red
    "348ABD",  # blue
    "988ED5",  # purple
    "777777",  # gray
    "FBC15E",  # yellow
    "8EBA42",  # green
    "FFB5B8",  # pink
]

STYLE = {
    "lines.linewidth": "1.0",
    "lines.markersize": "4",
    "patch.linewidth": "0.5",
    "patch.facecolor": "#348ABD",  # Blue
    "patch.edgecolor": "#EEEEEE",
    "patch.antialiased": True,
    "font.size": "14.0",
    "font.family": "serif",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": "1",
    "axes.grid": False,
    "axes.titlesize": "x-large",
    "axes.labelsize": "18",
    "axes.labelcolor": "black",
    "axes.formatter.use_mathtext": True,
    "axes.axisbelow": True,  # grid/ticks are below elements (e.g., lines, text)
    "axes.prop_cycle": cycler("color", COLOR_PALETTE),
    "xtick.color": "black",
    "xtick.direction": "out",
    "xtick.top": True,
    "xtick.bottom": True,
    "xtick.major.size": "4",
    "xtick.minor.size": "2",
    "ytick.color": "black",
    "ytick.direction": "out",
    "ytick.right": True,
    "ytick.left": True,
    "ytick.major.size": "4",
    "ytick.minor.size": "2",
    "grid.color": "#DDDDDD",
    "grid.linestyle": "-",  # solid line
    "figure.facecolor": "white",
    "figure.edgecolor": "0.50",
    "figure.dpi": "150",
    "legend.handlelength": "2",
    "legend.fontsize": "6",
    "legend.framealpha": "0.25",
    "mathtext.fontset": "cm",
    "savefig.dpi": "300",
    "savefig.bbox": "tight",
}
