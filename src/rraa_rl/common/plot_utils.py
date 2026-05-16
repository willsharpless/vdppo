from colour import hsl2hex
from matplotlib.colors import LinearSegmentedColormap


def get_BuRd():
    blue = hsl2hex([0.57, 0.5, 0.55])
    light_blue = hsl2hex([0.5, 1.0, 0.995])

    red = hsl2hex([0.028, 0.62, 0.59])
    light_red = hsl2hex([0.098, 1.0, 0.995])

    return LinearSegmentedColormap.from_list(
        "SDF",
        [(0, light_blue), (0.5, blue), (0.5, red), (1, light_red)],
        N=256,
    )


def get_BuRd_smooth():
    blue = hsl2hex([0.57, 0.5, 0.55])
    light_blue = hsl2hex([0.5, 1.0, 0.995])

    red = hsl2hex([0.028, 0.62, 0.59])
    light_red = hsl2hex([0.098, 1.0, 0.995])

    return LinearSegmentedColormap.from_list(
        "SDF",
        [(0, blue), (0.5, light_blue), (0.5, light_red), (1, red)],
        N=256,
    )


def get_BuRd_trunc(trunc_frac: float = 0.2):
    """Get only the middle part of the BuRd colormap."""
    sdf_cm = get_BuRd()

    light_blue = sdf_cm(0.5 * trunc_frac)
    light_red = sdf_cm(1.0 - 0.5 * trunc_frac)

    blue = hsl2hex([0.57, 0.5, 0.55])
    red = hsl2hex([0.028, 0.62, 0.59])

    return LinearSegmentedColormap.from_list(
        "SDF",
        [(0, light_blue), (0.5, blue), (0.5, red), (1, light_red)],
        N=256,
    )
