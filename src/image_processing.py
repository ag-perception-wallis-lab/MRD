from typing import Literal
import drjit as dr


def reinhard(x):
    # x >= 0
    return x / (1.0 + x)


def hable_filmic(x):
    # Uncharted 2 / Hable curve
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def linear_to_srgb(x):
    a = 0.055
    thr = 0.0031308
    # piecewise in Dr.Jit
    return dr.clamp(
        dr.select(
            x <= thr, 12.92 * x, (1 + a) * dr.power(dr.maximum(x, 0.0), 1 / 2.4) - a
        ),
        0.0,
        1.0,
    )


def linear_to_srgb_ldr(x, curve: Literal["filmic", "reinhard"] = "reinhard"):
    """
    img_linear: mi.TensorXf with shape (H, W, 3), linear RGB >= 0, requires grad
    Returns: mi.TensorXf (H, W, 3) in sRGB [0,1]
    """
    # white balance + exposure
    if curve == "filmic":
        tm = hable_filmic(x)
    else:
        tm = reinhard(x)

    x = dr.maximum(tm, 0.0)
    x = linear_to_srgb(x)
    return x
