from dataclasses import dataclass
from typing import Any
import numpy as np
from model import ModelMixin
import mitsuba as mi


@dataclass
class Config:
    dims: list[int]
    scene: dict[str, Any] | None = None
    model: ModelMixin | None = None
    envmap: str | None = None
    compute_forward: bool = False
    classify: bool = False
    spp: int | None = None
    seed: int | None = 42


@dataclass
class GeometryConfig:
    n_views: int
    lambda_reg: float
    lr: float
    remesh: list[int]
    epochs: int = 500
    alpha: float = 0.8
    class_idx: None | int = None
    # Geometric regularization losses (set to 0 to disable)
    lambda_lap: float = 0.0  # Laplacian smoothness
    lambda_edge: float = 0.0  # Uniform edge length
    lambda_area: float = 0.0  # Uniform triangle area
    lambda_arap: float = 0.0  # As-Rigid-As-Possible


class DragonConfig(GeometryConfig):
    def __init__(self):
        super().__init__(25, 15, 1e-1, [5, 25, 50, 100, 150, 250, 350, 450])


class LionConfig(GeometryConfig):
    def __init__(self):
        super().__init__(
            25, 15, 1e-1, [5, 25, 50, 100, 150, 250, 350, 450], class_idx=286
        )


class LionStatueConfig(GeometryConfig):
    def __init__(self):
        super().__init__(
            25, 15, 1e-1, [5, 25, 50, 100, 150, 250, 350, 450], class_idx=286
        )


class DogConfig(GeometryConfig):
    def __init__(self):
        super().__init__(
            25, 15, 1e-1, remesh=[5, 50, 100, 150, 250, 350, 450], class_idx=235
        )


class SuzanneConfig(GeometryConfig):
    def __init__(self):
        super().__init__(8, 25, 1e-1, [100, 200, 300, 400], class_idx=372)


@dataclass
class BSDFConfig:
    bsdf: dict | None = None
    params_to_optimize: list[str] | None = None
    n_views: int = 4
    lr: float = 3e-2
    epochs: int = 500


Translucent = BSDFConfig(
    bsdf={
        "type": "principled",
        "base_color": {
            "type": "bitmap",
            "data": mi.TensorXf(
                mi.Bitmap(np.zeros((480, 720, 3))).convert(
                    mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32
                )
            ),
        },
        "eta": 1.64,
        "spec_trans": 0.11,
        "roughness": 0.69,
    },
    params_to_optimize=[
        "bsdf.roughness.value",
        "bsdf.eta",
        "bsdf.base_color.data",
        "bsdf.spec_trans.value",
    ],
)

Diffuse = BSDFConfig(
    bsdf={
        "type": "principled",
        "base_color": {
            "type": "bitmap",
            "data": mi.TensorXf(
                mi.Bitmap(np.zeros((480, 720, 3))).convert(
                    mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32
                )
            ),
        },
    },
    params_to_optimize=[
        "bsdf.base_color.data",
    ],
)

BrushedMetal = BSDFConfig(
    bsdf={
        "type": "principled",
        "base_color": {
            "type": "bitmap",
            "data": mi.TensorXf(
                mi.Bitmap(np.zeros((480, 720, 3))).convert(
                    mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32
                )
            ),
        },
        "anisotropic": {
            "type": "bitmap",
            "data": mi.TensorXf(
                mi.Bitmap(np.zeros((480, 720, 3))).convert(
                    mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32
                )
            ),
        },
    },
    params_to_optimize=None,
)

Rosaline = BSDFConfig(
    bsdf={
        "type": "principled",
        "base_color": {
            "type": "bitmap",
            "data": mi.TensorXf(
                mi.Bitmap(np.zeros((480, 720, 3))).convert(
                    mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32
                )
            ),
        },
    },
    params_to_optimize=None,
)

Aurora = BSDFConfig(
    bsdf={
        "type": "principled",
        "anisotropic": 0.0,
        "base_color": {
            "type": "bitmap",
            "data": mi.TensorXf(
                mi.Bitmap(np.zeros((480, 720, 3))).convert(
                    mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32
                )
            ),
        },
    },
    params_to_optimize=None,
)
