from enum import Enum

import drjit as dr
import mitsuba as mi

T = mi.ScalarTransform4f()
MAX_BOUNCES = 10


def sample_camera_positions(number_of_views: int = 8, distance: int = 5) -> list[int]:
    """Samples camera origins around an object, on the Fibonacci lattice.
    https://observablehq.com/@meetamit/fibonacci-lattices

    Args:
        number_of_views (int): The number of views to create.
        distance (int, optional): The distance between the camera and the object.

    Returns:
        A list of camera origins.
    """
    positions = []
    golden_ratio = (1 + 5**0.5) / 2
    d = distance
    for i in range(number_of_views):
        theta = 2 * dr.pi * i / golden_ratio
        phi = dr.acos(1 - 2 * (i + 0.5) / number_of_views)

        positions.append(
            [
                d * dr.cos(theta) * dr.sin(phi),
                d * dr.sin(theta) * dr.sin(phi),
                d * dr.cos(phi),
            ]
        )

    return positions


def setup_views(
    n_views: int,
    fov: int = 45,
    width: int = 256,
    height: int = 256,
    to_grayscale: bool = False,
) -> list[mi.Sensor]:
    sensors = []
    for origin in sample_camera_positions(n_views):
        sensors.append(
            mi.load_dict(
                {
                    "type": "perspective",
                    "fov": fov,
                    "to_world": T.look_at(
                        target=[0, 0, 0], origin=origin, up=[0, 1, 0]
                    ),
                    "film": {
                        "type": "hdrfilm",
                        "width": width,
                        "height": height,
                        "filter": {"type": "gaussian"},
                        "sample_border": True,
                        "pixel_format": "rgb",
                    },
                    "sampler": {"type": "independent", "sample_count": 128},
                }
            )
        )
    return sensors


dragon_scene = {
    "type": "scene",
    "integrator": {
        "type": "direct_projective",
    },
    "emitter": "constant",
    "shape": {
        "type": "ply",
        "filename": "./assets/models/dragon.ply",
        "to_world": T.scale(0.018).translate([0, -15, 0]),
        "bsdf": {
            "type": "diffuse",
        },
    },
}

dog_scene = {
    "type": "scene",
    "integrator": {
        "type": "direct_projective",
    },
    "emitter": "constant",
    "shape": {
        "type": "obj",
        "filename": "./assets/models/dog.obj",
        "bsdf": {
            "type": "diffuse",
        },
    },
}

suzanne_scene = {
    "type": "scene",
    "integrator": {
        "type": "direct_projective",
    },
    "emitter": "constant",
    "shape": {
        "type": "obj",
        "filename": "./assets/models/monkey-subdiv.obj",
        "to_world": T,
        "bsdf": {
            "type": "diffuse",
        },
    },
}

lion_statue_scene = {
    "type": "scene",
    "integrator": {"type": "direct_projective"},
    "emitter": "constant",
    "shape": {
        "type": "obj",
        "filename": "./assets/models/lion.obj",
        "to_world": T.scale(0.15).translate([2.25, -7.5, 0]),
        "bsdf": {"type": "diffuse"},
    },
}

translucent = {
    "type": "scene",
    "integrator": {
        "type": "path",
        "max_depth": 12,
        # "rr_depth": 5,
    },
    "film": {
        "type": "hdrfilm",
        "width": 256,
        "height": 256,
        "sample_border": True,
    },
    # ---- Sensors ----
    "sensor_front": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 28.841546,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, 10.0])
        .rotate([0, 0, 1], 180.00000500895632)
        .rotate([0, 1, 0], 5.008956130975331e-06)
        .rotate([1, 0, 0], 179.999991348578),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_back": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, -10.0])
        .rotate([0, 0, 1], 2.1894845052756264e-13)
        .rotate([0, 1, 0], 4.785270367996859e-21)
        .rotate([1, 0, 0], -2.5044780654876655e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_right": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
        .rotate([0, 1, 0], -90.00000250447816)
        .rotate([1, 0, 0], 360.00001001791264),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    # "point_light": {
    #     'type': 'point',
    #     'to_world': mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
    #     .rotate([0, 1, 0], -90.00000250447816)
    #     .rotate([1, 0, 0], 360.00001001791264),
    #     'intensity': {
    #         'type': 'spectrum',
    #         'value': 100.0,
    #     }
    # },
    "sensor_left": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([-10.0, 2.0, 0.0])
        .rotate([0, 1, 0], 90.00000250447816)
        .rotate([1, 0, 0], -2.504477861932166e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    # ---- Materials / Textures ----
    "bsdf": {
        "type": "principled",
        "roughness": 0.001,
        "spec_trans": 1.0,
        "eta": 1.49,
        "base_color": {
            "type": "bitmap",
            "filename": "./assets/material-scene/textures/gradient.jpg",
        },
    },
    "mat-PlaneBsdf": {
        "type": "twosided",
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "ref", "id": "texture-checkerboard"},
        },
    },
    "texture-checkerboard": {
        "type": "checkerboard",
        "color0": {"type": "rgb", "value": 0.4},
        "color1": {"type": "rgb", "value": 0.2},
        "to_uv": mi.ScalarTransform4f.scale([8.0, 8.0, 1.0]),
    },
    # ---- Emitter ----
    "emitter-envmap": {
        "type": "envmap",
        "filename": "./assets/material-scene/envmap.exr",
    },
    # ---- Geometry ----
    "plane": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Plane.ply",
        "face_normals": True,
        "bsdf": {"type": "ref", "id": "mat-PlaneBsdf"},
    },
    "dragon": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Dragon.ply",
        "bsdf": {"type": "ref", "id": "bsdf"},
    },
}

diffuse = {
    "type": "scene",
    "integrator": {
        "type": "path",
        "max_depth": MAX_BOUNCES,
        "rr_depth": 5,
    },
    "film": {
        "type": "hdrfilm",
        "width": 256,
        "height": 256,
        "sample_border": True,
    },
    # ---- Sensors ----
    "sensor_front": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 28.841546,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, 10.0])
        .rotate([0, 0, 1], 180.00000500895632)
        .rotate([0, 1, 0], 5.008956130975331e-06)
        .rotate([1, 0, 0], 179.999991348578),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_back": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, -10.0])
        .rotate([0, 0, 1], 2.1894845052756264e-13)
        .rotate([0, 1, 0], 4.785270367996859e-21)
        .rotate([1, 0, 0], -2.5044780654876655e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_right": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
        .rotate([0, 1, 0], -90.00000250447816)
        .rotate([1, 0, 0], 360.00001001791264),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "point_light": {
        "type": "point",
        "to_world": mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
        .rotate([0, 1, 0], -90.00000250447816)
        .rotate([1, 0, 0], 360.00001001791264),
        "intensity": {
            "type": "spectrum",
            "value": 100.0,
        },
    },
    "sensor_left": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([-10.0, 2.0, 0.0])
        .rotate([0, 1, 0], 90.00000250447816)
        .rotate([1, 0, 0], -2.504477861932166e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    # ---- Materials / Textures ----
    "bsdf": {
        "type": "diffuse",
        "reflectance": {
            "type": "bitmap",
            "filename": "./assets/material-scene/textures/gradient.jpg",
        },
    },
    "mat-PlaneBsdf": {
        "type": "twosided",
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "ref", "id": "texture-checkerboard"},
        },
    },
    "texture-checkerboard": {
        "type": "checkerboard",
        "color0": {"type": "rgb", "value": 0.4},
        "color1": {"type": "rgb", "value": 0.2},
        "to_uv": mi.ScalarTransform4f.scale([8.0, 8.0, 1.0]),
    },
    # ---- Emitter ----
    "emitter-envmap": {
        "type": "envmap",
        "filename": "./assets/material-scene/envmap.exr",
    },
    # ---- Geometry ----
    "plane": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Plane.ply",
        "face_normals": True,
        "bsdf": {"type": "ref", "id": "mat-PlaneBsdf"},
    },
    "dragon": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Dragon.ply",
        "bsdf": {"type": "ref", "id": "bsdf"},
    },
}

brushed_metal = {
    "type": "scene",
    "integrator": {
        "type": "path",
        "max_depth": MAX_BOUNCES,
        "rr_depth": 5,
    },
    "film": {
        "type": "hdrfilm",
        "width": 256,
        "height": 256,
        "sample_border": True,
    },
    # ---- Sensors ----
    "sensor_front": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 28.841546,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, 10.0])
        .rotate([0, 0, 1], 180.00000500895632)
        .rotate([0, 1, 0], 5.008956130975331e-06)
        .rotate([1, 0, 0], 179.999991348578),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_back": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, -10.0])
        .rotate([0, 0, 1], 2.1894845052756264e-13)
        .rotate([0, 1, 0], 4.785270367996859e-21)
        .rotate([1, 0, 0], -2.5044780654876655e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_right": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
        .rotate([0, 1, 0], -90.00000250447816)
        .rotate([1, 0, 0], 360.00001001791264),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "point_light": {
        "type": "point",
        "to_world": mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
        .rotate([0, 1, 0], -90.00000250447816)
        .rotate([1, 0, 0], 360.00001001791264),
        "intensity": {
            "type": "spectrum",
            "value": 100.0,
        },
    },
    "sensor_left": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([-10.0, 2.0, 0.0])
        .rotate([0, 1, 0], 90.00000250447816)
        .rotate([1, 0, 0], -2.504477861932166e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    # ---- Materials / Textures ----
    "bsdf": {
        "type": "measured",
        "filename": "./assets/materials/weta_brushed_steel_satin_pink_rgb.bsdf",
    },
    "mat-PlaneBsdf": {
        "type": "twosided",
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "ref", "id": "texture-checkerboard"},
        },
    },
    "texture-checkerboard": {
        "type": "checkerboard",
        "color0": {"type": "rgb", "value": 0.4},
        "color1": {"type": "rgb", "value": 0.2},
        "to_uv": mi.ScalarTransform4f.scale([8.0, 8.0, 1.0]),
    },
    # ---- Emitter ----
    "emitter-envmap": {
        "type": "envmap",
        "filename": "./assets/material-scene/envmap.exr",
    },
    # ---- Geometry ----
    "plane": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Plane.ply",
        "face_normals": True,
        "bsdf": {"type": "ref", "id": "mat-PlaneBsdf"},
    },
    "dragon": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Dragon.ply",
        "bsdf": {"type": "ref", "id": "bsdf"},
    },
}

rosaline = {
    "type": "scene",
    "integrator": {
        "type": "path",
        "max_depth": MAX_BOUNCES,
        "rr_depth": 5,
    },
    "film": {
        "type": "hdrfilm",
        "width": 256,
        "height": 256,
        "sample_border": True,
    },
    # ---- Sensors ----
    "sensor_front": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 28.841546,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, 10.0])
        .rotate([0, 0, 1], 180.00000500895632)
        .rotate([0, 1, 0], 5.008956130975331e-06)
        .rotate([1, 0, 0], 179.999991348578),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_back": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, -10.0])
        .rotate([0, 0, 1], 2.1894845052756264e-13)
        .rotate([0, 1, 0], 4.785270367996859e-21)
        .rotate([1, 0, 0], -2.5044780654876655e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_right": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
        .rotate([0, 1, 0], -90.00000250447816)
        .rotate([1, 0, 0], 360.00001001791264),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "point_light": {
        "type": "point",
        "to_world": mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
        .rotate([0, 1, 0], -90.00000250447816)
        .rotate([1, 0, 0], 360.00001001791264),
        "intensity": {
            "type": "spectrum",
            "value": 100.0,
        },
    },
    "sensor_left": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([-10.0, 2.0, 0.0])
        .rotate([0, 1, 0], 90.00000250447816)
        .rotate([1, 0, 0], -2.504477861932166e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    # ---- Materials / Textures ----
    "bsdf": {
        "type": "measured",
        "filename": "./assets/materials/satin_rosaline_rgb.bsdf",
    },
    "mat-PlaneBsdf": {
        "type": "twosided",
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "ref", "id": "texture-checkerboard"},
        },
    },
    "texture-checkerboard": {
        "type": "checkerboard",
        "color0": {"type": "rgb", "value": 0.4},
        "color1": {"type": "rgb", "value": 0.2},
        "to_uv": mi.ScalarTransform4f.scale([8.0, 8.0, 1.0]),
    },
    # ---- Emitter ----
    "emitter-envmap": {
        "type": "envmap",
        "filename": "./assets/material-scene/envmap.exr",
    },
    # ---- Geometry ----
    "plane": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Plane.ply",
        "face_normals": True,
        "bsdf": {"type": "ref", "id": "mat-PlaneBsdf"},
    },
    "dragon": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Dragon.ply",
        "bsdf": {"type": "ref", "id": "bsdf"},
    },
}

northern_aurora = {
    "type": "scene",
    "integrator": {
        "type": "path",
        "max_depth": MAX_BOUNCES,
        "rr_depth": 5,
    },
    "film": {
        "type": "hdrfilm",
        "width": 256,
        "height": 256,
        "sample_border": True,
    },
    # ---- Sensors ----
    "sensor_front": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 28.841546,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, 10.0])
        .rotate([0, 0, 1], 180.00000500895632)
        .rotate([0, 1, 0], 5.008956130975331e-06)
        .rotate([1, 0, 0], 179.999991348578),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_back": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([0.0, 2.0, -10.0])
        .rotate([0, 0, 1], 2.1894845052756264e-13)
        .rotate([0, 1, 0], 4.785270367996859e-21)
        .rotate([1, 0, 0], -2.5044780654876655e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "sensor_right": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
        .rotate([0, 1, 0], -90.00000250447816)
        .rotate([1, 0, 0], 360.00001001791264),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    "point_light": {
        "type": "point",
        "to_world": mi.ScalarTransform4f.translate([10.0, 2.0, 0.0])
        .rotate([0, 1, 0], -90.00000250447816)
        .rotate([1, 0, 0], 360.00001001791264),
        "intensity": {
            "type": "spectrum",
            "value": 100.0,
        },
    },
    "sensor_left": {
        "type": "perspective",
        "fov_axis": "x",
        "fov": 39.597755,
        "principal_point_offset_x": 0.0,
        "principal_point_offset_y": -0.0,
        "near_clip": 0.1,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.translate([-10.0, 2.0, 0.0])
        .rotate([0, 1, 0], 90.00000250447816)
        .rotate([1, 0, 0], -2.504477861932166e-06),
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
        "film": {"type": "ref", "id": "film"},
    },
    # ---- Materials / Textures ----
    "bsdf": {
        "type": "measured",
        "filename": "./assets/materials/cc_nothern_aurora_rgb.bsdf",
    },
    "mat-PlaneBsdf": {
        "type": "twosided",
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "ref", "id": "texture-checkerboard"},
        },
    },
    "texture-checkerboard": {
        "type": "checkerboard",
        "color0": {"type": "rgb", "value": 0.4},
        "color1": {"type": "rgb", "value": 0.2},
        "to_uv": mi.ScalarTransform4f.scale([8.0, 8.0, 1.0]),
    },
    # ---- Emitter ----
    "emitter-envmap": {
        "type": "envmap",
        "filename": "./assets/material-scene/envmap.exr",
    },
    # ---- Geometry ----
    "plane": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Plane.ply",
        "face_normals": True,
        "bsdf": {"type": "ref", "id": "mat-PlaneBsdf"},
    },
    "dragon": {
        "type": "ply",
        "filename": "./assets/material-scene/meshes/Dragon.ply",
        "bsdf": {"type": "ref", "id": "bsdf"},
    },
}


class Scene(Enum):
    # Shape
    DRAGON = dragon_scene
    DOG = dog_scene
    LIONSTATUE = lion_statue_scene
    SUZANNE = suzanne_scene

    # BSDF
    TRANSLUCENT = translucent
    DIFFUSE = diffuse
    BRUSHED_METAL = brushed_metal
    AURORA = northern_aurora


class Envmap(Enum):
    HALLSTATT = "./assets/envmap/hallstatt4_hd.hdr"
    SKYBOX = "./assets/envmap/autumn_field.exr"
    GARDEN = "./assets/envmap/aloe_farm_shade_house_1k.hdr"
    CONSTANT = {"type": "constant", "radiance": 1.0}
