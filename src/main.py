from argparse import ArgumentParser
from collections import defaultdict
import os
import pickle

from bsdf import reconstruct_bsdf
from config import (
    Aurora,
    BrushedMetal,
    Config,
    Diffuse,
    DogConfig,
    DragonConfig,
    LionConfig,
    LionStatueConfig,
    Rosaline,
    SuzanneConfig,
    Translucent,
)
from model import DINO, CLIPVision, Model
from scenes import Scene, Envmap
from shape import GeometryConfig, reconstruct_geometry


def main():
    parser = ArgumentParser()
    # required
    parser.add_argument(
        "scene",
        choices=[e.name.lower() for e in Scene],
        help="The scene to load.",
    )
    parser.add_argument(
        "envmap",
        choices=[e.name.lower() for e in Envmap],
        help="The environment map to load.",
    )
    parser.add_argument(
        "model",
        choices=[e.name.lower() for e in Model],
        help="Specify the model used for the reconstruction.",
    )

    # optional args
    experiment = parser.add_argument_group(
        "Experiment Settings",
        "These settings will be passed to experiment configuration and mainly handle hyperparameters and some flags.",
    )

    experiment.add_argument(
        "--spp", type=int, default=64, help="The number per samples for each pixel."
    )

    experiment.add_argument("--seed", type=int, default=None, help="The seed to use.")

    experiment.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=500,
        help="The number of epochs to run the experiment for.",
    )

    experiment.add_argument(
        "-d",
        "--dims",
        default=[256, 256],
        help="Image dimensions of the rendered images. Must match the original training size of the model used.",
    )
    experiment.add_argument(
        "-n",
        "--nviews",
        type=int,
        help="Number of views",
    )
    experiment.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    experiment.add_argument(
        "--forward",
        action="store_true",
        help="Whether to compute and visualize the forward gradients (requires Wandb logging).",
    )
    experiment.add_argument(
        "--classify",
        action="store_true",
        help="Whether to use a classification loss for ResNets.",
    )
    experiment.add_argument(
        "--mask",
        action="store_true",
        help="Whether to use shape masks when computing similarity and correlation (excludes background pixels).",
    )
    experiment.add_argument(
        "--baseline-rsa",
        action="store_true",
        help="Compute and write RSA files during baseline runs using DINO latents.",
    )

    shape = parser.add_argument_group(
        "Shape experiment",
        "These are parameters only relevant for the shape reconstruction.",
    )
    shape.add_argument(
        "-l",
        type=int,
        help="Regularization factor for Large Steps gradient conditioning (controls the smoothness).",
    )
    shape.add_argument(
        "--remesh",
        help="The remeshing steps passed as a space delimited string, i.e. 10 20, defines remeshing steps at epoch 10 and 20.",
    )

    wandb = parser.add_argument_group("Logging")
    wandb.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb logging or not. Sources the credentials from the environment variables.",
    )
    wandb.add_argument(
        "--wandb-name",
        help="The experiment name.",
    )
    wandb.add_argument(
        "--wandb-project",
        default="mrd",
        help="The name of the wandb project.",
    )

    # entry point
    args = parser.parse_args()
    print(args)

    scene = getattr(Scene, args.scene.upper()).value
    is_shape_exp = "bsdf" not in scene.keys()
    envmap = getattr(Envmap, args.envmap.upper()).value
    # init objective
    model = getattr(Model, args.model.upper()).value
    model = model()
    args.dims = args.dims if not isinstance(model, (CLIPVision, DINO)) else [224, 224]

    # Setup experiment config
    cfg = Config(
        args.dims,
        scene,
        model,
        envmap,
        args.forward,
        args.classify,
        args.mask,
        args.spp,
        args.seed if args.seed is not None else 42,
        args.baseline_rsa,
    )

    run = reconstruct_geometry if is_shape_exp else reconstruct_bsdf
    # setup shape scene
    if is_shape_exp:
        match args.scene:
            case "dragon":
                exp_cfg = DragonConfig()
            case "lion":
                exp_cfg = LionConfig()
            case "lionstatue":
                exp_cfg = LionStatueConfig()
            case "dog":
                exp_cfg = DogConfig()
            case "suzanne":
                exp_cfg = SuzanneConfig()
            case _:
                remesh = [int(i) for i in args.remesh.split()] if args.remesh else []
                exp_cfg = GeometryConfig(
                    args.nviews,
                    args.l,
                    args.lr,
                    remesh,
                    args.epochs,
                )
    else:
        match args.scene:
            case "translucent":
                exp_cfg = Translucent
            case "diffuse":
                exp_cfg = Diffuse
            case "brushed_metal":
                exp_cfg = BrushedMetal
            case "rosaline":
                exp_cfg = Rosaline
            case "aurora":
                exp_cfg = Aurora

    if args.lr:
        exp_cfg.lr = args.lr

    if args.epochs:
        exp_cfg.epochs = args.epochs

    if args.l:
        exp_cfg.lambda_reg = args.l

    if args.nviews:
        exp_cfg.n_views = args.nviews

    logs = defaultdict(lambda: defaultdict(list))
    experiment_name = f"{args.scene}-{args.model}-{args.envmap}"
    if args.wandb:
        if not args.wandb_name:
            experiment_name = f"{args.scene}-{args.model}-{args.envmap}"
            if is_shape_exp and args.seed is not None:
                experiment_name += f"_seed{args.seed}"
        else:
            experiment_name = args.wandb_name
        if not is_shape_exp:
            experiment_name = experiment_name.replace(f"-{args.envmap}", "")
        res = run(cfg, exp_cfg, logs, args.wandb_project, experiment_name)  # type: ignore
    else:
        res = run(cfg, exp_cfg, logs)  # type: ignore

    if not args.wandb_name:
        os.mkdir(f"./results/{experiment_name}")
        with open(f"./results/{experiment_name}/metrics.pickle", "wb+") as fp:
            pickle.dump(dict(res), fp)


if __name__ == "__main__":
    main()
