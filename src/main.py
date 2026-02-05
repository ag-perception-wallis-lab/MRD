from argparse import ArgumentParser
from collections import defaultdict

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
from scenes import Envmap, Scene
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

    experiment.add_argument("--seed", type=int, default=42, help="The seed to use.")

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
        args.spp,
        seed=args.seed,
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

    logs = defaultdict(list)
    # Always create an experiment name for both wandb and local saving
    experiment_name = (
        f"{args.scene}-{args.model}-{args.envmap}"
        if not args.wandb_name
        else args.wandb_name
    )
    if not is_shape_exp:
        experiment_name = experiment_name.replace(f"-{args.envmap}", "")

    if args.wandb:
        res = run(cfg, exp_cfg, logs, args.wandb_project, experiment_name)  # type: ignore
    else:
        res = run(cfg, exp_cfg, logs, None, experiment_name)  # type: ignore


if __name__ == "__main__":
    main()
