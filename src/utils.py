from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable
from os import getenv
from gpytoolbox.remesh_botsch import remesh_botsch
import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, spearmanr, pearsonr
import torch
import wandb

from image_processing import linear_to_srgb_ldr

mi.set_variant(getenv("DEFAULT_MI_VARIANT"))

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb as wb


def write_render_to_disk(img: mi.TensorXf, path: str) -> None:
    bmp = mi.util.convert_to_bitmap(img)
    bmp.write_async(path)


def remesh(
    params: mi.SceneParameters,
    vertex_key: str,
    faces_key: str,
    edge_length: float | None = None,
    **kwargs,
) -> tuple[np.ndarray]:
    """Tessellate the mesh of an object in a scene and return the new vertices and faces.
    This can also be used prior to optimization or within the optimization loop.

    Args:
        scene (SceneDescription): The scene that contains the object to tessellate.
        vertex_key (str, optional): The dictionary key of the shape's vertices.
        faces_key (str, optional): The dictionary key of the shape's faces.
        edge_length (float, optional): The desired edge length, if `None` take average edge length.

    Returns:
        tuple[np.ndarray]: The new vertices and faces.
    """

    verts = params[vertex_key].numpy().reshape((-1, 3)).astype(np.float64)
    faces = params[faces_key].numpy().reshape((-1, 3))

    # h = None will compute average edge length
    verts, faces = remesh_botsch(
        verts, faces, i=5, h=edge_length, project=True, **kwargs
    )

    return verts, faces


def log_bsdf_parameters(opt: mi.ad.Adam) -> dict[str, float]:
    """Extracts current BSDF parameters and returns them as a dict.

    Args:
        opt: The optimizer with the parameter state.

    Returns:
        The BSDF parameters mapped from name to their value.
    """
    bsdf = {}
    for name, val in opt.items():
        # make sure the values do not go out of parameters' bounds
        name = (
            name.replace(".value", "")
            .replace("shape.bsdf.", "")
            .replace("bsdf.", "")
            .replace("bsdf", "")
        )
        k = f"bsdf/{name}"
        if name.endswith("data"):
            # Texture

            if "base_color" in name:
                tex_key = "render/Texture"
            elif "anisotropic" in name:
                tex_key = "render/Anisotropic"

            bsdf[tex_key] = wandb.Image(
                manual_detachment(
                    mi.TensorXf(
                        mi.util.convert_to_bitmap(val).convert(
                            mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32
                        )
                    )
                    .torch()
                    .permute(2, 0, 1)
                )
            )
            continue
        if name.endswith("eta"):
            bsdf[k] = dr.maximum(val, 0).torch().cpu().item()
            continue
        bsdf[k] = torch.clamp(val.torch(), 0.0, 1.0).cpu().item()

    return bsdf


def update_bsdf_parameter(opt: mi.ad.Adam):
    """Updates the BSDF parameters to be physically plausible.
    Defaults to best practices:
        - metallic should be either 0 or 1
        - values should generally be within the range of [0, 1]

    Args:
        opt: The optimizer with the parameter state.

    Returns:
        The optimizer with the updated state.
    """
    for k, v in opt.items():
        if k.endswith("eta"):
            opt[k] = dr.maximum(opt[k], 0)
            continue
        if k.endswith("data"):
            continue
        opt[k] = dr.clip(v, 0.0, 1.0)

    return opt


@torch.no_grad()
def compute_similarity(
    render: mi.TensorXf,
    target: mi.TensorXf | torch.Tensor,
    model: nn.Module,
    verbose: bool = False,
    shape: tuple = (1, 3, 256, 256),
    crop: Callable = None,
) -> float:
    """Computes the cosine similarity on a hypersphere.
    Only use this for models that are producing a latent space.

    Args:
        render: The current render.
        target: The target given the view.
        model: The model to produce the latent space with.

    Returns:
        The cosine similarity measured on an unit hypersphere.
    """
    render = render.torch().view(shape)
    if not torch.is_tensor(target):
        target = target.torch()
    target = target.view(shape)

    if crop:
        render = crop(render)
        target = crop(target)

    # LPIPS is inverted loss = 0 -> higher similarity
    model_name = model.__class__.__name__
    if model_name in ("LPIPS", "LPIPSVGG"):
        sim = 1 - model.lossfn(render, target).numpy().flatten()
        return float(sim)

    # CLIP and DINO are using cosine similarity loss
    # L(x, y) -> [0, 2]; lower is better
    if model_name in ("DINO", "CLIPVision"):
        cossim_loss = model.lossfn(render, target).numpy()
        sim = 1 - cossim_loss
        return float(sim)

    latent = model(render).cpu()
    target_latent = model(target).cpu()

    if model_name == "VGG":
        # accumulates features throughout layers
        latent = latent.view(1, -1)
        target_latent = target_latent.view(1, -1)

    render_hypersphere = map_to_hypersphere(latent)
    target_hypersphere = map_to_hypersphere(target_latent)

    # Optional sanity checks during debugging
    if verbose:
        print(
            render_hypersphere.norm(dim=-1).min().item(),
            render_hypersphere.norm(dim=-1).max().item(),
        )
        print(
            target_hypersphere.norm(dim=-1).min().item(),
            target_hypersphere.norm(dim=-1).max().item(),
        )

    # It is safe to take the item, because the resulting shape should be 1x1.
    sim = render_hypersphere @ target_hypersphere.T
    sim = sim.squeeze().cpu().numpy()
    return float(sim)


@torch.no_grad()
def map_to_hypersphere(z: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Maps the latent representation to a hypersphere to measure similarity with direction only.

    Args:
        z (torch.Tensor): The latent representation to map.

    Returns:
        A latent vector mapped to the unit hypersphere.
    """
    return F.normalize(z, p=2, dim=-1, eps=eps).cpu()


def manual_detachment(x: torch.Tensor) -> torch.Tensor:
    """Manually detaches the gradient, by setting the required grad to False.
    Source of issue: https://github.com/mitsuba-renderer/drjit/issues/114

    Args:
        x: the tensor to detach from the tape

    Returns:
        Tensor with disabled gradients and moved to the CPU.
    """

    x.requires_grad = False
    return x.cpu()


def compute_pca(latent: torch.Tensor, return_model: bool = True) -> [PCA, np.ndarray]:
    """PCA projection with optional model return.

    Args:
        latent: A batch of latent vectors.
        return_model: Whether or not to return the fitted model.

    Returns:
        model: A fitted PCA model.
        trafo: The transformed latent representations.
    """
    model = PCA(n_components=3)
    latent = latent.detach().cpu().numpy()  # pyright: ignore
    trafo = model.fit_transform(latent)

    if return_model:
        return model, trafo

    return trafo


def refit_pca_and_project(latent: torch.Tensor, model: PCA) -> np.ndarray:
    """Fits the existing PCA model with a new batch of latent vectors
    and returns the transformed vectors.

    Args:
        latent: A batch of latent vectors.
        model: An initialized PCA model.
    """
    latent = latent.detach().cpu().numpy()  # pyright: ignore
    return model.fit_transform(latent)


def flat(x: list[torch.Tensor]):
    return torch.hstack([l.flatten() for l in x])

def load_all_models() -> list:
    from model import ModelMixin, Resnet, ResnetSIN, LPIPSVGG, LPIPS, CLIPVision, DINO
    resnet = Resnet()
    resnet_sin = ResnetSIN()
    vgg = LPIPSVGG()
    lpips = LPIPS()
    clip = CLIPVision()
    dino = DINO()

    return [resnet, resnet_sin, vgg, lpips, clip, dino]


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim > 1:
        x = x.squeeze()
        y = y.squeeze()
    x_mean = x.mean()
    y_mean = y.mean()
    cov = ((x - x_mean) * (y - y_mean)).mean()
    std_x = x.std(unbiased=False)
    std_y = y.std(unbiased=False)
    return cov / (std_x * std_y)


def spearman_correlation(
    logits1: torch.Tensor, logits2: torch.Tensor, ties: bool = False
) -> torch.Tensor:
    """
    Compute Spearman's rank correlation between two tensors of shape (1, N) or (N,).

    Args:
        logits1 (torch.Tensor): Tensor of shape (1, N) or (N,)
        logits2 (torch.Tensor): Tensor of shape (1, N) or (N,)

    Returns:
        torch.Tensor: Scalar tensor representing the Spearman correlation.
    """
    logits1 = logits1.squeeze()
    logits2 = logits2.squeeze()
    if logits1.ndim != 1:
        logits1 = logits1.flatten()
    if logits2.ndim != 1:
        logits2 = logits2.flatten()
    assert logits1.shape == logits2.shape and logits1.ndim == 1, (
        "Inputs must be 1D and same shape after squeeze"
    )
    n = logits1.shape[0]

    def rank_tensor(x):
        sorted_indices = x.argsort(descending=True)
        ranks = torch.zeros_like(x, dtype=torch.float)
        ranks[sorted_indices] = torch.arange(
            1, n + 1, dtype=torch.float, device=x.device
        )
        return ranks

    rank1 = rank_tensor(logits1)
    rank2 = rank_tensor(logits2)
    if ties:
        return pearson_correlation(rank1, rank2)

    d = rank1 - rank2
    spearman = 1 - 6 * torch.sum(d**2) / (n * (n**2 - 1))
    return spearman


def to_minus1_1(x):
    x = x / x.max()
    # assumes x in [0,1]; if not, clamp/scale upstream
    return x * 2.0 - 1.0


def flatten_nobatch(t):
    return t.view(t.shape[0], -1)


def diag_gauss_w2(mu1, logvar1, mu2, logvar2, eps=1e-8):
    var1 = logvar1.exp().clamp_min(eps)
    var2 = logvar2.exp().clamp_min(eps)
    std1 = var1.sqrt()
    std2 = var2.sqrt()
    # average over all non-batch dims
    dif_mu = (mu1 - mu2).pow(2)
    dif_std = (std1 - std2).pow(2)
    return (dif_mu + dif_std).mean(dim=list(range(1, mu1.ndim)))


def diag_gauss_skl(mu1, logvar1, mu2, logvar2, eps=1e-8):
    var1 = logvar1.exp().clamp_min(eps)
    var2 = logvar2.exp().clamp_min(eps)
    # KL(q1||q2)
    kl12 = 0.5 * ((var1 / var2) + (mu2 - mu1).pow(2) / var2 - 1 + (logvar2 - logvar1))
    # KL(q2||q1)
    kl21 = 0.5 * ((var2 / var1) + (mu1 - mu2).pow(2) / var1 - 1 + (logvar1 - logvar2))
    skl = kl12 + kl21
    return skl.mean(dim=list(range(1, mu1.ndim)))


@dataclass
class EarlyStoppingConfig:
    patience: int = 10  # epochs to wait without improvement
    min_delta: float = 1e-4  # required improvement amount
    mode: str = "min"  # "min" for loss, "max" for similarity
    restore_best: bool = True  # rewind to best params on stop
    grace: int = 1  # do not stop early once


class EarlyStopping:
    def __init__(self, cfg: EarlyStoppingConfig):
        assert cfg.mode in ("min", "max")
        self.cfg = cfg
        self.best = float("inf") if cfg.mode == "min" else -float("inf")
        self.num_bad = 0
        self.best_state: dict[str, Any] | None = None
        self.best_epoch: int | None = None
        self.n_grace = cfg.grace
        self.n_grace_crossed = 0

    def _is_improved(self, value: float) -> bool:
        if self.cfg.mode == "min":
            return value < (self.best - self.cfg.min_delta)
        else:
            return value > (self.best + self.cfg.min_delta)

    @staticmethod
    def _clone_param(x):
        # Try to make a value copy that works with Dr.Jit/Mitsuba arrays.
        # .copy() is available on many array types; fall back to +0 trick.
        try:
            return x.copy()
        except Exception:
            return x + 0

    def crossed_grace(self) -> bool:
        return self.n_grace_crossed > self.n_grace

    def maybe_save_best(self, optimizer: Any, epoch: int):
        # Keep a lightweight snapshot of the current optimizer parameters.
        self.best_state = {k: self._clone_param(v) for k, v in optimizer.items()}
        self.best_epoch = epoch

    def restore(self, optimizer: Any, params: Any):
        if not (self.cfg.restore_best and self.best_state):
            return
        for k, v in self.best_state.items():
            optimizer[k] = v
        # push values into the scene
        params.update(optimizer)

    def step(self, value: float, optimizer: Any, params: Any, epoch: int) -> bool:
        """Return True if we should stop."""
        if self._is_improved(value):
            self.best = value
            self.num_bad = 0
            self.maybe_save_best(optimizer, epoch)
        else:
            self.num_bad += 1

        if self.num_bad >= self.cfg.patience:
            # Optional rewind to best
            if not self.crossed_grace():
                self.n_grace_crossed += 1
                self.num_bad = 0
                return True
            # Only restore if grace is crossed
            self.restore(optimizer, params)
            return True
        return False


def forward_render(
    scene: mi.Scene,
    params: mi.SceneParameters,
    parameter_key: str,
    sensor: mi.Sensor | None = None,
    *,
    new_param: Any | None = None,
    opt: mi.ad.Optimizer | None = None,
) -> list[torch.Tensor]:
    """
    Computes the forward-mode gradient from parameters to render given a certain parameter key.

    Args:
        scene - the Mitsuba scene to render
        params - the scene parameters that make the scene
        parameter_key - the key of the parameter that will be forward propagated
        sensor - the sensor to render the scene from (optional if specified within the scene)
        new_param - the new value of the parameter
        opt - the optimizer to update the scene parameters with

    Returns:

    """
    if opt:
        params.update(opt)
    else:
        params[parameter_key] = new_param
        params.update()

    if sensor:
        render = mi.render(scene, params, sensor=sensor, spp=8, seed=42)
    else:
        render = mi.render(scene, params, spp=8, seed=42)

    render = linear_to_srgb_ldr(render)
    dr.forward(params[parameter_key])

    grad = dr.grad(render).torch()
    mag = torch.linalg.norm(grad, dim=2)
    s_ldr = mag.pow(2).mean().sqrt().clamp_min(1e-8)
    mag = (mag / s_ldr).clamp(0, 1)

    # to CHW 3-channel grayscale
    grad_img = (
        torch.tensor(mi.util.convert_to_bitmap(grad)).permute(2, 0, 1).contiguous()
    )
    mag = mag.unsqueeze(0).repeat(3, 1, 1).contiguous()
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=mag.device, dtype=mag.dtype)
    gl_ldr = (grad * w).sum(dim=2)
    sl = gl_ldr.pow(2).mean().sqrt().clamp_min(1e-8)
    xl = (gl_ldr / sl).clamp(-1, 1)
    # red = positive, blue = negative, mid-gray = ~zero
    signed_grad = torch.stack(
        [torch.relu(xl), torch.zeros_like(xl), torch.relu(-xl)], dim=0
    )
    signed_grad = (0.5 + 0.5 * signed_grad).contiguous()

    return [grad_img, mag, signed_grad]

def exponential_decay(lr: float, epoch: int, decay: float = 5e-2):
    """
    Exponential decay for the learning rate.
    """
    return lr * (1 - decay) ** epoch

def apply_new_lr(optimizer: mi.ad.Optimizer, epoch: int, warmup: int = 100):
    if isinstance(optimizer.learning_rate(), float):
        lr = {p: optimizer.lr for p in optimizer.keys()}
    else:
        lr = {p: optimizer.learning_rate(p) for p in optimizer.keys()}

    #if warmup != 0 and epoch < warmup:
    lr.pop('bsdf.base_color.data')

    new_lr = {p: max(9e-2, exponential_decay(lr[p], epoch)) for p in lr.keys()}
    optimizer.set_learning_rate(new_lr)
    print('[INFO]\tNew learning rates set.')


def create_local_output_dir(exp_name: str, seed: int | None = None) -> Path:
    """
    Creates a local output directory for storing progress images when wandb is not used.

    Args:
        exp_name: Name of the experiment
        seed: Optional seed value to append to directory name

    Returns:
        Path to the created output directory
    """
    if seed:
        exp_name += f'-{seed}'

    output_dir = Path('./results') / exp_name
    (output_dir / 'render').mkdir(parents=True, exist_ok=True)
    (output_dir / 'render' / 'Step').mkdir(exist_ok=True)
    (output_dir / 'render' / 'Pixel').mkdir(exist_ok=True)

    return output_dir


def save_progress_images(
    output_dir: Path,
    epoch: int,
    image_grid: np.ndarray,
    pixel_error_grid: np.ndarray | None = None,
    grad_images: dict[str, np.ndarray] | None = None
) -> None:
    """
    Saves progress images to disk during optimization when wandb is not used.

    Args:
        output_dir: Base output directory
        epoch: Current epoch number
        image_grid: Rendered image grid (HWC numpy array)
        pixel_error_grid: Optional pixel error visualization (HWC numpy array)
        grad_images: Optional dictionary of gradient visualizations
    """
    from PIL import Image

    # Save rendered images
    img = Image.fromarray((image_grid * 255).astype(np.uint8))
    img.save(output_dir / 'render' / 'Step' / f'Step_{epoch:05d}.png')

    # Save pixel error if provided
    if pixel_error_grid is not None:
        err_img = Image.fromarray(pixel_error_grid.astype(np.uint8))
        err_img.save(output_dir / 'render' / 'Pixel' / f'Pixel_{epoch:05d}.png')

    # Save gradient images if provided
    if grad_images:
        for name, grad_array in grad_images.items():
            grad_dir = output_dir / 'grad' / name
            grad_dir.mkdir(parents=True, exist_ok=True)
            grad_img = Image.fromarray((grad_array * 255).astype(np.uint8))
            grad_img.save(grad_dir / f'{name}_{epoch:05d}.png')


def create_videos_from_local_images(
    output_dir: Path, fps: int = 24, subdirs: dict[str, list[str]] | None = None
) -> None:
    """
    Creates videos from locally saved images using ffmpeg.

    Args:
        output_dir: Base output directory containing the images
        fps: Frames per second for the video
        subdirs: Dictionary mapping category to list of prefixes (e.g., {"render": ["Step", "Pixel"]})
    """
    import subprocess
    import glob

    if subdirs is None:
        subdirs = {"render": ["Pixel", "Step"]}

    for k, v in subdirs.items():
        for prefix in v:
            pattern = str(output_dir / k / prefix / f"{prefix}_*.png")
            files = glob.glob(pattern)

            if not files:
                print(f"No files found for pattern: {pattern}")
                continue

            output = str(output_dir / k / f"{prefix}.mp4")

            cmd = [
                "ffmpeg",
                "-framerate",
                str(fps),
                "-pattern_type",
                "glob",
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",  # Overwrite output file if it exists
                output,
            ]

            print(f"Creating video: {output}")
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"Video created successfully: {output}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to create video: {e.stderr.decode()}")


def rename_log_files_and_create_video(
    run: wb.Run, exp_name: str, fps: int = 24, record_texture: bool = False, seed: int | None = None
):
    """
    Cleans up the run's artifacts stored in Weights and Biases folder.
    Also utilizes ffmpeg to create videos of the optimization process.
    Lastly, copyies the files into the project's root results.
    This requires ffmpeg to be in the PATH.
    """
    import re
    import os
    import glob
    import shutil
    import subprocess

    imgpath = f"{run.dir}/media/images/"
    # subdirs = {"grad": ["Image", "Magnitude", "Signed"], "render": ["Pixel", "Step"]}
    subdirs = {"render": ["Pixel", "Step"]}
    if record_texture:
        subdirs["render"].append("Texture")

    for k, v in subdirs.items():
        for prefix in v:
            for file in glob.glob(f"{imgpath}{k}/{prefix}*.png"):
                if match := re.match(r".*?_(\d+)_.*", file):
                    number = f"{int(match.group(1)):05d}"
                    new_name = f"{imgpath}{k}/{prefix}_{number}.png"
                    os.rename(file, new_name)

    for k, v in subdirs.items():
        for prefix in v:
            # Parameters
            pattern = f"{imgpath}{k}/{prefix}*.png"
            output = f"{imgpath}{k}/{prefix}.mp4"

            cmd = [
                "ffmpeg",
                "-framerate",
                str(fps),
                "-pattern_type",
                "glob",
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                output,
            ]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    if seed:
        exp_name += f'-{seed}'

    res_path = Path('./results')

    if not res_path.exists():
        res_path.mkdir()

    shutil.copytree(f"{run.dir}", f"./results/{exp_name}")


def compute_rdm(X, metric="correlation"):
    """
    X: array of shape (N, D) – N items, D features
    metric: distance metric for pdist, e.g. "correlation", "euclidean", "cosine"
    returns: RDM of shape (N, N)
    """
    # pdist gives condensed distance vector for all pairs
    dists = pdist(X, metric=metric)
    # squareform converts to full N x N matrix
    rdm = squareform(dists)
    return rdm


def compute_rsa_similarity(
    rdm_x: np.ndarray, rdm_y: np.ndarray, method: str = "kendall"
) -> list[np.ndarray]:
    i, j = np.triu_indices(rdm_x.shape[0], k=1)
    a = rdm_x[i, j]
    b = rdm_y[i, j]

    if method == "spearman":
        r, p = spearmanr(a, b)
    elif method == "kendall":
        r, p = kendalltau(a, b)
    else:
        r, p = pearsonr(a, b)

    return r, p


def get_label(idx: int) -> str:
    lut = {
        0: "tench, Tinca tinca",
        1: "goldfish, Carassius auratus",
        2: "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
        3: "tiger shark, Galeocerdo cuvieri",
        4: "hammerhead, hammerhead shark",
        5: "electric ray, crampfish, numbfish, torpedo",
        6: "stingray",
        7: "cock",
        8: "hen",
        9: "ostrich, Struthio camelus",
        10: "brambling, Fringilla montifringilla",
        11: "goldfinch, Carduelis carduelis",
        12: "house finch, linnet, Carpodacus mexicanus",
        13: "junco, snowbird",
        14: "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
        15: "robin, American robin, Turdus migratorius",
        16: "bulbul",
        17: "jay",
        18: "magpie",
        19: "chickadee",
        20: "water ouzel, dipper",
        21: "kite",
        22: "bald eagle, American eagle, Haliaeetus leucocephalus",
        23: "vulture",
        24: "great grey owl, great gray owl, Strix nebulosa",
        25: "European fire salamander, Salamandra salamandra",
        26: "common newt, Triturus vulgaris",
        27: "eft",
        28: "spotted salamander, Ambystoma maculatum",
        29: "axolotl, mud puppy, Ambystoma mexicanum",
        30: "bullfrog, Rana catesbeiana",
        31: "tree frog, tree-frog",
        32: "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
        33: "loggerhead, loggerhead turtle, Caretta caretta",
        34: "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
        35: "mud turtle",
        36: "terrapin",
        37: "box turtle, box tortoise",
        38: "banded gecko",
        39: "common iguana, iguana, Iguana iguana",
        40: "American chameleon, anole, Anolis carolinensis",
        41: "whiptail, whiptail lizard",
        42: "agama",
        43: "frilled lizard, Chlamydosaurus kingi",
        44: "alligator lizard",
        45: "Gila monster, Heloderma suspectum",
        46: "green lizard, Lacerta viridis",
        47: "African chameleon, Chamaeleo chamaeleon",
        48: "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
        49: "African crocodile, Nile crocodile, Crocodylus niloticus",
        50: "American alligator, Alligator mississipiensis",
        51: "triceratops",
        52: "thunder snake, worm snake, Carphophis amoenus",
        53: "ringneck snake, ring-necked snake, ring snake",
        54: "hognose snake, puff adder, sand viper",
        55: "green snake, grass snake",
        56: "king snake, kingsnake",
        57: "garter snake, grass snake",
        58: "water snake",
        59: "vine snake",
        60: "night snake, Hypsiglena torquata",
        61: "boa constrictor, Constrictor constrictor",
        62: "rock python, rock snake, Python sebae",
        63: "Indian cobra, Naja naja",
        64: "green mamba",
        65: "sea snake",
        66: "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
        67: "diamondback, diamondback rattlesnake, Crotalus adamanteus",
        68: "sidewinder, horned rattlesnake, Crotalus cerastes",
        69: "trilobite",
        70: "harvestman, daddy longlegs, Phalangium opilio",
        71: "scorpion",
        72: "black and gold garden spider, Argiope aurantia",
        73: "barn spider, Araneus cavaticus",
        74: "garden spider, Aranea diademata",
        75: "black widow, Latrodectus mactans",
        76: "tarantula",
        77: "wolf spider, hunting spider",
        78: "tick",
        79: "centipede",
        80: "black grouse",
        81: "ptarmigan",
        82: "ruffed grouse, partridge, Bonasa umbellus",
        83: "prairie chicken, prairie grouse, prairie fowl",
        84: "peacock",
        85: "quail",
        86: "partridge",
        87: "African grey, African gray, Psittacus erithacus",
        88: "macaw",
        89: "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
        90: "lorikeet",
        91: "coucal",
        92: "bee eater",
        93: "hornbill",
        94: "hummingbird",
        95: "jacamar",
        96: "toucan",
        97: "drake",
        98: "red-breasted merganser, Mergus serrator",
        99: "goose",
        100: "black swan, Cygnus atratus",
        101: "tusker",
        102: "echidna, spiny anteater, anteater",
        103: "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus",
        104: "wallaby, brush kangaroo",
        105: "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
        106: "wombat",
        107: "jellyfish",
        108: "sea anemone, anemone",
        109: "brain coral",
        110: "flatworm, platyhelminth",
        111: "nematode, nematode worm, roundworm",
        112: "conch",
        113: "snail",
        114: "slug",
        115: "sea slug, nudibranch",
        116: "chiton, coat-of-mail shell, sea cradle, polyplacophore",
        117: "chambered nautilus, pearly nautilus, nautilus",
        118: "Dungeness crab, Cancer magister",
        119: "rock crab, Cancer irroratus",
        120: "fiddler crab",
        121: "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
        122: "American lobster, Northern lobster, Maine lobster, Homarus americanus",
        123: "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
        124: "crayfish, crawfish, crawdad, crawdaddy",
        125: "hermit crab",
        126: "isopod",
        127: "white stork, Ciconia ciconia",
        128: "black stork, Ciconia nigra",
        129: "spoonbill",
        130: "flamingo",
        131: "little blue heron, Egretta caerulea",
        132: "American egret, great white heron, Egretta albus",
        133: "bittern",
        134: "crane",
        135: "limpkin, Aramus pictus",
        136: "European gallinule, Porphyrio porphyrio",
        137: "American coot, marsh hen, mud hen, water hen, Fulica americana",
        138: "bustard",
        139: "ruddy turnstone, Arenaria interpres",
        140: "red-backed sandpiper, dunlin, Erolia alpina",
        141: "redshank, Tringa totanus",
        142: "dowitcher",
        143: "oystercatcher, oyster catcher",
        144: "pelican",
        145: "king penguin, Aptenodytes patagonica",
        146: "albatross, mollymawk",
        147: "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus",
        148: "killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
        149: "dugong, Dugong dugon",
        150: "sea lion",
        151: "Chihuahua",
        152: "Japanese spaniel",
        153: "Maltese dog, Maltese terrier, Maltese",
        154: "Pekinese, Pekingese, Peke",
        155: "Shih-Tzu",
        156: "Blenheim spaniel",
        157: "papillon",
        158: "toy terrier",
        159: "Rhodesian ridgeback",
        160: "Afghan hound, Afghan",
        161: "basset, basset hound",
        162: "beagle",
        163: "bloodhound, sleuthhound",
        164: "bluetick",
        165: "black-and-tan coonhound",
        166: "Walker hound, Walker foxhound",
        167: "English foxhound",
        168: "redbone",
        169: "borzoi, Russian wolfhound",
        170: "Irish wolfhound",
        171: "Italian greyhound",
        172: "whippet",
        173: "Ibizan hound, Ibizan Podenco",
        174: "Norwegian elkhound, elkhound",
        175: "otterhound, otter hound",
        176: "Saluki, gazelle hound",
        177: "Scottish deerhound, deerhound",
        178: "Weimaraner",
        179: "Staffordshire bullterrier, Staffordshire bull terrier",
        180: "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
        181: "Bedlington terrier",
        182: "Border terrier",
        183: "Kerry blue terrier",
        184: "Irish terrier",
        185: "Norfolk terrier",
        186: "Norwich terrier",
        187: "Yorkshire terrier",
        188: "wire-haired fox terrier",
        189: "Lakeland terrier",
        190: "Sealyham terrier, Sealyham",
        191: "Airedale, Airedale terrier",
        192: "cairn, cairn terrier",
        193: "Australian terrier",
        194: "Dandie Dinmont, Dandie Dinmont terrier",
        195: "Boston bull, Boston terrier",
        196: "miniature schnauzer",
        197: "giant schnauzer",
        198: "standard schnauzer",
        199: "Scotch terrier, Scottish terrier, Scottie",
        200: "Tibetan terrier, chrysanthemum dog",
        201: "silky terrier, Sydney silky",
        202: "soft-coated wheaten terrier",
        203: "West Highland white terrier",
        204: "Lhasa, Lhasa apso",
        205: "flat-coated retriever",
        206: "curly-coated retriever",
        207: "golden retriever",
        208: "Labrador retriever",
        209: "Chesapeake Bay retriever",
        210: "German short-haired pointer",
        211: "vizsla, Hungarian pointer",
        212: "English setter",
        213: "Irish setter, red setter",
        214: "Gordon setter",
        215: "Brittany spaniel",
        216: "clumber, clumber spaniel",
        217: "English springer, English springer spaniel",
        218: "Welsh springer spaniel",
        219: "cocker spaniel, English cocker spaniel, cocker",
        220: "Sussex spaniel",
        221: "Irish water spaniel",
        222: "kuvasz",
        223: "schipperke",
        224: "groenendael",
        225: "malinois",
        226: "briard",
        227: "kelpie",
        228: "komondor",
        229: "Old English sheepdog, bobtail",
        230: "Shetland sheepdog, Shetland sheep dog, Shetland",
        231: "collie",
        232: "Border collie",
        233: "Bouvier des Flandres, Bouviers des Flandres",
        234: "Rottweiler",
        235: "German shepherd, German shepherd dog, German police dog, alsatian",
        236: "Doberman, Doberman pinscher",
        237: "miniature pinscher",
        238: "Greater Swiss Mountain dog",
        239: "Bernese mountain dog",
        240: "Appenzeller",
        241: "EntleBucher",
        242: "boxer",
        243: "bull mastiff",
        244: "Tibetan mastiff",
        245: "French bulldog",
        246: "Great Dane",
        247: "Saint Bernard, St Bernard",
        248: "Eskimo dog, husky",
        249: "malamute, malemute, Alaskan malamute",
        250: "Siberian husky",
        251: "dalmatian, coach dog, carriage dog",
        252: "affenpinscher, monkey pinscher, monkey dog",
        253: "basenji",
        254: "pug, pug-dog",
        255: "Leonberg",
        256: "Newfoundland, Newfoundland dog",
        257: "Great Pyrenees",
        258: "Samoyed, Samoyede",
        259: "Pomeranian",
        260: "chow, chow chow",
        261: "keeshond",
        262: "Brabancon griffon",
        263: "Pembroke, Pembroke Welsh corgi",
        264: "Cardigan, Cardigan Welsh corgi",
        265: "toy poodle",
        266: "miniature poodle",
        267: "standard poodle",
        268: "Mexican hairless",
        269: "timber wolf, grey wolf, gray wolf, Canis lupus",
        270: "white wolf, Arctic wolf, Canis lupus tundrarum",
        271: "red wolf, maned wolf, Canis rufus, Canis niger",
        272: "coyote, prairie wolf, brush wolf, Canis latrans",
        273: "dingo, warrigal, warragal, Canis dingo",
        274: "dhole, Cuon alpinus",
        275: "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
        276: "hyena, hyaena",
        277: "red fox, Vulpes vulpes",
        278: "kit fox, Vulpes macrotis",
        279: "Arctic fox, white fox, Alopex lagopus",
        280: "grey fox, gray fox, Urocyon cinereoargenteus",
        281: "tabby, tabby cat",
        282: "tiger cat",
        283: "Persian cat",
        284: "Siamese cat, Siamese",
        285: "Egyptian cat",
        286: "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
        287: "lynx, catamount",
        288: "leopard, Panthera pardus",
        289: "snow leopard, ounce, Panthera uncia",
        290: "jaguar, panther, Panthera onca, Felis onca",
        291: "lion, king of beasts, Panthera leo",
        292: "tiger, Panthera tigris",
        293: "cheetah, chetah, Acinonyx jubatus",
        294: "brown bear, bruin, Ursus arctos",
        295: "American black bear, black bear, Ursus americanus, Euarctos americanus",
        296: "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
        297: "sloth bear, Melursus ursinus, Ursus ursinus",
        298: "mongoose",
        299: "meerkat, mierkat",
        300: "tiger beetle",
        301: "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
        302: "ground beetle, carabid beetle",
        303: "long-horned beetle, longicorn, longicorn beetle",
        304: "leaf beetle, chrysomelid",
        305: "dung beetle",
        306: "rhinoceros beetle",
        307: "weevil",
        308: "fly",
        309: "bee",
        310: "ant, emmet, pismire",
        311: "grasshopper, hopper",
        312: "cricket",
        313: "walking stick, walkingstick, stick insect",
        314: "cockroach, roach",
        315: "mantis, mantid",
        316: "cicada, cicala",
        317: "leafhopper",
        318: "lacewing, lacewing fly",
        319: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
        320: "damselfly",
        321: "admiral",
        322: "ringlet, ringlet butterfly",
        323: "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
        324: "cabbage butterfly",
        325: "sulphur butterfly, sulfur butterfly",
        326: "lycaenid, lycaenid butterfly",
        327: "starfish, sea star",
        328: "sea urchin",
        329: "sea cucumber, holothurian",
        330: "wood rabbit, cottontail, cottontail rabbit",
        331: "hare",
        332: "Angora, Angora rabbit",
        333: "hamster",
        334: "porcupine, hedgehog",
        335: "fox squirrel, eastern fox squirrel, Sciurus niger",
        336: "marmot",
        337: "beaver",
        338: "guinea pig, Cavia cobaya",
        339: "sorrel",
        340: "zebra",
        341: "hog, pig, grunter, squealer, Sus scrofa",
        342: "wild boar, boar, Sus scrofa",
        343: "warthog",
        344: "hippopotamus, hippo, river horse, Hippopotamus amphibius",
        345: "ox",
        346: "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
        347: "bison",
        348: "ram, tup",
        349: "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
        350: "ibex, Capra ibex",
        351: "hartebeest",
        352: "impala, Aepyceros melampus",
        353: "gazelle",
        354: "Arabian camel, dromedary, Camelus dromedarius",
        355: "llama",
        356: "weasel",
        357: "mink",
        358: "polecat, fitch, foulmart, foumart, Mustela putorius",
        359: "black-footed ferret, ferret, Mustela nigripes",
        360: "otter",
        361: "skunk, polecat, wood pussy",
        362: "badger",
        363: "armadillo",
        364: "three-toed sloth, ai, Bradypus tridactylus",
        365: "orangutan, orang, orangutang, Pongo pygmaeus",
        366: "gorilla, Gorilla gorilla",
        367: "chimpanzee, chimp, Pan troglodytes",
        368: "gibbon, Hylobates lar",
        369: "siamang, Hylobates syndactylus, Symphalangus syndactylus",
        370: "guenon, guenon monkey",
        371: "patas, hussar monkey, Erythrocebus patas",
        372: "baboon",
        373: "macaque",
        374: "langur",
        375: "colobus, colobus monkey",
        376: "proboscis monkey, Nasalis larvatus",
        377: "marmoset",
        378: "capuchin, ringtail, Cebus capucinus",
        379: "howler monkey, howler",
        380: "titi, titi monkey",
        381: "spider monkey, Ateles geoffroyi",
        382: "squirrel monkey, Saimiri sciureus",
        383: "Madagascar cat, ring-tailed lemur, Lemur catta",
        384: "indri, indris, Indri indri, Indri brevicaudatus",
        385: "Indian elephant, Elephas maximus",
        386: "African elephant, Loxodonta africana",
        387: "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
        388: "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
        389: "barracouta, snoek",
        390: "eel",
        391: "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
        392: "rock beauty, Holocanthus tricolor",
        393: "anemone fish",
        394: "sturgeon",
        395: "gar, garfish, garpike, billfish, Lepisosteus osseus",
        396: "lionfish",
        397: "puffer, pufferfish, blowfish, globefish",
        398: "abacus",
        399: "abaya",
        400: "academic gown, academic robe, judge's robe",
        401: "accordion, piano accordion, squeeze box",
        402: "acoustic guitar",
        403: "aircraft carrier, carrier, flattop, attack aircraft carrier",
        404: "airliner",
        405: "airship, dirigible",
        406: "altar",
        407: "ambulance",
        408: "amphibian, amphibious vehicle",
        409: "analog clock",
        410: "apiary, bee house",
        411: "apron",
        412: "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
        413: "assault rifle, assault gun",
        414: "backpack, back pack, knapsack, packsack, rucksack, haversack",
        415: "bakery, bakeshop, bakehouse",
        416: "balance beam, beam",
        417: "balloon",
        418: "ballpoint, ballpoint pen, ballpen, Biro",
        419: "Band Aid",
        420: "banjo",
        421: "bannister, banister, balustrade, balusters, handrail",
        422: "barbell",
        423: "barber chair",
        424: "barbershop",
        425: "barn",
        426: "barometer",
        427: "barrel, cask",
        428: "barrow, garden cart, lawn cart, wheelbarrow",
        429: "baseball",
        430: "basketball",
        431: "bassinet",
        432: "bassoon",
        433: "bathing cap, swimming cap",
        434: "bath towel",
        435: "bathtub, bathing tub, bath, tub",
        436: "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
        437: "beacon, lighthouse, beacon light, pharos",
        438: "beaker",
        439: "bearskin, busby, shako",
        440: "beer bottle",
        441: "beer glass",
        442: "bell cote, bell cot",
        443: "bib",
        444: "bicycle-built-for-two, tandem bicycle, tandem",
        445: "bikini, two-piece",
        446: "binder, ring-binder",
        447: "binoculars, field glasses, opera glasses",
        448: "birdhouse",
        449: "boathouse",
        450: "bobsled, bobsleigh, bob",
        451: "bolo tie, bolo, bola tie, bola",
        452: "bonnet, poke bonnet",
        453: "bookcase",
        454: "bookshop, bookstore, bookstall",
        455: "bottlecap",
        456: "bow",
        457: "bow tie, bow-tie, bowtie",
        458: "brass, memorial tablet, plaque",
        459: "brassiere, bra, bandeau",
        460: "breakwater, groin, groyne, mole, bulwark, seawall, jetty",
        461: "breastplate, aegis, egis",
        462: "broom",
        463: "bucket, pail",
        464: "buckle",
        465: "bulletproof vest",
        466: "bullet train, bullet",
        467: "butcher shop, meat market",
        468: "cab, hack, taxi, taxicab",
        469: "caldron, cauldron",
        470: "candle, taper, wax light",
        471: "cannon",
        472: "canoe",
        473: "can opener, tin opener",
        474: "cardigan",
        475: "car mirror",
        476: "carousel, carrousel, merry-go-round, roundabout, whirligig",
        477: "carpenter's kit, tool kit",
        478: "carton",
        479: "car wheel",
        480: "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
        481: "cassette",
        482: "cassette player",
        483: "castle",
        484: "catamaran",
        485: "CD player",
        486: "cello, violoncello",
        487: "cellular telephone, cellular phone, cellphone, cell, mobile phone",
        488: "chain",
        489: "chainlink fence",
        490: "chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour",
        491: "chain saw, chainsaw",
        492: "chest",
        493: "chiffonier, commode",
        494: "chime, bell, gong",
        495: "china cabinet, china closet",
        496: "Christmas stocking",
        497: "church, church building",
        498: "cinema, movie theater, movie theatre, movie house, picture palace",
        499: "cleaver, meat cleaver, chopper",
        500: "cliff dwelling",
        501: "cloak",
        502: "clog, geta, patten, sabot",
        503: "cocktail shaker",
        504: "coffee mug",
        505: "coffeepot",
        506: "coil, spiral, volute, whorl, helix",
        507: "combination lock",
        508: "computer keyboard, keypad",
        509: "confectionery, confectionary, candy store",
        510: "container ship, containership, container vessel",
        511: "convertible",
        512: "corkscrew, bottle screw",
        513: "cornet, horn, trumpet, trump",
        514: "cowboy boot",
        515: "cowboy hat, ten-gallon hat",
        516: "cradle",
        517: "crane",
        518: "crash helmet",
        519: "crate",
        520: "crib, cot",
        521: "Crock Pot",
        522: "croquet ball",
        523: "crutch",
        524: "cuirass",
        525: "dam, dike, dyke",
        526: "desk",
        527: "desktop computer",
        528: "dial telephone, dial phone",
        529: "diaper, nappy, napkin",
        530: "digital clock",
        531: "digital watch",
        532: "dining table, board",
        533: "dishrag, dishcloth",
        534: "dishwasher, dish washer, dishwashing machine",
        535: "disk brake, disc brake",
        536: "dock, dockage, docking facility",
        537: "dogsled, dog sled, dog sleigh",
        538: "dome",
        539: "doormat, welcome mat",
        540: "drilling platform, offshore rig",
        541: "drum, membranophone, tympan",
        542: "drumstick",
        543: "dumbbell",
        544: "Dutch oven",
        545: "electric fan, blower",
        546: "electric guitar",
        547: "electric locomotive",
        548: "entertainment center",
        549: "envelope",
        550: "espresso maker",
        551: "face powder",
        552: "feather boa, boa",
        553: "file, file cabinet, filing cabinet",
        554: "fireboat",
        555: "fire engine, fire truck",
        556: "fire screen, fireguard",
        557: "flagpole, flagstaff",
        558: "flute, transverse flute",
        559: "folding chair",
        560: "football helmet",
        561: "forklift",
        562: "fountain",
        563: "fountain pen",
        564: "four-poster",
        565: "freight car",
        566: "French horn, horn",
        567: "frying pan, frypan, skillet",
        568: "fur coat",
        569: "garbage truck, dustcart",
        570: "gasmask, respirator, gas helmet",
        571: "gas pump, gasoline pump, petrol pump, island dispenser",
        572: "goblet",
        573: "go-kart",
        574: "golf ball",
        575: "golfcart, golf cart",
        576: "gondola",
        577: "gong, tam-tam",
        578: "gown",
        579: "grand piano, grand",
        580: "greenhouse, nursery, glasshouse",
        581: "grille, radiator grille",
        582: "grocery store, grocery, food market, market",
        583: "guillotine",
        584: "hair slide",
        585: "hair spray",
        586: "half track",
        587: "hammer",
        588: "hamper",
        589: "hand blower, blow dryer, blow drier, hair dryer, hair drier",
        590: "hand-held computer, hand-held microcomputer",
        591: "handkerchief, hankie, hanky, hankey",
        592: "hard disc, hard disk, fixed disk",
        593: "harmonica, mouth organ, harp, mouth harp",
        594: "harp",
        595: "harvester, reaper",
        596: "hatchet",
        597: "holster",
        598: "home theater, home theatre",
        599: "honeycomb",
        600: "hook, claw",
        601: "hoopskirt, crinoline",
        602: "horizontal bar, high bar",
        603: "horse cart, horse-cart",
        604: "hourglass",
        605: "iPod",
        606: "iron, smoothing iron",
        607: "jack-o'-lantern",
        608: "jean, blue jean, denim",
        609: "jeep, landrover",
        610: "jersey, T-shirt, tee shirt",
        611: "jigsaw puzzle",
        612: "jinrikisha, ricksha, rickshaw",
        613: "joystick",
        614: "kimono",
        615: "knee pad",
        616: "knot",
        617: "lab coat, laboratory coat",
        618: "ladle",
        619: "lampshade, lamp shade",
        620: "laptop, laptop computer",
        621: "lawn mower, mower",
        622: "lens cap, lens cover",
        623: "letter opener, paper knife, paperknife",
        624: "library",
        625: "lifeboat",
        626: "lighter, light, igniter, ignitor",
        627: "limousine, limo",
        628: "liner, ocean liner",
        629: "lipstick, lip rouge",
        630: "Loafer",
        631: "lotion",
        632: "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
        633: "loupe, jeweler's loupe",
        634: "lumbermill, sawmill",
        635: "magnetic compass",
        636: "mailbag, postbag",
        637: "mailbox, letter box",
        638: "maillot",
        639: "maillot, tank suit",
        640: "manhole cover",
        641: "maraca",
        642: "marimba, xylophone",
        643: "mask",
        644: "matchstick",
        645: "maypole",
        646: "maze, labyrinth",
        647: "measuring cup",
        648: "medicine chest, medicine cabinet",
        649: "megalith, megalithic structure",
        650: "microphone, mike",
        651: "microwave, microwave oven",
        652: "military uniform",
        653: "milk can",
        654: "minibus",
        655: "miniskirt, mini",
        656: "minivan",
        657: "missile",
        658: "mitten",
        659: "mixing bowl",
        660: "mobile home, manufactured home",
        661: "Model T",
        662: "modem",
        663: "monastery",
        664: "monitor",
        665: "moped",
        666: "mortar",
        667: "mortarboard",
        668: "mosque",
        669: "mosquito net",
        670: "motor scooter, scooter",
        671: "mountain bike, all-terrain bike, off-roader",
        672: "mountain tent",
        673: "mouse, computer mouse",
        674: "mousetrap",
        675: "moving van",
        676: "muzzle",
        677: "nail",
        678: "neck brace",
        679: "necklace",
        680: "nipple",
        681: "notebook, notebook computer",
        682: "obelisk",
        683: "oboe, hautboy, hautbois",
        684: "ocarina, sweet potato",
        685: "odometer, hodometer, mileometer, milometer",
        686: "oil filter",
        687: "organ, pipe organ",
        688: "oscilloscope, scope, cathode-ray oscilloscope, CRO",
        689: "overskirt",
        690: "oxcart",
        691: "oxygen mask",
        692: "packet",
        693: "paddle, boat paddle",
        694: "paddlewheel, paddle wheel",
        695: "padlock",
        696: "paintbrush",
        697: "pajama, pyjama, pj's, jammies",
        698: "palace",
        699: "panpipe, pandean pipe, syrinx",
        700: "paper towel",
        701: "parachute, chute",
        702: "parallel bars, bars",
        703: "park bench",
        704: "parking meter",
        705: "passenger car, coach, carriage",
        706: "patio, terrace",
        707: "pay-phone, pay-station",
        708: "pedestal, plinth, footstall",
        709: "pencil box, pencil case",
        710: "pencil sharpener",
        711: "perfume, essence",
        712: "Petri dish",
        713: "photocopier",
        714: "pick, plectrum, plectron",
        715: "pickelhaube",
        716: "picket fence, paling",
        717: "pickup, pickup truck",
        718: "pier",
        719: "piggy bank, penny bank",
        720: "pill bottle",
        721: "pillow",
        722: "ping-pong ball",
        723: "pinwheel",
        724: "pirate, pirate ship",
        725: "pitcher, ewer",
        726: "plane, carpenter's plane, woodworking plane",
        727: "planetarium",
        728: "plastic bag",
        729: "plate rack",
        730: "plow, plough",
        731: "plunger, plumber's helper",
        732: "Polaroid camera, Polaroid Land camera",
        733: "pole",
        734: "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
        735: "poncho",
        736: "pool table, billiard table, snooker table",
        737: "pop bottle, soda bottle",
        738: "pot, flowerpot",
        739: "potter's wheel",
        740: "power drill",
        741: "prayer rug, prayer mat",
        742: "printer",
        743: "prison, prison house",
        744: "projectile, missile",
        745: "projector",
        746: "puck, hockey puck",
        747: "punching bag, punch bag, punching ball, punchball",
        748: "purse",
        749: "quill, quill pen",
        750: "quilt, comforter, comfort, puff",
        751: "racer, race car, racing car",
        752: "racket, racquet",
        753: "radiator",
        754: "radio, wireless",
        755: "radio telescope, radio reflector",
        756: "rain barrel",
        757: "recreational vehicle, RV, R.V.",
        758: "reel",
        759: "reflex camera",
        760: "refrigerator, icebox",
        761: "remote control, remote",
        762: "restaurant, eating house, eating place, eatery",
        763: "revolver, six-gun, six-shooter",
        764: "rifle",
        765: "rocking chair, rocker",
        766: "rotisserie",
        767: "rubber eraser, rubber, pencil eraser",
        768: "rugby ball",
        769: "rule, ruler",
        770: "running shoe",
        771: "safe",
        772: "safety pin",
        773: "saltshaker, salt shaker",
        774: "sandal",
        775: "sarong",
        776: "sax, saxophone",
        777: "scabbard",
        778: "scale, weighing machine",
        779: "school bus",
        780: "schooner",
        781: "scoreboard",
        782: "screen, CRT screen",
        783: "screw",
        784: "screwdriver",
        785: "seat belt, seatbelt",
        786: "sewing machine",
        787: "shield, buckler",
        788: "shoe shop, shoe-shop, shoe store",
        789: "shoji",
        790: "shopping basket",
        791: "shopping cart",
        792: "shovel",
        793: "shower cap",
        794: "shower curtain",
        795: "ski",
        796: "ski mask",
        797: "sleeping bag",
        798: "slide rule, slipstick",
        799: "sliding door",
        800: "slot, one-armed bandit",
        801: "snorkel",
        802: "snowmobile",
        803: "snowplow, snowplough",
        804: "soap dispenser",
        805: "soccer ball",
        806: "sock",
        807: "solar dish, solar collector, solar furnace",
        808: "sombrero",
        809: "soup bowl",
        810: "space bar",
        811: "space heater",
        812: "space shuttle",
        813: "spatula",
        814: "speedboat",
        815: "spider web, spider's web",
        816: "spindle",
        817: "sports car, sport car",
        818: "spotlight, spot",
        819: "stage",
        820: "steam locomotive",
        821: "steel arch bridge",
        822: "steel drum",
        823: "stethoscope",
        824: "stole",
        825: "stone wall",
        826: "stopwatch, stop watch",
        827: "stove",
        828: "strainer",
        829: "streetcar, tram, tramcar, trolley, trolley car",
        830: "stretcher",
        831: "studio couch, day bed",
        832: "stupa, tope",
        833: "submarine, pigboat, sub, U-boat",
        834: "suit, suit of clothes",
        835: "sundial",
        836: "sunglass",
        837: "sunglasses, dark glasses, shades",
        838: "sunscreen, sunblock, sun blocker",
        839: "suspension bridge",
        840: "swab, swob, mop",
        841: "sweatshirt",
        842: "swimming trunks, bathing trunks",
        843: "swing",
        844: "switch, electric switch, electrical switch",
        845: "syringe",
        846: "table lamp",
        847: "tank, army tank, armored combat vehicle, armoured combat vehicle",
        848: "tape player",
        849: "teapot",
        850: "teddy, teddy bear",
        851: "television, television system",
        852: "tennis ball",
        853: "thatch, thatched roof",
        854: "theater curtain, theatre curtain",
        855: "thimble",
        856: "thresher, thrasher, threshing machine",
        857: "throne",
        858: "tile roof",
        859: "toaster",
        860: "tobacco shop, tobacconist shop, tobacconist",
        861: "toilet seat",
        862: "torch",
        863: "totem pole",
        864: "tow truck, tow car, wrecker",
        865: "toyshop",
        866: "tractor",
        867: "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi",
        868: "tray",
        869: "trench coat",
        870: "tricycle, trike, velocipede",
        871: "trimaran",
        872: "tripod",
        873: "triumphal arch",
        874: "trolleybus, trolley coach, trackless trolley",
        875: "trombone",
        876: "tub, vat",
        877: "turnstile",
        878: "typewriter keyboard",
        879: "umbrella",
        880: "unicycle, monocycle",
        881: "upright, upright piano",
        882: "vacuum, vacuum cleaner",
        883: "vase",
        884: "vault",
        885: "velvet",
        886: "vending machine",
        887: "vestment",
        888: "viaduct",
        889: "violin, fiddle",
        890: "volleyball",
        891: "waffle iron",
        892: "wall clock",
        893: "wallet, billfold, notecase, pocketbook",
        894: "wardrobe, closet, press",
        895: "warplane, military plane",
        896: "washbasin, handbasin, washbowl, lavabo, wash-hand basin",
        897: "washer, automatic washer, washing machine",
        898: "water bottle",
        899: "water jug",
        900: "water tower",
        901: "whiskey jug",
        902: "whistle",
        903: "wig",
        904: "window screen",
        905: "window shade",
        906: "Windsor tie",
        907: "wine bottle",
        908: "wing",
        909: "wok",
        910: "wooden spoon",
        911: "wool, woolen, woollen",
        912: "worm fence, snake fence, snake-rail fence, Virginia fence",
        913: "wreck",
        914: "yawl",
        915: "yurt",
        916: "web site, website, internet site, site",
        917: "comic book",
        918: "crossword puzzle, crossword",
        919: "street sign",
        920: "traffic light, traffic signal, stoplight",
        921: "book jacket, dust cover, dust jacket, dust wrapper",
        922: "menu",
        923: "plate",
        924: "guacamole",
        925: "consomme",
        926: "hot pot, hotpot",
        927: "trifle",
        928: "ice cream, icecream",
        929: "ice lolly, lolly, lollipop, popsicle",
        930: "French loaf",
        931: "bagel, beigel",
        932: "pretzel",
        933: "cheeseburger",
        934: "hotdog, hot dog, red hot",
        935: "mashed potato",
        936: "head cabbage",
        937: "broccoli",
        938: "cauliflower",
        939: "zucchini, courgette",
        940: "spaghetti squash",
        941: "acorn squash",
        942: "butternut squash",
        943: "cucumber, cuke",
        944: "artichoke, globe artichoke",
        945: "bell pepper",
        946: "cardoon",
        947: "mushroom",
        948: "Granny Smith",
        949: "strawberry",
        950: "orange",
        951: "lemon",
        952: "fig",
        953: "pineapple, ananas",
        954: "banana",
        955: "jackfruit, jak, jack",
        956: "custard apple",
        957: "pomegranate",
        958: "hay",
        959: "carbonara",
        960: "chocolate sauce, chocolate syrup",
        961: "dough",
        962: "meat loaf, meatloaf",
        963: "pizza, pizza pie",
        964: "potpie",
        965: "burrito",
        966: "red wine",
        967: "espresso",
        968: "cup",
        969: "eggnog",
        970: "alp",
        971: "bubble",
        972: "cliff, drop, drop-off",
        973: "coral reef",
        974: "geyser",
        975: "lakeside, lakeshore",
        976: "promontory, headland, head, foreland",
        977: "sandbar, sand bar",
        978: "seashore, coast, seacoast, sea-coast",
        979: "valley, vale",
        980: "volcano",
        981: "ballplayer, baseball player",
        982: "groom, bridegroom",
        983: "scuba diver",
        984: "rapeseed",
        985: "daisy",
        986: "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
        987: "corn",
        988: "acorn",
        989: "hip, rose hip, rosehip",
        990: "buckeye, horse chestnut, conker",
        991: "coral fungus",
        992: "agaric",
        993: "gyromitra",
        994: "stinkhorn, carrion fungus",
        995: "earthstar",
        996: "hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
        997: "bolete",
        998: "ear, spike, capitulum",
        999: "toilet tissue, toilet paper, bathroom tissue",
    }
    return lut[idx]
