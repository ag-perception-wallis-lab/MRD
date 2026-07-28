"""Shared building blocks for the reconstruction loops in shape.py and bsdf.py.

Both `reconstruct_geometry` and `reconstruct_bsdf` render a set of reference/
heldout views, optionally track a fixed panel of baseline similarity models
alongside the model actually being optimized, and log RSA/FLIP diagnostics to
Weights & Biases every epoch. This module factors out the pieces that are
identical between the two loops.
"""

from collections import defaultdict
import pickle
from typing import Any

from flip_evaluator import evaluate as FLIP
import numpy as np
import torch
from torchvision.transforms import CenterCrop
from torchvision.utils import make_grid

from plot import plot_rdm, plot_rsa_scatter
from utils import compute_rdm, compute_rsa_similarity, compute_similarity, load_all_models


def init_wandb_run(wandb_project: str | None, wandb_experiment_name: str | None, config: Any):
    """Starts a wandb run if both a project and an experiment name are given."""
    if wandb_project and wandb_experiment_name:
        import wandb

        return wandb.init(name=wandb_experiment_name, project=wandb_project, config=config)
    return None


def setup_baseline_models():
    """Loads the fixed panel of models (Resnet, ResnetSIN, CLIP, DINO) used for
    baseline-run RSA/similarity tracking, plus an empty history to fill in.
    """
    models = load_all_models()
    baseline_history = defaultdict(lambda: defaultdict(list))
    rsa_models = [models[0], models[1], models[4], models[5]]
    rsa_model_names = [str(model) for model in rsa_models]
    return models, baseline_history, rsa_models, rsa_model_names


def _model_crop(model) -> CenterCrop | None:
    return CenterCrop(224) if model.__class__.__name__ in ("DINO", "CLIPVision") else None


def compute_baseline_target_latents(
    rsa_models: list,
    rsa_model_names: list[str],
    imgs,
    heldout_imgs,
    img_masks: list | None = None,
    heldout_masks: list | None = None,
) -> dict[str, list]:
    """Precomputes reference/heldout latents for each baseline RSA model."""

    def latent(model, crop, img, mask):
        x = img * mask if mask is not None else img
        x = crop(x) if crop else x
        return model(x).detach().cpu().flatten().numpy()

    baseline_target_latent = {}
    for name, model in zip(rsa_model_names, rsa_models):
        crop = _model_crop(model)
        latents = [
            latent(model, crop, img, img_masks[i] if img_masks else None)
            for i, img in enumerate(imgs)
        ]
        heldout_latents = [
            latent(model, crop, img, heldout_masks[i] if heldout_masks else None)
            for i, img in enumerate(heldout_imgs)
        ]
        baseline_target_latent[f"{name}/main"] = latents
        baseline_target_latent[f"{name}/heldout"] = heldout_latents
    return baseline_target_latent


def record_baseline_metrics(
    models: list,
    render,
    target,
    heldout,
    cfg,
    sensor_idx: int,
    baseline_history: dict,
    render_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    heldout_mask: torch.Tensor | None = None,
) -> None:
    """Records cosine/pearson/spearman similarity for every baseline model at one epoch."""
    with torch.no_grad():
        for model in models:
            crop = _model_crop(model)
            metrics = compute_similarity(
                render, target, model, shape=(1, 3, *cfg.dims), crop=crop, is_baseline=True,
                render_mask=render_mask, target_mask=target_mask,
            )
            heldout_metrics = compute_similarity(
                render, heldout, model, shape=(1, 3, *cfg.dims), crop=crop, is_baseline=True,
                render_mask=render_mask, target_mask=heldout_mask,
            )
            for metric, value in metrics.items():
                baseline_history[metric][sensor_idx].append(value)
                baseline_history[f"heldout/{metric}"][sensor_idx].append(heldout_metrics[metric])


def compute_baseline_rsa(
    rsa_models: list,
    rsa_model_names: list[str],
    batch_renders: list[torch.Tensor],
    baseline_target_latent: dict[str, list],
    baseline_history: dict,
) -> dict[str, float]:
    """RSA of each baseline model's current-epoch renders against its precomputed targets."""
    rsa_log = {}
    for name, model in zip(rsa_model_names, rsa_models):
        crop = _model_crop(model)
        current_latents = np.stack([
            model(crop(view.unsqueeze(0)) if crop else view).detach().cpu().flatten().numpy()
            for view in batch_renders
        ])
        target_latents = np.stack(baseline_target_latent[f"{name}/main"])
        target_latents_heldout = np.stack(baseline_target_latent[f"{name}/heldout"])

        rdm_latent = compute_rdm(current_latents)
        rdm_target = compute_rdm(target_latents)
        rdm_heldout = compute_rdm(target_latents_heldout)
        r, p = compute_rsa_similarity(rdm_latent, rdm_target)
        heldout_r, heldout_p = compute_rsa_similarity(rdm_latent, rdm_heldout)

        baseline_history[f"rsa/{name}/rsa"][0].append(float(r))
        baseline_history[f"rsa/{name}/p"][0].append(float(p))
        baseline_history[f"heldout/rsa/{name}/rsa"][0].append(float(heldout_r))
        baseline_history[f"heldout/rsa/{name}/p"][0].append(float(heldout_p))
        rsa_log[f"RSA/{name}/Correlation"] = r
        rsa_log[f"RSA/{name}/Heldout Correlation"] = heldout_r
    return rsa_log


def compute_latent_rsa(
    model,
    current_latents: np.ndarray,
    target_latents: np.ndarray,
    heldout_latents: np.ndarray,
    n_views: int,
    logs: dict,
) -> tuple[dict, float, float, float, float]:
    """RSA/pearson/spearman of the optimized model's current-epoch latents.

    Returns (rsa_log, correlation, significance, heldout_correlation, heldout_significance).
    """
    for i in range(n_views):
        logs["pearson_latent"][i].append(
            float(model.pearson(current_latents[i], target_latents[i]))
        )
        logs["spearman_latent"][i].append(
            float(model.spearman(current_latents[i], target_latents[i]))
        )
        logs["heldout/pearson_latent"][i].append(
            float(model.pearson(current_latents[i], heldout_latents[i]))
        )
        logs["heldout/spearman_latent"][i].append(
            float(model.spearman(current_latents[i], heldout_latents[i]))
        )

    rdm_latent = compute_rdm(current_latents)
    rdm_target = compute_rdm(target_latents)
    rdm_heldout = compute_rdm(heldout_latents)
    correlation, significance = compute_rsa_similarity(rdm_latent, rdm_target)
    heldout_corr, heldout_sig = compute_rsa_similarity(rdm_latent, rdm_heldout)

    rsa_log = {
        "RSA/RDM Latent": plot_rdm(rdm_latent),
        "RSA/RDM Target": plot_rdm(rdm_target),
        "RSA/Heldout RDM": plot_rdm(rdm_heldout),
        "RSA/RSA": plot_rsa_scatter(rdm_latent, rdm_target),
        "RSA/Heldout RSA": plot_rsa_scatter(rdm_latent, rdm_heldout),
        "RSA/Correlation": correlation,
        "RSA/Significance": significance,
        "RSA/Heldout Correlation": heldout_corr,
        "RSA/Heldout Significance": heldout_sig,
    }
    return rsa_log, correlation, significance, heldout_corr, heldout_sig


def compute_flip_error(render, target) -> tuple[np.ndarray, float]:
    """FLIP error map and mean error between a render and its target, either of
    which may be an mi.TensorXf-like object or an already-converted torch tensor.
    """
    target_np = target.cpu().numpy() if torch.is_tensor(target) else target.numpy()
    flip_err_map, flip_err, _ = FLIP(
        render.numpy(), target_np, "ldr", inputsRGB=True, computeMeanError=True
    )
    return flip_err_map, flip_err


def build_render_flip_grids(
    batch_renders: list[torch.Tensor], batch_flip: list[np.ndarray], nrow: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Tiles the per-view renders and FLIP error maps into two logging grids."""
    renders = torch.stack(batch_renders)
    image_grid = make_grid(renders, nrow, normalize=True, value_range=(0, 1))
    image_grid = image_grid.permute(1, 2, 0).cpu().numpy()

    flip = torch.stack([torch.tensor(f) for f in batch_flip]).permute(0, 3, 1, 2)
    flip_grid = make_grid(flip, nrow, normalize=True, value_range=(0, 1))
    flip_grid = flip_grid.permute(1, 2, 0).cpu().numpy()
    return image_grid, flip_grid


def save_wandb_artifacts(
    wb_log,
    write_rsa_files: bool,
    rsa_lists: tuple[list, list, list, list],
    is_baseline_run: bool,
    baseline_history: dict,
) -> None:
    """Writes the per-run RSA time series and, for baseline runs, the full
    baseline similarity history — used downstream to compute the ECDF plots.
    """
    if write_rsa_files:
        rsa, sig, rsa_heldout, sig_heldout = rsa_lists
        with open(f"{wb_log.dir}/rsa.npy", "wb+") as f:
            pickle.dump(rsa, f)
        with open(f"{wb_log.dir}/rsa-sig.npy", "wb+") as f:
            pickle.dump(sig, f)
        with open(f"{wb_log.dir}/heldout-rsa.npy", "wb+") as f:
            pickle.dump(rsa_heldout, f)
        with open(f"{wb_log.dir}/heldout-rsa-sig.npy", "wb+") as f:
            pickle.dump(sig_heldout, f)

    if is_baseline_run:
        with open(f"{wb_log.dir}/baseline_model_metrics.npy", "wb+") as f:
            pickle.dump(dict(baseline_history), f)


def report_early_stop(es, epoch: int) -> None:
    print(
        f"Early stopping triggered at epoch {epoch} "
        f"(best @ epoch {es.best_epoch}, best={es.best:.6f})."
    )
