from typing import Any

from config import Config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import drjit as dr
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from image_processing import linear_to_srgb_ldr
from experiment_common import (
    build_render_flip_grids,
    compute_baseline_rsa,
    compute_baseline_target_latents,
    compute_flip_error,
    compute_latent_rsa,
    init_wandb_run,
    record_baseline_metrics,
    report_early_stop,
    save_wandb_artifacts,
    setup_baseline_models,
)
from utils import (
    EarlyStopping,
    EarlyStoppingConfig,
    compute_similarity,
    log_bsdf_parameters,
    rename_log_files_and_create_video,
    apply_new_lr
)
from config import BSDFConfig


def reconstruct_bsdf(
    cfg: Config,
    bsdf_cfg: BSDFConfig,
    logs: dict[str, dict[int, list]],
    wandb_project: str | None = None,
    wandb_experiment_name: str | None = None,
) -> dict[str, Any]:
    wb_log = init_wandb_run(wandb_project, wandb_experiment_name, bsdf_cfg)
    if wb_log:
        import wandb

    is_baseline_run = cfg.model.__str__() in ['MSE','MAE','DualBuffer']
    is_torch = cfg.model.is_torch
    has_cuda = mi.variant().startswith("cuda")
    scene_dict = cfg.scene
    lr = bsdf_cfg.lr
    dims = cfg.dims
    scene_dict["film"]["width"] = dims[0]
    scene_dict["film"]["height"] = dims[1]

    if is_baseline_run:
        models, baseline_history, rsa_models, rsa_model_names = setup_baseline_models()

    if has_cuda:
        denoiser = mi.OptixDenoiser(dims)

    ref_scene: mi.Scene = mi.load_dict(scene_dict)

    # render reference images
    ref_images = [
            linear_to_srgb_ldr(mi.render(ref_scene, sensor=s, spp=cfg.spp, seed=cfg.seed))
        for s in ref_scene.sensors()
    ]

    if has_cuda:
        ref_images = [denoiser(rndr) for rndr in ref_images]

    # render heldout views with the shape rotated 10° around y-axis
    original_transform = scene_dict['dragon'].get('to_world')
    if original_transform is not None:
        scene_dict['dragon']['to_world'] = original_transform.rotate([0, 1, 0], 10)
    else:
        scene_dict['dragon']['to_world'] = mi.ScalarTransform4f.rotate([0, 1, 0], 10)
    heldout_scene = mi.load_dict(scene_dict)
    if original_transform is not None:
        scene_dict['dragon']['to_world'] = original_transform
    else:
        scene_dict['dragon'].pop('to_world')
    heldout_views = [
        linear_to_srgb_ldr(mi.render(heldout_scene, sensor=s, spp=cfg.spp, seed=cfg.seed))
        for s in heldout_scene.sensors()
    ]

    init_imgs = torch.stack([img.torch().permute(2, 0, 1).contiguous() for img in ref_images])
    target_grid = make_grid(init_imgs, 5).permute(1, 2, 0).cpu().numpy()

    heldout_imgs = torch.stack([img.torch().permute(2, 0, 1).contiguous() for img in heldout_views])
    heldout_grid = make_grid(heldout_imgs, 5).permute(1, 2, 0).cpu().numpy()

    if wb_log:
        wb_log.log({
            "render/Target": wandb.Image(target_grid),
            "render/Heldout": wandb.Image(heldout_grid),
        })

    if is_torch:
        ref_images = [img.torch() for img in ref_images]

    scene_dict["bsdf"] = bsdf_cfg.bsdf
    scene = mi.load_dict(scene_dict)  # pyright: ignore
    params = mi.traverse(scene)  # pyright: ignore
    if bsdf_cfg.params_to_optimize:
        params.keep(bsdf_cfg.params_to_optimize)
    else:
        params.keep(r"^bsdf\.(?:[\w\.]+\.value|[\w\.]*data|specular|eta)$")

    print(params)
    # initialize optimizer and parameters
    optimizer = mi.ad.Adam(lr=lr)
    for k, v in params.items():
        optimizer[k] = v

    es = EarlyStopping(
        EarlyStoppingConfig(
            patience=50,
            min_delta=1e-4,
            mode="min",
            restore_best=True,
        )
    )

    # Precompute target latents for RSA tracking. Only models that produce a
    # latent representation (i.e. not pure pixel losses) support this.
    try:
        collect_latents = True
        if is_baseline_run and cfg.baseline_rsa:
            baseline_target_latent = compute_baseline_target_latents(
                rsa_models, rsa_model_names, init_imgs, heldout_imgs
            )
        else:
            latents = [
                cfg.model(render).flatten().detach().cpu().numpy()
                for render in ref_images
            ]
            heldout_latents = [
                cfg.model(img).flatten().detach().cpu().numpy()
                for img in heldout_views
            ]
            target_latents = np.stack(latents[: len(scene.sensors())])
            heldout_latents = np.stack(heldout_latents)
            rsa = []
            rsa_heldout = []
            sig = []
            sig_heldout = []
    except:  # noqa: E722
        collect_latents = False

    for epoch in tqdm(
        range(bsdf_cfg.epochs), desc="Optimization", total=bsdf_cfg.epochs, unit="epoch"
    ):
        batch_loss = 0.0
        batch_sim = 0.0
        batch_renders = []
        batch_flip = []

        for sensor_idx, sensor in enumerate(scene.sensors()):
            params.update(optimizer)

            target = ref_images[sensor_idx]
            heldout = heldout_imgs[sensor_idx]

            render = mi.render(
                scene, params, sensor=sensor, spp=cfg.spp, seed=cfg.seed * sensor_idx
            )
            render = linear_to_srgb_ldr(render)
            batch_renders.append(
                render.torch().permute(2, 0, 1).contiguous().detach()
            )
            if collect_latents and not (is_baseline_run and cfg.baseline_rsa):
                latents.append(cfg.model(render).detach().cpu().flatten().numpy())

            if str(cfg.model) == "DualBuffer":
                other_render = mi.render(
                    scene, params, sensor=sensor, spp=cfg.spp, seed=cfg.seed * sensor_idx + 1
                )
                other_render = linear_to_srgb_ldr(other_render)
                loss = cfg.model.lossfn(render, other_render, target)
            else:
                loss = cfg.model.lossfn(render, target)

            if is_baseline_run:
                target = target.torch() if not torch.is_tensor(target) else target

            flip_err_map, flip_err = compute_flip_error(render, target)
            batch_flip.append(flip_err_map)

            # Track for plotting
            logs["flip"][sensor_idx].append(float(flip_err))
            logs["loss"][sensor_idx].append(float(loss.torch().item()))

            if is_baseline_run:
                record_baseline_metrics(
                    models, render, target, heldout, cfg, sensor_idx, baseline_history
                )

            dr.backward(loss)
            batch_loss += loss.torch().detach().cpu().item()

            # we can only compute similarity for model latent representations
            # therefore we do not compute it for baselines (mean absolute) and
            # LPIPS (is a similarity measure itself).
            if not is_baseline_run:
                sim = compute_similarity(
                    render, target, cfg.model, shape=(1, 3, *cfg.dims)
                )  # pyright: ignore
                logs["cosine"][sensor_idx].append(sim)
                batch_sim += sim
                sim = compute_similarity(
                    render, heldout, cfg.model, shape=(1, 3, *cfg.dims)
                )  # pyright: ignore
                logs["heldout/cosine"][sensor_idx].append(sim)

        optimizer.step()
        for k in optimizer.keys():
            if k.endswith("eta"):
                optimizer[k] = dr.clip(optimizer[k], 0.001, 4.1)
                continue
            optimizer[k] = dr.clip(optimizer[k], 1e-3, 1.)

        if collect_latents:
            if is_baseline_run and cfg.baseline_rsa:
                rsa_log = compute_baseline_rsa(
                    rsa_models, rsa_model_names, batch_renders, baseline_target_latent, baseline_history
                )
            else:
                current_latents = np.stack(latents[len(scene.sensors()) * (epoch + 1):])
                rsa_log, correlation, significance, heldout_corr, heldout_sig = compute_latent_rsa(
                    cfg.model, current_latents, target_latents, heldout_latents, bsdf_cfg.n_views, logs
                )
                rsa.append(float(correlation))
                sig.append(float(significance))
                rsa_heldout.append(float(heldout_corr))
                sig_heldout.append(float(heldout_sig))

        apply_new_lr(optimizer, lr, epoch)

        image_grid, flip_grid = build_render_flip_grids(batch_renders, batch_flip)

        tqdm.write(
            f"Epoch {epoch + 1} – Loss: {batch_loss / bsdf_cfg.n_views:.6f}, Similarity: {batch_sim / bsdf_cfg.n_views:.6f}"
        )
        bsdf_params = log_bsdf_parameters(optimizer)

        if wb_log:
            images = {
                "render/Step": wandb.Image(image_grid),
                "render/FLIP Error": wandb.Image(flip_grid),
                "Epoch": epoch,
            }
            vals = dict()
            for k, v in logs.items():
                for kk, vv in v.items():
                    vals[f"{k}/view_{kk}"] = vv[-1]
            vals.update(images)
            vals.update(bsdf_params)
            if collect_latents:
                vals.update(rsa_log)

            wb_log.log(vals)

        plt.close("all")

        should_stop = es.step(
            value=batch_loss / bsdf_cfg.n_views,
            optimizer=optimizer,
            params=params,
            epoch=epoch,
        )

        if has_cuda:
            torch.cuda.empty_cache()

        if should_stop:
            report_early_stop(es, epoch)
            break

    if wb_log:
        save_wandb_artifacts(
            wb_log, collect_latents and not (is_baseline_run and cfg.baseline_rsa),
            (rsa, sig, rsa_heldout, sig_heldout), is_baseline_run, baseline_history,
        )
        rename_log_files_and_create_video(wb_log, wandb_experiment_name, seed=None)
        wb_log.finish()
    return logs
