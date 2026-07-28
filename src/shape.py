from typing import Any
import drjit as dr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from tqdm import tqdm
import torch
from torchvision.utils import make_grid
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
    compute_similarity,
    forward_render,
    remesh,
    EarlyStopping,
    EarlyStoppingConfig,
    rename_log_files_and_create_video,
)
from image_processing import linear_to_srgb_ldr
from config import GeometryConfig, Config
from scenes import setup_views


def reconstruct_geometry(
    cfg: Config,
    geom_cfg: GeometryConfig,
    logs: dict[str, dict[int, list]],
    wandb_project: str | None = None,
    wandb_experiment_name: str | None = None,
) -> dict[str, Any]:
    """
    Reconstructs geometry based on configuration, logs, and optional integration with
    Weights & Biases (wandb) logging. The function performs a rendering and optimization
    process, enabling updates to a geometry via optimization techniques and latent
    representations, while optionally saving progress for visualization and analysis.

    Parameters:
        cfg (Config): The general configuration object containing model and scene details.
        geom_cfg (GeometryConfig): The geometry configuration specifying optimization and
            rendering parameters.
        logs (dict[str, list]): A dictionary for storing log data across the processing steps.
        wandb_project (str | None, optional): The name of the wandb project for logging rendered
            outputs. Default is None.
        wandb_experiment_name (str | None, optional): The experiment name used in wandb if enabled.
            Default is None.

    Returns:
        dict[str, Any]: A dictionary containing logged data, computed similarities, or trained
            model outputs depending on the operation flow.

    Raises:
        ValueError: If any invalid configuration values are set during execution.
        RuntimeError: If external library calls encounter issues like insufficient resources.
    """
    wb_log = init_wandb_run(wandb_project, wandb_experiment_name, geom_cfg)
    if wb_log:
        import wandb

    is_baseline_run = cfg.model.__str__() == "MAE"
    is_torch = cfg.model.is_torch
    scene = cfg.scene
    if cfg.use_masks:
        # depth > 0 for hits, 0 for background — works for any number of shapes.
        # shape_index is 0-based so index 0 is indistinguishable from background.
        # The depth AOV is appended after the nested integrator's RGB channels,
        # so it sits at the last channel of the output tensor.
        mask_integrator = mi.load_dict({
            'type': 'aov',
            'aovs': 'd:depth'
        })
    scene["emitter"] = (
        dict(type="envmap", filename=cfg.envmap)
        if not isinstance(cfg.envmap, dict)  # constant case
        else cfg.envmap
    )
    lr = geom_cfg.lr

    if is_baseline_run:
        models, baseline_history, rsa_models, rsa_model_names = setup_baseline_models()

    # Factor 2 because we use heldout views to evaluate the scores.
    sensors = setup_views(geom_cfg.n_views, width=cfg.dims[0], height=cfg.dims[1])
    target_scene = mi.load_dict(scene)  # pyright: ignore

    # render reference images
    with dr.suspend_grad():
        ref_images = [
            linear_to_srgb_ldr(
                mi.render(target_scene, sensor=sensors[i], spp=cfg.spp, seed=cfg.seed)
            )
            for i in range(geom_cfg.n_views)
        ]

        if cfg.use_masks:
            ref_masks = [
                mi.render(target_scene, sensor=sensors[i], integrator=mask_integrator)
                for i in range(geom_cfg.n_views)
            ]


        # add 10 deg rotation around y-axis for heldout views
        if 'to_world' in scene['shape']:
            scene['shape']['to_world'] = scene['shape']['to_world'].rotate([0, 1, 0], 10)
        else:
            scene['shape']['to_world'] = mi.ScalarTransform4f.rotate([0, 1, 0], 10)
        heldout_scene = mi.load_dict(scene)
        heldout_views = [
            linear_to_srgb_ldr(
                mi.render(heldout_scene, sensor=sensors[i], spp=cfg.spp, seed=cfg.seed)
            )
            for i in range(geom_cfg.n_views)
        ]

        if cfg.use_masks:
            heldout_masks = [
                mi.render(heldout_scene, sensor=sensors[i], integrator=mask_integrator)
                for i in range(geom_cfg.n_views)
            ]

        init_imgs = torch.stack(
            [img.torch().permute(2, 0, 1).contiguous() for img in ref_images]
        )

        heldout_imgs = torch.stack(
            [img.torch().permute(2, 0, 1).contiguous() for img in heldout_views]
        )

        if cfg.use_masks:
            # binary mask tensors per sensor: (1, 1, H, W) float, used to zero background
            ref_mask_tensors = [
                (m.torch()[..., -1:] > 0).float().permute(2, 0, 1).unsqueeze(0)
                for m in ref_masks
            ]
            heldout_mask_tensors = [
                (m.torch()[..., -1:] > 0).float().permute(2, 0, 1).unsqueeze(0)
                for m in heldout_masks
            ]
            target_mask_grid = make_grid(
                torch.cat(ref_mask_tensors, dim=0), 5
            ).permute(1, 2, 0).cpu().numpy()

        target_grid = make_grid(init_imgs, 5).permute(1, 2, 0).cpu().numpy()
        heldout_grid = make_grid(heldout_imgs, 5).permute(1, 2, 0).cpu().numpy()

        if cfg.baseline_rsa and is_baseline_run:
            baseline_target_latent = compute_baseline_target_latents(
                rsa_models, rsa_model_names, init_imgs, heldout_imgs,
                img_masks=ref_mask_tensors if cfg.use_masks else None,
                heldout_masks=heldout_mask_tensors if cfg.use_masks else None,
            )

    if wb_log:
        static_images = {
            "render/Target": wandb.Image(target_grid),
            "render/Heldout": wandb.Image(heldout_grid),
        }
        if cfg.use_masks:
            heldout_mask_grid = make_grid(
                torch.stack([m.squeeze(0) for m in heldout_mask_tensors]), 5
            ).permute(1, 2, 0).cpu().numpy()
            static_images["render/Target Mask"] = wandb.Image(target_mask_grid)
            static_images["render/Heldout Mask"] = wandb.Image(heldout_mask_grid)
        wb_log.log(static_images)

    if is_torch:
        ref_images = [img.torch() for img in ref_images]

    scene["shape"]["filename"] = "assets/models/ico_10k.ply"
    scene["shape"]["type"] = "ply"

    try:
        scene["shape"].pop("to_world")
    except:
        pass

    scene = mi.load_dict(scene)  # pyright: ignore
    params = mi.traverse(scene)  # pyright: ignore

    es = EarlyStopping(
        EarlyStoppingConfig(
            patience=50,
            min_delta=1e-5,
            mode="min",
            restore_best=True,
        )
    )

    # init large steps
    ls = mi.ad.largesteps.LargeSteps(
        params["shape.vertex_positions"], params["shape.faces"], geom_cfg.lambda_reg
    )

    optimizer = mi.ad.Adam(lr=geom_cfg.lr, uniform=True)
    optimizer["u"] = ls.to_differential(params["shape.vertex_positions"])

    # Should only fail for single runs for LPIPS and VGG.
    try:
        collect_latents = True
        rsa = []
        rsa_heldout = []
        sig = []
        sig_heldout = []
        if cfg.use_masks:
            latents = [
                cfg.model(
                    (img.torch() if not torch.is_tensor(img) else img)
                    .permute(2, 0, 1).unsqueeze(0) * ref_mask_tensors[i]
                ).detach().cpu().flatten().numpy()
                for i, img in enumerate(ref_images)
            ]
            heldout_latents = [
                cfg.model(
                    heldout_imgs[i].unsqueeze(0) * heldout_mask_tensors[i]
                ).detach().cpu().flatten().numpy()
                for i in range(geom_cfg.n_views)
            ]
            target_latents = np.stack(latents)
            heldout_latents = np.stack(heldout_latents)
        else:
            latents = [
                cfg.model(render).detach().cpu().flatten().numpy() for render in ref_images
            ]
            heldout_latents = [
                cfg.model(render).detach().cpu().flatten().numpy() for render in heldout_imgs
            ]
            target_latents = np.stack(latents)
            heldout_latents = np.stack(heldout_latents)
    except:
        collect_latents = False

    for epoch in tqdm(
        range(geom_cfg.epochs), desc="Optimization", total=geom_cfg.epochs, unit="epoch"
    ):
        batch_loss = 0.0
        batch_sim = 0.0
        batch_renders = []
        batch_render_masks = []
        batch_flip = []
        if cfg.compute_forward:
            grad_renders = []
            mag_ldrs = []
            signed_ldrs = []

        remeshing = True if epoch in geom_cfg.remesh else False
        for sensor_idx, sensor in enumerate(sensors):
            params["shape.vertex_positions"] = ls.from_differential(optimizer["u"])
            params.update()
            if cfg.compute_forward:
                with dr.isolate_grad():
                    grad, mag_grad, signed_grad = forward_render(
                        scene,
                        params,
                        "shape.vertex_positions",
                        sensor,
                        new_param=ls.from_differential(optimizer["u"]),
                    )
                grad_renders.append(grad)
                mag_ldrs.append(mag_grad)
                signed_ldrs.append(signed_grad)

            target = ref_images[sensor_idx]
            heldout = heldout_imgs[sensor_idx]

            render = mi.render(
                scene, params, sensor=sensor, spp=cfg.spp, seed=cfg.seed * sensor_idx
            )

            if cfg.use_masks:
                with dr.suspend_grad():
                    mask = mi.render(
                        scene, params, sensor, integrator=mask_integrator
                    )
                render_mask_tensor = (mask.torch()[..., -1:] > 0).float().permute(2, 0, 1).unsqueeze(0)
                batch_render_masks.append(render_mask_tensor.squeeze(0))
            else:
                render_mask_tensor = None

            render = linear_to_srgb_ldr(render)
            render_torch = render.torch().permute(2, 0, 1).contiguous()
            batch_renders.append(render_torch)
            loss = cfg.model.lossfn(render, target)
            if is_baseline_run:
                target = target.torch()

            flip_err_map, flip_err = compute_flip_error(render, target)
            batch_flip.append(flip_err_map)

            logs["flip"][sensor_idx].append(float(flip_err))
            logs["loss"][sensor_idx].append(float(loss.torch().item()))

            dr.backward(loss)
            optimizer.step()

            if is_baseline_run:
                record_baseline_metrics(
                    models, render, target, heldout, cfg, sensor_idx, baseline_history,
                    render_mask=render_mask_tensor,
                    target_mask=ref_mask_tensors[sensor_idx] if cfg.use_masks else None,
                    heldout_mask=heldout_mask_tensors[sensor_idx] if cfg.use_masks else None,
                )

            if collect_latents:
                render_chw = render.torch().permute(2, 0, 1).unsqueeze(0)
                if cfg.use_masks and render_mask_tensor is not None:
                    render_chw = render_chw * render_mask_tensor
                else:
                    latents.append(cfg.model(render_chw).detach().cpu().flatten().numpy())

            batch_loss += float(loss.torch().item())

            if not is_baseline_run:
                sim = compute_similarity(
                    render, target, cfg.model, shape=(1, 3, *cfg.dims),
                    render_mask=render_mask_tensor,
                    target_mask=ref_mask_tensors[sensor_idx] if cfg.use_masks else None,
                )  # pyright: ignore
                logs["cosine"][sensor_idx].append(sim)
                batch_sim += sim
                sim = compute_similarity(
                    render, heldout, cfg.model, shape=(1, 3, *cfg.dims),
                    render_mask=render_mask_tensor,
                    target_mask=heldout_mask_tensors[sensor_idx] if cfg.use_masks else None,
                )  # pyright: ignore
                logs['heldout/cosine'][sensor_idx].append(sim)

        if collect_latents:
            current_latents = np.stack([
                cfg.model(view * view_mask).detach().cpu().flatten().numpy()
                for view, view_mask in zip(batch_renders, batch_render_masks)
            ])
            rsa_log, correlation, significance, heldout_corr, heldout_sig = compute_latent_rsa(
                cfg.model, current_latents, target_latents, heldout_latents, geom_cfg.n_views, logs
            )
            rsa.append(float(correlation))
            sig.append(float(significance))
            rsa_heldout.append(float(heldout_corr))
            sig_heldout.append(float(heldout_sig))

        if cfg.baseline_rsa and is_baseline_run:
            compute_baseline_rsa(
                rsa_models, rsa_model_names, batch_renders, baseline_target_latent, baseline_history
            )

        # remesh
        if remeshing:
            vertices, faces = remesh(params, "shape.vertex_positions", "shape.faces")
            params["shape.vertex_positions"] = mi.Float(
                vertices.flatten().astype(np.float32)
            )
            params["shape.faces"] = mi.UInt(faces.flatten())
            params.update()
            ls = mi.ad.largesteps.LargeSteps(
                params["shape.vertex_positions"],
                params["shape.faces"],
                geom_cfg.lambda_reg,
            )
            lr *= 8e-1
            optimizer = mi.ad.Adam(lr=lr, uniform=True)
            optimizer["u"] = ls.to_differential(params["shape.vertex_positions"])

        # Epoch End
        image_grid, flip_grid = build_render_flip_grids(batch_renders, batch_flip)

        tqdm.write(
            f"Epoch {epoch + 1} – Loss: {batch_loss / geom_cfg.n_views:.6f}, Similarity: {batch_sim / geom_cfg.n_views:.6f}"
        )

        if wb_log:
            # tile
            if cfg.compute_forward:
                grid_grad = make_grid(grad_renders, nrow=5)
                grid_ldr = make_grid(
                    torch.stack(mag_ldrs), nrow=5
                )  # (3, H_grid, W_grid)
                grid_signed_ldr = make_grid(
                    torch.stack(signed_ldrs), nrow=5
                )  # (3, H_grid, W_grid)
                grad_log = {
                    "grad/Image": wandb.Image(grid_grad.permute(1, 2, 0).cpu().numpy()),
                    "grad/Magnitude LDR": wandb.Image(
                        grid_ldr.permute(1, 2, 0).cpu().numpy()
                    ),
                    "grad/Signed LDR": wandb.Image(
                        grid_signed_ldr.permute(1, 2, 0).cpu().numpy()
                    ),
                }

            # -> HWC for wandb.Image
            images = {
                "render/Step": wandb.Image(image_grid),
                "render/FLIP Error": wandb.Image(flip_grid),
                "Epoch": epoch,
            }
            if cfg.use_masks and batch_render_masks:
                render_mask_grid = make_grid(
                    torch.stack(batch_render_masks), 5
                ).permute(1, 2, 0).cpu().numpy()
                images["render/Mask"] = wandb.Image(render_mask_grid)
            vals = dict()
            for k, v in logs.items():
                for kk, vv in v.items():
                    vals[f"{k}/view_{kk}"] = vv[-1]

            vals.update(images)
            if collect_latents:
                vals.update(rsa_log)
            if cfg.compute_forward:
                vals.update(grad_log)

            wb_log.log(vals)

        plt.close("all")

        should_stop = es.step(
            value=batch_loss / geom_cfg.n_views,
            optimizer=optimizer,
            params=params,
            epoch=epoch,
        )

        if should_stop:
            report_early_stop(es, epoch)
            if wb_log:
                wb_log.log({"Best epoch": es.best_epoch})
            break

    if wb_log:
        save_wandb_artifacts(
            wb_log, collect_latents, (rsa, sig, rsa_heldout, sig_heldout),
            is_baseline_run, baseline_history,
        )
        rename_log_files_and_create_video(wb_log, wandb_experiment_name, seed=None)
        wb_log.finish()
    return logs
