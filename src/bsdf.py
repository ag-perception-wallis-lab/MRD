import pickle
from typing import Any

from plot import pixel_error, plot_rdm, plot_rsa_scatter

from config import Config
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import drjit as dr
import torch
from tqdm import tqdm
from torchvision.transforms import CenterCrop
from torchvision.utils import make_grid
from image_processing import linear_to_srgb_ldr
from utils import (
    EarlyStopping,
    EarlyStoppingConfig,
    compute_rdm,
    compute_rsa_similarity,
    compute_similarity,
    load_all_models,
    log_bsdf_parameters,
    rename_log_files_and_create_video,
    apply_new_lr,
    create_local_output_dir,
    save_progress_images,
    create_videos_from_local_images,
)
from config import BSDFConfig


def reconstruct_bsdf(
    cfg: Config,
    bsdf_cfg: BSDFConfig,
    logs: dict[str, list],
    wandb_project: str | None = None,
    wandb_experiment_name: str | None = None,
) -> dict[str, Any]:
    """
    Reconstructs a principled BSDF based on configuration, logs, and optional integration with
    Weights & Biases (wandb) logging. The function performs a rendering and optimization
    process, enabling updates to a geometry via optimization techniques and latent
    representations, while optionally saving progress for visualization and analysis.

    Parameters:
        cfg (Config): The general configuration object containing model and scene details.
        bsdf_cfg (BSDFConfig): The BSDF configuration specifying optimization and
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
    if wandb_project and wandb_experiment_name:
        import wandb

        config = bsdf_cfg
        wb_log = wandb.init(
            name=wandb_experiment_name, project=wandb_project, config=config
        )
        output_dir = None
    else:
        wb_log = None
        # Create local output directory for saving progress images
        # Use the experiment name that was passed in (includes material-model)
        output_dir = create_local_output_dir(
            wandb_experiment_name, seed=cfg.seed
        )

    is_baseline_run = cfg.model.__str__() == "DualBuffer"
    is_torch = cfg.model.is_torch
    has_cuda = mi.variant().startswith("cuda")
    scene_dict = cfg.scene
    lr = bsdf_cfg.lr
    dims = cfg.dims
    scene_dict["film"]["width"] = dims[0]
    scene_dict["film"]["height"] = dims[1]

    if is_baseline_run:
        models = load_all_models()
        similarities = {
            str(k): {m.__str__(): [] for m in models} for k in range(bsdf_cfg.n_views)
        }

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

    init_imgs = (
        torch.stack([img.torch().permute(2, 0, 1).contiguous() for img in ref_images])
        .detach()
        .cpu()
    )
    target_grid = make_grid(init_imgs, 4).permute(1, 2, 0).cpu().numpy()

    if wb_log:
        wb_log.log({"render/Target": wandb.Image(target_grid)})

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

    # Fit PCA over all latent representations. This requires the model to produce a latent representation.
    try:
        collect_latents = True
        latents = [
            cfg.model(render).flatten().detach().cpu().numpy() for render in ref_images
        ]
        target_latents = np.stack(latents[: len(scene.sensors())])
        rsa = []
        sig = []
    except:  # noqa: E722
        collect_latents = False

    # mem = 0
    for epoch in tqdm(
        range(bsdf_cfg.epochs), desc="Optimization", total=bsdf_cfg.epochs, unit="epoch"
    ):
        batch_loss = 0.0
        batch_sim = 0.0
        batch_renders = []

        for sensor_idx, sensor in enumerate(scene.sensors()):
            target = ref_images[sensor_idx]
            # update and clip parameters
            for k in optimizer.keys():
                if k.endswith("eta"):
                    optimizer[k] = dr.clip(optimizer[k], 0.001, 4.1)
                    continue
                # if k.endswith("spec_trans.value"):
                #     optimizer[k] = dr.clip(optimizer[k], 0.11, 0.999)
                #     continue
                # Apply denoiser to texture, so we can keep sample size low
                # if k.endswith(".data"):
                #     optimizer[k] = tex_denoiser(optimizer[k])

                optimizer[k] = dr.clip(optimizer[k], 1e-3, 1.0)

            params.update(optimizer)
            render = mi.render(
                scene, params, sensor=sensor, spp=cfg.spp, seed=cfg.seed * sensor_idx
            )
            render = linear_to_srgb_ldr(render)
            batch_renders.append(
                render.torch().permute(2, 0, 1).contiguous().detach().cpu()
            )
            if collect_latents:
                latents.append(cfg.model(render).detach().cpu().flatten().numpy())

            if str(cfg.model) == "DualBuffer":
                other_render = mi.render(
                    scene,
                    params,
                    sensor=sensor,
                    spp=cfg.spp,
                    seed=cfg.seed * sensor_idx + 1,
                )
                other_render = linear_to_srgb_ldr(other_render)
                loss = cfg.model.lossfn(render, other_render, target)
            else:
                loss = cfg.model.lossfn(render, target)

            if is_baseline_run:
                with torch.no_grad():
                    for model in models:
                        crop = (
                            CenterCrop(224)
                            if model.__class__.__name__ in ["DINO", "CLIPVision"]
                            else None
                        )
                        sim = compute_similarity(
                            render, target, model, shape=(1, 3, *cfg.dims), crop=crop
                        )
                        similarities[str(sensor_idx)][model.__str__()].append(sim)

            dr.backward(loss)
            batch_loss += loss.torch().detach().cpu().item()

            # we can only compute similarity for model latent representations
            # therefore we do not compute it for baselines (mean absolute) and
            # LPIPS (is a similarity measure itself).
            if str(cfg.model) != "DualBuffer":
                sim = compute_similarity(
                    render, target, cfg.model, shape=(1, 3, *cfg.dims)
                )  # pyright: ignore
                batch_sim += sim

        optimizer.step()

        if collect_latents:
            current_latents = np.stack(latents[len(scene.sensors()) * (epoch + 1) :])

            rdm_latent = compute_rdm(current_latents)
            rdm_target = compute_rdm(target_latents)
            correlation, significance = compute_rsa_similarity(rdm_latent, rdm_target)
            rsa.append(float(correlation))
            sig.append(float(significance))
            rdm_x_fig = plot_rdm(rdm_latent)
            rdm_y_fig = plot_rdm(rdm_target)
            rsa_fig = plot_rsa_scatter(rdm_latent, rdm_target)
            rsa_log = {
                "RSA/RDM Latent": rdm_x_fig,
                "RSA/RDM Target": rdm_y_fig,
                "RSA/RSA": rsa_fig,
                "RSA/Correlation": correlation,
                "RSA/Significance": significance,
            }

        # Epoch End
        # Warmup
        if epoch != 0 and str(cfg.model) == "DualBuffer":
            apply_new_lr(optimizer, epoch)

        batch_renders = torch.stack(batch_renders).detach().cpu()
        image_grid = make_grid(batch_renders, 5, normalize=True, value_range=(0, 1))
        image_grid = image_grid.permute(1, 2, 0).cpu().numpy()

        tqdm.write(
            f"Epoch {epoch + 1} – Loss: {batch_loss / 4:.6f}, Similarity: {batch_sim / 4:.6f}"
        )
        bsdf_params = log_bsdf_parameters(optimizer)

        logs["loss"].append(batch_loss / bsdf_cfg.n_views)
        logs["similarity"].append(batch_sim / bsdf_cfg.n_views)

        # Prepare pixel error visualization for both wandb and local saving
        pixel_diffs = []
        for i, rndr in enumerate(batch_renders):
            s = rndr.cpu().permute(1, 2, 0).numpy()  # HWC (RGB)
            t = init_imgs[i, ...].cpu().permute(1, 2, 0).numpy()  # HWC (RGB)
            err_np = pixel_error(s, t)  # HWC, uint8 (RGB)
            pixel_diffs.append(torch.from_numpy(err_np).detach().cpu())  # HWC uint8

        # -> NCHW for make_grid
        errs_nchw = (
            torch.stack([img.permute(2, 0, 1) for img in pixel_diffs])
            .detach()
            .cpu()
        )
        # (N,3,H,W)
        grid_chw = make_grid(errs_nchw, nrow=4)  # (3, H_grid, W_grid)
        diffs = grid_chw.permute(1, 2, 0).cpu().numpy()  # (H_grid, W_grid, 3)

        if wb_log:
            images = {
                "render/Step": wandb.Image(image_grid),
                "render/Pixel Error": wandb.Image(diffs),
                "Epoch": epoch,
            }
            vals = {k: v[-1] for k, v in logs.items()}
            vals.update(images)
            vals.update(bsdf_params)
            if collect_latents:
                vals.update(rsa_log)

            wb_log.log(vals)
            plt.close("all")
        else:
            # Save images locally when wandb is not enabled
            save_progress_images(
                output_dir=output_dir,
                epoch=epoch,
                image_grid=image_grid,
                pixel_error_grid=diffs,
            )

        should_stop = es.step(
            value=batch_loss / bsdf_cfg.n_views,
            optimizer=optimizer,
            params=params,
            epoch=epoch,
        )

        if has_cuda:
            torch.cuda.empty_cache()

        if should_stop:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best @ epoch {es.best_epoch}, best={es.best:.6f})."
            )
            break

    if wb_log:
        if collect_latents:
            pca = PCA(n_components=3)
            latents = np.stack(latents)
            pca.fit(np.stack(latents))

            print("[PCA] EV Ratio: ", pca.explained_variance_ratio_)
            Z = pca.transform(latents[bsdf_cfg.n_views :, ...])
            Z_target = pca.transform(latents[: bsdf_cfg.n_views, ...])
            with open(f"{wb_log.dir}/pca_latent.npy", "wb+") as f:
                np.save(f, Z)
            with open(f"{wb_log.dir}/pca_target.npy", "wb+") as f:
                np.save(f, Z_target)
            with open(f"{wb_log.dir}/rsa.npy", "wb+") as f:
                pickle.dump(rsa, f)
            with open(f"{wb_log.dir}/rsa-sig.npy", "wb+") as f:
                pickle.dump(sig, f)

        # Write similarities to file. This is only done for baseline runs to compute the ECDF.
        if is_baseline_run:
            with open(f"{wb_log.dir}/similarities.npy", "wb+") as f:
                pickle.dump(similarities, f)
            with open(f"{wb_log.dir}/loss.npy", "wb+") as f:
                pickle.dump(logs["loss"], f)
            with open(f"{wb_log.dir}/hypersphere.npy", "wb+") as f:
                pickle.dump(logs["similarity"], f)

        rename_log_files_and_create_video(wb_log, wandb_experiment_name, seed=None)
        wb_log.finish()
    else:
        # Create videos from locally saved images
        print(f"Creating videos from saved images in {output_dir}...")
        create_videos_from_local_images(output_dir, fps=24, subdirs={"render": ["Step", "Pixel"]})

        # Save final data
        with open(f"{output_dir}/loss.npy", "wb+") as f:
            pickle.dump(logs["loss"], f)
        with open(f"{output_dir}/hypersphere.npy", "wb+") as f:
            pickle.dump(logs["similarity"], f)

        if collect_latents:
            pca = PCA(n_components=3)
            latents = np.stack(latents)
            pca.fit(np.stack(latents))

            print("[PCA] EV Ratio: ", pca.explained_variance_ratio_)
            Z = pca.transform(latents[bsdf_cfg.n_views :, ...])
            Z_target = pca.transform(latents[: bsdf_cfg.n_views, ...])
            with open(f"{output_dir}/pca_latent.npy", "wb+") as f:
                np.save(f, Z)
            with open(f"{output_dir}/pca_target.npy", "wb+") as f:
                np.save(f, Z_target)
            with open(f"{output_dir}/rsa.npy", "wb+") as f:
                pickle.dump(rsa, f)
            with open(f"{output_dir}/rsa-sig.npy", "wb+") as f:
                pickle.dump(sig, f)

        if is_baseline_run:
            with open(f"{output_dir}/similarities.npy", "wb+") as f:
                pickle.dump(similarities, f)

        print(f"Results saved to: {output_dir}")

    return logs
