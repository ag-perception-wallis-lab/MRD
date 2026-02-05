import pickle
from typing import Any
import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from sklearn.decomposition import PCA
from torchvision.transforms import CenterCrop
from tqdm import tqdm
import torch
from torchvision.utils import make_grid
from model import DEVICE, MAE, LaplacianLoss, EdgeLengthLoss, TriangleAreaLoss, ARAPLoss
from plot import pixel_error, plot_rdm, plot_rsa_scatter
from utils import (
    compute_rdm,
    compute_rsa_similarity,
    compute_similarity,
    forward_render,
    get_label,
    load_all_models,
    remesh,
    EarlyStopping,
    EarlyStoppingConfig,
    rename_log_files_and_create_video,
    create_local_output_dir,
    save_progress_images,
    create_videos_from_local_images,
)
from image_processing import linear_to_srgb_ldr
from config import GeometryConfig, Config
from scenes import setup_views


def reconstruct_geometry(
        cfg: Config,
        geom_cfg: GeometryConfig,
        logs: dict[str, list],
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
    if wandb_project and wandb_experiment_name:
        import wandb

        # INFO: general config to store
        config = geom_cfg
        wb_log = wandb.init(
            name=wandb_experiment_name, project=wandb_project, config=config
        )
        output_dir = None
    else:
        wb_log = None
        # Create local output directory for saving progress images
        # Use the experiment name that was passed in (includes scene-model-envmap)
        output_dir = create_local_output_dir(
            wandb_experiment_name, seed=cfg.seed
        )

    is_baseline_run = cfg.model.__str__() == "MAE"
    is_torch = cfg.model.is_torch
    has_cuda = mi.variant().startswith("cuda")
    scene = cfg.scene
    scene["emitter"] = (
        dict(type="envmap", filename=cfg.envmap)
        if not isinstance(cfg.envmap, dict)  # constant case
        else cfg.envmap
    )
    lr = geom_cfg.lr

    if is_baseline_run:
        # More or less the question if we have enough VRAM
        models = load_all_models()
        similarities = {str(k): {m.__str__(): [] for m in models} for k in range(geom_cfg.n_views)}

    if has_cuda:
        denoiser = mi.OptixDenoiser(cfg.dims)

    sensors = setup_views(geom_cfg.n_views, width=cfg.dims[0], height=cfg.dims[1])
    target_scene = mi.load_dict(scene)  # pyright: ignore

    # render reference images
    if has_cuda:
        ref_images = [
            linear_to_srgb_ldr(
                mi.render(target_scene, sensor=sensors[i], spp=cfg.spp, seed=cfg.seed)
            )
            for i in range(geom_cfg.n_views)
        ]
    else:
        ref_images = [
            mi.render(target_scene, sensor=sensors[i], spp=cfg.spp, seed=cfg.seed)
            for i in range(geom_cfg.n_views)
        ]

    init_imgs = torch.stack(
        [img.torch().permute(2, 0, 1).contiguous() for img in ref_images]
    )
    target_grid = make_grid(init_imgs, 5).permute(1, 2, 0).cpu().numpy()
    if wb_log:
        wb_log.log({"render/Target": wandb.Image(target_grid)})

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
    total_renders = []

    # Initialize geometric regularization losses
    geom_losses = {}
    if geom_cfg.lambda_lap > 0:
        geom_losses["laplacian"] = LaplacianLoss()
        tqdm.write(f"Using Laplacian loss with λ={geom_cfg.lambda_lap}")
    if geom_cfg.lambda_edge > 0:
        geom_losses["edge"] = EdgeLengthLoss()
        tqdm.write(f"Using Edge Length loss with λ={geom_cfg.lambda_edge}")
    if geom_cfg.lambda_area > 0:
        geom_losses["area"] = TriangleAreaLoss()
        tqdm.write(f"Using Triangle Area loss with λ={geom_cfg.lambda_area}")
    if geom_cfg.lambda_arap > 0:
        geom_losses["arap"] = ARAPLoss()
        # Initialize ARAP with initial vertex positions
        geom_losses["arap"].initialize(
            params["shape.vertex_positions"], params["shape.faces"]
        )
        tqdm.write(f"Using ARAP loss with λ={geom_cfg.lambda_arap}")

    # Fit PCA over all latent representations. This requires the model to produce a latent representation.
    try:
        collect_latents = True
        latents = [
            cfg.model(render).detach().cpu().flatten().numpy() for render in ref_images
        ]
        target_latents = np.stack(latents[: geom_cfg.n_views])
        rsa = []
        sig = []
    except:
        collect_latents = False

    for epoch in tqdm(
            range(geom_cfg.epochs), desc="Optimization", total=geom_cfg.epochs, unit="epoch"
    ):
        batch_loss = 0.0
        batch_sim = 0.0
        batch_renders = []
        if cfg.compute_forward:
            grad_renders = []
            mag_ldrs = []
            signed_ldrs = []

        remeshing = True if epoch in geom_cfg.remesh else False

        for sensor_idx, sensor in enumerate(sensors):
            # visualize gradient
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
            params["shape.vertex_positions"] = ls.from_differential(optimizer["u"])
            params.update()

            render = mi.render(
                scene, params, sensor=sensor, spp=cfg.spp, seed=cfg.seed * sensor_idx
            )
            render = linear_to_srgb_ldr(render)
            batch_renders.append(render.torch().permute(2, 0, 1).contiguous())
            loss = cfg.model.lossfn(render, target)

            # Run inference for each model and compute similarity
            if is_baseline_run:
                with torch.no_grad():
                    for model in models:
                        crop = CenterCrop(224) if model.__class__.__name__ in ['DINO', 'CLIPVision'] else None
                        sim = compute_similarity(
                            render, target, model, shape=(1, 3, *cfg.dims), crop=crop
                        )  # pyright: ignore
                        similarities[str(sensor_idx)][model.__str__()].append(sim)

            # Add geometric regularization losses
            if geom_losses:
                verts = params["shape.vertex_positions"]
                faces = params["shape.faces"]

                if "laplacian" in geom_losses:
                    lap_loss = geom_losses["laplacian"].lossfn(verts, faces)
                    loss = loss + geom_cfg.lambda_lap * lap_loss

                if "edge" in geom_losses:
                    edge_loss = geom_losses["edge"].lossfn(verts, faces)
                    loss = loss + geom_cfg.lambda_edge * edge_loss

                if "area" in geom_losses:
                    area_loss = geom_losses["area"].lossfn(verts, faces)
                    loss = loss + geom_cfg.lambda_area * area_loss

                if "arap" in geom_losses:
                    arap_loss = geom_losses["arap"].lossfn(verts)
                    loss = loss + geom_cfg.lambda_arap * arap_loss

            # classify
            if cfg.model.is_imagenet and wb_log and cfg.classify:
                label = torch.tensor([geom_cfg.class_idx], device=DEVICE)
                # introduce a weighting factor for the right class label
                class_weights = torch.ones(1000, device=DEVICE) / 100
                class_weights[geom_cfg.class_idx] = (
                        class_weights[geom_cfg.class_idx] * 99
                )
                pred_loss, probs = cfg.model.classify(render, label, class_weights)
                probs = probs.torch().detach()
                with torch.no_grad():
                    top_probs, top_idxs = probs.topk(5, dim=-1)
                    top_probs = top_probs.squeeze().tolist()
                    top_idxs = top_idxs.squeeze().tolist()
                    top_labels = [get_label(i) for i in top_idxs]

                table = wandb.Table(columns=["label", "probability"])
                for label, probability in zip(top_labels, top_probs):
                    table.add_data(label, probability)

                barplot = wandb.plot.bar(
                    table=table,
                    label="label",
                    value="probability",
                    title="Top-5 classification",
                )
                # fig, ax = plt.subplots(figsize=(5, 3))
                # ax.bar(range(len(top_labels)), top_probs)
                # ax.set_xticks(range(len(top_labels)))
                # ax.set_xticklabels(top_labels)  # , rotation=45, ha="right")
                # ax.set_ylabel("Probability")
                # ax.set_title("Top-5 probabilities")  # weighted loss components

                reconstruction_loss = loss
                loss = ((1 - geom_cfg.alpha) * pred_loss) + geom_cfg.alpha * loss
                pred_log = {
                    "Prediction/Probabilities": barplot,
                    "Prediction/Loss": pred_loss.torch().item(),
                    "Prediction/Reconstruction Loss": reconstruction_loss.torch().item(),
                }

            dr.backward(loss)
            optimizer.step()

            if collect_latents:
                latents.append(cfg.model(render).detach().cpu().flatten().numpy())

            batch_loss += loss.torch().item()

            # we can only compute similarity for model latent representations
            # therefore we do not compute it for baselines (mean absolute) and
            # LPIPS (is a similarity measure itself).
            if not isinstance(cfg.model, MAE):
                sim = compute_similarity(
                    render, target, cfg.model, shape=(1, 3, *cfg.dims)
                )  # pyright: ignore
                batch_sim += sim

        if collect_latents:
            current_latents = np.stack(latents[geom_cfg.n_views * (epoch + 1):])

            rdm_latent = compute_rdm(current_latents)
            rdm_target = compute_rdm(target_latents)
            correlation, significance = compute_rsa_similarity(rdm_latent, rdm_target)
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
            rsa.append(float(correlation))
            sig.append(float(significance))

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

            # Reinitialize geometric losses after remeshing (topology changed)
            if "laplacian" in geom_losses:
                geom_losses["laplacian"] = LaplacianLoss()
            if "edge" in geom_losses:
                geom_losses["edge"] = EdgeLengthLoss()
            if "area" in geom_losses:
                geom_losses["area"] = TriangleAreaLoss()
            if "arap" in geom_losses:
                geom_losses["arap"] = ARAPLoss()
                geom_losses["arap"].initialize(
                    params["shape.vertex_positions"], params["shape.faces"]
                )

        # Epoch End
        batch_renders = torch.stack(batch_renders)
        image_grid = make_grid(batch_renders, 5, normalize=True, value_range=(0, 1))
        image_grid = image_grid.permute(1, 2, 0).cpu().numpy()
        total_renders.append(image_grid)

        tqdm.write(
            f"Epoch {epoch + 1} – Loss: {batch_loss / geom_cfg.n_views:.6f}, Similarity: {batch_sim / geom_cfg.n_views:.6f}"
        )

        logs["loss"].append(batch_loss / geom_cfg.n_views)
        logs["similarity"].append(batch_sim / geom_cfg.n_views)

        # Prepare pixel error visualization for both wandb and local saving
        pixel_diffs = []
        for i, rndr in enumerate(batch_renders):
            s = rndr.cpu().permute(1, 2, 0).numpy()  # HWC (RGB)
            t = init_imgs[i, ...].cpu().permute(1, 2, 0).numpy()  # HWC (RGB)
            err_np = pixel_error(s, t)  # HWC, uint8 (RGB)
            pixel_diffs.append(torch.from_numpy(err_np))  # HWC uint8

        # -> NCHW for make_grid
        errs_nchw = torch.stack(
            [img.permute(2, 0, 1) for img in pixel_diffs]
        )  # (N,3,H,W)
        grid_chw = make_grid(errs_nchw, nrow=5)  # (3, H_grid, W_grid)
        diffs = grid_chw.permute(1, 2, 0).cpu().numpy()  # (H_grid, W_grid, 3)

        # Process gradient visualizations (for both wandb and local saving)
        if cfg.compute_forward:
            grid_grad = make_grid(grad_renders, nrow=5)
            grid_ldr = make_grid(
                torch.stack(mag_ldrs), nrow=5
            )  # (3, H_grid, W_grid)
            grid_signed_ldr = make_grid(
                torch.stack(signed_ldrs), nrow=5
            )  # (3, H_grid, W_grid)

        if wb_log:
            if cfg.compute_forward:
                grad_log = {
                    "grad/Image": wandb.Image(grid_grad.permute(1, 2, 0).cpu().numpy()),
                    "grad/Magnitude LDR": wandb.Image(
                        grid_ldr.permute(1, 2, 0).cpu().numpy()
                    ),
                    "grad/Signed LDR": wandb.Image(
                        grid_signed_ldr.permute(1, 2, 0).cpu().numpy()
                    ),
                }

            images = {
                "render/Step": wandb.Image(image_grid),
                "render/Pixel Error": wandb.Image(diffs),
                "Epoch": epoch,
            }
            vals = {k: v[-1] for k, v in logs.items()}
            vals.update(images)
            if collect_latents:
                vals.update(rsa_log)

            if cfg.model.is_imagenet and cfg.classify and wb_log:
                vals.update(pred_log)
            if cfg.compute_forward:
                vals.update(grad_log)

            wb_log.log(vals)
            plt.close("all")
        else:
            # Save images locally when wandb is not enabled
            grad_images = None
            if cfg.compute_forward:
                grad_images = {
                    "Image": grid_grad.permute(1, 2, 0).cpu().numpy(),
                    "Magnitude_LDR": grid_ldr.permute(1, 2, 0).cpu().numpy(),
                    "Signed_LDR": grid_signed_ldr.permute(1, 2, 0).cpu().numpy(),
                }

            save_progress_images(
                output_dir=output_dir,
                epoch=epoch,
                image_grid=image_grid,
                pixel_error_grid=diffs,
                grad_images=grad_images,
            )

        should_stop = es.step(
            value=batch_loss / geom_cfg.n_views,
            optimizer=optimizer,
            params=params,
            epoch=epoch,
        )

        if should_stop:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best @ epoch {es.best_epoch}, best={es.best:.6f})."
            )
            wb_log.log({'Best epoch': es.best_epoch})
            break

    # write optimized mesh
    # logs["scene"] = scene.shapes()[0].write_ply(f"{cfg.path}/optimized_mesh.ply")


    if wb_log:
        if collect_latents:
            pca = PCA(n_components=3)
            latents = np.stack(latents)
            pca.fit(np.stack(latents))

            print("[PCA] EV Ratio: ", pca.explained_variance_ratio_)
            Z = pca.transform(latents[geom_cfg.n_views:, ...])
            Z_target = pca.transform(latents[: geom_cfg.n_views, ...])
            with open(f"{wb_log.dir}/pca_latent.npy", "wb+") as f:
                np.save(f, Z)
            with open(f"{wb_log.dir}/pca_target.npy", "wb+") as f:
                np.save(f, Z_target)
            with open(f'{wb_log.dir}/rsa.npy', 'wb+') as f:
                pickle.dump(rsa, f)
            with open(f'{wb_log.dir}/rsa-sig.npy', 'wb+') as f:
                pickle.dump(sig, f)

        # Write similarities to file. This is only done for baseline runs to compute the ECDF.
        if is_baseline_run:
            with open(f'{wb_log.dir}/similarities.npy', 'wb+') as f:
                pickle.dump(similarities, f)
            with open(f'{wb_log.dir}/loss.npy', 'wb+') as f:
                pickle.dump(logs['loss'], f)
            with open(f'{wb_log.dir}/hypersphere.npy', 'wb+') as f:
                pickle.dump(logs["similarity"], f)

        rename_log_files_and_create_video(wb_log, wandb_experiment_name, seed=None)
        wb_log.finish()
    else:
        # Create videos from locally saved images
        print(f"Creating videos from saved images in {output_dir}...")
        subdirs = {"render": ["Step", "Pixel"]}
        if cfg.compute_forward:
            subdirs["grad"] = ["Image", "Magnitude_LDR", "Signed_LDR"]
        create_videos_from_local_images(output_dir, fps=24, subdirs=subdirs)

        # Save final data
        with open(f"{output_dir}/loss.npy", "wb+") as f:
            pickle.dump(logs['loss'], f)
        with open(f"{output_dir}/similarity.npy", "wb+") as f:
            pickle.dump(logs["similarity"], f)

        if collect_latents:
            pca = PCA(n_components=3)
            latents = np.stack(latents)
            pca.fit(np.stack(latents))

            print("[PCA] EV Ratio: ", pca.explained_variance_ratio_)
            Z = pca.transform(latents[geom_cfg.n_views:, ...])
            Z_target = pca.transform(latents[: geom_cfg.n_views, ...])
            with open(f"{output_dir}/pca_latent.npy", "wb+") as f:
                np.save(f, Z)
            with open(f"{output_dir}/pca_target.npy", "wb+") as f:
                np.save(f, Z_target)
            with open(f'{output_dir}/rsa.npy', 'wb+') as f:
                pickle.dump(rsa, f)
            with open(f'{output_dir}/rsa-sig.npy', 'wb+') as f:
                pickle.dump(sig, f)

        if is_baseline_run:
            with open(f'{output_dir}/similarities.npy', 'wb+') as f:
                pickle.dump(similarities, f)

        print(f"Results saved to: {output_dir}")

    return logs
