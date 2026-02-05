"""This module introduces the different objectives and models used.
* MAE.lossfn(x,y) → pixel L1 mean.
* DualBuffer.lossfn(x,x_other,y) → mean((x−y)*(x_other−y)).
* Resnet.lossfn(x,y) → L1(f_resnet(x), f_resnet(y)).
* ResnetSIN.lossfn(x,y) → L1(f_resnet_sin(x), f_resnet_sin(y)).
* DINO.lossfn(x,y) → mean(1 − cosine(f_dino(x), f_dino(y))) (dense patches by default).
* CLIP.lossfn(x, target_embed[,prompt]) → L1(clip_img_embed(x[,prompt]), target_embed).
* LPIPS.lossfn(x,y) → LPIPS(x,y).
* VGG.lossfn(x,y) → MSE(VGG(x), VGG(y)).
"""

from abc import ABC, abstractmethod
from enum import Enum
from math import sqrt

import numpy as np
from transformers import (
    AutoModel,
    CLIPVisionModel,
)

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import drjit as dr
from modelvshuman.models.pytorch.model_zoo import (
    resnet50_trained_on_SIN,
)
import torch
from torch import nn
from torch.nn import functional as F
import torchvision as tv
from torchvision.models import (
    vgg19,
    VGG19_Weights,
    resnet50,
    ResNet50_Weights,
)
from lpips import LPIPS as LPIPSModel

from utils import (
    spearman_correlation,
    pearson_correlation,
    to_minus1_1,
    flatten_nobatch,
    diag_gauss_skl,
    diag_gauss_w2,
)

import mitsuba as mi

mi.set_variant("cuda_ad_rgb" if torch.cuda.is_available() else "llvm_ad_rgb")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ModelMixin(ABC):
    is_torch: bool = True
    is_imagenet: bool = False

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError('Implement this in your own model.')

    @abstractmethod
    def lossfn(self, render: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Implement this in your own model.')

    @abstractmethod
    def __call__(self, render: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Implement this in your own model.')

    @torch.no_grad()
    def spearman(self, render, target, ties: bool = False):
        latent = self(render)
        return spearman_correlation(latent, target, ties)

    @torch.no_grad()
    def pearson(self, render, target):
        latent = self(render)
        return pearson_correlation(latent, target)

    @torch.no_grad()
    def l2(self, render: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.norm(self(render) - target, p=2)


class DualBuffer(ModelMixin):
    """
    Loss Function mentioned in:
    Reconstructing Translucent Objects Using Differentiable Rendering, Deng et al.
    """

    is_torch: bool = False

    def __str__(self):
        return "DualBuffer"

    def __call__(
        self,
        render: mi.TensorXf,
        other_render: mi.TensorXf,
        target: mi.TensorXf,  # pyright: ignore
    ) -> mi.TensorXf:
        return dr.mean((render - target) * (other_render - target))

    def lossfn(
        self, render: mi.TensorXf, other_render: mi.TensorXf, target: mi.TensorXf
    ) -> mi.TensorXf:
        return self(render, other_render, target)


class MAE(ModelMixin):
    is_torch: bool = False

    def __str__(self):
        return "MAE"

    def __call__(self, render: torch.Tensor, target: torch.Tensor):
        return dr.mean(dr.abs(render - target))

    def lossfn(self, render, target):
        return self.__call__(render, target)


class VGGLoss(ModelMixin):
    """VGG Loss introduced in
    Perceptual Losses for Real-Time Style Transfer and Super-Resolution
    https://arxiv.org/pdf/1603.08155
    """

    def __init__(self, layer=8, shift=0, reduction="mean"):
        super().__init__()
        self.model = vgg19(weights=VGG19_Weights).to(DEVICE).features[: layer + 1]
        self.shift = shift
        self.reduction = reduction
        self.normalize = tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __str__(self):
        return "vggloss"

    def prep(self):
        self.model.eval()
        self.model.requires_grad_(False)

    def __call__(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            x = x.torch()

        x = x.view(1, 3, 256, 256)
        return self.model(self.normalize(x))

    @dr.wrap("drjit", "torch")
    def lossfn(
        self,
        render: torch.Tensor,
        target: torch.Tensor,
    ):
        latent = self(render)
        with torch.no_grad():
            target = self(target)

        return F.mse_loss(latent, target, reduction=self.reduction)


class Resnet(ModelMixin):
    def __init__(self):
        super().__init__()
        self.is_imagenet = True
        self.model = resnet50(weights=ResNet50_Weights).eval().to(DEVICE)
        self.fc = self.model.fc
        self.model.fc = nn.Identity()
        self.normalize = tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __str__(self):
        return "ResNet50"

    def prep(self):
        self.model.eval()
        self.model.requires_grad_(False)

    @dr.wrap("drjit", "torch")
    def classify(
        self, x: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self.fc(self(x))
        probs = torch.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels, weight=weight)
        return loss, probs

    @dr.wrap("drjit", "torch")
    def lossfn(
        self,
        render: torch.Tensor,
        target: torch.Tensor,
    ):
        latent = self(render)
        with torch.no_grad():
            target = self(target)
        return F.l1_loss(latent, target)

    def __call__(self, render: torch.Tensor):
        if not torch.is_tensor(render):
            render = render.torch()
        render = render.view(1, 3, 256, 256)
        latent = self.model(self.normalize(render))
        return latent


class ResnetLatent(ModelMixin):
    def __init__(self, layer: int = 0):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights).eval().to(DEVICE)
        self.selected_layers = [
            "relu",  # after conv1 + bn1
            "layer1.0.conv2",  # inside 1st residual block
            "layer2.1.conv3",  # deeper
            "layer3.1.conv3",  # deeper
            "layer4.1.conv3",  # deepest
        ]
        self.selected_layer = layer
        self.selected_layers = self.selected_layers[layer]
        self.features = []
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

        # Register hooks
        for name, module in self.model.named_modules():
            if name in self.selected_layers:
                module.register_forward_hook(self.save_output())

    def __str__(self):
        return "ResnetLatent"

    def __call__(self, render):
        self.features = []

        if not torch.is_tensor(render):
            render = render.torch()

        render = render.view(1, 3, 256, 256)
        render = render / (render + 1)
        latent = self.model((render - self.mean) / self.std)
        self.features.append(latent)
        return self.features[0]

    @dr.wrap("drjit", "torch")
    def lossfn(self, render: torch.Tensor, target: torch.Tensor):
        render = self(render)
        return F.mse_loss(render, target)

    def save_output(self):
        def hook(module, input, output):
            self.features.append(output.flatten())

        return hook


class ResnetSIN(ModelMixin):
    def __init__(self) -> None:
        super().__init__()
        self.is_imagenet = True
        self.model = resnet50_trained_on_SIN("resnet50_trained_on_SIN").model.module
        self.fc = self.model.fc
        self.model.fc = nn.Identity()
        self.normalize = tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __str__(self):
        return "ResNet50 on SIN"

    def prep(self):
        self.model.eval()
        self.model.requires_grad_(False)

    @dr.wrap("drjit", "torch")
    def classify(
        self, x: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self.fc(self(x))
        probs = torch.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels, weight=weight)
        return loss, probs

    @dr.wrap("drjit", "torch")
    def lossfn(
        self,
        render: torch.Tensor,
        target: torch.Tensor,
    ):
        latent = self(render)
        with torch.no_grad():
            target = self(target)
        return F.l1_loss(latent, target)

    def __call__(self, render: torch.Tensor):
        if not torch.is_tensor(render):
            render = render.torch()
        render = render.view(1, 3, 256, 256)
        latent = self.model(self.normalize(render))
        return latent


class DINO(ModelMixin):
    def __init__(
        self, ckpt: str = "facebook/dinov2-base", normalize_feats: bool = True
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(ckpt).to(DEVICE)
        self.cfg = self.model.config
        self.image_size = 224
        self.patch_size = getattr(self.cfg, "patch_size", 14)  # ViT-L/14 -> 14
        self.normalize_feats = normalize_feats

        # CLIP normalization used in your code
        self.mean = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1).to(DEVICE)
        self.std = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1).to(DEVICE)

        self.prep()

    def prep(self):
        self.model.eval()
        self.model.requires_grad_(False)

    def __str__(self) -> str:
        return "dino"

    def __call__(self, render: mi.TensorXf) -> torch.Tensor:
        if not torch.is_tensor(render):
            render = render.torch()

        render = render.view(1, 3, self.image_size, self.image_size)
        render = (render - self.mean) / self.std

        out = self.model(pixel_values=render, output_hidden_states=False)
        tokens = out.last_hidden_state  # (1, 1 + H_p*W_p, D); first is CLS
        patch_tokens = tokens[:, 1:, :]  # drop CLS -> (1, N, D)

        # Compute grid size
        H_p = self.image_size // self.patch_size
        W_p = self.image_size // self.patch_size
        N = patch_tokens.shape[1]
        if H_p * W_p != N:  # fall back to square if needed
            g = int(sqrt(N))
            H_p = W_p = g

        dense = patch_tokens.view(1, H_p, W_p, -1)  # (1, H_p, W_p, D)
        if self.normalize_feats:
            dense = F.normalize(dense, dim=-1)
        return dense

    @dr.wrap("drjit", "torch")
    def lossfn(
        self, render: torch.Tensor, target: torch.Tensor, mode="cosine"
    ) -> torch.Tensor:
        """
        Patchwise loss between dense features:
          - mode="cosine"  → mean(1 - dot(render_feat, target_feat))
          - mode="l1"      → mean L1 across all patches and channels
        Expect shapes:
          - render: (..., 3, 224, 224)
          - target_dense: (1, H_p, W_p, D) or broadcastable
        """
        pred = self(render)  # (1, H_p, W_p, D)

        with torch.no_grad():
            target = self(target)

        if mode == "cosine":
            loss = (1.0 - (pred * target).sum(dim=-1)).mean()
        else:
            loss = F.l1_loss(pred, target)

        return loss


class VAE(ModelMixin):
    def __init__(self, repo="zelaki/eq-vae-ema", dtype=torch.float32):
        self.model = AutoencoderKL.from_pretrained(repo, torch_dtype=dtype)
        self.model.requires_grad_(False)  # freeze

    def __str__(self):
        return "gpvae"

    def __call__(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            x = x.torch()
        x = x.view(-1, 3, 256, 256)
        x = to_minus1_1(x)
        latent = self.model.encode(x).latent_dist
        return latent

    def prep(self):
        self.model = self.model.to(DEVICE)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def _encode_raw(self, x):
        # x: (B,3,256,256) in [-1,1]
        posterior = self.model.encode(x).latent_dist
        # Diffusers' DiagonalGaussianDistribution exposes .mean and .logvar
        mu = posterior.mean
        logvar = posterior.logvar
        return mu, logvar

    def encode_stats(self, x):
        """
        x: (...,3,256,256), pixels in [0,1] (or already scaled); returns (mu, logvar)
        """
        if not torch.is_tensor(x):
            x = x.torch()
        x = x.view(-1, 3, 256, 256).to(DEVICE)
        x = to_minus1_1(x)
        mu, logvar = self._encode_raw(x)
        return mu, logvar

    # -------- metrics (deterministic; use means) ----------
    @torch.no_grad()
    def pearson(self, render, target):
        mu, _ = self.encode_stats(render)
        a = flatten_nobatch(mu)
        b = flatten_nobatch(target.mean)
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)
        corr = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
        return corr.mean()

    @torch.no_grad()
    def spearman(self, render, target, ties=False):
        mu, _ = self.encode_stats(render)
        a = flatten_nobatch(mu)
        b = flatten_nobatch(target.mean)
        # simple rank transform (ties optional)
        if ties:
            # average ranks for ties
            def rankavg(t):
                sorted_idx = torch.argsort(t, dim=1)
                ranks = torch.zeros_like(sorted_idx, dtype=torch.float32)
                ranks.scatter_(
                    1,
                    sorted_idx,
                    torch.arange(t.shape[1], device=t.device).float().unsqueeze(0),
                )
                # naive tie handling via group averaging
                # (for speed, you can plug your existing spearman_correlation)
                return ranks

            ra, rb = rankavg(a), rankavg(b)
        else:
            ra = torch.argsort(torch.argsort(a, dim=1), dim=1).float()
            rb = torch.argsort(torch.argsort(b, dim=1), dim=1).float()
        ra = ra - ra.mean(dim=1, keepdim=True)
        rb = rb - rb.mean(dim=1, keepdim=True)
        corr = (ra * rb).sum(dim=1) / (ra.norm(dim=1) * rb.norm(dim=1) + 1e-8)
        return corr.mean()

    @torch.no_grad()
    def l2(self, x, y):
        mu, _ = self.encode_stats(x)
        return (mu - y.mean).pow(2).mean().sqrt()

    # -------- losses (use distributions; differentiable in x if x requires_grad) ----------
    @dr.wrap("drjit", "torch")
    def loss_w2(self, x, y):
        # x can require_grad; encodes inside (no detach)
        y_mu, y_logvar = y.mean, y.logvar
        if not torch.is_tensor(x):
            x = x.torch()
        x = x.view(-1, 3, 256, 256)
        x = to_minus1_1(x)
        mu, logvar = (
            self.model.encode(x).latent_dist.mean,
            self.model.encode(x).latent_dist.logvar,
        )
        return diag_gauss_w2(mu, logvar, y_mu, y_logvar).mean()

    @dr.wrap("drjit", "torch")
    def loss_skl(self, x, y):
        y_mu, y_logvar = y.mean, y.logvar
        if not torch.is_tensor(x):
            x = x.torch()
        x = x.view(-1, 3, 256, 256)
        x = to_minus1_1(x)
        posterior = self.model.encode(x).latent_dist
        mu, logvar = posterior.mean, posterior.logvar
        return diag_gauss_skl(mu, logvar, y_mu, y_logvar).mean()


class CLIPVision(ModelMixin, nn.Module):
    """
    Vision-only CLIP that returns dense patch embeddings as a 2D grid.
    - Input: image tensor shaped (..., 3, 224, 224) in float32
    - Output: dense features of shape (1, H_p, W_p, D) where H_p=W_p=224/patch_size
    - Loss: mean cosine distance (DINO-style) between normalized dense features.
    """

    def __init__(
        self,
        checkpoint: str = "openai/clip-vit-large-patch14",
        normalize_feats: bool = True,
    ):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(checkpoint).to(DEVICE)
        self.cfg = self.model.config
        self.image_size = getattr(self.cfg, "image_size", 224)
        self.patch_size = getattr(self.cfg, "patch_size", 14)  # ViT-L/14 -> 14
        self.normalize_feats = normalize_feats

        # CLIP normalization used in your code
        self.mean = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1).to(DEVICE)
        self.std = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1).to(DEVICE)

        self.prep()

    def __str__(self):
        return "CLIPVision"

    def prep(self):
        self.model.requires_grad_(False)
        self.model.eval()

    def __call__(self, x: mi.TensorXf) -> torch.Tensor:
        """
        Returns dense patch features as a grid: (1, H_p, W_p, D).
        """
        if not torch.is_tensor(x):
            x = x.torch()

        x = x.view(1, 3, self.image_size, self.image_size)
        x = (x - self.mean) / self.std

        out = self.model(pixel_values=x, output_hidden_states=False)
        tokens = out.last_hidden_state  # (1, 1 + H_p*W_p, D); first is CLS
        patch_tokens = tokens[:, 1:, :]  # drop CLS -> (1, N, D)

        # Compute grid size
        H_p = self.image_size // self.patch_size
        W_p = self.image_size // self.patch_size
        N = patch_tokens.shape[1]
        if H_p * W_p != N:  # fall back to square if needed
            g = int(sqrt(N))
            H_p = W_p = g

        dense = patch_tokens.view(1, H_p, W_p, -1)  # (1, H_p, W_p, D)
        if self.normalize_feats:
            dense = F.normalize(dense, dim=-1)
        return dense

    @dr.wrap("drjit", "torch")
    def lossfn(
        self, render: torch.Tensor, target: torch.Tensor, mode="cosine"
    ) -> torch.Tensor:
        """
        Patchwise loss between dense features:
          - mode="cosine"  → mean(1 - dot(render_feat, target_feat))
          - mode="l1"      → mean L1 across all patches and channels
        Expect shapes:
          - render: (..., 3, 224, 224)
          - target_dense: (1, H_p, W_p, D) or broadcastable
        """
        pred = self(render)  # (1, H_p, W_p, D)

        with torch.no_grad():
            target = self(target)

        if mode == "cosine":
            loss = (1.0 - (pred * target).sum(dim=-1)).mean()
        else:
            loss = F.l1_loss(pred, target)

        return loss


class LPIPS(ModelMixin):
    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        self.model = LPIPSModel(net=net).to(DEVICE)  # pyright: ignore

    def __str__(self) -> str:
        return "lpips"

    def __call__(self, render: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "LPIPS is a similarity metric but only one image can be provided."
        )

    @torch.no_grad()
    def spearman(self, render, target, ties: bool = False):
        raise RuntimeError("Cannot compute stats on LPIPS")

    @torch.no_grad()
    def pearson(self, render, target):
        raise RuntimeError("Cannot compute stats on LPIPS")

    @torch.no_grad()
    def l2(self, render: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Cannot compute stats on LPIPS")

    @dr.wrap("drjit", "torch")
    def lossfn(self, render: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        render = render.view(-1, 3, 256, 256)
        target = target.view(-1, 3, 256, 256)

        return self.model(render, target, normalize=True)


class LPIPSVGG(ModelMixin):
    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        self.model = LPIPSModel(lpips=False, net=net).to(DEVICE)  # pyright: ignore

    def __str__(self) -> str:
        return "lpipsvgg"

    def __call__(self, render: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "LPIPS is a similarity metric but only one image can be provided."
        )

    @torch.no_grad()
    def spearman(self, render, target, ties: bool = False):
        raise RuntimeError("Cannot compute stats on LPIPS")

    @torch.no_grad()
    def pearson(self, render, target):
        raise RuntimeError("Cannot compute stats on LPIPS")

    @torch.no_grad()
    def l2(self, render: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Cannot compute stats on LPIPS")

    @dr.wrap("drjit", "torch")
    def lossfn(self, render: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        render = render.view(-1, 3, 256, 256)
        target = target.view(-1, 3, 256, 256)

        _, res = self.model(render, target, True, True)
        return sum(res) / len(res)


class LaplacianLoss(ModelMixin):
    """
    Laplacian smoothing loss
    Penalizes deviations of each vertex from the average position of its neighboring vertices.
    ℒ_lap = 1/|V| ∑(i=1 to |V|) ||v_i - 1/|𝒩(i)| ∑(j∈𝒩(i)) v_j||²
    """

    is_torch: bool = False

    def __init__(self):
        super().__init__()
        self.adjacency_list = None
        self.vertex_indices = None
        self.neighbor_counts = None

    def __str__(self):
        return "LaplacianLoss"

    def build_adjacency(self, faces: mi.UInt, num_vertices: int):
        """Build vertex adjacency structure from faces."""
        faces_np = faces.numpy().reshape(-1, 3).astype(np.int32)
        adjacency = [set() for _ in range(num_vertices)]

        for face in faces_np:
            i, j, k = face
            adjacency[i].update([j, k])
            adjacency[j].update([i, k])
            adjacency[k].update([i, j])

        # Convert to flat arrays for efficient DrJit gathering
        vertex_indices = []
        neighbor_starts = [0]
        neighbor_indices = []
        neighbor_counts = []

        for i in range(num_vertices):
            neighbors = sorted(list(adjacency[i]))
            neighbor_counts.append(len(neighbors))
            neighbor_indices.extend(neighbors)
            neighbor_starts.append(len(neighbor_indices))

        self.vertex_indices = mi.UInt(np.arange(num_vertices, dtype=np.uint32))
        self.neighbor_starts = np.array(neighbor_starts[:-1], dtype=np.int32)
        self.neighbor_ends = np.array(neighbor_starts[1:], dtype=np.int32)
        self.neighbor_indices = mi.UInt(np.array(neighbor_indices, dtype=np.uint32))
        self.neighbor_counts = mi.Float(np.array(neighbor_counts, dtype=np.float32))
        self.num_vertices = num_vertices

    def __call__(self, vertices: mi.Float, faces: mi.UInt = None):
        """
        vertices: flattened vertex positions (num_vertices * 3,)
        faces: flattened face indices (num_faces * 3,) - only needed for initialization
        """
        if self.adjacency_list is None:
            if faces is None:
                raise ValueError("Must provide faces for first call")
            num_verts = len(vertices) // 3
            self.build_adjacency(faces, num_verts)

        # Reshape vertices to (num_vertices, 3)
        verts = mi.Float(vertices)

        # Compute loss for each vertex
        total_loss = mi.Float(0.0)

        for i in range(self.num_vertices):
            # Get neighbors for this vertex
            start = self.neighbor_starts[i]
            end = self.neighbor_ends[i]
            n_neighbors = end - start

            if n_neighbors == 0:
                continue

            # Current vertex position
            v_i = mi.Vector3f(verts[i*3], verts[i*3+1], verts[i*3+2])

            # Compute mean of neighbor positions
            neighbor_sum = mi.Vector3f(0.0, 0.0, 0.0)
            for idx in range(start, end):
                j = int(self.neighbor_indices[idx])
                v_j = mi.Vector3f(verts[j*3], verts[j*3+1], verts[j*3+2])
                neighbor_sum += v_j

            neighbor_mean = neighbor_sum / float(n_neighbors)

            # Compute squared deviation
            diff = v_i - neighbor_mean
            total_loss += dr.dot(diff, diff)

        return total_loss / float(self.num_vertices)

    def lossfn(self, vertices: mi.Float, faces: mi.UInt = None):
        return self(vertices, faces)


class EdgeLengthLoss(ModelMixin):
    """
    Uniform edge length loss
    Discourages disproportionate edge lengths by penalizing variance around the mean edge length.
    ℒ_edge = 1/|E| ∑((i,j)∈E) (||v_i - v_j|| - ℓ̄)²
    """

    is_torch: bool = False

    def __init__(self):
        super().__init__()
        self.edges = None

    def __str__(self):
        return "EdgeLengthLoss"

    def build_edges(self, faces: mi.UInt):
        """Extract unique edges from faces."""
        faces_np = faces.numpy().reshape(-1, 3).astype(np.int32)

        edges = set()
        for face in faces_np:
            i, j, k = face
            edges.add(tuple(sorted([i, j])))
            edges.add(tuple(sorted([j, k])))
            edges.add(tuple(sorted([k, i])))

        edges = np.array(list(edges), dtype=np.uint32)
        self.edge_i = mi.UInt(edges[:, 0])
        self.edge_j = mi.UInt(edges[:, 1])
        self.num_edges = len(edges)

    def __call__(self, vertices: mi.Float, faces: mi.UInt = None):
        """
        vertices: flattened vertex positions (num_vertices * 3,)
        faces: flattened face indices (num_faces * 3,) - only needed for initialization
        """
        if self.edges is None:
            if faces is None:
                raise ValueError("Must provide faces for first call")
            self.build_edges(faces)

        verts = mi.Float(vertices)

        # Compute all edge lengths
        edge_lengths = mi.Float(0.0)
        length_sum = mi.Float(0.0)

        for idx in range(self.num_edges):
            i = int(self.edge_i[idx])
            j = int(self.edge_j[idx])

            v_i = mi.Vector3f(verts[i*3], verts[i*3+1], verts[i*3+2])
            v_j = mi.Vector3f(verts[j*3], verts[j*3+1], verts[j*3+2])

            edge_vec = v_j - v_i
            length = dr.norm(edge_vec)
            length_sum += length

        mean_length = length_sum / float(self.num_edges)

        # Compute variance
        variance = mi.Float(0.0)
        for idx in range(self.num_edges):
            i = int(self.edge_i[idx])
            j = int(self.edge_j[idx])

            v_i = mi.Vector3f(verts[i*3], verts[i*3+1], verts[i*3+2])
            v_j = mi.Vector3f(verts[j*3], verts[j*3+1], verts[j*3+2])

            edge_vec = v_j - v_i
            length = dr.norm(edge_vec)
            diff = length - mean_length
            variance += diff * diff

        return variance / float(self.num_edges)

    def lossfn(self, vertices: mi.Float, faces: mi.UInt = None):
        return self(vertices, faces)


class TriangleAreaLoss(ModelMixin):
    """
    Triangle area loss
    Enforces uniformity in triangular face sizes by penalizing area variance.
    ℒ_area = 1/|F| ∑(t∈F) (A_t - Ā)²
    """

    is_torch: bool = False

    def __init__(self):
        super().__init__()
        self.face_indices = None

    def __str__(self):
        return "TriangleAreaLoss"

    def build_face_indices(self, faces: mi.UInt):
        """Store face indices for area computation."""
        faces_np = faces.numpy().reshape(-1, 3).astype(np.uint32)
        self.face_i = mi.UInt(faces_np[:, 0])
        self.face_j = mi.UInt(faces_np[:, 1])
        self.face_k = mi.UInt(faces_np[:, 2])
        self.num_faces = len(faces_np)

    def __call__(self, vertices: mi.Float, faces: mi.UInt = None):
        """
        vertices: flattened vertex positions (num_vertices * 3,)
        faces: flattened face indices (num_faces * 3,) - only needed for initialization
        """
        if self.face_indices is None:
            if faces is None:
                raise ValueError("Must provide faces for first call")
            self.build_face_indices(faces)

        verts = mi.Float(vertices)

        # Compute all triangle areas
        area_sum = mi.Float(0.0)

        for idx in range(self.num_faces):
            i = int(self.face_i[idx])
            j = int(self.face_j[idx])
            k = int(self.face_k[idx])

            v_i = mi.Vector3f(verts[i*3], verts[i*3+1], verts[i*3+2])
            v_j = mi.Vector3f(verts[j*3], verts[j*3+1], verts[j*3+2])
            v_k = mi.Vector3f(verts[k*3], verts[k*3+1], verts[k*3+2])

            # Area = 0.5 * ||(v_j - v_i) × (v_k - v_i)||
            edge1 = v_j - v_i
            edge2 = v_k - v_i
            cross_prod = dr.cross(edge1, edge2)
            area = 0.5 * dr.norm(cross_prod)
            area_sum += area

        mean_area = area_sum / float(self.num_faces)

        # Compute variance
        variance = mi.Float(0.0)
        for idx in range(self.num_faces):
            i = int(self.face_i[idx])
            j = int(self.face_j[idx])
            k = int(self.face_k[idx])

            v_i = mi.Vector3f(verts[i*3], verts[i*3+1], verts[i*3+2])
            v_j = mi.Vector3f(verts[j*3], verts[j*3+1], verts[j*3+2])
            v_k = mi.Vector3f(verts[k*3], verts[k*3+1], verts[k*3+2])

            edge1 = v_j - v_i
            edge2 = v_k - v_i
            cross_prod = dr.cross(edge1, edge2)
            area = 0.5 * dr.norm(cross_prod)
            diff = area - mean_area
            variance += diff * diff

        return variance / float(self.num_faces)

    def lossfn(self, vertices: mi.Float, faces: mi.UInt = None):
        return self(vertices, faces)


class ARAPLoss(ModelMixin):
    """
    As-Rigid-As-Possible (ARAP) loss
    Preserves original edge orientations by projecting deformed edges onto their initial directions.
    ℒ_arap = 1/|E| ∑((i,j)∈E) ||(v_i - v_j) - proj_ê^(0)_ij(v_i - v_j)||²
    """

    is_torch: bool = False

    def __init__(self):
        super().__init__()
        self.initial_edge_dirs = None
        self.edge_i = None
        self.edge_j = None

    def __str__(self):
        return "ARAPLoss"

    def initialize(self, vertices: mi.Float, faces: mi.UInt):
        """
        Initialize with the rest pose (initial vertex positions).
        This should be called once before optimization starts.
        """
        verts_np = vertices.numpy().reshape(-1, 3)
        faces_np = faces.numpy().reshape(-1, 3).astype(np.int32)

        # Extract unique edges
        edges = set()
        for face in faces_np:
            i, j, k = face
            edges.add(tuple(sorted([i, j])))
            edges.add(tuple(sorted([j, k])))
            edges.add(tuple(sorted([k, i])))

        edges = list(edges)
        self.num_edges = len(edges)

        # Store edge indices
        edge_array = np.array(edges, dtype=np.uint32)
        self.edge_i = mi.UInt(edge_array[:, 0])
        self.edge_j = mi.UInt(edge_array[:, 1])

        # Compute and store initial edge directions (normalized)
        initial_dirs = []
        for i, j in edges:
            edge_vec = verts_np[j] - verts_np[i]
            edge_dir = edge_vec / (np.linalg.norm(edge_vec) + 1e-8)
            initial_dirs.append(edge_dir)

        initial_dirs = np.array(initial_dirs, dtype=np.float32).flatten()
        self.initial_edge_dirs = mi.Float(initial_dirs)

    def __call__(self, vertices: mi.Float, faces: mi.UInt = None):
        """
        vertices: flattened vertex positions (num_vertices * 3,)
        faces: flattened face indices (only needed if not initialized)
        """
        if self.initial_edge_dirs is None:
            if faces is None:
                raise ValueError("Must call initialize() first or provide faces")
            self.initialize(vertices, faces)
            return mi.Float(0.0)

        verts = mi.Float(vertices)

        # Compute ARAP loss
        total_loss = mi.Float(0.0)

        for idx in range(self.num_edges):
            i = int(self.edge_i[idx])
            j = int(self.edge_j[idx])

            # Current edge vector
            v_i = mi.Vector3f(verts[i*3], verts[i*3+1], verts[i*3+2])
            v_j = mi.Vector3f(verts[j*3], verts[j*3+1], verts[j*3+2])
            edge_curr = v_j - v_i

            # Initial edge direction
            edge_init = mi.Vector3f(
                self.initial_edge_dirs[idx*3],
                self.initial_edge_dirs[idx*3+1],
                self.initial_edge_dirs[idx*3+2]
            )

            # Project current edge onto initial direction
            dot_prod = dr.dot(edge_curr, edge_init)
            projection = edge_init * dot_prod

            # Compute deviation from projection
            deviation = edge_curr - projection
            total_loss += dr.dot(deviation, deviation)

        return total_loss / float(self.num_edges)

    def lossfn(self, vertices: mi.Float, faces: mi.UInt = None):
        return self(vertices, faces)


class Model(Enum):
    MAE = MAE
    DUAL_BUFFER = DualBuffer
    DINO = DINO
    CLIP = CLIPVision
    RESNET = Resnet
    RESNET_SIN = ResnetSIN
    VGGLoss = VGGLoss
    LPIPS = LPIPS
    VGG = LPIPSVGG
