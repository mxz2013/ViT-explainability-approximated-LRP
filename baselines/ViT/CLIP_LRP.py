"""CLIP Vision Transformer (ViT-B-16) with LRP (Layer-wise Relevance Propagation)
Adapted from OpenAI CLIP implementation for LRP support.
LRP method from: "Transformer Interpretability Beyond Attention Visualization" (Chefer et al., CVPR 2021)
"""
import torch
import torch.nn as nn
from einops import rearrange
from modules.layers_ours import *
from collections import OrderedDict


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    """Compute attention rollout across layers."""
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                    for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


class CLIPPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding using Conv2d (CLIP style)
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        # Use LRP-aware Conv2d
        self.proj = Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, grid, grid) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1, 2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                         self.image_size[0] // self.patch_size[0],
                         self.image_size[1] // self.patch_size[1])
        return self.proj.relprop(cam, **kwargs)


class CLIPAttention(nn.Module):
    """
    Multi-head self-attention with LRP support (CLIP style).
    CLIP uses in_proj_weight/in_proj_bias for combined QKV.
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection (CLIP style)
        self.qkv = Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = Linear(embed_dim, embed_dim)

        self.attn_drop = Dropout(dropout)
        self.proj_drop = Dropout(dropout)

        # Custom operations for LRP
        self.matmul1 = einsum('bhid,bhjd->bhij')
        self.matmul2 = einsum('bhij,bhjd->bhid')
        self.softmax = Softmax(dim=-1)

        # Storage for attention and gradients
        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        if attn.requires_grad:
            attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)


class CLIPMLP(nn.Module):
    """
    MLP block with LRP support (CLIP style uses GELU activation).
    """
    def __init__(self, embed_dim=768, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.c_fc = Linear(embed_dim, hidden_dim)
        self.act = GELU()
        self.c_proj = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.c_proj.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.c_fc.relprop(cam, **kwargs)
        return cam


class CLIPResidualAttentionBlock(nn.Module):
    """
    Transformer block with pre-norm, attention, MLP and residual connections.
    CLIP uses pre-norm (LayerNorm before attention/MLP).
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.ln_1 = LayerNorm(embed_dim)
        self.attn = CLIPAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.ln_2 = LayerNorm(embed_dim)
        self.mlp = CLIPMLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio)

        # For LRP - handle residual connections
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        # First residual: attention
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.ln_1(x2))])

        # Second residual: MLP
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.ln_2(x2))])

        return x

    def relprop(self, cam, **kwargs):
        # Backprop through second residual (MLP)
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.ln_2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        # Backprop through first residual (attention)
        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.ln_1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)

        return cam


class CLIPVisionTransformer(nn.Module):
    """
    CLIP Vision Transformer with LRP support.
    Architecture matches CLIP ViT-B-16:
    - Patch size: 16
    - Hidden dim: 768
    - Layers: 12
    - Heads: 12
    - MLP ratio: 4
    - Projection dim: 512 (CLIP projects to this before contrastive learning)
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        proj_dim=512,
        dropout=0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = CLIPPatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CLIPResidualAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Pre-logits layer norm (CLIP applies LN before projection)
        self.ln_pre = LayerNorm(embed_dim)
        self.ln_post = LayerNorm(embed_dim)

        # CLIP projection layer (768 -> 512)
        self.proj = Linear(embed_dim, proj_dim, bias=False)

        # Classification head (on projected features)
        self.head = Linear(proj_dim, num_classes)

        # For LRP
        self.pool = IndexSelect()
        self.add = Add()
        self.inp_grad = None

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = self.add([x, self.pos_embed])

        # Pre-LN (CLIP style)
        x = self.ln_pre(x)

        if x.requires_grad:
            x.register_hook(self.save_inp_grad)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Post-LN
        x = self.ln_post(x)

        # Extract CLS token
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)

        # CLIP projection (768 -> 512)
        x = self.proj(x)

        # Classification head
        x = self.head(x)

        return x

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        """
        Relevance propagation for different attribution methods.
        """
        # Propagate through classification head
        cam = self.head.relprop(cam, **kwargs)

        # Propagate through CLIP projection
        cam = self.proj.relprop(cam, **kwargs)

        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.ln_post.relprop(cam, **kwargs)

        # Propagate through transformer blocks
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        if method == "full":
            cam = self.ln_pre.relprop(cam, **kwargs)
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn()
                if attn_heads is None:
                    raise ValueError("Attention weights not found. Run forward pass first.")
                attn_heads = attn_heads.clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam

        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                attn_cam = blk.attn.get_attn_cam()

                if attn_cam is None or grad is None:
                    raise ValueError("Attention or gradients not found. Run forward+backward pass first.")

                attn_cam = attn_cam[0].reshape(-1, attn_cam.shape[-1], attn_cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam_attn = grad * attn_cam
                cam_attn = cam_attn.mean(dim=0).clamp(min=0)
                cams.append(cam_attn.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam

        elif method == "last_layer":
            attn = self.blocks[-1].attn.get_attn()
            if attn is None:
                raise ValueError("Attention weights not found. Run forward pass first.")
            attn = attn[0].reshape(-1, attn.shape[-1], attn.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                if grad is not None:
                    grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                    attn = grad * attn
            attn = attn.clamp(min=0).mean(dim=0)
            cam = attn[0, 1:]
            return cam

        elif method == "last_layer_attn":
            attn = self.blocks[-1].attn.get_attn()
            if attn is None:
                raise ValueError("Attention weights not found. Run forward pass first.")
            attn = attn[0].reshape(-1, attn.shape[-1], attn.shape[-1])
            attn = attn.clamp(min=0).mean(dim=0)
            cam = attn[0, 1:]
            return cam

        else:
            raise ValueError(f"Unknown method: {method}")


def load_clip_weights_to_lrp_model(lrp_model, checkpoint_path, num_classes=10):
    """
    Load weights from CLIP linear probe checkpoint into our LRP model.

    The checkpoint contains:
    - clip_visual.* : Frozen CLIP visual encoder weights
    - head.* : Trained linear classification head

    Args:
        lrp_model: Our CLIPVisionTransformer with LRP support
        checkpoint_path: Path to the linear probe checkpoint
        num_classes: Number of classes (should match checkpoint)

    Returns:
        lrp_model with loaded weights
    """
    print(f"Loading CLIP checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    lrp_state_dict = {}

    for key, value in state_dict.items():
        new_key = None

        # Map CLIP visual encoder weights
        if key.startswith("clip_visual."):
            clip_key = key.replace("clip_visual.", "")

            # Patch embedding: conv1 -> patch_embed.proj
            if clip_key == "conv1.weight":
                new_key = "patch_embed.proj.weight"

            # Class token
            elif clip_key == "class_embedding":
                new_key = "cls_token"
                value = value.unsqueeze(0).unsqueeze(0)  # (768,) -> (1, 1, 768)

            # Positional embedding
            elif clip_key == "positional_embedding":
                new_key = "pos_embed"
                value = value.unsqueeze(0)  # (197, 768) -> (1, 197, 768)

            # Pre-LN
            elif clip_key == "ln_pre.weight":
                new_key = "ln_pre.weight"
            elif clip_key == "ln_pre.bias":
                new_key = "ln_pre.bias"

            # Post-LN (CLIP calls it ln_post)
            elif clip_key == "ln_post.weight":
                new_key = "ln_post.weight"
            elif clip_key == "ln_post.bias":
                new_key = "ln_post.bias"

            # Transformer blocks
            elif clip_key.startswith("transformer.resblocks."):
                block_key = clip_key.replace("transformer.resblocks.", "")
                parts = block_key.split(".", 1)
                block_idx = parts[0]
                rest = parts[1] if len(parts) > 1 else ""

                # Attention: in_proj_weight/in_proj_bias -> qkv.weight/qkv.bias
                if rest == "attn.in_proj_weight":
                    new_key = f"blocks.{block_idx}.attn.qkv.weight"
                elif rest == "attn.in_proj_bias":
                    new_key = f"blocks.{block_idx}.attn.qkv.bias"
                elif rest == "attn.out_proj.weight":
                    new_key = f"blocks.{block_idx}.attn.proj.weight"
                elif rest == "attn.out_proj.bias":
                    new_key = f"blocks.{block_idx}.attn.proj.bias"

                # LayerNorm 1
                elif rest == "ln_1.weight":
                    new_key = f"blocks.{block_idx}.ln_1.weight"
                elif rest == "ln_1.bias":
                    new_key = f"blocks.{block_idx}.ln_1.bias"

                # LayerNorm 2
                elif rest == "ln_2.weight":
                    new_key = f"blocks.{block_idx}.ln_2.weight"
                elif rest == "ln_2.bias":
                    new_key = f"blocks.{block_idx}.ln_2.bias"

                # MLP: c_fc -> c_fc, c_proj -> c_proj
                elif rest == "mlp.c_fc.weight":
                    new_key = f"blocks.{block_idx}.mlp.c_fc.weight"
                elif rest == "mlp.c_fc.bias":
                    new_key = f"blocks.{block_idx}.mlp.c_fc.bias"
                elif rest == "mlp.c_proj.weight":
                    new_key = f"blocks.{block_idx}.mlp.c_proj.weight"
                elif rest == "mlp.c_proj.bias":
                    new_key = f"blocks.{block_idx}.mlp.c_proj.bias"

            # Projection layer (768 -> 512)
            elif clip_key == "proj":
                new_key = "proj.weight"
                # CLIP stores proj as (768, 512), we need (512, 768) for Linear
                value = value.t()

        # Map classification head
        elif key == "head.weight":
            new_key = "head.weight"
        elif key == "head.bias":
            new_key = "head.bias"

        if new_key is not None:
            lrp_state_dict[new_key] = value

    # Load weights
    msg = lrp_model.load_state_dict(lrp_state_dict, strict=False)
    print("Loaded CLIP weights:")
    print(f"  Missing keys: {msg.missing_keys}")
    print(f"  Unexpected keys: {msg.unexpected_keys}")

    return lrp_model


def clip_vit_base_patch16_224(checkpoint_path, num_classes=10, **kwargs):
    """
    Load CLIP ViT-B-16 model with LRP support from a linear probe checkpoint.

    Model configuration (CLIP ViT-B-16):
    - Image size: 224x224
    - Patch size: 16x16 (results in 14x14 patch grid)
    - Hidden size: 768
    - Num layers: 12
    - Num heads: 12
    - MLP ratio: 4.0

    Args:
        checkpoint_path: Path to CLIP linear probe checkpoint (.pth file)
        num_classes: Number of output classes (default: 10 for ImageNet subset)
        **kwargs: Additional arguments

    Returns:
        CLIPVisionTransformer model with LRP capabilities and loaded weights

    Example:
        >>> model = clip_vit_base_patch16_224(
        ...     checkpoint_path='clip/output_dir_small_batch/checkpoint-14.pth',
        ...     num_classes=10
        ... )
        >>> model.eval()
    """
    model = CLIPVisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        dropout=0.0,
        **kwargs
    )

    model = load_clip_weights_to_lrp_model(model, checkpoint_path, num_classes)

    return model
