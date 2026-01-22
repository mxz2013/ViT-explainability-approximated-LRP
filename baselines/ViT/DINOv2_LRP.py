""" DINOv2 Vision Transformer with LRP (Layer-wise Relevance Propagation)
Adapted from HuggingFace transformers DINOv2 implementation
Original paper: "DINOv2: Learning Robust Visual Features without Supervision"
LRP method from: "Transformer Interpretability Beyond Attention Visualization" (Chefer et al., CVPR 2021)
"""
import torch
import torch.nn as nn
from einops import rearrange
from modules.layers_ours import *

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # Normalize after adding residual
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


class Dinov2PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding using Conv2d
    """
    def __init__(self, image_size=224, patch_size=14, num_channels=3, hidden_size=768):
        super().__init__()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (self.image_size[1] // self.patch_size[1]) * (self.image_size[0] // self.patch_size[0])
        self.num_channels = num_channels

        # Use LRP-aware Conv2d from modules.layers_ours
        self.projection = Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1, 2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                         (self.image_size[0] // self.patch_size[0]),
                         (self.image_size[1] // self.patch_size[1]))
        return self.projection.relprop(cam, **kwargs)


class Dinov2Embeddings(nn.Module):
    """
    Construct the CLS token, position embeddings and patch embeddings.
    """
    def __init__(self, image_size=224, patch_size=14, num_channels=3, hidden_size=768, hidden_dropout_prob=0.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.patch_embeddings = Dinov2PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.dropout = Dropout(hidden_dropout_prob)

        # For LRP
        self.add = Add()

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = self.add([embeddings, self.position_embeddings])
        embeddings = self.dropout(embeddings)

        return embeddings

    def relprop(self, cam, **kwargs):
        cam = self.dropout.relprop(cam, **kwargs)
        (cam, _) = self.add.relprop(cam, **kwargs)
        # Remove CLS token
        cam = cam[:, 1:]
        return self.patch_embeddings.relprop(cam, **kwargs)


class Dinov2Attention(nn.Module):
    """
    Multi-head self-attention with LRP support.
    Note: DINOv2 uses separate Q, K, V projections (not combined QKV like ViT).
    """
    def __init__(self, hidden_size=768, num_attention_heads=12, qkv_bias=True, attention_probs_dropout_prob=0.0):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scale = self.attention_head_size ** -0.5

        # DINOv2 uses separate Q, K, V (not combined)
        # We'll combine them for LRP compatibility
        self.qkv = Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = Linear(hidden_size, hidden_size)

        self.attn_drop = Dropout(attention_probs_dropout_prob)
        self.proj_drop = Dropout(attention_probs_dropout_prob)

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

    def forward(self, hidden_states):
        b, n, _, h = *hidden_states.shape, self.num_attention_heads
        qkv = self.qkv(hidden_states)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        # Only register hook if gradients are enabled
        if attn.requires_grad:
            attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v]) # shape 1, 12, 257, 64
        out = rearrange(out, 'b h n d -> b n (h d)') # shape 1, 257, 768  heads merged

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_attention_heads)

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

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_attention_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)


class Dinov2MLP(nn.Module):
    """
    MLP (Feed-forward) block with LRP support
    """
    def __init__(self, hidden_size=768, mlp_ratio=4.0):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = Linear(hidden_size, hidden_features, bias=True)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, hidden_size, bias=True)

    def forward(self, hidden_state):
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state

    def relprop(self, cam, **kwargs):
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Dinov2LayerScale(nn.Module):
    """
    Layer scale module for residual connections.
    For LRP, we treat this as a simple element-wise multiplication.
    """
    def __init__(self, hidden_size=768, layerscale_value=1.0):
        super().__init__()
        self.lambda1 = nn.Parameter(layerscale_value * torch.ones(hidden_size))

    def forward(self, hidden_state):
        return hidden_state * self.lambda1

    def relprop(self, cam, **kwargs):
        # Layer scale is just element-wise multiplication
        # For LRP, we propagate through by dividing by the scale
        return cam


class Dinov2Layer(nn.Module):
    """
    Transformer block with pre-norm, attention, MLP, layer scale and residual connections.
    """
    def __init__(self, hidden_size=768, num_attention_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0,
                 layerscale_value=1.0, drop_path_rate=0.0):
        super().__init__()

        self.norm1 = LayerNorm(hidden_size, eps=1e-6)
        self.attention = Dinov2Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            qkv_bias=qkv_bias,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        self.layer_scale1 = Dinov2LayerScale(hidden_size, layerscale_value)

        self.norm2 = LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Dinov2MLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)
        self.layer_scale2 = Dinov2LayerScale(hidden_size, layerscale_value)

        # For LRP - handle residual connections
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

        # Drop path (stochastic depth) - for now we use identity during eval
        # Can add proper DropPath if needed
        self.drop_path = nn.Identity()

    def forward(self, hidden_states):
        # First residual: attention
        x1, x2 = self.clone1(hidden_states, 2)
        attn_output = self.attention(self.norm1(x2))
        attn_output = self.layer_scale1(attn_output)
        attn_output = self.drop_path(attn_output)
        hidden_states = self.add1([x1, attn_output])

        # Second residual: MLP
        x1, x2 = self.clone2(hidden_states, 2)
        mlp_output = self.mlp(self.norm2(x2))
        mlp_output = self.layer_scale2(mlp_output)
        mlp_output = self.drop_path(mlp_output)
        hidden_states = self.add2([x1, mlp_output])

        return hidden_states

    def relprop(self, cam, **kwargs):
        # Backprop through second residual (MLP)
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.layer_scale2.relprop(cam2, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        # Backprop through first residual (attention)
        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.layer_scale1.relprop(cam2, **kwargs)
        cam2 = self.attention.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)

        return cam


class Dinov2Model(nn.Module):
    """
    DINOv2 base model (without classification head)
    """
    def __init__(self, image_size=224, patch_size=14, num_channels=3,
                 hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0, layerscale_value=1.0):
        super().__init__()

        self.embeddings = Dinov2Embeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob
        )

        self.layers = nn.ModuleList([
            Dinov2Layer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                layerscale_value=layerscale_value
            )
            for _ in range(num_hidden_layers)
        ])

        self.norm = LayerNorm(hidden_size, eps=1e-6)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def relprop(self, cam, **kwargs):
        cam = self.norm.relprop(cam, **kwargs)

        for layer in reversed(self.layers):
            cam = layer.relprop(cam, **kwargs)

        return self.embeddings.relprop(cam, **kwargs)


class Dinov2ForImageClassification(nn.Module):
    """
    DINOv2 Model with image classification head.
    The classification head uses: concat(CLS_token, mean(patch_tokens))
    """
    def __init__(self, image_size=224, patch_size=14, num_channels=3,
                 num_classes=1000, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                 layerscale_value=1.0):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.dinov2 = Dinov2Model(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layerscale_value=layerscale_value
        )

        # Classification head: Linear(hidden_size * 2, num_classes)
        # Input is concat of CLS token and mean of patch tokens
        self.classifier = Linear(hidden_size * 2, num_classes)

        # For LRP
        self.pool_cls = IndexSelect()
        self.pool_patches = IndexSelect()

        self.inp_grad = None

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def forward(self, pixel_values):
        hidden_states = self.dinov2(pixel_values)

        # Extract CLS token (first token)
        cls_token = self.pool_cls(hidden_states, dim=1, indices=torch.tensor(0, device=hidden_states.device))
        cls_token = cls_token.squeeze(1)  # (batch_size, hidden_size)

        # Extract patch tokens (all except first) and compute mean
        patch_tokens = hidden_states[:, 1:]  # (batch_size, num_patches, hidden_size)
        patch_mean = patch_tokens.mean(dim=1)  # (batch_size, hidden_size)

        # Concatenate CLS and patch mean
        linear_input = torch.cat([cls_token, patch_mean], dim=1)  # (batch_size, hidden_size * 2)

        # Register hook for gradients (only if requires_grad)
        if linear_input.requires_grad:
            linear_input.register_hook(self.save_inp_grad)

        # Classification
        logits = self.classifier(linear_input)

        return logits

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        """
        Relevance propagation for different methods.

        Args:
            cam: Initial relevance (usually from output logits)
            method: One of ["transformer_attribution", "grad", "rollout", "full", "last_layer", "last_layer_attn"]
            is_ablation: Whether to use ablation mode
            start_layer: Starting layer for rollout methods
        """
        # Propagate through classifier
        cam = self.classifier.relprop(cam, **kwargs)

        # Split back into CLS and patch mean components
        # cam shape: (batch_size, hidden_size * 2)
        cam_cls = cam[:, :self.hidden_size]
        cam_patch_mean = cam[:, self.hidden_size:]

        # For patch mean, we need to expand it to all patch positions
        # This is a simplification - mean aggregation makes exact LRP complex
        num_patches = self.dinov2.embeddings.patch_embeddings.num_patches
        cam_patches = cam_patch_mean.unsqueeze(1).expand(-1, num_patches, -1) / num_patches

        # Reconstruct full cam with CLS token
        cam_cls = cam_cls.unsqueeze(1)  # (batch_size, 1, hidden_size)
        cam = torch.cat([cam_cls, cam_patches], dim=1)  # (batch_size, num_patches+1, hidden_size)

        if method == "full":
            # Full LRP back to pixels
            cam = self.dinov2.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # Attention rollout - use actual attention weights from forward pass
            attn_cams = []
            for layer in self.dinov2.layers:
                attn_heads = layer.attention.get_attn()
                if attn_heads is None:
                    raise ValueError("Attention weights not found. Make sure to run forward pass first.")
                attn_heads = attn_heads.clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]  # CLS to patches
            return cam

        elif method == "transformer_attribution" or method == "grad":
            # Gradient-weighted attention (our method)
            # First, run LRP to populate attn_cam values
            cam_copy = cam.clone()
            for layer in reversed(self.dinov2.layers):
                cam_copy = layer.relprop(cam_copy, **kwargs)

            # Now use LRP-propagated attention (attn_cam) with gradients
            cams = []
            for layer in self.dinov2.layers:
                grad = layer.attention.get_attn_gradients()
                attn_cam = layer.attention.get_attn_cam()

                if attn_cam is None or grad is None:
                    raise ValueError("Attention or gradients not found. Make sure to run forward+backward pass first.")

                attn_cam = attn_cam[0].reshape(-1, attn_cam.shape[-1], attn_cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam_attn = grad * attn_cam
                # IMPORTANT: Average across heads first, then clamp
                # This allows positive and negative gradients to cancel out properly
                cam_attn = cam_attn.mean(dim=0).clamp(min=0)
                cams.append(cam_attn.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam

        elif method == "last_layer":
            attn = self.dinov2.layers[-1].attention.get_attn()
            if attn is None:
                raise ValueError("Attention weights not found. Make sure to run forward pass first.")
            attn = attn[0].reshape(-1, attn.shape[-1], attn.shape[-1])
            if is_ablation:
                grad = self.dinov2.layers[-1].attention.get_attn_gradients()
                if grad is not None:
                    grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                    attn = grad * attn
            attn = attn.clamp(min=0).mean(dim=0)
            cam = attn[0, 1:]
            return cam

        elif method == "last_layer_attn":
            attn = self.dinov2.layers[-1].attention.get_attn()
            if attn is None:
                raise ValueError("Attention weights not found. Make sure to run forward pass first.")
            attn = attn[0].reshape(-1, attn.shape[-1], attn.shape[-1])
            attn = attn.clamp(min=0).mean(dim=0)
            cam = attn[0, 1:]
            return cam

        else:
            raise ValueError(f"Unknown method: {method}")


def load_dinov2_pretrained_weights(lrp_model, hf_model_name='facebook/dinov2-base-imagenet1k-1-layer',
                                   interpolate_pos_encoding=True):
    """
    Load pretrained weights from HuggingFace DINOv2 model into our LRP model.

    Args:
        lrp_model: Our custom Dinov2ForImageClassification with LRP support
        hf_model_name: HuggingFace model identifier
        interpolate_pos_encoding: If True, interpolate position embeddings to match target size

    Returns:
        lrp_model with loaded weights
    """
    from transformers import AutoModelForImageClassification
    import torch.nn.functional as F

    print(f"Loading pretrained weights from {hf_model_name}...")
    hf_model = AutoModelForImageClassification.from_pretrained(hf_model_name)
    hf_state_dict = hf_model.state_dict()

    # Create mapping between HuggingFace parameter names and our LRP model names
    lrp_state_dict = {}

    for hf_key, hf_value in hf_state_dict.items():
        # Map HuggingFace keys to LRP model keys
        lrp_key = hf_key

        # Remove 'dinov2.' prefix
        if lrp_key.startswith('dinov2.'):
            lrp_key = lrp_key.replace('dinov2.', 'dinov2.')

        # Map embeddings
        lrp_key = lrp_key.replace('embeddings.patch_embeddings.projection.', 'embeddings.patch_embeddings.projection.')
        lrp_key = lrp_key.replace('embeddings.cls_token', 'embeddings.cls_token')
        lrp_key = lrp_key.replace('embeddings.position_embeddings', 'embeddings.position_embeddings')

        # Map encoder layers
        lrp_key = lrp_key.replace('encoder.layer.', 'layers.')

        # Map attention: HF uses separate Q, K, V while we use combined QKV
        # We need to handle this specially
        if '.attention.attention.query.' in hf_key or '.attention.attention.key.' in hf_key or '.attention.attention.value.' in hf_key:
            # Skip for now, we'll handle Q, K, V separately
            continue

        # Map attention output projection
        lrp_key = lrp_key.replace('.attention.output.dense.', '.attention.proj.')

        # Map MLP
        lrp_key = lrp_key.replace('.mlp.fc1.', '.mlp.fc1.')
        lrp_key = lrp_key.replace('.mlp.fc2.', '.mlp.fc2.')

        # Map layer scale
        lrp_key = lrp_key.replace('.layer_scale1.lambda1', '.layer_scale1.lambda1')
        lrp_key = lrp_key.replace('.layer_scale2.lambda1', '.layer_scale2.lambda1')

        # Map final norm
        lrp_key = lrp_key.replace('dinov2.layernorm.', 'dinov2.norm.')

        # Map classifier
        lrp_key = lrp_key.replace('classifier.', 'classifier.')

        lrp_state_dict[lrp_key] = hf_value

    # Special handling for Q, K, V -> QKV concatenation
    num_layers = len([k for k in hf_state_dict.keys() if '.attention.attention.query.weight' in k])
    for layer_idx in range(num_layers):
        q_weight = hf_state_dict[f'dinov2.encoder.layer.{layer_idx}.attention.attention.query.weight']
        k_weight = hf_state_dict[f'dinov2.encoder.layer.{layer_idx}.attention.attention.key.weight']
        v_weight = hf_state_dict[f'dinov2.encoder.layer.{layer_idx}.attention.attention.value.weight']

        q_bias = hf_state_dict[f'dinov2.encoder.layer.{layer_idx}.attention.attention.query.bias']
        k_bias = hf_state_dict[f'dinov2.encoder.layer.{layer_idx}.attention.attention.key.bias']
        v_bias = hf_state_dict[f'dinov2.encoder.layer.{layer_idx}.attention.attention.value.bias']

        # Concatenate Q, K, V
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

        lrp_state_dict[f'dinov2.layers.{layer_idx}.attention.qkv.weight'] = qkv_weight
        lrp_state_dict[f'dinov2.layers.{layer_idx}.attention.qkv.bias'] = qkv_bias

    # Handle position embeddings - may need interpolation
    if interpolate_pos_encoding and 'dinov2.embeddings.position_embeddings' in lrp_state_dict:
        pretrained_pos_embed = lrp_state_dict['dinov2.embeddings.position_embeddings']
        target_pos_embed = lrp_model.dinov2.embeddings.position_embeddings

        # Check if sizes match
        if pretrained_pos_embed.shape != target_pos_embed.shape:
            print(f"Interpolating position embeddings from {pretrained_pos_embed.shape} to {target_pos_embed.shape}...")

            # Separate CLS token and patch embeddings
            cls_pos_embed = pretrained_pos_embed[:, 0:1, :]  # (1, 1, hidden_size)
            patch_pos_embed = pretrained_pos_embed[:, 1:, :]  # (1, num_patches, hidden_size)

            # Get source and target grid sizes
            src_num_patches = patch_pos_embed.shape[1]
            tgt_num_patches = target_pos_embed.shape[1] - 1  # Subtract CLS token

            # Reshape to 2D grid for interpolation
            src_grid_size = int(src_num_patches ** 0.5)
            tgt_grid_size = int(tgt_num_patches ** 0.5)

            hidden_size = patch_pos_embed.shape[2]
            patch_pos_embed = patch_pos_embed.reshape(1, src_grid_size, src_grid_size, hidden_size)
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, hidden_size, H, W)

            # Interpolate
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                size=(tgt_grid_size, tgt_grid_size),
                mode='bicubic',
                align_corners=False
            )

            # Reshape back
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, H, W, hidden_size)
            patch_pos_embed = patch_pos_embed.reshape(1, -1, hidden_size)  # (1, num_patches, hidden_size)

            # Concatenate with CLS token
            interpolated_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
            lrp_state_dict['dinov2.embeddings.position_embeddings'] = interpolated_pos_embed
            print(f"Position embeddings interpolated successfully!")

    # Load state dict (with strict=False to handle minor mismatches)
    missing_keys, unexpected_keys = lrp_model.load_state_dict(lrp_state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5

    print("Weights loaded successfully!")
    return lrp_model


def dinov2_base_imagenet1k_1layer_lrp(pretrained=True, num_classes=1000, checkpoint_path=None, **kwargs):
    """
    Load DINOv2-base model with LRP support.

    Model configuration:
    - Image size: 224x224
    - Patch size: 14x14 (results in 16x16 patch grid)
    - Hidden size: 768
    - Num layers: 12
    - Num heads: 12
    - MLP ratio: 4.0
    - Num classes: 1000 (ImageNet)

    Args:
        pretrained: If True, load weights from HuggingFace hub
        num_classes: Number of output classes (default: 1000)
        checkpoint_path: Ignored (for API compatibility with other models)
        **kwargs: Additional arguments for model configuration

    Returns:
        Dinov2ForImageClassification model with LRP capabilities
    """
    # checkpoint_path is ignored - DINOv2 loads from HuggingFace
    model = Dinov2ForImageClassification(
        image_size=224,
        patch_size=14,
        num_channels=3,
        num_classes=num_classes,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layerscale_value=1.0,
    )

    if pretrained:
        model = load_dinov2_pretrained_weights(
            model,
            hf_model_name='facebook/dinov2-base-imagenet1k-1-layer'
        )

    return model
