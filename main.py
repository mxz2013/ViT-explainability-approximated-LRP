from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import argparse
from dataclasses import dataclass
from typing import Callable
from utils.CLS2IDX import CLS2IDX

from baselines.ViT.ViT_LRP import vit_base_patch16_224, mae_vit_base_patch16_224
from baselines.ViT.ViT_explanation_generator import LRP
from baselines.ViT.DINOv2_LRP import dinov2_base_imagenet1k_1layer_lrp
from baselines.ViT.CLIP_LRP import clip_vit_base_patch16_224

from logging import basicConfig, getLogger

logger = getLogger(__name__)
basicConfig(level="INFO")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelConfig:
    """Configuration for a Vision Transformer model."""

    factory: Callable
    patch_size: int
    image_size: int = 224
    model_path: str = ""
    num_classes: int = 1000
    normalize_mean: list[float] = (0.5, 0.5, 0.5)
    normalize_std: list[float] = (0.5, 0.5, 0.5)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "vit_base_patch16_224": ModelConfig(
        factory=vit_base_patch16_224,
        patch_size=16,
        num_classes=1000,
    ),
    "mae_vit_base_patch16_224": ModelConfig(
        factory=mae_vit_base_patch16_224,
        patch_size=16,
        model_path="mae/output_dir_small_batch/checkpoint-44.pth",
        num_classes=300,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    ),
    "dinov2_base_imagenet1k_1layer_lrp": ModelConfig(
        factory=dinov2_base_imagenet1k_1layer_lrp,
        patch_size=14,
        num_classes=1000,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    ),
    "clip_vit_base_patch16_224": ModelConfig(
        factory=clip_vit_base_patch16_224,
        patch_size=16,
        model_path="clip/output_dir_small_batch/checkpoint-37.pth",
        num_classes=300,
        normalize_mean=[0.48145466, 0.4578275, 0.40821073],
        normalize_std=[0.26862954, 0.26130258, 0.27577711],
    ),
}


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    """
    Docstring for show_cam_on_image

    :param img: Description
    :param mask: Description
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(
    original_image,
    attribution_generator,
    config: ModelConfig,
    method="transformer_attribution",
    class_index=None,
    use_thresholding=False,
):
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).to(device), method=method, index=class_index
    ).detach()

    # Compute grid size from image_size and patch_size
    grid_size = config.image_size // config.patch_size
    transformer_attribution = transformer_attribution.reshape(
        1, 1, grid_size, grid_size
    )
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=config.patch_size, mode="bilinear"
    )
    transformer_attribution = (
        transformer_attribution.reshape(config.image_size, config.image_size)
        .data.cpu()
        .numpy()
    )
    transformer_attribution = (
        transformer_attribution - transformer_attribution.min()
    ) / (transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
        transformer_attribution = transformer_attribution * 255
        transformer_attribution = transformer_attribution.astype(np.uint8)
        ret, transformer_attribution = cv2.threshold(
            transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        transformer_attribution[transformer_attribution == 255] = 1

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (
        image_transformer_attribution - image_transformer_attribution.min()
    ) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    return vis


def get_top_classes(predictions, topk=5):
    """
    Get top-k predicted classes with probabilities.
    :param predictions: model output logits, shape (1, num_classes)
    :param topk: Number of top classes to return
    :return: List of tuples (class_index, class_name, probability)
    """
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(topk, dim=1)[1][0].tolist()
    results = []
    for cls_idx in class_indices:
        class_name = CLS2IDX[cls_idx]
        class_prob = prob[0, cls_idx].item()
        results.append((cls_idx, class_name, class_prob))
        logger.info(
            f"ID {cls_idx}, Class: {class_name} , Probability: {class_prob:.4f}"
        )
    return results


def main(model_name: str, image_path: str, method: str, use_thresholding: bool = False):
    """
    Docstring for main

    :param model_name: Description
    :type model_name: str
    :param image_path: Description
    :type image_path: str
    :param method: Description
    :type method: str
    :param use_thresholding: Description
    :type use_thresholding: bool
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    config = MODEL_REGISTRY[model_name]

    # be careful about normalization values for different models
    normalize = transforms.Normalize(
        mean=config.normalize_mean, std=config.normalize_std
    )

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    # Initialize model
    logger.info(f"Loading model: {model_name}")
    model = config.factory(
        checkpoint_path=config.model_path, num_classes=config.num_classes
    ).to(device)
    model.eval()
    attribution_generator = LRP(model)

    # Get predictions and top-3 classes
    output = model(image_tensor.unsqueeze(0).to(device))
    top_classes = get_top_classes(output, topk=5)

    # Create 2x2 plot: original + top-3 attributions
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    fig.suptitle(f"Model: {model_name} | Method: {method}", fontsize=12)
    fig.tight_layout()

    # Original image
    axs[0, 0].set_title("Original Image")
    axs[0, 0].imshow(image)
    axs[0, 0].axis("off")

    # Top-5 class attributions
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for i, (cls_idx, cls_name, prob) in enumerate(top_classes):
        row, col = positions[i]
        vis = generate_visualization(
            image_tensor,
            attribution_generator,
            config,
            method=method,
            class_index=cls_idx,
            use_thresholding=use_thresholding,
        )
        axs[row, col].set_title(f"#{i + 1}: {cls_name.split(',')[0]} ({prob:.2%})")
        axs[row, col].imshow(vis)
        axs[row, col].axis("off")

    output_path = f"outputs/{model_name}_{method}.png"
    plt.savefig(output_path)
    logger.info(f"Saved visualization to {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ViT Explainability with LRP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available models:\n  " + "\n  ".join(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use for explainability",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="input_images/catdog.png",
        help="Path to input image",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="transformer_attribution",
        choices=[
            "transformer_attribution",
            "rollout",
            "full",
            "last_layer",
            "last_layer_attn",
        ],
        help="Attribution method to use",
    )
    parser.add_argument(
        "--apply_threshold",
        action="store_true",
        help="Apply Otsu thresholding to attribution",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Arguments: {args}")
    main(
        model_name=args.model,
        image_path=args.image,
        method=args.method,
        use_thresholding=args.apply_threshold,
    )
