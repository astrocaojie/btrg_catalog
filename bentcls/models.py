import torch.nn as nn
import torchvision.models as models

def build_model(model_name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """
    Returns a binary classifier producing logit (B,1).
    Input is 3-channel (dataset replicates single-channel to 3ch).
    """
    name = model_name.lower()

    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m

    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m

    if name in ("efficientnet_b0", "effb0", "efficientnet-b0"):
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        return m

    if name in ("convnext_tiny", "convnext-tiny"):
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        return m

    if name in ("vit_b_16", "vit-base", "vit"):
        # ViT 对小样本更容易不稳，但你既然要对比，我们也支持
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, 1)
        return m

    raise ValueError(f"Unknown model_name: {model_name}")
