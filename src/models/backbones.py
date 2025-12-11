import torch.nn as nn
import torchvision.models as models

def build_backbone(name: str, pretrained: bool = True):

    name = name.lower()

    if name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        feat_dim = model.fc.in_features     # 2048
        model.fc = nn.Identity()            # remove classifier
        return model, feat_dim

    elif name == "densenet121":
        model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        feat_dim = model.classifier.in_features
        model.classifier = nn.Identity()
        return model, feat_dim

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        feat_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feat_dim

    else:
        raise ValueError(f"Unsupported backbone: {name}")
