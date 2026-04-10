import argparse
import torch.nn as nn
import torch

from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights

# jersey visible is 2 mean invisible and visible
# jersey number is 20 mean 20 numbers
# jersey color is 2 mean 2 colors
class player_classifier(nn.Module):
    def __init__(self, n_jersey_numbers=20, n_jersey_color=2, number_visible=2, orig_model_path=None):
        super(player_classifier, self).__init__()
        self.backbone = efficientnet_v2_s()

        if orig_model_path is not None:
            self.backbone.load_state_dict(torch.load(orig_model_path))
        else:
            self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        in_features = self.backbone.classifier[1].in_features

        # Delete old classification layer
        del self.backbone.classifier

        # Initialize new FC layers
        self.backbone.number_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features, n_jersey_numbers)
        )

        self.backbone.color_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, n_jersey_color)
        )

        self.backbone.visible_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, number_visible)
        )

    def forward(self, x):
        x = self.backbone.features(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x1 = self.backbone.visible_head(x)
        x2 = self.backbone.number_head(x)
        x3 = self.backbone.color_head(x)

        return x1, x2, x3


