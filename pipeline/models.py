import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super(ResNetClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=False)
        self.base_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base_model.maxpool = nn.Identity()  # Remove downsampling
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, n_classes)

    def forward(self, x):
        return self.base_model(x)
