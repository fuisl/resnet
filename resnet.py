import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Any

class ResNetGlobalMaxPool(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(512, 3)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x.squeeze(-1).squeeze(-1))

        return x

def resnet34_global_max_pool():
    return ResNetGlobalMaxPool(BasicBlock, [3, 4, 6, 3], num_classes=3)

if __name__ == "__main__":
    resnet = ResNetGlobalMaxPool(BasicBlock, [3, 4, 6, 3], num_classes=3)
    input_tensor = torch.randn(1, 1, 300, 300)
    out = resnet(input_tensor)
    print(out.shape)