import timm
from torch import nn
from torch.nn import functional as F


class ResNet50_v4(nn.Module):
    def __init__(
        self, arch, layer=None, dropout_prob=0.6, pretrained=False, testing=True
    ):
        super().__init__()
        self.testing = testing
        self.arch = timm.create_model(arch, pretrained=pretrained)
        self.layer = layer
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(307200, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def forward_features(self, x):
        x = self.arch.conv1(x)
        x = self.arch.bn1(x)
        x = self.arch.act1(x)
        x = self.arch.maxpool(x)

        x1 = self.arch.layer1(x)
        x2 = self.arch.layer2(x1)
        x3 = self.arch.layer3(x2)
        x4 = self.arch.layer4(x3)

        if self.testing:
            features = {}
            for layer in self.layer:
                if layer == 'layer1':
                    features[layer] = x1
                elif layer == 'layer2':
                    features[layer] = x2
                elif layer == 'layer3':
                    features[layer] = x3
                elif layer == 'layer4':
                    features[layer] = x4
            return features
        return x4

    def forward_head(self, x):
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.last_linear(x)
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.testing:
            return x
        x = self.forward_head(x)
        return x


if __name__ == '__main__':
    import torch

    # Extraction PatchCore
    model = ResNet50_v4(layer=['layer2', 'layer3'], pretrained=False, testing=True)
    # Training
    # model = ResNet50_v4(pretrained=True, testing=False)
    input = torch.rand((2, 3, 300, 450))

    output = model(input)
    # print(output.shape)
    print(output['layer2'].shape)
    print(output['layer3'].shape)
