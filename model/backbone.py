import torch
from torchvision import models


class symmetric_backbone(torch.nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        if name == 'resnet18':
            features = models.resnet18(pretrained=pretrained)
            features_d = models.resnet18(pretrained=pretrained)

            self.conv1 = features.conv1
            self.bn1 = features.bn1
            self.relu = features.relu
            self.maxpool1 = features.maxpool
            self.layer1 = features.layer1
            self.layer2 = features.layer2
            self.layer3 = features.layer3
            self.layer4 = features.layer4

            self.conv1_d = features_d.conv1
            self.bn1_d = features_d.bn1
            self.relu_d = features_d.relu
            self.maxpool1_d = features_d.maxpool
            self.layer1_d = features_d.layer1
            self.layer2_d = features_d.layer2
            self.layer3_d = features_d.layer3
            self.layer4_d = features_d.layer4
        elif name == 'resnet101':
            features = models.resnet101(pretrained=pretrained)
            features_d = models.resnet101(pretrained=pretrained)

            self.conv1 = features.conv1
            self.bn1 = features.bn1
            self.relu = features.relu
            self.maxpool1 = features.maxpool
            self.layer1 = features.layer1
            self.layer2 = features.layer2
            self.layer3 = features.layer3
            self.layer4 = features.layer4

            self.conv1_d = features_d.conv1
            self.bn1_d = features_d.bn1
            self.relu_d = features_d.relu
            self.maxpool1_d = features_d.maxpool
            self.layer1_d = features_d.layer1
            self.layer2_d = features_d.layer2
            self.layer3_d = features_d.layer3
            self.layer4_d = features_d.layer4
        # add mobilenet
        elif name == 'mobilenet':
            features = models.mobilenet_v2(pretrained=pretrained)
            features_d = models.mobilenet_v2(pretrained=pretrained)

            self.layer0 = features.features[0:2]  # 1/2
            self.layer1 = features.features[2:4]  # 1/4
            self.layer2 = features.features[4:7]  # 1/8
            self.layer3 = features.features[7:14]  # 1/16
            self.layer4 = features.features[14:18]  # 1/32

            self.layer0_d = features_d.features[0:2]
            self.layer1_d = features_d.features[2:4]
            self.layer2_d = features_d.features[4:7]
            self.layer3_d = features_d.features[7:14]
            self.layer4_d = features_d.features[14:18]
        else:
            print('Error: unspported backbone \n')

    def forward(self, input_rgb, input_depth, name):
        if name == 'resnet18' or name == 'resnet101':
            # Symmetric Network
            x = self.conv1(input_rgb)
            x = self.relu(self.bn1(x))
            feature0 = self.maxpool1(x)

            feature1 = self.layer1(feature0)  # 1 / 4
            feature2 = self.layer2(feature1)  # 1 / 8
            feature3 = self.layer3(feature2)  # 1 / 16
            feature4 = self.layer4(feature3)  # 1 / 32

            y = self.conv1_d(input_depth)
            y = self.relu_d(self.bn1_d(y))
            feature0_d = self.maxpool1_d(y)

            feature1_d = self.layer1_d(feature0_d)  # 1 / 4
            feature2_d = self.layer2_d(feature1_d)  # 1 / 8
            feature3_d = self.layer3_d(feature2_d)  # 1 / 16
            feature4_d = self.layer4_d(feature3_d)  # 1 / 32
        elif name == 'mobilenet':
            feature0 = self.layer0(input_rgb)

            feature1 = self.layer1(feature0)  # 1 / 4
            feature2 = self.layer2(feature1)  # 1 / 8
            feature3 = self.layer3(feature2)  # 1 / 16
            feature4 = self.layer4(feature3)  # 1 / 32

            feature0_d = self.layer0_d(input_depth)

            feature1_d = self.layer1_d(feature0_d)  # 1 / 4
            feature2_d = self.layer2_d(feature1_d)  # 1 / 8
            feature3_d = self.layer3_d(feature2_d)  # 1 / 16
            feature4_d = self.layer4_d(feature3_d)  # 1 / 32

        return feature1, feature2, feature3, feature4, feature1_d, feature2_d, feature3_d, feature4_d
