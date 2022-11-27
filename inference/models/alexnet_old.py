"""
This file contains variants of our AlexnetGrasp model,
each with slightly modified architectures or weight initialization.
The current best model is trained using the class: 'AlexnetMap_v5'.

Alexnet:
    - Conventional Alexnet architecture with reduced no. of channels
      and fc-layers
myAlexNet:
    - Modified Alexnet architecture with added BatchNorm layers
PretrainedAlexnet:
    - Exact copy of Alexnet architecture with reduced fc layer
      (removed dropout layer proven to have better performance)
    - First two layers loaded with Imagenet pretraining weights
AlexnetMap:
    - Alexnet first two layers pretrained and frozen
    - Pseudo alexnet architecture for the rest of encoder
    - Decoder uses deconv with parameters mirroring encoder
    - Outputs a map with same dimension as input, with each pixel
            containing 6 values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import alexnet


class AlexNet(nn.Module):
    def __init__(self, input_channels=3, channel_size=16, n_cls=5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, channel_size, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(channel_size, 2*channel_size, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(2*channel_size, 4*channel_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*channel_size, 8*channel_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*channel_size, 16*channel_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=.2),
            nn.Linear(16*channel_size * 6 * 6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class myAlexNet(nn.Module):
    def __init__(self, input_channels=4, channel_size=16, n_cls=5):
        super(myAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(channel_size, 2*channel_size, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2*channel_size)
        self.conv3 = nn.Conv2d(2*channel_size, 4*channel_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(4*channel_size)
        self.conv4 = nn.Conv2d(4*channel_size, 8*channel_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(8*channel_size)
        self.conv5 = nn.Conv2d(8*channel_size, 16*channel_size, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16*channel_size)
        self.fc1 = nn.Linear(16*channel_size, 5)
        #self.fc2 = nn.Linear(64, 5)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x=F.adaptive_avg_pool2d(x,1)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        return x


class PretrainedAlexnet(nn.Module):
    def __init__(self, n_cls=5):
        super(PretrainedAlexnet, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.features = pretrained_alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_cls)
        )

        for i, param in enumerate(self.features.parameters()):
            if i < 4:
                param.requires_grad = False

        for i, m in enumerate(self.features.modules()):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and i > 4:
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


class PretrainedAlexnetv2(nn.Module):
    def __init__(self, n_cls=5):
        super(PretrainedAlexnetv2, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features = nn.Sequential(
            nn.Conv2d(192+48, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            #nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((12, 12))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 12 * 12, 64),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.2),
            #nn.Linear(256, 64),
            #nn.ReLU(inplace=True),
            nn.Linear(64, n_cls)
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False

        # xavier initialization for depth feature extractor
        for m in self.d_features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PretrainedAlexnetv3(nn.Module):
    def __init__(self, n_cls=5):
        super(PretrainedAlexnetv3, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = pretrained_alexnet.features[:6]
        self.features = nn.Sequential(
            nn.Conv2d(192+192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            #nn.Dropout(p=0.2),
            nn.Linear(32 * 118 * 118, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64), #
            nn.ReLU(inplace=True), #
            nn.Linear(64, n_cls)
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False
        for param in self.d_features.parameters():
            param.requires_grad = False

        # xavier initialization for depth feature extractor
        for m in self.d_features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.d_features.parameters():
            param.requires_grad = True


class PretrainedAlexnetv4(nn.Module):
    def __init__(self, n_cls=5):
        super(PretrainedAlexnetv4, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features = nn.Sequential(
            nn.Conv2d(192+48, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 118 * 118, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_cls)
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False

        # xavier initialization for depth feature extractor
        for m in self.d_features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True


class PretrainedAlexnetv5(nn.Module):
    def __init__(self, n_cls=5):
        super(PretrainedAlexnetv5, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:3]
        self.d_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features = nn.Sequential(
            nn.Conv2d(64+16, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 13 * 13, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_cls)
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False

        # xavier initialization for depth feature extractor
        for m in self.d_features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True


class AlexnetMap(nn.Module):
    def __init__(self, n_cls=5):
        super(AlexnetMap, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features = nn.Sequential(
            nn.Conv2d(192+48, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 6, kernel_size=11, stride=4, output_padding=1)
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False

        # xavier initialization for depth feature extractor
        for m in self.d_features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)

        return x

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True


class AlexnetMap_v2(nn.Module):
    def __init__(self, n_cls=5):
        super(AlexnetMap_v2, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features = nn.Sequential(
            nn.Conv2d(192+48, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.grasp = nn.ConvTranspose2d(32, 5, kernel_size=11, stride=4, output_padding=1)
        self.confidence = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=11, stride=4, output_padding=1),
            nn.Sigmoid()
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False

        # xavier initialization for depth feature extractor
        for m in self.d_features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        grasp = self.grasp(x)
        confidence = self.confidence(x)
        out = torch.cat((grasp, confidence), dim=1)

        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True


class AlexnetMap_v3(nn.Module):
    def __init__(self, n_cls=5):
        super(AlexnetMap_v3, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features = nn.Sequential(
            nn.Conv2d(192+48, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.grasp = nn.ConvTranspose2d(64, 5, kernel_size=11, stride=4, output_padding=1)
        self.confidence = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=11, stride=4, output_padding=1),
            nn.Sigmoid()
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False

        # xavier initialization for depth feature extractor
        for m in self.d_features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        grasp = self.grasp(x)
        confidence = self.confidence(x)
        out = torch.cat((grasp, confidence), dim=1)

        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True


class AlexnetMap_v4(nn.Module):
    def __init__(self, n_cls=5):
        super(AlexnetMap_v4, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.features = nn.Sequential(
            nn.Conv2d(192+48, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # Dropout added to v4.3 or later
            nn.Dropout2d(p=.2,),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # Dropout added to v4.3 or later
            nn.Dropout2d(p=.2,),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.grasp = nn.ConvTranspose2d(64, 5, kernel_size=11, stride=4, output_padding=1)
        self.confidence = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=11, stride=4, output_padding=1),
            nn.Sigmoid()
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False

        # xavier initialization for depth feature extractor
        for m in self.d_features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        grasp = self.grasp(x)
        confidence = self.confidence(x)
        out = torch.cat((grasp, confidence), dim=1)

        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True


class AlexnetMap_v5(nn.Module):
    """v5.3 onwards has 128 channles in the first .features layer."""
    def __init__(self, n_cls=5):
        super(AlexnetMap_v5, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = pretrained_alexnet.features[:6]
        self.features = nn.Sequential(
            nn.Conv2d(192+192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.grasp = nn.ConvTranspose2d(32, 5, kernel_size=11, stride=4, output_padding=1)

        self.confidence = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=11, stride=4, output_padding=1),
            nn.Sigmoid()
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False
        for param in self.d_features.parameters():
            param.requires_grad = False

        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        grasp = self.grasp(x)
        confidence = self.confidence(x)
        out = torch.cat((grasp, confidence), dim=1)

        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    # Mistake: didn't unfree d_features
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True


class AlexnetMap_v6(nn.Module):
    """v5.3 onwards has 128 channles in the first .features layer."""
    def __init__(self, n_cls=5):
        super(AlexnetMap_v6, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.rgb_features = pretrained_alexnet.features[:6]
        self.d_features = pretrained_alexnet.features[:6]
        self.features = nn.Sequential(
            nn.Conv2d(192+192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.grasp = nn.ConvTranspose2d(64, 5, kernel_size=5, stride=2, output_padding=1)
        self.confidence = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, output_padding=1),
            nn.Sigmoid()
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False
        for param in self.d_features.parameters():
            param.requires_grad = False

        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        grasp = self.grasp(x)
        confidence = self.confidence(x)
        out = torch.cat((grasp, confidence), dim=1)

        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True