from __future__ import absolute_import


import torch
import torch.nn as nn


__all__ = ["AlexNetV1", "AlexNetV2", "AlexNetV3", "RecurrentAlex"]


class _BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs
        )


class _AlexNet(nn.Module):
    def forward(self, x, search=False, reset_hidden=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1), _BatchNorm2d(384), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, 3, 1, groups=2))


class AlexNetV2(_AlexNet):
    output_stride = 4

    def __init__(self):
        super(AlexNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1), _BatchNorm2d(384), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(nn.Conv2d(384, 32, 3, 1, groups=2))


class AlexNetV3(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1), _BatchNorm2d(768), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1), _BatchNorm2d(768), nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(nn.Conv2d(768, 512, 3, 1), _BatchNorm2d(512))


class RecurrentAlex(AlexNetV1):
    def __init__(self):
        print("init")

        super(RecurrentAlex, self).__init__()

        self.transform = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=3,
                kernel_size=127,
                stride=6,
                padding=1,
            ),
            _BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        self.last = torch.zeros(1, 3, 239, 239).to("cuda")

    def forward(self, x, search=False, reset_hidden=None):
        if x.dim() != 4:
            x = torch.unsqueeze(x, 0)

        batch_size = x.size(dim=0)

        if search:
            transformed_images = torch.zeros(batch_size, 256, 20, 20)

            for i in range(batch_size):
                single_image = torch.unsqueeze(x[i], 0)

                if reset_hidden[i]:
                    self.last.zero_

                assert not torch.isnan(self.last).any(), "Last contains NaNs"

                single_image = single_image + self.last
                single_image = torch.clamp(single_image, min=0, max=255)

                assert not torch.isnan(single_image).any(), "single_image contains NaNs"

                single_image = super().forward(single_image)

                assert not torch.isnan(
                    single_image
                ).any(), "output from single_image contains NaNs"

                self.last = self.transform(single_image).detach()

                transformed_images[i] = torch.squeeze(single_image, 0)

            x = transformed_images
        else:
            x = super(RecurrentAlex, self).forward(x)

        return x
