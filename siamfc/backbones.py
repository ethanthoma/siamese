from __future__ import absolute_import


import torch
import torch.nn as nn


__all__ = [
    "AlexNetV1",
    "AlexNetV2",
    "AlexNetV3",
    "RecurrentAlexAdd",
    "RecurrentAlexMul",
]


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


class RecurrentAlexAdd(AlexNetV1):
    def __init__(self):
        print("init")

        super().__init__()

        self.conv5 = nn.Sequential(
            nn.Sequential(nn.Conv2d(384, 256, 3, 1, groups=2)),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

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
        if not search:
            batch_size = x.size(dim=0)

            transformed_images = torch.zeros(batch_size, 256, 20, 20)

            for i in range(batch_size):
                single_image = torch.unsqueeze(x[i], 0)

                if reset_hidden is not None and reset_hidden[i]:
                    self.last.zero_()
                else:
                    assert not torch.isnan(self.last).any(), "Last contains NaNs"
                    single_image = single_image + self.last
                    assert not torch.isnan(
                        single_image
                    ).any(), "single_image contains NaNs"

                single_image = super().forward(single_image)

                assert not torch.isnan(
                    single_image
                ).any(), "output from single_image contains NaNs"

                self.last = self.transform(single_image).detach()

                transformed_images[i] = torch.squeeze(single_image, 0)

            x = transformed_images
        else:
            if reset_hidden is not None:
                self.last.zero_()
            x = super().forward(x)

        return x

    def reset_hidden(self):
        self.last.zero_()


class RecurrentAlexMul(AlexNetV1):
    def __init__(self):
        print("init")

        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                groups=2,
            ),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                groups=2,
            ),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
            ),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2), mode="bilinear", align_corners=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=96,
                kernel_size=5,
                stride=1,
                groups=2,
            ),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2), mode="bilinear", align_corners=True),
            nn.ConvTranspose2d(
                in_channels=96,
                out_channels=3,
                kernel_size=11,
                stride=2,
            ),
            _BatchNorm2d(3),
            nn.Sigmoid(),
        )

        self.last = None

    def forward(self, x, search=False, reset_hidden=None):
        if not search:
            if x.dim() == 3:
                x = torch.unsqueeze(x, 0)

            batch_size = x.size(dim=0)

            if self.last is None:
                self.last = torch.ones(1, *x.shape[1:]).to(
                    x.get_device() if x.get_device() != -1 else "CPU"
                )

            transformed_images = []

            for i in range(batch_size):
                single_image = torch.unsqueeze(x[i], 0)

                if reset_hidden is not None and reset_hidden[i]:
                    self.last.fill_(1)
                else:
                    assert not torch.isnan(self.last).any(), "Last contains NaNs"
                    single_image = torch.mul(single_image, self.last)
                    assert not torch.isnan(
                        single_image
                    ).any(), "single_image contains NaNs"

                single_image = super().forward(single_image)

                assert not torch.isnan(single_image).any(), "Forward contains NaNs"

                self.last = nn.functional.interpolate(
                    self.deconv(single_image).detach(), size=list(x.shape[2:])
                )

                assert not torch.isnan(single_image).any(), "self.last contains NaNs"

                transformed_images.append(torch.squeeze(single_image, 0))

            x = torch.stack(transformed_images)
        else:
            if self.last is not None and reset_hidden is not None:
                self.last.fill_(1)
            x = super().forward(x)

        return x

    def reset_hidden(self):
        self.last.fill_(1)
