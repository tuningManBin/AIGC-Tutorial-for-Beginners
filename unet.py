"""
U-Net

Reference:
    [1] Ronneberger O, Fischer P, Brox T.
        U-net: Convolutional networks for biomedical image segmentation[C]//
        Medical image computing and computer-assisted intervention–MICCAI
        2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18.
        Springer International Publishing, 2015: 234-241.

    [2] https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """
    This module defines the basic convolutional block of U-Net.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        """
        Arguments:
            in_channels (int): the number of channels in the input tensor.
            out_channels (int): the number of channels in the output tensor.
        """
        super(ConvBlock, self).__init__()
        # conv + bn --> bias = False
        # kernel_size = 3, stride = 1, padding = 1 : [batch, c, h, w] --> [batch, c', h, w]
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.double_conv(x)
        return x


class DownSampling(nn.Module):
    """
    This module defines down-sampling.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        """
        Arguments:
            in_channels (int): the number of channels in the input tensor.
            out_channels (int): the number of channels in the output tensor.
        """
        super(DownSampling, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ConvBlock(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x


class UpSampling(nn.Module):
    """
    This module defines up-sampling.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        """
        Arguments:
            in_channels (int): the number of channels in the input tensor.
            out_channels (int): the number of channels in the output tensor.
        """
        super(UpSampling, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        self.conv = ConvBlock(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self,
                x: torch.Tensor,
                res: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UpSampling block.

        Arguments:
            x (torch.Tensor): the input tensor.
            res (torch.Tensor): the residual tensor from the encoder.

        Returns:
            x (torch.Tensor): the output tensor obtained after up-sampling and concatenation with the residual tensor.
        """
        x = self.deconv(x)
        x = torch.cat([x, res], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    This module is the main body of U-Net.
    """
    def __init__(self,
                 in_channels: int,
                 scale_factor: int,
                 depth: int,
                 input_size: int = 32,
                 out_channels: Optional[int] = None,
                 dataset: Optional[str] = None) -> None:
        """
        Arguments:
            in_channels (int): the number of channels in the input tensor.
            scale_factor (int): the factor by which the number of channels in the input tensor is multiplied to
                                obtain the number of channels in the output tensor.
            depth (int): depth of Encoder and Decoder.
            input_size (int): the size of the input tensor.
            out_channels (Optional[int]): the number of channels in the output tensor.
            dataset (Optional[str]): a control variable used for normalizing the data.
        """
        super(UNet, self).__init__()
        out_channels = out_channels or in_channels
        self.input_size = input_size
        self.dataset = dataset
        if self.dataset == "cifar10":
            self.input_size = 32
        elif self.dataset == "mnist":
            self.input_size = 28

        # --- Encoder & Decoder
        self.depth = depth
        self.encoder = nn.ModuleList([])
        self.encoder.append(ConvBlock(in_channels=in_channels, out_channels=in_channels))
        self.decoder = nn.ModuleList([])
        for i in range(self.depth):
            self.encoder.append(
                DownSampling(in_channels=in_channels * scale_factor**i, out_channels=in_channels * scale_factor**(i+1))
            )
            self.decoder.append(
                UpSampling(in_channels=in_channels * scale_factor**(i+1), out_channels=in_channels * scale_factor**i)
            )

        self.output_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(1, 1), stride=(1, 1))

    def forward_loss(self,
                     reconstruction: torch.Tensor,
                     ground_truth: torch.Tensor,
                     reduction: str = "mean") -> torch.Tensor:
        """
        This function calculates the mean squared error loss.

        Arguments:
            reconstruction (torch.Tensor): the output tensor of U-Net.
            ground_truth (torch.Tensor): the ground truth tensor to compare with the reconstructed output tensor.
            reduction (str): specifies the reduction to apply to the output.
                             "none" | "mean" | "sum"
                             - "none": no reduction will be applied.
                             - "mean": the sum of the output will be divided by the number of elements in the output.
                             - "sum": the output will be summed.

        Returns:
            loss (torch.Tensor): the calculated mean squared error loss.
        """
        # --- Preprocessing
        ground_truth = F.interpolate(ground_truth, size=(self.input_size, self.input_size), mode="bilinear",
                                     align_corners=True)
        if self.dataset == "cifar10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=ground_truth.device)
            mean = mean.expand(ground_truth.shape[0], -1)
            mean = mean.unsqueeze(-1).unsqueeze(-1)

            std = torch.tensor([0.2471, 0.2435, 0.2616], device=ground_truth.device)
            std = std.expand(ground_truth.shape[0], -1)
            std = std.unsqueeze(-1).unsqueeze(-1)

            ground_truth = torch.clamp(ground_truth, min=0.0, max=1.0)
            ground_truth = (ground_truth - mean) / std

        elif self.dataset == "mnist":
            mean = torch.tensor([0.1307], device=ground_truth.device)
            mean = mean.expand(ground_truth.shape[0], -1)
            mean = mean.unsqueeze(-1).unsqueeze(-1)

            std = torch.tensor([0.3081], device=ground_truth.device)
            std = std.expand(ground_truth.shape[0], -1)
            std = std.unsqueeze(-1).unsqueeze(-1)

            ground_truth = torch.clamp(ground_truth, min=0.0, max=1.0)
            ground_truth = (ground_truth - mean) / std

        # --- MSE Loss
        loss = nn.MSELoss(reduction=reduction)(reconstruction, ground_truth)
        return loss

    def forward_get_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        This function extracts the latent feature vectors.

        Arguments:
            images (torch.Tensor): the input tensor with shape [batch, in_channels, height, width].

        Returns:
            res_encoder[-1].flatten(1) (torch.Tensor): the latent feature vectors obtained from the last layer of the encoder.
        """
        # --- Preprocessing
        images = F.interpolate(images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=True)
        if self.dataset == "cifar10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device)
            mean = mean.expand(images.shape[0], -1)
            mean = mean.unsqueeze(-1).unsqueeze(-1)

            std = torch.tensor([0.2471, 0.2435, 0.2616], device=images.device)
            std = std.expand(images.shape[0], -1)
            std = std.unsqueeze(-1).unsqueeze(-1)

            images = torch.clamp(images, min=0.0, max=1.0)
            images = (images - mean) / std

        elif self.dataset == "mnist":
            mean = torch.tensor([0.1307], device=images.device)
            mean = mean.expand(images.shape[0], -1)
            mean = mean.unsqueeze(-1).unsqueeze(-1)

            std = torch.tensor([0.3081], device=images.device)
            std = std.expand(images.shape[0], -1)
            std = std.unsqueeze(-1).unsqueeze(-1)

            images = torch.clamp(images, min=0.0, max=1.0)
            images = (images - mean) / std

        res_encoder = []
        x = self.encoder[0](images)
        res_encoder.append(x)
        for i in range(self.depth):
            x = self.encoder[i+1](x)
            res_encoder.append(x)
        return res_encoder[-1].flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert self.input_size % (2 ** self.depth) == 0, \
            f"input_size {self.input_size} should be divided by 2**depth {2 ** self.depth}."

        # --- Preprocessing
        images = F.interpolate(images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=True)
        if self.dataset == "cifar10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device)
            mean = mean.expand(images.shape[0], -1)
            mean = mean.unsqueeze(-1).unsqueeze(-1)

            std = torch.tensor([0.2471, 0.2435, 0.2616], device=images.device)
            std = std.expand(images.shape[0], -1)
            std = std.unsqueeze(-1).unsqueeze(-1)

            images = torch.clamp(images, min=0.0, max=1.0)
            images = (images - mean) / std

        elif self.dataset == "mnist":
            mean = torch.tensor([0.1307], device=images.device)
            mean = mean.expand(images.shape[0], -1)
            mean = mean.unsqueeze(-1).unsqueeze(-1)

            std = torch.tensor([0.3081], device=images.device)
            std = std.expand(images.shape[0], -1)
            std = std.unsqueeze(-1).unsqueeze(-1)

            images = torch.clamp(images, min=0.0, max=1.0)
            images = (images - mean) / std

        res_encoder = []
        x = self.encoder[0](images)
        res_encoder.append(x)
        for i in range(self.depth):
            x = self.encoder[i+1](x)
            res_encoder.append(x)

        for j in range(self.depth):
            x = self.decoder[-(j+1)](x, res_encoder[-(j+1+1)])

        images = self.output_conv(x)
        return images
    

def unet_cifar10(in_channels=3,
                 scale_factor=2,
                 depth=3,
                 input_size=32,
                 out_channels=3,
                 dataset="cifar10") -> nn.Module:
    """
    A version that deploys U-Net to CIFAR-10.
    """
    return UNet(in_channels=in_channels,
                scale_factor=scale_factor,
                depth=depth,
                input_size=input_size,
                out_channels=out_channels,
                dataset=dataset)


def unet_mnist(in_channels=1,
               scale_factor=2,
               depth=2,
               input_size=28,
               out_channels=1,
               dataset="mnist") -> nn.Module:
    """
    A version that deploys U-Net to MNIST.
    """
    return UNet(in_channels=in_channels,
                scale_factor=scale_factor,
                depth=depth,
                input_size=input_size,
                out_channels=out_channels,
                dataset=dataset)


if __name__ == "__main__":
    net0 = unet_cifar10()
    test_sample0 = torch.rand([2, 3, 32, 32])
    pred0 = net0(test_sample0)
    print(pred0.shape)

    loss0 = net0.forward_loss(pred0, torch.rand([2, 3, 32, 32]))
    print(loss0)

    pred0 = net0.forward_get_latent(test_sample0)
    print(pred0.shape)

    net1 = unet_mnist()
    test_sample1 = torch.rand([2, 1, 28, 28])
    pred1 = net1(test_sample1)
    print(pred1.shape)

    loss1 = net1.forward_loss(pred1, torch.rand([2, 1, 28, 28]))
    print(loss1)

    pred1 = net1.forward_get_latent(test_sample1)
    print(pred1.shape)

