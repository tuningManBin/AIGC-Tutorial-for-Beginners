"""
Vision Transformer

Reference:
    [1] Dosovitskiy A, Beyer L, Kolesnikov A, et al.
        An image is worth 16x16 words: Transformers for image recognition at scale[J].
        arXiv preprint arXiv:2010.11929, 2020.

    [2] https://github.com/asyml/vision-transformer-pytorch
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type


class ParchEmbedding(nn.Module):
    """
    This module converts image pixel patches to embeddings according to the patch size.
    """
    def __init__(self,
                 in_channels: int,
                 input_size: int,
                 patch_size: int,
                 embedding_dim: int,
                 dropout_ratio: float = 0.1) -> None:
        """
        Arguments:
            in_channels (int): the number of channels in the input tensor.
            input_size (int): the size of the input tensor.
            patch_size (int): the size of the pixel patch.
            embedding_dim (int): the dimension of the embedding.
            dropout_ratio (float): the probability of a token being zero.
        """
        super(ParchEmbedding, self).__init__()
        self.project = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                 kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.num_patches = (input_size // patch_size) ** 2 + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim))

        self.drop = nn.Dropout(p=dropout_ratio)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # --- Patch Embedding
        tokens = self.project(images)
        # output shape = [batch_size, embedding_dim, input_size // patch_size, input_size // patch_size]
        tokens = tokens.flatten(2).transpose(1, 2)
        # output shape = [batch_size, num_patches, embedding_dim]

        # --- Position Embedding
        tokens = tokens + self.position_embedding[:, 1:, :]

        # --- Add class token
        cls_token = self.cls_token + self.position_embedding[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        # output shape = [batch_size, num_patches + 1, embedding_dim]
        tokens = self.drop(tokens)
        return tokens


class SelfAttention(nn.Module):
    """
    This module performs Multi-head Self-attention.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 attn_dropout_ratio: float = 0.1,
                 proj_dropout_ratio: float = 0.1) -> None:
        """
        Arguments:
            embedding_dim (int): the dimension of the embedding.
            num_heads (int): the number of self-attention heads.
            attn_dropout_ratio (float): the probability of an element being zero in the attention map.
            proj_dropout_ratio (float): the probability of an element being zero in the projection.
        """
        super(SelfAttention, self).__init__()
        assert embedding_dim % num_heads == 0, \
            f"embedding_dim {embedding_dim} should be divided by num_heads {num_heads}."
        self.norm = nn.LayerNorm(embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_matrices = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)

        self.project = nn.Linear(embedding_dim, embedding_dim)
        self.proj_drop = nn.Dropout(p=attn_dropout_ratio)
        self.attn_drop = nn.Dropout(p=proj_dropout_ratio)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        res = tokens

        batch_size, num_patches = tokens.shape[0], tokens.shape[1]

        tokens = self.norm(tokens)

        qkv = self.qkv_matrices(tokens).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # output shape = [batch_size, num_heads, num_patches, head_dim]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = scores.softmax(dim=-1)
        attn_map = self.attn_drop(attn_map)
        dots = attn_map @ v
        # output shape = [batch_size, num_heads, num_patches, head_dim]
        dots = dots.transpose(1, 2)
        # output shape = [batch_size, num_patches, num_heads, head_dim]
        dots = dots.reshape(batch_size, num_patches, self.embedding_dim)
        # output shape = [batch_size, num_patches, embedding_dim]
        dots = self.project(dots)
        dots = self.proj_drop(dots)

        dots = dots + res
        return dots


class MLP(nn.Module):
    """
    This module defines Multi-layer Perceptron.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 activation: Type[nn.Module] = nn.GELU,
                 dropout_ratio: float = 0.1) -> None:
        """
        Arguments:
            input_dim (int): the dimension of the input tensor.
            hidden_dim (Optional[int]): the dimension of the output tensor in the hidden layer.
            output_dim (Optional[int]): the dimension of the output tensor.
            activation (Type[nn.Module]): the activation function.
            dropout_ratio (float): the probability of an element being zero.
        """
        super(MLP, self).__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim

        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation()
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout_ratio)

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        x = self.norm(x)

        x = self.fc0(x)
        x = self.activation(x)
        x = self.drop(x)

        x = self.fc1(x)
        x = self.drop(x)

        x = x + res
        return x


class BasicBlock(nn.Module):
    """
    This module defines the basic block of Vision Transformer.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 mlp_ratio: int) -> None:
        """
        Arguments:
            embedding_dim (int): the dimension of the embedding.
            num_heads (int): the number of self-attention heads.
            mlp_ratio (int): the ratio of the hidden layer dimension to the input layer dimension.
        """
        super(BasicBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.project = MLP(input_dim=embedding_dim, hidden_dim=embedding_dim * mlp_ratio)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.attention(tokens)
        tokens = self.project(tokens)
        # output shape = [batch_size, num_patches, embedding_dim]
        return tokens


class VisionTransformer(nn.Module):
    """
    This module is the main body of vision transformer.
    """
    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 embedding_dim: int,
                 num_heads: int,
                 depth: int,
                 mlp_ratio: int,
                 logits_dim: int,
                 input_size: int = 32,
                 dataset: Optional[str] = None) -> None:
        """
        Arguments:
            in_channels (int): the number of channels in the input tensor.
            patch_size (int): the size of the pixel patch.
            embedding_dim (int): the dimension of the embedding.
            num_heads (int): the number of self-attention heads.
            depth (int): the number of basic blocks (attention + mlp).
            mlp_ratio (int): the ratio of the hidden layer dimension to the input layer dimension.
            logits_dim (int): the dimension of the output tensor.
            input_size (int): the size of the input tensor.
            dataset (Optional[str]): a control variable used for normalizing the data.
        """
        super(VisionTransformer, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size

        self.dataset = dataset
        if self.dataset == "cifar10":
            self.input_size = 32
        elif self.dataset == "mnist":
            self.input_size = 28

        assert self.input_size % patch_size == 0, \
            f"input_size // pool_stride {input_size} should be divided by patch_size {patch_size}."

        self.embedding = ParchEmbedding(in_channels=in_channels, input_size=input_size,
                                        embedding_dim=embedding_dim, patch_size=patch_size)

        self.blocks = nn.Sequential(*[
            BasicBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        # nn.Sequential(*layers) represents the extraction of elements from layers to form of sequential

        self.cls_project = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, logits_dim)
        )

    def forward_loss(self,
                     logits: torch.Tensor,
                     ground_truth: torch.Tensor,
                     reduction: str = "mean") -> torch.Tensor:
        """
        This function calculates the cross entropy loss.

        Arguments:
            logits (torch.Tensor): the output tensor of Vision Transformer (before applying softmax).
            ground_truth (torch.Tensor): true labels for the input tensor.
            reduction (str): specifies the reduction to apply to the output.
                             "none" | "mean" | "sum"
                             - "none": no reduction will be applied.
                             - "mean": the sum of the output will be divided by the number of elements in the output.
                             - "sum": the output will be summed.

        Returns:
            loss (torch.Tensor): the computed cross entropy loss.
        """
        loss = nn.CrossEntropyLoss(reduction=reduction)(logits, ground_truth)
        return loss

    def forward(self, images: torch.Tensor) -> torch.Tensor:
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

        tokens = self.embedding(images)
        tokens = self.blocks(tokens)
        # output shape = [batch_size, num_patches, embedding_dim]

        cls_token = tokens[:, 0]
        logits = self.cls_project(cls_token)
        return logits


def ViT_tiny_cifar10(in_channels=3,
                     patch_size=2,
                     embedding_dim=192,
                     num_heads=3,
                     depth=12,
                     mlp_ratio=4,
                     logits_dim=10,
                     input_size=32,
                     dataset="cifar10") -> nn.Module:
    """
    A version that deploys tiny ViT to CIFAR-10.
    """
    return VisionTransformer(in_channels=in_channels,
                             patch_size=patch_size,
                             embedding_dim=embedding_dim,
                             num_heads=num_heads,
                             depth=depth,
                             mlp_ratio=mlp_ratio,
                             logits_dim=logits_dim,
                             input_size=input_size,
                             dataset=dataset)


def ViT_tiny_mnist(in_channels=1,
                   patch_size=2,
                   embedding_dim=192,
                   num_heads=3,
                   depth=12,
                   mlp_ratio=4,
                   logits_dim=10,
                   input_size=28,
                   dataset="mnist") -> nn.Module:
    """
    A version that deploys tiny ViT to MNIST.
    """
    return VisionTransformer(in_channels=in_channels,
                             patch_size=patch_size,
                             embedding_dim=embedding_dim,
                             num_heads=num_heads,
                             depth=depth,
                             mlp_ratio=mlp_ratio,
                             logits_dim=logits_dim,
                             input_size=input_size,
                             dataset=dataset)


if __name__ == "__main__":
    net0 = ViT_tiny_cifar10()
    test_samples0 = torch.rand([2, 3, 32, 32])
    test_labels0 = torch.empty(2, dtype=torch.long).random_(10)  # [0, 10)
    pred0 = net0(test_samples0)
    print(pred0.shape)

    loss0 = net0.forward_loss(pred0, test_labels0)
    print(loss0)

    net1 = ViT_tiny_mnist()
    test_samples1 = torch.rand([2, 1, 28, 28])
    test_labels1 = torch.empty(2, dtype=torch.long).random_(10)  # [0, 10)
    pred1 = net1(test_samples1)
    print(pred1.shape)

    loss1 = net1.forward_loss(pred1, test_labels1)
    print(loss1)
