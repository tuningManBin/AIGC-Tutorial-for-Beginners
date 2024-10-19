"""
Masked Auto-encoder with Vision Transformer backbone

Reference:
    [1] He K, Chen X, Xie S, et al.
        Masked autoencoders are scalable vision learners[C]//
        Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
        2022: 16000-16009.

    [2] https://github.com/facebookresearch/mae
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import BasicBlock
from typing import Optional


class MaskedAutoEncoder(nn.Module):
    """
    This module is the main body of Masked Auto-encoder with Vision Transformer backbone.
    """
    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 encoder_embedding_dim: int,
                 encoder_attn_num_heads: int,
                 encoder_attn_depth: int,
                 encoder_mlp_ratio: int,
                 encoder_mask_ratio: float,
                 decoder_embedding_dim: int,
                 decoder_attn_num_heads: int,
                 decoder_attn_depth: int,
                 decoder_mlp_ratio: int,
                 input_size: int = 32,
                 dataset: Optional[str] = None) -> None:
        """
        Arguments:
            in_channels (int): the number of channels in the input tensor.
            patch_size (int): the size of the image pixel patch.
            encoder_embedding_dim (int): the dimension of the embedding in Encoder.
            encoder_attn_num_heads (int): the number of self-attention heads in Encoder.
            encoder_attn_depth (int): the number of basic blocks (attention + mlp) in Encoder.
            encoder_mlp_ratio (int): the ratio of the hidden layer dimension to the input layer dimension in Encoder MLP.
            encoder_mask_ratio (float): the ratio of masked patches to all patches in Encoder.
            decoder_embedding_dim (int): the dimension of the embedding in Decoder.
            decoder_attn_num_heads (int): the number of self-attention heads in Decoder.
            decoder_attn_depth (int): the number of basic blocks (attention + mlp) in Decoder.
            decoder_mlp_ratio (int): the ratio of the hidden layer dimension to the input layer dimension in Decoder MLP.
            input_size (int): the size of the input tensor.
            dataset (Optional[str]): a control variable used for normalizing the data.
        """
        super(MaskedAutoEncoder, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dataset = dataset
        if self.dataset == "cifar10":
            self.input_size = 32
        elif self.dataset == "mnist":
            self.input_size = 28
        assert self.input_size % patch_size == 0, \
            f"input_size // pool_stride {input_size} should be divided by patch_size {patch_size}."

        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.num_patches = (self.input_size // patch_size) ** 2 + 1

        # --- Encoder params
        self.encoder_mask_ratio = encoder_mask_ratio
        self.patch_embedding = nn.Conv2d(in_channels=self.in_channels, out_channels=encoder_embedding_dim,
                                         kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.encoder_position_embedding = nn.Parameter(torch.randn(1, self.num_patches, encoder_embedding_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embedding_dim))
        self.encoder_attn_blocks = nn.Sequential(*[
            BasicBlock(embedding_dim=encoder_embedding_dim,
                       num_heads=encoder_attn_num_heads,
                       mlp_ratio=encoder_mlp_ratio)
            for _ in range(encoder_attn_depth)
        ])  # nn.Sequential(*layers) represents the extraction of elements from layers to form of sequential
        self.encoder_norm = nn.LayerNorm(encoder_embedding_dim)

        # --- Decoder params
        self.decoder_embedding_project = nn.Linear(encoder_embedding_dim, decoder_embedding_dim)
        self.tokens_masked = nn.Parameter(torch.zeros(1, 1, decoder_embedding_dim))
        self.decoder_position_embedding = nn.Parameter(torch.randn(1, self.num_patches, decoder_embedding_dim))
        self.decoder_attn_blocks = nn.Sequential(*[
            BasicBlock(embedding_dim=decoder_embedding_dim,
                       num_heads=decoder_attn_num_heads,
                       mlp_ratio=decoder_mlp_ratio)
            for _ in range(decoder_attn_depth)
        ])  # nn.Sequential(*layers) represents the extraction of elements from layers to form of sequential
        self.decoder_norm = nn.LayerNorm(decoder_embedding_dim)

        self.reconstruct = nn.Linear(decoder_embedding_dim, patch_size ** 2 * self.in_channels)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        This function converts images to spectrum patches according to the patch size.

        Arguments:
            images (torch.Tensor): the input tensor with shape [batch, in_channels, height, width].

        Returns:
            tokens (torch.Tensor): the tokenized patches with shape
                                   [batch, (input_size // patch_size)**2, patch_size**2 * in_channels].
        """
        assert self.input_size % self.patch_size == 0, \
            f"input_size {self.input_size} should be divided by patch_size {self.patch_size}."
        h = w = self.input_size // self.patch_size
        tokens = images.reshape(shape=(images.shape[0], self.in_channels, h, self.patch_size, w, self.patch_size))
        tokens = torch.einsum("bchpwq->bhwpqc", tokens)
        tokens = tokens.reshape(shape=(images.shape[0], h * w, self.patch_size**2 * self.in_channels))
        return tokens

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        This function converts image pixel patches back to the full images.

        Arguments:
            tokens (torch.Tensor): the input tensor representing the tokenized patches with shape
                                   [batch, (input_size // patch_size)**2, patch_size**2 * in_channels].

        Returns:
            images (torch.Tensor): the reconstructed images with shape
                                   [batch, in_channels, input_size, input_size].
        """
        h = w = int(tokens.shape[1] ** 0.5)
        assert h * self.patch_size == self.input_size, \
            f"input_size // patch_size * patch_size {h * self.patch_size} " \
            f"should be the same as input_size {self.input_size}."

        tokens = tokens.reshape(shape=(tokens.shape[0], h, w, self.patch_size, self.patch_size, self.in_channels))
        tokens = torch.einsum("nhwpqc->nchpwq", tokens)
        images = tokens.reshape(shape=(tokens.shape[0], self.in_channels, h * self.patch_size, w * self.patch_size))
        return images

    def masking(self, tokens: torch.Tensor) -> tuple:
        """
        This function masks tokenized patches.

        Arguments:
            tokens (torch.Tensor): the input tensor representing the tokenized patches with shape
                                   [batch, num_patches, embedding_dim].

        Returns:
            tuple: a tuple containing the following elements:
                   - tokens_keep (torch.Tensor): the unmasked tokens.
                   - mask (torch.Tensor): the binary mask indicating which tokens are kept and which are masked.
                   - ids_restore (torch.Tensor): the indices used to restore the original order of tokens.
        """
        batch_size, num_patches, embedding_dim = tokens.shape
        num_patches_keep = int(num_patches * (1 - self.encoder_mask_ratio))

        # --- Mask tokens randomly
        # descend & keep the first subset: keep the large ones, remove the small ones.
        noise = torch.rand(tokens.shape[0], tokens.shape[1], device=tokens.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1, descending=True)
        ids_keep = ids_shuffle[:, :num_patches_keep]
        tokens_keep = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, embedding_dim))

        # --- Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, num_patches], device=tokens.device)
        mask[:, :num_patches_keep] = 0

        # --- Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return tokens_keep, mask, ids_restore

    def forward_encoder(self, images: torch.Tensor) -> tuple:
        """
        This function converts unmasked spectrum patches to embeddings.

        Arguments:
            images (torch.Tensor): the input tensor with shape [batch, in_channels, height, width].

        Returns:
            tuple: a tuple containing the following elements:
                   - tokens_keep (torch.Tensor): the unmasked tokens.
                   - mask (torch.Tensor): the binary mask indicating which tokens were kept and which were masked.
                   - ids_restore (torch.Tensor): the indices used to restore the original order of tokens.
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

        # --- Patch Embedding
        tokens = self.patch_embedding(images)
        tokens = tokens.flatten(2).transpose(1, 2)
        # output shape = [batch_size, num_patches, embedding_dim]

        # --- Position Embedding
        tokens = tokens + self.encoder_position_embedding[:, 1:, :]
        # output shape = [batch_size, num_patches, embedding_dim]

        # --- Masking
        tokens_keep, mask, ids_restore = self.masking(tokens)
        # output shape = [batch_size, num_patches * (1-encoder_mask_ratio), embedding_dim]

        # --- Add class token
        cls_token = self.cls_token + self.encoder_position_embedding[:, :1, :]
        cls_tokens = cls_token.expand(tokens_keep.shape[0], -1, -1)
        tokens_keep = torch.cat([cls_tokens, tokens_keep], dim=1)
        # output shape = [batch_size, num_patches_unmasked, embedding_dim]

        # --- Attention
        tokens_keep = self.encoder_attn_blocks(tokens_keep)
        tokens_keep = self.encoder_norm(tokens_keep)
        # output shape = [batch_size, num_patches_unmasked, embedding_dim]
        return tokens_keep, mask, ids_restore

    def forward_decoder(self,
                        tokens_keep: torch.Tensor,
                        ids_restore: torch.Tensor) -> torch.Tensor:
        """
        This function reconstructs the masked image pixel patches based on the unmasked image pixel patches.

        Arguments:
            tokens_keep (torch.Tensor): the input tensor.
            ids_restore (torch.Tensor): the indices used to restore the original order of tokens.

        Returns:
            tokens (torch.Tensor): the reconstructed image pixel patches.
        """
        tokens_keep = self.decoder_embedding_project(tokens_keep)

        # --- Integrate unmasked tokens with masked tokens
        tokens_masked = self.tokens_masked.repeat(tokens_keep.shape[0], ids_restore.shape[1] + 1 - tokens_keep.shape[1], 1)
        temp = torch.cat([tokens_keep[:, 1:, :], tokens_masked], dim=1)  # no cls token
        temp = torch.gather(temp, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, tokens_keep.shape[2]))  # unshuffle
        tokens = torch.cat([tokens_keep[:, :1, :], temp], dim=1)  # append cls token

        # --- Add position embedding
        tokens = tokens + self.decoder_position_embedding

        # --- Attention
        tokens = self.decoder_attn_blocks(tokens)
        tokens = self.decoder_norm(tokens)

        # --- Reconstruct
        tokens = self.reconstruct(tokens)

        # --- Remove class token
        tokens = tokens[:, 1:, :]
        return tokens

    def forward_loss(self,
                     reconstruction: torch.Tensor,
                     ground_truth: torch.Tensor,
                     reduction: str = "mean") -> torch.Tensor:
        """
        This function calculates the mean squared error loss.

        Arguments:
            reconstruction (torch.Tensor): the output tensor of Masked Auto-encoder.
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
        ground_truth = F.interpolate(ground_truth, size=(self.input_size, self.input_size), mode="bilinear", align_corners=True)
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

    def forward(self, images: torch.Tensor) -> tuple:
        """
        Arguments:
            images (torch.Tensor): the input tensor with shape [batch, in_channels, height, width].

        Returns:
            tuple: a tuple containing the following elements:
                   - tokens_keep (torch.Tensor): the unmasked tokens.
                   - tokens_reconstruct (torch.Tensor): the reconstructed tokens.
                   - mask (torch.Tensor): the binary mask indicating which tokens are kept and which are masked.
                   - reconstruction (torch.Tensor): the reconstructed images.
        """
        # --- Encoder
        tokens_keep, mask, ids_restore = self.forward_encoder(images)
        mask = self.unpatchify(mask.unsqueeze(-1).repeat(1, 1, self.patch_size ** 2 * self.in_channels))
        # --- Decoder
        tokens_reconstruct = self.forward_decoder(tokens_keep, ids_restore)
        reconstruction = self.unpatchify(tokens_reconstruct)
        return tokens_keep, tokens_reconstruct, mask, reconstruction


def MAE_cifar10(in_channels=3,
                patch_size=2,
                encoder_embedding_dim=192,
                encoder_attn_num_heads=3,
                encoder_attn_depth=12,
                encoder_mlp_ratio=4,
                encoder_mask_ratio=0.75,
                decoder_embedding_dim=192,
                decoder_attn_num_heads=3,
                decoder_attn_depth=8,
                decoder_mlp_ratio=4,
                input_size=32,
                dataset="cifar10") -> nn.Module:
    """
    A version deploying MAE to CIFAR-10.
    """
    return MaskedAutoEncoder(in_channels=in_channels,
                             patch_size=patch_size,
                             encoder_embedding_dim=encoder_embedding_dim,
                             encoder_attn_num_heads=encoder_attn_num_heads,
                             encoder_attn_depth=encoder_attn_depth,
                             encoder_mlp_ratio=encoder_mlp_ratio,
                             encoder_mask_ratio=encoder_mask_ratio,
                             decoder_embedding_dim=decoder_embedding_dim,
                             decoder_attn_num_heads=decoder_attn_num_heads,
                             decoder_attn_depth=decoder_attn_depth,
                             decoder_mlp_ratio=decoder_mlp_ratio,
                             input_size=input_size,
                             dataset=dataset)


def MAE_mnist(in_channels=1,
              patch_size=2,
              encoder_embedding_dim=192,
              encoder_attn_num_heads=3,
              encoder_attn_depth=12,
              encoder_mlp_ratio=4,
              encoder_mask_ratio=0.75,
              decoder_embedding_dim=192,
              decoder_attn_num_heads=3,
              decoder_attn_depth=8,
              decoder_mlp_ratio=4,
              input_size=28,
              dataset="mnist") -> nn.Module:
    """
    A version deploying MAE to MNIST.
    """
    return MaskedAutoEncoder(in_channels=in_channels,
                             patch_size=patch_size,
                             encoder_embedding_dim=encoder_embedding_dim,
                             encoder_attn_num_heads=encoder_attn_num_heads,
                             encoder_attn_depth=encoder_attn_depth,
                             encoder_mlp_ratio=encoder_mlp_ratio,
                             encoder_mask_ratio=encoder_mask_ratio,
                             decoder_embedding_dim=decoder_embedding_dim,
                             decoder_attn_num_heads=decoder_attn_num_heads,
                             decoder_attn_depth=decoder_attn_depth,
                             decoder_mlp_ratio=decoder_mlp_ratio,
                             input_size=input_size,
                             dataset=dataset)


if __name__ == "__main__":
    net0 = MAE_cifar10()
    test_sample0 = torch.rand([2, 3, 32, 32])
    pred0 = net0(test_sample0)
    print(pred0[0].shape, pred0[1].shape, pred0[2].shape, pred0[3].shape, sep="\n")

    loss0 = net0.forward_loss(pred0[3], torch.rand([2, 3, 32, 32]))
    print(loss0)

    net1 = MAE_mnist()
    test_sample1 = torch.rand([2, 1, 28, 28])
    pred1 = net1(test_sample1)
    print(pred1[0].shape, pred1[1].shape, pred1[2].shape, pred1[3].shape, sep="\n")

    loss1 = net1.forward_loss(pred1[3], torch.rand([2, 1, 28, 28]))
    print(loss1)
