# Non-causal discriminator. It is used only during training to provide a richer
# gradient signal to the generator without introducing any latency at inference.
#
# A non-causal discriminator can look at the entire context window (past and
# future) to detect phase artifacts, spectral blurring, and temporal
# inconsistencies.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class NonCausalDiscriminator(nn.Module):
    """A non-causal 1D waveform discriminator for adversarial training.

    The module intentionally uses symmetric padding so each convolution can see
    both past and future context. This makes it suitable for detecting
    temporal inconsistencies and phase-related artifacts in generated audio
    while keeping the deployed model causal and latency-free.

    Args:
        in_channels: Number of input channels (e.g. 1 for mono, 2 for stereo).
        base_channels: Number of channels in the first feature block.
        num_layers: Number of convolutional blocks in the discriminator.
        kernel_size: Kernel size for every convolutional layer.
        stride: Stride used in each convolutional block.
        leaky_relu_slope: Negative slope for LeakyReLU activation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 15,
        stride: int = 2,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if not 0.0 <= leaky_relu_slope < 1.0:
            raise ValueError("leaky_relu_slope must lie in [0, 1)")

        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.leaky_relu_slope = leaky_relu_slope

        channels = [in_channels]
        for layer_idx in range(num_layers):
            next_channels = min(base_channels * (2**layer_idx), 512)
            channels.append(next_channels)

        blocks: list[nn.Module] = []
        for idx in range(num_layers):
            in_dim = channels[idx]
            out_dim = channels[idx + 1]
            blocks.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            in_channels=in_dim,
                            out_channels=out_dim,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            bias=True,
                        )
                    ),
                    #nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=False),
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a discriminator score for each input waveform.

        Args:
            x: Input tensor of shape ``(batch_size, in_channels, time_steps)``.

        Returns:
            A tensor of shape ``(batch_size, 1)`` containing logits for each
            sample.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input tensor of shape (batch, channels, time), got {tuple(x.shape)}"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.size(1)}"
            )
        if x.size(2) < 1:
            raise ValueError("Input sequence must contain at least one timestep")

        features = x
        fmaps = []
        for block in self.blocks:
            features = block(features)
            features = F.leaky_relu(features, self.leaky_relu_slope)
            fmaps.append(features)

        pooled = self.pool(features).flatten(1) #score
        
        return self.classifier(pooled), fmaps


__all__ = ["NonCausalDiscriminator"]
        