import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class CustomResNet(nn.Module):
    def __init__(self, blocks_per_group, channels_per_group, flatten_superpixels=False):
        super(CustomResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            resnet.BasicBlock, channels_per_group[0], blocks_per_group[0]
        )
        self.layer2 = self._make_layer(
            resnet.BasicBlock, channels_per_group[1], blocks_per_group[1], stride=2
        )
        self.layer3 = self._make_layer(
            resnet.BasicBlock, channels_per_group[2], blocks_per_group[2], stride=2
        )
        self.layer4 = self._make_layer(
            resnet.BasicBlock, channels_per_group[3], blocks_per_group[3], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            channels_per_group[3] * resnet.BasicBlock.expansion, channels_per_group[3]
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        # seq_len = x.size(0)
        # embedded_seq = []
        # for i in range(seq_len):

        # Torch nn.conv2D expects input of shape (batch_size, channels, height, width)
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.maxpool(z)

        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)

        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        z = self.fc(z)
        #     embedded_seq.append(z)

        # return torch.stack([embedded_seq])
        return z


if __name__ == "__main__":
    blocks_per_group = [2, 2, 2, 2]
    channels_per_group = [16, 32, 32, 27]
    custom_resnet = CustomResNet(blocks_per_group, channels_per_group)

    input_tensor = torch.randn(1, 3, 224, 224)

    output = custom_resnet(input_tensor)
    print("Output shape:", output.shape)


# # Copyright 2022 DeepMind Technologies Limited
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #    http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================

# """Resnet without logits. Forked from Haiku version."""

# from typing import Mapping, Optional, Sequence, Union

# import haiku as hk
# import jax
# import jax.numpy as jnp


# FloatStrOrBool = Union[str, float, bool]


# class BlockV1(hk.Module):
#   """ResNet V1 block with optional bottleneck."""

#   def __init__(
#       self,
#       channels: int,
#       stride: Union[int, Sequence[int]],
#       use_projection: bool,
#       bn_config: Mapping[str, FloatStrOrBool],
#       bottleneck: bool,
#       name: Optional[str] = None,
#   ):
#     super().__init__(name=name)
#     self.use_projection = use_projection

#     bn_config = dict(bn_config)
#     bn_config.setdefault("create_scale", True)
#     bn_config.setdefault("create_offset", True)
#     bn_config.setdefault("decay_rate", 0.999)

#     if self.use_projection:
#       self.proj_conv = hk.Conv2D(
#           output_channels=channels,
#           kernel_shape=1,
#           stride=stride,
#           with_bias=False,
#           padding="SAME",
#           name="shortcut_conv")

#       self.proj_batchnorm = hk.BatchNorm(name="shortcut_batchnorm", **bn_config)

#     channel_div = 4 if bottleneck else 1
#     conv_0 = hk.Conv2D(
#         output_channels=channels // channel_div,
#         kernel_shape=1 if bottleneck else 3,
#         stride=1 if bottleneck else stride,
#         with_bias=False,
#         padding="SAME",
#         name="conv_0")
#     bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)

#     conv_1 = hk.Conv2D(
#         output_channels=channels // channel_div,
#         kernel_shape=3,
#         stride=stride if bottleneck else 1,
#         with_bias=False,
#         padding="SAME",
#         name="conv_1")

#     bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
#     layers = ((conv_0, bn_0), (conv_1, bn_1))

#     if bottleneck:
#       conv_2 = hk.Conv2D(
#           output_channels=channels,
#           kernel_shape=1,
#           stride=1,
#           with_bias=False,
#           padding="SAME",
#           name="conv_2")

#       bn_2 = hk.BatchNorm(name="batchnorm_2", scale_init=jnp.zeros, **bn_config)
#       layers = layers + ((conv_2, bn_2),)

#     self.layers = layers

#   def __call__(self, inputs, is_training, test_local_stats):
#     out = shortcut = inputs

#     if self.use_projection:
#       shortcut = self.proj_conv(shortcut)
#       shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)

#     for i, (conv_i, bn_i) in enumerate(self.layers):
#       out = conv_i(out)
#       out = bn_i(out, is_training, test_local_stats)
#       if i < len(self.layers) - 1:  # Don't apply relu on last layer
#         out = jax.nn.relu(out)

#     return jax.nn.relu(out + shortcut)


# class BlockV2(hk.Module):
#   """ResNet V2 block with optional bottleneck."""

#   def __init__(
#       self,
#       channels: int,
#       stride: Union[int, Sequence[int]],
#       use_projection: bool,
#       bn_config: Mapping[str, FloatStrOrBool],
#       bottleneck: bool,
#       name: Optional[str] = None,
#   ):
#     super().__init__(name=name)
#     self.use_projection = use_projection

#     bn_config = dict(bn_config)
#     bn_config.setdefault("create_scale", True)
#     bn_config.setdefault("create_offset", True)

#     if self.use_projection:
#       self.proj_conv = hk.Conv2D(
#           output_channels=channels,
#           kernel_shape=1,
#           stride=stride,
#           with_bias=False,
#           padding="SAME",
#           name="shortcut_conv")

#     channel_div = 4 if bottleneck else 1
#     conv_0 = hk.Conv2D(
#         output_channels=channels // channel_div,
#         kernel_shape=1 if bottleneck else 3,
#         stride=1 if bottleneck else stride,
#         with_bias=False,
#         padding="SAME",
#         name="conv_0")

#     bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)

#     conv_1 = hk.Conv2D(
#         output_channels=channels // channel_div,
#         kernel_shape=3,
#         stride=stride if bottleneck else 1,
#         with_bias=False,
#         padding="SAME",
#         name="conv_1")

#     bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
#     layers = ((conv_0, bn_0), (conv_1, bn_1))

#     if bottleneck:
#       conv_2 = hk.Conv2D(
#           output_channels=channels,
#           kernel_shape=1,
#           stride=1,
#           with_bias=False,
#           padding="SAME",
#           name="conv_2")

#       # NOTE: Some implementations of ResNet50 v2 suggest initializing
#       # gamma/scale here to zeros.
#       bn_2 = hk.BatchNorm(name="batchnorm_2", **bn_config)
#       layers = layers + ((conv_2, bn_2),)

#     self.layers = layers

#   def __call__(self, inputs, is_training, test_local_stats):
#     x = shortcut = inputs

#     for i, (conv_i, bn_i) in enumerate(self.layers):
#       x = bn_i(x, is_training, test_local_stats)
#       x = jax.nn.relu(x)
#       if i == 0 and self.use_projection:
#         shortcut = self.proj_conv(x)
#       x = conv_i(x)

#     return x + shortcut


# class BlockGroup(hk.Module):
#   """Higher level block for ResNet implementation."""

#   def __init__(
#       self,
#       channels: int,
#       num_blocks: int,
#       stride: Union[int, Sequence[int]],
#       bn_config: Mapping[str, FloatStrOrBool],
#       resnet_v2: bool,
#       bottleneck: bool,
#       use_projection: bool,
#       name: Optional[str] = None,
#   ):
#     super().__init__(name=name)

#     block_cls = BlockV2 if resnet_v2 else BlockV1

#     self.blocks = []
#     for i in range(num_blocks):
#       self.blocks.append(
#           block_cls(channels=channels,
#                     stride=(1 if i else stride),
#                     use_projection=(i == 0 and use_projection),
#                     bottleneck=bottleneck,
#                     bn_config=bn_config,
#                     name="block_%d" % (i,)))

#   def __call__(self, inputs, is_training, test_local_stats):
#     out = inputs
#     for block in self.blocks:
#       out = block(out, is_training, test_local_stats)
#     return out


# def check_length(length, value, name):
#   if len(value) != length:
#     raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


# class SimpleResNet(hk.Module):
#   """Simple ResNet model."""

#   BlockGroup = BlockGroup  # pylint: disable=invalid-name
#   BlockV1 = BlockV1  # pylint: disable=invalid-name
#   BlockV2 = BlockV2  # pylint: disable=invalid-name

#   def __init__(
#       self,
#       blocks_per_group: Sequence[int],
#       bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
#       resnet_v2: bool = False,
#       bottleneck: bool = True,
#       channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
#       use_projection: Sequence[bool] = (True, True, True, True),
#       name: Optional[str] = None,
#       initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
#       strides: Sequence[int] = (1, 2, 2, 2),
#       flatten_superpixels: bool = False,
#   ):
#     """Constructs a ResNet model.

#     Args:
#       blocks_per_group: A sequence of length 4 that indicates the number of
#         blocks created in each group.
#       bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
#         passed on to the :class:`~haiku.BatchNorm` layers. By default the
#         ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
#       resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
#         ``False``.
#       bottleneck: Whether the block should bottleneck or not. Defaults to
#         ``True``.
#       channels_per_group: A sequence of length 4 that indicates the number
#         of channels used for each block in each group.
#       use_projection: A sequence of length 4 that indicates whether each
#         residual block should use projection.
#       name: Name of the module.
#       initial_conv_config: Keyword arguments passed to the constructor of the
#         initial :class:`~haiku.Conv2D` module.
#       strides: A sequence of length 4 that indicates the size of stride
#         of convolutions for each block in each group.
#       flatten_superpixels: Whether to flatten (instead of average) super-pixels.
#     """
#     super().__init__(name=name)
#     self.resnet_v2 = resnet_v2
#     self.flatten_superpixels = flatten_superpixels

#     bn_config = dict(bn_config or {})
#     bn_config.setdefault("decay_rate", 0.9)
#     bn_config.setdefault("eps", 1e-5)
#     bn_config.setdefault("create_scale", True)
#     bn_config.setdefault("create_offset", True)

#     # Number of blocks in each group for ResNet.
#     check_length(4, blocks_per_group, "blocks_per_group")
#     check_length(4, channels_per_group, "channels_per_group")
#     check_length(4, strides, "strides")

#     initial_conv_config = dict(initial_conv_config or {})
#     initial_conv_config.setdefault("output_channels", 64)
#     initial_conv_config.setdefault("kernel_shape", 7)
#     initial_conv_config.setdefault("stride", 2)
#     initial_conv_config.setdefault("with_bias", False)
#     initial_conv_config.setdefault("padding", "SAME")
#     initial_conv_config.setdefault("name", "initial_conv")

#     self.initial_conv = hk.Conv2D(**initial_conv_config)

#     if not self.resnet_v2:
#       self.initial_batchnorm = hk.BatchNorm(name="initial_batchnorm",
#                                             **bn_config)

#     self.block_groups = []
#     for i, stride in enumerate(strides):
#       self.block_groups.append(
#           BlockGroup(channels=channels_per_group[i],
#                      num_blocks=blocks_per_group[i],
#                      stride=stride,
#                      bn_config=bn_config,
#                      resnet_v2=resnet_v2,
#                      bottleneck=bottleneck,
#                      use_projection=use_projection[i],
#                      name="block_group_%d" % (i,)))

#     if self.resnet_v2:
#       self.final_batchnorm = hk.BatchNorm(name="final_batchnorm", **bn_config)

#   def __call__(self, inputs, is_training, test_local_stats=False):
#     out = inputs
#     out = self.initial_conv(out)
#     if not self.resnet_v2:
#       out = self.initial_batchnorm(out, is_training, test_local_stats)
#       out = jax.nn.relu(out)

#     out = hk.max_pool(out,
#                       window_shape=(1, 3, 3, 1),
#                       strides=(1, 2, 2, 1),
#                       padding="SAME")

#     for block_group in self.block_groups:
#       out = block_group(out, is_training, test_local_stats)

#     if self.resnet_v2:
#       out = self.final_batchnorm(out, is_training, test_local_stats)
#       out = jax.nn.relu(out)
#     if self.flatten_superpixels:
#       out = hk.Flatten()(out)
#     else:
#       out = jnp.mean(out, axis=(1, 2))
#     return out
