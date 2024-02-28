import math
import random
import functools
import operator
import itertools
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd import Variable

from op import conv2d_gradfix
if torch.cuda.is_available():
    from op.fused_act import FusedLeakyReLU, fused_leaky_relu
    from op.upfirdn2d import upfirdn2d
else:
    from op.fused_act_cpu import FusedLeakyReLU, fused_leaky_relu
    from op.upfirdn2d_cpu import upfirdn2d


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def get_special_params(self, ids):
        params = []
        for idx in ids:
            if idx == 0:
                params += list(self.conv1.parameters())
                params += list(self.to_rgb1.parameters())
            else:
                params += list(self.convs[idx*2-2:idx*2].parameters())
                params += list(self.to_rgbs[idx-1].parameters())
        return params

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    @torch.no_grad()
    def mean_latent(self, n_latent):
        # print(self.input.input.device)
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input, truncation=1., truncation_latent=None):
        latent = self.style(input)
        if truncation < 1:
            latent_truncation = truncation_latent + truncation * (latent - truncation_latent)
            return latent_truncation
        else:
            return latent

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

            if truncation < 1:
                style_t = []

                for style in styles:
                    style_t.append(
                        truncation_latent + truncation * (style - truncation_latent)
                    )

                styles = style_t
            latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
        else:
            latent = styles

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        return image


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.ModuleList(convs)
        self.layers = (1,3,4,5)

    def forward(self, input):
        output = []
        out = input
        for i, block in enumerate(self.convs):
            out = block(out)
            if i in self.layers:
                output.append(out)
            if i == max(self.layers):
                break
        return output


class DiscriminatorPatch(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, inp, ind=None, extra=None, flag=None, p_ind=None, real=False):

        feat = []
        for i in range(len(self.convs)):
            if i == 0:
                inp = self.convs[i](inp)
            else:
                temp1 = self.convs[i].conv1(inp)
                if (flag > 0) and (temp1.shape[1] == 512) and (temp1.shape[2] == 32 or temp1.shape[2] == 16):
                    feat.append(temp1)
                temp2 = self.convs[i].conv2(temp1)
                if (flag > 0) and (temp2.shape[1] == 512) and (temp2.shape[2] == 32 or temp2.shape[2] == 16):
                    feat.append(temp2)
                inp = self.convs[i](inp)
                if (flag > 0) and len(feat) == 4:
                    # We use 4 possible intermediate feature maps to be used for patch-based adversarial loss. Any
                    # one of them is selected randomly during training.
                    inp = extra(feat[p_ind], p_ind)
                    return inp

        out = inp
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        feat.append(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out


class Extra(nn.Module):
    # to apply the patch-level adversarial loss, we take the intermediate discriminator feature maps of size [N x N x
    # D], and convert them into [N x N x 1]

    def __init__(self):
        super().__init__()

        self.new_conv = nn.ModuleList()
        self.new_conv.append(ConvLayer(512, 1, 3))
        self.new_conv.append(ConvLayer(512, 1, 3))
        self.new_conv.append(ConvLayer(512, 1, 3))
        self.new_conv.append(ConvLayer(512, 1, 3))

    def forward(self, inp, ind):
        out = self.new_conv[ind](inp)
        return out


class LocalisationNet(nn.Module):
    def __init__(self, res, num_output, target_control_points, kernel_size: list = None, need_pool: list = None):
        super().__init__()
        if kernel_size is None:
            kernel_size = [5, 5]
        if need_pool is None:
            need_pool = [True, True]
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512,
            128: 256,
            256: 128,
            512: 64,
            1024: 32,
        }
        in_channels = self.channels[res]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=kernel_size[0]),
            *[nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU()] if need_pool[0] else [nn.ReLU()],
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size[1]),
            # nn.Dropout2d(p=0.1),
            *[nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU()] if need_pool[1] else [nn.ReLU()]
        )

        with torch.no_grad():
            inp = torch.randn([1, in_channels, res, res])
            out = self.conv(inp)
            _, c, h, w = out.shape
        self.size = c * h * w
        self.fc = nn.Sequential(
            nn.Linear(self.size, 64),
            nn.ReLU(),
            # nn.Dropout(p=0.1)
            nn.Linear(64, num_output)
        )
        self.bias = target_control_points.view(-1)
        self.fc[-1].bias.data.copy_(self.bias)
        self.fc[-1].weight.data.zero_()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = x.view(-1, self.size)
        x = self.fc(x)
        return x.view(batch_size, -1, 2)


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


class TPSGridGenerator(nn.Module):
    def __init__(self, target_control_points, target_height, target_width):
        super().__init__()
        N = target_control_points.shape[0]
        self.num_points = N
        self.height = target_height
        self.width = target_width
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target coordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], dim=1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate.view(batch_size, self.height, self.width, 2)


class TPSSpatialTransformerBlock(nn.Module):
    def __init__(self, res, target_control_points, grid_size=10):
        super().__init__()
        nums = grid_size**2
        kernel_size = [3 if res == 8 else 5, 5]
        need_pool = [False if res == 8 else True, True]
        self.target_control_points = target_control_points
        self.loc_net = LocalisationNet(res=res, num_output=nums*2, target_control_points=target_control_points,
                                       kernel_size=kernel_size, need_pool=need_pool)
        self.tps_grid_gen = TPSGridGenerator(target_control_points=target_control_points,
                                             target_height=res, target_width=res)

    def forward(self, x, alpha=1.):
        batch = x.shape[0]
        source_control_points = self.loc_net(x)
        if alpha != 1.:
            source = self.loc_net.bias.to(x.device).view(1, -1, 2).repeat(batch, 1, 1)
            source_control_points = source * (1-alpha) + source_control_points * alpha
        grid = self.tps_grid_gen(source_control_points)
        transformed_x = F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)

        return transformed_x, grid


class TPSSpatialTransformer(nn.Module):
    def __init__(self, grid_size=10, resolutions: list = None):
        super().__init__()
        assert isinstance(grid_size, int) or isinstance(grid_size, list)
        if isinstance(grid_size, int):
            grid_size = [grid_size] * len(resolutions)
        if resolutions is None:
            resolutions = [8*(2**i) for i in range(4)]
        self.resolutions = resolutions
        r1 = r2 = 0.95

        stns = []
        for res, gs in zip(resolutions, grid_size):
            target_control_points = torch.Tensor(list(itertools.product(
                np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (gs - 1)),
                np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (gs - 1)),
            )))
            stns.append(TPSSpatialTransformerBlock(res=res,
                                                   target_control_points=target_control_points, grid_size=gs))

        self.stns = nn.Sequential(*stns)


class RTSpatialTransformerBlock(nn.Module):
    def __init__(self, resolution=1024):
        super().__init__()
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512,
            128: 256,
            256: 128,
            512: 64,
            1024: 32,
        }
        in_channels = self.channels[resolution]
        self.in_channels = in_channels
        self.resolution = resolution
        self.loc_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5), nn.MaxPool2d(2, stride=2), nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5), nn.MaxPool2d(2, stride=2), nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(self.get_size(), 32), nn.ReLU(True), nn.Linear(32, 6)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def get_size(self):
        x = torch.randn([1, self.in_channels, self.resolution, self.resolution])
        b, c, h, w = self.loc_net(x).shape
        return c * h * w

    def forward(self, x):
        xs = self.loc_net(x)
        b, _, _, _ = x.shape
        xs = xs.view(b, -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)
        return x


class RTSpatialTransformer(nn.Module):
    def __init__(self, resolutions: list = None):
        super().__init__()
        if resolutions is None:
            resolutions = [8*(2**i) for i in range(4)]
        self.resolutions = resolutions

        stns = []
        for res in resolutions:
            stns.append(RTSpatialTransformerBlock(resolution=res))

        self.stns = nn.Sequential(*stns)


class DeformAwareGenerator(Generator):
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, resolutions: list = None,
                 rt_resolutions: list = None):
        super().__init__(size=size, style_dim=style_dim, n_mlp=n_mlp, channel_multiplier=channel_multiplier)
        resolutions = [8 * (2 ** i) for i in range(4)] if resolutions is None else resolutions
        rt_resolutions = [8 * (2 ** i) for i in range(4)] if rt_resolutions is None else rt_resolutions
        self.resolutions = resolutions
        self.n_stn = len(resolutions)
        self.n_rtstn = len(rt_resolutions)

    def forward(
            self,
            styles,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            stns=None,
            rt_stns=None,
            alpha=1.
    ):
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        if not input_is_latent:
            styles = [self.style(s) for s in styles]
            if truncation < 1:
                style_t = []
                for style in styles:
                    style_t.append(truncation_latent + truncation * (style - truncation_latent))

                styles = style_t
            latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
        else:
            latent = styles

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        idx_stn = 0 if stns is None else - int(math.log2(stns.resolutions[0])) + 3
        idx_rtstn = 0 if rt_stns is None else - int(math.log2(rt_stns.resolutions[0])) + 3
        flows = []
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)

            if 0 <= idx_rtstn < self.n_rtstn and rt_stns is not None:
                out = rt_stns.stns[idx_rtstn](out)

            if 0 <= idx_stn < self.n_stn and stns is not None:
                temp = stns.stns[idx_stn](out, alpha)
                out, flow = temp
                flows.append(flow)

            idx_stn += 1
            idx_rtstn += 1

            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip
        return image, flows
