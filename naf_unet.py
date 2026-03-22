import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from abc import abstractmethod

"""
Authors: Ruijie He, Botong Cai, Ziqi Yang
Reference: NAFNet (Megvii Research) - https://github.com/megvii-research/NAFNet.git
Description: Multi-image conditional UNet using Nonlinear Activation Free (NAF) blocks.
"""


class EmbedBlock(nn.Module):
    """Base class for modules accepting embeddings in forward pass"""

    @abstractmethod
    def forward(self, x, emb):
        pass


class EmbedSequential(nn.Sequential, EmbedBlock):
    """Sequential wrapper that propagates embeddings to compatible layers"""

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


def gamma_embedding(gammas, dim, max_period=10000):
    """Sinusoidal positional encoding for diffusion timesteps/gammas"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm forward/backward for 2D feature maps"""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=(0, 2, 3)), grad_output.sum(dim=(0, 2, 3)), None


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 2D inputs"""

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Feature gating via element-wise multiplication of split channels"""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class CondNAFBlock(nn.Module):
    """NAFBlock for reference/condition images (no time embedding)"""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        # Simplified Channel Attention (SCA)
        self.sca_avg = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 4, dw_channel // 4, 1))
        self.sca_max = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Conv2d(dw_channel // 4, dw_channel // 4, 1))
        self.sg = SimpleGate()

        # Feed-forward Network (FFN)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x_avg, x_max = x.chunk(2, dim=1)
        x = torch.cat([self.sca_avg(x_avg) * x_avg, self.sca_max(x_max) * x_max], dim=1)
        x = self.conv3(x)
        y = inp + self.dropout1(x) * self.beta

        # FFN stage
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.dropout2(x) * self.gamma


class NAFBlock(EmbedBlock):
    """NAFBlock for main noisy path with Time Embedding injection"""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        self.sca_avg = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 4, dw_channel // 4, 1))
        self.sca_max = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Conv2d(dw_channel // 4, dw_channel // 4, 1))
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

        # Project time embedding to channel dimension
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(256, c))

    def forward(self, inp, t):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x_avg, x_max = x.chunk(2, dim=1)
        x = torch.cat([self.sca_avg(x_avg) * x_avg, self.sca_max(x_max) * x_max], dim=1)
        x = self.conv3(x)
        y = inp + self.dropout1(x) * self.beta

        # Time injection
        y = y + self.time_emb(t)[..., None, None]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.dropout2(x) * self.gamma


class UNet(nn.Module):
    """Time-conditional UNet for multi-reference image restoration"""

    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.cond_intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

        self.encoders, self.cond_encoders = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.middle_blks = nn.ModuleList(), nn.ModuleList()
        self.ups, self.downs, self.cond_downs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(EmbedSequential(*[NAFBlock(chan) for _ in range(num)]))
            self.cond_encoders.append(nn.Sequential(*[CondNAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            self.cond_downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = EmbedSequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(EmbedSequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)
        self.emb = partial(gamma_embedding, dim=64)
        self.map = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 256))

    def forward(self, inp, gammas):
        t = self.map(self.emb(gammas.view(-1, )))
        inp = self.check_image_size(inp)

        # Split 12-channel input: 3 reference images + 1 noisy image
        x1, x2, x3, x = inp.chunk(4, dim=1)
        cond = torch.stack([x1, x2, x3], dim=1)
        b, n, c, h, w = cond.shape
        cond = cond.view(b * n, c, h, w)  # Stack reference images in batch dim

        x, cond = self.intro(x), self.cond_intro(cond)
        encs = []

        # Encoder path: Main image absorbs reference features
        for encoder, down, cond_encoder, cond_down in zip(self.encoders, self.downs, self.cond_encoders,
                                                          self.cond_downs):
            x = encoder(x, t)
            cond = cond_encoder(cond)
            # Merge 3 reference features via summation
            tmp_cond = cond.view(b, n, -1, h, w).sum(dim=1)
            x = x + tmp_cond
            encs.append(x)
            x, cond = down(x), cond_down(cond)
            h, w = h // 2, w // 2

        x = self.middle_blks(x, t)

        # Decoder path with skip connections
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x, t)

        return self.ending(x)

    def check_image_size(self, x):
        """Pad image to be divisible by downsampling factor"""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


if __name__ == '__main__':
    # Test forward pass: Input(1, 12, 256, 256), Gamma(1,)
    model = UNet()
    dummy_input = torch.Tensor(1, 12, 256, 256)
    dummy_gamma = torch.ones(1, )
    print(model(dummy_input, dummy_gamma).shape)