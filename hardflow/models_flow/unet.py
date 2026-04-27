import math
import torch
import torch.nn as nn
import einops
from einops import rearrange
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample2d(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.conv = nn.Conv2d(dim_in, dim_out, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim_in, dim_out, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h d i", attn, v)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        return self.to_out(out)


class Conv2dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=3, n_groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock2d(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, n_groups=1):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv2dBlock(inp_channels, out_channels, 3, n_groups),
                Conv2dBlock(out_channels, out_channels, 3, n_groups),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )

        self.residual_conv = (
            nn.Conv2d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        time_emb = einops.rearrange(time_emb, "b c -> b c 1 1")
        out = self.blocks[0](x) + time_emb
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        output_channels=2,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        n_groups=1,
    ):
        super().__init__()

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, time_dim),
        )

        self.init_conv = nn.Conv2d(output_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualBlock2d(
                            dim_in, dim_in, embed_dim=time_dim, n_groups=n_groups
                        ),
                        ResidualBlock2d(
                            dim_in, dim_in, embed_dim=time_dim, n_groups=n_groups
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample2d(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock2d(
            mid_dim, mid_dim, embed_dim=time_dim, n_groups=n_groups
        )
        self.mid_attn = Residual(
            PreNorm(mid_dim, Attention(mid_dim, heads=4, dim_head=32))
        )
        self.mid_block2 = ResidualBlock2d(
            mid_dim, mid_dim, embed_dim=time_dim, n_groups=n_groups
        )

        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualBlock2d(
                            dim_out + dim_in,
                            dim_out,
                            embed_dim=time_dim,
                            n_groups=n_groups,
                        ),
                        ResidualBlock2d(
                            dim_out + dim_in,
                            dim_out,
                            embed_dim=time_dim,
                            n_groups=n_groups,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample2d(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        self.final_res_block = ResidualBlock2d(
            dim * 2, dim, embed_dim=time_dim, n_groups=n_groups
        )
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

    def forward(self, x, time):

        b = x.shape[0]

        while time.dim() > 1:
            time = time[..., 0]
        if time.dim() == 0:
            time = time.repeat(b)

        t = self.time_mlp(time)

        x = self.init_conv(x)
        r = x.clone()

        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            h.append(x)
            x = attn(x)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        return x


class WrappedFlowUnet(TemporalUnet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_info(self, temporal_dim=11, spatial_dim=128, padded_temporal_dim=16):

        assert temporal_dim <= padded_temporal_dim

        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.padded_temporal_dim = padded_temporal_dim

    def forward(self, src):
        # src: (batch_size, temporal_dim * spatial_dim + (temporal_dim - 1) * spatial_dim + 1)
        # output: (batch_size, temporal_dim * spatial_dim + (temporal_dim - 1) * spatial_dim)

        batch_size, v_dim = src.shape

        assert (
            v_dim
            == self.temporal_dim * self.spatial_dim
            + (self.temporal_dim - 1) * self.spatial_dim
            + 1
        ), "Dimension mismatch!"

        time = src[:, -1:].squeeze(-1)  # (batch_size, )

        src_state = src[:, : self.temporal_dim * self.spatial_dim]
        src_control = src[:, self.temporal_dim * self.spatial_dim : -1]

        src_state = src_state.view(batch_size, self.temporal_dim, self.spatial_dim)
        src_state = F.pad(
            src_state,
            (0, 0, 0, self.padded_temporal_dim - self.temporal_dim),
            "constant",
            0,
        )

        src_control = src_control.view(
            batch_size, self.temporal_dim - 1, self.spatial_dim
        )
        src_control = F.pad(
            src_control,
            (0, 0, 0, self.padded_temporal_dim - (self.temporal_dim - 1)),
            "constant",
            0,
        )

        # add channel dimension
        src_state = src_state.unsqueeze(1)  # [batch_size, 1, 16, 128]
        src_control = src_control.unsqueeze(1)  # [batch_size, 1, 16, 128]

        src_original = torch.cat(
            (src_state, src_control), dim=1
        )  # [batch_size, 2, 16, 128]

        output_original = super().forward(src_original, time)

        output_original_state = output_original[:, 0, : self.temporal_dim, :]
        output_original_control = output_original[:, 1, : self.temporal_dim - 1, :]

        output_state = output_original_state.view(
            batch_size, self.temporal_dim * self.spatial_dim
        )
        output_control = output_original_control.view(
            batch_size, (self.temporal_dim - 1) * self.spatial_dim
        )

        output = torch.cat((output_state, output_control), dim=1)

        return output
