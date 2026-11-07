"""A float32 PyTorch transcription of the 3D splat model's three networks.

Written from a reading of the model's published SDK (Apache-2.0; see
PROVENANCE.md) so the reference stages can run where the SDK itself is not at
hand. The module and parameter names are the checkpoint's, and every network
is loaded with `strict=True`, so a name or shape this transcription got wrong
is refused rather than silently initialised. It shares no code with the Rust
port: it is torch's own linear, attention and norm arithmetic over the
checkpoint's tensors.

Where the SDK runs the transformer trunks in float16, this runs float32; that
is the point (see idol_reference.py).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm32(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x):
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class FeedForwardNet(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, channels))

    def forward(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, channels, inner, out, layers=2):
        super().__init__()
        mods = []
        for i in range(layers - 1):
            mods += [nn.Linear(channels if i == 0 else inner, inner), nn.GELU(approximate="tanh")]
        mods.append(nn.Linear(inner, out))
        self.mlp = nn.Sequential(*mods)

    def forward(self, x):
        return self.mlp(x)


def exact_pow2(exponents):
    """2^e rounded once from float64.

    The SDK writes `2 ** e` in float32, and torch's vectorised pow is not
    correctly rounded: 4 of the denoiser's 170 frequencies (and 6 of the
    decoders') land one ulp below, and torch's float32 linspace is not
    the exact grid either. At the denoiser's top frequencies (2^16,
    angles to 4e5 rad) one ulp moves a sinusoid by 0.03, which the velocity
    carries as ~3.7e-4 of its scale. The port rounds correctly, and so does
    this transcription, so the grading measures the port rather than torch's
    pow; the gap to the SDK is the figure above."""
    return torch.from_numpy(np.array([2.0 ** float(e) for e in exponents.tolist()], dtype=np.float64)
                            .astype(np.float32))


def point_embed_v1(x, channels, max_res=16):
    """Sinusoids of 3D positions, frequencies 2^0..2^15 then a ramp; zero-padded to `channels`."""
    freq_dim = channels // 3 // 2
    exps = torch.arange(max_res, dtype=torch.float32)
    extra = max(0, freq_dim - max_res)
    ramp = torch.arange(extra, dtype=torch.float32) / extra * max_res
    freqs = exact_pow2(torch.cat([exps, ramp])[:freq_dim])
    *dims, _ = x.shape
    out = torch.outer(x.reshape(-1), freqs) * 2 * torch.pi
    embed = torch.cat([torch.sin(out), torch.cos(out)], dim=-1).reshape(*dims, -1)
    if embed.shape[-1] < channels:
        embed = torch.cat([embed, torch.zeros(*dims, channels - embed.shape[-1])], dim=-1)
    return embed


def point_embed_v2(x, channels, max_res=10):
    """Sinusoids of 3D points: 2^linspace(0, max_res) times pi; zero-padded to `channels`."""
    freq_dim = channels // 3 // 2
    # linspace(0, max_res, freq_dim), each exponent taken in float64.
    freqs = exact_pow2(torch.tensor([max_res * i / (freq_dim - 1) for i in range(freq_dim)], dtype=torch.float64))
    n = x.shape[0]
    ang = x.unsqueeze(-1) * freqs * torch.pi
    embed = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(n, -1)
    if embed.shape[1] < channels:
        embed = torch.cat([embed, torch.zeros(n, channels - embed.shape[1])], dim=-1)
    return embed


class SinusoidalEmbedder(nn.Module):
    def __init__(self, hidden, freq_size=256, max_period=10000, two_pi=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_size, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.freq_size, self.max_period, self.two_pi = freq_size, max_period, two_pi

    def forward(self, t):
        half = self.freq_size // 2
        freqs = torch.exp(-np.log(self.max_period) * torch.arange(half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        if self.two_pi:
            args = args * 2 * torch.pi
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


def rotate(x, freqs):
    """x [B, L, H, D] turned pairwise (2i, 2i+1) by the complex freqs [B, L, H, D/2]."""
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * freqs
    return torch.view_as_real(z).reshape(*x.shape).type_as(x)


class RePo3D(nn.Module):
    def __init__(self, channels, heads, head_dim, ratio=0.125, max_freq=16.0):
        super().__init__()
        self.heads = heads
        inner = int(channels * ratio)
        self.norm = LayerNorm32(channels)
        self.gate_map = nn.Linear(channels, inner, bias=False)
        self.content_map = nn.Linear(channels, inner, bias=False)
        self.final_map = nn.Linear(inner, 3 * heads, bias=False)
        d0 = 2 * (head_dim // 6)
        dims = [d0, d0, head_dim - 2 * d0]
        self.freqs_0 = nn.Parameter(torch.linspace(1.0, max_freq, dims[0] // 2))
        self.freqs_1 = nn.Parameter(torch.linspace(1.0, max_freq, dims[1] // 2))
        self.freqs_2 = nn.Parameter(torch.linspace(1.0, max_freq, dims[2] // 2))

    def forward(self, x):
        h = self.norm(x.float())
        out = self.final_map(F.silu(self.gate_map(h)) * self.content_map(h))
        b, n, _ = out.shape
        delta = out.reshape(b, n, self.heads, 3)
        # clamp_mul(x, f) is x * f in value (its tanh split only shapes the gradient).
        ang = torch.cat([delta[..., 0:1] * self.freqs_0, delta[..., 1:2] * self.freqs_1,
                         delta[..., 2:3] * self.freqs_2], dim=-1) * torch.pi
        return torch.polar(torch.ones_like(ang), ang)


def attend(q, k, v):
    """[B, L, H, D] queries against [B, S, H, D] keys and values."""
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


class RopeAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads, self.head_dim = heads, channels // heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.q_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.k_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, freqs):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).unbind(2)
        q = self.q_norm(rotate(q, freqs))
        k = self.k_norm(rotate(k, freqs))
        return self.out(attend(q, k, v).reshape(b, n, c))


class DenoiserBlock(nn.Module):
    def __init__(self, channels, heads, mlp_ratio=4.0, modulation=True):
        super().__init__()
        self.modulation = modulation
        self.norm1 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.attn = RopeAttention(channels, heads)
        self.mlp = FeedForwardNet(channels, mlp_ratio)
        if modulation:
            self.shift_table = nn.Parameter(torch.zeros(1, 6 * channels))

    def forward(self, x, mod, freqs):
        if not self.modulation:
            x = x + self.attn(self.norm1(x), freqs)
            return x + self.mlp(self.norm2(x))
        mod = mod + self.shift_table
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + self.attn(h, freqs) * gate1.unsqueeze(1)
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        return x + self.mlp(h) * gate2.unsqueeze(1)


class Denoiser(nn.Module):
    def __init__(self, q_token_length, in_channels, cam_channels, model_channels, cond_channels, cond2_channels,
                 num_blocks, num_refiner_blocks=2, num_heads=16, mlp_ratio=4):
        super().__init__()
        c = model_channels
        self.q_token_length, self.in_channels, self.cam_channels = q_token_length, in_channels, cam_channels

        def blocks(n, modulation=True):
            return nn.ModuleList([DenoiserBlock(c, num_heads, mlp_ratio, modulation) for _ in range(n)])

        def repos(n):
            return nn.ModuleList([RePo3D(c, num_heads, 64) for _ in range(n)])

        self.t_embedder = SinusoidalEmbedder(c)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c, 6 * c))
        self.input_layer = nn.Linear(in_channels, c)
        self.cond_embedder = nn.Linear(cond_channels, c)
        self.cond_embedder2 = nn.Linear(cond2_channels, c)
        self.register_buffer("pos_pe", torch.zeros(1, q_token_length, 3))
        self.noise_repo_layers = repos(num_refiner_blocks)
        self.context_repo_layers = repos(num_refiner_blocks)
        self.repo_layers = repos(num_blocks)
        self.noise_refiner = blocks(num_refiner_blocks)
        self.context_refiner = blocks(num_refiner_blocks, modulation=False)
        self.cam_refiner = MLP(cam_channels, c, c, num_refiner_blocks)
        self.blocks = blocks(num_blocks)
        self.shift_table = nn.Parameter(torch.zeros(1, 2, c))
        self.out_layer = nn.Linear(c, in_channels)
        self.cam_out_layer = nn.Linear(c, cam_channels)

    def forward(self, latent, camera, t, feature1, feature2, taps=None):
        h_x = self.input_layer(latent)
        h_cond = self.cond_embedder(feature1) + self.cond_embedder2(feature2)
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb)
        h_x = h_x + point_embed_v1(self.pos_pe, h_x.shape[-1])
        for repo, block in zip(self.noise_repo_layers, self.noise_refiner):
            h_x = block(h_x, t_mod, repo(h_x))
        for repo, block in zip(self.context_repo_layers, self.context_refiner):
            h_cond = block(h_cond, None, repo(h_cond))
        h_cam = self.cam_refiner(camera)
        if taps is not None:
            taps["noise"], taps["context"] = h_x, h_cond
        h = torch.cat([h_x, h_cond, h_cam], dim=1)
        for index, (repo, block) in enumerate(zip(self.repo_layers, self.blocks)):
            h = block(h, t_mod, repo(h))
            if taps is not None and index == 0:
                taps["block0"] = h
        h_x = F.layer_norm(h[:, :latent.shape[1]], h.shape[-1:])
        h_cam = F.layer_norm(h[:, -camera.shape[1]:], h.shape[-1:])
        shift, scale = (self.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
        h_x = h_x * (1 + scale) + shift
        h_cam = h_cam * (1 + scale) + shift
        return self.out_layer(h_x), self.cam_out_layer(h_cam)


class Attention(nn.Module):
    def __init__(self, channels, heads, ctx_channels=None, cross=False):
        super().__init__()
        self.heads, self.cross = heads, cross
        head_dim = channels // heads
        if cross:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(ctx_channels, channels * 2)
        else:
            self.to_qkv = nn.Linear(channels, channels * 3)
        self.q_rms_norm = MultiHeadRMSNorm(head_dim, heads)
        self.k_rms_norm = MultiHeadRMSNorm(head_dim, heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x, context=None):
        b, n, c = x.shape
        if self.cross:
            q = self.to_q(x).reshape(b, n, self.heads, -1)
            k, v = self.to_kv(context).reshape(b, context.shape[1], 2, self.heads, -1).unbind(dim=2)
        else:
            q, k, v = self.to_qkv(x).reshape(b, n, 3, self.heads, -1).unbind(dim=2)
        h = attend(self.q_rms_norm(q), self.k_rms_norm(k), v)
        return self.to_out(h.reshape(b, n, -1))


class CrossBlock(nn.Module):
    """Gaussian decoder block: self attention, cross attention to the latent, FFN."""

    def __init__(self, channels, ctx_channels, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(channels, heads)
        self.cross_attn = Attention(channels, heads, ctx_channels, cross=True)
        self.mlp = FeedForwardNet(channels, mlp_ratio)

    def forward(self, x, context):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        return x + self.mlp(self.norm3(x))


class ModulatedCrossBlock(nn.Module):
    """Octree decoder block: cross attention to the latent and an FFN, both level-modulated."""

    def __init__(self, channels, ctx_channels, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(channels, heads, ctx_channels, cross=True)
        self.mlp = FeedForwardNet(channels, mlp_ratio)

    def forward(self, x, mod, context):
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + self.cross_attn(h, context) * gate1.unsqueeze(1)
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        return x + self.mlp(h) * gate2.unsqueeze(1)


class OctreeDecoder(nn.Module):
    def __init__(self, model_channels, cond_channels, num_blocks, num_heads, mlp_ratio=4.0):
        super().__init__()
        c = model_channels
        self.channels = c
        self.in_proj = nn.Linear(3, c)
        self.input_layer = nn.Linear(c, c)
        self.l_embedder = SinusoidalEmbedder(c, max_period=1024, two_pi=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c, 6 * c))
        self.blocks = nn.ModuleList([ModulatedCrossBlock(c, cond_channels, num_heads, mlp_ratio)
                                     for _ in range(num_blocks)])
        self.out_proj = nn.Linear(c, 8)

    def forward(self, x, level, cond):
        """x: cell centres [B, L, 3] in [0, 1]; level: the children's resolution [B]."""
        b, n, _ = x.shape
        h = self.input_layer(self.in_proj(x) + point_embed_v2(x.view(-1, 3), self.channels).view(b, n, -1))
        mod = self.adaLN_modulation(self.l_embedder(level.float()))
        for block in self.blocks:
            h = block(h, mod, cond)
        return self.out_proj(F.layer_norm(h, h.shape[-1:]))


def systematic_sample(probs, counts, rand):
    """Split counts[r] items among the bins of probs[r] by systematic resampling; rand(r) -> r uniforms."""
    rows, bins = probs.shape
    probs = probs.to(torch.float32).clamp_min(0)
    sums = probs.sum(1, keepdim=True)
    zero = sums.eq(0)
    probs = probs / sums.clamp_min(1)
    probs = torch.where(zero.expand_as(probs), torch.full_like(probs, 1.0 / bins), probs)
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
    out = torch.zeros(rows, bins, dtype=torch.long)
    draws = torch.zeros(rows)
    unique_n, inv = counts.unique(sorted=False, return_inverse=True)
    for index, n in enumerate(unique_n.tolist()):
        if n == 0:
            continue
        sel = (inv == index).nonzero(as_tuple=False).squeeze(1)
        u = rand(sel.numel())
        draws[sel] = u
        us = (u[:, None] / float(n) + torch.arange(n, dtype=torch.float32)[None, :] / float(n)).clamp(max=1.0 - 1e-12)
        idx = torch.searchsorted(cdf.index_select(0, sel), us).clamp_max(bins - 1)
        buf = torch.zeros(sel.numel(), bins)
        buf.scatter_add_(1, idx, torch.ones_like(idx, dtype=buf.dtype))
        out.index_copy_(0, sel, buf.to(torch.long))
    return out, draws


def place_anchors(model, cond, num_points, level, rand):
    """The octree descent for one latent; returns anchors [num_points, 3] and the per-level draws."""
    child = torch.tensor([[i, j, k] for k in [0, 1] for j in [0, 1] for i in [0, 1]], dtype=torch.long)
    coords = torch.zeros(1, 3, dtype=torch.long)
    counts = torch.full((1,), num_points, dtype=torch.long)
    level_draws = []
    for depth in range(1, level + 1):
        parents = (coords.float() + 0.5) / (1 << (depth - 1))
        probs = torch.softmax(model(parents[None], torch.tensor([1 << depth]), cond)[0], dim=-1)
        split, draws = systematic_sample(probs, counts, rand)
        level_draws.append(draws)
        children = (coords[:, None, :] * 2 + child[None]).reshape(-1, 3)
        split = split.reshape(-1)
        keep = split > 0
        coords, counts = children[keep], split[keep]
    leaf = torch.repeat_interleave(coords, counts, dim=0)
    jitter = rand(leaf.numel()).reshape(leaf.shape)
    return (leaf.float() + jitter) / (1 << level), level_draws, jitter


class GaussianDecoder(nn.Module):
    LAYOUT = {"_xyz": 3, "_features_dc": 3, "_scaling": 3, "_rotation": 4, "_opacity": 1, "_offset_scale": 1}

    def __init__(self, in_channels, model_channels, cond_channels, num_blocks, num_heads, representation_config,
                 mlp_ratio=4.0):
        super().__init__()
        c = model_channels
        self.channels = c
        self.rep = representation_config
        per = representation_config["num_gaussians"]
        self.ranges, start = {}, 0
        for name, width in self.LAYOUT.items():
            self.ranges[name] = (start, start + per * width)
            start += per * width
        self.in_proj = nn.Linear(in_channels, c)
        self.input_layer = nn.Linear(c, c)
        self.blocks = nn.ModuleList([CrossBlock(c, cond_channels, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(c, start)
        self.register_buffer("points_offset_perturbation", torch.zeros(per, 3))
        self.register_buffer("base_offset_scale", torch.zeros(()))

    def forward(self, points, cond):
        b, n, c = points.shape
        h = self.input_layer(self.in_proj(points) + point_embed_v2(points.view(-1, c), self.channels).view(b, n, -1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(F.layer_norm(h, h.shape[-1:]))

    def activate(self, points, features):
        """One cloud's anchors [L, 3] and outputs [L, D] -> xyz, colour, scaling, rotation, opacity."""
        rep, per = self.rep, self.rep["num_gaussians"]

        def part(name):
            start, end = self.ranges[name]
            return features[:, start:end].reshape(features.shape[0], per, self.LAYOUT[name])

        lr = rep["lr"]
        offset_scale = F.softplus(part("_offset_scale") + self.base_offset_scale)
        offset = torch.tanh(part("_xyz") * lr["_xyz"] + self.points_offset_perturbation)
        offset = offset * 0.5 * rep["perturbe_size"] * offset_scale
        xyz = (offset + points[:, None, :]).reshape(-1, 3) - 0.5
        colour = (part("_features_dc") * lr["_features_dc"]).reshape(-1, 3)
        scaling_bias = torch.tensor(rep["scaling_bias"])
        scale_offset = scaling_bias + torch.log(-torch.expm1(-scaling_bias))
        scaling = F.softplus((part("_scaling") * lr["_scaling"]).reshape(-1, 3) + scale_offset)
        scaling = torch.sqrt(scaling ** 2 + rep["3d_filter_kernel_size"] ** 2)
        rotation = F.normalize((part("_rotation") * lr["_rotation"]).reshape(-1, 4)
                               + torch.tensor([1.0, 0.0, 0.0, 0.0]))
        p = torch.tensor(rep["opacity_bias"])
        opacity = torch.sigmoid((part("_opacity") * lr["_opacity"]).reshape(-1) + torch.log(p / (1 - p)))
        return xyz, colour, scaling, rotation, opacity


def ply_bytes(xyz, colour, scaling, rotation, opacity):
    """The standard 3DGS PLY: y-up turn, clamped inverse sigmoid and log, seventeen little-endian floats."""
    eps = 1e-6
    n = xyz.shape[0]
    logit = torch.log(opacity.clamp(eps, 1 - eps) / (1 - opacity.clamp(eps, 1 - eps)))
    log_scale = torch.log(scaling.clamp_min(eps))
    turned = torch.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], dim=1)
    a = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
    w, x, y, z = rotation.unbind(1)
    rot = torch.stack([a[0] * w - a[1] * x - a[2] * y - a[3] * z, a[0] * x + a[1] * w + a[2] * z - a[3] * y,
                       a[0] * y - a[1] * z + a[2] * w + a[3] * x, a[0] * z + a[1] * y - a[2] * x + a[3] * w], dim=1)
    rows = torch.cat([turned, torch.zeros(n, 3), colour, logit[:, None], log_scale, rot], dim=1)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
             "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join("property float %s\n" % name for name in names) + "end_header\n"
    return header.encode("ascii") + rows.numpy().astype("<f4").tobytes()


# ---------------------------------------------------------------------------------------------------------------------
# The background remover: a Swin-L backbone run at two scales, an ASPP-deformable squeeze block and a four-level
# decoder that re-reads the picture at every level. Names are the checkpoint's; loaded strictly.
# ---------------------------------------------------------------------------------------------------------------------

def _window_partition(x, ws):
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, c)


def _window_reverse(windows, ws, h, w):
    b = int(windows.shape[0] / (h * w / ws / ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class _SwinMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads):
        super().__init__()
        self.ws, self.heads = ws, heads
        self.scale = (dim // heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid([torch.arange(ws), torch.arange(ws)], indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n, n, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = (attn.view(b // nw, nw, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)).view(-1, self.heads, n, n)
        return self.proj((attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c))


class _SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, ws, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _SwinMlp(dim, dim * 4)

    def forward(self, x, h, w, mask):
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        pad_r, pad_b = (self.ws - w % self.ws) % self.ws, (self.ws - h % self.ws) % self.ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        hp, wp = x.shape[1], x.shape[2]
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = self.attn(_window_partition(x, self.ws).view(-1, self.ws * self.ws, c),
                            mask=mask if self.shift > 0 else None)
        x = _window_reverse(windows.view(-1, self.ws, self.ws, c), self.ws, hp, wp)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x[:, :h, :w, :].reshape(b, h * w, c)
        return x + self.mlp(self.norm2(x))


class _PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, h, w):
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.view(b, -1, 4 * c)))


class _SwinLayer(nn.Module):
    def __init__(self, dim, depth, heads, ws, downsample):
        super().__init__()
        self.ws, self.shift = ws, ws // 2
        self.blocks = nn.ModuleList([_SwinBlock(dim, heads, ws, 0 if i % 2 == 0 else ws // 2) for i in range(depth)])
        self.downsample = _PatchMerging(dim) if downsample else None

    def forward(self, x, h, w):
        hp, wp = math.ceil(h / self.ws) * self.ws, math.ceil(w / self.ws) * self.ws
        img = torch.zeros((1, hp, wp, 1))
        cnt = 0
        for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
            for wsl in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                img[:, hs, wsl, :] = cnt
                cnt += 1
        mw = _window_partition(img, self.ws).view(-1, self.ws ** 2)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        for blk in self.blocks:
            x = blk(x, h, w, mask)
        if self.downsample is not None:
            return x, h, w, self.downsample(x, h, w), (h + 1) // 2, (w + 1) // 2
        return x, h, w, x, h, w


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 192, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(192)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class _SwinLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()
        dims, depths, heads = [192, 384, 768, 1536], [2, 2, 18, 2], [6, 12, 24, 48]
        self.dims = dims
        self.layers = nn.ModuleList([_SwinLayer(dims[i], depths[i], heads[i], 12, i < 3) for i in range(4)])
        for i in range(4):
            self.add_module(f"norm{i}", nn.LayerNorm(dims[i]))

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        outs = []
        for i in range(4):
            x_out, hh, ww, x, h, w = self.layers[i](x, h, w)
            x_out = getattr(self, f"norm{i}")(x_out)
            outs.append(x_out.view(-1, hh, ww, self.dims[i]).permute(0, 3, 1, 2).contiguous())
        return tuple(outs)


class _DeformableConv2d(nn.Module):
    def __init__(self, cin, cout, k, padding):
        super().__init__()
        self.padding = padding
        self.offset_conv = nn.Conv2d(cin, 2 * k * k, kernel_size=k, padding=padding)
        self.modulator_conv = nn.Conv2d(cin, k * k, kernel_size=k, padding=padding)
        self.regular_conv = nn.Conv2d(cin, cout, kernel_size=k, padding=padding, bias=False)

    def forward(self, x):
        from torchvision.ops import deform_conv2d
        return deform_conv2d(x, self.offset_conv(x), self.regular_conv.weight, self.regular_conv.bias,
                             padding=self.padding, mask=2.0 * torch.sigmoid(self.modulator_conv(x)))


class _ASPPModule(nn.Module):
    def __init__(self, cin, planes, k, padding):
        super().__init__()
        self.atrous_conv = _DeformableConv2d(cin, planes, k, padding)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return F.relu(self.bn(self.atrous_conv(x)))


class _ASPPDeformable(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.aspp1 = _ASPPModule(cin, 256, 1, 0)
        self.aspp_deforms = nn.ModuleList([_ASPPModule(cin, 256, k, k // 2) for k in (1, 3, 7)])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(cin, 256, 1, bias=False),
                                             nn.BatchNorm2d(256), nn.ReLU())
        self.conv1 = nn.Conv2d(256 * 5, cin, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)

    def forward(self, x):
        x1 = self.aspp1(x)
        rest = [m(x) for m in self.aspp_deforms]
        x5 = F.interpolate(self.global_avg_pool(x), size=x1.shape[2:], mode="bilinear", align_corners=True)
        return F.relu(self.bn1(self.conv1(torch.cat((x1, *rest, x5), dim=1))))


class _BasicDecBlk(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv_in = nn.Conv2d(cin, 64, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(64)
        self.dec_att = _ASPPDeformable(64)
        self.conv_out = nn.Conv2d(64, cout, 3, 1, padding=1)
        self.bn_out = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn_out(self.conv_out(self.dec_att(F.relu(self.bn_in(self.conv_in(x))))))


class _Lateral(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.conv(x)


class _SimpleConvs(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, 64, 3, 1, 1)
        self.conv_out = nn.Conv2d(64, cout, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


def image_to_patches(image, ref):
    """(b, c, hg*h, wg*w) -> (b, c*hg*wg, h, w): the picture cut into ref-sized patches stacked on channels."""
    b, c, hf, wf = image.shape
    hg, wg = hf // ref.shape[-2], wf // ref.shape[-1]
    return image.view(b, c, hg, hf // hg, wg, wf // wg).permute(0, 1, 2, 4, 3, 5).reshape(b, c * hg * wg,
                                                                                        hf // hg, wf // wg)


def _up(x, ref):
    return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=True)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        c = (3072, 1536, 768, 384)
        self.ipt_blk5 = _SimpleConvs(2 ** 10 * 3, c[0] // 8)
        self.ipt_blk4 = _SimpleConvs(2 ** 8 * 3, c[0] // 8)
        self.ipt_blk3 = _SimpleConvs(2 ** 6 * 3, c[1] // 8)
        self.ipt_blk2 = _SimpleConvs(2 ** 4 * 3, c[2] // 8)
        self.ipt_blk1 = _SimpleConvs(3, c[3] // 8)
        self.decoder_block4 = _BasicDecBlk(c[0] + c[0] // 8, c[1])
        self.decoder_block3 = _BasicDecBlk(c[1] + c[0] // 8, c[2])
        self.decoder_block2 = _BasicDecBlk(c[2] + c[1] // 8, c[3])
        self.decoder_block1 = _BasicDecBlk(c[3] + c[2] // 8, c[3] // 2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(c[3] // 2 + c[3] // 8, 1, 1))
        self.lateral_block4 = _Lateral(c[1])
        self.lateral_block3 = _Lateral(c[2])
        self.lateral_block2 = _Lateral(c[3])
        for i, cin in ((4, c[1]), (3, c[2]), (2, c[3])):
            setattr(self, f"gdt_convs_{i}", nn.Sequential(nn.Conv2d(cin, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU()))
            setattr(self, f"gdt_convs_attn_{i}", nn.Sequential(nn.Conv2d(16, 1, 1)))

    def forward(self, x, x1, x2, x3, x4, taps=None):
        x4 = torch.cat((x4, self.ipt_blk5(image_to_patches(x, x4))), 1)
        p4 = self.decoder_block4(x4)
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        p3 = _up(p4, x3) + self.lateral_block4(x3)
        p3 = torch.cat((p3, self.ipt_blk4(image_to_patches(x, p3))), 1)
        p3 = self.decoder_block3(p3)
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
        p2 = _up(p3, x2) + self.lateral_block3(x2)
        p2 = torch.cat((p2, self.ipt_blk3(image_to_patches(x, p2))), 1)
        p2 = self.decoder_block2(p2)
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
        p1 = _up(p2, x1) + self.lateral_block2(x1)
        p1 = torch.cat((p1, self.ipt_blk2(image_to_patches(x, p1))), 1)
        p1 = _up(self.decoder_block1(p1), x)
        if taps is not None:
            taps["p4"], taps["p1"] = p4, p1
        p1 = torch.cat((p1, self.ipt_blk1(image_to_patches(x, p1))), 1)
        return self.conv_out1(p1)


class BiRefNet(nn.Module):
    INPUT_SIZE = (1024, 1024)
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self.bb = _SwinLarge()
        self.squeeze_module = nn.Sequential(_BasicDecBlk(3072 + 384 + 768 + 1536, 3072))
        self.decoder = _Decoder()

    def forward(self, x, taps=None):
        x1, x2, x3, x4 = self.bb(x)
        h, w = x.shape[2:]
        y1, y2, y3, y4 = self.bb(F.interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True))
        x1 = torch.cat([x1, _up(y1, x1)], 1)
        x2 = torch.cat([x2, _up(y2, x2)], 1)
        x3 = torch.cat([x3, _up(y3, x3)], 1)
        x4 = torch.cat([x4, _up(y4, x4)], 1)
        x4 = torch.cat([_up(x1, x4), _up(x2, x4), _up(x3, x4), x4], 1)
        if taps is not None:
            taps["x1"], taps["x4"] = x1, x4
        x4 = self.squeeze_module(x4)
        if taps is not None:
            taps["squeezed"] = x4
        return torch.sigmoid(self.decoder(x, x1, x2, x3, x4, taps))

    def alpha(self, rgb_u8, taps=None):
        """(H, W, 3) uint8 -> (H, W) uint8 matte: resized to 1024 and back, bilinear with aligned corners."""
        h, w = rgb_u8.shape[:2]
        t = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=self.INPUT_SIZE, mode="bilinear", align_corners=True)
        t = (t - torch.tensor(self.MEAN).view(1, 3, 1, 1)) / torch.tensor(self.STD).view(1, 3, 1, 1)
        out = self(t, taps)
        if taps is not None:
            taps["input"], taps["mask"] = t, out
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)[0, 0]
        return (out.clamp(0, 1) * 255).to(torch.uint8).numpy()
