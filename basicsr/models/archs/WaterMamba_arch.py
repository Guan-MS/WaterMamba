import numbers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

# use GN for norm layer
def norm_layer(channels):
    return nn.GroupNorm(channels//2, channels)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

## CCOSS
class ChannelMamba(nn.Module):
    def __init__(
        self,
        d_model,
        dim=None,
        d_state=16,
        d_conv=4,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        bimamba_type="v2",
        if_devide_out=False
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.if_devide_out = if_devide_out
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.bimamba_type = bimamba_type
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(normalized_shape=dim)
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.conv2d = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            bias=conv_bias,
            kernel_size=3,
            groups=dim,
            padding=1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

    def forward(self, u):
        """
        u: (B, H, W, C)
        Returns: same shape as hidden_states
        """
        b, d, h, w = u.shape
        l = h * w
        u = rearrange(u, "b d h w-> b (h w) d").contiguous()

        conv_state, ssm_state = None, None

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(u, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=l,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        x, z = xz.chunk(2, dim=1)
        x = rearrange(self.conv2d(rearrange(x, "b l d -> b d 1 l")), "b d 1 l -> b l d")

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=l)
        B = rearrange(B, "(b l) d -> b d l", l=l).contiguous()
        C = rearrange(C, "(b l) d -> b d l", l=l).contiguous()

        x_dbl_b = self.x_proj_b(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b = self.dt_proj_b.weight @ dt_b.t()
        dt_b = rearrange(dt_b, "d (b l) -> b d l", l=l)
        B_b = rearrange(B_b, "(b l) d -> b d l", l=l).contiguous()
        C_b = rearrange(C_b, "(b l) d -> b d l", l=l).contiguous()
        if self.bimamba_type == "v1":
            A_b = -torch.exp(self.A_b_log.float())
            out = selective_scan_fn(
                x,
                dt,
                A_b,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
        elif self.bimamba_type == "v2":
            A_b = -torch.exp(self.A_b_log.float())
            out = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            out_b = selective_scan_fn(
                x.flip([-1]),
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            out = self.ln(out) * self.act(z)
            out_b = self.ln1(out_b) * self.act(z)
            if not self.if_devide_out:
                out = rearrange(out + out_b.flip([-1]), "b l (h w) -> b l h w", h=h, w=w)
            else:
                out = rearrange(out + out_b.flip([-1]), "b l (h w) -> b l h w", h=h, w=w) / 2

        return out

class CCOSS(nn.Module):
    def __init__(self,channel=3, w=128, h=128, d_state=16, expand=1, d_conv=4, mam_block=2):
        super().__init__()

        self.H_CSSM = nn.Sequential(*[
            ChannelMamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=h,
                dim=channel,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
            ) for i in
            range(mam_block)])

        self.W_CSSM = nn.Sequential(*[
            ChannelMamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=w,
                dim=channel,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
            ) for i in
            range(mam_block)])

        self.channel = channel
        self.ln = nn.LayerNorm(normalized_shape=channel)
        self.softmax = nn.Softmax(1)

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,
                                  bias=False)

        self.dwconv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, groups=channel,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel)

        self.silu_h = nn.SiLU()
        self.silu_w = nn.SiLU()

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_s = x.contiguous()
        b, c, w, h = x.shape
        x = rearrange(x, "b c h w-> b (h w) c ")
        x = (self.ln(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x_in = x
        x_shotcut = self.softmax(self.dwconv(x))
        x_h = torch.mean(x_in, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x_in, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        x_h = x_cat_conv_split_h.permute(0, 3, 2, 1)#input=[b, h, 1, c]
        x_h= self.H_CSSM(x_h).permute(0, 3, 2, 1)
        x_w = x_cat_conv_split_w.permute(0, 3, 2, 1)
        x_w = self.W_CSSM(x_w).permute(0, 3, 2, 1)  #input=[b, w, 1, c]
        s_h = self.sigmoid_h(x_h.permute(0, 1, 3, 2))
        s_w = self.sigmoid_w(x_w)
        out = s_h.expand_as(x) * s_w.expand_as(x) * x_shotcut

        return out + x_s

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out

class SCOSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 48,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            d_conv=4,
            h=64,
            w=64,
            mam_block=2,
            **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ln_1 = norm_layer(hidden_dim)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.SOSS = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.conv_blk = MSFFN(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.CCOSS = CCOSS(channel=hidden_dim, h=h, w=w, d_state=d_state, d_conv=d_conv, mam_block=mam_block)

    def forward(self, input):
        # x [B,C,H,W]

        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)
        x = input + self.drop_path1(self.SOSS(x))
        x = x.permute(0, 3, 1, 2)
        x = ((input * self.skip_scale).permute(0, 3, 1, 2) + self.drop_path2(self.CCOSS(x))).permute(0, 2, 3, 1)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2)

class MSFFN(nn.Module):
    def __init__(self, in_channels):
        super(MSFFN, self).__init__()
        self.a = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
        nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, groups=in_channels)
        )
        self.a1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
        nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)
        )
        self.relu1 = nn.ReLU()

        self.b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )
        self.relu2 = nn.ReLU()

        self.c = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
        nn.Conv2d(in_channels, in_channels, 5, stride=1, padding=2, groups=in_channels)
        )
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
        nn.Conv2d(in_channels, in_channels, 5, stride=1, padding=2, groups=in_channels)
        )
        self.relu3 = nn.ReLU()
        self.conv_out = nn.Conv2d(in_channels * 3, in_channels,1)
        self.ln = nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, x):
        x_in = x
        b, c, h, w = x.shape
        x = (self.ln(x.view(b, -1, c))).view(b, c, w, h)
        x1 = self.a1(self.relu1(self.a(x)))
        x2 = self.b1(self.b(x))
        x3 = self.c1(self.c(x))
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.conv_out(out)
        return out + x_in

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)

def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)

    return x

##########################################################################
class WaterMamba(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=24,
                 drop_path=0.,
                 attn_drop_rate=0.,
                 resolution=[256, 256],
                 num_blocks=[1, 1, 1, 1],
                 mam_blocks=[2, 2, 2, 2],
                 d_state=16,
                 expand=2,
                 d_conv=4,
                 bias=False,
                 ):

        super(WaterMamba, self).__init__()
        h = resolution[0]
        w = resolution[1]
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            SCOSSBlock(hidden_dim=dim, h=h, w=w, d_state=d_state, expand=expand, mam_block=mam_blocks[0],
                       d_conv=d_conv,drop_path=drop_path, attn_drop_rate=attn_drop_rate) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2 , init_value, heads_range, value_factor=1
        self.encoder_level2 = nn.Sequential(*[
            SCOSSBlock(hidden_dim=dim * 2, h=h//2, w=w//2, d_state=d_state, expand=expand,  mam_block=mam_blocks[1],
                       d_conv=d_conv,drop_path=drop_path, attn_drop_rate=attn_drop_rate) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            SCOSSBlock(hidden_dim=int(dim * 2 ** 2), h=h//4, w=w//4, d_state=d_state, expand=expand, mam_block=mam_blocks[2],
                       d_conv=d_conv,drop_path=drop_path, attn_drop_rate=attn_drop_rate) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            SCOSSBlock(hidden_dim=int(dim * 2 ** 3), h=h//8, w=w//8, d_state=d_state, expand=expand, mam_block=mam_blocks[3],
                       d_conv=d_conv,drop_path=drop_path, attn_drop_rate=attn_drop_rate) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            SCOSSBlock(hidden_dim=int(dim * 2 ** 2), h=h//4, w=w//4, d_state=d_state, expand=expand, mam_block=mam_blocks[2],
                       d_conv=d_conv,drop_path=drop_path, attn_drop_rate=attn_drop_rate) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            SCOSSBlock(hidden_dim=int(dim * 2 ** 1), h=h//2, w=w//2, d_state=d_state, expand=expand, mam_block=mam_blocks[1],
                       d_conv=d_conv,drop_path=drop_path, attn_drop_rate=attn_drop_rate) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[
            SCOSSBlock(hidden_dim=dim, h=h, w=w, d_state=d_state, expand=expand, mam_block=mam_blocks[0],
                       d_conv=d_conv,drop_path=drop_path, attn_drop_rate=attn_drop_rate) for i in range(num_blocks[0])])

        self.output = nn.Sequential(
            norm_layer(int(dim)),
            nn.SiLU(),
            nn.Conv2d(int(dim), out_channels, kernel_size=1,bias=bias),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = cat(inp_dec_level3, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = cat(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = cat(inp_dec_level1, out_enc_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return self.output(out_dec_level1) + inp_img

from thop import profile
if __name__ == '__main__':
    data = torch.randn([1, 3, 256, 256]).cuda()
    model = WaterMamba(3,resolution=[256,256]).cuda()
    out = model(data)
    print(out.shape)
    flops, params = profile(model, (data, ))
    print("flops: ", flops / 1e9, "params: ", params / 1e6)


