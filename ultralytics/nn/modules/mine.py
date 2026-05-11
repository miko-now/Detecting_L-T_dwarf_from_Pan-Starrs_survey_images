import torch
from torch import nn
import torch.nn.functional as F
from .block import C3,C2f
from typing import Sequence, Optional
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7]):
        super().__init__()
        self.eca=ECA(dim_xh,k_size=k_size)
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size+(k_size-1)*(d_list[0]-1))//2,
                      dilation=d_list[0], groups=group_size )
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
                      padding=(k_size+(k_size-1)*(d_list[1]-1))//2,
                      dilation=d_list[1], groups=group_size )
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
                      padding=(k_size+(k_size-1)*(d_list[2]-1))//2,
                      dilation=d_list[2], groups=group_size )
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
                      padding=(k_size+(k_size-1)*(d_list[3]-1))//2,
                      dilation=d_list[3], groups=group_size )
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 , data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 , dim_xl, 1)
        )
    def forward(self, xh, xl):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = self.eca(xl)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0]), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1]), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2]), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3]), dim=1))
        x = torch.cat((x0,x1,x2,x3), dim=1)
        x = self.tail_conv(x)
        return x
class ECA(nn.Module):
    def __init__(self, c, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):                 # x: [B, C, H, W]
        y = F.adaptive_avg_pool2d(x, 1)   # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))   # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(6, 12, 18, 24), use_pool=True, use_pool_bn=False):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1 if d == 1 else 3, padding=0 if d == 1 else d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])

        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.AdaptiveAvgPool2d(1)
            # 🚫 关键：池化分支默认不用 BN，避免 [1, C, 1, 1] 报错
            if use_pool_bn:
                norm = nn.BatchNorm2d(out_ch)          # 仅在你明确需要时打开
            else:
                norm = nn.Identity()                    # 或者换成 GroupNorm/LayerNorm
                # norm = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
                # norm = LayerNorm(out_ch, data_format='channels_first')  # 若你已实现

            self.pool_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                norm,
                nn.ReLU(inplace=True),
            )

        fuse_in = out_ch * (len(dilations) + (1 if use_pool else 0))
        self.project = nn.Sequential(
            nn.Conv2d(fuse_in, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        if self.use_pool:
            p = self.pool(x)
            p = self.pool_proj(p)                       # 这里已经不会触发 BN 报错
            p = F.interpolate(p, size=x.shape[-2:], mode='bilinear', align_corners=False)
            feats.append(p)
        y = torch.cat(feats, dim=1)
        return self.project(y)
class ECA_ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=None, dilations=(1, 6, 12, 18), eca_k=3):
        super().__init__()
        out_ch = out_ch or in_ch   # None -> 保持通道不变
        self.aspp = ASPP(in_ch, out_ch, dilations=dilations, use_pool=True)
        self.eca  = ECA(out_ch, k_size=eca_k)
    def forward(self, x):
        return self.eca(self.aspp(x))


class SKConv(nn.Module):
    def __init__(self, channels, M=2, G=32, r=16, L=32,
                 kernel_sizes=None, dilations=(1, 2), stride=1, bias=False):
        super().__init__()
        # ------- 自修正开始 -------
        # kernel_sizes 与 dilations 二选一；若给了 kernel_sizes，则忽略 dilations
        if kernel_sizes is not None:
            ks = list(kernel_sizes)
            if len(ks) < M:
                ks += [ks[-1]] * (M - len(ks))
            elif len(ks) > M:
                ks = ks[:M]
            kernel_sizes = ks
            dilations = None
        else:
            if dilations is None or len(dilations) == 0:
                dilations = tuple(range(1, M + 1))  # (1,2,...,M)
            elif len(dilations) < M:
                dilations = tuple(list(dilations) + [dilations[-1]] * (M - len(dilations)))
            elif len(dilations) > M:
                dilations = tuple(dilations[:M])
        # ------- 自修正结束 -------

        self.channels = channels
        self.M = M
        self.G = max(1, min(G, channels))
        self.d = max(L, channels // r)

        self.branches = nn.ModuleList()
        if kernel_sizes is not None:
            for k in kernel_sizes:
                p = k // 2
                self.branches.append(nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k, stride=stride,
                              padding=p, groups=self.G, bias=bias),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ))
        else:
            for d in dilations:
                k = 3
                p = d
                self.branches.append(nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k, stride=stride,
                              padding=p, dilation=d, groups=self.G, bias=bias),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ))
        # 其余保持不变...


        # --- 融合后的通道注意力（共享降维 + 分支专属升维）---
        self.fc = nn.Sequential(
            nn.Conv2d(channels, self.d, kernel_size=1, bias=True),  # 共享：C -> d
            nn.SiLU(inplace=True),
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(self.d, channels, kernel_size=1, bias=True)   # 每个分支都有一条升维 d -> C
            for _ in range(M)
        ])
        self.softmax = nn.Softmax(dim=1)  # 在分支维 M 上做 softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多分支特征
        feats = [op(x) for op in self.branches]              # list of [B,C,H,W]
        U = sum(feats)                                       # 聚合特征 U
        s = F.adaptive_avg_pool2d(U, 1)                      # [B,C,1,1]
        z = self.fc(s)                                       # [B,d,1,1]

        # 生成每个分支的注意力向量
        a_list = [fc(z) for fc in self.fcs]                  # M × [B,C,1,1]
        a = torch.stack(a_list, dim=1)                       # [B,M,C,1,1]
        a = self.softmax(a)                                  # 分支维 softmax

        # 融合：对每个分支乘权后求和
        V = sum(a[:, i, ...] * feats[i] for i in range(self.M))
        return V


class SKLayer(nn.Module):
    """
    YAML 友好封装：在 C2f(15) 之后插入
    - 自动读取输入通道 c1，默认保持通道不变；如需改通道可传 c2。
    """
    def __init__(self, c1, c2=None, M=2, G=32, r=16, L=32, kernel_sizes=None, dilations=(1,2), stride=1):
        super().__init__()
        c2 = c1 if c2 is None else c2
        # 若 c1 != c2，则先 1x1 对齐到 c2
        self.pre = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True)
        )
        self.sk = SKConv(channels=c2, M=M, G=G, r=r, L=L,
                         kernel_sizes=kernel_sizes, dilations=dilations, stride=stride, bias=False)

    def forward(self, x):
        x = self.pre(x)
        return self.sk(x)

class GABridge(nn.Module):
    """
    YAML 友好封装：多输入版 group_aggregation_bridge
    - from: [xh_index, xl_index]
    - args: [dim_xh, dim_xl, k_size, d_list]
    注意：dim_xh/ dim_xl 要与你接入的两个分支通道一致
    """
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=(1,2,5,7)):
        super().__init__()
        self.mod = group_aggregation_bridge(dim_xh=dim_xh, dim_xl=dim_xl,
                                            k_size=k_size, d_list=list(d_list))

    def forward(self, x):  # x 为 list/tuple: [xh, xl]
        xh, xl = x
        return self.mod(xh, xl)


class ChannelAttention(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        c_ = max(1, c1 // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c1, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))
        w = self.sigmoid(a + m)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        assert k in (3, 7)
        p = (k - 1) // 2
        self.conv = nn.Conv2d(2, 1, k, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([mean, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """CBAM for YOLO: keep shape, no extra args required beyond channels."""
    def __init__(self, c1, reduction=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(c1, reduction=reduction)
        self.sa = SpatialAttention(k=k)

    def forward(self, x):
        return self.sa(self.ca(x))

from ultralytics.nn.modules.conv import Conv
class Res2Conv(nn.Module):
    """
    Residual Double 3x3 Conv Block
    - main: 3x3 Conv -> 3x3 Conv
    - skip: identity if shape matches else 3x3 Conv projection
    Args in YAML (c2, s=1, g=1, d=1, act=True)
      - c2: output channels
      - s : stride for the first conv (downsample if s=2)
      - g : groups (usually 1)
      - d : dilation (>=1)
      - act: whether use activation in Conv layers
    """
    def __init__(self, c1, c2, s=1, g=1, d=1, act=True):
        super().__init__()

        # 主分支：两次 3x3
        # 第1个 3x3 负责可能的下采样（stride=s）
        self.cv1 = Conv(c1, c2, k=3, s=s, p=None, g=g, d=d, act=act)
        # 第2个 3x3 stride 固定 1
        self.cv2 = Conv(c2, c2, k=3, s=1, p=None, g=g, d=d, act=act)

        # 捷径分支：默认 identity；若不匹配则用 3x3 投影对齐
        self.use_proj = (c1 != c2) or (s != 1)
        self.proj = Conv(c1, c2, k=3, s=s, p=None, g=1, d=d, act=False) if self.use_proj else nn.Identity()

        # 最终激活（残差相加后）
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        skip = self.proj(x)
        return self.act(y + skip)


class EMCA(nn.Module):
    """
    EMCA: Efficient Multi-scale Channel Attention
    - Global AvgPool -> [B, C, 1]
    - Multiple 1D convs with different kernel sizes -> fuse -> sigmoid -> channel weights
    - Apply as residual attention: y = x * (1 + w)

    Args:
        c1 (int): input channels (Ultralytics parse_model will inject this)
        ks (tuple[int]): kernel sizes for multi-scale 1D conv (must be odd)
        fuse (str): 'sum' or 'concat' (sum is lighter)
        residual (bool): use residual attention x*(1+w) instead of x*w
    """
    def __init__(self, c1, ks=(3, 5, 7), fuse="sum", residual=True):
        super().__init__()
        assert fuse in ("sum", "concat")
        for k in ks:
            assert k % 2 == 1, f"EMCA kernel size must be odd, got {k}"

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = fuse
        self.residual = residual

        # 1D conv operates on channel descriptor [B, 1, C]
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
            for k in ks
        ])

        if fuse == "concat":
            self.fuse_conv = nn.Conv1d(len(ks), 1, kernel_size=1, bias=False)
        else:
            self.fuse_conv = None

        self.act = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, 1, c)  # [B, 1, C]

        outs = [conv(y) for conv in self.convs]  # each: [B, 1, C]

        if self.fuse == "sum":
            z = outs[0]
            for t in outs[1:]:
                z = z + t
        else:  # concat
            z = torch.cat(outs, dim=1)          # [B, K, C]
            z = self.fuse_conv(z)               # [B, 1, C]

        w = self.act(z).view(b, c, 1, 1)        # [B, C, 1, 1]

        if self.residual:
            return x * (1.0 + w)
        else:
            return x * w


class PConv(nn.Module):
    """
    Partial Convolution (PConv)
    Only apply kxk conv on a portion of channels, remaining channels are bypassed,
    then fuse with a pointwise (1x1) conv.

    Args:
        c1 (int): input channels
        c2 (int): output channels
        n_div (int): channel split divisor. conv_channels = c1 // n_div
        k (int): kernel size for partial conv (default 3)
        s (int): stride for partial conv (default 1)
        g (int): groups for partial conv (default 1). (Usually 1)
        act (bool): activation
        fuse_k (int): fusion conv kernel size, default 1 (recommended). You may set 3 if you insist.
    """
    def __init__(self, c1, c2, n_div=4, k=3, s=1, g=1, act=True, fuse_k=1):
        super().__init__()
        assert n_div >= 2, "n_div should be >= 2"
        assert k in (1, 3, 5, 7), "k should be a common odd kernel"
        assert fuse_k in (1, 3), "fuse_k should be 1 or 3"

        self.c1 = c1
        self.c2 = c2
        self.n_div = n_div

        # split channels
        self.c_partial = max(1, c1 // n_div)
        self.c_bypass = c1 - self.c_partial

        # partial conv on subset channels (kxk)
        self.partial_conv = Conv(self.c_partial, self.c_partial, k=k, s=s, g=g, act=act)

        # If stride>1, bypass branch needs downsample to match spatial size
        self.bypass_down = None
        if s != 1 and self.c_bypass > 0:
            # use 1x1 downsample to match size (lightweight)
            self.bypass_down = Conv(self.c_bypass, self.c_bypass, k=1, s=s, act=False)

        # fuse conv to mix channels and adjust to c2
        self.fuse = Conv(c1, c2, k=fuse_k, s=1, act=act)

    def forward(self, x):
        # split
        x1 = x[:, :self.c_partial, :, :]
        x2 = x[:, self.c_partial:, :, :] if self.c_bypass > 0 else None

        y1 = self.partial_conv(x1)

        if x2 is None:
            y = y1
        else:
            y2 = self.bypass_down(x2) if self.bypass_down is not None else x2
            y = torch.cat((y1, y2), dim=1)

        return self.fuse(y)

class DWConv(nn.Module):
    """
    Depthwise Separable Convolution
    = Depthwise kxk (groups=c1) + Pointwise 1x1

    Args:
        c1 (int): input channels
        c2 (int): output channels
        k (int): depthwise kernel size, default 3
        s (int): stride for depthwise conv, default 1
        d (int): dilation for depthwise conv, default 1
        act (bool): whether to use activation, default True
    """
    def __init__(self, c1, c2, k=3, s=1, d=1, act=True):
        super().__init__()
        # depthwise: groups=c1
        self.dw = Conv(c1, c1, k=k, s=s, g=c1, d=d, act=act)
        # pointwise: 1x1 mixes channels
        self.pw = Conv(c1, c2, k=1, s=1, g=1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class DPBottleneck(nn.Module):
    def __init__(self, c1, c2, s=1, e=0.5, n_div=4, act=True, shortcut=True):
        super().__init__()
        self.shortcut = shortcut

        # hidden channels for main branch
        c_ = max(1, int(round(c2 * e)))

        # main: DWConv -> DWConv
        # stride is only on the first DWConv for downsampling
        self.dw1 = DWConv(c1, c_, k=3, s=s, d=1, act=act)
        self.dw2 = DWConv(c_, c2, k=3, s=1, d=1, act=act)

        # side: PConv from input x (must use same stride s to match spatial size)
        self.pconv = PConv(c1, c2, n_div=n_div, k=3, s=s, g=1, act=act, fuse_k=1)

    def forward(self, x):
        y_main = self.dw2(self.dw1(x))
        if self.shortcut:
            y_side = self.pconv(x)
            return y_main + y_side
        return y_main
class DPC3k(C3):
    """
    DPC3k: C3 variant using DPBottleneck blocks (DWConv->DWConv with PConv(x) branch).

    Notes:
      - Keeps the same init signature as DSC3k for compatibility (k1/k2/d2 kept but unused).
      - DPBottleneck defaults to 3x3 DWConv blocks + PConv branch from input.
    """
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        k1=3,   # kept for compatibility, unused in DPBottleneck path
        k2=5,   # kept for compatibility, unused
        d2=1,   # kept for compatibility, unused
        pconv_div=4,  # ✅ 新增：PConv 的分流比例 n_div
        dp_e=1.0      # ✅ 新增：DPBottleneck 内部扩展比（建议 1.0 与原一致）
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        # Replace DSBottleneck -> DPBottleneck
        self.m = nn.Sequential(
            *(
                DPBottleneck(
                    c_, c_,
                    s=1,                 # C3 内部块一般不做下采样
                    e=dp_e,              # 与原 DSBottleneck e=1.0 对齐
                    n_div=pconv_div,
                    act=True,
                    shortcut=shortcut
                )
                for _ in range(n)
            )
        )
class DPC3k2(C2f):

    def __init__(
        self,
        c1,
        c2,
        n=1,
        dpc3k=False,     # ✅ rename flag (was dsc3k)
        e=0.5,
        g=1,
        shortcut=True,
        k1=3,
        k2=7,
        d2=1,
        pconv_div=4,     # ✅ PConv channel divisor for DP blocks
        dp_e=1.0         # ✅ expansion inside DPBottleneck
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        if dpc3k:
            # ✅ DSC3k -> DPC3k
            self.m = nn.ModuleList(
                DPC3k(
                    self.c, self.c,
                    n=2,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    k1=k1,        # 若你的 DPC3k 保留了这些参数签名，这里可传；否则删掉这三行
                    k2=k2,
                    d2=d2,
                    pconv_div=pconv_div,
                    dp_e=dp_e
                )
                for _ in range(n)
            )
        else:
            # ✅ DSBottleneck -> DPBottleneck
            self.m = nn.ModuleList(
                DPBottleneck(
                    self.c, self.c,
                    s=1,
                    e=dp_e,
                    n_div=pconv_div,
                    act=True,
                    shortcut=shortcut
                )
                for _ in range(n)
            )



class DynamicConv(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=None,
        g=1,
        d=1,
        K=4,
        act=True,
        bn=True,
        temperature=1.0,
    ):
        super().__init__()
        assert K >= 2, "DynamicConv K (experts) must be >= 2"
        assert c1 % g == 0 and c2 % g == 0, "c1 and c2 must be divisible by groups g"
        self.c1, self.c2 = c1, c2
        self.k, self.s, self.d, self.g = k, s, d, g
        self.p = (k // 2) if p is None else p
        self.K = K
        self.temperature = temperature

        # ---- gating: GAP -> FC -> softmax over K experts ----
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c1, K, kernel_size=1, bias=True)

        # init gating bias to make uniform softmax at start (stable from scratch)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # ---- expert weights ----
        # weight: [K, c2, c1/g, k, k]
        self.weight = nn.Parameter(torch.randn(K, c2, c1 // g, k, k) * 0.02)
        self.bias = nn.Parameter(torch.zeros(K, c2))

        # ---- BN + Act (Ultralytics-like) ----
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape

        # gating weights: [B, K]
        gate = self.gap(x)                 # [B, C, 1, 1]
        gate = self.fc(gate).view(b, self.K)
        gate = F.softmax(gate / self.temperature, dim=1)  # [B, K]

        # Build per-sample conv by grouping batch into one conv call:
        # x: [1, B*C1, H, W]
        x_ = x.reshape(1, b * self.c1, h, w)

        # mix expert weights per sample:
        # W_mix: [B, C2, C1/g, k, k]
        W = torch.einsum("bk,kocij->bocij", gate, self.weight)
        b_bias = torch.einsum("bk,kc->bc", gate, self.bias)

        # reshape for grouped conv over batch:
        # W_group: [B*C2, C1/g, k, k]
        W_group = W.reshape(b * self.c2, self.c1 // self.g, self.k, self.k)
        b_group = b_bias.reshape(b * self.c2)

        # groups = B * g
        y = F.conv2d(
            x_,
            W_group,
            b_group,
            stride=self.s,
            padding=self.p,
            dilation=self.d,
            groups=b * self.g,
        )
        # y: [1, B*C2, H', W'] -> [B, C2, H', W']
        y = y.view(b, self.c2, y.shape[-2], y.shape[-1])

        y = self.act(self.bn(y))
        return y


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


