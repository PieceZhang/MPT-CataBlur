import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
from ptflops import get_model_complexity_info



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, downpool):
        super().__init__()
        self.downpool = downpool
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in1 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_in2 = nn.Linear(dim, hidden_features, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Linear(hidden_features, dim, bias=bias)

    def forward(self, x):
        x1 = self.project_in1(x[0])
        x1 = self.dwconv1(x1)
        x1 = x1.flatten(-2).transpose(-2, -1)

        x2 = x[-1]
        b, c, h, w = x2.shape
        x2 = x2.flatten(-2).transpose(-2, -1)
        x2 = self.project_in2(x2)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = x.transpose(-2, -1).view(b, c, h, w)
        return x


class FPDownSamp(nn.Module):
    def __init__(self, dim, downpool):
        super().__init__()
        self.downsamp = nn.AvgPool2d(kernel_size=int(1 / downpool), stride=int(1 / downpool))  # ablation: AVP or MP or stride conv or interpolation
        self.downproj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.downsamp(x)
        return x + self.downproj(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, downpool, input_res, shift, config=None, window_size=8):
        super(Attention, self).__init__()
        self.config = config
        self.num_heads = num_heads
        self.temperature = nn.ParameterList(nn.Parameter(torch.ones(num_heads, 1, 1)) for _ in range(len(downpool)))

        self.q = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=1, bias=bias), nn.Linear(dim, dim, bias=bias)])
        self.q_dwconv = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
                                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)])
        self.kv = nn.ModuleList([nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias), nn.Linear(dim, dim*2, bias=bias)])
        self.kv_dwconv = nn.ModuleList([nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias),
                                        nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)])
        self.project_out = nn.ModuleList(nn.Linear(dim, dim, bias=bias) for _ in range(len(downpool)))
        # self.proj = nn.Linear(dim, dim, bias=bias)
        self.window_size = window_size
        if shift:
            self.shift_size = window_size // 2
        else:
            self.shift_size = 0
        self.input_resolution = (input_res, input_res)
        self.downpool = downpool
        if downpool[-1] != 1:
            self.downsamp = FPDownSamp(dim, downpool[-1])
        else:
            self.downsamp = nn.Identity()

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if self.shift_size > 0:
            # attn_mask = self.calculate_mask(self.input_resolution)
            attn_mask_r = self.calculate_mask((int(self.input_resolution[0]*downpool[-1]), int(self.input_resolution[1]*downpool[-1])))
            attn_mask_r = attn_mask_r.repeat_interleave(int(1/downpool[-1] ** 2), dim=0)
            if config is not None:
                # attn_mask_val = self.calculate_mask((config.val_height, config.val_width))
                attn_mask_val_r = self.calculate_mask((int(config.val_height*downpool[-1]), int(config.val_width*downpool[-1])))
                attn_mask_val_r = attn_mask_val_r.repeat_interleave(int(1/downpool[-1] ** 2), dim=0)
            else:
                # attn_mask_val = None
                attn_mask_val_r = None
        else:
            # attn_mask = None
            attn_mask_r = None
            # attn_mask_val = None
            attn_mask_val_r = None

        self.attn_mask_r = attn_mask_r
        self.attn_mask_val_r = attn_mask_val_r
        # self.register_buffer("attn_mask", attn_mask)  # this will lead to a very large weight file
        # self.register_buffer("attn_mask_r", attn_mask_r)
        # self.register_buffer("attn_mask_val", attn_mask_val)
        # self.register_buffer("attn_mask_val_r", attn_mask_val_r)

    @torch.no_grad()
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        b,c,h,w = x.shape
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if self.input_resolution == (h, w):
            if self.attn_mask_r is not None and self.attn_mask_r.device != x.device:
                self.attn_mask_r = self.attn_mask_r.to(x.device)
            # attn_mask = self.attn_mask
            attn_mask_r = self.attn_mask_r
        elif self.attn_mask_val_r is not None and (self.config.val_height, self.config.val_width) == (h, w):
            if self.attn_mask_val_r is not None and self.attn_mask_val_r.device != x.device:
                self.attn_mask_val_r = self.attn_mask_val_r.to(x.device)
            # attn_mask = self.attn_mask_val
            attn_mask_r = self.attn_mask_val_r
        else:
            # attn_mask = self.calculate_mask((h, w)).to(x.device)
            attn_mask_r = self.calculate_mask((int(h*self.downpool[-1]), int(w*self.downpool[-1]))).to(x.device)
            attn_mask_r = attn_mask_r.repeat_interleave(int(1 / self.downpool[-1] ** 2), dim=0)

        outlist = []
        pool = [x, self.downsamp(x)]
        for i, xresize in enumerate(pool):
            if i == 0:
                q = self.q[i](xresize)  # not recursive
                q = self.q_dwconv[i](q)

                kv = self.kv[i](xresize)
                kv = self.kv_dwconv[i](kv)

                q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # nWin, nHead, C/nHead, hWin*wWin
                q = torch.nn.functional.normalize(q, dim=-1)  # normalize winsize dim

                k, v = kv.chunk(2, dim=1)  # nWin, C, hWin, wWin

                k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                k = torch.nn.functional.normalize(k, dim=-1)  # normalize winsize dim

                attn = (q @ k.transpose(-2, -1)) * self.temperature[i]  # window channel attn
                attn = attn.softmax(dim=-1)

                out = (attn @ v)
                out = rearrange(out, 'b head c hw -> b hw (head c)', head=self.num_heads)
                out = self.project_out[i](out)
                out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
            else:
                q = window_partition(x.permute(0, 2, 3, 1), self.window_size)  # nWin, hWin, wWin, C
                q = q.view(q.shape[0], -1, q.shape[-1])  # nWin, hWin*wWin, C
                q = self.q[i](q)  # not recursive
                q = window_reverse(q, self.window_size, int(h * self.downpool[i]), int(w * self.downpool[i])).permute(0, 3, 1, 2)  # N, C, H, W
                q = self.q_dwconv[i](q)

                xresize = window_partition(xresize.permute(0, 2, 3, 1), self.window_size)  # nWin, hWin, wWin, C
                xresize = xresize.view(xresize.shape[0], -1, xresize.shape[-1])  # nWin, hWin*wWin, C
                kv = self.kv[i](xresize)
                kv = window_reverse(kv, self.window_size, int(h * self.downpool[i]), int(w * self.downpool[i])).permute(0, 3, 1, 2)  # N, C*2, H, W
                kv = self.kv_dwconv[i](kv)

                if self.shift_size > 0:
                    q = torch.roll(q, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
                    kv = torch.roll(kv, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
                kv = window_partition(kv.permute(0, 2, 3, 1), self.window_size).permute(0, 3, 1, 2)
                k, v = kv.chunk(2, dim=1)  # nWin, C, hWin, wWin

                q = window_partition(q.permute(0, 2, 3, 1), self.window_size).permute(0, 3, 1, 2)  # nWin, C, hWin, wWin
                B_, N, C = (q.shape[0], int(q.shape[2] * q.shape[3]), q.shape[1])  # (B*num_windows_H*num_windows_W, (winsize*winsize), C)
                q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # nWin, nHead, C/nHead, hWin*wWin
                q = torch.nn.functional.normalize(q, dim=-2)  # normalize C dim

                arearate = int(1/self.downpool[i] ** 2)
                k = k.repeat_interleave(arearate, dim=0)
                v = v.repeat_interleave(arearate, dim=0)

                k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                k = torch.nn.functional.normalize(k, dim=-2)  # normalize C dim

                attn = (q.transpose(-2, -1) @ k) * self.temperature[i]  # window spatial attn
                attn = attn + relative_position_bias.unsqueeze(0)

                if attn_mask_r is not None:
                    nW = attn_mask_r.shape[0]
                    attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + attn_mask_r.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, self.num_heads, N, N)
                    attn = attn.softmax(dim=-1)
                else:
                    attn = attn.softmax(dim=-1)

                out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
                out = rearrange(out, 'b head c hw -> b hw (head c)', head=self.num_heads)
                out = self.project_out[i](out)
                out = window_reverse(out, self.window_size, int(h), int(w)).permute(0, 3, 1, 2)
                if self.shift_size > 0:
                    out = torch.roll(out.permute(0, 2, 3, 1), shifts=(self.shift_size, self.shift_size), dims=(1, 2)).permute(0, 3, 1, 2)
            outlist.append(out)  # B C H' W' append first map
        return outlist



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, downpool, input_res, config=None, shift=False):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, downpool, input_res, shift,  config=config)
        self.norm2 = nn.ModuleList([LayerNorm(dim, LayerNorm_type), LayerNorm(dim, LayerNorm_type)])
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, downpool)

    def forward(self, x):
        x_shortcut = x
        x = list(map(lambda _: _ + x_shortcut, self.attn(self.norm1(x))))  # x: list[downpool]
        x_shortcut = x[0]
        x = self.ffn([self.norm2[0](x[0]), self.norm2[1](x[1])]) + x_shortcut

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class MPT(nn.Module):
    """ Multi-pyramid Transformer """
    def __init__(self,
        config=None,
        inp_channels=3,
        out_channels=3,
        dim = 40,
        num_blocks = [6,6,6,6],
        downpool=[[[1, 0.125], [1, 0.125], [1, 0.25], [1, 0.25], [1, 1], [1, 1]],
                  [[1, 0.25], [1, 0.25], [1, 0.5], [1, 0.5], [1, 1], [1, 1]],
                  [[1, 0.5], [1, 0.5], [1, 0.5], [1, 0.5], [1, 1], [1, 1]],
                  [[1, 0.5], [1, 0.5], [1, 0.5], [1, 0.5], [1, 1], [1, 1]]],  # 76.206768128 B 19.799356 M ws=8
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.6,
        bias = False,
        LayerNorm_type = 'WithBias',   ## 'BiasFree'
        input_res=256
    ):

        super().__init__()
        if config is not None:
            inp_channels = config.inch
            out_channels = config.inch

        self.inproj = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, downpool=downpool[0][i],
                                                               LayerNorm_type=LayerNorm_type, input_res=input_res, config=config,
                                                               shift=False if i % 2 == 0 else True) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, downpool=downpool[1][i],
                                                               LayerNorm_type=LayerNorm_type, input_res=input_res//2, config=config,
                                                               shift=False if i % 2 == 0 else True) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, downpool=downpool[2][i],
                                                               LayerNorm_type=LayerNorm_type, input_res=input_res//4, config=config,
                                                               shift=False if i % 2 == 0 else True) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, downpool=downpool[3][i],
                                                       LayerNorm_type=LayerNorm_type, input_res=input_res//8, config=config,
                                                       shift=False if i % 2 == 0 else True) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, downpool=downpool[2][i],
                                                               LayerNorm_type=LayerNorm_type, input_res=input_res//4, config=config,
                                                               shift=False if i % 2 == 0 else True) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, downpool=downpool[1][i],
                                                               LayerNorm_type=LayerNorm_type, input_res=input_res//2, config=config,
                                                               shift=False if i % 2 == 0 else True) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, downpool=downpool[0][i],
                                                               LayerNorm_type=LayerNorm_type, input_res=input_res, config=config,
                                                               shift=False if i % 2 == 0 else True) for i in range(num_blocks[0])])

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.output = nn.Sequential(nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=1, bias=bias),
        #                             nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias))

    def forward(self, x):

        inp_enc_level1 = self.inproj(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = inp_dec_level3 + out_enc_level3
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = inp_dec_level2 + out_enc_level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + x

        outs = collections.OrderedDict()
        outs['result'] = out_dec_level1
        return outs


if __name__ == '__main__':
    height = 256
    width = 256
    model = MPT().cuda()

    x = torch.randn((2, 3, height, width))
    Macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=False, print_per_layer_stat=True)
    print('\t{:<30}  {:<8} B'.format('Computational complexity (Macs): ', Macs / 1000 ** 3))
    print('\t{:<30}  {:<8} M'.format('Number of parameters: ', params / 1000 ** 2, '\n'))
