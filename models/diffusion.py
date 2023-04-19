import logging
import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    # emb = emb.to(device='cpu')
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Model(nn.Module):
    def __init__(self, config, in_channels=0, out_channels=0, resolution=0, ts_type='discrete'):
        super().__init__()
        self.config = config
        ch, ch_mult = config.model.ch, tuple(config.model.ch_mult)
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = in_channels or config.model.in_channels
        out_ch = out_channels or config.model.out_ch
        resolution = resolution or config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.ch_mult_len = len(ch_mult)
        self.num_res_blocks = config.model.num_res_blocks
        self.resolution = resolution
        if ts_type in ['discrete', 'continuous']:
            self.ts_type = ts_type
        else:
            raise ValueError(f"Unknown ts_type: {ts_type}")
        self.in_channels = in_channels
        logging.info(f"models.Diffusion() ======================")
        logging.info(f"  ch             : {self.ch}")
        logging.info(f"  temb_ch        : {self.temb_ch}")
        logging.info(f"  ch_mult_len    : {self.ch_mult_len}")
        logging.info(f"  num_res_blocks : {self.num_res_blocks}")
        logging.info(f"  resolution     : {self.resolution}")
        logging.info(f"  ts_type        : {self.ts_type}")

        # timestep embedding
        self.temb = nn.Module()
        # The "dense" below is not a property; it's just an arbitrary name.
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution       # 32
        in_ch_mult = (1,)+ch_mult   # [1, 1, 2, 2, 2]
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.ch_mult_len):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:  # attn_resolutions: [16, ]
                    attn.append(AttnBlock(block_in))
            # for i_block
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.ch_mult_len-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        # for i_level

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.ch_mult_len)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution  # config.data.image_size

        if self.ts_type == 'continuous':
            t = t.clone() * 1000.  # if t *= 1000., it will affect t itself.
        # if

        # timestep embedding
        # self.ch: config.model.ch. Usually 128
        temb = get_timestep_embedding(t, self.ch)   # shape [256, 128]
        temb = self.temb.dense[0](temb)             # shape [256, 512]
        temb = nonlinearity(temb)                   # shape [256, 512]
        temb = self.temb.dense[1](temb)             # shape [256, 512]

        # down-sampling
        hs = [self.conv_in(x)]
        for i_level in range(self.ch_mult_len):     # ch_mult = [1, 2, 2, 2]
            for i_block in range(self.num_res_blocks):  # num_res_blocks = 2
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.ch_mult_len-1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # after each level, the len(hs):
        #   i_level: len(hs)
        #        0 : 4
        #        1 : 7
        #        2 : 10
        #        3 : 12
        # Shape of each element in hs, from hs[0] to hs[11]
        #   [250, 128, 32, 32]
        #   [250, 128, 32, 32]
        #   [250, 128, 32, 32]
        #   [250, 128, 16, 16]
        #   [250, 256, 16, 16]
        #   [250, 256, 16, 16]
        #   [250, 256, 8,  8 ]
        #   [250, 256, 8,  8 ]
        #   [250, 256, 8,  8 ]
        #   [250, 256, 4,  4 ]
        #   [250, 256, 4,  4 ]
        #   [250, 256, 4,  4 ]

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # up-sampling
        for i_level in reversed(range(self.ch_mult_len)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # shape of h during up-sampling:
        #   [250, 256, 4, 4]
        #   [250, 256, 4, 4]
        #   [250, 256, 4, 4]
        #   [250, 256, 8, 8] ------ after self.up[i_level].upsample(h)
        #   [250, 256, 8, 8]
        #   [250, 256, 8, 8]
        #   [250, 256, 8, 8]
        #   [250, 256, 16, 16] ------ after self.up[i_level].upsample(h)
        #   [250, 256, 16, 16]
        #   [250, 256, 16, 16]
        #   [250, 256, 16, 16]
        #   [250, 256, 32, 32] ------ after self.up[i_level].upsample(h)
        #   [250, 128, 32, 32]
        #   [250, 128, 32, 32]
        #   [250, 128, 32, 32]

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ModelStack(nn.Module):
    """
    Model stack. we put some model together to make a stack. And call each model 'brick'.
    """
    def __init__(self, config, stack_size=5):
        super().__init__()
        self.config = config
        self.stack_sz = stack_size
        self.ts_cnt = config.diffusion.num_diffusion_timesteps
        self.model_stack = nn.ModuleList()
        self.tsr_stack = []          # timestamp range stack
        self.brick_hit_counter = []  # track hit count of each brick
        self.brick_cvg = self.ts_cnt // self.stack_sz  # brick coverage: one brick cover how many timesteps
        for i in range(self.stack_sz):
            model = Model(config)
            self.model_stack.append(model)
            self.tsr_stack.append([i*self.brick_cvg, (i+1)*self.brick_cvg])
            self.brick_hit_counter.append(0)
        logging.info(f"ModelStack()...")
        logging.info(f"  stack_sz : {self.stack_sz}")
        logging.info(f"  ts_cnt   : {self.ts_cnt}")
        logging.info(f"  brick_cvg: {self.brick_cvg}")
        logging.info(f"  ts 0     : -> idx {self.get_brick_idx_by_ts(0)}")
        logging.info(f"  ts 500   : -> idx {self.get_brick_idx_by_ts(500)}")
        logging.info(f"  ts 999   : -> idx {self.get_brick_idx_by_ts(999)}")

    def get_brick_idx_by_ts(self, ts):
        """
        Get brick index by timestep number. the index should be within self.stack_sz
        :param ts: timestep
        :return:  brick index
        """
        for i, (stt, end) in enumerate(self.tsr_stack):
            if stt <= ts < end:
                return i
        raise Exception(f"Not found brick index by ts: {ts}")

    def forward(self, x, t):
        t0 = t
        # usually, t is an array of random int. To calculate brick index, we only take the first element.
        if hasattr(t0, '__len__') and len(t0) > 1:
            t0 = t0[0]
        if isinstance(t0, torch.Tensor):
            t0 = int(t0)
        if t0 < 0 or t0 >= self.ts_cnt:
            raise ValueError(f"illegal ts {t0}. Should be in range [0, {self.ts_cnt})")
        b_idx = self.get_brick_idx_by_ts(t0)  # brick index
        # If stack_sz is 5, and ts_cnt is 1000, then brick_cvg will be 200.
        # Given the above condition, here is how we do:
        #   brick 0 handles ts [1, 199);
        #   brick 1 handles ts [200, 399);
        #   brick 2 handles ts [400, 599);
        #   ...
        self.brick_hit_counter[b_idx] += 1
        return self.model_stack[b_idx](x, t)
# class ModelStack
