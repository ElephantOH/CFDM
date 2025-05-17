
from . import dfm_layers
import torch.nn as nn
import functools
import torch
import numpy as np


conv3x3 = dfm_layers.conv3x3
conv1x1 = dfm_layers.conv1x1
dense = dfm_layers.dense
default_init = dfm_layers.default_init
get_t_embed = dfm_layers.get_t_embed
get_coord_embed = dfm_layers.get_2d_spe
ResnetBlockBigGAN = dfm_layers.ResnetBlockBigGAN
AttnBlock = dfm_layers.AttnBlock
CrossAttnBlock = dfm_layers.CrossAttnBlock
PixelNorm = dfm_layers.PixelNorm
DownSample = dfm_layers.DownSample

class Config:
    def __init__(self):
        self.image_size = 256
        self.input_channels = 3
        self.num_channels = 64
        self.conditional = True
        self.level_channels = [1, 1, 2, 2, 4, 4]
        self.z_emb_dim = 100
        self.z_emb_channels = [256, 256, 256, 256]
        self.t_emb_dim = 64
        self.t_emb_channels = [256, 256, 256]
        self.num_resblocks = 2
        self.attn_levels = [16, ]
        self.use_cross_attn = True
        self.dropout = 0.1
        self.use_tanh_final = True
        self.fir_kernel = [1, 3, 3, 1]
        self.skip_rescale = True
        self.resblock_type = 'biggan'
        self.output_complete = True
        self.phase = 'train'

@dfm_layers.register_model(name='fdm')
class FractalDiffusionModel(nn.Module):

    def __init__(self, config, image_size=256, num_channels=64, level_channels=[1,1,2,2,4,4], attn_levels=(16)):
        super().__init__()
        if config is None:
            config = self.get_default_config()

        self.image_size = image_size
        self.input_channels = config.input_channels
        self.output_channels = config.input_channels
        self.num_channels = num_channels
        self.conditional = config.conditional
        self.conditional_type = config.conditional_type
        self.level_channels = level_channels

        self.num_levels = len(self.level_channels)
        self.all_levels = [self.image_size // (2 ** i) for i in range(self.num_levels)]

        self.z_emb_dim = config.z_emb_dim
        self.z_emb_channels = config.z_emb_channels
        self.t_emb_dim = config.t_emb_dim
        self.t_emb_channels = config.t_emb_channels

        self.num_resblocks = config.num_resblocks
        self.attn_levels = attn_levels
        self.dropout = config.dropout
        self.use_tanh_final = config.use_tanh_final
        self.act = nn.SiLU()

        self.fir_kernel = config.fir_kernel
        self.skip_rescale = config.skip_rescale

        self.output_complete = config.output_complete
        self.init_scale = 0.
        self.phase = config.phase

        self.ResnetBlock = functools.partial(ResnetBlockBigGAN, act=self.act, dropout=self.dropout,
                                             fir_kernel=self.fir_kernel,
                                             init_scale=self.init_scale, skip_rescale=self.skip_rescale,
                                             t_emb_dim=self.t_emb_channels[-1], z_emb_dim=self.z_emb_channels[-1])
        self.AttnBlock = functools.partial(AttnBlock, init_scale=self.init_scale, skip_rescale=self.skip_rescale)
        self.CrossAttnBlock = functools.partial(CrossAttnBlock, init_scale=self.init_scale, skip_rescale=self.skip_rescale)
        self.DownSample = functools.partial(DownSample, fir=True, fir_kernel=self.fir_kernel, with_conv=True)

        modules = []
        modules.append(nn.Linear(self.t_emb_dim, self.t_emb_channels[-1]))
        modules[-1].weight.data = default_init()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(self.t_emb_channels[-1], self.t_emb_channels[-1]))
        modules[-1].weight.data = default_init()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        # Downsampling block

        channels = config.input_channels * 2
        input_pyramid_ch = channels
        modules.append(conv3x3(channels, self.num_channels))
        hs_c = [self.num_channels]

        in_ch = self.num_channels
        for i_level in range(self.num_levels):
            # Residual blocks for this resolution
            for i_block in range(self.num_resblocks):
                out_ch = self.num_channels * self.level_channels[i_level]
                modules.append(self.ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if self.all_levels[i_level] in  self.attn_levels:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != self.num_levels - 1:
                modules.append(self.ResnetBlock(down=True, in_ch=in_ch))
                modules.append(self.DownSample(in_ch=input_pyramid_ch, out_ch=in_ch))
                input_pyramid_ch = in_ch
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(self.ResnetBlock(in_ch=in_ch))
        modules.append(self.AttnBlock(channels=in_ch))
        modules.append(self.ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(self.num_levels)):
            for i_block in range(self.num_resblocks + 1):
                out_ch = self.num_channels * self.level_channels[i_level]
                modules.append(self.ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if self.all_levels[i_level] in self.attn_levels:
                modules.append(AttnBlock(channels=in_ch))

            if i_level != 0:
                modules.append(self.ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c


        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=self.init_scale))

        self.all_modules = nn.ModuleList(modules)

        mapping_layers = [PixelNorm(), dense(100, self.z_emb_channels[-1]), self.act, ]
        for _ in range(3):
            mapping_layers.append(dense(self.z_emb_channels[-1], self.z_emb_channels[-1]))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)

    def forward(self, x, time_cond, z):
        zemb = self.z_transform(z)
        modules = self.all_modules
        m_idx = 0
        timesteps = time_cond
        if self.conditional_type == "coord":
            temb = get_coord_embed(timesteps, self.t_emb_dim)
        else:
            temb = get_t_embed(timesteps, self.t_emb_dim)
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        # Downsampling block
        input_pyramid = x
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_levels):
            for i_block in range(self.num_resblocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                if h.shape[-1] in self.attn_levels:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_levels - 1:
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                input_pyramid = modules[m_idx](input_pyramid)
                m_idx += 1
                input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                h = input_pyramid
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_levels)):
            for i_block in range(self.num_resblocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1

            if h.shape[-1] in self.attn_levels:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                h = modules[m_idx](h, temb, zemb)
                m_idx += 1

        assert not hs

        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)

        out = h
        if self.use_tanh_final:
            out = torch.tanh(out)

        if self.output_complete and self.phase == 'train':
            if self.output_channels == 3:
                return out[:, [0, 1, 2], ...], out[:, [3, 4, 5], ...]
            elif self.output_channels == 1:
                return out[:, [0], ...], out[:, [1], ...]
        else:
            if self.output_channels == 3:
                return out[:, [0, 1, 2], ...], None
            elif self.output_channels == 1:
                return out[:, [0], ...], None

    def get_default_config(self):
        return Config()