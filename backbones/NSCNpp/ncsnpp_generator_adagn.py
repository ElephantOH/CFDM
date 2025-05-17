# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

from backbones.NSCNpp import layerspp, layers, dense_layer, utils
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = False
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = 256
        self.out_channels = config.input_channels
        self.nf = nf = 64
        ch_mult = [1, 1, 2, 2, 4, 4]
        self.num_res_blocks = num_res_blocks = 2
        self.attn_resolutions = attn_resolutions = (16,)
        dropout = config.dropout
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]
        self.conditional = conditional = config.conditional  # noise-conditional
        fir = True
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        init_scale = 0.
        modules = []
        embed_dim = nf
        modules.append(nn.Linear(embed_dim, nf * 4))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(nf * 4, nf * 4))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale)
        pyramid_downsample = functools.partial(layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)
        ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act, dropout=dropout, fir=fir, fir_kernel=fir_kernel, init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4, zemb_dim=z_emb_dim)

        # Downsampling block
        channels = config.input_channels * 2
        input_pyramid_ch = channels
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                modules.append(ResnetBlock(down=True, in_ch=in_ch))
                modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                input_pyramid_ch = in_ch
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if i_level != 0:
                modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c


        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                    num_channels=in_ch, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

        mapping_layers = [PixelNorm(), dense(100, 256), self.act, ]
        for _ in range(3):
            mapping_layers.append(dense(256, 256))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)

    def forward(self, x, time_cond, z):
        # timestep/noise_level embedding; only for continuous training
        zemb = self.z_transform(z)
        modules = self.all_modules
        m_idx = 0
        # Sinusoidal positional embeddings.
        timesteps = time_cond
        temb = layers.get_timestep_embedding(timesteps, self.nf)
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        # Downsampling block
        input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
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

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
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
        if not self.not_use_tanh:
            out = torch.tanh(h)

        if self.out_channels == 1:
            return out[:, [0], ...], out[:, [1], ...]
        elif self.out_channels == 3:
            return out[:, [0, 1, 2], ...], out[:, [3, 4, 5], ...]
        else:
            assert False
