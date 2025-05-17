import math
import string
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from utils.op import upfirdn2d


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')


def dense(in_channels, out_channels, init_scale=1.):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################

def get_t_embed(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def get_2d_spe(coords, embedding_dim=64, max_positions=10000):
    assert coords.dim() == 2 and coords.size(1) == 2, "input_size : [batch_size, 2]"
    scaled_coords = coords * max_positions  # [batch_size, 2]
    x = scaled_coords[:, 0]  # [batch_size]
    y = scaled_coords[:, 1]
    half_dim = embedding_dim // 4
    omega = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=coords.device)
        * (-math.log(max_positions) / (half_dim - 1))
    )  # [half_dim]
    x_emb = x.unsqueeze(-1) * omega.unsqueeze(0)  # [batch_size, half_dim]
    y_emb = y.unsqueeze(-1) * omega.unsqueeze(0)
    sin_x = torch.sin(x_emb)  # [batch_size, half_dim]
    cos_x = torch.cos(x_emb)
    sin_y = torch.sin(y_emb)
    cos_y = torch.cos(y_emb)
    emb = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=1)
    if embedding_dim % 2 != 0:
        emb = F.pad(emb, (0, 1), mode='constant')  # 补零对齐
    return emb  # [batch_size, embedding_dim]


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


# %%
# Resnet
class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, in_channel, style_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense(style_dim, in_channel * 2)
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def upsample_2d(x, k=None, factor=2, gain=1):
    r"""Upsample a batch of 2D images with the given filter.

      Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
      and upsamples each image with the given filter. The filter is normalized so
      that
      if the input pixels are constant, they will be scaled by the specified
      `gain`.
      Pixels outside the image are assumed to be zero, and the filter is padded
      with
      zeros so that its shape is a multiple of the upsampling factor.
      Args:
          x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
          k:            FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            nearest-neighbor upsampling.
          factor:       Integer upsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).

      Returns:
          Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device), up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
    r"""Downsample a batch of 2D images with the given filter.

      Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
      and downsamples each image with the given filter. The filter is normalized
      so that
      if the input pixels are constant, they will be scaled by the specified
      `gain`.
      Pixels outside the image are assumed to be zero, and the filter is padded
      with
      zeros so that its shape is a multiple of the downsampling factor.
      Args:
          x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
          k:            FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            average pooling.
          factor:       Integer downsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).

      Returns:
          Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2))


def _shape(x, dim):
    return x.shape[dim]


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

       Padding is performed only once at the beginning, not between the
       operations.
       The fused op is considerably more efficient than performing the same
       calculation
       using standard TensorFlow ops. It supports gradients of arbitrary order.
       Args:
         x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
           C]`.
         w:            Weight tensor of the shape `[filterH, filterW, inChannels,
           outChannels]`. Grouped convolution can be performed by `inChannels =
           x.shape[0] // numGroups`.
         k:            FIR filter of the shape `[firH, firW]` or `[firN]`
           (separable). The default is `[1] * factor`, which corresponds to
           nearest-neighbor upsampling.
         factor:       Integer upsampling factor (default: 2).
         gain:         Scaling factor for signal magnitude (default: 1.0).

       Returns:
         Tensor of the shape `[N, C, H * factor, W * factor]` or
         `[N, H * factor, W * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    assert len(w.shape) == 4
    convH = w.shape[2]
    convW = w.shape[3]
    inC = w.shape[1]
    outC = w.shape[0]

    assert convW == convH

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor)

    # Determine data dimensions.
    stride = [1, 1, factor, factor]
    output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
    output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
                      output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = _shape(x, 1) // inC

    # Transpose weights.
    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

    x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
    ## Original TF code.
    # x = tf.nn.conv2d_transpose(
    #     x,
    #     w,
    #     output_shape=output_shape,
    #     strides=stride,
    #     padding='VALID',
    #     data_format=data_format)
    ## JAX equivalent

    return upfirdn2d(x, torch.tensor(k, device=x.device),
                     pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

      Padding is performed only once at the beginning, not between the operations.
      The fused op is considerably more efficient than performing the same
      calculation
      using standard TensorFlow ops. It supports gradients of arbitrary order.
      Args:
          x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
          w:            Weight tensor of the shape `[filterH, filterW, inChannels,
            outChannels]`. Grouped convolution can be performed by `inChannels =
            x.shape[0] // numGroups`.
          k:            FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            average pooling.
          factor:       Integer downsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).

      Returns:
          Tensor of the shape `[N, C, H // factor, W // factor]` or
          `[N, H // factor, W // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device),
                  pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


class Conv2d(nn.Module):
    """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

    def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
                 resample_kernel=(1, 3, 3, 1),
                 use_bias=True,
                 kernel_init=None):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        if self.up:
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)

        return x


class DownSample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                 fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            if with_conv:
                self.Conv2d_0 = Conv2d(in_ch, out_ch, kernel=3, down=True, resample_kernel=fir_kernel, use_bias=True,
                                       kernel_init=default_init())
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)
        return x


class ResnetBlockBigGAN(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, t_emb_dim=None, z_emb_dim=None,
                 up=False, down=False, dropout=0.1, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, z_emb_dim)
        self.up = up
        self.down = down
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if t_emb_dim is not None:
            self.Dense_0 = nn.Linear(t_emb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, z_emb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, t_emb=None, z_emb=None):
        h = self.act(self.GroupNorm_0(x, z_emb))
        if self.up:
            h = upsample_2d(h, self.fir_kernel, factor=2)
            x = upsample_2d(x, self.fir_kernel, factor=2)
        elif self.down:
            h = downsample_2d(h, self.fir_kernel, factor=2)
            x = downsample_2d(x, self.fir_kernel, factor=2)
        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if t_emb is not None:
            h += self.Dense_0(self.act(t_emb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h, z_emb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


# %%
# Transformer
def _einsum(a, b, c, x, y):
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


class AttnBlock(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


class CrossAttnBlock(nn.Module):
    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)
        self.NIN_layers = nn.ModuleList([NIN(channels // 2, channels // 2) for _ in range(3)])
        self.NIN_layers1 = nn.ModuleList([NIN(channels // 2, channels // 2) for _ in range(3)])
        self.NIN_mid_layers = nn.ModuleList([NIN(channels // 2, channels // 2, init_scale=init_scale),
                                             NIN(channels // 2, channels // 2, init_scale=init_scale)])
        # self.NIN_final_layers = NIN(channels, channels)
        self.skip_rescale = True

    def cross_attention(self, query, key, value, B, H, W):
        attention_weights = torch.einsum('bchw,bcij->bhwij', query, key) * (query.size(1) ** (-0.5))
        attention_weights = attention_weights.reshape(B, H, W, H * W)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.reshape(B, H, W, H, W)
        return torch.einsum('bhwij,bcij->bchw', attention_weights, value)

    def forward(self, x):
        B, C, H, W = x.shape
        sh = self.GroupNorm_0(x)

        half_channels = C // 2
        q = self.NIN_layers[0](sh[:, :half_channels, :, :])
        k = self.NIN_layers[1](sh[:, half_channels:, :, :])
        v = self.NIN_layers[2](sh[:, half_channels:, :, :])

        h = self.cross_attention(q, k, v, B, H, W)
        h = self.NIN_mid_layers[0](h)

        q1 = self.NIN_layers1[0](sh[:, half_channels:, :, :])
        k1 = self.NIN_layers1[1](sh[:, :half_channels, :, :])
        v1 = self.NIN_layers1[2](sh[:, :half_channels, :, :])

        h1 = self.cross_attention(q1, k1, v1, B, H, W)
        h1 = self.NIN_mid_layers[1](h1)

        h = torch.cat((h, h1), dim=1)

        # h = self.NIN_final_layers(h)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


# %%
# Util

_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)
