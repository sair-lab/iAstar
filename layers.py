import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

def scaled_dot_product(q, k, v):
    '''
    Input:
        (B, num_head, len_seq, head_dim)
    Output:
        attention: (B, num_head, len_seq, len_seq)
        values: (B, num_head, len_seq, head_dim)
    '''
    dim = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(dim)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values

class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        # x: (B, C, H, W)
        y = torch.einsum('bchw,co->bohw', x, self.W) + self.b[None, :, None, None]
        if y.stride()[1] == 1:
            y = y.contiguous()
        return y

class MultiheadAttnBlock(nn.Module):
    def __init__(self, channels, num_heads, head_dim=None):
        """
            Multihead attention block.
        Args:
            channels (int): number of channels
            num_heads (int): number of attn heads
            head_dim (int, optional): head dimension, if specified, it will overwrite the number of heads. 
        """
        super(MultiheadAttnBlock, self).__init__()
        if head_dim is None:
            assert channels % num_heads == 0
            self.head_dim = channels // num_heads
            self.num_heads = num_heads
        else:
            assert channels % head_dim == 0
            self.head_dim = head_dim
            self.num_heads = channels // head_dim
        
        self.ch = channels
        self.NINs = NIN(channels, 3 * channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.0)
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.GroupNorm_0(x)

        h = self.NINs(h)    # N, 3C, H, W
        h = h.reshape(B, self.num_heads, 3 * self.head_dim, H * W)
        h = h.permute(0, 1, 3, 2)       # (N, num_heads, H * W, 3 * head_dim)
        q, k, v = h.chunk(3, dim=-1)     # each chunk has shape (B, num_heads, H * W, head_dim)
        h = scaled_dot_product(q, k, v) # (N, num_heads, H * W, head_dim)
        h = h.permute(0, 1, 3, 2)       # (N, num_heads, head_dim, H * W)
        h = h.reshape(B, self.num_heads * self.head_dim, H, W)
        h = self.NIN_3(h)
        return x + h.reshape(B, C, H, W)
    
@torch.jit.script
def compl_mul1d(a, b):
    # (B, M, in_ch, H, W), (in_ch, out_ch, M) -> (B, M, out_channel, H, W)
    return torch.einsum("bmihw,iom->bmohw", a, b)

class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = (1 / (in_ch*out_ch))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float))

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        with torch.autocast(device_type='cuda', enabled=False):
            x_ft = torch.fft.rfftn(x.float(), dim=[1])
            # Multiply relevant Fourier modes
            out_ft = compl_mul1d(x_ft[:, :self.modes1], torch.view_as_complex(self.weights1))
            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=[T], dim=[1])
        return x

class TimeConv(nn.Module):
    def __init__(self, in_ch, out_ch, modes, act, with_nin=False):
        super(TimeConv, self).__init__()
        self.with_nin = with_nin
        self.t_conv = SpectralConv1d(in_ch, out_ch, modes)
        if with_nin:
            self.nin = NIN(in_ch, out_ch)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        h = self.t_conv(x)
        if self.with_nin:
            x = self.nin(x)
        out = self.act(h)
        return x + out
    
class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""

    def __init__(self, channels):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.)

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum('bcthw,bctij->bthwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, T, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, T, H, W, H, W))
        h = torch.einsum('bthwij,bctij->bcthw', w, v)
        h = self.NIN_3(h)
        return x + h