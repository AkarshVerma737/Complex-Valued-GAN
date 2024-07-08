import torch
import torch.nn as nn

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ComplexConv2d, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        real = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag = self.real_conv(x.imag) + self.imag_conv(x.real)
        return torch.complex(real, imag)

# Complex Transposed Convolutional Layer
class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super(ComplexConvTranspose2d, self).__init__()
        self.real_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
        self.imag_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)

    def forward(self, x):
        real = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag = self.real_conv(x.imag) + self.imag_conv(x.real)
        return torch.complex(real, imag)

# Complex Batch Normalization Layer
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm2d, self).__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.imag_bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        real = self.real_bn(x.real)
        imag = self.imag_bn(x.imag)
        return torch.complex(real, imag)