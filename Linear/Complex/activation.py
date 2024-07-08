import torch 
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,z):
        a=z.real
        b=z.imag
        x = 1+torch.exp(-a)*torch.cos(b)
        y = torch.exp(-a)*torch.sin(b)
        denominator = x**2 + y**2
        c = (a*x-b*y)/denominator  #a*(1 + e^-a * cosb) - b*(e^-a * sinb) / denominator
        d=(b*x+a*y)/denominator    #b*(1 + e^-a * cosb) + a*(e^-a * sinb) / denominator
        return torch.complex(c,d)

class CTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,z):
        cs = torch.cos(2*z.imag)
        sn = torch.sin(2*z.imag)
        ex = torch.exp(2*z.real)
        mag = (1+ex*cs)**2 + (ex*sn)**2
        real = torch.clamp((ex**2 -1)/mag,min = -1,max = 1)
        imag = torch.clamp((2*ex*sn)/mag, min = -1, max =1)
        return torch.complex(real,imag)

class CTanhR(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, z):
        real = self.tanh(z.real)
        imag = self.tanh(z.imag)
        return real*imag

class Sigmoidmag(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,z):
        mag = torch.abs(z)
        phase = torch.atan(z.imag/z.real)
        sigmag = self.sigmoid(mag)
        sigphase = self.sigmoid(phase)
        return (sigmag+sigphase)/2
    
    
class CReLUR(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, z):
        
        real = self.relu(z.real)
        imag = self.relu(z.imag)
        zz = torch.complex(real,imag)
        mag = torch.abs(zz)
        return mag