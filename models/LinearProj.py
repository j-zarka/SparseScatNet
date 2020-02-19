import torch.nn as nn
import torch


class LinearProj(nn.Module):
    def __init__(self, standardization, proj, L_kernel_size=3):
        super(LinearProj, self).__init__()
        self.standardization = standardization
        self.proj = proj
        self.L_kernel_size = L_kernel_size

    def forward(self, x):
        output = self.standardization(x)
        if self.L_kernel_size > 1:
            output = nn.functional.pad(output, ((self.L_kernel_size-1)//2,)*4, mode='reflect')
        output = self.proj(output)

        output = torch.div(output, output.norm(p=2, dim=1, keepdim=True))
        return output