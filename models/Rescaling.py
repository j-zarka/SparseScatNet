import torch.nn as nn


class Rescaling(nn.Module):
    def __init__(self, bias, scaling_mat):
        super(Rescaling, self).__init__()
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.scaling_mat = nn.Parameter(scaling_mat, requires_grad=False)

    def forward(self, x):
        output = x - self.bias
        output = nn.functional.conv2d(output, self.scaling_mat)
        return output
