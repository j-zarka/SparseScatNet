import torch.nn as nn


class SparseScatNet(nn.Module):
    def __init__(self, scattering, linear_proj, istc, classifier, return_full_inf=False):
        super(SparseScatNet, self).__init__()
        self.scattering = scattering
        self.linear_proj = linear_proj
        self.istc = istc
        self.classifier = classifier
        self.return_full_inf = return_full_inf

    def forward(self, x, return_proj=False):
        output = self.scattering(x)
        output = self.linear_proj(output)
        if return_proj:
            return output
        if self.return_full_inf:
            output, lambda_0_max_batch, sparsity, support_size, support_diff, rec_loss_rel = \
                self.istc(output, self.return_full_inf)
        else:
            output = self.istc(output)

        output = self.classifier(output)

        if self.return_full_inf:
            return output, lambda_0_max_batch, sparsity, support_size, support_diff, rec_loss_rel
        else:
            return output