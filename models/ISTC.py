import torch.nn as nn
import torch
import numpy as np


class ISTC(nn.Module):

    def __init__(self, nb_channels_in, dictionary_size=1024, n_iterations=12, lambda_0=0.3, lambda_star=0.05,
                 lambda_star_lb=0.05, grad_lambda_star=True, epsilon_lambda_0=1., output_rec=False, use_W=False):

        super(ISTC, self).__init__()

        self.dictionary_size = dictionary_size
        self.n_iterations = n_iterations
        self.use_W = use_W

        dictionary = nn.Conv2d(dictionary_size, nb_channels_in, kernel_size=1, stride=1, padding=0, bias=False). \
            weight.data
        if use_W:
            w_matrix = nn.Conv2d(dictionary_size, nb_channels_in, kernel_size=1, stride=1, padding=0, bias=False). \
                weight.data

        dictionary.normal_()
        dictionary /= dictionary.norm(p=2, dim=0, keepdim=True)
        self.dictionary_weight = nn.Parameter(dictionary)

        if use_W:
            w_matrix.normal_()
            w_matrix /= w_matrix.norm(p=2, dim=0, keepdim=True)
            self.w_weight = nn.Parameter(w_matrix)
            with torch.no_grad():
                self.w_weight.data += self.dictionary_weight.data - \
                                             (self.dictionary_weight.data *
                                              self.w_weight.data).sum(dim=0, keepdim=True) * \
                                             self.dictionary_weight.data

        else:
            self.w_weight = self.dictionary_weight

        self.output_rec = output_rec

        self.log_lambda_0 = nn.Parameter(torch.FloatTensor(1).fill_(np.log(lambda_0)), requires_grad=False)
        self.log_lambdas = nn.Parameter(torch.FloatTensor(n_iterations - 1), requires_grad=False)
        self.log_lambda_star = nn.Parameter(torch.FloatTensor(1).fill_(np.log(lambda_star)),
                                            requires_grad=grad_lambda_star)
        self.log_gamma = nn.Parameter(torch.FloatTensor(1).fill_((1. / self.n_iterations) *
                                                                 np.log(lambda_star / lambda_0)), requires_grad=False)

        self.lambda_0 = nn.Parameter(torch.FloatTensor(1).fill_(lambda_0), requires_grad=False)
        self.lambdas = nn.Parameter(torch.FloatTensor(n_iterations - 1), requires_grad=False)
        self.lambda_star = nn.Parameter(torch.FloatTensor(1).fill_(lambda_star), requires_grad=False)
        self.gamma = nn.Parameter(torch.FloatTensor(1).fill_(np.power(lambda_star / lambda_0, 1. / self.n_iterations)),
                                  requires_grad=False)

        with torch.no_grad():
            for i in range(n_iterations - 1):
                self.log_lambdas.data[i] = np.log(lambda_0) + (i+1) * np.log(self.gamma)
                self.lambdas.data[i] = lambda_0 * (self.gamma**(i+1))

        self.grad_lambda_star = grad_lambda_star
        self.lambda_star_lb = lambda_star_lb
        self.epsilon_lambda_0 = epsilon_lambda_0

    def forward(self, x, return_full_inf=False):
        with torch.no_grad():
            if return_full_inf:
                sparsity = torch.zeros(self.n_iterations).cuda()
                support_diff = torch.zeros(self.n_iterations).cuda()

            if self.grad_lambda_star:
                if self.log_lambda_star.data[0] < np.log(self.lambda_star_lb):
                    self.log_lambda_star.data[0] = np.log(self.lambda_star_lb)
                elif self.log_lambda_star.data[0] > torch.log(self.lambda_0.data[0]):
                    self.log_lambda_star.data[0] = torch.log(self.lambda_0.data[0]) + self.n_iterations * np.log(0.99)

        if self.grad_lambda_star or self.epsilon_lambda_0 > 0:
            log_lambdas_fwd = torch.FloatTensor(self.log_lambdas.size()).cuda()
            lambdas_fwd = torch.FloatTensor(self.lambdas.size()).cuda()
            for i in range(self.n_iterations - 1):
                log_lambdas_fwd[i][...] = (1-(i + 1)/self.n_iterations)*self.log_lambda_0 \
                                          + ((i + 1)/self.n_iterations)*self.log_lambda_star
                lambdas_fwd[i][...] = torch.exp(log_lambdas_fwd[i][...])

                with torch.no_grad():
                    self.log_lambdas.data = log_lambdas_fwd.data
                    self.lambdas.data = lambdas_fwd.data
                    self.lambda_0.data = torch.exp(self.log_lambda_0.data)
                    self.lambda_star.data = torch.exp(self.log_lambda_star.data)
                    self.gamma.data = torch.pow(self.lambda_star.data/self.lambda_0.data, 1. / self.n_iterations)
                    self.log_gamma.data = torch.log(self.gamma.data)

        self.dictionary_weight.data /= self.dictionary_weight.data.norm(p=2, dim=0, keepdim=True)

        if self.use_W:
            self.w_weight.data += self.dictionary_weight.data - \
                                         (self.dictionary_weight.data * self.w_weight.data).sum(dim=0,
                                                                                                       keepdim=True) \
                                         * self.dictionary_weight.data

        WT_x = nn.functional.conv2d(x, self.w_weight.transpose(0, 1).contiguous())

        if return_full_inf:
            with torch.no_grad():
                lambda_0_max_batch = torch.max(-WT_x.min(), WT_x.max())

        # Initialize z
        z = torch.zeros(WT_x.size()).cuda()

        for i_iter in range(self.n_iterations):
            if i_iter == self.n_iterations-1:
                lambda_i = torch.exp(self.log_lambda_star)

            else:
                if self.grad_lambda_star or self.epsilon_lambda_0 > 0:
                    lambda_i = lambdas_fwd[i_iter]
                else:
                    lambda_i = torch.exp(self.log_lambdas[i_iter])

            D_z = nn.functional.conv2d(z, self.dictionary_weight)
            WT_D_z = nn.functional.conv2d(D_z, self.w_weight.transpose(0, 1).contiguous())
            z_prev = z
            z = relu(z - WT_D_z + WT_x, lambda_i)

            if return_full_inf:
                with torch.no_grad():
                    sparsity[i_iter] = 100 * (z != 0).float().mean()
                    support_diff[i_iter] = ((z != 0) ^ (z_prev != 0)).float().mean()

        if self.output_rec:
            D_z = nn.functional.conv2d(z, self.dictionary_weight)
            output = D_z

        else:
            output = z

        if return_full_inf:
            with torch.no_grad():
                support_size = sparsity * self.dictionary_size / 100
                support_diff = support_diff * self.dictionary_size

                reconstructed_proj = nn.functional.conv2d(z, self.dictionary_weight)
                rec_loss_rel = ((reconstructed_proj - x).norm(p=2, dim=1) / x.norm(p=2, dim=1)).mean()

            return output, lambda_0_max_batch, sparsity, support_size, support_diff, rec_loss_rel

        else:
            return output


def relu(x, lambd):
    return nn.functional.relu(x - lambd)
