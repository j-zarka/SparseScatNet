# Modified from Kymatio scattering2d with frontend torch

import torch
import torch.nn.functional as F
import copy
import numpy as np

from kymatio.scattering2d.frontend.base_frontend import ScatteringBase2D
from kymatio.frontend.torch_frontend import ScatteringTorch


class ScatteringTorch2D_wph(ScatteringTorch, ScatteringBase2D):
    def __init__(self, J, shape, L=8, A=4, max_order=2, pre_pad=False, backend='torch'):
        ScatteringTorch.__init__(self)
        scat_args = locals()
        scat_args.pop('A')
        ScatteringBase2D.__init__(**scat_args)
        self.A = A
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)
        if A != 4:
            alphas = np.linspace(0, (A - 1) * 2 * np.pi / A, A)
            self.phases = np.exp(1j * alphas)
            self.phases = torch.FloatTensor([(np.real(phase), np.imag(phase)) for phase in self.phases])
        else:
            self.phases = None

        self.register_filters()

    def register_single_filter(self, v, n):
        current_filter = torch.from_numpy(v).unsqueeze(-1)
        self.register_buffer('tensor' + str(n), current_filter)
        return current_filter

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        # Create the filters

        n = 0

        for c, phi in self.phi.items():
            if not isinstance(c, int):
                continue

            self.register_single_filter(phi, n)
            n = n + 1

        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if not isinstance(k, int):
                    continue

                self.register_single_filter(v, n)
                n = n + 1

    def load_single_filter(self, n, buffer_dict):
        return buffer_dict['tensor' + str(n)]

    def load_filters(self):
        """ This function loads filters from the module's buffers """
        # each time scattering is run, one needs to make sure self.psi and self.phi point to
        # the correct buffers
        buffer_dict = dict(self.named_buffers())

        n = 0

        phis = copy.deepcopy(self.phi)
        for c, phi in phis.items():
            if not isinstance(c, int):
                continue

            phis[c] = self.load_single_filter(n, buffer_dict)
            n = n + 1

        psis = copy.deepcopy(self.psi)
        for j in range(len(psis)):
            for k, v in psis[j].items():
                if not isinstance(k, int):
                    continue

                psis[j][k] = self.load_single_filter(n, buffer_dict)
                n = n + 1

        return phis, psis

    def scattering(self, input):
        """Forward pass of the scattering.

            Parameters
            ----------
            input : tensor
                Tensor with k+2 dimensions :math:`(n_1, ..., n_k, M, N)` where :math:`(n_1, ...,n_k)` is
                arbitrary. Currently, k=2 is hardcoded. :math:`n_1` typically is the batch size, whereas
                :math:`n_2` is the number of input channels.

            Raises
            ------
            RuntimeError
                In the event that the input does not have at least two
                dimensions, or the tensor is not contiguous, or the tensor is
                not of the correct spatial size, padded or not.
            TypeError
                In the event that the input is not a Torch tensor.

            Returns
            -------
            S : tensor
                Scattering of the input, a tensor with k+3 dimensions :math:`(n_1, ...,n_k, D, Md, Nd)`
                where :math:`D` corresponds to a new channel dimension and :math:`(Md, Nd)` are
                downsampled sizes by a factor :math:`2^J`. Currently, k=2 is hardcoded.

        """
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')

        if (input.shape[-1] != self.N or input.shape[-2] != self.M) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.M, self.N))

        if (input.shape[-1] != self.N_padded or input.shape[-2] != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self.M_padded, self.N_padded))

        phi, psi = self.load_filters()

        batch_shape = input.shape[:-2]
        signal_shape = input.shape[-2:]

        input = input.reshape((-1,) + signal_shape)

        S = phase_scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.L, self.A, phi, psi,
                               self.max_order, self.phases)

        scattering_shape = S.shape[-3:]

        S = S.reshape(batch_shape + scattering_shape)

        return S


def phase_scattering2d(x, pad, unpad, backend, J, L, A, phi, psi, max_order, phases):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate

    order0_size = 1
    order1_size = L * J * A
    order2_size = L ** 2 * J * (J - 1) // 2
    output_size = order0_size + order1_size

    if max_order == 2:
        output_size += order2_size

    out_S_0, out_S_1, out_S_2 = [], [], []

    U_r = pad(x)

    U_0_c = fft(U_r, 'C2C')

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi[0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    S_0 = fft(U_1_c, 'C2R', inverse=True)
    S_0 = unpad(S_0)

    out_S_0.append(S_0)

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        U_1_c = cdgmm(U_0_c, psi[n1][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)

        U_1_c = fft(U_1_c, 'C2C', inverse=True)

        if A == 4:
            U_1_c_ph = U_1_c.permute(0, 3, 1, 2)
            U_1_c_ph_ = F.relu(-U_1_c_ph)
            U_1_c_ph = F.relu(U_1_c_ph)
            U_1_c_ph = torch.cat((U_1_c_ph, U_1_c_ph_), 1)
            U_1_c_ph = U_1_c_ph.reshape(-1, U_1_c_ph.shape[-2], U_1_c_ph.shape[-1])
            U_1_c_ph = add_imaginary_part(U_1_c_ph)
            U_1_c_ph = fft(U_1_c_ph, 'C2C')

            # Second low pass filter
            S_1_c = cdgmm(U_1_c_ph, phi[j1])
            S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))

            S_1_r = fft(S_1_c, 'C2R', inverse=True)
            S_1_r = unpad(S_1_r)
            S_1_r = S_1_r.reshape(-1, 4, S_1_r.shape[-2], S_1_r.shape[-1])
            for i in range(4):
                out_S_1.append(S_1_r[:, i, :, :])

        else:
            for phase in phases:
                U_1_c_ph = complex_multiplication(phase, U_1_c)
                U_1_c_ph = U_1_c_ph[..., 0]  # take the real part
                U_1_c_ph = F.relu(U_1_c_ph)
                U_1_c_ph = add_imaginary_part(U_1_c_ph)
                U_1_c_ph = fft(U_1_c_ph, 'C2C')

                # Second low pass filter
                S_1_c = cdgmm(U_1_c_ph, phi[j1])
                S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))

                S_1_r = fft(S_1_c, 'C2R', inverse=True)
                S_1_r = unpad(S_1_r)

                out_S_1.append(S_1_r)

        U_1_c = modulus(U_1_c)
        U_1_c = fft(U_1_c, 'C2C')

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            if j2 <= j1:
                continue
            U_2_c = cdgmm(U_1_c, psi[n2][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = fft(U_2_c, 'C2C', inverse=True)
            U_2_c = modulus(U_2_c)
            U_2_c = fft(U_2_c, 'C2C')

            # Third low pass filter
            S_2_c = cdgmm(U_2_c, phi[j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))

            S_2_r = fft(S_2_c, 'C2R', inverse=True)
            S_2_r = unpad(S_2_r)

            out_S_2.append(S_2_r)

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    out_S = concatenate(out_S)
    return out_S

def add_imaginary_part(x):
    output = x.new_zeros(x.shape + (2,))
    output[..., 0] = x
    return output

# Complex tensor x Nx2 tensor
def complex_multiplication(t1, t2):
    real1, imag1 = t1
    real2, imag2 = (t2[..., 0], t2[..., 1])
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


__all__ = ['ScatteringTorch2D_wph']