import argparse
import os

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from kymatio import Scattering2D
from phase_scattering2d_torch import ScatteringTorch2D_wph
from models.ISTC import ISTC, relu
from models.Rescaling import Rescaling
from models.LinearProj import LinearProj
from models.SparseScatNet import SparseScatNet

model_names = ['sparsescatnet', 'sparsescatnetw']

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='sparsescatnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: sparsescatnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Model path
parser.add_argument('--model-checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint to load')

# Subsample of classes the model has been trained on
parser.add_argument('--nb-classes', default=1000, type=int, help='number of classes randomly chosen '
                    'used for training and validation (default: 1000 = whole train/val dataset)')
parser.add_argument('--class-indices', default=None, help='numpy array of indices used in case nb-classes < 1000')

# Nb of images used for convergence analysis
parser.add_argument('--nb-images', default=256, type=int, help='number of images for convergence analysis')

# Scattering parameters
parser.add_argument('--scattering-order2', help='Compute order 2 scattering coefficients',
            action='store_true')
parser.add_argument('--scattering-wph', help='Use phase scattering',
            action='store_true')
parser.add_argument('--scat-angles', default=8, type=int, help='number of orientations for scattering')
parser.add_argument('--backend', default='torch_skcuda', type=str, help='scattering backend')
parser.add_argument('--scattering-nphases', default=4, type=int,
        help='number of phases in the first order of the phase harmonic scattering transform')
parser.add_argument('--scattering-J', default=4, type=int,
        help='j value (= maximum scale) for the scattering transform')

# ISTC(W)
parser.add_argument('--n-iterations', default=8, type=int, help='number of iterations for ISTC')
parser.add_argument('--dictionary-size', default=1024, type=int,
        help='size of the sparse coding dictionary')
parser.add_argument('--lambda-0', default=0.3, type=float, help='lambda_0')
parser.add_argument('--lambda-star', default=0.05, type=float, help='lambda_star')
parser.add_argument('--lambda-star-lb', default=0.05, type=float, help='lambda_star lower bound')
parser.add_argument('--epsilon-lambda-0', default=1., type=float, help='epsilon for lambda_0 adjustment')
parser.add_argument('--grad-lambda-star', help='gradient on lambda_star', action='store_true')

# Linear projection parameters
parser.add_argument('--L-proj-size', default=512, type=int,
                    help='dimension of the linear projection')
parser.add_argument('--L-kernel-size', default=1, type=int, help='kernel size of L')


def main():
    args = parser.parse_args()

    model, val_loader, dictionary, w_matrix = load_model(args)
    return model, val_loader, dictionary, w_matrix


def load_model(args):

    # Data loading code
    ###########################################################################################
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    # can use a subset of all classes (specified in a file or randomly chosen)
    if args.nb_classes < 1000:
        val_indices = list(np.load('utils_sampling/imagenet_val_class_indices.npy'))
        if args.class_indices is not None:
            class_indices = torch.load(args.class_indices)
        else:
            perm = torch.randperm(1000)
            class_indices = perm[:args.nb_classes].tolist()
        val_indices_full = [x for i in range(len(class_indices)) for x in range(val_indices[class_indices[i]],
                                                                                val_indices[class_indices[i] + 1])]

        val_dataset = torch.utils.data.Subset(val_dataset, val_indices_full)

    # Use a subset of images
    perm = torch.randperm(len(val_dataset))[:args.nb_images]

    val_dataset = torch.utils.data.Subset(val_dataset, perm)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    ###########################################################################################

    # Model creation
    ###########################################################################################
    if args.arch in model_names:

        # create scattering
        J = args.scattering_J
        L_ang = args.scat_angles
        A = args.scattering_nphases

        max_order = 2 if args.scattering_order2 else 1

        if args.scattering_wph:
            scattering = ScatteringTorch2D_wph(J=J, shape=(224, 224), L=L_ang, A=A, max_order=max_order,
                                               backend=args.backend)
        else:
            scattering = Scattering2D(J=args.scattering_J, shape=(224, 224), L=L_ang, max_order=max_order,
                                      backend=args.backend)

        # Flatten scattering
        scattering = nn.Sequential(scattering, nn.Flatten(1, 2))

        nb_channels_in = 3
        if args.scattering_wph:
            nb_channels_in += 3 * A * L_ang * J
        else:
            nb_channels_in += 3 * L_ang * J

        if max_order == 2:
            nb_channels_in += 3 * (L_ang ** 2) * J * (J - 1) // 2
    ###########################################################################################

        # create linear proj
        # Standardization (can also be performed with BatchNorm2d(affine=False))
        std_file = 'standardization/ImageNet2012_scattering_J{}_order{}_wph_{}_nphases_{}_nb_classes_{}.pth.tar'.format(
            args.scattering_J, 2 if args.scattering_order2 else 1, args.scattering_wph,
            args.scattering_nphases if args.scattering_wph else 0, args.nb_classes)

        if os.path.isfile(std_file):
            print("=> loading scattering mean and std '{}'".format(std_file))
            std_dict = torch.load(std_file)
            mean_std = std_dict['mean']
            stding_mat = std_dict['matrix']
        else:
            print("Mistake in the model parameters, standardization has not been performed".format(std_file))
            return

        standardization = Rescaling(mean_std, stding_mat)
        # standardization = nn.BatchNorm2d(nb_channels_in, affine=False)

        proj = nn.Conv2d(nb_channels_in, args.L_proj_size, kernel_size=args.L_kernel_size, stride=1,
                         padding=0, bias=False)
        nb_channels_in = args.L_proj_size

        linear_proj = LinearProj(standardization, proj, args.L_kernel_size)
        ###########################################################################################

        # Create ISTC
        ###########################################################################################
        if args.arch == 'sparsescatnet':
            print("=> loading model with phase scattering {}, linear projection" \
                       "(projection dimension {}), ISTC' with {} iterations, dictionary size {}".
                  format(args.scattering_wph, args.L_proj_size, args.n_iterations, args.dictionary_size))

            istc = ISTC(nb_channels_in, dictionary_size=args.dictionary_size, n_iterations=args.n_iterations,
                        lambda_0=args.lambda_0, lambda_star=args.lambda_star, lambda_star_lb=args.lambda_star_lb,
                        grad_lambda_star=args.grad_lambda_star, epsilon_lambda_0=args.epsilon_lambda_0,
                        output_rec=False)

        elif args.arch == 'sparsescatnetw':
            print("=> loading model with phase scattering {}, linear projection" \
                       "(projection dimension {}), ISTCW' with {} iterations, dictionary size {}".
                  format(args.scattering_wph, args.L_proj_size, args.n_iterations, args.dictionary_size))

            istc = ISTC(nb_channels_in, dictionary_size=args.dictionary_size, n_iterations=args.n_iterations,
                        lambda_0=args.lambda_0, lambda_star=args.lambda_star, lambda_star_lb=args.lambda_star_lb,
                        grad_lambda_star=args.grad_lambda_star, epsilon_lambda_0=args.epsilon_lambda_0,
                        output_rec=False, use_W=True)

        # Create model
        ###########################################################################################
        model = SparseScatNet(scattering, linear_proj, istc, nn.Identity(), return_full_inf=False)

    else:
        print("Unknown model")
        return

    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # load model from a checkpoint
    if args.model_checkpoint:
        if os.path.isfile(args.model_checkpoint):
            print("=> loading model from checkpoint '{}'".format(args.model_checkpoint))
            checkpoint = torch.load(args.model_checkpoint)
            model_dict = model.state_dict()
            # For the convergence analysis, we do not need classifiers, hence we do not load it and keep Identity
            checkpoint_dict = checkpoint['state_dict']
            checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            dictionary = model.module.istc.dictionary_weight.data
            w_matrix = model.module.istc.w_weight.data

            print("dictionary matrix size {}".format(dictionary.size()))
            print("=> loaded checkpoint '{}' (epoch {})".format(args.model_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_checkpoint))
            return

    model.eval()

    return model, val_loader, dictionary, w_matrix


def compute_conv_model(model, val_loader, dictionary, w_matrix, override_lambdas=None):
    with torch.no_grad():
        input_size, dict_size = dictionary.size(0), dictionary.size(1)

        if override_lambdas is not None:
            n_iterations = len(override_lambdas)
        else:
            n_iterations = model.module.istc.n_iterations

        loss_curve = np.zeros(n_iterations)
        conv = np.zeros(n_iterations)
        x_star_norm = np.zeros(1)
        support_size_x_star = np.zeros(n_iterations)
        support_size_x_star_curve = np.zeros(200)
        support_size_model = np.zeros(n_iterations)
        support_incl = np.zeros(n_iterations)
        support_diff = np.zeros(n_iterations)

        if override_lambdas is not None:
            lambdas = override_lambdas
        else:
            lambdas = torch.zeros(n_iterations)
            lambdas[-1] = model.module.istc.lambda_star
            for i in range(n_iterations-1):
                lambdas[i] = model.module.istc.lambdas[i]

        for _, (input_batch, target) in enumerate(val_loader):
            input_batch = input_batch.cuda()
            input_batch = model(input_batch, return_proj=True)
            batch_size, M, N = input_batch.size(0), input_batch.size(2), input_batch.size(3)
            # Use FISTA to compute the l1 problem solution
            x_star, support_size_x_star_batch = compute_sparse_code_FISTA(input_batch, dictionary, lambdas[-1],
                                                                          maxiter=200, tol=1e-3)
            x_star_norm += (x_star.norm(p=2) ** 2).sum().item()
            support_size_x_star_curve += support_size_x_star_batch

            support_size_x_star[:] += (x_star != 0).sum().item()

            x = input_batch.new_zeros(batch_size, dict_size, M, N)

            for i_iter in range(n_iterations):
                D_x = nn.functional.conv2d(x, dictionary)
                x = x + nn.functional.conv2d(input_batch - D_x, w_matrix.transpose(0, 1).contiguous())
                x = relu(x, lambdas[i_iter])

                rec_error = 0.5*((nn.functional.conv2d(x, dictionary) - input_batch).norm(p=2, dim=1) ** 2).sum().item()
                sparsity_loss = (lambdas[-1]*x).norm(p=1, dim=1).sum().item()
                loss_curve[i_iter] += rec_error + sparsity_loss
                conv[i_iter] += ((x - x_star).norm(p=2) ** 2).sum().item()
                support_size_model[i_iter] += (x != 0).sum()
                support_incl[i_iter] += (((x != 0) * (x_star != 0)).sum(dim=1) /
                                         torch.max(torch.ones(1).cuda(), (x_star != 0).sum(dim=1).type(torch.cuda.FloatTensor))).sum().item()
                support_diff[i_iter] += (((x != 0) * (x_star == 0)).sum(dim=1) /
                                         torch.max(torch.ones(1).cuda(), (x_star != 0).sum(dim=1).type(torch.cuda.FloatTensor))).sum().item()

        support_incl /= (M * N * len(val_loader.dataset))
        support_diff /= (M * N * len(val_loader.dataset))
        support_size_x_star /= (M * N * len(val_loader.dataset))
        support_size_x_star_curve /= (M * N * len(val_loader.dataset))
        support_size_model /= (M * N * len(val_loader.dataset))
        loss_curve /= (M * N * len(val_loader.dataset))
        conv_rel = conv / x_star_norm

    return loss_curve, conv_rel, support_incl, support_diff, support_size_x_star, support_size_x_star_curve, \
           support_size_model


def compute_conv_FISTA(model, val_loader, dictionary, nb_iter, override_lambda_star=None):
    with torch.no_grad():
        L = torch.symeig(torch.mm(dictionary[..., 0, 0].t(), dictionary[..., 0, 0]))[0][-1]
        input_size, dict_size = dictionary.size(0), dictionary.size(1)

        if override_lambda_star is not None:
            lambda_star = override_lambda_star
        else:
            lambda_star = model.module.istc.lambda_star

        loss_curve = np.zeros(nb_iter)
        conv = np.zeros(nb_iter)
        x_star_norm = np.zeros(1)

        for _, (input_batch, target) in enumerate(val_loader):
            input_batch = input_batch.cuda()
            input_batch = model(input_batch, return_proj=True)
            batch_size, M, N = input_batch.size(0), input_batch.size(2), input_batch.size(3)
            # Use FISTA to compute the l1 problem solution
            x_star = compute_sparse_code_FISTA(input_batch, dictionary, lambda_star, maxiter=200, tol=1e-3)[0]
            x_star_norm += (x_star.norm(p=2) ** 2).sum().item()

            x = input_batch.new_zeros(batch_size, dict_size, M, N)
            t = 1
            y = x.clone()
            for i_iter in range(nb_iter):
                x_old = x.clone()
                D_y = nn.functional.conv2d(y, dictionary)
                y = y + nn.functional.conv2d(input_batch - D_y, dictionary.transpose(0, 1).contiguous()) / L
                x = relu(y, lambda_star / L)
                t0 = t
                t = (1. + math.sqrt(1. + 4. * t ** 2)) / 2.
                y = x + ((t0 - 1.) / t) * (x - x_old)

                rec_error = 0.5 * ((nn.functional.conv2d(x, dictionary) - input_batch).norm(p=2, dim=1) ** 2).sum().item()
                sparsity_loss = (lambda_star * x).norm(p=1, dim=1).sum().item()
                loss_curve[i_iter] += rec_error + sparsity_loss
                conv[i_iter] += ((x - x_star).norm(p=2) ** 2).sum().item()

        loss_curve /= (M * N * len(val_loader.dataset))
        conv_rel = conv / x_star_norm

    return loss_curve, conv_rel


def compute_conv_ISTA(model, val_loader, dictionary, nb_iter, override_lambda_star=None):
    with torch.no_grad():
        L = torch.symeig(torch.mm(dictionary[..., 0, 0].t(), dictionary[..., 0, 0]))[0][-1]
        input_size, dict_size = dictionary.size(0), dictionary.size(1)

        if override_lambda_star is not None:
            lambda_star = override_lambda_star
        else:
            lambda_star = model.module.istc.lambda_star

        loss_curve = np.zeros(nb_iter)
        conv = np.zeros(nb_iter)
        x_star_norm = np.zeros(1)

        for _, (input_batch, target) in enumerate(val_loader):
            input_batch = input_batch.cuda()
            input_batch = model(input_batch, return_proj=True)
            batch_size, M, N = input_batch.size(0), input_batch.size(2), input_batch.size(3)
            # Use FISTA to compute the l1 problem solution
            x_star = compute_sparse_code_FISTA(input_batch, dictionary, lambda_star, maxiter=200, tol=1e-3)[0]
            x_star_norm += (x_star.norm(p=2) ** 2).sum().item()

            x = input_batch.new_zeros(batch_size, dict_size, M, N)
            for i_iter in range(nb_iter):
                D_x = nn.functional.conv2d(x, dictionary)
                x = x + nn.functional.conv2d(input_batch - D_x, dictionary.transpose(0, 1).contiguous()) / L
                x = relu(x, lambda_star / L)

                rec_error = 0.5 * ((nn.functional.conv2d(x, dictionary) - input_batch).norm(p=2, dim=1) ** 2).sum().item()
                sparsity_loss = (lambda_star * x).norm(p=1, dim=1).sum().item()
                loss_curve[i_iter] += rec_error + sparsity_loss
                conv[i_iter] += ((x - x_star).norm(p=2) ** 2).sum().item()

        loss_curve /= (M * N * len(val_loader.dataset))
        conv_rel = conv / x_star_norm

    return loss_curve, conv_rel


def compute_sparse_code_FISTA(input_batch, dictionary, lambda_star, maxiter, tol=1e-3):
    with torch.no_grad():
        L = torch.symeig(torch.mm(dictionary[..., 0, 0].t(), dictionary[..., 0, 0]))[0][-1]
        input_size, dict_size = dictionary.size(0), dictionary.size(1)

        support_size_FISTA_curve = np.zeros(maxiter)

        batch_size, M, N = input_batch.size(0), input_batch.size(2), input_batch.size(3)
        x = input_batch.new_zeros(batch_size, dict_size, M, N)
        t = 1
        y = x.clone()
        for i_iter in range(maxiter):
            x_old = x.clone()
            D_y = nn.functional.conv2d(y, dictionary)
            y = y + nn.functional.conv2d(input_batch - D_y, dictionary.transpose(0, 1).contiguous()) / L
            x = relu(y, lambda_star / L)
            t0 = t
            t = (1. + math.sqrt(1. + 4. * t ** 2)) / 2.
            y = x + ((t0 - 1.) / t) * (x - x_old)

            support_size_FISTA_curve[i_iter] = (x != 0).sum().item()

            diff = ((x - x_old).norm(p=2, dim=1) / (x.norm(p=2, dim=1) + 1e-10))

            if tol is not None and diff.max() < tol:
                support_size_FISTA_curve[i_iter:] = support_size_FISTA_curve[i_iter - 1]
                break

    return x, support_size_FISTA_curve


def compute_sparse_code_ISTA(input_batch, dictionary, lambda_star, maxiter, tol=1e-3):
    with torch.no_grad():
        L = torch.symeig(torch.mm(dictionary[..., 0, 0].t(), dictionary[..., 0, 0]))[0][-1]
        input_size, dict_size = dictionary.size(0), dictionary.size(1)

        support_size_ISTA_curve = np.zeros(maxiter)

        batch_size, M, N = input_batch.size(0), input_batch.size(2), input_batch.size(3)
        x = input_batch.new_zeros(batch_size, dict_size, M, N)
        for i_iter in range(maxiter):
            x_old = x.clone()
            D_x = nn.functional.conv2d(x, dictionary)
            x = x + nn.functional.conv2d(input_batch - D_x, dictionary.transpose(0, 1).contiguous()) / L
            x = relu(x, lambda_star / L)

            support_size_ISTA_curve[i_iter] = (x != 0).sum().item()

            diff = ((x - x_old).norm(p=2, dim=1) / (x.norm(p=2, dim=1) + 1e-10))

            if tol is not None and diff.max() < tol:
                support_size_ISTA_curve[i_iter:] = support_size_ISTA_curve[i_iter - 1]
                break

    return x, support_size_ISTA_curve


if __name__ == '__main__':
    model, val_loader, dictionary, w_matrix = main()
