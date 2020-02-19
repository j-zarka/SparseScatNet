# Modified from PyTorch example for ImageNet classification:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from kymatio import Scattering2D
from phase_scattering2d_torch import ScatteringTorch2D_wph
from models.ISTC import ISTC
from models.Rescaling import Rescaling
from models.LinearProj import LinearProj
from models.Classifier import Classifier
from models.SparseScatNet import SparseScatNet

from torch.utils.tensorboard import SummaryWriter
from utils import print_and_write, compute_stding_matrix

model_names = ['sparsescatnet', 'sparsescatnetw', 'scatnet']

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
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# Additional training args
parser.add_argument('--learning-rate-adjust-frequency', default=30, type=int,
                    help='number of epoch after which learning rate is decayed by 10 (default: 30)')
parser.add_argument('--logdir', default='./training_logs', type=str,
                    help='directory for training logs')
parser.add_argument('--savedir', default='./checkpoints', type=str,
                    help='directory to save checkpoints')

# Train on a subsample of classes
parser.add_argument('--nb-classes', default=1000, type=int, help='number of classes randomly chosen '
                    'used for training and validation (default: 1000 = whole train/val dataset)')
parser.add_argument('--class-indices', default=None, help='numpy array of indices used in case nb-classes < 1000')

# Scattering parameters
parser.add_argument('--scattering-order2', help='Compute order 2 scattering coefficients',
            action='store_true')
parser.add_argument('--scattering-wph', help='Use phase scattering',
            action='store_true')
parser.add_argument('--scat-angles', default=8, type=int, help='number of orientations for scattering')
parser.add_argument('--backend', default='torch', type=str, help='scattering backend')
parser.add_argument('--scattering-nphases', default=4, type=int,
        help='number of phases in the first order of the phase harmonic scattering transform')
parser.add_argument('--scattering-J', default=4, type=int,
        help='maximum scale for the scattering transform')

# Linear projection parameters
parser.add_argument('--L-proj-size', default=256, type=int,
                    help='dimension of the linear projection')
parser.add_argument('--L-kernel-size', default=1, type=int, help='kernel size of L')

# ISTC(W) parameters
parser.add_argument('--n-iterations', default=12, type=int, help='number of iterations for ISTC')
parser.add_argument('--dictionary-size', default=2048, type=int,
        help='size of the sparse coding dictionary')
parser.add_argument('--output-rec', help='output reconstruction', action='store_true')
parser.add_argument('--lambda-0', default=0.3, type=float, help='lambda_0')
parser.add_argument('--lambda-star', default=0.05, type=float, help='lambda_star')
parser.add_argument('--lambda-star-lb', default=0.05, type=float, help='lambda_star lower bound')
parser.add_argument('--epsilon-lambda-0', default=1., type=float, help='epsilon for lambda_0 adjustment')
parser.add_argument('--l0-inf-init', help='initialization of lambda_0 as W^Tx inf', action='store_true')
parser.add_argument('--grad-lambda-star', help='gradient on lambda_star', action='store_true')

# Classifier parameters
parser.add_argument('--avg-ker-size', default=1, type=int, help='size of averaging kernel')
parser.add_argument('--classifier-type', default='mlp', type=str, help='classifier type')
parser.add_argument('--nb-hidden-units', default=2048, type=int, help='number of hidden units for mlp classifier')
parser.add_argument('--dropout-p-mlp', default=0.3, type=float, help='dropout probability in mlp')
parser.add_argument('--nb-l-mlp', default=2, type=int, help='number of hidden layers in mlp')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def main_worker(args):
    best_acc1 = 0
    best_acc5 = 0
    best_epoch_acc1 = 0
    best_epoch_acc5 = 0

    logs_dir = args.logdir
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    checkpoint_savedir = args.savedir
    if not os.path.exists(checkpoint_savedir):
        os.makedirs(checkpoint_savedir)

    logfile = os.path.join(logs_dir, 'training_{}_b_{}_lrfreq_{}.log'.format(
        args.arch, args.batch_size, args.learning_rate_adjust_frequency))

    summaryfile = os.path.join(logs_dir, 'summary_file.txt')

    checkpoint_savefile = os.path.join(checkpoint_savedir, '{}_batchsize_{}_lrfreq_{}.pth.tar'.format(
        args.arch, args.batch_size, args.learning_rate_adjust_frequency))

    best_checkpoint_savefile = os.path.join(checkpoint_savedir,'{}_batchsize_{}_lrfreq_{}_best.pth.tar'.format(
                                                args.arch, args.batch_size, args.learning_rate_adjust_frequency))

    writer = SummaryWriter(logs_dir)

    # Data loading code
    ###########################################################################################
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    # can use a subset of all classes (specified in a file or randomly chosen)
    if args.nb_classes < 1000:
        train_indices = list(np.load('utils_sampling/imagenet_train_class_indices.npy'))
        val_indices = list(np.load('utils_sampling/imagenet_val_class_indices.npy'))
        classes_names = torch.load('utils_sampling/labels_dict')
        if args.class_indices is not None:
            class_indices = torch.load(args.class_indices)
        else:
            perm = torch.randperm(1000)
            class_indices = perm[:args.nb_classes].tolist()
        train_indices_full = [x for i in range(len(class_indices)) for x in range(train_indices[class_indices[i]],
                                                                                  train_indices[class_indices[i] + 1])]
        val_indices_full = [x for i in range(len(class_indices)) for x in range(val_indices[class_indices[i]],
                                                                                val_indices[class_indices[i] + 1])]
        classes_indices_file = os.path.join(logs_dir, 'classes_indices_selected')
        selected_classes_names = [classes_names[i] for i in class_indices]
        torch.save(class_indices, classes_indices_file)
        print_and_write('Selected {} classes indices:  {}'.format(args.nb_classes, class_indices), logfile,
                        summaryfile)
        print_and_write('Selected {} classes names:  {}'.format(args.nb_classes, selected_classes_names), logfile,
                        summaryfile)
        if args.random_seed is not None:
            print_and_write('Random seed used {}'.format(args.random_seed), logfile, summaryfile)

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices_full)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices_full)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    ###########################################################################################

    # Model creation
    ###########################################################################################
    if args.arch in model_names:

        n_space = 224
        nb_channels_in = 3

        # create scattering
        J = args.scattering_J
        L_ang = args.scat_angles

        max_order = 2 if args.scattering_order2 else 1

        if args.scattering_wph:
            A = args.scattering_nphases
            scattering = ScatteringTorch2D_wph(J=J, shape=(224, 224), L=L_ang, A=A, max_order=max_order,
                                               backend=args.backend)
        else:
            scattering = Scattering2D(J=J, shape=(224, 224), L=L_ang, max_order=max_order,
                                      backend=args.backend)
        # Flatten scattering
        scattering = nn.Sequential(scattering, nn.Flatten(1, 2))

        if args.scattering_wph:
            nb_channels_in += 3 * A * L_ang * J
        else:
            nb_channels_in += 3 * L_ang * J

        if max_order == 2:
            nb_channels_in += 3 * (L_ang ** 2) * J * (J - 1) // 2

        n_space = n_space // (2 ** J)
    ###########################################################################################

        # create linear proj
        # Standardization (can also be performed with BatchNorm2d(affine=False))
        if not os.path.exists('standardization'):
            os.makedirs('standardization')
        std_file = 'standardization/ImageNet2012_scattering_J{}_order{}_wph_{}_nphases_{}_nb_classes_{}.pth.tar'.format(
            args.scattering_J, 2 if args.scattering_order2 else 1, args.scattering_wph,
            args.scattering_nphases if args.scattering_wph else 0, args.nb_classes)

        if os.path.isfile(std_file):
            print_and_write("=> loading scattering mean and std '{}'".format(std_file), logfile)
            std_dict = torch.load(std_file)
            mean_std = std_dict['mean']
            stding_mat = std_dict['matrix']
        else:
            mean_std, stding_mat, std = compute_stding_matrix(train_loader, scattering, logfile)
            print_and_write("=> saving scattering mean and std '{}'".format(std_file), logfile)
            std_dict = {'mean': mean_std, 'std': std, 'matrix': stding_mat}
            torch.save(std_dict, std_file)

        standardization = Rescaling(mean_std, stding_mat)
        # standardization = nn.BatchNorm2d(nb_channels_in, affine=False)

        if args.arch in ['sparsescatnet', 'sparsescatnetw']:
            proj = nn.Conv2d(nb_channels_in, args.L_proj_size, kernel_size=args.L_kernel_size, stride=1,
                             padding=0, bias=False)
            nb_channels_in = args.L_proj_size
            linear_proj = LinearProj(standardization, proj, args.L_kernel_size)
        else:  # scatnet
            proj = nn.Identity()
            linear_proj = LinearProj(standardization, proj, 0)

        ###########################################################################################

        # Create ISTC (when applicable)
        if args.arch in ['sparsescatnet', 'sparsescatnetw']:
        ###########################################################################################
            if args.arch == 'sparsescatnet':
                arch_log = "=> creating model SparseScatNet with phase scattering {}, linear projection " \
                           "(projection dimension {}), ISTC with {} iterations, dictionary size {}, classifier {} " \
                           "pipeline".format(args.scattering_wph, args.L_proj_size, args.n_iterations,
                                             args.dictionary_size, args.classifier_type)

                istc = ISTC(nb_channels_in, dictionary_size=args.dictionary_size, n_iterations=args.n_iterations,
                            lambda_0=args.lambda_0, lambda_star=args.lambda_star, lambda_star_lb=args.lambda_star_lb,
                            grad_lambda_star=args.grad_lambda_star, epsilon_lambda_0=args.epsilon_lambda_0,
                            output_rec=args.output_rec)

            elif args.arch == 'sparsescatnetw':
                arch_log = "=> creating model SparseScatNetW with phase scattering {}, linear projection " \
                           "(projection dimension {}), ISTCW with {} iterations, dictionary size {}, classifier {} " \
                           "pipeline".format(args.scattering_wph, args.L_proj_size, args.n_iterations,
                                             args.dictionary_size, args.classifier_type)

                istc = ISTC(nb_channels_in, dictionary_size=args.dictionary_size, n_iterations=args.n_iterations,
                            lambda_0=args.lambda_0, lambda_star=args.lambda_star, lambda_star_lb=args.lambda_star_lb,
                            grad_lambda_star=args.grad_lambda_star, epsilon_lambda_0=args.epsilon_lambda_0,
                            output_rec=args.output_rec, use_W=True)

            if not args.output_rec:
                nb_channels_in = args.dictionary_size

        elif args.arch == 'scatnet':
            arch_log = "=> creating model ScatNet with phase scattering {} and classifier {}".\
                format(args.scattering_wph, args.classifier_type)

        # Create classifier
        ###########################################################################################

        classifier = Classifier(n_space, nb_channels_in, classifier_type=args.classifier_type,
                                nb_classes=1000, nb_hidden_units=args.nb_hidden_units, nb_l_mlp=args.nb_l_mlp,
                                dropout_p_mlp=args.dropout_p_mlp, avg_ker_size=args.avg_ker_size)

        # Create model
        ###########################################################################################
        if args.arch in ['sparsescatnet', 'sparsescatnetw']:
            model = SparseScatNet(scattering, linear_proj, istc, classifier, return_full_inf=True)  # print model info

        elif args.arch == 'scatnet':
            model = nn.Sequential(scattering, linear_proj, classifier)
    else:
        print_and_write("Unknown model", logfile, summaryfile)
        return

    print_and_write(arch_log, logfile, summaryfile)
    print_and_write('Number of epochs {}, learning rate decay epochs {}'.format(args.epochs,
                                                                                args.learning_rate_adjust_frequency),
                                                                                logfile, summaryfile)

    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_and_write("=> loading checkpoint '{}'".format(args.resume), logfile, summaryfile)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_and_write("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), logfile,
                            summaryfile)
        else:
            print_and_write("=> no checkpoint found at '{}'".format(args.resume), logfile, summaryfile)

    cudnn.benchmark = True

    if args.arch in ['sparsescatnet', 'sparsescatnetw']:

        lambda_0_max = compute_lambda_0(train_loader, model).item()

        if args.l0_inf_init:
            with torch.no_grad():
                model.module.istc.lambda_0.data.fill_(lambda_0_max)
                model.module.istc.log_lambda_0.data = torch.log(model.module.istc.lambda_0.data)
                model.module.istc.gamma.data.fill_(np.power(args.lambda_star / lambda_0_max, 1. / args.n_iterations))
                model.module.istc.log_gamma.data = torch.log(model.module.istc.gamma.data)
                for i in range(args.n_iterations - 1):
                    model.module.istc.lambdas.data[i] = lambda_0_max * (model.module.istc.gamma.data ** (i + 1))
                    model.module.istc.log_lambdas.data[i] = torch.log(model.module.istc.lambdas.data[i])

        print_and_write('Lambda star lower bound {:.3f}'.format(args.lambda_star_lb), logfile, summaryfile)
        print_and_write('epsilon lambda_0 {}'.format(args.epsilon_lambda_0), logfile, summaryfile)

        with torch.no_grad():
            with np.printoptions(precision=2, suppress=True):
                    lambda_0_init = model.module.istc.lambda_0.data.cpu().item()
                    print_and_write('Lambda_0 init {:.2f}'.format(lambda_0_init), logfile, summaryfile)

                    lambdas_init = model.module.istc.lambdas.data.cpu().numpy()
                    print_and_write('Lambdas init {}'.format(lambdas_init), logfile, summaryfile)

                    print_and_write('Lambda_star init {:.2f}'.format(args.lambda_star), logfile, summaryfile)

                    gamma_init = model.module.istc.gamma.data.cpu().item()
                    print_and_write('Gamma init {:.2f}'.format(gamma_init), logfile, summaryfile)

            count = 0
            for i in range(args.dictionary_size):
                if model.module.istc.dictionary_weight.data[:, i].norm(p=2) < 0.99 or \
                        model.module.istc.dictionary_weight.data[:, i].norm(p=2) > 1.01:
                    count += 1

            if count == 0:
                print_and_write("Dictionary atoms initially well normalized", logfile,summaryfile)
            else:
                print_and_write("{} dictionary atoms not initially well normalized".format(count), logfile, summaryfile)

            gram = torch.matmul(model.module.istc.w_weight.data[..., 0, 0].t(),
                                      model.module.istc.dictionary_weight.data[..., 0, 0]).cpu().numpy()

            if args.arch == 'sparsescatnetw':
                count = 0
                for i in range(args.dictionary_size):
                    if gram[i, i] < 0.99 or gram[i, i] > 1.01:
                        count += 1
                if count == 0:
                    print_and_write("W^T D diagonal elements well equal to 1", logfile, summaryfile)
                else:
                    print_and_write("{} W^T D diagonal elements not equal to 1".format(count),
                                    logfile, summaryfile)

            gram = np.abs(gram)
            for i in range(args.dictionary_size):
                gram[i, i] = 0

            print_and_write("Initial max coherence {:.3f}, median coherence {:.3f}".
                            format(np.max(gram), np.median(gram)), logfile, summaryfile)

    if args.evaluate:
        print_and_write("Evaluating model at epoch {}...".format(args.start_epoch), logfile)
        validate(val_loader, model, criterion, args.start_epoch, args, logfile, summaryfile, writer)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, logfile, writer)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, epoch, args, logfile, summaryfile, writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch_acc1 = epoch
        if acc5 > best_acc5:
            best_acc5 = acc5
            best_epoch_acc5 = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_filename=checkpoint_savefile, best_checkpoint_filename=best_checkpoint_savefile)

    if args.arch in ['sparsescatnet', 'sparsescatnetw']:
        with torch.no_grad():
            with np.printoptions(precision=2, suppress=True):

                lambda_0_final = model.module.istc.lambda_0.data.cpu().item()
                print_and_write('Lambda_0 final {:.2f}'.format(lambda_0_final), logfile, summaryfile)

                lambdas_final = model.module.istc.lambdas.data.cpu().numpy()
                print_and_write('Lambdas final {}'.format(lambdas_final), logfile, summaryfile)

                lambda_star_final = model.module.istc.lambda_star.data.cpu().item()
                print_and_write('Lambda_star final {:.2f}'.format(lambda_star_final), logfile, summaryfile)

                gamma_final = model.module.istc.gamma.data.cpu().item()
                print_and_write('Gamma final {:.2f}'.format(gamma_final), logfile, summaryfile)

            count = 0
            for i in range(args.dictionary_size):
                if model.module.istc.dictionary_weight.data[:, i].norm(p=2) < 0.99 or \
                        model.module.istc.dictionary_weight.data[:, i].norm(p=2) > 1.01:
                    count += 1

            if count == 0:
                print_and_write("Dictionary atoms finally well normalized", logfile, summaryfile)
            else:
                print_and_write("{} dictionary atoms not finally well normalized".format(count), logfile, summaryfile)

            gram = torch.matmul(model.module.istc.w_weight.data[..., 0, 0].t(),
                                model.module.istc.dictionary_weight.data[..., 0, 0]).cpu().numpy()

            if args.arch == 'sparsescatnetw':
                count = 0
                for i in range(args.dictionary_size):
                    if gram[i, i] < 0.99 or gram[i, i] > 1.01:
                        count += 1
                if count == 0:
                    print_and_write("W^T D diagonal elements well equal to 1", logfile, summaryfile)
                else:
                    print_and_write("{} W^T D diagonal elements not equal to 1".format(count),
                                    logfile, summaryfile)

            gram = np.abs(gram)
            for i in range(args.dictionary_size):
                gram[i, i] = 0

            print_and_write("Final max coherence {:.3f}, median coherence {:.3f}".
                            format(np.max(gram), np.median(gram)), logfile, summaryfile)

    print_and_write(
        "Best top 1 accuracy {:.2f} at epoch {}, best top 5 accuracy {:.2f} at epoch {}".format(best_acc1,
                                                                                                best_epoch_acc1,
                                                                                                best_acc5,
                                                                                                best_epoch_acc5),
        logfile, summaryfile)


def train(train_loader, model, criterion, optimizer, epoch, args, logfile, writer):
    batch_time = AverageMeter('Time', ':.1f')
    data_time = AverageMeter('Data', ':.1f')
    losses = AverageMeter('Loss', ':.2f')
    top1 = AverageMeter('Acc@1', ':.1f')
    top5 = AverageMeter('Acc@5', ':.1f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    if args.arch in ['sparsescatnet', 'sparsescatnetw']:
        rec_losses_rel = AverageMeter('Rec. losses', ':.2f')
        progress.add(rec_losses_rel)
        sparsities = AverageMeterTensor(args.n_iterations)
        support_sizes = AverageMeterTensor(args.n_iterations)
        support_diffs = AverageMeterTensor(args.n_iterations)


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        if args.arch in ['sparsescatnet', 'sparsescatnetw']:
            output, lambda_0_max_batch, sparsity, support_size, support_diff, rec_loss_rel = model(input)
        else:
            output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # Record useful indicators for ISTC
        if args.arch in ['sparsescatnet', 'sparsescatnetw']:
            lambda_0_max = lambda_0_max_batch.max().item()
            rec_losses_rel.update(rec_loss_rel.mean().item(), input.size(0))
            if len(sparsity) > args.n_iterations:  # multi-GPU
                sparsities.update(sparsity.reshape(-1, args.n_iterations).mean(dim=0).cpu().numpy(), input.size(0))
                support_sizes.update(support_size.reshape(-1, args.n_iterations).mean(dim=0).cpu().numpy(), input.size(0))
                support_diffs.update(support_diff.reshape(-1, args.n_iterations).mean(dim=0).cpu().numpy(), input.size(0))

            else:
                sparsities.update(sparsity.cpu().numpy(), input.size(0))
                support_sizes.update(support_size.cpu().numpy(), input.size(0))
                support_diffs.update(support_diff.cpu().numpy(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_and_write('\n', logfile)
            progress.display(i, logfile)

            # Print ISTC model information
            if args.arch in ['sparsescatnet', 'sparsescatnetw']:
                print_and_write('Sparsity {}\t Support size {}\t Support diff {}'
                                .format(np.array2string(sparsities.avg, formatter={'float_kind':lambda x: "%.1f" % x}),
                                        np.array2string(support_sizes.avg, formatter={'float_kind': lambda x: "%.0f" % x}),
                                        np.array2string(support_diffs.avg, formatter={'float_kind': lambda x: "%.0f" % x})),
                                        logfile)

                with torch.no_grad():
                    lambda_0 = model.module.istc.lambda_0.data.clone().cpu()
                    lambdas = model.module.istc.lambdas.data.clone().cpu().numpy()
                    log_lambda_0 = model.module.istc.log_lambda_0.data.clone()  # Need to keep it on GPU for
                    # log_lambda_0 update with epsilon lambda 0. Other information can be put on CPU
                    lambda_star = model.module.istc.lambda_star.data.clone().cpu()
                    gamma = model.module.istc.gamma.data.clone().cpu()

                    gram = torch.matmul(model.module.istc.w_weight.data[..., 0, 0].t(),
                                        model.module.istc.dictionary_weight.data[..., 0, 0]).cpu().numpy()

                    gram = np.abs(gram)
                    for k in range(args.dictionary_size):
                        gram[k, k] = 0

                    max_coherence = np.max(gram)
                    median_coherence = np.median(gram)

                    print_and_write('Lambda_0_max: {:.3f}'.format(lambda_0_max), logfile)
                    print_and_write('Lambda_0: {:.3f}'.format(lambda_0.item()), logfile)
                    with np.printoptions(precision=2, suppress=True):
                        print_and_write('Lambdas: {}'.format(lambdas.reshape(args.n_iterations - 1)), logfile)
                    print_and_write('Lambda_star: {:.3f}'.format(lambda_star.item()), logfile)
                    print_and_write('Gamma: {:.2f}'.format(gamma.item()), logfile)
                    print_and_write("Max coherence {:.3f}, median coherence {:.3f}".format(max_coherence,
                                                                                           median_coherence), logfile)

        # update lambda_0
        if args.arch in ['sparsescatnet', 'sparsescatnetw']:
            if args.epsilon_lambda_0 > 0:
                model.module.istc.log_lambda_0.data = log_lambda_0 + args.epsilon_lambda_0 * \
                                                      (np.log(lambda_0_max) - log_lambda_0)

    # Print statistics summary
    print_and_write('\n Train Epoch {}, * Acc@1 {:.2f} Acc@5 {:.2f}'.format(epoch, top1.avg, top5.avg), logfile)

    if args.arch in ['sparsescatnet', 'sparsescatnetw']:
        print_and_write('Rec loss rel {:.2f}\t Sparsity {}\t Support size {}\t Support diff {}'.
                        format(rec_losses_rel.avg, np.array2string(sparsities.avg, formatter={
                        'float_kind': lambda x: "%.1f" % x}), np.array2string(support_sizes.avg, formatter={
                        'float_kind': lambda x: "%.0f" % x}), np.array2string(support_diffs.avg, formatter={
                        'float_kind': lambda x: "%.0f" % x})), logfile)

    if writer is not None:
        writer.add_scalar('top5_train', top5.avg, global_step=epoch)
        writer.add_scalar('top1_train', top1.avg, global_step=epoch)


def validate(val_loader, model, criterion, epoch, args, logfile, summaryfile, writer):
    batch_time = AverageMeter('Time', ':.1f')
    data_time = AverageMeter('Data', ':.1f')
    losses = AverageMeter('Loss', ':.2f')
    top1 = AverageMeter('Acc@1', ':.1f')
    top5 = AverageMeter('Acc@5', ':.1f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Validation Epoch: [{}]".format(epoch))

    if args.arch in ['sparsescatnet', 'sparsescatnetw']:
        rec_losses_rel = AverageMeter('Rec. losses', ':.2f')
        progress.add(rec_losses_rel)
        sparsities = AverageMeterTensor(args.n_iterations)
        support_sizes = AverageMeterTensor(args.n_iterations)
        support_diffs = AverageMeterTensor(args.n_iterations)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            if args.arch in ['sparsescatnet', 'sparsescatnetw']:
                output, _, sparsity, support_size, support_diff, rec_loss_rel = model(input)
            else:
                output = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # Record useful indicators for ISTC
            if args.arch in ['sparsescatnet', 'sparsescatnetw']:
                rec_losses_rel.update(rec_loss_rel.mean().item(), input.size(0))
                if len(sparsity) > args.n_iterations:  # multi-GPU
                    sparsities.update(sparsity.reshape(-1, args.n_iterations).mean(dim=0).cpu().numpy(), input.size(0))
                    support_sizes.update(support_size.reshape(-1, args.n_iterations).mean(dim=0).cpu().numpy(),
                                         input.size(0))
                    support_diffs.update(support_diff.reshape(-1, args.n_iterations).mean(dim=0).cpu().numpy(),
                                         input.size(0))

                else:
                    sparsities.update(sparsity.cpu().numpy(), input.size(0))
                    support_sizes.update(support_size.cpu().numpy(), input.size(0))
                    support_diffs.update(support_diff.cpu().numpy(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_and_write('\n', logfile)
                progress.display(i, logfile)

                # Print ISTC model information
                if args.arch in ['sparsescatnet', 'sparsescatnetw']:
                    print_and_write('Sparsity {}\t Support size {}\t Support diff {}'
                                    .format(
                        np.array2string(sparsities.avg, formatter={'float_kind': lambda x: "%.1f" % x}),
                        np.array2string(support_sizes.avg, formatter={'float_kind': lambda x: "%.0f" % x}),
                        np.array2string(support_diffs.avg, formatter={'float_kind': lambda x: "%.0f" % x})),
                                    logfile)

                    lambda_0 = model.module.istc.lambda_0.data.clone().cpu()
                    lambdas = model.module.istc.lambdas.data.clone().cpu().numpy()
                    lambda_star = model.module.istc.lambda_star.data.clone().cpu()
                    gamma = model.module.istc.gamma.data.clone().cpu()

                    gram = torch.matmul(model.module.istc.w_weight.data[..., 0, 0].t(),
                                        model.module.istc.dictionary_weight.data[..., 0, 0]).cpu().numpy()

                    gram = np.abs(gram)
                    for k in range(args.dictionary_size):
                        gram[k, k] = 0

                    max_coherence = np.max(gram)
                    median_coherence = np.median(gram)

                    print_and_write('Lambda_0: {:.3f}'.format(lambda_0.item()), logfile)
                    with np.printoptions(precision=2, suppress=True):
                        print_and_write('Lambdas: {}'.format(lambdas.reshape(args.n_iterations - 1)), logfile)
                    print_and_write('Lambda_star: {:.3f}'.format(lambda_star.item()), logfile)
                    print_and_write('Gamma: {:.2f}'.format(gamma.item()), logfile)
                    print_and_write("Max coherence {:.3f}, median coherence {:.3f}".format(max_coherence,
                                                                                           median_coherence),
                                    logfile)

    # Print statistics summary
    print_and_write('\n Validation Epoch {}, * Acc@1 {:.2f} Acc@5 {:.2f}'.format(epoch, top1.avg, top5.avg), logfile)

    if args.arch in ['sparsescatnet', 'sparsescatnetw']:
        print_and_write('Rec loss rel {:.2f}\t Sparsity {}\t Support size {}\t Support diff {}'.
                        format(rec_losses_rel.avg, np.array2string(sparsities.avg, formatter={
            'float_kind': lambda x: "%.1f" % x}), np.array2string(support_sizes.avg, formatter={
            'float_kind': lambda x: "%.0f" % x}), np.array2string(support_diffs.avg, formatter={
            'float_kind': lambda x: "%.0f" % x})), logfile)

        count = 0
        for i in range(args.dictionary_size):
            if model.module.istc.dictionary_weight.data[:, i].norm(p=2) < 0.99 or \
                    model.module.istc.dictionary_weight.data[:, i].norm(p=2) > 1.01:
                count += 1

        if count == 0:
            print_and_write("Dictionary atoms well normalized", logfile)
        else:
            print_and_write("{} dictionary atoms not well normalized".format(count), logfile)

        gram = torch.matmul(model.module.istc.w_weight.data[..., 0, 0].t(),
                            model.module.istc.dictionary_weight.data[..., 0, 0]).cpu().numpy()

        if args.arch == 'sparsescatnetw':
            count = 0
            for i in range(args.dictionary_size):
                if gram[i, i] < 0.99 or gram[i, i] > 1.01:
                    count += 1
            if count == 0:
                print_and_write("W^T D diagonal elements well equal to 1", logfile)
            else:
                print_and_write("{} W^T D diagonal elements not equal to 1".format(count),
                                logfile)

        gram = np.abs(gram)
        for i in range(args.dictionary_size):
            gram[i, i] = 0

        print_and_write("Max coherence {:.3f}, median coherence {:.3f}".
                        format(np.max(gram), np.median(gram)), logfile)

    if (epoch % args.learning_rate_adjust_frequency) == (args.learning_rate_adjust_frequency-1):
        print_and_write('Validation Epoch {} before learning rate adjustment n° {}, * Acc@1 {:.2f} Acc@5 {:.2f}'
                        .format(epoch, epoch // args.learning_rate_adjust_frequency+1, top1.avg, top5.avg), summaryfile)
        if args.arch in ['sparsescatnet', 'sparsescatnetw']:
            print_and_write('Rec loss rel {:.2f}\t Sparsity {}\t Support size {}\t Support diff {}'.
                            format(rec_losses_rel.avg, np.array2string(sparsities.avg, formatter={
                'float_kind': lambda x: "%.1f" % x}), np.array2string(support_sizes.avg, formatter={
                'float_kind': lambda x: "%.0f" % x}), np.array2string(support_diffs.avg, formatter={
                'float_kind': lambda x: "%.0f" % x})), summaryfile)

    if epoch == 0:
        print_and_write('First epoch, * Acc@1 {:.2f} Acc@5 {:.2f}'.format(top1.avg, top5.avg), summaryfile)
        if args.arch in ['sparsescatnet', 'sparsescatnetw']:
            print_and_write('Rec loss rel {:.2f}\t Sparsity {}\t Support size {}\t Support diff {}'.
                            format(rec_losses_rel.avg, np.array2string(sparsities.avg, formatter={
                'float_kind': lambda x: "%.1f" % x}), np.array2string(support_sizes.avg, formatter={
                'float_kind': lambda x: "%.0f" % x}), np.array2string(support_diffs.avg, formatter={
                'float_kind': lambda x: "%.0f" % x})), summaryfile)

    if epoch == args.epochs-1:
        print_and_write('Final epoch {}, * Acc@1 {:.2f} Acc@5 {:.2f}'.format(epoch, top1.avg, top5.avg), summaryfile)
        if args.arch in ['sparsescatnet', 'sparsescatnetw']:
            print_and_write('Rec loss rel {:.2f}\t Sparsity {}\t Support size {}\t Support diff {}'.
                            format(rec_losses_rel.avg, np.array2string(sparsities.avg, formatter={
                'float_kind': lambda x: "%.1f" % x}), np.array2string(support_sizes.avg, formatter={
                'float_kind': lambda x: "%.0f" % x}), np.array2string(support_diffs.avg, formatter={
                'float_kind': lambda x: "%.0f" % x})), summaryfile)

    if writer is not None:
        writer.add_scalar('top5_val', top5.avg, global_step=epoch)
        writer.add_scalar('top1_val', top1.avg, global_step=epoch)

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, checkpoint_filename='checkpoint.pth.tar',
                    best_checkpoint_filename='model_best.pth.tar'):
    torch.save(state, checkpoint_filename)
    if is_best:
        shutil.copyfile(checkpoint_filename, best_checkpoint_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logfile):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_and_write('\t'.join(entries), logfile)

    def add(self, *meters):
        for meter in meters:
            self.meters.append(meter)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeterTensor(object):
    def __init__(self, size):
        self.reset(size)

    def reset(self, size):
        self.avg = np.zeros(size)
        self.sum = np.zeros(size)
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every learning_rate_adjust_frequency epochs"""
    lr = args.lr * (0.1 ** (epoch // args.learning_rate_adjust_frequency))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_lambda_0(loader, model, nb_batches=1):
    with torch.no_grad():
        best_lambda = torch.zeros(1).cuda()

        for i, (input, target) in enumerate(loader):
            if i >= nb_batches:
                break
            input = input.cuda()
            input_proj = model(input, return_proj=True)
            WT_x_proj = nn.functional.conv2d(
                input_proj, model.module.istc.w_weight.data.transpose(0, 1).contiguous())
            abs_WT_x_proj = torch.abs(WT_x_proj)

            lambda_max = abs_WT_x_proj.max()
            if lambda_max > best_lambda[0]:
                best_lambda[0] = lambda_max

        return best_lambda


if __name__ == '__main__':
    main()
