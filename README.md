### Code implementation of paper: Deep Network Classification by Scattering and Homotopy Dictionary Learning
This repository contains the code to reproduce experiments in the paper: J. Zarka, L. Thiry, T. Angles and S. Mallat,
[Deep Network Classification by Scattering and Homotopy Dictionary Learning (2019)](https://arxiv.org/abs/1910.03561), arXiv preprint arXiv:1910.03561

### Requirements
Our code is designed to run on GPU using [PyTorch](https://pytorch.org/) framework, while scattering transforms are computed using the [Kymatio software package](https://github.com/kymatio/kymatio)
which supports torch and scikit-cuda ('skcuda') backends (skcuda being faster).
In order to run our experiments you will need the following packages: 
- For PyTorch: torch, torchvision, tensorboard
- For skcuda: scikit-cuda, cupy

and a multi-GPU version of Kymatio.

You can install the PyTorch and skcuda packages by:

`pip install torch torchvision tensorboard scikit-cuda cupy`

For the multi-GPU version of Kymatio, until the release of Kymatio v0.2 (planned next spring) which shall support 
this feature, you can use the _multigpu_ branch of the Kymatio fork https://github.com/j-zarka/kymatio.

To install this branch:

```
git clone -b multigpu https://github.com/j-zarka/kymatio.git
cd kymatio
pip install -r requirements.txt
pip install .
```

Multi-GPU scattering transform computation may occasionally lead to a [segmentation fault due to torch.fft](https://github.com/pytorch/pytorch/issues/24176),
especially when numerous GPUs are used. If this is the case, just relaunch the script as those issues shall seldom occur.

### Phase scattering
The phase scattering described in section 2. of the paper is implemented 
in a separate torch module phase_scattering2d_torch (possible backends 'torch' or 'torch_skcuda').

### ImageNet
Download ImageNet dataset from http://www.image-net.org/challenges/LSVRC/2012/downloads (registration required).
Then move validation images to labeled subfolders, using [the PyTorch shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

Model can optionally be trained on a subset of ImageNet classes using the --nb-classes option (with --nb-classes < 1000).
In this case, selected classes' indices are either randomly chosen or determined in a file whose path is specified using the --class-indices
option. Examples of such file for 10, 20 and 100 classes are provided in the utils_sampling folder.

### Setup
Results in the last version of the paper were produced using torch 1.4.0, torchvision 0.5.0 and cuda 10.1
 
### Usage
To train a model, run main.py with the desired model architecture, the below options and the path to the ImageNet dataset.

To reproduce the paper's experiments, run the following commands:

- For classification on Scat. + ISTC α with W = D:

```
python main.py -a sparsescatnet -p 100 --scattering-J 4 --scattering-order2 --scattering-wph --L-kernel-size 3 --dictionary-size 2048
--L-proj-size 256 --epochs 160 --learning-rate-adjust-frequency 60 --lr 0.01 --lambda-star 0.05 --grad-lambda-star
--lambda-star-lb 0.05 --l0-inf-init --epsilon-lambda-0 1 --logdir path/to/training_dir --n-iterations 12 --backend torch_skcuda 
--classifier mlp --nb-l-mlp 2 --dropout-p-mlp 0.3 --nb-hidden-units 4096 --avg-ker-size 5 --savedir path/to/checkpoint_dir 
-j 10  path/to/ImageNet
```

- For classification on Scat. + ISTC α with flexible W, simply change the arch option from sparsescatnet to sparsescatnetw,
for classification on Scat + ISTC Dα with  W = D, simply add the --output-rec option, and finally for Scat + ISTC α with W = D
and for a linear classifier, simply change --classifier option from 'mlp' to 'fc' and remove mlp-linked options (nb-l-mlp,
dropout-p-mlp, nb-hidden-units).

For more details, please see below usage.

```
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e]
               [--seed SEED]
               [--learning-rate-adjust-frequency LEARNING_RATE_ADJUST_FREQUENCY]
               [--logdir LOGDIR] [--savedir SAVEDIR] [--nb-classes NB_CLASSES]
               [--class-indices CLASS_INDICES] [--scattering-order2]
               [--scattering-wph] [--scat-angles SCAT_ANGLES]
               [--backend BACKEND] [--scattering-nphases SCATTERING_NPHASES]
               [--scattering-J SCATTERING_J] [--L-proj-size L_PROJ_SIZE]
               [--L-kernel-size L_KERNEL_SIZE] [--n-iterations N_ITERATIONS]
               [--dictionary-size DICTIONARY_SIZE] [--output-rec]
               [--lambda-0 LAMBDA_0] [--lambda-star LAMBDA_STAR]
               [--lambda-star-lb LAMBDA_STAR_LB]
               [--epsilon-lambda-0 EPSILON_LAMBDA_0] [--l0-inf-init]
               [--grad-lambda-star] [--avg-ker-size AVG_KER_SIZE]
               [--classifier-type CLASSIFIER_TYPE]
               [--nb-hidden-units NB_HIDDEN_UNITS]
               [--dropout-p-mlp DROPOUT_P_MLP] [--nb-l-mlp NB_L_MLP]
               DIR

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: sparsescatnet | sparsescatnetw |
                        scatnet (default: sparsescatnet)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --seed SEED           seed for initializing training.
  --learning-rate-adjust-frequency LEARNING_RATE_ADJUST_FREQUENCY
                        number of epoch after which learning rate is decayed
                        by 10 (default: 30)
  --logdir LOGDIR       directory for the training logs
  --savedir SAVEDIR     directory to save checkpoints
  --nb-classes NB_CLASSES
                        number of classes randomly chosen used for training
                        and validation (default: 1000 = whole train/val
                        dataset)
  --class-indices CLASS_INDICES
                        numpy array of indices used in case nb-classes < 1000
  --scattering-order2   Compute order 2 scattering coefficients
  --scattering-wph      Use phase scattering
  --scat-angles SCAT_ANGLES
                        number of orientations for scattering
  --backend BACKEND     scattering backend
  --scattering-nphases SCATTERING_NPHASES
                        number of phases in the first order of the phase
                        harmonic scattering transform
  --scattering-J SCATTERING_J
                        maximum scale for the scattering transform
  --L-proj-size L_PROJ_SIZE
                        dimension of the linear projection
  --L-kernel-size L_KERNEL_SIZE
                        kernel size of L
  --n-iterations N_ITERATIONS
                        number of iterations for ISTC
  --dictionary-size DICTIONARY_SIZE
                        size of the sparse coding dictionary
  --output-rec          output reconstruction
  --lambda-0 LAMBDA_0   lambda_0
  --lambda-star LAMBDA_STAR
                        lambda_star
  --lambda-star-lb LAMBDA_STAR_LB
                        lambda_star lower bound
  --epsilon-lambda-0 EPSILON_LAMBDA_0
                        epsilon for lambda_0 adjustment
  --l0-inf-init         initialization of lambda_0 as W^Tx inf
  --grad-lambda-star    gradient on lambda_star
  --avg-ker-size AVG_KER_SIZE
                        size of averaging kernel
  --classifier-type CLASSIFIER_TYPE
                        classifier type
  --nb-hidden-units NB_HIDDEN_UNITS
                        number of hidden units for mlp classifier
  --dropout-p-mlp DROPOUT_P_MLP
                        dropout probability in mlp
  --nb-l-mlp NB_L_MLP   number of hidden layers in mlp
  ```

 