import torch
import torch.nn as nn


def print_and_write(log, *logfiles):
    print(log)
    for logfile in logfiles:
        with open(logfile, 'a') as f:
            f.write(log + '\n')


def compute_batch_mean_var(output, var=False):
    n_channels = output.size(1)
    output = output.transpose(1, -1).contiguous().view(-1, n_channels)
    samples_size = output.size(0)

    batch_mean = torch.mean(output, 0).cpu()
    if var:
        batch_var = torch.var(output, 0).cpu()

    if var:
        return batch_mean, batch_var, samples_size
    else:
        return batch_mean, samples_size


def compute_stding_matrix(data_loader, transformation, logfile, print_freq=100):
    # Compute a convolutional standardization after a transform

    # Put transform on GPU
    transformation = torch.nn.DataParallel(transformation).cuda()

    with torch.no_grad():
        print_and_write('Computing mean and standardization matrix...', logfile)
        print_and_write('Starting by computing mean and std dev...', logfile)
        mean_var_meter = AverageVarMeter()
        for i, (input, _) in enumerate(data_loader):
            input = input.cuda()
            if transformation is not None:
                input = transformation(input)
            if len(input.size()) == 5:
                B, nc1, nc2, H, W = input.size()
                input = input.view(B, nc1 * nc2, H, W).contiguous()

            batch_mean, batch_var, samples_size = compute_batch_mean_var(input, var=True)
            mean_var_meter.update(batch_mean, batch_var, samples_size)

            if i % print_freq == 0:
                print_and_write('batch: [{0}/{1}]'.format(i, len(data_loader)), logfile)

        mean = mean_var_meter.avg
        std = torch.sqrt(mean_var_meter.var)

        print_and_write('Creating stding matrix...', logfile)
        W = torch.diag(std ** (-1)).unsqueeze(-1).unsqueeze(-1)
        mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    return mean, W, std


class AverageVarMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.var = 0
        self.sum_avg = 0
        self.sum_var = 0
        self.count = 0

    def update(self, avg, var=0, n=1):
        self.sum_avg += avg * n
        self.sum_var += var * n
        product = self.count*n
        self.count += n
        self.var = self.sum_var / self.count + (product/self.count**2)*(self.avg-avg)**2
        self.avg = self.sum_avg / self.count
