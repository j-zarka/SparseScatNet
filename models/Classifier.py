import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, n_space, nb_channels_in, classifier_type='mlp', nb_classes=1000,
                 nb_hidden_units=2048, nb_l_mlp=2, dropout_p_mlp=0.3, avg_ker_size=1):
        super(Classifier, self).__init__()

        assert (classifier_type in ['fc', 'mlp'])

        self.bn = nn.BatchNorm2d(nb_channels_in)

        self.avg_ker_size = avg_ker_size
        if self.avg_ker_size > 1:
            n = n_space - avg_ker_size + 1
        else:
            n = n_space

        in_planes = nb_channels_in * (n ** 2)

        if classifier_type == 'mlp':
            classif_modules = [nn.Linear(in_planes, nb_hidden_units)]

            for i in range(nb_l_mlp-1):
                classif_modules.append(nn.ReLU(inplace=True))
                classif_modules.append(nn.Dropout(p=dropout_p_mlp))
                classif_modules.append(nn.Linear(nb_hidden_units, nb_hidden_units))

            classif_modules.append(nn.ReLU(inplace=True))
            classif_modules.append(nn.Dropout(p=dropout_p_mlp))
            classif_modules.append(nn.Linear(nb_hidden_units, nb_classes))

            self.classifier = nn.Sequential(*classif_modules)

        elif classifier_type == 'fc':
            self.classifier = nn.Linear(in_planes, nb_classes)

    def forward(self, x):
        x = self.bn(x)
        if self.avg_ker_size > 1:
            x = nn.functional.avg_pool2d(x, self.avg_ker_size, stride=1)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output
