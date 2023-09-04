from torch import nn


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups, bias=False),
        nn.BatchNorm1d(out_planes),
        nn.ReLU()
    )


class OneDCNN(nn.Module):
    def __init__(self, model_arch=[64, 64, 128, 128, 256]):
        super(OneDCNN, self).__init__()
        self.feature = self._make_layer(model_arch)
        self.fc = nn.Linear(model_arch[-1], 1)

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _make_layer(self, model_arch=[64, 64, 128, 128, 256]):
        layers = []
        in_channels = 1
        flag_stride = True
        arch_temp = model_arch[0]
        for arch in model_arch:
            if flag_stride and arch > arch_temp:
                stride = 2
                flag_stride = False
            else:
                stride = 1
            arch_temp = arch

            layers += [conv_block(in_channels, arch, stride=stride)]
            in_channels = arch

        layers += [nn.AdaptiveAvgPool1d(1)]

        return nn.Sequential(*layers)
