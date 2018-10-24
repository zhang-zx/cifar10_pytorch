import torch
import torch.nn as nn

NUM_CLASS = 10

config = {
    'VGG_11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG_13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG_16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG_19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, VGG_name):
        super(VGG, self).__init__()
        self.config = config[VGG_name]
        self.in_channels = 3
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=NUM_CLASS),
        )
        self._initialize_weights()
        pass

    def _make_layers(self):
        layers = list()
        for conf in self.config:
            if conf == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels=self.in_channels, out_channels=conf, kernel_size=3, padding=1, bias=True))
                layers.append(nn.BatchNorm2d(num_features=conf))
                layers.append(nn.ReLU(inplace=True))
                self.in_channels = conf
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def VGG_11():
    return VGG('VGG_11')

def VGG_13():
    return VGG('VGG_13')

def VGG_16():
    return VGG('VGG_16')

def VGG_19():
    return VGG('VGG_19')

def test():
    net = VGG('VGG_11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()