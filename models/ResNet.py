import torch.nn as nn
import math
NUM_CLASSES = 10

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride = 1, down_sample = None):
        super(BasicBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.BatchNorm2d(num_features=channels)
        )
        self.down_sample = down_sample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.residual(x)
        if self.down_sample:
            residual = self.down_sample(residual)
        out += residual
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride = 1, down_sample = None):
        super(Bottleneck, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels*self.expansion, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=channels*self.expansion)
        )

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.residual(x)
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.__make_layers(block, 64, layers[0], stride=1)
        self.layer2 = self.__make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self.__make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self.__make_layers(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512 * block.expansion, NUM_CLASSES)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def __make_layers(self, block, channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:

            downsample = nn.Sequential(
                # nn.AvgPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=self.in_channels, out_channels=channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=channels * block.expansion)
            )
        layers = list()
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)