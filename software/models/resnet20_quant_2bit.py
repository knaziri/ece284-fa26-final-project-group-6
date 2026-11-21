'''
2-bit ResNet-20 for CIFAR-10 (all convs quantized, including the first one).
Based on the 4-bit starter code.
'''
 
import math
import torch.nn as nn
from models.quant_layer_2bit import QuantConv2d
 
BIT = 2
# Initial clipping values (tunable). Starter used 8.0 for activations and 3.0 for weights.
FIRST_ACT_ALPHA = 2.0    # input is the normalized image, about -2 .. 2.1
HIDDEN_ACT_ALPHA = 3.0   # post-BN/ReLU activations; 8.0 is too coarse at 2 bits (step 2.7)
WGT_ALPHA = 1.0          # ternary weights: threshold = alpha/2 std; 3.0 would zero ~87% of weights
 
 
def qconv(cin, cout, k, stride=1, padding=0, act_alpha_init=HIDDEN_ACT_ALPHA):
    return QuantConv2d(cin, cout, kernel_size=k, stride=stride, padding=padding, bias=False,
                       bit=BIT, act_alpha_init=act_alpha_init, wgt_alpha_init=WGT_ALPHA)
 
 
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = qconv(inplanes, planes, 3, stride, 1)
        self.conv2 = qconv(planes, planes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)
 
 
class ResNet_Cifar(nn.Module):
 
    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        # first conv is now a 2-bit QuantConv2d (was nn.Conv2d(3, 16, 3, 1, 1, bias=False))
        self.conv1 = qconv(3, 16, 3, 1, 1, act_alpha_init=FIRST_ACT_ALPHA)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                qconv(self.inplanes, planes * block.expansion, 1, stride, 0),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
 
    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()
 
 
def resnet20_quant_2bit(**kwargs):
    return ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)