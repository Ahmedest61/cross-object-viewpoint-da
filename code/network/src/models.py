"""
BUILT OFF OF RESNET
"""

from torch.autograd import Function
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ViewpointNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ViewpointNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_azi = nn.Linear(num_classes, 360)
        self.fc_ele = nn.Linear(num_classes, 360)

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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        azimuth = self.fc_azi(x)
        elevation = self.fc_ele(x)

        return azimuth, elevation

class GradReverse(Function):

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -1)

class VCDNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(VCDNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes) # intermediate embedding

        # Viewpoint classifier
        self.fc_viewpoint = nn.Linear(num_classes, 64)
        #self.fc2_viewpoint = nn.Linear(num_classes, num_classes)
        self.fc_azi = nn.Linear(64, 360)
        self.fc_ele = nn.Linear(64, 360)

        # Object classifier
        self.grad_r_class = GradReverse()
        self.fc_class = nn.Linear(num_classes, num_classes)
        self.fc2_class = nn.Linear(num_classes, 12)

        # Domain classifier
        self.grad_r_domain = GradReverse()
        self.fc_domain = nn.Linear(num_classes, num_classes)
        self.fc2_domain = nn.Linear(num_classes, 1)

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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Viewpoint
        x_viewpoint = self.fc_viewpoint(x)
        #x_viewpoint = self.fc2_viewpoint(x_viewpoint)
        azimuth = self.fc_azi(x_viewpoint)
        elevation = self.fc_ele(x_viewpoint)
        
        # Object class
        x_class = self.grad_r_class(x)
        x_class = self.fc_class(x_class)
        class_ = self.fc2_class(x_class)

        # Domain
        x_domain = self.grad_r_domain(x)
        x_domain = self.fc_domain(x_domain)
        x_domain = self.fc2_domain(x_domain)
        domain = nn.functional.sigmoid(x_domain)

        return azimuth, elevation, class_, domain

class SimpleNetwork(nn.Module):

    def __init__(self, bottleneck_size, block, layers, num_classes=1000):
        self.inplanes = 64
        super(SimpleNetwork, self).__init__()
        self.hook_store = []

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Intermediate embedding generation
        if bottleneck_size == 0:
            embedding_size = num_classes
        else:
            embedding_size = bottleneck_size
        self.fc_embedding = nn.Linear(num_classes, embedding_size)
        self.fc_embedding.register_forward_hook(self.store_embedding)
        self.fc2 = nn.Linear(embedding_size, num_classes)

        # Output
        self.fc_azi = nn.Linear(num_classes, 360)
        self.fc_ele = nn.Linear(num_classes, 360)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def store_embedding(self, _, input, output):
        self.hook_store.append(output)

    def get_embedding(self):
        out = self.hook_store.pop(0)
        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        x = self.relu(self.fc_embedding(x))
        x = self.fc2(x)

        azimuth = self.fc_azi(x)
        elevation = self.fc_ele(x)

        return azimuth, elevation


def viewpoint_net(layers=18, pretrained=False, **kwargs):
    if layers == 18:
        layer_struct = [2,2,2,2]
        component = BasicBlock
    elif layers == 34:
        layer_struct = [3,4,6,3]
        component = BasicBlock
    elif layers == 50:
        layer_struct = [3,4,6,3]
        component = Bottleneck
    elif layers == 101:
        layer_struct = [3,4,23,3]
        component = Bottleneck
    elif layers == 152:
        layer_struct = [3,8,36,3]
        component = Bottleneck

    model = ViewpointNet(component, layer_struct, **kwargs)
    if pretrained:
        state = model.state_dict()
        state.update(model_zoo.load_url(model_urls['resnet%i' % layers]))
        model.load_state_dict(state)
    return model

def vcd_net(layers=18, pretrained=False, **kwargs):
    if layers == 18:
        layer_struct = [2,2,2,2]
        component = BasicBlock
    elif layers == 34:
        layer_struct = [3,4,6,3]
        component = BasicBlock
    elif layers == 50:
        layer_struct = [3,4,6,3]
        component = Bottleneck
    elif layers == 101:
        layer_struct = [3,4,23,3]
        component = Bottleneck
    elif layers == 152:
        layer_struct = [3,8,36,3]
        component = Bottleneck

    model = VCDNet(component, layer_struct, **kwargs)
    if pretrained:
        state = model.state_dict()
        state.update(model_zoo.load_url(model_urls['resnet%i' % layers]))
        model.load_state_dict(state)
    return model

def resnet_experiment(bottleneck_size=0, layers=18, pretrained=False, **kwargs):
    if layers == 18:
        layer_struct = [2,2,2,2]
        component = BasicBlock
    elif layers == 34:
        layer_struct = [3,4,6,3]
        component = BasicBlock
    elif layers == 50:
        layer_struct = [3,4,6,3]
        component = Bottleneck
    elif layers == 101:
        layer_struct = [3,4,23,3]
        component = Bottleneck
    elif layers == 152:
        layer_struct = [3,8,36,3]
        component = Bottleneck

    model = SimpleNetwork(bottleneck_size, component, layer_struct, **kwargs)

    if pretrained:
        state = model.state_dict()
        state.update(model_zoo.load_url(model_urls['resnet%i' % layers]))
        model.load_state_dict(state)
    return model
