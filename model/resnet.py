import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision.models.resnet as torch_resnet
from typing import Optional, Callable
from torchvision.models.resnet import conv3x3, conv1x1
from torch import Tensor
import sys

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class ResNet(torch_resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

    def modify(self, remove_layers=[], padding=''):
        # Set stride of layer3 and layer 4 to 1 (from 2)
        filter_layers = lambda x: [l for l in x if getattr(self, l) is not None]
        for layer in filter_layers(['layer3', 'layer4']):
            for m in getattr(self, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = tuple(1 for _ in m.stride)
        # Set padding (zeros or reflect, doesn't change much; 
        # zeros requires lower temperature)
        if padding != '' and padding != 'no':
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) and sum(m.padding) > 0:
                    m.padding_mode = padding
        elif padding == 'no':
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) and sum(m.padding) > 0:
                    m.padding = (0,0)

        # Remove extraneous layers
        remove_layers += ['fc', 'avgpool']
        for layer in filter_layers(remove_layers):
            setattr(self, layer, None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x if self.maxpool is None else self.maxpool(x) 

        x = self.layer1(x)
        x = F.avg_pool2d(x,(2,2)) if self.layer2 is None else self.layer2(x)
        x = x if self.layer3 is None else self.layer3(x) 
        x = x if self.layer4 is None else self.layer4(x) 
    
        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs) -> ResNet:
    from energy_consumption_calculation.engine import add_syops_counting_methods, print_model_with_syops
    from energy_consumption_calculation.flops_counter import get_energy_cost
    dataLen = kwargs.pop('dataLen', None)
    record_inout = kwargs.pop('record_inout', None)
    count = 0
    def custom_forward_hook(module, input, output):
        nonlocal dataLen,count
        # print("count",count,"dataLen",dataLen)
        if count == dataLen - 1:
            ssa_info = {'depth': 12, 'Nheads': 6, 'embSize': 384, 'patchSize': 16, 'Tsteps': 32}  # small
            syops_count, params_count = module.compute_average_syops_cost()
            print_model_with_syops(module,syops_count,params_count,ost=sys.stdout,syops_units='GMac',param_units="M",precision=4)
            module.stop_syops_count()
            get_energy_cost(module,ssa_info)
            count = count + 1
        else:
            count = count + 1
    
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

    if record_inout:
        model = add_syops_counting_methods(model)
        model.start_syops_count(ost=sys.stdout, verbose=False,
                                ignore_list=[])
        model.eval()
        model.register_forward_hook(custom_forward_hook)    

    return model
    
def spikeresnet50(pretrained=False, progress=True, **kwargs) -> ResNet:
    # need args: QANNPath, level, weight_quantization_bit, time_step, encoding_type, record_inout, log_dir
    from .spike_quan_wrapper import myquan_replace_resnet,SNNWrapper
    # from energy_consumption_calculation.engine import add_syops_counting_methods, print_model_with_syops
    # from energy_consumption_calculation.flops_counter import get_energy_cost
        
    # get the args and remove them from kwargs
    QANNPath = kwargs.pop('QANNPath', None)
    level = kwargs.pop('level', None)
    weight_quantization_bit = kwargs.pop('weight_quantization_bit', None)
    time_step = kwargs.pop('time_step', None)
    encoding_type = kwargs.pop('encoding_type', None)
    record_inout = kwargs.pop('record_inout', None)
    log_dir = kwargs.pop('log_dir', None)
    dataLen = kwargs.pop('dataLen', None)
    count = 0

    # def custom_forward_hook(module, input, output):
    #     nonlocal dataLen,count
    #     # print("count",count,"dataLen",dataLen)
    #     if count == dataLen - 1:
    #         ssa_info = {'depth': 12, 'Nheads': 6, 'embSize': 384, 'patchSize': 16, 'Tsteps': 32}  # small
    #         syops_count, params_count = module.compute_average_syops_cost()
    #         print_model_with_syops(module,syops_count,params_count,ost=sys.stdout,syops_units='GMac',param_units="M",precision=4)
    #         module.stop_syops_count()
    #         get_energy_cost(module,ssa_info)
    #         count = count + 1
    #     else:
    #         count = count + 1
    
    # define ANN
    ANNresnet50 =  _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

    # convert ANN to QANN
    myquan_replace_resnet(ANNresnet50,level=level,weight_bit=weight_quantization_bit)
    
    # load QANN model
    checkpoint = torch.load(QANNPath, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % QANNPath)
    checkpoint_model = checkpoint['model']
    msg = ANNresnet50.load_state_dict(checkpoint_model, strict=True)
    print(msg)

    # # convert QANN to SNN
    SNNresnet50 = SNNWrapper(ann_model=ANNresnet50, cfg=None, time_step=time_step, \
                           Encoding_type=encoding_type, level=level, neuron_type="ST-BIF", \
                           model_name="SpikeResNet50", is_softmax = False, \
                           record_inout=False, record_dir=log_dir+f"/output_bin_snn_AppearanceModel_w{weight_quantization_bit}_a{int(torch.log2(torch.tensor(level+0.0)))}_T{time_step}/")
    
    # if record_inout:
    #     SNNresnet50 = add_syops_counting_methods(SNNresnet50)
    #     SNNresnet50.start_syops_count(ost=sys.stdout, verbose=False,
    #                             ignore_list=[])
    #     SNNresnet50.eval()
    #     SNNresnet50.register_forward_hook(custom_forward_hook)
    # # print(SNNresnet50)
    return SNNresnet50


def spikeresnet50IF(pretrained=False, progress=True, **kwargs) -> ResNet:
    # need args: QANNPath, level, weight_quantization_bit, time_step, encoding_type, record_inout, log_dir
    from .spike_quan_wrapper import myquan_replace_resnet,SNNWrapperIF
    from energy_consumption_calculation.engine import add_syops_counting_methods, print_model_with_syops
    from energy_consumption_calculation.flops_counter import get_energy_cost
        
    # get the args and remove them from kwargs
    QANNPath = kwargs.pop('QANNPath', None)
    level = kwargs.pop('level', None)
    weight_quantization_bit = kwargs.pop('weight_quantization_bit', None)
    time_step = kwargs.pop('time_step', None)
    encoding_type = kwargs.pop('encoding_type', None)
    record_inout = kwargs.pop('record_inout', None)
    log_dir = kwargs.pop('log_dir', None)
    dataLen = kwargs.pop('dataLen', None)
    count = 0

    def custom_forward_hook(module, input, output):
        nonlocal dataLen,count
        # print("count",count,"dataLen",dataLen)
        if count == dataLen - 1:
            ssa_info = {'depth': 12, 'Nheads': 6, 'embSize': 384, 'patchSize': 16, 'Tsteps': 32}  # small
            syops_count, params_count = module.compute_average_syops_cost()
            print_model_with_syops(module,syops_count,params_count,ost=sys.stdout,syops_units='GMac',param_units="M",precision=4)
            module.stop_syops_count()
            get_energy_cost(module,ssa_info)
            count = count + 1
        else:
            count = count + 1
    
    # define ANN
    ANNresnet50 =  _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

    # convert ANN to QANN
    myquan_replace_resnet(ANNresnet50,level=level,weight_bit=weight_quantization_bit)
    
    # load QANN model
    checkpoint = torch.load(QANNPath, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % QANNPath)
    checkpoint_model = checkpoint['model']
    msg = ANNresnet50.load_state_dict(checkpoint_model, strict=True)
    print(msg)

    # convert QANN to SNN
    SNNresnet50 = SNNWrapperIF(ann_model=ANNresnet50, cfg=None, time_step=time_step, \
                           Encoding_type=encoding_type, level=level, neuron_type="ST-BIF", \
                           model_name="SpikeResNet50", is_softmax = False, \
                           record_inout=False, record_dir=log_dir+f"/output_bin_snn_AppearanceModel_w{weight_quantization_bit}_a{int(torch.log2(torch.tensor(level+0.0)))}_T{time_step}/")
    
    if record_inout:
        SNNresnet50 = add_syops_counting_methods(SNNresnet50)
        SNNresnet50.start_syops_count(ost=sys.stdout, verbose=False,
                                ignore_list=[])
        SNNresnet50.eval()
        SNNresnet50.register_forward_hook(custom_forward_hook)
    # print(SNNresnet50)
    return SNNresnet50


def spikeresnet50LCC(pretrained=False, progress=True, **kwargs) -> ResNet:
    # need args: QANNPath, level, weight_quantization_bit, time_step, encoding_type, record_inout, log_dir
    from .spike_quan_wrapper import myquan_replace_resnet,SNNWrapperLCC
    # from energy_consumption_calculation.engine import add_syops_counting_methods, print_model_with_syops
    # from energy_consumption_calculation.flops_counter import get_energy_cost
        
    # get the args and remove them from kwargs
    QANNPath = kwargs.pop('QANNPath', None)
    level = kwargs.pop('level', None)
    weight_quantization_bit = kwargs.pop('weight_quantization_bit', None)
    time_step = kwargs.pop('time_step', None)
    encoding_type = kwargs.pop('encoding_type', None)
    record_inout = kwargs.pop('record_inout', None)
    log_dir = kwargs.pop('log_dir', None)
    dataLen = kwargs.pop('dataLen', None)
    count = 0    
    
    # def custom_forward_hook(module, input, output):
    #     nonlocal dataLen,count
    #     if count == dataLen - 1:
    #         ssa_info = {'depth': 12, 'Nheads': 6, 'embSize': 384, 'patchSize': 16, 'Tsteps': 32}  # small
    #         syops_count, params_count = module.compute_average_syops_cost()
    #         print_model_with_syops(module,syops_count,params_count,ost=sys.stdout,syops_units='GMac',param_units="M",precision=4)
    #         module.stop_syops_count()
    #         get_energy_cost(module,ssa_info)
    #         count = count + 1
    #     else:
    #         count = count + 1

    # define ANN
    ANNresnet50 = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

    # convert ANN to QANN
    myquan_replace_resnet(ANNresnet50,level=level,weight_bit=weight_quantization_bit)
    
    # load QANN model
    checkpoint = torch.load(QANNPath, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % QANNPath)
    checkpoint_model = checkpoint['model']
    msg = ANNresnet50.load_state_dict(checkpoint_model, strict=True)
    print(msg)

    # convert QANN to SNN
    SNNresnet50 = SNNWrapperLCC(ann_model=ANNresnet50, cfg=None, time_step=time_step, \
                           Encoding_type=encoding_type, level=level, neuron_type="ST-BIF", \
                           model_name="SpikeResNet50", is_softmax = False, \
                           record_inout=False, record_dir=log_dir+f"/output_bin_snn_AppearanceModel_w{weight_quantization_bit}_a{int(torch.log2(torch.tensor(level+0.0)))}_T{time_step}/")
    
    # if record_inout:
        # SNNresnet50 = add_syops_counting_methods(SNNresnet50)
        # SNNresnet50.start_syops_count(ost=sys.stdout, verbose=False,
        #                         ignore_list=[])
        # SNNresnet50.eval()
        # SNNresnet50.register_forward_hook(custom_forward_hook)
    return SNNresnet50


def resnet101(pretrained=False, progress=True, **kwargs): 
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
