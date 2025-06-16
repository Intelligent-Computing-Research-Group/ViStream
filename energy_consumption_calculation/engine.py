'''
Copyright (C) 2022 Guangyao Chen. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
from functools import partial

import numpy as np
import time
import torch
import torch.nn as nn
import cv2
try:
    from spikingjelly.clock_driven import surrogate, neuron, functional
except:
    from spikingjelly.activation_based import surrogate, neuron, functional

from .ops import CUSTOM_MODULES_MAPPING, MODULES_MAPPING,IFNeuron
from .utils import syops_to_string, params_to_string

from timm.utils import *
from timm.utils.metrics import *  # AverageMeter, accuracy
import os
# sys.path.insert(0,"/home/kang_you/SpikeZIP_transformer/")
from model.spike_quan_wrapper import SNNWrapper,open_dropout,MyQuan

def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img

def visualize(img, 
              bboxes, 
              scores, 
              cls_inds, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.8
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(cls_inds[i])
            if dataset_name == 'coco':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img

def get_syops_pytorch(args, model, device, 
                      dataset, transform, 
                      vis_thresh,
                      print_per_layer_stat=True,
                      ost=sys.stdout,
                      verbose=False, ignore_modules=[],
                      custom_modules_hooks={},
                      output_precision=3,
                      syops_units='GMac',
                      param_units='M', 
                      class_colors=None, 
                      class_names=None, 
                      class_indexs=None, 
                      dataset_name='voc',
                      mode = "QANN"):
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    syops_model = add_syops_counting_methods(model)  # dir(syops_model)
    syops_model.eval()
    open_dropout(model)
    syops_model.start_syops_count(ost=ost, verbose=verbose,
                                ignore_list=ignore_modules)

    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, f"{args.version}+{args.mode}")

    for index in range(num_images):
        print("index",index)
        # if index > 0:
        #     break
        if hasattr(syops_model,"reset"):
            syops_model.reset()
        
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)
        h, w, _ = image.shape
        scale = np.array([[w, h, w, h]])

        # to tensor
        x = torch.from_numpy(transform(image)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        if args.double:
            x = x.double()
        bboxesList, scoresList, cls_indsList = syops_model(x,verbose=True)
        print("detection time used ", time.time() - t0, "s")
        
        for t in range(len(bboxesList)):
            # rescale
            bboxesList[t] *= scale

            # vis detection
            img_processed = visualize(
                                img=image.copy(),
                                bboxes=bboxesList[t],
                                scores=scoresList[t],
                                cls_inds=cls_indsList[t],
                                vis_thresh=vis_thresh,
                                class_colors=class_colors,
                                class_names=class_names,
                                class_indexs=class_indexs,
                                dataset_name=dataset_name
                                )
            if args.show:
                cv2.imshow('detection', img_processed)
                cv2.waitKey(0)
            # save result
            cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +f"_{t}_"+mode+'.jpg'), img_processed)
    
    syops_count, params_count = syops_model.compute_average_syops_cost()  # 整个网络的操作数加和除以累积的batchsize、总的参数和。未对网络中子模块操作。


    if print_per_layer_stat:
        print_model_with_syops(
            syops_model,
            syops_count,
            params_count,
            ost=ost,
            syops_units=syops_units,
            param_units=param_units,
            precision=output_precision
        )
    syops_model.stop_syops_count()
    CUSTOM_MODULES_MAPPING = {}

    return syops_count, params_count, syops_model


def accumulate_syops(self):  # 如果本module在MODULES_MAPPING或CUSTOM_MODULES_MAPPING中，则直接输出记录的__syops__，否则将其子module的__syops__累积加和，即该函数只返回self.__syops__（如果有子模块，则是累积和）
    if is_supported_instance(self):
        return self.__syops__
    else:
        sum = np.array([0.0, 0.0, 0.0, 0.0])
        for m in self.children():
            sum += m.accumulate_syops()  #循环递归调用，对整个网络结构进行计算
        return sum


def print_model_with_syops(model, total_syops, total_params, syops_units='GMac',
                           param_units='M', precision=3, ost=sys.stdout):

    for i in range(3):
        if total_syops[i] < 1:
            total_syops[i] = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(self):  # 如果本module在MODULES_MAPPING或CUSTOM_MODULES_MAPPING中，则直接输出记录的__params__，否则将其子module的__params__累积加和，即该函数只返回self.__params__（如果有子模块，则是累积和）
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()  #循环递归调用，对整个网络结构进行计算
            return sum

    def syops_repr(self):
        accumulated_params_num = self.accumulate_params()  # 返回self.__params__（如果有子模块，则是累积和）
        accumulated_syops_cost = self.accumulate_syops()  # 返回self.__syops__（如果有子模块，则是累积和）
        # print("model.__batch_counter__",model.__batch_counter__,"model.__times_counter__",model.__times_counter__)
        accumulated_syops_cost[0] /= (model.__batch_counter__)  # 取均值
        accumulated_syops_cost[1] /= (model.__batch_counter__)
        accumulated_syops_cost[2] /= (model.__batch_counter__)
        accumulated_syops_cost[3] /= model.__times_counter__  # 取均值
        # store info for later analysis
        
        self.accumulated_params_num = accumulated_params_num
        self.accumulated_syops_cost = accumulated_syops_cost
        return ', '.join([self.original_extra_repr(),
                          params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          syops_to_string(accumulated_syops_cost[0],
                                          units=syops_units, precision=precision),
                          '{:.3%} oriMACs'.format(accumulated_syops_cost[0] / total_syops[0]),
                          syops_to_string(accumulated_syops_cost[1],
                                          units=syops_units, precision=precision),
                          '{:.3%} ACs'.format(accumulated_syops_cost[1] / total_syops[1]),
                          syops_to_string(accumulated_syops_cost[2],
                                          units=syops_units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_syops_cost[2] / total_syops[2]),
                          '{:.3%} Spike Rate'.format(accumulated_syops_cost[3] / 100.),
                          'SpkStat: {}'.format(self.__spkhistc__)])  # print self.__spkhistc__
                          #self.original_extra_repr()])
        # return ', '.join([params_to_string(accumulated_params_num,
        #                                    units=param_units, precision=precision),
        #                   '{:.3%} Params'.format(accumulated_params_num / total_params),
        #                   syops_to_string(accumulated_syops_cost[1],
        #                                   units=syops_units, precision=precision),
        #                   '{:.3%} ACs'.format(accumulated_syops_cost[1] / total_syops[1]),
        #                   syops_to_string(accumulated_syops_cost[2],
        #                                   units=syops_units, precision=precision),
        #                   '{:.3%} MACs'.format(accumulated_syops_cost[2] / total_syops[2]),
        #                   '{:.3%} Spike Rate'.format(accumulated_syops_cost[3] / 100.)])
        #                   #self.original_extra_repr()])
    
    def syops_repr_empty(self):
        return ''

    def add_extra_repr(m):
        m.accumulate_syops = accumulate_syops.__get__(m)  # 为module增加属性accumulate_syops
        m.accumulate_params = accumulate_params.__get__(m)
        if is_supported_instance(m):
            syops_extra_repr = syops_repr.__get__(m)
            print(m,is_supported_instance(m))
        else:
            syops_extra_repr = syops_repr_empty.__get__(m)
        if m.extra_repr != syops_extra_repr:
            m.original_extra_repr = m.extra_repr  # backup原先的extra_repr
            m.extra_repr = syops_extra_repr  # 将syops_extra_repr作为extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)  # 输出整个网络，且在输出过程中计算各子模块的__syops__、__params__
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_syops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_syops_count = start_syops_count.__get__(net_main_module)
    net_main_module.stop_syops_count = stop_syops_count.__get__(net_main_module)
    net_main_module.reset_syops_count = reset_syops_count.__get__(net_main_module)
    net_main_module.compute_average_syops_cost = compute_average_syops_cost.__get__(
                                                    net_main_module)

    net_main_module.reset_syops_count()  # 此处通过调用add_syops_counter_variable_or_reset(module)对各子module计算module.__params__，后面无需重新计算

    return net_main_module


def compute_average_syops_cost(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Returns current mean syops consumption per image.

    """

    for m in self.modules():  # 递归式的为每个子模块加上属性accumulate_syops
        m.accumulate_syops = accumulate_syops.__get__(m)

    syops_sum = self.accumulate_syops()  # 整个网络的操作数加和。未对网络中子模块操作。self.accumulate_syops()只返回self的__syops__
    syops_sum = np.array([item / self.__batch_counter__ for item in syops_sum])

    for m in self.modules():
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    params_sum = get_model_parameters_number(self)  # 整个网络的参数加和
    return syops_sum, params_sum


def start_syops_count(self, **kwargs):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean syops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_syops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__syops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__syops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_syops_counter_hook_function, **kwargs))


def stop_syops_count(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Stops computing the mean syops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_syops_counter_hook_function)
    # self.apply(remove_syops_counter_variables)  # keep this for later analyses


def reset_syops_count(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_syops_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size
    module.__times_counter__ += 32


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0
    module.__times_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_syops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__') or hasattr(module, '__params__'):
            print('Warning: variables __syops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' syops can affect your code!')
            module.__syops_backup_syops__ = module.__syops__
            module.__syops_backup_params__ = module.__params__
        module.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
        module.__params__ = get_model_parameters_number(module)
        # add __spkhistc__ for each module (by yult 2023.4.18)
        module.__spkhistc__ = None #np.zeros(20)  # assuming there are no more than 20 spikes for one neuron


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_syops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops_handle__'):
            module.__syops_handle__.remove()
            del module.__syops_handle__


def remove_syops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__'):
            del module.__syops__
            if hasattr(module, '__syops_backup_syops__'):
                module.__syops__ = module.__syops_backup_syops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__syops_backup_params__'):
                module.__params__ = module.__syops_backup_params__
        # remove module.__spkhistc__ after print
        if hasattr(module, '__spkhistc__'):
            del module.__spkhistc__
