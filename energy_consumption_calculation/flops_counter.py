'''
Copyright (C) 2022 Guangyao Chen - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys

import torch.nn as nn

from .engine import get_syops_pytorch
from .utils import syops_to_string, params_to_string
import re
# from .ops import SpikeInferConv2dFuseBN, SpikeInferAvgPool, SpikeResidualAddNew, SpikeInferLinear, STBIFNeuron
from .ops import MyQuan, IFNeuron, LLConv2d, LLLinear

# ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 384, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-384
# ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 512, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-512
# ssa_info = {'depth': 12, 'Nheads': 12, 'embSize': 768, 'patchSize': 16, 'Tsteps': 64}  # base
ssa_info = {'depth': 12, 'Nheads': 6, 'embSize': 384, 'patchSize': 16, 'Tsteps': 15}  # small
# ssa_info = {'depth': 24, 'Nheads': 16, 'embSize': 1024, 'patchSize': 16, 'Tsteps': 32}  # large

def replace_decimal_strings(input_string):
    pattern = r'\.(\d+)'
    
    replaced_string = re.sub(pattern, r'[\1]', input_string)

    return replaced_string

def get_energy_cost(model, ssa_info):
    # calculate energy consumption according to E_mac = 4.6 pJ and E_ac = 0.9 pJ
    print('Calculating energy consumption ...')
    conv_linear_layers_info = []
    Nac = 0
    Nmac = 0
    for name, module in model.named_modules():
        if isinstance(module,nn.Conv2d)  or isinstance(module,nn.Linear)  or isinstance(module,nn.BatchNorm2d) or isinstance(module,IFNeuron): # SpikeZIP: linear in name
            # print(name)
            accumulated_syops_cost = eval(replace_decimal_strings(f'model.{name}.accumulated_syops_cost'))
            if "conv" in name:
                accumulated_syops_cost[3] = accumulated_syops_cost[3]*ssa_info['Tsteps']
            tinfo = (name, module, accumulated_syops_cost)
            conv_linear_layers_info.append(tinfo)
            if abs(accumulated_syops_cost[3] - 100) < 1e-4:  # fr = 100%
                Nmac += accumulated_syops_cost[2]
            else:
                Nac += accumulated_syops_cost[1]
    Nmac = Nmac / 1e9 # G
    Nac = Nac / 1e9 # G
    E_mac = Nmac * 4.6 # mJ
    E_ac = Nac * 0.9 # mJ
    E_all = E_mac + E_ac
    print(f"Number of operations: {Nmac} G MACs, {Nac} G ACs")
    print(f"Energy consumption: {E_all} mJ")
    return


def get_model_complexity_info(args, model, device, dataset, transform, vis_thresh,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}, backend='pytorch',
                              syops_units=None, param_units=None, class_colors=None, class_names=None, class_indexs=None, dataset_name='voc', mode = "QANN",
                              output_precision=2):

    assert isinstance(model, nn.Module)


    if backend == 'pytorch':
        syops_count, params_count, syops_model = get_syops_pytorch(args,model,device,dataset,transform,vis_thresh,
                                                      print_per_layer_stat, ost,
                                                      verbose, ignore_modules,
                                                      custom_modules_hooks,
                                                      output_precision=output_precision,
                                                      syops_units=syops_units,
                                                      param_units=param_units,
                                                      class_colors=class_colors,
                                                      class_names=class_names,
                                                      class_indexs=class_indexs,
                                                      dataset_name=dataset_name,
                                                      mode=mode)
        # calculate energy consumption according to E_mac = 4.6 pJ and E_ac = 0.9 pJ
        get_energy_cost(syops_model, ssa_info)
    else:
        raise ValueError('Wrong backend name')

    if as_strings:
        syops_string = syops_to_string(
            syops_count[0],
            units=syops_units,
            precision=output_precision
        )
        ac_syops_string = syops_to_string(
            syops_count[1],
            units=syops_units,
            precision=output_precision
        )
        mac_syops_string = syops_to_string(
            syops_count[2],
            units=syops_units,
            precision=output_precision
        )
        params_string = params_to_string(
            params_count,
            units=param_units,
            precision=output_precision
        )
        return [syops_string, ac_syops_string, mac_syops_string], params_string

    return syops_count, params_count
