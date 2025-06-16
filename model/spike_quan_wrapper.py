
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .spike_quan_layer import MyQuan,IFNeuron,LLConv2d,LLLinear,QAttention,SAttention,Spiking_LayerNorm,QuanConv2d,QuanLinear,Attention_no_softmax,ORIIFNeuron, save_module_inout, Addition, SpikeMaxPooling, spiking_BatchNorm2d, ORIIFNeuron
import sys
from timm.models.vision_transformer import Attention,Mlp,Block
from copy import deepcopy
import os

def get_subtensors(tensor,mean,std,sample_grain=255,output_num=4):
    for i in range(int(sample_grain)):
        output = (tensor/sample_grain).unsqueeze(0)
        # output = (tensor).unsqueeze(0)
        if i == 0:
            accu = output
        else:
            accu = torch.cat((accu,output),dim=0)
    return accu

def reset_model(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, IFNeuron) or isinstance(child, LLConv2d) or isinstance(child, LLLinear) or isinstance(child, SAttention) or isinstance(child, Spiking_LayerNorm) or isinstance(child, ORIIFNeuron) or isinstance(child, SpikeMaxPooling) or isinstance(child, spiking_BatchNorm2d) or isinstance(child,ORIIFNeuron):
            model._modules[name].reset()
            is_need = True
        if not is_need:
            reset_model(child)

class Judger():
	def __init__(self):
		self.network_finish=True

	def judge_finish(self,model):
		children = list(model.named_children())
		for name, child in children:
			is_need = False
			if isinstance(child, IFNeuron) or isinstance(child, LLLinear) or isinstance(child, LLConv2d) or isinstance(child, ORIIFNeuron):
				self.network_finish = self.network_finish and (not model._modules[name].is_work)
				# print("child",child,"network_finish",self.network_finish,"model._modules[name].is_work",(model._modules[name].is_work))
				is_need = True
			if not is_need:
				self.judge_finish(child)

	def reset_network_finish_flag(self):
		self.network_finish = True

def attn_convert(QAttn:QAttention,SAttn:SAttention,level,neuron_type):
    SAttn.qkv = LLLinear(linear = QAttn.qkv,neuron_type = "ST-BIF",level = level)
    SAttn.proj = LLLinear(linear = QAttn.proj,neuron_type = "ST-BIF",level = level)

    SAttn.q_IF.neuron_type= neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min
    SAttn.q_IF.is_init = False

    SAttn.k_IF.neuron_type= neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min
    SAttn.k_IF.is_init = False

    SAttn.v_IF.neuron_type= neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min
    SAttn.v_IF.is_init = False

    SAttn.attn_IF.neuron_type= neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold.data = QAttn.attn_quan.s.data
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min
    SAttn.attn_IF.is_init = False

    SAttn.attn_softmax_IF.neuron_type= neuron_type
    SAttn.attn_softmax_IF.level = level
    SAttn.attn_softmax_IF.q_threshold.data = QAttn.attn_softmax_quan.s.data
    SAttn.attn_softmax_IF.pos_max = QAttn.attn_softmax_quan.pos_max
    SAttn.attn_softmax_IF.neg_min = QAttn.attn_softmax_quan.neg_min
    SAttn.attn_softmax_IF.is_init = False

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min
    SAttn.after_attn_IF.is_init = False

    SAttn.proj_IF.neuron_type= neuron_type
    SAttn.proj_IF.level = level
    SAttn.proj_IF.q_threshold.data = QAttn.quan_proj.s.data
    SAttn.proj_IF.pos_max = QAttn.quan_proj.pos_max
    SAttn.proj_IF.neg_min = QAttn.quan_proj.neg_min
    SAttn.proj_IF.is_init = False

    SAttn.attn_drop = QAttn.attn_drop
    SAttn.proj_drop = QAttn.proj_drop

def open_dropout(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, nn.Dropout):
            child.train()
            print(child)
            is_need = True
        if not is_need:
            open_dropout(child)



def cal_l1_loss(model):
    l1_loss = 0.0
    def _cal_l1_loss(model):
        nonlocal l1_loss
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, MyQuan):
                l1_loss = l1_loss + child.act_loss
                is_need = True
            if not is_need:
                _cal_l1_loss(child)
    _cal_l1_loss(model)
    return l1_loss

class SNNWrapperIF(nn.Module):
    
    def __init__(self, ann_model, cfg, time_step = 2000,Encoding_type="rate",**kwargs):
        super(SNNWrapperIF, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model
        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_softmax = kwargs["is_softmax"]
        self.record_inout = False
        self.record_dir = kwargs["record_dir"]
        self.max_T = 0
        self.visualize = False
        self.calEnergy = False
        # self.model_reset = None
        if self.model_name.count("vit") > 0:
            self.pos_embed = deepcopy(self.model.pos_embed.data)
            self.cls_token = deepcopy(self.model.cls_token.data)

        self._replace_weight(self.model)
        # self.model_reset = deepcopy(self.model)        
        if self.record_inout:
            self.calOrder = []
            self._record_inout(self.model)
            self.set_snn_save_name(self.model)
            local_rank = torch.distributed.get_rank()
            glo._init()
            if local_rank == 0:
                if not os.path.exists(self.record_dir):
                    os.mkdir(self.record_dir)
                glo.set_value("output_bin_snn_dir",self.record_dir)
                f = open(f"{self.record_dir}/calculationOrder.txt","w+")
                for order in self.calOrder:
                    f.write(order+"\n")
                f.close()
        self.count2 = 0
    
    def hook_mid_feature(self):
        self.feature_list = []
        self.input_feature_list = []
        def _hook_mid_feature(module, input, output):
            self.feature_list.append(output)
            self.input_feature_list.append(input[0])
        self.model.register_forward_hook(_hook_mid_feature)
        # self.model.blocks[11].attn.attn_IF.register_forward_hook(_hook_mid_feature)
    
    def get_mid_feature(self):
        self.feature_list = torch.stack(self.feature_list,dim=0)
        self.input_feature_list = torch.stack(self.input_feature_list,dim=0)
        print("self.feature_list",self.feature_list.shape) 
        print("self.input_feature_list",self.input_feature_list.shape) 
            
    def reset(self):
        # self.model = deepcopy(self.model_reset).cuda()
        if self.model_name.count("vit")>0:
            self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()
            self.model.cls_token.data = deepcopy(self.cls_token).cuda()
        # print(self.model.pos_embed)
        # print(self.model.cls_token)
        reset_model(self)
    
    def _record_inout(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, SAttention):
                model._modules[name].first = True
                model._modules[name].T = self.T
                is_need = True
            elif isinstance(child, nn.Sequential) and isinstance(child[1], IFNeuron):
                model._modules[name] = save_module_inout(m=child,T=self.T)
                model._modules[name].first = True
                is_need = True
            if not is_need:            
                self._record_inout(child)            

    def set_snn_save_name(self, model):
        children = list(model.named_modules())
        for name, child in children:
            if isinstance(child, save_module_inout):
                child.name = name
                self.calOrder.append(name)
            if isinstance(child, SAttention):
                child.name = name
                self.calOrder.append(name)
    
    def _replace_weight(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=IFNeuron)
                attn_convert(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Conv2d) or isinstance(child, QuanConv2d):
                model._modules[name] = LLConv2d(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear) or isinstance(child, QuanLinear):
                model._modules[name] = LLLinear(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling(child)
                is_need = True
            elif isinstance(child,nn.BatchNorm2d):
                model._modules[name] = spiking_BatchNorm2d(bn=child,level=self.level)
                is_need = True
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.normalized_shape[0])
                if child.elementwise_affine:
                    SNN_LN.layernorm.weight.data = child.weight.data
                    SNN_LN.layernorm.bias.data = child.bias.data                
                model._modules[name] = SNN_LN
                is_need = True
            elif isinstance(child, MyQuan):
                neurons = ORIIFNeuron(q_threshold = torch.tensor(1.0),sym=child.sym,level = child.pos_max)
                neurons.q_threshold=child.s.data
                neurons.neuron_type=self.neuron_type
                neurons.level = self.level
                neurons.pos_max = child.pos_max
                neurons.neg_min = child.neg_min
                neurons.is_init = False
                model._modules[name] = neurons
                is_need = True
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
            if not is_need:            
                self._replace_weight(child)

    def forward(self,x, verbose=False):
        accu = None
        count1 = 0
        self.count2 = self.count2 + 1
        accu_per_timestep = []
        if self.visualize:
            self.hook_mid_feature()
        if self.Encoding_type == "rate":
            self.mean = 0.0
            self.std  = 0.0
            x = get_subtensors(x,self.mean,self.std,sample_grain=self.level)
        while(1):
            self.finish_judger.reset_network_finish_flag()
            self.finish_judger.judge_finish(self)
            network_finish = self.finish_judger.network_finish
            if (count1 > 0 and network_finish) or count1 >= self.T:
                self.max_T = max(count1, self.max_T)
                break
            if self.model_name.count("vit")>0 and count1 > 0:
                self.model.pos_embed = nn.Parameter(torch.zeros(1, self.model.patch_embed.num_patches + 1, self.model.
                                                                embed_dim).to(x.device))
                self.model.cls_token = nn.Parameter(torch.zeros(1, 1, self.model.embed_dim).to(x.device))
            if self.Encoding_type == "rate":
                if count1 < x.shape[0]:
                    input = x[count1]
                else:
                    input = torch.zeros(x[0].shape).to(x.device)            
            else:
                if count1 == 0:
                    input = x
                else:
                    input = torch.zeros(x.shape).to(x.device)

            output = self.model(input)
            
            if count1 == 0:
                accu = output+0.0
            else:
                accu = accu+output
            if verbose:
                accu_per_timestep.append(accu)
            # print("accu",accu.sum(),"output",output.sum())
            count1 = count1 + 1
            if count1 % 100 == 0:
                print(count1)
        # print("Time Step:",count1)
        if self.visualize:
            self.get_mid_feature()
            torch.save(self.feature_list,"spikeresnet50IF_output_feature.pth")
            torch.save(self.input_feature_list,"spikeresnet50IF_input_feature.pth")
        self.reset()
        if self.count2 > 0:
            self.visualize = True
        if verbose:
            accu_per_timestep = torch.stack(accu_per_timestep,dim=0)
            return accu,accu_per_timestep
        else:
            # print(accu.sum())
            return accu



class SNNWrapper(nn.Module):
    
    def __init__(self, ann_model, cfg, time_step = 2000,Encoding_type="rate",**kwargs):
        super(SNNWrapper, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model
        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_softmax = kwargs["is_softmax"]
        self.record_inout = kwargs["record_inout"]
        self.record_dir = kwargs["record_dir"]
        self.max_T = 0
        self.visualize = False
        self.calEnergy = False
        self.count2 = 0
        # self.model_reset = None
        if self.model_name.count("vit") > 0:
            self.pos_embed = deepcopy(self.model.pos_embed.data)
            self.cls_token = deepcopy(self.model.cls_token.data)

        self._replace_weight(self.model)
        # self.model_reset = deepcopy(self.model)        
        # if self.record_inout:
        #     self.calOrder = []
        #     self._record_inout(self.model)
        #     self.set_snn_save_name(self.model)
        #     local_rank = torch.distributed.get_rank()
        #     glo._init()
        #     if local_rank == 0:
        #         if not os.path.exists(self.record_dir):
        #             os.mkdir(self.record_dir)
        #         glo.set_value("output_bin_snn_dir",self.record_dir)
        #         f = open(f"{self.record_dir}/calculationOrder.txt","w+")
        #         for order in self.calOrder:
        #             f.write(order+"\n")
        #         f.close()
    
    def hook_mid_feature(self):
        self.feature_list = []
        self.input_feature_list = []
        def _hook_mid_feature(module, input, output):
            self.feature_list.append(output)
            self.input_feature_list.append(input[0])
        self.model.register_forward_hook(_hook_mid_feature)
        # self.model.blocks[11].attn.attn_IF.register_forward_hook(_hook_mid_feature)
    
    def get_mid_feature(self):
        self.feature_list = torch.stack(self.feature_list,dim=0)
        self.input_feature_list = torch.stack(self.input_feature_list,dim=0)
        print("self.feature_list",self.feature_list.shape) 
        print("self.input_feature_list",self.input_feature_list.shape) 
            
    def reset(self):
        # self.model = deepcopy(self.model_reset).cuda()
        if self.model_name.count("vit")>0:
            self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()
            self.model.cls_token.data = deepcopy(self.cls_token).cuda()
        # print(self.model.pos_embed)
        # print(self.model.cls_token)
        reset_model(self)
    
    def _record_inout(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, SAttention):
                model._modules[name].first = True
                model._modules[name].T = self.T
                is_need = True
            elif isinstance(child, nn.Sequential) and isinstance(child[1], IFNeuron):
                model._modules[name] = save_module_inout(m=child,T=self.T)
                model._modules[name].first = True
                is_need = True
            if not is_need:            
                self._record_inout(child)            

    def set_snn_save_name(self, model):
        children = list(model.named_modules())
        for name, child in children:
            if isinstance(child, save_module_inout):
                child.name = name
                self.calOrder.append(name)
            if isinstance(child, SAttention):
                child.name = name
                self.calOrder.append(name)
    
    def _replace_weight(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=ORIIFNeuron)
                attn_convert(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Conv2d) or isinstance(child, QuanConv2d):
                model._modules[name] = LLConv2d(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear) or isinstance(child, QuanLinear):
                model._modules[name] = LLLinear(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling(child)
                is_need = True
            elif isinstance(child,nn.BatchNorm2d):
                model._modules[name] = spiking_BatchNorm2d(bn=child,level=self.level)
                is_need = True
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.normalized_shape[0])
                if child.elementwise_affine:
                    SNN_LN.layernorm.weight.data = child.weight.data
                    SNN_LN.layernorm.bias.data = child.bias.data                
                model._modules[name] = SNN_LN
                is_need = True
            elif isinstance(child, MyQuan):
                neurons = ORIIFNeuron(q_threshold = torch.tensor(1.0),sym=child.sym,level = self.level)
                neurons.q_threshold=child.s.data
                neurons.neuron_type=self.neuron_type
                neurons.level = self.level
                neurons.pos_max = child.pos_max
                neurons.neg_min = child.neg_min
                neurons.is_init = False
                model._modules[name] = neurons
                is_need = True
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
            if not is_need:            
                self._replace_weight(child)

    def forward(self,x, verbose=False):
        accu = None
        count1 = 0
        self.count2 = self.count2 + 1
        accu_per_timestep = []
        # if self.visualize:
            # self.hook_mid_feature()
        if self.Encoding_type == "rate":
            self.mean = 0.0
            self.std  = 0.0
            x = get_subtensors(x,self.mean,self.std,sample_grain=self.level)
        while(1):
            self.finish_judger.reset_network_finish_flag()
            self.finish_judger.judge_finish(self)
            network_finish = self.finish_judger.network_finish
            network_finish = False
            if (count1 > 0 and network_finish) or count1 >= self.T:
                self.max_T = max(count1, self.max_T)
                break
            if self.model_name.count("vit")>0 and count1 > 0:
                self.model.pos_embed = nn.Parameter(torch.zeros(1, self.model.patch_embed.num_patches + 1, self.model.
                                                                embed_dim).to(x.device))
                self.model.cls_token = nn.Parameter(torch.zeros(1, 1, self.model.embed_dim).to(x.device))
            if self.Encoding_type == "rate":
                if count1 < x.shape[0]:
                    input = x[count1]
                else:
                    input = torch.zeros(x[0].shape).to(x.device)            
            else:
                if count1 == 0:
                    input = x
                else:
                    input = torch.zeros(x.shape).to(x.device)

            output = self.model(input)
            
            if count1 == 0:
                accu = output+0.0
            else:
                accu = accu+output
            if verbose:
                accu_per_timestep.append(accu)
            # print("accu",accu.sum(),"output",output.sum())
            count1 = count1 + 1
            if count1 % 100 == 0:
                print(count1)
        # print("Time Step:",count1)
        # if self.visualize:
            # self.get_mid_feature()
            # torch.save(self.feature_list,"spikeresnet50_output_feature.pth")
            # torch.save(self.input_feature_list,"spikeresnet50_input_feature.pth")
        self.reset()
        if self.count2 > 0:
            self.visualize = True
        if verbose:
            accu_per_timestep = torch.stack(accu_per_timestep,dim=0)
            return accu,accu_per_timestep
        else:
            # print(accu.sum())
            return accu




class SNNWrapperLCC(nn.Module):
    
    def __init__(self, ann_model, cfg, time_step = 2000,Encoding_type="rate",**kwargs):
        super(SNNWrapperLCC, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model
        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_softmax = kwargs["is_softmax"]
        self.record_inout = kwargs["record_inout"]
        self.record_dir = kwargs["record_dir"]
        self.max_T = 0
        self.visualize = False
        self.last_input = torch.tensor(0.0)
        # self.model_reset = None
        if self.model_name.count("vit") > 0:
            self.pos_embed = deepcopy(self.model.pos_embed.data)
            self.cls_token = deepcopy(self.model.cls_token.data)
        self.accu = torch.tensor(0.0)
        self.count1 = 0
        self.count2 = 0

        self._replace_weight(self.model)
        # self.model_reset = deepcopy(self.model)        
        # if self.record_inout:
        #     self.calOrder = []
        #     self._record_inout(self.model)
        #     self.set_snn_save_name(self.model)
        #     local_rank = torch.distributed.get_rank()
        #     glo._init()
        #     if local_rank == 0:
        #         if not os.path.exists(self.record_dir):
        #             os.mkdir(self.record_dir)
        #         glo.set_value("output_bin_snn_dir",self.record_dir)
        #         f = open(f"{self.record_dir}/calculationOrder.txt","w+")
        #         for order in self.calOrder:
        #             f.write(order+"\n")
        #         f.close()
    
    def hook_mid_feature(self):
        self.feature_list = []
        self.input_feature_list = []
        def _hook_mid_feature(module, input, output):
            self.feature_list.append(output)
            self.input_feature_list.append(input[0])
        self.model.register_forward_hook(_hook_mid_feature)
        # self.model.blocks[11].attn.attn_IF.register_forward_hook(_hook_mid_feature)
    
    def get_mid_feature(self):
        self.feature_list = torch.stack(self.feature_list,dim=0)
        self.input_feature_list = torch.stack(self.input_feature_list,dim=0)
        print("self.feature_list",self.feature_list.shape) 
        print("self.input_feature_list",self.input_feature_list.shape) 
            
    def reset(self):
        print("LCC Reset!!!")
        # self.model = deepcopy(self.model_reset).cuda()
        if self.model_name.count("vit")>0:
            self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()
            self.model.cls_token.data = deepcopy(self.cls_token).cuda()
        # print(self.model.pos_embed)
        # print(self.model.cls_token)
        self.accu = torch.tensor(0.0)
        self.count1 = 0
        self.last_input = torch.tensor(0.0)
        reset_model(self)
    
    def _record_inout(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, SAttention):
                model._modules[name].first = True
                model._modules[name].T = self.T
                is_need = True
            elif isinstance(child, nn.Sequential) and isinstance(child[1], IFNeuron):
                model._modules[name] = save_module_inout(m=child,T=self.T)
                model._modules[name].first = True
                is_need = True
            if not is_need:            
                self._record_inout(child)            

    def set_snn_save_name(self, model):
        children = list(model.named_modules())
        for name, child in children:
            if isinstance(child, save_module_inout):
                child.name = name
                self.calOrder.append(name)
            if isinstance(child, SAttention):
                child.name = name
                self.calOrder.append(name)
    
    def _replace_weight(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=IFNeuron)
                attn_convert(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Conv2d) or isinstance(child, QuanConv2d):
                model._modules[name] = LLConv2d(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear) or isinstance(child, QuanLinear):
                model._modules[name] = LLLinear(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling(child)
                is_need = True
            elif isinstance(child,nn.BatchNorm2d):
                model._modules[name] = spiking_BatchNorm2d(bn=child,level=self.level)
                is_need = True
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.normalized_shape[0])
                if child.elementwise_affine:
                    SNN_LN.layernorm.weight.data = child.weight.data
                    SNN_LN.layernorm.bias.data = child.bias.data                
                model._modules[name] = SNN_LN
                is_need = True
            elif isinstance(child, MyQuan):
                neurons = ORIIFNeuron(q_threshold = torch.tensor(1.0),sym=child.sym,level = self.level)
                neurons.q_threshold=child.s.data
                neurons.neuron_type=self.neuron_type
                neurons.level = self.level
                neurons.pos_max = child.pos_max
                neurons.neg_min = child.neg_min
                neurons.is_init = False
                model._modules[name] = neurons
                is_need = True
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
            if not is_need:            
                self._replace_weight(child)

    def forward(self, x):
        accuList = []
        accuperTime = []
        self.count2 = self.count2 + 1
        B = x.shape[0]
        # if self.visualize:
            # self.hook_mid_feature()
        for idx in range(B):
            accuperTime1 = []
            count2 = 0
            x1 = x[idx]
            # if self.count1 > 0:
                # self.T = 32
            
            # 用于存储在 T 时刻的累计输出
            accu_at_T = None
            
            while(1):
                self.finish_judger.reset_network_finish_flag()
                self.finish_judger.judge_finish(self)
                network_finish = self.finish_judger.network_finish
                
                # 只有在 judger 判断 finish 且 count2 > 0 时才停止
                if count2 > 0 and network_finish or count2 >= 16:
                    self.max_T = max(count2, self.max_T)
                    break

                if count2 == 0:
                    input = x1.unsqueeze(0) - self.last_input
                else:
                    input = torch.zeros((x1.unsqueeze(0)).shape).to(x1.device)

                output = self.model(input)
                
                if self.count1 == 0:
                    self.accu = output + 0.0
                    self.count1 = self.count1 + 1
                else:
                    self.accu = self.accu + output
                    self.count1 = self.count1 + 1

                count2 = count2 + 1
                accuperTime1.append(output + 0.0)
                
                # 在 count2 == self.T 时保存累计结果，但继续推理
                if count2 == self.T:
                    accu_at_T = self.accu + 0.0  # 深拷贝当前累计结果

            # 使用在 T 时刻的累计输出，如果循环在 T 之前结束，则使用最终累计输出
            if accu_at_T is not None:
                accuList.append(accu_at_T)
            else:
                accuList.append(self.accu + 0.0)
                
            self.last_input = x1 + 0.0
            accuperTime.append(torch.cat(accuperTime1, dim=0))
        
        # if self.visualize:
        #     self.get_mid_feature()
        #     # torch.save(torch.stack(accuperTime, dim=0),f"spikeresnet50LCC_output_feature{self.count1}.pth")
        #     torch.save(self.feature_list,"spikeresnet50LCC_output_feature.pth")
        #     torch.save(self.input_feature_list,"spikeresnet50LCC_input_feature.pth")
        # print(torch.stack(accuList, dim=0).sum())
        # print(torch.cat(accuList, dim=0).shape)
        # if self.count2 > 0:
            # self.visualize = True
        return torch.cat(accuList, dim=0)


def remove_softmax(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, Attention):
            reluattn = Attention_no_softmax(dim=child.num_heads*child.head_dim,num_heads=child.num_heads)
            reluattn.qkv = child.qkv
            reluattn.attn_drop = child.attn_drop
            reluattn.proj = child.proj
            reluattn.proj_drop = child.proj_drop
            is_need = True
            model._modules[name] = reluattn
        # elif isinstance(child, nn.LayerNorm):
        #     LN = MyBatchNorm1d(num_features = child.normalized_shape[0])
        #     # LN.weight.data = child.weight
        #     # LN.bias.data = child.bias
        #     model._modules[name] = LN
        if not is_need:
            remove_softmax(child)



def myquan_replace(model,level,weight_bit=32, is_softmax = True):
    index = 0
    cur_index = 0
    def get_index(model):
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                index = index + 1
                is_need = True
            if not is_need:
                get_index(child)

    def _myquan_replace(model,level):
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, Block):
                # print(children)
                qattn = QAttention(dim=child.attn.num_heads*child.attn.head_dim,num_heads=child.attn.num_heads,level=level,is_softmax=is_softmax)
                qattn.qkv = child.attn.qkv
                # qattn.q_norm = child.q_norm
                # qattn.k_norm = child.k_norm
                qattn.attn_drop = child.attn.attn_drop
                qattn.proj = child.attn.proj
                qattn.proj_drop = child.attn.proj_drop
                model._modules[name].attn = qattn
                # model._modules[name].act1 = MyQuan(level, sym=True)
                # model._modules[name].act2 = MyQuan(level, sym=True)
                model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=True))
                model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=True))
                model._modules[name].mlp.fc1 = nn.Sequential(child.mlp.fc1,MyQuan(level, sym=False))
                model._modules[name].mlp.fc2 = nn.Sequential(child.mlp.fc2,MyQuan(level, sym=True))
                model._modules[name].addition1 = nn.Sequential(Addition(),MyQuan(level, sym=True))
                model._modules[name].addition2 = nn.Sequential(Addition(),MyQuan(level, sym=True))
                print("model._modules[name].addition1",model._modules[name].addition1)
                print("index",cur_index,"myquan replace finish!!!!")
                cur_index = cur_index + 1
                is_need = True
            # if isinstance(child, Attention):
            #     # print(children)
            #     qattn = QAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=level)
            #     qattn.qkv = child.qkv
            #     # qattn.q_norm = child.q_norm
            #     # qattn.k_norm = child.k_norm
            #     qattn.attn_drop = child.attn_drop
            #     qattn.proj = child.proj
            #     qattn.proj_drop = child.proj_drop
            #     model._modules[name] = qattn
            #     print("index",cur_index,"myquan replace finish!!!!")
            #     cur_index = cur_index + 1
            #     is_need = True
            # elif isinstance(child,Mlp):
            #     model._modules[name].act = nn.Sequential(MyQuan(level,sym = False),child.act)
            #     model._modules[name].fc2 = nn.Sequential(child.fc2,MyQuan(level,sym = True))
            #     is_need = True
            elif isinstance(child, nn.Conv2d):
                model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
                is_need = True
            # elif isinstance(child, Block):
            #     model._modules[name].norm1 = nn.Sequential(child.norm1,MyQuan(level,sym = True))
            #     model._modules[name].norm2 = nn.Sequential(child.norm2,MyQuan(level,sym = True))
            #     is_need = False
            elif isinstance(child, nn.LayerNorm):
                model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
                is_need = True
            if not is_need:
                _myquan_replace(child,level)
    
    def _weight_quantization(model,weight_bit):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.Conv2d):
                model._modules[name] = QuanConv2d(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = QuanLinear(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            if not is_need:
                _weight_quantization(child,weight_bit)
                
    get_index(model)
    _myquan_replace(model,level)
    if weight_bit < 32:
        _weight_quantization(model,weight_bit)



def myquan_replace_resnet(model,level,weight_bit=32, is_softmax = True):
    index = 0
    cur_index = 0
    def get_index(model):
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                index = index + 1
                is_need = True
            if not is_need:
                get_index(child)

    def _myquan_replace(model,level):
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.ReLU):
                model._modules[name] = MyQuan(level,sym = False)
                is_need = True
            if not is_need:
                _myquan_replace(child,level)
    
    def _weight_quantization(model,weight_bit):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.Conv2d):
                model._modules[name] = QuanConv2d(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = QuanLinear(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            if not is_need:
                _weight_quantization(child,weight_bit)
                
    get_index(model)
    _myquan_replace(model,level)
    if weight_bit < 32:
        _weight_quantization(model,weight_bit)

