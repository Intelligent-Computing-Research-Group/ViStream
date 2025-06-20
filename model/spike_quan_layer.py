import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy
import numpy as np
import scipy

# torch.set_default_dtype(torch.double)
# torch.set_default_tensor_type(torch.DoubleTensor)

class ORIIFNeuron(nn.Module):
    """
    改进版 IF 神经元：
    1. q、cur_output 用 register_buffer() 绑定到模块，跟随 .cuda() / DDP 自动搬迁；
    2. 前向采用 in-place add_ / zero_，彻底复用显存；
    3. 推理阶段不记录梯度；
    """

    def __init__(self, q_threshold: float | torch.Tensor, level: int, sym: bool = False):
        super().__init__()
        # 常量参数
        self.q_threshold = torch.as_tensor(q_threshold, dtype=torch.float32)  # 可是标量，也可加载量化标尺
        self.level = level
        self.sym = sym
        self.pos_max = level - 1
        self.neg_min = 0
        self.eps = 0.0

        # 状态缓冲区（懒初始化）
        self.register_buffer("q", None, persistent=False)          # 积分电荷
        self.register_buffer("cur_output", None, persistent=False) # 本步脉冲
        self.is_work = False

    # ------------------------------------------------------------
    # 实用函数
    # ------------------------------------------------------------
    def _lazy_init_state(self, x: torch.Tensor):
        """按输入尺寸一次性申请显存，之后一直复用"""
        if self.q is None or self.q.shape != x.shape:
            self.q = torch.zeros_like(x, dtype=torch.float32) + 0.5
            self.cur_output = torch.zeros_like(x, dtype=x.dtype)

    # ------------------------------------------------------------
    # 接口
    # ------------------------------------------------------------
    def reset(self):
        """清空内部状态；训练／推理新序列前调用"""
        if isinstance(self.q, torch.Tensor):
            self.q.zero_().add_(0.5)
        if isinstance(self.cur_output, torch.Tensor):
            self.cur_output.zero_()
        self.is_work = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        输入：实数电流  (B, …)  
        输出：脉冲张量，同尺寸，数值 ∈ {0, q_threshold}
        """
        # 归一化
        x = input / self.q_threshold

        # 第一次调用时按输入形状初始化缓存
        self._lazy_init_state(x)

        # 积分（不需要梯度，且 in-place）
        with torch.no_grad():
            self.q.add_(x)

        # 生成脉冲
        spike_pos = self.q >= 1
        self.cur_output.zero_()
        self.cur_output[spike_pos] = 1

        # 电荷回落
        self.q[spike_pos] -= 1

        # 判断是否仍在工作（可选逻辑）
        self.is_work = spike_pos.any().item()

        return self.cur_output * self.q_threshold


class NeuronCircuit(nn.Module):
    def __init__(self,pos_max, neg_min):
        super(NeuronCircuit,self).__init__()
        self.pos_max = pos_max
        self.neg_min = neg_min
    
    def forward(self, q, acc_q):
        spike_position = (q - 1 >= 0) & (acc_q < self.pos_max)
        neg_spike_position = (q < 0) & (acc_q > self.neg_min)

        cur_output = spike_position.float() - neg_spike_position.float()

        acc_q = acc_q + cur_output
        q = q - cur_output        
        return q, acc_q, cur_output

class IFNeuron(nn.Module):
    def __init__(self,q_threshold,level,sym=False):
        super(IFNeuron,self).__init__()
        self.q = 0.5
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.level = torch.tensor(level)
        self.sym = sym
        if sym:
            self.pos_max = torch.tensor(level//2 - 1)
            self.neg_min = torch.tensor(-level//2)
        else:
            self.pos_max = torch.tensor(level//2 - 1)
            self.neg_min = torch.tensor(0)
        self.init = False
        self.eps = 0
        self.neuralCircuit = torch.jit.script(NeuronCircuit(self.pos_max,self.neg_min))
    # def __repr__(self):
    #         return f"ST-BIFNeuron(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.5
        self.acc_q = 0.0
        self.is_work = False
        self.init = False
        self.spike_position = None
        self.neg_spike_position = None

    def forward(self,input):
        with torch.no_grad():
            x = input/self.q_threshold
            if (not torch.is_tensor(x)) and x == 0.0 and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
                self.is_work = False
                return x*self.q_threshold
            
            self.is_work = True
            self.q = self.q + x
            if self.init == False:
                self.acc_q = torch.zeros(x.shape,dtype=torch.float32).to(x.device)
            self.init = True
            
            self.q, self.acc_q, self.cur_output = self.neuralCircuit(self.q, self.acc_q)

            # spike_position = (self.q - 1 >= 0) & (self.acc_q < self.pos_max)
            # neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

            # self.cur_output = spike_position.float() - neg_spike_position.float()

            # self.acc_q = self.acc_q + self.cur_output
            # self.q = self.q - self.cur_output

            # print((x == 0).all(), (self.cur_output==0).all())
            if x.abs().sum() == 0 and self.cur_output.abs().sum() == 0:
                self.is_work = False
            
            # print("self.cur_output",self.cur_output)
            
        return self.cur_output*self.q_threshold

class spiking_BatchNorm2d(nn.Module):
    def __init__(self,bn,level):
        super(spiking_BatchNorm2d, self).__init__()
        self.level = level
        self.fire_time = self.level
        self.bn = bn
        self.running_mean = bn.running_mean.unsqueeze(-1).unsqueeze(-1).cuda()
        self.running_var = bn.running_var.unsqueeze(-1).unsqueeze(-1).cuda()
        self.weight = bn.weight.unsqueeze(-1).unsqueeze(-1).cuda()
        self.bias = bn.bias.unsqueeze(-1).unsqueeze(-1).cuda()
        self.eps = bn.eps
    
    def reset(self):
        # print("spiking_BatchNorm2d reset")
        self.fire_time = self.level
    
    def forward(self, input):
        if self.fire_time == self.level:
            self.fire_time = self.fire_time - 1
            output = ((input - self.running_mean)/torch.sqrt(self.running_var+self.eps))*self.weight + self.bias
            return output
        else:
            return ((input)/torch.sqrt(self.running_var+self.eps))*self.weight



class Spiking_LayerNorm(nn.Module):
    def __init__(self,dim):
        super(Spiking_LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.X = 0.0
        self.Y_pre = None
        self.weight = self.layernorm.weight
        self.bias = self.layernorm.bias        

    def reset(self):
        # print("Spiking_LayerNorm reset")
        self.X = 0.0
        self.Y_pre = None
        
    def forward(self,input):
        self.X = self.X + input
        Y = self.layernorm(self.X)
        if self.Y_pre is not None:
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = 0.0
        self.Y_pre = Y
        return Y - Y_pre

class spiking_softmax(nn.Module):
    def __init__(self):
        super(spiking_softmax, self).__init__()
        self.X = 0.0
        self.Y_pre = 0.0
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = 0.0
        self.Y_pre = 0.0        
    
    def forward(self, input):
        self.X = input + self.X
        Y = F.softmax(self.X,dim=-1)
        Y_pre = deepcopy(self.Y_pre)
        self.Y_pre = Y
        return Y - Y_pre

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def threshold_optimization(data, quantization_level=255, n_trial=300, eps=1e-10):
    '''
    This function collect the activation data and find the optimized clipping
    threshold using KL_div as metric. Since this method is originated from
    post-training quantization which adopted in Tensor-RT, we keep the number of
    bits here.
    Args:
        data(numpy array): activation data
        n_bit(int):
        n_trial(int): the searching steps.
        eps(float): add eps at the average bin step for numberical stability.

    '''

    n_lvl = quantization_level  # quantization levels
    n_half_lvls = (quantization_level)//2
    n_bin_edge = n_lvl * n_trial + 1

    data_max = np.max(np.abs(data))
    hist, bin_edge = np.histogram(data.flatten(),
                                  bins=np.linspace(-data_max,
                                                   data_max,
                                                   num=n_bin_edge))

    mid_idx = int((len(hist)) / 2)
    start_idx = 100
    # log the threshold and corresponding KL-divergence
    kl_result = np.empty([len(range(start_idx, n_trial + 1)), 2])

    for i in range(start_idx, n_trial + 1):
        ref_dist = np.copy(hist[mid_idx - i * n_half_lvls:mid_idx +
                                i * n_half_lvls])
        # merge the outlier
        ref_dist[0] += hist[:mid_idx - i * n_half_lvls].sum()
        ref_dist[-1] += hist[mid_idx + i * n_half_lvls:].sum()
        # perform quantization: bins merge and expansion
        reshape_dim = int(len(ref_dist) / n_lvl)
        ref_dist_reshape = ref_dist.reshape(n_lvl, i)
        # merge bins for quantization
        ref_dist_merged = ref_dist_reshape.sum(axis=1)
        nonzero_mask = (ref_dist_reshape != 0
                        )  # obtain the mask of non-zero bins
        # in each merged large bin, get the average bin_count
        average_bin_count = ref_dist_merged / (nonzero_mask.sum(1) + eps)
        # expand the merged bins back
        expand_bin_count = np.expand_dims(average_bin_count,
                                          axis=1).repeat(i, axis=1)
        candidate_dist = (nonzero_mask * expand_bin_count).flatten()
        kl_div = scipy.stats.entropy(candidate_dist / candidate_dist.sum(),
                                     ref_dist / ref_dist.sum())
        #log threshold and KL-divergence
        current_th = np.abs(
            bin_edge[mid_idx - i * n_half_lvls])  # obtain current threshold
        kl_result[i -
                  start_idx, 0], kl_result[i -
                                           start_idx, 1] = current_th, kl_div

    # based on the logged kl-div result, find the threshold correspond to the smallest kl-div
    th_sel = kl_result[kl_result[:, 1] == kl_result[:, 1].min()][0, 0]
    print(f"Threshold calibration of current layer finished!, calculate threshold {th_sel}")

    return th_sel

# class MyQuan(nn.Module):
#     def __init__(self,level,sym = False,**kwargs):
#         super(MyQuan,self).__init__()
#         # self.level_init = level
#         self.s_init = 0.0
#         self.level = level
#         self.sym = sym
#         if level >= 256:
#             print("level",level)
#             self.pos_max = 'full'
#             self.neg_min = 'full'
#         else:
#             print("level",level)
#             self.pos_max = torch.tensor(level)
#             if sym:
#                 self.pos_max = torch.tensor(float(level//2 - 1))
#                 self.neg_min = torch.tensor(float(-level//2))
#             else:
#                 self.pos_max = torch.tensor(float(level - 1))
#                 self.neg_min = torch.tensor(float(0))
#
#         self.s = nn.Parameter(torch.tensor(1.0))
#         self.batch_init = 20
#         self.init_state = 0
#         self.debug = False
#         self.tfwriter = None
#         self.global_step = 0.0
#         self.name = "myquan"
#
#     def __repr__(self):
#         return f"MyQuan(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, s={self.s.data})"
#
#
#     def reset(self):
#         self.history_max = torch.tensor(0.0)
#         self.init_state = 0
#         self.is_init = True
#
#     def profiling(self,name,tfwriter,global_step):
#         self.debug = True
#         self.name = name
#         self.tfwriter = tfwriter
#         self.global_step = global_step
#
#     def forward(self, x):
#         # print("self.pos_max",self.pos_max)
#         if self.pos_max == 'full':
#             return x
#         # print("self.Q_thr in Quan",self.Q_thr,"self.T:",self.T)
#         if str(self.neg_min.device) == 'cpu':
#             self.neg_min = self.neg_min.to(x.device)
#         if str(self.pos_max.device) == 'cpu':
#             self.pos_max = self.pos_max.to(x.device)
#         min_val = self.neg_min
#         max_val = self.pos_max
#         # x = F.hardtanh(x, min_val=min_val, max_val=max_val.item())
#
#         # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
#         # s_grad_scale = 1.0 / ((self.level * x.numel()) ** 0.5)
#         s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)
#
#         # s_grad_scale = s_grad_scale / ((self.level_init)/(self.pos_max))
#
#         # print("self.init_state",self.init_state)
#         # print("self.init_state<self.batch_init",self.init_state<self.batch_init)
#         # print("self.training",self.training)
#         if self.init_state == 0 and self.training:
#             threshold = threshold_optimization(np.array(x.detach().cpu()), quantization_level=int(self.level), n_trial=300, eps=1e-10)
#             self.s.data = torch.tensor(threshold / (self.level),dtype=torch.float32).cuda()
#             self.init_state += 1
#         # if self.init_state == 0 and self.training:
#         #     self.s.data = torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max.detach().abs().mean() ** 0.5),dtype=torch.float32).cuda()
#         #     self.init_state += 1
#         # elif self.init_state<self.batch_init and self.training:
#         #     self.s.data = 0.9*self.s.data + 0.1*torch.tensor(torch.mean(torch.abs(x.detach()))*2/(math.sqrt(max_val.detach().abs().mean())),dtype=torch.float32)
#         #     self.init_state += 1
#
#         # elif self.init_state==self.batch_init and self.training:
#         #     # self.s = torch.nn.Parameter(self.s)
#         #     self.init_state += 1
#         #     print("initialize finish!!!!")
#
#         s_scale = grad_scale(self.s, s_grad_scale)
#         # s_scale = s_scale * ((self.level_init)/(self.pos_max))
#         output = torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)*s_scale
#
#         if self.debug and self.tfwriter is not None:
#             self.tfwriter.add_histogram(tag="before_quan/".format(s_scale.item())+self.name+'_data', values=(x).detach().cpu(), global_step=self.global_step)
#             # self.tfwriter.add_histogram(tag="after_clip/".format(s_scale.item())+self.name+'_data', values=(floor_pass(x/s_scale)).detach().cpu(), global_step=self.global_step)
#             self.tfwriter.add_histogram(tag="after_quan/".format(s_scale.item())+self.name+'_data', values=((torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))).detach().cpu(), global_step=self.global_step)
#             # print("(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))",(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)))
#             self.debug = False
#             self.tfwriter = None
#             self.name = ""
#             self.global_step = 0.0
#
#         # output = floor_pass(x/s_scale)*s_scale
#         return output

def set_init_false(model):
    def set_init_false_inner(model):
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, MyQuan):
                model._modules[name].init_state = model._modules[name].batch_init
            else:
                set_init_false_inner(child)
    set_init_false_inner(model)

def l2_regularization(model):
    l2_loss = 0.0
    abs_value = 0.0
    def l2_regularization_inner(model):
        nonlocal l2_loss
        nonlocal abs_value
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, MyQuan):
                l2_loss += model._modules[name].l2_loss
                abs_value += model._modules[name].absvalue.detach()
            else:
                l2_regularization_inner(child)
    l2_regularization_inner(model)
    return l2_loss, abs_value

class MyQuan(nn.Module):
    def __init__(self,level,sym = False,**kwargs):
        super(MyQuan,self).__init__()
        # self.level_init = level
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        if level >= 512:
            # print("level",level)
            self.pos_max = 'full'
        else:
            # print("level",level)
            self.pos_max = torch.tensor(level)
            if sym:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(-level//2))
            else:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(0))

        self.s = nn.Parameter(torch.tensor(1.0))
        self.batch_init = 20
        self.init_state = 0
        self.debug = False
        self.tfwriter = None
        self.global_step = 0.0
        self.name = "myquan"

    def __repr__(self):
        return f"MyQuan(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, s={self.s.data})"


    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def profiling(self,name,tfwriter,global_step):
        self.debug = True
        self.name = name
        self.tfwriter = tfwriter
        self.global_step = global_step

    def forward(self, x):
        # print("self.pos_max",self.pos_max)
        if self.pos_max == 'full':
            return x
        # print("self.Q_thr in Quan",self.Q_thr,"self.T:",self.T)
        if str(self.neg_min.device) == 'cpu':
            self.neg_min = self.neg_min.to(x.device)
        if str(self.pos_max.device) == 'cpu':
            self.pos_max = self.pos_max.to(x.device)
        min_val = self.neg_min
        max_val = self.pos_max
        # x = F.hardtanh(x, min_val=min_val, max_val=max_val.item())

        # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
        s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)
        # s_grad_scale = s_grad_scale / ((self.level_init)/(self.pos_max))

        # print("self.init_state",self.init_state)
        # print("self.init_state<self.batch_init",self.init_state<self.batch_init)
        # print("self.training",self.training)
        if self.init_state == 0 and self.training:
            self.s.data = torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max.detach().abs().mean() ** 0.5),dtype=torch.float32).cuda()
            self.init_state += 1
        # elif self.init_state<self.batch_init and self.training:
        #     self.s.data = 0.9*self.s.data + 0.1*torch.tensor(torch.mean(torch.abs(x.detach()))*2/(math.sqrt(max_val.detach().abs().mean())),dtype=torch.float32)
        #     self.init_state += 1

        # elif self.init_state==self.batch_init and self.training:
        #     # self.s = torch.nn.Parameter(self.s)
        #     self.init_state += 1
        #     # print("initialize finish!!!!")
        l2_loss1 = 0
        s_scale = grad_scale(self.s, s_grad_scale)
        # s_scale = s_scale * ((self.level_init)/(self.pos_max))
        output = torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)*s_scale

        if self.debug and self.tfwriter is not None:
            self.tfwriter.add_histogram(tag="before_quan/".format(s_scale.item())+self.name+'_data', values=(x).detach().cpu(), global_step=self.global_step)
            # self.tfwriter.add_histogram(tag="after_clip/".format(s_scale.item())+self.name+'_data', values=(floor_pass(x/s_scale)).detach().cpu(), global_step=self.global_step)
            self.tfwriter.add_histogram(tag="after_quan/".format(s_scale.item())+self.name+'_data', values=((torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))).detach().cpu(), global_step=self.global_step)
            # print("(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))",(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)))
            self.debug = False
            self.tfwriter = None
            self.name = ""
            self.global_step = 0.0

        # x_abs = torch.abs(output)/self.s
        # self.l2_loss = l2_loss1 + (x_abs - (1/147)*x_abs*x_abs*x_abs).sum()
        # self.absvalue = (torch.abs(output)/self.s).sum()
        # output = floor_pass(x/s_scale)*s_scale
        return output

class QAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            level = 2,
            is_softmax = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.level = level
        self.is_softmax = is_softmax

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.quan_q = MyQuan(self.level,sym=True)
        self.quan_k = MyQuan(self.level,sym=True)
        self.quan_v = MyQuan(self.level,sym=True)
        # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim,bias=True)
        self.quan_proj = MyQuan(self.level,sym=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_quan = MyQuan(self.level,sym=True)
        if self.is_softmax:
            self.attn_softmax_quan = MyQuan(self.level,sym=False)
        self.after_attn_quan = MyQuan(self.level,sym=True)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # q, k = self.q_norm(q), self.k_norm(k)
        q = self.quan_q(q)
        k = self.quan_k(k)
        v = self.quan_v(v)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.is_softmax:
            attn = self.attn_quan(attn)
            attn = attn.softmax(dim=-1)
            attn = self.attn_softmax_quan(attn)
        else:
            # print("no softmax!!!!")
            attn = self.attn_quan(attn)/N
        
        attn = self.attn_drop(attn)
        x = attn @ v
        x = self.after_attn_quan(x)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.quan_proj(x)

        return x

def multi(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return x1_sum_t @ x2_t.transpose(-2, -1)  + x1_t @ x2_sum_t.transpose(-2, -1) - x1_t @ x2_t.transpose(-2, -1)

def multi1(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return x1_sum_t @ x2_t + x1_t @ x2_sum_t - x1_t @ x2_t

class SAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            neuron_layer = IFNeuron,
            level = 2,
            is_softmax = True,
            T = 32,
            
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.neuron_layer = neuron_layer
        self.level = level
        self.is_softmax = is_softmax

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.k_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.v_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.attn_ReLU = nn.ReLU()
        self.attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        if self.is_softmax:
            self.attn_softmax_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=False)
        self.after_attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.proj = nn.Linear(dim, dim,bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        if self.is_softmax:
            self.Ssoftmax = spiking_softmax()
        self.T = T

        # saving mid feature
        self.t = 0
        self.first = False        
        self.accu_input = []
        self.accu_qkv = []
        self.accu_q = []
        self.accu_k = []
        self.accu_v = []
        self.accu_q_scale = []
        self.accu_q_scale_acc = []
        self.accu_k_acc = []
        self.accu_v_acc = []
        self.accu_qk = []
        self.accu_qk_softmax = []
        self.accu_qk_acc = []
        self.accu_attn = []
        self.accu_proj_input = []
        self.accu_proj = []
        self.accu = []
        self.accu1 = []
        self.name = ""

    def reset(self):
        # print("SAttention reset")
        self.q_IF.reset()
        self.k_IF.reset()
        self.v_IF.reset()
        self.attn_IF.reset()
        self.attn_softmax_IF.reset()
        self.after_attn_IF.reset()
        self.proj_IF.reset()
        if self.is_softmax:
            self.Ssoftmax.reset()
        self.qkv.reset()
        self.proj.reset()
        self.t = 0

    def forward(self, x):
        self.t = self.t + 1
        B, N, C = x.shape
        # print("qkv:", self.qkv(x).shape, self.qkv.out_features)
        if self.first:
            self.accu_input.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                # print(self.qkv)
                # print(self.accu_input[0][0,0,:])
                # print(self.qkv.linear.quan_w_fn(self.qkv.linear.weight)[0,:64])
                save_input_for_bin_snn_4dim(torch.stack(self.accu_input), glo.get_value("output_bin_snn_dir"),self.name+"_qkv.in")
                del self.accu_input

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3 B self.num_heads N self.head_dim
        # if self.t == 1:
        #     qkv1 = torch.nn.functional.linear(x, self.qkv.linear.quan_w_fn(self.qkv.linear.weight), bias=self.qkv.linear.bias).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # else:
        #     qkv1 = torch.nn.functional.linear(x, self.qkv.linear.quan_w_fn(self.qkv.linear.weight), bias=None).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv.unbind(0)
        # q1, k1, v1 = qkv1.unbind(0)
        # if self.first:
        #     self.accu.append(v[0].unsqueeze(0)+0)
        #     self.accu1.append(v1[0].unsqueeze(0)+0)
        #     if self.t == self.T:
        #         mask = (torch.stack(self.accu) - torch.stack(self.accu1).abs() > 1e-3)                
        #         print(len(mask))
        #         print(torch.stack(self.accu)[mask])
                # print(torch.stack(self.accu1)[mask])
                # print(torch.stack(self.accu)[0,0,0,0,:64])
                # print(torch.stack(self.accu1)[0,0,0,0,:64])
            
        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)
        
        if self.first:
            self.accu_q.append(q[0].unsqueeze(0)+0)
            self.accu_k.append(k[0].unsqueeze(0)+0)
            self.accu_v.append(v[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn_5dim(torch.stack(self.accu_q), glo.get_value("output_bin_snn_dir"),self.name+"_qkv_q.out")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_k), glo.get_value("output_bin_snn_dir"),self.name+"_qkv_k.out")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_v), glo.get_value("output_bin_snn_dir"),self.name+"_qkv_v.out")
                del self.accu_q
        
        
        q = q * self.scale
        q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold
        if self.first:
            self.accu_q_scale.append(q[0].unsqueeze(0)+0)
            self.accu_q_scale_acc.append(q_acc[0].unsqueeze(0)+0)
            self.accu_k_acc.append((self.k_IF.acc_q*self.k_IF.q_threshold)[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn_5dim(torch.stack(self.accu_q_scale), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_q.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_q_scale_acc), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_q_accu.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_k), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_k.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_k_acc), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_k_accu.in")
                del self.accu_q_scale
                del self.accu_q_scale_acc
                del self.accu_k
                del self.accu_k_acc

        attn = multi(q,k,q_acc.float(),(self.k_IF.acc_q*self.k_IF.q_threshold).float())

        attn = self.attn_IF(attn)

        if self.first:
            self.accu_qk.append(attn[0].unsqueeze(0)+0)
            if self.t == self.T:    
                save_input_for_bin_snn_5dim(torch.stack(self.accu_qk), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti.out")        
        
        if self.is_softmax:
            attn = self.Ssoftmax(attn)
            attn = self.attn_softmax_IF(attn)
            if self.first:
                self.accu_qk_softmax.append(attn[0].unsqueeze(0)+0)
                if self.t == self.T:    
                    save_input_for_bin_snn_5dim(torch.stack(self.accu_qk_softmax), glo.get_value("output_bin_snn_dir"),self.name+"_qkMulti_softmax.out")



        if not self.is_softmax:
            attn = attn/N
            acc_attn = self.attn_IF.acc_q*self.attn_IF.q_threshold/N

        attn = self.attn_drop(attn)

        if self.first:
            if not self.is_softmax:
                self.accu_qk_acc.append((self.attn_IF.acc_q*self.attn_IF.q_threshold)[0].unsqueeze(0)+0)
            else:
                self.accu_qk_acc.append((self.attn_softmax_IF.acc_q*self.attn_softmax_IF.q_threshold)[0].unsqueeze(0)+0)
            self.accu_v_acc.append((self.v_IF.acc_q*self.v_IF.q_threshold)[0].unsqueeze(0)+0)
            if self.t == self.T:
                if not self.is_softmax:
                    save_input_for_bin_snn_5dim(torch.stack(self.accu_qk), glo.get_value("output_bin_snn_dir"),self.name+"_attn_qk.in")
                else:
                    save_input_for_bin_snn_5dim(torch.stack(self.accu_qk_softmax), glo.get_value("output_bin_snn_dir"),self.name+"_attn_qk.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_qk_acc), glo.get_value("output_bin_snn_dir"),self.name+"_attn_qk_acc.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_v), glo.get_value("output_bin_snn_dir"),self.name+"_attn_v.in")
                save_input_for_bin_snn_5dim(torch.stack(self.accu_v_acc), glo.get_value("output_bin_snn_dir"),self.name+"_attn_v_acc.in")
                del self.accu_qk
                del self.accu_qk_acc
                del self.accu_v
                del self.accu_v_acc

        if not self.is_softmax:
            x = multi1(attn,v,(acc_attn).float(),(self.v_IF.acc_q*self.v_IF.q_threshold).float())
        else:
            x = multi1(attn,v,(self.attn_softmax_IF.acc_q*self.attn_softmax_IF.q_threshold).float(),(self.v_IF.acc_q*self.v_IF.q_threshold).float())

        x = self.after_attn_IF(x)
        if self.first:
            self.accu_attn.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:    
                save_input_for_bin_snn_5dim(torch.stack(self.accu_attn), glo.get_value("output_bin_snn_dir"),self.name+"_attn.out")
                del self.accu_attn

        x = x.transpose(1, 2).reshape(B, N, C)

        if self.first:
            self.accu_proj_input.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:    
                save_input_for_bin_snn_4dim(torch.stack(self.accu_proj_input), glo.get_value("output_bin_snn_dir"),self.name+"_proj.in")
                del self.accu_proj_input

        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.proj_IF(x)

        if self.first:
            self.accu_proj.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:    
                save_input_for_bin_snn_4dim(torch.stack(self.accu_proj), glo.get_value("output_bin_snn_dir"),self.name+"_proj.out")
                del self.accu_proj
                self.first = False
                local_rank = torch.distributed.get_rank()
                if local_rank == 0:
                    torch.save(self.qkv.quan_w_fn(self.qkv.weight),f'{glo.get_value("output_bin_snn_dir")}/{self.name}_qkv_weight.pth')
                    torch.save(self.qkv.bias,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_qkv_bias.pth')
                    torch.save(self.proj.quan_w_fn(self.proj.weight),f'{glo.get_value("output_bin_snn_dir")}/{self.name}_proj_weight.pth')
                    torch.save(self.proj.bias,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_proj_bias.pth')

        return x

class SpikeMaxPooling(nn.Module):
    def __init__(self,maxpool):
        super(SpikeMaxPooling,self).__init__()
        self.maxpool = maxpool
        
        self.accumulation = None
    
    def reset(self):
        self.accumulation = None

    def forward(self,x):
        old_accu = deepcopy(self.accumulation)
        if self.accumulation is None:
            self.accumulation = x
        else:
            self.accumulation = self.accumulation + x
        
        if old_accu is None:
            output = self.maxpool(self.accumulation)
        else:
            output = self.maxpool(self.accumulation) - self.maxpool(old_accu)

        # print("output.shape",output.shape)
        # print(output[0][0][0:4][0:4])
        
        return output


class Addition(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    
    def forward(self,x):
        return x[0]+x[1]


class QuanConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        else:
            self.bias = None

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return self._conv_forward(x, quantized_weight, self.bias)


class QuanLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, quan_w_fn=None):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return torch.nn.functional.linear(x, quantized_weight, self.bias)


class LLConv2d(nn.Module):
    def __init__(self,conv,**kwargs):
        super(LLConv2d,self).__init__()
        self.conv = conv
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = 1
        self.realize_time = self.steps
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        # self.quan_w_fn = self.conv.quan_w_fn
        
    def reset(self):
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self,input):
        # print("LLConv2d.steps",self.steps)
        x = input
        N,C,H,W = x.shape
        F_h,F_w = self.conv.kernel_size
        S_h,S_w = self.conv.stride
        P_h,P_w = self.conv.padding
        C = self.conv.out_channels
        H = math.floor((H - F_h + 2*P_h)/S_h)+1
        W = math.floor((W - F_w + 2*P_w)/S_w)+1

        if self.zero_output is None:
            # self.zero_output = 0.0
            self.zero_output = torch.zeros(size=(N,C,H,W),device=x.device,dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        output = self.conv(x)

        if self.neuron_type == 'IF':
            pass
        else:
            if self.conv.bias is None:
                pass
            else:
                # if not self.first:
                #     output = output - self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                output = output - (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if self.conv.bias is not None else 0.0)
                if self.realize_time > 0:
                    output = output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
                    self.realize_time = self.realize_time - 1
                    # print("conv2d self.realize_time",self.realize_time)
                    

        self.is_work = True
        self.first = False

        return output

class LLLinear(nn.Module):
    def __init__(self,linear,**kwargs):
        super(LLLinear,self).__init__()
        self.linear = linear
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = 1
        self.realize_time = self.steps
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        # self.quan_w_fn = self.linear.quan_w_fn
        
    def reset(self):
        # print("LLLinear reset")
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self,input):
        # print("LLLinear.steps",self.steps)
        x = input
        # if x.ndim == 2:
        #     B,N = x.shape
        # elif x.ndim == 3:
        #     B,C,N = x.shape
        # N = self.linear.out_features
        if x.dim() == 3:
            B, N, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, N, D)
        elif x.dim() == 2:
            B, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, D)
        if self.zero_output is None:
            self.zero_output = torch.zeros(size=shape_new,device=x.device,dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            return self.zero_output

        output = self.linear(x)

        if self.neuron_type == 'IF':
            pass
        else:
            if self.linear.bias is None:
                pass
            else:
                if self.realize_time == self.steps:
                    output = output
                    self.realize_time = 0
                else:
                    output = output - (self.linear.bias.data.unsqueeze(0) if self.linear.bias is not None else 0.0)


        self.is_work = True
        self.first = False

        return output


class save_module_inout(nn.Module):
    def __init__(self, m, T):
        super().__init__()
        self.m = m
        self.T = T
        self.name = ""
        self.t = 0
        self.accu = []
        self.accu2 = []
        self.accu1 = []
        self.first = False
    
    def forward(self,x):
        if isinstance(self.m[0],Addition):
            dimNum = len(x[0].shape) + 1
        else:
            dimNum = len(x.shape) + 1
        self.t = self.t + 1
        if self.first:
            if isinstance(self.m[0],Addition):
                self.accu.append(x[0][0].unsqueeze(0)+0)
            else:
                self.accu.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                if dimNum == 3:
                    save_fc_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                if dimNum == 4:
                    save_input_for_bin_snn_4dim(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                if dimNum == 5:
                    save_input_for_bin_snn_5dim(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
            if isinstance(self.m[0],Addition):
                self.accu2.append(x[1][0].unsqueeze(0)+0)
                if self.t == self.T:
                    if dimNum == 3:
                        save_fc_input_for_bin_snn(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+"input2.in")
                    if dimNum == 4:
                        save_input_for_bin_snn_4dim(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+"input2.in")
                    if dimNum == 5:
                        save_input_for_bin_snn_5dim(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+"input2.in")
                    del self.accu2
                
        x = self.m(x)
        if self.first:
            self.accu1.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                if dimNum == 3:
                    save_fc_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                if dimNum == 4:
                    save_input_for_bin_snn_4dim(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                if dimNum == 5:
                    save_input_for_bin_snn_5dim(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                self.first = False

                # saving weight and bias
                local_rank = torch.distributed.get_rank()
                if local_rank == 0 and not isinstance(self.m[0],Addition):
                    if hasattr(self.m[0], "quan_w_fn"):
                        torch.save(self.m[0].quan_w_fn(self.m[0].weight),f'{glo.get_value("output_bin_snn_dir")}/{self.name}_weight.pth')
                    else:
                        torch.save(self.m[0].weight,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_weight.pth')
                        
                    if self.m[0].bias is not None:
                        torch.save(self.m[0].bias,f'{glo.get_value("output_bin_snn_dir")}/{self.name}_bias.pth')
                        
                del self.accu
                del self.accu1
        return x

        

class Attention_no_softmax(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_Relu = nn.ReLU(inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_Relu(attn)/N
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, dim, **kwargs):
        super(MyBatchNorm1d, self).__init__(dim, **kwargs)
    
    def forward(self,x):
        x = x.transpose(1,2)
        F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,self.training,self.momentum,self.eps)
        x = x.transpose(1,2)
        return x
        
    

class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)
        self.running_mean = None
        self.running_var = None
        self.momentum = 0.9
        self.eps = 1e-6
    
    def forward(self,x):        
        if self.training:
            if self.running_mean is None:
                self.running_mean = nn.Parameter((1-self.momentum) * x.mean([-1], keepdim=True),requires_grad=False)
                self.running_var = nn.Parameter((1-self.momentum) * x.var([-1], keepdim=True),requires_grad=False)
            else:
                self.running_mean.data = (1-self.momentum) * x.mean([-1], keepdim=True) + self.momentum * self.running_mean # mean: [1, max_len, 1]
                self.running_var.data = (1-self.momentum) * x.var([-1], keepdim=True) + self.momentum * self.running_var # std: [1, max_len, 1]
            return self.weight * (x - self.running_mean) / (self.running_var + self.eps) + self.bias
        else:
            # if self.running_mean is None:
            self.running_mean = nn.Parameter(x.mean([-1], keepdim=True),requires_grad=False)
            self.running_var = nn.Parameter(x.var([-1], keepdim=True),requires_grad=False)
            running_mean = self.running_mean
            running_var = self.running_var
            return self.weight * (x) / (running_var + self.eps).sqrt() + self.bias    
        # 注意这里也在最后一个维度发生了广播

def save_input_for_bin_snn_5dim(input,dir,name):
    T,B,L1,L2,N = input.shape
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        torch.save(input,f'{dir}/act_{name}_T={T}_B={B}_L1={L1}_L2={L2}_N={N}.pth')

    
def save_input_for_bin_snn_4dim(input,dir,name):
    T,B,L,N = input.shape
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        torch.save(input,f'{dir}/act_{name}_T={T}_B={B}_L={L}_N={N}.pth')
    
def save_fc_input_for_bin_snn(input,dir,name):
    T,B,N = input.shape
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        torch.save(input,f'{dir}/act_{name}_T={T}_B={B}_N={N}.pth')

