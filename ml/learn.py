import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from circuits import System,Props
from models import DynamicModule,SystemModule
from data import Data

class Stability():
    '''takes attributes from solver and determines if the all attributes have 
    converged to a stable value.'''
    def __init__(self, tensor_list:list[Tensor], pu_threshold:float) -> None:
        self.tensor_list = []
        if(len(tensor_list) > 0):
            for tensor in tensor_list:
                self.tensor_list.append(tensor.detach().clone())
        self.pu_threshold = pu_threshold
    
    def is_stable(self,tensor_list:list[Tensor]):
        assert len(self.tensor_list) > 0
        ret_bool = True
        for index in range(len(tensor_list)):
            new_tensor = tensor_list[index].detach().clone()
            abs_val = torch.abs(new_tensor - self.tensor_list[index])
            prev_abs = torch.abs(self.tensor_list[index])
            pu_change = abs_val / prev_abs
            pu_change_max = torch.max(pu_change)
            if(pu_change_max > self.pu_threshold):
                ret_bool = False
        self.tensor_list = tensor_list
        return ret_bool

class Trainer():
    def __init__(self, system:System, learn_rate:float) -> None:
        self.model = DynamicModule(system)
        self.data = Data(system,self.model)
        self.learn_rate = learn_rate
        self.optimizer = None
        self.loss_fn = nn.MSELoss()

    def init_optimizer(self):
        self.optimizer = Adam(params=self.model.parameters(),lr=self.learn_rate)
    
    def step_sequence(self):
        '''Returns the total loss, the model parameters, and the output of the 
        model for all time steps.'''
        system_sequence,sw_err_out,delta_err_out = self.model.forward()
        sequence_loss_list = []
        for time,sys_t in system_sequence.items():
            for c,ckt_t_out in enumerate(sys_t):
                ckt_t_data = self.data.sequence[time].circuits[c]
                ckt_t_mask = self.data.masks[c]
                for key in ckt_t_mask:
                    if(key==Props.A):
                        continue
                    pred_mask = ckt_t_mask[key]
                    pred_prop = ckt_t_out[key]
                    pred_knowns = pred_prop[pred_mask]
                    if(len(pred_knowns)==0):continue
                    knowns = ckt_t_data[key]
                    loss = self.loss_fn(pred_knowns, knowns)
                    sequence_loss_list.append(loss)
        sw_loss = self.loss_fn(sw_err_out, torch.zeros_like(sw_err_out))
        delta_loss = self.loss_fn(delta_err_out, torch.zeros_like(delta_err_out))
        total_loss = sum(sequence_loss_list) + sw_loss + delta_loss
        self.model.zero_grad()
        # if(sw_loss.requires_grad):
        #     sw_loss.backward(retain_graph=True)
        # if(delta_loss.requires_grad):
        #     delta_loss.backward(retain_graph=True)
        # if(seq_loss.requires_grad):
        #     seq_loss.backward()
        total_loss.backward()
        self.init_optimizer()
        self.optimizer.step()
        self.model.clamp_params()
        return total_loss, system_sequence
    
    def run(self, epochs, stable_threshold:float):
        # self.model.train()
        # params = list(self.model.parameters())
        loss, pred_list = self.step_sequence()
        # param_stability = Stability(params, stable_threshold)
        epoch = 0
        while(epoch < epochs):
            loss, pred_list = self.step_sequence()
            epoch += 1
            # params = list(self.model.parameters())
            # if(epoch % 200 == 199):
            #     if(param_stability.is_stable(params)):
            #         print('threshold met')
            #         break
        return pred_list, loss, epoch
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        
    def set_lr(self,optimizer,lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
