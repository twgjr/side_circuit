import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from models import Cell
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
        return ret_bool

class Trainer():
    def __init__(self, data:Data, init_learn_rate:float) -> None:
        self.data = data
        self.model = Cell(data=data)
        self.optimizer = Adam(params=self.model.parameters(),lr=init_learn_rate)
        self.loss_fn = nn.MSELoss()
        self.dataset = data.init_dataset()
        self.mask = torch.tensor(data.data_mask_list()).to(torch.bool)

    def run(self, epochs, stable_threshold:float):
        self.model.train()
        loss = None
        params = self.model.params
        out_list = []
        attr_stability = Stability(params, stable_threshold)
        preds_stability = Stability(out_list, stable_threshold)
        epoch = 0
        reset_count = 0
        while(epoch < epochs):
            if(epoch % 2 == 1):
                if(attr_stability.is_stable([params]) and 
                    preds_stability.is_stable(out_list)):
                    print('threshold met')
                    break
                # self.model.set_r_from_knowns(state,target,target_mask)
                reset_count += 1
            loss, out_list = self.step_sequence()
            epoch += 1
            params = self.model.params
        print(f'reset_count = {reset_count}')
        # self.model.set_r_from_knowns(state,target,target_mask)
        i_sol, v_sol = self.data.denorm_input_output(out_list)
        a_sol = self.data.denorm_params(params)
        return i_sol, v_sol, a_sol, loss, epoch

    def step_cell(self,input:Tensor):
        '''processes one time step of the sequence'''
        out = self.model.forward(input)
        loss = self.loss_fn(out[self.mask], input[self.mask])
        return loss,out
    
    def step_sequence(self):
        '''Calls step() for each item in self.dataset. Returns the total
          loss, the model parameters, and the output of the model for each time
          step.'''
        loss_list = []
        input = None
        out_list = []
        for input in self.dataset:
            loss, out = self.step_cell(input)
            out_list.append(out)
            loss_list.append(loss)
        total_loss = sum(loss_list)
        self.model.zero_grad()
        total_loss.backward()
        self.model.zero_known_grads()
        self.optimizer.step()
        self.model.clamp_params()
        return total_loss, out_list
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        
    def set_lr(self,optimizer,lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
