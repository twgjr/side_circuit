import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from models import Solver
from data import Data

class Stability():
    '''takes attributes from solver and determines if the all attributes have 
    converged to a stable value.'''
    def __init__(self, init_tensor:Tensor, pu_threshold:float) -> None:
        self.tensor = init_tensor.detach().clone()
        self.pu_threshold = pu_threshold
    
    def is_stable(self,tensor:Tensor):
        new_tensor = tensor.detach().clone()
        abs_val = torch.abs(new_tensor - self.tensor)
        prev_abs = torch.abs(self.tensor)
        pu_change = abs_val / prev_abs
        pu_change_max = torch.max(pu_change)
        ret_bool = pu_change_max < self.pu_threshold
        self.tensor = new_tensor
        return ret_bool

class Trainer():
    def __init__(self, data:Data, init_learn_rate:float) -> None:
        self.data = data
        self.model = Solver(data=data)
        self.optimizer = Adam(params=self.model.parameters(),lr=init_learn_rate)
        self.loss_fn = nn.MSELoss()

    def run(self, epochs, stable_threshold:float):
        self.model.train()
        target = torch.tensor(
            self.model.data.target_list(
                self.model.i_base, self.model.v_base
            )).to(torch.float).unsqueeze(dim=1)
        target_mask = torch.tensor(
            self.model.data.target_mask_list()).to(torch.bool).unsqueeze(dim=1)
        loss, attr, preds = self.step(target,target_mask)
        attr_stability = Stability(attr, stable_threshold)
        preds_stability = Stability(preds, stable_threshold)
        epoch = 0
        reset_count = 0
        while(epoch < epochs):
            if(epoch % 2 == 1):
                if(attr_stability.is_stable(attr) and 
                    preds_stability.is_stable(preds)):
                    print('threshold met')
                    break
                self.model.set_r_from_knowns(preds,target,target_mask)
                reset_count += 1
            loss, attr, preds = self.step(target,target_mask)
            epoch += 1
        print(f'reset_count = {reset_count}')
        self.model.set_r_from_knowns(preds,target,target_mask)
        i_sol, v_sol = self.model.denorm_solution(preds)
        a_sol = self.model.denorm_attr()
        return i_sol.squeeze(dim=1), v_sol.squeeze(dim=1), a_sol, loss, epoch

    def step(self,target:Tensor,target_mask:Tensor):
        preds = self.model.forward()
        loss = self.loss_fn(preds[target_mask], target[target_mask])
        self.model.zero_grad()
        loss.backward()
        self.model.zero_known_grads()
        self.optimizer.step()
        self.model.clamp_attr()
        return loss,self.model.attr,preds
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        
    def set_lr(self,optimizer,lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
