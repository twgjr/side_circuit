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
    def __init__(self, data:Data) -> None:
        self.data = data
        self.model = Solver(data=data)
        self.optimizer = Adam(params=self.model.parameters(),lr=self.init_LR())
        self.loss_fn = nn.MSELoss()

    def init_LR(self):
        max_base = max(self.model.v_base,self.model.i_base,self.model.r_base)
        rate = 1/(10*max_base)
        if(0.01 < rate):
            rate = 0.01
        if(rate < 0.00001):
            rate = 0.00001
        print(rate)
        return rate

    def run(self, epochs, stable_threshold:float, loss_threshold:float):
        loss, attr, preds = self.step()
        attr_stability = Stability(attr, stable_threshold)
        preds_stability = Stability(preds, stable_threshold)
        epoch = 0
        while(epoch < epochs):
            loss, attr, preds = self.step()
            if(loss < loss_threshold):
                    break
            if(attr_stability.is_stable(attr) and 
                preds_stability.is_stable(preds)):
                 break
            epoch += 1
        i_sol, v_sol = self.model.denorm_solution(self.model())
        a_sol = self.model.attr*self.model.r_base_from_i_v()
        return i_sol, v_sol, a_sol, loss, epoch

    def step(self):
        self.model.train()
        preds = self.model()
        target = torch.tensor(self.model.data.target).to(torch.float).unsqueeze(dim=1)
        mask = torch.tensor(self.model.data.target_mask).to(torch.bool).unsqueeze(dim=1)
        loss = self.loss_fn(preds[mask[:-1]], target[mask])
        self.model.zero_grad()
        loss.backward()
        self.model.zero_known_grads()
        self.optimizer.step()
        self.model.clamp_attr()
        return loss,self.model.attr,preds
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
