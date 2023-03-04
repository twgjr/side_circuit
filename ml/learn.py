import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from torch import Tensor
from models import Solver
from models import State
from data import Data

class Trainer():
    def __init__(self,data:Data) -> None:
        self.input = data
        self.model:Solver = Solver(data=data, state=State.Solve)
        self.optimizer:Adam = Adam(params=self.model.parameters(),lr=self.init_LR())
        self.loss_fn:nn.MSELoss = nn.MSELoss()

    def init_LR(self):
        max_base = max(self.model.v_base,self.model.i_base,self.model.r_base)
        rate = 1/(10*max_base)
        if(0.01 < rate):
            rate = 0.01
        if(rate < 0.00001):
            rate = 0.00001
        print(rate)
        return rate

    def run(self, epochs):
        loss = self.step()
        epoch = 0
        while(epoch < epochs):
            loss, _ = self.step()
            if(loss < 1e-20):
                    break
            epoch += 1
        i_sol, v_sol = self.model.denorm_solution(self.model())
        a_sol = self.model.attr*self.model.r_base_from_i_v()
        return i_sol, v_sol, a_sol, loss, epoch

    def step(self):
        self.model.train()
        loss = None
        if(self.model.state == State.Solve):
            preds = self.model()
            target = torch.tensor(self.model.data.target).to(torch.float).unsqueeze(dim=1)
            mask = torch.tensor(self.model.data.target_mask).to(torch.bool).unsqueeze(dim=1)
            loss = self.loss_fn(preds[mask[:-1]], target[mask])
        elif(self.model.state == State.Lstsq):
            A,preds,b = self.model()
            loss = self.loss_fn(A @ preds, b)
        else:
            assert()
        self.model.zero_grad()
        loss.backward()
        self.model.zero_known_grads()
        self.optimizer.step()
        self.model.clamp_attr()
        return loss,self.model.attr
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def is_stable(self,prev_tpl:tuple[nn.Parameter], updated_tpl:tuple[nn.Parameter], 
                pu_threshold:float):
        updated = torch.cat(tensors=self.list_params_filter_none(updated_tpl))
        prev = torch.cat(tensors=self.list_params_filter_none(prev_tpl))
        abs_val = torch.abs(updated - prev)
        prev_abs = torch.abs(prev)
        pu_change = abs_val / prev_abs
        pu_change_max = torch.max(pu_change)
        ret_bool = pu_change_max < pu_threshold
        return ret_bool
