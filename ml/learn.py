import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from torch import Tensor
from models import Solver
from models import State
from data import Preprocess

class Trainer():
    def __init__(self,input:Preprocess) -> None:
        self.input = input
        self.model:Solver = Solver(input=input, state=State.Solve)
        self.optimizer:Adam = Adam(params=self.model.parameters(),lr=self.init_LR())
        self.scheduler:MultiStepLR = None
        # self.scheduler:MultiStepLR = MultiStepLR(self.optimizer, milestones=[250], gamma=0.01)
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
        loss, _ = self.step()
        print(f'init target: {self.model.input.target}')
        print(f'init params: {self.model.attr}')
        print(f'init loss: {loss.item()}')
        prev_loss = 0.1
        epoch = 0
        while(epoch < epochs):
            loss, _ = self.step()
            if(loss < 1e-20):
                    break
            epoch += 1
            loss_change = abs(loss - prev_loss) / prev_loss
            prev_loss = loss

        print(f'Done! at {epoch} passes, {loss.item()} loss')
        i_sol, v_sol = self.model.denorm_solution(self.model())
        a_sol = self.model.attr*self.model.r_base_from_i_v()
        print('i, v, attr')
        print(torch.cat(tensors=(i_sol,v_sol,a_sol.unsqueeze(dim=1)),dim=1))

    def step(self):
        self.model.train()
        loss = None
        if(self.model.state == State.Solve):
            preds = self.model()
            target = torch.tensor(self.model.input.target).to(torch.float).unsqueeze(dim=1)
            mask = torch.tensor(self.model.input.target_mask).to(torch.bool).unsqueeze(dim=1)
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
        if(self.scheduler!=None):
            self.scheduler.step()
        return loss,self.model.attr
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def list_params_filter_none(self,param_tpl:tuple[nn.Parameter]):
        param_lst = []
        for param in param_tpl:
            if(param == None):
                continue
            param_lst.append(param[param!=None])
        return param_lst

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

class Tensor_Op():
    def __init__(self, input:Preprocess) -> None:
        self.input = input

    def errors(self, prediction:Tensor, known_input: Tensor, known_mask:Tensor):
        errors = known_input[:-1] - prediction
        errors[~known_mask[:-1]] = 0
        return errors
    
    def split(self, ivp:Tensor):
        '''split a tensor that is (currents, voltages, potentials)
        Could be errors or predictions in that format'''
        num_elem = self.input.circuit.num_elements()
        i = ivp[:num_elem,:]
        v = ivp[num_elem:num_elem*2,:]
        p = ivp[2*num_elem:,:]
        return i,v,p

    def diffuse(self, prediction:Tensor, self_loops:bool):
        A = self.input.circuit.A_edge_row_norm(self_loops = self_loops, torch_type=torch.float)
        return A.T @ A @ prediction
    
    def reset_known_error(self, error, diffusion, mask):
        diffusion[mask] = error[mask]
        return diffusion

    def propagate(self, errors: Tensor):
        num_elem = self.input.circuit.num_elements()
        i_errors = errors[:num_elem,:]
        v_errors = errors[num_elem:num_elem*2,:]
        return v_errors / i_errors
