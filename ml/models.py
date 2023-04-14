import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from data import Data
from torch.linalg import solve

class Switch(nn.Module):
    def __init__(self, data:Data):
        super().__init__()
        self.data = data

    def forward(self, triggers:Tensor):
        triggers = triggers.float()
        return torch.diag(triggers) @ torch.diag(self.data.vcsw_mask).float()
    
class VoltageControl(nn.Module):
    def __init__(self, data:Data):
        super().__init__()
        self.data = data
    
    def forward(self):
        return torch.diag(self.data.vc_mask).float()

class Impedance(nn.Module):
    def __init__(self, data:Data):
        super().__init__()
        self.data = data
        self.Z_vcsw = Switch(data)
        self.Z_vc = VoltageControl(data)
        self.Z_ics = torch.diag(self.data.ics_mask)
    
    def forward(self,params:Parameter,triggers:Tensor):
        Z_r = torch.diag(params) @ torch.diag(self.data.r_mask).to(torch.float)
        return -Z_r + self.Z_ics + self.Z_vcsw(triggers) + self.Z_vc()

class Admittance(nn.Module):
    def __init__(self, data:Data):
        super().__init__()
        self.data = data
        self.Y_r = torch.diag(data.r_mask).float()
        self.Y_ivs = torch.diag(data.ivs_mask).float()
        self.Y_vcsw = Switch(data)
    
    def forward(self, triggers:Tensor):
        return self.Y_r + self.Y_ivs + self.Y_vcsw(~triggers)

class Elements(nn.Module):
    def __init__(self, data:Data):
        super().__init__()
        self.data = data
        self.params = data.init_params()
        self.Z = Impedance(data)
        self.Y = Admittance(data)
        
    def forward(self, i_in:Tensor, v_in:Tensor):
        triggers = self.triggers(i_in,v_in)
        return torch.cat(tensors=(
            self.Z(self.params,triggers),self.Y(triggers)),dim=1)
    
    def triggers(self, i_in:Tensor, v_in:Tensor):
        v_in_flat = v_in.flatten()
        i_in_flat = i_in.flatten()
        v_ctrl_idx = torch.tensor(self.data.v_control_list).to(torch.int64)
        i_ctrl_idx = torch.tensor(self.data.i_control_list).to(torch.int64)
        v_in_reordered = torch.gather(dim=0, index=v_ctrl_idx, input=v_in_flat)
        i_in_reordered = torch.gather(dim=0, index=i_ctrl_idx, input=i_in_flat)
        v_ctrl = v_in_reordered * self.data.vcsw_mask
        i_ctrl = i_in_reordered * self.data.ccsw_mask
        v_is_trig = v_ctrl > 0
        i_is_trig = i_ctrl > 0
        is_trig = v_is_trig | i_is_trig
        return is_trig
    
    def zero_known_grads(self):
        if(self.params != None and self.params.grad != None):
            self.params.grad[self.data.attrs_mask] = 0

    def clamp_params(self):
        min = 1e-15
        max = 1e15
        for p in self.parameters():
            p.data.clamp_(min, max)

    def set_param_vals(self, params: Tensor):
        assert params.shape == self.params.shape
        for p in self.parameters():
            p.data = params

class Coefficients(nn.Module):
    def __init__(self, data: Data):
        super().__init__()
        M_zeros = torch.zeros_like(data.M)
        kvl_coef = torch.tensor(data.circuit.kvl_coef()).to(torch.float)
        kvl_zeros = torch.zeros_like(kvl_coef)
        self.kcl = torch.cat(tensors=(data.M,M_zeros),dim=1)
        self.kvl = torch.cat(tensors=(kvl_zeros,kvl_coef),dim=1)
        self.elements = Elements(data)
    
    def forward(self, i_in:Tensor, v_in:Tensor):
        return torch.cat(tensors=(
            self.kcl,self.kvl,self.elements(i_in,v_in)), dim=0)

class Sources(nn.Module):
    def __init__(self, data: Data):
        super().__init__()
        self.data = data

    def forward(self, i_in:Tensor, v_in:Tensor):
        ics_out = i_in * self.data.ics_mask.unsqueeze(1)
        ivs_out = v_in * self.data.ivs_mask.unsqueeze(1)
        return ics_out + ivs_out
    
class Constants(nn.Module):
    def __init__(self, data: Data):
            super().__init__()
            num_nodes = data.circuit.num_nodes()
            num_elements = data.circuit.num_elements()
            self.kcl = torch.zeros(size=(num_nodes,1))
            self.kvl = torch.zeros(size=(num_elements - num_nodes + 1,1))
            self.sources = Sources(data)
    
    def forward(self, i_in:Tensor, v_in:Tensor):
        s = self.sources(i_in, v_in)
        b = torch.cat(tensors=(self.kcl,self.kvl,s), dim=0)
        return b

class Cell(nn.Module):
    def __init__(self, data: Data):
        super().__init__()
        self.data = data
        self.A = Coefficients(data)
        self.b = Constants(data)

    @property
    def params(self):
        return self.A.elements.params

    def forward(self,input:Tensor):
        i_in, v_in = self.data.split_input_output(input)
        A = self.A(i_in, v_in)
        b = self.b(i_in, v_in)
        solution_out = solve(A[1:,:],b[1:,:])
        return solution_out
    
    def zero_known_grads(self):
        self.A.elements.zero_known_grads()

    def clamp_params(self):
        self.A.elements.clamp_params()

    def set_param_vals(self, params: Tensor):
        self.A.elements.set_param_vals(params)