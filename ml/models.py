import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from data import Data
from torch.linalg import solve

class Z(nn.Module):
    def __init__(self, data:Data):
        super().__init__()
        self.data = data
    
    def forward(self,params:Parameter):
        #TODO: make this more efficient by eliminating non-resistor params
        Z_r = torch.diag(params) @ torch.diag(self.data.r_mask).to(torch.float)
        Z_ics = torch.diag(self.data.ics_mask)
        return -Z_r + Z_ics

class Y(nn.Module):
    def __init__(self, data:Data):
        super().__init__()
        self.Y_r = torch.diag(data.r_mask).float()
        self.Y_ivs = torch.diag(data.ivs_mask).float()
    
    def forward(self):
        return self.Y_r + self.Y_ivs

class E(nn.Module):
    def __init__(self, data:Data):
        super().__init__()
        self.Z = Z(data)
        self.Y = Y(data)
        
    def forward(self,params):
        return torch.cat(tensors=(self.Z(params),self.Y()),dim=1)

class A(nn.Module):
    def __init__(self, data: Data):
        super().__init__()
        M_zeros = torch.zeros_like(data.M)
        kvl_coef = torch.tensor(data.circuit.kvl_coef()).to(torch.float)
        kvl_zeros = torch.zeros_like(kvl_coef)
        self.kcl = torch.cat(tensors=(data.M,M_zeros),dim=1)
        self.kvl = torch.cat(tensors=(kvl_zeros,kvl_coef),dim=1)
        self.elements = E(data)
    
    def forward(self,params):
        return torch.cat(tensors=(self.kcl,self.kvl,self.elements(params)), dim=0)

class S(nn.Module):
    def __init__(self, data: Data):
        super().__init__()
        self.data = data

    def forward(self,input:Tensor):
        i,v = self.data.split_input_output(input)
        ics = i * self.data.ics_mask.unsqueeze(1)
        ivs = v * self.data.ivs_mask.unsqueeze(1)
        return ics + ivs
    
class B(nn.Module):
    def __init__(self, data: Data):
            super().__init__()
            num_nodes = data.circuit.num_nodes()
            num_elements = data.circuit.num_elements()
            self.kcl = torch.zeros(size=(num_nodes,1))
            self.kvl = torch.zeros(size=(num_elements - num_nodes + 1,1))
            self.sources = S(data)
    
    def forward(self,input:Tensor):
        s = self.sources(input)
        b = torch.cat(tensors=(self.kcl,self.kvl,s), dim=0)
        return b

class Cell(nn.Module):
    def __init__(self, data: Data):
        super().__init__()
        self.data = data
        self.params = data.init_params()
        self.A = A(data)
        self.b = B(data)

    def forward(self,input):
        A = self.A(self.params)
        b = self.b(input)
        out = solve(A[1:,:],b[1:,:])
        return out
    
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
        # self.params = Parameter(params)
