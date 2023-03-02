import torch
import torch.nn as nn
from torch import Tensor
from circuits import Props,Kinds
from data import Preprocess
from enum import Enum

class State(Enum):
    Init = 0
    Solve = 1
    Lstsq = 2

class Solver(nn.Module):
    ''' 
    Sparse Tableau Formulation of circuit analysis, modeled as a machine learning
    problem to learn element attributes using backprop and optimization.
    '''
    def __init__(self, input: Preprocess, state: State):
        super().__init__()
        self.input = input
        self.ics_mask = self.init_mask(Kinds.ICS)
        self.ivs_mask = self.init_mask(Kinds.IVS)
        self.r_mask = self.init_mask(Kinds.R)
        self.all_knowns_mask = self.init_known_attr_mask()
        self.attr = nn.Parameter(self.init_attr())
        self.state = state

    def init_attr(self):
        ret_tensor = torch.tensor(
            self.input.attr_list(
            Kinds.ICS,replace_nones=True, rand_unknowns=True)).to(torch.float)
        ivs = torch.tensor(
            self.input.attr_list(
            Kinds.IVS,replace_nones=True,rand_unknowns=True)).to(torch.float)
        ret_tensor[self.ivs_mask] = ivs[self.ivs_mask]
        r = torch.tensor(
            self.input.attr_list(
            Kinds.R,replace_nones=True,rand_unknowns=True)).to(torch.float)
        ret_tensor[self.r_mask] = r[self.r_mask]
        return ret_tensor
    
    def init_mask(self, kind:Kinds):
        return torch.tensor(self.input.mask_of_kind(kind))
    
    def init_known_attr_mask(self):
        ics = torch.tensor(self.input.mask_of_attr(Kinds.ICS))
        ivs = torch.tensor(self.input.mask_of_attr(Kinds.IVS))
        r = torch.tensor(self.input.mask_of_attr(Kinds.R))
        return torch.logical_or(ics,torch.logical_or(ivs,r))
    
    def zero_known_grads(self):
        if(self.attr != None and self.attr.grad != None):
            self.attr.grad[self.all_knowns_mask] = 0

    def clamp_attr(self):
        min = 1e-12
        max = 1e9
        for p in self.parameters():
            p.data.clamp_(min, max)

    def forward(self):
        if(self.state == State.Init):
            pass
        elif(self.state == State.Solve):
            A,b = self.build()
            return torch.linalg.solve(A[1:,:-1],b[1:,:])
        elif(self.state == State.Lstsq):
            A,b = self.build()
            return A,torch.linalg.lstsq(A,b).solution,b

    def build(self):
        # inputs
        M = self.input.M
        num_elements = self.input.circuit.num_elements()
        num_nodes = self.input.circuit.num_nodes()
                
        # A matrix
        kcl_row = torch.cat(tensors=(M,
                                    torch.zeros_like(M),
                                    torch.zeros_like(M)),dim=1)
        kvl_row = torch.cat(tensors=(torch.zeros_like(M),
                                    torch.eye(num_elements),
                                    -M.T),dim=1)
        e_row = self.E()

        A = torch.cat(tensors=(
                kcl_row,
                kvl_row,
                e_row,
            ), dim=0)
                
        # b matrix
        kcl_zeros = torch.zeros(num_nodes)
        kvl_zeros = torch.zeros(num_elements)
        b = torch.cat(tensors=(
                kvl_zeros,
                kcl_zeros,
                self.source_attr()
            ), dim=0)
        
        return A,b.unsqueeze(dim=1)
    
    def E(self):
        Z = self.Z()
        Y = self.Y()
        return torch.cat(tensors=(Z,Y,torch.zeros_like(Z)),dim=1)

    def source_attr(self):
        source_mask = torch.logical_or(self.ics_mask,self.ivs_mask)
        return self.attr * source_mask

    def Z(self):
        Z_r = torch.diag(self.attr) @ torch.diag(self.r_mask).to(torch.float)
        Z_ics = torch.diag(self.ics_mask)
        return -Z_r + Z_ics

    def Y(self):
        Y_r = torch.diag(self.r_mask)
        Y_ivs = torch.diag(self.ivs_mask)
        return Y_r + Y_ivs
    
    def list_to_diag(self, input:list):
        vector = self.list_to_vec(input, to_bool=False)
        return torch.diag(vector)