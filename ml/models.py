import torch
import torch.nn as nn
from torch import Tensor
from circuits import Props,Kinds
from data import Data
from torch.linalg import solve

class Solver(nn.Module):
    def __init__(self, data: Data):
        super().__init__()
        self.data = data
        self.ics_attr_mask = self.init_mask(Kinds.ICS)
        self.ivs_attr_mask = self.init_mask(Kinds.IVS)
        self.r_attr_mask = self.init_mask(Kinds.R)
        self.known_attr_mask = self.init_known_attr_mask()
        (self.i_base,
        self.v_base,
        self.r_base) = self.data.init_base()
        self.attr = nn.Parameter(self.init_attr())

    def init_mask(self, kind:Kinds):
        return torch.tensor(self.data.mask_of_kind(kind)).to(torch.bool)
    
    def init_known_attr_mask(self):
        ics = torch.tensor(self.data.mask_of_attr(Kinds.ICS))
        ivs = torch.tensor(self.data.mask_of_attr(Kinds.IVS))
        r = torch.tensor(self.data.mask_of_attr(Kinds.R))
        return torch.logical_or(ics,torch.logical_or(ivs,r))

    def init_attr(self):
        ics_list = self.data.attr_list(Kinds.ICS,1)
        ivs_list = self.data.attr_list(Kinds.IVS,1)
        r_list = self.data.attr_list(Kinds.R,1)
        ics_tensor = torch.tensor(ics_list).to(torch.float)
        ics_tensor[~self.ics_attr_mask] = 0
        ics_tensor = ics_tensor/self.i_base
        ivs_tensor = torch.tensor(ivs_list).to(torch.float)
        ivs_tensor[~self.ivs_attr_mask] = 0
        ivs_tensor = ivs_tensor/self.v_base
        r_tensor = torch.tensor(r_list).to(torch.float)
        r_tensor[~self.r_attr_mask] = 0
        r_tensor = r_tensor/self.r_base
        attr_tensor = ics_tensor + ivs_tensor + r_tensor
        attr_tensor[~self.known_attr_mask] = 1
        return attr_tensor

    def split_solution(self, solution:Tensor):
        split = self.data.circuit.num_elements()
        i = solution[:split,:]
        v = solution[split:2*split,:]
        return i,v
    
    def denorm_solution(self,solution):
        i_norm,v_norm = self.split_solution(solution)
        i = self.denorm(i_norm,self.i_base)
        v = self.denorm(v_norm,self.v_base)
        return i,v

    def norm(self, input:Tensor, base:float):
        clone = input.detach().clone()
        return clone/base

    def denorm(self, input:Tensor, base:float):
        clone = input.detach().clone()
        return clone*base
    
    def denorm_attr(self):
        attr = self.attr.clone()
        attr[self.ics_attr_mask] = self.denorm(attr[self.ics_attr_mask],self.i_base)
        attr[self.ivs_attr_mask] = self.denorm(attr[self.ivs_attr_mask],self.v_base)
        attr[self.r_attr_mask] = self.denorm(attr[self.r_attr_mask],self.r_base)
        return attr
    
    def forward(self):
        A,b = self.build()
        return solve(A[1:,:],b[1:,:])
    
    def zero_known_grads(self):
        if(self.attr != None and self.attr.grad != None):
            self.attr.grad[self.known_attr_mask] = 0

    def clamp_attr(self):
        min = 1e-15
        max = 1e15
        for p in self.parameters():
            p.data.clamp_(min, max)

    def M(self):
        return self.data.M

    def build(self):
        num_nodes = self.data.circuit.num_nodes()
        kcl_coef = self.M()
        M_zeros = torch.zeros_like(kcl_coef)
        kcl_block = torch.cat(tensors=(kcl_coef,M_zeros),dim=1)
        kvl_coef = torch.tensor(self.data.circuit.kvl_coef()).to(torch.float)
        kvl_zeros = torch.zeros_like(kvl_coef)
        kvl_block = torch.cat(tensors=(kvl_zeros,kvl_coef),dim=1)
        element_coef = self.E()
        A = torch.cat(tensors=(
                kcl_block,
                kvl_block,
                element_coef
            ), dim=0)
        kcl_zeros = torch.zeros(num_nodes)
        kvl_zeros = torch.zeros(kvl_block.shape[0])
        s_const = self.source_attr()
        b = torch.cat(tensors=(
                kcl_zeros,
                kvl_zeros,
                s_const
            ), dim=0)
        return A,b.unsqueeze(dim=1)
    
    def E(self):
        Z = self.Z()
        Y = self.Y()
        return torch.cat(tensors=(Z,Y),dim=1)

    def source_attr(self):
        source_mask = torch.logical_or(self.ics_attr_mask,self.ivs_attr_mask)
        return self.attr * source_mask

    def Z(self):
        Z_r = torch.diag(self.attr) @ torch.diag(self.r_attr_mask).to(torch.float)
        Z_ics = torch.diag(self.ics_attr_mask)
        return -Z_r + Z_ics

    def Y(self):
        Y_r = torch.diag(self.r_attr_mask)
        Y_ivs = torch.diag(self.ivs_attr_mask)
        return Y_r + Y_ivs
    
    def set_r_from_knowns(self, preds:Tensor,target:Tensor,target_mask:Tensor):
        with torch.no_grad():
            i,v = self.split_solution(preds)
            i = i.flatten()
            v = v.flatten()
            i_known,v_known = self.split_solution(target)
            i_known = i_known.flatten()
            v_known = v_known.flatten()
            i_known_mask,v_known_mask = self.split_solution(target_mask)
            i_known_mask = i_known_mask.flatten()
            v_known_mask = v_known_mask.flatten()
            i[i_known_mask] = i_known[i_known_mask]
            v[v_known_mask] = v_known[v_known_mask]
            self.attr[self.r_attr_mask] = v[self.r_attr_mask]/i[self.r_attr_mask]

