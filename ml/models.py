import torch
import torch.nn as nn
import circuits as ckt


class Solver(nn.Module):
    def __init__(self, input:ckt.Input):
        super().__init__()
        self.input = input
        self.i:nn.Parameter = None
        self.v:nn.Parameter = None
        self.pot:nn.Parameter = None
        self.attr:nn.Parameter = None

    def get_params(self):
        return (self.get_one_param(self.i),
                self.get_one_param(self.v),
                self.get_one_param(self.pot),
                self.get_one_param(self.attr))

    def get_one_param(self,param:nn.Parameter):
        if(param != None):
            return param
        else:
            return None
    
    def kcl(self):
        return self.input.M @ self.i

    def kvl(self):
        return self.v - self.input.M.T @ self.pot

    def resistor(self):
        res_mask = self.input.kinds_map[ckt.Kinds.R]
        return self.i[res_mask] * self.attr[res_mask] - self.v[res_mask]

    def zero_known_grads(self):
        if(self.i != None and self.i.grad != None):
            self.i.grad[self.input.knowns_map[ckt.Props.I]] = 0
        if(self.v != None and self.v.grad != None):
            self.v.grad[self.input.knowns_map[ckt.Props.V]] = 0
        if(self.attr != None and self.attr.grad != None):
            self.attr.grad[self.input.knowns_map[ckt.Props.Attr]] = 0

class KCL(Solver):
    def __init__(self, input:ckt.Input, params:tuple[nn.Parameter]):
        super().__init__(input)
        self.i = nn.Parameter(params[0])
        self.v = params[1].clone().detach().requires_grad_(False)
        self.pot = params[2].clone().detach().requires_grad_(False)
        self.attr = params[3].clone().detach().requires_grad_(False)

    def forward(self):
        return self.input.M @ self.i

class KVL(Solver):
    def __init__(self, input:ckt.Input, params:tuple[nn.Parameter]):
        super().__init__(input)
        self.i = params[0].clone().detach().requires_grad_(False)
        self.v = nn.Parameter(params[1])
        self.pot = nn.Parameter(params[2])
        self.attr = params[3].clone().detach().requires_grad_(False)

    def forward(self):
        return self.v - self.input.M.T @ self.pot

class Full(Solver):
    def __init__(self, input:ckt.Input, params:tuple[nn.Parameter]):
        super().__init__(input)
        self.i = nn.Parameter(params[0])
        self.v = nn.Parameter(params[1])
        self.pot = nn.Parameter(params[2])
        self.attr = nn.Parameter(params[3])

    def forward(self):
        return torch.cat(tensors = (self.kcl(), self.kvl(), self.resistor()))