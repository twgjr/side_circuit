import torch
import torch.nn as nn
from circuits import Props
from inputs import Input
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
    def __init__(self, input: Input, attr:nn.Parameter):
        super().__init__()
        self.input = input
        self.attr = attr
        self.state = State.Solve
    
    def get_params(self):
        return self.attr
    
    def zero_known_grads(self):
        if(self.attr != None and self.attr.grad != None):
            self.attr.grad[self.input.knowns_map[Props.Attr]] = 0

    def forward(self):
        '''
            Returns prediction given the IVS, ICS, and element attributes.  Uses 
            the linear algebra solution to the Sparse Tableau Formulation of the
            circuit.
            Prediction contains element currents, element voltages, and node 
            potentials. Node potentials are missing the reference node since it 
            is removed from the STF to avoid singular matrix A.
            Output is 2D tensor of shape ( 2 * elements + nodes - 1, 1)
        '''
        if(self.state == State.Init):
            pass
        elif(self.state == State.Solve):
            A,b = self.build(with_constants=False)
            return A,torch.linalg.solve(A,b),b
        elif(self.state == State.Lstsq):
            A,b = self.build(with_constants=True)
            return A,torch.linalg.lstsq(A,b).solution,b

    def build(self,with_constants):
        # inputs
        M = self.input.M
        M_red = self.input.M_red
        num_elements = self.input.circuit.num_elements()
        num_nodes = self.input.circuit.num_nodes()
                
        # A matrix
        kcl_row = torch.cat(tensors=(M_red,
                                    torch.zeros_like(M_red),
                                    torch.zeros_like(M_red[:,:-1])),dim=1)
        kvl_row = torch.cat(tensors=(torch.zeros_like(M),
                                    torch.eye(num_elements),
                                    -M_red.T),dim=1)
        e_row = self.input.E()
        A = None
        if(with_constants):
            A = torch.cat(tensors=(
                    kcl_row,
                    kvl_row,
                    e_row,
                    self.input.I_knowns(),
                    self.input.V_knowns()
                ), dim=0)
        else:
            A = torch.cat(tensors=(
                    kcl_row,
                    kvl_row,
                    e_row,
                ), dim=0)
                
        # b matrix
        kcl_zeros = torch.zeros(size=(num_nodes - 1,1))
        kvl_zeros = torch.zeros(size=(num_elements,1))
        b = None
        if(with_constants):
            b = torch.cat(tensors=(
                    kvl_zeros,
                    kcl_zeros,
                    self.input.src_const(),
                    self.input.known_const()
                ), dim=0)
        else:
            b = torch.cat(tensors=(
                    kvl_zeros,
                    kcl_zeros,
                    self.input.src_const()
                ), dim=0)
        
        return A,b