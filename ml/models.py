import torch
import torch.nn as nn
import circuits as ckt


class Solver(nn.Module):
    ''' 
    Sparse Tableau Formulation of circuit analysis, modeled as a machine learning
    problem to learn element attributes using backprop and optimization.
    '''
    def __init__(self, input: ckt.Input, attr:nn.Parameter):
        super().__init__()
        self.input = input
        self.attr = attr
    
    def get_params(self):
        return self.attr
    
    def zero_known_grads(self):
        if(self.attr != None and self.attr.grad != None):
            self.attr.grad[self.input.knowns_map[ckt.Props.Attr]] = 0

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
        A,b = self.build()
        return A,torch.linalg.lstsq(A,b).solution,b

    def build(self):
        # inputs
        s = self.input.s(self.attr)
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
        e_row = self.input.X_row(self.attr)

        A = torch.cat(tensors=(kcl_row,kvl_row,e_row,
                        torch.tensor([[1,0,0,0,0]]).to(torch.float)
                               ), dim=0)
                
        # b matrix
        kcl_zeros = torch.zeros(size=(num_nodes - 1,1))
        kvl_zeros = torch.zeros(size=(num_elements,1))
        b = torch.cat(tensors=(kvl_zeros,kcl_zeros,s,
                               torch.tensor([-10]).to(torch.float).unsqueeze(dim=1).T
                               ), dim=0)
        
        return A,b