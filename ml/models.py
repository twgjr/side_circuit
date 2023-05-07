import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from circuits import System,Kinds,Circuit,Element,Props
from torch.linalg import solve

class Coefficients(nn.Module):
    def __init__(self, circuit:Circuit):
        super().__init__()
        num_elements = circuit.num_elements()
        num_nodes = circuit.num_nodes()
        M = circuit.M()
        M_zeros = torch.zeros_like(M)
        element_eye = torch.eye(num_elements)
        element_zeros = torch.zeros_like(element_eye)
        node_zeros = torch.zeros((num_nodes,num_nodes))
        self.kcl = torch.cat(tensors=(M,M_zeros,node_zeros),dim=1)
        self.kvl = torch.cat(tensors=(element_zeros,element_eye,-M.T),dim=1)
        self.elements = self.init_elements(circuit)

    def init_elements(self, circuit:Circuit) -> nn.ModuleList:
        mod_list = []
        for element in circuit.elements:
            if(element.kind == Kinds.R):
                mod_list.append(ResistorCoeff(element))
            elif(element.kind == Kinds.IVS):
                mod_list.append(VoltageSourceCoeff(element))
            elif(element.kind == Kinds.VC):
                mod_list.append(VoltageControlCoeff(element))
            elif(element.kind == Kinds.SW):
                mod_list.append(SwitchCoeff(element))
        return  nn.ModuleList(mod_list)
    
    def forward(self):
        coeff = torch.cat(tensors=(self.kcl,self.kvl), dim=0)
        for element in self.elements:
            el_out = element()
            coeff = torch.cat(tensors=(coeff,el_out), dim=0)
        return coeff

class ElementModule(nn.Module):
    def __init__(self, element:Element):
        super().__init__()
        self.element = element
        self.num_elements = element.circuit.num_elements()
        self.num_nodes = element.circuit.num_nodes()
        self.pos = element.index
        self.is_known:bool = None
        self.params:Parameter = None

    def zero_known_grads(self):
        if(self.is_known == True and self.params != None):
            if(self.params != None and self.params.grad != None):
                self.params.grad = torch.zeros_like(self.params)

    def clamp_params(self):
        if(self.params!=None):
            min = 1e-15
            max = 1e15
            for p in self.parameters():
                p.data.clamp_(min, max)

    def set_param_vals(self, params: Tensor):
        if(self.params != None):
            assert params.shape == self.params.shape
            for p in self.parameters():
                p.data = params

class ResistorCoeff(ElementModule):
    def __init__(self, element:Element):
        super().__init__(element)
        init_val = element.a
        if(init_val == None):
            init_val = 1.0
            self.is_known = False
        else:
            init_val /= element.circuit.system.r_base
            self.is_known = True
        self.params = Parameter(torch.tensor([init_val]))

    def forward(self):
        p = torch.zeros(self.num_nodes)
        z = torch.zeros(self.num_elements)
        z[self.pos] = -self.params
        y = torch.zeros(self.num_elements)
        y[self.pos] = 1.0
        return torch.cat((z,y,p)).unsqueeze(0)
    
class VoltageSourceCoeff(ElementModule):
    def __init__(self, element):
        super().__init__(element)

    def forward(self):
        p = torch.zeros(self.num_nodes)
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        y[self.pos] = 1.0
        return torch.cat((z,y,p)).unsqueeze(0)
    
class VoltageControlCoeff(ElementModule):
    def __init__(self, element):
        super().__init__(element)

    def forward(self):
        p = torch.zeros(self.num_nodes)
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        z[self.pos] = 1.0
        return torch.cat((z,y,p)).unsqueeze(0)
    
class SwitchCoeff(ElementModule):
    def __init__(self, element:Element):
        super().__init__(element)
        self.params = Parameter(torch.tensor([0.0]))

    def forward(self):
        p = torch.zeros(self.num_nodes)
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        if(self.params > 0):
            y[self.pos] = 1.0
        else:
            z[self.pos] = 1.0
        return torch.cat((z,y,p)).unsqueeze(0)
    
class Constants(nn.Module):
    def __init__(self, circuit: Circuit):
            super().__init__()
            num_nodes = circuit.num_nodes()
            num_elements = circuit.num_elements()
            self.kcl = torch.zeros(size=(num_nodes,1))
            self.kvl = torch.zeros(size=(num_elements,1))
            self.elements = self.init_elements(circuit)

    def init_elements(self, circuit:Circuit) -> nn.ModuleList:
        mod_list = []
        for element in circuit.elements:
            if(element.kind == Kinds.IVS):
                mod_list.append(VoltageSourceConstant(element))
            else:
                mod_list.append(ZeroConstant(element))
        return  nn.ModuleList(mod_list)
    
    def forward(self, time:int):
        consts = torch.cat(tensors=(self.kcl,self.kvl), dim=0)
        for element in self.elements:
            el_out = element(time)
            consts = torch.cat(tensors=(consts,el_out), dim=0)
        return consts
    
class ZeroConstant(ElementModule):
    def __init__(self, element:Element):
        super().__init__(element)

    def forward(self,time):
        return torch.zeros((1,1))
    
class VoltageSourceConstant(ElementModule):
    def __init__(self, element:Element):
        super().__init__(element)
        init_vals = element.v.get_data()
        if(init_vals == []):
            init_vals = element.circuit.system.init_signal_data()
            self.is_known = False
        else:
            for v in range(len(init_vals)):
                init_vals[v] /= element.circuit.system.v_base
            self.is_known = True
        self.params = Parameter(torch.tensor(init_vals))

    def forward(self,time:int):
        return self.params[time].unsqueeze(0).unsqueeze(0)

class CircuitModule(nn.Module):
    def __init__(self, circuit: Circuit):
        super().__init__()
        self.circuit = circuit
        self.A = Coefficients(circuit)
        self.b = Constants(circuit)

    def get_params(self,time:int) -> list[Tensor]:
        '''return a list of all element parameters order by position in circuit
        elements list.'''
        params_list = []
        indices = []
        for m,module in enumerate(self.A.elements):
            module:ElementModule
            if(module.params == None):
                indices.append(m)
                params_list.append(None)
            else:
                params_list.append(module.params[0])
        for index in indices:
            module = self.b.elements[index]
            if(module.params == None):
                params_list[index] = None
            else:
                params_list[index] = module.params[time]
        return params_list

    def forward(self,time:int):
        A = self.A()
        b = self.b(time)
        solution_out:Tensor = solve(A[1:,:-1],b[1:,:])
        split = self.circuit.num_elements()
        i_out = solution_out[:split,:].squeeze()
        v_out = solution_out[split:2*split,:].squeeze()
        a_out = self.get_params(time)
        out_map = {Props.I:i_out,Props.V:v_out,Props.A:a_out}
        return out_map
    
    def zero_known_grads(self):
        for module in self.A.elements:
            module:ElementModule
            module.zero_known_grads()
        for module in self.b.elements:
            module:ElementModule
            module.zero_known_grads()

    def clamp_params(self):
        for module in self.A.elements:
            module:ElementModule
            module.clamp_params()
        for module in self.b.elements:
            module:ElementModule
            module.clamp_params()

    def set_param_vals(self, params: Tensor):
        for module in self.A.elements:
            module:ElementModule
            module.set_param_vals(params)
        for module in self.b.elements:
            module:ElementModule
            module.set_param_vals(params)

class SwitchError(nn.Module):
    def __init__(self, parent:Element, child:Element) -> None:
        super().__init__()
        self.parent = parent
        self.child = child
        self.p_prop = self.init_prop(parent)
        self.p_idx = parent.index
        self.p_ckt_idx = parent.circuit.index
        self.c_idx = child.index
        self.c_ckt_idx = child.circuit.index

    def init_prop(self, element:Element):
        ret_prop = None
        if(element.kind == Kinds.VC):
            ret_prop = Props.V
        return ret_prop

    def forward(self, ckt_list:list[tuple[Tensor]]) -> Tensor:
        parent_val = ckt_list[self.p_ckt_idx][self.p_prop][self.p_idx]
        child_param = ckt_list[self.c_ckt_idx][Props.A][self.c_idx]
        return parent_val - child_param

class SystemModule(nn.Module):
    def __init__(self, system: System):
        super().__init__()
        self.system = system
        system.update_bases()
        self.circuits = self.init_circuit()
        self.sw_err_list = self.init_sw_error_list()

    def init_sw_error_list(self):
        sw_error_list = []
        for circuit in self.system.circuits:
            for element in circuit.elements:
                if(element.has_parent()):
                    sw_error_list.append(SwitchError(element.parent,element))
        return nn.ModuleList(sw_error_list)

    def init_circuit(self):
        mod_list = []
        for circuit in self.system.circuits:
            mod_list.append(CircuitModule(circuit))
        return nn.ModuleList(mod_list)
    
    def forward(self, time:int):
        ckt_out_list = []
        sw_err_out_list = []
        for circuit in self.circuits:
            circuit_output = circuit(time)
            ckt_out_list.append(circuit_output)
        for sw_err in self.sw_err_list:
            sw_err_out_list.append(sw_err(ckt_out_list))
        return ckt_out_list, sw_err_out_list
    
    def zero_known_grads(self):
        for module in self.circuits:
            module:ElementModule
            module.zero_known_grads()

    def clamp_params(self):
        for module in self.circuits:
            module:ElementModule
            module.clamp_params()

    def set_param_vals(self, params: Tensor):
        for module in self.circuits:
            module:ElementModule
            module.set_param_vals(params)