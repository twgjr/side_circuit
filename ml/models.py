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
            elif(element.kind == Kinds.C):
                mod_list.append(CapacitorCoeff(element))
            elif(element.kind == Kinds.L):
                mod_list.append(InductorCoeff(element))
            else:
                assert()
        return  nn.ModuleList(mod_list)
    
    def forward(self, time:int):
        coeff = torch.cat(tensors=(self.kcl,self.kvl), dim=0)
        for element in self.elements:
            el_out = element(time)
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

    def forward(self,time:int):
        p = torch.zeros(self.num_nodes)
        z = torch.zeros(self.num_elements)
        z[self.pos] = -self.params[0]
        y = torch.zeros(self.num_elements)
        y[self.pos] = 1.0
        return torch.cat((z,y,p)).unsqueeze(0)
    
class CapacitorCoeff(ElementModule):
    def __init__(self, element:Element):
        super().__init__(element)
        init_val = element.a
        if(init_val == None):
            init_val = 1.0
            self.is_known = False
        else:
            init_val /= element.circuit.system.c_base
            self.is_known = True
        self.params = Parameter(torch.tensor([init_val]))

    def forward(self,time:int):
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        p = torch.zeros(self.num_nodes)
        dt = self.element.circuit.system.dt
        z[self.pos] = -dt/self.params[0]
        y[self.pos] = 1.0
        return torch.cat((z,y,p)).unsqueeze(0)
    
class InductorCoeff(ElementModule):
    def __init__(self, element:Element):
        super().__init__(element)
        init_val = element.a
        if(init_val == None):
            init_val = 1.0
            self.is_known = False
        else:
            init_val /= element.circuit.system.l_base
            self.is_known = True
        self.params = Parameter(torch.tensor([init_val]))

    def forward(self,time:int):
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        p = torch.zeros(self.num_nodes)
        dt = self.element.circuit.system.dt
        z[self.pos] = 1.0
        y[self.pos] = -dt/self.params[0]
        return torch.cat((z,y,p)).unsqueeze(0)
    
class VoltageSourceCoeff(ElementModule):
    def __init__(self, element):
        super().__init__(element)

    def forward(self,time:int):
        p = torch.zeros(self.num_nodes)
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        y[self.pos] = 1.0
        return torch.cat((z,y,p)).unsqueeze(0)
    
class VoltageControlCoeff(ElementModule):
    def __init__(self, element):
        super().__init__(element)

    def forward(self,time:int):
        p = torch.zeros(self.num_nodes)
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        z[self.pos] = 1.0
        return torch.cat((z,y,p)).unsqueeze(0)
    
class SwitchCoeff(ElementModule):
    def __init__(self, element:Element):
        super().__init__(element)
        sig_len = self.element.circuit.system.signal_len
        self.params = Parameter(torch.tensor([0.0]*sig_len))

    def forward(self,time:int):
        p = torch.zeros(self.num_nodes)
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        if(torch.sigmoid(self.params[time]) > 0.5):
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
    
class DynamicConstant(ElementModule):
    '''Stores a parameter that represents the previous time step value (i or v).
    Used for calculating the differential voltage or current.'''
    def __init__(self, element:Element):
        super().__init__(element)
        self.params = Parameter(torch.tensor([1.0]))

    def forward(self,time:int):
        return self.params[0].unsqueeze(0).unsqueeze(0)

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
        A = self.A(time)
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
        sw_err_out = torch.tensor(0.0)
        dyn_params_out_list = []
        for circuit in self.circuits:
            circuit_output = circuit(time)
            ckt_out_list.append(circuit_output)
        for sw_err in self.sw_err_list:
            sw_err_out += sw_err(ckt_out_list)
        
        return {'circuits_t':ckt_out_list, 'sw_err_t':sw_err_out, 
                'dyn_params': dyn_params_out_list}
    
    def zero_known_grads(self):
        for module in self.circuits:
            module:ElementModule
            module.zero_known_grads()

    def clamp_params(self):
        for module in self.circuits:
            module:ElementModule
            module.clamp_params()

class DeltaError(nn.Module):
    def __init__(self, element:Element) -> None:
        super().__init__()
        self.element = element
        self.prop = self.init_prop(element)
        self.idx = element.index
        self.ckt_idx = element.circuit.index

    def init_prop(self, element:Element):
        ret_prop = None
        if(element.kind == Kinds.C):
            ret_prop = Props.V
        if(element.kind == Kinds.L):
            ret_prop = Props.I
        return ret_prop

    def forward(self, ckt_t_prev_pred:list[tuple[Tensor]],
                ckt_t_dyn_params:list[tuple[Tensor]]) -> Tensor:
        prev_val = ckt_t_prev_pred[self.ckt_idx][self.prop][self.idx]
        prev_val_param = ckt_t_dyn_params[self.ckt_idx][self.idx]
        return prev_val_param - prev_val

class DynamicModule(nn.Module):
    def __init__(self, system: System):
        super().__init__()
        self.system = system
        self.system_mod = SystemModule(system)
        self.delta_errors = self.init_delta_errors()

    def init_delta_errors(self):
        delta_errors = []
        for element in self.system.elements:
            if(element.kind == Kinds.C or element.kind == Kinds.L):
                delta_errors.append(DeltaError(element))
        return nn.ModuleList(delta_errors)

    def forward(self):
        sys_t_out_list = []
        delta_err_out = torch.tensor(0.0)
        sw_err_out = torch.tensor(0.0)
        for t in range(self.system.signal_len):
            sys_t = self.system_mod.forward(t)
            ckts_t = sys_t['circuits_t']
            sw_err_t = sys_t['sw_err_t']
            sw_err_out += sw_err_t
            sys_t_out_list.append(ckts_t)
            for delta_err in self.delta_errors:
                delta_err:DeltaError
                if(t>0):
                    delta_err_out += delta_err.forward(sys_t_out_list[t-1],
                                                        sys_t_out_list[t])
        return {'sys_seq':sys_t_out_list, 'delta_err':delta_err_out, 
                'sw_err':sw_err_out}
    
    def zero_known_grads(self):
        self.system_mod.zero_known_grads()

    def clamp_params(self):
        self.system_mod.clamp_params()