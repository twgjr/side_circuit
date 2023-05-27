import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from circuits import System,Kinds,Circuit,Element,Props,Signal
from torch.linalg import solve

class Coefficients(nn.Module):
    def __init__(self, circuit:Circuit,time_set:'TimeSet'):
        super().__init__()
        self.time_set = time_set
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
            el_coeff = ElementCoeff(element,self.time_set)
            mod_list.append(el_coeff)
        return  nn.ModuleList(mod_list)
    
    def forward(self, time:float, time_prev:float):
        coeff = torch.cat(tensors=(self.kcl,self.kvl), dim=0)
        for element in self.elements:
            element:ElementCoeff
            el_out = element.forward(time,time_prev)
            coeff = torch.cat(tensors=(coeff,el_out), dim=0)
        return coeff

class ElementCoeff(nn.Module):
    def __init__(self, element:Element, time_set:'TimeSet'):
        super().__init__()
        self.time_set = time_set
        self.element = element
        self.num_elements = element.circuit.num_elements()
        self.num_nodes = element.circuit.num_nodes()
        self.ckt_idx = element.circuit.index
        self.idx = element.index
        self.p_ckt_idx = None
        self.p_idx = None
        self.p_prop = None
        if(element.has_parent()):
            self.p_ckt_idx = element.parent.circuit.index
            self.p_idx = element.parent.index
            if(element.parent.kind == Kinds.VC): self.p_prop = Props.V 
            elif(element.parent.kind == Kinds.CC): self.p_prop = Props.I
            else: assert()
        self.is_known:bool = False
        self.values = TimeSeries(self.time_set)
        for time,value in element.a:
            self.values.set_param(time,value)

    def forward(self, time:float, time_prev:float):
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        p = torch.zeros(self.num_nodes)
        if(self.element.kind == Kinds.R):
            z[self.idx] = self.values(0.0)
            y[self.idx] = -1.0
        elif(self.element.kind == Kinds.C):
            dt = time - time_prev
            z[self.idx] = dt/self.values(0.0)
            y[self.idx] = -1.0
        elif(self.element.kind == Kinds.L):
            dt = time - time_prev
            z[self.idx] = -1.0
            y[self.idx] = dt/self.values(0.0)
        elif(self.element.kind == Kinds.IVS or self.element.kind == Kinds.VG 
             or self.element.kind == Kinds.CC):
            y[self.idx] = 1.0
        elif(self.element.kind == Kinds.ICS or self.element.kind == Kinds.CG
             or self.element.kind == Kinds.VC):
            z[self.idx] = 1.0
        elif(self.element.kind == Kinds.SW):
            if(torch.sigmoid(self.values(time)) > 0.5):
                y[self.idx] = 1.0
            else:
                z[self.idx] = 1.0
        else: assert()
        return torch.cat((z,y,p)).unsqueeze(0)

    def clamp_params(self):
        # if(self.values != None):
        min = 1e-18
        max = 1e18
        for p in self.parameters():
            p.data.clamp_(min, max)
    
class Constants(nn.Module):
    def __init__(self, circuit: Circuit,time_set:'TimeSet'):
        super().__init__()
        self.time_set = time_set
        num_nodes = circuit.num_nodes()
        num_elements = circuit.num_elements()
        self.kcl = torch.zeros(size=(num_nodes,1))
        self.kvl = torch.zeros(size=(num_elements,1))
        self.elements = self.init_elements(circuit)

    def init_elements(self, circuit:Circuit) -> nn.ModuleList:
        mod_list = []
        for element in circuit.elements:
            mod_list.append(ElementConstant(element,self.time_set))
        return  nn.ModuleList(mod_list)
    
    def forward(self, time:float, time_prev:float):
        consts = torch.cat(tensors=(self.kcl,self.kvl), dim=0)
        for element in self.elements:
            element:ElementConstant
            el_out = element.forward(time,time_prev).unsqueeze(0).unsqueeze(0)
            consts = torch.cat(tensors=(consts,el_out), dim=0)
        return consts
    
class ElementConstant(nn.Module):
    def __init__(self, element:Element,time_set:'TimeSet'):
        super().__init__()
        self.time_set = time_set
        self.element = element
        self.ckt_idx = element.circuit.index
        self.idx = element.index
        self.gain = TimeSeries(time_set)
        self.init_gain()
        self.values = TimeSeries(time_set)
        self.init_values()

    def init_gain(self):
        if(self.element.kind == Kinds.VG or self.element.kind == Kinds.CG):
            for time in self.element.a:
                self.gain.set_param(time,self.element.a[time])
    
    def init_values(self):
        signal = None
        if(self.element.kind == Kinds.IVS):
            signal = self.element.v
        elif(self.element.kind == Kinds.ICS):
            signal = self.element.i
        elif(self.element.kind == Kinds.VG):
            signal = self.element.parent.v
        elif(self.element.kind == Kinds.CG):
            signal = self.element.parent.i
        if(signal == None): return
        for time in signal:
            self.values.set_param(time,signal[time])

    def forward(self, time:float, time_prev:float):
        if(self.element.kind == Kinds.IVS or self.element.kind == Kinds.ICS):
            return self.values(time)
        if(self.element.kind == Kinds.L or self.element.kind == Kinds.C):
            return self.values(time_prev) * self.gain(time)
        else:
            return torch.tensor(0.0)

class CircuitModule(nn.Module):
    def __init__(self, circuit: Circuit,time_set:'TimeSet'):
        super().__init__()
        self.time_set = time_set
        self.circuit = circuit
        self.A = Coefficients(circuit,self.time_set)
        self.b = Constants(circuit,self.time_set)

    def get_attrs(self, time:float, coeff:bool) -> list[Tensor]:
        '''return a list of all element attr params ordered by position in 
        circuit elements list. If coeff == True parameters are from A matrix,
        else from b vector'''
        attr_list = []
        elements = self.A.elements
        if(coeff == False):
            elements = self.b.elements
        for module in elements:
            module:ElementCoeff
            attr_list.append(module.values.get_param(time,static=True))
        return attr_list

    def forward(self, time:float, time_prev:float):
        A = self.A.forward(time,time_prev)
        b = self.b.forward(time,time_prev)
        solution_out:Tensor = solve(A[1:,:-1],b[1:,:])
        split = self.circuit.num_elements()
        i_out = solution_out[:split,:].squeeze()
        v_out = solution_out[split:2*split,:].squeeze()
        a_out = self.get_attrs(time,coeff=True)
        b_out = self.get_attrs(time,coeff=False)
        out_map = {Props.I:i_out,Props.V:v_out,Props.A:a_out,Props.B:b_out}
        return out_map

    def clamp_params(self):
        for module in self.A.elements:
            module:ElementCoeff
            if(module.element.kind == Kinds.R or 
               module.element.kind == Kinds.L or
               module.element.kind == Kinds.C):
                module.clamp_params()

class ControlledElementError(nn.Module):
    def __init__(self, element:Element) -> None:
        super().__init__()
        if(element.kind != Kinds.CG and element.kind != Kinds.VG and 
           element.kind != Kinds.SW):
            assert()
        self.element = element
        self.idx = element.index
        self.ckt_idx = element.circuit.index
        self.parent = element.parent
        self.p_idx = self.parent.index
        self.p_ckt_idx = self.parent.circuit.index

    def forward(self, ckt_list:list[tuple[Tensor]]) -> Tensor:
        a = ckt_list[self.ckt_idx][Props.A][self.idx]
        if(self.element.kind == Kinds.SW or self.element.kind == Kinds.VG or
           self.element.kind == Kinds.CG):
            if(self.parent.kind == Kinds.VC):
                parent_val = ckt_list[self.p_ckt_idx][Props.V][self.p_idx]
                return torch.square(parent_val - a)
            elif(self.parent.kind == Kinds.CC):
                parent_val = ckt_list[self.p_ckt_idx][Props.I][self.p_idx]
                return torch.square(parent_val - a)

class SystemModule(nn.Module):
    def __init__(self, system: System,time_set:'TimeSet'):
        super().__init__()
        self.time_set = time_set
        self.system = system
        # system.update_bases()
        self.circuits = self.init_circuit()
        self.ctrl_el_err_list = self.init_ctrl_el_err_list()
        self.i_base = 1.0
        self.v_base = 1.0
        self.r_base = 1.0
        # self.l_base = 1.0
        # self.c_base = 1.0

    def init_ctrl_el_err_list(self):
        ctrl_el_error_list = []
        for circuit in self.system.circuits:
            for element in circuit.elements:
                if(element.kind == Kinds.SW or element.kind == Kinds.VG or
                   element.kind == Kinds.CG):
                    ctrl_el_error_list.append(ControlledElementError(element))
        return nn.ModuleList(ctrl_el_error_list)

    def init_circuit(self):
        mod_list = []
        for circuit in self.system.circuits:
            mod_list.append(CircuitModule(circuit,self.time_set))
        return nn.ModuleList(mod_list)
    
    def forward(self, time:float, time_prev:float):
        sys_out = []
        ctrl_el_err_out = torch.tensor(0.0)
        for circuit in self.circuits:
            circuit:CircuitModule
            circuit_output = circuit.forward(time,time_prev)
            sys_out.append(circuit_output)
        for ctrl_el_err in self.ctrl_el_err_list:
            ctrl_el_err:ControlledElementError
            ctrl_el_err_out += ctrl_el_err.forward(time,sys_out)
        return (sys_out, ctrl_el_err_out)

    def clamp_params(self):
        for module in self.circuits:
            module:ElementCoeff
            module.clamp_params()

class TimeDeltaError(nn.Module):
    '''Let t0, t1, and t2 be times.  There are two time deltas t1-t0, and t2-t1.
    But the model actually creats a parameter that represents the t1 when it is 
    used as a previous time (i.e. t2-t1_pri and t1-t0_pri).  This module 
    reconciles t0 with t0_pri and t1 with t1_pri so that they converge to the 
    same value.'''
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

    def forward(self, sys_pred_prev:list[tuple[Tensor]],
                sys_pred:list[tuple[Tensor]]) -> Tensor:
        prev_val = sys_pred_prev[self.ckt_idx][self.prop][self.idx]
        prev_val_param = sys_pred[self.ckt_idx][Props.B][self.idx]
        return torch.square(prev_val_param - prev_val)
    
class DynamicModule(nn.Module):
    def __init__(self, system: System):
        super().__init__()
        self.timeset = TimeSet()
        self.system = system
        self.system_mod = SystemModule(system,self.timeset)
        self.time_errors = self.init_time_errors()
        # self.dyn_el_errors = self.init_element_errors()
        self.timeset.sort()

    def init_time_errors(self):
        delta_errors = []
        for element in self.system.elements:
            if(element.kind == Kinds.C or element.kind == Kinds.L):
                delta_errors.append(TimeDeltaError(element))
        return nn.ModuleList(delta_errors)

    def forward(self):
        system_sequence:dict[float:Tensor] = {}
        delta_err_out = torch.tensor(0.0)
        ctrl_el_err_out = torch.tensor(0.0)
        time = 0.0
        time_prev = time
        for t,time in enumerate(self.timeset.times):
            system_t,ctrl_el_err_t = self.system_mod.forward(time,time_prev)
            ctrl_el_err_out += ctrl_el_err_t
            system_sequence[time] = system_t
            for delta_err in self.time_errors:
                delta_err:TimeDeltaError
                if(time == time_prev): continue
                sys = system_sequence[time]
                sys_prev = system_sequence[time_prev]
                delta_err_out += delta_err.forward(sys_prev,sys)
            time_prev = time
        return (system_sequence, delta_err_out, ctrl_el_err_out)

    def clamp_params(self):
        self.system_mod.clamp_params()

class TimeSet():
    def __init__(self):
        self.times = []
        self._times_dict = {}

    def add(self, time):
        self._times_dict[time] = None

    def sort(self):
        self.times = sorted(self._times_dict.keys())

    def prev_time(self, time_idx):
        assert(time_idx != 0)
        return self.times[time_idx-1]

class TimeSeries(nn.Module):
    def __init__(self, time_set:TimeSet):
        super().__init__()
        self._params = nn.ParameterDict()
        self.time_set = time_set

    def set_param(self, time:float, value:float=None):
        assert(isinstance(time,float))
        self.time_set.add(time)
        key = str(time).replace(".", "_")
        if(value == None):
            self._params[key] = Parameter(torch.tensor(1.0))
        else:
            self._params[key] = Parameter(torch.tensor(value))
            param:Parameter = self._params[key]
            param.requires_grad = False

    def get_param(self,time:float, static:bool=False):
        key = str(time).replace(".", "_")
        if(key in self._params):
            return self._params[key]
        else:
            if(static == True):
                return None
            self._params[key] = Parameter(torch.tensor(1.0))
            return self._params[key]
        
    def reinforce_param(self, time:float, value:float):
        assert(isinstance(time,float))
        key = str(time).replace(".", "_")
        self._params[key] = Parameter(torch.tensor(value))
        
    def forward(self,time:float):
        return self.get_param(time)