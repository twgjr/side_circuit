import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from circuits import System,Kinds,Circuit,Element,Props
from torch.linalg import solve
import bisect

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
            mod_list.append(ElementCoeff(element,self.time_set))
        return  nn.ModuleList(mod_list)
    
    def forward(self, time:float):
        coeff = torch.cat(tensors=(self.kcl,self.kvl), dim=0)
        for element in self.elements:
            element:ElementCoeff
            el_out = element.forward(time)
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
        self.z = torch.zeros(self.num_elements)
        self.y = torch.zeros(self.num_elements)
        self.p = torch.zeros(self.num_nodes)
        self.values = TimeSeries(self.time_set)
        if(self.element.kind == Kinds.R or self.element.kind == Kinds.C or
           self.element.kind == Kinds.L):
            for time,value in element.a:
                self.values[time] = value

    def forward(self, time:float):
        if(self.element.kind == Kinds.R):
            self.z[self.idx] = -self.values[0.0]
            self.y[self.idx] = 1.0
        elif(self.element.kind == Kinds.C):
            dt = self.element.circuit.system.dt
            self.z[self.idx] = -dt/self.values[0.0]
            self.y[self.idx] = 1.0
        elif(self.element.kind == Kinds.L):
            dt = self.element.circuit.system.dt
            self.z[self.idx] = 1.0
            self.y[self.idx] = -dt/self.values[0.0]
        elif(self.element.kind == Kinds.IVS or self.element.kind == Kinds.VG 
             or self.element.kind == Kinds.CC):
            self.y[self.idx] = 1.0
        elif(self.element.kind == Kinds.ICS or self.element.kind == Kinds.IG
             or self.element.kind == Kinds.VC):
            self.z[self.idx] = 1.0
        elif(self.element.kind == Kinds.SW):
            if(torch.sigmoid(self.values[time]) > 0.5):
                self.y[self.idx] = 1.0
                self.z[self.idx] = 0.0
            else:
                self.y[self.idx] = 0.0
                self.z[self.idx] = 1.0
        else: assert()
        return torch.cat((self.z,self.y,self.p)).unsqueeze(0)

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
    
    def forward(self, time:float, time_idx:int):
        consts = torch.cat(tensors=(self.kcl,self.kvl), dim=0)
        for element in self.elements:
            element:ElementConstant
            el_out = element.forward(time,time_idx).unsqueeze(0).unsqueeze(0)
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
        if(self.element.kind == Kinds.VG or self.element.kind == Kinds.IG):
            for time in self.element.a:
                self.gain[time] = self.element.a[time]
    
    def init_values(self):
        signal = None
        if(self.element.kind == Kinds.IVS):
            signal = self.element.v
        elif(self.element.kind == Kinds.ICS):
            signal = self.element.i
        elif(self.element.kind == Kinds.VG):
            signal = self.element.parent.v
        elif(self.element.kind == Kinds.IG):
            signal = self.element.parent.i
        if(signal == None): return
        for time in signal:
            self.values[time] = signal[time]

    def forward(self, time:float, time_idx:int):
        if(self.element.kind == Kinds.IVS or self.element.kind == Kinds.ICS):
            return self.values[time]
        if(self.element.kind == Kinds.VG or self.element.kind == Kinds.IG):
            prev_time = self.time_set.prev_time(time_idx)
            return self.values[str(prev_time)] * self.gain[str(time)]
        else:
            return torch.tensor(0.0)

class CircuitModule(nn.Module):
    def __init__(self, circuit: Circuit,time_set:'TimeSet'):
        super().__init__()
        self.time_set = time_set
        self.circuit = circuit
        self.A = Coefficients(circuit,self.time_set)
        self.b = Constants(circuit,self.time_set)

    def get_attrs(self, coeff:bool) -> list[Tensor]:
        '''return a list of all element parameters ordered by position in 
        circuit elements list. If coeff == True parameters are from A matrix,
        else from b vector'''
        attr_list = []
        elements = self.A.elements
        if(coeff == False):
            elements = self.b.elements
        for module in elements:
            module:ElementCoeff
            attr = module.values[0.0]
            attr_list.append(attr)
        return attr_list

    def forward(self, time:float, time_idx:int):
        A = self.A.forward(time)
        b = self.b.forward(time,time_idx)
        solution_out:Tensor = solve(A[1:,:-1],b[1:,:])
        split = self.circuit.num_elements()
        i_out = solution_out[:split,:].squeeze()
        v_out = solution_out[split:2*split,:].squeeze()
        a_out = self.get_attrs(coeff=True)
        b_out = self.get_attrs(coeff=False)
        out_map = {Props.I:i_out,Props.V:v_out,Props.A:a_out,Props.B:b_out}
        return out_map

    def clamp_params(self):
        for module in self.A.elements:
            module:ElementCoeff
            if(module.element.kind == Kinds.R or 
               module.element.kind == Kinds.L or
               module.element.kind == Kinds.C):
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
        return torch.square(parent_val - child_param)

class SystemModule(nn.Module):
    def __init__(self, system: System,time_set:'TimeSet'):
        super().__init__()
        self.time_set = time_set
        self.system = system
        # system.update_bases()
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
            mod_list.append(CircuitModule(circuit,self.time_set))
        return nn.ModuleList(mod_list)
    
    def forward(self, time:float, time_idx:int):
        ckt_out_list = []
        sw_err_out = torch.tensor(0.0)
        for circuit in self.circuits:
            circuit:CircuitModule
            circuit_output = circuit.forward(time,time_idx)
            ckt_out_list.append(circuit_output)
        for sw_err in self.sw_err_list:
            sw_err:SwitchError
            sw_err_out += sw_err.forward(ckt_out_list)
        return (ckt_out_list, sw_err_out)

    def clamp_params(self):
        for module in self.circuits:
            module:ElementCoeff
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
        self.delta_errors = self.init_delta_errors()
        self.timeset.sort()

    def init_delta_errors(self):
        delta_errors = []
        for element in self.system.elements:
            if(element.kind == Kinds.C or element.kind == Kinds.L):
                delta_errors.append(DeltaError(element))
        return nn.ModuleList(delta_errors)

    def forward(self):
        ckts_t_outs:dict[float:Tensor] = {}
        delta_err_out = torch.tensor(0.0)
        sw_err_out = torch.tensor(0.0)
        time = 0
        time_prev = time
        for t,time in enumerate(self.timeset.times):
            ckts_t,sw_err_t = self.system_mod.forward(time,t)
            sw_err_out += sw_err_t
            ckts_t_outs[time] = ckts_t
            for delta_err in self.delta_errors:
                delta_err:DeltaError
                if(time == time_prev): continue
                sys = ckts_t_outs[time]
                sys_prev = ckts_t_outs[time_prev]
                delta_err_out += delta_err.forward(sys_prev,sys)
            time_prev = time
        return (ckts_t_outs, delta_err_out, sw_err_out)

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
        self.params = nn.ParameterDict()
        self.time_set = time_set

    def __setitem__(self, time:float, value:float=None):
        assert(isinstance(time,float))
        self.time_set.add(time)
        key = str(time).replace(".", "_")
        if(value == None):

            self.params[key] = Parameter(torch.tensor(1.0))
        else:
            self.params[key] = Parameter(torch.tensor(value))
            param:Parameter = self.params[key]
            param.requires_grad = False

    def __getitem__(self,time:float):
        key = str(time).replace(".", "_")
        if(key in self.params):
            return self.params[key]
        else:
            self.params[key] = Parameter(torch.tensor(1.0))
            return self.params[key]
        
    def forward(self,time:float):
        return self[time]