import torch
from torch import Tensor,nn
from torch.nn import Parameter
from circuits import System,Kinds,Circuit,Element,Props
from torch.linalg import solve
from math import isclose
from enum import Enum

class Modes(Enum):
    INIT = 0
    TR = 1

class StepChange(Enum):
    Init = 0
    Unknown = 1
    Increasing = 2
    Decreasing = 3
    Stop = 4

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

    def init_elements(self, circuit:Circuit) -> list['ElementCoeff']:
        mod_list = []
        for element in circuit.elements:
            el_coeff = ElementCoeff(element)
            mod_list.append(el_coeff)
        return  mod_list
    
    def forward(self, time:float, dt:float, mode:Modes):
        coeff = torch.cat(tensors=(self.kcl,self.kvl), dim=0)
        for element in self.elements:
            element:ElementCoeff
            el_out = element.forward(time,dt,mode)
            coeff = torch.cat(tensors=(coeff,el_out), dim=0)
        return coeff

class ElementCoeff(nn.Module):
    def __init__(self, element:Element):
        super().__init__()
        self.element = element
        self.num_elements = element.circuit.num_elements()
        self.num_nodes = element.circuit.num_nodes()
        self.idx = element.index()
        self.ckt_idx = element.circuit_index()
        self.p_idx = element.parent_index()
        self.p_ckt_idx = element.parent_circuit_index()
        self.param = None
        if(self.element.kind == Kinds.SW):
            self.param = Parameter(0.0)

    def forward(self, time:float, dt:float, mode:Modes):
        z = torch.zeros(self.num_elements)
        y = torch.zeros(self.num_elements)
        p = torch.zeros(self.num_nodes)

        if(self.element.kind == Kinds.R):
            z[self.idx] = self.element.a[time]
            y[self.idx] = -1.0
        elif(self.element.kind == Kinds.L):
            if(mode==Modes.TR):
                z[self.idx] = 1.0
                y[self.idx] = -dt/self.element.a[time]
            elif(mode==Modes.INIT):
                z[self.idx] = 1.0
            else:
                assert()
        elif(self.element.kind == Kinds.C):
            if(mode==Modes.TR):
                z[self.idx] = -dt/self.element.a[time]
                y[self.idx] = 1.0
            elif(mode==Modes.INIT):
                y[self.idx] = 1.0
            else:
                assert()
        elif(self.element.kind == Kinds.VS or self.element.kind == Kinds.VG 
             or self.element.kind == Kinds.CC):
            y[self.idx] = 1.0
        elif(self.element.kind == Kinds.CS or self.element.kind == Kinds.CG
             or self.element.kind == Kinds.VC):
            z[self.idx] = 1.0
        elif(self.element.kind == Kinds.SW):
            if(torch.sigmoid(self.param) > 0.5):
                y[self.idx] = 1.0
            else:
                z[self.idx] = 1.0
        else: assert()
        return torch.cat((z,y,p)).unsqueeze(0)
    
class Constants(nn.Module):
    def __init__(self, circuit: Circuit):
        super().__init__()
        num_nodes = circuit.num_nodes()
        num_elements = circuit.num_elements()
        self.kcl = torch.zeros(size=(num_nodes,1))
        self.kvl = torch.zeros(size=(num_elements,1))
        self.elements = self.init_elements(circuit)

    def init_elements(self, circuit:Circuit) -> list['ElementConstant']:
        mod_list = []
        for element in circuit.elements:
            mod_list.append(ElementConstant(element))
        return  mod_list
    
    def forward(self, time:float, ckt_sols_prev, mode:Modes):
        consts = torch.cat(tensors=(self.kcl,self.kvl), dim=0)
        for element in self.elements:
            element:ElementConstant
            el_out = element.forward(time,ckt_sols_prev,mode).unsqueeze(0).unsqueeze(0)
            consts = torch.cat(tensors=(consts,el_out), dim=0)
        return consts
    
class ElementConstant(nn.Module):
    def __init__(self, element:Element):
        super().__init__()
        self.element = element
        self.ckt_idx = element.circuit.index()
        self.idx = element.index()
        self.param = None
        if(self.element.kind == Kinds.VG or self.element.kind == Kinds.CG):
            self.param = Parameter(0.0)

    def forward(self, time:float, ckt_sols_prev:dict[Props,Tensor], mode:Modes):
        if(self.element.kind == Kinds.VS):
            return torch.tensor(self.element.v[time])
        elif(self.element.kind == Kinds.CS):
            return torch.tensor(self.element.i[time])
        elif(self.element.kind == Kinds.VG or self.element.kind == Kinds.CG):
            return torch.tensor(self.param*self.element.a[0.0])
        elif(self.element.kind == Kinds.L):
            if(mode==Modes.TR):
                return ckt_sols_prev[self.ckt_idx][Props.I][self.idx]
            elif(mode==Modes.INIT):
                return torch.tensor(0.0)
            else:
                assert()
        elif(self.element.kind == Kinds.C):
            if(mode==Modes.TR):
                return ckt_sols_prev[self.ckt_idx][Props.V][self.idx]
            elif(mode==Modes.INIT):
                return torch.tensor(0.0)
            else:
                assert()
        else:
            return torch.tensor(0.0)

class CircuitModel(nn.Module):
    def __init__(self, circuit: Circuit):
        super().__init__()
        self.circuit = circuit
        self.index = circuit.index()
        self.A = Coefficients(circuit)
        self.b = Constants(circuit)

    def forward(self, time:float, dt:float, ckt_sols_prev, mode:Modes):
        A = self.A.forward(time,dt,mode)
        b = self.b.forward(time,ckt_sols_prev,mode)
        solution_out:Tensor = solve(A[1:,:-1],b[1:,:])
        split = self.circuit.num_elements()
        i_out = solution_out[:split,:].squeeze()
        v_out = solution_out[split:2*split,:].squeeze()
        return {Props.I:i_out,Props.V:v_out}
    
class ControlError(nn.Module):
    def __init__(self,element:Element):
        self.p_ckt_idx = element.parent_circuit_index()
        self.p_prop = Props.I if (element.parent.kind==Kinds.CC) else (Props.V)
        self.p_idx = element.parent_index()
        self.param = None
        self.prev_val = None

    def forward(self, ckt_sols:list[dict]):
        sol_val = ckt_sols[self.p_ckt_idx][self.p_prop][self.p_idx]
        is_stable = self.is_stable(sol_val.item())
        return torch.square(sol_val - self.param), is_stable
    
    def is_stable(self, new_val):
        is_close = isclose(new_val,self.prev_val)
        self.prev_val = new_val
        return is_close
    
class SystemSolve():
    '''solves a single time step'''
    def __init__(self, system: System, lr:float):
        super().__init__()
        self.system = system
        self.circuits = self.init_circuit()
        self.errors = self.init_errors()
        self.opt = None
        if(len(list(self.circuits.parameters())) > 0):
            self.opt = torch.optim.Adam(self.circuits.parameters(),lr=lr)
        self.loss_fn = nn.MSELoss()

    def init_errors(self):
        mod_list = []
        for circuit in self.system.circuits:
            for element in circuit.elements:
                if(element.kind != Kinds.SW and element.kind != Kinds.CG and
                   element.kind != Kinds.VG):
                    continue
                error = ControlError(element)
                ckt_mod:CircuitModel = self.circuits[element]
                if(element.kind == Kinds.SW):
                    error.param = ckt_mod.A.elements[element.index()].param
                if(element.kind == Kinds.CG or element.kind == Kinds.VG):
                    error.param = ckt_mod.b.elements[element.index()].param
                mod_list.append(error)
        return nn.ModuleList(mod_list)

    def init_circuit(self):
        mod_list = []
        for circuit in self.system.circuits:
            mod_list.append(CircuitModel(circuit))
        return nn.ModuleList(mod_list)
    
    def solve(self, time:float, dt:float, ckt_sols_in,mode:Modes):
        ckt_sols, all_stable = self.step(time,dt,ckt_sols_in,mode)
        while(not all_stable):
            ckt_sols, all_stable = self.step(time,dt,ckt_sols_in,mode)
        return ckt_sols

    def step(self, time:float, dt:float, ckt_sols_prev, mode:Modes):
        ckt_sols = []
        err_outs = []
        all_stable = True
        for circuit in self.circuits:
            circuit:CircuitModel
            ckt_sols.append(circuit.forward(time,dt,ckt_sols_prev,mode))
        for err in self.errors:
            err:ControlError
            err_out,is_stable  = err.forward(ckt_sols)
            if(is_stable==False): all_stable = False
            err_outs.append(err_out)
        if(self.opt != None):
            loss = self.loss_fn(torch.sum(err_outs))
            loss.backward()
            self.opt.zero_grad()
            self.opt.step()
        return ckt_sols, all_stable

    
class Simulator():
    '''steps through transient simulation of each system solution'''
    def __init__(self, system: System):
        super().__init__()
        self.system = system
        self.system_mod = SystemSolve(system,0.5)

    def run(self, stop:float, init_step_size:float, d_threshold:float,
            min_step_size:float):
        sol_prev_t = self.system_mod.solve(0.0,init_step_size,None,mode=Modes.INIT)
        self.system.load(sol_prev_t,0.0)
        time = init_step_size
        prev_time = 0.0
        step_size = init_step_size
        while(time < stop):
            di_prev_t = None
            dv_prev_t = None
            sc_state = StepChange.Unknown
            while(not sc_state == StepChange.Stop):
                sol_t = self.system_mod.solve(time,step_size,sol_prev_t,Modes.TR)
                di_t = []
                dv_t = []
                for c in range(len(sol_t)):
                    di_t.append(sol_t[c][Props.I] - sol_prev_t[c][Props.I])
                    dv_t.append(sol_t[c][Props.V] - sol_prev_t[c][Props.V])
                    cat_t = torch.cat((di_t[c],dv_t[c]))
                    cat_t.data[cat_t == 0] = 1e-18
                    if(di_prev_t == None or dv_prev_t == None):
                        break
                    cat_prev_t = torch.cat((di_prev_t[c],dv_prev_t[c]))
                    cat_prev_t.data[cat_prev_t == 0] = 1e-18
                    sub = torch.sub(cat_t,cat_prev_t)/cat_prev_t
                    min_change = torch.min(sub)
                    max_change = torch.max(sub)
                    if(sc_state == StepChange.Unknown):
                        if(max_change > d_threshold):
                            sc_state = StepChange.Decreasing
                            step_size /= 2
                        else:
                            sc_state = StepChange.Increasing
                            step_size *= 2
                    elif(sc_state == StepChange.Decreasing):
                        if(max_change > d_threshold):
                            step_size /= 2
                        else:
                            sc_state = StepChange.Stop
                            break
                    elif(sc_state == StepChange.Increasing):
                        if(min_change < -d_threshold and 
                            max_change < d_threshold):
                            step_size *= 2
                        else:
                            sc_state = StepChange.Stop
                            break
                di_prev_t = di_t
                dv_prev_t = dv_t
                time = max(prev_time + step_size,min_step_size)
            sc_state = StepChange.Unknown
            prev_time = time
            sol_prev_t = sol_t
            self.system.load(sol_t,time)