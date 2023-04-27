from circuits import Circuit,Props,Kinds,Signal,Element,System
from torch.utils.data import Dataset
import torch
from torch.nn import Parameter
from torch import Tensor

class SystemData():
    def __init__(self, system:System):
        self.system = system
        self.circuit_data_list = self.init_circuit_data_list()
        self.circuit_masks = self.init_circuit_masks()
        self.v_control_list = self.system.parent_indices(Kinds.VC)
        self.i_control_list = self.system.parent_indices(Kinds.CC)

    def init_circuit_data_list(self):
        return [CircuitData(circuit,self) 
                for circuit in self.system.circuits.values()]
    
    def init_circuit_masks(self)->list[list[bool]]:
        '''returns a list of circuit masks.  Each mask is a list of int indices.
        The order of the list corresponds to the order of the elements in the 
        respective circuit.  The index value at each position refers to the 
        element somewhere in the system that is a control input to that element
        in the circuit.  '''
        circuit_masks = []
        for circuit in self.system.circuits.values():
            circuit_masks.append(circuit.circuit_mask())
        return circuit_masks
    
class CircuitData():
    def __init__(self, circuit:Circuit, system_data:SystemData) -> None:
        self.system_data = system_data
        self.circuit = circuit
        self.M = self.circuit.M()
        self.i_base, self.v_base, self.r_base = self.init_base()
        #TODO attrs_mask and Trainer.mask (v/i) should be in same place
        self.attrs_mask = self.init_attrs_mask()
        self.r_mask = self.init_mask(Kinds.R)
        self.ics_mask = self.init_mask(Kinds.ICS)
        self.ivs_mask = self.init_mask(Kinds.IVS)
        self.vcsw_mask = self.init_mask(Kinds.SW, Kinds.VC)
        self.ccsw_mask = self.init_mask(Kinds.SW, Kinds.CC)
        self.vc_mask = self.init_mask(Kinds.VC)
        self.cc_mask = self.init_mask(Kinds.CC)
        self.circuit_mask = self.system_data.circuit_masks[self.circuit.index]

    def init_mask(self, kind:Kinds, with_control:Kinds=None):
        return torch.tensor(
            self.circuit.kind_list(kind,with_control)).to(torch.bool)
    
    def signals_base(self, signals:list[Signal], eps:float=1e-12) -> float:
        input_max = 0
        for signal in signals:
            if(signal.is_empty()):
                continue
            else:
                for v in range(len(signal)):
                    val = signal[v]
                    abs_val = abs(val)
                    if(abs_val > input_max):
                        input_max = abs_val
        if(input_max < eps):
            return eps
        else:
            return input_max
    
    def values_base(self, values:list[float], eps:float=1e-12) -> float:
        input_max = 0
        for val in values:
            if(val == None):
                continue
            abs_val = abs(val)
            if(abs_val > input_max):
                input_max = abs_val
        if(input_max < eps):
            return eps
        else:
            return input_max
        
    def init_base(self):
        i_sigs = self.circuit.prop_list(Props.I)
        i_knowns = self.prop_mask(Props.I)
        i_has_knowns = True in i_knowns
        v_sigs = self.circuit.prop_list(Props.V)
        v_knowns = self.prop_mask(Props.V)
        v_has_knowns = True in v_knowns
        r_vals = self.circuit.attr_list(Kinds.R)
        r_knowns = self.attr_mask(Kinds.R)
        r_has_knowns = True in r_knowns
        i_base = self.signals_base(i_sigs)
        v_base = self.signals_base(v_sigs)
        r_base = self.values_base(r_vals)
        if(not i_has_knowns and not v_has_knowns and not r_has_knowns):
            i_base = 1
            v_base = 1
            r_base = 1
        elif(not i_has_knowns and not v_has_knowns and r_has_knowns):
            i_base = 1/r_base
            v_base = r_base
        elif(not i_has_knowns and v_has_knowns and not r_has_knowns):
            i_base = v_base
            r_base = v_base
        elif(not i_has_knowns and v_has_knowns and r_has_knowns):
            i_base = v_base/r_base
        elif(i_has_knowns and not v_has_knowns and not r_has_knowns):
            v_base = i_base
            r_base = 1/i_base
        elif(i_has_knowns and not v_has_knowns and r_has_knowns):
            v_base = i_base*r_base
        elif(i_has_knowns and v_has_knowns and not r_has_knowns):
            r_base = v_base/i_base
        elif(i_has_knowns and v_has_knowns and r_has_knowns):
            pass
        return (i_base,v_base,r_base)
    
    def init_attrs_mask(self):
        r = torch.tensor(self.attr_mask(Kinds.R))
        return r
    
    def replace_none(self, l:list[float], val:float) -> list[float]:
        ret_list = []
        for x in l:
            if(x == None):
                ret_list.append(val)
            else:
                ret_list.append(x)
        return ret_list
    
    def init_params(self):
        r_list = self.norm_attrs(Kinds.R,self.r_base)
        r_list = self.replace_none(r_list,1.0)
        r_tensor = torch.tensor(r_list).to(torch.float)
        attr_tensor = r_tensor
        attr_params = torch.nn.Parameter(attr_tensor)
        return attr_params
    
    def norm_signals(self, prop:Props, base:int) -> list[Signal]:
        prop_sigs = self.circuit.prop_list(prop)
        ret_list = []
        for signal in prop_sigs:
            sig_copy = signal.copy()
            if(not sig_copy.is_empty()):
                for d in range(len(sig_copy )):
                    sig_copy[d] = sig_copy[d]/base
            ret_list.append(sig_copy)
        return ret_list
    
    def norm_attrs(self, kind:Kinds, base:int) -> list[float]:
        attr_vals = self.circuit.attr_list(kind)
        ret_list = []
        for val in attr_vals:
            if(val == None):
                ret_list.append(None)
            else:
                ret_list.append(val/base)
        return ret_list
    
    def split_input_output(self, input:Tensor):
        split = self.circuit.num_elements()
        assert input.shape == (2*split,1)
        i = input[:split,:]
        v = input[split:2*split,:]
        return i,v
    
    def denorm_input_output(self,io_list:list[Tensor])->tuple[list[Tensor],list[Tensor]]:
        assert isinstance(io_list,list)
        i = []
        v = []
        for io in io_list:
            assert isinstance(io,Tensor)
            i_norm,v_norm = self.split_input_output(io)
            i.append(self.denorm(i_norm,self.i_base))
            v.append(self.denorm(v_norm,self.v_base))
        return i,v

    @staticmethod
    def denorm(input:Tensor, base:float):
        clone = input.detach().clone()
        return clone*base
    
    def denorm_params(self, params:Parameter):
        with torch.no_grad():
            params[self.r_mask] = self.denorm(params[self.r_mask],self.r_base)
        return params
    
    def extract_data(self) -> tuple[list[list[float]],list[list[float]]]:
        '''Returns two lists ordered by time: currents and voltages.  Each row 
        represents time.  Each column represents a property value for an element
          (i or v).'''
        elements_i = self.norm_signals(Props.I,self.i_base)
        elements_v = self.norm_signals(Props.V,self.v_base)
        data_i = []
        data_v = []
        assert self.circuit.signal_len > 0
        for t in range(self.circuit.signal_len):
            data_i_t = []
            for signal in elements_i:
                if(signal.is_empty()):
                    data_i_t.append(None)
                else:
                    data_i_t.append(signal[t])
            data_v_t = []
            for signal in elements_v:
                if(signal.is_empty()):
                    data_v_t.append(None)
                else:
                    data_v_t.append(signal [t])
            data_i.append(data_i_t)
            data_v.append(data_v_t)
        return (data_i,data_v)

    def extract_attributes(self) -> list[float]:
        '''Returns a list ordered by element: resistances.'''
        r_attrs = self.norm_attrs(Kinds.R,self.r_base)
        return r_attrs

    def init_dataset(self)->list[Tensor]:
        '''Returns a tensor list dataset ordered by time. Only known properties
        are non-zero.  Shape = (signal length, 2*elements)'''
        i_data,v_data = self.extract_data()
        data_zip = zip(i_data,v_data)
        data_list = []
        for i,v in data_zip:
            sample = i+v
            clean_sample = []
            for item in sample:
                if(item == None):
                    clean_sample.append(0)
                else:
                    clean_sample.append(item)
            data_list.append(clean_sample)
        dataset = []
        for sample in data_list:
            sample_tensor = torch.tensor(sample).float().unsqueeze(1)
            dataset.append(sample_tensor)
        return dataset

    def data_mask_list(self):
        '''returns boolean mask of target values ordered by element'''
        currents = self.prop_mask(Props.I)
        voltages = self.prop_mask(Props.V)
        return currents + voltages
    
    def prop_mask(self, prop:Props) -> list[bool]:
        '''returns boolean mask of known element properties ordered by element'''
        assert isinstance(prop,Props)
        signals = self.circuit.prop_list(prop)
        ret_list = []
        for signal in signals:
            assert isinstance(signal,Signal)
            if(signal.is_empty()):
                ret_list.append(False)
            else:
                ret_list.append(True)
        return ret_list
    
    def attr_mask(self, kind:Kinds) -> list[bool]:
        '''returns boolean mask of known element attributes ordered by element'''
        assert isinstance(kind,Kinds)
        assert kind != Kinds.ICS and kind != Kinds.IVS
        attrs = self.circuit.attr_list(kind)
        ret_list = []
        for attr in attrs:
            assert isinstance(attr,float) or attr == None
            ret_list.append(attr != None)
        return ret_list
    
    def extract_input(self, 
                      system_i:Tensor, 
                      system_v:Tensor) -> tuple[Tensor,Tensor]:
        '''returns circuit_i, circuit_v that are inputs to the next iteration
        of the circuit solver.  Circuit_i and circuit_v are indexed from 
        system_i and system_v by the circuit mask.  The circuit mask is a list
        indices of the inputs to the circuit, such as current sources currents, 
        voltage sources voltages, and switch control voltage.'''
        assert isinstance(system_i,Tensor)
        assert isinstance(system_v,Tensor)
        assert system_i.shape == system_v.shape
        assert system_i.shape[0] == self.system_data.system.num_elements()
        assert system_i.shape[1] == 1
        circuit_i = system_i[self.circuit_mask]
        circuit_v = system_v[self.circuit_mask]
        return circuit_i,circuit_v
    
class CircuitDataset(Dataset):
    def __init__(self,dataset:list[Tensor]):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]