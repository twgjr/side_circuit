from circuits import Circuit,Props,Kinds,Signal,Element
from torch.utils.data import Dataset
import torch
from torch.nn import Parameter
from torch import Tensor

class Data():
    def __init__(self, circuit:Circuit) -> None:
        self.circuit = circuit
        self.M = self.circuit.M()
        self.elements = self.circuit.export()
        self.i_base, self.v_base, self.r_base = self.init_base()
        self.known_attr_mask = self.init_known_attr_mask()
        self.r_mask = self.init_mask(Kinds.R)
        self.ics_mask = self.init_mask(Kinds.ICS)
        self.ivs_mask = self.init_mask(Kinds.IVS)

    def init_mask(self, kind:Kinds):
        return torch.tensor(self.kind_one_hot(kind)).to(torch.bool)
    
    def signals_base(self, signals:list[Signal], eps:float=1e-12) -> float:
        input_max = 0
        for signal in signals:
            if(signal.is_empty()):
                continue
            else:
                for val in signal.data:
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
        i_sigs = self.prop_list(Props.I)
        i_knowns = self.prop_mask(Props.I)
        i_has_knowns = True in i_knowns
        v_sigs = self.prop_list(Props.V)
        v_knowns = self.prop_mask(Props.V)
        v_has_knowns = True in v_knowns
        r_vals = self.attr_list(Kinds.R)
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
    
    def init_known_attr_mask(self):
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
        prop_sigs = self.prop_list(prop)
        ret_list = []
        for signal in prop_sigs:
            sig_copy = signal.copy()
            if(not sig_copy.is_empty()):
                for d in range(len(sig_copy.data)):
                    sig_copy.data[d] = sig_copy.data[d]/base
            ret_list.append(sig_copy)
        return ret_list
    
    def norm_attrs(self, kind:Kinds, base:int) -> list[float]:
        attr_vals = self.attr_list(kind)
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
            params[self.ics_mask] = self.denorm(params[self.ics_mask],self.i_base)
            params[self.ivs_mask] = self.denorm(params[self.ivs_mask],self.v_base)
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
                    data_i_t.append(signal.data[t])
            data_v_t = []
            for signal in elements_v:
                if(signal.is_empty()):
                    data_v_t.append(None)
                else:
                    data_v_t.append(signal.data[t])
            data_i.append(data_i_t)
            data_v.append(data_v_t)
        return (data_i,data_v)

    def extract_attributes(self) -> list[list[float]]:
        '''Returns a list ordered by time: resistances.  Each row represents time.  
        Each column represents an attribute value for an element (r).'''
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

    def kind_one_hot(self, kind:Kinds) -> list[bool]:
        '''returns boolean mask of element kinds ordered by element'''
        assert isinstance(kind,Kinds)
        return self.elements['kinds'][kind]

    def prop_list(self, prop:Props) -> list[Signal]:
        '''return list ordered by element of properties (i,v). Uknowns 
        initialized to 1'''
        assert isinstance(prop,Props)
        return self.elements['properties'][prop]
    
    def attr_list(self,kind:Kinds) -> list[float]:
        '''return list of element attributes with None for unknowns'''
        assert isinstance(kind,Kinds)
        return self.elements['attributes'][kind]
    
    def prop_mask(self, prop:Props) -> list[bool]:
        '''returns boolean mask of known element properties ordered by element'''
        assert isinstance(prop,Props)
        signals = self.elements['properties'][prop]
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
        attrs = self.elements['attributes'][kind]
        ret_list = []
        for attr in attrs:
            assert isinstance(attr,float) or attr == None
            ret_list.append(attr != None)
        return ret_list
    
class CircuitDataset(Dataset):
    def __init__(self,dataset:list[Tensor]):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]