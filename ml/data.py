from circuits import Circuit,Props,System
from models import DynamicModule
import torch

class Data():
    def __init__(self, system:System, model:DynamicModule):
        self.model = model
        self.system = system
        self.sequence = self.init_sequence()
        self.masks = self.init_masks()
    
    def init_sequence(self) -> dict[float:'SystemDataT']:
        return {time:SystemDataT(self.system,time) for time in self.model.timeset.times}
    
    def init_masks(self):
        '''returns a list of lists.  Outer list (rows) is circuits.  Inner 
        list contains a tuple of three boolean masks to index predicted i,v,a
        for loss calculation with a sequence.'''
        circuit_masks = []
        for circuit in self.system.circuits:
            i_prop_mask = circuit.prop_mask(Props.I)
            v_prop_mask =circuit.prop_mask(Props.V)
            a_prop_mask = circuit.prop_mask(Props.A)
            mask_map = {Props.I:i_prop_mask,Props.V:v_prop_mask,
                        Props.A:a_prop_mask}
            circuit_masks.append(mask_map)
        return circuit_masks

class SystemDataT():
    def __init__(self, system:System, time:float):
        assert(isinstance(time,float))
        self.system = system
        self.circuits = self.init_data_list(time)

    def init_data_list(self,time:float) -> list['CircuitDataT']:
        assert(isinstance(time,float))
        return [CircuitDataT(circuit,time) for circuit in self.system.circuits]
    
class CircuitDataT():
    def __init__(self, circuit:Circuit, time:float) -> None:
        assert(isinstance(time,float))
        self.circuit = circuit
        self._data = {Props.I: self.init_tensor(Props.I,time),
                      Props.V: self.init_tensor(Props.V,time),
                      Props.A: self.init_tensor(Props.A,time)}

    def __getitem__(self, key):
        return self._data[key]

    def init_tensor(self,prop:Props,time:float):
        assert(isinstance(time,float))
        prop_list = []
        for element in self.circuit.elements:
            if(prop == Props.I):
                if(len(element.i) > 0):
                    prop_list.append(element.i[time])
            elif(prop == Props.V):
                if(len(element.v) > 0):
                    prop_list.append(element.v[time])
            elif(prop == Props.A):
                if(len(element.a) > 0):
                    prop_list.append(element.a[time])
        return torch.tensor(prop_list).float()