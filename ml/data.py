from circuits import Circuit,Props,Kinds
import random

class Preprocess():
    def __init__(self, circuit:Circuit) -> None:
        self.circuit = circuit
        self.M = self.circuit.M()
        self.elements = self.circuit.extract_elements()
        self.truth = self.truth_list()
        self.truth_mask = self.truth_mask_list()
    
    def truth_list(self) -> list[float]:
        '''inputs values including source attributes in respective i and v props'''
        currents = self.prop_list(Props.I,include_attr=True,replace_nones=True)
        voltages = self.prop_list(Props.V,include_attr=True,replace_nones=True)
        potentials = self.prop_list(Props.Pot,include_attr=False,replace_nones=True)
        return currents + voltages + potentials
    
    def truth_mask_list(self):
        '''mask of known inputs values including source attributes in respective
          i and v props'''
        currents = self.mask_of_prop(Props.I,include_attr=True)
        voltages = self.mask_of_prop(Props.V,include_attr=True)
        potentials = self.mask_of_prop(Props.Pot,include_attr=False)
        return currents + voltages + potentials
    
    def mask_of_kind(self, kind:Kinds):
        '''returns boolean mask of element kinds ordered by element'''
        return self.kind_list(kind)
    
    def mask_of_prop(self, prop:Props, include_attr:bool):
        '''returns boolean mask of known element properties ordered by element'''
        return self.to_bool_mask(
            self.prop_list(prop,include_attr,replace_nones=False))
    
    def mask_of_attr(self, kind:Kinds):
        '''returns boolean mask of known element attributes ordered by element'''
        return self.to_bool_mask(self.attr_list(kind,replace_nones=False))

    def kind_list(self, kind:Kinds) -> list:
        kind_list = self.elements['kinds'][kind]
        return kind_list
    
    def prop_list(self, prop:Props, include_attr:bool, 
                  replace_nones:bool=True) -> list:
        prop_list = None
        if(replace_nones):
            prop_list = self.replace_nones(self.elements['properties'][prop])
        else:
            prop_list = self.elements['properties'][prop]
        if(include_attr):
            ret_list = []
            if(prop == Props.I):
                ics_attr_list = self.attr_list(Kinds.ICS, replace_nones)
                for p in range(len(prop_list)):
                    if(self.kind_list(Kinds.ICS)[p]):
                        ret_list.append(ics_attr_list[p])
                    else:
                        ret_list.append(prop_list[p])
            elif(prop == Props.V):
                ivs_attr_list = self.attr_list(Kinds.IVS, replace_nones)
                for p in range(len(prop_list)):
                    if(self.kind_list(Kinds.IVS)[p]):
                        ret_list.append(ivs_attr_list[p])
                    else:
                        ret_list.append(prop_list[p])
            elif(prop == Props.Pot):
                ret_list = prop_list
            return ret_list
        else:
            return prop_list
    
    def attr_list(self,kind:Kinds, replace_nones:bool=True, 
                  rand_unknowns:bool = True) -> list:
        if(replace_nones):
            return self.replace_nones(
                self.elements['attributes'][kind],rand_unknowns)
        else:
            return self.elements['attributes'][kind]
    
    def replace_nones(self, input_list, rand_unknowns:bool=True):
        '''replaces None type items in list with False (bool) or 0 (float)'''
        ret_list = []
        for i in range(len(input_list)):
            if(input_list[i] == None):
                if(rand_unknowns):
                    ret_list.append(random.random())
                else:
                    ret_list.append(0)
            else:
                ret_list.append(input_list[i])
        return ret_list
    
    def to_bool_mask(self, input:list):
        ret_list = []
        for item in input:
            if(item == None):
                ret_list.append(False)
            else:
                ret_list.append(True)
        return ret_list