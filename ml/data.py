from circuits import Circuit,Props,Kinds

class Data():
    def __init__(self, circuit:Circuit) -> None:
        self.circuit = circuit
        self.M = self.circuit.M()
        self.elements = self.circuit.extract_elements()
        self.target = self.target_list()
        self.target_mask = self.target_mask_list()

    def base(self, input:list):
        input_max = 0
        for val in input:
            abs_val = abs(val)
            if(abs_val > input_max):
                input_max = abs_val
        if(input_max > 1):
            return input_max
        else:
            return 1
    
    def normalize(self, base:int, input:list):
        ret_list = []
        for val in input:
            ret_list.append(val/base)
        return ret_list
    
    def target_list(self) -> list[float]:
        '''inputs values including source attributes in respective i and v props'''
        i_vals = self.prop_list(Props.I,True)
        v_vals = self.prop_list(Props.V,True)
        p_vals = self.prop_list(Props.Pot,True)
        i_base = self.base(i_vals)
        v_base = self.base(v_vals)
        p_base = self.base(p_vals)
        i_norm = self.normalize(i_base,i_vals)
        v_norm = self.normalize(v_base,v_vals)
        p_norm = self.normalize(p_base,p_vals)
        return i_norm + v_norm + p_norm
    
    def target_mask_list(self):
        '''mask of known inputs values including source attributes in respective
          i and v props'''
        currents = self.mask_of_prop(Props.I,True)
        voltages = self.mask_of_prop(Props.V,True)
        potentials = self.mask_of_prop(Props.Pot,True)
        return currents + voltages + potentials
    
    def mask_of_kind(self, kind:Kinds):
        '''returns boolean mask of element kinds ordered by element'''
        return self.kind_list(kind)
    
    def mask_of_prop(self, prop:Props, include_attr:bool):
        '''returns boolean mask of known element properties ordered by element'''
        if(include_attr):
            return self.nones_to_bool_mask(self.prop_with_attrs(prop))
        else:
            return self.nones_to_bool_mask(self.elements['properties'][prop])
    
    def mask_of_attr(self, kind:Kinds):
        '''returns boolean mask of known element attributes ordered by element'''
        return self.nones_to_bool_mask(self.elements['attributes'][kind])

    def kind_list(self, kind:Kinds) -> list:
        kind_list = self.elements['kinds'][kind]
        return kind_list
    
    def prop_with_attrs_of_kind(self,kind:Kinds,to_prop:Props):
        to_prop_list = self.elements['properties'][to_prop]
        list_with_attr = []
        attr_list = self.replace_nones(self.attr_list(kind))
        for p in range(len(to_prop_list)):
            if(self.kind_list(kind)[p]):
                list_with_attr.append(attr_list[p])
            else:
                list_with_attr.append(to_prop_list[p])
        return list_with_attr
    
    def prop_with_attrs(self,prop:Props):
        if(prop == Props.I):
            return self.prop_with_attrs_of_kind(Kinds.ICS,prop)
        elif(prop == Props.V):
            return self.prop_with_attrs_of_kind(Kinds.IVS,prop)
        elif(prop == Props.Pot):
            return self.elements['properties'][prop]

    def prop_list(self, prop:Props, include_attr:bool) -> list:
        '''return list of element properties (i,v,pot) with unknowns initialized
          to 1'''
        if(include_attr):
            return self.replace_nones(self.prop_with_attrs(prop))
        else:
            return self.replace_nones(self.elements['properties'][prop])
    
    def attr_list(self,kind:Kinds) -> list:
        '''return list of element attributes with unknowns initialized to 1'''
        return self.replace_nones(self.elements['attributes'][kind])
    
    def replace_nones(self, input_list):
        '''replaces None type items in list with False (bool) or 0 (float)'''
        ret_list = []
        for i in range(len(input_list)):
            if(input_list[i] == None):
                ret_list.append(1)
            else:
                ret_list.append(input_list[i])
        return ret_list
    
    def nones_to_bool_mask(self, input:list):
        '''converts the list into a boolean list where True has a value and 
        Falseis None type.  Used to indicate a "unknown" circuit values of None 
        type."'''
        ret_list = []
        for item in input:
            if(item == None):
                ret_list.append(False)
            else:
                ret_list.append(True)
        return ret_list