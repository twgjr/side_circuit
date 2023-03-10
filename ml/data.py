from circuits import Circuit,Props,Kinds

class Data():
    def __init__(self, circuit:Circuit) -> None:
        self.circuit = circuit
        self.M = self.circuit.M()
        self.elements = self.circuit.export()

    def base(self, input:list, eps:float=1e-12) -> float:
        input_max = 0
        for val in input:
            abs_val = abs(val)
            if(abs_val > input_max):
                input_max = abs_val
        if(input_max < eps):
            return eps
        else:
            return input_max
        
    def init_base(self):
        i_data = self.prop_list(Props.I,True,0)
        i_knowns = self.mask_of_prop(Props.I,True)
        i_has_knowns = True in i_knowns
        v_data = self.prop_list(Props.V,True,0)
        v_knowns = self.mask_of_prop(Props.V,True)
        v_has_knowns = True in v_knowns
        r_data = self.attr_list(Kinds.R,0)
        r_knowns = self.mask_of_attr(Kinds.R)
        r_has_knowns = True in r_knowns
        i_base = self.base(i_data)
        v_base = self.base(v_data)
        r_base = self.base(r_data)
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

    
    def normalize(self, base:int, input:list):
        ret_list = []
        for v in range(len(input)):
            ret_list.append(input[v]/base)
        return ret_list
    
    def target_list(self, i_base, v_base) -> list[float]:
        '''returns list of normalized target values ordered by element'''
        i_vals = self.prop_list(Props.I,True,0)
        v_vals = self.prop_list(Props.V,True,0)
        i_mask = self.mask_of_prop(Props.I,True)
        v_mask = self.mask_of_prop(Props.V,True)
        i_norm = self.normalize(i_base,i_vals)
        v_norm = self.normalize(v_base,v_vals)
        for m in i_mask:
            if(not i_mask[m]):
                i_norm[m] = 1
            if(not v_mask[m]):
                v_norm[m] = 1
        vals_norm = i_norm + v_norm
        return vals_norm
    
    def target_mask_list(self):
        '''returns boolean mask of target values ordered by element'''
        currents = self.mask_of_prop(Props.I,True)
        voltages = self.mask_of_prop(Props.V,True)
        return currents + voltages
    
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
        attr_list = self.elements['attributes'][kind]
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

    def prop_list(self, prop:Props, include_attr:bool, init_val:int) -> list:
        '''return list of element properties (i,v,pot) with unknowns initialized
          to 1'''
        if(include_attr):
            return self.replace_nones(self.prop_with_attrs(prop),init_val)
        else:
            return self.replace_nones(self.elements['properties'][prop],init_val)
    
    def attr_list(self,kind:Kinds,init_val:float) -> list:
        '''return list of element attributes with unknowns initialized to 1'''
        return self.replace_nones(self.elements['attributes'][kind],init_val)
    
    def replace_nones(self, input_list, init_val:int):
        '''replaces None type items in list with False (bool) or init_val (float)'''
        ret_list = []
        for i in range(len(input_list)):
            if(input_list[i] == None):
                ret_list.append(init_val)
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