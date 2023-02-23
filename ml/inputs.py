from circuits import Circuit,Props,Kinds
import torch
import torch.nn as nn

class Input():
    def __init__(self, circuit:Circuit) -> None:
        self.circuit = circuit
        self.M = self.circuit.M()
        self.M_red = self.M[:-1,:]
        self.kinds_map, self.inputs_map, self.knowns_map = self.circuit.extract_elements()
        self.inputs_map[Props.Pot] = [0]*self.circuit.num_nodes()
        self.knowns_map[Props.Pot] = [False]*self.circuit.num_nodes()
    
    def list_to_vec_mask(self, input_list:list):
        size = len(input_list)
        vector = torch.tensor(input_list).reshape(size,1).to(torch.float)
        return vector
    
    def list_to_diag_mask(self, input_list:list):
        size = len(input_list)
        eye = torch.eye(size)
        vector_mask = self.list_to_vec_mask(input_list)
        matrix_mask =  vector_mask @ vector_mask.T * eye
        return matrix_mask
    
    def X_r(self, element_attrs):
        num_elem = self.circuit.num_elements()
        num_nodes = self.circuit.num_nodes()
        is_r_mask_m = self.list_to_diag_mask(self.kinds_map[Kinds.R])
        X_mask = torch.cat(tensors= (
                            - is_r_mask_m * element_attrs,
                            is_r_mask_m,
                            torch.zeros(size=(num_elem,num_nodes-1)),
                        ),dim=1)
        return X_mask
    
    def X_ivs(self):
        num_elem = self.circuit.num_elements()
        num_nodes = self.circuit.num_nodes()
        is_s_mask_m = self.list_to_diag_mask(self.kinds_map[Kinds.IVS])
        X_mask = torch.cat(tensors= (
                            torch.zeros(size=(num_elem,num_elem)),
                            is_s_mask_m,
                            torch.zeros(size=(num_elem,num_nodes-1))
                        ),dim=1)
        return X_mask

    def init_params(self):
        num_elem = self.circuit.num_elements()
        attr_tensor = torch.tensor(self.inputs_map[Props.Attr]).reshape(num_elem,1)
        attr_param = nn.Parameter(attr_tensor)
        return attr_param

    def X_row(self, element_attrs):
        X = self.X_r(element_attrs) + self.X_ivs()
        return X

    def sources(self,element_attrs:nn.Parameter):
        s = torch.zeros(element_attrs.size(dim=0))
        s = s.reshape(s.size(dim=0),1)
        s_v = element_attrs[self.kinds_map[Kinds.IVS]]
        s_i = element_attrs[self.kinds_map[Kinds.ICS]]
        if(s_v.nelement() > 0):
            s[self.kinds_map[Kinds.IVS]] = s_v
        if(s_i.nelement() > 0):
            s[self.kinds_map[Kinds.ICS]] = s_i
        return s
    
    # def constants(self):
    #     inputs = self.
    #     s = torch.zeros(element_attrs.size(dim=0))
    #     s = s.reshape(s.size(dim=0),1)
    #     s_v = element_attrs[self.kinds_map[Kinds.IVS]]
    #     s_i = element_attrs[self.kinds_map[Kinds.ICS]]
    #     if(s_v.nelement() > 0):
    #         s[self.kinds_map[Kinds.IVS]] = s_v
    #     if(s_i.nelement() > 0):
    #         s[self.kinds_map[Kinds.ICS]] = s_i
    #     return s