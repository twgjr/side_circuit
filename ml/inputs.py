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
        self.attr_param = self.init_params()
    
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

    def init_params(self):
        num_elem = self.circuit.num_elements()
        attr_tensor = torch.tensor(self.inputs_map[Props.Attr])\
                                .reshape(num_elem,1)
        attr_param = nn.Parameter(attr_tensor)
        return attr_param

    def E(self):
        return torch.cat(tensors=(
            self.Z(),self.Y(),self.constants(False,0))
                         ,dim=1)
    
    def I_knowns(self):
        Z_k = self.Z_known()
        return torch.cat(tensors=(
            Z_k,torch.zeros_like(Z_k),torch.zeros_like(Z_k)[:,:-1])
                        ,dim=1)
    
    def V_knowns(self):
        Y_k = self.Y_known()
        return torch.cat(tensors=(
            torch.zeros_like(Y_k),Y_k,torch.zeros_like(Y_k)[:,:-1])
                        ,dim=1)

    def src_const(self):
        attrs = self.attr_param
        s_v = attrs[self.kinds_map[Kinds.IVS]]
        s_i = attrs[self.kinds_map[Kinds.ICS]]
        s = torch.zeros(size = (attrs.size(dim=0), 1))
        if(s_v.nelement() > 0):
            s[self.kinds_map[Kinds.IVS]] = s_v
        if(s_i.nelement() > 0):
            s[self.kinds_map[Kinds.ICS]] = s_i
        return s
    
    def known_const(self):
        i_knowns_mask = torch.tensor(self.knowns_map[Props.I])
        v_knowns_mask = torch.tensor(self.knowns_map[Props.V])
        i_tensor = torch.tensor(self.inputs_map[Props.I])
        v_tensor = torch.tensor(self.inputs_map[Props.V])
        kc_i = i_tensor[i_knowns_mask]
        kc_v = v_tensor[v_knowns_mask]
        kc = torch.zeros(size = (self.circuit.num_elements(), 1))
        if(kc_i.nelement() > 0):
            kc[i_knowns_mask] = kc_i
        if(kc_v.nelement() > 0):
            kc[v_knowns_mask] = kc_v
        i_v_knowns_mask = i_knowns_mask + v_knowns_mask
        return kc[i_v_knowns_mask]

    def constants(self,e_prop,c_type):
        '''e_prop = True means the zeros will be sized for element properties
        current and voltag; False means it will be sized for the pots.
        c_type = 0: zeros, 1: ones, 2:identity'''
        if(e_prop):
            num_elem = self.circuit.num_elements()
            return self.constant_type(c_type, num_elem, num_elem)
        else:
            num_elem = self.circuit.num_elements()
            num_nodes = self.circuit.num_nodes()
            return self.constant_type(c_type, num_elem, num_nodes-1)
        
    def constant_type(self, c_type, rows, cols):
        if(c_type == 0):
            return torch.zeros(size=(rows,cols))
        elif(c_type == 1):
            return torch.ones(size=(rows,cols))
        elif(c_type == 2):
            return torch.eye(n=rows)
        else:
            assert()

    def Z(self):
        R_mask = self.list_to_diag_mask(self.kinds_map[Kinds.R])
        Z_r = - R_mask * self.attr_param
        Z_ics = self.list_to_diag_mask(self.kinds_map[Kinds.ICS])
        return Z_r + Z_ics

    def Y(self):
        Y_r = self.list_to_diag_mask(self.kinds_map[Kinds.R])
        Y_ivs = self.list_to_diag_mask(self.kinds_map[Kinds.IVS])
        return Y_r + Y_ivs

    def Z_known(self):
        i_knowns_mask_list = self.knowns_map[Props.I]
        kc_eye = torch.eye(n = self.circuit.num_elements())
        return kc_eye[i_knowns_mask_list]

    def Y_known(self):
        v_knowns_mask_list = self.knowns_map[Props.V]
        kc_eye = torch.eye(n = self.circuit.num_elements())
        return kc_eye[v_knowns_mask_list]
