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
        self.attr_param = nn.Parameter(self.init_tensor(Props.Attr))
    
    def list_to_vec_mask(self, input_list:list, allow_bool:bool):
        if(len(input_list) == 0):
            assert()
        list_type = type(input_list[0])
        torch_type = torch.float
        if(list_type == bool and allow_bool):
            torch_type = torch.bool
        size = len(input_list)
        vector = torch.tensor(input_list).reshape(size,1).to(torch_type)
        return vector
    
    def list_to_diag_mask(self, input_list:list):
        size = len(input_list)
        eye = torch.eye(size)
        vector_mask = self.list_to_vec_mask(input_list, allow_bool=False)
        matrix_mask =  vector_mask @ vector_mask.T * eye
        return matrix_mask
    
    def init_tensor(self, prop:Props):
        rows = None
        if(prop == Props.Pot):
            rows = self.circuit.num_nodes()
        else:
            rows = self.circuit.num_elements()
        out_tensor = torch.tensor(self.inputs_map[prop])\
                                .reshape(rows,1)
        return out_tensor
    
    def ivp_inputs(self):
        '''inputs values including source attributes in respective i and v props'''
        attrs = self.init_tensor(Props.Attr)
        currents = self.init_tensor(Props.I)
        voltages = self.init_tensor(Props.V)
        potentials = self.init_tensor(Props.Pot)
        i_known_mask = self.list_to_vec_mask(self.knowns_map[Props.I],True)
        v_known_mask = self.list_to_vec_mask(self.knowns_map[Props.V],True)
        ics_mask = self.list_to_vec_mask(self.kinds_map[Kinds.ICS],True)
        ivs_mask = self.list_to_vec_mask(self.kinds_map[Kinds.IVS],True)
        currents[~i_known_mask] = 0
        currents[ics_mask] = attrs[ics_mask]
        voltages[~v_known_mask] = 0
        voltages[ivs_mask] = attrs[ivs_mask]
        return torch.cat(tensors=(currents,voltages,potentials),dim=0)
    
    def ivp_knowns_mask(self):
        '''mask of known inputs values including source attributes in respective
          i and v props'''
        ics_mask = self.list_to_vec_mask(self.kinds_map[Kinds.ICS],True)
        ivs_mask = self.list_to_vec_mask(self.kinds_map[Kinds.IVS],True)
        attrs = self.list_to_vec_mask(self.knowns_map[Props.Attr],True)
        currents = self.list_to_vec_mask(self.knowns_map[Props.I],True)
        voltages = self.list_to_vec_mask(self.knowns_map[Props.V],True)
        potentials = self.list_to_vec_mask(self.knowns_map[Props.Pot],True)
        currents[ics_mask] = attrs[ics_mask]
        voltages[ivs_mask] = attrs[ivs_mask]
        return torch.cat(tensors=(currents,voltages,potentials),dim=0)

    def E(self):
        Z = self.Z()
        Y = self.Y()
        return torch.cat(tensors=(Z,Y,torch.zeros_like(Z)),dim=1)
    
    def I_knowns(self):
        Z_k = self.Z_known()
        return torch.cat(tensors=(
            Z_k,torch.zeros_like(Z_k),torch.zeros_like(Z_k)),dim=1)
    
    def V_knowns(self):
        Y_k = self.Y_known()
        return torch.cat(tensors=(
            torch.zeros_like(Y_k),Y_k,torch.zeros_like(Y_k)),dim=1)

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
    
class Process():
    def __init__(self, input:Input) -> None:
        self.input = input

    def errors(self, prediction:torch.Tensor):
        inputs = self.input.ivp_inputs()
        knowns_mask = self.input.ivp_knowns_mask()
        errors = inputs[:-1] - prediction
        errors[~knowns_mask[:-1]] = 0
        return errors
    
    def split(self, ivp:torch.Tensor):
        '''split a tensor that is (currents, voltages, potentials)
        Could be errors or predictions in that format'''
        num_elem = self.input.circuit.num_elements()
        i = ivp[:num_elem,:]
        v = ivp[num_elem:num_elem*2,:]
        p = ivp[2*num_elem:,:]
        return i,v,p

    def diffuse(self, prediction:torch.Tensor):
        M = self.input.circuit.M()
        i,v,_ = self.split(self.errors(prediction))
        return (M @ i).T @ M

    def propagate(self, errors: torch.Tensor):
        num_elem = self.input.circuit.num_elements()
        i_errors = errors[:num_elem,:]
        v_errors = errors[num_elem:num_elem*2,:]
        return v_errors / i_errors
