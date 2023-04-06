import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from models import Cell
from data import Data

class Stability():
    '''takes attributes from solver and determines if the all attributes have 
    converged to a stable value.'''
    def __init__(self, tensor_list:list[Tensor], pu_threshold:float) -> None:
        self.tensor_list = []
        if(len(tensor_list) > 0):
            for tensor in tensor_list:
                self.tensor_list.append(tensor.detach().clone())
        self.pu_threshold = pu_threshold
    
    def is_stable(self,tensor_list:list[Tensor]):
        assert len(self.tensor_list) > 0
        ret_bool = True
        for index in range(len(tensor_list)):
            new_tensor = tensor_list[index].detach().clone()
            abs_val = torch.abs(new_tensor - self.tensor_list[index])
            prev_abs = torch.abs(self.tensor_list[index])
            pu_change = abs_val / prev_abs
            pu_change_max = torch.max(pu_change)
            if(pu_change_max > self.pu_threshold):
                ret_bool = False
        self.tensor_list = tensor_list
        return ret_bool

class Trainer():
    def __init__(self, data:Data, init_learn_rate:float) -> None:
        self.data = data
        self.model = Cell(data=data)
        self.optimizer = Adam(params=self.model.parameters(),lr=init_learn_rate)
        self.loss_fn = nn.MSELoss()
        self.dataset = data.init_dataset()
        self.mask = torch.tensor(data.data_mask_list()).to(torch.bool).unsqueeze(1)
        (self.i_mask,self.v_mask) = self.data.split_input_output(self.mask)

    def step_cell(self,input:Tensor):
        '''processes one time step of the sequence'''
        out = self.model.forward(input)
        loss = self.loss_fn(out[self.mask], input[self.mask])
        return loss,out
    
    def step_sequence(self):
        '''Calls step() for each item in self.dataset. Returns the total
          loss, the model parameters, and the output of the model for each time
          step.'''
        loss_list = []
        input = None
        out_list = []
        for input in self.dataset:
            loss, out = self.step_cell(input)
            out_list.append(out)
            loss_list.append(loss)
        total_loss = sum(loss_list)
        self.model.zero_grad()
        total_loss.backward()
        self.model.zero_known_grads()
        self.optimizer.step()
        self.model.clamp_params()
        return total_loss, out_list
    
    def calc_params(self, preds_list:list[Tensor], knowns_list:list[Tensor]):
        '''Directly calculates the attributes of the circuit given the predicted
          values and the known values.  Returns the average of the parameters 
          across all time steps.'''
        avg_list = []
        for preds, knowns in zip(preds_list, knowns_list):
            avg_list.append(self.calc_params_single(preds,knowns))
        combined = torch.cat(avg_list,dim=1)
        mean_params = torch.mean(combined,dim=1,keepdim=False)
        return mean_params
    
    def calc_params_single(self, preds:Tensor, knowns:Tensor):
        '''Directly calculates the circuit attribute values based on the known
        values and the most recent predictions based on one time step.'''
        assert(preds.shape == (2*self.data.circuit.num_elements(),1))
        assert(preds.shape == knowns.shape)
        with torch.no_grad():
            i,v = self.data.split_input_output(preds)
            i_known,v_known = self.data.split_input_output(knowns)
            # i_known_mask,v_known_mask = self.data.split_input_output(knowns_mask)
            i[self.i_mask] = i_known[self.i_mask]
            v[self.v_mask] = v_known[self.v_mask]
            r_mask = self.data.r_mask
            # self.params[r_mask] = v[r_mask]/i[r_mask]
            r_params = torch.zeros(size=(self.data.circuit.num_elements(),1))
            eps = 1e-15
            r_params[r_mask] = (v[r_mask]+eps)/(i[r_mask]+eps)
            return r_params


    def run(self, epochs, stable_threshold:float):
        self.model.train()
        loss = None
        params = self.model.params
        out_list = [torch.tensor([0.0]*2*len(self.data.circuit.elements))
                    .unsqueeze(1).float()]*self.data.circuit.signal_len
        attr_stability = Stability([params], stable_threshold)
        preds_stability = Stability(out_list, stable_threshold)
        epoch = 0
        reset_count = 0
        while(epoch < epochs):
            loss, out_list = self.step_sequence()
            epoch += 1
            params = self.model.params
            if(epoch % 2 == 1):
                if(attr_stability.is_stable([params]) and 
                    preds_stability.is_stable(out_list)):
                    print('threshold met')
                    break
                params_recalc = self.calc_params(out_list, self.dataset)
                self.model.set_param_vals(params_recalc)
                reset_count += 1
        print(f'reset_count = {reset_count}')
        params_recalc = self.calc_params(out_list, self.dataset)
        self.model.set_param_vals(params_recalc)
        i_sol, v_sol = self.data.denorm_input_output(out_list)
        a_sol = self.data.denorm_params(params)
        return i_sol, v_sol, a_sol, loss, epoch
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        
    def set_lr(self,optimizer,lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
