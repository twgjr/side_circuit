import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from circuits import System,Props,Kinds
from models import DynamicModule,CircuitModule,ElementCoeff,ElementConstant
from data import Data
from enum import Enum

class States(Enum):
    UNSTABLE = 0
    COURSE = 1
    FINE = 2

class ParamStability():
    '''takes parameters from model and determines if the each parameter has 
    crossed a stability threshold.'''
    def __init__(self, tensor_list:list[Tensor], pu_threshold:float) -> None:
        self.param_list = []
        if(len(tensor_list) > 0):
            for tensor in tensor_list:
                self.param_list.append(tensor.detach().clone())
        self.pu_threshold = pu_threshold
        self.course_threshold = max(self.pu_threshold,1e10*self.pu_threshold)
        self.state = States.UNSTABLE
    
    def check_stable(self,tensor_list:list[Tensor]):
        assert len(self.param_list) > 0
        is_course = True
        is_fine = True
        for index in range(len(tensor_list)):
            new_tensor = tensor_list[index].detach().clone()
            abs_val = torch.abs(new_tensor - self.param_list[index])
            prev_abs = torch.abs(self.param_list[index])
            pu_change = abs_val / prev_abs
            pu_change_max = torch.max(pu_change)
            if(pu_change_max > self.pu_threshold):
                is_fine = False
            if(pu_change_max > self.course_threshold):
                is_course = False
        self.param_list = tensor_list
        if(self.state == States.UNSTABLE):
            if(is_course or is_fine):
                self.state = States.COURSE
        elif(self.state == States.COURSE):
            if(is_fine):
                self.state = States.FINE
            else:
                self.state = States.UNSTABLE
        return self.state

class Trainer():
    def __init__(self, system:System, learn_rate:float) -> None:
        self.model = DynamicModule(system)
        self.data = Data(system,self.model)
        self.learn_rate = learn_rate
        self.optimizer = None
        self.init_optimizer()
        self.loss_fn = nn.MSELoss()

    def init_optimizer(self):
        self.optimizer = Adam(params=self.model.parameters(),lr=self.learn_rate)#,
                              #weight_decay=1-self.learn_rate)
    
    def step_sequence(self):
        '''Returns the total loss, the model parameters, and the output of the 
        model for all time steps.'''
        model_out = self.model.forward()
        system_sequence = model_out[0]
        delta_err_out = model_out[1]
        ctrl_el_err_out = model_out[2]
        sequence_loss_list = []
        for time,sys_t in system_sequence.items():
            for c,ckt_t_out in enumerate(sys_t):
                ckt_t_data = self.data.sequence[time].circuits[c]
                ckt_t_mask = self.data.masks[c]
                for prop in ckt_t_mask:
                    if(prop==Props.A):
                        continue
                    pred_mask = ckt_t_mask[prop]
                    pred_prop = ckt_t_out[prop]
                    pred = pred_prop[pred_mask]
                    if(len(pred)==0):continue
                    knowns = ckt_t_data[prop]
                    loss = self.loss_fn(pred, knowns)
                    sequence_loss_list.append(loss)
        ctrl_loss = self.loss_fn(ctrl_el_err_out, torch.zeros_like(ctrl_el_err_out))
        delta_loss = self.loss_fn(delta_err_out, torch.zeros_like(delta_err_out))
        sequence_loss = sum(sequence_loss_list)
        total_loss = sequence_loss + ctrl_loss + delta_loss
        num_optimizer_params = sum(p.numel() for p in self.optimizer.param_groups[0]['params'])
        num_model_params = sum(p.numel() for p in self.model.parameters())
        if(num_optimizer_params != num_model_params):
            self.init_optimizer()
        #################################
        # from torchviz import make_dot
        # # Visualize the computation graph
        # dot = make_dot(total_loss, params=dict(self.model.named_parameters()))
        # dot.render("computation_graph")
        #################################
        total_loss.backward()
        self.optimizer.step()
        # self.model.clamp_params()
        return total_loss, system_sequence
    
    def run(self, epochs, stable_threshold:float):
        loss_prev, system_sequence = self.step_sequence()
        params = list(self.model.parameters())
        param_stability = ParamStability(params, stable_threshold)
        epoch = 0
        while(epoch < epochs):
            loss, system_sequence = self.step_sequence()
            if(epoch % 10 == 9):
                with torch.no_grad():
                    self.reinforce(system_sequence)
            params = list(self.model.parameters())
            if(len(params)>len(param_stability.param_list)):
                param_stability = ParamStability(params, stable_threshold)
                continue
            elif(epoch % 10 == 9):
                state = param_stability.check_stable(params)
                if(state==States.FINE):
                    break
            epoch += 1
        return system_sequence, loss, epoch
    
    def reinforce(self,system_sequence:dict):
        '''Manually set an element parameter based on known values and recent 
        predictions.  Help reduce epochs to reach solution.'''
        for time in system_sequence:
            for ckt_t_mod in self.model.system_mod.circuits:
                ckt_t_mod:CircuitModule
                for element in ckt_t_mod.A.elements:
                    static_param:nn.Parameter = element.values.get_param_static(time)
                    if(static_param==None):
                        continue
                    if(static_param.requires_grad == False):
                        continue
                    element:ElementCoeff
                    i=None
                    if(time in element.element.i):
                        #get known value
                        i=element.element.i[time]
                    else:
                        #get predicted value
                        ckt_idx=element.element.circuit.index
                        el_idx=element.element.index
                        i=system_sequence[time][ckt_idx][Props.I][el_idx]
                    v=None
                    if(time in element.element.v):
                        #get known value
                        v=element.element.v[time]
                    else:
                        #get predicted value
                        ckt_idx=element.element.circuit.index
                        el_idx=element.element.index
                        v=system_sequence[time][ckt_idx][Props.V][el_idx]
                    if(element.element.kind == Kinds.R):
                        element.values.reinforce_param(time,v/i)
                        
                for element in ckt_t_mod.b.elements:
                    element:ElementConstant
                    if(element.element.kind == Kinds.VG or 
                       element.element.kind == Kinds.CG):
                        assert()
    
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def set_lr(self,lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
