import torch
import torch.nn as nn
from models import Solver
from models import State

def train(model:Solver,optimizer:torch.optim.Adam,loss_fn:nn.MSELoss):
    model.train()
    loss = None
    if(model.state == State.Solve):
        preds = model()
        inputs = model.input.ivp_inputs() #TODO input class attrib to speed up training
        knowns = model.input.ivp_knowns_mask() #TODO input class attrib to speed up training
        loss = loss_fn(preds[knowns[:-1]], inputs[knowns])
    elif(model.state == State.Lstsq):
        A,preds,b = model()
        loss = loss_fn(A @ preds, b)
    else:
        assert()
    model.zero_grad()
    loss.backward()
    model.zero_known_grads()
    optimizer.step()
    params = model.get_params()
    return loss,params

def list_params_filter_none(param_tpl:tuple[nn.Parameter]):
    param_lst = []
    for param in param_tpl:
        if(param == None):
            continue
        param_lst.append(param[param!=None])
    return param_lst

def is_stable(prev_tpl:tuple[nn.Parameter], updated_tpl:tuple[nn.Parameter], 
              pu_threshold:float):
    updated = torch.cat(tensors=list_params_filter_none(updated_tpl))
    prev = torch.cat(tensors=list_params_filter_none(prev_tpl))
    abs_val = torch.abs(updated - prev)
    prev_abs = torch.abs(prev)
    pu_change = abs_val / prev_abs
    pu_change_max = torch.max(pu_change)
    ret_bool = pu_change_max < pu_threshold
    return ret_bool

