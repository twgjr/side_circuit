import torch
import torch.nn as nn
import circuits as ckt
import models as mdl

def train(model:mdl.Solver,optimizer:torch.optim.Adam,loss_fn:nn.MSELoss,
          truths,selection):
    model.train()
    A,preds,b = model()
    if(truths == None and selection == None):
        loss = loss_fn(A @ preds, b)
    else:
        loss = loss_fn(preds[selection], truths[selection])
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

def next_state(in_state, max_state):
    if(in_state < max_state):
        out_state = in_state + 1
    else:
        out_state = 0
    return out_state

def process_state(params:tuple[nn.Parameter], input:ckt.Input, solver:mdl.Solver,
                  state, max_state, threshold = 0.1, state_limit = 1000, 
                  device='cpu'):
        model:mdl.Solver = solver(input, params).to(device)
        opt = torch.optim.Adam(params=model.parameters(),lr=0.9)
        count = 0
        num_elements = solver.input.circuit.num_elements()
        num_nodes = solver.input.circuit.num_nodes()
        len_pred = num_elements + num_nodes - 1

        while(True):
            #save previous state of params before next learning step
            for param_item in params:
                prev_params = param_item.clone().detach()
            loss,_ = train(model,opt,nn.MSELoss(),torch.zeros(size=(len_pred)))
            if(is_stable(prev_params, params, threshold)
                or 
                count > state_limit):
                return loss, next_state(state,max_state)
            count +=1