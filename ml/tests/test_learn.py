import unittest
from circuits import Circuit,Kinds
from data import Data
from learn import Trainer
from torch.nn import MSELoss
from models import Cell
from torch.optim import Adam
import torch

class TestTrainer(unittest.TestCase):
    def test_Trainer(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v = [3.0]
        circuit.elements[1].a = 2.0
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        self.assertTrue(isinstance(trainer.data,Data))
        self.assertTrue(isinstance(trainer.model,Cell))
        self.assertTrue(isinstance(trainer.optimizer,Adam))
        self.assertTrue(isinstance(trainer.loss_fn,MSELoss))

    def test_step_cell(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [3.0]
        circuit.elements[1].i= [1.5]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        input = trainer.dataset[0]
        loss,out = trainer.step_cell(input)
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(trainer.model.params[1],torch.tensor(1.0)))
        test_out = torch.tensor([-1,1,1,1]).unsqueeze(1).to(torch.float)
        self.assertTrue(torch.allclose(out,test_out))

    def test_step_cell_with_neg(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [-3.0]
        circuit.elements[1].i= [-1.5]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        input = trainer.dataset[0]
        loss,out = trainer.step_cell(input)
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(trainer.model.params[1],torch.tensor(1.0)))
        test_out = torch.tensor([1,-1,-1,-1]).unsqueeze(1).to(torch.float)
        self.assertTrue(torch.allclose(out,test_out))

    def test_step_cell_with_zero(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [0.0]
        circuit.elements[1].i= [0.0]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        input = trainer.dataset[0]
        loss,out = trainer.step_cell(input)
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(trainer.model.params[1],torch.tensor(1.0)))
        test_out = torch.tensor([0,0,0,0]).unsqueeze(1).to(torch.float)
        self.assertTrue(torch.allclose(out,test_out))
        
    def test_step_sequence(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [3.0]
        circuit.elements[1].a = 2.0
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        self.assertTrue(len(out_list) == 1)
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(trainer.model.params[1],torch.tensor(1.0)))
        test_out_list = [torch.tensor([-1,1,1,1]).unsqueeze(1).to(torch.float)]
        for out,test_out in zip(out_list,test_out_list):
            self.assertTrue(torch.allclose(out,test_out))

    def test_step_sequence_many(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [4.0,8.0]
        circuit.elements[1].a = 2.0
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        self.assertTrue(len(out_list) == 2)
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(trainer.model.params[1],torch.tensor(1.0)))
        test_out_list = [
            torch.tensor([-0.5,0.5,0.5,0.5]).unsqueeze(1).to(torch.float),
            torch.tensor([-1,1,1,1]).unsqueeze(1).to(torch.float)]
        for out,test_out in zip(out_list,test_out_list):
            self.assertTrue(torch.allclose(out,test_out))

    def test_step_sequence_many_with_zero(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [4.0,0.0]
        circuit.elements[1].a = 2.0
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        self.assertTrue(len(out_list) == 2)
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(trainer.model.params[1],torch.tensor(1.0)))
        test_out_list = [
            torch.tensor([-1,1,1,1]).unsqueeze(1).to(torch.float),
            torch.tensor([0,0,0,0]).unsqueeze(1).to(torch.float)]
        for out,test_out in zip(out_list,test_out_list):
            self.assertTrue(torch.allclose(out,test_out))

    def test_step_sequence_many_with_neg(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [4.0,-4.0]
        circuit.elements[1].a = 2.0
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        self.assertTrue(len(out_list) == 2)
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(trainer.model.params[1],torch.tensor(1.0)))
        test_out_list = [
            torch.tensor([-1,1,1,1]).unsqueeze(1).to(torch.float),
            torch.tensor([1,-1,-1,-1]).unsqueeze(1).to(torch.float)]
        for out,test_out in zip(out_list,test_out_list):
            self.assertTrue(torch.allclose(out,test_out))

    def test_calc_params_single(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [3.0]
        circuit.elements[1].i = [1.5]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        params = trainer.calc_params_single(out_list[0],trainer.dataset[0])
        self.assertTrue(torch.allclose(params[1],torch.tensor(1.0)))

    def test_calc_params_single_with_zero(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [0.0]
        circuit.elements[1].i = [0.0]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        params = trainer.calc_params_single(out_list[0],trainer.dataset[0])
        self.assertTrue(torch.allclose(params[1],torch.tensor(1.0)))

    def test_calc_params_single_with_neg(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [-3.0]
        circuit.elements[1].i = [-1.5]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        params = trainer.calc_params_single(out_list[0],trainer.dataset[0])
        self.assertTrue(torch.allclose(params[1],torch.tensor(1.0)))

    def test_calc_params(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [3.0,6.0]
        circuit.elements[1].i = [1.5,3.0]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        params = trainer.calc_params(out_list,trainer.dataset)
        self.assertTrue(torch.allclose(params[1],torch.tensor(1.0)))

    def test_calc_params_with_neg(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [3.0,-6.0]
        circuit.elements[1].i = [1.5,-3.0]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        params = trainer.calc_params(out_list,trainer.dataset)
        self.assertTrue(torch.allclose(params[1],torch.tensor(1.0)))

    def test_calc_params_with_zero(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v= [3.0,0.0]
        circuit.elements[1].i = [1.5,0.0]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        params = trainer.calc_params(out_list,trainer.dataset)
        self.assertTrue(torch.allclose(params[1],torch.tensor(1.0)))

    def test_run_single(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v = [3.0]
        circuit.elements[1].i = [1.5]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        i_sol, v_sol, a_sol, loss, epoch = trainer.run(1000,1e-3)
        self.assertTrue(len(i_sol) == 1)
        self.assertTrue(len(v_sol) == 1)
        i_sol_test = torch.tensor([-1.5,1.5]).unsqueeze(1).to(torch.float)
        v_sol_test = torch.tensor([3.0,3.0]).unsqueeze(1).to(torch.float)
        a_sol_test = torch.tensor([0,2.0]).unsqueeze(1).to(torch.float)
        self.assertTrue(torch.allclose(i_sol[0],i_sol_test))
        self.assertTrue(torch.allclose(v_sol[0],v_sol_test))
        self.assertTrue(torch.allclose(a_sol[1],a_sol_test[1]))

    def test_run_multiple(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v = [3.0,6.0]
        circuit.elements[1].i = [1.5,3.0]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        i_sol, v_sol, a_sol, loss, epoch = trainer.run(1000,1e-3)
        self.assertTrue(len(i_sol) == 2)
        self.assertTrue(len(v_sol) == 2)
        i_sol_test = [torch.tensor([-1.5,1.5]).unsqueeze(1).to(torch.float),
                      torch.tensor([-3.0,3.0]).unsqueeze(1).to(torch.float)]
        v_sol_test = [torch.tensor([3.0,3.0]).unsqueeze(1).to(torch.float),
                      torch.tensor([6.0,6.0]).unsqueeze(1).to(torch.float)]
        a_sol_test = torch.tensor([0,2.0]).unsqueeze(1).to(torch.float)
        self.assertTrue(torch.allclose(i_sol[0],i_sol_test[0]))
        self.assertTrue(torch.allclose(i_sol[1],i_sol_test[1]))
        self.assertTrue(torch.allclose(v_sol[0],v_sol_test[0]))
        self.assertTrue(torch.allclose(v_sol[1],v_sol_test[1]))
        self.assertTrue(torch.allclose(a_sol[1],a_sol_test[1]))

    def test_run_multiple_with_zeros(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v = [3.0,6.0,0.0]
        circuit.elements[1].i = [1.5,3.0,0.0]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        i_sol, v_sol, a_sol, loss, epoch = trainer.run(1000,1e-3)
        self.assertTrue(len(i_sol) == 3)
        self.assertTrue(len(v_sol) == 3)
        i_sol_test = [torch.tensor([-1.5,1.5]).unsqueeze(1).to(torch.float),
                      torch.tensor([-3.0,3.0]).unsqueeze(1).to(torch.float),
                      torch.tensor([0.0,0.0]).unsqueeze(1).to(torch.float)]
        v_sol_test = [torch.tensor([3.0,3.0]).unsqueeze(1).to(torch.float),
                      torch.tensor([6.0,6.0]).unsqueeze(1).to(torch.float),
                      torch.tensor([0.0,0.0]).unsqueeze(1).to(torch.float)]
        a_sol_test = torch.tensor([0,2.0]).unsqueeze(1).to(torch.float)
        self.assertTrue(torch.allclose(i_sol[0],i_sol_test[0]))
        self.assertTrue(torch.allclose(i_sol[1],i_sol_test[1]))
        self.assertTrue(torch.allclose(v_sol[0],v_sol_test[0]))
        self.assertTrue(torch.allclose(v_sol[1],v_sol_test[1]))
        self.assertTrue(torch.allclose(a_sol[1],a_sol_test[1]))