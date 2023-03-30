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
        circuit.elements[0].v.data = [3.0]
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
        circuit.elements[0].v.data = [3.0]
        circuit.elements[1].i.data = [1.5]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        input = trainer.dataset[0]
        loss,out = trainer.step_cell(input)
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(trainer.model.params[1],torch.tensor(1.0)))
        test_out = torch.tensor([-1,1,1,1]).unsqueeze(1).to(torch.float)
        self.assertTrue(torch.allclose(out,test_out))
        
    def test_step_sequence(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v.data = [3.0]
        circuit.elements[1].a = 2.0
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        loss,out_list = trainer.step_sequence()
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(trainer.model.params[1],torch.tensor(1.0)))
        test_out_list = [
            torch.tensor([-1,1,1,1]).unsqueeze(1).to(torch.float)
            ]
        for out,test_out in zip(out_list,test_out_list):
            self.assertTrue(torch.allclose(out,test_out))

    def test_run_single(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v.data = [3.0]
        circuit.elements[1].i.data = [1.5]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        i_sol, v_sol, a_sol, loss, epoch = trainer.run(1,1e-3)
        i_sol_test = torch.tensor([-1.5,1.5]).unsqueeze(1).to(torch.float)
        v_sol_test = torch.tensor([3.0,3.0]).unsqueeze(1).to(torch.float)
        a_sol_test = torch.tensor([0,2.0]).unsqueeze(1).to(torch.float)
        self.assertTrue(torch.allclose(i_sol[0],i_sol_test))
        self.assertTrue(torch.allclose(v_sol[0],v_sol_test))
        self.assertTrue(torch.allclose(a_sol[1],a_sol_test[1]))

    def test_run_multiple(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v.data = [3.0,6.0]
        circuit.elements[1].i.data = [1.5,3.0]
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        i_sol, v_sol, a_sol, loss, epoch = trainer.run(2,1e-3)
        i_sol_test = [torch.tensor([-1.5,1.5]).unsqueeze(1).to(torch.float),
                      torch.tensor([-3.0,3.0]).unsqueeze(1).to(torch.float)]
        v_sol_test = [torch.tensor([3.0,3.0]).unsqueeze(1).to(torch.float),
                      torch.tensor([6.0,6.0]).unsqueeze(1).to(torch.float)]
        a_sol_test = torch.tensor([0,2.0]).unsqueeze(1).to(torch.float)
        self.assertTrue(torch.allclose(i_sol[0],i_sol_test[0]))
        self.assertTrue(torch.allclose(i_sol[1],i_sol_test[0]))
        self.assertTrue(torch.allclose(v_sol[0],v_sol_test[0]))
        self.assertTrue(torch.allclose(v_sol[1],v_sol_test[1]))
        self.assertTrue(torch.allclose(a_sol[1],a_sol_test[1]))