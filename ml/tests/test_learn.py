import unittest
from circuits import Circuit,Kinds
from data import Data
from learn import Trainer
from torch.nn import MSELoss
from models import Solver
from torch.optim import Adam
import torch

class TestTrainer(unittest.TestCase):
    def test_Trainer(self):
        circuit = Circuit()
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        self.assertTrue(isinstance(trainer.data,Data))
        self.assertTrue(isinstance(trainer.model,Solver))
        self.assertTrue(isinstance(trainer.optimizer,Adam))
        self.assertTrue(isinstance(trainer.loss_fn,MSELoss))

    def test_step(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 1
        circuit.elements[1].attr = 2
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        target = torch.tensor(
            data.target_list(trainer.model.i_base, trainer.model.v_base
            )).to(torch.float).unsqueeze(dim=1)
        target_mask = torch.tensor(
            trainer.model.data.target_mask_list()).to(torch.bool).unsqueeze(dim=1)
        loss,attr,preds = trainer.step(target,target_mask)
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(torch.allclose(attr,torch.ones_like(attr)))
        self.assertTrue(
            torch.allclose(
                preds,
                torch.tensor([-1,1,1,1]).unsqueeze(dim=1).to(torch.float)))
        
    def test_run(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 1
        circuit.elements[1].attr = 2
        data = Data(circuit)
        trainer = Trainer(data,0.01)
        i_sol, v_sol, a_sol, loss, epoch = trainer.run(100,0.01,0.01)
        self.assertTrue(torch.allclose(i_sol,torch.tensor([-0.5,0.5])
                                       .to(torch.float)))
        self.assertTrue(torch.allclose(v_sol,torch.tensor([1,1])
                                       .to(torch.float)))
        self.assertTrue(torch.allclose(a_sol,torch.tensor([1,2])
                                       .to(torch.float)))
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(epoch < 100)




