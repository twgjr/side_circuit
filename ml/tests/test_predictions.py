import unittest
from circuits import Circuit,Kinds
from data import Data
from learn import Trainer
import torch

class TestSolutions(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.max_epochs = 1000
        self.learning_rate = 1e-1
        self.stable_threshold = 1e-5
        self.loss_threshold = 1e-20

    def run_and_check(self, c:Circuit, i:list, v:list, a:list):
        d = Data(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol, loss, epoch = trainer.run(self.max_epochs,
                             self.stable_threshold,
                             self.loss_threshold)
        i_test = torch.tensor(i).to(torch.float)
        v_test = torch.tensor(v).to(torch.float)
        a_test = torch.tensor(a).to(torch.float)
        sols = torch.cat((i_sol,v_sol,a_sol))
        tests = torch.cat((i_test,v_test,a_test))
        self.assertTrue(torch.allclose(sols,tests))
        self.assertTrue(torch.allclose(loss,torch.zeros_like(loss)))
        self.assertTrue(epoch < 100)

    def test_ring_1IVS_1R(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,1)
        c.elements[0].attr = 1
        c.elements[1].attr = 2
        i_test = [-0.500,  0.500]
        v_test = [1., 1.]
        a_test = [1., 2.]
        self.run_and_check(c,i_test,v_test,a_test)

    def test_ring_1IVS_1R_R_unknown(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,1)
        c.elements[0].attr = 1
        c.elements[1].i = 0.5
        i_test = [-0.500,  0.500]
        v_test = [1., 1.]
        a_test = [1., 2.]
        self.run_and_check(c,i_test,v_test,a_test)

    def test_ring_1IVS_1R_IVS_unknown(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,1)
        c.elements[1].v = 1
        c.elements[1].attr = 2
        i_test = [-0.500,  0.500]
        v_test = [1., 1.]
        a_test = [1., 2.]
        self.run_and_check(c,i_test,v_test,a_test)

    def test_ring_1IVS_1R_IVS_and_R_unknown(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,1)
        c.elements[0].i = -0.5
        c.elements[1].v = 1
        i_test = [-0.500,  0.500]
        v_test = [1., 1.]
        a_test = [1., 2.]
        self.run_and_check(c,i_test,v_test,a_test)

    def test_ladder_1IVS_2R(self):
        c = Circuit()
        c.ladder(Kinds.IVS,Kinds.R,2)
        c.elements[0].attr = 1
        c.elements[1].attr = 1
        c.elements[2].attr = 1
        i_test = [-2.,  1.,  1.]
        v_test = [1., 1., 1.]
        a_test = [1., 1., 1.]
        self.run_and_check(c,i_test,v_test,a_test)
