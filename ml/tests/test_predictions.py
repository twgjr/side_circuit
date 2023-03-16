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

    def is_close(self, a, b, tol=1e-6):
        return abs(a-b)/max(a,b) < tol

    def test_ring_1IVS_1R_R_unknown(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,1)
        c.elements[0].attr = 1
        c.elements[1].i = 0.5
        d = Data(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold,
                             self.loss_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(2,c.elements[1].a_pred))

    def test_ring_1IVS_1R_IVS_unknown(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,1)
        c.elements[1].v = 1
        c.elements[1].attr = 2
        d = Data(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold,
                             self.loss_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(c.elements[1].v,c.elements[1].v_pred))

    def test_ring_1IVS_1R_IVS_and_R_unknown(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,1)
        c.elements[0].i = -0.5
        c.elements[1].v = 1
        d = Data(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold,
                             self.loss_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(c.elements[0].i,c.elements[0].i_pred))
        self.assertTrue(self.is_close(c.elements[1].v,c.elements[1].v_pred))

    def test_voltage_divider(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,2)
        c.elements[0].attr = 1
        c.elements[2].v = 0.1
        d = Data(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold,
                             self.loss_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(c.elements[0].attr,c.elements[0].v_pred))
        self.assertTrue(self.is_close(c.elements[2].v,c.elements[2].v_pred))
        
    def test_voltage_divider_large_delta(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,2)
        c.elements[0].attr = 1
        c.elements[2].v = 1e-12
        d = Data(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold,
                             self.loss_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(c.elements[2].v,c.elements[2].v_pred))

    def test_current_divider_large_delta(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,2)
        c.elements[0].attr = 1
        c.elements[1].i = 9e-12
        c.elements[2].i = 1e-12
        d = Data(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold,
                             self.loss_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(c.elements[1].i,c.elements[1].i_pred))
        self.assertTrue(self.is_close(c.elements[2].i,c.elements[2].i_pred))
