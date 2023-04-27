import unittest
from circuits import Circuit,Kinds,Signal
from data import CircuitData
from learn import Trainer
import torch

class TestSolutions(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.max_epochs = 1000
        self.learning_rate = 1e-1
        self.stable_threshold = 1e-3

    def is_close(self, a:Signal, b:Signal, tol=1e-6):
        assert type(a) == type(b)
        a_vals = []
        b_vals = []
        if(isinstance(a,Signal)):
            a_vals = a.get_data()
            b_vals = b.get_data()
        elif(isinstance(a,float)):
            a_vals = [a]
            b_vals = [b]
        else:
            raise ValueError("a and b must be of type Signal or float")
        for a,b in zip(a_vals,b_vals):
            if abs(a-b)/max(abs(a),abs(b),tol) > tol:
                return False
        return True

    def test_voltage_divider(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,2)
        c.elements[0].v = [1]
        c.elements[2].v = [0.1]
        d = CircuitData(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                                              self.stable_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(-c.elements[0].i_pred,c.elements[1].i_pred))
        self.assertTrue(self.is_close(c.elements[1].i_pred,c.elements[2].i_pred))
        self.assertTrue(self.is_close(c.elements[0].v,c.elements[0].v_pred))
        self.assertTrue(self.is_close(c.elements[2].v,c.elements[2].v_pred))

    def test_voltage_divider_sequence(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,2)
        c.elements[0].v = [1,0,-1]
        c.elements[2].v = [0.1,0,-0.1]
        d = CircuitData(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                                              self.stable_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(c.elements[0].v,c.elements[0].v_pred))
        self.assertTrue(self.is_close(c.elements[2].v,c.elements[2].v_pred))

    def test_voltage_divider_large(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,100)
        c.elements[0].v = [1.0]
        c.elements[-1].v = [0.1]
        d = CircuitData(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(c.elements[0].v,c.elements[0].v_pred))
        self.assertTrue(self.is_close(c.elements[-1].v,c.elements[-1].v_pred))
        
    def test_voltage_divider_large_delta(self):
        c = Circuit()
        c.ring(Kinds.IVS,Kinds.R,2)
        c.elements[0].v = [1.0]
        c.elements[2].v = [1e-12]
        d = CircuitData(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(c.elements[2].v,c.elements[2].v_pred))

    def test_current_divider_large_delta(self):
        c = Circuit()
        c.ladder(Kinds.IVS,Kinds.R,2)
        c.elements[0].v = [1.0]
        c.elements[1].i = [1.0]
        c.elements[2].i = [1e-12]
        d = CircuitData(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(c.elements[1].i,c.elements[1].i_pred))
        self.assertTrue(self.is_close(c.elements[2].i,c.elements[2].i_pred))
        self.assertTrue(self.is_close(c.elements[0].v,c.elements[1].v_pred))
        self.assertTrue(self.is_close(c.elements[0].v,c.elements[2].v_pred))

    def test_switched_resistor(self):
        c = Circuit()
        src, ctl_src, res, ctl_res, ctl, sw = c.switched_resistor()
        src.v = [1.0]
        ctl_src.v = [0.1]
        res.a = 1.0
        ctl_res.a = 1.0
        d = CircuitData(c)
        trainer = Trainer(d,self.learning_rate)
        i_sol, v_sol, a_sol,_,_ = trainer.run(self.max_epochs,
                             self.stable_threshold)
        c.load(i_sol,v_sol,a_sol)
        self.assertTrue(self.is_close(src.v_pred,res.v_pred))
        self.assertTrue(self.is_close(src.i,-res.i))
