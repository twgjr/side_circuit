import unittest
from circuits import Kinds,Signal,System
from learn import Trainer
import torch

class TestLearning(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.max_epochs = 1000
        self.learning_rate = 0.5e-1
        self.stable_threshold = 1e-10

    def is_close(self, a:Signal, b:Signal, tol=1e-3):
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
        system = System()
        circuit = system.ring(Kinds.IVS,Kinds.R,2)
        ivs = circuit.elements[0]
        res_top = circuit.elements[1]
        load = circuit.elements[2]
        ivs.v = [1]
        load.v = [0.1]
        trainer = Trainer(system,self.learning_rate)
        pred,loss,epoch = trainer.run(self.max_epochs,self.stable_threshold)
        system.load(pred)
        self.assertTrue(self.is_close(-ivs.i_pred,res_top.i_pred))
        self.assertTrue(self.is_close(res_top.i_pred,load.i_pred))
        self.assertTrue(self.is_close(ivs.v,ivs.v_pred))
        self.assertTrue(self.is_close(load.v,load.v_pred))

    def test_switched_resistor(self):
        system = System()
        parent, child = system.switched_resistor()
        par_ivs = parent.elements[0]
        par_r = parent.elements[1]
        par_vc = parent.elements[2]
        ch_ivs = child.elements[0]
        ch_sw = child.elements[1]
        ch_r = child.elements[2]
        par_ivs.v = [1.0,0.0]
        ch_ivs.v = [0.1,0.1]
        par_r.a = 1.0
        ch_r.a = 1.0
        trainer = Trainer(system,self.learning_rate)
        pred,loss,epoch = trainer.run(self.max_epochs, self.stable_threshold)
        system.load(pred)
        test_par_r_i = Signal(None,[1.0,0.0])
        self.assertTrue(self.is_close(par_r.i_pred,test_par_r_i))
        test_par_r_v = Signal(None,[1.0,0.0])
        self.assertTrue(self.is_close(par_r.v_pred,test_par_r_v))
        test_ch_r_i = Signal(None,[0.1,0.0])
        self.assertTrue(self.is_close(ch_r.i_pred,test_ch_r_i))
        test_ch_r_v = Signal(None,[0.1,0.0])
        self.assertTrue(self.is_close(ch_r.v_pred,test_ch_r_v))
