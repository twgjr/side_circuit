import unittest
from circuits import Kinds,Signal,System
from learn import Trainer
import torch
from math import pi

class TestLearning(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.max_epochs = 1000
        self.learning_rate = 0.5e-1
        self.stable_threshold = 1e-10

    def test_voltage_divider(self):
        system = System()
        circuit = system.ring(Kinds.IVS,Kinds.R,2)
        ivs = circuit.elements[0]
        res_top = circuit.elements[1]
        res_low = circuit.elements[2]
        ivs.v[0.0] = 1.0
        res_low.v[0.0] = 0.1
        trainer = Trainer(system,self.learning_rate)
        pred,loss,epoch = trainer.run(self.max_epochs,self.stable_threshold)
        system.load(pred)
        self.assertTrue(-ivs.i_pred==res_top.i_pred)
        self.assertTrue(res_top.i_pred==res_low.i_pred)
        self.assertTrue(ivs.v==ivs.v_pred)
        self.assertTrue(res_low.v==res_low.v_pred)

    def test_gain(self):
        system = System()
        ivs = system.add_element_of(Kinds.IVS)
        vc_res = system.add_element_of(Kinds.R)
        vc, vg = system.add_element_pair(Kinds.VC,Kinds.VG)
        res = system.add_element_of(Kinds.R)
        system.connect(ivs.high,vc_res.high)
        system.connect(ivs.low,vc_res.low)
        system.connect(ivs.high,vc.high)
        system.connect(ivs.low,vc.low)
        system.connect(vg.high,res.high)
        system.connect(vg.low,res.low)
        ivs.v[0.0] = 0.1
        res.v[0.0] = 0.5
        trainer = Trainer(system,self.learning_rate)
        pred,loss,epoch = trainer.run(self.max_epochs,self.stable_threshold)
        system.load(pred)
        self.assertTrue(vg.a==Signal(None,{0:10.0}))

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
        test_par_r_i = Signal(None,{0:1.0,1:0.0})
        self.assertTrue(self.is_close(par_r.i_pred,test_par_r_i))
        test_par_r_v = Signal(None,{0:1.0,1:0.0})
        self.assertTrue(self.is_close(par_r.v_pred,test_par_r_v))
        test_ch_r_i = Signal(None,{0:0.1,1:0.0})
        self.assertTrue(self.is_close(ch_r.i_pred,test_ch_r_i))
        test_ch_r_v = Signal(None,{0:0.1,1:0.0})
        self.assertTrue(self.is_close(ch_r.v_pred,test_ch_r_v))

    def test_RC_low_pass_steady_state(self):
        system = System()
        ivs = system.add_element_of(Kinds.IVS)
        r = system.add_element_of(Kinds.R)
        c = system.add_element_of(Kinds.C)
        system.connect(ivs.high,r.high)
        system.connect(r.low,c.high)
        system.connect(c.low,ivs.low)
        # using the default sample dt of 1us (1kHz)
        # num_periods = 1
        period_samples = 10
        freq = period_samples/system.dt
        # p = 2*math.pi/period_samples # increment of the period in radians
        # ps = math.pi/2 # phase shift of RC 90deg in radians
        # per_rads = [i*p+ps for i in range(period_samples)] * num_periods
        ivs.v = int(period_samples/2)*[1] + int(period_samples/2)*[-1]  # 100Hz square wave
        r.a = 1.0
        RC = 1/(2*pi*freq)
        c.a = RC/r.a
        # c_v_test = [math.sin(rad)*cv_scale for rad in per_rads] * num_periods # 100Hz cos wave
        trainer = Trainer(system,self.learning_rate)
        pred,loss,epoch = trainer.run(self.max_epochs,self.stable_threshold)
        system.load(pred)
        self.assertTrue(self.is_close(ivs.v,ivs.v_pred))
        c.v_pred
        # self.assertTrue(self.is_close(c_v_test,c.v_pred))