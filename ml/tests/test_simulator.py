import unittest
from circuits import Kinds,System
from simulator import Simulator
from math import pi as PI
from math import sqrt,exp

class TestSimulator(unittest.TestCase):
    def test_voltage_divider(self):
        system = System()
        circuit = system.ring(Kinds.VS,Kinds.R,2)
        ivs = circuit.elements[0]
        res_top = circuit.elements[1]
        res_low = circuit.elements[2]
        ivs.v[0.0] = 10
        ivs.v[0.5] = 20
        res_top.a[0.0] = 9
        res_low.a[0.0] = 1
        trainer = Simulator(system)
        trainer.run(stop=1,step_size=0.1)
        v_step0 = res_low.v_sol[0.0]
        self.assertTrue(v_step0==1.0)
        v_step1 = res_low.v_sol[0.5]
        self.assertTrue(v_step1==2.0)

    def test_RC(self):
        system = System()
        ivs = system.add_element_of(Kinds.VS)
        res = system.add_element_of(Kinds.R)
        cap = system.add_element_of(Kinds.C)
        system.connect(ivs.high,res.high)
        system.connect(res.low,cap.high)
        system.connect(cap.low,ivs.low)
        
        R = 1e3
        tau = 1e-6
        stop = 5*tau
        Vpk = 10
        ivs.v[0.0] = Vpk
        res.a[0.0] = R
        cap.a[0.0] = tau/R
        trainer = Simulator(system)
        trainer.run(stop=stop,step_size=tau)
        for i in range(1,6):
            cap_v = cap.v_sol[tau*i]
            cap_v_test = Vpk*(1-exp(-tau*i/tau))
            self.assertTrue(cap_v==cap_v_test)