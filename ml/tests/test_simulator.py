import unittest
from circuits import Kind,System,Quantity,Voltage,Current,Resistor
from simulator import Simulator
from math import pi as PI
from math import sqrt,exp    

class TestSimulator(unittest.TestCase):
    def test_voltage_divider(self):
        system = System()
        circuit = system.ring(Kind.V,Kind.R,2)
        ivs = circuit.elements[0]
        res_top = circuit.elements[1]
        res_low = circuit.elements[2]
        ivs.add_data(Quantity.V,0.0,10)
        ivs:Voltage
        ivs.dc = 10
        res_top:Resistor
        res_top.parameter = 9
        res_low:Resistor
        res_low.parameter = 1
        simulator = Simulator(system)
        simulator.set_tran(tstop=1,tstep=0.1)
        simulator.run()
        v_step0 = res_low.data[Quantity.V][0.0]
        self.assertTrue(v_step0==1.0)

    def test_RC(self):
        system = System()
        ivs = system.add_element_of(Kind.V)
        res = system.add_element_of(Kind.R)
        cap = system.add_element_of(Kind.C)
        system.connect(ivs.high,res.high)
        system.connect(res.low,cap.high)
        system.connect(cap.low,ivs.low)
        
        R = 1e3
        tau = 1e-6
        stop = 5*tau
        Vpk = 10
        ivs.v[0.0] = Vpk
        res.data[0.0] = R
        cap.data[0.0] = tau/R
        trainer = Simulator(system)
        trainer.run(stop=stop,step_size=tau)
        for i in range(1,6):
            cap_v = cap.v_sol[tau*i]
            cap_v_test = Vpk*(1-exp(-tau*i/tau))
            self.assertTrue(cap_v==cap_v_test)