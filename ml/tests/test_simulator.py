import unittest
from circuits import Kind,System,Quantity,Voltage,Resistor
from simulator import Simulator
from math import sqrt,exp    

class TestSimulator(unittest.TestCase):
    def test_dc_voltage_divider(self):
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
    
    def test_sine_voltage_divider(self):
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