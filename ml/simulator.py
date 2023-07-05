from circuits import System,Kind,Circuit,Element,Quantity,Voltage,Resistor,Pulse
from ngspice_lib import ngspice_api as ngsp
from enum import Enum
import pprint

class Modes(Enum):
    DC = 0
    tran = 1

class Simulator():
    '''steps through transient simulation of each system solution'''
    def __init__(self, system: System):
        self.system = system
        self.spice = ngsp.Spice(system,dll_path=r'.\ngspice_lib\Spice64_dll\dll-vs\ngspice.dll')
        self.load_system()

    def load_system(self):
        self.spice.circ_by_line("Title")
        for element in self.system.elements:
            line = element.to_spice()
            self.spice.circ_by_line(line)

    def set_tran(self, tstep:float, tstop:float, tmax:float=None, 
                 uic:bool=False) -> None:
        if(tmax == None):
            tmax = tstep
        self.spice.trans(tstep, tstop, tmax, uic)

    def run(self):
        self.spice.end_circuit()
        self.spice.run()
        self.spice.quit()

if(__name__=="__main__"):
    '''
        R1 in out 1k   ; Resistor with resistance 1k Ohm
        C1 out 0 1u ; Capacitor with capacitance 1uF
        Vin in 0 PULSE(0 10 0 0.1ms 0.1ms 10ms)
    '''
    # system = System()
    # rc = system.rc()
    # v = rc.elements[0]
    # v
    # r = rc.elements[1]
    # r.add_data(Quantity.A,0.0,1e3)
    # c = rc.elements[2]
    # r.add_data(Quantity.A,0.0,1e-6)
    # simulator = Simulator(system)
    # simulator.load_system()
    # simulator.set_tran(tstep=0.1e-3,tstop=10e-3)
    # simulator.run()
    system = System()
    circuit = system.ring(Kind.V,Kind.R,2)
    ivs = circuit.elements[0]
    res_top = circuit.elements[1]
    res_low = circuit.elements[2]
    ivs:Voltage
    # ivs.dc = 10
    ivs.sig_func = Pulse(val1=0,val2=1,freq=1e3)
    res_top:Resistor
    res_top.parameter = 9
    res_low:Resistor
    res_low.parameter = 1
    simulator = Simulator(system)
    simulator.set_tran(tstop=1e-3,tstep=1e-3/50)
    simulator.run()
    for element in system.elements:
        print(f'element{element}')
        for qty in element.data:
            if(len(element.data[qty]) > 0):
                print(f'has {qty}')
    for node in system.nodes:
        print(f'node{node}')
        for qty in node.data:
            if(len(node.data[qty]) > 0):
                print(f'has {qty}')