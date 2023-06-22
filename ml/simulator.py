from circuits import System,Kinds,Circuit,Element,Props
from ngspice_lib import ngspice_api
from enum import Enum

class Modes(Enum):
    DC = 0
    TR = 1
    
class Simulator():
    '''steps through transient simulation of each system solution'''
    def __init__(self, system: System):
        self.system = system

    def run(self, stop:float, init_step:float,):
            # insert code here for running simulation
            sol_t = None
            time = None
            self.system.load(sol_t,time)