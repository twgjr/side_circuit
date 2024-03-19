from modules.simulator import Simulator
from modules.system import System, Wire
from modules.elements import *

import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()


def test_system_to_pyspice():
    # create a voltage resistor circuit
    system = System()
    voltage = V(system).DC(10)
    resistor = R(system, 100)
    Wire(system, voltage.p, resistor.p)
    Wire(system, voltage.n, system.gnd)
    Wire(system, resistor.n, system.gnd)
    # convert to PySpice
    simulator = Simulator(system)
    circuit = simulator.system_to_pyspice("test")
    assert len(circuit.elements) == 2


def test_simulator():
    system = System()
    voltage = V(system).DC(10)
    top_resistor = R(system, 9)
    bottom_resistor = R(system, 1)
    Wire(system, voltage.p, top_resistor.p)
    Wire(system, voltage.n, system.gnd)
    Wire(system, top_resistor.n, bottom_resistor.p)
    Wire(system, bottom_resistor.n, system.gnd)

    # convert to PySpice
    sim = Simulator(system)
    circuit = sim.system_to_pyspice("test")

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    analysis = simulator.operating_point()
    print('\n')
    for node in (analysis[voltage.p.neighbor().deep_id()], analysis[top_resistor.p.neighbor().deep_id()]):
        print('Node {}: {} V'.format(str(node), float(node)))
