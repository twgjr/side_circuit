from modules.simulator import Simulator
from modules.system import System, Wire
from modules.elements import *

import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()


def test_system_to_pyspice():
    # create a voltage resistor circuit
    system = System()
    voltage = Voltage(system).DC(10)
    resistor = Resistor(system, 100)
    Wire(system, voltage.p, resistor.p)
    Wire(system, voltage.n, system.ground)
    Wire(system, resistor.n, system.ground)
    # convert to PySpice
    simulator = Simulator(system)
    circuit = simulator.system_to_pyspice("test")
    assert len(circuit.elements) == 2


def test_simulator():
    system = System()
    voltage = Voltage(system).DC(10)
    top_resistor = Resistor(system, 9)
    bottom_resistor = Resistor(system, 1)
    Wire(system, voltage.p, top_resistor.p)
    Wire(system, voltage.n, system.ground)
    Wire(system, top_resistor.n, bottom_resistor.p)
    Wire(system, bottom_resistor.n, system.ground)

    # convert to PySpice
    sim = Simulator(system)
    circuit = sim.system_to_pyspice("test")

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    analysis = simulator.operating_point()
    print('\n')
    for node in (analysis[voltage.p.neighbor().deep_id()], analysis[top_resistor.p.neighbor().deep_id()]):
        print('Node {}: {} V'.format(str(node), float(node)))
