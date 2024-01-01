from modules.elements import Voltage, Resistor
from modules.system import *


def test_merge_circuit_nodes():
    system = System()
    assert len(system.edges) == 0
    assert len(system.nodes) == 1  # 1 ground
    cn1 = CircuitNode(system)
    cn2 = CircuitNode(system)
    wire = Wire(system, cn1, cn2)
    assert len(system.edges) == 0  # wire not created when merging
    assert len(system.nodes) == 2  # 1 ground + 1 merged circuit node
    assert wire not in system.edges
    assert cn1 in system.nodes
    assert cn2 not in system.nodes


def test_split_wire():
    system = System()
    assert len(system.edges) == 0
    assert len(system.nodes) == 1  # 1 ground
    voltage = Voltage(system).DC(10)
    resistor = Resistor(system, 1)

    wire1 = Wire(system, voltage.p, resistor.p)
    wire2 = Wire(system, voltage.n, resistor.n)
    assert len(system.edges) == 4
    assert wire1 not in system.edges
    assert wire2 not in system.edges
    assert voltage.p.edge() in system.edges
    assert voltage.n.edge() in system.edges
    assert resistor.p.edge() in system.edges
    assert resistor.n.edge() in system.edges
    assert resistor.p.edge() != voltage.p.edge()
    assert resistor.n.edge() != voltage.n.edge()
    assert len(system.nodes) == 5  # 1 ground + 2 elements + 2 circuit node in between
    assert voltage in system.nodes
    assert resistor in system.nodes
    assert voltage.p.edge().lo in system.nodes
    assert voltage.n.edge().lo in system.nodes
    assert system.ground in system.nodes
    assert isinstance(voltage.p.edge().lo, CircuitNode)
    assert isinstance(voltage.n.edge().lo, CircuitNode)
    assert voltage.p.edge().lo == resistor.p.edge().hi
    assert voltage.n.edge().lo == resistor.n.edge().hi




def test_check_complete():
    system = System()
    ss2 = System(system, [Port(name="p"), Port(name="n")])
    Wire(system, hi=system.ground, lo=ss2["p"])
    Wire(system, hi=system.ground, lo=ss2["n"])
    cn3 = CircuitNode(ss2)
    voltage = Voltage(ss2).DC(10)
    Wire(ss2, cn3, voltage.p)
    Wire(ss2, cn3, voltage.n)
    system.check_complete()
