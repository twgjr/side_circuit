import unittest

from elements import Voltage, Resistor
from system import *


class TestSystem(unittest.TestCase):
    def test_merge_circuit_nodes(self):
        system = System()
        assert system.num_edges() == 0
        assert system.num_nodes() == 2  # grund and top level subsystem
        cn1 = system.subsystem.add_circuit_node()
        cn2 = system.subsystem.add_circuit_node()
        try:
            system.subsystem.add_wire(cn1, cn2)
            assert False
        except ValueError:
            pass

    # def test_split_wire(self):
    #     system = System()
    #     assert len(system.edges) == 0
    #     assert len(system.__circuit_nodes) == 1  # 1 ground
    #     voltage = Voltage(system).DC(10)
    #     resistor = Resistor(system, 1)

    #     wire1 = Wire(system, voltage.p, resistor.p)
    #     wire2 = Wire(system, voltage.n, resistor.n)
    #     assert len(system.edges) == 4
    #     assert wire1 not in system.edges
    #     assert wire2 not in system.edges
    #     assert voltage.p.edge() in system.edges
    #     assert voltage.n.edge() in system.edges
    #     assert resistor.p.edge() in system.edges
    #     assert resistor.n.edge() in system.edges
    #     assert resistor.p.edge() != voltage.p.edge()
    #     assert resistor.n.edge() != voltage.n.edge()
    #     assert len(system.__circuit_nodes) == 5  # 1 ground + 2 elements + 2 circuit node in between
    #     assert voltage in system.__circuit_nodes
    #     assert resistor in system.__circuit_nodes
    #     assert voltage.p.edge().lo in system.__circuit_nodes
    #     assert voltage.n.edge().lo in system.__circuit_nodes
    #     assert system.ground in system.__circuit_nodes
    #     assert isinstance(voltage.p.edge().lo, CircuitNode)
    #     assert isinstance(voltage.n.edge().lo, CircuitNode)
    #     assert voltage.p.edge().lo == resistor.p.edge().hi
    #     assert voltage.n.edge().lo == resistor.n.edge().hi




    # def test_check_complete(self):
    #     system = System()
    #     ss2 = System(system, [Port(name="p"), Port(name="n")])
    #     Wire(system, hi=system.ground, lo=ss2["p"])
    #     Wire(system, hi=system.ground, lo=ss2["n"])
    #     cn3 = CircuitNode(ss2)
    #     voltage = Voltage(ss2).DC(10)
    #     Wire(ss2, cn3, voltage.p)
    #     Wire(ss2, cn3, voltage.n)
    #     system.check_complete()
