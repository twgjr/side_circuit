"""unit tests for system.py"""
import unittest
import random
from app.system import (
    System,
    Port,
    Node,
    Wire,
    CircuitNode,
)

class TestSystem(unittest.TestCase):
    """class for testing system.py"""

    def test_system(self):
        """test system"""
        system = System()
        self.assertEqual(len(system.sub_systems), 0)
        self.assertEqual(len(system.elements), 0)
        self.assertEqual(len(system.ports), 0)
        self.assertEqual(len(system.circuit_nodes), 0)
        self.assertEqual(len(system.wires), 0)

    def test_add_sub_system(self):
        """test add_sub_system"""
        system = System()
        system.add_sub_system(System())
        self.assertEqual(len(system.sub_systems), 1)

    def test_add_element(self):
        """test add_element"""
        system = System()
        system.add_element(Node())
        self.assertEqual(len(system.elements), 1)

    def test_add_port(self):
        """test add_port"""
        system = System()
        system.add_port("test")
        self.assertEqual(len(system.ports), 1)

    def test_add_circuit_node(self):
        """test add_circuit_node"""
        system = System()
        system.add_circuit_node()
        self.assertEqual(len(system.circuit_nodes), 1)

    def test_add_wire(self):
        """test add_wire"""
        system = System()
        system.add_wire(Node(), Node())
        self.assertEqual(len(system.wires), 1)

    def test_split_wire_to(self):
        """test split_wire_to"""
        system = System()
        port1 = system.add_port("test1")
        port2 = system.add_port("test2")
        system.add_wire(port1, port2)
        circuit_node = system.split_wire_to(port1)
        self.assertEqual(len(system.circuit_nodes), 1)
        self.assertEqual(len(system.wires), 2)

    def test_connect(self):
        """test connect"""
        system = System()
        port1 = system.add_port("test1")
        port2 = system.add_port("test2")
        system.connect(port1, port2)
        self.assertEqual(len(system.circuit_nodes), 0)
        self.assertEqual(len(system.wires), 1)
        self.assertEqual(port1.wire.p, port1)
        self.assertEqual(port1.wire.n, port2)
        self.assertEqual(port2.wire, port1.wire)