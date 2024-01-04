import unittest

from elements import Voltage, Resistor
from system import *


class TestSystem(unittest.TestCase):
    def test_merge_circuit_nodes(self):
        system = System()
        self.assertEqual(system.num_nodes(), 2) # grund and top level subsystem
        self.assertEqual(system.root.num_circuit_nodes(), 0)
        cn1 = system.root.add_circuit_node()
        cn2 = system.root.add_circuit_node()
        try:
            system.root.add_wire(cn1, cn2)
            self.assertTrue(False)
        except ValueError:
            pass

    def test_subsystem_tree(self):
        """make a tree that looks like:
                         root
                        /    \
                     sub1    sub2
                            /   
                         sub21  """
        system = System()
        self.assertEqual(system.num_edges(), 0)
        self.assertEqual(system.num_nodes(), 2)
        self.assertEqual(system.root.num_subsystems(), 0)
        sub1 = system.root.add_subsystem()
        sub2 = system.root.add_subsystem()
        sub21 = sub2.add_subsystem()
        self.assertEqual(system.num_edges(), 0)
        self.assertEqual(system.num_nodes(), 5)
        self.assertEqual(system.root.num_subsystems(), 2)
        self.assertEqual(sub1.num_subsystems(), 0)
        self.assertEqual(sub2.num_subsystems(), 1)
        self.assertEqual(sub21.num_subsystems(), 0)


    def test_make_system_with_elements(self):
        """make a system that looks like:
            system
               |   
               root-----------------
                    |              |
                voltage -> cn -> resistor
                    |______gnd_____|
        """        
        system = System()
        voltage = Voltage().DC(10)
        system.root.add_element(voltage)
        resistor = Resistor(10)
        system.root.add_element(resistor)
        try:
            system.root.add_wire(resistor.p, voltage.p)
            self.assertTrue(False)
        except ValueError:
            pass
        self.assertEqual(system.num_edges(), 0)
        cn = system.root.add_circuit_node()
        try:
            system.root.add_wire(cn, system.gnd)
            self.assertTrue(False)
        except ValueError:
            pass
        system.root.add_wire(resistor.p, cn)
        system.root.add_wire(voltage.p, cn)
        system.root.add_wire(resistor.n, system.gnd)
        system.root.add_wire(voltage.n, system.gnd)
        self.assertEqual(system.num_edges(), 4)
        self.assertEqual(system.num_nodes(), 5)
        self.assertEqual(system.root.num_elements(), 2)
        self.assertEqual(system.root.num_circuit_nodes(), 1)
        self.assertEqual(system.root.num_ports(), 0)
        self.assertEqual(system.root.num_wires(), 4)
        self.assertEqual(system.root.num_subsystems(), 0)

