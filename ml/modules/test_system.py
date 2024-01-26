import unittest

from elements import V, R, DC
from system import *


class TestSystem(unittest.TestCase):
    def test_wiring(self):
        # create a basic system
        s = System("wiring", [], [V("1", DC(10)), R("1", 1e3)])
        # make wires
        s.link(s.V["1"]["n"], s.R["1"]["n"])
        s.link(s.V["1"]["p"], s.R["1"]["p"])

        # test number of components
        self.assertEqual(s.num_subsystems(), 0)
        self.assertEqual(s.num_v(), 1)
        self.assertEqual(s.num_r(), 1)
        self.assertEqual(s.num_interfaces(), 0)
        self.assertEqual(s.num_wires(), 2)
        self.assertEqual(s.num_nodes(), 0)

    def test_adding_node(self):
        # create a basic system
        s = System("wiring", ["in", "out"], [V("1", DC(10)), R("1", 1e3)])
        # make wires
        s.link(s.V["1"]["n"], s.R["1"]["n"])
        s.link(s.V["1"]["p"], s.R["1"]["p"])

        # add a node by adding an interface
        s.link(s.ii["in"], s.V["1"]["p"])

        # test number of components after adding a node
        self.assertEqual(s.num_subsystems(), 0)
        self.assertEqual(s.num_v(), 1)
        self.assertEqual(s.num_r(), 1)
        self.assertEqual(s.num_interfaces(), 2)
        self.assertEqual(s.num_wires(), 4)
        self.assertEqual(s.num_nodes(), 1)

        # add a second node by adding an interface
        s.link(s.R["1"]["n"], s.ii["out"])

        # test number of components after adding second node
        self.assertEqual(s.num_subsystems(), 0)
        self.assertEqual(s.num_v(), 1)
        self.assertEqual(s.num_r(), 1)
        self.assertEqual(s.num_interfaces(), 2)
        self.assertEqual(s.num_wires(), 6)
        self.assertEqual(s.num_nodes(), 2)

    def test_using_subsystem_template(self):
        # create a resistor series subsystem template
        rr = System(
            "series resistors",
            ["in", "out", "ref"],
            [
                R("1", 1e3),
                R("2", 1e3),
            ],
        )

        # start adding wires in the subsystem
        rr.link(rr.ii["in"], rr.R["1"]["p"])
        rr.link(rr.R["1"]["n"], rr.R["2"]["p"])
        rr.link(rr.R["2"]["p"], rr.ii["out"])
        rr.link(rr.R["2"]["n"], rr.ii["ref"])

        # make two copies of the subsystem
        rr1 = rr("resistors1")

        # test that they are unique
        self.assertNotEqual(id(rr), id(rr1))

        # test that r1 is unique but has same attributes as rr
        self.assertNotEqual(id(rr.R["1"]), id(rr1.R["1"]))
        self.assertEqual(rr.R["1"].value, rr1.R["1"].value)

        # test that r2 is unique but has same attributes as rr
        self.assertNotEqual(id(rr.R["2"]), id(rr1.R["2"]))
        self.assertEqual(rr.R["2"].value, rr1.R["2"].value)

        # test that the interfaces are unique but have same attributes as rr
        self.assertNotEqual(id(rr.ii["in"]), id(rr1.ii["in"]))
        self.assertEqual(str(rr.ii["in"]), str(rr1.ii["in"]))

        # test that the wires are unique but have same attributes as rr
        wires_copied = 0
        for wire in rr.Wires:
            for wire1 in rr1.Wires:
                if str(wire) == str(wire1):
                    self.assertNotEqual(id(wire), id(wire1))
                    wires_copied += 1

        self.assertEqual(wires_copied, 5)

    def test_wire_subsystem_to_subsystem(self):
        # create a resistor series subsystem template
        rr = System(
            "series resistors",
            ["in", "out"],
            [
                R("1", 1e3),
                R("2", 1e3),
            ],
        )

        # make wires in the subsystem
        rr.link(rr.ii["in"], rr.R["1"]["p"])
        rr.link(rr.R["1"]["n"], rr.R["2"]["p"])
        rr.link(rr.R["2"]["n"], rr.ii["out"])

        # voltage divider system definition
        vd = System(
            "voltage divider",
            [],
            [
                V("1", DC(10)),
                rr("bank1"),
                rr("bank2"),
            ],
        )

        # voltage divider top system wiring
        vd.link(vd.V["1"]["p"], vd.X["bank1"].ei["in"])
        vd.link(vd.X["bank1"].ei["out"], vd.X["bank2"].ei["in"])
        vd.link(vd.X["bank2"].ei["out"], vd.V["1"]["n"])

        # test number of components in divider system
        self.assertEqual(vd.num_subsystems(), 2)
        self.assertEqual(vd.num_v(), 1)
        self.assertEqual(vd.num_interfaces(), 0)
        self.assertEqual(vd.num_wires(), 3)
        self.assertEqual(vd.num_nodes(), 0)

        # test number of components in bank1
        self.assertEqual(vd.X["bank1"].num_subsystems(), 0)
        self.assertEqual(vd.X["bank1"].num_r(), 2)
        self.assertEqual(vd.X["bank1"].num_interfaces(), 2)
        self.assertEqual(vd.X["bank1"].num_wires(), 3)
        self.assertEqual(vd.X["bank1"].num_nodes(), 0)

        # test number of components in bank2
        self.assertEqual(vd.X["bank2"].num_subsystems(), 0)
        self.assertEqual(vd.X["bank2"].num_r(), 2)
        self.assertEqual(vd.X["bank2"].num_interfaces(), 2)
        self.assertEqual(vd.X["bank2"].num_wires(), 3)
        self.assertEqual(vd.X["bank2"].num_nodes(), 0)