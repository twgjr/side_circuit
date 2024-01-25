import unittest

from elements import V, R, DC
from system import *


class TestSystem(unittest.TestCase):
    def test_wiring(self):
        # create a basic system
        s = System("wiring", ["in", "out"], [V("1", DC(10)), R("1", 1e3)])
        # make wires
        s.link(s["in"], s.V["1"]["p"])
        s.link(s.V["1"]["n"], s.R["1"]["p"])
        s.link(s.R["1"]["n"], s["out"])

        # test number of components
        self.assertEqual(s.num_subsystems(), 0)
        self.assertEqual(s.num_v(), 1)
        self.assertEqual(s.num_r(), 1)
        self.assertEqual(s.num_interfaces(), 2)
        self.assertEqual(s.num_wires(), 3)

    def test_resistor_subsystem(self):
        # create a resistor series subsystem template
        rr = System(
            "series resistors",
            ["in", "out", "ref"],
            [
                R("1", 1e3),
                R("2", 1e3),
            ],
        )

        # make wires in the subsystem
        rr.link(rr["in"], rr.R["1"]["p"])
        rr.link(rr.R["1"]["n"], rr.R["2"]["p"])
        rr.link(rr.R["2"]["n"], rr["out"])

        # make two copies of the subsystem
        rr1 = rr("resistors1")

        # test that they are unique
        self.assertNotEqual(id(rr), id(rr1))

        # test that r1 is unique but has same attributes as rr
        self.assertNotEqual(id(rr.R["1"]), id(rr1.R["1"]))
        self.assertEqual(rr.R["1"].value, rr1.R["1"].value)

    def test_voltage_divider(self):
        # create a resistor series subsystem template
        rr = System(
            "series resistors",
            ["in", "out", "ref"],
            [
                R("1", 1e3),
                R("2", 1e3),
            ],
        )

        # make wires in the subsystem
        rr.link(rr["in"], rr.R["1"]["p"])
        rr.link(rr.R["1"]["n"], rr.R["2"]["p"])
        rr.link(rr.R["2"]["n"], rr["out"])

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
        vd.link(vd.V["1"]["p"], vd.X["bank1"]["in"]) 
        vd.link(vd.X["bank1"]["out"], vd.X["bank2"]["in"])
        vd.link(vd.X["bank2"]["out"], vd.V["1"]["n"])

        # test number of components in divider system
        self.assertEqual(vd.num_subsystems(), 2)
        self.assertEqual(vd.num_v(), 1)
        self.assertEqual(vd.num_interfaces(), 0)
        self.assertEqual(vd.num_wires(), 3)

        # test number of components in bank1
        self.assertEqual(vd.X["bank1"].num_subsystems(), 0)
        self.assertEqual(vd.X["bank1"].num_r(), 2)
        self.assertEqual(vd.X["bank1"].num_interfaces(), 3)
        self.assertEqual(vd.X["bank1"].num_wires(), 3)

        # test number of components in bank2
        self.assertEqual(vd.X["bank2"].num_subsystems(), 0)
        self.assertEqual(vd.X["bank2"].num_r(), 2)
        self.assertEqual(vd.X["bank2"].num_interfaces(), 3)
        self.assertEqual(vd.X["bank2"].num_wires(), 3)
