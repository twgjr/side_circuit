"""unittest units tests for the element module"""
import unittest
from app.element import (
    Kind,
    Terminal,
    Element,
    TwoTerminalElement,
    FourTerminalElement,
    Voltage,
    Current,
    Resistor,
    Inductor,
    Capacitor,
    VoltageControlledSwitch,
    CurrentControlledSwitch,
    VoltageControlledVoltageSource,
    VoltageControlledCurrentSource,
    CurrentControlledVoltageSource,
    CurrentControlledCurrentSource,
)


class TestElement(unittest.TestCase):
    """class for testing element module"""

    def test_two_terminals(self):
        """test two terminal elements"""
        tte_list: list[TwoTerminalElement] = [
            Voltage(),
            Current(),
            Resistor(),
            Inductor(),
            Capacitor(),
        ]

        for tte in tte_list:
            assert len(tte.terminals) == 2
            assert tte.terminals["p"].name == "p"
            assert tte.terminals["n"].name == "n"

    def test_four_terminals(self):
        """test four terminal elements"""
        fte_list: list[FourTerminalElement] = [
            VoltageControlledSwitch(),
            CurrentControlledSwitch(),
            VoltageControlledVoltageSource(),
            VoltageControlledCurrentSource(),
            CurrentControlledVoltageSource(),
            CurrentControlledCurrentSource(),
        ]

        for fte in fte_list:
            assert len(fte.terminals) == 4
            assert fte.terminals["p"].name == "p"
            assert fte.terminals["n"].name == "n"
            assert fte.terminals["cp"].name == "cp"
            assert fte.terminals["cn"].name == "cn"

if __name__ == '__main__':
    unittest.main()