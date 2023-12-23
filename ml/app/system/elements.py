from enum import Enum

from app.system.system import Element, Terminal


class Kind(Enum):
    V = "voltage source"
    I = "current source"
    R = "resistor"
    L = "inductor"
    C = "capacitor"
    VCSW = "voltage controlled switch"
    CCSW = "current controlled switch"
    VCVS = "voltage controlled voltage source"
    CCVS = "current controlled voltage source"
    CCCS = "current controlled current source"
    VCCC = "voltage controlled current source"


class TwoTerminalElement(Element):
    def __init__(self, system, kind: Kind) -> None:
        slots = [Terminal(self, "p"), Terminal(self, "n")]
        super().__init__(system, slots, kind)

    @property
    def p(self) -> Terminal:
        return self.slots[0]

    @property
    def n(self) -> Terminal:
        return self.slots[1]


class FourTerminalElement(TwoTerminalElement):
    def __init__(self, system, kind: Kind) -> None:
        super().__init__(system, kind)
        self.slots.append(Terminal(self, "cp"))
        self.slots.append(Terminal(self, "cn"))

    @property
    def cp(self) -> Terminal:
        return self.slots[2]

    @property
    def cn(self) -> Terminal:
        return self.slots[3]


class Voltage(TwoTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.V)


class Current(TwoTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.I)


class Resistor(TwoTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.R)


class Inductor(TwoTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.L)


class Capacitor(TwoTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.C)


class VoltageControlledSwitch(FourTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.VCSW)


class CurrentControlledSwitch(FourTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.CCSW)


class VoltageControlledVoltageSource(FourTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.VCVS)


class CurrentControlledVoltageSource(FourTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.CCVS)


class CurrentControlledCurrentSource(FourTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.CCCS)


class VoltageControlledCurrentSource(FourTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.VCCC)
