from enum import Enum

from app.system.system import Element, Terminal


class Kind(Enum):
    V = "voltage source"
    I = "current source"
    R = "resistor"
    L = "inductor"
    C = "capacitor"


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

    def __repr__(self) -> str:
        return f"Voltage({self.deep_id()})"


class Current(TwoTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.I)

    def __repr__(self) -> str:
        return f"Current({self.deep_id()})"


class Resistor(TwoTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.R)

    def __repr__(self) -> str:
        return f"Resistor({self.deep_id()})"


class Inductor(TwoTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.L)

    def __repr__(self) -> str:
        return f"Inductor({self.deep_id()})"


class Capacitor(TwoTerminalElement):
    def __init__(self, system) -> None:
        super().__init__(system, Kind.C)

    def __repr__(self) -> str:
        return f"Capacitor({self.deep_id()})"
