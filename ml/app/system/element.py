from enum import Enum

from app.graph.graph import Node, Slot
from app.system.interface import Terminal


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


class Element(Node):
    """An intrinsic circuit element viewed as a single node with terminals.
    No internal sub-systems, elements, wires, or circuit nodes."""

    def __init__(self, slots: list[Slot], kind: Kind) -> None:
        super().__init__(slots)
        assert isinstance(kind, Kind)
        self.kind = kind

    # def get_terminal(self, name: str) -> Terminal:
    #     interface = self.get_interface(name)
    #     if interface is None:
    #         raise Exception(f"Element has no terminal named {name}")
    #     if isinstance(interface, Terminal):
    #         return interface
    #     else:
    #         raise TypeError(f"expected Terminal, got {type(interface)}")


class TwoTerminalElement(Element):
    def __init__(self, kind: Kind) -> None:
        slots = [Terminal("p"), Terminal("n")]
        super().__init__(slots, kind)

    @property
    def p(self) -> Terminal:
        return self.slots[0]

    @property
    def n(self) -> Terminal:
        return self.slots[1]


class FourTerminalElement(TwoTerminalElement):
    def __init__(self, kind: Kind) -> None:
        super().__init__(kind)
        self.slots.append(Terminal("cp"))
        self.slots.append(Terminal("cn"))

    @property
    def cp(self) -> Terminal:
        return self.slots[2]

    @property
    def cn(self) -> Terminal:
        return self.slots[3]


class Voltage(TwoTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.V)


class Current(TwoTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.I)


class Resistor(TwoTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.R)


class Inductor(TwoTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.L)


class Capacitor(TwoTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.C)


class VoltageControlledSwitch(FourTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.VCSW)


class CurrentControlledSwitch(FourTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.CCSW)


class VoltageControlledVoltageSource(FourTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.VCVS)


class CurrentControlledVoltageSource(FourTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.CCVS)


class CurrentControlledCurrentSource(FourTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.CCCS)


class VoltageControlledCurrentSource(FourTerminalElement):
    def __init__(self) -> None:
        super().__init__(Kind.VCCC)
