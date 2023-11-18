from enum import Enum
from app.graph import Node


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


class Terminal(Node):
    """A terminal is a single node with no ports.  Acts as the interface to
    elements.  Terminals may connect to other terminals, circuit nodes, or
    ports"""

    def __init__(self, name: str) -> None:
        super().__init__(max_edges=1)
        self.name = name


class Element(Node):
    """An intrinsic circuit element viewed as a single node with ports.
    No internal sub-systems, elements, wires, or circuit nodes."""

    def __init__(self, kind: Kind) -> None:
        super().__init__(max_edges=0)
        assert isinstance(kind, Kind)
        self.kind = kind
        self.terminals: dict[str, Terminal] = {}


class TwoTerminalElement(Element):
    def __init__(self, kind: Kind) -> None:
        super().__init__(kind)
        self.terminals["p"] = Terminal("p")
        self.terminals["n"] = Terminal("n")


class FourTerminalElement(Element):
    def __init__(self, kind: Kind) -> None:
        super().__init__(kind)
        self.terminals["p"] = Terminal("p")
        self.terminals["n"] = Terminal("n")
        self.terminals["cp"] = Terminal("cp")
        self.terminals["cn"] = Terminal("cn")


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
