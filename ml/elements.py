from enum import Enum
from graph import Node
from system import Port

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
    """An intrinsic circuit element viewed as a single node with ports.
    No internal sub-systems, elements, wires, or circuit nodes."""

    def __init__(self, kind: Kind) -> None:
        super().__init__()
        assert isinstance(kind, Kind)
        self.kind = kind
        self.ports: dict[str, Port] = {}
        if (
            kind == Kind.R
            or kind == Kind.L
            or kind == Kind.C
            or kind == Kind.V
            or kind == Kind.I
        ):
            self.ports["p"] = Port(-2, "p")
            self.ports["n"] = Port(-1, "n")
        elif (
            kind == Kind.VCSW
            or kind == Kind.CCSW
            or kind == Kind.VCVS
            or kind == Kind.CCVS
            or kind == Kind.CCCS
            or kind == Kind.VCCC
        ):
            self.ports["p"] = Port(-2, "p")
            self.ports["n"] = Port(-1, "n")
            self.ports["cp"] = Port(0, "cp")
            self.ports["cn"] = Port(1, "cn")

class Voltage(Element):
    def __init__(self) -> None:
        super().__init__(Kind.V)

class Current(Element):
    def __init__(self) -> None:
        super().__init__(Kind.I)

class Resistor(Element):
    def __init__(self) -> None:
        super().__init__(Kind.R)

class Inductor(Element):
    def __init__(self) -> None:
        super().__init__(Kind.L)

class Capacitor(Element):
    def __init__(self) -> None:
        super().__init__(Kind.C)

class VoltageControlledSwitch(Element):
    def __init__(self) -> None:
        super().__init__(Kind.VCSW)

class CurrentControlledSwitch(Element):
    def __init__(self) -> None:
        super().__init__(Kind.CCSW)

class VoltageControlledVoltageSource(Element):
    def __init__(self) -> None:
        super().__init__(Kind.VCVS)

class CurrentControlledVoltageSource(Element):
    def __init__(self) -> None:
        super().__init__(Kind.CCVS)

class CurrentControlledCurrentSource(Element):
    def __init__(self) -> None:
        super().__init__(Kind.CCCS)
        
class VoltageControlledCurrentSource(Element):
    def __init__(self) -> None:
        super().__init__(Kind.VCCC)