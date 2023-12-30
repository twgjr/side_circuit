from enum import Enum

from modules.system import Element, Terminal


class Kind(Enum):
    V = "voltage source"
    R = "resistor"
    C = "capacitor"
    L = "inductor"


class TwoTerminal(Element):
    def __init__(self, system, kind: Kind) -> None:
        slots = [Terminal(self, "p"), Terminal(self, "n")]
        super().__init__(system, slots, kind)

    @property
    def p(self) -> Terminal:
        return self.slots[0]

    @property
    def n(self) -> Terminal:
        return self.slots[1]


class FourTerminal(TwoTerminal):
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

class DC:
    def __init__(self, value: float) -> None:
        self.value = value

    def __repr__(self) -> str:
        return "DC"

class Pulse:
    def __init__(self, initial_value: float, pulsed_value: float, freq: float, duty: float) -> None:
        self.initial_value = initial_value
        self.pulsed_value = pulsed_value
        self.freq = freq
        self.duty = duty

    def __repr__(self) -> str:
        return "Pulse"

class Source(TwoTerminal):
    """Base source"""

    def __init__(self, system, kind):
        super().__init__(system, kind)
        self.__config = None

    def DC(self, value: float):
        self.__config = DC(value)
        return self

    def Pulse(self, initial_value: float, pulsed_value: float, freq: float, duty: float):
        self.__config = Pulse(initial_value, pulsed_value, freq, duty)
        return self

    @property
    def config(self):
        if self.__config is None:
            raise ValueError("Source not configured")
        return self.__config

class Voltage(Source):
    """Base voltage source"""

    def __init__(self, system) -> None:
        super().__init__(system, Kind.V)

    def __repr__(self) -> str:
        return f"{self.config}Voltage({self.deep_id()})"


class Sink(TwoTerminal):
    """Base sink"""

    def __init__(self, system, kind, value) -> None:
        super().__init__(system, kind)
        self.value = value

class Resistor(Sink):
    def __init__(self, system, value: float) -> None:
        super().__init__(system, Kind.R, value)

    def __repr__(self) -> str:
        return f"Resistor({self.deep_id()})"
    
class Capacitor(Sink):
    def __init__(self, system, value: float) -> None:
        super().__init__(system, Kind.C, value)

    def __repr__(self) -> str:
        return f"Capacitor({self.deep_id()})"
    
class Inductor(Sink):
    def __init__(self, system, value: float) -> None:
        super().__init__(system, Kind.L, value)

    def __repr__(self) -> str:
        return f"Inductor({self.deep_id()})"
