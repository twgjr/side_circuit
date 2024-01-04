from system import Element, Terminal


class TwoTerminal(Element):
    def __init__(self):
        super().__init__([Terminal(self, "p"), Terminal(self, "n")])

    @property
    def p(self) -> Terminal:
        return super().terminals[0]

    @property
    def n(self) -> Terminal:
        return super().terminals[1]


class DC:
    def __init__(self, value: float) -> None:
        self.value = value

    def __repr__(self) -> str:
        return "DC"


class Pulse:
    def __init__(
        self, initial_value: float, pulsed_value: float, freq: float, duty: float
    ) -> None:
        self.initial_value = initial_value
        self.pulsed_value = pulsed_value
        self.freq = freq
        self.duty = duty

    def __repr__(self) -> str:
        return "Pulse"


class Source(TwoTerminal):
    """Base source"""

    def __init__(self):
        super().__init__()
        self.__config = None

    def DC(self, value: float):
        self.__config = DC(value)
        return self

    def Pulse(
        self, initial_value: float, pulsed_value: float, freq: float, duty: float
    ):
        self.__config = Pulse(initial_value, pulsed_value, freq, duty)
        return self

    @property
    def config(self):
        if self.__config is None:
            raise ValueError("Source not configured")
        return self.__config


class Voltage(Source):
    """Base voltage source"""

    def __init__(self) -> None:
        super().__init__()


class Sink(TwoTerminal):
    """Base sink"""

    def __init__(self, value) -> None:
        super().__init__()
        self.value = value


class Resistor(Sink):
    def __init__(self, value: float) -> None:
        super().__init__(value)


class Capacitor(Sink):
    def __init__(self, value: float) -> None:
        super().__init__(value)


class Inductor(Sink):
    def __init__(self, value: float) -> None:
        super().__init__(value)
