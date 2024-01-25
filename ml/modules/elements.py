from abc import abstractmethod
from common import CommonObject, Interface, Wire


class Element(CommonObject):
    """node that only connects to Terminal nodes"""

    def __init__(self, idx: str, interfaces: list[Interface]) -> None:
        super().__init__(idx)
        self.__interfaces: dict[str, Interface] = {}
        for interface in interfaces:
            interface.parent = self
            self.__interfaces[str(interface)] = interface

    def __getitem__(self, key: str) -> Interface:
        return self.__interfaces[key]

    def __contains__(self, item: Interface) -> bool:
        return str(item) in self.__interfaces

    def add_wire(self, wire: Wire) -> None:
        parent = self.parent
        from system import System
        if isinstance(parent, System):
            parent.add_wire(wire)
    
    @abstractmethod
    def copy(self):
        raise NotImplementedError


class TwoTerminal(Element):
    def __init__(self, id: str):
        super().__init__(id, [Interface("p"), Interface("n")])

    @abstractmethod
    def copy(self) -> "TwoTerminal":
        raise NotImplementedError

class SourceConfig:
    @abstractmethod
    def copy(self) -> "SourceConfig":
        raise NotImplementedError


class DC(SourceConfig):
    def __init__(self, value: float) -> None:
        self.value = value

    def copy(self) -> "DC":
        return DC(self.value)


class Pulse(SourceConfig):
    def __init__(
        self, initial_value: float, pulsed_value: float, freq: float, duty: float
    ) -> None:
        self.initial_value = initial_value
        self.pulsed_value = pulsed_value
        self.freq = freq
        self.duty = duty

    def copy(self) -> "Pulse":
        return Pulse(
            self.initial_value,
            self.pulsed_value,
            self.freq,
            self.duty,
        )


class Source(TwoTerminal):
    """Base source"""

    def __init__(self, id: str, config: DC | Pulse) -> None:
        super().__init__(id)
        self.__config = config

    @property
    def config(self):
        if self.__config is None:
            raise ValueError("Source not configured")
        return self.__config

    def copy(self) -> "Source":
        return Source(self.idx, self.config)


class V(Source):
    """Base voltage source"""

    def __init__(self, id: str, config: DC | Pulse) -> None:
        super().__init__(id, config)

    def copy(self) -> "V":
        return V(self.idx, self.config)


class Sink(TwoTerminal):
    """Base sink"""

    def __init__(self, idx: str, value) -> None:
        super().__init__(idx)
        self.value = value

    def copy(self) -> "Sink":
        raise NotImplementedError


class R(Sink):
    def __init__(self, idx: str, value: float) -> None:
        super().__init__(idx, value)

    def copy(self) -> "R":
        return R(self.idx, self.value)
