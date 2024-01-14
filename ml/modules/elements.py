from common import SystemObject, Interface, Wire


class Element(SystemObject):
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

    def add_wire(self, wire: "Wire") -> None:
        parent = self.parent
        from system import System
        if isinstance(parent, System):
            parent.add_wire(wire)


class ElementDict:
    def __init__(self, parent, elements: list["Element"]) -> None:
        self.__v: dict[str, "V"] = {}
        self.__r: dict[str, "R"] = {}
        for element in elements:
            from system import System

            if not isinstance(parent, System):
                raise ValueError(f"invalid parent type {type(parent)}")
            element.parent = parent
            if isinstance(element, V):
                self.__v[str(element)] = element
            elif isinstance(element, R):
                self.__r[str(element)] = element

    def __contains__(self, item) -> bool:
        if isinstance(item, V):
            return item in self.__v
        if isinstance(item, R):
            return item in self.__r
        return False

    def __len__(self) -> int:
        return len(self.__v) + len(self.__r)

    @property
    def v(self) -> dict[str, "V"]:
        return self.__v

    @property
    def r(self) -> dict[str, "R"]:
        return self.__r

    @property
    def elements(self) -> list["V | R"]:
        list_v = list(self.__v.values())
        list_r = list(self.__r.values())
        return list_v + list_r


class TwoTerminal(Element):
    def __init__(self, id: str):
        super().__init__(id, [Interface("p"), Interface("n")])


class SourceConfig:
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
        raise NotImplementedError


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
