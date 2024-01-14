from common import SystemObject, Wire, Interface
from elements import Element, ElementDict


class System(SystemObject):
    """create a system by defining wires between nodes
    and elements and subsystems in the constructor.
    """

    def __init__(
        self,
        idx: str = "",
        interfaces: list[str] = [],
        objects: list = [],
    ) -> None:
        super().__init__(idx)
        self.__interfaces: InterfaceDict = None  # type: ignore
        self.__elementdict: ElementDict = None  # type: ignore
        self.__subsystemdict: SystemDict = None  # type: ignore
        self.__wires: dict[str, Wire] = {}
        self.__process_init(interfaces, objects)

    def __process_init(
        self, interfaces: list[str], objects: list["SystemObject"]
    ) -> None:
        elements = []
        subsystems = []
        for item in objects:
            if isinstance(item, System):
                subsystems.append(item)
                item.parent = self
            elif isinstance(item, Element):
                elements.append(item)
                item.parent = self
            else:
                raise ValueError(f"invalid object type {type(item)}")

        self.__interfaces = InterfaceDict(self, interfaces)
        self.__elementdict = ElementDict(self, elements)
        self.__subsystemdict = SystemDict(self, subsystems)

    def __contains__(self, item) -> bool:
        if isinstance(item, Interface):
            return item in self.__interfaces

        if isinstance(item, Element):
            return item in self.__elementdict

        if isinstance(item, System):
            return item in self.__subsystemdict

        if isinstance(item, Wire):
            return item in self.__wires

        return False

    def __getitem__(self, key: str) -> "Interface":
        return self.__interfaces[key]

    def __call__(self, idx: str) -> "System":
        system_copy = self.copy()
        system_copy.idx = idx
        return system_copy

    def copy(self) -> "System":
        sys = System(
            self.idx,
            [str(interface) for interface in self.__interfaces.interfaces],
            [element.copy() for element in self.__elementdict.elements]
            + [subsystem.copy() for subsystem in self.__subsystemdict.subsystems],
        )
        sys.copy_wires(self.__wires)
        return sys

    def __get_wire_arg(self, old: Interface) -> Interface:
        if isinstance(old, Interface):
            return self.__interfaces[str(old)]
        elif isinstance(old, Interface):
            elements = self.__elementdict.elements
            for element in elements:
                if old in element:
                    return element[str(old)]

        raise ValueError(f"{old} not in system")

    def copy_wires(self, wires: dict[str, Wire]) -> None:
        for key, wire in wires.items():
            hi = self.__get_wire_arg(wire.hi)
            lo = self.__get_wire_arg(wire.lo)

            self.__wires[key] = hi >> lo

    @property
    def s(self) -> "SystemDict":
        return self.__subsystemdict

    @property
    def e(self) -> "ElementDict":
        return self.__elementdict

    def add_wire(self, wire: Wire) -> None:
        self.__wires[str(wire)] = wire

    def num_subsystems(self) -> int:
        return len(self.__subsystemdict)

    def num_elements(self) -> int:
        return len(self.__elementdict)

    def num_wires(self) -> int:
        return len(self.__wires)

    def num_interfaces(self) -> int:
        return len(self.__interfaces)


class SystemDict:
    def __init__(self, system: System, subsystems: list["System"]) -> None:
        self.__subsystems: dict[str, "System"] = {}
        for subsystem in subsystems:
            self.__subsystems[str(subsystem)] = subsystem
            subsystem.parent = system

    def __getitem__(self, key: str) -> "System":
        return self.__subsystems[key]

    def __contains__(self, item) -> bool:
        return item in self.__subsystems

    def __iter__(self):
        return iter(self.__subsystems.keys())

    def __len__(self):
        return len(self.__subsystems)

    @property
    def subsystems(self) -> list["System"]:
        return list(self.__subsystems.values())


class InterfaceDict:
    def __init__(self, system: System, interfaces: list[str]) -> None:
        self.__interfaces: dict[str, Interface] = {}
        for interface in interfaces:
            self.__interfaces[interface] = Interface(interface)
            self.__interfaces[interface].parent = system

    def __getitem__(self, key: str) -> Interface:
        return self.__interfaces[key]

    def __contains__(self, item) -> bool:
        if isinstance(item, Interface):
            return str(item) in self.__interfaces

        if isinstance(item, str):
            return item in self.__interfaces

        return False

    def __len__(self) -> int:
        return len(self.__interfaces)

    @property
    def interfaces(self) -> list[Interface]:
        return list(self.__interfaces.values())
