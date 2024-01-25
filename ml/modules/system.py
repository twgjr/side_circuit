from common import CommonObject, Wire, Interface
from elements import *


class System(CommonObject):
    """create a system by defining wires between nodes
    and elements and subsystems in the constructor.
    """

    def __init__(
        self,
        idx: str = "",
        interfaces: list[str] = [],
        objects: list[CommonObject] = [],
    ) -> None:
        super().__init__(idx)

        self.__interfaces: dict[str, Interface] = {}
        for interface in interfaces:
            self.__interfaces[interface] = Interface(interface)
            self.__interfaces[interface].parent = self
        self.__v_dict: dict[str, V] = {}
        self.__r_dict: dict[str, R] = {}
        self.__x_dict: dict[str, "System"] = {}
        for obj in objects:
            obj.parent = self
            if isinstance(obj, V):
                self.__v_dict[str(obj)] = obj
            elif isinstance(obj, R):
                self.__r_dict[str(obj)] = obj
            elif isinstance(obj, System):
                self.__x_dict[str(obj)] = obj
            else:
                raise TypeError(f"invalid object type {type(obj)}")
        self.__wires: dict[str, Wire] = {}

    def __contains__(self, item) -> bool:
        if isinstance(item, Interface):
            return item in self.__interfaces

        if isinstance(item, V):
            return item in self.__v_dict
        
        if isinstance(item, R):
            return item in self.__r_dict

        if isinstance(item, System):
            return item in self.__x_dict

        if isinstance(item, Wire):
            return item in self.__wires

        return False

    def __getitem__(self, item) -> "Interface":
        if item in self.__interfaces:
            intx = self.__interfaces[item]
            if isinstance(intx, Interface):
                return intx
            raise TypeError(f"invalid interface {item}")
        raise KeyError(f"invalid interface {item}")

    def __call__(self, idx: str) -> "System":
        system_copy = self.copy()
        system_copy.idx = idx
        return system_copy

    def copy(self) -> "System":
        elements = []
        for v in self.__v_dict.values():
            elements.append(v.copy())
        for r in self.__r_dict.values():
            elements.append(r.copy())
        sys = System(self.idx, list(self.__interfaces.keys()), elements)
        sys.copy_wires(self.__wires)
        return sys

    def __get_matching(self, interface: Interface) -> Interface:
        parent = interface.parent
        if isinstance(parent, System):
            return self[interface.idx]

        if isinstance(parent, V):
            v = self.V[str(parent)]
            if isinstance(v, V):
                return v[str(interface)]
            
        if isinstance(parent, R):
            r = self.R[str(parent)]
            if isinstance(r, R):
                return r[str(interface)]

        raise TypeError(f"invalid interface {interface}")

    def copy_wires(self, wires: dict[str, Wire]) -> None:
        for key, wire in wires.items():
            # get matching interfaces
            hi = self.__get_matching(wire.hi)
            lo = self.__get_matching(wire.lo)
            # create new wire
            new_wire = self.link(hi, lo)
            new_wire.idx = wire.idx
            # add new wire to system
            self.add_wire(new_wire)

    @property
    def X(self) -> dict[str, "System"]:
        return self.__x_dict

    @property
    def V(self) -> dict[str, V]:
        return self.__v_dict
    
    @property
    def R(self) -> dict[str, R]:
        return self.__r_dict

    def add_wire(self, wire: Wire) -> None:
        self.__wires[str(wire)] = wire

    def num_subsystems(self) -> int:
        return len(self.__x_dict)

    def num_v(self) -> int:
        return len(self.__v_dict)
    
    def num_r(self) -> int:
        return len(self.__r_dict)

    def num_wires(self) -> int:
        return len(self.__wires)

    def num_interfaces(self) -> int:
        return len(self.__interfaces)

    def link(self, hi: Interface, lo: Interface) -> Wire:
        wire = Wire(hi, lo)
        hi.add_wire(wire)
        lo.add_wire(wire)
        self.add_wire(wire)
        return wire
