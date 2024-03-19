from abc import ABC, abstractmethod


class CommonObject(ABC):
    def __init__(self, idx: str) -> None:
        self.__idx = idx
        self.__parent: CommonObject | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.idx})"

    def __str__(self) -> str:
        return self.idx

    @property
    def idx(self) -> str:
        return self.__idx

    @idx.setter
    def idx(self, idx: str) -> None:
        self.__idx = idx

    @property
    def parent(self) -> "CommonObject | None":
        return self.__parent

    @parent.setter
    def parent(self, parent: "CommonObject | None") -> None:
        self.__parent = parent

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    def cls(self) -> str:
        return self.__class__.__name__


class Junction(CommonObject):
    def __init__(self, idx: str) -> None:
        super().__init__(idx)
        self.__wires: list[Wire] = []

    def add_wire(self, wire: "Wire") -> None:
        self.__wires.append(wire)

    def remove_wire(self, wire: "Wire") -> None:
        self.__wires.remove(wire)

    def num_wires(self) -> int:
        return len(self.__wires)

    def set_wire_at(self, idx: int, wire: "Wire") -> None:
        self.__wires[idx] = wire

    @abstractmethod
    def copy(self) -> "Junction":
        raise NotImplementedError

    def __neighbor(self, wire: "Wire") -> "Junction":
        if wire.hi == self:
            return wire.lo
        else:
            return wire.hi

    def neighbors(self) -> list["Junction"]:
        neighbors = []
        for wire in self.__wires:
            neighbor = self.__neighbor(wire)
            neighbors.append(neighbor)
        return neighbors

    def neighbors_in(self, system) -> list["Junction"]:
        neighbors = []
        from system import System

        for wire in self.__wires:
            neighbor = self.__neighbor(wire)
            if neighbor:
                neighbors.append(neighbor)
        return neighbors



class Interface(Junction):
    """Up to one wire per interface"""

    def __init__(self, idx: str) -> None:
        super().__init__(idx)

    def copy(self) -> "Interface":
        copy = Interface(self.idx)
        return copy

    def add_wire(self, wire: "Wire") -> None:
        if self.num_wires() > 0:
            self.set_wire_at(0, wire)
        else:
            super().add_wire(wire)

    def nodes(self) -> list["Node"]:
        neighbors = self.neighbors()
        nodes = []
        for neighbor in neighbors:
            if isinstance(neighbor, Node):
                nodes.append(neighbor)

        if len(nodes) > 1:
            raise ValueError(f"interface {self} connected to multiple nodes")

        return nodes


class Node(Junction):
    """Unlimited number of wires per node"""

    def __init__(self, idx: str) -> None:
        super().__init__(idx)

    def copy(self) -> "Node":
        copy = Node(self.idx)
        return copy


class Wire(CommonObject):
    def __init__(
        self,
        hi: Junction | None = None,
        lo: Junction | None = None,
    ) -> None:
        if hi is None and lo is None:
            super().__init__("")
        elif isinstance(hi, Junction) and isinstance(lo, Junction):
            super().__init__(self.make_idx(hi, lo))
        else:
            raise ValueError(f"invalid parameters {hi}, {lo}")
        self.__hi = hi
        self.__lo = lo

    @staticmethod
    def make_idx(hi: Junction, lo: Junction) -> str:
        return f"{repr(hi)} -> {repr(lo)}"

    @property
    def hi(self) -> Junction:
        if self.__hi is None:
            raise ValueError(f"hi not connected")
        return self.__hi

    @property
    def lo(self) -> Junction:
        if self.__lo is None:
            raise ValueError(f"lo not connected")
        return self.__lo

    def copy(self) -> "Wire":
        copy = Wire(self.hi.copy(), self.lo.copy())
        copy.idx = self.idx
        return copy
