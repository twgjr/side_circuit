class SystemObject:
    def __init__(self, idx: str) -> None:
        self.__idx = idx
        self.__parent: SystemObject | None = None

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
    def parent(self) -> "SystemObject | None":
        return self.__parent

    @parent.setter
    def parent(self, parent: "SystemObject | None") -> None:
        self.__parent = parent


class Interface(SystemObject):
    def __init__(self, idx: str) -> None:
        super().__init__(idx)
        self.__wires: list["Wire"] = []

    def __rshift__(self, other: "Interface") -> "Wire":
        parent = self.parent

        from system import System
        from elements import Element

        if not isinstance(parent, System) and not isinstance(parent, Element):
            raise ValueError(f"invalid parent type {type(parent)}")

        wire = Wire(self, other)
        wire.parent = parent
        self.add_wire(wire)
        other.add_wire(wire)
        parent.add_wire(wire)
        return wire

    def add_wire(self, wire: "Wire") -> None:
        self.__wires.append(wire)


class Wire(SystemObject):
    def __init__(
        self,
        hi: Interface | None = None,
        lo: Interface | None = None,
    ) -> None:
        if hi is None and lo is None:
            super().__init__("")
        elif isinstance(hi, Interface) and isinstance(lo, Interface):
            super().__init__(self.make_idx(hi, lo))
        else:
            raise ValueError(f"invalid parameters {hi}, {lo}")
        self.__hi = hi
        self.__lo = lo

    def make_idx(self, hi: Interface, lo: Interface) -> str:
        return f"{repr(hi)} -> {repr(lo)}"

    @property
    def hi(self) -> Interface:
        if self.__hi is None:
            raise ValueError(f"hi not connected")
        return self.__hi

    @property
    def lo(self) -> Interface:
        if self.__lo is None:
            raise ValueError(f"lo not connected")
        return self.__lo

    def copy(
        self,
    ) -> "Wire":
        return Wire(self.hi, self.lo)
