from app.graph.graph import Graph, Node, Edge, Slot


class System(Graph):
    def __init__(self, system: 'System' = None, ports: list[str] = None) -> None:
        super().__init__(system, ports)
        self.ground = CircuitNode(self)  # ground node is always 0th node at top level of system

    # methods to manage connections between system objects
    def split_wire(self, wire: 'Wire') -> 'CircuitNode':
        p = wire.hi
        n = wire.lo
        wire.remove()
        ckt_node = CircuitNode(self)
        Wire(system=self, hi=p, lo=ckt_node, hi_slot=wire.hi_slot)
        Wire(system=self, hi=ckt_node, lo=n, lo_slot=wire.lo_slot)
        return ckt_node


class Wire(Edge):
    """A connection between two CircuitNodes, two Elements, or Element to
    CircuitNode"""

    def __init__(self, system: System, hi: Node, lo: Node, hi_slot: Slot = None, lo_slot: Slot = None) -> None:
        super().__init__(graph=system, hi=hi, lo=lo, hi_slot=hi_slot, lo_slot=lo_slot)


class Element(Node):
    """An intrinsic circuit element viewed as a single node with terminals.
    No internal sub-systems, elements, wires, or circuit nodes."""

    def __init__(self, system: System, slots: list[Slot], kind) -> None:
        super().__init__(system, slots)
        from app.system.elements import Kind
        if not isinstance(kind, Kind):
            raise ValueError("Invalid kind for element")
        self.kind = kind


class CircuitNode(Node):
    """wire to wire connection point within a circuit"""

    def __init__(self, system: System) -> None:
        super().__init__(graph=system, slots=[])


class Terminal(Slot):
    """A connection point for an Element or CircuitNode"""

    def __init__(self, node: Node, name: str) -> None:
        super().__init__(node, name)


class Port(Slot):
    """A connection point for a System"""

    def __init__(self, system: System, name: str) -> None:
        super().__init__(system, name)
