from app.graph.graph import Graph, Node, Edge, Slot


class System(Graph):
    def __init__(self, system: 'System' = None, ports: list['Port'] = None) -> None:
        if system is not None and ports is None:
            raise ValueError("Ports must be specified for sub-System")
        if ports is None:
            ports = []
        for port in ports:
            port.node = self
        super().__init__(system, ports)
        if system is None:
            self.ground = CircuitNode(self)  # ground node is always 0th node at top level of system
        else:
            self.ground = system.ground

    # methods to manage connections between system objects
    def split_wire(self, wire: 'Wire') -> 'CircuitNode':
        wire.remove()
        ckt_node = CircuitNode(self)
        if wire.hi_slot is None:
            Wire(system=self, hi=wire.hi, lo=ckt_node)
        else:
            Wire(system=self, hi=wire.hi_slot, lo=ckt_node)

        if wire.lo_slot is None:
            Wire(system=self, hi=ckt_node, lo=wire.lo)
        else:
            Wire(system=self, hi=ckt_node, lo=wire.lo_slot)
        return ckt_node

    def is_complete(self) -> bool:
        """Return True if the system is complete:
            - All nodes (CircuitNodes, Elements, and Systems) are connected by at least two wires
            - The system is fully connected (no unconnected or isolated nodes)
            - No empty sub-systems
            """
        # get the set of all nodes in the system
        nodes = self.deep_nodes()

        def check(node: Node) -> bool:
            """Check conditions for a node to be complete:"""
            wires_condition = len(node.edges) > 1
            subsystem_condition = True
            if isinstance(node, System):
                subsystem_condition = len(node.nodes) > 0
            return wires_condition and subsystem_condition

        connected_nodes = self.breadth_first_search(False, check)
        return len(nodes) == len(connected_nodes)


class Wire(Edge):
    """A connection between two CircuitNodes, two Elements, or Element to
    CircuitNode"""

    def __init__(self, system: System, hi, lo) -> None:
        if isinstance(hi, System):
            raise ValueError("Ports must be specified for System")
        if isinstance(hi, Element):
            raise ValueError("Terminals must be specified for Element")
        super().__init__(graph=system, hi=hi, lo=lo)


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

    def __init__(self, element: Element = None, name: str = "") -> None:
        super().__init__(element, name)


class Port(Slot):
    """A connection point for a System"""

    def __init__(self, system: System = None, name: str = "") -> None:
        super().__init__(system, name)
