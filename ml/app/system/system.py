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

    def __repr__(self) -> str:
        nodes_ids = []
        for node in self.nodes:
            nodes_ids.append(node.deep_id())
        edges_ids = []
        for edge in self.edges:
            edge_id = edge.deep_id()
            if edge_id not in edges_ids:
                edges_ids.append(edge.deep_id())
        return f"System(id: {self.deep_id()})"

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

    def check_complete(self) -> None:
        """Return True if the system is complete:
            - All nodes (CircuitNodes, Elements, and Systems) are connected by at least two wires
            - The system is fully connected (no unconnected or isolated nodes)
            - No empty sub-systems
            - sub-Systems and Elements must not be directly connected with a Wire (need a CircuitNode in between)
            """
        # get the set of all nodes in the system
        nodes = self.deep_nodes()

        def node_check(node: Node) -> None:
            """raise ValueErrors for each condition that is not met"""

            # node is connected by at least two wires
            if len(node.edges) < 2:
                raise ValueError(f"{node} is not connected by at least two wires")

            # System must not be empty
            if isinstance(node, System) and len(node.nodes) == 0:
                raise ValueError(f"{node} is empty")

            # nodes must not be neighbors of the same type
            for neighbor in node.neighbors():
                if isinstance(node, type(neighbor)):
                    raise ValueError(f"{node} and {neighbor} are neighbors of the same type")

        connected_nodes = self.breadth_first_search(node_check, lambda node: False)
        if len(nodes) != len(connected_nodes):
            raise ValueError("System is not fully connected")


class Wire(Edge):
    """A connection between two CircuitNodes, two Elements, or Element to
    CircuitNode"""

    def __init__(self, system: System, hi, lo) -> None:
        if isinstance(hi, System):
            raise ValueError("Ports must be specified for System")
        if isinstance(hi, Element):
            raise ValueError("Terminals must be specified for Element")
        super().__init__(graph=system, hi=hi, lo=lo)

    def __repr__(self) -> str:
        return f"Wire({self.deep_id()})"


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

    def __repr__(self) -> str:
        return f"CircuitNode({self.deep_id()})"


class Terminal(Slot):
    """A connection point for an Element or CircuitNode"""

    def __init__(self, element: Element = None, name: str = "") -> None:
        super().__init__(element, name)


class Port(Slot):
    """A connection point for a System"""

    def __init__(self, system: System = None, name: str = "") -> None:
        super().__init__(system, name)
