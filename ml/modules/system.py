from modules.graph import Graph, Node, Edge, Slot
import json


class System(Graph):
    def __init__(self, system: "System" = None, ports: list["Port"] = None) -> None:
        if system is not None and ports is None:
            raise ValueError("Ports must be specified for sub-System")
        if ports is None:
            ports = []
        for port in ports:
            port.node = self
        super().__init__(system, ports)
        if system is None:
            self.ground = CircuitNode(
                self
            )  # ground node is always 0th node at top level of system
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

    def to_dict(self) -> dict:
        """Return a dictionary representation of the system"""
        return {
            "id": self.deep_id(),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_json(self) -> str:
        """Return a JSON representation of the system"""
        return json.dumps(self.to_dict(), indent=4)

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
                    raise ValueError(
                        f"{node} and {neighbor} are neighbors of the same type"
                    )

        connected_nodes = self.breadth_first_search(node_check, lambda node: False)
        if len(nodes) != len(connected_nodes):
            raise ValueError("System is not fully connected")


class Wire(Edge):
    """A connection between two CircuitNodes, two Elements, or Element to
    CircuitNode"""

    def __init__(self, system: System, hi, lo) -> None:
        if isinstance(hi, System) or isinstance(lo, System):
            raise ValueError("Ports must be specified for System")
        if isinstance(hi, Element) or isinstance(lo, Element):
            raise ValueError("Terminals must be specified for Element")
        
        # merge CircuitNodes if connecting two CircuitNodes
        if isinstance(hi, CircuitNode) and isinstance(lo, CircuitNode):
            hi.merge(lo)
            # but don't create a new Wire as an Edge
        else:
            # if not merging, create a new Wire as an Edge
            super().__init__(graph=system, hi=hi, lo=lo)

        # add a CircuitNode if connecting two Elements/Systems
        if (not isinstance(hi, CircuitNode)) and (not isinstance(lo, CircuitNode)):
            self.split()

    def to_dict(self) -> dict:
        """Return a dictionary representation of the wire"""
        return {
            "id": self.deep_id(),
            "hi": self.hi.deep_id(),
            "lo": self.lo.deep_id(),
        }
    
    def __repr__(self) -> str:
        return f"Wire({self.hi.deep_id()}->{self.lo.deep_id()})"

    def deep_id(self) -> str:
        return f"{self.hi.deep_id()}->{self.lo.deep_id()}"

    def split(self) -> "CircuitNode":
        """Split the wire into two wires and return the new CircuitNode"""
        if not isinstance(self.graph, System):
            raise ValueError("Wire must be part of a System")
        ckt_node = CircuitNode(self.graph)

        if self.hi_slot is None:
            Wire(system=self.graph, hi=self.hi, lo=ckt_node)
        else:
            Wire(system=self.graph, hi=self.hi_slot, lo=ckt_node)

        if self.lo_slot is None:
            Wire(system=self.graph, hi=ckt_node, lo=self.lo)
        else:
            Wire(system=self.graph, hi=ckt_node, lo=self.lo_slot)
        
        self.remove()
        
        return ckt_node


class Element(Node):
    """An intrinsic circuit element viewed as a single node with terminals.
    No internal sub-systems, elements, wires, or circuit nodes."""

    def __init__(self, system: System, slots: list[Slot], kind) -> None:
        super().__init__(system, slots)
        from modules.elements import Kind

        if not isinstance(kind, Kind):
            raise ValueError("Invalid kind for element")
        self.kind = kind

    def to_dict(self) -> dict:
        """Return a dictionary representation of the element"""
        return {
            "id": self.deep_id(),
            "kind": self.kind.name,
            "slots": [slot.to_dict() for slot in self.slots],
        }

    def __str__(self):
        return f"{self.kind.name}{self.deep_id()}"


class CircuitNode(Node):
    """wire to wire connection point within a circuit"""

    def __init__(self, system: System) -> None:
        super().__init__(graph=system, slots=[])

    def __repr__(self) -> str:
        return f"CircuitNode({self.deep_id()})"

    def merge(self, other: "CircuitNode") -> None:
        '''Replace node with self in all edges and remove node from graph'''
        if not isinstance(other, CircuitNode):
            raise ValueError("Must merge CircuitNodes")
        if other.graph != self.graph and not (isinstance(other.graph, System) and isinstance(self.graph, System)):
            raise ValueError("CircuitNodes must be part of same System")
        if other == self:
            raise ValueError("Cannot merge node with itself")

        if other == self.graph.ground:
            # merge self into ground (other) node
            for edge in self.edges:
                other.edges.append(edge)
                if edge.hi == self:
                    edge.hi = other
                else:
                    edge.lo = other
            self.remove()
        else:
            # merge other node into self
            for edge in other.edges:
                self.edges.append(edge)
                if edge.hi == other:
                    edge.hi = self
                else:
                    edge.lo = self
            other.remove()

    def to_dict(self) -> dict:
        """Return a dictionary representation of the circuit node"""
        return {"id": self.deep_id()}


class Terminal(Slot):
    """A connection point for an Element or CircuitNode"""

    def __init__(self, element: Element = None, name: str = "") -> None:
        super().__init__(element, name)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the terminal"""
        return {"id": self.node.slots.index(self), "name": self.name}


class Port(Slot):
    """A connection point for a System"""

    def __init__(self, system: System = None, name: str = "") -> None:
        super().__init__(system, name)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the port"""
        return {"id": self.node.slots.index(self), "name": self.name}
