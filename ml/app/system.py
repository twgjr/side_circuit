from enum import Enum
from app.graph import Node, Edge
from app.element import Element


class Quantity(Enum):
    I = "current"
    V = "voltage"
    P = "potential"


class Wire(Edge):
    """A connection between two CircuitNodes, two Elements, or Element to
    CircuitNode"""

    def __init__(self, p: Node, n: Node) -> None:
        super().__init__(p, n)

    def disconnect(self, node: Node) -> None:
        """Disconnect the wire from the node"""
        if self.p == node:
            self.p = None
        elif self.n == node:
            self.n = None
        else:
            raise ValueError("Node not connected to wire")

    def connect_to(self, node: Node) -> None:
        """Connect a node to one of the unconnected ends of the wire"""
        if self.p is None:
            self.p = node
            node.connect_to(self)
        elif self.n is None:
            self.n = node
        else:
            raise ValueError("Wire already fully connected")


class Port(Node):
    """An interface point at the edge of a sub-system connecting to outside
    sub-Systems or Elements"""

    def __init__(self) -> None:
        super().__init__(max_edges=2)

    def disconnected(self) -> bool:
        """Return True if the port is not connected to a wire"""
        return self.edges[0] is None and self.edges[1] is None
    
    @property
    def wire_in(self) -> Wire:
        """Return the wire inside the sub-system of this port"""
        return self.edges[0]
        
    @wire_in.setter
    def wire_in(self, wire: Wire) -> None:
        """Set the wire inside the sub-system of this port"""
        self.edges[0] = wire

    @property
    def wire_out(self) -> Wire:
        """Return the wire outside the sub-system of this port"""
        return self.edges[1]
    
    @wire_out.setter
    def wire_out(self, wire: Wire) -> None:
        """Set the wire outside the sub-system of this port"""
        self.edges[1] = wire


class CircuitNode(Node):
    """wire to wire connection point within a circuit.  Has no distinct ports"""

    def __init__(self) -> None:
        super().__init__(max_edges=None)


class System(Node):
    """From outside it is viewed as a single node with ports (part of parent
    inherited Node class).  Internally
    """

    def __init__(self) -> None:
        super().__init__(max_edges=0)
        self.sub_systems: list[System] = []
        self.elements: list[Element] = []
        self.ports: dict[str, Port] = {}
        self.circuit_nodes: list[CircuitNode] = []
        self.wires: list[Wire] = []

    def add_sub_system(self, sub_system: 'System') -> None:
        """Add a sub-system to the system"""
        self.sub_systems.append(sub_system)

    def add_element(self, element: Element) -> None:
        """Add an element to the system"""
        self.elements.append(element)

    def add_port(self, name: str) -> Port:
        """Add a port to the system"""
        port = Port()
        self.ports[name] = port
        return port

    def add_circuit_node(self) -> CircuitNode:
        """Add a circuit node to the system"""
        circuit_node = CircuitNode()
        self.circuit_nodes.append(circuit_node)
        return circuit_node

    def add_wire(self, p: Node, n: Node) -> Wire:
        """Add a wire to the system"""
        wire = Wire(p, n)
        p.connect_to(wire)
        n.connect_to(wire)
        self.wires.append(wire)
        return wire

    def split_wire_to(self, port: Port) -> CircuitNode:
        """Split a wire connected to port into two wires and connect them with
        a circuit node"""
        wire = port.wire
        wire.disconnect(port)
        circuit_node = self.add_circuit_node()
        wire.connect_to(circuit_node)
        new_wire = self.add_wire(circuit_node, port)
        port.wire = new_wire
        return circuit_node

    def connect(self, from_port: Port, to_port: Port) -> None:
        """Connect two ports together.  Manage ports that may already be
        connected to wires."""
        raise NotImplementedError("connect method needs different way to check disconnected")
        if from_port.disconnected():
            if to_port.disconnected():
                # both ports are disconnected, connect them with a new wire
                wire = self.add_wire(from_port, to_port)
                from_port.wire = wire
                to_port.wire = wire
            else:
                # connect new wire from from_port to to_port wire
                ckt_node = self.split_wire_to(to_port)
                self.add_wire(from_port, ckt_node)
        else:
            # from_port is connected to a wire
            if to_port.disconnected():
                # connect new wire from to_port to from_port wire
                ckt_node = self.split_wire_to(from_port)
                self.add_wire(to_port, ckt_node)
            else:
                # both ports are connected to wires, connect the two wires
                ckt_node1 = self.split_wire_to(from_port)
                ckt_node2 = self.split_wire_to(to_port)
                self.add_wire(ckt_node1, ckt_node2)
