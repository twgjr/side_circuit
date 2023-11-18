from enum import Enum
from graph import Node, Edge
from elements import Element


class Quantity(Enum):
    I = "current"
    V = "voltage"
    P = "potential"


class Wire(Edge):
    """A connection between two CircuitNodes, two Elements, or Element to
    CircuitNode"""

    def __init__(self, from_node: Node, to_node: Node) -> None:
        super().__init__(from_node, to_node)

    def disconnect(self, node: Node) -> None:
        """Disconnect the wire from the node"""
        if self.from_node == node:
            self.from_node = None
        elif self.to_node == node:
            self.to_node = None
        else:
            raise ValueError("Node not connected to wire")

    def connect(self, node: Node) -> None:
        """Connect a node to one of the unconnected ends of the wire"""
        if self.from_node is None:
            self.from_node = node
        elif self.to_node is None:
            self.to_node = node
        else:
            raise ValueError("Wire already fully connected")


class Port(Node):
    """An interface point at the edge of a sub-system connecting to outside
    sub-Systems or Elements"""

    def __init__(self) -> None:
        super().__init__()
        self.wire: Wire = None


class CircuitNode(Node):
    """wire to wire connection point within a circuit.  Has no distinct ports"""

    def __init__(self) -> None:
        super().__init__()


class System(Node):
    """From outside it is viewed as a single node with ports (part of parent
    inherited Node class).  Internally
    """

    def __init__(self) -> None:
        super().__init__()
        self.sub_systems: list[System] = []
        self.elements: list[Element] = []
        self.ports: dict[str, Port] = {}
        self.circuit_nodes: list[CircuitNode] = []
        self.wires: list[Wire] = []

    def split_wire(self, port: Port) -> CircuitNode:
        """Split a wire connected to port into two wires and connect them with
        a circuit node"""
        wire = port.wire
        wire.disconnect(port)
        circuit_node = CircuitNode()
        self.circuit_nodes.append(circuit_node)
        wire.connect(circuit_node)
        new_wire = Wire(circuit_node, port)
        self.wires.append(new_wire)
        port.wire = new_wire
        return circuit_node

    def connect(self, from_port: Port, to_port: Port) -> None:
        """Connect two ports together.  Manage ports that may already be
        connected to wires."""
        if from_port.wire is None and to_port.wire is None:
            # neither port is connected to a wire, connect them with a new wire
            wire = Wire(from_port, to_port)
            from_port.wire = wire
            to_port.wire = wire
            self.wires.append(wire)
        elif from_port.wire is not None and to_port.wire is None:
            # from_port is connected to a wire but to_port is not
            circuit_node = self.split_wire(from_port)
            # connect to_port to circuit node with a new wire
            wire = Wire(circuit_node, to_port)
            self.wires.append(wire)
            to_port.wire = wire
        elif from_port.wire is None and to_port.wire is not None:
            # from_port is not connected to a wire but to_port is
            circuit_node = self.split_wire(to_port)
            # connect from_port to circuit node with a new wire
            wire = Wire(circuit_node, from_port)
            self.wires.append(wire)
            from_port.wire = wire
        else:
            # both ports are connected to wires, split both wires and connect
            # the two circuit nodes with a new wire
            circuit_node1 = self.split_wire(from_port)
            circuit_node2 = self.split_wire(to_port)
            wire = Wire(circuit_node1, circuit_node2)
            self.wires.append(wire)
