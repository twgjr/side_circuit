from graph import Graph, Node, Edge


class System(Graph):
    def __init__(self) -> None:
        super().__init__()
        self.__subsystem = SubSystem(self)
        super().set_node(self.__subsystem)
        self.__ground = CircuitNode()
        super().set_node(self.__ground)

    @property
    def subsystem(self) -> "SubSystem":
        return self.__subsystem

    @property
    def gnd(self) -> "CircuitNode":
        return self.__ground


class SubSystem(Node):
    """graph node that acts as interface for a cluster of nodes and edges"""

    def __init__(self, system: System) -> None:
        super().__init__()
        self.__system: System = system
        self.__circuit_nodes: list[CircuitNode] = []
        self.__ports: list[Port] = []
        self.__elements: list[Element] = []
        self.__wires: list[Wire] = []
        self.__subsystems: list[SubSystem] = []

    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, o: object) -> bool:
        return hash(self) == hash(o)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{id(self)}"

    def __contains__(self, item) -> bool:
        return (
            item in self.__circuit_nodes
            or item in self.__ports
            or item in self.__elements
            or item in self.__wires
            or item in self.__subsystems
        )

    def add_circuit_node(self) -> "CircuitNode":
        node = CircuitNode()
        self.__system.set_node(node)
        self.__circuit_nodes.append(node)
        return node

    def add_wire(self, item1: "Node | Slot", item2: "Node | Slot") -> "Wire":
        if isinstance(item1, CircuitNode) and isinstance(item2, CircuitNode):
            raise ValueError("Cannot connect two CircuitNodes")
        edge = Wire(None, None)  # type: ignore
        if isinstance(item1, Slot) and isinstance(item2, Slot):
            edge = Wire(item1, item2)
            self.__system.set_edge(edge, item1.parent, item2.parent)
        elif isinstance(item1, Slot) and isinstance(item2, Node):
            edge = Wire(item1, None)  # type: ignore
            self.__system.set_edge(edge, item1.parent, item2)
        elif isinstance(item1, Node) and isinstance(item2, Slot):
            edge = Wire(None, item2)  # type: ignore
            self.__system.set_edge(edge, item1, item2.parent)
        elif isinstance(item1, Node) and isinstance(item2, Node):
            self.__system.set_edge(edge, item1, item2)

        self.__wires.append(edge)
        return edge

    def add_element(self, element: "Element") -> "Element":
        self.__system.set_node(element)
        self.__elements.append(element)
        return element

    def add_subsystem(self) -> "SubSystem":
        node = SubSystem(self.__system)
        self.__system.set_node(node)
        self.__subsystems.append(node)
        return node


class Wire(Edge):
    """graph edge connecting CircuitNodes, Elements and SubSystems.
    Wires hold references to connected Ports of SubSystems or"""

    def __init__(self, slot_a: "Slot", slot_b: "Slot") -> None:
        super().__init__()
        self.__slots = (slot_a, slot_b)

    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, o: object) -> bool:
        return hash(self) == hash(o)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{id(self)})"
        


class Element(Node):
    """graph node that only connects to Terminal nodes"""

    def __init__(self, terminals: list["Terminal"]) -> None:
        super().__init__()
        self.terminals = terminals


class CircuitNode(Node):
    """graph node that connects directly with only Wire edges"""

    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, o: object) -> bool:
        return hash(self) == hash(o)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{id(self)})"

class Slot:
    """graph node that acts as interface for a Wire edge"""

    def __init__(self, parent) -> None:
        self.__parent: SubSystem | Element = parent

    @property
    def parent(self) -> SubSystem | Element:
        return self.__parent


class Terminal(Slot):
    """graph node that acts as the label for incoming/outgoing edges of an Element node"""

    def __init__(self, parent: Element, name: str) -> None:
        super().__init__(parent)
        self.name = name


class Port(Slot):
    """graph node that acts as the label for incoming/outgoing edges of an SubSystem node"""

    def __init__(self, parent: SubSystem, name: str) -> None:
        super().__init__(parent)
        self.name = name
