class Node:
    def __init__(self, slots: list['Slot']) -> None:
        self.slots: list['Slot'] = slots
        self.edges: list[Edge] = []

    def slot_is_valid(self, slot: 'Slot') -> bool:
        if slot in self.slots:
            return True
        if len(self.slots) == 0 and slot is None:
            return True
        return False

    def __getitem__(self, key: str) -> 'Slot':
        for slot in self.slots:
            if slot.slot == key:
                return slot
        raise KeyError("Slot not found")


class Slot:
    def __init__(self, slot: str) -> None:
        self.slot: str = slot


class Edge:
    """An edge is a connection between two nodes. It stores a reference to the nodes it connects, but leaves updating
    the nodes to the graph."""
    def __init__(self, hi: Node, lo: Node, hi_slot: Slot = None, lo_slot: Slot = None) -> None:
        self.hi: Node = hi
        self.lo: Node = lo
        if not hi.slot_is_valid(hi_slot):
            raise ValueError("Invalid slot for node")
        self.hi_slot: Slot = hi_slot
        if not lo.slot_is_valid(lo_slot):
            raise ValueError("Invalid slot for node")
        self.lo_slot: Slot = lo_slot


class Graph(Node):
    def __init__(self, slots: list[Slot]) -> None:
        super().__init__(slots)
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def add_node_from(self, node: Node) -> None:
        self.nodes.append(node)

    def remove_node(self, node: Node) -> None:
        """Removing a node from a graph also removes all edges connected to it."""
        for edge in node.edges:
            self.remove_edge(edge)
        self.nodes.remove(node)

    def add_edge_from(self, edge: Edge) -> None:
        edge.hi.edges.append(edge)
        edge.lo.edges.append(edge)
        self.edges.append(edge)

    def remove_edge(self, edge: Edge) -> None:
        """Remove an edge from the graph while also removing references to it from the nodes it connects."""
        edge.hi.edges.remove(edge)
        edge.lo.edges.remove(edge)
        self.edges.remove(edge)
