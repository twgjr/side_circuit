class Node:
    def __init__(self, graph: 'Graph', slots: list['Slot'] = None) -> None:
        self.graph: 'Graph' = graph
        self.slots = slots
        if self.slots is None:
            self.slots = []
        self.edges: list[Edge] = []
        if self.graph is not None:
            self.graph.nodes.append(self)

    def remove(self) -> None:
        """Remove this node from the graph and all connected edges."""
        for edge in self.edges:
            edge.remove()
        self.graph.nodes.remove(self)

    def slot_is_valid(self, slot: 'Slot') -> bool:
        if slot in self.slots:
            return True
        if len(self.slots) == 0 and slot is None:
            return True
        return False

    def __getitem__(self, key: str) -> 'Slot':
        for slot in self.slots:
            if slot.name == key:
                return slot
        raise KeyError("Slot not found")

    def deep_id(self) -> str:
        # recursively find the index of the graph this node belongs to
        # append indices as we go
        if self.graph is None:
            return ""
        return self.graph.deep_id() + "_" + str(self.graph.nodes.index(self))


class Slot:
    def __init__(self, node: Node = None, name: str = "") -> None:
        self.node: Node = node
        self.name: str = name

    def edge(self) -> 'Edge':
        """Return the edge connected to this slot."""
        for edge in self.node.edges:
            if edge.hi_slot == self or edge.lo_slot == self:
                return edge
        raise ValueError("No edge connected to slot")


class Edge:
    """An edge is a connection between two nodes. It stores a reference to the nodes it connects, but leaves updating
    the nodes to the graph."""

    def __init__(self, graph: 'Graph', hi: Node, lo: Node, hi_slot: Slot = None, lo_slot: Slot = None) -> None:
        self.graph: Graph = graph
        self.hi: Node = hi
        self.lo: Node = lo
        if not hi.slot_is_valid(hi_slot):
            raise ValueError("Invalid slot for node")
        self.hi_slot: Slot = hi_slot
        if not lo.slot_is_valid(lo_slot):
            raise ValueError("Invalid slot for node")
        self.lo_slot: Slot = lo_slot
        self.graph.edges.append(self)
        self.hi.edges.append(self)
        self.lo.edges.append(self)

    def remove(self) -> None:
        """Remove this edge from the graph and disconnect it from the nodes it connects."""
        self.graph.edges.remove(self)
        self.hi.edges.remove(self)
        self.lo.edges.remove(self)

    def deep_id(self) -> str:
        # recursively find the index of the graph this node belongs to
        # append indices as we go
        if self.graph is None:
            return ""
        return self.graph.deep_id() + "_" + str(self.graph.edges.index(self))


class Graph(Node):
    def __init__(self, graph: 'Graph' = None, slots: list[Slot] = None) -> None:
        super().__init__(graph, slots)
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
