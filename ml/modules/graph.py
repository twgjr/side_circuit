from typing import Callable


class Node:
    def __init__(self, graph: 'Graph', slots: list['Slot'] = None, name: str = "") -> None:
        self.name = name
        self.graph: 'Graph' = graph
        self.slots = slots
        if self.slots is None:
            self.slots = []
        for slot in self.slots:
            slot.node = self
        self.edges: list[Edge] = []
        if self.graph is not None:
            self.graph.nodes.append(self)

    def __repr__(self):
        return f"Node({self.deep_id()})"

    def __str__(self):
        return f"{self.deep_id()}"

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
            # stop when we reach the top level graph (node with no parent sub-graph)
            if isinstance(self, Graph):
                return ""
        elif self.graph.graph is None:
            return f"{self.graph.nodes.index(self)}"
        else:
            return f"{self.graph.deep_id()}_{self.graph.nodes.index(self)}"

    def neighbors(self) -> list['Node']:
        """Return a list of all nodes connected to this node."""
        neighbors = []
        for edge in self.edges:
            if edge.hi == self:
                neighbors.append(edge.lo)
            else:
                neighbors.append(edge.hi)
        return neighbors


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

    def neighbor(self) -> Node:
        """Return the node connected to this slot."""
        edge = self.edge()
        if edge.hi_slot == self:
            return edge.lo
        else:
            return edge.hi


class Edge:
    """An edge is a connection between two nodes. It stores a reference to the nodes it connects, but leaves updating
    the nodes to the graph."""

    def __init__(self, graph: 'Graph', hi, lo) -> None:
        self.graph: Graph = graph
        self.hi: Node = hi
        if isinstance(hi, Slot):
            self.hi: Node = hi.node
        self.lo: Node = lo
        if isinstance(lo, Slot):
            self.lo: Node = lo.node
        self.hi_slot: Slot = None
        if isinstance(hi, Slot):
            self.hi_slot: Slot = hi
        self.lo_slot: Slot = None
        if isinstance(lo, Slot):
            self.lo_slot: Slot = lo
        self.graph.edges.append(self)
        self.hi.edges.append(self)
        self.lo.edges.append(self)

    def remove(self) -> None:
        """Remove this edge from the graph and disconnect it from the nodes it connects."""
        self.graph.edges.remove(self)
        self.hi.edges.remove(self)
        self.lo.edges.remove(self)

    def __repr__(self):
        return f"Edge(hi: {self.hi}, lo: {self.lo})"


class Graph(Node):
    def __init__(self, graph: 'Graph' = None, slots: list[Slot] = None) -> None:
        super().__init__(graph, slots)
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def __repr__(self):
        return f"Graph(id: {self.deep_id()}, nodes: {self.nodes}, edges: {self.edges})"

    def deep_nodes(self) -> list[Node]:
        """Return a list of all nodes in the graph regardless of edges."""
        nodes = []
        subgraphs = []

        for node in self.nodes:
            if isinstance(node, Graph):
                subgraphs.append(node)
            nodes.append(node)

        while subgraphs:
            subgraph = subgraphs.pop(0)
            for node in subgraph.nodes:
                if isinstance(node, Graph):
                    subgraphs.append(node)
                nodes.append(node)
        return nodes

    def breadth_first_search(self, node_check: Callable[[Node], None],
                             node_match: Callable[[Node], bool]) -> list[Node]:
        """Return the nodes found by breadth first search that satisfies the callback. If first_match_only is False,
        return all nodes that satisfy the callback, otherwise return the first node that satisfies the callback.
        node_check is a callback that takes a node as an argument and raises a ValueError if the node does not
        satisfy"""

        visited = []
        queue = [self.nodes[0]]
        matching = []
        while queue:
            node = queue.pop(0)

            if node_match(node):
                return [node]

            node_check(node)
            matching.append(node)

            if node not in visited:
                visited.append(node)
                for neighbor in node.neighbors():
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)

        return matching
