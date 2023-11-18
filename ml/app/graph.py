class Node:
    """Generic graph node base class"""

    def __init__(self, max_edges: int = None) -> None:
        self.max_edges = max_edges
        self.edges: list[Edge] = [] 
        if max_edges is not None:
            self.edges = [None] * max_edges

    def connect_to(self, edge: "Edge") -> None:
        """Connect this node to an edge"""
        if self.max_edges is None or len(self.edges) < self.max_edges:
            self.edges.append(edge)
        else:
            raise ValueError("Node already has maximum number of edges")

    def disconnect(self, edge: "Edge") -> None:
        """Disconnect this node from an edge"""
        self.edges.remove(edge)
    

class Edge:
    """Generic graph edge base class"""

    def __init__(self, p: Node, n: Node) -> None:
        self.p: Node = p
        self.n: Node = n

    def connect_to_p(self, node: Node) -> None:
        """Connect this edge to the positive node"""
        if self.p is None:
            self.p = node
        else:
            self.p.disconnect(self)
            self.p = node

    def connect_to_n(self, node: Node) -> None:
        """Connect this edge to the negative node"""
        if self.n is None:
            self.n = node
        else:
            self.n.disconnect(self)
            self.n = node
