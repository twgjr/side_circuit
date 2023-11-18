class Graph:
    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []


class Node:
    """Generic graph node base class"""


class Edge:
    """Generic graph edge base class"""

    def __init__(self, from_node: Node, to_node: Node) -> None:
        super().__init__()
        self.from_node: Node = from_node
        self.to_node: Node = to_node