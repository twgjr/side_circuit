import unittest
from graph import Graph, Node, Edge

class TestEdge(Edge):
    def __init__(self, value: int) -> None:
        self.value = value

    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, o: object) -> bool:
        return hash(self) == hash(o)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.value})"    
    
class TestNode(Node):
    def __init__(self, value: int) -> None:
        self.value = value

    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, o: object) -> bool:
        return hash(self) == hash(o)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.value})"


class TestGraph(unittest.TestCase):
    def test_add_remove_edge_node_with_mixed_slots(self):
        graph = Graph()
        # add a node to the graph with two slots
        n0 = TestNode(10)
        graph.set_node(n0)
        assert graph.num_nodes() == 1
        n1 = TestNode(20)
        graph.set_node(n1)
        assert graph.num_nodes() == 2

        # add an edge to the graph
        edge = TestEdge(30)
        graph.set_edge(edge, n0, n1)
        assert graph.num_nodes() == 2
        assert graph.num_edges() == 1

        # remove edge from graph, which should disconnect it from the node
        graph.delete_edge(str(edge))
        assert graph.num_nodes() == 2
        assert graph.num_edges() == 0

        # add edge back to graph
        graph.set_edge(edge, n0, n1)
        assert graph.num_nodes() == 2
        assert graph.num_edges() == 1

        # remove the node from the graph, which should remove the edge
        graph.delete_node(str(n1))
        assert graph.num_nodes() == 1
        assert graph.num_edges() == 0
