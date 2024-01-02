from modules.graph import Graph, Node, Edge, Slot


def test_add_remove_edge_node_with_mixed_slots():
    graph = Graph([])
    # add a node to the graph with two slots
    node = Node([])
    graph.add_node(node)
    slot0 = Slot("p")
    slot1 = Slot("n")
    slot_node = Node([slot0, slot1])
    graph.add_node(slot_node)
    assert graph.num_nodes() == 2
    assert node in graph
    assert slot_node in graph

    # create an edge from the node to itself using slots, then add it to the graph
    edge = Edge()
    graph.add_edge(edge, node, slot_node["p"])
    assert graph.num_nodes() == 2
    assert graph.num_edges() == 1
    assert node in graph
    assert slot_node in graph
    assert edge in graph

    # remove edge from graph, which should disconnect it from the node
    graph.remove_edge(edge)
    assert graph.num_nodes() == 2
    assert graph.num_edges() == 0
    assert node in graph
    assert slot_node in graph
    assert edge not in graph

    # add edge back to graph
    graph.add_edge(edge, node, slot_node["p"])
    assert graph.num_nodes() == 2
    assert graph.num_edges() == 1
    assert node in graph
    assert slot_node in graph
    assert edge in graph
    
    # remove the node from the graph, which should remove the edge
    graph.remove_node(slot_node)
    assert graph.num_nodes() == 1
    assert graph.num_edges() == 0
    assert node in graph
    assert slot_node not in graph
    assert edge not in graph


def test_node_id():
    graph = Graph([])
    assert graph.deep_id() == ""
    node0 = Node(graph)
    assert node0.deep_id() == "0"
    subgraph1 = Graph(graph)
    assert subgraph1.deep_id() == "1"
    node_1_0 = Node(subgraph1)
    assert node_1_0.deep_id() == "1_0"


def test_deep_nodes():
    graph = Graph()
    node0 = Node(graph)
    node1 = Node(graph)
    subgraph2 = Graph(graph)
    node_1_0 = Node(subgraph2)
    node_1_1 = Node(subgraph2)
    assert graph.deep_nodes() == [node0, node1, subgraph2, node_1_0, node_1_1]
    assert subgraph2.deep_nodes() == [node_1_0, node_1_1]


def test_breadth_first_search():
    graph = Graph()
    node0 = Node(graph)
    node1 = Node(graph)
    subgraph2 = Graph(graph)
    Edge(graph, node0, node1)
    Edge(graph, node0, subgraph2)
    Edge(graph, node1, subgraph2)
    node_1_0 = Node(subgraph2)
    Edge(subgraph2, node_1_0, subgraph2)
    Edge(subgraph2, node_1_0, subgraph2)
    assert graph.breadth_first_search(lambda node: None, lambda node: False) == [node0, node1, subgraph2, node_1_0]
    assert graph.breadth_first_search(lambda node: None, lambda node: True) == [node0]

    def node_check(node: Node) -> None:
        if node == node0:
            raise ValueError("node0")

    try:
        graph.breadth_first_search(node_check, lambda node: False)
        assert False
    except ValueError as e:
        assert str(e) == "node0"


def test_slot_neighbor():
    graph = Graph()
    node = Node(graph)
    slotted_node = Node(graph, [Slot(name="p"), Slot(name="n")])
    Edge(graph, node, slotted_node["p"])
    assert slotted_node["p"].neighbor() == node
