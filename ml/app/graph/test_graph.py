from app.graph.graph import Graph, Node, Edge, Slot


def test_add_remove_edge_node_with_mixed_slots():
    graph = Graph()
    # add a node to the graph with two slots
    node = Node(graph)
    slot0 = Slot(name="p")
    slot1 = Slot(name="n")
    slot_node = Node(graph, [slot0, slot1])
    slot0.node = slot_node
    slot1.node = slot_node
    assert node in graph.nodes
    assert len(graph.nodes) == 2
    assert slot_node in graph.nodes
    # create an edge from the node to itself using slots, then add it to the graph
    edge1 = Edge(graph=graph, hi=node, lo=slot_node, lo_slot=slot_node["n"])
    assert len(graph.edges) == 1
    assert edge1 in graph.edges
    assert edge1 in node.edges
    assert edge1.hi == node
    assert edge1.lo == slot_node
    assert edge1.hi_slot is None
    assert edge1.lo_slot is slot_node["n"]
    # remove edge from graph, which should disconnect it from the node
    edge1.remove()
    assert len(graph.edges) == 0
    assert edge1 not in graph.edges
    assert node in graph.nodes
    assert slot_node in graph.nodes
    assert edge1 not in node.edges
    # create a new edge same as the old one and add it to the graph
    Edge(graph=graph, hi=node, lo=slot_node, lo_slot=slot_node["n"])
    assert len(graph.edges) == 1
    # remove the node from the graph, which should remove the edge
    node.remove()
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0
    assert node not in graph.nodes
    assert edge1 not in graph.edges
    slot_node.remove()
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
    assert slot_node not in graph.nodes
    assert edge1 not in graph.edges


def test_node_id():
    graph = Graph()
    assert graph.deep_id() == ""
    node0 = Node(graph)
    assert node0.deep_id() == "_0"
    subgraph1 = Graph(graph)
    assert subgraph1.deep_id() == "_1"
    node_1_0 = Node(subgraph1)
    assert node_1_0.deep_id() == "_1_0"


def test_edge_id():
    graph = Graph()
    node0 = Node(graph)
    node1 = Node(graph)
    edge0 = Edge(graph, node0, node1)
    assert edge0.deep_id() == "_0"
    subgraph1 = Graph(graph)
    node_1_0 = Node(subgraph1)
    node_1_1 = Node(subgraph1)
    edge_1_0 = Edge(subgraph1, node_1_0, node_1_1)
    assert edge_1_0.deep_id() == "_2_0"
    edge_1_1 = Edge(subgraph1, node_1_0, node_1_1)
    assert edge_1_1.deep_id() == "_2_1"
