from app.graph.graph import Graph, Node, Edge, Slot


def test_add_remove_edge_node_with_mixed_slots():
    graph = Graph([])
    # add a node to the graph with two slots
    node = Node([])
    slot_node = Node([Slot("p"), Slot("n")])
    graph.add_node_from(node)
    graph.add_node_from(slot_node)
    assert node in graph.nodes
    assert len(graph.nodes) == 2
    assert slot_node in graph.nodes
    # create an edge from the node to itself using slots, then add it to the graph
    edge1 = Edge(hi=node, lo=slot_node, lo_slot=slot_node["n"])
    graph.add_edge_from(edge1)
    assert len(graph.edges) == 1
    assert edge1 in graph.edges
    assert edge1 in node.edges
    assert edge1.hi == node
    assert edge1.lo == slot_node
    assert edge1.hi_slot is None
    assert edge1.lo_slot is slot_node["n"]
    # remove edge from graph, which should disconnect it from the node
    graph.remove_edge(edge1)
    assert len(graph.edges) == 0
    assert edge1 not in graph.edges
    assert node in graph.nodes
    assert slot_node in graph.nodes
    assert edge1 not in node.edges
    # create a new edge same as the old one and add it to the graph
    graph.add_edge_from(edge1)
    assert len(graph.edges) == 1
    # remove the node from the graph, which should remove the edge
    graph.remove_node(node)
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0
    assert node not in graph.nodes
    assert edge1 not in graph.edges
    graph.remove_node(slot_node)
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
    assert slot_node not in graph.nodes
    assert edge1 not in graph.edges
