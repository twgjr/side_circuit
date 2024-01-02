from modules.graph import Graph, Node, Edge, Slot


def test_add_remove_edge_node_with_mixed_slots():
    graph = Graph()
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
    graph = Graph()
    node0 = Node([])
    graph.add_node(node0)
    assert graph.node_id(node0) == "0"
    node1 = Node([])
    graph.add_node(node1)
    assert graph.node_id(node1) == "1"

