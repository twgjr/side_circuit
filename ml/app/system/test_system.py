from app.system.system import System
from app.system.circuit_node import CircuitNode
from app.system.wire import Wire


def test_split_wire():
    system = System([])
    # add a wire to the system
    cn1 = CircuitNode()
    system.add_node_from(cn1)
    cn2 = CircuitNode()
    system.add_node_from(cn2)
    wire = Wire(cn1, cn2)
    system.add_edge_from(wire)
    assert len(system.edges) == 1
    assert wire in system.edges
    assert cn1 in system.nodes
    assert cn2 in system.nodes
    # split the wire
    ckt_node = system.split_wire(wire)
    assert len(system.nodes) == 3
    assert len(system.edges) == 2
    assert wire not in system.edges
    assert ckt_node in system.nodes
    assert cn1 in system.nodes
    assert cn2 in system.nodes
    assert ckt_node in system.nodes

