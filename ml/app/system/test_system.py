from app.system.system import *


def test_split_wire():
    system = System()
    assert len(system.edges) == 0
    assert len(system.nodes) == 1  # ground
    # add a wire to the system
    cn1 = CircuitNode(system)
    cn2 = CircuitNode(system)
    wire = Wire(system, cn1, cn2)
    assert len(system.edges) == 1
    assert len(system.nodes) == 3  # ground and two circuit nodes
    assert wire in system.edges
    assert cn1 in system.nodes
    assert cn2 in system.nodes
    # split the wire
    ckt_node = system.split_wire(wire)
    assert len(system.nodes) == 4  # ground and three circuit nodes
    assert len(system.edges) == 2
    assert wire not in system.edges
    assert ckt_node in system.nodes
    assert cn1 in system.nodes
    assert cn2 in system.nodes
    assert ckt_node in system.nodes