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


def test_is_complete():
    system = System()
    assert not system.is_complete()  # ground is not connected to anything
    # add a wire to the system
    ss2 = System(system, [Port(name="p"), Port(name="n")])
    Wire(system, hi=system.ground, lo=ss2["p"])
    assert not system.is_complete()
    # complete the system by adding a second wire
    Wire(system, hi=system.ground, lo=ss2["n"])
    assert not system.is_complete()
    # ss2 is empty, so the system is not truly complete, add nodes and edges to ss2
    cn3 = CircuitNode(ss2)
    cn4 = CircuitNode(ss2)
    Wire(ss2, cn3, cn4)
    assert not system.is_complete()
    Wire(ss2, cn3, cn4)
    assert system.is_complete()
