from app.graph.graph import Graph, Slot, Node
from app.system.wire import Wire
from app.system.circuit_node import CircuitNode
from app.system.element import Element
from app.system.interface import Port


class System(Graph):
    def __init__(self, ports: list[Port]) -> None:
        super().__init__(ports)

    # methods to manage connections between system objects
    def split_wire(self, wire: Wire) -> CircuitNode:
        p = wire.hi
        n = wire.lo
        self.remove_edge(wire)
        ckt_node = CircuitNode()
        self.add_node_from(ckt_node)
        wire0 = Wire(hi=p,
                     lo=ckt_node,
                     hi_slot=wire.hi_slot)
        self.add_edge_from(wire0)
        wire1 = Wire(hi=ckt_node,
                     lo=n,
                     lo_slot=wire.lo_slot)
        self.add_edge_from(wire1)
        return ckt_node
