from app.graph.graph import Edge, Slot, Node


class Wire(Edge):
    """A connection between two CircuitNodes, two Elements, or Element to
    CircuitNode"""

    def __init__(self, hi: Node, lo: Node, hi_slot: Slot = None, lo_slot: Slot = None) -> None:
        super().__init__(hi, lo, hi_slot, lo_slot)
