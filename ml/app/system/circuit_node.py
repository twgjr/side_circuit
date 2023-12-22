from app.graph.graph import Node, Slot


class CircuitNode(Node):
    """wire to wire connection point within a circuit"""

    def __init__(self) -> None:
        super().__init__([])
