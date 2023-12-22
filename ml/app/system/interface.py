from app.graph.graph import Slot


class Terminal(Slot):
    """A connection point for an Element or CircuitNode"""

    def __init__(self, name: str) -> None:
        super().__init__(name)


class Port(Slot):
    """A connection point for a System"""

    def __init__(self, name: str) -> None:
        super().__init__(name)
