class Node:
    def __init__(self, slots: list["Slot"]) -> None:
        self.__slots = slots

    @property
    def slots(self) -> list["Slot"]:
        return self.__slots

    def __getitem__(self, key: str) -> "Slot":
        for slot in self.slots:
            if slot.name == key:
                return slot
        raise KeyError(key)


class Slot:
    def __init__(self, name: str = "") -> None:
        self.name: str = name


class Edge:
    def __init__(self) -> None:
        pass


class Graph:
    def __init__(self) -> None:
        self.__nodes: list[Node] = []
        self.__edges: list[Edge] = []
        self.__slot_node_map: dict[Slot, Node] = {}
        self.__from_edge_map: dict[Edge, list[Slot | Node]] = {}
        self.__to_edge_map: dict[Slot | Node, Edge] = {}

    def __contains__(self, item) -> bool:
        if isinstance(item, Node):
            if item not in self.__nodes:
                return False
            for slot in item.slots:
                if slot not in self.__slot_node_map:
                    return False
                if self.__slot_node_map[slot] != item:
                    return False
            return True

        elif isinstance(item, Edge):
            if item not in self.__edges:
                return False
            if item not in self.__from_edge_map:
                return False
            hi, lo = self.__from_edge_map[item]
            if hi not in self.__to_edge_map:
                return False
            if self.__to_edge_map[hi] != item:
                return False
            if lo not in self.__to_edge_map:
                return False
            if self.__to_edge_map[lo] != item:
                return False
            return True

        return False

    # node methods

    def node_id(self, node: Node) -> str:
        return str(self.__nodes.index(node))
    
    def add_node(self, node: Node) -> None:
        self.__nodes.append(node)
        for slot in node.slots:
            self.__slot_node_map[slot] = node

    def remove_node(self, node: Node) -> None:
        # remove the node from the graph
        self.__nodes.remove(node)

        # remove slot mappings
        for slot in node.slots:
            del self.__slot_node_map[slot]

        # remove edges
        if node in self.__to_edge_map:
            self.remove_edge(self.__to_edge_map[node])
        for slot in node.slots:
            if slot in self.__to_edge_map:
                self.remove_edge(self.__to_edge_map[slot])

    def num_nodes(self) -> int:
        return len(self.__nodes)

    # edge methods

    def add_edge(self, edge: Edge, hi: Slot | Node, lo: Slot | Node) -> None:
        if hi == lo:
            raise ValueError("hi and lo cannot be the same")
        self.__edges.append(edge)
        self.__from_edge_map[edge] = [hi, lo]
        self.__to_edge_map[hi] = edge
        self.__to_edge_map[lo] = edge

    def remove_edge(self, edge: Edge) -> None:
        self.__edges.remove(edge)
        hi, lo = self.__from_edge_map[edge]
        del self.__from_edge_map[edge]
        del self.__to_edge_map[hi]
        del self.__to_edge_map[lo]

    def num_edges(self) -> int:
        return len(self.__edges)
