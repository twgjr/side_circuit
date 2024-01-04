from abc import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, o: object) -> bool:
        raise NotImplementedError


class Edge(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, o: object) -> bool:
        raise NotImplementedError


class Graph:
    def __init__(self) -> None:
        self.__nodes: dict[str, Node] = {}
        self.__edges: dict[str, Edge] = {}
        self.__node_edge_map: dict[str, list[str]] = {}
        self.__edge_node_map: dict[str, tuple[str, str]] = {}

    def __get_node_edge_map(self, node: str) -> list[str]:
        try:
            return self.__node_edge_map[node]
        except KeyError:
            raise KeyError(f"{node} not in node edge map")

    def __get_edge_node_map(self, edge: str) -> tuple[str, str]:
        try:
            return self.__edge_node_map[edge]
        except KeyError:
            raise KeyError(f"{edge} not in edge node map")

    def get_node(self, node: str) -> Node:
        try:
            return self.__nodes[node]
        except KeyError:
            raise KeyError(f"{node} not in nodes")

    def get_edge(self, edge: str) -> Edge:
        try:
            return self.__edges[edge]
        except KeyError:
            raise KeyError(f"{edge} not in edges")

    def neighbors(self, node: str) -> list[Node]:
        return [self.get_node(node) for node in self.__get_node_edge_map(node)]

    def set_node(self, node: Node) -> None:
        node_str = str(node)
        self.__nodes[node_str] = node
        self.__node_edge_map[node_str] = []

    def set_edge(self, edge: Edge, node1: Node, node2: Node) -> None:
        edge_str = str(edge)
        node1_str = str(node1)
        node2_str = str(node2)
        self.__edges[edge_str] = edge
        self.__node_edge_map[node1_str].append(edge_str)
        self.__node_edge_map[node2_str].append(edge_str)
        self.__edge_node_map[edge_str] = (node1_str, node2_str)

    def delete_node(self, node: str) -> None:
        for edge in self.__get_node_edge_map(node):
            self.delete_edge(edge)
        if node in self.__nodes:
            del self.__nodes[node]
        if node in self.__node_edge_map:
            del self.__node_edge_map[node]

    def delete_edge(self, edge: str) -> None:
        nodes = self.__edge_node_map[edge]
        for node in nodes:
            self.__node_edge_map[node].remove(edge)
        if edge in self.__edge_node_map:
            del self.__edge_node_map[edge]
        if edge in self.__edges:
            del self.__edges[edge]

    def num_nodes(self) -> int:
        return len(self.__nodes)

    def num_edges(self) -> int:
        return len(self.__edges)
