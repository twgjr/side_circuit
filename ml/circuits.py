import networkx as nx
from enum import Enum
import torch
import random

class Kinds(Enum):
    IVS = 0
    ICS = 1
    R = 2

class Props(Enum):
    I = 0
    V = 1
    Pot = 2
    Attr = 3

class Circuit():
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.elements: list[Element] = []

    def add_node(self, element:'Element') -> 'Node':
        '''create a node for a new elemenet and add to circuit.
            nodes are never created without an element. No floating nodes'''
        ckt_node = Node(self,[element])
        self.nodes.append(ckt_node)
        return ckt_node

    def remove_node(self, node: 'Node'):
        if(node in self.nodes):
            self.nodes.remove(node)

    def add_element(self, kind:Kinds) -> 'Element':
        element = Element(self,kind=kind)
        high_node = self.add_node(element)
        low_node = self.add_node(element)
        element.high = high_node
        element.low = low_node
        self.elements.append(element)
        return element

    def num_nodes(self):
        return len(self.nodes)

    def num_elements(self):
        return len(self.elements)
    
    def node_idx(self, node: 'Node'):
        return self.nodes.index(node)

    def element_idx(self, element: 'Element'):
        return self.elements.index(element)

    def draw(self):
        nx.draw(self.nx_graph(), with_labels = True)

    def nx_graph(self):
        graph = nx.MultiDiGraph()
        for element in self.elements:
            element = element.to_nx()
            graph.add_edges_from([element])
        return graph

    def M(self,dtype=torch.float) -> torch.Tensor:
        M_scipy = nx.incidence_matrix(G=self.nx_graph(),oriented=True)
        M_numpy = M_scipy.toarray()
        M_tensor = torch.tensor(M_numpy,dtype=dtype)
        return M_tensor

    def __repr__(self) -> str:
        return "Circuit with " + str(len(self.nodes)) + \
                " nodes and "+ str(len(self.elements)) + " elements"

    def elements_parallel_with(self, base_element:'Element'):
        parallels = []
        for element in self.elements:
            if (element.high == base_element.high and
                element.low == base_element.low):
                parallels.append(element)
        return parallels

    def extract_elements(self):
        '''
        return dictinaries of circuit inputs
        knowns map is {prop type: list(bool)} boolean list in same order as circuit
        inputs map is {prop type: list(float)} boolean list in same order as circuit
        kinds map is {prop type: list(bool)} boolean list in same order as circuit
        '''
        inputs_map: dict[Props,list[float]] = {}
        knowns_map: dict[Props,list[float]] = {}
        kinds_map: dict[Props,list[float]] = {}

        for kind in Kinds:
            kinds_map[kind] = []

        for prop in Props:
            inputs_map[prop] = []
            knowns_map[prop] = []

        for e in range(len(self.elements)):
            element = self.elements[e]

            for kind in Kinds:
                if(element.kind == kind):
                    kinds_map[kind].append(True)
                else:
                    kinds_map[kind].append(False)

            for prop in Props:
                value = None
                if(prop == Props.I):
                    if(element.kind == Kinds.ICS):
                        value = None
                    else:
                        value = element.i
                elif(prop == Props.V):
                    if(element.kind == Kinds.IVS):
                        value = None
                    else:
                        value = element.v
                elif(prop == Props.Pot):
                    pass
                elif(prop == Props.Attr):
                    value = element.attr
                else:
                    assert()

                if(value == None):# unknown
                    inputs_map[prop].append(random.random()) # initialize unknowns
                    knowns_map[prop].append(False)
                else: # known
                    inputs_map[prop].append(float(value))
                    knowns_map[prop].append(True)

        return kinds_map, inputs_map, knowns_map

class Node():
    def __init__(self, circuit: Circuit, elements: list['Element'], p = None) -> None:
        self.circuit = circuit
        self.elements = elements
        self.p = p
        assert(self.circuit != None)
        assert(self.elements != None)

    def __repr__(self) -> str:
        return str(self.idx)

    def to_nx(self):
        v = {'v':self.p}
        return (self.idx, v)

    @property
    def idx(self):
        return self.circuit.node_idx(self)

    def clear(self):
        self.circuit.remove_node(self)
        self.circuit = None
        self.elements.clear()
        self.p = None

    def add_element(self, element: 'Element'):
        if(element not in self.elements):
            self.elements.append(element)

class Element():
    def __init__(self, circuit: Circuit, kind:Kinds, low:Node = None, high:Node = None,
                 v = None, i = None, attr = None) -> None:
        assert(isinstance(kind,Kinds))
        self.circuit = circuit
        self.low = low
        self.high = high
        self.kind = kind
        self.i = i
        self.v = v
        self.attr = attr

    def __repr__(self) -> str:
        return "("+str(self.low.idx)+ " , "+str(self.high.idx)+")"

    def to_nx(self):
        kind = ('kind',self.kind)
        v = ('v',self.v)
        i = ('i',self.i)
        attr = ('attr',self.attr)
        return (self.low.idx, self.high.idx, self.key, (kind, i, v, attr))

    @property
    def key(self):
        parallels = self.circuit.elements_parallel_with(self)
        return parallels.index(self)

    @property
    def edge_key(self):
        return self.circuit.node_idx(self)

    def has_node(self, node:Node):
        return self.low == node or self.high == node

    def connect(self, from_node: Node, to_node: Node):
        assert(len(from_node.elements) == 1)
        if(from_node == self.high):
            self.high.clear()
            self.high = to_node
            self.high.add_element(self)
        elif(from_node == self.low):
            self.low.clear()
            self.low = to_node
            self.low.add_element(self)
        else:
            assert()