import networkx as nx
from enum import Enum
import torch
from torch import Tensor
from torch.nn.functional import normalize

class Kinds(Enum):
    IVS = 0
    ICS = 1
    R = 2

class Props(Enum):
    I = 0
    V = 1
    Pot = 2

class Circuit():
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.elements: list[Element] = []

    def clear(self):
        for node in self.nodes:
            node.clear()
        for element in self.elements:
            element.clear()
        self.nodes.clear()
        self.elements.clear()
    
    def add_element(self, element:'Element') -> None:
        assert(isinstance(element,Element))
        high_node = self.add_node([element])
        low_node = self.add_node([element])
        element.high = high_node
        element.low = low_node
        self.elements.append(element)
        return element
    
    def add_element_of(self, kind:Kinds) -> 'Element':
        assert(isinstance(kind,Kinds))
        element = Element(self,kind=kind)
        high_node = self.add_node([element])
        low_node = self.add_node([element])
        element.high = high_node
        element.low = low_node
        self.elements.append(element)
        return element
    
    def remove_element(self, element: 'Element'):
        if(element in self.elements):
            element.clear()
            self.elements.remove(element)
    
    def add_node(self, elements:list['Element']) -> 'Node':
        '''create a node for a new element and add to circuit.
            nodes are never created without an element. No floating nodes'''
        
        assert(isinstance(elements,list))
        for element in elements:
            assert(isinstance(element,Element))
        ckt_node = Node(self,elements)
        self.nodes.append(ckt_node)
        return ckt_node

    def remove_node(self, node: 'Node'):
        if(node in self.nodes):
            node.clear()
            self.nodes.remove(node)

    def connect(self, from_node: 'Node', to_node: 'Node'):
        assert(len(from_node.elements) == 1)
        element = from_node.elements[0]
        if(from_node == element.high):
            self.remove_node(element.high)
            element.high = to_node
            element.high.add_element(element)
        elif(from_node == element.low):
            self.remove_node(element.low)
            element.low = to_node
            element.low.add_element(element)
        else:
            assert()

    def num_nodes(self):
        return len(self.nodes)

    def num_elements(self):
        return len(self.elements)
    
    def node_idx(self, node: 'Node'):
        return self.nodes.index(node)

    def draw(self):
        nx.draw(self.nx_graph(), with_labels = True)

    def nx_graph(self):
        graph = nx.MultiDiGraph()
        for element in self.elements:
            element = element.to_nx()
            graph.add_edges_from([element])
        return graph

    def M(self,dtype=torch.float) -> Tensor:
        M_scipy = nx.incidence_matrix(G=self.nx_graph(),oriented=True)
        M_numpy = M_scipy.toarray()
        M_tensor = torch.tensor(M_numpy,dtype=dtype)
        return M_tensor

    def __repr__(self) -> str:
        return "Circuit with " + str(len(self.nodes)) + \
                " nodes and "+ str(len(self.elements)) + " elements"

    def parallel_elements(self, reference_element:'Element'):
        parallels = []
        for high_element in reference_element.high.elements:
            for low_element in reference_element.low.elements:
                if(high_element == low_element):
                    parallels.append(low_element)
        return parallels

    def extract_elements(self):
        '''
        return dictinaries of circuit data and other useful precomputed lists
        '''
        kinds_map: dict[Kinds,list[bool]] = {}
        props_map: dict[Props,list[float]] = {}
        attributes_map: dict[Kinds,list[float]] = {}
        for kind in Kinds:
            kinds_map[kind] = []
            attributes_map[kind] = []
        for prop in Props:
            props_map[prop] = []
        for e in range(len(self.elements)):
            element = self.elements[e]
            for kind in Kinds:
                if(element.kind == kind):
                    kinds_map[kind].append(True)
                    attributes_map[kind].append(element.attr)
                else:
                    kinds_map[kind].append(False)
                    attributes_map[kind].append(None)
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
                else:
                    assert()
                if(value == None):
                    props_map[prop].append(None)
                else:
                    props_map[prop].append(float(value))
        elements = {
            'kinds': kinds_map,
            'properties': props_map,
            'attributes': attributes_map,
        }
        return elements
    
    def ring(self, source_kind:Kinds, load_kind:Kinds, num_loads:int) -> 'Circuit':
        '''one source and all loads in parallel'''
        assert(num_loads > 0)
        self.clear()
        source = Element(self,source_kind)
        self.add_element(source)
        first_load = Element(self,load_kind)
        self.add_element(first_load)
        self.connect(source.high, first_load.high)
        prev_element = first_load
        for l in range(num_loads-1):
            new_load = Element(circuit=self, kind=load_kind)
            self.add_element(new_load)
            self.connect(prev_element.low, new_load.high)
            prev_element = new_load
        self.connect(source.low, prev_element.low)

class Element():
    def __init__(self, circuit: Circuit, kind:Kinds) -> None:
        assert(isinstance(kind,Kinds))
        self.circuit = circuit
        self.low:Node = None
        self.high:Node = None
        self.kind = kind
        self.i:float = None
        self.v:float = None
        self.attr:float = None

    def __repr__(self) -> str:
        return "("+str(self.kind.name)+", "+str(self.low.idx)+ ", "\
                    +str(self.high.idx)+")"

    def to_nx(self):
        kind = ('kind',self.kind)
        v = ('v',self.v)
        i = ('i',self.i)
        attr = ('attr',self.attr)
        return (self.low.idx, self.high.idx, self.key, (kind, i, v, attr))

    @property
    def key(self):
        parallels = self.circuit.parallel_elements(self)
        return parallels.index(self)

    def clear(self):
        self.circuit = None
        self.low = None
        self.high = None
        self.kind = None
        self.i = None
        self.v = None
        self.attr = None
    
class Node():
    def __init__(self, circuit: Circuit, elements: list[Element]) -> None:
        self.circuit = circuit
        self.elements = elements
        self.potential:float = None
        assert(self.circuit != None)
        assert(self.elements != None)
        assert(len(self.elements) != 0)

    def __repr__(self) -> str:
        return str(self.idx)

    def to_nx(self):
        v = {'v':self.potential}
        return (self.idx, v)

    @property
    def idx(self):
        return self.circuit.node_idx(self)

    def clear(self):
        self.circuit = None
        self.elements = None
        self.potential = None

    def add_element(self, element: Element):
        if(element not in self.elements):
            self.elements.append(element)