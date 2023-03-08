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
    
    def add_element(self, kind:Kinds) -> 'Element':
        assert(isinstance(kind,Kinds))
        element = Element(self,kind)
        high_node = self.add_node([element])
        low_node = self.add_node([element])
        element.high = high_node
        element.low = low_node
        self.elements.append(element)
        return element
    
    def remove_element(self, element: 'Element', merge_nodes:bool):
        if(element in self.elements):
            if(element in element.high.elements):
                self.remove_node(element.high)
            if(element in element.low.elements):
                self.remove_node(element.low)
            if(merge_nodes):
                self.connect(element.high,element.low)
            element.clear()
            self.elements.remove(element)
    
    def add_node(self, elements:list['Element']) -> 'Node':
        '''create a node for a new element and add to circuit.
            nodes are never created without an element. No floating nodes'''
        assert(isinstance(elements,list) or elements == None)
        if(elements == None):
            elements = []
        for element in elements:
            assert(isinstance(element,Element))
        ckt_node = Node(self,elements)
        self.nodes.append(ckt_node)
        return ckt_node

    def remove_node(self, node: 'Node'):
        if(node in self.nodes):
            self.nodes.remove(node)
            node.clear()

    def connect(self, from_node: 'Node', to_node: 'Node'):
        common_node = self.add_node([])
        for element in from_node.elements:
            common_node.add_element(element)
            if(from_node == element.high):
                element.high = common_node
            elif(from_node == element.low):
                element.low = common_node
            else:
                assert()
        for element in to_node.elements:
            common_node.add_element(element)
            if(to_node == element.high):
                element.high = common_node
            elif(to_node == element.low):
                element.low = common_node
            else:
                assert()
        self.remove_node(from_node)
        self.remove_node(to_node)

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

    def M(self,dtype=torch.float) -> Tensor:
        M_scipy = nx.incidence_matrix(G=self.nx_graph(),oriented=True)
        M_numpy = M_scipy.toarray()
        M_tensor = torch.tensor(M_numpy,dtype=dtype)
        return M_tensor
    
    def spanning_tree(self) -> list['Element']:
        '''Simple minimum spanning tree algorithm that returns list of elements
        in the minimum spanning tree.'''
        unvisited_nodes = self.nodes.copy()
        st_elements = []
        active_node = unvisited_nodes[-1]
        while(len(unvisited_nodes) > 0):
            for element in active_node.elements:
                if(element not in st_elements):
                    if(active_node == element.high):
                        matches = self.intersection(element.low.elements,
                                                    st_elements)
                        if(len(matches) == 0):
                            st_elements.append(element)
                    elif(active_node == element.low):
                        matches = self.intersection(element.high.elements,
                                                    st_elements)
                        if(len(matches) == 0):
                            st_elements.append(element)
            unvisited_nodes.pop()
            if(len(unvisited_nodes) > 0):
                active_node = unvisited_nodes[-1]
        return st_elements

    def intersection(self, list1, list2):
        return list(set(list1) & set(list2))
    
    def loops(self) -> list[list['Element']]:
        '''Returns a list of lists of elements. Each list of elements is part of 
        a loop in the circuit.  Each list of elements also contains elements 
        from the spanning tree that are not part of the loop.'''
        mst_elements = self.spanning_tree()
        non_st_elements = []
        for element in self.elements:
            if(element not in mst_elements):
                non_st_elements.append(element)
        loops_with_acyclics = []
        for key_element in non_st_elements:
            loop_with_acyclics = mst_elements.copy()
            loop_with_acyclics.append(key_element)
            loops_with_acyclics.append(loop_with_acyclics)
        return loops_with_acyclics

    def kvl_coefficients(self) -> list[list[int]]:
        '''Returns a list of lists of integers. Each list of integers represents
        the coefficients of the KVL equations for a loop in the circuit.'''
        loops = self.loops()
        kvl_coefficients = []
        for loop in loops:
            coefficients = [0]*self.num_elements()
            first_element = loop[0]
            active_element = first_element
            active_polarity = 1
            coefficients[self.element_idx(active_element)] = active_polarity
            next_element = self.next_in_loop(loop, active_element)
            while(next_element != first_element):
                if(next_element.high == active_element.low):
                    active_polarity = 1
                elif(next_element.low == active_element.low):
                    active_polarity = -1
                else:
                    assert()
                coefficients[self.element_idx(next_element)] = active_polarity
                active_element = next_element
                next_element = self.next_in_loop(loop, active_element)
            kvl_coefficients.append(coefficients)
        return kvl_coefficients
            
    def next_in_loop(self, loop:list['Element'], active_element:'Element') -> 'Element':
        '''Returns the next element in the loop after active_element.'''
        for low_element in active_element.low.elements:
            if(low_element == active_element):
                continue
            elif(low_element in loop):
                return low_element

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

    def load(self, i_tensor:Tensor, v_tensor:Tensor, attr_tensor:Tensor):
        '''Takes inputs i_sol, v_sol, a_sol from solver and loads them into the circuit'''
        attr_list = attr_tensor.tolist()
        i_list = i_tensor.squeeze(1).tolist()
        v_list = v_tensor.squeeze(1).tolist()
        assert(len(attr_list) == len(i_list) == len(v_list))
        for e in range(len(self.elements)):
            element = self.elements[e]
            element.attr = attr_list[e]
            element.i = i_list[e]
            element.v = v_list[e]

    def export(self):
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
        '''one source and all loads in series'''
        assert(num_loads > 0)
        self.clear()
        source = self.add_element(source_kind)
        first_load = self.add_element(load_kind)
        self.connect(source.high, first_load.high)
        prev_element = first_load
        for l in range(num_loads-1):
            new_load = self.add_element(load_kind)
            self.connect(prev_element.low, new_load.high)
            prev_element = new_load
        self.connect(source.low, prev_element.low)

    def ladder(self, source_kind:Kinds, load_kind:Kinds, num_loads:int) -> 'Circuit':
        '''one source and all loads in parallel'''
        assert(num_loads > 0)
        self.clear()
        source = self.add_element(source_kind)
        first_load = self.add_element(load_kind)
        self.connect(source.high, first_load.high)
        self.connect(source.low, first_load.low)
        prev_element = first_load
        for l in range(num_loads-1):
            new_load = self.add_element(load_kind)
            self.connect(prev_element.high, new_load.high)
            self.connect(prev_element.low, new_load.low)
            prev_element = new_load

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

    def disconnect_low(self):
        self.low.remove_element(self)
        self.low = Node(self.circuit, [self])

    def disconnect_high(self):
        self.high.remove_element(self)
        self.high = Node(self.circuit, [self])
    
class Node():
    def __init__(self, circuit: Circuit, elements: list[Element]) -> None:
        self.circuit = circuit
        self.elements = elements
        self.potential:float = None

    def __repr__(self) -> str:
        return str(self.idx)

    def to_nx(self):
        v = {'v':self.potential}
        return (self.idx, v)
    
    def num_elements(self):
        return len(self.elements)

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

    def remove_element(self, element: Element):
        if(element in self.elements):
            self.elements.remove(element)