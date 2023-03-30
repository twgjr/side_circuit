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

class Circuit():
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.elements: list[Element] = []
        self.signal_len = 0

    def clear(self):
        for node in self.nodes:
            node.delete()
        for element in self.elements:
            element.delete()
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
        assert element in self.elements
        self.elements.remove(element)
        for node in self.nodes:
            if(node.elements == []):
                self.remove_node(node)
        if(len(self.elements)==0):
            self.nodes = []
        elif(merge_nodes):
            self.connect(element.high,element.low)
        element.delete()
    
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
        assert node in self.nodes
        self.nodes.remove(node)
        node.delete()

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

    def kvl_coef(self) -> list[list[int]]:
        '''Returns a list of lists of integers. Each list of integers represents
        the coefficients of the KVL equations for a loop in the circuit.'''
        loops = self.loops()
        kvl_coefficients = []
        for loop in loops:
            coefficients = [0]*self.num_elements()
            first_element = loop[0]
            from_element = first_element
            from_node = first_element.low
            coefficients[self.element_idx(from_element)] = 1
            next_element = self.next_loop_element(loop, from_element, from_node)
            while(next_element != first_element):
                from_node,polarity = self.next_node_in(next_element, from_node)
                coefficients[self.element_idx(next_element)] = polarity
                from_element = next_element
                next_element = self.next_loop_element(loop, from_element, from_node)
            kvl_coefficients.append(coefficients)
        return kvl_coefficients
    
    def next_series_element(self, from_element:'Element', 
                            shared_node:'Node') -> 'Element':
        to_elements = shared_node.elements
        assert(len(to_elements) == 2)
        for element in to_elements:
            if(element != from_element):
                return element
            
    def next_loop_element(self, loop:list['Element'], from_element: 'Element',
                        from_node:'Node') -> 'Element':
        '''Returns the next element in the loop after active_element.  Exit from
        the exit node.'''
        for element in from_node.elements:
            if(element == from_element):
                continue
            elif(element in loop):
                return element

    def next_node_in(self, element:'Element', from_node:'Node') -> tuple['Node',int]:
        '''Returns the next node in the loop after active_node.  Exit from
        the exit node.'''
        if(from_node == element.high):
            return element.low, 1
        elif(from_node == element.low):
            return element.high, -1
        else:
            assert()
    
    def __repr__(self) -> str:
        return "Circuit with " + str(len(self.nodes)) + \
                " nodes and "+ str(len(self.elements)) + " elements"

    def elements_parallel_to(self, reference_element:'Element',
                             include_ref:bool)->list['Element']:
        parallels = []
        for high_element in reference_element.high.elements:
            for low_element in reference_element.low.elements:
                if(high_element == low_element):
                    if(low_element != reference_element or include_ref):
                        parallels.append(low_element)
                    break
        return parallels
    
    def elements_in_series_with(self, reference_element:'Element',
                                include_ref:bool)->list['Element']:
        series = []
        if(include_ref):
            series.append(reference_element)
        for ref_node in [reference_element.low, reference_element.high]:
            node = ref_node
            element = reference_element
            while(len(node.elements) == 2):
                node,_ = self.next_node_in(element, node)
                element = self.next_series_element(element, node)
                if(element == reference_element):
                    return series
                else:
                    series.append(element)
        return series

    def load(self, i_tensor:Tensor, v_tensor:Tensor, attr_tensor:Tensor):
        '''Takes inputs i_sol, v_sol, a_sol from solver and loads them into the circuit'''
        attr_list = attr_tensor.tolist()
        i_list = i_tensor.tolist()
        v_list = v_tensor.tolist()
        assert(len(attr_list) == len(i_list) == len(v_list))
        for e in range(len(self.elements)):
            element = self.elements[e]
            element.a_pred = attr_list[e]
            element.i_pred = i_list[e]
            element.v_pred = v_list[e]

    def export(self):
        '''
        return dictinaries of circuit data and other useful precomputed lists
        '''
        kinds_map: dict[Kinds,list[bool]] = {}
        props_map: dict[Props,list[Signal]] = {}
        attributes_map: dict[Kinds,list[float]] = {}
        for kind in Kinds:
            kinds_map[kind] = []
            attributes_map[kind] = []
        for prop in Props:
            props_map[prop] = []
        for element in self.elements:
            for kind in Kinds:
                if(element.kind == kind):
                    kinds_map[kind].append(True)
                    attributes_map[kind].append(element.a)
                else:
                    kinds_map[kind].append(False)
                    attributes_map[kind].append(None)
            for prop in Props:
                if(prop == Props.I):
                    props_map[prop].append(element.i)
                elif(prop == Props.V):
                    props_map[prop].append(element.v)
                else:
                    assert()

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

    def update_signal_len(self, signal_len:int):
        if(signal_len == 0):
            return
        max_sig_len = 0
        for element in self.elements:
            i_len = len(element.i.data)
            v_len = len(element.v.data)
            max_sig_len = max(max_sig_len, i_len, v_len)
        self.signal_len = signal_len

class Element():
    def __init__(self, circuit: Circuit, kind:Kinds) -> None:
        assert(isinstance(kind,Kinds))
        self.circuit = circuit
        self.low:Node = None
        self.high:Node = None
        self.kind = kind
        self._i:Signal = Signal(element=self)
        self._v:Signal = Signal(element=self)
        self._a:Signal = None
        self.i_pred:Signal = None
        self.v_pred:Signal = None
        self.a_pred:Signal = None

    def __repr__(self) -> str:
        return "("+str(self.kind.name)+", "+str(self.low.idx)+ ", "\
                    +str(self.high.idx)+")"

    def to_nx(self):
        kind = ('kind',self.kind)
        v = ('v',self.v)
        i = ('i',self.i)
        attr = None
        if(self.kind == Kinds.ICS or self.kind == Kinds.IVS):
            attr = ('attr',None)
        else:
            attr = ('attr',self.a)
        return (self.low.idx, self.high.idx, self.key, (kind, i, v, attr))
    
    @property 
    def i(self):
        return self._i
    
    @i.setter
    def i(self, value):
        if(value == None):
            self._i = value
            return
        assert isinstance(value,Signal)
        series = self.circuit.elements_in_series_with(self,False)
        i_defined = False
        for element in series:
            if(not element.i.is_empty()):
                i_defined = True
        if(i_defined):
            assert()

    @property
    def v(self):
        return self._v
    
    @v.setter
    def v(self, value):
        if(value == None):
            self._v = value
            return
        assert isinstance(value,Signal)
        parallels = self.circuit.elements_parallel_to(self,False)
        v_defined = False
        for element in parallels:
            if(not element.v.is_empty()):
                v_defined = True
        if(v_defined):
            assert()

    @property
    def a(self):
        return self._a
    
    @a.setter
    def a(self, value):
        assert self.kind != Kinds.ICS and self.kind != Kinds.IVS
        assert value == None or isinstance(value,float)
        self._a = value

    @property
    def key(self):
        parallels = self.circuit.elements_parallel_to(self,True)
        return parallels.index(self)

    def delete(self):
        self._i.clear()
        self._i = None
        self._v.clear()
        self._v = None
        self._a = None
        self.i_pred = None
        self.v_pred = None
        self.a_pred = None
        self.low.remove_element(self)
        self.low = None
        self.high.remove_element(self)
        self.high = None
        self.kind = None
        self.circuit = None
    
class Node():
    def __init__(self, circuit: Circuit, elements: list[Element]) -> None:
        self.circuit = circuit
        self.elements = elements

    def __repr__(self) -> str:
        return str(self.idx)
    
    def num_elements(self):
        return len(self.elements)

    @property
    def idx(self):
        return self.circuit.node_idx(self)

    def delete(self):
        self.circuit = None
        self.remove_self_from_elements()
        self.elements = None

    def remove_self_from_elements(self):
        for element in self.elements:
            if(element.low == self):
                element.low = None
            if(element.high == self):
                element.high = None

    def add_element(self, element: Element):
        if(element not in self.elements):
            self.elements.append(element)

    def remove_element(self, element: Element):
        if(element in self.elements):
            self.elements.remove(element)

class Signal():
    def __init__(self, element: Element = None, data:list = []) -> None:
        self.element = element
        self._data = data
        if(element != None):
            self.element.circuit.update_signal_len(len(self._data))

    def __repr__(self) -> str:
        return str(self._data)

    def clear(self):
        self._data = []
        self.element = None

    def is_empty(self):
        return len(self._data) == 0
    
    def copy(self):
        assert isinstance(self.element,Element)
        new_signal = Signal(self.element)
        new_signal.data = self.data.copy()
        return new_signal
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value:list):
        assert isinstance(value,list)
        if(self.element.a == self):
            if(self.element.kind == Kinds.ICS or self.element.kind == Kinds.IVS):
                assert()
        save_data = self._data
        self._data = value
        self.element.circuit.update_signal_len(len(self.data))
        data_len = len(self._data)
        ckt_sig_len = self.element.circuit.signal_len
        if(data_len != 0 and ckt_sig_len != 0):
            if(data_len != ckt_sig_len):
                self._data = save_data
                self.element.circuit.update_signal_len(len(self.data))
                assert()