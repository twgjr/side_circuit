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

    def load(self, i_tensor_list:list[Tensor], v_tensor_list:list[Tensor],
              attr_tensor:Tensor):
        '''Stores predictions from Trainer in Circuit'''
        assert(attr_tensor.shape[0] == self.num_elements())
        assert(len(i_tensor_list) == len(v_tensor_list) == self.signal_len)
        for e in range(len(self.elements)):
            element = self.elements[e]
            element.a_pred = attr_tensor[e].item()
            for t in range(self.signal_len):
                i_tensor = i_tensor_list[t]
                i_pred = i_tensor[e,:].item()
                element.i_pred.append(i_pred)
                v_tensor = v_tensor_list[t]
                v_pred = v_tensor[e,:].item()
                element.v_pred.append(v_pred)

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
            i_len = len(element.i)
            v_len = len(element.v)
            max_sig_len = max(max_sig_len, i_len, v_len)
        self.signal_len = signal_len

class Element():
    def __init__(self, circuit: Circuit, kind:Kinds) -> None:
        assert(isinstance(kind,Kinds) and isinstance(circuit,Circuit))
        self.circuit = circuit
        self.low:Node = None
        self.high:Node = None
        self.kind = kind
        self._i:Signal = Signal(self,[])
        self._v:Signal = Signal(self,[])
        self._a:Signal = None
        self._i_pred:Signal = Signal(self,[])
        self._v_pred:Signal = Signal(self,[])
        self._a_pred:Signal = None

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
    def i(self, values:list):
        assert isinstance(values,list)
        series = self.circuit.elements_in_series_with(self,False)
        for element in series:
            if(not element.i.is_empty()):
                assert()
        self.set_signal_data(values, self._i)

    @property
    def v(self):
        return self._v
    
    @v.setter
    def v(self, values:list):
        assert isinstance(values,list)
        parallels = self.circuit.elements_parallel_to(self,False)
        for element in parallels:
            if(not element.v.is_empty()):
                assert()
        self.set_signal_data(values, self._v)

    @property
    def a(self):
        return self._a
    
    @a.setter
    def a(self, value):
        assert self.kind != Kinds.ICS and self.kind != Kinds.IVS
        assert value == None or isinstance(value,float)
        self._a = value

    @property
    def i_pred(self):
        return self._i_pred
    
    @i_pred.setter
    def i_pred(self, values:list):
        assert isinstance(values,list)
        assert values != self._v_pred.get_data()
        self.set_signal_data(values, self._i_pred)

    @property
    def v_pred(self):
        return self._v_pred
    
    @v_pred.setter
    def v_pred(self, values:list):
        assert isinstance(values,list)
        assert values != self._i_pred.get_data()
        self.set_signal_data(values, self._v_pred)

    @property
    def a_pred(self):
        return self._a_pred
    
    @a_pred.setter
    def a_pred(self, value:float):
        assert isinstance(value,float)
        self._a_pred = value

    def set_signal_data(self, value:list, signal:'Signal'):
        assert isinstance(value,list)
        if(self.a == self):
            if(self.kind == Kinds.ICS or self.kind == Kinds.IVS):
                assert()
        signal.set_data(value)
        data_len = len(signal)
        self.circuit.update_signal_len(data_len)

    @property
    def key(self):
        parallels = self.circuit.elements_parallel_to(self,True)
        return parallels.index(self)

    def delete(self):
        self._i.prep_delete()
        self._i = None
        self._v.prep_delete()
        self._v = None
        self._a = None
        self._i_pred.prep_delete()
        self._i_pred = None
        self._v_pred.prep_delete()
        self._v_pred = None
        self._a_pred = None
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
    def __init__(self, element: Element,data) -> None:
        assert isinstance(element,Element) or element == None
        assert isinstance(data,list)
        self.element = element
        self._data = data
        if(self.element != None):
            self.element.circuit.update_signal_len(len(self._data))

    def get_data(self):
        return self._data
    
    def set_data(self, value:list):
        assert isinstance(value,list)
        self._data = value

    def __repr__(self) -> str:
        return str(self._data)

    def prep_delete(self):
        self._data = []
        self.element = None
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value

    def __eq__(self, other:'Signal') -> bool:
        assert isinstance(other,Signal)
        if(len(self) != len(other)):
            return False
        for i in range(len(self)):
            if(self[i] != other[i]):
                return False
        return True
    
    def __neg__(self):
        data = []
        for item in self._data:
            data.append(-item)
        return Signal(element=self.element, data=data)
    
    def clear(self):
        self._data = []
    
    def append(self, value:float):
        assert isinstance(value,float)
        self._data.append(value)

    def is_empty(self):
        return len(self._data) == 0
    
    def copy(self):
        assert isinstance(self.element,Element)
        data_copy = []
        for item in self._data:
            data_copy.append(item)
        return Signal(element=self.element, data=data_copy)