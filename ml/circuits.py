import networkx as nx
from enum import Enum
import torch
from torch import Tensor

class Kinds(Enum):
    IVS = 0
    ICS = 1
    R = 2
    VC = 3
    CC = 4
    SW = 5

class Props(Enum):
    I = 0
    V = 1

class System():
    '''Collection of isolated Circuits that are only connected by parent/child
    relationships between Elements. When an Element is added to the System,
    it is added to its own new Circuit.  When two Elements are connected, the
    Circuits are merged, retaining the larger Circuit of the two connecting
    Elements.'''
    def __init__(self) -> None:
        self.circuits: dict[int,Circuit] = {}

    def num_circuits(self) -> int:
        return len(self.circuits)
    
    def num_elements(self) -> int:
        return sum([self.circuits[c].num_elements() for c in self.circuits])
        
    def prep_for_delete(self):
        for c in self.circuits:
            self.circuits[c].prep_for_delete()
        self.circuits.clear()

    def add_circuit(self) -> 'Circuit':
        circuit = Circuit(self)
        self.circuits[len(self.circuits)] = circuit
        return circuit
    
    def remove_circuit(self, circuit: 'Circuit'):
        circuit.prep_for_delete()
        del self.circuits[circuit.index]
        for c in self.circuits:
            if(self.circuits[c] == circuit):
                del self.circuits[c]
                break
        self.renumber_circuits()
            
    def renumber_circuits(self):
        new_circuits = {}
        for c in self.circuits:
            new_circuits[len(new_circuits)] = self.circuits[c]
        self.circuits = new_circuits
    
    def add_element_of(self, kind:Kinds) -> 'Element':
        circuit = self.add_circuit()
        return circuit.add_element_of(kind)

    def add_ctl_element(self, parent_kind:Kinds, 
                        child_kind:Kinds) -> tuple['Element','Element']:
        assert(parent_kind == Kinds.VC)
        assert(child_kind == Kinds.SW)
        parent = self.add_element_of(parent_kind)
        child = self.add_element_of(child_kind)
        child.parent = parent
        parent.child = child
        return parent, child
    
    def connect(self, from_node: 'Node', to_node: 'Node'):
        '''Merge two nodes together.  If the nodes are in different circuits, 
        then the elements of the smaller circuit are moved to the larger circuit.
        if the nodes are in the same circuit, then the circuit is unchanged.'''
        assert(isinstance(from_node,Node))
        assert(isinstance(to_node,Node))
        assert(from_node.circuit != None)
        assert(to_node.circuit != None)
        if(from_node.circuit == to_node.circuit):
            from_node.circuit.connect(from_node,to_node)
        else:
            if(from_node.circuit.num_elements() > 
               to_node.circuit.num_elements()):
                # "from" circuit is larger, move "to" circuit to "from"
                large_circuit = from_node.circuit
                small_circuit = to_node.circuit
            else:
                # "to" circuit is equal or larger, move "from" circuit to "to"
                large_circuit = to_node.circuit
                small_circuit = from_node.circuit
            self.transfer_circuit(small_circuit,large_circuit)
            large_circuit.connect(from_node,to_node)
            self.renumber_circuits()
            

    def transfer_circuit(self, small: 'Circuit', large: 'Circuit'):
        for element in small.elements.values():
            element.circuit = large
            large.elements[len(large.elements)] = element
        for node in small.nodes.values():
            node.circuit = large
            large.nodes[len(large.nodes)] = node
        small.elements.clear()
        small.nodes.clear()
        del self.circuits[small.index]

    def flatten_elements(self) -> list['Element']:
        '''return a list of all elements in the system, ordered by circuit'''
        elements = []
        for circuit in self.circuits.values():
            elements += list(circuit.elements.values())
        return elements
    
    def parent_indices(self, kind:Kinds) -> list[int]:
        '''return a list of the indices of elements in the system.  If an 
        element has a parent, then the index of the parent is returned.  If
        an element has no parent, then the index of the element itself is
        returned.'''
        elements = self.flatten_elements()
        control_list = []
        for element in elements:
            if(element.parent == kind):
                control_list.append(elements.index(element.parent))
            else:
                control_list.append(elements.index(element))
        return control_list
    
    def control_mask(self) -> list[bool]:
        '''True if the element has a control, False otherwise.'''
        elements = self.flatten_elements()
        control_mask_list = []
        for element in elements:
            control_mask_list.append(element.parent != None)
        return control_mask_list

    def switched_resistor(self) -> tuple['Element', 'Element', 'Element', 
                                         'Element', 'Element', 'Element']:
        '''one source and one load, with a switch in series. The switch control
        has a separate voltage source  and resistor in parallel.'''
        self.prep_for_delete()
        child_src = self.add_element_of(Kinds.IVS)
        parent_src = self.add_element_of(Kinds.IVS)
        child_res = self.add_element_of(Kinds.R)
        parent_res = self.add_element_of(Kinds.R)
        parent, child = self.add_ctl_element(Kinds.VC, Kinds.SW)
        self.connect(child_src.high, child.high)
        self.connect(child.low, child_res.high)
        self.connect(child_src.low, child_res.low)
        self.connect(parent_src.high, parent_res.high)
        self.connect(parent_res.high, parent.high)
        self.connect(parent_src.low, parent_res.low)
        self.connect(parent_res.low, parent.low)
        return child_src, parent_src, child_res, parent_res, parent, child
    
    def ring(self, source_kind:Kinds, load_kind:Kinds, num_loads:int) -> 'Circuit':
        '''one source and all loads in series'''
        assert(num_loads > 0)
        self.prep_for_delete()
        circuit = self.add_circuit()
        source = circuit.add_element_of(source_kind)
        first_load = circuit.add_element_of(load_kind)
        circuit.connect(source.high, first_load.high)
        prev_element = first_load
        for l in range(num_loads-1):
            new_load = circuit.add_element_of(load_kind)
            circuit.connect(prev_element.low, new_load.high)
            prev_element = new_load
        circuit.connect(source.low, prev_element.low)
        return circuit

    def ladder(self, source_kind:Kinds, load_kind:Kinds, num_loads:int) -> 'Circuit':
        '''one source and all loads in parallel'''
        assert(num_loads > 0)
        self.prep_for_delete()
        circuit = self.add_circuit()
        source = circuit.add_element_of(source_kind)
        first_load = circuit.add_element_of(load_kind)
        circuit.connect(source.high, first_load.high)
        circuit.connect(source.low, first_load.low)
        prev_element = first_load
        for l in range(num_loads-1):
            new_load = circuit.add_element_of(load_kind)
            circuit.connect(prev_element.high, new_load.high)
            circuit.connect(prev_element.low, new_load.low)
            prev_element = new_load
        return circuit

class Circuit():
    def __init__(self, system:System) -> None:
        self.system = system
        self.nodes: dict[int,Node] = {}
        self.elements: dict[int,Element] = {}
        self.signal_len = 0

    @property
    def index(self) -> int:
        key_list = list(self.system.circuits.keys())
        val_list = list(self.system.circuits.values())
        position = val_list.index(self)
        return key_list[position]

    def prep_for_delete(self):
        for e in self.elements:
            self.elements[e].prep_for_delete()
        for n in self.nodes:
            self.nodes[n].prep_for_delete()
        self.clear()

    def clear(self):
        self.nodes.clear()
        self.elements.clear()
    
    def add_element_of(self, kind:Kinds) -> 'Element':
        assert(isinstance(kind,Kinds))
        element = Element(self,kind)
        element.high = self.add_node([element])
        element.low = self.add_node([element])
        self.elements[len(self.elements)] = element
        return element
    
    def remove_element(self, element: 'Element'):
        element.prep_for_delete()
        del self.elements[element.index]
        self.renumber_elements()
            
    def renumber_elements(self):
        new_elements = {}
        for e in self.elements:
            new_elements[len(new_elements)] = self.elements[e]
        self.elements = new_elements
    
    def add_node(self, elements:list['Element']) -> 'Node':
        '''create a node for a new element and add to circuit.
            nodes are never created without an element. No floating nodes'''
        assert(isinstance(elements,list))
        ckt_node = Node(self,elements)
        self.nodes[len(self.nodes)] = ckt_node
        return ckt_node

    def remove_node(self, node: 'Node'):
        del self.nodes[node.index]
        node.prep_for_delete()
        self.renumber_nodes()

    def renumber_nodes(self):
        new_nodes = {}
        for n in self.nodes:
            new_nodes[len(new_nodes)] = self.nodes[n]
        self.nodes = new_nodes

    def transfer_elements(self, from_node: 'Node', to_node: 'Node'):
        '''transfer all elements from one node to another.  This is used
        when a node is deleted and its elements are transferred to another
        node.'''
        for element in from_node.elements:
            to_node.add_element(element)
            if(from_node == element.high):
                element.high = to_node
            elif(from_node == element.low):
                element.low = to_node
            else:
                assert()
        self.remove_node(from_node)

    def connect(self, a: 'Node', b: 'Node'):
        c = self.add_node([])
        self.transfer_elements(a, c)
        self.transfer_elements(b, c)
        self.renumber_elements()
        self.renumber_nodes()
        return c

    def num_nodes(self):
        return len(self.nodes)

    def num_elements(self):
        return len(self.elements)

    def draw(self):
        nx.draw(self.nx_graph(), with_labels = True)

    def nx_graph(self):
        graph = nx.MultiDiGraph()
        for element in self.elements.values():
            element = element.to_nx()
            graph.add_edges_from([element])
        return graph

    def M(self,dtype=torch.float) -> Tensor:
        M_scipy = nx.incidence_matrix(G=self.nx_graph(),oriented=True)
        M_numpy = M_scipy.toarray()
        M_tensor = torch.tensor(M_numpy,dtype=dtype)
        return M_tensor
    
    def spanning_tree(self) -> tuple[list['Element'],list['Element'],
                                      list['Node']]:
        '''Returns a spanning tree in the form of a list of spanning tree 
        elements, missing elements, and leaves.'''
        active = self.nodes[0]
        pending: list[Node] = [active]
        st:list[Element] = []
        missing:list[Element] = []
        leaves:list[Node] = []
        while(len(pending) > 0):
            pending.pop()
            num_added = 0
            for element in active.elements:
                adjacent_node, polarity = self.next_node_in(element,active)
                if(element not in st):
                    if(adjacent_node in pending):
                        missing.append(element)
                    else:
                        if(element not in missing):
                            st.append(element)
                            num_added += 1
                            pending.append(adjacent_node)
            if(num_added == 0):
                leaves.append(active)
            if(len(pending) > 0):
                active = pending[-1]
        return st, missing, leaves
    
    def loops(self) -> list[list['Element']]:
        '''Returns a list of lists of elements. Each list of elements is part of 
        a loop in the circuit.  Each list of elements also contains elements 
        from the spanning tree that are not part of the loop.'''
        st, missing, leaves = self.spanning_tree()
        st_one_added_list = []
        for missing_element in missing:
            st_one_added = st.copy()
            st_one_added.append(missing_element)
            st_one_added_list.append(st_one_added)
        return st_one_added_list

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
            coefficients[from_element.index] = 1
            next_element = self.next_loop_element(loop, from_element, from_node)
            while(next_element != first_element):
                from_node,polarity = self.next_node_in(next_element, from_node)
                coefficients[next_element.index] = polarity
                from_element = next_element
                next_element = self.next_loop_element(loop, from_element, from_node)
            kvl_coefficients.append(coefficients)
        return kvl_coefficients
            
    def next_loop_element(self, loop:list['Element'], from_element: 'Element',
                        from_node:'Node') -> 'Element':
        '''Returns the next element in the loop after active_element.  Exit from
        the exit node.'''
        for element in from_node.elements:
            if(element == from_element):
                continue
            elif(element in loop):
                return element

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
    
    def next_node_in(self, element:'Element', from_node:'Node') -> tuple['Node',int]:
        '''Returns the next node of the element opposite of the from_node.  Also
        returns the polarity of the element with respect to the from_node.'''
        if(from_node == element.high):
            return element.low, 1
        elif(from_node == element.low):
            return element.high, -1
        else:
            assert()

    def next_series_element(self, from_element:'Element', 
                            shared_node:'Node') -> 'Element':
        to_elements = shared_node.elements
        assert(len(to_elements) == 2)
        for element in to_elements:
            if(element != from_element):
                return element

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
    
    def kind_list(self, kind:Kinds, with_control:Kinds=None) -> list[bool]:
        '''returns a list of booleans indicating which elements are of kind'''
        assert(kind != with_control)
        assert(with_control == Kinds.VC or with_control == Kinds.CC or 
               with_control == None)
        kind_list = []
        for element in self.elements.values():
            if(element.kind == kind):
                if(with_control == None):
                    kind_list.append(True)
                elif(element.has_parent_of(with_control)):
                    kind_list.append(True)
                else:
                    kind_list.append(False)
            else:
                kind_list.append(False)
        return kind_list
    
    def attr_list(self, kind:Kinds) -> list[float]:
        '''returns a list of attributes of elements of kind'''
        attrs = []
        for element in self.elements.values():
            if(element.kind == kind):
                attrs.append(element.a)
            else:
                attrs.append(None)
        return attrs
    
    def prop_list(self, prop:Props) -> list['Signal']:
        '''returns a list of properties matching prop of elements'''
        props = []
        for element in self.elements.values():
            if(prop == Props.I):
                props.append(element.i.copy())
            elif(prop == Props.V):
                props.append(element.v.copy())
            else:
                assert()
        return props

    # def ring(self, source_kind:Kinds, load_kind:Kinds, num_loads:int) -> 'Circuit':
    #     '''one source and all loads in series'''
    #     assert(num_loads > 0)
    #     self.prep_for_delete()
    #     source = self.add_element_of(source_kind)
    #     first_load = self.add_element_of(load_kind)
    #     self.connect(source.high, first_load.high)
    #     prev_element = first_load
    #     for l in range(num_loads-1):
    #         new_load = self.add_element_of(load_kind)
    #         self.connect(prev_element.low, new_load.high)
    #         prev_element = new_load
    #     self.connect(source.low, prev_element.low)

    # def ladder(self, source_kind:Kinds, load_kind:Kinds, num_loads:int) -> 'Circuit':
    #     '''one source and all loads in parallel'''
    #     assert(num_loads > 0)
    #     self.prep_for_delete()
    #     source = self.add_element_of(source_kind)
    #     first_load = self.add_element_of(load_kind)
    #     self.connect(source.high, first_load.high)
    #     self.connect(source.low, first_load.low)
    #     prev_element = first_load
    #     for l in range(num_loads-1):
    #         new_load = self.add_element_of(load_kind)
    #         self.connect(prev_element.high, new_load.high)
    #         self.connect(prev_element.low, new_load.low)
    #         prev_element = new_load

    def update_signal_len(self, signal_len:int):
        if(signal_len == 0):
            return
        max_sig_len = 0
        for element in self.elements.values():
            i_len = len(element.i)
            v_len = len(element.v)
            max_sig_len = max(max_sig_len, i_len, v_len)
        self.signal_len = signal_len

    def circuit_mask(self) -> list[bool]:
        '''returns a list of booleans indicating which elements are in the 
        circuit among all the system elements, if the system elements were
        flattened into a list ordered by circuit, then by element.'''
        start_index = self.index
        end_index = start_index + len(self.elements)
        circuit_mask = [False]*self.system.num_elements()
        circuit_mask[start_index:end_index] = True

class Element():
    def __init__(self, circuit: Circuit, kind:Kinds) -> None:
        assert(isinstance(kind,Kinds) and isinstance(circuit,Circuit))
        self.circuit = circuit
        self.low:Node = None
        self.high:Node = None
        self._parent:Element = None
        self._child:Element = None
        self.kind = kind
        self._i:Signal = Signal(self,[])
        self._v:Signal = Signal(self,[])
        self._a:float = None
        self._i_pred:Signal = Signal(self,[])
        self._v_pred:Signal = Signal(self,[])
        self._a_pred:float = None

    @property
    def index(self) -> list[int]:
        key_list = list(self.circuit.elements.keys())
        val_list = list(self.circuit.elements.values())
        position = val_list.index(self)
        return key_list[position]

    def to_nx(self):
        kind = ('kind',self.kind)
        v = ('v',self.v)
        i = ('i',self.i)
        attr = None
        if(self.kind == Kinds.ICS or self.kind == Kinds.IVS):
            attr = ('attr',None)
        else:
            attr = ('attr',self.a)
        return (self.low.index, self.high.index, self.key, (kind, i, v, attr))
    
    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self, element:'Element'):
        assert(isinstance(element,Element))
        assert(element.kind == Kinds.VC)
        self._parent = element

    def has_parent(self):
        return self.parent != None
    
    def has_parent_of(self, kind:Kinds):
        assert kind in [Kinds.VC, Kinds.CC]
        return self.has_parent() and self.parent.kind == kind
    
    @property
    def child(self):
        return self._child
    
    @child.setter
    def child(self, element:'Element'):
        assert(isinstance(element,Element))
        assert(element.kind == Kinds.SW)
        self._child = element

    def has_child(self):
        return self.child != None
    
    def has_child_of(self, kind:Kinds):
        return self.has_child() and self.child.kind == kind
    
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

    def prep_for_delete(self):
        self._i.prep_for_delete()
        self._i = None
        self._v.prep_for_delete()
        self._v = None
        self._a = None
        self._i_pred.prep_for_delete()
        self._i_pred = None
        self._v_pred.prep_for_delete()
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
        return str(self.index)
    
    def num_elements(self):
        return len(self.elements)

    @property
    def index(self) -> int:
        key_list = list(self.circuit.nodes.keys())
        val_list = list(self.circuit.nodes.values())
        position = val_list.index(self)
        return key_list[position]

    def prep_for_delete(self):
        '''remove all references to this Node and clear any data it holds'''
        # self.remove_self_from_elements()
        for element in self.elements:
            assert element.low != self and element.high != self
        self.clear()

    def clear(self):
        '''clear any data this Node holds'''
        self.circuit = None
        self.elements = []

    # def remove_self_from_elements(self):
    #     for element in self.elements:
    #         if(element.low == self):
    #             element.low = None
    #         if(element.high == self):
    #             element.high = None

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

    def prep_for_delete(self):
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