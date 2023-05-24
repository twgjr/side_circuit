import networkx as nx
from enum import Enum
import torch
from torch import Tensor
from math import isclose

class Kinds(Enum):
    IVS = 0
    ICS = 1
    R = 2
    VC = 3
    CC = 4
    SW = 5
    L = 6
    C = 7
    VG = 8
    IG = 9

class Props(Enum):
    I = 0
    V = 1
    A = 2
    B = 3

class System():
    '''Collection of isolated Circuits that are only connected by parent/child
    relationships between Elements. When an Element is added to the System,
    it is added to its own new Circuit.  When two Elements are connected, the
    Circuits are merged, retaining the larger Circuit of the two connecting
    Elements.'''
    def __init__(self) -> None:
        self.circuits: list[Circuit] = []
        self.elements: list[Element] = []
        self.nodes: list[Node] = []
        # self.i_base = 1
        # self.v_base = 1
        # self.r_base = 1
        # self.l_base = 1
        # self.c_base = 1
        self.dt = 1e-6

    def load(self, pred):
        for t,pred_t in enumerate(pred):
            for c,circuit in enumerate(self.circuits):
                circuit.load(pred_t[c],t)

    def num_circuits(self) -> int:
        return len(self.circuits)
    
    def num_elements(self) -> int:
        return sum([c.num_elements() for c in self.circuits])
        
    def prep_for_delete(self):
        for c in self.circuits:
            c.prep_for_delete()
        self.circuits.clear()

    def add_circuit(self) -> 'Circuit':
        circuit = Circuit(self)
        self.circuits.append(circuit)
        return circuit
    
    def remove_circuit(self, circuit: 'Circuit'):
        circuit.prep_for_delete()
        self.circuits.remove(circuit)
        for c in self.circuits:
            if(c == circuit):
                self.circuits.remove(c)
                break

    def add_node(self, circuit:'Circuit', elements:list['Element']) -> 'Node':
        assert(isinstance(elements,list))
        assert(circuit in self.circuits)
        ckt_node = Node(circuit,elements)
        self.nodes.append(ckt_node)
        return ckt_node
    
    def remove_node(self, node: 'Node'):
        node.circuit.remove_node(node)
        self.nodes.remove(node)
        if(node.circuit.is_empty()):
            self.remove_circuit(node.circuit)
    
    def add_element_of(self, kind:Kinds) -> 'Element':
        circuit = self.add_circuit()
        high_node = self.add_node(circuit,[])
        low_node = self.add_node(circuit,[])
        circuit.nodes.append(high_node)
        circuit.nodes.append(low_node)
        element = Element(circuit, high_node, low_node, kind)
        self.elements.append(element)
        high_node.elements.append(element)
        low_node.elements.append(element)
        circuit.elements.append(element)
        return element
    
    def remove_element(self, element: 'Element'):
        element.circuit.remove_element(element)
        if(element.circuit.is_empty()):
            self.remove_circuit(element.circuit)
        self.elements.remove(element)

    def add_element_pair(self, parent_kind:Kinds, 
                        child_kind:Kinds) -> tuple['Element','Element']:
        assert(parent_kind == Kinds.VC)
        assert(child_kind == Kinds.SW)
        parent = self.add_element_of(parent_kind)
        child = self.add_element_of(child_kind)
        child.parent = parent
        parent.child = child
        return parent, child
    
    def name_exists(self, name: str) -> bool:
        for c in self.circuits:
            if c.name_exists(name):
                return True
        return False

    def connect(self, a: 'Node', b: 'Node'):
        '''Merge two nodes together into one. If the nodes are in different
        circuits, then merge the smaller circuit into the larger circuit.  If
        the nodes are in the same circuit, then transfer all elements from the
        smaller node to the larger node.  Finally remove the smaller node.'''
        assert(isinstance(a, Node))
        assert(isinstance(b, Node))
        if(a == b):
            return
        if(a.circuit != b.circuit):
            self.merge_circuits(a.circuit, b.circuit)
        self.merge_nodes(a, b)

    def merge_nodes(self, a: 'Node', b: 'Node'):
        '''Merge two nodes together into one.  Transfer all elements from the 
        smaller node to the larger node.  Finally remove the smaller node.'''
        assert(isinstance(a, Node))
        assert(isinstance(b, Node))
        if(a == b):
            return
        if(a.circuit != b.circuit):
            raise Exception('Cannot merge nodes in different circuits')
        if(a.num_elements() < b.num_elements()):
            small = a
            large = b
        else:
            small = b
            large = a
        for element in small.elements:
            if(element.high == small):
                element.high = large
            if(element.low == small):
                element.low = large
            large.elements.append(element)
        small.elements.clear()
        small.circuit.nodes.remove(small)
        self.nodes.remove(small)

    def merge_circuits(self, a: 'Circuit', b: 'Circuit'):
        '''Merge two circuits together into one.  Transfer all elements from
        the smaller circuit to the larger circuit.  Finally remove the smaller
        circuit.'''
        assert(isinstance(a, Circuit))
        assert(isinstance(b, Circuit))
        if(a == b):
            return
        if(a.num_elements() < b.num_elements()):
            small = a
            large = b
        else:
            small = b
            large = a
        for element in small.elements:
            element.circuit = large
            large.elements.append(element)
        for node in small.nodes:
            node.circuit = large
            large.nodes.append(node)
        small.elements.clear()
        small.nodes.clear()
        self.circuits.remove(small)
    
    def switched_resistor(self) -> tuple['Circuit', 'Circuit']:
        '''one source and one load, with a switch in series. The switch control
        has a separate voltage source  and resistor in parallel.'''
        self.prep_for_delete()
        child_src = self.add_element_of(Kinds.IVS)
        parent_src = self.add_element_of(Kinds.IVS)
        child_res = self.add_element_of(Kinds.R)
        parent_res = self.add_element_of(Kinds.R)
        parent, child = self.add_element_pair(Kinds.VC, Kinds.SW)
        self.connect(child_src.high, child.high)
        self.connect(child.low, child_res.high)
        self.connect(child_src.low, child_res.low)
        self.connect(parent_src.high, parent_res.high)
        self.connect(parent_res.high, parent.high)
        self.connect(parent_src.low, parent_res.low)
        self.connect(parent_res.low, parent.low)
        return parent.circuit, child.circuit
    
    def ring(self, source_kind:Kinds, load_kind:Kinds, num_loads:int) -> 'Circuit':
        '''one source and all loads in series'''
        assert(num_loads > 0)
        self.prep_for_delete()
        source = self.add_element_of(source_kind)
        first_load = self.add_element_of(load_kind)
        self.connect(source.high, first_load.high)
        prev_element = first_load
        for l in range(num_loads-1):
            new_load = self.add_element_of(load_kind)
            self.connect(prev_element.low, new_load.high)
            prev_element = new_load
        self.connect(source.low, prev_element.low)
        return source.circuit

    def ladder(self, source_kind:Kinds, load_kind:Kinds, num_loads:int) -> 'Circuit':
        '''one source and all loads in parallel'''
        assert(num_loads > 0)
        self.prep_for_delete()
        source = self.add_element_of(source_kind)
        first_load = self.add_element_of(load_kind)
        self.connect(source.high, first_load.high)
        self.connect(source.low, first_load.low)
        prev_element = first_load
        for l in range(num_loads-1):
            new_load = self.add_element_of(load_kind)
            self.connect(prev_element.high, new_load.high)
            self.connect(prev_element.low, new_load.low)
            prev_element = new_load
        return source.circuit
    
    def update_bases(self):
        i_sigs = []
        v_sigs = []
        r_vals = []
        l_vals = []
        c_vals = []
        for circuit in self.circuits:
            i_sigs += circuit.prop_list(Props.I)
            v_sigs += circuit.prop_list(Props.V)
            r_vals += circuit.attr_list(Kinds.R)
            l_vals += circuit.attr_list(Kinds.L)
            c_vals += circuit.attr_list(Kinds.C)
        i_base = self.signals_base(i_sigs)
        v_base = self.signals_base(v_sigs)
        r_base = self.values_base(r_vals)
        l_base = self.values_base(l_vals)
        c_base = self.values_base(c_vals)
        return {'i':i_base,'v':v_base,'r':r_base,'l':l_base,'c':c_base}
    
    def signals_base(self, signals:list['Signal']) -> float:
        input_max = 1
        for signal in signals:
            if(signal.is_empty()):
                continue
            else:
                for v in range(len(signal)):
                    val = signal[v]
                    abs_val = abs(val)
                    if(abs_val > input_max):
                        input_max = abs_val
        return input_max
        
    def values_base(self, values:list[float]) -> float:
        input_max = 1
        for val in values:
            if(val == None):
                continue
            abs_val = abs(val)
            if(abs_val > input_max):
                input_max = abs_val
        return input_max

class Circuit():
    '''A Circuit is a subset collection of Nodes and Elements from the System'''
    def __init__(self, system:System) -> None:
        self.system = system
        self.nodes: list[Node] = []
        self.elements: list[Element] = []
        self.signal_len = 0

    @property
    def index(self) -> int:
        return self.system.circuits.index(self)
    
    def name_exists(self, name:str) -> bool:
        for e in self.elements:
            if e.name == name:
                return False
        return True
    
    def is_empty(self) -> bool:
        return len(self.elements) == 0 and len(self.nodes) == 0

    def prep_for_delete(self):
        for e in self.elements:
            e.prep_for_delete()
        for n in self.nodes:
            n.prep_for_delete()
        self.clear()

    def clear(self):
        self.nodes.clear()
        self.elements.clear()
    
    def remove_element(self, element: 'Element'):
        element.prep_for_delete()
        self.elements.remove(element)
    
    def add_node(self, node: 'Node'):
        self.nodes.append(node)

    def remove_node(self, node: 'Node'):
        self.nodes.remove(node)
        node.prep_for_delete()

    def num_nodes(self):
        return len(self.nodes)

    def num_elements(self):
        return len(self.elements)

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

    def load(self, pred_ckt_t:dict[str:Tensor], time:int):
        '''Stores predictions from Trainer in Circuit'''
        for e,element in enumerate(self.elements):
            # denorm_i = pred_ckt_t[Props.I][e].item() * self.system.i_base
            # denorm_v = pred_ckt_t[Props.V][e].item() * self.system.v_base
            denorm_i = pred_ckt_t[Props.I][e].item()
            denorm_v = pred_ckt_t[Props.V][e].item()
            element.i_pred[time] = denorm_i
            element.v_pred[time] = denorm_v
            if(time > 0): continue
            if(pred_ckt_t[Props.A][e] != None):
                norm_a = None
                if(element.kind != Kinds.VC):
                    norm_a = pred_ckt_t[Props.A][e].item()  
                if(element.kind == Kinds.R):
                    # element.a_pred = norm_a * self.system.r_base
                    element.a_pred = norm_a
                elif(element.kind == Kinds.IVS):
                    # element.a_pred = norm_a * self.system.v_base
                    element.a_pred = norm_a
                elif(element.kind == Kinds.VC):
                    pass
                elif(element.kind == Kinds.ICS):
                    # element.a_pred = norm_a * self.system.i_base
                    element.a_pred = norm_a
                elif(element.kind == Kinds.SW):
                    pass
                elif(element.kind == Kinds.C):
                    # element.a_pred = norm_a * self.system.c_base
                    element.a_pred = norm_a
                elif(element.kind == Kinds.L):
                    # element.a_pred = norm_a * self.system.l_base
                    element.a_pred = norm_a
                else:
                    assert()
    
    def kind_list(self, kind:Kinds, with_control:Kinds=None) -> list[bool]:
        '''returns a list of booleans indicating which elements are of kind'''
        assert(kind != with_control)
        assert(with_control == Kinds.VC or with_control == Kinds.CC or 
               with_control == None)
        kind_list = []
        for element in self.elements:
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
        for element in self.elements:
            if(element.kind == kind):
                attrs.append(element.a)
            else:
                attrs.append(None)
        return attrs
    
    def prop_list(self, prop:Props) -> list['Signal']:
        '''returns a list of properties matching prop of elements'''
        props = []
        for element in self.elements:
            if(prop == Props.I):
                props.append(element.i.copy())
            elif(prop == Props.V):
                props.append(element.v.copy())
            else:
                assert()
        return props
    
    def prop_mask(self, prop:Props) -> list[bool]:
        '''returns boolean mask of known element properties ordered by element'''
        assert isinstance(prop,Props)
        ret_list = []
        for element in self.elements:
            if(prop == Props.I):
                ret_list.append(len(element.i)>0)
            elif(prop == Props.V):
                ret_list.append(len(element.v)>0)
            elif(prop == Props.A):
                ret_list.append(element.a != None)
            else:
                assert()
        return ret_list
    
    def attr_mask(self, kind:Kinds) -> list[bool]:
        '''returns boolean mask of known element attributes ordered by element'''
        assert isinstance(kind,Kinds)
        assert kind != Kinds.ICS and kind != Kinds.IVS
        attrs = self.attr_list(kind)
        ret_list = []
        for attr in attrs:
            assert isinstance(attr,float) or attr == None
            ret_list.append(attr != None)
        return ret_list

class Element():
    def __init__(self, circuit: Circuit, high:'Node', low:'Node', kind:Kinds, 
                 name:str=None) -> None:
        assert(isinstance(kind,Kinds) and isinstance(circuit,Circuit))
        self.system = circuit.system
        self.circuit = circuit
        self.low:Node = low
        self.high:Node = high
        self.parent:Element = None
        self.child:Element = None
        self.kind = kind
        self._i:Signal = Signal(self,{})
        self._v:Signal = Signal(self,{})
        self.a:Signal = Signal(self,{})
        self.i_pred:Signal = Signal(self,{})
        self.v_pred:Signal = Signal(self,{})
        self.a_pred:Signal = Signal(self,{})
        self._name = name

    @property
    def index(self) -> list[int]:
        return self.circuit.elements.index(self)
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name:str):
        assert(isinstance(name,str))
        if(self.circuit.system.name_exists(name)):
            raise ValueError('name already exists')
        self._name = name
    
    @property
    def id(self) -> str:
        return str(self.kind.name) \
            + '_' + str(self.circuit.index) + '_' + str(self.index)
    
    def __repr__(self) -> str:
        return self.name + '_' + self.id

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

    def has_parent(self):
        return self.parent != None
    
    def has_parent_of(self, kind:Kinds):
        assert kind in [Kinds.VC, Kinds.CC]
        return self.has_parent() and self.parent.kind == kind
    
    @property 
    def i(self):
        return self._i
    
    @i.setter
    def i(self, key, value):
        series = self.circuit.elements_in_series_with(self,False)
        for element in series:
            if(not element.i.is_empty()):
                assert()
        self._i[key] = value

    @property
    def v(self):
        return self._v
    
    @v.setter
    def v(self, key, value):
        assert isinstance(key,float)
        assert isinstance(value,float)
        parallels = self.circuit.elements_parallel_to(self,False)
        for element in parallels:
            if(not element.v.is_empty()):
                assert()
        self._v[key] = value

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
        self.i_pred.prep_for_delete()
        self._i_pred = None
        self.v_pred.prep_for_delete()
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
        self.system = circuit.system
        self.circuit = circuit
        self.elements = elements

    def __repr__(self) -> str:
        el_str = ''
        for element in self.elements:
            el_str += str(element) + ','
        el_str = el_str[:-1]    # remove last comma
        return f'Node({self.index}:{el_str})'
    
    def num_elements(self):
        return len(self.elements)

    @property
    def index(self) -> int:
        return self.circuit.nodes.index(self)

    def prep_for_delete(self):
        for element in self.elements:
            assert element.low != self and element.high != self
        self.clear()

    def clear(self):
        self.circuit = None
        self.elements = []

    def add_element(self, element: Element):
        if(element not in self.elements):
            self.elements.append(element)

    def remove_element(self, element: Element):
        if(element in self.elements):
            self.elements.remove(element)

class Signal():
    def __init__(self, element: Element,data:dict[float:float]) -> None:
        assert isinstance(element,Element) or element == None
        assert isinstance(data,dict)
        self.element = element
        self._data = data

    def values(self):
        return self._data.values()
    
    def keys(self):
        return self._data.keys()
    
    def items(self):
        return self._data.items()

    def __repr__(self) -> str:
        return str(self._data)

    def prep_for_delete(self):
        self._data = {}
        self.element = None
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, key):
        assert(isinstance(key,float))
        return self._data[key]
    
    def __setitem__(self, key, value):
        assert(isinstance(key,float))
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)
    
    def __eq__(self, obj) -> bool:
        if(not isinstance(obj,Signal)):
            return False
        if(len(self.items() != len(obj.items()))):
            return False
        for time,value in self.items():
            if(time not in obj.items()):
                return False
            if(not isclose(obj[time],value)):
                return False
        return True
    
    def __neg__(self):
        data = {}
        for key,value in self._data.items():
            data[key] = -value
        return Signal(element=self.element, data=data)
    
    def clear(self):
        self._data = {}

    def is_empty(self):
        return len(self._data) == 0
    
    def copy(self):
        assert isinstance(self.element,Element)
        data_copy = {}
        for key,value in self._data.items():
            data_copy[key] = value
        return Signal(element=self.element, data=data_copy)