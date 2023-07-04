import networkx as nx
from enum import Enum
import torch
from torch import Tensor
from math import isclose,sqrt
from abc import abstractmethod

class Kind(Enum):
    V = 0
    I = 1
    R = 2
    VC = 3
    CC = 4
    SW = 5
    L = 6
    C = 7
    VG = 8
    CG = 9

class Quantity(Enum):
    I = 0
    V = 1

class SignalClass(Enum):
    PULSE = 0
    SIN = 1

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

    def load(self, solution:dict, time:float):
        for c,circuit in enumerate(self.circuits):
            circuit.load(solution[c],time)

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
    
    def add_element_of(self, kind:Kind):
        circuit = self.add_circuit()
        high_node = self.add_node(circuit,[])
        low_node = self.add_node(circuit,[])
        circuit.nodes.append(high_node)
        circuit.nodes.append(low_node)
        element = self.new_element(kind,circuit,high_node,low_node)
        self.elements.append(element)
        high_node.elements.append(element)
        low_node.elements.append(element)
        circuit.elements.append(element)
        return element
    
    def new_element(self, kind:Kind, circuit, high, low):
        if(kind==Kind.V):
            return Voltage(circuit, high, low)
        if(kind==Kind.I):
            return Current(circuit, high, low)
        if(kind==Kind.R):
            return Resistor(circuit, high, low)
        raise NotImplementedError()
    
    def remove_element(self, element: 'Element'):
        element.circuit.remove_element(element)
        if(element.circuit.is_empty()):
            self.remove_circuit(element.circuit)
        self.elements.remove(element)

    def add_element_pair(self, parent_kind:Kind, 
                        child_kind:Kind) -> tuple['Element','Element']:
        assert(parent_kind == Kind.VC or parent_kind == Kind.CC)
        assert(child_kind == Kind.SW or child_kind == Kind.VG or
               child_kind == Kind.CG)
        parent = self.add_element_of(parent_kind)
        child = self.add_element_of(child_kind)
        child.parent = parent
        parent.child = child
        return parent, child

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
        child_src = self.add_element_of(Kind.V)
        parent_src = self.add_element_of(Kind.V)
        child_res = self.add_element_of(Kind.R)
        parent_res = self.add_element_of(Kind.R)
        parent, child = self.add_element_pair(Kind.VC, Kind.SW)
        self.connect(child_src.high, child.high)
        self.connect(child.low, child_res.high)
        self.connect(child_src.low, child_res.low)
        self.connect(parent_src.high, parent_res.high)
        self.connect(parent_res.high, parent.high)
        self.connect(parent_src.low, parent_res.low)
        self.connect(parent_res.low, parent.low)
        return parent.circuit, child.circuit
    
    def ring(self, source_kind:Kind, load_kind:Kind, num_loads:int) -> 'Circuit':
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

    def ladder(self, source_kind:Kind, load_kind:Kind, num_loads:int) -> 'Circuit':
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
    
    def rc(self):
        self.prep_for_delete()
        v = self.add_element_of(Kind.V)
        r = self.add_element_of(Kind.R)
        c = self.add_element_of(Kind.C)
        self.connect(v.high, r.high)
        self.connect(r.low, c.high)
        self.connect(v.low, c.low)
        return v.circuit

class Circuit():
    '''A Circuit is a subset collection of Nodes and Elements from the System'''
    def __init__(self, system:System) -> None:
        self.system = system
        self.nodes: list[Node] = []
        self.elements: list = []

    def index(self) -> int:
        return self.system.circuits.index(self)
    
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

    def load(self, pred_ckt_t, time:float):
        '''Stores solutions or predictions as a time series'''
        for e,element in enumerate(self.elements):
            element.data[Quantity.I][time] = pred_ckt_t[Quantity.I][e].item()
            element.data[Quantity.V][time] = pred_ckt_t[Quantity.V][e].item()
    


class SignalFunction():
    def __init__(self, kind:SignalClass):
        self.kind = kind
    
    def to_spice(self) -> str:
        args = []
        for value in self.__dict__.values():
            args.append(str(value))
        return ' '.join(args)

class Pulse(SignalFunction):
        def __init__(self, off_value, on_value, freq, delay=0.0, rise_time=None,
              fall_time=None, duty_ratio=0.5, num_pulses=0):
            super().__init__()
            '''Delay determines when the on_value begins, otherise off_value.
            PULSE ( V1 V2 TD TR TF PW PER NP )'''
            self.V1 = off_value
            self.V2 = on_value
            self.TD = delay
            self.TR = rise_time
            self.TF = fall_time
            period = 1/freq
            self.PW = period*duty_ratio
            self.PER = period
            self.NP = num_pulses
            if(rise_time == None):
                if(fall_time != None):
                    self.TR = fall_time
                else:
                    self.TR = self.PER/1000
            if(fall_time == None):
                if(rise_time != None):
                    self.TF = rise_time
                else:
                    self.TF = self.PER/1000

class Element():
    def __init__(self, circuit: Circuit, high:'Node', low:'Node', 
                 kind:Kind) -> None:
        assert(isinstance(kind,Kind) and isinstance(circuit,Circuit))
        self.circuit = circuit
        self.low:Node = low
        self.high:Node = high
        self.parent:Element = None
        self.child:Element = None
        self.kind = kind
        self.data:dict[Quantity:dict[float:float]] = {}

    @abstractmethod
    def behavior(self) -> str:
        pass

    def to_spice(self)->str:
        id = self.kind.name + str(self.index())
        n_low = str(self.low.index)
        n_high = str(self.high.index)
        return id+" "+n_high+" "+n_low+" "+self.behavior()

    def add_data(self, prop:Quantity, time:float, value:float):
        if(prop not in self.data):
            self.data[prop] = {}
        self.data[prop][time] = value

    def index(self) -> int:
        return self.circuit.elements.index(self)
    
    def parent_index(self) -> int:
        if(self.parent == None):
            return None
        return self.parent.index()
    
    def circuit_index(self) -> int:
        return self.circuit.index()
    
    def parent_circuit_index(self) -> int:
        if(self.parent == None):
            return None
        return self.parent.circuit.index()
    
    @property
    def id(self) -> str:
        return str(self.kind.name) \
            + '_' + str(self.circuit.index()) + '_' + str(self.index)
    
    def __repr__(self) -> str:
        return self.id

    def has_parent(self):
        return self.parent != None
    
    def has_parent_of(self, kind:Kind):
        assert kind in [Kind.VC, Kind.CC]
        return self.has_parent() and self.parent.kind == kind

    @property
    def edge_key(self):
        parallels = self.circuit.elements_parallel_to(self,True)
        return parallels.index(self)

    def prep_for_delete(self):
        self.data.clear()
        self.child:Element = None
        self.parent:Element = None
        self.low.remove_element(self)
        self.low = None
        self.high.remove_element(self)
        self.high = None
        self.kind = None
        self.circuit = None

class IndependentSource(Element):
    '''
    General form:
    VXXXXXXX N+ N- <<DC> DC/TRAN VALUE> <AC <ACMAG <ACPHASE>>> <periodic signal>
    IYYYYYYY N+ N- <<DC> DC/TRAN VALUE> <AC <ACMAG <ACPHASE>>> <periodic signal>
    Examples:
    VCC 10 0 DC 6
    VIN 13 2 0.001 AC 1 SIN(0 1 1 MEG )
    ISRC 23 21 AC 0.333 45.0 SFFM(0 1 10 K 5 1 K )
    VMEAS 12 9
    '''
    def __init__(self, circuit:Circuit, high:'Node', low:'Node', kind:Kind) -> None:
        assert(kind==Kind.V or kind==Kind.I)
        super().__init__(circuit, high, low, kind)
        self.dc = None
        self.ac_mag = None
        self.ac_phase = None
        self.sig_func:SignalFunction = None

    def behavior(self) -> str:
        args = []
        if(self.dc != None):
            args.append("DC")
            args.append(str(self.dc))
        if(self.ac_mag != None): 
            args.append("AC")
            args.append(str(self.ac_mag))
        if(self.ac_phase != None):
            args.append(str(self.ac_phase))
        if(self.sig_func != None): args.append(self.sig_func.to_spice())
        return ' '.join(args)
    
class Voltage(IndependentSource):
    def __init__(self, circuit: Circuit, high: 'Node', low: 'Node') -> None:
        kind = Kind.V
        super().__init__(circuit, high, low, kind)

class Current(IndependentSource):
    def __init__(self, circuit: Circuit, high: 'Node', low: 'Node') -> None:
        kind = Kind.I
        super().__init__(circuit, high, low, kind)
    
class Resistor(Element):
    '''
    General form:
    RXXXXXXX n+ n- <resistance | r => value <ac = val> <m = val>
    + <scale = val> <temp = val> <dtemp = val> <tc1 = val> <tc2 = val>
    + <noisy =0|1>
    '''
    def __init__(self, circuit:Circuit, high:'Node', low:'Node') -> None:
        kind = Kind.R
        super().__init__(circuit, high, low, kind)
        self.parameter = None

    def behavior(self) -> str:
        assert(self.parameter != None)
        assert(self.parameter > 0)
        return str(self.parameter)
    
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
    '''Signal is for operations and analysis on time valued data.  It can 
    interpolate between defined points. Used to store predicted or calculated 
    voltage and current signals.'''
    def __init__(self, data:dict[float:float]) -> None:
        assert isinstance(data,dict)
        self._data = data
        self.lower = 0.0
        self.upper = 0.0
        self.max = 0.0
        self.is_periodic = False
        '''True periodic means to repeat pattern of signal, False means to 
        continue the last value in the sequence at DC.'''

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
    
    def __getitem__(self, time):
        assert(isinstance(time,float))
        if(self.is_periodic == True):
            if(self.max > 0.0):
                time = time % self.max
        if(time < self.lower or self.upper < time):
            self.set_window(time)
        if(self.is_periodic == False and self.upper < time):
            assert(self.lower <= self.upper)
            return self._data[self.max]
        if time in self._data:
            assert(self.lower <= self.upper)
            return self._data[time]
        return self.interpolate(time)
    
    def interpolate(self,time:float):
        low_val = self._data[self.lower]
        high_val = self._data[self.upper]
        ratio = (time - self.lower) / (self.upper - self.lower)
        assert(self.lower <= self.upper)
        return low_val + ratio * (high_val - low_val)
    
    def set_window(self,time:float):
        prev_key = 0.0
        for time_key in self._data:
            if(prev_key <= time <= time_key):
                self.lower = prev_key
                self.upper = time_key
                break
            prev_key = time_key
    
    def __setitem__(self, key, value):
        assert(isinstance(key,float))
        self._data[key] = value
        if(key > self.max):
            self.max = key

    def __iter__(self):
        return iter(self._data)
    
    def __eq__(self, signal) -> bool:
        if(not isinstance(signal,Signal)):
            return False
        if(len(self) != len(signal)):
            return False
        for time in self:
            if(time not in signal):
                return False
            if(not isclose(signal[time],self[time],rel_tol=1e-6)):
                return False
        return True
    
    def __neg__(self):
        data = {}
        for key,value in self._data.items():
            data[key] = -value
        return Signal(element=self.element, data=data)
    
    def __add__(self,signal):
        assert(isinstance(signal,Signal))
        assert(len(self)==len(signal) or len(self)==1 or len(signal)==1)
        sig_sum = {}
        if(len(self)==1 and len(signal)>1):
            for time in self:
                sig_sum[time] = self[0.0] + signal[time]
        if(len(self)>1 and len(signal)==1):
            for time in self:
                sig_sum[time] = self[time] + signal[0.0]
        if(len(self)==len(signal)):
            for time in self:
                sig_sum[time] = self[time] + signal[time]
        return Signal(None,sig_sum)

    def __sub__(self,signal):
        assert(isinstance(signal,Signal))
        assert(len(self)==len(signal) or len(self)==1 or len(signal)==1)
        sig_sum = {}
        if(len(self)==1 and len(signal)>1):
            for time in self:
                sig_sum[time] = self[0.0] - signal[time]
        if(len(self)>1 and len(signal)==1):
            for time in self:
                sig_sum[time] = self[time] - signal[0.0]
        if(len(self)==len(signal)):
            for time in self:
                sig_sum[time] = self[time] - signal[time]
        return Signal(None,sig_sum)
    
    def __mul__(self,signal):
        assert(isinstance(signal,Signal))
        assert(len(self)==len(signal) or len(self)==1 or len(signal)==1)
        sig_sum = {}
        if(len(self)==1 and len(signal)>1):
            for time in self:
                sig_sum[time] = self[0.0] * signal[time]
        if(len(self)>1 and len(signal)==1):
            for time in self:
                sig_sum[time] = self[time] * signal[0.0]
        if(len(self)==len(signal)):
            for time in self:
                sig_sum[time] = self[time] * signal[time]
        return Signal(None,sig_sum)
    
    def __truediv__(self,signal):
        assert(isinstance(signal,Signal))
        assert(len(self)==len(signal) or len(self)==1 or len(signal)==1)
        sig_sum = {}
        if(len(self)==1 and len(signal)>1):
            for time in self:
                sig_sum[time] = self[0.0] / signal[time]
        if(len(self)>1 and len(signal)==1):
            for time in self:
                sig_sum[time] = self[time] / signal[0.0]
        if(len(self)==len(signal)):
            for time in self:
                sig_sum[time] = self[time] / signal[time]
        return Signal(None,sig_sum)

    def rms(self):
        sum_of_squares = sum([self._data[time] ** 2 for time in self._data])
        count = len(self._data)
        return sqrt(sum_of_squares / count)