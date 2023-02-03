import networkx as nx
from enum import Enum
import numpy as np
import torch

class Kinds(Enum):
    IVS = 0
    R = 1

class Props(Enum):
    I = 0
    V = 1
    Attr = 2

def remove_tensor_row(tensor:torch.Tensor,row:int):
    ''' 
    return the tensor with the column at index "col" removed.

    must be 2x1 tensor
    '''
    max_rows = tensor.shape[1]
    assert(row <= max_rows)
    if(row < max_rows):
        return torch.cat((tensor[:row,:],tensor[row+1:,:]))
    else:
        return tensor[:row,:]

class Circuit():
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.elements: list[Element] = []
        # self.ref_node: Node = None

    def add_node(self, element:'Element') -> 'Node':
        '''create a node for a new elemenet and add to circuit.
            nodes are never created without an element. No floating nodes'''
        ckt_node = Node(self,[element])
        self.nodes.append(ckt_node)
        return ckt_node

    def remove_node(self, node: 'Node'):
        if(node in self.nodes):
            self.nodes.remove(node)

    def add_element(self, kind=None) -> 'Element':
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
        # assert(self.ref_node != None)
        # return remove_tensor_row(M_tensor,self.ref_node.idx)
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
        prop_matrix = []
        knowns_matrix = []
        kind_matrix = []
        
        num_kinds = len(Kinds)
        for e in range(len(self.elements)):
            element = self.elements[e]
            properties = (element.i,element.v, element.attr)
            num_props = len(properties)
            kind_oh = [0]*num_kinds
            values = [0]*num_props
            knowns_oh = [0]*num_props

            for kind in Kinds:
                if(element.kind == kind.value):
                    kind_oh[kind.value] = 1

            for p in range(num_props):
                prop = properties[p]
                if(prop == None):
                    values[p] = 0.0
                    knowns_oh[p] = 0
                else:
                    values[p] = float(prop)
                    knowns_oh[p] = 1

            prop_matrix.append(values)
            knowns_matrix.append(knowns_oh)
            kind_matrix.append(kind_oh)

        prop_tensor = torch.tensor(prop_matrix)
        knowns_tensor = torch.tensor(knowns_matrix)
        kind_tensor = torch.tensor(kind_matrix)
            
        return (kind_tensor, prop_tensor, knowns_tensor)

    def extract_nodes(self):
        pot_tensor = []
        knowns_tensor = []

        for n in range(len(self.nodes)):
            node = self.nodes[n]
            # if(node == self.ref_node):
            #     continue
            values = 0
            knowns_oh = 0
            if(node.p == None):
                values = 0.0
                knowns_oh = 0
            else:
                values = float(node.p)
                knowns_oh = 1
            pot_tensor.append(values)
            knowns_tensor.append(knowns_oh)
        pot_tensor = torch.tensor(pot_tensor)
        knowns_tensor = torch.tensor(knowns_tensor)
        pot_tensor = pot_tensor.reshape(shape=(self.num_nodes(),1))
        knowns_tensor = knowns_tensor.reshape(shape=(self.num_nodes(),1))
            
        return (pot_tensor, knowns_tensor)

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

    def set_ground(self):
        self.p = 0
        self.circuit.set_ground(self)

class Element():
    def __init__(self, circuit: Circuit, low:Node = None, high:Node = None, 
                kind = None, v = None, i = None, attr = None) -> None:
        self.circuit = circuit
        self.low:Node = low
        self.high:Node = high
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
        # from node, to node, edge key, attributes
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

class Input():
    def __init__(self, circuit:Circuit) -> None:
        self.circuit = circuit
        self.M = self.circuit.M()

    def mask(self,tensor,column):
        return tensor[:,column].reshape(self.circuit.num_elements(),1)
        
    def known_masks(self):
        _, _, e_knowns_tensor = self.circuit.extract_elements()
        _, n_knowns_tensor = self.circuit.extract_nodes()
        
        current = self.mask(e_knowns_tensor,Props.I.value)
        voltage = self.mask(e_knowns_tensor,Props.V.value)
        pot = n_knowns_tensor
        attr = self.mask(e_knowns_tensor,Props.Attr.value)

        return (current, voltage, pot, attr)

    def kind_map(self):
        kind_map = {}
        
        kind_map[Kinds.IVS.value] = self.is_kind_mask(Kinds.IVS)
        kind_map[Kinds.R.value] = self.is_kind_mask(Kinds.R)

        # kind_map[Kinds.IVS.value].requires_grad=True
        # kind_map[Kinds.R.value].requires_grad=True

        return kind_map

    def rs_mask(self, kind_map):
        num_elem = self.circuit.num_elements()
        
        # is source
        s_mask = kind_map[Kinds.IVS.value]
        is_s_mask_m = s_mask @ s_mask.T
        is_s_mask_y = torch.cat(tensors= (
                            torch.zeros(size=(num_elem,num_elem)),
                            is_s_mask_m,
                            torch.zeros(size=(num_elem,num_elem))
                        ),dim=1)

        # is resistor
        r_mask = kind_map[Kinds.R.value]
        is_r_mask_m = r_mask @ r_mask.T
        is_r_mask_z = torch.cat(tensors= (
                            is_r_mask_m,
                            torch.zeros(size=(num_elem,num_elem)),
                            torch.zeros(size=(num_elem,num_elem)),
                        ),dim=1)

        is_r_mask_y = torch.cat(tensors= (
                            torch.zeros(size=(num_elem,num_elem)),
                            is_r_mask_m,
                            torch.zeros(size=(num_elem,num_elem))
                        ),dim=1)

        return is_r_mask_z, is_r_mask_y, is_s_mask_y


    def is_kind_mask(self,kind:Kinds):
        element_kinds, _, _ = self.circuit.extract_elements()
        is_kind = element_kinds[:,kind.value].to(torch.float)
        is_kind = is_kind.reshape(self.circuit.num_elements(),1)
        # is_kind.requires_grad=True
        # print(is_kind)
        return is_kind

    def z(self):
        pass

    def y(self):
        pass

    def prop_tensors(self):
        num_elem = self.circuit.num_elements()
        num_nodes = self.circuit.num_nodes()
        pot_tensor,_ = self.circuit.extract_nodes()
        pot_tensor = pot_tensor.reshape(num_nodes,1)
        _,element_attr,_ = self.circuit.extract_elements()
        i_tensor = element_attr[:,Props.I.value].reshape(num_elem,1)
        v_tensor = element_attr[:,Props.V.value].reshape(num_elem,1)
        attr_tensor = element_attr[:,Props.Attr.value].reshape(num_elem,1)
        return i_tensor, v_tensor, pot_tensor, attr_tensor

    def element_row(self):
        _,_,_,element_attrs = self.prop_tensors()
        mask_rz,mask_ry,mask_sy = self.rs_mask(self.kind_map())
        z_r = mask_rz * -element_attrs
        y_r = mask_ry
        y_s = mask_sy
        row = z_r + y_r + y_s
        return row

    def s(self):
        _,_,_,element_attrs = self.prop_tensors()
        kind_map = self.kind_map()
        # i_src = None
        s = torch.zeros_like(element_attrs)
        v_bool = kind_map[Kinds.IVS.value].to(torch.bool)
        v_src = element_attrs[v_bool].unsqueeze(dim=1)
        s[v_bool] = v_src
        return s

    def num_elements(self):
        return self.circuit.num_elements()

class Learn():
    def __init__(self) -> None:
        pass