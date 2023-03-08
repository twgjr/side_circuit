import unittest
from circuits import Element,Node,Circuit,Kinds,Props
from torch import tensor
import torch

class Test_Kinds(unittest.TestCase):
    def test_Kinds(self):
        self.assertTrue(Kinds.ICS)
        self.assertTrue(Kinds.IVS)
        self.assertTrue(Kinds.R)
        self.assertTrue(len(Kinds)==3)

class Test_Props(unittest.TestCase):
    def test_Props(self):
        self.assertTrue(Props.I)
        self.assertTrue(Props.V)
        self.assertTrue(Props.Pot)
        self.assertTrue(len(Props)==3)

class Test_Circuit(unittest.TestCase):
    def test_Circuit(self):
        circuit = Circuit()
        self.assertTrue(len(circuit.nodes)==0)
        self.assertTrue(len(circuit.elements)==0)

    def test_add_element(self):
        circuit = Circuit()
        self.assertTrue(len(circuit.elements) == 0)
        resistor = circuit.add_element(Kinds.R)
        self.assertTrue(isinstance(resistor,Element))
        self.assertTrue(circuit.elements[0].kind == Kinds.R)
        self.assertTrue(len(circuit.elements) == 1)

    def test_remove_element(self):
        circuit = Circuit()
        ics = circuit.add_element(Kinds.ICS)
        self.assertTrue(len(circuit.elements) == 1)
        circuit.remove_element(ics,False)
        self.assertTrue(len(circuit.elements) == 0)

    def test_add_node(self):
        circuit = Circuit()
        self.assertTrue(len(circuit.nodes) == 0)
        ivs = Element(circuit,Kinds.IVS)
        ics = Element(circuit,Kinds.ICS)
        r = Element(circuit,Kinds.R)
        elements = [ivs,ics,r]
        node = circuit.add_node(elements)
        self.assertTrue(len(circuit.nodes) == 1)
        self.assertTrue(node.elements[0].kind == Kinds.IVS)
        self.assertTrue(node.elements[1].kind == Kinds.ICS)
        self.assertTrue(node.elements[2].kind == Kinds.R)

    def test_remove_node(self):
        circuit = Circuit()
        ivs = Element(circuit,Kinds.IVS)
        elements = [ivs]
        self.assertTrue(len(circuit.nodes) == 0)
        node0 = circuit.add_node(elements)
        node1 = circuit.add_node(elements)
        node2 = circuit.add_node(elements)
        self.assertTrue(len(circuit.nodes) == 3)
        circuit.remove_node(node1)
        self.assertTrue(len(circuit.nodes) == 2)
        self.assertTrue(circuit.nodes[0] == node0)
        self.assertTrue(circuit.nodes[1] == node2)

    def test_connect(self):
        circuit = Circuit()
        ivs = circuit.add_element(Kinds.IVS)
        r = circuit.add_element(Kinds.R)
        self.assertTrue(ivs.high != r.high)
        circuit.connect(ivs.high, r.high)
        self.assertTrue(ivs.high == r.high)
        self.assertTrue(ivs.low != r.high)
        self.assertTrue(ivs.high != r.low)
        self.assertTrue(ivs.low != r.low)

    def test_num_nodes(self):
        circuit = Circuit()
        ivs = Element(circuit,Kinds.IVS)
        circuit.add_node([ivs])
        self.assertTrue(circuit.num_nodes() == 1)

    def test_num_elements(self):
        circuit = Circuit()
        circuit.add_element(Kinds.ICS)
        self.assertTrue(circuit.num_elements() == 1)

    def test_node_idx(self):
        circuit = Circuit()
        circuit.add_node([Element(circuit,Kinds.IVS)])
        test_node = circuit.add_node([Element(circuit,Kinds.IVS)])
        circuit.add_node([Element(circuit,Kinds.IVS)])
        self.assertTrue(circuit.node_idx(test_node) == 1)

    def test_M(self):
        circuit = Circuit()
        source = circuit.add_element(Kinds.IVS)
        load = circuit.add_element(Kinds.R)
        circuit.connect(source.high, load.high)
        circuit.connect(source.low, load.low)
        M = circuit.M()
        M_test = tensor([[-1,-1],
                         [ 1, 1]])
        self.assertTrue(torch.all(torch.eq(M,M_test)))

    def test_spanning_tree_ladder(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS,Kinds.R,3)
        mst = circuit.spanning_tree()
        self.assertTrue(len(mst) == 1)
    
    def test_spanning_tree_ring(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,3)
        mst = circuit.spanning_tree()
        self.assertTrue(len(mst) == 3)

    def test_loops_ladder(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS,Kinds.R,3)
        loops = circuit.loops()
        s1 = circuit.elements[0]
        r1 = circuit.elements[1]
        r2 = circuit.elements[2]
        r3 = circuit.elements[3]
        loops_test = [[ s1, r1],
                      [ s1, r2],
                      [ s1, r3]]
        self.assertTrue(loops == loops_test)

    def test_loops_ring(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,3)
        loops = circuit.loops()
        s1 = circuit.elements[0]
        r1 = circuit.elements[1]
        r2 = circuit.elements[2]
        r3 = circuit.elements[3]
        loops_test = [[ s1, r3, r2, r1]]
        self.assertTrue(loops == loops_test)

    def test_kvl_coefficients_ladder(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS,Kinds.R,3)
        kvl = circuit.kvl_coefficients()
        kvl_test = [[ 1,-1, 0, 0],
                    [ 1, 0,-1, 0],
                    [ 1, 0, 0,-1]]
        self.assertTrue(kvl == kvl_test)

    def test_kvl_coefficients_ring(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,3)
        kvl = circuit.kvl_coefficients()
        kvl_test = [[ 1,-1,-1,-1]]
        self.assertTrue(kvl == kvl_test)

    def test_elements_parallel_with_1(self):
        circuit = Circuit()
        source = circuit.add_element(Kinds.IVS)
        load0 = circuit.add_element(Kinds.R)
        load1 = circuit.add_element(Kinds.R)
        circuit.connect(load0.high, source.high)
        circuit.connect(load0.low, source.low)
        circuit.connect(load1.high, source.high)
        circuit.connect(load1.low, source.low)
        par_elems = circuit.parallel_elements(source)
        self.assertTrue(len(par_elems) == 3)
        
    def test_elements_parallel_with_2(self):
        circuit = Circuit()
        source = circuit.add_element(Kinds.IVS)
        load0 = circuit.add_element(Kinds.R)
        load1 = circuit.add_element(Kinds.R)
        load2 = circuit.add_element(Kinds.R)
        circuit.connect(load0.low, source.high)
        circuit.connect(load1.high, load0.high)
        circuit.connect(load1.low, source.low)
        circuit.connect(load2.high, load1.high)
        circuit.connect(load2.low, load1.low)
        par_elems = circuit.parallel_elements(load2)
        self.assertTrue(len(par_elems) == 2)

    def test_extract_elements(self):
        circuit = Circuit()
        ivs = circuit.add_element(Kinds.IVS)
        r = circuit.add_element(Kinds.R)
        circuit.connect(ivs.high, r.high)
        circuit.connect(ivs.low, r.low)
        ivs.attr = 1
        r.i = 0.5
        extract = circuit.export() 
        extract_test = {
            'kinds': {
                Kinds.IVS: [True, False],
                Kinds.ICS: [False, False],
                Kinds.R: [False, True]
                },
            'properties': {
                Props.I: [None, 0.5],
                Props.V: [None, None],
                Props.Pot: [None, None]
                },
            'attributes': {
                Kinds.IVS: [1, None],
                Kinds.ICS: [None, None],
                Kinds.R: [None, None]
                }
            }
        self.assertTrue(extract == extract_test)

    def test_ring_2_element(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        self.assertTrue(circuit.num_elements() == 2)
        self.assertTrue(circuit.num_nodes() == 2)
        self.assertTrue(circuit.elements[0].kind == Kinds.IVS)
        self.assertTrue(circuit.elements[1].kind == Kinds.R)
        self.assertTrue(len(circuit.parallel_elements(circuit.elements[0])) == 2)

    def test_ring_3_element(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,2)
        self.assertTrue(circuit.num_elements() == 3)
        self.assertTrue(circuit.num_nodes() == 3)
        self.assertTrue(circuit.elements[0].kind == Kinds.IVS)
        self.assertTrue(circuit.elements[1].kind == Kinds.R)
        self.assertTrue(circuit.elements[2].kind == Kinds.R)
        self.assertTrue(len(circuit.parallel_elements(circuit.elements[0])) == 1)

class Test_Element(unittest.TestCase):
    def test_Element(self):
        circuit = Circuit()
        ivs = Element(circuit=circuit,kind=Kinds.IVS)
        self.assertTrue(ivs.circuit == circuit)
        self.assertTrue(ivs.low == None)
        self.assertTrue(ivs.high == None)
        self.assertTrue(ivs.kind == Kinds.IVS)
        self.assertTrue(ivs.i == None)
        self.assertTrue(ivs.v == None)
        self.assertTrue(ivs.attr == None)

class Test_Node(unittest.TestCase):
    def test_Node(self):
        circuit = Circuit()
        ivs = Element(circuit,Kinds.IVS)
        r = Element(circuit,Kinds.R)
        elements = [ivs,r]
        node = Node(circuit,elements)
        self.assertTrue(node.circuit == circuit)
        self.assertTrue(node.elements[0] == ivs)
        self.assertTrue(node.elements[1] == r)
        self.assertTrue(node.potential == None)