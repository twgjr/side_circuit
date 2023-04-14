import unittest
from circuits import Element,Node,Circuit,Kinds,Props,Signal
from torch import tensor
import torch

class Test_Kinds(unittest.TestCase):
    def test_Kinds(self):
        self.assertTrue(Kinds.ICS)
        self.assertTrue(Kinds.IVS)
        self.assertTrue(Kinds.R)
        self.assertTrue(Kinds.CC)
        self.assertTrue(Kinds.VC)
        self.assertTrue(Kinds.SW)
        self.assertTrue(len(Kinds)==6)

class Test_Props(unittest.TestCase):
    def test_Props(self):
        self.assertTrue(Props.I)
        self.assertTrue(Props.V)
        self.assertTrue(len(Props)==2)

class Test_Circuit(unittest.TestCase):
    def test_Circuit(self):
        circuit = Circuit()
        self.assertTrue(len(circuit.nodes)==0)
        self.assertTrue(len(circuit.elements)==0)

    def test_max_signal_len(self):
        circuit = Circuit()
        self.assertTrue(circuit.signal_len==0)
        circuit.add_element(Kinds.IVS)
        self.assertTrue(circuit.signal_len==0)
        circuit.add_element(Kinds.R)
        circuit.elements[0].v = [1,2,3]
        self.assertTrue(circuit.signal_len==3)
        circuit.elements[1].v = [1,2,3,4]
        self.assertTrue(circuit.signal_len==4)
        circuit.elements[1].v = [1,2,3]
        self.assertTrue(circuit.signal_len==3)

    def test_add_element(self):
        circuit = Circuit()
        self.assertTrue(len(circuit.elements) == 0)
        resistor = circuit.add_element(Kinds.R)
        self.assertTrue(isinstance(resistor,Element))
        self.assertTrue(circuit.elements[0].kind == Kinds.R)
        self.assertTrue(circuit.elements[0].i.is_empty())
        self.assertTrue(circuit.elements[0].v.is_empty())
        self.assertTrue(circuit.elements[0].a == None)
        self.assertTrue(len(circuit.elements) == 1)
        source = circuit.add_element(Kinds.IVS)
        resistor.a = 1.0
        resistor.i = [2.0]
        resistor.v = [3.0]
        self.assertTrue(len(circuit.elements) == 2)
        self.assertTrue(circuit.elements[0] != circuit.elements[1])
        self.assertTrue(circuit.elements[0].kind != circuit.elements[1].kind)
        self.assertTrue(circuit.elements[0].i != circuit.elements[1].i)
        self.assertTrue(circuit.elements[0].v != circuit.elements[1].v)
        self.assertTrue(circuit.elements[0].a != circuit.elements[1].a)

    def test_add_controlled_element(self):
        circuit = Circuit()
        self.assertTrue(len(circuit.elements) == 0)
        control, element = circuit.add_controlled_element(Kinds.VC,Kinds.SW)
        self.assertTrue(isinstance(control,Element))
        self.assertTrue(control in circuit.elements)
        self.assertTrue(isinstance(element,Element))
        self.assertTrue(element in circuit.elements)
        self.assertTrue(circuit.elements[0].kind == Kinds.VC)
        self.assertTrue(circuit.elements[1].kind == Kinds.SW)
        self.assertTrue(len(circuit.elements) == 2)
        self.assertTrue(circuit.elements[0].parent == None)
        self.assertTrue(circuit.elements[1].parent == circuit.elements[0])

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
        loops_test = [[ s1, r1, r2, r3]]
        self.assertTrue(loops == loops_test)

    def test_kvl_coefficients_ladder(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS,Kinds.R,1)
        kvl = circuit.kvl_coef()
        kvl_test = [[ 1,-1]]
        self.assertTrue(kvl == kvl_test)
        circuit.ladder(Kinds.IVS,Kinds.R,2)
        kvl = circuit.kvl_coef()
        kvl_test = [[ 1,-1, 0],
                    [ 1, 0,-1]]
        self.assertTrue(kvl == kvl_test)
        circuit.ladder(Kinds.IVS,Kinds.R,3)
        kvl = circuit.kvl_coef()
        kvl_test = [[ 1,-1, 0, 0],
                    [ 1, 0,-1, 0],
                    [ 1, 0, 0,-1]]
        self.assertTrue(kvl == kvl_test)

    def test_kvl_coefficients_ring(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,3)
        kvl = circuit.kvl_coef()
        kvl_test = [[ 1,-1,-1,-1]]
        self.assertTrue(kvl == kvl_test)

    def test_elements_series_elements_ring(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,2)
        source = circuit.elements[0]
        ser_elems = circuit.elements_in_series_with(source,include_ref=False)
        self.assertTrue(len(ser_elems) == 2)

    def test_elements_series_elements_ladder(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS,Kinds.R,3)
        source = circuit.elements[0]
        ser_elems = circuit.elements_in_series_with(source,include_ref=True)
        self.assertTrue(len(ser_elems) == 1)

    def test_elements_parallel_with_1(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS,Kinds.R,2)
        source = circuit.elements[0]
        par_elems = circuit.elements_parallel_to(source,False)
        self.assertTrue(len(par_elems) == 2)
        
    def test_elements_parallel_with_2(self):
        circuit = Circuit()
        source = circuit.add_element(Kinds.IVS)
        load0 = circuit.add_element(Kinds.R)
        load1 = circuit.add_element(Kinds.R)
        load2 = circuit.add_element(Kinds.R)
        circuit.connect(source.high, load0.low)
        circuit.connect(load0.high, load1.high)
        circuit.connect(load1.high, load2.high)
        circuit.connect(load1.low, load2.low)
        circuit.connect(load1.low, source.low)
        par_elems = circuit.elements_parallel_to(load2, True)
        self.assertTrue(len(par_elems) == 2)

    def test_load(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v = [3.0,6.0]
        circuit.elements[1].i = [1.5,3.0]
        i_pred = [torch.tensor([-1.5,1.5]).unsqueeze(1).to(torch.float),
                      torch.tensor([-3.0,3.0]).unsqueeze(1).to(torch.float)]
        v_pred = [torch.tensor([3.0,3.0]).unsqueeze(1).to(torch.float),
                      torch.tensor([6.0,6.0]).unsqueeze(1).to(torch.float)]
        a_pred = torch.tensor([0,2.0]).unsqueeze(1).to(torch.float)
        circuit.load(i_pred,v_pred,a_pred)
        i_test = [Signal(None,[-1.5,-3.0]),Signal(None,[1.5,3.0])]
        v_test = [Signal(None,[3.0,6.0]),Signal(None,[3.0,6.0])]
        a_test = [0,2.0]
        for e in range(circuit.num_elements()):
            self.assertTrue((circuit.elements[e].i_pred == i_test[e]))
            self.assertTrue((circuit.elements[e].v_pred == v_test[e]))
            self.assertTrue((circuit.elements[e].a_pred == a_test[e]))

    def test_kind_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        self.assertTrue(circuit.kind_list(Kinds.IVS) == [True,False])
        self.assertTrue(circuit.kind_list(Kinds.ICS) == [False,False])
        self.assertTrue(circuit.kind_list(Kinds.R) == [False,True])
        src, ctl_src, res, ctl_res, ctl_el, sw = circuit.switched_resistor()
        self.assertTrue(circuit.kind_list(Kinds.R) == 
                        [False,False,True,True,False,False])
        self.assertTrue(circuit.kind_list(Kinds.SW) ==
                        [False,False,False,False,False,True])
        self.assertTrue(circuit.kind_list(Kinds.SW,Kinds.VC) ==
                        [False,False,False,False,False,True])
        self.assertTrue(circuit.kind_list(Kinds.SW,Kinds.CC) ==
                        [False,False,False,False,False,False])

    def test_prop_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,2)
        el0 = circuit.elements[0]
        el1 = circuit.elements[1]
        el2 = circuit.elements[2]
        el0.v = [2]
        el1.i = [3]
        el2.v = [4]
        v = circuit.prop_list(Props.V)
        self.assertTrue(v[0] == el0.v)
        self.assertTrue(v[1].is_empty())
        self.assertTrue(v[2] == el2.v)
        i = circuit.prop_list(Props.I)
        self.assertTrue(i[0].is_empty())
        self.assertTrue(i[1] == el1.i)
        self.assertTrue(i[2].is_empty())

    def test_control_list(self):
        circuit = Circuit()
        src, ctl_src, res, ctl_res, ctl_el, sw = circuit.switched_resistor()
        v_controls = circuit.control_list(Kinds.VC)
        self.assertTrue(v_controls == [0,1,2,3,4,4])
        i_controls = circuit.control_list(Kinds.CC)
        self.assertTrue(i_controls == [0,1,2,3,4,5])

    def test_attr_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        el0 = circuit.elements[0]
        el1 = circuit.elements[1]
        el0.v = [2.0]
        el1.a = 3.0
        ivs = circuit.attr_list(Kinds.IVS)
        ics = circuit.attr_list(Kinds.ICS)
        r = circuit.attr_list(Kinds.R)
        self.assertTrue(ivs[0] == None)
        self.assertTrue(ivs[1] == None)
        self.assertTrue(r[0] == None)
        self.assertTrue(r[1] == el1.a)
        self.assertTrue(ics[0] == None)
        self.assertTrue(ics[1] == None)

    def test_ring_2_element(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        self.assertTrue(circuit.num_elements() == 2)
        self.assertTrue(circuit.num_nodes() == 2)
        self.assertTrue(circuit.elements[0].kind == Kinds.IVS)
        self.assertTrue(circuit.elements[1].kind == Kinds.R)
        parallels = circuit.elements_parallel_to(circuit.elements[0],True)
        self.assertTrue(len(parallels) == 2)

    def test_ring_3_element(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,2)
        self.assertTrue(circuit.num_elements() == 3)
        self.assertTrue(circuit.num_nodes() == 3)
        self.assertTrue(circuit.elements[0].kind == Kinds.IVS)
        self.assertTrue(circuit.elements[1].kind == Kinds.R)
        self.assertTrue(circuit.elements[2].kind == Kinds.R)
        parallels = circuit.elements_parallel_to(circuit.elements[0],True)
        self.assertTrue(len(parallels) == 1)

class Test_Element(unittest.TestCase):
    def test_Element(self):
        circuit = Circuit()
        ivs = Element(circuit=circuit,kind=Kinds.IVS)
        self.assertTrue(ivs.circuit == circuit)
        self.assertTrue(ivs.low == None)
        self.assertTrue(ivs.high == None)
        self.assertTrue(ivs.kind == Kinds.IVS)
        self.assertTrue(isinstance(ivs.i, Signal))
        self.assertTrue(isinstance(ivs.v, Signal))
        self.assertTrue(ivs.a == None)
        self.assertTrue(isinstance(ivs._i, Signal))
        self.assertTrue(isinstance(ivs._v, Signal))
        self.assertTrue(ivs._a == None)

    def test_append_signals(self):
        circuit = Circuit()
        ivs = Element(circuit=circuit,kind=Kinds.IVS)
        r = Element(circuit=circuit,kind=Kinds.R)
        self.assertTrue(id(ivs.i) != id(r.i))
        self.assertTrue(id(ivs.i._data) != id(r.i._data))
        self.assertTrue(id(ivs.v) != id(r.v))
        ivs.i.append(10.0)
        ivs_i_data = ivs.i.get_data()
        r_i_data = r.i.get_data()
        self.assertTrue(ivs_i_data != r_i_data)

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