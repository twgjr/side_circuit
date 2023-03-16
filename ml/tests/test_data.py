import unittest
from data import Data
from circuits import Circuit,Kinds,Props
from torch import Tensor

class Test_Preprocess(unittest.TestCase):
    def test_Preprocess(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        input = Data(circuit)
        self.assertTrue(type(input.circuit)==Circuit)
        self.assertTrue(type(input.M)==Tensor)
        self.assertTrue(type(input.elements) == dict)

    def test_base(self):
        input = Data(Circuit())
        self.assertTrue(input.base([0,1,2]) == 2)
        self.assertTrue(input.base([-1,1,2]) == 2)
        self.assertTrue(input.base([0,0,0]) == 1e-12)
        self.assertTrue(input.base([-1,-1]) == 1)

    def test_normalize(self):
        data = Data(Circuit())
        test_list = [1,1,2,3]
        base = data.base(test_list)
        self.assertTrue(data.normalize(base,test_list) == [1/3,1/3,2/3,1])

    def test_target_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[-1].i = 0.5
        data = Data(circuit)
        target_list_test = [1,1,1,1]
        i_base,v_base,_ = data.init_base()
        target_list = data.target_list(i_base,v_base)
        self.assertTrue(target_list == target_list_test)
    
    def test_target_mask_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[1].i = 0.5
        input = Data(circuit)
        test_mask = [False, True, True, False]
        target_mask_list = input.target_mask_list()
        self.assertTrue(target_mask_list == test_mask)

    def test_prop_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,2)
        circuit.elements[0].attr = 2
        circuit.elements[1].i = 3
        circuit.elements[2].v = 4
        input = Data(circuit)
        v_attr_F = input.prop_list(Props.V,False,1)
        self.assertTrue(v_attr_F == [1,1,4])
        v_attr_T = input.prop_list(Props.V,True,1)
        self.assertTrue(v_attr_T == [2,1,4])
        i_attr_F = input.prop_list(Props.I,False,1)
        self.assertTrue(i_attr_F == [1,3,1])
        i_attr_T = input.prop_list(Props.I,True,1)
        self.assertTrue(i_attr_T == [1,3,1])

    def test_mask_of_prop(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[1].i = 3
        input = Data(circuit)
        i_mask = input.mask_of_prop(Props.I,False)
        self.assertTrue(i_mask == [False,True])
        i_mask_with_attr = input.mask_of_prop(Props.I,True)
        self.assertTrue(i_mask_with_attr == [False,True])
        v_mask = input.mask_of_prop(Props.V,False)
        self.assertTrue(v_mask == [False,False])
        v_mask_with_attr = input.mask_of_prop(Props.V,True)
        self.assertTrue(v_mask_with_attr == [True,False])

    def test_to_bool_mask(self):
        preprocess = Data(Circuit())
        input = [None,0,1.1]
        mask = preprocess.nones_to_bool_mask(input)
        self.assertTrue(mask == [False,True,True])

    def test_replace_nones(self):
        preprocess = Data(Circuit())
        unknowns = [None,2,None,1.1]
        change = preprocess.replace_nones(unknowns,1)
        self.assertTrue(change == [1,2,1,1.1])
        change = preprocess.replace_nones(unknowns,0)
        self.assertTrue(change == [0,2,0,1.1])
        all_knowns = [3,10.5]
        no_change = preprocess.replace_nones(all_knowns,1)
        self.assertTrue(no_change == all_knowns)

    def test_attr_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[1].attr = 3
        preprocess = Data(circuit)
        ivs = preprocess.attr_list(Kinds.IVS,1)
        ics = preprocess.attr_list(Kinds.ICS,1)
        r = preprocess.attr_list(Kinds.R,1)
        self.assertTrue(ivs+ics+r == [2,1]+[1,1]+[1,3])

    def test_mask_of_attr(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[1].attr = 4

        preprocess = Data(circuit)

        ivs = preprocess.mask_of_attr(Kinds.IVS)
        ics = preprocess.mask_of_attr(Kinds.ICS)
        r = preprocess.mask_of_attr(Kinds.R)
        self.assertTrue(ivs+ics+r == [True,False,]+[False,False]+[False,True])

    def test_mask_of_kind(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 1
        preprocess = Data(circuit)

        mask = preprocess.mask_of_kind(Kinds.IVS)
        self.assertTrue(mask == [True,False])
        mask = preprocess.mask_of_kind(Kinds.ICS)
        self.assertTrue(mask == [False,False])
        mask = preprocess.mask_of_kind(Kinds.R)
        self.assertTrue(mask == [False,True])