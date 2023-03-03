import unittest
from data import Preprocess
from circuits import Circuit,Kinds,Props
from torch import Tensor

class Test_Preprocess(unittest.TestCase):
    def test_Preprocess(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        input = Preprocess(circuit)
        self.assertTrue(type(input.circuit)==Circuit)
        self.assertTrue(type(input.M)==Tensor)
        self.assertTrue(type(input.elements) == dict)
        self.assertTrue(type(input.target) == list)
        self.assertTrue(type(input.target_mask) == list)

    def test_base(self):
        input = Preprocess(Circuit())
        self.assertTrue(input.base([0,1,2]) == 2)
        self.assertTrue(input.base([-1,1,2]) == 2)
        self.assertTrue(input.base([0,0,0]) == 1)
        self.assertTrue(input.base([-1,-1]) == 1)

    def test_normalize(self):
        input = Preprocess(Circuit())
        test_list = [0,1,2]
        base = input.base(test_list)
        self.assertTrue(input.normalize(base,test_list) == [0,0.5,1])

    def test_target_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[-1].i = 0.5
        input = Preprocess(circuit)
        target = [1,0.5,1,0.5,1,1]
        self.assertTrue(input.target_list() == target)
    
    def test_target_mask_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 1
        circuit.elements[1].i = 0.5
        input = Preprocess(circuit)
        test_mask = [False, True, True, False, False, False]
        target_mask_list = input.target_mask_list()
        self.assertTrue(target_mask_list == test_mask)

    def test_prop_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[1].i = 3
        circuit.elements[1].v = 4
        input = Preprocess(circuit)
        v_attr_F = input.prop_list(Props.V,False)
        self.assertTrue(v_attr_F == [1,4])
        v_attr_T = input.prop_list(Props.V,True)
        self.assertTrue(v_attr_T == [2,4])
        i_attr_F = input.prop_list(Props.I,False)
        self.assertTrue(i_attr_F == [1,3])
        i_attr_T = input.prop_list(Props.I,True)
        self.assertTrue(i_attr_T == [1,3])
        pot_attr_F = input.prop_list(Props.Pot,False)
        self.assertTrue(pot_attr_F == [1,1])
        pot_attr_T = input.prop_list(Props.Pot,True)
        self.assertTrue(pot_attr_T == [1,1])

    def test_mask_of_prop(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[1].i = 3
        input = Preprocess(circuit)
        i_mask = input.mask_of_prop(Props.I,False)
        self.assertTrue(i_mask == [False,True])
        i_mask_with_attr = input.mask_of_prop(Props.I,True)
        self.assertTrue(i_mask_with_attr == [False,True])
        v_mask = input.mask_of_prop(Props.V,False)
        self.assertTrue(v_mask == [False,False])
        v_mask_with_attr = input.mask_of_prop(Props.V,True)
        self.assertTrue(v_mask_with_attr == [True,False])
        p_mask = input.mask_of_prop(Props.Pot,False)
        self.assertTrue(p_mask == [False,False])
        p_mask_with_attr = input.mask_of_prop(Props.Pot,True)
        self.assertTrue(p_mask_with_attr == [False,False])

    def test_to_bool_mask(self):
        preprocess = Preprocess(Circuit())
        input = [None,0,1.1]
        mask = preprocess.nones_to_bool_mask(input)
        self.assertTrue(mask == [False,True,True])

    def test_replace_nones(self):
        preprocess = Preprocess(Circuit())
        unknowns = [None,2,None,1.1]
        change = preprocess.replace_nones(unknowns)
        self.assertTrue(change == [1,2,1,1.1])
        all_knowns = [3,10.5]
        no_change = preprocess.replace_nones(all_knowns)
        self.assertTrue(no_change == all_knowns)

    def test_attr_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[1].attr = 3
        preprocess = Preprocess(circuit)

        ivs = preprocess.attr_list(Kinds.IVS)
        ics = preprocess.attr_list(Kinds.ICS)
        r = preprocess.attr_list(Kinds.R)
        self.assertTrue(ivs+ics+r == [2,1]+[1,1]+[1,3])

    def test_mask_of_attr(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 2
        circuit.elements[1].attr = 4

        preprocess = Preprocess(circuit)

        ivs = preprocess.mask_of_attr(Kinds.IVS)
        ics = preprocess.mask_of_attr(Kinds.ICS)
        r = preprocess.mask_of_attr(Kinds.R)
        self.assertTrue(ivs+ics+r == [True,False,]+[False,False]+[False,True])

    def test_mask_of_kind(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 1
        preprocess = Preprocess(circuit)

        mask = preprocess.mask_of_kind(Kinds.IVS)
        self.assertTrue(mask == [True,False])
        mask = preprocess.mask_of_kind(Kinds.ICS)
        self.assertTrue(mask == [False,False])
        mask = preprocess.mask_of_kind(Kinds.R)
        self.assertTrue(mask == [False,True])