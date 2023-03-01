import unittest
from data import Preprocess
from circuits import Circuit,Kinds,Props
from torch import Tensor

def is_rand(vals:list):
    assert(isinstance(vals,list))
    for val in vals:
        assert(isinstance(val,float))
        if(not (0 < val and val <= 1)):
            return False
    return True
    

class Test_Preprocess(unittest.TestCase):
    def test_Preprocess(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        input = Preprocess(circuit)
        self.assertTrue(type(input.circuit)==Circuit)
        self.assertTrue(type(input.M)==Tensor)
        self.assertTrue(type(input.elements) == dict)
        self.assertTrue(type(input.truth) == list)
        self.assertTrue(type(input.truth_mask) == list)

    def test_truth_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 1
        circuit.elements[-1].i = 0.5
        input = Preprocess(circuit)
        self.assertTrue(is_rand([input.truth[0]]))
        self.assertTrue(input.truth[1] == 0.5)
        self.assertTrue(input.truth[2] == 1)
        self.assertTrue(is_rand([input.truth[3]]))
        self.assertTrue(is_rand([input.truth[4]]))
        self.assertTrue(is_rand([input.truth[5]]))
    
    def test_truth_mask_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 1
        circuit.elements[-1].i = 0.5
        input = Preprocess(circuit)
        test_mask = [False, True, True, False, False, False]
        self.assertTrue(input.truth_mask == test_mask)

    def test_prop_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        known_ivs = 1
        circuit.elements[0].attr = known_ivs
        known_i = 2
        circuit.elements[1].i = known_i
        input = Preprocess(circuit)
        attr_F_none_F = input.prop_list(Props.V,False,False)
        self.assertTrue(attr_F_none_F == [None,None])
        attr_F_none_T = input.prop_list(Props.V,False,True)
        self.assertTrue(is_rand(attr_F_none_T))
        attr_T_none_T = input.prop_list(Props.V,True,True)
        self.assertTrue(attr_T_none_T[0] == known_ivs)
        self.assertTrue(is_rand([attr_T_none_T[1]]))
        attr_T_none_F = input.prop_list(Props.V,True,False)
        self.assertTrue(attr_T_none_F == [known_ivs,None])

        attr_F_none_F = input.prop_list(Props.I,False,False)
        self.assertTrue(attr_F_none_F == [None,known_i])
        attr_F_none_T = input.prop_list(Props.I,False,True)
        self.assertTrue(is_rand([attr_F_none_T[0]]))
        self.assertTrue(attr_F_none_T[1] == known_i)
        attr_T_none_T = input.prop_list(Props.I,True,True)
        self.assertTrue(is_rand([attr_T_none_T[0]]))
        self.assertTrue(attr_T_none_T[1] == known_i)
        attr_T_none_F = input.prop_list(Props.I,True,False)
        self.assertTrue(attr_T_none_F == [None,known_i])
        
        attr_F_none_F = input.prop_list(Props.Pot,False,False)
        self.assertTrue(attr_F_none_F == [None,None])
        attr_F_none_T = input.prop_list(Props.Pot,False,True)
        self.assertTrue(is_rand(attr_F_none_T))
        attr_T_none_T = input.prop_list(Props.Pot,True,True)
        self.assertTrue(is_rand(attr_T_none_T))
        attr_T_none_F = input.prop_list(Props.Pot,True,False)
        self.assertTrue(attr_T_none_F == [None,None])

    def test_mask_of_prop(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        known_ivs = 1
        circuit.elements[0].attr = known_ivs
        known_i = 2
        circuit.elements[1].i = known_i
        input = Preprocess(circuit)
        i_mask = input.mask_of_prop(Props.I,False)
        self.assertTrue(i_mask == [False,True])
        i_mask_attr = input.mask_of_prop(Props.I,True)
        self.assertTrue(i_mask_attr == [False,True])
        v_mask = input.mask_of_prop(Props.V,False)
        self.assertTrue(v_mask == [False,False])
        v_mask_attr = input.mask_of_prop(Props.V,True)
        self.assertTrue(v_mask_attr == [True,False])
        p_mask = input.mask_of_prop(Props.Pot,False)
        self.assertTrue(p_mask == [False,False])
        p_mask_attr = input.mask_of_prop(Props.Pot,True)
        self.assertTrue(p_mask_attr == [False,False])

    def test_to_bool_mask(self):
        preprocess = Preprocess(Circuit())
        input = [None,0,1.1]
        mask = preprocess.to_bool_mask(input)
        self.assertTrue(mask == [False,True,True])

    def test_replace_nones(self):
        preprocess = Preprocess(Circuit())
        input = [None,2,None,1.1]
        zeros = preprocess.replace_nones(input,False)
        self.assertTrue(zeros == [0,2,0,1.1])
        rands = preprocess.replace_nones(input,True)
        self.assertTrue(is_rand([rands[0],rands[2]]))
        self.assertTrue(rands[1] == input[1])
        self.assertTrue(rands[3] == input[3])

    def test_attr_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 1
        preprocess = Preprocess(circuit)

        replace_nones = True
        rand_unknowns = True
        ivs = preprocess.attr_list(Kinds.IVS,replace_nones,rand_unknowns)
        ics = preprocess.attr_list(Kinds.ICS,replace_nones,rand_unknowns)
        r = preprocess.attr_list(Kinds.R,replace_nones,rand_unknowns)
        self.assertTrue(ivs[0] == 1)
        self.assertTrue(is_rand([ivs[1]]+ics+r))

        replace_nones = True
        rand_unknowns = False
        ivs = preprocess.attr_list(Kinds.IVS,replace_nones,rand_unknowns)
        ics = preprocess.attr_list(Kinds.ICS,replace_nones,rand_unknowns)
        r = preprocess.attr_list(Kinds.R,replace_nones,rand_unknowns)
        self.assertTrue(ivs+ics+r == [1,0]+[0,0]+[0,0])

        replace_nones = False
        rand_unknowns = False
        ivs = preprocess.attr_list(Kinds.IVS,replace_nones,rand_unknowns)
        ics = preprocess.attr_list(Kinds.ICS,replace_nones,rand_unknowns)
        r = preprocess.attr_list(Kinds.R,replace_nones,rand_unknowns)
        self.assertTrue(ivs+ics+r == [1,None]+[None,None]+[None,None])

        replace_nones = False
        rand_unknowns = True
        ivs = preprocess.attr_list(Kinds.IVS,replace_nones,rand_unknowns)
        ics = preprocess.attr_list(Kinds.ICS,replace_nones,rand_unknowns)
        r = preprocess.attr_list(Kinds.R,replace_nones,rand_unknowns)
        self.assertTrue(ivs+ics+r == [1,None]+[None,None]+[None,None])

    def test_mask_of_attr(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].attr = 1
        preprocess = Preprocess(circuit)

        mask = preprocess.mask_of_attr(Kinds.IVS)
        self.assertTrue(mask == [True,False])
        mask = preprocess.mask_of_attr(Kinds.ICS)
        self.assertTrue(mask == [False,False])
        mask = preprocess.mask_of_attr(Kinds.R)
        self.assertTrue(mask == [False,False])

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