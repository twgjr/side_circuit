import unittest
from circuits import Circuit,Kinds
from data import Data
from models import Solver
from torch.nn import Parameter
import torch


class Test_Solver(unittest.TestCase):
    def test_Solver(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].attr = 2
        circuit.elements[1].i = 3
        data = Data(circuit)
        solver = Solver(data)
        self.assertTrue(type(solver.data) == Data)
        ics_tester = torch.tensor([False, False])
        self.assertFalse(False in torch.eq(solver.ics_attr_mask,ics_tester))
        ivs_tester = torch.tensor([True, False])
        self.assertFalse(False in torch.eq(solver.ivs_attr_mask, ivs_tester))
        r_tester = torch.tensor([False, True])
        self.assertFalse(False in torch.eq(solver.r_attr_mask, r_tester))
        known_tester = torch.tensor([True, False])
        self.assertFalse(False in torch.eq(solver.known_attr_mask, known_tester))
        self.assertTrue(solver.base == 3)
        attr_tester = Parameter(torch.tensor([2/3,1]).to(torch.float))
        self.assertFalse(False in torch.eq(solver.attr, attr_tester))

    def test_M(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].attr = 2
        circuit.elements[1].i = 3
        data = Data(circuit)
        solver = Solver(data)
        M = solver.M()
        M_tester = torch.tensor([[-1, -1,], 
                                 [ 1,  1]])
        self.assertFalse(False in torch.eq(M, M_tester))

    def test_build_2_ladder(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].attr = 2
        circuit.elements[1].attr = 3
        data = Data(circuit)
        solver = Solver(data)
        A,b = solver.build()
        A_tester = torch.tensor([[-1,-1, 0, 0, 0, 0], 
                                 [ 1, 1, 0, 0, 0, 0],
                                 [ 0, 0, 1, 0, 1,-1],
                                 [ 0, 0, 0, 1, 1,-1],
                                 [ 0, 0, 1, 0, 0, 0],
                                 [ 0,-1, 0, 1, 0, 0]]).to(torch.float)
        self.assertFalse(False in torch.eq(A, A_tester))
        b_tester = torch.tensor([0, 0, 0, 0, 2/3, 0]).unsqueeze(1).to(torch.float)
        self.assertFalse(False in torch.eq(b, b_tester))