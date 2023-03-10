import unittest
from circuits import Circuit,Kinds
from data import Data
from models import Solver
from torch.nn import Parameter
import torch
from torch import Tensor


class Test_Solver(unittest.TestCase):
    def test_Solver(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].attr = 2
        circuit.elements[1].i = 3
        data = Data(circuit)
        solver = Solver(data)
        self.assertTrue(isinstance(solver.data,Data))
        self.assertTrue(isinstance(solver.ics_attr_mask,Tensor))
        self.assertTrue(isinstance(solver.ivs_attr_mask,Tensor))
        self.assertTrue(isinstance(solver.r_attr_mask,Tensor))
        self.assertTrue(isinstance(solver.known_attr_mask,Tensor))
        self.assertTrue(isinstance(solver.i_base,int) or 
                        isinstance(solver.i_base,float))
        self.assertTrue(isinstance(solver.v_base,int) or 
                        isinstance(solver.v_base,float))
        self.assertTrue(isinstance(solver.r_base,int) or 
                        isinstance(solver.r_base,float))
        self.assertTrue(isinstance(solver.attr,Tensor))

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

    def test_build_1IVS_1R_Ladder(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].attr = 2
        circuit.elements[1].attr = 3
        data = Data(circuit)
        solver = Solver(data)
        A,b = solver.build()
        A_tester = torch.tensor([[-1,-1, 0, 0], 
                                 [ 1, 1, 0, 0],
                                 [ 0, 0, 1,-1],
                                 [ 0, 0, 1, 0],
                                 [ 0,-1, 0, 1]]).to(torch.float)
        self.assertFalse(False in torch.eq(A, A_tester))
        b_tester = torch.tensor([0, 0, 0, 1, 0]).unsqueeze(1).to(torch.float)
        self.assertFalse(False in torch.eq(b, b_tester))

    def test_forward(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].attr = 2
        circuit.elements[1].attr = 0.5
        data = Data(circuit)
        solver = Solver(data)
        x = solver.forward()
        x_tester = torch.tensor([-1,1,1,1]).unsqueeze(1).to(torch.float)
        self.assertFalse(False in torch.eq(x, x_tester))