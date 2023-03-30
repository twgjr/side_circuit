import unittest
from circuits import Circuit,Kinds
from data import Data
from models import Cell,Z,Y,E,A,S,B
from torch.nn import Parameter
import torch
from torch import Tensor


class Test_Cell(unittest.TestCase):
    def test_Cell(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v.data = [2]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        init_state = data.init_dataset()
        cell = Cell(data)
        self.assertTrue(isinstance(cell.data,Data))
        self.assertTrue(isinstance(cell.params,Tensor))
        self.assertTrue(data.v_base == 2)
        self.assertTrue(data.r_base == 3)
        state = cell(init_state[0])
        state_test = torch.tensor([-1,1,1,1]).float().unsqueeze(dim=1)
        self.assertTrue(torch.allclose(state,state_test))

class Test_Z(unittest.TestCase):
    def test_Z(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v.data = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        z = Z(data)
        self.assertTrue(isinstance(z.data,Data))
        params = data.init_params()
        z_out = z(params)
        z_out_test = torch.tensor([[0, 0],
                                   [0,-1]]).float()
        self.assertTrue(torch.allclose(z_out,z_out_test))

class Test_Y(unittest.TestCase):
    def test_Y(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v.data = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        y = Y(data)
        y_out = y()
        y_out_test = torch.tensor([[1, 0],
                                   [0, 1]]).float()
        self.assertTrue(torch.allclose(y_out,y_out_test))

class Test_E(unittest.TestCase):
    def test_E_known_r(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v.data = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        e = E(data)
        params = data.init_params()
        e_out = e(params)
        e_out_test = torch.tensor([[0, 0, 1, 0],
                                   [0,-1, 0, 1]]).float()
        self.assertTrue(torch.allclose(e_out,e_out_test))

    def test_E_missing_r(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v.data = [2.0]
        circuit.elements[1].i.data = [1.5]
        data = Data(circuit)
        e = E(data)
        params = data.init_params()
        e_out = e(params)
        e_out_test = torch.tensor([[0, 0, 1, 0],
                                   [0,-1, 0, 1]]).float()
        self.assertTrue(torch.allclose(e_out,e_out_test))

class Test_A(unittest.TestCase):
    def test_A(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v.data = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        a = A(data)
        params = data.init_params()
        a_out = a(params)
        a_out_test = torch.tensor([[-1,-1, 0, 0],
                                   [ 1, 1, 0, 0],
                                   [ 0, 0, 1,-1],
                                   [ 0, 0, 1, 0],
                                   [ 0,-1, 0, 1]]).float()
        self.assertTrue(torch.allclose(a_out,a_out_test))

class Test_S(unittest.TestCase):
    def test_S(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v.data = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        s = S(data)
        iv_in = torch.tensor([0.1, 0.2, 0.3, 0.4]).float().unsqueeze(1)
        s_out = s(iv_in)
        s_out_test = torch.tensor([0.3, 0.0]).float().unsqueeze(1)
        self.assertTrue(torch.allclose(s_out,s_out_test))

class Test_B(unittest.TestCase):
    def test_B(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v.data = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        b = B(data)
        iv_in = torch.tensor([0.1, 0.2, 0.3, 0.4]).float().unsqueeze(1)
        b_out = b(iv_in)
        b_out_test = torch.tensor([0, 0, 0, 0.3, 0]).float().unsqueeze(1)
        self.assertTrue(torch.allclose(b_out,b_out_test))