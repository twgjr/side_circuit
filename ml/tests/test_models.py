import unittest
from circuits import Circuit,Kinds
from data import Data
from models import Cell,Impedance,Admittance,Elements,Coefficients,Sources,\
                    Constants,Switch
import torch

class Test_Cell(unittest.TestCase):
    def test_Cell(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v = [2]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        init_dataset = data.init_dataset()
        cell = Cell(data)
        self.assertTrue(isinstance(cell.data,Data))
        self.assertTrue(data.v_base == 2)
        self.assertTrue(data.r_base == 3)
        out = cell(init_dataset[0])
        out_test = torch.tensor([-1,1,1,1]).float().unsqueeze(dim=1)
        self.assertTrue(torch.allclose(out,out_test))

class Test_Impedance(unittest.TestCase):
    def test_Impedance(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        z = Impedance(data)
        self.assertTrue(isinstance(z.data,Data))
        iv_in = data.init_dataset()[0]
        i_in, v_in = data.split_input_output(iv_in)
        e = Elements(data)
        triggers = e.triggers(i_in, v_in)
        params = data.init_params()
        z_out = z.forward(params,triggers)
        z_out_test = torch.tensor([[0, 0],
                                   [0,-1]]).float()
        self.assertTrue(torch.allclose(z_out,z_out_test))

class Test_Admittance(unittest.TestCase):
    def test_Y(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        y = Admittance(data)
        iv_in = data.init_dataset()[0]
        i_in, v_in = data.split_input_output(iv_in)
        e = Elements(data)
        triggers = e.triggers(i_in, v_in)
        triggers_toggle = torch.zeros_like(triggers)
        triggers_toggle[data.vcsw_mask] = ~triggers[data.vcsw_mask]
        y_out = y(triggers_toggle)
        y_out_test = torch.tensor([[1, 0],
                                   [0, 1]]).float()
        self.assertTrue(torch.allclose(y_out,y_out_test))

class Test_SW(unittest.TestCase):
    def test_SW(self):
        circuit = Circuit()
        src, ctl_src, res, ctl_res, ctl, sw = circuit.switched_resistor()
        ctl.v = [1.0]
        data = Data(circuit)
        sw = Switch(data)
        iv_in = data.init_dataset()[0]
        i_in, v_in = data.split_input_output(iv_in)
        e = Elements(data)
        triggers = e.triggers(i_in, v_in)
        # as used in Z model
        sw_out = sw.forward(triggers)
        sw_out_test = torch.tensor([[0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1],]).float()
        self.assertTrue(torch.allclose(sw_out,sw_out_test))
        # as used in Y model
        triggers_toggle = torch.zeros_like(triggers)
        triggers_toggle[data.vcsw_mask] = ~triggers[data.vcsw_mask]
        sw_out = sw.forward(triggers_toggle)
        sw_out_test = torch.tensor([[0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],]).float()
        self.assertTrue(torch.allclose(sw_out,sw_out_test))

class Test_Elements(unittest.TestCase):
    def test_E_known_r(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        iv_in = data.init_dataset()[0]
        i_in, v_in = data.split_input_output(iv_in)
        e = Elements(data)
        e_out = e(i_in, v_in)
        e_out_test = torch.tensor([[0, 0, 1, 0],
                                   [0,-1, 0, 1]]).float()
        self.assertTrue(torch.allclose(e_out,e_out_test))

    def test_E_missing_r(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].i = [1.5]
        data = Data(circuit)
        e = Elements(data)
        iv_in = data.init_dataset()[0]
        i_in, v_in = data.split_input_output(iv_in)
        e_out = e(i_in, v_in)
        e_out_test = torch.tensor([[0, 0, 1, 0],
                                   [0,-1, 0, 1]]).float()
        self.assertTrue(torch.allclose(e_out,e_out_test))

    def test_triggers(self):
        circuit = Circuit()
        src, ctl_src, res, ctl_res, ctl_el, sw = circuit.switched_resistor()
        ctl_el.v = [1.0]
        data = Data(circuit)
        iv_in = data.init_dataset()[0]
        i_in, v_in = data.split_input_output(iv_in)
        e = Elements(data)
        triggers = e.triggers(i_in, v_in)
        triggers_test = torch.tensor([0, 0, 0, 0, 0, 1]).float()
        self.assertTrue(torch.equal(triggers,triggers_test))

    def test_switched_resistor(self):
        circuit = Circuit()
        src, ctl_src, res, ctl_res, ctl_el, sw = circuit.switched_resistor()
        res.a = 1.0
        ctl_res.a = 2.0
        ctl_el.v = [1.0]
        data = Data(circuit)
        iv_in = data.init_dataset()[0]
        i_in, v_in = data.split_input_output(iv_in)
        e = Elements(data)
        e_out = e.forward(i_in, v_in)
        e_out_test = torch.tensor([[0, 0,   0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0,   0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0,-0.5, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0,   0,-1, 0, 0, 0, 0, 0, 1, 0, 0],
                                   [0, 0,   0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0,   0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]).float()
        self.assertTrue(torch.allclose(e_out,e_out_test))

class Test_Coefficients(unittest.TestCase):
    def test_Coefficients(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        a = Coefficients(data)
        iv_in = data.init_dataset()[0]
        i_in, v_in = data.split_input_output(iv_in)
        a_out = a.forward(i_in, v_in)
        a_out_test = torch.tensor([[-1,-1, 0, 0],
                                   [ 1, 1, 0, 0],
                                   [ 0, 0, 1,-1],
                                   [ 0, 0, 1, 0],
                                   [ 0,-1, 0, 1]]).float()
        self.assertTrue(torch.allclose(a_out,a_out_test))

class Test_Sources(unittest.TestCase):
    def test_Sources(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        s = Sources(data)
        iv_in = torch.tensor([0.1, 0.2, 0.3, 0.4]).float().unsqueeze(1)
        i_in, v_in = data.split_input_output(iv_in)
        s_out = s(i_in, v_in)
        s_out_test = torch.tensor([0.3, 0.0]).float().unsqueeze(1)
        self.assertTrue(torch.allclose(s_out,s_out_test))

class Test_Constants(unittest.TestCase):
    def test_Constants(self):
        circuit = Circuit()
        circuit.ladder(Kinds.IVS, Kinds.R, 1)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].a = 3.0
        data = Data(circuit)
        b = Constants(data)
        iv_in = torch.tensor([0.1, 0.2, 0.3, 0.4]).float().unsqueeze(1)
        i_in, v_in = data.split_input_output(iv_in)
        b_out = b.forward(i_in, v_in)
        b_out_test = torch.tensor([0, 0, 0, 0.3, 0]).float().unsqueeze(1)
        self.assertTrue(torch.allclose(b_out,b_out_test))