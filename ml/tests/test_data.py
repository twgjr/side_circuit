import unittest
from data import Data
from circuits import Circuit,Kinds,Props,Signal
import torch
from torch import Tensor

class Test_Data(unittest.TestCase):
    def test_Data(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        data = Data(circuit)
        self.assertTrue(type(data.circuit)==Circuit)
        self.assertTrue(type(data.M)==Tensor)

    def test_signals_base(self):
        data = Data(Circuit())
        self.assertTrue(data.signals_base([Signal(None,[0,1,2])]) == 2)
        self.assertTrue(data.signals_base([Signal(None,[-1,1,2])]) == 2)
        self.assertTrue(data.signals_base([Signal(None,[0,0,0])]) == 1e-12)
        self.assertTrue(data.signals_base([Signal(None,[-1,-1])]) == 1)
        self.assertTrue(data.signals_base([Signal(None,[0,0]),
                                     Signal(None,[0,0])]) == 1e-12)
        self.assertTrue(data.signals_base([Signal(None,[0,2]),
                                     Signal(None,[0,1])]) == 2)
        
    def test_init_base(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,2)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].a = 3.0
        circuit.elements[2].a = 6.0
        circuit.elements[1].i = [1.5,4.0]
        data = Data(circuit)
        self.assertTrue(data.v_base == 2.0)
        self.assertTrue(data.r_base == 6.0)
        self.assertTrue(data.i_base == 4.0)

    def test_init_attrs_mask(self):
        assert len(Kinds) <= 3
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,2)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].a = 4.0
        data = Data(circuit)
        attrs_mask = data.init_attrs_mask()
        self.assertTrue(torch.equal(attrs_mask, torch.tensor([False,True,False])))
        
    def test_values_base(self):
        data = Data(Circuit())
        self.assertTrue(data.values_base([0,1,2]) == 2)
        self.assertTrue(data.values_base([-1,1,2]) == 2)
        self.assertTrue(data.values_base([0,0,0]) == 1e-12)
        self.assertTrue(data.values_base([-1,-1]) == 1)
        self.assertTrue(data.values_base([0,0,0,0]) == 1e-12)
        self.assertTrue(data.values_base([0,2,0,1]) == 2)

    def test_norm_signals(self):
        circuit = Circuit()
        r = circuit.add_element(Kinds.R)
        ivs = circuit.add_element(Kinds.IVS)
        ivs.v = [1,1,2,4]
        r.v = [1,1,2,4]
        r.i = [1,1,2,3]
        data = Data(circuit)
        i_base = data.signals_base(circuit.prop_list(Props.I))
        norm_i = data.norm_signals(Props.I,i_base)
        self.assertTrue(norm_i[0] == Signal(None,[1/3,1/3,2/3,1.0]))
        v_base = data.signals_base(circuit.prop_list(Props.V))
        norm_v = data.norm_signals(Props.V,v_base)
        self.assertTrue(norm_v[0] == Signal(None,[1/4,1/4,2/4,1]))
        self.assertTrue(norm_v[1] == Signal(None,[1/4,1/4,2/4,1]))
        self.assertTrue(ivs.v == Signal(None,[1,1,2,4]))
        self.assertTrue(r.v == Signal(None,[1,1,2,4]))
        self.assertTrue(r.i == Signal(None,[1,1,2,3]))

    def test_norm_attrs(self):
        circuit = Circuit()
        r1 = circuit.add_element(Kinds.R)
        r2 = circuit.add_element(Kinds.R)
        r1.a = 2.0
        r2.a = 4.0
        data = Data(circuit)
        attrs = circuit.attr_list(Kinds.R)
        base = data.values_base(attrs)
        norm_attrs = data.norm_attrs(Kinds.R,base)
        norm_attrs_test = [0.5,1]
        self.assertTrue(norm_attrs == norm_attrs_test)

    def test_split_input_output(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        data = Data(circuit)
        input = torch.tensor([[1],
                              [2],
                              [3],
                              [4]])
        i_test = torch.tensor([[1],
                               [2]])
        v_test = torch.tensor([[3],
                               [4]])
        i_list,v_list = data.split_input_output(input)
        self.assertTrue(torch.equal(i_list,i_test))
        self.assertTrue(torch.equal(v_list,v_test))
    
    def test_denorm_input_output(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        data = Data(circuit)
        data.i_base = 10
        data.v_base = 10
        input = [torch.tensor([[0.1],[0.2],[0.3],[0.4]]).float()]
        i_test = [torch.tensor([[1],[2]]).float()]
        v_test = [torch.tensor([[3],[4]]).float()]
        i_tensor_list,v_tensor_list = data.denorm_input_output(input)
        for i in range(len(i_tensor_list)):
            self.assertTrue(torch.allclose(i_tensor_list[i],i_test[i]))
            self.assertTrue(torch.allclose(v_tensor_list[i],v_test[i]))

    def test_denorm(self):
        base = 10
        test_tensor = torch.tensor([[3,4,5,6],
                                    [7,8,9,10]])
        input_tensor = torch.tensor([[0.3,0.4,0.5,0.6],
                                     [0.7,0.8,0.9,1]])
        denorm_tensor = Data.denorm(input_tensor,base)
        self.assertTrue(torch.equal(denorm_tensor,test_tensor))
        
    def test_denorm_params(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        data = Data(circuit)
        data.i_base = 100
        data.v_base = 10
        data.r_base = 2
        norm_params = torch.tensor([[0.1],[0.2]]).float()
        params_test = torch.tensor([[0.1],[0.4]]).float()
        denorm_params = data.denorm_params(norm_params)
        self.assertTrue(torch.allclose(denorm_params,params_test))

    def dataset_output(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        el0 = circuit.elements[0]
        el_last = circuit.elements[1]
        el0.a = [2]
        el_last.i = [0.5]
        data = Data(circuit)
        data_list_test = [[],[1.0],[],[]]
        data_list = data.extract_data()
        self.assertTrue(data_list == data_list_test)

    def test_extract_data(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        el0 = circuit.elements[0]
        el_last = circuit.elements[1]
        el0.v = [1,2]
        el_last.i = [4,2]
        data = Data(circuit)
        test_data_i = [[None,1.0],
                       [None,0.5]]
        test_data_v = [[0.5,None],
                       [1.0,None]]
        data_i,data_v = data.extract_data()
        self.assertTrue(data_i == test_data_i)
        self.assertTrue(data_v == test_data_v)

    def test_extract_attributes(self):
        circuit = Circuit()
        circuit.add_element(Kinds.IVS)
        r1 = circuit.add_element(Kinds.R)
        r2 = circuit.add_element(Kinds.R)
        r1.a = 2.0
        r2.a = 4.0
        data = Data(circuit)
        attrs = data.extract_attributes()
        attrs_test = [None,0.5,1]
        self.assertTrue(attrs == attrs_test)
        
    def test_init_dataset(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        el0 = circuit.elements[0]
        el_last = circuit.elements[1]
        el0.v = [1,2]
        el_last.i = [4,2]
        data = Data(circuit)
        init_data_input = data.init_dataset()
        init_data_test = [torch.tensor([0,1.0,0.5,0]).float().unsqueeze(1),
                          torch.tensor([0,0.5,1.0,0]).float().unsqueeze(1)]
        for i in range(len(init_data_test)):
            self.assertTrue(torch.equal(init_data_test[i],init_data_input[i]))
        
    def test_data_mask_list(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v = [2]
        circuit.elements[1].i = [0.5]
        data = Data(circuit)
        test_mask = [False, True, True, False]
        data_mask_list = data.data_mask_list()
        self.assertTrue(data_mask_list == test_mask)

    def test_prop_mask(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        el0 = circuit.elements[0]
        el1 = circuit.elements[1]
        el0.v = [2]
        el1.i = [3]
        input = Data(circuit)
        i_mask = input.prop_mask(Props.I)
        self.assertTrue(i_mask == [False,True])
        v_mask = input.prop_mask(Props.V)
        self.assertTrue(v_mask == [True,False])

    def test_attr_mask(self):
        circuit = Circuit()
        circuit.ring(Kinds.IVS,Kinds.R,1)
        circuit.elements[0].v = [2.0]
        circuit.elements[1].a = 4.0
        data = Data(circuit)
        r = data.attr_mask(Kinds.R)
        self.assertTrue(r == [False,True])