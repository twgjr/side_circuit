from torch.utils.data import Dataset
import numpy as np
from signal_generator import SignalGenerator
import h5py

class SignalDataset(Dataset):
    def __init__(self):
        # open the hdf5 file to write to, create the file if it doesn't exist
        self.file = h5py.File("dataset.hdf5", "a")

        # create the initial 1000 example dataset if it doesn't exist
        if("sine" not in self.file):
            self.create_datasets(10)

        self.file.close()

    def __del__(self):
        self.file.close()
    
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, idx):
        return self.file[str(idx)]
    
    def create_datasets(self, num_examples):
        generator = SignalGenerator()

        for i in range(num_examples):
            group = self.file.create_group(str(i))
            signal = generator.generate_sine()
            group.create_dataset("signal", data = signal["signal"])
            group.create_dataset("times", data = signal["times"])

            mean_coeff, mean_exp = signal["mean"]
            group.attrs["mean_coeff"] = mean_coeff
            group.create_dataset("mean_exp", data = mean_exp)
            
            amp_coeff, amp_exp = signal["amplitude"]
            group.attrs["amp_coeff"] = amp_coeff
            group.create_dataset("amp_exp", data = amp_exp)

            group.attrs["frequency"] = signal["frequency"]

            group.attrs["phase"] = signal["phase"]
            
    def append_datasets(self, num_examples):
        # generate the examples
        generator = SignalGenerator()
        for i in range(len(self.file), len(self.file) + num_examples):
            group = self.file.create_group(str(i))
            signal = generator.generate_sine()
            group.create_dataset("signal", data = signal["signal"])
            group.create_dataset("times", data = signal["times"])

            mean_coeff, mean_exp = generator.to_ml_format(signal["mean"])
            group.attrs["mean_coeff"] = mean_coeff
            group.create_dataset("mean_exp", data = mean_exp)
            
            amp_coeff, amp_exp = generator.to_ml_format(signal["amplitude"])
            group.attrs["amp_coeff"] = amp_coeff
            group.create_dataset("amp_exp", data = amp_exp)

            group.attrs["frequency"] = signal["frequency"]

            group.attrs["phase"] = signal["phase"]

    #print the dataset
    def print_dataset(self):
        for i in range(len(self)):
            print(self[i])