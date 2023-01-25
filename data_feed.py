import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


def create_samples(root, portion=1., shuffle=False, nat_sort=False):
    f = pd.read_csv(root, na_values='')
    f = f.fillna(-99)
    data_samples = []
    pred_beam = []
    inp_beam = []
    for idx, row in f.iterrows():
        lidar_data = row['seq1':'seq8'].tolist()

        data_samples.append(lidar_data)
        future_beam = row['Future_Beam1':'Future_Beam3'].tolist()
        future_beam = np.asarray([np.argmax(np.loadtxt('./scenario8/DEV[95%]/' + pwr[1:])) for pwr in future_beam])
        pred_beam.append(future_beam)

        input_beam = row['Beam1':'Beam8'].tolist()
        input_beam = np.asarray([np.argmax(np.loadtxt('./scenario8/DEV[95%]/' + pwr[1:])) for pwr in input_beam])
        inp_beam.append(input_beam)
        
    print('list is ready')
    num_data = len(data_samples)
    num_data = int(num_data * portion)
    return data_samples[:num_data], inp_beam[:num_data], pred_beam[:num_data]


class DataFeed(Dataset):
    def __init__(self, root_dir, n, init_shuffle=True, portion=1.):
    
        self.root = root_dir
        self.samples, self.inp_val, self.pred_val = create_samples(self.root, shuffle=init_shuffle, portion=portion)
        self.seq_len = n
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        beam_val = self.pred_val[idx]
        input_beam = self.inp_val[idx]

        sample = sample[:self.seq_len]
        input_beam = input_beam[:self.seq_len] 
        out_beam = torch.zeros((3,))
        lidar_val = torch.zeros((self.seq_len, 216))
        input_data = torch.zeros((self.seq_len,))
        for i, s in enumerate(sample):
            data = s 
            lidar_data = loadmat('./scenario8/DEV[95%]'+data[1:])['data'][:, 0] / 10
            lidar_val[i] = torch.tensor(lidar_data, requires_grad=False) 

        for i, s in enumerate(input_beam):
            x = s 
            input_data[i] = torch.tensor(x, requires_grad=False) - 1

        for i, s in enumerate(beam_val):
            x = s
            out_beam[i] = torch.tensor(x, requires_grad=False) - 1
        
        return lidar_val, input_data.long(), torch.squeeze(out_beam.long())

if __name__ == "__main__":
    num_classes = 64
    batch_size = 32
    val_batch_size = 64
    train_dir = './train_seqs.csv'
    val_dir = './test_seqs.csv'
    seq_len = 8
    train_loader = DataLoader(DataFeed(train_dir, seq_len, portion=1.), batch_size=batch_size, shuffle=True)
    data = next(iter(train_loader))
    print('done')