import numpy as np
import pandas as pd


data_root = 'scenario8/DEV[95%]/scenario8.csv'
in_len = 8
out_len = 3

all_data = pd.read_csv(data_root)
all_seq_idx = all_data['seq_index'].unique()

all_seq_split = []

for i in all_seq_idx:
    tmp = all_data[all_data['seq_index'] == i]
    tmp = tmp[['unit1_pwr_60ghz', 'unit1_lidar_SCR', 'seq_index']]
    all_seq_split.append(tmp)

all_seqs = []
for seq in all_seq_split:
    start = 0
    while start+in_len+out_len < seq.shape[0]:
        lidar = seq['unit1_lidar_SCR'][start:start+in_len].tolist()
        in_beam = seq['unit1_pwr_60ghz'][start:start+in_len].tolist()
        out_beam = seq['unit1_pwr_60ghz'][start+in_len:start+in_len+out_len].tolist()
        seq_idx = seq['seq_index'][0:1].tolist()
        all_seqs.append(lidar+out_beam+in_beam+seq_idx)
        start += 1

col_namse = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8'] + ['Future_Beam1', 'Future_Beam2', 'Future_Beam3'] + ['Beam1', 'Beam2', 'Beam3', 'Beam4', 'Beam5', 'Beam6', 'Beam7', 'Beam8'] + ['seq_index']

all_seqs = pd.DataFrame(all_seqs, columns = col_namse)

train_seq_idx = np.sort(all_seq_idx[:int(0.8*all_seq_idx.shape[0])])
test_seq_idx = np.sort(all_seq_idx[int(0.8*all_seq_idx.shape[0]):])

train_seqs = all_seqs[all_seqs['seq_index'].isin(train_seq_idx)]
test_seqs = all_seqs[all_seqs['seq_index'].isin(test_seq_idx)]

train_seqs.to_csv('train_seqs.csv', index=False)
test_seqs.to_csv('test_seqs.csv', index=False)

print('done')
