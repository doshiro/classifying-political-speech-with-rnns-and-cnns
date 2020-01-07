# Dataset loader for the presidential dataset found in https://github.com/doshiro/2020-presidential-candidate-dataset
# Makes it easier for tensorflow loading / batching

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchnlp.word_to_vector import GloVe

class PresidentialCandidateDataset(Dataset):

    def __init__(self, candidates, train=True, cnn=False):
        dataset = 'C:\\Users\\usr1\\python\\final project\\merged-filtered-candidate-data\\tokenized\\'
        if(train):
            dataset += 'training'
        else:
            dataset += 'testing'
        i, j = 0, 0
        self.len = 3230 if train else 1489
        self.x_data = [None for x in range(self.len)]
        self.y_data = [None for x in range(self.len)]
        self.vectors = GloVe('6B')
        self.id_candidate_map = [None for x in range(12)]
        for filename in os.listdir(dataset):
            f = open(dataset + '\\' + filename)
            self.id_candidate_map[j] = filename[:-18]
            lines = f.readlines()
            for line in lines:
                if (line):
                    self.x_data[i] = self.convert_line(line) if cnn else line
                    self.y_data[i] = j
                    i += 1
            j += 1
        
        # finally convert to tensor
        if cnn:
            self.x_data = self.zero_pad(self.x_data)
        # self.y_data = torch.LongTensor(self.y_data)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def get_candidate(self, id):
        return self.id_candidate_map[id]

    # adds zero-padding
    def zero_pad(self, data):
        data_lengths = torch.tensor([len(seq) for seq in data])
        zero_padded_tensor = torch.zeros((len(data), 394, 300))
        for x, (sequence, length) in enumerate(zip(data, data_lengths)):
            zero_padded_tensor[x, :length] = sequence
        return zero_padded_tensor

    # expects line as string, returns a #words x 300 (GloVe dim) tensor
    def convert_line(self, line):
        return torch.stack([self.vectors[word] for word in line.split()])