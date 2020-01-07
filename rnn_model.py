# Pretty simple GRU-implemented RNN model for the
# sentence classification task. Achieves 40% accurate classification.
# For more info see the pdf in root or presentation

import os
import numpy as np
import torch
import torch.nn as nn
import PresidentialCandidateDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnlp.word_to_vector import GloVe
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

RNN_PARAM_PATH = 'C:\\Users\\usr1\\python\\final project\\rnn.params'
LEN = 3230
HIDDEN_DIM = 30
N_LAYERS = 2
BATCH_SIZE = 256
N_EPOCHS = 100
CANDIDATES = np.array(np.eye(12).tolist())
LEARNING_RATE = 0.001
vectors = GloVe('6B')

# adds zero-padding for batch learning
def zero_pad(paragraphs, y_data):
    x_data = [convert_line(paragraph) for paragraph in paragraphs]
    data_lengths = torch.tensor([len(seq) for seq in x_data])
    zero_padded_tensors = torch.zeros((len(x_data), data_lengths.max(), 300))
    for x, (sequence, length) in enumerate(zip(x_data, data_lengths)):
        zero_padded_tensors[x, :length] = sequence

    #sort
    data_lengths, indices = data_lengths.sort(0, descending=True)
    zero_padded_tensors = zero_padded_tensors[indices]
    y_data = torch.LongTensor(y_data)[indices]

    return Variable(zero_padded_tensors), Variable(y_data), Variable(data_lengths)

# expects line as string, returns a #words x 300 (GloVe dim) tensor
def convert_line(line):
    return torch.stack([vectors[word] * 100 for word in line.split()])

train_dataset = PresidentialCandidateDataset.PresidentialCandidateDataset(CANDIDATES, train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = PresidentialCandidateDataset.PresidentialCandidateDataset(CANDIDATES, train=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class RNNClassifier(nn.Module):

    def __init__(self, hidden_dim, n_layers=2):
        super(RNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # words are embedded into 300 dimensional space
        self.rnn = nn.GRU(input_size=300, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=False, batch_first=True)
        # there are 12 candidates to choose from
        self.fc = nn.Linear(hidden_dim, 12)

    # input is batch_size x #numwords in sentence (340) x dim_word_embedding (300)
    def forward(self, input, input_lengths):
        batch_size = input.size(0)
        hidden_state = self._init_hidden_state(batch_size)

        gru_input = pack_padded_sequence(input, input_lengths.data.cpu().numpy(), batch_first=True)

        #print('input', input, input.size())
        self.rnn.flatten_parameters()
        output, hidden_state = self.rnn(gru_input, hidden_state)
        #print('final hidden_state', hidden_state[-1], hidden_state[-1].size())

        # only train on the last hidden state
        candidate_pred = self.fc(hidden_state[-1])
        # print('fc out', candidate_pred)

        return candidate_pred

    # zero vector as initial hidden state for RNN
    def _init_hidden_state(self, batch_size):
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))

def train():
    sum_loss = 0

    # paragraph is a string, candidates is a number
    for i, (paragraphs, candidates) in enumerate(train_loader, 1):
        padded_paragraphs, candidates, paragraph_lengths = zero_pad(paragraphs, candidates) 
        output = classifier(padded_paragraphs, paragraph_lengths)
        loss = criterion(output, candidates)
        #print('output', output)
        #print('candidates', candidates)
        print("\t loss for this batch:", loss.data.item())
        sum_loss += loss.data.item()

        classifier.zero_grad()
        loss.backward()
        optimizer.step()

    return sum_loss
    
def test():

    print('evaluating model')
    num_correct = 0
    training_data_size = len(test_loader.dataset)

    for paragraphs, candidates in test_loader:
        padded_paragraphs, candidates, paragraph_lengths = zero_pad(paragraphs, candidates) 
        output = classifier(padded_paragraphs, paragraph_lengths)
        candidate_pred = output.data.max(1, keepdim=True)[1]
        num_correct += candidate_pred.eq(candidates.data.view_as(candidate_pred)).cpu().sum()

    print('The model correctly predicted', num_correct, '//', training_data_size)


if __name__ == '__main__':
    classifier = RNNClassifier(HIDDEN_DIM, n_layers=N_LAYERS)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # Change if needed

    for epoch in range(N_EPOCHS):
        print("epoch", epoch)
        total_loss = train()
        print("loss for this epoch:", total_loss)

    # save now that training is done!
    torch.save(classifier.state_dict(), RNN_PARAM_PATH)

    test()