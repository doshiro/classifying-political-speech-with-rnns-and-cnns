# CNN model described here: https://arxiv.org/abs/1408.5882
# For more info see the paper and pdf in root

import os
import numpy as np
import torch
import torch.nn as nn
import PresidentialCandidateDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnlp.word_to_vector import GloVe
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CNN_PARAM_PATH = 'C:\\Users\\usr1\\python\\final project\\cnn.params'
LEN = 3230
HIDDEN_DIM = 30
N_LAYERS = 2
BATCH_SIZE = 50
N_EPOCHS = 100
CANDIDATES = np.array(np.eye(12).tolist())
LEARNING_RATE = 0.001
vectors = GloVe('6B')

train_dataset = PresidentialCandidateDataset.PresidentialCandidateDataset(CANDIDATES, train=True, cnn=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = PresidentialCandidateDataset.PresidentialCandidateDataset(CANDIDATES, train=False, cnn=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class CustomConvLayer(nn.Module):

    def __init__(self,depth):
        super(CustomConvLayer, self).__init__()
        self.depth = depth
        self.W1 = torch.nn.Parameter(data=torch.randn(depth, 3*300), requires_grad=True)
        self.W2 = torch.nn.Parameter(data=torch.randn(depth, 4*300), requires_grad=True)
        self.W3 = torch.nn.Parameter(data=torch.randn(depth, 5*300), requires_grad=True)

        self.b1 = torch.nn.Parameter(data=torch.randn(depth,1), requires_grad=True)
        self.b2 = torch.nn.Parameter(data=torch.randn(depth,1), requires_grad=True)
        self.b3 = torch.nn.Parameter(data=torch.randn(depth,1), requires_grad=True)

    def convolve(self, input, mat, b):
        n = int(mat.size(1) / 300)
        maximum = torch.zeros(input.size(0), self.depth,1)
        for x in range(input.size(0)):
            paragraph = input[x]
            for i in range(394 - n):
                #get the next n words
                c_i = torch.flatten(paragraph[i:i+n,:])
                for j in range(self.depth):
                    maximum[x,j] = max(maximum[x,j], nn.functional.relu(torch.dot(c_i, mat[j]) + b[j]))
        return maximum

    def forward(self, input):
        in_size = input.size(0)

        input1 = nn.functional.relu(self.convolve(input, self.W1, self.b1))
        input2 = nn.functional.relu(self.convolve(input, self.W2, self.b2))
        input3 = nn.functional.relu(self.convolve(input, self.W3, self.b3))

        input = torch.cat([input1, input2, input3],1).view(in_size,self.depth*3)
        return input

class CNNClassifier(nn.Module):

    def __init__(self, depth):
        super(CNNClassifier, self).__init__()
        self.depth = depth
        
        self.conv = CustomConvLayer(depth)
        # there are 12 candidates to choose from
        self.hidden = nn.Linear(3, 100)
        self.output = nn.Linear(100, 12)
    
    # input is batch_size x #numwords in sentence (340) x dim_word_embedding (300)
    def forward(self, input):
        in_size = input.size(0)

        input = self.conv(input)
        #input = self.dropout(input)
        input = nn.functional.relu(self.hidden(input))
        input = self.output(input)

        return input




def train():
    sum_loss = 0

    # paragraph is a string, candidates is a number
    for i, (paragraphs, candidates) in enumerate(train_loader, 1):
        output = classifier(paragraphs)
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
        output = classifier(paragraphs)
        candidate_pred = output.data.max(1, keepdim=True)[1]
        num_correct += candidate_pred.eq(candidates.data.view_as(candidate_pred)).cpu().sum()

    print('The model correctly predicted', num_correct, '/', training_data_size)


if __name__ == '__main__':
    classifier = CNNClassifier(20)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # Change if needed

    for epoch in range(N_EPOCHS):
        print("epoch", epoch)
        total_loss = train()
        print("loss for this epoch:", total_loss)

    # save now that training is done!
    torch.save(classifier.state_dict(), CNN_PARAM_PATH)

    test()