import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


class TripletWnut(Dataset):

    def __init__(self, wnut_dataset):
        self.wnut_dataset = wnut_dataset
        self.train_data = [t[0] for t in self.wnut_dataset]
        self.train_labels = [t[1] for t in self.wnut_dataset]
        self.labels_set = set(np.array(self.train_labels))
        self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        word1, word2, word3 = triplet_random_miner(self.train_data, self.train_labels, index, self.label_to_indices, self.labels_set)
        return (word1, word2, word3),[]

    def __len__(self):
        return len(self.wnut_dataset)


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(TripletNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding_net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, data):

        x1,x2,x3 = data
        output1 = F.normalize(self.embedding_net(x1),p=2,dim=-1)
        output2 = F.normalize(self.embedding_net(x2),p=2,dim=-1)
        output3 = F.normalize(self.embedding_net(x3),p=2,dim=-1)
        return output1, output2, output3

    def get_embedding(self,encoder_featune):
        return self.embedding_net(encoder_featune)


def triplet_random_miner(train_data,train_labels,index, label_to_indices, labels_set):

    with torch.no_grad():
        word1, label1 = train_data[index], train_labels[index]
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(label_to_indices[label1])
        negative_label = np.random.choice(list(labels_set - {label1}))
        negative_index = np.random.choice(label_to_indices[negative_label])
        word2 = train_data[positive_index]
        word3 = train_data[negative_index]

    return word1, word2, word3

