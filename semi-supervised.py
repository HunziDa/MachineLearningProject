import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import jieba
from sklearn.metrics import f1_score


labels = {
    '1. just happened':1,
    '2. happened late':2,
    '3. lower than expected':3 ,
    '4. necessary condition':4 ,
    '5. emphasize a fact':5 ,
    '0. not adverb':0
}

output_size = 6
embedding_dim = 300
hidden_dim = 128
n_layers = 2
drop_prob = 0.5
num_epochs = 20
adverb_token = '才'
threshold = 0.9

def label_to_index(s):
    return labels.get(s, 0)

def index_to_label(n):
    for label, index in labels.items():
        if index == n:
            return label
    return 'error'

# Load pretrained word vectors
def load_word_vectors(filename):
    word_vectors = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word, vector = line.split(' ', 1)
            word_vectors[word] = np.fromstring(vector, sep=' ')
    return word_vectors

word_vectors = load_word_vectors('sgns.literature.char')

# Load data
train_data = pd.read_csv('.\data\dictionary.csv')
unlabeled_data  = pd.read_csv('.\data\corpus.csv')

# Set unknown words
vector_zeros = np.zeros((1, embedding_dim))
word_vectors['<unk>'] = vector_zeros

# Generate word2idx and embedding_matrix
vocabulary = list(word_vectors.keys())

word2idx = {word: i for i, word in enumerate(vocabulary)}
vocab_size = len(vocabulary)
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for i, word in enumerate(vocabulary):
    embedding_matrix[i, :] = word_vectors[word]

def find_positions(text, char):
    positions = []
    pos = text.find(char)
    while pos != -1:
        positions.append(pos)
        pos = text.find(char, pos+1)
    return positions

# Define the training model
class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(RNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = input.long()
        embeds = self.embedding(input)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.log_softmax(out)
        return out

# Initial the model
model = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob)
device = torch.device('cuda')
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Train labeled_data")
#Train labeled_data
def train_labeled(model, optimizer, criterion, train_data):
    model.train()
    total_loss = 0.0
    for i, row in train_data.iterrows():
        text = row['text']
        label = row['label_meaning']
        input = [word2idx.get(word, word2idx['<unk>']) for word in text]
        input_tensor = torch.LongTensor(input).unsqueeze(0)
        label_tensor = torch.LongTensor([label_to_index(label)])
        input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)
        optimizer.zero_grad()
        outputs = model(input_tensor)
        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_data)

for epoch in range(num_epochs):
    labeled_loss = train_labeled(model, optimizer, criterion, train_data)
    print("Epoch %d, labeled loss: %f" % (epoch+1, labeled_loss))










