import csv
import torch
import pandas as pd
import numpy as np
import os
import joblib
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans, AgglomerativeClustering

train_path = "final/corpus/unlabelled.csv"
num_clusters = 5

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data['text'][i]
        return text

train_data = TextDataset(train_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') 
model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)

char = '才'
def get_word_vector(text, char):
    tokens = tokenizer.tokenize(text)
    index = tokens.index(char)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_tensor)
    hidden_state = outputs.hidden_states[-1][0][index].numpy()
    return hidden_state

vector_list = []
for i in range(len(train_data)):
    text = train_data[i]
    tensor_tmp = get_word_vector(text, char)
    vector_list.append(tensor_tmp)
    if i%50 == 0:
        print('Index [{}/{}]'.format(i + 1, len(train_data)))
vectors = np.array(vector_list)

os.environ["OMP_NUM_THREADS"] = "1"

max_iterations = 100

init_centers = []
init_centers.append(vectors[0])
init_centers.append(vectors[1])
init_centers.append(vectors[2])
init_centers.append(vectors[3])
init_centers.append(vectors[4])


# 使用指定的初始质心
kmeans = KMeans(n_clusters=num_clusters, init=init_centers, random_state=0)
kmeans.fit(vectors)

cluster_labels = kmeans.labels_ # shape: (n,)
joblib.dump(kmeans, 'final/model/kmeans_5.pkl')

