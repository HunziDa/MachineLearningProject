import csv
import torch
import pandas as pd
import numpy as np
import os
import joblib
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans, AgglomerativeClustering

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data['text'][i]
        return text

train_data = TextDataset('data\corpus_才.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') 
model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)

df = pd.read_csv('data\corpus_才.csv', encoding="utf-8")

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
vectors = np.array(vector_list)

print("mid")
os.environ["OMP_NUM_THREADS"] = "1"

num_clusters = 3
max_iterations = 100

init_centers = []
init_centers.append(vectors[1])
init_centers.append(vectors[2])
init_centers.append(vectors[5])

# 使用指定的初始质心
kmeans = KMeans(n_clusters=num_clusters, init=init_centers, random_state=0)
kmeans.fit(vectors)

cluster_labels = kmeans.labels_ # shape: (n,)
joblib.dump(kmeans, 'kmeans_才_3.pkl')

new_column = pd.Series(cluster_labels)
df['label_predicted_3'] = new_column
df.to_csv('data\corpus_才.csv', index=False)

print(cluster_labels)

