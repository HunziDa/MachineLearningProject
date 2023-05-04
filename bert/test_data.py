import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import joblib
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') 
model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)
kmeans = joblib.load('kmeans_才.pkl')
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

df = pd.read_csv('data\dictionary_才.csv', encoding="utf-8")
text_data = df['text']
label_data = df['label_meaning']
label_list = [((int(s[0]) - 1)%6) for s in label_data]


vector_list = []
for i in range(len(text_data)):
    text = text_data[i]
    tensor_tmp = get_word_vector(text, char)
    vector_list.append(tensor_tmp)
vectors = np.array(vector_list)

cluster_labels = kmeans.predict(vectors)
df['label_predicted'] = cluster_labels

# 将带有预测标签的数据写回CSV文件
df.to_csv('data\dictionary_才.csv', index=False)
# 计算准确度和F1得分
accuracy = accuracy_score(label_list, cluster_labels)
f1 = f1_score(label_list, cluster_labels, average='macro')

print("准确度：{:.2f}".format(accuracy))
print("F1得分：{:.2f}".format(f1))

#准确度：0.36
#F1得分：0.42
