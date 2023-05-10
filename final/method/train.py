import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import torch.nn.functional as F
import random

train_path = "final/corpus/train_1000.csv"
model_path = "final/model/model_"
num_label = 5
num_epochs = 5
task_index = "3_"
# 读取训练集数据
train_df = pd.read_csv(train_path)

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_label)
#config = BertConfig.from_pretrained("model2/config.json")
#model = BertForSequenceClassification.from_pretrained("model2/pytorch_model.bin", config=config)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义标签转换函数
def get_label(origin_label):
    return origin_label

# 定义训练函数
def train(model, tokenizer, train_df, num_epochs=num_epochs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(num_epochs):
        running_loss = 0.0
        random_indexes = random.sample(range(len(train_df)), len(train_df))
        count = 0
        for i in random_indexes:
            input_text = train_df.iloc[i]['text']
            label = get_label(train_df.iloc[i]['label_meaning'])
            
            # 将文本编码为BERT输入格式
            encoded = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            label = torch.tensor([label]).long().to(device) 
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss  # 取第一个损失
            logits = outputs.logits
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if count%50 == 0:
                print('Epoch [{}/{}], Index [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, count+1, len(train_df), running_loss/(i+1)))
            count+=1
        model.save_pretrained(model_path+task_index+str(epoch+1))
    return model
# 训练模型
model = train(model, tokenizer, train_df)


