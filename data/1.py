import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# 加载数据
df = pd.read_csv('data/test.csv')

# 初始化BertTokenizer和BertForSequenceClassification模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=6)

# 定义数据集类
class WSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本转换为Bert模型的输入格式
        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, padding='max_length', return_tensors='pt')
        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

# 定义训练和测试数据集
train_texts = df['text'][:4]
train_labels = df['label_meaning'][:4]
train_dataset = WSDataset(train_texts, train_labels, tokenizer)

test_texts = df['text'][4:]
test_labels = df['label_meaning'][4:]
test_dataset = WSDataset(test_texts, test_labels, tokenizer)

# 定义批量大小和数据加载器
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练模型
model.train()

epochs = 10
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# 测试模型
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy}')
