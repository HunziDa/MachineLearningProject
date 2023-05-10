import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

# 加载本地模型
model_path = "final/model/model_3_5"
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
config = BertConfig.from_pretrained(model_path + "/config.json")
model = BertForSequenceClassification.from_pretrained(model_path + "/pytorch_model.bin", config=config)
model.eval()

def predict_label(sentence):
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels.item()
while(1):
    s = input("输入句子：")
    predicted_label = predict_label(s)
    print(predicted_label)
