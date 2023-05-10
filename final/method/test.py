import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import accuracy_score, f1_score

model_path = "final/model/model_3_5"
test_path = "final/corpus/test_100.csv"
result_path = "final/result/model_3_5.csv"
# 加载本地模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
config = BertConfig.from_pretrained(model_path + "/config.json")
model = BertForSequenceClassification.from_pretrained(model_path + "/pytorch_model.bin", config=config)

def get_label(origin_label):
    return origin_label

# 读取测试集csv文件
test_df = pd.read_csv(test_path)

# 添加新的一列，用于存储预测结果
test_df["prediction"] = ""
label_data = test_df['label_meaning']
cluster_labels = []
# 遍历测试集的每一行，逐行进行预测
for i, row in test_df.iterrows():
    text = row["text"]
    encoded_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    input_ids = encoded_text["input_ids"]
    attention_mask = encoded_text["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, axis=1)
    test_df.at[i, "prediction"] = predicted_label.item()
    cluster_labels.append(predicted_label.item())

accuracy = accuracy_score(label_data, cluster_labels)
f1 = f1_score(label_data, cluster_labels, average='macro')

print("准确度：{:.2f}".format(accuracy))
print("F1得分：{:.2f}".format(f1))

# 将预测结果写回csv文件
test_df.to_csv(result_path, index=False)
