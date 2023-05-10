from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

result_path = "final/result/kmeans_5.csv"
data = pd.read_csv(result_path, skiprows=1)
# 获取正确标签和预测标签
y_true = data.iloc[:, 1]
y_pred = data.iloc[:, 2]

label = []
predict = []
for i in range(len(y_true)):
    if y_true[i] == 2 or y_true[i] ==3:
        label.append(y_true[i])
        if y_true[i] != y_pred[i]:
            if y_true[i] == 2:
                predict.append(3)
            else:
                predict.append(2)
        else:
            predict.append(y_pred[i])

# 计算准确率
accuracy = accuracy_score(label, predict)

# 计算F1分数
f1 = f1_score(label, predict, pos_label=2)

print("准确率: {:.2f}".format(accuracy))
print("F1分数: {:.2f}".format(f1))
