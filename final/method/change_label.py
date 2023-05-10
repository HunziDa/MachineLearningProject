import pandas as pd
import numpy as np
path = 'final/corpus/test_100.csv'

# 读取csv文件
df = pd.read_csv(path, header=0)

# 定义新的标签映射
new_labels = {
    1: 1,
    2: 2,
    3: 1,
    4: 3,
    5: 4,
    0: 0
}

# 使用map()函数将标签映射到新的标签
df['label_meaning'] = df['label_meaning'].astype(int)
#df['label_meaning'] = df['label_meaning'].map(new_labels)



# 将更新后的DataFrame保存回原文件
df.to_csv(path, index=False)

# 统计每个标签出现的数量
label_counts = df['label_meaning'].value_counts()

# 输出标签出现的数量
print(label_counts)
