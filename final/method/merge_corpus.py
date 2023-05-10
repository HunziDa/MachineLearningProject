import pandas as pd

first_corpus = 'final/subcorpus/直到.csv'
second_corpus = 'final/subcorpus/只有.csv'
result = "final/subcorpus/merge_直到_只有.csv"
# 读取第一个csv文件
df1 = pd.read_csv(first_corpus)

# 读取第二个csv文件
df2 = pd.read_csv(second_corpus)

# 合并两个数据框
merged_df = pd.concat([df1, df2])

# 将合并后的数据框写入新的csv文件
merged_df.to_csv(result, index=False)
