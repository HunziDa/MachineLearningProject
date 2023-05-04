import pandas as pd

# 读取csv文件
df = pd.read_csv('data\corpus.csv')

# 删除包含两个及以上“才”字的行
df = df[~(df['text'].str.count('才') >= 2)]

# 将结果写回csv文件
df.to_csv('data\corpus.csv', index=False)
