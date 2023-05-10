import pandas as pd
import re
# 读取CSV文件
df = pd.read_csv("weakly/到.csv")


# 使用字符串方法和条件选择删除包含“直到”的行
df = df[~df["text"].str.contains("直到")]

# 保存更改回源文件
df.to_csv("weakly/到.csv", index=False)
