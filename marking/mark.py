import csv

# 标签字典，将标签的序号映射到完整的标签
label_dict = {
    '1': '1. just happened',
    '2': '2. happened late',
    '3': '3. lower than expected',
    '4': '4. necessary condition',
    '5': '5. emphasize a fact',
    '6': '6. not adverb'
}

# 打开CSV文件并读取数据
with open('./marking/Unclassified.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

# 获取上次停止的位置（如果有）
resume_index = 0
try:
    with open('./marking/progress.txt', 'r') as f:
        resume_index = int(f.read().strip())
        print(f'恢复标注进度：从第{resume_index}个句子继续')
except FileNotFoundError:
    print('没有找到保存的进度文件，从头开始标注')

# 循环遍历每个句子并添加标签
for i in range(resume_index, len(rows)):
    row = rows[i]
    # 如果已经标注过，则跳过
    if row[1]:
        continue
    sentence = row[0]
    print(f'[{i+1}/{len(rows)}] 句子：{sentence}')
    label = input('请输入标签序号（1-6）：')
    while label not in label_dict:
        label = input('输入有误，请重新输入标签序号（1-6）：')

    row[1] = label_dict[label]

    # 每标注20个句子保存一次进度
    if (i + 1) % 20 == 0:
        with open('./marking/progress.txt', 'w') as f:
            f.write(str(i + 1))
        if i < len(rows) - 1:
            quit_input = input('是否退出标注？（y/n）')
            if quit_input.lower() == 'y':
                with open('./marking/progress.txt', 'w') as f:
                    f.write(str(i + 1))
                with open('file.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(rows)
                break

# 将带有标签的数据写回CSV文件
with open('./marking/Unclassified.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)

print('标注完成！')
