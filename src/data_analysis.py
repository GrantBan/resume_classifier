"""
数据分析
"""

import pandas as pd
from config import Config

config = Config()



id2name = {}
# 读取类别文件
with open(config.class_data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        index, name = line.split("\t")
        id2name[index] = name
    # print(f"id2name:{id2name}")
            


train_data = {}
with open(config.res_data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        text, label = line.split("\t")
        class_name = id2name[label]
        # print(f"text:{text}, class_name:{class_name}")
        train_data[text] = class_name

with open(config.processed_data_path+"fasttext_train_data.txt", "w", encoding="utf-8") as f:
    for text, class_name in train_data.items():
        f.write(f"__label__{class_name} {text}\n")

print(f"train_data:行数:{len(train_data)}")
print(f"train_data前5条内容:{list(train_data.items())[:5]}")


