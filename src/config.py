"""
配置文件
"""

from transformers import AutoTokenizer


class Config():
    def __init__(self):
        # 原始数据路径
        self.res_data_path = "../data/train.txt"
        # 类别数据路径
        self.class_data_path = "../data/category_map.txt"
        # 划分后的训练集
        self.train_data_path = "../data/train_data.txt"
        # 划分后的测试集
        self.test_data_path = "../data/test_data.txt"

        # 批次大小
        self.batch_size = 16

        # 预处理后的数据路径
        self.processed_data_path = "../data/"

        # 模型保存路径
        self.model_save_path = "../model/"

        # Hierarchical BERT
        self.bert_model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.pad_size = 512  # 每个 chunk 的 token 长度（含 [CLS]/[SEP]）
        self.max_chunks = 8  # 每份简历最多 chunk 数
        self.max_tokens_per_chunk = 510  # 切 chunk 时的内容 token 上限
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    config = Config()
    import pandas as pd
    data = pd.read_csv(config.train_data_path, sep="\t", names=["text", "label"])
    print(f"data.head(5):{data.head(5)}")