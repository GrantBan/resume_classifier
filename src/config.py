"""
简历分类实验的统一配置文件。
"""

from pathlib import Path

import torch
from transformers import AutoTokenizer


class Config:
    """集中管理数据路径、模型路径、训练参数和标签信息。"""

    def __init__(self):
        self.src_dir = Path(__file__).resolve().parent
        self.project_dir = self.src_dir.parent
        self.data_dir = self.project_dir / "data"
        self.model_dir = self.project_dir / "model"

        # 数据路径。
        self.res_data_path = str(self.data_dir / "train.txt")
        self.class_data_path = str(self.data_dir / "category_map.txt")
        self.train_data_path = str(self.data_dir / "train_data.txt")
        self.test_data_path = str(self.data_dir / "test_data.txt")
        self.processed_data_path = str(self.data_dir)
        self.result_dir = self.project_dir / "result"

        # 模型保存路径。
        self.model_save_path = str(self.model_dir / "best_hierarchical_bert.pt")

        # 训练超参数。
        self.batch_size = 2
        self.gradient_accumulation_steps = 4
        self.num_epochs = 5
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.log_interval = 10
        self.debug_mode = False
        self.dropout = 0.3
        self.warmup_ratio = 0.1
        self.use_dev_split = True
        self.dev_size = 0.1
        self.split_seed = 42
        self.early_stopping_patience = 2

        # Hierarchical BERT 相关参数。
        self.bert_model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.pad_size = 512
        self.max_chunks = 4
        self.max_tokens_per_chunk = 510
        self.pooling_type = "attention"

        # 过长样本过滤参数。
        # 当前训练集和测试集约 98% 的样本长度不超过 2040 个正文 token，
        # 因此用 4 个 chunk 作为窗口，并过滤超出窗口容量的极端长样本。
        self.filter_overlong_samples = True
        self.max_total_tokens = self.max_chunks * self.max_tokens_per_chunk

        # 过短样本过滤参数。
        # 少量只有几十个字符的简历通常缺少有效经历、技能和教育信息，
        # 更接近噪声样本，因此在进入模型前过滤。
        self.filter_short_samples = True
        self.min_text_chars = 200
        self.min_total_tokens = 50

        # 类别名称列表，顺序需要和标签 id 保持一致。
        self.class_list = self._load_class_list()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_class_list(self):
        """从 category_map.txt 中读取类别名称。"""
        class_path = Path(self.class_data_path)
        if not class_path.exists():
            return [str(i) for i in range(20)]

        label_pairs = []
        with class_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "\t" in line:
                    label_id, label_name = line.split("\t", 1)
                elif "," in line:
                    label_id, label_name = line.split(",", 1)
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    label_id, label_name = parts

                if label_id.strip().isdigit():
                    label_pairs.append((int(label_id.strip()), label_name.strip()))

        if not label_pairs:
            return [str(i) for i in range(20)]

        label_pairs.sort(key=lambda item: item[0])
        return [label_name for _, label_name in label_pairs]


if __name__ == "__main__":
    config = Config()
    print(f"train_data_path: {config.train_data_path}")
    print(f"test_data_path: {config.test_data_path}")
    print(f"class_list: {config.class_list}")
