"""
用于简历分类的 Hierarchical BERT 模型。

这里没有直接使用 HuggingFace 上的 Common Core 层级分类模型，因为那个模型
是面向教育标准标签训练的。当前项目使用普通 BERT 作为 chunk 编码器，
再把多个 chunk 的表示聚合成整份简历的表示。
"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

try:
    from config import Config
except ImportError:
    Config = None


def infer_num_labels(category_map_path):
    """如果类别映射文件存在，则从文件中推断类别数量。"""
    path = Path(category_map_path)
    if not path.exists():
        return None

    labels = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                parts = line.split("\t")
            elif "," in line:
                parts = line.split(",")
            else:
                parts = line.split()

            for part in parts:
                part = part.strip()
                if part.isdigit():
                    labels.add(int(part))

    return len(labels) if labels else None


class HierarchicalBERTClassifier(nn.Module):
    """
    chunk 级 BERT 编码器 + 文档级池化分类器。

    期望输入形状：
        input_ids:      [batch_size, max_chunks, chunk_size]
        attention_mask: [batch_size, max_chunks, chunk_size]
        chunk_mask:     [batch_size, max_chunks]

    输出形状：
        logits:         [batch_size, num_labels]
    """

    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        num_labels=20,
        dropout=0.2,
        freeze_bert=False,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        if freeze_bert:
            # 冻结 BERT 编码层时，只训练最后的分类层。
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, chunk_mask=None):
        batch_size, max_chunks, chunk_size = input_ids.size()

        flat_input_ids = input_ids.view(batch_size * max_chunks, chunk_size)
        flat_attention_mask = attention_mask.view(batch_size * max_chunks, chunk_size)

        outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        # 每个 chunk 使用 [CLS] 向量作为该片段的语义表示。
        chunk_embeddings = outputs.last_hidden_state[:, 0, :]
        chunk_embeddings = chunk_embeddings.view(batch_size, max_chunks, -1)

        if chunk_mask is None:
            document_embedding = chunk_embeddings.mean(dim=1)
        else:
            # chunk_mask 用于忽略 padding 出来的空 chunk。
            mask = chunk_mask.unsqueeze(-1).float()
            masked_embeddings = chunk_embeddings * mask
            valid_chunk_count = mask.sum(dim=1).clamp(min=1.0)
            document_embedding = masked_embeddings.sum(dim=1) / valid_chunk_count

        document_embedding = self.dropout(document_embedding)
        logits = self.classifier(document_embedding)

        return logits


def HierarchicalBERT_model(config=None, num_labels=None, freeze_bert=False):
    """
    根据 Config 构建简历分类用的 Hierarchical BERT 模型。
    """
    if config is None:
        if Config is None:
            raise RuntimeError("Config is not available. Please pass config manually.")
        config = Config()

    if num_labels is None:
        num_labels = infer_num_labels(config.class_data_path) or 20

    return HierarchicalBERTClassifier(
        bert_model_name=config.bert_model_name,
        num_labels=num_labels,
        freeze_bert=freeze_bert,
    )


if __name__ == "__main__":
    config = Config() if Config is not None else None
    model = HierarchicalBERT_model(config)

    batch_size = 2
    max_chunks = config.max_chunks if config is not None else 8
    chunk_size = config.pad_size if config is not None else 512

    input_ids = torch.zeros(batch_size, max_chunks, chunk_size, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_chunks, chunk_size, dtype=torch.long)
    chunk_mask = torch.ones(batch_size, max_chunks, dtype=torch.long)

    logits = model(input_ids, attention_mask, chunk_mask)
    print(f"logits shape: {logits.shape}")
