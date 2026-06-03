"""
用于简历分类的 Base BERT + 层级分块聚合模型。

当前项目使用 bert-base-uncased 作为每个 chunk 的编码器，并在其外部
实现文档级聚合和分类层。这里的 Hierarchical BERT 指自定义的长文本
分块分类结构，不是直接加载现成的 Hierarchical BERT 预训练模型。
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
    chunk 级 BERT 编码器 + 文档级聚合分类器。

    输入形状：
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
        dropout=0.3,
        pooling_type="attention",
        freeze_bert=False,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.pooling_type = pooling_type
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if pooling_type == "attention":
            self.chunk_attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False),
            )
        elif pooling_type != "mean":
            raise ValueError("pooling_type 只能是 'mean' 或 'attention'")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def mean_pooling(self, chunk_embeddings, chunk_mask=None):
        """对所有真实 chunk 做平均池化。"""
        if chunk_mask is None:
            return chunk_embeddings.mean(dim=1)

        mask = chunk_mask.unsqueeze(-1).float()
        masked_embeddings = chunk_embeddings * mask
        valid_chunk_count = mask.sum(dim=1).clamp(min=1.0)
        return masked_embeddings.sum(dim=1) / valid_chunk_count

    def attention_pooling(self, chunk_embeddings, chunk_mask=None):
        """学习每个 chunk 的重要性，再加权得到整篇简历向量。"""
        scores = self.chunk_attention(chunk_embeddings).squeeze(-1)

        if chunk_mask is not None:
            scores = scores.masked_fill(chunk_mask == 0, -1e4)

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (chunk_embeddings * weights).sum(dim=1)

    def forward(self, input_ids, attention_mask, chunk_mask=None):
        batch_size, max_chunks, chunk_size = input_ids.size()

        flat_input_ids = input_ids.view(batch_size * max_chunks, chunk_size)
        flat_attention_mask = attention_mask.view(batch_size * max_chunks, chunk_size)

        outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        chunk_embeddings = outputs.last_hidden_state[:, 0, :]
        chunk_embeddings = chunk_embeddings.view(batch_size, max_chunks, -1)

        if self.pooling_type == "attention":
            document_embedding = self.attention_pooling(chunk_embeddings, chunk_mask)
        else:
            document_embedding = self.mean_pooling(chunk_embeddings, chunk_mask)

        document_embedding = self.dropout(document_embedding)
        logits = self.classifier(document_embedding)
        return logits


def HierarchicalBERT_model(config=None, num_labels=None, freeze_bert=False):
    """根据 Config 构建简历分类模型。"""
    if config is None:
        if Config is None:
            raise RuntimeError("Config is not available. Please pass config manually.")
        config = Config()

    if num_labels is None:
        num_labels = infer_num_labels(config.class_data_path) or 20

    return HierarchicalBERTClassifier(
        bert_model_name=config.bert_model_name,
        num_labels=num_labels,
        dropout=config.dropout,
        pooling_type=config.pooling_type,
        freeze_bert=freeze_bert,
    )


if __name__ == "__main__":
    config = Config() if Config is not None else None
    model = HierarchicalBERT_model(config)

    batch_size = 2
    max_chunks = config.max_chunks if config is not None else 4
    chunk_size = config.pad_size if config is not None else 512

    input_ids = torch.zeros(batch_size, max_chunks, chunk_size, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_chunks, chunk_size, dtype=torch.long)
    chunk_mask = torch.ones(batch_size, max_chunks, dtype=torch.long)

    logits = model(input_ids, attention_mask, chunk_mask)
    print(f"logits shape: {logits.shape}")
