"""
构建 Hierarchical BERT 训练和测试所需的 Dataset 与 DataLoader。
"""

import torch
from torch.utils.data import DataLoader, Dataset

from build_datasets import build_hbert_chunks, get_token_counter
from config import Config


config = Config()
_count_tokens = None


def _get_count_tokens():
    """延迟初始化 tokenizer 计数器，避免每次 collate 都重复加载。"""
    global _count_tokens
    if _count_tokens is None:
        _count_tokens = get_token_counter(config.bert_model_name)
    return _count_tokens


def load_raw_data(data_path):
    """读取 text<TAB>label 格式的数据，兼容有表头和无表头两种情况。"""
    data = []

    need_count_tokens = config.filter_overlong_samples or config.filter_short_samples
    count_tokens = _get_count_tokens() if need_count_tokens else None
    removed_short_count = 0
    removed_long_count = 0

    def add_sample(text, label):
        nonlocal removed_short_count, removed_long_count
        text = text.strip()

        if config.filter_short_samples and len(text) < config.min_text_chars:
            removed_short_count += 1
            return

        token_count = count_tokens(text) if count_tokens is not None else 0
        if config.filter_short_samples and token_count < config.min_total_tokens:
            removed_short_count += 1
            return

        if config.filter_overlong_samples and token_count > config.max_total_tokens:
            removed_long_count += 1
            return

        data.append((text, int(label)))

    with open(data_path, "r", encoding="utf-8") as f:
        header = f.readline()
        if header and header.strip().lower() == "text\tlabel":
            pass
        elif header.strip():
            text, label = header.strip().split("\t", 1)
            add_sample(text, label)

        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.split("\t", 1)
            add_sample(text, label)

    if removed_short_count or removed_long_count:
        print(
            f"{data_path} 已过滤过短样本 {removed_short_count} 条，"
            f"过长样本 {removed_long_count} 条，"
            f"保留样本 {len(data)} 条，"
            f"最短阈值 {config.min_text_chars} 字符 / {config.min_total_tokens} token，"
            f"最长阈值 {config.max_total_tokens} token。",
            flush=True,
        )
    return data


class TextDataset(Dataset):
    """保存简历文本和对应类别标签。"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item][0]
        label = self.data[item][1]
        return text, label


def collate_fn(batch):
    """
    将一个 batch 的长简历转换成 Hierarchical BERT 输入。

    输出形状：
        input_ids:      [batch_size, max_chunks, chunk_size]
        attention_mask: [batch_size, max_chunks, chunk_size]
        chunk_mask:     [batch_size, max_chunks]
        labels:         [batch_size]
    """
    texts = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    max_chunks = config.max_chunks
    chunk_size = config.pad_size
    count_tokens = _get_count_tokens()

    flat_chunks = []
    chunk_masks = []

    for text in texts:
        chunks = build_hbert_chunks(
            text,
            count_tokens,
            config.max_tokens_per_chunk,
            max_chunks,
        )

        if not chunks:
            chunks = [""]

        # 真实 chunk 记为 1，补齐出来的空 chunk 记为 0。
        num_real = min(len(chunks), max_chunks)
        chunks = chunks[:max_chunks]
        while len(chunks) < max_chunks:
            chunks.append("")

        flat_chunks.extend(chunks)
        chunk_masks.append([1] * num_real + [0] * (max_chunks - num_real))

    token_encode = config.tokenizer(
        flat_chunks,
        padding="max_length",
        max_length=chunk_size,
        truncation=True,
    )

    batch_size = len(texts)
    input_ids = torch.tensor(token_encode["input_ids"]).view(
        batch_size,
        max_chunks,
        chunk_size,
    )
    attention_mask = torch.tensor(token_encode["attention_mask"]).view(
        batch_size,
        max_chunks,
        chunk_size,
    )
    chunk_mask = torch.tensor(chunk_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, chunk_mask, labels


def build_dataloader():
    """构建训练集和测试集的 DataLoader。"""
    train_data = load_raw_data(config.train_data_path)
    test_data = load_raw_data(config.test_data_path)

    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    train_loader, test_loader = build_dataloader()
    batch = next(iter(train_loader))
    input_ids, attention_mask, chunk_mask, labels = batch
    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"chunk_mask: {chunk_mask.shape}")
    print(f"labels: {labels.shape}")
    print(f"chunk_mask sample: {chunk_mask[0].tolist()}")
